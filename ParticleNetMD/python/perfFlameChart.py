#!/usr/bin/env python
"""Build time-ordered flame charts (one per thread) from a perf.data file.

A flame graph merges identical stacks and sorts them alphabetically, so its x-axis is a
share of samples. A flame chart keeps the samples in time order instead. perf only samples
while a thread is on a CPU, so this script fills the gap between two consecutive samples of
a thread with an "[off-CPU]" frame whenever the gap is longer than the nominal sampling
interval. The x-axis of every chart is then wall-clock time of that thread.

Usage:
    perfFlameChart.py --perf-data profiling/perf/GA_MHc130_MA90_Combined.perf.data

Outputs (next to the perf.data file unless --outdir is given):
    <stem>.flamechart/<pid>_<tid>_<comm>.{folded,svg}   one chart per kept thread
    <stem>.flamechart/index.tsv                       thread table (start, duration, on-CPU)
    <stem>.flamechart/timeline.svg                    all processes on one time axis, by activity

Run it on the node that recorded the data if the Python frames matter: perf script
resolves them from /tmp/perf-<pid>.map, which only exists there.
"""

import argparse
import html
import math
import os
import re
import subprocess
import sys
from collections import Counter

OFF_CPU = "[off-CPU]"

# Activity categories for the combined timeline, in stacking order (bottom -> top).
# Colors: validated categorical slots 1-5 (light, dark) + neutral "Other".
CATEGORIES = [
    ("python", "Python interpreter", "#2a78d6", "#3987e5"),
    ("torch", "PyTorch C++ dispatch/ops", "#eb6834", "#d95926"),
    ("sync", "Waiting on GPU (stream sync)", "#1baf7a", "#199e70"),
    ("d2h", "GPU→CPU copy (.item/.cpu)", "#eda100", "#c98500"),
    ("gc", "Python garbage collector", "#e87ba4", "#d55181"),
    ("other", "Other (libc, CUDA driver, unresolved)", "#a3a29b", "#6f6e68"),
]
CATEGORY_INDEX = {key: i for i, (key, _, _, _) in enumerate(CATEGORIES)}
SYNC_RE = re.compile(r"cuStreamSynchronize|cuCtxSynchronize|cuEventSynchronize")
D2H_RE = re.compile(r"cuMemcpyDtoH|_local_scalar_dense")
GC_RE = re.compile(r"^(gc_collect|deduce_unreachable|visit_reachable|visit_decref|move_unreachable|"
                   r"subtract_refs|update_refs)")
PY_RE = re.compile(r"^(_?Py|_Py|py::)|dealloc|vectorcall|frame|take_gil|drop_gil|method_get|^object_|"
                   r"initialize_locals|^func_|dict|PyTraceBack|^tuple|^list_|^long_|^unicode")
TORCH_RE = re.compile(r"c10::|at::|torch::|THP")
HEADER_RE = re.compile(r"^(?P<comm>.+?)\s+(?P<pid>\d+)/(?P<tid>\d+)\s+(?P<time>\d+\.\d+):")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Per-thread flame charts from perf.data")
    parser.add_argument("--perf-data", required=True, help="perf.data file from perf record")
    parser.add_argument("--outdir", default=None,
                        help="Output directory (default: <perf-data without .perf.data>.flamechart)")
    parser.add_argument("--flamegraph-dir",
                        default=os.environ.get("FLAMEGRAPH_DIR", os.path.join(os.getcwd(), "FlameGraph")),
                        help="Directory with flamegraph.pl (default: $FLAMEGRAPH_DIR or ./FlameGraph)")
    parser.add_argument("--freq", type=float, default=99.0,
                        help="Sampling frequency used in perf record -F (default: 99)")
    parser.add_argument("--min-oncpu-sec", type=float, default=60.0,
                        help="Skip threads with less on-CPU time than this (default: 60 s)")
    parser.add_argument("--width", type=int, default=1800, help="SVG width in pixels (default: 1800)")
    return parser.parse_args()


def frame_name(line):
    """Turn one perf script callchain line ('addr sym (dso)') into a frame label."""
    parts = line.strip().split(None, 1)
    if len(parts) < 2:
        return "[unknown]"
    rest = parts[1]
    dso = ""
    if rest.endswith(")") and " (" in rest:
        rest, dso = rest.rsplit(" (", 1)
        dso = dso[:-1]
    sym = rest.strip()
    if sym and sym != "[unknown]":
        return sym.replace(";", ":")
    if dso and dso != "[unknown]":
        return f"[{os.path.basename(dso)}]"
    return "[kernel]" if parts[0].startswith("ffff") else "[unknown]"


def read_samples(perf_data):
    """Yield (pid, tid, comm, time_sec, stack_root_first) for every sample."""
    cmd = ["perf", "script", "-i", perf_data, "-F", "comm,pid,tid,time,ip,sym,dso"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            text=True, errors="replace", bufsize=1 << 20)
    header, frames = None, []
    for line in proc.stdout:
        if not line.strip():
            if header is not None:
                yield (*header, frames[::-1])
            header, frames = None, []
            continue
        if line[0] not in " \t":
            match = HEADER_RE.match(line)
            if match is None:
                continue
            header = (int(match["pid"]), int(match["tid"]), match["comm"].strip(), float(match["time"]))
            frames = []
        elif header is not None:
            frames.append(frame_name(line))
    if header is not None:
        yield (*header, frames[::-1])
    if proc.wait() != 0:
        raise RuntimeError(f"perf script failed on {perf_data} (exit code {proc.returncode})")


def build_thread_timelines(samples, interval_us):
    """Group samples per thread, in time order, as run-length encoded (stack, weight_us) lists."""
    threads = {}
    for pid, tid, comm, t_sec, stack in samples:
        key = (pid, tid)
        t_us = t_sec * 1e6
        th = threads.get(key)
        if th is None:
            th = threads[key] = {"pid": pid, "tid": tid, "comm": comm, "start": t_us,
                                 "last": t_us, "last_stack": None, "segments": [], "oncpu": 0.0,
                                 "cuda": False}
        else:
            # The previous sample stands for at most one interval of CPU time; the rest of
            # the gap until this sample is time the thread spent off the CPU.
            dt = max(t_us - th["last"], 0.0)
            on = min(dt, interval_us)
            append_segment(th["segments"], th["last_stack"], on)
            th["oncpu"] += on
            if dt > on:
                append_segment(th["segments"], OFF_CPU, dt - on)
        th["comm"] = comm  # threads are renamed after start (python -> pt_main_thread)
        th["last"] = t_us
        th["last_stack"] = ";".join(stack) if stack else "[no stack]"
        # A thread that ever samples inside the CUDA driver/runtime belongs to a training worker
        if not th["cuda"] and ("libcuda" in th["last_stack"] or "libcudart" in th["last_stack"]):
            th["cuda"] = True
    for th in threads.values():
        append_segment(th["segments"], th["last_stack"], interval_us)
        th["oncpu"] += interval_us
    return threads


def append_segment(segments, stack, weight):
    if weight <= 0:
        return
    if segments and segments[-1][0] == stack:
        segments[-1][1] += weight
    else:
        segments.append([stack, weight])


def classify(stack):
    """Map a root-first stack string to a CATEGORIES index (same rules as the flame graph study)."""
    if SYNC_RE.search(stack):
        return CATEGORY_INDEX["sync"]
    if D2H_RE.search(stack):
        return CATEGORY_INDEX["d2h"]
    leaf = stack.rsplit(";", 1)[-1]
    if GC_RE.search(leaf):
        return CATEGORY_INDEX["gc"]
    if PY_RE.search(leaf):
        return CATEGORY_INDEX["python"]
    if TORCH_RE.search(leaf):
        return CATEGORY_INDEX["torch"]
    return CATEGORY_INDEX["other"]


def bin_thread(th, run_start, bin_us, n_bins):
    """Split a thread's time-ordered segments into fixed time bins.

    Returns (fractions[n_bins][n_cat], leaf_counters[n_bins]) where fractions are shares of
    the bin's wall time and leaf_counters hold on-CPU time per leaf frame (for tooltips).
    """
    n_cat = len(CATEGORIES)
    weights = [[0.0] * n_cat for _ in range(n_bins)]
    leaves = [Counter() for _ in range(n_bins)]
    cache = {}
    t = th["start"] - run_start
    for stack, weight in th["segments"]:
        if stack == OFF_CPU:
            t += weight
            continue
        cat = cache.get(stack)
        if cat is None:
            cat = cache[stack] = classify(stack)
        leaf = stack.rsplit(";", 1)[-1]
        end = t + weight
        while t < end:
            b = min(int(t // bin_us), n_bins - 1)
            chunk = min(end, (b + 1) * bin_us) - t
            if chunk <= 0:  # last bin: put the remainder there
                chunk = end - t
            weights[b][cat] += chunk
            leaves[b][leaf] += chunk
            t += chunk
    fractions = [[min(w / bin_us, 1.0) for w in row] for row in weights]
    return fractions, leaves


def render_timeline(threads, svg_path, title, width=1500, target_bins=420):
    """One SVG: a lane per process main thread on a shared wall-clock axis, stacked by activity."""
    kept = sorted(threads, key=lambda th: th["start"])
    run_start = kept[0]["start"]
    run_end = max(th["last"] for th in kept)
    total_s = (run_end - run_start) / 1e6
    bin_s = max(1, math.ceil(total_s / target_bins))
    bin_us = bin_s * 1e6
    n_bins = int(math.ceil(total_s / bin_s)) or 1

    label_w, summary_w, pad_r = 150, 250, 16
    plot_x0 = label_w
    plot_w = width - label_w - summary_w - pad_r
    lane_h, lane_gap = 22, 6
    top = 112
    height = top + len(kept) * (lane_h + lane_gap) + 56
    x_of = lambda sec: plot_x0 + plot_w * sec / (n_bins * bin_s)

    out = []
    css_light = "".join(f".c{i}{{fill:{light}}}" for i, (_, _, light, _) in enumerate(CATEGORIES))
    css_dark = "".join(f".c{i}{{fill:{dark}}}" for i, (_, _, _, dark) in enumerate(CATEGORIES))
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="system-ui, -apple-system, Segoe UI, sans-serif">\n'
        "<style>"
        ".bg{fill:#fcfcfb}.t1{fill:#0b0b0b}.t2{fill:#52514e}.t3{fill:#8a8983}"
        ".track{fill:#f0efec}.grid{stroke:#e4e3de;stroke-width:1}.mark{stroke:#52514e;stroke-width:1;"
        "stroke-dasharray:4 3}.hit{fill:transparent}.hit:hover{fill:#0b0b0b;fill-opacity:.08}"
        f"{css_light}"
        "@media (prefers-color-scheme: dark){"
        ".bg{fill:#1a1a19}.t1{fill:#f0efec}.t2{fill:#c3c2b7}.t3{fill:#8f8e86}"
        ".track{fill:#262624}.grid{stroke:#33332f}.mark{stroke:#c3c2b7}.hit:hover{fill:#f0efec;fill-opacity:.1}"
        f"{css_dark}}}"
        "text{font-size:12px}.title{font-size:16px;font-weight:600}.small{font-size:11px}"
        "</style>\n"
        f'<rect class="bg" width="{width}" height="{height}"/>\n'
        f'<text class="t1 title" x="16" y="26">{html.escape(title)}</text>\n'
        f'<text class="t2" x="16" y="46">Each row is one process (main thread) on a shared wall-clock axis. '
        f'Row height = share of each {bin_s} s bin spent on the CPU, split by activity; '
        f'empty track = off-CPU (sleeping or blocked). Hover a bin for details.</text>\n'
    )

    # Legend
    lx = 16
    for i, (_, name, _, _) in enumerate(CATEGORIES):
        out.append(f'<rect class="c{i}" x="{lx}" y="62" width="12" height="12" rx="2"/>'
                   f'<text class="t1" x="{lx + 17}" y="72">{html.escape(name)}</text>\n')
        lx += 24 + 7 * len(name)
    out.append(f'<rect class="track" x="{lx}" y="62" width="12" height="12" rx="2"/>'
               f'<text class="t1" x="{lx + 17}" y="72">Off-CPU</text>\n')

    # Time grid
    step = next(s for s in (10, 30, 60, 120, 300, 600, 1200) if total_s / s <= 12)
    lanes_bottom = top + len(kept) * (lane_h + lane_gap) - lane_gap
    for sec in range(0, int(n_bins * bin_s) + 1, step):
        x = x_of(sec)
        out.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 8}" x2="{x:.1f}" y2="{lanes_bottom + 4}"/>'
                   f'<text class="t2 small" x="{x:.1f}" y="{lanes_bottom + 18}" text-anchor="middle">'
                   f'{sec // 60}:{sec % 60:02d}</text>\n')
    out.append(f'<text class="t2" x="{plot_x0 + plot_w / 2:.0f}" y="{lanes_bottom + 38}" '
               f'text-anchor="middle">Wall-clock time since start (min:s)</text>\n')
    out.append(f'<text class="t2 small" x="{width - summary_w - pad_r + 12}" y="{top - 12}">'
               f'share of lifetime · share of CPU time</text>\n')

    # Row roles: the earliest process is the parent; processes that used CUDA are training
    # workers; any other process (e.g. the dataset-loading pool) is a loader.
    labels, n_worker, n_loader = [], 0, 0
    for lane, th in enumerate(kept):
        if lane == 0:
            labels.append("parent")
        elif th["cuda"]:
            labels.append(f"worker {n_worker}")
            n_worker += 1
        else:
            labels.append(f"loader {n_loader}")
            n_loader += 1

    # Mark the first training-worker start (spawn)
    first_worker = next((th for th, lab in zip(kept, labels) if lab.startswith("worker")), None)
    if first_worker is not None:
        x = x_of((first_worker["start"] - run_start) / 1e6)
        out.append(f'<line class="mark" x1="{x:.1f}" y1="{top - 22}" x2="{x:.1f}" y2="{lanes_bottom + 4}"/>'
                   f'<text class="t2 small" x="{x + 4:.1f}" y="{top - 14}">first worker starts</text>\n')

    for lane, th in enumerate(kept):
        y0 = top + lane * (lane_h + lane_gap)
        base = y0 + lane_h
        label = labels[lane]
        out.append(f'<text class="t1" x="{label_w - 10}" y="{y0 + 11}" text-anchor="end">{label}</text>'
                   f'<text class="t3 small" x="{label_w - 10}" y="{y0 + 23}" text-anchor="end">'
                   f'pid {th["pid"]}</text>\n')
        t_start = (th["start"] - run_start) / 1e6
        t_end = (th["last"] - run_start) / 1e6
        out.append(f'<rect class="track" x="{x_of(t_start):.1f}" y="{y0}" '
                   f'width="{max(x_of(t_end) - x_of(t_start), 1):.1f}" height="{lane_h}" rx="2"/>\n')

        fractions, leaves = bin_thread(th, run_start, bin_us, n_bins)
        lower = [0.0] * n_bins
        for c in range(len(CATEGORIES)):
            upper = [lower[b] + fractions[b][c] for b in range(n_bins)]
            if max(fractions[b][c] for b in range(n_bins)) <= 0:
                lower = upper
                continue
            pts_top, pts_bot = [], []
            for b in range(n_bins):
                xa, xb = x_of(b * bin_s), x_of((b + 1) * bin_s)
                pts_top += [(xa, base - lane_h * min(upper[b], 1)), (xb, base - lane_h * min(upper[b], 1))]
                pts_bot += [(xa, base - lane_h * min(lower[b], 1)), (xb, base - lane_h * min(lower[b], 1))]
            pts = pts_top + pts_bot[::-1]
            d = "M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in pts) + "Z"
            out.append(f'<path class="c{c}" d="{d}"/>\n')
            lower = upper

        # Hover targets with exact numbers (also the relief for low-contrast fills)
        for b in range(n_bins):
            on = sum(fractions[b])
            if on <= 0.005:
                continue
            parts = [f"{CATEGORIES[c][1]}: {100 * fractions[b][c]:.0f}%"
                     for c in range(len(CATEGORIES)) if fractions[b][c] >= 0.005]
            tops = [f"  {html.escape(name[:90])} ({w / 1e6:.1f} s)" for name, w in leaves[b].most_common(3)]
            tip = (f"{label} (pid {th['pid']}), {b * bin_s // 60}:{b * bin_s % 60:02d}"
                   f"–{(b + 1) * bin_s // 60}:{(b + 1) * bin_s % 60:02d}\n"
                   f"on-CPU {100 * min(on, 1):.0f}%, off-CPU {100 * max(1 - on, 0):.0f}%\n"
                   + "\n".join(html.escape(p) for p in parts) + "\ntop leaf frames:\n" + "\n".join(tops))
            xa = x_of(b * bin_s)
            out.append(f'<rect class="hit" x="{xa:.1f}" y="{y0}" width="{x_of((b + 1) * bin_s) - xa:.1f}" '
                       f'height="{lane_h}"><title>{tip}</title></rect>\n')

        # Direct summary label per lane: on-CPU share of its lifetime, GPU-wait share of its CPU time
        totals = [sum(fractions[b][c] for b in range(n_bins)) * bin_s for c in range(len(CATEGORIES))]
        life = max(t_end - t_start, 1e-9)
        cpu = max(sum(totals), 1e-9)
        gpu_wait = totals[CATEGORY_INDEX["sync"]] + totals[CATEGORY_INDEX["d2h"]]
        out.append(f'<text class="t1 small" x="{width - summary_w - pad_r + 12}" y="{y0 + 15}">'
                   f'on-CPU {100 * min(cpu / life, 1):.0f}% · GPU wait {100 * gpu_wait / cpu:.0f}% · '
                   f'Python {100 * totals[CATEGORY_INDEX["python"]] / cpu:.0f}%</text>\n')

    out.append("</svg>\n")
    with open(svg_path, "w") as f:
        f.write("".join(out))


def main():
    args = parse_arguments()
    if not os.path.isfile(args.perf_data):
        raise FileNotFoundError(f"perf data not found: {args.perf_data}")
    flamegraph_pl = os.path.join(args.flamegraph_dir, "flamegraph.pl")
    if not os.access(flamegraph_pl, os.X_OK):
        raise FileNotFoundError(f"flamegraph.pl not found or not executable: {flamegraph_pl}")
    if args.freq <= 0:
        raise ValueError(f"--freq must be positive, got {args.freq}")

    stem = re.sub(r"\.perf\.data$", "", args.perf_data)
    outdir = args.outdir or f"{stem}.flamechart"
    os.makedirs(outdir, exist_ok=True)

    print(f"Reading samples from {args.perf_data} ...", flush=True)
    threads = build_thread_timelines(read_samples(args.perf_data), 1e6 / args.freq)
    if not threads:
        raise RuntimeError(f"No samples found in {args.perf_data}")
    run_start = min(th["start"] for th in threads.values())

    rows = []
    for th in sorted(threads.values(), key=lambda x: (x["start"], x["tid"])):
        duration = th["last"] - th["start"] + 1e6 / args.freq
        oncpu_sec = th["oncpu"] / 1e6
        if oncpu_sec < args.min_oncpu_sec:
            continue
        comm_safe = re.sub(r"[^A-Za-z0-9_.-]", "_", th["comm"])
        name = f"{th['pid']}_{th['tid']}_{comm_safe}"
        root = f"{th['comm']} pid {th['pid']} tid {th['tid']}"
        folded_path = os.path.join(outdir, f"{name}.folded")
        with open(folded_path, "w") as f:
            for stack, weight_us in th["segments"]:
                f.write(f"{root};{stack} {int(round(weight_us))}\n")

        offset = (th["start"] - run_start) / 1e6
        title = (f"Flame chart: {root} | starts at +{offset:.0f} s, lasts {duration / 1e6:.0f} s, "
                 f"on-CPU {oncpu_sec:.0f} s")
        svg_path = os.path.join(outdir, f"{name}.svg")
        with open(svg_path, "w") as svg:
            subprocess.run([flamegraph_pl, "--flamechart", "--title", title, "--countname", "us",
                            "--width", str(args.width), folded_path], stdout=svg, check=True)
        rows.append((name, offset, duration / 1e6, oncpu_sec))
        print(f"  {svg_path}")

    if not rows:
        raise RuntimeError(f"No thread has more than {args.min_oncpu_sec} s on-CPU time")
    index_path = os.path.join(outdir, "index.tsv")
    with open(index_path, "w") as f:
        f.write("thread\tstart_offset_s\tduration_s\toncpu_s\toncpu_fraction\n")
        for name, offset, duration, oncpu in rows:
            f.write(f"{name}\t{offset:.1f}\t{duration:.1f}\t{oncpu:.1f}\t{oncpu / duration:.3f}\n")
    print(f"Wrote {len(rows)} flame charts ({len(threads) - len(rows)} short threads skipped): {index_path}")

    # Combined view: main thread (tid == pid) of every kept process on one time axis
    kept_names = {name for name, _, _, _ in rows}
    main_threads = [th for th in threads.values()
                    if th["tid"] == th["pid"]
                    and f"{th['pid']}_{th['tid']}_{re.sub(r'[^A-Za-z0-9_.-]', '_', th['comm'])}" in kept_names]
    if not main_threads:
        raise RuntimeError("No process main thread passed --min-oncpu-sec; cannot build the timeline")
    timeline_path = os.path.join(outdir, "timeline.svg")
    render_timeline(main_threads, timeline_path, f"CPU activity timeline: {os.path.basename(stem)}")
    print(f"Wrote combined timeline: {timeline_path}")


if __name__ == "__main__":
    sys.exit(main())
