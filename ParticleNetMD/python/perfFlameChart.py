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

Run it on the node that recorded the data if the Python frames matter: perf script
resolves them from /tmp/perf-<pid>.map, which only exists there.
"""

import argparse
import os
import re
import subprocess
import sys
from collections import defaultdict

OFF_CPU = "[off-CPU]"
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
                                 "last": t_us, "last_stack": None, "segments": [], "oncpu": 0.0}
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


if __name__ == "__main__":
    sys.exit(main())
