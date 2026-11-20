#!/usr/bin/env python3
"""
plotStepProfile.py — distribution of the DataLoader fetch ("collate") time per training batch.

Reads the per-batch CSV written by trainMultiClass.py in profile mode
(<output>/<model_name>_step_times.csv, from StepProfiler.save_csv) and draws, with ROOT + cmsstyle:

    step_collate_hist.png        distribution of the fetch time and its split into collate_fn / fetch_other
    step_collate_vs_batch.png    fetch time against the batch number over all epochs (GC batches highlighted)
    step_collate_vs_nodes.png    fetch time against the number of nodes in the batch
    step_collate_per_epoch.png   mean / median / max of the fetch time per epoch

and prints how much of the total fetch time the slow batches carry and where they are
(first epoch, GC during the fetch, collate_fn or pin_memory side).

Usage:
    python python/plotStepProfile.py --signal MHc130_MA90 --channel Combined
    python python/plotStepProfile.py --csv profile/Combined/MHc130_MA90/pilot/<model_name>_step_times.csv
"""

import argparse
import csv
import math
import os
import sys
from array import array
from pathlib import Path

import numpy as np

import ROOT
ROOT.gROOT.SetBatch(True)

# Common/Tools is added to PYTHONPATH by setup.sh
import cmsstyle as CMS

PARTICLENETMD_DIR = Path(__file__).resolve().parents[1]
COLORS = {"black": "#000000", "blue": "#0099b4", "red": "#d81b60", "grey": "#9c9ca1"}
SLOW_FACTOR = 5.0  # a batch is "slow" if its fetch takes more than this times the median


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the per-batch fetch (collate) time distribution")
    parser.add_argument("--csv", default=None, type=str,
                        help="Per-batch CSV (<model_name>_step_times.csv); default: look it up from "
                             "--results-dir/--channel/--signal/--leaf")
    parser.add_argument("--signal", default="MHc130_MA90", type=str)
    parser.add_argument("--channel", default="Combined", type=str)
    parser.add_argument("--results-dir", default="profile", type=str,
                        help="Directory below ParticleNetMD/ that the training wrote to (default: profile)")
    parser.add_argument("--leaf", default="pilot", type=str,
                        help="Last path component of the run (pilot or fold-N, default: pilot)")
    parser.add_argument("--output-dir", default=None, type=str,
                        help="Where to write the plots (default: <csv directory>/plots)")
    return parser.parse_args()


def find_csv(args: argparse.Namespace) -> Path:
    if args.csv:
        path = Path(args.csv)
        if not path.is_file():
            raise FileNotFoundError(f"CSV not found: {path}")
        return path
    run_dir = PARTICLENETMD_DIR / args.results_dir / args.channel / args.signal / args.leaf
    matches = sorted(run_dir.glob("*_step_times.csv"))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one *_step_times.csv in {run_dir}, found {len(matches)}: "
                         f"{[m.name for m in matches]}; pass --csv")
    return matches[0]


def load_rows(path: Path) -> dict:
    """CSV columns as numpy arrays (empty fields -> nan)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        columns = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name, value in row.items():
                columns[name].append(float(value) if value != "" else math.nan)
    data = {name: np.array(values) for name, values in columns.items()}
    if not len(data["epoch"]):
        raise ValueError(f"{path} has no rows")
    return data


def configure_cms_style() -> None:
    CMS.setCMSStyle()
    # Not an approved result: the CMS logo is replaced by the private-work label
    CMS.SetExtraText("")
    CMS.SetCmsText("Private work (CMS simulation)", font=52, size=0.75 * 0.76)
    CMS.SetLumi(None, run="")
    CMS.SetEnergy(0, unit="train batches, 512 events")


def color(name: str) -> int:
    return ROOT.TColor.GetColor(COLORS[name])


def make_canvas(xmin, xmax, ymin, ymax, xtitle, ytitle, logx=False, logy=True):
    canvas = CMS.cmsCanvas("", xmin, xmax, ymin, ymax, xtitle, ytitle,
                           square=False, iPos=11, extraSpace=0.0)
    canvas.SetLogx(logx)
    canvas.SetLogy(logy)
    return canvas


def make_graph(x, y, marker_color: str, marker_style: int = 20, marker_size: float = 0.5):
    if len(x) == 0:
        raise ValueError("make_graph: no points (PyROOT cannot convert an empty buffer)")
    graph = ROOT.TGraph(len(x), array("d", x), array("d", y))
    graph.SetMarkerColor(color(marker_color))
    graph.SetLineColor(color(marker_color))
    graph.SetMarkerStyle(marker_style)
    graph.SetMarkerSize(marker_size)
    return graph


def make_legend(entries, x1=0.60, y1=0.62, x2=0.93, y2=0.86):
    legend = ROOT.TLegend(x1, y1, x2, y2)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.04)
    for obj, label, option in entries:
        legend.AddEntry(obj, label, option)
    legend.Draw()
    return legend


def draw_text(lines, x=0.22, y=0.75, size=0.036):
    keep = []
    for i, text in enumerate(lines):
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(size)
        latex.SetTextFont(42)
        latex.DrawLatex(x, y - i * 1.4 * size, text)
        keep.append(latex)
    return keep


def save(canvas, keepalive, path: Path) -> None:
    canvas._keepalive = keepalive
    os.makedirs(path.parent, exist_ok=True)
    canvas.SaveAs(str(path))
    canvas.Close()
    print(f"wrote {path}")


def plot_hist(data, used, out: Path) -> None:
    series = [("fetch total", data["collate_ms"][used], "black", 1),
              ("collate_fn", data["collate_fn_ms"][used], "blue", 2),
              ("fetch other (pin_memory, ...)", data["fetch_other_ms"][used], "red", 3)]
    positive = np.concatenate([v[v > 0] for _, v, _, _ in series])
    edges = np.logspace(np.log10(positive.min() * 0.9), np.log10(positive.max() * 1.1), 41)
    hists = []
    for name, values, col, _style in series:
        hist = ROOT.TH1D(f"h_{len(hists)}", "", len(edges) - 1, array("d", edges))
        for v in values:
            hist.Fill(v)
        hist.SetDirectory(0)
        hists.append(hist)
    ymax = max(h.GetMaximum() for h in hists)
    canvas = make_canvas(edges[0], edges[-1], 0.5, ymax * 30, "Fetch time per batch [ms]",
                         "Batches", logx=True)
    keepalive = list(hists)
    for hist, (_n, _v, col, style) in zip(hists, series):
        CMS.cmsObjectDraw(hist, "hist", LineColor=color(col), LineWidth=3, LineStyle=style)
    total = data["collate_ms"][used]
    keepalive.append(make_legend([(h, s[0], "l") for h, s in zip(hists, series)]))
    keepalive += draw_text([f"N = {len(total)}, mean {total.mean():.1f} ms, median {np.median(total):.1f} ms",
                            f"std {total.std(ddof=1):.1f} ms, max {total.max():.0f} ms"])
    save(canvas, keepalive, out / "step_collate_hist.png")


def plot_vs_batch(data, out: Path) -> None:
    index = np.arange(len(data["epoch"]), dtype=float)
    ms = data["collate_ms"]
    # gen-0 collections run during practically every fetch; only a full (gen 2)
    # collection, which walks all live objects, is a candidate for a slow batch
    gc_hit = data["gc_gen2"] > 0
    canvas = make_canvas(-0.02 * len(index), 1.02 * len(index), ms.min() * 0.7, ms.max() * 4,
                         "Training batch (all epochs)", "Fetch time [ms]")
    keepalive = []
    boundaries = index[np.r_[False, np.diff(data["epoch"]) != 0]]
    for x in boundaries:
        line = ROOT.TLine(x, ms.min() * 0.7, x, ms.max() * 4)
        line.SetLineStyle(2)
        line.SetLineColor(color("grey"))
        line.Draw()
        keepalive.append(line)
    entries = []
    if (~gc_hit).any():
        normal = make_graph(index[~gc_hit], ms[~gc_hit], "black")
        normal.Draw("P SAME")
        keepalive.append(normal)
        entries.append((normal, "no full GC during fetch", "p"))
    if gc_hit.any():
        hit = make_graph(index[gc_hit], ms[gc_hit], "red", marker_style=29, marker_size=1.6)
        hit.Draw("P SAME")
        keepalive.append(hit)
        entries.append((hit, "full (gen 2) GC during fetch", "p"))
    keepalive.append(make_legend(entries, 0.60, 0.74, 0.93, 0.86))
    keepalive += draw_text(["dashed: epoch boundaries (all batches, incl. warm-up)"], y=0.70)
    save(canvas, keepalive, out / "step_collate_vs_batch.png")


def plot_vs_nodes(data, out: Path) -> None:
    nodes, ms = data["num_nodes"], data["collate_ms"]
    span = nodes.max() - nodes.min()
    canvas = make_canvas(nodes.min() - 0.05 * span, nodes.max() + 0.05 * span,
                         ms.min() * 0.7, ms.max() * 4, "Nodes in batch", "Fetch time [ms]")
    graph = make_graph(nodes, ms, "black")
    graph.Draw("P SAME")
    save(canvas, [graph], out / "step_collate_vs_nodes.png")


def plot_per_epoch(data, used, out: Path) -> None:
    epochs = np.unique(data["epoch"][used])
    ms = data["collate_ms"]
    stats = {"mean": [], "median": [], "max": []}
    for epoch in epochs:
        values = ms[used & (data["epoch"] == epoch)]
        stats["mean"].append(values.mean())
        stats["median"].append(np.median(values))
        stats["max"].append(values.max())
    top = max(stats["max"])
    bottom = min(stats["median"])
    canvas = make_canvas(epochs.min() - 0.5, epochs.max() + 0.5, bottom * 0.5, top * 4,
                         "Epoch", "Fetch time per batch [ms]")
    keepalive, entries = [], []
    for key, col, style in (("max", "red", 22), ("mean", "black", 20), ("median", "blue", 21)):
        graph = make_graph(epochs, stats[key], col, marker_style=style, marker_size=1.2)
        graph.Draw("PL SAME")
        keepalive.append(graph)
        entries.append((graph, key, "pl"))
    keepalive.append(make_legend(entries, 0.60, 0.70, 0.93, 0.86))
    save(canvas, keepalive, out / "step_collate_per_epoch.png")


def print_diagnosis(data, used) -> None:
    ms = data["collate_ms"][used]
    median = np.median(ms)
    slow = ms > SLOW_FACTOR * median
    total = ms.sum()
    print(f"\nfetch time over {len(ms)} used batches: mean {ms.mean():.2f} ms, median {median:.2f} ms, "
          f"max {ms.max():.1f} ms")
    print(f"slowest 1% of the batches carry {100 * np.sort(ms)[::-1][:max(1, len(ms) // 100)].sum() / total:.1f}% "
          f"of the total fetch time")
    print(f"slow batches (> {SLOW_FACTOR:g} x median = {SLOW_FACTOR * median:.1f} ms): {slow.sum()} "
          f"({100 * slow.mean():.1f}% of batches) carry {100 * ms[slow].sum() / total:.1f}% of the total")
    if not slow.any():
        return
    epoch = data["epoch"][used][slow]
    with_gc = data["gc_gen2"][used][slow] > 0
    in_fn = data["collate_fn_ms"][used][slow] > data["fetch_other_ms"][used][slow]
    print(f"  of the slow batches: in epoch 0: {(epoch == 0).sum()}, with a full (gen 2) GC during the fetch: "
          f"{with_gc.sum()}, dominated by collate_fn: {in_fn.sum()}, dominated by fetch_other: {(~in_fn).sum()}")
    print(f"  first batch of an epoch: {(data['batch'][used][slow] == 0).sum()}; slow batches per epoch: "
          + ", ".join(f"{int(e)}:{int((epoch == e).sum())}" for e in np.unique(epoch)))


def print_per_epoch(data, used) -> None:
    """Median and max fetch time per epoch, with the collate_fn / fetch_other split."""
    print("\nper epoch (used batches):  median fetch / collate_fn / fetch_other [ms], max fetch [ms], "
          "full GCs, GC time [ms]")
    for epoch in np.unique(data["epoch"][used]):
        m = used & (data["epoch"] == epoch)
        print(f"  epoch {int(epoch):>2}: {np.median(data['collate_ms'][m]):7.1f} / "
              f"{np.median(data['collate_fn_ms'][m]):6.1f} / {np.median(data['fetch_other_ms'][m]):6.1f}, "
              f"max {data['collate_ms'][m].max():8.1f}, full GCs {int(data['gc_gen2'][m].sum())}, "
              f"GC {data['gc_ms'][m].sum():8.1f}")


def main() -> None:
    args = parse_arguments()
    csv_path = find_csv(args)
    data = load_rows(csv_path)
    used = data["used"] == 1
    if not used.any():
        raise ValueError(f"{csv_path}: no row has used=1")
    out = Path(args.output_dir) if args.output_dir else csv_path.parent / "plots"

    print(f"reading {csv_path}")
    configure_cms_style()
    plot_hist(data, used, out)
    plot_vs_batch(data, out)
    plot_vs_nodes(data, out)
    plot_per_epoch(data, used, out)
    print_diagnosis(data, used)
    print_per_epoch(data, used)


if __name__ == "__main__":
    main()
