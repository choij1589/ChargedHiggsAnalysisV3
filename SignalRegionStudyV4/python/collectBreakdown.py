#!/usr/bin/env python3
"""Collect the grouped uncertainty breakdown into a single JSON.

`plot1DScan.py` renders the components but keeps the numbers only as text
inside a TPaveText, and on a negative quadrature subtraction it silently
substitutes 0.  This collector recomputes them from the scan ROOTs so the
analysis has a git-tracked source of record, using the SAME spline and
crossing finder the plot uses (HiggsAnalysis.CombinedLimit.util.plotting)
so the JSON and the PDF agree by construction rather than by coincidence.

A negative subtraction is recorded as null, not zero, and reported -- see
docs/BREAKDOWN.md.

Usage:
  python3 python/collectBreakdown.py --template-points
  python3 python/collectBreakdown.py --point Baseline:MHc130_MA90
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import srspaths
import interpolation_config
import nuisanceGroups

import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True
try:
    import HiggsAnalysis.CombinedLimit.util.plotting as plot
except ImportError as exc:  # pragma: no cover - environment guard
    raise RuntimeError(
        "HiggsAnalysis.CombinedLimit.util.plotting is not importable; "
        "source setup.sh (CMSSW_14_1_0_pre4 + Combine) first") from exc

from functools import partial

Y_CUT = 100.0
CROSSING = 1.0  # 2*deltaNLL = 1 <=> the 1 sigma interval


def breakdown_dir(masspoint, method, era, channel, source):
    base = srspaths.template_dir(masspoint, method, era, channel,
                                 source=source)
    if source == "interp-signal":
        seed = interpolation_config.group_seed(masspoint, method)
        if seed != masspoint:
            base = srspaths.interp_member_dir(seed, masspoint, era, channel,
                                              method=method)
    return os.path.join(base, "combine_output", "breakdown")


def _eval(obj, x, params):
    return obj.Eval(x[0])


_NAMECOUNTER = [0]


def scan_interval(path):
    """(best_fit, +hi, -lo) at 2*deltaNLL = 1, or None if unusable.

    Mirrors plot1DScan.BuildScan: same graph selection, same TSpline3,
    same FindCrossingsWithSpline, same "interval containing the best fit"
    rule.
    """
    if not plot.TFileIsGood(path):
        return None
    limit = plot.MakeTChain([path], "limit")
    graph = plot.TGraphFromTree(limit, "r", "2*deltaNLL",
                               "quantileExpected > -1.5")
    graph.Sort()
    plot.RemoveGraphXDuplicates(graph)
    plot.RemoveGraphYAbove(graph, Y_CUT)
    if graph.GetN() <= 1:
        return None
    bestfit = None
    for i in range(graph.GetN()):
        if graph.GetY()[i] == 0.0:
            bestfit = graph.GetX()[i]
    if bestfit is None:
        return None
    spline = ROOT.TSpline3("spline3", graph)
    method = partial(_eval, spline)
    func = ROOT.TF1(f"splinefn{_NAMECOUNTER[0]}", method,
                    graph.GetX()[0], graph.GetX()[graph.GetN() - 1], 1)
    func._method = method
    _NAMECOUNTER[0] += 1
    for cr in plot.FindCrossingsWithSpline(graph, func, CROSSING):
        if cr["lo"] <= bestfit and cr["hi"] >= bestfit:
            if not (cr["valid_lo"] and cr["valid_hi"]):
                # The scan window clipped the interval: the sigma would be
                # a lower bound, not a measurement.
                return None
            return (bestfit, cr["hi"] - bestfit, cr["lo"] - bestfit)
    return None


def subtract(outer, inner):
    """Quadrature subtraction, mirroring plot1DScan.py.

    Returns None where the subtraction is negative instead of zeroing it:
    that means the freeze did not shrink the interval (minimizer noise, or
    a scan too coarse to resolve the difference) and the component is not
    measured, which is a different statement from "this component is
    negligible".
    """
    if abs(inner) > abs(outer):
        return None
    return math.sqrt(outer * outer - inner * inner)


def collect_point(masspoint, method, era, channel, source):
    """The breakdown of one (point, channel), or (None, reason)."""
    bdir = breakdown_dir(masspoint, method, era, channel, source)
    groups_file = os.path.join(bdir, "freeze_groups.txt")
    if not os.path.exists(groups_file):
        return None, groups_file
    with open(groups_file) as f:
        groups = [line.strip() for line in f if line.strip()]
    if not groups:
        return None, f"{groups_file}::empty"

    method_segment = srspaths.method_segment(method)
    tag = f"{masspoint}.{method_segment}"
    files = [(("total"), os.path.join(
        bdir, f"higgsCombine.{tag}.total.MultiDimFit.mH120.root"))]
    cumulative = []
    for group in groups:
        cumulative.append(group)
        cum_tag = "_".join(cumulative)
        files.append((f"freeze_{cum_tag}", os.path.join(
            bdir,
            f"higgsCombine.{tag}.freeze_{cum_tag}.MultiDimFit.mH120.root")))

    intervals = {}
    for name, path in files:
        if not os.path.exists(path):
            return None, path
        iv = scan_interval(path)
        if iv is None:
            return None, f"{path}::unusable scan"
        intervals[name] = iv

    order = [name for name, _ in files]
    best = intervals["total"][0]
    entry = {
        "best_fit": best,
        "total": {"up": intervals["total"][1], "dn": intervals["total"][2]},
        "cumulative": {name: {"up": intervals[name][1],
                              "dn": intervals[name][2]} for name in order},
    }
    negatives = []
    for i, group in enumerate(groups):
        outer, inner = intervals[order[i]], intervals[order[i + 1]]
        up = subtract(outer[1], inner[1])
        dn = subtract(outer[2], inner[2])
        if up is None or dn is None:
            negatives.append(group)
        entry[group] = {"up": up, "dn": None if dn is None else -dn}
    # Whatever still floats after the last freeze: data statistics plus
    # the autoMCStats prop_bin parameters, which are never grouped.
    residual = nuisanceGroups.load_config()["residual"]["name"]
    last = intervals[order[-1]]
    entry[residual] = {"up": last[1], "dn": last[2]}
    if negatives:
        entry["negative_subtraction"] = negatives
    return entry, None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--point", action="append", default=[],
                        metavar="METHOD:MASSPOINT",
                        help="repeatable, e.g. Baseline:MHc130_MA90")
    parser.add_argument("--template-points", action="store_true",
                        help="the curated bundle set "
                             "(collectTemplatePlots.DEFAULT_POINTS)")
    parser.add_argument("--era", default="All")
    parser.add_argument("--channels", nargs="+", default=["Combined"])
    parser.add_argument("--signal-source", default="interp-signal",
                        choices=list(srspaths.SIGNAL_SOURCES))
    parser.add_argument("--output")
    args = parser.parse_args()

    specs = []
    if args.template_points:
        from collectTemplatePlots import DEFAULT_POINTS
        for method in sorted(DEFAULT_POINTS):
            for mp in DEFAULT_POINTS[method]:
                specs.append((method, mp))
    for spec in args.point:
        method, _, mp = spec.partition(":")
        if method not in ("Baseline", "ParticleNet") or not mp:
            raise SystemExit(f"ERROR: bad --point spec {spec!r}")
        specs.append((method, mp))
    specs = list(dict.fromkeys(specs))
    if not specs:
        raise SystemExit("ERROR: no points. Use --template-points or "
                         "--point METHOD:MASSPOINT.")

    payload = {}
    missing = []
    negative = []
    parsed = total = 0
    for method, mp in specs:
        for channel in args.channels:
            total += 1
            entry, why = collect_point(mp, method, args.era, channel,
                                       args.signal_source)
            if entry is None:
                missing.append(why)
                continue
            payload.setdefault(method, {}).setdefault(mp, {})[channel] = entry
            if entry.get("negative_subtraction"):
                negative.append(f"{method}/{mp}/{channel}: "
                                f"{', '.join(entry['negative_subtraction'])}")
            parsed += 1

    source_infix = ("" if args.signal_source == "mc-signal"
                    else f".{args.signal_source}")
    out_path = args.output or os.path.join(
        srspaths.module_dir(), "results", "json",
        f"breakdown.{args.era}{source_infix}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Merge rather than replace, so a later run on other points adds to
    # the record instead of truncating it.
    merged = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            merged = json.load(f)
    for method, points in payload.items():
        for mp, channels in points.items():
            merged.setdefault(method, {}).setdefault(mp, {}).update(channels)
    with open(out_path, "w") as f:
        json.dump(merged, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Parsed {parsed}/{total} (point, channel) -> {out_path}")
    if negative:
        print(f"WARNING: negative quadrature subtraction at {len(negative)} "
              "(point, channel); those components are null, not zero:")
        for line in negative:
            print(f"  {line}")
    if missing:
        print(f"Missing {len(missing)} source(s):")
        for line in missing[:20]:
            print(f"  {line}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
