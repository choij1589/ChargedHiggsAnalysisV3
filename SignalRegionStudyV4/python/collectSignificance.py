#!/usr/bin/env python3
"""Collect the observed local significances into one JSON.

Mirrors collectLimits.py for the Significance method: reads each point's
Combine output and writes a single keyed file that everything downstream
(the template-point bundles, any summary table) reads instead of
re-opening ROOT.

    templates/{seed}/{method}/{source}/{era}/{channel}/combine_output/
        significance/higgsCombine.{mp}.{method}.Significance.mH120.root
 -> results/json/significance.{era}[.{source}].json

The stored `Z` is the UNCAPPED significance runSignificance.sh computes,
so a deficit keeps its sign instead of being floored at zero; `p` is the
one-sided tail 0.5*erfc(Z/sqrt2), which for Z < 0 is correspondingly
above 0.5.  Both are LOCAL: no look-elsewhere correction is applied, and
on this scan the trials factor is large.

    python3 python/collectSignificance.py --template-points
    python3 python/collectSignificance.py --point Baseline:MHc160_MA17p5
"""
import argparse
import json
import math
import os

import ROOT

import interpolation_config
import srspaths

CHANNELS = ["Combined", "SR1E2Mu", "SR3Mu"]


def significance_root(masspoint, method, era, channel, source):
    """Path of the Significance output, nesting members under their seed."""
    base = srspaths.template_dir(masspoint, method, era, channel,
                                 source=source)
    if source == "interp-signal":
        seed = interpolation_config.group_seed(masspoint, method)
        if seed != masspoint:
            base = srspaths.interp_member_dir(seed, masspoint, era, channel,
                                              method=method)
    return os.path.join(base, "combine_output", "significance",
                        f"higgsCombine.{masspoint}.{method}"
                        f".Significance.mH120.root")


def read_significance(path):
    """Uncapped Z from the single-entry `limit` tree."""
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise RuntimeError(f"cannot open {path}")
    try:
        tree = f.Get("limit")
        if not tree or not tree.GetEntries():
            raise RuntimeError(f"no 'limit' entries in {path}")
        tree.GetEntry(0)
        return float(tree.limit)
    finally:
        f.Close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--point", action="append", metavar="METHOD:MASSPOINT",
                        help="point to collect; repeatable")
    parser.add_argument("--template-points", action="store_true",
                        help="add the curated template-point set")
    parser.add_argument("--era", default="All")
    parser.add_argument("--channels", nargs="+", default=CHANNELS)
    parser.add_argument("--signal-source", default="interp-signal",
                        choices=["interp-signal", "mc-signal"])
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    specs = list(args.point or ())
    if args.template_points:
        from collectTemplatePlots import DEFAULT_POINTS
        specs += [f"{method}:{mp}" for method in sorted(DEFAULT_POINTS)
                  for mp in DEFAULT_POINTS[method]]
    if not specs:
        raise ValueError("no points: use --template-points and/or --point")

    ROOT.gROOT.SetBatch(True)
    payload, missing = {}, []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"--point expects METHOD:MASSPOINT, got {spec!r}")
        method, masspoint = spec.split(":", 1)
        for channel in args.channels:
            path = significance_root(masspoint, method, args.era, channel,
                                     args.signal_source)
            if not os.path.exists(path):
                missing.append(path)
                continue
            z = read_significance(path)
            entry = payload.setdefault(method, {}).setdefault(masspoint, {})
            entry[channel] = {"Z": z, "p": 0.5 * math.erfc(z / math.sqrt(2.0))}

    source_infix = ("" if args.signal_source == "mc-signal"
                    else f".{args.signal_source}")
    out = args.output or os.path.join(
        srspaths.module_dir(), "results", "json",
        f"significance.{args.era}{source_infix}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # Merge into any existing file: the point list is ad hoc by design, so
    # a later run on different points must not drop the earlier ones.
    if os.path.exists(out):
        with open(out) as f:
            merged = json.load(f)
        for method, points in payload.items():
            merged.setdefault(method, {}).update(points)
        payload = merged
    with open(out, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    n = sum(len(chs) for pts in payload.values() for chs in pts.values())
    print(f"Wrote {n} (point, channel) significances -> {out}")
    if missing:
        print(f"{len(missing)} missing sources:")
        for path in missing:
            print(f"  {path}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
