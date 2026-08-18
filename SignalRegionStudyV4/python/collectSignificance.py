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
    python3 python/collectSignificance.py --grid            # the whole scan

`--grid`/`--pnet-grid` collect the full scan an `automize/significance.sh
--grid` run produced.  That is the look-elsewhere input — the trials
estimate counts upcrossings of the observed Z(mA) curve, so it needs every
scan point (see docs/LEE.md) — and it is why this script reports
parsed/total and exits non-zero on a partial collection: a short DAG must
not turn into a silently truncated curve.
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


def grid_points(method, mhc_filter=None):
    """Every scan point of an arm, from the frozen grid config."""
    cfg = (srspaths.grid_config() if method == "Baseline"
           else srspaths.pnet_grid_config())
    wanted = set(mhc_filter or ())
    points = []
    for key in sorted(cfg["grids"], key=lambda k: int(k[3:])):
        mhc = int(key[3:])
        if wanted and mhc not in wanted:
            continue
        for grp in cfg["grids"][key]["groups"]:
            for ma in grp["members"]:
                points.append(f"{method}:"
                              f"{interpolation_config.masspoint_name(ma, mhc)}")
    return points


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--point", action="append", metavar="METHOD:MASSPOINT",
                        help="point to collect; repeatable")
    parser.add_argument("--template-points", action="store_true",
                        help="add the curated template-point set")
    parser.add_argument("--grid", action="store_true",
                        help="add every Baseline scan point (LEE input)")
    parser.add_argument("--pnet-grid", action="store_true",
                        help="add every ParticleNet scan point (LEE input)")
    parser.add_argument("--mhc", type=int, nargs="+", default=None,
                        help="restrict --grid/--pnet-grid to these mHc")
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
    if args.grid:
        specs += grid_points("Baseline", args.mhc)
    if args.pnet_grid:
        specs += grid_points("ParticleNet", args.mhc)
    # A point can be named twice (a template point is also a grid point);
    # collect it once, keeping the order for a readable missing-list.
    specs = list(dict.fromkeys(specs))
    if not specs:
        raise ValueError("no points: use --template-points, --point, "
                         "--grid and/or --pnet-grid")

    ROOT.gROOT.SetBatch(True)
    payload, missing = {}, []
    parsed = 0
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
            parsed += 1

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
    requested = len(specs) * len(args.channels)
    print(f"Parsed {parsed}/{requested} (point, channel) significances; "
          f"file now holds {n} -> {out}")
    if missing:
        # On a scan collect this list can be thousands long; the count is
        # the signal, the first entries are the diagnosis.
        print(f"{len(missing)} missing sources:")
        for path in missing[:20]:
            print(f"  {path}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
