#!/usr/bin/env python3
"""Promote the MC-vs-interpolation template closure figures.

The per-category plots are written into the gitignored template tree by
plotTemplateClosure.py; this lifts them into the tracked results tree, the
same rule collectPnetScorePlots.py applies to the LR panels:

    templates/{mp}/{method}/interp-signal/{era}/{channel}/closure/
        closure.{channel}_{era}.{png,pdf}
      ->
    results/plots/closure/{method}/{masspoint}/closure.{channel}_{era}.{png,pdf}

Scope is the mass points that have MC -- configs/masspoints.json 'baseline'
(78) and 'particlenet' (17) -- times {Run2, Run3} x {SR1E2Mu, SR3Mu}.
Group members are resolved through their seed, as everywhere else. Only the
figures are promoted: the per-category JSON stays in the template tree.

The script exits 1 listing every missing source, so a partial campaign
cannot pass silently.

  python3 python/collectTemplateClosure.py
  python3 python/collectTemplateClosure.py --point Baseline:MHc130_MA90
"""
import argparse
import os
import shutil

import interpolation_config
import run_period_utils
import srspaths

METHODS = {"Baseline": "baseline", "ParticleNet": "particlenet"}
ERAS = ["Run2", "Run3"]
CHANNELS = ["SR1E2Mu", "SR3Mu"]
SUFFIXES = ["png", "pdf"]


class Collector:
    """Copies with a running tally of what was found and what was not."""

    def __init__(self):
        self.copied = 0
        self.missing = []

    def copy(self, src, outdir, name):
        if not os.path.exists(src):
            self.missing.append(src)
            return False
        os.makedirs(outdir, exist_ok=True)
        shutil.copy2(src, os.path.join(outdir, name))
        self.copied += 1
        return True


def default_points():
    """The MC mass points, per arm, from configs/masspoints.json."""
    config = srspaths.masspoints_config()
    return {method: list(config[key]) for method, key in METHODS.items()}


def leaf_dir(masspoint, method, era, channel):
    seed = interpolation_config.group_seed(masspoint, method)
    if seed == masspoint:
        return srspaths.template_dir(masspoint, method, era, channel,
                                     source="interp-signal")
    return srspaths.interp_member_dir(seed, masspoint, era, channel,
                                      method=method)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--point", action="append", metavar="METHOD:MASSPOINT",
                        help="override the mass-point lists; repeatable")
    parser.add_argument("--eras", nargs="+", default=ERAS)
    parser.add_argument("--channels", nargs="+", default=CHANNELS)
    parser.add_argument("--output-dir",
                        default=os.path.join(srspaths.module_dir(),
                                             "results", "plots", "closure"))
    args = parser.parse_args()

    if args.point:
        points = {}
        for spec in args.point:
            if ":" not in spec:
                raise ValueError(f"--point expects METHOD:MASSPOINT, "
                                 f"got {spec!r}")
            method, masspoint = spec.split(":", 1)
            if method not in METHODS:
                raise ValueError(f"unknown method {method!r}")
            points.setdefault(method, []).append(masspoint)
    else:
        points = default_points()

    coll = Collector()
    for method in sorted(points):
        for masspoint in points[method]:
            outdir = os.path.join(args.output_dir, method, masspoint)
            for era in args.eras:
                for channel in args.channels:
                    cat = run_period_utils.category_name(channel, era)
                    base = os.path.join(
                        leaf_dir(masspoint, method, era, channel),
                        "closure", f"closure.{cat}")
                    for suffix in SUFFIXES:
                        coll.copy(f"{base}.{suffix}", outdir,
                                  f"closure.{cat}.{suffix}")

    n_points = sum(len(v) for v in points.values())
    print(f"Copied {coll.copied} files from {n_points} mass points "
          f"-> {args.output_dir}")
    if coll.missing:
        print(f"{len(coll.missing)} missing sources:")
        for src in coll.missing:
            print(f"  {src}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
