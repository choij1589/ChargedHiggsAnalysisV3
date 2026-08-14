#!/usr/bin/env python3
"""Collect the ParticleNet interp-signal score plots into the tracked tree.

Per-seed score plots live in the gitignored template dirs; following V3's
collector rule, only the production discriminant panel (LR_modified) is
promoted, one per (seed, era, channel) target plus the TTZ2E1Mu control
region emitted by the per-channel jobs:

    templates/{seed}/ParticleNet/interp-signal/{era}/{channel}/scores/
        {region}/LR_modified.png
 -> results/plots/scores/{seed}/LR_modified_{era}_{channel}[_TTZ2E1Mu].png

  python3 python/collectPnetScorePlots.py [--eras Run2 Run3 All]
"""
import argparse
import os
import shutil

import pnet_interp_config as pic
import srspaths

ERAS = ["Run2", "Run3", "All"]
CHANNELS = ["SR1E2Mu", "SR3Mu", "Combined"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eras", nargs="+", default=ERAS)
    parser.add_argument("--output-dir",
                        default=os.path.join(srspaths.module_dir(),
                                             "results", "plots", "scores"))
    args = parser.parse_args()

    n_copied, missing = 0, []
    for mhc in pic.pn_mhc_list():
        grids = srspaths.pnet_grid_config()["grids"][mhc]
        for grp in grids["groups"]:
            import interpolation_config
            seed = interpolation_config.masspoint_name(
                grp["seed"], pic.mhc_int(mhc))
            outdir = os.path.join(args.output_dir, seed)
            for era in args.eras:
                for channel in CHANNELS:
                    tdir = srspaths.template_dir(
                        seed, "ParticleNet", era, channel,
                        source="interp-signal")
                    regions = [(channel, f"LR_modified_{era}_{channel}.png")]
                    if channel != "Combined":
                        regions.append(
                            ("TTZ2E1Mu",
                             f"LR_modified_{era}_{channel}_TTZ2E1Mu.png"))
                    for region, outname in regions:
                        src = os.path.join(tdir, "scores", region,
                                           "LR_modified.png")
                        if not os.path.exists(src):
                            missing.append(src)
                            continue
                        os.makedirs(outdir, exist_ok=True)
                        shutil.copy2(src, os.path.join(outdir, outname))
                        n_copied += 1

    print(f"Copied {n_copied} plots -> {args.output_dir}")
    if missing:
        print(f"{len(missing)} missing sources; first: {missing[0]}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
