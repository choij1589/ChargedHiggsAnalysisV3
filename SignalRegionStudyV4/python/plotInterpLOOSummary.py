#!/usr/bin/env python3
"""Summary plots of the leave-one-out yield closures: overlay every
point's LOO-predicted window yield on the measured MC yields, and the
matching relative-residual series — the visual counterpart of the
loo_uncertainties.json norm tables.

Reads the adopted yields.json (measured) and the 78 per-point
tests/interpolation/MHc{X}_MA{Y}/yields/yield_closure.json (predictions);
JSON-only, no sample access. Outputs into
tests/interpolation/MHc{X}/plots/yields/:

  loo_grid.{channel}.{era}.png        measured vs LOO-predicted N_window
  loo_residuals.{channel}.{period}.png  LOO (pred-meas)/meas vs mA per era

  python3 plotInterpLOOSummary.py --mhc 130
  python3 plotInterpLOOSummary.py --all
"""
import argparse
import json
import os

import interp_plot_utils
import interpolation_config
import srspaths
from interpolation_config import masspoint_name


def load_loo(mhc, grid):
    """{mA: yield_closure entry} from the per-point LOO dirs; incomplete
    sweeps are a hard error."""
    out, missing = {}, []
    for mA in grid:
        path = os.path.join(srspaths.interpolation_loo_dir(mhc, mA),
                            "yields", "yield_closure.json")
        if not os.path.exists(path):
            missing.append(path)
            continue
        with open(path) as f:
            payload = json.load(f)
        if payload["meta"].get("loo_ma") != mA:
            raise RuntimeError(f"{path} is not a LOO result for mA={mA}")
        out[mA] = payload["closure"][masspoint_name(mA, mhc)]
    if missing:
        raise FileNotFoundError(
            f"LOO sweep incomplete for mHc={mhc}; missing:\n  "
            + "\n  ".join(missing))
    return out


def plot_one(mhc):
    grid = interpolation_config.study(mhc)["all"]
    with open(os.path.join(srspaths.interpolation_dir(mhc),
                           "yields", "yields.json")) as f:
        yields = json.load(f)["results"]
    loo = load_loo(mhc, grid)
    outdir = srspaths.interpolation_plots_dir(mhc, "yields")
    for channel in interpolation_config.STUDY_CHANNELS:
        interp_plot_utils.plot_yield_loo_grid(mhc, channel, yields, loo, outdir)
        interp_plot_utils.plot_yield_loo_residuals(mhc, channel, loo, outdir)
    print(f"[MHc{mhc}] wrote loo_grid/loo_residuals PNGs to {outdir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", type=int, help="one mHc study")
    parser.add_argument("--all", action="store_true",
                        help="every mHc in configs/interpolation.json")
    args = parser.parse_args()
    if not args.mhc and not args.all:
        parser.error("pass --mhc N and/or --all")
    if args.all:
        mhcs = sorted(int(k) for k in
                      srspaths.interpolation_config()["fit_points"])
    else:
        mhcs = [args.mhc]
    for mhc in mhcs:
        plot_one(mhc)


if __name__ == "__main__":
    main()
