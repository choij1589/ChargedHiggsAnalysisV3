#!/usr/bin/env python3
"""Summary plots of the leave-one-out yield closures — the visual
counterpart of the loo_uncertainties.json norm tables.

One two-pad PNG per (channel, era): measured MC yields, the adopted
full-grid model curve with its 1-sigma band and every point's LOO
prediction on top; the LOO relative residual (with the model band in
relative terms) below.

Reads the adopted yields.json + yield_model.json + polynomials.json and
the 78 per-point closure/interpolation/loo/MHc{X}_MA{Y}/yields/yield_closure.json;
JSON-only, no sample access. Outputs into
closure/interpolation/MHc{X}/plots/yields/loo_grid.{channel}.{era}.png.

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
    from fitInterpYieldModel import predict_yield

    grid = interpolation_config.study(mhc)["all"]
    yields_dir = os.path.join(srspaths.interpolation_fits_dir(mhc),
                              "yields")
    with open(os.path.join(yields_dir, "yields.json")) as f:
        yields = json.load(f)["results"]
    with open(os.path.join(yields_dir, "yield_model.json")) as f:
        model = json.load(f)["model"]
    loo = load_loo(mhc, grid)
    outdir = srspaths.interpolation_closure_plots_dir(mhc, "yields")
    for channel in interpolation_config.STUDY_CHANNELS:
        interp_plot_utils.plot_yield_loo_grid(
            mhc, channel, yields, loo, outdir,
            model=model, predict_yield=predict_yield)
    print(f"[MHc{mhc}] wrote loo_grid PNGs to {outdir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", type=int, help="one mHc study")
    parser.add_argument("--all", action="store_true",
                        help="every mHc in the baseline grid")
    args = parser.parse_args()
    if not args.mhc and not args.all:
        parser.error("pass --mhc N and/or --all")
    if args.all:
        mhcs = sorted(int(k) for k in
                      interpolation_config.mhc_grid())
    else:
        mhcs = [args.mhc]
    for mhc in mhcs:
        plot_one(mhc)


if __name__ == "__main__":
    main()
