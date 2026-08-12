#!/usr/bin/env python3
"""Global plots of the (mHc, mA) surfaces the chain fits.

Everything in the interpolation chain that depends on mA is now one surface
across all seven mHc studies, sliced per study. The per-study plots show a
single slice each, which hides the thing that actually makes the model work
— that the dense grids constrain the sparse ones. These plots show all seven
slices on a common mA axis with each study's own points in the matching
colour:

  plots/surfaces/surface.shape.{category}.{param}.png   DCB parameters
  plots/surfaces/surface.G.{period}.{total_channel}.png per-period total yield
  plots/surfaces/surface.k_era.{period}.{total_channel}.png  era shares

Reads the per-study polynomials.json / yield_model.json (each carries the
full surface under joint_surface) plus the measured fits and yields.
JSON-only apart from the ROOT canvases; safe to run on the login node.

  python3 plotInterpSurfaces.py --all
  python3 plotInterpSurfaces.py --kind shape
"""
import argparse
import json
import os

import numpy as np

import interp_plot_utils
import interpolation_config
import run_period_utils
import srspaths
from interpolation_config import ALL_PARAM_ORDER

TOTAL_CHANNEL_SRC = {"SR1E2Mu": "SR1E2Mu", "SR3Mu": "SR3Mu_lowM"}


def _surface_curve(rec, grid, logspace=False, logit=False):
    """Evaluate a sliced record over a mA grid."""
    y = np.polyval(np.asarray(rec["coeffs"]), grid)
    if logspace:
        return np.exp(y)
    if logit:
        return 1.0 / (1.0 + np.exp(-y))
    return y


def plot_shape_surfaces(mhcs, outdir):
    polys = {}
    for mhc in mhcs:
        path = os.path.join(srspaths.interpolation_dir(mhc),
                            "polynomials.json")
        with open(path) as f:
            polys[mhc] = json.load(f)["polynomials"]

    categories = sorted({c for p in polys.values() for c in p})
    for cat in categories:
        params = [p for p in ALL_PARAM_ORDER
                  if any(p in polys[m].get(cat, {}) for m in mhcs)]
        for param in params:
            per_mhc = {}
            for mhc in mhcs:
                rec = polys[mhc].get(cat, {}).get(param)
                if rec is None or rec.get("frozen"):
                    continue
                pu = rec["points_used"]
                if not pu["mA"]:
                    continue
                grid = np.linspace(min(pu["mA"]) - 2, max(pu["mA"]) + 2, 150)
                logit = rec.get("form") == "logitpoly"
                curve = _surface_curve(rec, grid, logit=logit)
                per_mhc[mhc] = {"curve": (grid, curve),
                                "points": (pu["mA"], pu["value"], pu["error"])}
            if not per_mhc:
                continue
            period = cat.rsplit("_", 1)[1]
            interp_plot_utils.plot_surface_slices(
                f"{cat}  {param}", "m_{A} [GeV]", param, per_mhc, outdir,
                f"surface.shape.{cat}.{param}", period)


def plot_yield_surfaces(mhcs, outdir):
    models, yields = {}, {}
    for mhc in mhcs:
        ydir = os.path.join(srspaths.interpolation_dir(mhc), "yields")
        with open(os.path.join(ydir, "yield_model.json")) as f:
            models[mhc] = json.load(f)["model"]
        with open(os.path.join(ydir, "yields.json")) as f:
            yields[mhc] = json.load(f)["results"]

    for period, eras in run_period_utils.RUN_PERIODS.items():
        for tot_channel, src in TOTAL_CHANNEL_SRC.items():
            # --- G: period-summed total yield
            per_mhc = {}
            for mhc in mhcs:
                rec = models[mhc]["totals"][period][tot_channel]["G"]
                pu = rec["points_used"]
                grid = np.linspace(min(pu["x"]) - 2, max(pu["x"]) + 2, 150)
                per_mhc[mhc] = {
                    "curve": (grid, _surface_curve(rec, grid, logspace=True)),
                    "points": (pu["x"], list(np.exp(pu["y"])),
                               [np.exp(v) * e for v, e in zip(pu["y"],
                                                              pu["err"])])}
            interp_plot_utils.plot_surface_slices(
                f"G  {tot_channel}  {period}", "m_{A} [GeV]", "N_{total}",
                per_mhc, outdir, f"surface.G.{period}.{tot_channel}", period,
                logy=True)

            # --- k_era: era shares
            for era in eras:
                per_mhc = {}
                for mhc in mhcs:
                    rec = models[mhc]["totals"][period][tot_channel]["k"][era]
                    pts_x, pts_y = [], []
                    for mp, entry in sorted(yields[mhc].items(),
                                            key=lambda kv: kv[1]["mA"]):
                        rows = {e: entry["channels"].get(src, {}).get(e)
                                for e in eras}
                        if any(r is None for r in rows.values()):
                            continue
                        total = sum(r["sumw_total"] for r in rows.values())
                        pts_x.append(entry["mA"])
                        pts_y.append(rows[era]["sumw_total"] / total)
                    if not pts_x:
                        continue
                    grid = np.linspace(min(pts_x) - 2, max(pts_x) + 2, 150)
                    per_mhc[mhc] = {
                        "curve": (grid, _surface_curve(rec, grid)),
                        "points": (pts_x, pts_y, [0.0] * len(pts_x))}
                if not per_mhc:
                    continue
                interp_plot_utils.plot_surface_slices(
                    f"k_{{{era}}}  {tot_channel}", "m_{A} [GeV]",
                    "era share of the period total", per_mhc, outdir,
                    f"surface.k_era.{era}.{tot_channel}", era)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="both shape and yield surfaces (default)")
    parser.add_argument("--kind", choices=("shape", "yield"),
                        help="only one family")
    args = parser.parse_args()

    mhcs = interpolation_config.mhc_grid()
    outdir = srspaths.interpolation_global_plots_dir("surfaces")
    os.makedirs(outdir, exist_ok=True)
    if args.kind in (None, "shape") or args.all:
        plot_shape_surfaces(mhcs, outdir)
    if args.kind in (None, "yield") or args.all:
        plot_yield_surfaces(mhcs, outdir)
    n = len([f for f in os.listdir(outdir) if f.endswith(".png")])
    print(f"Wrote {n} surface plots into {outdir}")


if __name__ == "__main__":
    main()
