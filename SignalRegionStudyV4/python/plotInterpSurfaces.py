#!/usr/bin/env python3
"""Global plots of the (mHc, mA) surfaces the chain fits.

Everything in the interpolation chain that depends on mA is now one surface
across all seven mHc studies, sliced per study. The per-study plots show a
single slice each, which hides the thing that actually makes the model work
— that the dense grids constrain the sparse ones. These plots show all seven
slices on a common mA axis with each study's own points in the matching
colour:

The yield model is N_win = k_era * G_period(mA) * f_category(mA), one panel
family per factor:

  fits/params/surface.shape.{category}.{param}.png      DCB parameters
  fits/yield/surface.G.{period}.{total_channel}.png     per-period total yield
  fits/yield/surface.f.{period}.{category}.png          window fraction

The era shares are the exception: their mA dependence is much weaker than
the spread between eras, so each plot fixes one mHc study and draws the
eras of a run period together, the way they add up to one:

  fits/yield/surface.k_era.{period}.{total_channel}.MHc{mhc}.png

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
from fitInterpYieldModel import period_merged_fraction
from interpolation_config import ALL_PARAM_ORDER

TOTAL_CHANNEL_SRC = {"SR1E2Mu": "SR1E2Mu", "SR3Mu": "SR3Mu_lowM"}
# G spans well under a decade, so a factor 2 above the maximum is a sliver of
# log axis; a decade is what actually clears the information text.
G_HEADROOM_FACTOR = 10.0
# f and k_era are fractions: fixed axes on their natural scale, with the top
# half of the frame left to the information text and the legend. Every panel
# of a family then shares one axis and can be compared by eye.
F_WINDOW_YRANGE = (0.0, 2.0)
K_ERA_YRANGE = (0.0, 1.0)


def _surface_curve(rec, grid, logspace=False, logit=False):
    """Evaluate a sliced record over a mA grid."""
    y = np.polyval(np.asarray(rec["coeffs"]), grid)
    if logspace:
        return np.exp(y)
    if logit:
        return 1.0 / (1.0 + np.exp(-y))
    return y


def plot_shape_surfaces(mhcs, outdir, max_error_frac):
    polys = {}
    for mhc in mhcs:
        path = os.path.join(srspaths.interpolation_fits_dir(mhc),
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
            # Only the region and final state are written on the plot: the
            # run period is in the lumi header and the parameter is the
            # y-axis title.
            interp_plot_utils.plot_surface_slices(
                (interp_plot_utils.REGION_TAG,
                 interp_plot_utils.channel_label(cat)), "m_{A} [GeV]",
                interp_plot_utils.param_title(param), per_mhc, outdir,
                f"surface.shape.{cat}.{param}", period, headroom_factor=2.0,
                max_error_frac=max_error_frac)


def _measured_era_shares(study_yields, src, eras):
    """[(mA, {era: share of the period total})] over the study's mass points
    that have every era of the period."""
    rows_by_ma = []
    for entry in sorted(study_yields.values(), key=lambda e: e["mA"]):
        rows = {e: entry["channels"].get(src, {}).get(e) for e in eras}
        if any(r is None for r in rows.values()):
            continue
        total = sum(r["sumw_total"] for r in rows.values())
        rows_by_ma.append((entry["mA"],
                           {e: rows[e]["sumw_total"] / total for e in eras}))
    return rows_by_ma


def plot_yield_surfaces(mhcs, outdir, max_error_frac):
    models, yields = {}, {}
    for mhc in mhcs:
        ydir = os.path.join(srspaths.interpolation_fits_dir(mhc), "yields")
        with open(os.path.join(ydir, "yield_model.json")) as f:
            models[mhc] = json.load(f)["model"]
        with open(os.path.join(ydir, "yields.json")) as f:
            yields[mhc] = json.load(f)["results"]

    plot_fraction_surfaces(mhcs, models, yields, outdir, max_error_frac)

    for period, eras in run_period_utils.RUN_PERIODS.items():
        for tot_channel, src in TOTAL_CHANNEL_SRC.items():
            channel = interp_plot_utils.channel_label(tot_channel)
            # --- G: period-summed total yield, one slice per mHc study
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
                (interp_plot_utils.REGION_TAG, channel), "m_{A} [GeV]",
                "G [events]", per_mhc, outdir,
                f"surface.G.{period}.{tot_channel}", period, logy=True,
                headroom_factor=G_HEADROOM_FACTOR,
                max_error_frac=max_error_frac)

            # --- k_era: the shares of one period, all its eras together at
            # a fixed mHc. The era is the interesting axis here (the shares
            # of a period sum to one), and their mA dependence is far weaker
            # than the spread between eras, so mHc labels the plot instead
            # of being drawn as seven near-degenerate slices per era.
            for mhc in mhcs:
                shares = _measured_era_shares(yields[mhc], src, eras)
                if not shares:
                    continue
                ma = [m for m, _ in shares]
                grid = np.linspace(min(ma) - 2, max(ma) + 2, 150)
                per_era = {}
                for era in eras:
                    rec = models[mhc]["totals"][period][tot_channel]["k"][era]
                    per_era[era] = {
                        "curve": (grid, _surface_curve(rec, grid)),
                        "points": (ma, [s[era] for _, s in shares],
                                   [0.0] * len(ma))}
                interp_plot_utils.plot_surface_slices(
                    (interp_plot_utils.REGION_TAG, channel,
                     interp_plot_utils.mhc_legend_label(mhc)), "m_{A} [GeV]",
                    "k_{era}", per_era, outdir,
                    f"surface.k_era.{period}.{tot_channel}.MHc{mhc}", period,
                    yrange=K_ERA_YRANGE, max_error_frac=max_error_frac,
                    legend_label=str, key_order=eras)


def _fraction_curve(fractions, channel, grid):
    """f(mA) of one category from the period's fraction sub-model."""
    if channel == "SR1E2Mu":
        return _surface_curve(fractions["f_sr1e2mu"], grid)
    S = _surface_curve(fractions["S"], grid)
    p_high = _surface_curve(fractions["p_high_logit"], grid, logit=True)
    return S * (p_high if channel == "SR3Mu_highM" else 1.0 - p_high)


def plot_fraction_surfaces(mhcs, models, yields, outdir, max_error_frac):
    """The third factor of N = k_era * G * f, per category.

    f is the one factor that is NOT a joint surface — it is a per-study 1D
    fit (SR1E2Mu directly, SR3Mu through the S / p_high decomposition), so
    these panels overlay seven independent fits rather than slices of one
    model. That is exactly what makes them worth drawing: they show how
    little the window fraction moves with mHc.
    """
    for period, eras in run_period_utils.RUN_PERIODS.items():
        for channel in interpolation_config.STUDY_CHANNELS:
            per_mhc = {}
            for mhc in mhcs:
                merged = period_merged_fraction(yields[mhc], channel, eras)
                if not merged:
                    continue
                ma = sorted(merged)
                grid = np.linspace(min(ma) - 2, max(ma) + 2, 150)
                per_mhc[mhc] = {
                    "curve": (grid, _fraction_curve(
                        models[mhc]["fractions"][period], channel, grid)),
                    "points": (ma, [merged[m]["f"] for m in ma],
                               [merged[m]["ferr"] for m in ma])}
            if not per_mhc:
                continue
            interp_plot_utils.plot_surface_slices(
                (interp_plot_utils.REGION_TAG,
                 interp_plot_utils.channel_label(channel)), "m_{A} [GeV]",
                "f_{window}", per_mhc, outdir,
                f"surface.f.{period}.{channel}", period,
                yrange=F_WINDOW_YRANGE, max_error_frac=max_error_frac)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true",
                        help="both shape and yield surfaces (default)")
    parser.add_argument("--kind", choices=("shape", "yield"),
                        help="only one family")
    parser.add_argument("--max-error-frac", type=float, default=0.3,
                        help="hide points whose error bar exceeds this "
                             "fraction of the plotted value range "
                             "(0 = draw every point); display only")
    args = parser.parse_args()

    mhcs = interpolation_config.mhc_grid()
    total = 0
    if args.kind in (None, "shape") or args.all:
        outdir = srspaths.interpolation_global_plots_dir("params")
        os.makedirs(outdir, exist_ok=True)
        plot_shape_surfaces(mhcs, outdir, args.max_error_frac)
        n = len([f for f in os.listdir(outdir) if f.endswith(".png")])
        print(f"Wrote {n} shape-surface plots into {outdir}")
        total += n
    if args.kind in (None, "yield") or args.all:
        outdir = srspaths.interpolation_global_plots_dir("yield")
        os.makedirs(outdir, exist_ok=True)
        plot_yield_surfaces(mhcs, outdir, args.max_error_frac)
        n = len([f for f in os.listdir(outdir) if f.endswith(".png")])
        print(f"Wrote {n} yield-surface plots into {outdir}")
        total += n
    print(f"{total} surface plots in total")


if __name__ == "__main__":
    main()
