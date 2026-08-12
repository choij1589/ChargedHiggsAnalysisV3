#!/usr/bin/env python3
"""Stage 1 of the mA-interpolation chain: per-masspoint DCB(+cheb2) fits.

For every masspoint x category, fit the signal 'Central' trees and record
parameters + errors + fit quality into
tests/interpolation/MHc{X}/fits/dcb_fits_{pass}.json, with a diagnostic
MC-vs-fit PNG per fit. Adopted fit model (frozen): pure DCB for SR1E2Mu and
SR3Mu_lowM, DCB + 2nd-order Chebychev combinatoric background for
SR3Mu_highM alone (interpolation_config.channel_has_bkg) — lowM's
few-percent wrong-pairing continuum is absorbed by the DCB tails.

Two-pass structure (part of the adopted method, docs/interpolation/EXPERIMENTS.md S5):
``--pass floating`` fits nL/nR freely (source of the per-category median
used to freeze them); ``--pass frozen`` fixes nL/nR to that median, which
breaks the alpha-n degeneracy and is what the shape polynomials are fitted
on.

Missing samples are skipped with a warning, never fatal. Runs as
per-masspoint condor jobs via the interpolation wrapper (--output writes a
part JSON; merge with mergeInterpResults.py --stage fits-floating|fits).

  python3 fitInterpShapes.py --mhc 160 --pass floating \\
      [--masspoints MP1,MP2] [--categories SR3Mu_lowM_Run2,...] [--output F]
"""
import argparse
import datetime
import json
import os
import sys
from collections import OrderedDict

import interpolation_config
import srspaths
from interpolation_config import LOW_STAT_ENTRIES, masspoint_name

import ROOT  # noqa: E402

from dcb_fit_utils import (bkg_components, canvas_config,  # noqa: E402
                           draw_dcb_params, fit_dcb_bkg,
                           fit_dcb_with_errors, fit_quality, make_mc_hist,
                           model_label)
from plotter import FitCanvasWithRatio  # noqa: E402  (Common/Tools)

ROOT.gROOT.SetBatch(True)


def make_fit_plot(chain, fit, era, channel, masspoint, outdir):
    """Diagnostic plot: 100-bin MC hist + frozen-parameter model overlay."""
    from dcb_fit_utils import build_model

    fit_lo, fit_hi = fit["fit_lo"], fit["fit_hi"]
    p = {k: v["value"] for k, v in fit["params"].items()}

    hist_name = f"h_fit_{era}_{channel}_{masspoint}".replace("-", "_")
    hist = make_mc_hist(chain, hist_name, fit_lo, fit_hi)

    mass_plot = ROOT.RooRealVar("mass_plot", "mass", fit_lo, fit_hi)
    roo_data = ROOT.RooDataHist("data_plot", "", ROOT.RooArgList(mass_plot),
                                hist)

    pdf, _keep = build_model("plot", mass_plot, p)
    npar = sum(1 for pv in fit["params"].values() if pv["error"] > 0)
    label = model_label(fit)
    fit_models = OrderedDict([(label, (pdf, npar, ROOT.kSolid))])
    config = canvas_config(era, channel, masspoint,
                           bkg_components(fit, label, "plot"))

    canvas = FitCanvasWithRatio(roo_data, mass_plot, hist, fit_models, config)
    canvas.drawPadUp()
    canvas.drawPadDown()
    canvas.drawMasspoint()
    canvas.canv.cd(1)
    draw_dcb_params(fit["params"], sigma_eff=fit["sigma_eff"])
    os.makedirs(outdir, exist_ok=True)
    canvas.canv.SaveAs(
        os.path.join(outdir, f"signal_fit.{channel}_{era}.{masspoint}.png"))
    canvas.canv.Close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", type=int, required=True,
                        help="mHc study to run")
    parser.add_argument("--pass", dest="fit_pass", required=True,
                        choices=["floating", "frozen"],
                        help="floating: nL/nR free (source of the median); "
                             "frozen: nL/nR fixed to that median")
    parser.add_argument("--masspoints", default="",
                        help="comma-separated masspoint filter")
    parser.add_argument("--categories", default="",
                        help="comma-separated category filter")
    parser.add_argument("--output", default="",
                        help="write results to this explicit path instead of "
                             "fits/dcb_fits_{pass}.json (no merging; used by "
                             "per-masspoint condor jobs)")
    args = parser.parse_args()

    study = interpolation_config.study(args.mhc)
    fixed_n = (interpolation_config.fixed_n_values(args.mhc)
               if args.fit_pass == "frozen" else {})

    masspoints = interpolation_config.filter_csv(
        [masspoint_name(m, args.mhc) for m in study["all"]],
        args.masspoints, "masspoint")
    cats = interpolation_config.categories()
    known_keys = {interpolation_config.category_key(ch, p) for ch, p, _ in cats}
    requested = set(interpolation_config.filter_csv(sorted(known_keys),
                                             args.categories, "category"))
    cats = [(ch, p, s) for ch, p, s in cats
            if interpolation_config.category_key(ch, p) in requested]

    results = {}
    warnings = []
    for channel, period, suberas in cats:
        cat_key = interpolation_config.category_key(channel, period)
        results.setdefault(cat_key, {})
        for mp in masspoints:
            mA = interpolation_config.mA_of(mp)
            chain, missing = interpolation_config.build_signal_chain(
                suberas, channel, mp, ROOT)
            if chain is None:
                warnings.append(f"[{cat_key}/{mp}] skipped, missing sample: "
                                f"{missing}")
                print(f"WARNING: {warnings[-1]}")
                continue

            print(f"[{cat_key}/{mp}] fitting (mA nominal = {mA} GeV, "
                  f"{chain.GetEntries()} entries)...")
            use_bkg = ("cheb2"
                       if interpolation_config.channel_has_bkg(channel)
                       else None)
            if use_bkg or args.fit_pass == "frozen":
                nfix = fixed_n[cat_key] if args.fit_pass == "frozen" else {}
                fit = fit_dcb_bkg(chain, float(mA),
                                  nL_fixed=nfix.get("nL"),
                                  nR_fixed=nfix.get("nR"),
                                  bkg=use_bkg)
            else:
                fit = fit_dcb_with_errors(chain, float(mA))
            quality, reasons = fit_quality(fit)
            fit["quality"] = quality
            fit["quality_reasons"] = reasons
            fit["low_stat"] = fit["entries"] < LOW_STAT_ENTRIES
            fit["mA"] = mA
            if quality != "good":
                warnings.append(f"[{cat_key}/{mp}] BAD fit: {'; '.join(reasons)}")
                print(f"WARNING: {warnings[-1]}")

            make_fit_plot(chain, fit, period, channel, mp,
                          srspaths.interpolation_plots_dir(
                              args.mhc, "fits"))
            results[cat_key][mp] = fit

    # The frozen pass is the adopted direct fit (dcb_fits.json); the
    # floating pass only exists to derive the per-category median n
    # (dcb_fits_floating.json).
    basename = "dcb_fits.json" if args.fit_pass == "frozen" else "dcb_fits_floating.json"

    if args.output:
        outpath = args.output
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    else:
        fits_dir = os.path.join(
            srspaths.interpolation_dir(args.mhc), "fits")
        os.makedirs(fits_dir, exist_ok=True)
        outpath = os.path.join(fits_dir, basename)
        # A filtered rerun merges into the existing JSON instead of
        # clobbering the fits it did not redo.
        if (args.masspoints or args.categories) and os.path.exists(outpath):
            with open(outpath) as f:
                previous = json.load(f).get("results", {})
            for cat_key, fits in results.items():
                previous.setdefault(cat_key, {}).update(fits)
            results = previous

    payload = {
        "meta": {
            "mhc": args.mhc,
            "fit_pass": args.fit_pass,
            "fit_ma": study["fit"],
            "held_out_ma": study["held_out"],
            "fixed_n": fixed_n or None,
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "results": results,
        "warnings": warnings,
    }
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)

    n_fits = sum(len(v) for v in results.values())
    n_bad = sum(1 for cat in results.values() for fit in cat.values()
                if fit["quality"] != "good")
    print(f"\nWrote {outpath}: {n_fits} fits "
          f"({n_bad} flagged bad, {len(warnings)} warnings).")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(f"  {w}")


if __name__ == "__main__":
    main()
