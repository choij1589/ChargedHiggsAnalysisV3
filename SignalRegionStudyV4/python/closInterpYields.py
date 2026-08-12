#!/usr/bin/env python3
"""Stage 3 of the yield-interpolation chain: validation.

Runs over ALL study mass points; fit points are the self-consistency test
(in_sample=true), held-out points the interpolation test:

  - scalar test, per era x channel: predicted window yield
    N_pred = k_era * G_period(mA) * f_category(mA) (the physics model of
    fitInterpYieldModel.py; error from the component fit covariances)
    against the measured N_meas +- err;
    pull = (N_pred - N_meas)/sqrt(err^2 + err_pred^2) and relative
    residual N_pred/N_meas - 1.
  - template test, per merged category: the interpolated shape model
    normalized to the summed per-era predicted yields — ABSOLUTE
    normalization, no rescaling to MC — compared to the 100-bin MC
    histogram in the same interpolated window: chi2/ndf + a cmsstyle
    overlay/ratio PNG.

Per-masspoint condor sharding: --masspoints + --output write a part JSON
(plots are written directly); merge with mergeInterpResults.py
--stage yield-closure, which also prints the summary tables.

  python3 closInterpYields.py --mhc 160 [--masspoints MHc160_MA85] [--output F]
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import ROOT

import interp_plot_utils
import interpolation_config
import run_period_utils
import srspaths
from interpolation_config import (ALL_PARAM_ORDER, BKG_PARAMS, PARAM_CEILINGS,
                                  PARAM_FLOORS, masspoint_name)

from dcb_fit_utils import build_model, make_mc_hist
from fitInterpYieldModel import fit_range, predict_yield

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError

TEMPLATE_NBINS = 100


def predict_shape_params(cat_polys, mA):
    """Predicted shape parameters at mA, mirroring closInterpShapes.py:
    floors/ceilings clip, background dropped when fsig -> 1."""
    predicted = {}
    clipped = []
    for param in [p for p in ALL_PARAM_ORDER if p in cat_polys]:
        value = float(interpolation_config.eval_param(cat_polys[param], mA))
        floor = PARAM_FLOORS.get(param)
        ceiling = PARAM_CEILINGS.get(param)
        if floor is not None and value < floor:
            clipped.append(param)
            value = floor
        if ceiling is not None and value > ceiling:
            clipped.append(param)
            value = ceiling
        predicted[param] = value
    if predicted.get("fsig", 0.0) >= interpolation_config.FSIG_DROP_THRESHOLD:
        for bkg_param in BKG_PARAMS + ("fsig",):
            predicted.pop(bkg_param, None)
    return predicted, clipped


def template_hist_from_model(pdf, mass_var, n_pred, tag):
    """Predicted-template bin contents: pdf shape scaled to N_pred."""
    h_model = pdf.createHistogram(f"h_tmpl_{tag}", mass_var,
                                  ROOT.RooFit.Binning(TEMPLATE_NBINS))
    values = np.array([h_model.GetBinContent(i)
                       for i in range(1, TEMPLATE_NBINS + 1)])
    h_model.Delete()
    total = values.sum()
    if total <= 0:
        raise RuntimeError(f"Non-positive model integral for {tag}")
    return values * (n_pred / total)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", type=int, required=True,
                        help="mHc study to run")
    parser.add_argument("--masspoints", default="",
                        help="comma-separated masspoint filter")
    parser.add_argument("--output", default="",
                        help="output JSON path (default: "
                             "tests/interpolation/MHc{X}/yields/yield_closure.json)")
    parser.add_argument("--loo-ma", type=int, default=None,
                        help="leave-one-out mode: evaluate ONLY this mA, "
                             "against the LOO yield_model/polynomials in the "
                             "per-point dir tests/interpolation/MHc{X}_MA{Y}/ "
                             "(where yield_closure.json and plots are written "
                             "too); measured yields come from the adopted "
                             "yields.json")
    args = parser.parse_args()

    if args.loo_ma is not None and args.masspoints:
        raise ValueError("--loo-ma already selects its single mass point; "
                         "do not combine with --masspoints")
    study = interpolation_config.study(args.mhc, loo_ma=args.loo_ma)
    fit_ma = study["fit"]
    yields_dir = os.path.join(srspaths.interpolation_dir(args.mhc), "yields")
    loo_dir = (srspaths.interpolation_loo_dir(args.mhc, args.loo_ma)
               if args.loo_ma is not None else None)
    plot_base = (os.path.join(loo_dir, "plots", "yields")
                 if loo_dir is not None
                 else srspaths.interpolation_plots_dir(args.mhc, "yields"))

    with open(os.path.join(yields_dir, "yields.json")) as f:
        yields = json.load(f)["results"]
    model_path = os.path.join(loo_dir, "yields", "yield_model.json") \
        if loo_dir is not None else os.path.join(yields_dir, "yield_model.json")
    with open(model_path) as f:
        model_payload = json.load(f)
    model = model_payload["model"]
    if args.loo_ma is not None and model_payload["meta"].get("loo_ma") != args.loo_ma:
        raise RuntimeError(
            f"{model_path} was not produced with --loo-ma {args.loo_ma} "
            f"(meta.loo_ma={model_payload['meta'].get('loo_ma')})")
    shape_polys, _ = interpolation_config.load_shape_polynomials(
        args.mhc, loo_ma=args.loo_ma)

    if args.loo_ma is not None:
        masspoints = [masspoint_name(args.loo_ma, args.mhc)]
    else:
        masspoints = interpolation_config.filter_csv(
            [masspoint_name(m, args.mhc) for m in study["all"]],
            args.masspoints, "masspoint")

    output = {}
    warnings = []
    for mp in masspoints:
        mA = interpolation_config.mA_of(mp)
        in_sample = mA in fit_ma
        entry = {"mA": mA, "in_sample": in_sample,
                 "scalar": {}, "template": {}}

        # Scalar test: per era x channel yield prediction vs measurement.
        for channel in interpolation_config.STUDY_CHANNELS:
            entry["scalar"][channel] = {}
            for period, suberas in run_period_utils.RUN_PERIODS.items():
                fit_lo, fit_hi = fit_range(model, period)
                for era in suberas:
                    meas = yields[mp]["channels"].get(channel, {}).get(era)
                    if meas is None:
                        warnings.append(f"[{mp}/{channel}/{era}] no measured "
                                        "yield; skipped")
                        continue
                    n_pred, err_pred = predict_yield(model, shape_polys,
                                                     channel, era, mA)
                    # Measured error floored like the fit inputs: per-sample
                    # normalization noise dominates over MC stat.
                    err_floor = interpolation_config.REL_YIELD_ERR_FLOOR[
                        interpolation_config.period_of(era)]
                    err_meas = max(meas["err"], meas["sumw"] * err_floor)
                    pull = (n_pred - meas["sumw"]) / np.hypot(err_meas,
                                                              err_pred)
                    entry["scalar"][channel][era] = {
                        "n_pred": n_pred, "err_pred": err_pred,
                        "n_meas": meas["sumw"], "err_meas": meas["err"],
                        "pull": float(pull),
                        "rel": n_pred / meas["sumw"] - 1.0,
                        "extrapolation": mA < fit_lo or mA > fit_hi,
                    }

        # Template test: absolute-normalized interpolated model vs MC per
        # merged category.
        for channel, period, suberas in interpolation_config.categories():
            cat_key = interpolation_config.category_key(channel, period)
            per_era = entry["scalar"][channel]
            missing_eras = [e for e in suberas if e not in per_era]
            if missing_eras:
                warnings.append(f"[{mp}/{cat_key}] template check skipped "
                                f"(no yield for {', '.join(missing_eras)})")
                continue
            n_pred = sum(per_era[e]["n_pred"] for e in suberas)
            err_pred = float(np.hypot.reduce(
                [per_era[e]["err_pred"] for e in suberas]))

            lo, hi = interpolation_config.interp_window(shape_polys[cat_key], mA)
            chain, missing = interpolation_config.build_signal_chain(
                suberas, channel, mp, ROOT)
            if chain is None:
                warnings.append(f"[{mp}/{cat_key}] missing sample {missing}; "
                                "template check skipped")
                continue
            htag = f"{cat_key}_{mp}".replace("-", "_")
            hist = make_mc_hist(chain, f"h_yc_{htag}", lo, hi)

            predicted, clipped = predict_shape_params(shape_polys[cat_key],
                                                      mA)
            if clipped:
                warnings.append(f"[{mp}/{cat_key}] predicted shape "
                                f"parameter(s) clipped: {', '.join(clipped)}")
            mass_var = ROOT.RooRealVar(f"mass_{htag}", "mass", lo, hi)
            pdf, _keep = build_model("interp", mass_var, predicted)
            pred = template_hist_from_model(pdf, mass_var, n_pred, htag)

            mc = np.array([hist.GetBinContent(i)
                           for i in range(1, hist.GetNbinsX() + 1)])
            mc_err = np.array([hist.GetBinError(i)
                               for i in range(1, hist.GetNbinsX() + 1)])
            ok = mc_err > 0
            chi2 = float(np.sum(((mc[ok] - pred[ok]) / mc_err[ok]) ** 2))
            ndf = int(ok.sum())

            interp_plot_utils.plot_yield_template_closure(
                cat_key, mp, hist, pred, n_pred, err_pred, chi2, ndf,
                period, plot_base)
            entry["template"][cat_key] = {
                "n_pred": n_pred, "err_pred": err_pred,
                "n_mc": float(mc.sum()),
                "window": [lo, hi],
                "chi2": chi2, "ndf": ndf,
            }
        output[mp] = entry
        print(f"[{mp}] scalar: "
              f"{sum(len(v) for v in entry['scalar'].values())} era yields, "
              f"template: {len(entry['template'])} categories")

    payload = {
        "meta": {
            "mhc": args.mhc,
            "fit_ma": fit_ma,
            "loo_ma": args.loo_ma,
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "closure": output,
        "warnings": warnings,
    }
    if loo_dir is not None:
        default_out = os.path.join(loo_dir, "yields", "yield_closure.json")
    else:
        default_out = os.path.join(yields_dir, "yield_closure.json")
    outpath = args.output or default_out
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {outpath}")
    for w in warnings:
        print(f"  warning: {w}")


if __name__ == "__main__":
    main()
