#!/usr/bin/env python3
"""Stage 3 of the mA-interpolation chain: shape closure over the FULL mass
grid (interpolation_config.study(mhc)["all"]; fit points double as
in-sample checks, held-out points are the interpolation test).

At each mA and category, evaluate the stage-2 parametrizations to predict
the model parameters, then compare against the stage-1 direct (frozen-n)
fit and the MC histogram:

  - per-parameter pull = (predicted - direct) / direct_error;
  - overlay canvas: MC hist + direct model (solid) + interpolated model
    (dashed), with signal/background component curves;
  - chi2/ndf of each shape against the MC histogram (pdf normalized to the
    MC integral in the fit window);
  - x0_direct, sigma_eff_direct, sigma_eff_pred — feed the interpolation-
    uncertainty exporter (exportInterpUncertainties.py) directly, so it does
    not have to re-derive them from params.

Outputs tests/interpolation/MHc{X}/closure.json, canvases under
tests/interpolation/MHc{X}/plots/closure/, and a summary table. Runs as
per-masspoint condor jobs (--output writes a part JSON; merge with
mergeInterpResults.py --stage closure).

  python3 closInterpShapes.py --mhc 160 [--masspoints MP1,MP2] [--output F]
"""
import argparse
import datetime
import json
import math
import os
import sys
from collections import OrderedDict

import ROOT

import interpolation_config
import srspaths
from interpolation_config import (ALL_PARAM_ORDER, PARAM_CEILINGS,
                                  PARAM_FLOORS, masspoint_name)

from dcb_fit_utils import (bkg_components, build_model,
                           canvas_config, draw_dcb_param_comparison,
                           make_mc_hist, model_label)
from plotter import FitCanvasWithRatio  # Common/Tools

ROOT.gROOT.SetBatch(True)


def pdf_chi2_vs_hist(pdf, mass_var, hist, n_free_params):
    """chi2 of a pdf against the MC hist, pdf scaled to the MC integral."""
    nbins = hist.GetNbinsX()
    h_model = pdf.createHistogram(f"h_chi2_{pdf.GetName()}_{id(pdf)}",
                                  mass_var, ROOT.RooFit.Binning(nbins))
    if h_model.Integral() > 0:
        h_model.Scale(hist.Integral() / h_model.Integral())
    chi2 = 0.0
    n_used = 0
    for i in range(1, nbins + 1):
        err = hist.GetBinError(i)
        if err <= 0:
            continue
        chi2 += ((hist.GetBinContent(i) - h_model.GetBinContent(i)) / err) ** 2
        n_used += 1
    h_model.Delete()
    ndf = max(n_used - n_free_params, 1)
    return chi2, ndf


def sigma_eff_of(params):
    sL = params["sigmaL"]["value"] if isinstance(params.get("sigmaL"), dict) else params.get("sigmaL")
    sR = params["sigmaR"]["value"] if isinstance(params.get("sigmaR"), dict) else params.get("sigmaR")
    return math.sqrt(0.5 * (sL * sL + sR * sR))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", type=int, required=True,
                        help="mHc study to run")
    parser.add_argument("--masspoints", default="",
                        help="comma-separated closure masspoint filter")
    parser.add_argument("--output", default="",
                        help="write results to this explicit path instead of "
                             "closure.json (no merging; used by per-masspoint "
                             "condor jobs)")
    parser.add_argument("--loo-ma", type=int, default=None,
                        help="leave-one-out mode: evaluate ONLY this mA, "
                             "against the LOO polynomials in the per-point "
                             "dir tests/interpolation/MHc{X}_MA{Y}/ (where "
                             "closure.json and plots are written too)")
    args = parser.parse_args()

    if args.loo_ma is not None and args.masspoints:
        raise ValueError("--loo-ma already selects its single mass point; "
                         "do not combine with --masspoints")
    study = interpolation_config.study(args.mhc, loo_ma=args.loo_ma)
    interp_dir = srspaths.interpolation_dir(args.mhc)
    loo_dir = (srspaths.interpolation_loo_dir(args.mhc, args.loo_ma)
               if args.loo_ma is not None else None)

    with open(os.path.join(interp_dir, "fits", "dcb_fits.json")) as f:
        dcb = json.load(f)["results"]
    poly_path = os.path.join(loo_dir or interp_dir, "polynomials.json")
    with open(poly_path) as f:
        poly_payload = json.load(f)
    polys = poly_payload["polynomials"]
    if args.loo_ma is not None and poly_payload["meta"].get("loo_ma") != args.loo_ma:
        raise RuntimeError(
            f"{poly_path} was not produced with --loo-ma {args.loo_ma} "
            f"(meta.loo_ma={poly_payload['meta'].get('loo_ma')})")

    if args.loo_ma is not None:
        closure_points = [masspoint_name(args.loo_ma, args.mhc)]
    else:
        closure_points = interpolation_config.filter_csv(
            [masspoint_name(m, args.mhc) for m in study["all"]],
            args.masspoints, "closure masspoint")
    fit_mp = {masspoint_name(m, args.mhc) for m in study["fit"]}

    output = {}
    warnings = []
    rows = []
    for channel, period, suberas in interpolation_config.categories():
        cat_key = interpolation_config.category_key(channel, period)
        if cat_key not in polys:
            warnings.append(f"[{cat_key}] no parametrization; category skipped")
            continue
        output[cat_key] = {}
        # This category's actual fit range: the points the x0
        # parametrization used.
        cat_fit_ma = polys[cat_key]["x0"]["points_used"]["mA"]
        for mp in closure_points:
            mA = interpolation_config.mA_of(mp)
            direct = dcb.get(cat_key, {}).get(mp)
            if direct is None:
                warnings.append(f"[{cat_key}/{mp}] no direct fit; skipped")
                continue
            extrapolation = mA < min(cat_fit_ma) or mA > max(cat_fit_ma)

            # Predicted parameters from the parametrizations.
            cat_params = [p for p in ALL_PARAM_ORDER if p in polys[cat_key]]
            predicted = {}
            clipped = []
            for param in cat_params:
                value = float(interpolation_config.eval_param(
                    polys[cat_key][param], mA))
                floor = PARAM_FLOORS.get(param)
                ceiling = PARAM_CEILINGS.get(param)
                if floor is not None and value < floor:
                    clipped.append(param)
                    value = floor
                if ceiling is not None and value > ceiling:
                    clipped.append(param)
                    value = ceiling
                predicted[param] = value
            missing_poly = [p for p in direct["params"] if p not in predicted]
            if missing_poly:
                warnings.append(f"[{cat_key}/{mp}] no parametrization for "
                                f"{','.join(missing_poly)}; skipped")
                continue
            if clipped:
                warnings.append(f"[{cat_key}/{mp}] predicted parameter(s) "
                                f"clipped: {', '.join(clipped)}")

            # Per-parameter pulls vs the direct fit.
            direct_good = direct["quality"] == "good"
            pulls = {}
            for param in cat_params:
                pv = direct["params"][param]
                if direct_good and pv["error"] > 0:
                    pulls[param] = (predicted[param] - pv["value"]) / pv["error"]
                else:
                    pulls[param] = None

            x0_direct = direct["params"]["x0"]["value"]
            sigma_eff_direct = sigma_eff_of(direct["params"])
            sigma_eff_pred = sigma_eff_of(predicted)

            # MC histogram in the direct fit window.
            fit_lo, fit_hi = direct["fit_lo"], direct["fit_hi"]
            chain, missing = interpolation_config.build_signal_chain(
                suberas, channel, mp, ROOT)
            if chain is None:
                warnings.append(f"[{cat_key}/{mp}] missing sample {missing}; "
                                "skipped")
                continue

            hist_name = f"h_closure_{cat_key}_{mp}".replace("-", "_")
            hist = make_mc_hist(chain, hist_name, fit_lo, fit_hi)

            mass_var = ROOT.RooRealVar("mass_plot", "mass", fit_lo, fit_hi)
            roo_data = ROOT.RooDataHist("data_plot", "",
                                        ROOT.RooArgList(mass_var), hist)
            direct_params = {k: v["value"] for k, v in direct["params"].items()}
            pdf_direct, keep_d = build_model("direct", mass_var, direct_params)
            pdf_interp, keep_i = build_model("interp", mass_var, predicted)

            n_free_direct = sum(1 for pv in direct["params"].values()
                                if pv["error"] > 0)
            chi2_direct, ndf_direct = pdf_chi2_vs_hist(
                pdf_direct, mass_var, hist, n_free_direct)
            chi2_interp, ndf_interp = pdf_chi2_vs_hist(
                pdf_interp, mass_var, hist, 0)

            label = model_label(direct)
            fit_models = OrderedDict([
                (f"{label} direct", (pdf_direct, n_free_direct, ROOT.kSolid)),
                (f"{label} interpolated", (pdf_interp, 0, ROOT.kDashed)),
            ])
            config = canvas_config(
                period, channel, mp,
                bkg_components(direct, f"{label} direct", "direct"),
                legend=[0.58, 0.58, 0.90, 0.78],
                colors=[ROOT.kRed, ROOT.kBlue])
            canvas = FitCanvasWithRatio(roo_data, mass_var, hist,
                                        fit_models, config)
            canvas.drawPadUp()
            canvas.drawPadDown()
            canvas.drawMasspoint()
            canvas.canv.cd(1)
            draw_dcb_param_comparison(direct["params"], predicted)
            outdir = (os.path.join(loo_dir, "plots", "closure")
                      if loo_dir is not None
                      else srspaths.interpolation_plots_dir(
                          args.mhc, "closure"))
            os.makedirs(outdir, exist_ok=True)
            canvas.canv.SaveAs(
                os.path.join(outdir, f"closure.{cat_key}.{mp}.png"))
            canvas.canv.Close()

            defined_pulls = {k: v for k, v in pulls.items() if v is not None}
            worst_param, worst_pull = (None, None)
            if defined_pulls:
                worst_param = max(defined_pulls,
                                  key=lambda k: abs(defined_pulls[k]))
                worst_pull = defined_pulls[worst_param]

            output[cat_key][mp] = {
                "mA": mA,
                "is_fit_point": mp in fit_mp,
                "extrapolation": extrapolation,
                "predicted": predicted,
                "clipped": clipped,
                "pulls": pulls,
                "x0_direct": x0_direct,
                "sigma_eff_direct": sigma_eff_direct,
                "sigma_eff_pred": sigma_eff_pred,
                "chi2_interp": chi2_interp,
                "ndf_interp": ndf_interp,
                "chi2_direct": chi2_direct,
                "ndf_direct": ndf_direct,
                "direct_quality": direct["quality"],
            }
            rows.append((cat_key, mA, extrapolation, worst_param, worst_pull,
                         chi2_interp / ndf_interp, chi2_direct / ndf_direct))

    if args.output:
        outpath = args.output
        os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    else:
        outpath = os.path.join(loo_dir or interp_dir, "closure.json")

    payload = {
        "meta": {
            "mhc": args.mhc,
            "all_ma": study["all"],
            "fit_ma": poly_payload["meta"]["fit_ma"],
            "held_out_ma": study["held_out"],
            "loo_ma": args.loo_ma,
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "closure": output,
        "warnings": warnings,
    }
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {outpath}\n")

    header = (f"{'category':<18} {'mA':>4} {'type':<8} {'worst pull':>22} "
              f"{'chi2/ndf interp':>16} {'chi2/ndf direct':>16}")
    print(header)
    print("-" * len(header))
    for cat_key, mA, extrap, wp, wpull, c2i, c2d in sorted(rows):
        kind = "extrap" if extrap else "interp"
        pull_txt = f"{wp}={wpull:+.2f}" if wp is not None else "n/a"
        print(f"{cat_key:<18} {mA:>4} {kind:<8} {pull_txt:>22} "
              f"{c2i:>16.2f} {c2d:>16.2f}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  {w}")


if __name__ == "__main__":
    main()
