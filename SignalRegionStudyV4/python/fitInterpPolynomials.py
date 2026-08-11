#!/usr/bin/env python3
"""Stage 2 of the mA-interpolation chain: parametrization vs mA.

For each category and fit-model parameter, fit the designated fit points
(interpolation_config.study(mhc)["fit"]) whose stage-1 (frozen-n) fit
quality is good: error-weighted polynomials with per-parameter orders from
interpolation_config.POLY_ORDERS (single-entry = fixed order; multi-entry =
up-only F-test ladder), and a logit-space polynomial for fsig (turnover
capable; linear-space polynomial fallback below FSIG_LOGISTIC_MIN_POINTS).
Parameters frozen in the fits (fixed n) become constant records.

Outputs tests/interpolation/MHc{X}/polynomials.json and per-category
parameter-vs-mA cmsstyle PNGs under tests/interpolation/MHc{X}/plots/params/.

  python3 fitInterpPolynomials.py --mhc 160
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
from scipy import stats

import interp_plot_utils
import interpolation_config
import srspaths
from interpolation_config import ALL_PARAM_ORDER, F_TEST_PVALUE


def weighted_polyfit(x, y, err, deg):
    """Error-weighted least-squares polynomial fit.

    Returns dict with coeffs (numpy convention, highest degree first),
    covariance, chi2, ndf.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    err = np.asarray(err, dtype=float)
    if deg == 0:
        w = 1.0 / err**2
        mean = np.sum(w * y) / np.sum(w)
        var = 1.0 / np.sum(w)
        coeffs = np.array([mean])
        cov = np.array([[var]])
    else:
        coeffs, cov = np.polyfit(x, y, deg, w=1.0 / err, cov="unscaled")
    resid = (y - np.polyval(coeffs, x)) / err
    chi2 = float(np.sum(resid**2))
    ndf = len(x) - (deg + 1)
    return {"coeffs": coeffs.tolist(), "cov": np.asarray(cov).tolist(),
            "chi2": chi2, "ndf": ndf}


def select_order(x, y, err, orders):
    """Walk orders upward; accept a higher order only on F-test p < cut.

    If no requested order is feasible for the point count (need
    ndf >= 1, i.e. npoints >= deg+2), fall back to the highest feasible
    order. Single-entry order lists fit exactly that order.
    """
    if not any(len(x) >= deg + 2 for deg in orders):
        orders = [max(len(x) - 2, 0)]
    tried = {}
    chosen = None
    for deg in orders:
        if len(x) < deg + 2:  # need ndf >= 1
            break
        fit = weighted_polyfit(x, y, err, deg)
        tried[deg] = fit
        if chosen is None:
            chosen = deg
            continue
        prev = tried[chosen]
        d_dof = deg - chosen
        if fit["ndf"] <= 0 or fit["chi2"] <= 0:
            break
        f_stat = ((prev["chi2"] - fit["chi2"]) / d_dof) / (fit["chi2"] / fit["ndf"])
        p_value = 1.0 - stats.f.cdf(max(f_stat, 0.0), d_dof, fit["ndf"])
        fit["f_test_p_vs_lower"] = float(p_value)
        if p_value < F_TEST_PVALUE:
            chosen = deg
    return chosen, tried


def fit_logit_poly(x, y, err):
    """Adopted fsig form: F-test polynomial fit of logit(fsig) — kept in
    (0,1) but able to turn over (the true fsig falls again as mA -> mHc).
    Anchor points (fsig = 1, background dropped) are pinned at
    logit(1 - FSIG_LOGIT_CLIP) with a fixed logit-space error so the anchor
    region stays above the drop threshold without dominating the fit."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    err = np.asarray(err, dtype=float)
    eps = interpolation_config.FSIG_LOGIT_CLIP
    p = np.clip(y, eps, 1.0 - eps)
    z = np.log(p / (1.0 - p))
    zerr = err / (p * (1.0 - p))
    anchor = y >= interpolation_config.FSIG_DROP_THRESHOLD
    zerr[anchor] = interpolation_config.FSIG_LOGIT_ANCHOR_SIGMA
    chosen, tried = select_order(x, z, zerr,
                                 interpolation_config.FSIG_LOGIT_ORDERS)
    if chosen is None:
        return None
    result = dict(tried[chosen])
    result["form"] = "logitpoly"
    result["chosen_order"] = chosen
    result["orders_tried"] = {
        str(deg): {k: v for k, v in info.items() if k != "cov"}
        for deg, info in tried.items()}
    return result


def poly_band(coeffs, cov, xgrid):
    """1-sigma uncertainty band of a polynomial via its covariance."""
    deg = len(coeffs) - 1
    V = np.vander(xgrid, deg + 1)
    var = np.einsum("ij,jk,ik->i", V, np.asarray(cov), V)
    return np.sqrt(np.clip(var, 0.0, None))


def form_label(fit_info):
    form = fit_info.get("form")
    if form == "logitpoly":
        return f"logit-pol{fit_info['chosen_order']}"
    return f"pol{fit_info['chosen_order']}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", type=int, required=True,
                        help="mHc study to run")
    parser.add_argument("--exclude-ma", default="",
                        help="comma-separated mA values to drop from the fit")
    parser.add_argument("--suffix", default="",
                        help="appended to the output filename (e.g. '_ex90'), "
                             "so a leave-one-out refit does not clobber the "
                             "adopted polynomials.json")
    args = parser.parse_args()
    excluded = {int(m) for m in args.exclude_ma.split(",") if m.strip()}
    study = interpolation_config.study(args.mhc)
    fit_ma = study["fit"]
    all_ma = study["all"]
    interp_dir = srspaths.interpolation_dir(args.mhc)
    params_plot_base = srspaths.interpolation_plots_dir(args.mhc, "params")

    with open(os.path.join(interp_dir, "fits", "dcb_fits_frozen.json")) as f:
        fits = json.load(f)["results"]

    output = {}
    warnings = []
    for cat_key, cat_fits in sorted(fits.items()):
        output[cat_key] = {}
        cat_params = [p for p in ALL_PARAM_ORDER
                      if p in next(iter(cat_fits.values()))["params"]]
        for param in cat_params:
            # A parameter frozen in the fit (fixed n) becomes a constant
            # record: exact value, zero covariance.
            frozen = [fit["fixed_n"][param]
                      for fit in cat_fits.values()
                      if param in fit.get("fixed_n", {})]
            if frozen:
                output[cat_key][param] = {
                    "coeffs": [frozen[0]], "cov": [[0.0]],
                    "chi2": 0.0, "ndf": 0, "chosen_order": 0,
                    "mhc": args.mhc, "frozen": True,
                    "orders_tried": {}, "points_used":
                        {"mA": [], "value": [], "error": []},
                }
                continue

            used = {"mA": [], "value": [], "error": []}
            held = {"mA": [], "value": [], "error": []}
            for mp, fit in sorted(cat_fits.items(),
                                  key=lambda kv: kv[1]["mA"]):
                mA = fit["mA"]
                pv = fit["params"][param]
                good = fit["quality"] == "good" and pv["error"] > 0
                if mA in fit_ma and mA not in excluded and good:
                    used["mA"].append(mA)
                    used["value"].append(pv["value"])
                    used["error"].append(pv["error"])
                else:
                    if mA in fit_ma and (not good or mA in excluded):
                        warnings.append(
                            f"[{cat_key}] {param}: fit point mA={mA} dropped "
                            f"({'excluded' if mA in excluded else 'bad fit'})")
                    if pv["error"] > 0:
                        held["mA"].append(mA)
                        held["value"].append(pv["value"])
                        held["error"].append(pv["error"])

            result = None
            if (param == "fsig"
                    and len(used["mA"]) >= interpolation_config.FSIG_LOGISTIC_MIN_POINTS):
                result = fit_logit_poly(used["mA"], used["value"], used["error"])
            if result is None:
                chosen, tried = select_order(
                    used["mA"], used["value"], used["error"],
                    interpolation_config.POLY_ORDERS[param])
                if chosen is None:
                    warnings.append(f"[{cat_key}] {param}: no parametrization "
                                    "possible (too few points); skipped")
                    continue
                result = dict(tried[chosen])
                result["chosen_order"] = chosen
                result["orders_tried"] = {
                    str(deg): {k: v for k, v in info.items() if k != "cov"}
                    for deg, info in tried.items()}
            result["mhc"] = args.mhc
            result["points_used"] = used
            output[cat_key][param] = result

            channel, period = cat_key.rsplit("_", 1)
            interp_plot_utils.plot_parameter_vs_mA(
                cat_key, param, {"used": used, "held_out": held}, result,
                os.path.join(params_plot_base, cat_key), all_ma, args.mhc, period)

    payload = {
        "meta": {
            "mhc": args.mhc,
            "fit_ma": [m for m in fit_ma if m not in excluded],
            "excluded_ma": sorted(excluded),
            "f_test_pvalue": F_TEST_PVALUE,
            "poly_orders": {p: interpolation_config.POLY_ORDERS[p]
                            for p in ALL_PARAM_ORDER},
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "polynomials": output,
        "warnings": warnings,
    }
    os.makedirs(interp_dir, exist_ok=True)
    outpath = os.path.join(interp_dir, f"polynomials{args.suffix}.json")
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {outpath}")
    print("\nChosen parametrizations (chi2/ndf):")
    for cat_key, params in sorted(output.items()):
        row = ", ".join(
            f"{p}: {form_label(info)} ({info['chi2']:.1f}/{info['ndf']})"
            for p, info in params.items())
        print(f"  {cat_key}: {row}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  {w}")


if __name__ == "__main__":
    main()
