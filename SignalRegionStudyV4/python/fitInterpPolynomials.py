#!/usr/bin/env python3
"""Stage 2 of the mA-interpolation chain: parametrization vs mA.

Every shape parameter is fitted as ONE error-weighted surface in (mHc, mA)
across ALL seven mHc studies and sliced at this study's mHc; fsig is fitted
in logit space. Parameters frozen in the fits (fixed n) become constant
records, per study.

Fitting across studies is what makes the sparse low-mA grids usable: a
per-study polynomial swings by +-20% when one point moves, and no 1D basis
fixes it, while borrowing the shape across mHc halves the worst-case scale
error. Interpolation is still in mA only. Because the slice of a surface is
a polynomial in mA, polynomials.json keeps its previous record shape and no
downstream consumer had to change.

**Reads every study's fits/dcb_fits.json**, so all seven must have completed
stage 1 before this step runs (see automize/interpolation.sh --stop-after).

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
        try:
            coeffs, cov = np.polyfit(x, y, deg, w=1.0 / err, cov="unscaled")
        except np.linalg.LinAlgError as exc:
            # Almost always one anchor with a collapsed error dominating the
            # weights (see credible_error). Report the offending inputs
            # instead of a bare "Singular matrix" from deep inside numpy.
            spread = [f"mA={a}: {v:.6g} +- {e:.3g}" for a, v, e in zip(x, y, err)]
            raise RuntimeError(
                f"weighted_polyfit(deg={deg}) hit a singular matrix. Anchors:\n  "
                + "\n  ".join(spread)) from exc
    resid = (y - np.polyval(coeffs, x)) / err
    chi2 = float(np.sum(resid**2))
    ndf = len(x) - (deg + 1)
    return {"coeffs": coeffs.tolist(), "cov": np.asarray(cov).tolist(),
            "chi2": chi2, "ndf": ndf}


def credible_error(pv):
    """False when a parameter's error is too small to be a real uncertainty.

    A collapsed Hesse error (seen down to 1e-13 on an O(0.1) coefficient)
    would enter the weighted parametrization fit with weight 1/err ~ 1e13 and
    make the design matrix singular. The relative test is only applied where
    it is defined; a value consistent with zero is left to the caller's
    error > 0 check.
    """
    value, err = pv["value"], pv["error"]
    if not abs(value) > 0:
        return True
    return err / abs(value) >= interpolation_config.MIN_REL_PARAM_ERROR


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


def load_joint_shape_fits(loo_mhc=None, loo_ma=None):
    """{cat_key: {param: (mhc[], mA[], value[], error[])}} across EVERY study.

    In leave-one-out mode the excluded point is dropped from the study it
    belongs to only — the other studies keep their full grids, which is the
    whole point of borrowing the shape across mHc.
    """
    raw = {}
    for mhc in interpolation_config.mhc_grid():
        path = os.path.join(srspaths.interpolation_dir(mhc), "fits",
                            "dcb_fits.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found — the joint shape surfaces need the "
                "frozen-n fits of every mHc study (run stage 1 for all of "
                "them first)")
        with open(path) as f:
            fits = json.load(f)["results"]
        for cat_key, cat_fits in fits.items():
            for fit in cat_fits.values():
                if mhc == loo_mhc and fit["mA"] == loo_ma:
                    continue
                if fit["quality"] != "good":
                    continue
                for param, pv in fit["params"].items():
                    if pv["error"] <= 0 or not credible_error(pv):
                        continue
                    raw.setdefault(cat_key, {}).setdefault(param, []).append(
                        (float(mhc), float(fit["mA"]), pv["value"],
                         pv["error"]))
    # -> (mhc[], mA[], value[], error[]) per (category, parameter)
    return {cat: {p: np.array(pts, dtype=float).T
                  for p, pts in params.items()}
            for cat, params in raw.items()}


def fit_param_surface(points, param, slice_at):
    """Surface record for one (category, parameter), sliced at slice_at.

    fsig is fitted in logit space: bounded in (0,1) and able to turn over,
    since the true fsig rises past the naive-logistic plateau and falls
    again as mA -> mHc where the two OS pairings converge.
    """
    mhc, mA, value, error = points
    if param in interpolation_config.FSIG_LOGIT_PARAMS:
        eps = 1e-6
        p = np.clip(value, eps, 1.0 - eps)
        y = np.log(p / (1.0 - p))
        err = error / (p * (1.0 - p))
    else:
        y, err = value, error
    rec = interpolation_config.fit_surface(
        mhc, mA, y, err, interpolation_config.SHAPE_SURFACE_DEGREES, slice_at)
    if param in interpolation_config.FSIG_LOGIT_PARAMS:
        rec["form"] = "logitpoly"
    return rec


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
                        help="mHc study to slice the surfaces at")
    parser.add_argument("--loo-ma", type=int, default=None,
                        help="leave-one-out mode: drop this study's mA from "
                             "the joint fit (other studies keep their full "
                             "grids); outputs go to the per-point dir "
                             "tests/interpolation/MHc{X}_MA{Y}/")
    args = parser.parse_args()

    study = interpolation_config.study(args.mhc, loo_ma=args.loo_ma)
    fit_ma = study["fit"]
    all_ma = study["all"]
    interp_dir = srspaths.interpolation_dir(args.mhc)
    out_dir = (srspaths.interpolation_loo_dir(args.mhc, args.loo_ma)
               if args.loo_ma is not None else interp_dir)
    params_plot_base = os.path.join(out_dir, "plots", "params")

    with open(os.path.join(interp_dir, "fits", "dcb_fits.json")) as f:
        local_fits = json.load(f)["results"]
    joint = load_joint_shape_fits(loo_mhc=args.mhc, loo_ma=args.loo_ma)

    output = {}
    warnings = []
    degenerate = []
    for cat_key, cat_fits in sorted(local_fits.items()):
        output[cat_key] = {}
        if not cat_fits:
            # Every mass point in this category failed to build a signal
            # chain (missing samples): report it rather than dying on the
            # empty iterator below.
            warnings.append(f"[{cat_key}] no fit records; category skipped")
            continue
        cat_params = [p for p in ALL_PARAM_ORDER
                      if p in next(iter(cat_fits.values()))["params"]]
        for param in cat_params:
            # A parameter frozen in the fit (fixed n) becomes a constant
            # record: exact value, zero covariance. Frozen n is a per-study
            # median, so it stays per-study.
            frozen = [fit["fixed_n"][param]
                      for fit in cat_fits.values()
                      if param in fit.get("fixed_n", {})]
            if frozen:
                output[cat_key][param] = {
                    "coeffs": [frozen[0]], "cov": [[0.0]],
                    "chi2": 0.0, "ndf": 0, "chosen_order": 0,
                    "mhc": args.mhc, "frozen": True,
                    "points_used": {"mA": [], "value": [], "error": []},
                }
                continue

            points = joint.get(cat_key, {}).get(param)
            if points is None or len(points[0]) < 3 * (
                    interpolation_config.SHAPE_SURFACE_DEGREES[1] + 1):
                n = 0 if points is None else len(points[0])
                degenerate.append(
                    f"[{cat_key}] {param}: only {n} usable point(s) across "
                    "all studies; the surface is underdetermined")
                continue

            result = fit_param_surface(points, param, args.mhc)

            # This study's own points, for the plot and the audit trail.
            mhc_arr, mA_arr, val_arr, err_arr = points
            here = mhc_arr == float(args.mhc)
            used = {"mA": [float(v) for v in mA_arr[here]],
                    "value": [float(v) for v in val_arr[here]],
                    "error": [float(v) for v in err_arr[here]]}
            held = {"mA": [], "value": [], "error": []}
            for mp, fit in sorted(cat_fits.items(), key=lambda kv: kv[1]["mA"]):
                if fit["mA"] in used["mA"]:
                    continue
                pv = fit["params"].get(param)
                if pv and pv["error"] > 0:
                    held["mA"].append(fit["mA"])
                    held["value"].append(pv["value"])
                    held["error"].append(pv["error"])
                if fit["mA"] in fit_ma and fit["quality"] != "good":
                    warnings.append(f"[{cat_key}] {param}: fit point "
                                    f"mA={fit['mA']} dropped (bad fit)")
            result["mhc"] = args.mhc
            result["points_used"] = used
            output[cat_key][param] = result

            channel, period = cat_key.rsplit("_", 1)
            interp_plot_utils.plot_parameter_vs_mA(
                cat_key, param, {"used": used, "held_out": held}, result,
                os.path.join(params_plot_base, cat_key), all_ma, args.mhc,
                period)

        # Structural invariant: dcb_fit_utils.build_model reads c1/c2 whenever
        # fsig is present, so a category carrying fsig without both background
        # coefficients is an incomplete model. It would survive this step and
        # then kill every downstream consumer with a bare KeyError.
        if "fsig" in output[cat_key]:
            absent = [p for p in interpolation_config.BKG_PARAMS
                      if p not in output[cat_key]]
            if absent:
                degenerate.append(
                    f"[{cat_key}] carries fsig but no {','.join(absent)}: "
                    "the DCB+background model is incomplete and build_model "
                    "would fail downstream")

    if args.loo_ma is not None:
        # Leak check: the excluded point must not have anchored anything,
        # here or anywhere in the joint fit.
        leaked = [(cat_key, p) for cat_key, params in output.items()
                  for p, info in params.items()
                  if args.loo_ma in info.get("points_used", {}).get("mA", [])]
        for cat_key, params in joint.items():
            for p, (mh, ma, _v, _e) in params.items():
                if np.any((mh == float(args.mhc)) & (ma == float(args.loo_ma))):
                    leaked.append((cat_key, p, "joint"))
        if leaked:
            raise RuntimeError(
                f"LOO leak: excluded mA={args.loo_ma} entered the fit for "
                f"{leaked}")

    payload = {
        "meta": {
            "mhc": args.mhc,
            "fit_ma": fit_ma,
            "loo_ma": args.loo_ma,
            "model": "one (mHc, mA) surface per parameter across all studies, "
                     "sliced at this mHc; fsig in logit space",
            "surface_degrees": list(interpolation_config.SHAPE_SURFACE_DEGREES),
            "mhc_pooled": interpolation_config.mhc_grid(),
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "polynomials": output,
        "warnings": warnings,
        "degenerate": degenerate,
    }
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, "polynomials.json")
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {outpath}")
    print("\nSliced surfaces (chi2 over this study's points / n points):")
    for cat_key, params in sorted(output.items()):
        row = ", ".join(
            f"{p}: {form_label(info)} ({info['chi2']:.1f}/{info['ndf']})"
            for p, info in params.items())
        print(f"  {cat_key}: {row}")
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  {w}")

    if degenerate:
        # The payload is written first so the failure is diagnosable, but the
        # step must not report success: a missing or underdetermined
        # parametrization produces mass windows that miss the peak entirely,
        # and the bad yields flow into the exported nuisance sizes.
        raise RuntimeError(
            f"{len(degenerate)} degenerate parametrization(s) for mHc="
            f"{args.mhc}:\n  " + "\n  ".join(degenerate)
            + f"\n\nWrote {outpath} for inspection.")


if __name__ == "__main__":
    main()
