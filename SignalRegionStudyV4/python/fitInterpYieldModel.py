#!/usr/bin/env python3
"""Stage 2 of the yield-interpolation chain: the physics-structured yield
model.

  N_win(era, mA) = k_era * G_period(mA) * f_category(mA)

  - G: per-period total yield, a log-space SURFACE in (mHc, mA) fitted
    across ALL seven studies and sliced at this mHc. One G per
    (total-channel x period), total-channel in {SR1E2Mu, SR3Mu} (the two
    pairings share identical totals). Summing the four eras first averages
    the per-sample normalization scatter.
  - k_era: era share, a plane in (mHc, mA) also pooled across studies —
    the shares carry a real smooth mHc drift, and pooling averages the
    per-sample noise over 78 points instead of one study's 6-23. Predicted
    shares are renormalized to sum to one over a period.
  - f: window fraction per category, a per-study 1D fit. SR1E2Mu: pol0/1 on
    the period-merged fraction. SR3Mu: the pairing decomposition on the RAW
    measured fractions, S = f_low + f_high and p_low + p_high = 1, with S a
    low-order poly and p_high fitted in logit space.

**Reads every study's yields/yields.json** for the surfaces, so all seven
must have completed stage 1 of this chain before this step runs (see
automize/interpolation.sh --stop-after).

Outputs tests/interpolation/MHc{X}/yields/yield_model.json, per-period
model component plots and per-(channel,era) measured-vs-model plots under
tests/interpolation/MHc{X}/plots/yields/.

  python3 fitInterpYieldModel.py --mhc 160
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

import interp_plot_utils
import interpolation_config
import run_period_utils
import srspaths
from fitInterpPolynomials import poly_band, select_order

TOTAL_CHANNEL = {"SR1E2Mu": "SR1E2Mu",
                 "SR3Mu_lowM": "SR3Mu", "SR3Mu_highM": "SR3Mu"}


def logit(p):
    return np.log(np.asarray(p) / (1.0 - np.asarray(p)))


def inv_logit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z)))


def fit_record(x, y, err, orders):
    """F-test poly fit returning a record with points_used attached."""
    chosen, tried = select_order(np.asarray(x, float), np.asarray(y, float),
                                 np.asarray(err, float), orders)
    if chosen is None:
        return None
    rec = dict(tried[chosen])
    rec["chosen_order"] = chosen
    rec["orders_tried"] = {
        str(deg): {k: v for k, v in info.items() if k != "cov"}
        for deg, info in tried.items()}
    rec["points_used"] = {"x": list(map(float, x)),
                          "y": list(map(float, y)),
                          "err": list(map(float, err))}
    return rec


def eval_rec(rec, x):
    return np.polyval(np.asarray(rec["coeffs"]), x)


def rec_band(rec, x):
    return poly_band(np.array(rec["coeffs"]), np.array(rec["cov"]),
                     np.atleast_1d(np.asarray(x, float)))


def period_merged_fraction(yields, channel, eras):
    """mA -> merged window-fraction point over the available eras."""
    out = {}
    for mp, rec in yields.items():
        rows = [rec["channels"].get(channel, {}).get(e) for e in eras]
        rows = [r for r in rows if r is not None]
        if not rows:
            continue
        sw = sum(r["sumw"] for r in rows)
        st = sum(r["sumw_total"] for r in rows)
        et = float(np.hypot.reduce([r["err_total"] for r in rows]))
        f = sw / st
        n_eff = (st / et) ** 2
        ferr = max(float(np.sqrt(max(f * (1 - f), 1e-12) / n_eff)),
                   interpolation_config.YIELD_F_ABS_ERR_FLOOR)
        out[rec["mA"]] = {"f": f, "ferr": ferr, "n_eras": len(rows)}
    return out


def fit_fractions(yields, period, eras, fit_ma, warnings, orders):
    """Fraction sub-model of one run period: f_sr1e2mu, S, p_high.

    The SR3Mu pairing decomposition is taken on the RAW measured window
    fractions: S = f_low + f_high, p_high = f_high/S. Dividing by the shape
    fit's fsig first is an exact reparametrization that buys no smoothness
    (helps 0, hurts 1, mixed 13 of 14 datasets) and coupled this model to
    the shape chain, which pure-DCB lowM cannot support."""
    merged = {ch: period_merged_fraction(yields, ch, eras)
              for ch in interpolation_config.STUDY_CHANNELS}

    ma1 = sorted(m for m in merged["SR1E2Mu"] if m in fit_ma)
    f_rec = fit_record(ma1, [merged["SR1E2Mu"][m]["f"] for m in ma1],
                       [merged["SR1E2Mu"][m]["ferr"] for m in ma1],
                       orders["f_sr1e2mu"])

    common = sorted(set(merged["SR3Mu_lowM"]) & set(merged["SR3Mu_highM"]))
    s_pts, p_pts = [], []
    for m in [m for m in common if m in fit_ma]:
        lo, hi = merged["SR3Mu_lowM"][m], merged["SR3Mu_highM"][m]
        ql, qh = lo["f"], hi["f"]
        sql, sqh = lo["ferr"], hi["ferr"]
        S = ql + qh
        p = qh / S
        s_pts.append((m, S, max(float(np.hypot(sql, sqh)), 0.003)))
        p_err = max(float(np.hypot(ql * sqh, qh * sql) / S**2), 0.004)
        p_pts.append((m, logit(p), p_err / (p * (1 - p))))
    s_rec = fit_record(*zip(*[(m, v, e) for m, v, e in s_pts]),
                       orders=orders["S"])
    p_rec = fit_record(*zip(*[(m, v, e) for m, v, e in p_pts]),
                       orders=orders["p_high_logit"])
    if f_rec is None or s_rec is None or p_rec is None:
        raise RuntimeError(f"[{period}] fraction sub-model fit failed")
    return {"f_sr1e2mu": f_rec, "S": s_rec, "p_high_logit": p_rec}, merged


def period_totals(yields, period, eras, src, fit_ma, warnings, tag):
    """mA -> {era: row} for points with a complete set of era samples."""
    per_ma = {}
    for mp, rec in yields.items():
        rows = {era: rec["channels"].get(src, {}).get(era) for era in eras}
        rows = {e: r for e, r in rows.items() if r is not None}
        if len(rows) < len(eras):
            if rec["mA"] in fit_ma:
                warnings.append(f"[{tag}] mA={rec['mA']} dropped from G fit "
                                "(missing era sample)")
            continue
        per_ma[rec["mA"]] = rows
    return per_ma


def fit_joint_G(joint_data, period, tot_channel, mhc, err):
    """Period total-yield surface across every study, sliced at this mHc."""
    pts = joint_data[(period, tot_channel)]["totals"]
    degrees = interpolation_config.JOINT_G_DEGREES
    if len(pts) < 3 * (degrees[1] + 1):
        raise RuntimeError(f"[{period}/{tot_channel}] joint G surface needs "
                           f"more points than {len(pts)}")
    return interpolation_config.fit_surface(
        pts[:, 0], pts[:, 1], pts[:, 2], np.full(len(pts), err), degrees, mhc)


def fit_k_surface(joint_data, period, tot_channel, era, mhc):
    """Era share as a plane in (mHc, mA), pooled across every study and
    sliced at this study's mHc.

    The shares drift smoothly with mHc, so four constants per study both
    miss a real trend and carry the per-sample noise of that study's 6-23
    points alone. The quoted error is the SCATTER of this study's points
    about the surface — the predictive error for one mass point, unlike the
    old std/sqrt(N), which is the error on the mean and understates it by
    sqrt(N) = 2.4-4.8.
    """
    degrees = interpolation_config.JOINT_K_DEGREES
    block = joint_data[(period, tot_channel)]
    pts, shares = block["totals"], block["shares"][era]
    rec = interpolation_config.fit_surface(
        pts[:, 0], pts[:, 1], shares, np.ones(len(shares)), degrees, mhc)
    here = pts[:, 0] == float(mhc)
    amat, _ = interpolation_config.joint_design(pts[:, 0], pts[:, 1], degrees)
    coeffs = np.array(rec["joint_surface"]["coeffs"])
    resid = (amat @ coeffs - shares)[here]
    mean = float(shares[here].mean())
    rec["value"] = mean
    rec["err_rel"] = float(np.sqrt((resid ** 2).mean()) / mean)
    return rec


def k_value(tot, era, mA):
    """Era share at mA, renormalized so the period's shares sum to one."""
    rec = tot["k"][era]
    if "coeffs" not in rec:
        return rec["value"]
    total = sum(float(np.polyval(np.asarray(r["coeffs"]), mA))
                for r in tot["k"].values())
    return float(np.polyval(np.asarray(rec["coeffs"]), mA)) / total


def fit_totals(yields, period, eras, fit_ma, warnings, joint_data, mhc):
    """Total sub-model of one run period: per total-channel G + k_era."""
    floor = interpolation_config.REL_YIELD_ERR_FLOOR[period]
    # The period sum averages the eras' independent sample normalizations.
    g_err = floor / 2.0
    out = {}
    for tot_channel, src in (("SR1E2Mu", "SR1E2Mu"),
                             ("SR3Mu", "SR3Mu_lowM")):
        per_ma = period_totals(yields, period, eras, src, fit_ma, warnings,
                               f"{period}/{tot_channel}")
        ma_fit = sorted(m for m in per_ma if m in fit_ma)
        g_pts = {m: sum(r["sumw_total"] for r in per_ma[m].values())
                 for m in ma_fit}
        g_rec = fit_joint_G(joint_data, period, tot_channel, mhc, g_err)
        g_rec.setdefault("points_used", {})
        g_rec["points_used"] = {"x": [float(m) for m in ma_fit],
                                "y": [float(np.log(g_pts[m])) for m in ma_fit],
                                "err": [g_err] * len(ma_fit)}
        k = {era: fit_k_surface(joint_data, period, tot_channel, era, mhc)
             for era in eras}
        out[tot_channel] = {"G": g_rec, "k": k}
    return out


def load_joint_totals(loo_mhc=None, loo_ma=None):
    """Period totals and era shares of EVERY mHc study, for the joint
    G and k_era surfaces.

    Reads each study's adopted yields.json. In leave-one-out mode the
    excluded point is dropped from the study it belongs to only — the
    other studies keep their full grids, which is the whole point of
    borrowing shape across mHc.

    Returns {(period, total-channel): {"totals": [(mHc, mA, log N)],
                                       "shares": {era: [N_era/N]}}}
    with the rows of "totals" and every "shares" list index-aligned.
    """
    out = {}
    for mhc in interpolation_config.mhc_grid():
        path = os.path.join(srspaths.interpolation_dir(mhc), "yields",
                            "yields.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found — the joint yield surface needs the "
                "measured yields of every mHc study")
        with open(path) as f:
            res = json.load(f)["results"]
        for period, eras in run_period_utils.RUN_PERIODS.items():
            for tot_channel, src in (("SR1E2Mu", "SR1E2Mu"),
                                     ("SR3Mu", "SR3Mu_lowM")):
                block = out.setdefault((period, tot_channel),
                                       {"totals": [],
                                        "shares": {e: [] for e in eras}})
                for rec in res.values():
                    if mhc == loo_mhc and rec["mA"] == loo_ma:
                        continue
                    era_rows = {e: rec["channels"].get(src, {}).get(e)
                                for e in eras}
                    if any(r is None for r in era_rows.values()):
                        continue
                    total = sum(r["sumw_total"] for r in era_rows.values())
                    block["totals"].append((float(mhc), float(rec["mA"]),
                                            float(np.log(total))))
                    for e in eras:
                        block["shares"][e].append(
                            era_rows[e]["sumw_total"] / total)
    return {k: {"totals": np.array(v["totals"]),
                "shares": {e: np.array(s) for e, s in v["shares"].items()}}
            for k, v in out.items()}


def predict_yield(model, channel, era, mA):
    """(N_pred, err_pred) of the physics model for one era x channel."""
    period = interpolation_config.period_of(era)
    fr = model["fractions"][period]
    if channel == "SR1E2Mu":
        f = float(eval_rec(fr["f_sr1e2mu"], mA))
        f_relerr = float(rec_band(fr["f_sr1e2mu"], mA)[0]) / f
    else:
        S = float(eval_rec(fr["S"], mA))
        z = float(eval_rec(fr["p_high_logit"], mA))
        p_high = float(inv_logit(z))
        p = p_high if channel == "SR3Mu_highM" else 1.0 - p_high
        f = S * p
        # logit-space band sz -> sigma_p = sz*p_high*(1-p_high); relative
        # error of the used p is sigma_p / p.
        sz = float(rec_band(fr["p_high_logit"], mA)[0])
        f_relerr = float(np.hypot(rec_band(fr["S"], mA)[0] / S,
                                  sz * p_high * (1.0 - p_high) / p))
    tot = model["totals"][period][TOTAL_CHANNEL[channel]]
    g = float(np.exp(eval_rec(tot["G"], mA)))
    g_relerr = float(rec_band(tot["G"], mA)[0])   # log-space band ~ rel
    k = tot["k"][era]
    n = k_value(tot, era, mA) * g * f
    relerr = float(np.hypot.reduce([g_relerr, k["err_rel"], f_relerr]))
    return n, n * relerr


def fit_range(model, period):
    ma = model["totals"][period]["SR3Mu"]["G"]["points_used"]["x"]
    return min(ma), max(ma)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", type=int, required=True,
                        help="mHc study to run")
    parser.add_argument("--loo-ma", type=int, default=None,
                        help="leave-one-out mode: drop this study's mA from "
                             "the joint surfaces and the per-study fits "
                             "(other studies keep their full grids); outputs "
                             "go to tests/interpolation/MHc{X}_MA{Y}/")
    args = parser.parse_args()

    study = interpolation_config.study(args.mhc, loo_ma=args.loo_ma)
    fit_ma = study["fit"]
    orders = interpolation_config.YIELD_ORDERS
    yields_dir = os.path.join(srspaths.interpolation_dir(args.mhc), "yields")
    if args.loo_ma is not None:
        out_base = srspaths.interpolation_loo_dir(args.mhc, args.loo_ma)
        out_dir = os.path.join(out_base, "yields")
        plot_base = os.path.join(out_base, "plots", "yields")
    else:
        out_dir = yields_dir
        plot_base = srspaths.interpolation_plots_dir(args.mhc, "yields")

    with open(os.path.join(yields_dir, "yields.json")) as f:
        yields = json.load(f)["results"]
    joint_data = load_joint_totals(args.mhc, args.loo_ma)

    model = {"fractions": {}, "totals": {}}
    warnings = []
    merged_by_period = {}
    for period, eras in run_period_utils.RUN_PERIODS.items():
        model["fractions"][period], merged = fit_fractions(
            yields, period, list(eras), fit_ma, warnings, orders)
        model["totals"][period] = fit_totals(
            yields, period, list(eras), fit_ma, warnings, joint_data,
            args.mhc)
        merged_by_period[period] = merged

    for period in model["fractions"]:
        interp_plot_utils.plot_yield_period_model(
            args.mhc, period, model, merged_by_period[period],
            fit_ma, plot_base, eval_rec, rec_band, inv_logit)
    for channel in interpolation_config.STUDY_CHANNELS:
        interp_plot_utils.plot_yield_era_grid(
            args.mhc, channel, yields, model, fit_ma, plot_base,
            predict_yield)

    payload = {
        "meta": {
            "mhc": args.mhc,
            "fit_ma": fit_ma,
            "loo_ma": args.loo_ma,
            "model": "k_era * G_period(mA) * f_category(mA), with G and "
                     "k_era (mHc, mA) surfaces sliced at this mHc and "
                     "f_SR3Mu = S * p_pairing on the raw window fractions",
            "orders": orders,
            "surface_degrees": {
                "G": list(interpolation_config.JOINT_G_DEGREES),
                "k_era": list(interpolation_config.JOINT_K_DEGREES)},
            "mhc_pooled": interpolation_config.mhc_grid(),
            "rel_yield_err_floor": interpolation_config.REL_YIELD_ERR_FLOOR,
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "model": model,
        "warnings": warnings,
    }
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, "yield_model.json")
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {outpath}\n")
    for period in model["fractions"]:
        fr = model["fractions"][period]
        tots = model["totals"][period]
        print(f"{period}: "
              f"f1e2u pol{fr['f_sr1e2mu']['chosen_order']}"
              f"({fr['f_sr1e2mu']['chi2']:.1f}/{fr['f_sr1e2mu']['ndf']}), "
              f"S pol{fr['S']['chosen_order']}"
              f"({fr['S']['chi2']:.1f}/{fr['S']['ndf']}), "
              f"p_high logit-pol{fr['p_high_logit']['chosen_order']}"
              f"({fr['p_high_logit']['chi2']:.1f}/{fr['p_high_logit']['ndf']})"
              " | G: "
              + ", ".join(
                  f"{ch} surf-deg{t['G']['chosen_order']}"
                  f"({t['G']['chi2']:.1f}/{t['G']['ndf']})"
                  for ch, t in tots.items()))
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  {w}")


if __name__ == "__main__":
    main()
