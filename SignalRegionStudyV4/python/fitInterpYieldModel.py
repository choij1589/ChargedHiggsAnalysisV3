#!/usr/bin/env python3
"""Stage 2 of the yield-interpolation chain: the physics-structured yield
model.

  N_win(era, mA) = k_era * G_period(mA) * f_category(mA)

  - G: shared per-period baseline-selection yield shape, log-space poly
    (YIELD_G_ORDERS) fitted on the period-summed totals; the sum over four
    eras averages the per-sample normalization scatter. One G per
    (total-channel x period), total-channel in {SR1E2Mu, SR3Mu} (the two
    pairings share identical totals).
  - k_era: constant era share (era shares were shown flat in mA).
  - f: window fraction per category. SR1E2Mu: pol0/1 on the period-merged
    fraction. SR3Mu: derived from the shape fit's fsig via the pairing
    decomposition f_variant = S * p_variant / fsig with p_low + p_high = 1;
    S (shared containment) is a low-order poly, p_high is fitted in logit
    space.

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


def fsig_of(polys, cat_key, mA):
    """Interpolated fsig clipped to (0,1]; 1.0 when the category has no
    background component."""
    rec = polys[cat_key].get("fsig")
    if rec is None:
        return 1.0
    return float(np.clip(interpolation_config.eval_param(rec, mA), 1e-3, 1.0))


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


def fit_fractions(yields, polys, period, eras, fit_ma, warnings, orders):
    """Fraction sub-model of one run period: f_sr1e2mu, S, p_high."""
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
        fsl = fsig_of(polys, f"SR3Mu_lowM_{period}", m)
        fsh = fsig_of(polys, f"SR3Mu_highM_{period}", m)
        ql, qh = lo["f"] * fsl, hi["f"] * fsh
        sql, sqh = lo["ferr"] * fsl, hi["ferr"] * fsh
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


def fit_totals(yields, period, eras, fit_ma, warnings, orders):
    """Total sub-model of one run period: per total-channel G + k_era."""
    floor = interpolation_config.REL_YIELD_ERR_FLOOR[period]
    out = {}
    for tot_channel, src in (("SR1E2Mu", "SR1E2Mu"),
                             ("SR3Mu", "SR3Mu_lowM")):
        per_ma = {}
        for mp, rec in yields.items():
            rows = {era: rec["channels"].get(src, {}).get(era)
                    for era in eras}
            rows = {e: r for e, r in rows.items() if r is not None}
            if len(rows) < len(eras):
                if rec["mA"] in fit_ma:
                    warnings.append(
                        f"[{period}/{tot_channel}] mA={rec['mA']} dropped "
                        f"from G fit (missing era sample)")
                continue
            per_ma[rec["mA"]] = rows
        ma_fit = sorted(m for m in per_ma if m in fit_ma)
        g_pts = {m: sum(r["sumw_total"] for r in per_ma[m].values())
                 for m in ma_fit}
        # Period sum averages four independent sample normalizations.
        g_err = floor / 2.0
        g_rec = fit_record(ma_fit, np.log([g_pts[m] for m in ma_fit]),
                           [g_err] * len(ma_fit), orders["G"])
        if g_rec is None:
            raise RuntimeError(f"[{period}/{tot_channel}] G fit failed")
        k = {}
        for era in eras:
            shares = np.array([per_ma[m][era]["sumw_total"] / g_pts[m]
                               for m in ma_fit])
            k[era] = {"value": float(shares.mean()),
                      "err_rel": float(shares.std(ddof=1)
                                       / np.sqrt(len(shares))
                                       / shares.mean())}
        out[tot_channel] = {"G": g_rec, "k": k}
    return out


def predict_yield(model, polys, channel, era, mA):
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
        f = S * p / fsig_of(polys, f"{channel}_{period}", mA)
        # logit-space band sz -> sigma_p = sz*p_high*(1-p_high); relative
        # error of the used p is sigma_p / p.
        sz = float(rec_band(fr["p_high_logit"], mA)[0])
        f_relerr = float(np.hypot(rec_band(fr["S"], mA)[0] / S,
                                  sz * p_high * (1.0 - p_high) / p))
    tot = model["totals"][period][TOTAL_CHANNEL[channel]]
    g = float(np.exp(eval_rec(tot["G"], mA)))
    g_relerr = float(rec_band(tot["G"], mA)[0])   # log-space band ~ rel
    k = tot["k"][era]
    n = k["value"] * g * f
    relerr = float(np.hypot.reduce([g_relerr, k["err_rel"], f_relerr]))
    return n, n * relerr


def fit_range(model, period):
    ma = model["totals"][period]["SR3Mu"]["G"]["points_used"]["x"]
    return min(ma), max(ma)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", type=int, required=True,
                        help="mHc study to run")
    parser.add_argument("--exclude-ma", default="",
                        help="comma-separated mA values to drop from every "
                             "sub-model fit (leave-one-out); requires --suffix")
    parser.add_argument("--suffix", default="",
                        help="appended to the output filename (e.g. '_ex90'), "
                             "so a leave-one-out refit does not clobber the "
                             "adopted yield_model.json")
    args = parser.parse_args()

    study = interpolation_config.study(args.mhc)
    excluded = {int(m) for m in args.exclude_ma.split(",") if m.strip()}
    fit_ma = [m for m in study["fit"] if m not in excluded]
    orders = interpolation_config.YIELD_ORDERS
    if excluded and not args.suffix:
        raise ValueError("--exclude-ma would overwrite the adopted "
                         "yield_model.json; pass a --suffix "
                         "(e.g. the matching leave-one-out shape suffix)")
    yields_dir = os.path.join(srspaths.interpolation_dir(args.mhc), "yields")
    plot_base = srspaths.interpolation_plots_dir(args.mhc, "yields")

    with open(os.path.join(yields_dir, "yields.json")) as f:
        yields_payload = json.load(f)
    yields = yields_payload["results"]
    shape_suffix = args.suffix if args.exclude_ma else ""
    polys, polys_path = interpolation_config.load_shape_polynomials(
        args.mhc, shape_suffix)

    model = {"fractions": {}, "totals": {}}
    warnings = []
    merged_by_period = {}
    for period, eras in run_period_utils.RUN_PERIODS.items():
        model["fractions"][period], merged = fit_fractions(
            yields, polys, period, list(eras), fit_ma, warnings, orders)
        model["totals"][period] = fit_totals(
            yields, period, list(eras), fit_ma, warnings, orders)
        merged_by_period[period] = merged

    for period in model["fractions"]:
        interp_plot_utils.plot_yield_period_model(
            args.mhc, period, model, polys, merged_by_period[period],
            fit_ma, plot_base, eval_rec, rec_band, inv_logit, fsig_of)
    for channel in interpolation_config.STUDY_CHANNELS:
        interp_plot_utils.plot_yield_era_grid(
            args.mhc, channel, yields, model, polys, fit_ma, plot_base,
            predict_yield)

    payload = {
        "meta": {
            "mhc": args.mhc,
            "fit_ma": fit_ma,
            "excluded_ma": sorted(excluded),
            "model": "k_era * G_period(mA) * f_category(mA); "
                     "f_SR3Mu = S * p_pairing / fsig",
            "orders": orders,
            "rel_yield_err_floor": interpolation_config.REL_YIELD_ERR_FLOOR,
            "shape_polynomials": polys_path,
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "model": model,
        "warnings": warnings,
    }
    outpath = os.path.join(yields_dir, f"yield_model{args.suffix}.json")
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
                  f"{ch} pol{t['G']['chosen_order']}"
                  f"({t['G']['chi2']:.1f}/{t['G']['ndf']})"
                  for ch, t in tots.items()))
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  {w}")


if __name__ == "__main__":
    main()
