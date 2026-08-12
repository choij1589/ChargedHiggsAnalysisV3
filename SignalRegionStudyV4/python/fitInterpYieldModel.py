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
from scipy import stats

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


def fsig_of(polys, cat_key, mA, use_fsig=True):
    """Interpolated fsig clipped to (0,1]; 1.0 when the category has no
    background component, or when the pairing decomposition is run on the
    raw window fractions (yield variant 'joint')."""
    if not use_fsig:
        return 1.0
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


def fit_fractions(yields, polys, period, eras, fit_ma, warnings, orders,
                  use_fsig=True):
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
        fsl = fsig_of(polys, f"SR3Mu_lowM_{period}", m, use_fsig)
        fsh = fsig_of(polys, f"SR3Mu_highM_{period}", m, use_fsig)
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


def joint_design(mhc, mA, dh, da):
    """Total-degree-truncated tensor basis in scaled (mHc, mA)."""
    mh0, mhs = interpolation_config.JOINT_G_MHC_SCALE
    ma0, mas = interpolation_config.JOINT_G_MA_SCALE
    u = (np.asarray(mhc, float) - mh0) / mhs
    v = (np.asarray(mA, float) - ma0) / mas
    cols, powers = [], []
    for i in range(dh + 1):
        for j in range(da + 1):
            if i + j > da:
                continue
            cols.append(u ** i * v ** j)
            powers.append((i, j))
    return np.vstack(cols).T, powers


def slice_surface(coeffs, cov, powers, mhc, da):
    """Collapse the (mHc, mA) surface at fixed mHc into a plain polynomial
    in mA (numpy descending convention) with a propagated covariance.

    The slice of a polynomial surface is a polynomial, so the sliced record
    is drop-in compatible with the adopted per-mHc G record: eval_rec and
    rec_band keep working unchanged downstream.
    """
    mh0, mhs = interpolation_config.JOINT_G_MHC_SCALE
    ma0, mas = interpolation_config.JOINT_G_MA_SCALE
    u = (float(mhc) - mh0) / mhs
    base = np.array([1.0 / mas, -ma0 / mas])   # v as a polynomial in mA
    kmat = np.zeros((da + 1, len(powers)))
    for k, (i, j) in enumerate(powers):
        vj = np.array([1.0])
        for _ in range(j):
            vj = np.polymul(vj, base)
        kmat[da + 1 - len(vj):, k] = (u ** i) * vj
    beta = kmat @ np.asarray(coeffs)
    return beta, kmat @ np.asarray(cov) @ kmat.T


def fit_joint_G(joint_data, period, tot_channel, mhc, err):
    """One surface across every mHc study, sliced at this study's mHc."""
    dh = interpolation_config.JOINT_G_MHC_DEGREE
    da = interpolation_config.JOINT_G_MA_DEGREE
    pts = joint_data[(period, tot_channel)]
    if len(pts) < 3 * (da + 1):
        raise RuntimeError(f"[{period}/{tot_channel}] joint G surface needs "
                           f"more points than {len(pts)}")
    mh, ma, logn = pts[:, 0], pts[:, 1], pts[:, 2]
    amat, powers = joint_design(mh, ma, dh, da)
    coeffs, *_ = np.linalg.lstsq(amat, logn, rcond=None)
    resid = amat @ coeffs - logn
    cov = err * err * np.linalg.pinv(amat.T @ amat)
    beta, beta_cov = slice_surface(coeffs, cov, powers, mhc, da)
    here = mh == float(mhc)
    return {
        "coeffs": [float(c) for c in beta],
        "cov": [[float(c) for c in row] for row in beta_cov],
        "chosen_order": da,
        "chi2": float(((resid[here] / err) ** 2).sum()),
        "ndf": int(here.sum()),
        "joint_surface": {
            "mhc_degree": dh, "ma_degree": da,
            "n_points": int(len(pts)), "n_params": int(len(coeffs)),
            "mhc_values": sorted({int(v) for v in mh}),
            "chi2_all": float(((resid / err) ** 2).sum()),
            "ndf_all": int(len(pts) - len(coeffs)),
            "rms_resid_rel": float(np.sqrt((resid ** 2).mean())),
            "coeffs": [float(c) for c in coeffs],
            "powers": [[int(i), int(j)] for i, j in powers],
        },
    }


def fit_k_era(shares, ma_fit, orders):
    """Era share vs mA: F-tested pol0/pol1, error = the observed scatter.

    The adopted model quotes std/sqrt(N) — the error on the MEAN, which
    understates the predictive error for a single mass point by sqrt(N).
    """
    x = np.asarray(ma_fit, float)
    y = np.asarray(shares, float)
    chosen, coeffs = None, None
    for deg in sorted(orders):
        if len(x) < deg + 2:
            continue
        c = np.polyfit(x, y, deg)
        rss = float(((np.polyval(c, x) - y) ** 2).sum())
        if chosen is None:
            chosen, coeffs, prev_rss = deg, c, rss
            continue
        ndf = len(x) - (deg + 1)
        if ndf <= 0 or rss <= 0:
            break
        f_stat = ((prev_rss - rss) / (deg - chosen)) / (rss / ndf)
        p_value = 1.0 - stats.f.cdf(max(f_stat, 0.0), deg - chosen, ndf)
        if p_value < interpolation_config.F_TEST_PVALUE:
            chosen, coeffs, prev_rss = deg, c, rss
    resid = np.polyval(coeffs, x) - y
    mean = float(y.mean())
    scatter = float(np.sqrt((resid ** 2).mean()))
    return {"value": mean,
            "err_rel": scatter / mean,
            "coeffs": [float(c) for c in coeffs],
            "chosen_order": int(chosen)}


def k_value(tot, era, mA):
    """Era share at mA, renormalized so the period's shares sum to one."""
    rec = tot["k"][era]
    if "coeffs" not in rec:
        return rec["value"]
    total = sum(float(np.polyval(np.asarray(r["coeffs"]), mA))
                for r in tot["k"].values())
    return float(np.polyval(np.asarray(rec["coeffs"]), mA)) / total


def fit_totals(yields, period, eras, fit_ma, warnings, orders,
               joint_data=None, mhc=None, k_orders=None):
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
        if joint_data is not None:
            g_rec = fit_joint_G(joint_data, period, tot_channel, mhc, g_err)
        else:
            g_rec = fit_record(ma_fit, np.log([g_pts[m] for m in ma_fit]),
                               [g_err] * len(ma_fit), orders["G"])
            if g_rec is None:
                raise RuntimeError(f"[{period}/{tot_channel}] G fit failed")
        g_rec.setdefault("points_used", {})
        g_rec["points_used"] = {"x": [float(m) for m in ma_fit],
                                "y": [float(np.log(g_pts[m])) for m in ma_fit],
                                "err": [g_err] * len(ma_fit)}
        k = {}
        for era in eras:
            shares = [per_ma[m][era]["sumw_total"] / g_pts[m] for m in ma_fit]
            if k_orders:
                k[era] = fit_k_era(shares, ma_fit, k_orders)
            else:
                s = np.array(shares)
                k[era] = {"value": float(s.mean()),
                          "err_rel": float(s.std(ddof=1) / np.sqrt(len(s))
                                           / s.mean())}
        out[tot_channel] = {"G": g_rec, "k": k}
    return out


def load_joint_totals(loo_mhc=None, loo_ma=None):
    """Period-summed totals of EVERY mHc study, for the joint G surface.

    Reads each study's adopted yields.json. In leave-one-out mode the
    excluded point is dropped from the study it belongs to only — the
    other studies keep their full grids, which is the whole point of
    borrowing shape across mHc.
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
                rows = out.setdefault((period, tot_channel), [])
                for rec in res.values():
                    if mhc == loo_mhc and rec["mA"] == loo_ma:
                        continue
                    era_rows = [rec["channels"].get(src, {}).get(e)
                                for e in eras]
                    if any(r is None for r in era_rows):
                        continue
                    rows.append((float(mhc), float(rec["mA"]),
                                 float(np.log(sum(r["sumw_total"]
                                                  for r in era_rows)))))
    return {k: np.array(v) for k, v in out.items()}


def predict_yield(model, polys, channel, era, mA):
    """(N_pred, err_pred) of the physics model for one era x channel."""
    period = interpolation_config.period_of(era)
    use_fsig = model.get("options", {}).get("pairing_fsig", True)
    fr = model["fractions"][period]
    if channel == "SR1E2Mu":
        f = float(eval_rec(fr["f_sr1e2mu"], mA))
        f_relerr = float(rec_band(fr["f_sr1e2mu"], mA)[0]) / f
    else:
        S = float(eval_rec(fr["S"], mA))
        z = float(eval_rec(fr["p_high_logit"], mA))
        p_high = float(inv_logit(z))
        p = p_high if channel == "SR3Mu_highM" else 1.0 - p_high
        f = S * p / fsig_of(polys, f"{channel}_{period}", mA, use_fsig)
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
    parser.add_argument("--exclude-ma", default="",
                        help="comma-separated mA values to drop from every "
                             "sub-model fit (leave-one-out); requires --suffix")
    parser.add_argument("--suffix", default="",
                        help="appended to the output filename (e.g. '_ex90'), "
                             "so an anchor-exclusion refit does not clobber "
                             "the adopted yield_model.json")
    parser.add_argument("--loo-ma", type=int, default=None,
                        help="leave-one-out mode: fit anchors = full grid "
                             "minus this mA; reads the LOO polynomials from "
                             "and writes yield_model.json to the per-point "
                             "dir tests/interpolation/MHc{X}_MA{Y}/")
    parser.add_argument("--yield-variant", default=None,
                        help="yield-model variant test "
                             f"({'|'.join(sorted(interpolation_config.YIELD_VARIANTS))}); "
                             "outputs go to the variant tree, shape "
                             "polynomials still come from the adopted one")
    args = parser.parse_args()

    excluded = {int(m) for m in args.exclude_ma.split(",") if m.strip()}
    if args.loo_ma is not None and (excluded or args.suffix):
        raise ValueError("--loo-ma is a complete mode of its own; "
                         "do not combine with --exclude-ma/--suffix")
    if args.yield_variant is not None and (excluded or args.suffix):
        raise ValueError("--yield-variant does not combine with "
                         "--exclude-ma/--suffix")
    vcfg = (interpolation_config.yield_variant_config(args.yield_variant)
            if args.yield_variant else {})
    use_fsig = vcfg.get("pairing_fsig", True)
    k_orders = vcfg.get("k_era_orders")
    study = interpolation_config.study(args.mhc, loo_ma=args.loo_ma)
    fit_ma = [m for m in study["fit"] if m not in excluded]
    orders = interpolation_config.YIELD_ORDERS
    if excluded and not args.suffix:
        raise ValueError("--exclude-ma would overwrite the adopted "
                         "yield_model.json; pass a --suffix "
                         "(e.g. the matching leave-one-out shape suffix)")
    # Measured yields and shape polynomials always come from the adopted
    # tree (or its per-point LOO dirs): a yield variant changes the model,
    # not the measurement or the shape chain.
    yields_dir = os.path.join(srspaths.interpolation_dir(args.mhc), "yields")
    if args.loo_ma is not None:
        out_base = srspaths.interpolation_loo_dir(args.mhc, args.loo_ma,
                                                  variant=args.yield_variant)
        out_dir = os.path.join(out_base, "yields")
        plot_base = os.path.join(out_base, "plots", "yields")
    else:
        out_dir = os.path.join(
            srspaths.interpolation_dir(args.mhc, variant=args.yield_variant),
            "yields")
        plot_base = srspaths.interpolation_plots_dir(
            args.mhc, "yields", variant=args.yield_variant)

    with open(os.path.join(yields_dir, "yields.json")) as f:
        yields_payload = json.load(f)
    yields = yields_payload["results"]
    shape_suffix = args.suffix if args.exclude_ma else ""
    polys, polys_path = interpolation_config.load_shape_polynomials(
        args.mhc, shape_suffix, loo_ma=args.loo_ma)

    joint_data = (load_joint_totals(args.mhc, args.loo_ma)
                  if vcfg.get("joint_G") else None)

    model = {"fractions": {}, "totals": {},
             "options": {"pairing_fsig": use_fsig,
                         "joint_G": bool(vcfg.get("joint_G")),
                         "k_era_orders": k_orders}}
    warnings = []
    merged_by_period = {}
    for period, eras in run_period_utils.RUN_PERIODS.items():
        model["fractions"][period], merged = fit_fractions(
            yields, polys, period, list(eras), fit_ma, warnings, orders,
            use_fsig)
        model["totals"][period] = fit_totals(
            yields, period, list(eras), fit_ma, warnings, orders,
            joint_data=joint_data, mhc=args.mhc, k_orders=k_orders)
        merged_by_period[period] = merged

    def fsig_fn(cat_polys, cat_key, mA):
        return fsig_of(cat_polys, cat_key, mA, use_fsig)

    for period in model["fractions"]:
        interp_plot_utils.plot_yield_period_model(
            args.mhc, period, model, polys, merged_by_period[period],
            fit_ma, plot_base, eval_rec, rec_band, inv_logit, fsig_fn)
    for channel in interpolation_config.STUDY_CHANNELS:
        interp_plot_utils.plot_yield_era_grid(
            args.mhc, channel, yields, model, polys, fit_ma, plot_base,
            predict_yield)

    payload = {
        "meta": {
            "mhc": args.mhc,
            "fit_ma": fit_ma,
            "excluded_ma": sorted(excluded),
            "loo_ma": args.loo_ma,
            "yield_variant": args.yield_variant,
            "model": "k_era * G_period(mA) * f_category(mA); "
                     + ("f_SR3Mu = S * p_pairing"
                        if not use_fsig else
                        "f_SR3Mu = S * p_pairing / fsig")
                     + ("; G = joint (mHc,mA) surface" if joint_data
                        else ""),
            "orders": orders,
            "rel_yield_err_floor": interpolation_config.REL_YIELD_ERR_FLOOR,
            "shape_polynomials": polys_path,
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "model": model,
        "warnings": warnings,
    }
    os.makedirs(out_dir, exist_ok=True)
    outpath = os.path.join(out_dir, f"yield_model{args.suffix}.json")
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
