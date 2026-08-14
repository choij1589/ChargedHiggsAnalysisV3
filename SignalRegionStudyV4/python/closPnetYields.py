#!/usr/bin/env python3
"""ParticleNet yield closure: template normalization = (Baseline window
yield) x (threshold efficiency).

The production model reuses the frozen Baseline yield model

    N_win(era, mA) = k_era * G_period(mA) * f_category(mA)

(fits/MHc{X}/yields/yield_model.json) and multiplies it by a threshold
efficiency measured on the ParticleNet grid, so only eps carries a new
interpolation. Each mHc offers three anchors -- mA = 85, 90, 95 -- all
scored by the SAME seed net, which is what a template-sharing group
actually uses.

Per (mHc, channel, era, seed, member) this measures, inside the SEED's
mass window at the frozen working point:

    N_nocut  full signal yield      -- what the Baseline model should predict
    N_cut    after the seed's cut   -- the actual template normalization
    eps      N_cut / N_nocut

and reports three residuals, so a discrepancy can be attributed:

    r_model = N_base_pred / N_nocut - 1        Baseline model -> PN dir transfer
    r_eps   = eps_interp  / eps    - 1         efficiency interpolation
    r_total = N_base_pred * eps_interp / N_cut - 1

eps_interp is built from the anchors ONLY, leaving the evaluated point out
whenever it is itself an anchor (leave-one-out), so the anchor points are a
genuine interpolation test and not an exact-by-construction fit. MA87/MA92
are never in the anchor set -- they are the blind validation points. The
r_eps residuals feed exportPnetUncertainties.py (CMS_interp_eff_pnet); the
PRODUCTION eps model (all anchors, no LOO) is exported separately by
fitPnetEpsModel.py from this script's measured eps values.

Output: closure/pnet/MHc{X}/yield_interp.json.

  python3 python/closPnetYields.py --mhc MHc115 [--wp 'epsB=20%']
"""
import argparse
import json
import os
from collections import OrderedDict

import numpy as np

import ROOT

import pnet_interp_config as pic
import srspaths
from dcb_fit_utils import fit_dcb_with_errors
from fitInterpYieldModel import predict_yield
from makeBinnedTemplates import (
    getCategoryBackgroundWeights, optimizeCategoryParticleNetThreshold)
from pnet_interp_config import ANCHOR_MA, DEFAULT_WP, PERIODS, STUDY_CHANNEL
from template_utils import build_particlenet_score

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError


def signal_chain(basedirs, masspoint):
    chain = ROOT.TChain("Central")
    for basedir in basedirs:
        path = os.path.join(basedir, f"{masspoint}.root")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        chain.Add(path)
    return chain


def window_yields(path, lo, hi, formula, threshold):
    """(sumw_nocut, sumw_cut) for one era's signal file in one window."""
    base = ROOT.RDataFrame("Central", path).Filter(
        f"mass >= {lo} && mass <= {hi}")
    nocut = base.Sum("weight")
    cut = (base.Define("score_PN", formula)
               .Filter(f"score_PN >= {threshold}")
               .Sum("weight"))
    return float(nocut.GetValue()), float(cut.GetValue())


def interp_eps(anchors, mA):
    """Interpolate eps(mA) from {mA: eps} anchors (3 -> quadratic, 2 ->
    linear for the leave-one-out case, 1 -> flat)."""
    coeffs, _deg = pic.fit_eps_anchors(anchors)
    return float(np.polyval(np.asarray(coeffs), mA))


def run_mhc(mhc, channels, args, frozen_wp, warnings):
    group = pic.trained_masspoints(mhc)
    results = OrderedDict()

    model_path = os.path.join(
        srspaths.interpolation_fits_dir(pic.mhc_int(mhc)),
        "yields", "yield_model.json")
    if not os.path.exists(model_path):
        warnings.append(f"[{mhc}] no Baseline yield model at {model_path}")
        print(f"WARNING: {warnings[-1]}")
        return results
    with open(model_path) as fh:
        model = json.load(fh)["model"]

    workdir = os.path.join(srspaths.pnet_closure_dir(mhc), "thresholds")

    for channel in channels:
        study_channel = STUDY_CHANNEL[channel]
        for period, suberas in PERIODS.items():
            cat = f"{channel}_{period}"
            basedirs = [srspaths.mhc_sample_dir(e, channel, mhc)
                        for e in suberas]
            missing = [d for d in basedirs if not os.path.isdir(d)]
            if missing:
                warnings.append(f"[{mhc}/{cat}] missing sample dirs: "
                                f"{missing}")
                print(f"WARNING: {warnings[-1]}")
                continue

            for seed in group:
                seed_mA = pic.mA_of(seed)
                key = f"{mhc}/{cat}/seed{seed_mA}"
                if frozen_wp:
                    # The window is a property of the UNCUT seed fit, so
                    # the stored one is the same number this would refit;
                    # reusing it also guarantees the shape and yield
                    # studies share a byte-identical cut.
                    rec = frozen_wp.get(key)
                    if rec is None:
                        warnings.append(f"[{key}] absent from the "
                                        f"{args.wp} shards; skipped")
                        print(f"WARNING: {warnings[-1]}")
                        continue
                    lo, hi = rec["mass_window"]
                    bg_weights = rec["bg_weights"]
                    thr = rec["threshold"]
                else:
                    os.makedirs(workdir, exist_ok=True)
                    try:
                        fit = fit_dcb_with_errors(
                            signal_chain(basedirs, seed), seed_mA)
                    except Exception as exc:                # noqa: BLE001
                        warnings.append(f"[{key}] seed fit failed: {exc}")
                        print(f"WARNING: {warnings[-1]}")
                        continue
                    x0 = fit["params"]["x0"]["value"]
                    sig = fit["sigma_eff"]
                    lo, hi = max(x0 - 10.0 * sig, 12.0), x0 + 10.0 * sig
                    bg_weights = getCategoryBackgroundWeights(
                        basedirs, lo, hi, workdir,
                        f"{mhc}_{cat}_seed{seed_mA}")
                    thr, payload = optimizeCategoryParticleNetThreshold(
                        basedirs, seed, lo, hi, bg_weights, workdir,
                        f"{mhc}_{cat}_seed{seed_mA}")
                    if payload is None:
                        warnings.append(f"[{key}] no threshold; skipped")
                        print(f"WARNING: {warnings[-1]}")
                        continue
                formula = build_particlenet_score(seed, bg_weights)

                # --- measure every member in every sub-era of the period
                meas = OrderedDict()
                for mp in group:
                    for era, basedir in zip(suberas, basedirs):
                        path = os.path.join(basedir, f"{mp}.root")
                        if not os.path.exists(path):
                            warnings.append(f"[{key}/{mp}/{era}] missing "
                                            f"{path}")
                            continue
                        n0, n1 = window_yields(path, lo, hi, formula, thr)
                        meas[(mp, era)] = {"n_nocut": n0, "n_cut": n1,
                                           "eps": (n1 / n0) if n0 else None}

                entry = OrderedDict([
                    ("mhc", mhc), ("channel", channel), ("period", period),
                    ("study_channel", study_channel),
                    ("seed", seed), ("seed_mA", seed_mA),
                    ("mass_window", [lo, hi]), ("threshold", float(thr)),
                    ("wp", args.wp),
                    ("points", OrderedDict()),
                ])

                for mp in group:
                    mA = pic.mA_of(mp)
                    is_anchor = mA in ANCHOR_MA
                    for era in suberas:
                        rec = meas.get((mp, era))
                        if rec is None or rec["eps"] is None:
                            continue
                        # Anchor set for eps: the trained mA, minus this
                        # point when it is itself an anchor
                        # (leave-one-out).
                        anchors = {}
                        for other in group:
                            o_mA = pic.mA_of(other)
                            if o_mA not in ANCHOR_MA:
                                continue
                            if is_anchor and o_mA == mA:
                                continue
                            orec = meas.get((other, era))
                            if orec and orec["eps"] is not None:
                                anchors[float(o_mA)] = orec["eps"]
                        if not anchors:
                            warnings.append(f"[{key}/{mp}/{era}] no eps "
                                            "anchors")
                            continue
                        eps_pred = interp_eps(anchors, float(mA))

                        n_base, err_base = predict_yield(
                            model, study_channel, era, float(mA))
                        r_model = (n_base / rec["n_nocut"] - 1.0) \
                            if rec["n_nocut"] else None
                        r_eps = eps_pred / rec["eps"] - 1.0
                        r_total = (n_base * eps_pred / rec["n_cut"] - 1.0) \
                            if rec["n_cut"] else None

                        entry["points"][f"{mp}/{era}"] = OrderedDict([
                            ("mA", mA), ("era", era),
                            ("is_anchor", is_anchor),
                            ("loo", is_anchor),
                            ("n_nocut", rec["n_nocut"]),
                            ("n_cut", rec["n_cut"]),
                            ("eps", rec["eps"]),
                            ("eps_pred", eps_pred),
                            ("n_base_pred", n_base),
                            ("err_base_pred", err_base),
                            ("r_model", r_model),
                            ("r_eps", r_eps),
                            ("r_total", r_total),
                            ("n_anchors", len(anchors)),
                        ])
                        print(f"  {key} {mp:16s} {era:11s} "
                              f"eps={rec['eps']:.4f} "
                              f"pred={eps_pred:.4f}  r_eps={r_eps:+.4f}  "
                              f"r_model={r_model:+.4f}  "
                              f"r_total={r_total:+.4f}")

                results[key] = entry
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", default="all",
                        help="comma-separated mHc studies, or 'all'")
    parser.add_argument("--channels", default="SR1E2Mu,SR3Mu")
    parser.add_argument("--output", default=None,
                        help="override output path (single --mhc only); "
                             "default closure/pnet/MHc{X}/yield_interp.json")
    parser.add_argument("--wp", default=DEFAULT_WP,
                        help="frozen working-point label to read from "
                             "fits/pnet/*/threshold_wp.json (default: the "
                             "production eps_B=20%%). Pass 'optimized' to "
                             "re-derive the sensitivity-optimized "
                             "threshold instead.")
    args = parser.parse_args()

    mhcs = (pic.pn_mhc_list() if args.mhc == "all"
            else [m.strip() for m in args.mhc.split(",") if m.strip()])
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    if args.output and len(mhcs) != 1:
        parser.error("--output requires a single --mhc")

    frozen_wp = {}
    if args.wp != "optimized":
        frozen_wp = pic.wp_lookup(args.wp, mhcs)
        if not frozen_wp:
            raise SystemExit(
                f"--wp {args.wp!r} not found in fits/pnet/*/threshold_wp.json "
                f"(labels present: {pic.wp_labels(mhcs) or 'none - run measPnetThresholds.py first'})")
        print(f"using frozen working point {args.wp} "
              f"({len(frozen_wp)} categories)")

    all_results = OrderedDict()
    for mhc in mhcs:
        warnings = []
        results = run_mhc(mhc, channels, args, frozen_wp, warnings)
        out = args.output or os.path.join(
            srspaths.pnet_closure_dir(mhc), "yield_interp.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump({"results": results, "warnings": warnings},
                      fh, indent=2)
        if warnings:
            print(f"{mhc}: {len(warnings)} warning(s)")
        print(f"Wrote {out}")
        all_results.update(results)

    # ---------------- summary ----------------
    def gather(pred):
        return [p for e in all_results.values()
                for p in e["points"].values() if pred(p)]

    print("\n" + "=" * 84)
    print("YIELD INTERPOLATION -- residual summary (|median| / p90 / max)")
    print("=" * 84)
    rows = [
        ("anchors (leave-one-out)", gather(lambda p: p["is_anchor"])),
        ("validation (MA87/MA92)", gather(lambda p: not p["is_anchor"])),
    ]
    for label, pts in rows:
        if not pts:
            print(f"{label:26s}  (none)")
            continue
        print(f"\n{label}  [{len(pts)} points]")
        for field in ("r_model", "r_eps", "r_total"):
            vals = np.abs([p[field] for p in pts if p[field] is not None])
            if not len(vals):
                continue
            print(f"  {field:9s} median={np.median(vals):7.4f}  "
                  f"p90={np.percentile(vals, 90):7.4f}  max={vals.max():7.4f}")


if __name__ == "__main__":
    main()
