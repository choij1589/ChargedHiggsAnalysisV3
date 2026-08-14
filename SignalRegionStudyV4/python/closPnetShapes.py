#!/usr/bin/env python3
"""ParticleNet shape closure: is the Baseline signal mass shape reusable
after the score cut?

The nets are trained with the dimuon mass decorrelated, so the expectation
is that a score cut leaves the signal mass peak alone -- that is what lets
the frozen Baseline shape surfaces (fits/MHc{X}/polynomials.json) be reused
verbatim for ParticleNet templates, with only the NORMALIZATION carrying a
new model.

For every (mHc, channel, run period) category this fits the same DCB twice
per mass point -- once on the full signal, once after the SEED's score cut
at the frozen working point -- and reports the shifts

    d_scale = (x0_cut - x0_nocut) / sigma_eff_nocut
    d_res   = (sigma_eff_cut - sigma_eff_nocut) / sigma_eff_nocut

in the same units the Baseline interpolation uses for CMS_interp_scale/res.
The residuals feed exportPnetUncertainties.py (the CMS_interp_res_pnet
family; scale was judged refit noise at Gate U1 -- UNCERTAINTY.md).

The full seed x member matrix is scanned, not just the diagonal: a member
is selected by its SEED's net, so how the shape moves as the evaluated mass
walks away from the net's training mass is exactly the quantity at issue.
(Consumers restrict to |dmA| <= 2.5, the production-relevant pairs.)

Output: closure/pnet/MHc{X}/shape_reuse.json.

  python3 python/closPnetShapes.py --mhc MHc115 [--wp 'epsB=20%']
"""
import argparse
import json
import os
import shutil
import tempfile
from collections import OrderedDict

import ROOT

import pnet_interp_config as pic
import srspaths
from dcb_fit_utils import fit_dcb_with_errors
from makeBinnedTemplates import (
    getCategoryBackgroundWeights, optimizeCategoryParticleNetThreshold)
from pnet_interp_config import DEFAULT_WP, PERIODS
from template_utils import build_particlenet_score

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError


def signal_chain(basedirs, masspoint):
    """TChain('Central') over one mass point across a period's sub-eras."""
    chain = ROOT.TChain("Central")
    n = 0
    for basedir in basedirs:
        path = os.path.join(basedir, f"{masspoint}.root")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        chain.Add(path)
        n += 1
    if n == 0 or chain.GetEntries() <= 0:
        raise RuntimeError(f"empty chain for {masspoint}")
    return chain


def cut_chain(basedirs, masspoint, seed, threshold, bg_weights, tmpdir, tag):
    """TChain over the same signal AFTER the seed's ParticleNet cut.

    Snapshots mass/weight of the surviving events so the downstream fit is
    literally fit_dcb_with_errors on a 'Central' tree -- the cut and no-cut
    fits then differ by the selection alone, never by fit configuration.
    The cut mirrors binned_template_core: score_PN >= threshold, with the
    score built by the production helper from the SEED's branches."""
    out = os.path.join(tmpdir, f"cut_{tag}.root")
    formula = build_particlenet_score(seed, bg_weights)

    chain = ROOT.TChain("Central")
    for basedir in basedirs:
        chain.Add(os.path.join(basedir, f"{masspoint}.root"))

    rdf = (ROOT.RDataFrame(chain)
           .Define("score_PN", formula)
           .Filter(f"score_PN >= {threshold}"))
    cols = ROOT.std.vector("string")(["mass", "weight"])
    rdf.Snapshot("Central", out, cols)

    cut = ROOT.TChain("Central")
    cut.Add(out)
    return cut, out


def summarize(fit):
    return {
        "x0": fit["params"]["x0"]["value"],
        "x0_err": fit["params"]["x0"]["error"],
        "sigma_eff": fit["sigma_eff"],
        "sumw": fit["sumw"],
        "entries": fit["entries"],
        "status": fit["status"],
    }


def run_mhc(mhc, channels, args, frozen_wp, tmpdir, warnings):
    group = pic.trained_masspoints(mhc)
    results = OrderedDict()
    workdir = os.path.join(srspaths.pnet_closure_dir(mhc), "thresholds")

    for channel in channels:
        for period, suberas in PERIODS.items():
            cat = f"{channel}_{period}"
            basedirs = [srspaths.mhc_sample_dir(e, channel, mhc)
                        for e in suberas]
            missing = [d for d in basedirs if not os.path.isdir(d)]
            if missing:
                warnings.append(f"[{mhc}/{cat}] missing sample dirs: {missing}")
                print(f"WARNING: {warnings[-1]}")
                continue

            # No-cut fits are seed-independent: once per mass point.
            nocut = {}
            for mp in group:
                try:
                    nocut[mp] = fit_dcb_with_errors(
                        signal_chain(basedirs, mp), pic.mA_of(mp))
                except Exception as exc:                    # noqa: BLE001
                    warnings.append(f"[{mhc}/{cat}/{mp}] no-cut fit "
                                    f"failed: {exc}")
                    print(f"WARNING: {warnings[-1]}")

            for seed in group:
                if seed not in nocut:
                    continue
                seed_mA = pic.mA_of(seed)
                key = f"{mhc}/{cat}/seed{seed_mA}"
                # Window and threshold come from the SEED, exactly as a
                # template-sharing group is built in production.
                if frozen_wp:
                    rec = frozen_wp.get(key)
                    if rec is None:
                        warnings.append(f"[{key}] absent from the "
                                        f"{args.wp} shards; skipped")
                        print(f"WARNING: {warnings[-1]}")
                        continue
                    lo, hi = rec["mass_window"]
                    bg_weights = rec["bg_weights"]
                    thr = rec["threshold"]
                    payload = {"label": args.wp, "eff_bkg": rec["eff_bkg"]}
                else:
                    os.makedirs(workdir, exist_ok=True)
                    x0 = nocut[seed]["params"]["x0"]["value"]
                    sig = nocut[seed]["sigma_eff"]
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

                entry = OrderedDict([
                    ("mhc", mhc), ("channel", channel), ("period", period),
                    ("seed", seed), ("seed_mA", seed_mA),
                    ("mass_window", [lo, hi]),
                    ("bg_weights", bg_weights),
                    ("threshold", float(thr)),
                    ("wp", args.wp),
                    ("eff_bkg", payload.get("eff_bkg")),
                    ("sensitivity_gain",
                     payload.get("max_sensitivity", 0.0)
                     / payload["initial_sensitivity"]
                     if payload.get("initial_sensitivity") else None),
                    ("members", OrderedDict()),
                ])

                for mp in group:
                    if mp not in nocut:
                        continue
                    tag = f"{mhc}_{cat}_s{seed_mA}_m{pic.mA_of(mp)}"
                    try:
                        cut, tmp = cut_chain(basedirs, mp, seed, thr,
                                             bg_weights, tmpdir, tag)
                        fit_c = fit_dcb_with_errors(cut, pic.mA_of(mp))
                    except Exception as exc:                # noqa: BLE001
                        warnings.append(f"[{key}/{mp}] cut fit failed: {exc}")
                        print(f"WARNING: {warnings[-1]}")
                        continue
                    finally:
                        tmp_path = os.path.join(tmpdir, f"cut_{tag}.root")
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

                    f0 = nocut[mp]
                    d_scale = ((fit_c["params"]["x0"]["value"]
                                - f0["params"]["x0"]["value"])
                               / f0["sigma_eff"])
                    d_res = ((fit_c["sigma_eff"] - f0["sigma_eff"])
                             / f0["sigma_eff"])
                    eff = (fit_c["sumw"] / f0["sumw"]) if f0["sumw"] else None

                    entry["members"][mp] = OrderedDict([
                        ("mA", pic.mA_of(mp)),
                        ("d_scale_sigma", float(d_scale)),
                        ("d_res_rel", float(d_res)),
                        ("eff_window_sumw", float(eff) if eff else None),
                        ("nocut", summarize(f0)),
                        ("cut", summarize(fit_c)),
                    ])
                    print(f"  {key} m{pic.mA_of(mp):3d}  thr={thr:.3f}  "
                          f"d_scale={d_scale:+.4f} sig  d_res={d_res:+.4f}  "
                          f"eff={eff if eff is not None else float('nan'):.4f}")

                results[key] = entry
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", default="all",
                        help="comma-separated mHc studies, or 'all'")
    parser.add_argument("--channels", default="SR1E2Mu,SR3Mu")
    parser.add_argument("--output", default=None,
                        help="override output path (single --mhc only); "
                             "default closure/pnet/MHc{X}/shape_reuse.json")
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

    # A --wp run MUST find its shards: silently falling back to the
    # optimized threshold would mix two different cuts into one summary.
    frozen_wp = {}
    if args.wp != "optimized":
        frozen_wp = pic.wp_lookup(args.wp, mhcs)
        if not frozen_wp:
            raise SystemExit(
                f"--wp {args.wp!r} not found in fits/pnet/*/threshold_wp.json "
                f"(labels present: {pic.wp_labels(mhcs) or 'none - run measPnetThresholds.py first'})")
        print(f"using frozen working point {args.wp} "
              f"({len(frozen_wp)} categories)")

    tmpdir = tempfile.mkdtemp(prefix="pnet_shapes_")
    try:
        for mhc in mhcs:
            warnings = []
            results = run_mhc(mhc, channels, args, frozen_wp, tmpdir,
                              warnings)
            out = args.output or os.path.join(
                srspaths.pnet_closure_dir(mhc), "shape_reuse.json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with open(out, "w") as fh:
                json.dump({"results": results, "warnings": warnings},
                          fh, indent=2)
            if results:
                scales = [abs(m["d_scale_sigma"]) for e in results.values()
                          for m in e["members"].values()]
                reses = [abs(m["d_res_rel"]) for e in results.values()
                         for m in e["members"].values()]
                print(f"{mhc}: max |d_scale| = {max(scales):.4f} sigma_eff"
                      f"   max |d_res| = {max(reses):.4f}   "
                      f"(full seed x member matrix)")
            if warnings:
                print(f"{mhc}: {len(warnings)} warning(s)")
            print(f"Wrote {out}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
