#!/usr/bin/env python3
"""Derive interpolation-uncertainty nuisance sizes from the closure tests
and export them for the template-production step.

Two closure tests, two correlation models (user requirement,
docs/INTERPOLATION.md):

  - shape closure (closInterpShapes.py -> closure.json): interpolated vs
    direct-fit x0/sigma_eff at held-out mass points -> scale/res nuisances,
    CORRELATED WITHIN A RUN PERIOD (one shape parametrization per
    (channel, period), so every mA in a period shares it).
  - yield closure (closInterpYields.py -> yield_closure.json): predicted
    vs measured window yield at held-out mass points -> norm nuisance,
    DECORRELATED BETWEEN ERAS (the datacard's signal columns are per-era
    components).

Point selection (all three rules): held-out mass points only, restricted to
the production-relevant pairing variant (srspaths.pairing_variant) so
SR3Mu's lowM/highM study channels pool into the single production channel
"SR3Mu"; non-extrapolation; good-quality direct fit (shape) / a measured
scalar yield record (norm).

Envelope: the MAXIMUM over the selected held-out points, floored (conservative
approach, user decision 2026-08-11 — scale/res/norm all use the max, not an
RMS/median, even though for norm this absorbs the known Run3 per-sample
scatter into the exported nuisance size).

  python3 exportInterpUncertainties.py --mhc 160
  python3 exportInterpUncertainties.py --all
"""
import argparse
import datetime
import json
import os
import sys

import interpolation_config
import srspaths
from interpolation_config import masspoint_name


def _production_categories(mhc, held_out_ma, period):
    """For each held-out mA, (prod_channel, study_category_key) pairs
    relevant to production at this (mhc, mA, period)."""
    out = []
    for mA in held_out_ma:
        mp = masspoint_name(mA, mhc)
        out.append((mA, mp, "SR1E2Mu",
                    interpolation_config.category_key("SR1E2Mu", period)))
        sr3mu_channel = interpolation_config.study_channel_for("SR3Mu", mp)
        out.append((mA, mp, "SR3Mu",
                    interpolation_config.category_key(sr3mu_channel, period)))
    return out


def derive_shape_uncertainties(mhc, closure, held_out_ma, warnings):
    """(scale, res) = {prod_channel: {period: value}}, plus per-point
    detail for the audit trail."""
    scale, res, detail = {}, {}, {}
    for period, _eras in _run_periods():
        for mA, mp, prod_channel, cat_key in _production_categories(
                mhc, held_out_ma, period):
            rec = closure.get(cat_key, {}).get(mp)
            if rec is None:
                continue
            if rec.get("extrapolation") or rec.get("direct_quality") != "good":
                continue
            x0_direct = rec["x0_direct"]
            sigma_eff_direct = rec["sigma_eff_direct"]
            sigma_eff_pred = rec["sigma_eff_pred"]
            x0_pred = rec["predicted"]["x0"]
            if sigma_eff_direct <= 0:
                continue
            scale_pt = abs(x0_pred - x0_direct) / sigma_eff_direct
            res_pt = abs(sigma_eff_pred / sigma_eff_direct - 1.0)
            key = (prod_channel, period)
            detail.setdefault(key, {"scale": [], "res": []})
            detail[key]["scale"].append({"mA": mA, "value": scale_pt})
            detail[key]["res"].append({"mA": mA, "value": res_pt})

    for (prod_channel, period), pts in detail.items():
        scale_vals = [p["value"] for p in pts["scale"]]
        res_vals = [p["value"] for p in pts["res"]]
        if not scale_vals:
            warnings.append(f"[shape/{prod_channel}/{period}] no held-out "
                            "points survived selection; using the floor")
        scale.setdefault(prod_channel, {})[period] = max(
            [interpolation_config.UNCERTAINTY_SCALE_FLOOR] + scale_vals)
        res.setdefault(prod_channel, {})[period] = max(
            [interpolation_config.UNCERTAINTY_RES_FLOOR] + res_vals)
    return scale, res, detail


def derive_norm_uncertainties(mhc, yield_closure, held_out_ma, warnings):
    """norm = {prod_channel: {era: lnN}}, plus per-point detail."""
    norm, detail = {}, {}
    for period, eras in _run_periods():
        for mA, mp, prod_channel, _cat_key in _production_categories(
                mhc, held_out_ma, period):
            study_channel = ("SR1E2Mu" if prod_channel == "SR1E2Mu" else
                             interpolation_config.study_channel_for("SR3Mu", mp))
            entry = yield_closure.get(mp)
            if entry is None:
                continue
            for era in eras:
                rec = entry.get("scalar", {}).get(study_channel, {}).get(era)
                if rec is None or rec.get("extrapolation"):
                    continue
                key = (prod_channel, era)
                detail.setdefault(key, []).append(
                    {"mA": mA, "value": abs(rec["rel"])})

    for (prod_channel, era), pts in detail.items():
        vals = [p["value"] for p in pts]
        envelope = max([interpolation_config.UNCERTAINTY_NORM_FLOOR] + vals)
        if envelope > interpolation_config.UNCERTAINTY_NORM_WARN:
            warnings.append(
                f"[norm/{prod_channel}/{era}] envelope {envelope:.3f} exceeds "
                f"the {interpolation_config.UNCERTAINTY_NORM_WARN:g} warn "
                "threshold (expected for Run3: known per-sample scatter, or "
                "a sparse fit-grid gap)")
        norm.setdefault(prod_channel, {})[era] = 1.0 + envelope
    return norm, detail


def _run_periods():
    import run_period_utils
    return list(run_period_utils.RUN_PERIODS.items())


def build_nuisance_names(scale, res, norm):
    names = {"scale": {}, "res": {}, "norm": {}}
    for prod_channel, per_period in scale.items():
        for period in per_period:
            n = interpolation_config.interp_nuisance_names(prod_channel, period)
            names["scale"].setdefault(prod_channel, {})[period] = n["scale"]
    for prod_channel, per_period in res.items():
        for period in per_period:
            n = interpolation_config.interp_nuisance_names(prod_channel, period)
            names["res"].setdefault(prod_channel, {})[period] = n["res"]
    for prod_channel, per_era in norm.items():
        for era in per_era:
            period = interpolation_config.period_of(era)
            n = interpolation_config.interp_nuisance_names(
                prod_channel, period, era=era)
            names["norm"].setdefault(prod_channel, {})[era] = n["norm"]
    return names


def export_one(mhc):
    """Compute and write the per-mHc uncertainties.json; returns the
    (mhc_key, entry, input_paths) tuple for the consolidated export."""
    study = interpolation_config.study(mhc)
    interp_dir = srspaths.interpolation_dir(mhc)

    closure_path = os.path.join(interp_dir, "closure.json")
    with open(closure_path) as f:
        closure_payload = json.load(f)
    yield_closure_path = os.path.join(interp_dir, "yields", "yield_closure.json")
    with open(yield_closure_path) as f:
        yield_closure_payload = json.load(f)

    warnings = []
    scale, res, shape_detail = derive_shape_uncertainties(
        mhc, closure_payload["closure"], study["held_out"], warnings)
    norm, norm_detail = derive_norm_uncertainties(
        mhc, yield_closure_payload["closure"], study["held_out"], warnings)
    nuisances = build_nuisance_names(scale, res, norm)
    n_points = {
        "scale": {f"{ch}/{p}": len(pts["scale"])
                 for (ch, p), pts in shape_detail.items()},
        "res": {f"{ch}/{p}": len(pts["res"])
               for (ch, p), pts in shape_detail.items()},
        "norm": {f"{ch}/{e}": len(pts) for (ch, e), pts in norm_detail.items()},
    }

    entry = {
        "scale": scale, "res": res, "norm": norm,
        "nuisances": nuisances, "n_points": n_points,
    }
    detail = {
        "scale": {f"{ch}/{p}": pts["scale"] for (ch, p), pts in shape_detail.items()},
        "res": {f"{ch}/{p}": pts["res"] for (ch, p), pts in shape_detail.items()},
        "norm": {f"{ch}/{e}": pts for (ch, e), pts in norm_detail.items()},
    }

    payload = {
        "meta": {
            "mhc": mhc,
            "held_out_ma": study["held_out"],
            "rules": {
                "scale": "max(|x0_pred-x0_direct|/sigma_eff_direct) over "
                         f"held-out points, floor {interpolation_config.UNCERTAINTY_SCALE_FLOOR}",
                "res": "max(|sigma_eff_pred/sigma_eff_direct-1|) over "
                       f"held-out points, floor {interpolation_config.UNCERTAINTY_RES_FLOOR}",
                "norm": "max(|N_pred/N_meas-1|) over held-out points per era, "
                        f"floor {interpolation_config.UNCERTAINTY_NORM_FLOOR}, "
                        f"warn above {interpolation_config.UNCERTAINTY_NORM_WARN}",
            },
            "inputs": {"closure": closure_path,
                      "yield_closure": yield_closure_path},
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        **entry,
        "point_detail": detail,
        "warnings": warnings,
    }
    outpath = os.path.join(interp_dir, "uncertainties.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {outpath}")
    for w in warnings:
        print(f"  warning: {w}")
    return entry, {"closure": closure_path, "yield_closure": yield_closure_path}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", type=int, help="compute for one mHc")
    parser.add_argument("--all", action="store_true",
                        help="compute for every mHc in "
                             "configs/interpolation.json and write the "
                             "consolidated configs/interpolation_uncertainties.json")
    args = parser.parse_args()
    if not args.mhc and not args.all:
        parser.error("pass --mhc N and/or --all")

    if args.all:
        mhcs = sorted(int(k) for k in
                     srspaths.interpolation_config()["fit_points"])
    else:
        mhcs = [args.mhc]

    consolidated = {}
    inputs_meta = {}
    for mhc in mhcs:
        entry, inputs = export_one(mhc)
        consolidated[f"MHc{mhc}"] = entry
        inputs_meta[f"MHc{mhc}"] = inputs

    if args.all:
        payload = {
            "meta": {
                "rules": {
                    "scale": "max(|dx0|/sigma_eff) over held-out points per "
                             "(channel, period), floor "
                             f"{interpolation_config.UNCERTAINTY_SCALE_FLOOR}",
                    "res": "max(|dsigma_eff/sigma_eff|) over held-out points "
                           f"per (channel, period), floor "
                           f"{interpolation_config.UNCERTAINTY_RES_FLOOR}",
                    "norm": "max(|N_pred/N_meas-1|) over held-out points per "
                            f"(channel, era), floor "
                            f"{interpolation_config.UNCERTAINTY_NORM_FLOOR}",
                },
                "inputs": inputs_meta,
                "command": " ".join(sys.argv),
                "date": datetime.datetime.now().isoformat(timespec="seconds"),
            },
            **consolidated,
        }
        outpath = srspaths.interpolation_uncertainties_path()
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        with open(outpath, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote consolidated {outpath}")


if __name__ == "__main__":
    main()
