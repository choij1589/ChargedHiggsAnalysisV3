#!/usr/bin/env python3
"""Stage 1 of the shape-delta chain: systematic shape deltas.

A parametric signal template has no MC histogram to vary, so every shape
systematic must be expressed as a shift of the fit function. For each
(sub-era, study channel, systematic, direction) this script compresses the
systematic tree into three dimensionless numbers relative to the Central
tree of the same sample:

    dm   = <m>_var / <m>_cen - 1     core window x0 +- 2 sigma_eff
    dsig = rms_var / rms_cen - 1     core window
    dN   = sumw_var / sumw_cen - 1   full +-10 sigma template window

Windows come from the INTERPOLATED shape parametrization, so the recipe is
identical at donor points and at a target point that has no MC.

Weight-only systematic trees hold the same events as Central (verified:
identical entry counts and masses), so their deltas are paired differences
with much smaller errors than a naive quadrature estimate; kinematic trees
migrate events across the selection and fall back to the uncorrelated
error. Both are floored (DELTA_ERR_FLOOR).

Runs as per-masspoint condor jobs via the interpolation wrapper.

  python3 measInterpShapeDeltas.py --mhc 160 [--masspoints MP] [--output F]
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np
import ROOT

import interpolation_config
import srspaths
from interpolation_config import masspoint_name

from closInterpYields import predict_shape_params
from makeBinnedTemplates import load_systematics_block
from template_utils import categorize_systematics, iter_shape_directions

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError


def signal_trees(era, prod_channel):
    """(tree_name, syst_name, direction) for every signal-group shape tree
    of one (sub-era, channel), in production's own order."""
    cats = categorize_systematics(load_systematics_block(era, prod_channel))
    out = []
    for syst, variations, group in cats["preprocessed_shape"]:
        if "signal" not in group:
            continue
        for direction in iter_shape_directions(variations):
            out.append((f"{syst}_{direction}", syst, direction))
    for syst, variations, group in cats["multi_variation"]:
        if "signal" not in group:
            continue
        for var in variations:
            tree = (f"PDF_{int(var.replace('pdf_', ''))}"
                    if var.startswith("pdf_") else var)
            out.append((tree, syst, var))
    return out


def read_tree(path, tree_name):
    """(mass, weight) arrays of one tree, or None when it is absent."""
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        if f:
            f.Close()
        raise RuntimeError(f"Cannot open {path}")
    try:
        if not f.Get(tree_name):
            return None
    finally:
        f.Close()
    cols = ROOT.RDataFrame(tree_name, path).AsNumpy(["mass", "weight"])
    return np.asarray(cols["mass"], float), np.asarray(cols["weight"], float)


def moments(mass, weight, lo, hi, core_lo, core_hi):
    """Window sums and core-window weighted mean/rms."""
    win = (mass >= lo) & (mass <= hi)
    sumw = float(weight[win].sum())
    sumw2 = float((weight[win] ** 2).sum())
    core = (mass >= core_lo) & (mass <= core_hi)
    w, m = weight[core], mass[core]
    sw = float(w.sum())
    sw2 = float((w ** 2).sum())
    base = {"sumw": sumw, "sumw2": sumw2, "n_win": int(win.sum()),
            "sumw_core": sw, "sumw2_core": sw2, "n_core": int(core.sum())}
    if sw <= 0:
        return {**base, "mean": None, "rms": None}
    mean = float((w * m).sum() / sw)
    var = float((w * (m - mean) ** 2).sum() / sw)
    return {**base, "mean": mean, "rms": float(np.sqrt(max(var, 0.0)))}


def relative_error(num, den, num_err2, den_err2, paired_err2):
    """Relative error of num/den - 1: paired when the samples share
    events, uncorrelated quadrature otherwise."""
    if den <= 0:
        return None
    if paired_err2 is not None:
        return float(np.sqrt(max(paired_err2, 0.0)) / abs(den))
    r = num / den
    return float(abs(r) * np.sqrt(max(num_err2, 0.0) / max(num ** 2, 1e-300)
                                  + max(den_err2, 0.0)
                                  / max(den ** 2, 1e-300)))


def deltas(cen, var, paired, mass_cen=None, w_cen=None, w_var=None,
           lo=None, hi=None):
    """The three deltas plus errors for one variation tree."""
    out = {}

    # dN: relative change of the window sum. Paired trees share events, so
    # the correlated part of the MC statistical error cancels.
    if paired:
        win = (mass_cen >= lo) & (mass_cen <= hi)
        paired_err2 = float(((w_var[win] - w_cen[win]) ** 2).sum())
    else:
        paired_err2 = None
    out["dN"] = var["sumw"] / cen["sumw"] - 1.0 if cen["sumw"] > 0 else None
    out["dN_err"] = relative_error(var["sumw"], cen["sumw"],
                                   var["sumw2"], cen["sumw2"], paired_err2)

    # dm / dsig: relative change of the core-window mean and rms. Both get
    # the uncorrelated single-sample estimate; the residual-RMS rescaling
    # in fitInterpShapeDeltas.py absorbs the (large) over-estimate for
    # paired trees, so this only has to set the relative weight between
    # points.
    if cen["mean"] and var["mean"] and cen["rms"] and var["rms"]:
        out["dm"] = var["mean"] / cen["mean"] - 1.0
        out["dsig"] = var["rms"] / cen["rms"] - 1.0
        n_eff = cen["sumw_core"] ** 2 / max(cen["sumw2_core"], 1e-300)
        out["dm_err"] = float(cen["rms"] / np.sqrt(max(n_eff, 1.0))
                              / abs(cen["mean"]))
        out["dsig_err"] = float(np.sqrt(2.0 / max(n_eff, 1.0)))
    else:
        out["dm"] = out["dsig"] = None
        out["dm_err"] = out["dsig_err"] = None
    out["paired"] = bool(paired)
    return out


def measure_point(mp, mA, polys, channels, known_missing, warnings):
    """All (sub-era, study channel) delta records of one mass point."""
    out = {}
    for channel, period, suberas in interpolation_config.categories():
        if channel not in channels:
            continue
        cat_key = interpolation_config.category_key(channel, period)
        if cat_key not in polys:
            raise RuntimeError(f"No shape parametrization for {cat_key}")
        params, _clipped = predict_shape_params(polys[cat_key], mA)
        x0 = params["x0"]
        sigma_eff = float(np.sqrt(0.5 * (params["sigmaL"] ** 2
                                         + params["sigmaR"] ** 2)))
        lo, hi = interpolation_config.interp_window(polys[cat_key], mA)
        core_lo = x0 - interpolation_config.DELTA_CORE_NSIGMA * sigma_eff
        core_hi = x0 + interpolation_config.DELTA_CORE_NSIGMA * sigma_eff
        prod_channel = interpolation_config.production_channel(channel)

        for era in suberas:
            path = interpolation_config.signal_path(era, channel, mp)
            if not os.path.exists(path):
                if (mp, era, channel) in known_missing:
                    warnings.append(f"[{mp}/{era}/{channel}] known-missing "
                                    "sample skipped")
                    continue
                raise FileNotFoundError(f"Missing sample: {path}")

            central = read_tree(path, "Central")
            if central is None:
                raise RuntimeError(f"No Central tree in {path}")
            m_cen, w_cen = central
            cen = moments(m_cen, w_cen, lo, hi, core_lo, core_hi)

            systs, pdf_members = {}, {}
            for tree_name, syst, direction in signal_trees(era, prod_channel):
                got = read_tree(path, tree_name)
                if got is None:
                    warnings.append(f"[{mp}/{era}/{channel}] missing tree "
                                    f"{tree_name}")
                    continue
                m_var, w_var = got
                paired = (len(m_var) == len(m_cen)
                          and np.array_equal(m_var, m_cen))
                var = moments(m_var, w_var, lo, hi, core_lo, core_hi)
                rec = deltas(cen, var, paired, mass_cen=m_cen, w_cen=w_cen,
                             w_var=w_var, lo=lo, hi=hi)
                if direction in ("Up", "Down"):
                    systs.setdefault(syst, {})[direction] = rec
                else:
                    pdf_members.setdefault(syst, {})[direction] = rec

            out[interpolation_config.delta_key(era, channel)] = {
                "era": era, "channel": channel, "period": period,
                "window": [lo, hi], "core": [core_lo, core_hi],
                "x0": x0, "sigma_eff": sigma_eff,
                "central": cen, "systs": systs, "pdf_members": pdf_members,
            }
            print(f"[{mp}/{era}/{channel}] {len(systs)} systematics, "
                  f"{sum(len(v) for v in pdf_members.values())} pdf members")
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", type=int, required=True,
                        help="mHc study to run")
    parser.add_argument("--masspoints", default="",
                        help="comma-separated masspoint filter")
    parser.add_argument("--channels", default="",
                        help="comma-separated study channels (default: all)")
    parser.add_argument("--output", default="",
                        help="output JSON path (default: "
                             "tests/interpolation/MHc{X}/shape_deltas/shape_deltas.json)")
    args = parser.parse_args()

    study = interpolation_config.study(args.mhc)
    masspoints = interpolation_config.filter_csv(
        [masspoint_name(m, args.mhc) for m in study["all"]],
        args.masspoints, "masspoint")
    channels = interpolation_config.filter_csv(
        interpolation_config.STUDY_CHANNELS, args.channels, "channel")
    known_missing = interpolation_config.known_missing_samples()
    polys, polys_path = interpolation_config.load_shape_polynomials(args.mhc)

    results, warnings = {}, []
    for mp in masspoints:
        mA = interpolation_config.mA_of(mp)
        results[mp] = {"mA": mA,
                       "cats": measure_point(mp, mA, polys, channels,
                                            known_missing, warnings)}

    payload = {
        "meta": {
            "mhc": args.mhc,
            "fit_ma": study["fit"],
            "channels": channels,
            "shape_polynomials": polys_path,
            "core_nsigma": interpolation_config.DELTA_CORE_NSIGMA,
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "results": results,
        "warnings": warnings,
    }
    outpath = args.output or os.path.join(
        srspaths.interpolation_dir(args.mhc), "shape_deltas", "shape_deltas.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {outpath}")
    for w in warnings:
        print(f"  warning: {w}")


if __name__ == "__main__":
    main()
