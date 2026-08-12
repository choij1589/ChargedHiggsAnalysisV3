#!/usr/bin/env python3
"""Stage 1 of the yield-interpolation chain: per-sub-era window yields.

For each study channel x run period category and each of its sub-eras,
measure the Central-tree signal yield inside the production mass window
[max(x0 - 10*sigma_eff, 12), x0 + 10*sigma_eff], with x0/sigma_eff
evaluated from the INTERPOLATED shape parametrizations at the point's mA.
This is exactly the number a parametric signal template will be normalized
to: the datacard rate is -1, i.e. the nominal histogram integral of the
per-era signal component.

Yields are TTree::Draw weighted sums with err = sqrt(sum w^2); the
full-tree sum (no window) is recorded as a reference. The dcb_fits.json
sumw is NOT reused: it is per merged run period, restricted to the
per-point direct-fit window and weight-clipped by the RooFit observable
range.

  python3 measInterpYields.py --mhc 160 [--masspoints MHc160_MA90] [--output F]
"""
import argparse
import ctypes
import datetime
import json
import os
import sys

import ROOT

import interpolation_config
import srspaths
from interpolation_config import masspoint_name

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError


def measure_file(path, lo, hi, tag):
    """Window and full-tree weighted sums of the Central tree.

    Returns {sumw, err, entries, sumw_total, err_total}."""
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        if f:
            f.Close()
        raise RuntimeError(f"Cannot open {path}")
    try:
        tree = f.Get("Central")
        if not tree or not isinstance(tree, ROOT.TTree):
            raise RuntimeError(f"No 'Central' tree in {path}")

        h_win = ROOT.TH1D(f"h_win_{tag}", "", 1, lo, hi)
        h_win.Sumw2()
        tree.Draw(f"mass>>h_win_{tag}",
                  f"weight*(mass >= {lo} && mass <= {hi})", "goff")
        err = ctypes.c_double(0.0)
        sumw = h_win.IntegralAndError(1, 1, err)
        entries = int(tree.GetEntries(f"mass >= {lo} && mass <= {hi}"))
        h_win.Delete()

        h_tot = ROOT.TH1D(f"h_tot_{tag}", "", 1, 0.0, 1e6)
        h_tot.Sumw2()
        tree.Draw(f"mass>>h_tot_{tag}", "weight", "goff")
        err_tot = ctypes.c_double(0.0)
        sumw_tot = h_tot.IntegralAndError(0, 2, err_tot)  # incl. flows
        h_tot.Delete()
    finally:
        f.Close()
    return {"sumw": float(sumw), "err": float(err.value),
            "entries": entries,
            "sumw_total": float(sumw_tot), "err_total": float(err_tot.value)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", type=int, required=True,
                        help="mHc study to run")
    parser.add_argument("--masspoints", default="",
                        help="comma-separated masspoint filter")
    parser.add_argument("--output", default="",
                        help="output JSON path (default: "
                             "fits/MHc{X}/yields/yields.json)")
    args = parser.parse_args()

    study = interpolation_config.study(args.mhc)
    masspoints = interpolation_config.filter_csv(
        [masspoint_name(m, args.mhc) for m in study["all"]],
        args.masspoints, "masspoint")
    known_missing = interpolation_config.known_missing_samples()

    polys, polys_path = interpolation_config.load_shape_polynomials(args.mhc)

    results = {}
    warnings = []
    for mp in masspoints:
        mA = interpolation_config.mA_of(mp)
        results[mp] = {"mA": mA, "channels": {}}
        for channel, period, suberas in interpolation_config.categories():
            cat_key = interpolation_config.category_key(channel, period)
            if cat_key not in polys:
                raise RuntimeError(f"No shape parametrization for {cat_key} "
                                   f"in {polys_path}")
            lo, hi = interpolation_config.interp_window(polys[cat_key], mA)
            chan = results[mp]["channels"].setdefault(channel, {})
            for era in suberas:
                path = interpolation_config.signal_path(era, channel, mp)
                if not os.path.exists(path):
                    if (mp, era, channel) in known_missing:
                        warnings.append(f"[{mp}/{era}/{channel}] known-missing "
                                        "sample skipped (corrupt raw skim)")
                        continue
                    raise FileNotFoundError(f"Missing sample: {path}")
                tag = f"{mp}_{era}_{channel}".replace("-", "_")
                record = measure_file(path, lo, hi, tag)
                record["window"] = [lo, hi]
                record["period"] = period
                chan[era] = record
        print(f"[{mp}] measured "
              f"{sum(len(c) for c in results[mp]['channels'].values())} "
              "era yields")

    payload = {
        "meta": {
            "mhc": args.mhc,
            "fit_ma": study["fit"],
            "shape_polynomials": polys_path,
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "results": results,
        "warnings": warnings,
    }
    outpath = args.output or os.path.join(
        srspaths.interpolation_fits_dir(args.mhc), "yields", "yields.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {outpath}")
    for w in warnings:
        print(f"  warning: {w}")


if __name__ == "__main__":
    main()
