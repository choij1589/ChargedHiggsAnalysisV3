#!/usr/bin/env python3
"""Stage 2 of the shape-delta chain: delta model vs mA.

Every systematic delta measured in stage 1 is parametrized in mA with an
up-only F-test ladder (DELTA_ORDERS: the physics prior is an mA-independent
relative shift, and the F-test may only upgrade it to a straight line),
fitted ONCE per (era|channel key, systematic, direction, quantity) over the
fit-anchor mass points — evaluable at any mA, exactly like the shape and
yield polynomials, so the production template producer can interpolate a
delta at an arbitrary target point.

The measured errors are only used for relative weighting: a second pass
rescales them to the residual RMS of the first-pass fit, so the reported
band reflects the actual point-to-point scatter rather than the paired
MC-statistical error, which is far too small for weight-only trees.

Closure: the model evaluated at every HELD-OUT mass point (which also has
its own measured delta, since stage 1 runs over the full grid) against
that measured delta — the interpolation test.

Outputs fits/MHc{X}/shape_deltas/delta_model.json plus a
closure table and cmsstyle overview plots for a curated subset (worst
closers + JES/ps_isr/ps_fsr) under
fits/MHc{X}/plots/deltas/.

  python3 fitInterpShapeDeltas.py --mhc 160
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

import interp_plot_utils
import interpolation_config
import srspaths
from fitInterpPolynomials import poly_band, select_order, weighted_polyfit

N_PLOTTED_WORST = 8
CURATED_KEYWORDS = ("jes", "ps_isr", "ps_fsr")


def collect_points(results, key, path, quantity, fit_ma, ma_range):
    """(mA, value, error) donor points of one delta series."""
    lo, hi = ma_range
    pts = []
    for mp, rec in results.items():
        mA = rec["mA"]
        if mA not in fit_ma or not lo <= mA <= hi:
            continue
        cat = rec["cats"].get(key)
        if cat is None:
            continue
        node = cat[path[0]].get(path[1], {}).get(path[2])
        if node is None or node.get(quantity) is None:
            continue
        err = node.get(f"{quantity}_err")
        pts.append((mA, float(node[quantity]),
                    max(float(err) if err else 0.0,
                        interpolation_config.DELTA_ERR_FLOOR[quantity])))
    pts.sort()
    return pts


def measured_at(results, key, path, quantity, mA_target):
    """The measured delta at mA_target, or None."""
    for rec in results.values():
        if rec["mA"] != mA_target:
            continue
        cat = rec["cats"].get(key)
        if cat is None:
            return None
        node = cat[path[0]].get(path[1], {}).get(path[2])
        return None if node is None else node.get(quantity)
    return None


def fit_series(pts):
    """Two-pass ladder fit of one delta series (evaluable at any mA)."""
    if not pts:
        return None
    x = np.array([p[0] for p in pts], float)
    y = np.array([p[1] for p in pts], float)
    err = np.array([p[2] for p in pts], float)
    if len(x) == 1:
        return {"coeffs": [float(y[0])], "cov": [[0.0]], "order": 0,
                "chi2": 0.0, "ndf": 0, "npoints": 1, "err_scale": 1.0,
                "points": pts}

    order, tried = select_order(x, y, err, interpolation_config.DELTA_ORDERS)
    fit = tried[order]
    # Rescale the errors to the residual scatter so the covariance band is
    # meaningful, then refit at the same order.
    resid = y - np.polyval(np.array(fit["coeffs"]), x)
    ndf = max(len(x) - (order + 1), 1)
    rms = float(np.sqrt(float(np.sum(resid ** 2)) / ndf))
    scale = 1.0
    if rms > 0:
        scale = max(rms / float(np.mean(err)), 1.0)
        fit = weighted_polyfit(x, y, err * scale, order)

    return {"coeffs": fit["coeffs"], "cov": fit["cov"], "order": int(order),
            "chi2": fit["chi2"], "ndf": fit["ndf"], "npoints": len(x),
            "err_scale": scale, "points": pts}


def eval_series(rec, mA):
    coeffs = np.array(rec["coeffs"], float)
    value = float(np.polyval(coeffs, float(mA)))
    band = float(poly_band(coeffs, np.array(rec["cov"]),
                           np.array([float(mA)]))[0])
    return value, band


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", type=int, required=True,
                        help="mHc study to fit")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    deltas_dir = os.path.join(srspaths.interpolation_fits_dir(args.mhc),
                              "shape_deltas")
    with open(os.path.join(deltas_dir, "shape_deltas.json")) as f:
        payload = json.load(f)
    results = payload["results"]
    study = interpolation_config.study(args.mhc)
    fit_ma = set(study["fit"])
    held_out_ma = study["held_out"]

    keys = sorted({k for rec in results.values() for k in rec["cats"]})
    model, closure, warnings = {}, [], []
    for key in keys:
        sample = next(rec["cats"][key] for rec in results.values()
                      if key in rec["cats"])
        ma_range = interpolation_config.delta_ma_range(sample["channel"])
        entry = {"era": sample["era"], "channel": sample["channel"],
                 "period": sample["period"], "ma_range": list(ma_range),
                 "systs": {}, "pdf_members": {}}

        for bucket in ("systs", "pdf_members"):
            for syst, directions in sorted(sample[bucket].items()):
                for direction in sorted(directions):
                    path = (bucket, syst, direction)
                    recs = {}
                    for quantity in interpolation_config.DELTA_QUANTITIES:
                        pts = collect_points(results, key, path, quantity,
                                             fit_ma, ma_range)
                        fit = fit_series(pts)
                        if fit is None:
                            warnings.append(f"[{key}] {syst}/{direction}/"
                                            f"{quantity}: no donor points")
                            continue
                        recs[quantity] = fit
                        for mA_target in held_out_ma:
                            if not ma_range[0] <= mA_target <= ma_range[1]:
                                continue
                            mc = measured_at(results, key, path, quantity,
                                             mA_target)
                            if mc is None:
                                continue
                            value, band = eval_series(fit, mA_target)
                            closure.append({
                                "key": key, "syst": syst,
                                "direction": direction, "quantity": quantity,
                                "mA": mA_target,
                                "model": value, "mc": float(mc),
                                "band": band,
                            })
                    entry[bucket].setdefault(syst, {})[direction] = recs
        model[key] = entry

    out = {
        "meta": {
            "mhc": args.mhc,
            "fit_ma": sorted(fit_ma),
            "held_out_ma": held_out_ma,
            "orders": interpolation_config.DELTA_ORDERS,
            "err_floor": interpolation_config.DELTA_ERR_FLOOR,
            "core_nsigma": payload["meta"]["core_nsigma"],
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "model": model,
        "closure": closure,
        "warnings": warnings,
    }
    outpath = os.path.join(deltas_dir, "delta_model.json")
    with open(outpath, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {outpath}")

    # Closure summary: how well the interpolated delta reproduces the delta
    # measured on each held-out point's own MC.
    print(f"\nDelta closure over {len(held_out_ma)} held-out points "
          f"({len(closure)} series x point entries):")
    worst_by_syst = {}
    for quantity in interpolation_config.DELTA_QUANTITIES:
        rows = [c for c in closure if c["quantity"] == quantity]
        if not rows:
            continue
        diff = np.array([c["model"] - c["mc"] for c in rows])
        size = np.array([abs(c["mc"]) for c in rows])
        tol = np.maximum(0.2 * size, interpolation_config.DELTA_ERR_FLOOR[quantity])
        bad = int(np.sum(np.abs(diff) > tol))
        print(f"  {quantity:5s} n={len(rows):5d}  "
              f"median|d|={np.median(np.abs(diff)):.2e}  "
              f"p90={np.percentile(np.abs(diff), 90):.2e}  "
              f"max={np.max(np.abs(diff)):.2e}  out-of-tolerance={bad}")
        for c in rows:
            k = (c["key"], c["syst"])
            worst_by_syst[k] = max(worst_by_syst.get(k, 0.0),
                                   abs(c["model"] - c["mc"]))
    worst = sorted(closure, key=lambda c: -abs(c["model"] - c["mc"]))[:10]
    print("\n  worst 10:")
    for c in worst:
        print(f"    {c['key']:28s} {c['syst']:42s} {c['direction']:5s} "
              f"{c['quantity']:5s} mA={c['mA']:g} model={c['model']:+.4f} "
              f"mc={c['mc']:+.4f}")
    for w in warnings[:20]:
        print(f"  warning: {w}")

    if not args.no_plots:
        # Curated subset: the worst closers by (key, syst) plus the
        # perennial JES/ps_isr/ps_fsr systematics — the full series grid
        # (thousands of syst x key combinations) stays JSON-only.
        worst_keys = {k for k, _v in
                     sorted(worst_by_syst.items(), key=lambda kv: -kv[1])
                     [:N_PLOTTED_WORST]}
        curated_keys = {(key, syst) for key, entry in model.items()
                        for bucket in ("systs", "pdf_members")
                        for syst in entry[bucket]
                        if any(kw in syst.lower() for kw in CURATED_KEYWORDS)}
        plot_dir = srspaths.interpolation_fit_plots_dir(args.mhc, "deltas")
        for key, syst in sorted(worst_keys | curated_keys):
            interp_plot_utils.plot_shape_delta_series(
                key, syst, model[key], plot_dir)


if __name__ == "__main__":
    main()
