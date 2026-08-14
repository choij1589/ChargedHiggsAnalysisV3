#!/usr/bin/env python3
"""Derive the ParticleNet-layer nuisances from the eps_B=20% closure shards
and write configs/pnet_interpolation_uncertainties.json.

Gate U1 decides WHICH families get a nuisance, Gate U2 the rule; both are
recorded in docs/interpolation/particlenet/UNCERTAINTY.md. This script is
the single place the numbers quoted there are produced, so no value appears
in prose that is not reproducible from the committed shards.

Rule (adopted at Gate U2, mirroring the Baseline arm):

    rms within each mHc study -> max across studies holding >= 2 points
    -> floored by the cell's pooled rms -> floored by an absolute floor

The residuals are taken about ZERO, not about their mean, so a family that
carries a bias (res does: the cut narrows the peak) is covered by the same
rule -- bias and scatter together -- without a separate correction term.

Only production-relevant seed-member pairs enter (|dmA| <= 2.5): a grid
point joins its nearest seed, so wider pairs never occur.

  python3 python/exportPnetUncertainties.py
"""
import argparse
import datetime
import json
import os
import sys
from collections import OrderedDict

import numpy as np

import pnet_interp_config as pic
import srspaths
from pnet_interp_config import (
    CHANNELS, DEFAULT_WP, ENERGY, MAX_DELTA_MA, MIN_STUDY_POINTS,
    PERIODS, UNCERTAINTY_FLOORS)


def shard_path(mhc, stem):
    return os.path.join(srspaths.pnet_closure_dir(mhc), f"{stem}.json")


def load_res(mhcs, wp):
    """(mHc, channel, period, d_res) for production-relevant pairs."""
    rows = []
    for mhc in mhcs:
        path = shard_path(mhc, "shape_reuse")
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing; run closPnetShapes.py first")
        with open(path) as fh:
            payload = json.load(fh)
        for e in payload["results"].values():
            if e.get("wp") != wp:
                raise RuntimeError(f"{path}: wp={e.get('wp')!r}, expected "
                                   f"{wp!r} -- refusing to mix working "
                                   "points")
            for m in e["members"].values():
                if abs(m["mA"] - e["seed_mA"]) <= MAX_DELTA_MA:
                    rows.append((e["mhc"], e["channel"], e["period"],
                                 m["d_res_rel"]))
    return rows


def load_norm(mhcs, wp):
    """(mHc, channel, period, era, r_eps) for production-relevant points."""
    rows = []
    for mhc in mhcs:
        path = shard_path(mhc, "yield_interp")
        if not os.path.exists(path):
            raise SystemExit(f"{path} missing; run closPnetYields.py first")
        with open(path) as fh:
            payload = json.load(fh)
        for e in payload["results"].values():
            if e.get("wp") != wp:
                raise RuntimeError(f"{path}: wp={e.get('wp')!r}, expected "
                                   f"{wp!r} -- refusing to mix working "
                                   "points")
            for p in e["points"].values():
                if abs(p["mA"] - e["seed_mA"]) <= MAX_DELTA_MA:
                    rows.append((e["mhc"], e["channel"], e["period"],
                                 p["era"], p["r_eps"]))
    return rows


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def apply_rule(rows, select, value_idx, family):
    """rms within study -> max across studies -> pooled floor -> abs floor."""
    cell = [r for r in rows if select(r)]
    if not cell:
        return None
    per_study, skipped = {}, []
    for mhc in sorted({r[0] for r in cell}):
        v = [r[value_idx] for r in cell if r[0] == mhc]
        if len(v) >= MIN_STUDY_POINTS:
            per_study[mhc] = rms(v)
        else:
            skipped.append(mhc)
    if not per_study:
        return None
    driver = max(per_study, key=per_study.get)
    raw = per_study[driver]
    pooled = rms([r[value_idx] for r in cell])
    floor = UNCERTAINTY_FLOORS[family]
    value = max(raw, pooled, floor)
    on_floor = value == floor and floor > max(raw, pooled)
    return {
        "value": round(value, 4),
        "driver": driver,
        "rms_max": round(raw, 4),
        "pooled_rms": round(pooled, 4),
        "per_study_rms": {k: round(v, 4) for k, v in sorted(per_study.items())},
        "studies_below_min_points": skipped,
        "n_points": len(cell),
        "on_absolute_floor": bool(on_floor),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",
                        default=srspaths.pnet_uncertainties_path(),
                        help="default: the production config "
                             "configs/pnet_interpolation_uncertainties.json")
    args = parser.parse_args()

    mhcs = pic.pn_mhc_list()
    res_rows = load_res(mhcs, DEFAULT_WP)
    norm_rows = load_norm(mhcs, DEFAULT_WP)

    out = OrderedDict()
    out["meta"] = {
        "working_point": DEFAULT_WP,
        "rule": ("rms within each mHc study, then max across studies "
                 f"holding >= {MIN_STUDY_POINTS} points, floored by the "
                 "cell's pooled rms then by an absolute floor; residuals "
                 "taken about zero so a biased family is covered "
                 "bias+scatter"),
        "max_delta_mA": MAX_DELTA_MA,
        "floors": UNCERTAINTY_FLOORS,
        "mA_binning": ("none -- the ParticleNet reach [82.5, 97.5] lies "
                       "entirely inside the Baseline onZ bin"),
        "scale_family": ("absent by Gate U1: the d_scale spread is "
                         "consistent with refit statistics (median pull "
                         "~1), so a nuisance would double-count MC stat "
                         "that autoMCStats already carries"),
        "source": "closure/pnet/MHc{X}/{shape_reuse,yield_interp}.json",
        "command": " ".join(sys.argv),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    # res: one value per (channel, run period), like the Baseline res family.
    out["res"] = OrderedDict()
    for ch in CHANNELS:
        out["res"][ch] = OrderedDict()
        for period in PERIODS:
            rec = apply_rule(res_rows,
                             lambda r, c=ch, p=period: r[1] == c and r[2] == p,
                             3, "res")
            if rec:
                out["res"][ch][period] = rec

    # norm: per-era VALUES under a period-level NAME, mirroring the Baseline.
    out["norm"] = OrderedDict()
    for ch in CHANNELS:
        out["norm"][ch] = OrderedDict()
        for period, eras in PERIODS.items():
            for era in eras:
                rec = apply_rule(norm_rows,
                                 lambda r, c=ch, e=era: r[1] == c and r[3] == e,
                                 4, "norm")
                if rec:
                    rec["period"] = period
                    out["norm"][ch][era] = rec

    # Naming: the quantity token matches the Baseline family
    # (CMS_interp_res_*), with `_pnet` as a method qualifier before the
    # channel/COM suffix. `eff` rather than `norm` on purpose -- this covers
    # the ParticleNet threshold EFFICIENCY interpolation, not a second copy
    # of the Baseline yield-model norm, and naming it `norm` would invite
    # exactly the double-counting the split exists to prevent.
    out["nuisances"] = {
        "res": [pic.pn_nuisance_name("res", ch, p)
                for ch in CHANNELS for p in PERIODS],
        "eff": [pic.pn_nuisance_name("eff", ch, p)
                for ch in CHANNELS for p in PERIODS],
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"working point: {DEFAULT_WP}   res rows: {len(res_rows)}   "
          f"norm rows: {len(norm_rows)}\n")
    print("res  -> CMS_interp_res_pnet_{ch}_{COM}  (per channel x run period)")
    print(f"  {'channel':9s}{'period':7s}{'value':>8s}{'driver':>9s}"
          f"{'rms_max':>9s}{'pooled':>9s}{'n':>5s}")
    for ch in CHANNELS:
        for period, rec in out["res"][ch].items():
            print(f"  {ch:9s}{period:7s}{rec['value']:>8.4f}{rec['driver']:>9s}"
                  f"{rec['rms_max']:>9.4f}{rec['pooled_rms']:>9.4f}"
                  f"{rec['n_points']:>5d}")
    print("\neff  -> CMS_interp_eff_pnet_{ch}_{COM}"
          "  (per channel x era values under a period-level name)")
    print(f"  {'channel':9s}{'era':13s}{'value':>8s}{'driver':>9s}"
          f"{'rms_max':>9s}{'pooled':>9s}{'n':>5s}")
    for ch in CHANNELS:
        for era, rec in out["norm"][ch].items():
            print(f"  {ch:9s}{era:13s}{rec['value']:>8.4f}{rec['driver']:>9s}"
                  f"{rec['rms_max']:>9.4f}{rec['pooled_rms']:>9.4f}"
                  f"{rec['n_points']:>5d}")

    floored = [f"{fam}/{ch}/{k}" for fam in ("res", "norm") for ch in CHANNELS
               for k, rec in out[fam][ch].items() if rec["on_absolute_floor"]]
    print(f"\ncells resting on the absolute floor: {floored or 'none'}")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
