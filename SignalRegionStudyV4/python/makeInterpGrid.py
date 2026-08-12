#!/usr/bin/env python3
"""Fix the template-scan mA grid: configs/grid.json.

Requirements (user, 2026-08-13):
  1. the grid step is slightly SMALLER than the dimuon mass resolution
     everywhere;
  2. the grid CONTAINS every MC point, so direct-MC templates and fit
     templates can be compared at those mA.

Implementation: banded absolute steps, each band's step below the
minimum-over-categories sigma_eff at the band start (resolution grows
with mA, so the band start is the binding point):

    [15, 30)    0.1  GeV   (sigma_min(15)  ~ 0.13 GeV)
    [30, 60)    0.25 GeV   (sigma_min(30)  ~ 0.27 GeV)
    [60, 100)   0.5  GeV   (sigma_min(60)  ~ 0.62 GeV)
    [100, 155]  1.0  GeV   (sigma_min(100) ~ 1.19 GeV)

Every MC mA is an integer and every band step divides integers, so the
lattice contains all MC points by construction — verified anyway, along
with the resolution condition (evaluated from the fitted sigma_eff
surfaces, never assumed) and the p-notation name round-trip. Per-mHc
range: [15, max MC mA] — never beyond the MC endpoints (the chain's
no-extrapolation policy).

The grid is built on an integer lattice in ticks of 0.05 GeV, so the
emitted floats are exact and stable in JSON. Login-node safe (JSON only).

  python3 python/makeInterpGrid.py
"""
import datetime
import json
import os
import re
import sys

import numpy as np

import interpolation_config
import srspaths
from interpolation_config import masspoint_name, parse_ma

# (band lo, band hi, step) in GeV; hi of the last band is inclusive.
BANDS = [(15.0, 30.0, 0.1),
         (30.0, 60.0, 0.25),
         (60.0, 100.0, 0.5),
         (100.0, 155.0, 1.0)]
TICKS_PER_GEV = 20  # 0.05 GeV lattice; every band step is a multiple


def to_ticks(x):
    t = int(round(x * TICKS_PER_GEV))
    if abs(t / TICKS_PER_GEV - x) > 1e-9:
        raise ValueError(f"{x} is not on the 0.05 GeV lattice")
    return t


def build_grid(lo, hi):
    """Banded lattice over [lo, hi], band edges anchored at the band
    starts (integers), MC endpoints inclusive."""
    lo_t, hi_t = to_ticks(lo), to_ticks(hi)
    ticks = set()
    for blo, bhi, step in BANDS:
        blo_t, bhi_t, st_t = to_ticks(blo), to_ticks(bhi), to_ticks(step)
        start = max(blo_t, lo_t)
        stop = min(bhi_t, hi_t)
        if start > stop:
            continue
        # anchor at the band start so values stay on the band's lattice
        first = blo_t + ((start - blo_t + st_t - 1) // st_t) * st_t
        ticks.update(range(first, stop + 1, st_t))
    ticks.update((lo_t, hi_t))
    return [t / TICKS_PER_GEV for t in sorted(ticks)]


def sigma_min(polys, mA):
    """Minimum-over-categories sigma_eff from the sliced shape surfaces."""
    sigs = []
    for params in polys.values():
        sL = interpolation_config.eval_param(params["sigmaL"], mA)
        sR = interpolation_config.eval_param(params["sigmaR"], mA)
        sigs.append(float(np.sqrt(0.5 * (sL ** 2 + sR ** 2))))
    return min(sigs)


def main():
    grids = {}
    res_check = {}
    n_total = 0
    for mhc in interpolation_config.mhc_grid():
        mc_points = interpolation_config.study(mhc)["all"]
        if min(mc_points) != 15:
            raise RuntimeError(
                f"MHc{mhc}: lowest MC point is {min(mc_points)}, not 15 — "
                "revisit the grid range rule")
        grid = build_grid(min(mc_points), max(mc_points))

        # -- verification, per requirement ------------------------------
        gset = {to_ticks(v) for v in grid}
        missing = [m for m in mc_points if to_ticks(m) not in gset]
        if missing:
            raise RuntimeError(f"MHc{mhc}: MC points missing from grid: "
                               f"{missing}")
        polys, _ = interpolation_config.load_shape_polynomials(mhc)
        worst, worst_at = 0.0, None
        for a, b in zip(grid, grid[1:]):
            ratio = (b - a) / sigma_min(polys, a)
            if ratio > worst:
                worst, worst_at = ratio, a
        if worst >= 1.0:
            raise RuntimeError(
                f"MHc{mhc}: step {worst:.2f}x sigma_min at mA={worst_at} "
                "— the grid is NOT below the resolution there")
        for v in grid:
            name = masspoint_name(v, mhc)
            if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                raise RuntimeError(f"unsafe name {name!r}")
            back = parse_ma(name.split("_MA")[1])
            if abs(back - v) > 1e-9:
                raise RuntimeError(
                    f"name round-trip failed: {v} -> {name} -> {back}")

        grids[f"MHc{mhc}"] = {"grid": grid, "mc_points": mc_points}
        res_check[f"MHc{mhc}"] = {
            "max_step_over_sigma_min": round(worst, 3),
            "at_mA": worst_at,
        }
        n_total += len(grid)
        print(f"MHc{mhc}: {len(grid):4d} points "
              f"({len(mc_points)} MC), range [{min(mc_points)}, "
              f"{max(mc_points)}], worst step/sigma_min "
              f"{worst:.2f} at mA={worst_at}")

    payload = {
        "meta": {
            "rule": "banded absolute steps, each below the "
                    "minimum-over-categories dimuon sigma_eff at the band "
                    "start; range [15, max MC mA] per mHc (no "
                    "extrapolation); MC points are lattice members by "
                    "construction and verified",
            "bands": [[lo, hi, step] for lo, hi, step in BANDS],
            "tick_gev": 1.0 / TICKS_PER_GEV,
            "naming": "p-notation, exact: 90 -> MA90, 90.5 -> MA90p5, "
                      "30.25 -> MA30p25 (interpolation_config."
                      "masspoint_name / parse_ma)",
            "resolution_check": res_check,
            "sigma_source": "fits/MHc{X}/polynomials.json sliced surfaces",
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "grids": grids,
    }
    path = srspaths.config_path("grid.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {path}: {n_total} grid points over "
          f"{len(grids)} mHc studies")


if __name__ == "__main__":
    main()
