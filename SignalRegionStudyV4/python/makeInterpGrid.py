#!/usr/bin/env python3
"""Fix the template-scan mA grid: configs/grid.json.

Requirements (user, 2026-08-13):
  1. the grid step is of the order of the dimuon mass resolution
     (slightly above sigma is acceptable);
  2. the grid CONTAINS every MC point, so direct-MC templates and fit
     templates can be compared at those mA.

Implementation: banded absolute steps of the order of the
minimum-over-categories sigma_eff (user decision 2026-08-13: a step
slightly ABOVE sigma near a band start is acceptable — the ratio is
reported, never enforced):

    [15, 30)    0.1  GeV
    [30, 60)    0.25 GeV
    [60, 100)   0.5  GeV
    [100, -)    1.0  GeV

(user-fixed edges [15, 30, 60, 100, -]; the open top is the BAND
definition — the emitted grid is still clipped to [15, max MC mA]).

Every MC mA is an integer and every band step divides integers, so the
lattice contains all MC points by construction — verified anyway, along
with the p-notation name round-trip; the step/sigma_min ratio is
evaluated from the fitted sigma_eff surfaces and recorded in the meta
block as a diagnostic. Per-mHc
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

# (band lo, band hi, step) in GeV; hi None = open-ended (clipped by the
# per-mHc MC range). User-fixed edges: [15, 30, 60, 100, -].
BANDS = [(15.0, 30.0, 0.1),
         (30.0, 60.0, 0.25),
         (60.0, 100.0, 0.5),
         (100.0, None, 1.0)]
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
        blo_t, st_t = to_ticks(blo), to_ticks(step)
        bhi_t = to_ticks(bhi) if bhi is not None else hi_t
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


# Template-sharing groups: seeds on a regular lattice mirroring the grid
# bands, coarser where sigma is larger (user-fixed 2026-08-13). The seed's
# template dir holds the group's shared background templates (built with
# the SEED's mean/sigma) and hosts validation/GoF/impact jobs; members
# nest under it (srspaths.interp_member_dir). Worst member peak offset:
# 1.74 sigma, at the band starts.
SEED_SPACING = [(15.0, 30.0, 0.5),
                (30.0, 60.0, 1.0),
                (60.0, 100.0, 2.0),
                (100.0, None, 4.0)]


def build_groups(grid):
    """[(seed, [members incl. seed])] — every grid point in exactly one
    group, assigned to its nearest seed; per-mHc endpoints always seeds."""
    seed_ticks = set()
    for blo, bhi, spacing in SEED_SPACING:
        blo_t, sp_t = to_ticks(blo), to_ticks(spacing)
        for v in grid:
            t = to_ticks(v)
            if blo_t <= t < (to_ticks(bhi) if bhi is not None
                             else t + 1) and (t - blo_t) % sp_t == 0:
                seed_ticks.add(t)
    seed_ticks.update((to_ticks(grid[0]), to_ticks(grid[-1])))
    seeds = sorted(t / TICKS_PER_GEV for t in seed_ticks)
    members = {s: [] for s in seeds}
    for v in grid:
        members[min(seeds, key=lambda s: (abs(v - s), s))].append(v)
    return [(s, members[s]) for s in seeds if members[s]]


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
        for v in grid:
            name = masspoint_name(v, mhc)
            if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                raise RuntimeError(f"unsafe name {name!r}")
            back = parse_ma(name.split("_MA")[1])
            if abs(back - v) > 1e-9:
                raise RuntimeError(
                    f"name round-trip failed: {v} -> {name} -> {back}")

        groups = build_groups(grid)
        gmembers = [v for _s, ms in groups for v in ms]
        if sorted(gmembers) != grid:
            raise RuntimeError(f"MHc{mhc}: groups do not partition the grid")
        worst_off, worst_off_at = 0.0, None
        for s, ms in groups:
            for v in ms:
                r = abs(v - s) / sigma_min(polys, v)
                if r > worst_off:
                    worst_off, worst_off_at = r, v
        grids[f"MHc{mhc}"] = {
            "grid": grid,
            "mc_points": mc_points,
            "groups": [{"seed": s, "members": ms} for s, ms in groups],
        }
        res_check[f"MHc{mhc}"] = {
            "max_step_over_sigma_min": round(worst, 3),
            "at_mA": worst_at,
            "max_seed_offset_over_sigma_min": round(worst_off, 3),
            "offset_at_mA": worst_off_at,
        }
        n_total += len(grid)
        print(f"MHc{mhc}: {len(grid):4d} points "
              f"({len(mc_points)} MC), {len(groups):3d} groups, "
              f"range [{min(mc_points)}, {max(mc_points)}], worst "
              f"step/sigma {worst:.2f} at mA={worst_at}, worst seed "
              f"offset {worst_off:.2f} sigma at mA={worst_off_at}")

    payload = {
        "meta": {
            "rule": "banded absolute steps of the order of the "
                    "minimum-over-categories dimuon sigma_eff (steps "
                    "slightly above sigma near band starts accepted, "
                    "ratio reported); range [15, max MC mA] per mHc (no "
                    "extrapolation); MC points are lattice members by "
                    "construction and verified",
            "bands": [[lo, hi, step] for lo, hi, step in BANDS],  # hi null = open
            "seed_spacing": [[lo, hi, sp] for lo, hi, sp in SEED_SPACING],
            "grouping": "template-sharing groups: every grid point joins "
                        "its nearest seed; the seed's template dir holds "
                        "the shared background templates (built with the "
                        "seed's mean/sigma) and members nest under it "
                        "(points/{masspoint}); per-mHc endpoints are "
                        "always seeds",
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
