#!/usr/bin/env python3
"""Fix the ParticleNet-interpolation scan grid: configs/pnet_grid.json.

The ParticleNet arm's reach is set by the trained nets, not by choice:
seeds exist only at mA = 85/90/95 per mHc, and +-2.5 GeV around them
tiles [82.5, 97.5] exactly with no gap (an earlier +-2 GeV proposal
would have left 87.5 and 92.5 uncovered). The grid is a plain 0.5 GeV
lattice over that reach -- 31 points per mHc, 155 in total over the five
trained mHc -- partitioned into 3 template-sharing groups per mHc by
nearest-seed assignment. The boundaries 87.5 and 92.5 are equidistant
from two seeds and are assigned to the LOWER one, so grouping is
reproducible rather than dependent on comparison order (same tie rule as
the Baseline grid).

mc_points are the trained mA (85/90/95 everywhere, plus the blind
validation points MA87/MA92 in MHc115/MHc145), where direct-MC
comparison is possible. All are lattice members by construction --
verified anyway, with the p-notation name round-trip.

Login-node safe (JSON only).

  python3 python/makePnetGrid.py
"""
import datetime
import json
import os
import re
import sys

import pnet_interp_config as pic
import srspaths
from interpolation_config import masspoint_name, parse_ma

TICKS_PER_GEV = 20  # the shared 0.05 GeV lattice; 0.5 is a multiple


def to_ticks(x):
    t = int(round(x * TICKS_PER_GEV))
    if abs(t / TICKS_PER_GEV - x) > 1e-9:
        raise ValueError(f"{x} is not on the 0.05 GeV lattice")
    return t


def build_grid(mc_lo, mc_hi):
    """0.5 GeV lattice over the ParticleNet reach, CLIPPED to the study's
    Baseline MC range [mc_lo, mc_hi]: outside its MC endpoints both the
    Baseline yield model and the eps quadratic would be extrapolating
    (user decision 2026-08-14 — MHc100 stops at its mA = 95 endpoint)."""
    lo = max(pic.SEED_MA[0] - pic.GROUP_HALF_WIDTH, float(mc_lo))
    hi = min(pic.SEED_MA[-1] + pic.GROUP_HALF_WIDTH, float(mc_hi))
    st = to_ticks(pic.GRID_STEP)
    return [t / TICKS_PER_GEV
            for t in range(to_ticks(lo), to_ticks(hi) + 1, st)]


def build_groups(grid):
    """[(seed, [members incl. seed])] — nearest seed, ties to the lower
    one (min over (|dmA|, seed))."""
    members = {s: [] for s in pic.SEED_MA}
    for v in grid:
        members[min(pic.SEED_MA, key=lambda s: (abs(v - s), s))].append(v)
    return [(s, members[s]) for s in pic.SEED_MA if members[s]]


def main():
    import interpolation_config

    grids = {}
    n_total = 0
    for mhc in pic.pn_mhc_list():
        mc_points = [pic.mA_of(mp) for mp in pic.trained_masspoints(mhc)]
        baseline_mc = interpolation_config.study(pic.mhc_int(mhc))["all"]
        grid = build_grid(min(baseline_mc), max(baseline_mc))
        groups = build_groups(grid)

        # ---- verification ----------------------------------------------
        gmembers = sorted(v for _s, ms in groups for v in ms)
        if gmembers != grid:
            raise RuntimeError(f"{mhc}: groups do not partition the grid")
        for s, ms in groups:
            worst = max(abs(v - s) for v in ms)
            if worst > pic.GROUP_HALF_WIDTH + 1e-9:
                raise RuntimeError(
                    f"{mhc}: seed {s}: member offset {worst} exceeds the "
                    f"+-{pic.GROUP_HALF_WIDTH} GeV group window")
        gset = {to_ticks(v) for v in grid}
        missing = [m for m in mc_points if to_ticks(m) not in gset]
        if missing:
            raise RuntimeError(f"{mhc}: trained points missing from the "
                               f"grid: {missing}")
        for v in grid:
            name = masspoint_name(v, pic.mhc_int(mhc))
            if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                raise RuntimeError(f"unsafe name {name!r}")
            back = parse_ma(name.split("_MA")[1])
            if abs(back - v) > 1e-9:
                raise RuntimeError(
                    f"name round-trip failed: {v} -> {name} -> {back}")
        grids[mhc] = {
            "grid": grid,
            "mc_points": mc_points,
            "groups": [{"seed": s, "members": ms} for s, ms in groups],
        }
        n_total += len(grid)
        print(f"{mhc}: {len(grid)} points ({len(mc_points)} trained), "
              f"{len(groups)} groups, reach [{grid[0]}, {grid[-1]}]")

    payload = {
        "meta": {
            "rule": "0.5 GeV lattice over the ParticleNet reach "
                    "[82.5, 97.5] (seeds at the trained mA = 85/90/95, "
                    "groups +-2.5 GeV), CLIPPED per mHc to the Baseline "
                    "MC range -- beyond a study's MC endpoint both the "
                    "yield model and the eps quadratic would extrapolate "
                    "(MHc100 stops at mA = 95). Outside the reach only "
                    "Baseline templates exist. Model frozen 2026-08-14 -- "
                    "docs/interpolation/particlenet/METHOD.md",
            "seeds": list(pic.SEED_MA),
            "group_half_width": pic.GROUP_HALF_WIDTH,
            "grid_step": pic.GRID_STEP,
            "grouping": "nearest seed; the boundaries 87.5 and 92.5 are "
                        "equidistant and go to the LOWER seed. Members "
                        "nest under the seed's template dir "
                        "(points/{masspoint}), method ParticleNet",
            "tick_gev": 1.0 / TICKS_PER_GEV,
            "naming": "p-notation, exact: 90 -> MA90, 82.5 -> MA82p5 "
                      "(interpolation_config.masspoint_name / parse_ma)",
            "working_point": pic.DEFAULT_WP,
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "grids": grids,
    }
    path = srspaths.config_path("pnet_grid.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {path}: {n_total} grid points over {len(grids)} mHc")


if __name__ == "__main__":
    main()
