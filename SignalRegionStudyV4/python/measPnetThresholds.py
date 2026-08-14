#!/usr/bin/env python3
"""ParticleNet working points per (mHc, channel, run period, seed).

Measures, inside the SEED's mass window with the SEED's net, the weighted
signal/background efficiencies and Asimov significance of the
sensitivity-optimized threshold AND of fixed background-efficiency working
points -- and writes fits/pnet/MHc{X}/threshold_wp.json, the frozen
working-point source of truth every other ParticleNet-interpolation step
reads (production WP: eps_B = 20%, chosen at the METHOD.md gate).

Everything mirrors makeBinnedTemplates: score = s_sig / (s_sig + w1*s_np +
w2*s_db + w3*s_ttX) with the weights measured in the seed's window,
Z = sqrt(2*((S+B)*ln(1+S/B) - S)), strict '>' as in getOptimizedThreshold.

Each file is read ONCE per (mHc, channel, period) with every net's score
branches, then all seeds are evaluated in numpy -- the naive version
re-read all files per seed and took ~10 min per seed.

  python3 python/measPnetThresholds.py --mhc MHc115 [--wp 0.1,0.2,0.3]
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
from makeBinnedTemplates import PARTICLENET_CLASS_MAPPING
from pnet_interp_config import PERIODS, WeightedScores

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError

# Same list makeBinnedTemplates.optimizeCategoryParticleNetThreshold uses.
BKG_PROCESSES = ["nonprompt", "WZ", "ZZ", "ttW", "ttZ", "ttH", "tZq",
                 "conversion", "others"]
SCORE_CLASSES = ["signal", "nonprompt", "diboson", "ttZ"]
# ParticleNet score class -> the bg_weights key it is divided by.
WEIGHT_KEY = {"nonprompt": "nonprompt", "diboson": "diboson", "ttZ": "ttX"}

# Production scans thresholds on this grid (np.linspace(0, 1, 101)).
THRESHOLD_GRID = np.linspace(0.0, 1.0, 101)


def load_known_windows():
    """Seed mass windows already measured by closPnetYields.py, keyed
    identically. The window is a property of the seed's DCB fit alone, so
    refitting would burn ~15 s per seed to reproduce a number already on
    disk; absent keys fall back to the fit."""
    windows = {}
    for mhc in pic.pn_mhc_list():
        path = os.path.join(srspaths.pnet_closure_dir(mhc),
                            "yield_interp.json")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            for k, v in json.load(fh)["results"].items():
                windows[k] = v["mass_window"]
    return windows


def signal_chain(basedirs, mp):
    chain = ROOT.TChain("Central")
    for basedir in basedirs:
        path = os.path.join(basedir, f"{mp}.root")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        chain.Add(path)
    return chain


def read_file(path, group):
    """mass, weight and every net's four score branches, as numpy."""
    cols = ["mass", "weight"] + [f"score_{mp}_{c}" for mp in group
                                 for c in SCORE_CLASSES]
    arrs = ROOT.RDataFrame("Central", path).AsNumpy(cols)
    return {c: np.asarray(arrs[c], dtype=float) for c in cols}


def load_category(basedirs, group, processes):
    """Read every process of every sub-era once. {process: [per-era dict]}."""
    cache = OrderedDict()
    for proc in processes:
        per_era = []
        for basedir in basedirs:
            path = os.path.join(basedir, f"{proc}.root")
            if not os.path.exists(path):
                continue
            per_era.append(read_file(path, group))
        if per_era:
            cache[proc] = per_era
    return cache


def scores_for(rec, seed, bg_weights, mask):
    """Seed-net score of the masked events (mirrors build_particlenet_score)."""
    s0 = rec[f"score_{seed}_signal"][mask]
    denom = s0.copy()
    for cls in ("nonprompt", "diboson", "ttZ"):
        w = bg_weights.get(WEIGHT_KEY[cls], 1.0) if bg_weights else 1.0
        denom = denom + w * rec[f"score_{seed}_{cls}"][mask]
    out = np.zeros_like(s0)
    np.divide(s0, denom, out=out, where=denom > 0)
    return out


def background_weights(cache, lo, hi):
    """Per-PN-class weight fractions in the window
    (getCategoryBackgroundWeights, vectorized -- can differ in the last
    ulp, irrelevant for a score denominator)."""
    weights = {}
    for pn_class, procs in PARTICLENET_CLASS_MAPPING.items():
        total, found = 0.0, False
        for proc in procs:
            for rec in cache.get(proc, []):
                found = True
                m = (rec["mass"] >= lo) & (rec["mass"] <= hi)
                total += float(np.sum(rec["weight"][m]))
        weights[pn_class] = total if found else 1.0 / 3.0
    tot = sum(weights.values())
    if tot > 0:
        return {k: v / tot for k, v in weights.items()}
    return {k: 1.0 / 3.0 for k in weights}


def pooled(cache, procs, seed, bg_weights, lo, hi):
    """(scores, weights) of several processes pooled over the period."""
    s_all, w_all = [], []
    for proc in procs:
        for rec in cache.get(proc, []):
            m = (rec["mass"] >= lo) & (rec["mass"] <= hi)
            if not m.any():
                continue
            s_all.append(scores_for(rec, seed, bg_weights, m))
            w_all.append(rec["weight"][m])
    if not s_all:
        return np.array([]), np.array([])
    return np.concatenate(s_all), np.concatenate(w_all)


def asimov_z(S, B):
    if B <= 0 or S <= 0:
        return 0.0
    return float(np.sqrt(2.0 * ((S + B) * np.log(1.0 + S / B) - S)))


def optimized_threshold(sig, bkg):
    """makeBinnedTemplates.getOptimizedThreshold, without the logging."""
    z = np.array([asimov_z(sig.sum_above(t), bkg.sum_above(t))
                  for t in THRESHOLD_GRID])
    best = int(np.argmax(z))
    return float(THRESHOLD_GRID[best]), float(z[0]), float(z[best])


def run_mhc(mhc, channels, targets, known_windows, warnings):
    group = pic.trained_masspoints(mhc)
    results = OrderedDict()
    for channel in channels:
        for period, suberas in PERIODS.items():
            cat = f"{channel}_{period}"
            basedirs = [srspaths.mhc_sample_dir(e, channel, mhc)
                        for e in suberas]
            if any(not os.path.isdir(d) for d in basedirs):
                warnings.append(f"[{mhc}/{cat}] missing sample dirs")
                print(f"WARNING: {warnings[-1]}")
                continue

            print(f"[{mhc}/{cat}] reading {len(BKG_PROCESSES) + len(group)} "
                  f"processes x {len(basedirs)} eras ...")
            cache = load_category(basedirs, group,
                                  list(group) + BKG_PROCESSES)

            for seed in group:
                seed_mA = pic.mA_of(seed)
                key = f"{mhc}/{cat}/seed{seed_mA}"
                if key in known_windows:
                    lo, hi = known_windows[key]
                else:
                    try:
                        fit = fit_dcb_with_errors(
                            signal_chain(basedirs, seed), seed_mA)
                    except Exception as exc:                 # noqa: BLE001
                        warnings.append(f"[{key}] seed fit failed: {exc}")
                        print(f"WARNING: {warnings[-1]}")
                        continue
                    x0, sig = fit["params"]["x0"]["value"], fit["sigma_eff"]
                    lo, hi = max(x0 - 10.0 * sig, 12.0), x0 + 10.0 * sig

                bg_weights = background_weights(cache, lo, hi)
                bkg = WeightedScores(*pooled(cache, BKG_PROCESSES, seed,
                                             bg_weights, lo, hi))
                if bkg.n == 0:
                    warnings.append(f"[{key}] no background events")
                    print(f"WARNING: {warnings[-1]}")
                    continue
                B_tot = bkg.total

                sig_arrays = OrderedDict()
                for mp in group:
                    ws = WeightedScores(*pooled(cache, [mp], seed,
                                                bg_weights, lo, hi))
                    if ws.n:
                        sig_arrays[mp] = ws
                if seed not in sig_arrays:
                    warnings.append(f"[{key}] no seed signal events")
                    print(f"WARNING: {warnings[-1]}")
                    continue
                sig = sig_arrays[seed]
                S_tot = sig.total

                thr_opt, z_nocut, z_opt = optimized_threshold(sig, bkg)

                def working_point(thr, label):
                    eS = sig.eff_above(thr)
                    eB = bkg.eff_above(thr)
                    S, B = sig.sum_above(thr), bkg.sum_above(thr)
                    z = asimov_z(S, B)
                    return OrderedDict([
                        ("label", label), ("threshold", float(thr)),
                        ("eff_sig", eS), ("eff_bkg", eB),
                        ("S", S), ("B", B), ("Z", z),
                        ("Z_over_nocut", z / z_nocut if z_nocut else None),
                        # eps_S of every member under this same cut: the
                        # quantity the mA interpolation has to model.
                        ("eff_sig_members", OrderedDict(
                            (mp, ws.eff_above(thr))
                            for mp, ws in sig_arrays.items())),
                    ])

                wps = [working_point(thr_opt, "optimized")]
                for t in targets:
                    wps.append(working_point(bkg.threshold_for_eff(t),
                                             f"epsB={t:.0%}"))

                results[key] = OrderedDict([
                    ("mhc", mhc), ("channel", channel), ("period", period),
                    ("seed", seed), ("seed_mA", seed_mA),
                    ("mass_window", [lo, hi]), ("bg_weights", bg_weights),
                    ("S_total", S_tot), ("B_total", B_tot),
                    ("Z_nocut", z_nocut), ("working_points", wps),
                ])
                print(f"  {key}: S={S_tot:.2f} B={B_tot:.2f} "
                      f"Z_nocut={z_nocut:.3f} thr_opt={thr_opt:.2f}")

            cache.clear()
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", default="all",
                        help="comma-separated mHc studies, or 'all'")
    parser.add_argument("--channels", default="SR1E2Mu,SR3Mu")
    parser.add_argument("--wp", default="0.1,0.2,0.3",
                        help="comma-separated target background efficiencies")
    parser.add_argument("--output", default=None,
                        help="override output path (single --mhc only); "
                             "default fits/pnet/MHc{X}/threshold_wp.json")
    args = parser.parse_args()

    mhcs = (pic.pn_mhc_list() if args.mhc == "all"
            else [m.strip() for m in args.mhc.split(",") if m.strip()])
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    targets = [float(w) for w in args.wp.split(",") if w.strip()]
    if args.output and len(mhcs) != 1:
        parser.error("--output requires a single --mhc")

    known_windows = load_known_windows()
    if known_windows:
        print(f"reusing {len(known_windows)} seed windows from "
              "closure/pnet yield_interp.json")

    all_results, warnings = OrderedDict(), []
    for mhc in mhcs:
        mhc_warnings = []
        results = run_mhc(mhc, channels, targets, known_windows, mhc_warnings)
        out = args.output or pic.threshold_wp_path(mhc)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump({"results": results, "warnings": mhc_warnings},
                      fh, indent=2)
        print(f"Wrote {out} ({len(results)} categories)")
        all_results.update(results)
        warnings.extend(mhc_warnings)

    labels = ["optimized"] + [f"epsB={t:.0%}" for t in targets]
    print("\n" + "=" * 104)
    print("PARTICLENET WORKING POINTS -- weighted, in the seed's mass window")
    print("=" * 104)
    print(f"{'category':26s}{'seed':>5s}{'WP':>11s}{'thr':>7s}"
          f"{'effS':>8s}{'effB':>8s}{'S':>9s}{'B':>10s}{'Z':>8s}{'Z/Znocut':>10s}")
    for key, e in all_results.items():
        head = key.rsplit("/seed", 1)[0]
        for wp in e["working_points"]:
            print(f"{head:26s}{e['seed_mA']:>5d}{wp['label']:>11s}"
                  f"{wp['threshold']:>7.3f}{wp['eff_sig']:>8.4f}{wp['eff_bkg']:>8.4f}"
                  f"{wp['S']:>9.3f}{wp['B']:>10.3f}{wp['Z']:>8.4f}"
                  f"{(wp['Z_over_nocut'] or 0):>10.4f}")
        print()

    print("=" * 104)
    print("WP COMPARISON -- across all categories")
    print("=" * 104)
    print(f"{'WP':>11s}{'n':>5s}{'effS med':>10s}{'effS min':>10s}"
          f"{'effB med':>10s}{'Z/Znocut med':>14s}{'Z/Znocut min':>14s}"
          f"{'epsS spread':>13s}")
    for lab in labels:
        wps = [wp for e in all_results.values()
               for wp in e["working_points"] if wp["label"] == lab]
        if not wps:
            continue
        eS = np.array([w["eff_sig"] for w in wps])
        eB = np.array([w["eff_bkg"] for w in wps])
        zr = np.array([w["Z_over_nocut"] or 0.0 for w in wps])
        # How much eps_S moves across a group's members -- the
        # interpolation burden. Max-min over members, relative to the
        # seed's own value.
        spread = []
        for w in wps:
            vals = [v for v in w["eff_sig_members"].values() if v == v]
            if len(vals) > 1 and w["eff_sig"] > 0:
                spread.append((max(vals) - min(vals)) / w["eff_sig"])
        print(f"{lab:>11s}{len(wps):>5d}{np.median(eS):>10.4f}{eS.min():>10.4f}"
              f"{np.median(eB):>10.4f}{np.median(zr):>14.4f}{zr.min():>14.4f}"
              f"{(np.median(spread) if spread else float('nan')):>13.4f}")

    if warnings:
        print(f"\n{len(warnings)} warning(s)")


if __name__ == "__main__":
    main()
