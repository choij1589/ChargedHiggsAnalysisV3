#!/usr/bin/env python3
"""Combined report over the per-mHc ParticleNet-interpolation shards.

Merges fits/pnet/MHc*/threshold_wp.json and
closure/pnet/MHc*/{shape_reuse,yield_interp}.json and prints the tables
that matter -- including the Gate U1 bias-vs-refit-noise decomposition --
writing the same text to closure/pnet/summary.txt so the numbers quoted in
docs/interpolation/particlenet/ stay traceable to a tracked file.

Every table is restricted to PRODUCTION-RELEVANT seed-member pairs
(|dmA| <= 2.5): a grid point joins its NEAREST seed, so pairs further
apart never occur. Reading the unrestricted matrix is actively misleading
-- (seed85, mA95) shows an ~85% yield residual purely because the seed's
mass window clips the member's peak in half.

  python3 python/summarizePnetStudies.py
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

import pnet_interp_config as pic
import srspaths
from pnet_interp_config import ANCHOR_MA, MAX_DELTA_MA


def load_shards(stem):
    """Merge the per-mHc shards of one study into a single results dict."""
    merged, files = {}, []
    for mhc in pic.pn_mhc_list():
        if stem == "threshold_wp":
            path = pic.threshold_wp_path(mhc)
        else:
            path = os.path.join(srspaths.pnet_closure_dir(mhc),
                                f"{stem}.json")
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            payload = json.load(fh)
        if "results" not in payload:
            continue
        merged.update(payload["results"])
        files.append(os.path.relpath(path, srspaths.module_dir()))
    return merged, files


def stat_line(label, vals, width=30):
    v = np.abs(np.asarray([x for x in vals if x is not None and x == x]))
    if not len(v):
        return f"{label:{width}s}  (none)"
    return (f"{label:{width}s}n={len(v):>4d}  median={np.median(v):7.4f}  "
            f"p90={np.percentile(v, 90):7.4f}  max={v.max():7.4f}")


# --------------------------------------------------------------- shapes
def report_shapes(max_delta):
    res, files = load_shards("shape_reuse")
    if not res:
        print("shape_reuse: no shards found\n")
        return
    rows = []
    for e in res.values():
        for m in e["members"].values():
            d = abs(m["mA"] - e["seed_mA"])
            rows.append(dict(mhc=e["mhc"], ch=e["channel"], per=e["period"],
                             seed=e["seed_mA"], mA=m["mA"], d=d,
                             anchor=m["mA"] in ANCHOR_MA,
                             ds=m["d_scale_sigma"], dr=m["d_res_rel"],
                             eff=m["eff_window_sumw"]))
    prod = [r for r in rows if r["d"] <= max_delta]
    print("=" * 92)
    print(f"CHECK 1 -- SHAPE REUSABILITY AFTER THE SCORE CUT   [{len(files)} shards]")
    print(f"  {len(prod)} of {len(rows)} seed-member pairs are production-relevant "
          f"(|dmA| <= {max_delta})")
    print("=" * 92)
    for field, name in (("ds", "|d_scale| [sigma_eff]"), ("dr", "|d_res| [rel]")):
        print(f"\n{name}   (Baseline nuisance floor: "
              f"{'0.02' if field == 'ds' else '0.01'})")
        print("  " + stat_line("ALL", [r[field] for r in prod]))
        for mhc in sorted({r["mhc"] for r in prod}):
            print("  " + stat_line(f"  {mhc}",
                                   [r[field] for r in prod if r["mhc"] == mhc]))
        for key, sel in (("Run2", lambda r: r["per"] == "Run2"),
                         ("Run3", lambda r: r["per"] == "Run3"),
                         ("SR1E2Mu", lambda r: r["ch"] == "SR1E2Mu"),
                         ("SR3Mu", lambda r: r["ch"] == "SR3Mu"),
                         ("validation MA87/MA92",
                          lambda r: not r["anchor"])):
            print("  " + stat_line(f"  {key}", [r[field] for r in prod if sel(r)]))
    worst = sorted(prod, key=lambda r: -abs(r["ds"]))[:5]
    print("\n  worst |d_scale| rows:")
    for r in worst:
        print(f"    {r['mhc']}/{r['ch']}_{r['per']} seed{r['seed']}->mA{r['mA']}"
              f"  d_scale={r['ds']:+.4f}  d_res={r['dr']:+.4f}  eff={r['eff']:.3f}")

    # ---- Gate U1: is each family a BIAS or just refit noise? ----
    # The cut removes 40-80% of the events, so the post-cut DCB is refit on
    # a subset and its parameters move by pure statistics. A family whose
    # SIGNED mean is consistent with zero needs no nuisance -- sizing one
    # on its spread would double-count MC statistics that autoMCStats
    # already carries. A significant, coherent mean is a real bias.
    #
    # The cut sample is a SUBSET of the no-cut sample, so the variance of
    # the difference is approximately |err_cut^2 - err_nocut^2|, not their
    # sum.
    print("\n" + "-" * 92)
    print("  GATE U1 -- bias vs refit noise")
    print("-" * 92)
    print(f"  {'family':7s}{'channel':10s}{'period':7s}{'signed mean':>14s}"
          f"{'err':>9s}{'n_sigma':>9s}{'verdict':>12s}")
    for field, name in (("ds", "scale"), ("dr", "res")):
        for ch in ("SR1E2Mu", "SR3Mu"):
            for per in ("Run2", "Run3"):
                v = np.array([r[field] for r in prod
                              if r["ch"] == ch and r["per"] == per])
                if len(v) < 2:
                    continue
                mean, err = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
                nsig = abs(mean) / err if err > 0 else float("inf")
                verdict = "BIAS" if nsig >= 3 else "noise"
                print(f"  {name:7s}{ch:10s}{per:7s}{mean:>+14.4f}{err:>9.4f}"
                      f"{nsig:>9.1f}{verdict:>12s}")
        v = np.array([r[field] for r in prod])
        mean, err = v.mean(), v.std(ddof=1) / np.sqrt(len(v))
        print(f"  {name:7s}{'ALL':10s}{'':7s}{mean:>+14.4f}{err:>9.4f}"
              f"{abs(mean)/err:>9.1f}"
              f"{('BIAS' if abs(mean)/err >= 3 else 'noise'):>12s}")

    # d_scale against its own refit statistical error: a median pull near 1
    # means the spread IS the statistics.
    pulls = []
    for e in res.values():
        for m in e["members"].values():
            if abs(m["mA"] - e["seed_mA"]) > max_delta:
                continue
            c, n = m["cut"], m["nocut"]
            dvar = abs(c["x0_err"] ** 2 - n["x0_err"] ** 2)
            if dvar > 0 and n["sigma_eff"] > 0:
                pulls.append(abs(m["d_scale_sigma"])
                             / (np.sqrt(dvar) / n["sigma_eff"]))
    if pulls:
        pulls = np.array(pulls)
        print(f"\n  |d_scale| / its refit stat error:  median={np.median(pulls):.2f}"
              f"  p90={np.percentile(pulls, 90):.2f}"
              f"   |pull|<1: {np.mean(pulls < 1):.2f}   |pull|<2: {np.mean(pulls < 2):.2f}")
        print("  (median ~1 => the scale spread is statistics, not a shift)")
    print()


# --------------------------------------------------------------- yields
def report_yields(max_delta):
    res, files = load_shards("yield_interp")
    if not res:
        print("yield_interp: no shards found\n")
        return
    rows = []
    for e in res.values():
        for p in e["points"].values():
            rows.append(dict(mhc=e["mhc"], ch=e["channel"], per=e["period"],
                             era=p["era"], seed=e["seed_mA"], mA=p["mA"],
                             d=abs(p["mA"] - e["seed_mA"]), anchor=p["is_anchor"],
                             r_model=p["r_model"], r_eps=p["r_eps"],
                             r_total=p["r_total"]))
    prod = [r for r in rows if r["d"] <= max_delta]
    print("=" * 92)
    print(f"CHECK 2 -- YIELD = BASELINE MODEL x EPS(mA)   [{len(files)} shards]")
    print(f"  {len(prod)} of {len(rows)} points are production-relevant "
          f"(|dmA| <= {max_delta})")
    print("  r_model = Baseline model -> PN dir   r_eps = efficiency interpolation")
    print("=" * 92)
    groups = [
        ("anchors, leave-one-out", lambda r: r["anchor"]),
        ("validation MA87/MA92 (blind)", lambda r: not r["anchor"]),
        ("  validation, Run2", lambda r: not r["anchor"] and r["per"] == "Run2"),
        ("  validation, Run3", lambda r: not r["anchor"] and r["per"] == "Run3"),
    ]
    for label, sel in groups:
        pts = [r for r in prod if sel(r)]
        if not pts:
            continue
        print(f"\n{label}  [{len(pts)} points]")
        for f in ("r_model", "r_eps", "r_total"):
            print("  " + stat_line(f"  {f}", [p[f] for p in pts]))
    print("\nr_eps by mHc (the ParticleNet-specific term):")
    for mhc in sorted({r["mhc"] for r in prod}):
        print("  " + stat_line(f"  {mhc}",
                               [r["r_eps"] for r in prod if r["mhc"] == mhc]))
    print("\nr_model by period (inherited Baseline error):")
    for per in ("Run2", "Run3"):
        print("  " + stat_line(f"  {per}",
                               [r["r_model"] for r in prod if r["per"] == per]))
    print()


# ------------------------------------------------------------ workingpts
def report_wp():
    res, files = load_shards("threshold_wp")
    if not res:
        print("threshold_wp: no shards found\n")
        return
    labels = []
    for e in res.values():
        for w in e["working_points"]:
            if w["label"] not in labels:
                labels.append(w["label"])
    print("=" * 92)
    print(f"WORKING POINTS   [{len(files)} shards, {len(res)} categories]")
    print("=" * 92)
    print(f"{'WP':>11s}{'effS med':>10s}{'effS min':>10s}{'effB med':>10s}"
          f"{'Z/Zn med':>10s}{'Z/Zn min':>10s}{'B min':>9s}{'thr spread':>12s}"
          f"{'epsS spread':>13s}")
    per_cat = defaultdict(lambda: defaultdict(list))
    for key, e in res.items():
        cat = key.rsplit("/seed", 1)[0]
        for w in e["working_points"]:
            per_cat[cat][w["label"]].append(w["threshold"])
    for lab in labels:
        wps = [w for e in res.values()
               for w in e["working_points"] if w["label"] == lab]
        eS = np.array([w["eff_sig"] for w in wps])
        eB = np.array([w["eff_bkg"] for w in wps])
        zr = np.array([w["Z_over_nocut"] or 0.0 for w in wps])
        Bs = np.array([w["B"] for w in wps])
        thr_spread = [max(v[lab]) - min(v[lab]) for v in per_cat.values()
                      if lab in v and len(v[lab]) > 1]
        spread = []
        for w in wps:
            vals = [v for v in w["eff_sig_members"].values() if v == v]
            if len(vals) > 1 and w["eff_sig"] > 0:
                spread.append((max(vals) - min(vals)) / w["eff_sig"])
        print(f"{lab:>11s}{np.median(eS):>10.4f}{eS.min():>10.4f}"
              f"{np.median(eB):>10.4f}{np.median(zr):>10.4f}{zr.min():>10.4f}"
              f"{Bs.min():>9.2f}"
              f"{(np.median(thr_spread) if thr_spread else float('nan')):>12.3f}"
              f"{(np.median(spread) if spread else float('nan')):>13.4f}")

    print("\nlow-statistics spikes -- categories whose OPTIMIZED threshold "
          "leaves B < 10 events:")
    n_bad = 0
    for key, e in res.items():
        o = e["working_points"][0]
        if o["B"] >= 10:
            continue
        n_bad += 1
        alt = next((w for w in e["working_points"]
                    if w["label"] == "epsB=20%"), None)
        msg = (f"    {key:34s} thr={o['threshold']:.3f} effB={o['eff_bkg']:.4f} "
               f"S={o['S']:.2f} B={o['B']:.2f} Z={o['Z']:.3f}")
        if alt:
            msg += f"   | epsB=20%: S={alt['S']:.2f} B={alt['B']:.2f} Z={alt['Z']:.3f}"
        print(msg)
    print(f"  {n_bad} of {len(res)} categories")
    print()


class Tee:
    """Duplicate stdout into the tracked summary file."""

    def __init__(self, path):
        self.stdout = sys.stdout
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, "w")

    def write(self, text):
        self.stdout.write(text)
        self.fh.write(text)

    def flush(self):
        self.stdout.flush()
        self.fh.flush()

    def close(self):
        sys.stdout = self.stdout
        self.fh.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-delta", type=float, default=MAX_DELTA_MA,
                        help="max |mA_member - mA_seed| counted as "
                             "production-relevant")
    parser.add_argument("--output",
                        default=os.path.join(srspaths.pnet_closure_dir(),
                                             "summary.txt"))
    args = parser.parse_args()

    tee = Tee(args.output)
    sys.stdout = tee
    try:
        report_shapes(args.max_delta)
        report_yields(args.max_delta)
        report_wp()
    finally:
        tee.close()
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
