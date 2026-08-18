#!/usr/bin/env python3
"""Look-elsewhere effect: global significance of the mA scan maxima.

The trials correction is asymptotic (Gross-Vitells), not toy-based, and
that is a consequence of what V4's scan is.  V3 tested 35 mass points
spaced far above the mass resolution, so every point was an independent
test and a brute-force toy campaign over the tested set was both the
right object and affordable.  V4 samples the mA axis AT the resolution
(configs/grid.json, step/sigma_eff = 0.86-0.89 everywhere), so the number
of scan points is not a trials count at all -- a finer lattice would
inflate it without adding one independent test -- and the same campaign
would cost ~68,000 CPU-h.  What is invariant is how often the scan
statistic crosses a level: for a chi2_1 field,

    <N_u(u)> = N_0 * exp(-u/2),     p_global ~= p_local + <N_u(u_obs)>

with u = Z^2.  N_0 is measured here from the observed scan by counting
upcrossings at a ladder of low thresholds, where the exponential is
best-determined; N_u(0) is scale-invariant and anchors it.

SCOPE (frozen decision, docs/LEE.md): the search space is the mA scan
ALONE, evaluated per (arm, channel, mHc) column.  The extra multiplicity
from having also scanned 7 mHc columns and 3 channels is bounded in
docs/LEE.md from the measured curve correlations; it is not folded in
here.

    python3 python/estimateLEE.py                      # exact scan
    python3 python/estimateLEE.py --statistic bandpull # proxy, no fits
    python3 python/estimateLEE.py --method Baseline --mhc 145
"""
import argparse
import json
import math
import os
import re

import interpolation_config
import srspaths

CHANNELS = ["Combined", "SR1E2Mu", "SR3Mu"]
METHODS = ["Baseline", "ParticleNet"]
# Upcrossing thresholds in Z. Low levels are used because the sample of
# crossings is largest there and the e^{-u/2} law is being extrapolated
# UP to Z_max; u = 0 is scale-invariant and anchors the ladder.
THRESHOLDS = [0.0, 0.5, 1.0, 1.5, 2.0]
MP_RE = re.compile(r"^MHc(\d+)_MA([\dp]+)$")


def parse_masspoint(masspoint):
    m = MP_RE.match(masspoint)
    if not m:
        raise ValueError(f"not a mass point name: {masspoint!r}")
    return int(m.group(1)), float(m.group(2).replace("p", "."))


def z_to_p(z):
    """One-sided tail; the same convention collectSignificance.py stores."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def p_to_z(p):
    """Inverse of z_to_p by bisection (no scipy in the Combine env)."""
    if p <= 0.0:
        return float("inf")
    if p >= 1.0:
        return float("-inf")
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if z_to_p(mid) > p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def load_significance_curves(era, source):
    """{method: {channel: {mhc: [(mA, Z), ...]}}} from the collected scan."""
    infix = "" if source == "mc-signal" else f".{source}"
    path = os.path.join(srspaths.module_dir(), "results", "json",
                        f"significance.{era}{infix}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found -- run automize/significance.sh --grid and "
            "python3 python/collectSignificance.py --grid first")
    with open(path) as f:
        record = json.load(f)
    curves = {}
    for method, points in record.items():
        for masspoint, channels in points.items():
            mhc, ma = parse_masspoint(masspoint)
            for channel, entry in channels.items():
                (curves.setdefault(method, {}).setdefault(channel, {})
                       .setdefault(mhc, []).append((ma, entry["Z"])))
    return curves, path


def band_pull(entry):
    """(obs - exp0) over the half-width of the expected band on the side
    the observation falls -- the ranking metric of docs/SIGNIFICANCE.md."""
    obs, med = entry["obs"], entry["exp0"]
    if obs >= med:
        return (obs - med) / (entry["exp+1"] - med)
    return (obs - med) / (med - entry["exp-1"])


def load_bandpull_curves(era, source, calibration):
    """The same curves from the limit JSONs, as calibrated band pulls.

    A proxy: it needs no new fits, so it can be run on a scan whose
    Significance jobs have not been produced, and it is what the trials
    factor was first scoped with.  `calibration` scales pull -> Z."""
    curves = {}
    for method in METHODS:
        for channel in CHANNELS:
            path = srspaths.limits_json(era, channel, method, mode="BR",
                                        source=source)
            if not os.path.exists(path):
                continue
            with open(path) as f:
                limits = json.load(f)
            for masspoint, entry in limits.items():
                mhc, ma = parse_masspoint(masspoint)
                (curves.setdefault(method, {}).setdefault(channel, {})
                       .setdefault(mhc, [])
                       .append((ma, calibration * band_pull(entry))))
    return curves


def fit_bandpull_calibration(era, source):
    """Slope of Z(combine) vs band pull, through the origin, over every
    point where both exist.  Self-calibrating: no constant is frozen, and
    the fit quality is reported so a bad proxy cannot pass unnoticed."""
    infix = "" if source == "mc-signal" else f".{source}"
    path = os.path.join(srspaths.module_dir(), "results", "json",
                        f"significance.{era}{infix}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found -- cannot calibrate")
    with open(path) as f:
        record = json.load(f)
    xs, ys = [], []
    for method, points in record.items():
        for channel in CHANNELS:
            lpath = srspaths.limits_json(era, channel, method, mode="BR",
                                         source=source)
            if not os.path.exists(lpath):
                continue
            with open(lpath) as f:
                limits = json.load(f)
            for masspoint, channels in points.items():
                if channel not in channels or masspoint not in limits:
                    continue
                xs.append(band_pull(limits[masspoint]))
                ys.append(channels[channel]["Z"])
    if len(xs) < 5:
        raise RuntimeError(f"only {len(xs)} points to calibrate the proxy")
    slope = sum(x * y for x, y in zip(xs, ys)) / sum(x * x for x in xs)
    res = [y - slope * x for x, y in zip(xs, ys)]
    rms = math.sqrt(sum(r * r for r in res) / len(res))
    return {"slope": slope, "rms": rms, "max_abs_residual": max(abs(r) for r in res),
            "n_points": len(xs)}


def grid_size(method, mhc):
    """Scan points this arm's frozen grid holds at one mHc, or None if the
    arm has no study there.  An upcrossing rate read off a curve with
    holes in it is wrong, not merely imprecise, so the count is checked."""
    cfg = (srspaths.grid_config() if method == "Baseline"
           else srspaths.pnet_grid_config())
    grids = cfg["grids"].get(f"MHc{mhc}")
    if grids is None:
        return None
    return sum(len(g["members"]) for g in grids["groups"])


def upcrossings(curve, level):
    """Number of upward crossings of `level` by the mA-ordered curve."""
    return sum(1 for (_, a), (_, b) in zip(curve, curve[1:])
               if a < level <= b)


def n0_estimates(curve):
    """N_0 from each threshold of the ladder: N_u(z) * exp(z^2/2)."""
    out = {}
    for z in THRESHOLDS:
        n_u = upcrossings(curve, z)
        if n_u > 0:
            out[z] = n_u * math.exp(0.5 * z * z)
    return out


def sigma_eff_categories(channel, mhc):
    """Shape-surface categories that feed a fitted channel.  SR3Mu splits
    by the pairing rule (srspaths.pairing_variant); both variants are kept
    because a column spans mA on both sides of the mA = 60 boundary."""
    if channel == "SR1E2Mu":
        return ("SR1E2Mu_Run2", "SR1E2Mu_Run3")
    sr3 = ("SR3Mu_lowM_Run2", "SR3Mu_lowM_Run3",
           "SR3Mu_highM_Run2", "SR3Mu_highM_Run3")
    if channel == "SR3Mu":
        return sr3
    return ("SR1E2Mu_Run2", "SR1E2Mu_Run3") + sr3


def resolution_elements(mhc, channel, ma_lo, ma_hi, n_steps=400):
    """N_res = integral dmA / sigma_eff(mA), the number of resolution
    elements the scan range holds.  Used only for the Sidak bound: it is
    what the trials factor would be if every element were an independent
    test, so p_global must come out below it."""
    polys, _ = interpolation_config.load_shape_polynomials(mhc)
    cats = [c for c in sigma_eff_categories(channel, mhc) if c in polys]
    if not cats:
        raise KeyError(f"no shape surfaces for {channel} at mHc {mhc}")
    total, step = 0.0, (ma_hi - ma_lo) / n_steps
    for i in range(n_steps):
        ma = ma_lo + (i + 0.5) * step
        sigmas = []
        for cat in cats:
            sl = float(interpolation_config.eval_param(polys[cat]["sigmaL"], ma))
            sr = float(interpolation_config.eval_param(polys[cat]["sigmaR"], ma))
            sigmas.append(math.sqrt(0.5 * (sl * sl + sr * sr)))
        total += step / min(sigmas)
    return total


def gv_extreme(curve, sign, mhc, channel):
    """Gross-Vitells for ONE tail of the scan.

    sign = +1 treats the maximum of Z (the excess); sign = -1 the minimum,
    by running the identical calculation on the reflected curve -Z.  The
    excursion statistics of the field are the same in both directions -- a
    deficit is an upcrossing of the reflected field -- so a deficit gets a
    trials correction on exactly the same footing as an excess.  That is
    not cosmetic here: the deepest deficit of the scan is nearly as large
    as its highest peak, which is itself the signature of a scan whose
    extremes are driven by trials rather than by signal.
    """
    flipped = [(ma, sign * z) for ma, z in curve]
    ma_ext, z_ext = max(flipped, key=lambda t: t[1])
    p_local = z_to_p(z_ext)
    est = n0_estimates(flipped)

    out = {"ma": ma_ext, "z": sign * z_ext, "z_abs": z_ext,
           "p_local": p_local,
           "upcrossings": {f"{z:g}": upcrossings(flipped, z)
                           for z in THRESHOLDS},
           "n0_per_threshold": {f"{z:g}": v for z, v in est.items()}}
    if not est:
        # The curve never crosses any threshold upward in this direction.
        # Nothing to extrapolate from; say so rather than invent N_0.
        out["n0"] = None
        out["note"] = "no crossing at any threshold; N_0 not estimable"
        return out

    values = sorted(est.values())
    n0 = (values[len(values) // 2] if len(values) % 2 else
          0.5 * (values[len(values) // 2 - 1] + values[len(values) // 2]))
    out["n0"] = n0
    out["n0_range"] = [values[0], values[-1]]

    def globalise(n_0):
        """Expected crossings at the observed level, and the global p.

        The textbook form p_global = p_local + <N_u> is a first-order
        expansion: it is the EXPECTED NUMBER of excursions, so it runs
        past 1 as soon as <N_u> ~ 1 -- which happens here for the weaker
        column extremes.  The Poisson-clumping form below is the
        PROBABILITY of at least one excursion, agrees with the linear one
        to first order, and stays a probability.  Both are stored.
        """
        n_u = n_0 * math.exp(-0.5 * z_ext * z_ext)
        p = 1.0 - (1.0 - p_local) * math.exp(-n_u)
        return n_u, p, min(1.0, p_local + n_u), p_to_z(p)

    n_u, p_global, p_linear, z_global = globalise(n0)
    _, _, _, z_lo = globalise(values[-1])   # largest N_0 -> smallest Z
    _, _, _, z_hi = globalise(values[0])
    out.update({"expected_crossings_at_extreme": n_u,
                "trials_factor": p_global / p_local if p_local > 0 else None,
                "p_global": p_global, "p_global_linear": p_linear,
                "z_global": z_global, "z_global_range": [z_lo, z_hi]})

    n_res = resolution_elements(mhc, channel, curve[0][0], curve[-1][0])
    sidak = 1.0 - (1.0 - p_local) ** n_res
    out.update({"n_res": n_res, "sidak_bound": sidak})
    # Sidak treats every resolution element as an INDEPENDENT test, so a
    # correlated scan must land below it.  N_0 is measured from a handful
    # of crossings, though, so the comparison is only meaningful beyond
    # its own noise -- 20% here.  The ladder's own threshold-to-threshold
    # spread is a factor ~2 max/min (measured over all 36 columns), so
    # this tolerance is well inside it and the check only fires on a real
    # violation.
    out["checks"] = {
        "p_global_ge_p_local": p_global >= p_local - 1e-12,
        "p_global_le_sidak": p_global <= 1.2 * sidak + 1e-12,
    }
    return out


def analyse_column(method, channel, mhc, points):
    """Gross-Vitells for one mA scan, in both directions.

    The flat keys (z_max, p_global, z_global, ...) are the EXCESS, kept
    at the top level because that is what the headline quotes and what
    plotLEE.py reads; the deficit lives in its own block with the same
    field names.
    """
    curve = sorted(points)
    excess = gv_extreme(curve, +1.0, mhc, channel)
    deficit = gv_extreme(curve, -1.0, mhc, channel)

    row = {"method": method, "channel": channel, "mhc": mhc,
           "n_points": len(curve),
           "ma_range": [curve[0][0], curve[-1][0]],
           "ma_max": excess["ma"], "z_max": excess["z"],
           "ma_min": deficit["ma"], "z_min": deficit["z"],
           "excess": excess, "deficit": deficit}
    # Flat aliases for the excess (backwards compatible).
    for key in ("p_local", "upcrossings", "n0_per_threshold", "n0",
                "n0_range", "trials_factor", "p_global", "p_global_linear",
                "z_global", "z_global_range", "n_res", "sidak_bound",
                "checks", "note"):
        if key in excess:
            row[key] = excess[key]
    return row


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--statistic", default="significance",
                        choices=["significance", "bandpull"],
                        help="exact combine Z, or the calibrated band-pull "
                             "proxy (no fits needed)")
    parser.add_argument("--era", default="All")
    parser.add_argument("--signal-source", default="interp-signal",
                        choices=["interp-signal", "mc-signal"])
    parser.add_argument("--method", nargs="+", default=METHODS,
                        choices=METHODS)
    parser.add_argument("--channels", nargs="+", default=CHANNELS,
                        choices=CHANNELS)
    parser.add_argument("--mhc", type=int, nargs="+", default=None)
    parser.add_argument("--min-points", type=int, default=10,
                        help="skip columns with fewer scan points: an "
                             "upcrossing rate needs a curve, not a handful")
    parser.add_argument("--allow-partial", action="store_true",
                        help="analyse columns whose scan is incomplete "
                             "(reported either way; off by default)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    calibration = None
    if args.statistic == "significance":
        curves, src = load_significance_curves(args.era, args.signal_source)
    else:
        calibration = fit_bandpull_calibration(args.era, args.signal_source)
        curves = load_bandpull_curves(args.era, args.signal_source,
                                      calibration["slope"])
        src = "limits JSONs (band-pull proxy)"
        print(f"Band-pull calibration: Z = {calibration['slope']:.3f} x pull "
              f"(rms {calibration['rms']:.3f}, max |res| "
              f"{calibration['max_abs_residual']:.2f}, "
              f"n = {calibration['n_points']})")

    rows, skipped, partial = [], [], []
    for method in args.method:
        for channel in args.channels:
            for mhc, points in sorted(curves.get(method, {})
                                            .get(channel, {}).items()):
                if args.mhc and mhc not in args.mhc:
                    continue
                if len(points) < args.min_points:
                    skipped.append((method, channel, mhc, len(points)))
                    continue
                expected = grid_size(method, mhc)
                if expected is not None and len(points) != expected:
                    partial.append((method, channel, mhc, len(points),
                                    expected))
                    if not args.allow_partial:
                        continue
                row = analyse_column(method, channel, mhc, points)
                row["expected_points"] = expected
                rows.append(row)

    for method, channel, mhc, have, want in partial:
        print(f"INCOMPLETE {method}/{channel}/mHc{mhc}: {have}/{want} scan "
              f"points -- the upcrossing count on a curve with holes is "
              f"wrong, not just noisy")

    if not rows:
        raise RuntimeError("no columns with enough scan points -- has the "
                           "grid scan been collected?")

    # Both tails, side by side: the deficit is corrected the same way and
    # on this scan it is nearly as large, which is the point.
    header = (f"{'arm':12s}{'channel':9s}{'mHc':>5s}{'n':>5s}"
              f"|{'mA*':>7s}{'Zloc':>7s}{'N0':>6s}{'TF':>6s}"
              f"{'p_glob':>8s}{'Zglob':>7s}"
              f" |{'mA*':>7s}{'Zloc':>7s}{'N0':>6s}{'TF':>6s}"
              f"{'p_glob':>8s}{'Zglob':>7s}")
    print(f"\n{'':31s}|{'EXCESS':^41s} |{'DEFICIT':^41s}")
    print(header)

    def cell(block):
        """Zglob is the equivalent Z of the GLOBAL p, a magnitude: its
        direction is the column it sits in, not its sign.  p_glob is
        printed beside it so the two cannot be confused."""
        if block.get("n0") is None:
            return (f"{block['ma']:7.1f}{block['z']:+7.2f}{'--':>6s}"
                    f"{'--':>6s}{'--':>8s}{'--':>7s}")
        return (f"{block['ma']:7.1f}{block['z']:+7.2f}{block['n0']:6.1f}"
                f"{block['trials_factor']:6.0f}{block['p_global']:8.3f}"
                f"{block['z_global']:7.2f}")

    for r in rows:
        print(f"{r['method']:12s}{r['channel']:9s}{r['mhc']:5d}"
              f"{r['n_points']:5d}|{cell(r['excess'])} |{cell(r['deficit'])}")

    for method, channel, mhc, n in skipped:
        print(f"skipped {method}/{channel}/mHc{mhc}: only {n} scan points")

    # The headline: each arm's largest maximum, and what the trials
    # correction on ITS OWN mA scan does to it.
    print()
    for method in args.method:
        for label, key, pick in (("largest excess", "excess", max),
                                 ("largest deficit", "deficit", min)):
            arm = [r for r in rows if r["method"] == method
                   and r[key].get("n0")]
            if not arm:
                continue
            best = pick(arm, key=lambda r: r[key]["z"])
            b = best[key]
            lo, hi = b["z_global_range"]
            # The global Z is the equivalent Z of the global p, i.e. a
            # magnitude; the direction is `key`, so it is spelled out
            # rather than carried as a sign that would read backwards on
            # the deficit side.
            arrow = "upward" if key == "excess" else "downward"
            print(f"{method}: {label} Z = {b['z']:+.2f} "
                  f"(local p = {b['p_local']:.2e}) at mA = {b['ma']:g}, "
                  f"{best['channel']}, mHc {best['mhc']} -> global "
                  f"p = {b['p_global']:.3f}, equivalent to "
                  f"{b['z_global']:.2f} sigma {arrow} "
                  f"[{lo:.2f}, {hi:.2f}] (trials factor "
                  f"{b['trials_factor']:.0f})")

    failed = [(r, side) for r in rows for side in ("excess", "deficit")
              if r[side].get("checks")
              and not all(r[side]["checks"].values())]
    if failed:
        print(f"\n{len(failed)} column tails failed a consistency check:")
        for r, side in failed:
            b = r[side]
            print(f"  {r['method']}/{r['channel']}/mHc{r['mhc']} {side}: "
                  f"{b['checks']} (p_global {b['p_global']:.4f} vs Sidak "
                  f"{b['sidak_bound']:.4f}, implied trials "
                  f"{b['trials_factor']:.0f} vs N_res {b['n_res']:.0f})")
        print("  A GV p_global above the Sidak bound means N_0 is too large "
              "for that tail:\n"
              "  the bound assumes INDEPENDENT resolution elements, and a "
              "correlated scan\n"
              "  must sit below it. Check the N_0 ladder -- if it rises with "
              "threshold instead\n"
              "  of staying flat, the exp(-u/2) law is not being followed "
              "and the quoted\n"
              "  correction is an over-correction; the Sidak value is then "
              "the tighter statement.")

    infix = ("" if args.signal_source == "mc-signal"
             else f".{args.signal_source}")
    stat_infix = "" if args.statistic == "significance" else ".bandpull"
    # A restricted run is a VIEW of the same per-column analysis, not a
    # different one -- but it holds fewer columns, so it gets its own name
    # rather than truncating the full record in place.
    sel = []
    if sorted(args.channels) != sorted(CHANNELS):
        sel.append(".".join(args.channels))
    if sorted(args.method) != sorted(METHODS):
        sel.append(".".join(args.method))
    if args.mhc:
        sel.append("MHc" + "_".join(str(m) for m in sorted(args.mhc)))
    sel_infix = ("." + ".".join(sel)) if sel else ""
    out = args.output or os.path.join(
        srspaths.module_dir(), "results", "json",
        f"lee.{args.era}{infix}{sel_infix}{stat_infix}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"meta": {"era": args.era,
                            "signal_source": args.signal_source,
                            "statistic": args.statistic,
                            "source": src,
                            "scope": "mA scan alone, per (arm, channel, mHc)",
                            "method": "Gross-Vitells: p_global = p_local + "
                                      "N_0 exp(-Z^2/2)",
                            "thresholds": THRESHOLDS,
                            "bandpull_calibration": calibration},
                   "columns": rows}, f, indent=2)
        f.write("\n")
    print(f"\nWrote {len(rows)} columns -> {out}")
    # Only an EXCESS-tail failure is fatal: that is the number the analysis
    # quotes. Deficit-tail failures are reported above and discussed in
    # docs/LEE.md -- the downward direction is where the Gaussian-field
    # asymptotic is weakest, and hiding it behind exit 0 or behind a wider
    # tolerance would be the wrong fix.
    if any(side == "excess" for _, side in failed):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
