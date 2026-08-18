#!/usr/bin/env python3
"""Derive interpolation-uncertainty nuisance sizes from the closure tests
and export them for the template-production step.

Two closure tests, two correlation models (user requirement,
docs/interpolation/UNCERTAINTY.md):

  - shape closure (closInterpShapes.py -> closure.json): interpolated vs
    direct-fit x0/sigma_eff at held-out mass points -> scale/res nuisances,
    CORRELATED WITHIN A RUN PERIOD (one shape parametrization per
    (channel, period), so every mA in a period shares it).
  - yield closure (closInterpYields.py -> yield_closure.json): predicted
    vs measured window yield at held-out mass points -> norm nuisance,
    DECORRELATED BETWEEN ERAS (the datacard's signal columns are per-era
    components).

Point selection (all three rules): held-out mass points only, restricted to
the production-relevant pairing variant (srspaths.pairing_variant) so
SR3Mu's lowM/highM study channels pool into the single production channel
"SR3Mu"; non-extrapolation; good-quality direct fit (shape) / a measured
scalar yield record (norm).

Envelope: the MAXIMUM over the selected held-out points, floored (conservative
approach, user decision 2026-08-11 — scale/res/norm all use the max, not an
RMS/median, even though for norm this absorbs the known Run3 per-sample
scatter into the exported nuisance size).

  python3 exportInterpUncertainties.py --mhc 160
  python3 exportInterpUncertainties.py --all
"""
import argparse
import datetime
import decimal
import json
import os
import sys

import interpolation_config
import srspaths
from interpolation_config import masspoint_name


def _run_periods():
    import run_period_utils
    return list(run_period_utils.RUN_PERIODS.items())


def build_nuisance_names(scale, res, norm):
    """Nuisance names for the three families — all period-level:
    {channel: {period: name}}.

    Values are keyed by STUDY channel (SR3Mu_lowM / SR3Mu_highM) because a
    single mA bin at fixed mHc can contain both pairings; the names use the
    production channel SR3Mu, which is safe because one datacard holds one
    mass point and therefore exactly one pairing. norm VALUES stay keyed
    per (era, mA bin) — one period-level nuisance carries per-era lnN
    values in its datacard columns, and the target mA selects the bin."""
    names = {"scale": {}, "res": {}, "norm": {}}
    for kind, block in (("scale", scale), ("res", res)):
        for channel, per_period in block.items():
            prod = interpolation_config.production_channel(channel)
            for period in per_period:
                n = interpolation_config.interp_nuisance_names(prod, period)
                names[kind].setdefault(channel, {})[period] = n[kind]
    for channel, per_era in norm.items():
        prod = interpolation_config.production_channel(channel)
        for era in per_era:
            period = interpolation_config.period_of(era)
            n = interpolation_config.interp_nuisance_names(prod, period)
            names["norm"].setdefault(channel, {})[period] = n["norm"]
    return names


# The SR3Mu pairing variants stay SPLIT here: every mass point constrains
# both, and only the production_restricted block narrows each to the mA
# range where it is the production pairing.
LOO_CHANNELS = list(interpolation_config.STUDY_CHANNELS)


def _loo_point_files(mhc, grid):
    """{mA: (closure, yield_closure)} from the per-point LOO dirs; a
    partial sweep is a hard error, never a silently partial envelope."""
    missing, out = [], {}
    for mA in grid:
        loo_dir = srspaths.interpolation_loo_dir(mhc, mA)
        cpath = os.path.join(loo_dir, "closure.json")
        ypath = os.path.join(loo_dir, "yields", "yield_closure.json")
        absent = [p for p in (cpath, ypath) if not os.path.exists(p)]
        if absent:
            missing.extend(absent)
            continue
        with open(cpath) as f:
            closure = json.load(f)
        with open(ypath) as f:
            yield_closure = json.load(f)
        for payload, path in ((closure, cpath), (yield_closure, ypath)):
            if payload["meta"].get("loo_ma") != mA:
                raise RuntimeError(
                    f"{path} is not a LOO result for mA={mA} "
                    f"(meta.loo_ma={payload['meta'].get('loo_ma')})")
        out[mA] = (closure["closure"], yield_closure["closure"])
    if missing:
        raise FileNotFoundError(
            f"LOO sweep incomplete for mHc={mhc}; missing:\n  "
            + "\n  ".join(missing))
    return out


def _envelope(points, floor, production_only=False):
    """(max envelope, n points used) over the usable point records."""
    vals = [p["value"] for p in points
            if p.get("excluded") is None
            and (p["production_pairing"] or not production_only)]
    return max([floor] + vals), len(vals)


def collect_loo_points(mhc, signed=False):
    """Per-point LOO records of one mHc, before any envelope is taken.

    Returns (shape_detail, norm_detail, grid, warnings); every record
    carries mA and production_pairing so the caller can slice.

    `value` is the ABSOLUTE residual, which is what the envelope rule
    consumes. With signed=True each kept record additionally carries
    `signed` — the same residual with its sign — for plotters that need to
    show bias and scatter separately (plotInterpResiduals.py). The extra
    key is opt-in so the exported artifacts stay byte identical."""
    study = interpolation_config.study(mhc)
    grid = study["all"]
    per_point = _loo_point_files(mhc, grid)

    warnings = []
    shape_detail = {}   # (channel, period) -> {"scale": [...], "res": [...]}
    norm_detail = {}    # (channel, era) -> [...]
    for mA in grid:
        mp = masspoint_name(mA, mhc)
        closure, yield_closure = per_point[mA]
        pairing_channel = interpolation_config.study_channel_for("SR3Mu", mp)
        for period, eras in _run_periods():
            for channel in LOO_CHANNELS:
                in_production = (channel == "SR1E2Mu"
                                 or channel == pairing_channel)
                base = {"mA": mA, "production_pairing": in_production}

                # Norm points (independent of the shape-closure gates).
                entry = yield_closure.get(mp)
                for era in eras:
                    rec_n = (entry or {}).get("scalar", {}) \
                        .get(channel, {}).get(era)
                    if rec_n is None:
                        excluded_n, value = "missing_record", None
                    elif rec_n.get("extrapolation"):
                        excluded_n, value = "extrapolation", None
                    else:
                        excluded_n, value = None, abs(rec_n["rel"])
                    pt_n = {**base, "value": value, "excluded": excluded_n}
                    if signed and excluded_n is None:
                        pt_n["signed"] = float(rec_n["rel"])
                    norm_detail.setdefault((channel, era), []).append(pt_n)

                cat_key = interpolation_config.category_key(channel, period)
                rec = closure.get(cat_key, {}).get(mp)
                slot = shape_detail.setdefault(
                    (channel, period), {"scale": [], "res": []})
                if rec is None:
                    warnings.append(f"[shape/{channel}/{period}] mA={mA}: "
                                    "no LOO closure record")
                    excluded = "missing_record"
                elif rec.get("extrapolation"):
                    excluded = "extrapolation"
                elif rec.get("direct_quality") != "good":
                    excluded = f"direct_quality={rec.get('direct_quality')}"
                elif not rec["sigma_eff_direct"] > 0:
                    excluded = "sigma_eff_direct<=0"
                else:
                    excluded = None
                if excluded is not None:
                    pt = {**base, "value": None, "excluded": excluded}
                    slot["scale"].append(pt)
                    slot["res"].append(dict(pt))
                    continue
                d_scale = ((rec["predicted"]["x0"] - rec["x0_direct"])
                           / rec["sigma_eff_direct"])
                d_res = rec["sigma_eff_pred"] / rec["sigma_eff_direct"] - 1.0
                pt_s = {**base, "excluded": None, "value": abs(d_scale)}
                pt_r = {**base, "excluded": None, "value": abs(d_res)}
                if signed:
                    pt_s["signed"] = float(d_scale)
                    pt_r["signed"] = float(d_res)
                slot["scale"].append(pt_s)
                slot["res"].append(pt_r)

    return shape_detail, norm_detail, grid, warnings


def export_loo_one(mhc):
    """Aggregate the per-point LOO closures of one mHc into
    closure/interpolation/MHc{X}/loo_uncertainties.json.

    Its norm block keeps the plain per-study max and is a DIAGNOSTIC: the
    production norm nuisance is mA-binned, pooled over mHc and set by the
    per-study rms rule in loo_uncertainties.pooled.json (--pooled)."""
    shape_detail, norm_detail, grid, warnings = collect_loo_points(
        mhc)

    def blocks(production_only):
        scale, res, norm, n_points = {}, {}, {}, {"scale": {}, "res": {}, "norm": {}}
        for (channel, period), pts in shape_detail.items():
            v, n = _envelope(pts["scale"],
                             interpolation_config.UNCERTAINTY_SCALE_FLOOR,
                             production_only)
            scale.setdefault(channel, {})[period] = v
            n_points["scale"][f"{channel}/{period}"] = n
            v, n = _envelope(pts["res"],
                             interpolation_config.UNCERTAINTY_RES_FLOOR,
                             production_only)
            res.setdefault(channel, {})[period] = v
            n_points["res"][f"{channel}/{period}"] = n
        for (channel, era), pts in norm_detail.items():
            v, n = _envelope(pts, interpolation_config.UNCERTAINTY_NORM_FLOOR,
                             production_only)
            norm.setdefault(channel, {})[era] = 1.0 + v
            n_points["norm"][f"{channel}/{era}"] = n
        return scale, res, norm, n_points

    scale, res, norm, n_points = blocks(production_only=False)
    for channel, per_era in norm.items():
        for era, v in per_era.items():
            if v - 1.0 > interpolation_config.UNCERTAINTY_NORM_WARN:
                warnings.append(
                    f"[norm/{channel}/{era}] LOO envelope {v - 1.0:.3f} "
                    f"exceeds the "
                    f"{interpolation_config.UNCERTAINTY_NORM_WARN:g} warn "
                    "threshold")
    ps, pr, pn, pnp = blocks(production_only=True)

    payload = {
        "meta": {
            "mhc": mhc,
            "strategy": "leave-one-out: every grid point closed against "
                        "models refit on the full grid minus that point",
            "grid_ma": grid,
            "endpoint_ma": [grid[0], grid[-1]],
            "channels": LOO_CHANNELS,
            "rules": {
                "scale": "max(|x0_pred-x0_direct|/sigma_eff_direct) over "
                         "usable LOO points, floor "
                         f"{interpolation_config.UNCERTAINTY_SCALE_FLOOR}",
                "res": "max(|sigma_eff_pred/sigma_eff_direct-1|) over usable "
                       "LOO points, floor "
                       f"{interpolation_config.UNCERTAINTY_RES_FLOOR}",
                "norm": "max(|N_pred/N_meas-1|) over usable LOO points per "
                        "era, floor "
                        f"{interpolation_config.UNCERTAINTY_NORM_FLOOR}, "
                        f"warn above {interpolation_config.UNCERTAINTY_NORM_WARN}",
                "excluded": "extrapolation (grid endpoints), non-good direct "
                            "fit (shape only), missing records",
                "keying": "SR3Mu pairing variants SPLIT, evaluated over the "
                          "full mA grid; the production_restricted block "
                          "limits each variant to its production pairing "
                          "range (informational)",
            },
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "scale": scale, "res": res, "norm": norm,
        # No nuisance names here on purpose: this file is a per-study
        # DIAGNOSTIC under the old plain-max rule. The production values,
        # their mA binning and their names live in
        # loo_uncertainties.pooled.json / configs/interpolation_uncertainties.json.
        "n_points": n_points,
        "production_restricted": {
            "scale": ps, "res": pr, "norm": pn, "n_points": pnp,
        },
        "point_detail": {
            "scale": {f"{ch}/{p}": pts["scale"]
                      for (ch, p), pts in shape_detail.items()},
            "res": {f"{ch}/{p}": pts["res"]
                    for (ch, p), pts in shape_detail.items()},
            "norm": {f"{ch}/{e}": pts
                     for (ch, e), pts in norm_detail.items()},
        },
        "warnings": warnings,
    }
    outpath = os.path.join(srspaths.interpolation_closure_dir(mhc),
                           "loo_uncertainties.json")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {outpath}")

    for kind, block in (("scale", scale), ("res", res)):
        for channel in sorted(block):
            row = ", ".join(f"{p}={v:.4f}" for p, v in
                            sorted(block[channel].items()))
            print(f"  {kind:<5} {channel:<12} {row}")
    for channel in sorted(norm):
        row = ", ".join(f"{e}={v - 1.0:.1%}" for e, v in
                        sorted(norm[channel].items()))
        print(f"  norm  {channel:<12} {row}")
    for w in warnings:
        print(f"  warning: {w}")


# A max over one or two points is not an envelope; say so rather than
# quoting it as if it were measured.
MIN_ENVELOPE_POINTS = 3

# THE uncertainty rule, for all three families: the rms WITHIN each mHc
# study, then the MAX over studies (user decision 2026-08-12).
#
# The max over all pooled points was a ~3 sigma order statistic used as a
# 1 sigma lnN width: the residuals are unbiased (|mean| < 2% everywhere)
# and Gaussian-like (max/rms 1.8-3.6, against sqrt(2 ln N) = 1.8-2.9
# expected), so the max scaled with how many points a cell happened to
# hold rather than with model accuracy, and one mass point (MHc115_MA27)
# set 15 of 16 below-Z cells. The plain pooled rms is the right 1 sigma
# but hides a real effect: below the Z the per-study rms genuinely varies
# with mHc (observed spread 69-77% against 41% expected from statistics),
# tracking low-mA grid density — MHc160 samples it every 5 GeV and closes
# to 2.2%, MHc115/MHc130 have 15-25 GeV gaps and close to 11-18%.
#
# Taking the rms inside a study makes no single mass point set the value;
# taking the max across studies covers the sparse-grid ones instead of
# averaging them away. A study needs at least MIN_STUDY_POINTS mass points
# to define its own rms, otherwise a one-point study leaks its outlier
# straight back in; the pooled rms is the floor, which is also what a cell
# falls back to when no study qualifies (the pooled rms is a quadratic
# mean of the per-study values, hence never above their max).
MIN_STUDY_POINTS = 2


def _rms(values):
    return float(sum(v * v for v in values) / len(values)) ** 0.5


def _envelope_rms_then_max(points, floor, production_only=False):
    """(envelope, n points, diagnostics) under the rms-then-max rule.

    Used for scale, res and norm alike; the caller supplies the floor and
    decides the cell keying."""
    usable = [p for p in points if p.get("excluded") is None
              and (p["production_pairing"] or not production_only)]
    if not usable:
        return floor, 0, {}
    by_study = {}
    for p in usable:
        by_study.setdefault(p["mhc"], []).append(p["value"])
    per_study = {m: _rms(v) for m, v in sorted(by_study.items())
                 if len(v) >= MIN_STUDY_POINTS}
    pooled = _rms([p["value"] for p in usable])
    value = max([floor, pooled] + list(per_study.values()))
    diag = {
        "rule": "max over studies of the per-study rms",
        "per_study_rms": {f"MHc{m}": v for m, v in per_study.items()},
        "per_study_npoints": {f"MHc{m}": len(v)
                              for m, v in sorted(by_study.items())},
        "studies_below_min_points": [f"MHc{m}" for m, v
                                     in sorted(by_study.items())
                                     if len(v) < MIN_STUDY_POINTS],
        "pooled_rms": pooled,
        "pooled_max": max(p["value"] for p in usable),
        "driver": (max(per_study, key=per_study.get) if per_study
                   else "pooled_rms"),
    }
    return value, len(usable), diag


def _reachable(study_channel, bin_label):
    """Can any baseline mass point in this mA bin use this study channel as
    its production pairing? The SR3Mu rule (highM iff mHc>=100 and mA>=60)
    makes e.g. highM/belowZ impossible, so no nuisance is owed there."""
    for mp in srspaths.masspoints_config()["baseline"]:
        if not mp.startswith("MHc") or "_MA" not in mp:
            continue
        mA = float(mp[mp.index("_MA") + 3:])
        if interpolation_config.norm_ma_bin(mA) != bin_label:
            continue
        if study_channel == "SR1E2Mu":
            return True
        if interpolation_config.study_channel_for("SR3Mu", mp) == study_channel:
            return True
    return False


def export_loo_pooled(mhcs, write_config=False):
    """Pool every study's LOO points and bin the norm envelope in mA.

    The norm nuisance is keyed (channel, era, mA bin) with NO mHc
    dependence: the joint yield surface is one global model, so its error
    belongs to the (mHc, mA) plane rather than to a study, and a per-study
    max stops being an estimator once mA is binned (most split cells hold
    only a point or two).

    scale/res are pooled the same way and reported for reference only —
    the shape parametrizations are still fitted per study, so the per-mHc
    loo_uncertainties.json files stay authoritative for them.
    """
    shape_all = {}     # (channel, period) -> {"scale": [], "res": []}
    norm_all = {}      # (channel, era, ma_bin) -> [records]
    warnings, grids = [], {}
    for mhc in mhcs:
        shape_detail, norm_detail, grid, warn = collect_loo_points(
            mhc)
        grids[mhc] = grid
        warnings.extend(f"[MHc{mhc}] {w}" for w in warn)
        for key, slot in shape_detail.items():
            dst = shape_all.setdefault(key, {"scale": [], "res": []})
            for kind in ("scale", "res"):
                dst[kind].extend({**p, "mhc": mhc} for p in slot[kind])
        for (channel, era), pts in norm_detail.items():
            for p in pts:
                bin_label = interpolation_config.norm_ma_bin(p["mA"])
                norm_all.setdefault((channel, era, bin_label), []).append(
                    {**p, "mhc": mhc})

    def norm_block(production_only):
        floor = interpolation_config.UNCERTAINTY_NORM_FLOOR
        out, counts, spread = {}, {}, {}
        for (channel, era, bin_label), pts in norm_all.items():
            value, n, diag = _envelope_rms_then_max(
                pts, floor, production_only)
            key = f"{channel}/{era}/{bin_label}"
            counts[key] = n
            if not n:
                continue
            out.setdefault(channel, {}).setdefault(era, {})[bin_label] = \
                1.0 + value
            spread[key] = diag
        # An empty cell is either structurally unreachable (the pairing rule
        # means no production point can land there) or a genuine coverage
        # hole. The first needs no nuisance at all; the second must not
        # silently take the floor, so it inherits the channel's worst
        # populated bin and is flagged.
        for (channel, era, bin_label) in sorted(norm_all):
            if out.get(channel, {}).get(era, {}).get(bin_label) is not None:
                continue
            key = f"{channel}/{era}/{bin_label}"
            if production_only and not _reachable(channel, bin_label):
                spread[key] = {"unreachable": "no production point of this "
                                              "channel lies in this mA bin"}
                continue
            populated = out.get(channel, {}).get(era, {})
            if not populated:
                continue
            value = max(populated.values())
            out[channel][era][bin_label] = value
            spread[key] = {"fallback_from": sorted(populated)}
            warnings.append(
                f"[norm/{key}] no usable LOO point in this mA bin; "
                f"falling back to the channel's worst populated bin "
                f"({value - 1.0:.3f})")
        for key, n in counts.items():
            if 0 < n < MIN_ENVELOPE_POINTS and production_only:
                warnings.append(
                    f"[norm/{key}] envelope rests on {n} point(s); a max over "
                    f"fewer than {MIN_ENVELOPE_POINTS} is not an envelope")
        return out, counts, spread

    def shape_block(production_only):
        """scale/res under the same rms-then-max rule as norm.

        They are NOT binned in mA: res sits at its floor in every cell, and
        scale's region-to-region variation has no consistent pattern across
        channels (worst below-Z in some, on-Z in others) — noise across
        6-32-point cells rather than the grid-density effect that justifies
        binning norm."""
        floors = {"scale": interpolation_config.UNCERTAINTY_SCALE_FLOOR,
                  "res": interpolation_config.UNCERTAINTY_RES_FLOOR}
        out = {"scale": {}, "res": {}}
        counts, detail = {}, {}
        for (channel, period), pts in shape_all.items():
            for kind in ("scale", "res"):
                v, n, diag = _envelope_rms_then_max(
                    pts[kind], floors[kind], production_only)
                out[kind].setdefault(channel, {})[period] = v
                counts[f"{kind}/{channel}/{period}"] = n
                detail[f"{kind}/{channel}/{period}"] = diag
        return out["scale"], out["res"], counts, detail

    norm, n_norm, spread = norm_block(production_only=False)
    pnorm, pn_norm, pspread = norm_block(production_only=True)
    scale, res, n_shape, d_shape = shape_block(production_only=False)
    pscale, pres, pn_shape, pd_shape = shape_block(production_only=True)

    for channel, per_era in pnorm.items():
        for era, per_bin in per_era.items():
            for bin_label, v in per_bin.items():
                if v - 1.0 > interpolation_config.UNCERTAINTY_NORM_WARN:
                    warnings.append(
                        f"[norm/{channel}/{era}/{bin_label}] pooled envelope "
                        f"{v - 1.0:.3f} exceeds the "
                        f"{interpolation_config.UNCERTAINTY_NORM_WARN:g} warn "
                        "threshold")

    names = build_nuisance_names(pscale, pres, pnorm)

    payload = {
        "meta": {
            "strategy": "leave-one-out, POOLED over mHc; the norm envelope "
                        "is binned in mA",
            "mhc_pooled": sorted(mhcs),
            "grid_ma": {f"MHc{m}": g for m, g in sorted(grids.items())},
            "channels": LOO_CHANNELS,
            "norm_ma_bins": [[lab, lo, hi] for lab, lo, hi in
                             interpolation_config.NORM_MA_BINS],
            "rules": {
                "norm": "per (channel, era, mA bin), pooled over mHc: the "
                        "rms WITHIN each study, then the MAX over studies "
                        f"holding >= {MIN_STUDY_POINTS} mass points; floored "
                        "by the pooled rms and by "
                        f"{interpolation_config.UNCERTAINTY_NORM_FLOOR}, warn "
                        f"above {interpolation_config.UNCERTAINTY_NORM_WARN}. "
                        "per_study_detail keeps every input visible: the "
                        "per-study rms and point counts, the studies skipped "
                        "for having too few points, the pooled rms, the old "
                        "plain max, and which study drove the value",
                "scale_res": "same rms-then-max rule, per (channel, period), "
                             "NOT binned in mA and with no mHc dependence "
                             "(the rule already pools studies). The shape "
                             "parametrizations are themselves (mHc, mA) "
                             "surfaces, so a per-study envelope would be "
                             "measuring one slice of a global model.",
                "excluded": "extrapolation (grid endpoints), non-good direct "
                            "fit (shape only), missing records",
                "empty_bins": "a (channel, era, mA bin) with no usable point "
                              "takes the channel's worst populated bin and is "
                              "listed in warnings — never the bare floor",
            },
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "norm": norm,
        "scale": scale, "res": res,
        "nuisances": names,
        "n_points": {"norm": n_norm, **n_shape},
        "per_study_detail": {**spread, **d_shape},
        "production_restricted": {
            "norm": pnorm, "scale": pscale, "res": pres,
            "n_points": {"norm": pn_norm, **pn_shape},
            "per_study_detail": {**pspread, **pd_shape},
        },
        "warnings": warnings,
    }
    outdir = srspaths.interpolation_closure_dir()
    outpath = os.path.join(outdir, "loo_uncertainties.pooled.json")
    os.makedirs(outdir, exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {outpath}")
    print("  pooled norm envelopes (production pairing), % :")
    for channel in sorted(pnorm):
        for era in sorted(pnorm[channel]):
            row = ", ".join(
                f"{b}={100*(pnorm[channel][era][b] - 1.0):.1f}"
                f"({pn_norm.get(f'{channel}/{era}/{b}', 0)})"
                for b in sorted(pnorm[channel][era]))
            print(f"    {channel:<12} {era:<12} {row}")
    for w in warnings:
        print(f"  warning: {w}")
    return payload


def write_production_config(payload):
    """configs/interpolation_uncertainties.json — the production artifact.

    Takes the production-pairing block: those are the values a datacard for
    a real mass point will use.

    Values are CEIL-rounded to 3 decimals (0.0233 -> 0.024) — conservative
    by construction, never rounded down; norm rounds its envelope part
    (1 + ceil3(v - 1)). Full precision stays in the pooled diagnostic and
    in per_study_detail, which is the audit trail the rounded numbers are
    checked against."""

    def ceil3(v):
        return float(decimal.Decimal(str(v)).quantize(
            decimal.Decimal("0.001"), rounding=decimal.ROUND_CEILING))

    pr = payload["production_restricted"]
    scale = {ch: {p: ceil3(v) for p, v in per.items()}
             for ch, per in pr["scale"].items()}
    res = {ch: {p: ceil3(v) for p, v in per.items()}
           for ch, per in pr["res"].items()}
    norm = {ch: {era: {b: 1.0 + ceil3(v - 1.0) for b, v in bins.items()}
                 for era, bins in per.items()}
            for ch, per in pr["norm"].items()}
    out = {
        "meta": {
            **{k: payload["meta"][k] for k in
               ("strategy", "mhc_pooled", "channels", "norm_ma_bins",
                "rules", "command", "date")},
            "keying": "scale/res VALUES per (study channel, run period); "
                      "norm VALUES per (study channel, era, mA bin). NO mHc "
                      "dependence in any of them — the rule pools studies "
                      "and both the shape and yield models are (mHc, mA) "
                      "surfaces. NAMES are period-level for all three "
                      "families (CMS_interp_{scale,res,norm}_{ch}_"
                      "{13TeV|13p6TeV}): the residual is common-mode across "
                      "a period's eras (r=+0.99 Run2 / +0.80 Run3), so one "
                      "nuisance spans the four era columns, carrying each "
                      "era's own lnN value; the target mA selects the norm "
                      "bin, which never appears in the name. Values are "
                      "keyed by STUDY channel (SR3Mu_lowM/highM) because "
                      "one mA bin can hold both pairings; the nuisance "
                      "NAMES use the production channel SR3Mu, which is "
                      "safe because one datacard holds one mass point.",
            "floors": {
                "scale": interpolation_config.UNCERTAINTY_SCALE_FLOOR,
                "res": interpolation_config.UNCERTAINTY_RES_FLOOR,
                "norm": interpolation_config.UNCERTAINTY_NORM_FLOOR},
            "min_study_points": MIN_STUDY_POINTS,
            "rounding": "scale/res/norm ceil-rounded to 3 decimals "
                        "(never down); per_study_detail and the pooled "
                        "diagnostic keep full precision",
            "source": "closure/interpolation/loo_uncertainties.pooled.json",
        },
        "scale": scale, "res": res, "norm": norm,
        "nuisances": payload["nuisances"],
        "n_points": pr["n_points"],
        "per_study_detail": pr["per_study_detail"],
        "warnings": payload["warnings"],
    }
    path = srspaths.interpolation_uncertainties_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {path}")
    for kind in ("scale", "res"):
        for channel in sorted(out[kind]):
            row = ", ".join(f"{p}={v:.4f}"
                            for p, v in sorted(out[kind][channel].items()))
            print(f"  {kind:<5} {channel:<12} {row}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", type=int, help="compute for one mHc")
    parser.add_argument("--all", action="store_true",
                        help="every mHc in the baseline grid")
    parser.add_argument("--loo", action="store_true",
                        help="aggregate the leave-one-out sweep "
                             "(closure/interpolation/loo/MHc{X}_MA{Y}/ dirs) "
                             "into the per-study closure/interpolation/"
                             "MHc{X}/loo_uncertainties.json diagnostics")
    parser.add_argument("--pooled", action="store_true",
                        help="with --loo: also pool every study's LOO points "
                             "into closure/interpolation/"
                             "loo_uncertainties.pooled.json, where the norm "
                             "envelope is binned in mA and nothing carries an "
                             "mHc dependence (needs every mHc)")
    parser.add_argument("--write-config", action="store_true",
                        help="with --pooled: also write the production "
                             "configs/interpolation_uncertainties.json")
    args = parser.parse_args()
    if not args.mhc and not args.all:
        parser.error("pass --mhc N and/or --all")
    if not args.loo:
        parser.error("--loo is the only supported mode: with full-grid "
                     "anchors the leave-one-out sweep is where the "
                     "uncertainties come from")
    if args.write_config and not args.pooled:
        parser.error("--write-config needs --pooled")

    mhcs = interpolation_config.mhc_grid() if args.all else [args.mhc]

    for mhc in mhcs:
        export_loo_one(mhc)
    if args.pooled:
        if not args.all:
            parser.error("--pooled needs every study; pass --all")
        payload = export_loo_pooled(mhcs)
        if args.write_config:
            write_production_config(payload)


if __name__ == "__main__":
    main()
