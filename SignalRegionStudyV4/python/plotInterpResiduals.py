#!/usr/bin/env python3
"""The LOO residuals the interpolation nuisances are sized from.

One PNG per uncertainty cell: the signed leave-one-out residual against mA,
one colour per mHc study, with the adopted envelope drawn as a symmetric
band about zero and the cell's pooled rms dotted. Where `plotInterpNuisances`
shows the rms-then-max RULE (per-study summaries against mHc), this shows the
residuals the rule was applied to -- so the claims behind the choices become
checkable off the plot: that the residuals are unbiased (the signed mean is
printed on every panel), that below-Z is the wide region because the low-mA
grid is sparse, and that the ParticleNet res residual is a coherent negative
bias, which is why its rms is taken about zero rather than about its mean.

Covers both arms:

  Baseline      scale, res  per (study channel, run period)
                norm        per (study channel, era); the band steps at the
                            NORM_MA_BINS edges, so the mA binning is visible
                            sitting on the residual cloud it was chosen for
  ParticleNet   res         per (production channel, run period)
                eff         per (production channel, era)

Baseline residuals come from the raw LOO closures through
`exportInterpUncertainties.collect_loo_points(mhc, signed=True)`, so the
plotted set is gated exactly like the sized set; the adopted values come
from closure/interpolation/loo_uncertainties.pooled.json. ParticleNet
residuals are already signed in the closure/pnet shards.

Every panel asserts its own point count against the config's `n_points`, and
re-derives rms-then-max from the plotted points -- a figure that silently
disagreed with the number it justifies would be worse than no figure.

Runs after `exportInterpUncertainties.py --loo --all --pooled` and
`exportPnetUncertainties.py`. JSON-only apart from the ROOT canvases; safe
on the login node.

  python3 plotInterpResiduals.py --all
  python3 plotInterpResiduals.py --method Baseline --family scale \\
                                 --channel SR1E2Mu --key Run2
"""
import argparse
import json
import os
from collections import defaultdict

import exportInterpUncertainties as exp_base
import exportPnetUncertainties as exp_pnet
import interp_plot_utils
import interpolation_config
import pnet_interp_config
import run_period_utils
import srspaths

# Baseline: the NORM_MA_BINS span [15, 155], so a frame of 0-200 would push
# the three bin edges into the middle of the plot and flatten the step.
BASELINE_XRANGE = (10.0, 160.0)
# ParticleNet: the arm's own reach, mA in [82.5, 97.5].
PNET_XRANGE = (82.0, 98.0)


def _rms(values):
    return (sum(v * v for v in values) / len(values)) ** 0.5


def _mean(values):
    return sum(values) / len(values) if values else 0.0


def _driver_mhc(driver):
    """The Baseline pooled JSON stores an int, the ParticleNet config a
    string "MHc115"; both print as one label."""
    if driver is None or driver == "pooled_rms":
        return "pooled rms"
    return f"MHc{driver}" if isinstance(driver, int) else str(driver)


def _check_count(label, plotted, expected):
    """The panel must show exactly the points the envelope was built from."""
    if expected is not None and plotted != expected:
        raise RuntimeError(
            f"{label}: plotted {plotted} points but the config reports "
            f"n_points={expected} -- the figure would misrepresent the "
            "number it is meant to justify")


def _check_envelope(label, series, adopted):
    """Recompute rms-within-study -> max-across-studies from the plotted
    residuals; the adopted value must cover it. The config is ceil-rounded
    to 3 decimals, so allow that much slack in the covering direction."""
    per_study = [_rms([v for _, v in s["used"]]) for s in series.values()
                 if len(s["used"]) >= exp_base.MIN_STUDY_POINTS]
    if not per_study:
        return
    recomputed = max(per_study)
    if recomputed > adopted + 1e-3:
        raise RuntimeError(
            f"{label}: rms-then-max over the plotted residuals gives "
            f"{recomputed:.4f}, above the adopted {adopted:.4f}")


def _info(adopted, pooled, driver, values, pct):
    """Common info block: what the band is, what set it, and the signed mean
    (the unbiasedness / bias check the whole view exists to make visible)."""
    fmt = (lambda v: f"{100 * v:.2f}%") if pct else (lambda v: f"{v:.4f}")
    lines = [f"envelope {fmt(adopted)}, driver {_driver_mhc(driver)}"]
    tail = f"n = {len(values)}, mean = {fmt(_mean(values))}"
    if pooled is not None:
        tail = f"pooled rms {fmt(pooled)}, " + tail
    lines.append(tail)
    return lines


# ------------------------------------------------------------------ Baseline

def _baseline_points(production_only):
    """Signed residual records keyed by uncertainty cell.

    shape[(channel, period)][family][mhc] -> {"used": [...], "unused": [...]}
    norm[(channel, era)][mhc]             -> {"used": [...], "unused": [...]}

    "used" are the points the envelope was built from; "unused" are the
    other pairing variant of the same mass point, drawn open for context.
    """
    shape = defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: {"used": [], "unused": []})))
    norm = defaultdict(lambda: defaultdict(
        lambda: {"used": [], "unused": []}))
    for mhc in interpolation_config.mhc_grid():
        shape_detail, norm_detail, _, _ = exp_base.collect_loo_points(
            mhc, signed=True)
        for (channel, period), families in shape_detail.items():
            for family, points in families.items():
                for p in points:
                    if p["excluded"] is not None:
                        continue
                    kind = ("used"
                            if p["production_pairing"] or not production_only
                            else "unused")
                    shape[(channel, period)][family][mhc][kind].append(
                        (p["mA"], p["signed"]))
        for (channel, era), points in norm_detail.items():
            for p in points:
                if p["excluded"] is not None:
                    continue
                kind = ("used"
                        if p["production_pairing"] or not production_only
                        else "unused")
                norm[(channel, era)][mhc][kind].append((p["mA"], p["signed"]))
    return shape, norm


def _norm_band(per_bin, detail, channel, era):
    """Adopted-envelope segments of one norm panel, one per mA bin.

    A structurally unreachable bin contributes no segment -- no nuisance is
    emitted there, so painting a band would be a lie and the gap is the
    statement. A bin that inherited its value from another is flagged so it
    can be drawn as a dashed, fainter band.
    """
    segments = []
    for label, lo, hi in interpolation_config.NORM_MA_BINS:
        if label not in per_bin:
            continue
        diag = detail.get(f"{channel}/{era}/{label}", {})
        if "unreachable" in diag:
            continue
        segments.append((lo, hi, per_bin[label] - 1.0,
                         "fallback_from" in diag))
    return segments


def plot_baseline(block, outdir, production_only, keep_cell):
    detail = block["per_study_detail"]
    counts = block["n_points"]
    shape, norm = _baseline_points(production_only)
    written = defaultdict(int)

    for family, ytitle, pct in (
            ("scale",
             "(x_{0}^{pred} #minus x_{0}^{direct}) / #sigma_{eff}", False),
            ("res",
             "#sigma_{eff}^{pred} / #sigma_{eff}^{direct} #minus 1 [%]",
             True)):
        for channel in interpolation_config.STUDY_CHANNELS:
            for period in run_period_utils.RUN_PERIODS:
                series = shape[(channel, period)][family]
                if not series or not keep_cell(family, channel, period):
                    continue
                key = f"{family}/{channel}/{period}"
                adopted = block[family][channel][period]
                diag = detail.get(key, {})
                label = f"{family} {channel} {period}"
                used = [v for s in series.values() for _, v in s["used"]]
                _check_count(label, len(used), counts.get(key))
                _check_envelope(label, series, adopted)
                if interp_plot_utils.plot_residual_vs_mA(
                        f"{family}.{channel}.{period}",
                        interp_plot_utils.channel_label(channel),
                        series,
                        {"adopted": [(BASELINE_XRANGE[0], BASELINE_XRANGE[1],
                                      adopted, False)],
                         "pooled": diag.get("pooled_rms")},
                        outdir, period, BASELINE_XRANGE, ytitle,
                        info_lines=_info(adopted, diag.get("pooled_rms"),
                                         diag.get("driver"), used, pct),
                        yscale=100.0 if pct else 1.0):
                    written[family] += 1

    for channel in interpolation_config.STUDY_CHANNELS:
        for era in sorted(block["norm"].get(channel, {})):
            series = norm[(channel, era)]
            if not series or not keep_cell("norm", channel, era):
                continue
            per_bin = block["norm"][channel][era]
            segments = _norm_band(per_bin, detail, channel, era)
            label = f"norm {channel} {era}"
            # n_points for norm is nested and counted per mA bin, so each
            # bin is checked separately rather than against a panel total.
            for bin_label in per_bin:
                expected = counts["norm"].get(f"{channel}/{era}/{bin_label}")
                plotted = sum(
                    1 for s in series.values() for mA, _ in s["used"]
                    if interpolation_config.norm_ma_bin(mA) == bin_label)
                if expected:
                    _check_count(f"{label}/{bin_label}", plotted, expected)
            used = [v for s in series.values() for _, v in s["used"]]
            info = ["envelope " + ", ".join(
                f"{b} {100 * (per_bin[b] - 1.0):.1f}%"
                for b, _, _ in interpolation_config.NORM_MA_BINS
                if b in per_bin),
                f"n = {len(used)}, mean = {100 * _mean(used):.2f}%"]
            if any(inherited for _, _, _, inherited in segments):
                info.append("opposite hatch = value inherited from another bin")
            if interp_plot_utils.plot_residual_vs_mA(
                    f"norm.{channel}.{era}",
                    interp_plot_utils.channel_label(channel),
                    series, {"adopted": segments, "pooled": None},
                    outdir, era, BASELINE_XRANGE,
                    "N_{pred} / N_{meas} #minus 1 [%]",
                    info_lines=info, yscale=100.0):
                written["norm"] += 1
    return written


# --------------------------------------------------------------- ParticleNet

def _pnet_series(rows, select, value_idx, ma_idx):
    """{mhc: {"used": [...], "unused": []}} -- filled markers are the
    trained anchors, open the blind validation points (mA 87 / 92), which
    are the only genuinely out-of-sample tests of eps(mA)."""
    series = defaultdict(lambda: {"used": [], "unused": []})
    for row in rows:
        if not select(row):
            continue
        mhc = int(str(row[0]).replace("MHc", ""))
        kind = ("used" if row[ma_idx] in pnet_interp_config.SEED_MA
                else "unused")
        series[mhc][kind].append((row[ma_idx], row[value_idx]))
    return dict(series)


def plot_pnet(outdir, keep_cell):
    with open(srspaths.pnet_uncertainties_path()) as f:
        payload = json.load(f)
    mhcs = sorted(payload["res"]["SR1E2Mu"]["Run2"]["per_study_rms"],
                  key=lambda k: int(k.replace("MHc", "")))
    wp = pnet_interp_config.DEFAULT_WP
    res_rows = exp_pnet.load_res(mhcs, wp)
    norm_rows = exp_pnet.load_norm(mhcs, wp)
    written = defaultdict(int)

    for channel in pnet_interp_config.CHANNELS:
        for period in pnet_interp_config.PERIODS:
            rec = payload["res"].get(channel, {}).get(period)
            if rec is None or not keep_cell("res", channel, period):
                continue
            series = _pnet_series(
                res_rows,
                lambda r, c=channel, p=period: r[1] == c and r[2] == p, 3, 4)
            label = f"res_pnet {channel} {period}"
            used = [v for s in series.values()
                    for _, v in s["used"] + s["unused"]]
            _check_count(label, len(used), rec["n_points"])
            info = _info(rec["value"], rec["pooled_rms"], rec["driver"],
                         used, True)
            info.append("mean < 0: the cut keeps core events, "
                        "so the peak narrows")
            if interp_plot_utils.plot_residual_vs_mA(
                    f"res_pnet.{channel}.{period}",
                    interp_plot_utils.channel_label(channel),
                    series,
                    {"adopted": [(PNET_XRANGE[0], PNET_XRANGE[1],
                                  rec["value"], False)],
                     "pooled": rec["pooled_rms"]},
                    outdir, period, PNET_XRANGE,
                    "#sigma_{eff}^{cut} / #sigma_{eff}^{nocut} #minus 1 [%]",
                    info_lines=info):
                written["res_pnet"] += 1

        for eras in pnet_interp_config.PERIODS.values():
            for era in eras:
                rec = payload["norm"].get(channel, {}).get(era)
                if rec is None or not keep_cell("eff", channel, era):
                    continue
                series = _pnet_series(
                    norm_rows,
                    lambda r, c=channel, e=era: r[1] == c and r[3] == e, 4, 5)
                label = f"eff_pnet {channel} {era}"
                used = [v for s in series.values()
                        for _, v in s["used"] + s["unused"]]
                _check_count(label, len(used), rec["n_points"])
                info = _info(rec["value"], rec["pooled_rms"], rec["driver"],
                             used, True)
                if interp_plot_utils.plot_residual_vs_mA(
                        f"eff_pnet.{channel}.{era}",
                        interp_plot_utils.channel_label(channel),
                        series,
                        {"adopted": [(PNET_XRANGE[0], PNET_XRANGE[1],
                                      rec["value"], False)],
                         "pooled": rec["pooled_rms"]},
                        outdir, era, PNET_XRANGE,
                        "#varepsilon_{interp} / #varepsilon_{direct} "
                        "#minus 1 [%]",
                        info_lines=info):
                    written["eff_pnet"] += 1
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", action="append", default=None,
                        choices=["Baseline", "ParticleNet"],
                        help="interpolation arm to plot (repeatable)")
    parser.add_argument("--all", action="store_true",
                        help="both arms")
    parser.add_argument("--full-grid", action="store_true",
                        help="Baseline only: use the full-mA-grid block "
                             "instead of the production-pairing one")
    parser.add_argument("--family", help="single family (scale/res/norm or "
                                         "res/eff) for a smoke run")
    parser.add_argument("--channel", help="single channel for a smoke run")
    parser.add_argument("--key", help="single era or run period for a "
                                      "smoke run")
    args = parser.parse_args()

    methods = set(args.method or [])
    if args.all:
        methods = {"Baseline", "ParticleNet"}
    if not methods:
        parser.error("give --method Baseline/ParticleNet or --all")
    if args.full_grid and "Baseline" not in methods:
        parser.error("--full-grid applies to the Baseline arm only")

    def keep_cell(family, channel, key):
        return ((args.family is None or args.family == family)
                and (args.channel is None or args.channel == channel)
                and (args.key is None or args.key == key))

    total = {}
    if "Baseline" in methods:
        path = os.path.join(srspaths.interpolation_closure_dir(),
                            "loo_uncertainties.pooled.json")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} not found -- run exportInterpUncertainties.py "
                "--loo --all --pooled first")
        with open(path) as f:
            payload = json.load(f)
        block = payload if args.full_grid else payload["production_restricted"]
        outdir = srspaths.interpolation_global_plots_dir("residual")
        total.update(plot_baseline(block, outdir, not args.full_grid,
                                   keep_cell))
        print(f"Baseline    -> {outdir}")

    if "ParticleNet" in methods:
        if not os.path.exists(srspaths.pnet_uncertainties_path()):
            raise FileNotFoundError(
                f"{srspaths.pnet_uncertainties_path()} not found -- run "
                "exportPnetUncertainties.py first")
        outdir = srspaths.pnet_closure_plots_dir("residual")
        total.update(plot_pnet(outdir, keep_cell))
        print(f"ParticleNet -> {outdir}")

    for family, n in sorted(total.items()):
        print(f"  {family:10s} {n:3d} panels")
    written = sum(total.values())
    if not written:
        raise RuntimeError("no panel matched the --family/--channel/--key "
                           "selection -- refusing to exit quietly")
    print(f"Wrote {written} residual plots")


if __name__ == "__main__":
    main()
