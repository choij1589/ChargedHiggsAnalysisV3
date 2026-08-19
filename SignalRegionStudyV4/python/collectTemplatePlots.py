#!/usr/bin/env python3
"""Collect the per-mass-point diagnostic artifacts of the TEMPLATE points.

"Template points" are the curated handful of mass points kept as the
representative record of the analysis — one per corner of each arm's
reach — so that the fit diagnostics of the campaign can be read without
walking 572 gitignored template dirs.  Same promotion rule as
collectPnetScorePlots.py: the plots live in the template tree, only the
ones worth keeping are copied into the tracked results tree.

    templates/{seed}/{method}/interp-signal/{era}/{channel}/
        combine_output/gof/gof_plot.{pdf,png}
        combine_output/impacts_obs/impacts[_filtered][_summary].pdf
        combine_output/fitdiag/nuisance_pulls[_filtered]_{fit}.pdf
        combine_output/fitdiag/plots_mass/*.png
        scores/{region}/*.png                     (ParticleNet only)
 -> results/templates/{method}/{masspoint}/
        gof/gof_{era}_{channel}.{pdf,png}
        impacts/impacts[_filtered][_summary]_{era}_{channel}.pdf
        pulls/nuisance_pulls[_filtered]_{fit}_{era}_{channel}.pdf
        mass/{prefit,postfit_b,...}_mass_{era_scope}_{channel_scope}.png
        scores/{era}_{channel}[_TTZ2E1Mu]_{plot}.png
        limits.json                               (this point, per channel)
        significance.json                         (this point, per channel)

GoF runs at All x {SR1E2Mu, SR3Mu, Combined}; impacts, FitDiagnostics and
the mass plots at All/Combined only — so the per-artifact channel lists
below are not the same.  The mass plots' own filenames already carry the
era/channel SCOPE they project onto (Run2/Run3/All x the three channels),
which is a different axis from the fitted category and is left untouched.

    python3 python/collectTemplatePlots.py
    python3 python/collectTemplatePlots.py --point Baseline:MHc145_MA45
    python3 python/collectTemplatePlots.py --eras All Run2 Run3
"""
import argparse
import glob
import json
import os
import shutil

import interpolation_config
import srspaths

# The curated representative points, per arm.  Baseline spans the reach
# from the lowest mA study to the mA = mHc - 5 edge; ParticleNet covers
# its three trained anchors (mA = 85/90/95) at three different mHc.
DEFAULT_POINTS = {
    "Baseline": ["MHc70_MA15", "MHc100_MA60", "MHc130_MA90", "MHc160_MA155"],
    "ParticleNet": ["MHc160_MA85", "MHc130_MA90", "MHc100_MA95"],
}

GOF_CHANNELS = ["SR1E2Mu", "SR3Mu", "Combined"]
FIT_CHANNELS = ["Combined"]
SCORE_CHANNELS = ["SR1E2Mu", "SR3Mu", "Combined"]
LIMIT_CHANNELS = ["SR1E2Mu", "SR3Mu", "Combined"]
LIMIT_MODES = ["BR", "xsec"]

IMPACT_PLOTS = ["impacts", "impacts_filtered", "impacts_filtered_summary"]
# Written by runPullPlots.sh; the suffix is the --pull-fit mode it ran
# with, "_both" in the production DAGs and "" for the b-only default.
PULL_PLOTS = ["nuisance_pulls", "nuisance_pulls_filtered"]
PULL_SUFFIXES = ["_both", ""]


class Collector:
    """Copies with a running tally of what was found and what was not."""

    def __init__(self):
        self.copied = 0
        self.missing = []

    def copy(self, src, outdir, name, required=True):
        if not os.path.exists(src):
            if required:
                self.missing.append(src)
            return False
        os.makedirs(outdir, exist_ok=True)
        shutil.copy2(src, os.path.join(outdir, name))
        self.copied += 1
        return True


def point_dir(masspoint, method, era, channel, source):
    """Template dir of one point — nested under its group seed when the
    point is an interp-signal group member rather than the seed itself."""
    if source == "interp-signal":
        seed = interpolation_config.group_seed(masspoint, method)
        if seed != masspoint:
            return srspaths.interp_member_dir(seed, masspoint, era, channel,
                                              method=method)
    return srspaths.template_dir(masspoint, method, era, channel,
                                 source=source)


def seed_dir(masspoint, method, era, channel, source):
    """Template dir of the point's GROUP SEED.  The score plots are a
    property of the group (one net, one threshold, shared backgrounds),
    so they are only ever written at the seed."""
    if source == "interp-signal":
        masspoint = interpolation_config.group_seed(masspoint, method)
    return srspaths.template_dir(masspoint, method, era, channel,
                                 source=source)


def collect_gof(coll, masspoint, method, era, outroot, source):
    for channel in GOF_CHANNELS:
        base = os.path.join(point_dir(masspoint, method, era, channel, source),
                            "combine_output", "gof", "gof_plot")
        for ext in ("pdf", "png"):
            coll.copy(f"{base}.{ext}", os.path.join(outroot, "gof"),
                      f"gof_{era}_{channel}.{ext}")


def collect_impacts(coll, masspoint, method, era, outroot, source):
    for channel in FIT_CHANNELS:
        idir = os.path.join(point_dir(masspoint, method, era, channel, source),
                            "combine_output", "impacts_obs")
        for plot in IMPACT_PLOTS:
            coll.copy(os.path.join(idir, f"{plot}.pdf"),
                      os.path.join(outroot, "impacts"),
                      f"{plot}_{era}_{channel}.pdf")


def collect_pulls(coll, masspoint, method, era, outroot, source):
    for channel in FIT_CHANNELS:
        fdir = os.path.join(point_dir(masspoint, method, era, channel, source),
                            "combine_output", "fitdiag")
        for plot in PULL_PLOTS:
            # Take the first --pull-fit variant present; a point is run
            # with one mode, so at most one of these exists.
            for suffix in PULL_SUFFIXES:
                if coll.copy(os.path.join(fdir, f"{plot}{suffix}.pdf"),
                             os.path.join(outroot, "pulls"),
                             f"{plot}{suffix}_{era}_{channel}.pdf",
                             required=False):
                    break
            else:
                coll.missing.append(
                    os.path.join(fdir, f"{plot}{PULL_SUFFIXES[0]}.pdf"))


def collect_mass(coll, masspoint, method, era, outroot, source):
    """Prefit, postfit b/s and the prefit-vs-postfit overlays."""
    for channel in FIT_CHANNELS:
        mdir = os.path.join(point_dir(masspoint, method, era, channel, source),
                            "combine_output", "fitdiag", "plots_mass")
        plots = sorted(glob.glob(os.path.join(mdir, "*.png")))
        if not plots:
            coll.missing.append(os.path.join(mdir, "*.png"))
            continue
        for src in plots:
            coll.copy(src, os.path.join(outroot, "mass"),
                      os.path.basename(src))


def collect_scores(coll, masspoint, method, era, outroot, source):
    """ParticleNet score panels, including the TTZ2E1Mu control region
    the per-channel jobs emit alongside the signal region."""
    for channel in SCORE_CHANNELS:
        sdir = os.path.join(seed_dir(masspoint, method, era, channel, source),
                            "scores")
        regions = [(channel, f"{era}_{channel}")]
        if channel != "Combined":
            regions.append(("TTZ2E1Mu", f"{era}_{channel}_TTZ2E1Mu"))
        for region, tag in regions:
            plots = sorted(glob.glob(os.path.join(sdir, region, "*.png")))
            if not plots:
                coll.missing.append(os.path.join(sdir, region, "*.png"))
                continue
            for src in plots:
                name = os.path.splitext(os.path.basename(src))[0]
                coll.copy(src, os.path.join(outroot, "scores"),
                          f"{tag}_{name}.png")


def collect_limits(coll, masspoint, method, era, outroot, source):
    """This point's own limit, per channel and in both units, lifted out
    of the campaign JSONs collectLimits.py already writes.

    Taken from the collected JSON rather than re-read from the point's
    Combine output so the bundle can never disagree with the published
    limit curves — the same numbers, indexed by mass point instead of by
    channel.  A missing entry means the collection has not been rerun
    since this point was produced.
    """
    payload = {}
    for mode in LIMIT_MODES:
        for channel in LIMIT_CHANNELS:
            path = srspaths.limits_json(era, channel, method, mode=mode,
                                        source=source)
            if not os.path.exists(path):
                coll.missing.append(path)
                continue
            with open(path) as f:
                limits = json.load(f)
            if masspoint not in limits:
                coll.missing.append(f"{path}::{masspoint}")
                continue
            payload.setdefault(mode, {})[channel] = limits[masspoint]
    if not payload:
        return
    os.makedirs(outroot, exist_ok=True)
    with open(os.path.join(outroot, "limits.json"), "w") as f:
        json.dump({"masspoint": masspoint, "method": method, "era": era,
                   "signal_source": source, "limits": payload}, f, indent=2)
        f.write("\n")
    coll.copied += 1


def collect_significance(coll, masspoint, method, era, outroot, source):
    """This point's observed local significance, per channel, lifted out
    of the collectSignificance.py JSON for the same reason limits are
    lifted out of the limit JSONs.  Uncapped Z (a deficit keeps its sign)
    and the one-sided tail p; both LOCAL, no trials correction."""
    source_infix = "" if source == "mc-signal" else f".{source}"
    path = os.path.join(srspaths.module_dir(), "results", "json",
                        f"significance.{era}{source_infix}.json")
    if not os.path.exists(path):
        coll.missing.append(path)
        return
    with open(path) as f:
        record = json.load(f)
    entry = record.get(method, {}).get(masspoint)
    if not entry:
        coll.missing.append(f"{path}::{method}/{masspoint}")
        return
    os.makedirs(outroot, exist_ok=True)
    with open(os.path.join(outroot, "significance.json"), "w") as f:
        json.dump({"masspoint": masspoint, "method": method, "era": era,
                   "signal_source": source, "significance": entry},
                  f, indent=2, sort_keys=True)
        f.write("\n")
    coll.copied += 1


def collect_breakdown(coll, masspoint, method, era, outroot, source):
    """The grouped uncertainty breakdown: the plot1DScan panel, plus this
    point's component sigmas lifted out of the collectBreakdown.py JSON
    for the same reason limits and significance are lifted out of theirs
    -- the bundle can then never disagree with the published values."""
    src = point_dir(masspoint, method, era, "Combined", source)
    bdir = os.path.join(src, "combine_output", "breakdown")
    for name in ("breakdown.pdf", "breakdown.png"):
        coll.copy(os.path.join(bdir, name), outroot, name)

    source_infix = "" if source == "mc-signal" else f".{source}"
    path = os.path.join(srspaths.module_dir(), "results", "json",
                        f"breakdown.{era}{source_infix}.json")
    if not os.path.exists(path):
        coll.missing.append(path)
        return
    with open(path) as f:
        record = json.load(f)
    entry = record.get(method, {}).get(masspoint)
    if not entry:
        coll.missing.append(f"{path}::{method}/{masspoint}")
        return
    os.makedirs(outroot, exist_ok=True)
    with open(os.path.join(outroot, "breakdown.json"), "w") as f:
        json.dump({"masspoint": masspoint, "method": method, "era": era,
                   "signal_source": source, "breakdown": entry},
                  f, indent=2, sort_keys=True)
        f.write("\n")
    coll.copied += 1


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--point", action="append", metavar="METHOD:MASSPOINT",
                        help="override the curated list; repeatable")
    parser.add_argument("--eras", nargs="+", default=["All"],
                        help="fitted era targets to collect [All]")
    parser.add_argument("--signal-source", default="interp-signal",
                        choices=["interp-signal", "mc-signal"])
    parser.add_argument("--output-dir",
                        default=os.path.join(srspaths.module_dir(),
                                             "results", "templates"))
    args = parser.parse_args()

    if args.point:
        points = {}
        for spec in args.point:
            if ":" not in spec:
                raise ValueError(f"--point expects METHOD:MASSPOINT, got {spec!r}")
            method, masspoint = spec.split(":", 1)
            if method not in DEFAULT_POINTS:
                raise ValueError(f"unknown method {method!r}")
            points.setdefault(method, []).append(masspoint)
    else:
        points = DEFAULT_POINTS

    coll = Collector()
    for method in sorted(points):
        for masspoint in points[method]:
            outroot = os.path.join(args.output_dir, method, masspoint)
            for era in args.eras:
                collect_gof(coll, masspoint, method, era, outroot,
                            args.signal_source)
                collect_impacts(coll, masspoint, method, era, outroot,
                                args.signal_source)
                collect_pulls(coll, masspoint, method, era, outroot,
                              args.signal_source)
                collect_mass(coll, masspoint, method, era, outroot,
                             args.signal_source)
                collect_limits(coll, masspoint, method, era, outroot,
                               args.signal_source)
                collect_significance(coll, masspoint, method, era, outroot,
                                     args.signal_source)
                collect_breakdown(coll, masspoint, method, era, outroot,
                                  args.signal_source)
                if method == "ParticleNet":
                    collect_scores(coll, masspoint, method, era, outroot,
                                   args.signal_source)
            print(f"{method}/{masspoint} -> {outroot}")

    print(f"Copied {coll.copied} files -> {args.output_dir}")
    if coll.missing:
        print(f"{len(coll.missing)} missing sources:")
        for src in coll.missing:
            print(f"  {src}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
