#!/usr/bin/env python3
import os
import argparse
import json
import glob
import ROOT
import logging
import re

import srspaths

parser = argparse.ArgumentParser()
parser.add_argument("--era", type=str, required=True,
                    help="2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix, Run2, Run3, All")
parser.add_argument("--channel", type=str, default="Combined",
                    choices=["Combined", "SR1E2Mu", "SR3Mu"],
                    help="Analysis channel (default: Combined)")
parser.add_argument("--method", type=str, required=True, help="Baseline, ParticleNet")
parser.add_argument("--blind", action="store_true",
                    help="Collect limits from the {method}_blind template segment")
parser.add_argument("--signal-source", type=str, default="mc-signal",
                    choices=["mc-signal", "interp-signal"],
                    help="Template signal source (interp-signal: the scan "
                         "grid of configs/grid.json; Baseline only)")
parser.add_argument("--mode", type=str, default="BR", choices=["BR", "xsec"],
                    help="Limit unit: BR (relative branching ratio, default) or xsec (sigma(pp->ttbar) x B_sig in fb)")
parser.add_argument("--available-only", action="store_true",
                    help="Discover and collect mass points with existing Combine output files for the requested settings")
parser.add_argument("--masspoint", type=str, default=None,
                    help="Collect a single mass point only (must be in the configured set)")
parser.add_argument("--output", type=str, default=None,
                    help="Override the output JSON path (default: results/json/... standard path)")
parser.add_argument("--debug", action='store_true', default=False, help="Enable debug logging")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
logger = logging.getLogger(__name__)

# Validate era
VALID_ERAS = [
    "2016preVFP", "2016postVFP", "2017", "2018",
    "2022", "2022EE", "2023", "2023BPix",
    "Run2", "Run3", "All"
]
if args.era not in VALID_ERAS:
    raise ValueError(f"Invalid era: {args.era}. Must be one of {VALID_ERAS}")

_masspoints_config = srspaths.masspoints_config()

if args.signal_source == "interp-signal":
    if args.method not in ("Baseline", "ParticleNet"):
        raise ValueError(f"Invalid method: {args.method}")
    # The scan grid is the mass-point set; each point resolves to its
    # template-sharing group seed for path construction. Baseline scans
    # configs/grid.json (2467 points); ParticleNet scans
    # configs/pnet_grid.json (155 points, reach [82.5, 97.5]).
    from interpolation_config import masspoint_name
    _grid_cfg = (srspaths.grid_config() if args.method == "Baseline"
                 else srspaths.pnet_grid_config())
    MASSPOINTs = [
        masspoint_name(v, int(key.replace("MHc", "")))
        for key, entry in sorted(
            _grid_cfg["grids"].items(),
            key=lambda kv: int(kv[0].replace("MHc", "")))
        for v in entry["grid"]
    ]
elif args.method == "Baseline":
    MASSPOINTs = _masspoints_config["baseline"]
elif args.method == "ParticleNet":
    MASSPOINTs = _masspoints_config["particlenet"]
else:
    raise ValueError(f"Invalid method: {args.method}. Must be Baseline or ParticleNet")

# Reference cross-section and normalization constants
# Use 13 TeV ttbar cross-section for ALL eras (signal samples already scaled to 13 TeV reference)
REFERENCE_XSEC = 5.0  # fb
# NNLO+NNLL, https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
TTBAR_XEC_13TEV = 833.9e3  # fb
TTBAR_XEC_13p6TEV = 923.6e3  # fb
BR_TTBAR_TO_LEPTON = 2 * 0.5456  # 2 for charge conjugation, 0.5456 for non-hadronic decay of two W bosons


def _convert(r, mode):
    """Convert signal-strength limit r to BR or xsec units."""
    if mode == "BR":
        return r * REFERENCE_XSEC / TTBAR_XEC_13TEV / BR_TTBAR_TO_LEPTON
    if mode == "xsec":
        return r * REFERENCE_XSEC / BR_TTBAR_TO_LEPTON  # fb; equals B_sig * sigma_ttbar
    raise ValueError(f"Unknown mode: {mode}")


def _seed_of(masspoint):
    """Group seed for interp-signal path nesting (identity for mc-signal)."""
    if args.signal_source != "interp-signal":
        return None
    import interpolation_config
    return interpolation_config.group_seed(masspoint, args.method)


def parseAsymptoticLimit(masspoint, method, era, channel="Combined", mode="BR", blind=False):
    """Parse asymptotic limits from Combine ROOT output file."""
    root_file = srspaths.asymptotic_root(
        masspoint, method, era, channel, blind=blind,
        source=args.signal_source, seed_masspoint=_seed_of(masspoint))

    logger.debug(f"Reading limits from: {root_file}")

    f = ROOT.TFile.Open(root_file)
    if not f:
        raise FileNotFoundError(f"Limit file not found: {root_file}")
    if f.IsZombie():
        f.Close()
        raise RuntimeError(f"ROOT file is zombie/corrupt: {root_file}")

    limit = f.Get("limit")
    if not limit:
        f.Close()
        raise RuntimeError(f"TTree 'limit' not found in {root_file}")

    branching_ratios = {}
    try:
        for idx, entry in enumerate(limit):
            branching_ratios[idx] = _convert(entry.limit, mode)
    except Exception as e:
        logger.error(f"Error parsing {masspoint}: {e}")
        raise ValueError(e)
    finally:
        f.Close()

    # Map indices to limit types
    out = {
        "exp-2": branching_ratios[0],
        "exp-1": branching_ratios[1],
        "exp0": branching_ratios[2],
        "exp+1": branching_ratios[3],
        "exp+2": branching_ratios[4],
        "obs": branching_ratios[5]
    }

    return out


def masspoint_sort_key(masspoint):
    """Sort by (MHc, MA) incl. p-notation, stable fallback otherwise."""
    try:
        mhc, ma = srspaths.masspoint_mhc_ma(masspoint)
        return (mhc, float(ma), masspoint)
    except (ValueError, IndexError):
        return (10**9, 10**9.0, masspoint)


def discoverAvailableMasspoints(method, era, channel="Combined", blind=False):
    """Discover mass points with existing asymptotic output for the requested mode."""
    if args.signal_source == "interp-signal":
        # The grid is the authoritative point list; discovery just filters
        # by output existence (seed dirs and nested member dirs alike).
        return [mp for mp in MASSPOINTs if os.path.isfile(
            srspaths.asymptotic_root(mp, method, era, channel, blind=blind,
                                     source="interp-signal",
                                     seed_masspoint=_seed_of(mp)))]
    masspoints = set()
    pattern = os.path.join(
        srspaths.module_dir(), "templates", "*",
        srspaths.method_segment(method, blind), "mc-signal", era, channel,
        "combine_output", "asymptotic",
        "higgsCombine*.AsymptoticLimits.mH120.root",
    )
    templates_prefix = os.path.join(srspaths.module_dir(), "templates") + os.sep
    for path in glob.glob(pattern):
        masspoint = path[len(templates_prefix):].split(os.sep, 1)[0]
        expected_path = srspaths.asymptotic_root(masspoint, method, era, channel, blind=blind)
        if os.path.isfile(expected_path):
            masspoints.add(masspoint)
    return sorted(masspoints, key=masspoint_sort_key)


if __name__ == "__main__":
    logger.info(f"Collecting limits for era={args.era}, method={args.method}")

    if args.available_only:
        MASSPOINTs = discoverAvailableMasspoints(
            args.method, args.era, channel=args.channel, blind=args.blind,
        )
        logger.info(f"Found {len(MASSPOINTs)} available mass points for requested outputs")
        if not MASSPOINTs:
            raise RuntimeError("No existing Combine output files found for requested settings")

    if args.masspoint is not None:
        if args.masspoint not in MASSPOINTs:
            raise ValueError(
                f"Mass point {args.masspoint} is not in the configured set for method {args.method}"
            )
        MASSPOINTs = [args.masspoint]

    limits = {}
    failed_masspoints = []

    for masspoint in MASSPOINTs:
        try:
            limits[masspoint] = parseAsymptoticLimit(
                masspoint, args.method, args.era,
                channel=args.channel, mode=args.mode, blind=args.blind,
            )
            logger.debug(f"  {masspoint}: exp0 = {limits[masspoint]['exp0']:.2e}")
        except FileNotFoundError as e:
            logger.warning(f"  {masspoint}: SKIPPED - {e}")
            failed_masspoints.append(masspoint)
        except Exception as e:
            logger.error(f"  {masspoint}: ERROR - {e}")
            failed_masspoints.append(masspoint)

    if not limits:
        raise RuntimeError("No limits were successfully parsed")

    # Summary
    logger.info(f"Successfully parsed {len(limits)}/{len(MASSPOINTs)} mass points")
    if failed_masspoints:
        logger.warning(f"Failed mass points: {failed_masspoints}")

    # Save results
    if args.output is not None:
        outpath = args.output
    else:
        outpath = srspaths.limits_json(args.era, args.channel, args.method,
                                       mode=args.mode, blind=args.blind,
                                       source=args.signal_source)
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    with open(outpath, "w") as f:
        json.dump(limits, f, indent=4)

    logger.info(f"Saved limits to {outpath}")
