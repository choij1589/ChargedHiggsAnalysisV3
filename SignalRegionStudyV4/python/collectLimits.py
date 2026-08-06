#!/usr/bin/env python3
import os
import argparse
import json
import glob
import ROOT
import logging
import re

parser = argparse.ArgumentParser()
parser.add_argument("--era", type=str, required=True,
                    help="2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix, Run2, Run3, All")
parser.add_argument("--channel", type=str, default="Combined",
                    choices=["Combined", "SR1E2Mu", "SR3Mu"],
                    help="Analysis channel (default: Combined)")
parser.add_argument("--method", type=str, required=True, help="Baseline, ParticleNet")
parser.add_argument("--limit_type", type=str, default="Asymptotic",
                    help="Asymptotic or HybridNew")
parser.add_argument("--unblind", action="store_true", help="Collect limits from unblind templates")
parser.add_argument("--cnc", action="store_true", help="Collect CnC limits (uses asymptotic_cnc/ directory)")
parser.add_argument("--nsigma", type=float, default=3.0, help="CnC mass window half-width in sigma_voigt (default: 3.0)")
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

# Mass points (loaded from configs/masspoints.json)
_masspoints_json = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "masspoints.json")
with open(_masspoints_json) as _f:
    _masspoints_config = json.load(_f)

if args.method == "Baseline":
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


def parseAsymptoticLimit(masspoint, method, era, binning_suffix="extended", cnc=False, nsigma_tag="3sigma", channel="Combined", mode="BR"):
    """Parse asymptotic limits from Combine ROOT output file."""
    root_file = getAsymptoticLimitPath(masspoint, method, era, binning_suffix, cnc=cnc, nsigma_tag=nsigma_tag, channel=channel)

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


def getAsymptoticLimitPath(masspoint, method, era, binning_suffix="extended", cnc=False, nsigma_tag="3sigma", channel="Combined"):
    """Return the expected AsymptoticLimits ROOT path for one mass point."""
    base_dir = f"templates/{era}/{channel}/{masspoint}/{method}/{binning_suffix}"
    if cnc:
        return f"{base_dir}/combine_output/asymptotic_cnc_{nsigma_tag}/higgsCombine.{masspoint}.{method}.{binning_suffix}.CnC_{nsigma_tag}.AsymptoticLimits.mH120.root"
    return f"{base_dir}/combine_output/asymptotic/higgsCombine.{masspoint}.{method}.{binning_suffix}.AsymptoticLimits.mH120.root"


def getHybridNewLimitDir(masspoint, method, era, binning_suffix="extended", channel="Combined"):
    """Return the expected HybridNew partial-extract directory for one mass point."""
    return f"templates/{era}/{channel}/{masspoint}/{method}/{binning_suffix}/combine_output/hybridnew/partial_extract"


def masspoint_sort_key(masspoint):
    """Sort by (MHc, MA), with a stable fallback for unexpected names."""
    match = re.fullmatch(r"MHc(\d+)_MA(\d+)", masspoint)
    if match:
        return (int(match.group(1)), int(match.group(2)), masspoint)
    return (10**9, 10**9, masspoint)


def masspoint_from_template_path(path, era, channel):
    """Extract the mass-point directory from templates/{era}/{channel}/..."""
    prefix = os.path.join("templates", era, channel) + os.sep
    if not path.startswith(prefix):
        return None
    rest = path[len(prefix):]
    return rest.split(os.sep, 1)[0]


def discoverAvailableMasspoints(method, era, binning_suffix, limit_type, cnc=False, nsigma_tag="3sigma", channel="Combined"):
    """Discover mass points with existing output for the requested mode."""
    masspoints = set()
    if limit_type == "Asymptotic":
        asymptotic_dir = f"asymptotic_cnc_{nsigma_tag}" if cnc else "asymptotic"
        pattern = os.path.join(
            "templates", era, channel, "*", method, binning_suffix,
            "combine_output", asymptotic_dir,
            "higgsCombine*.AsymptoticLimits.mH120.root",
        )
        for path in glob.glob(pattern):
            masspoint = masspoint_from_template_path(path, era, channel)
            if not masspoint:
                continue
            expected_path = getAsymptoticLimitPath(
                masspoint, method, era, binning_suffix,
                cnc=cnc, nsigma_tag=nsigma_tag, channel=channel,
            )
            if os.path.isfile(expected_path):
                masspoints.add(masspoint)
    elif limit_type == "HybridNew":
        pattern = os.path.join(
            "templates", era, channel, "*", method, binning_suffix,
            "combine_output", "hybridnew", "partial_extract",
        )
        for path in glob.glob(pattern):
            if not os.path.isdir(path):
                continue
            masspoint = masspoint_from_template_path(path, era, channel)
            if masspoint:
                masspoints.add(masspoint)
    else:
        raise ValueError(f"Unknown limit_type: {limit_type}")
    return sorted(masspoints, key=masspoint_sort_key)


def parseHybridNewLimit(masspoint, method, era, binning_suffix="extended", channel="Combined", mode="BR"):
    """Parse HybridNew limits from partial_extract ROOT files."""
    partial_dir = f"templates/{era}/{channel}/{masspoint}/{method}/{binning_suffix}/combine_output/hybridnew/partial_extract"

    if not os.path.isdir(partial_dir):
        raise FileNotFoundError(f"partial_extract directory not found: {partial_dir}")

    # Map quantile tag → limit key
    quantile_map = {
        "quant0.025": "exp-2",
        "quant0.160": "exp-1",
        "quant0.500": "exp0",
        "quant0.840": "exp+1",
        "quant0.975": "exp+2",
    }

    branching_ratios = {}

    # Expected quantiles
    for quant_tag, limit_key in quantile_map.items():
        # e.g. higgsCombine.partial.exp0.025.HybridNew.mH120.quant0.025.root
        q_str = quant_tag.replace("quant", "exp")
        root_file = os.path.join(partial_dir, f"higgsCombine.partial.{q_str}.HybridNew.mH120.{quant_tag}.root")
        f = ROOT.TFile.Open(root_file)
        if not f:
            raise FileNotFoundError(f"HybridNew limit file not found: {root_file}")
        if f.IsZombie():
            f.Close()
            raise RuntimeError(f"ROOT file is zombie/corrupt: {root_file}")
        tree = f.Get("limit")
        if not tree or tree.GetEntries() == 0:
            f.Close()
            raise RuntimeError(f"TTree 'limit' empty or missing in {root_file}")
        tree.GetEntry(0)
        branching_ratios[limit_key] = _convert(tree.limit, mode)
        f.Close()

    # Observed
    obs_file = os.path.join(partial_dir, "higgsCombine.partial.obs.HybridNew.mH120.root")
    f = ROOT.TFile.Open(obs_file)
    if not f:
        raise FileNotFoundError(f"HybridNew observed file not found: {obs_file}")
    if f.IsZombie():
        f.Close()
        raise RuntimeError(f"ROOT file is zombie/corrupt: {obs_file}")
    tree = f.Get("limit")
    if not tree or tree.GetEntries() == 0:
        f.Close()
        raise RuntimeError(f"TTree 'limit' empty or missing in {obs_file}")
    tree.GetEntry(0)
    branching_ratios["obs"] = _convert(tree.limit, mode)
    f.Close()

    return branching_ratios


if __name__ == "__main__":
    logger.info(f"Collecting limits for era={args.era}, method={args.method}")

    binning_suffix = "extended_unblind" if args.unblind else "extended"
    nsigma_tag = f"{args.nsigma:g}sigma"

    if args.available_only:
        MASSPOINTs = discoverAvailableMasspoints(
            args.method, args.era, binning_suffix, args.limit_type,
            cnc=args.cnc, nsigma_tag=nsigma_tag, channel=args.channel,
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
            if args.limit_type == "Asymptotic":
                limits[masspoint] = parseAsymptoticLimit(masspoint, args.method, args.era, binning_suffix, cnc=args.cnc, nsigma_tag=nsigma_tag, channel=args.channel, mode=args.mode)
            elif args.limit_type == "HybridNew":
                limits[masspoint] = parseHybridNewLimit(masspoint, args.method, args.era, binning_suffix, channel=args.channel, mode=args.mode)
            else:
                raise ValueError(f"Unknown limit_type: {args.limit_type}")
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
    cnc_suffix = f".CnC_{nsigma_tag}" if args.cnc else ""
    suffix = ".unblind" if args.unblind else ""
    ch_suffix = "" if args.channel == "Combined" else f".{args.channel}"
    if args.output is not None:
        outpath = args.output
    else:
        outpath = f"results/json/{args.mode}/{args.era}/limits.{args.era}{ch_suffix}.{args.limit_type}.{args.method}{cnc_suffix}{suffix}.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    with open(outpath, "w") as f:
        json.dump(limits, f, indent=4)

    logger.info(f"Saved limits to {outpath}")
