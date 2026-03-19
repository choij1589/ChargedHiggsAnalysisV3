#!/usr/bin/env python3
import os
import argparse
import json
import ROOT
import logging

parser = argparse.ArgumentParser()
parser.add_argument("--era", type=str, required=True,
                    help="2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix, Run2, Run3, All")
parser.add_argument("--method", type=str, required=True, help="Baseline, ParticleNet")
parser.add_argument("--limit_type", type=str, default="Asymptotic",
                    help="Asymptotic or HybridNew")
parser.add_argument("--unblind", action="store_true", help="Collect limits from unblind templates")
parser.add_argument("--cnc", action="store_true", help="Collect CnC limits (uses asymptotic_cnc/ directory)")
parser.add_argument("--nsigma", type=float, default=3.0, help="CnC mass window half-width in sigma_voigt (default: 3.0)")
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
TTBAR_XEC_13TEV = 833.9e3  # fb
BR_TTBAR_TO_LEPTON = 2 * 0.5456  # 2 for charge conjugation, 0.5456 for non-hadronic decay of two W bosons


def parseAsymptoticLimit(masspoint, method, era, binning_suffix="extended", cnc=False, nsigma_tag="3sigma"):
    """Parse asymptotic limits from Combine ROOT output file."""
    base_dir = f"templates/{era}/Combined/{masspoint}/{method}/{binning_suffix}"
    if cnc:
        root_file = f"{base_dir}/combine_output/asymptotic_cnc_{nsigma_tag}/higgsCombine.{masspoint}.{method}.{binning_suffix}.CnC_{nsigma_tag}.AsymptoticLimits.mH120.root"
    else:
        root_file = f"{base_dir}/combine_output/asymptotic/higgsCombine.{masspoint}.{method}.{binning_suffix}.AsymptoticLimits.mH120.root"

    if not os.path.exists(root_file):
        raise FileNotFoundError(f"Limit file not found: {root_file}")

    logger.debug(f"Reading limits from: {root_file}")

    f = ROOT.TFile.Open(root_file)
    if not f or f.IsZombie():
        raise RuntimeError(f"Failed to open ROOT file: {root_file}")

    limit = f.Get("limit")
    if not limit:
        f.Close()
        raise RuntimeError(f"TTree 'limit' not found in {root_file}")

    branching_ratios = {}
    try:
        for idx, entry in enumerate(limit):
            # Convert signal strength to branching ratio
            branching_ratios[idx] = entry.limit * REFERENCE_XSEC / TTBAR_XEC_13TEV / BR_TTBAR_TO_LEPTON
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


def parseHybridNewLimit(masspoint, method, era, binning_suffix="extended"):
    """Parse HybridNew limits from partial_extract ROOT files."""
    partial_dir = f"templates/{era}/Combined/{masspoint}/{method}/{binning_suffix}/combine_output/hybridnew/partial_extract"

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
        if not os.path.exists(root_file):
            raise FileNotFoundError(f"HybridNew limit file not found: {root_file}")
        f = ROOT.TFile.Open(root_file)
        if not f or f.IsZombie():
            raise RuntimeError(f"Failed to open ROOT file: {root_file}")
        tree = f.Get("limit")
        if not tree or tree.GetEntries() == 0:
            f.Close()
            raise RuntimeError(f"TTree 'limit' empty or missing in {root_file}")
        tree.GetEntry(0)
        branching_ratios[limit_key] = tree.limit * REFERENCE_XSEC / TTBAR_XEC_13TEV / BR_TTBAR_TO_LEPTON
        f.Close()

    # Observed
    obs_file = os.path.join(partial_dir, "higgsCombine.partial.obs.HybridNew.mH120.root")
    if not os.path.exists(obs_file):
        raise FileNotFoundError(f"HybridNew observed file not found: {obs_file}")
    f = ROOT.TFile.Open(obs_file)
    if not f or f.IsZombie():
        raise RuntimeError(f"Failed to open ROOT file: {obs_file}")
    tree = f.Get("limit")
    if not tree or tree.GetEntries() == 0:
        f.Close()
        raise RuntimeError(f"TTree 'limit' empty or missing in {obs_file}")
    tree.GetEntry(0)
    branching_ratios["obs"] = tree.limit * REFERENCE_XSEC / TTBAR_XEC_13TEV / BR_TTBAR_TO_LEPTON
    f.Close()

    return branching_ratios


if __name__ == "__main__":
    logger.info(f"Collecting limits for era={args.era}, method={args.method}")

    binning_suffix = "extended_unblind" if args.unblind else "extended"
    nsigma_tag = f"{args.nsigma:g}sigma"

    limits = {}
    failed_masspoints = []

    for masspoint in MASSPOINTs:
        try:
            if args.limit_type == "Asymptotic":
                limits[masspoint] = parseAsymptoticLimit(masspoint, args.method, args.era, binning_suffix, cnc=args.cnc, nsigma_tag=nsigma_tag)
            elif args.limit_type == "HybridNew":
                limits[masspoint] = parseHybridNewLimit(masspoint, args.method, args.era, binning_suffix)
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
    outpath = f"results/json/limits.{args.era}.{args.limit_type}.{args.method}{cnc_suffix}{suffix}.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    with open(outpath, "w") as f:
        json.dump(limits, f, indent=4)

    logger.info(f"Saved limits to {outpath}")
