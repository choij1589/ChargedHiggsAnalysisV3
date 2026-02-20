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

# Mass points
if args.method == "Baseline":
    MASSPOINTs = [
        "MHc70_MA18", "MHc70_MA40", "MHc70_MA55", "MHc70_MA65",
        "MHc85_MA15", "MHc85_MA70", "MHc85_MA80",
        "MHc100_MA24", "MHc100_MA60", "MHc100_MA75", "MHc100_MA95",
        "MHc115_MA27", "MHc115_MA87", "MHc115_MA110",
        "MHc130_MA30", "MHc130_MA83", "MHc130_MA90", "MHc130_MA100", "MHc130_MA125",
        "MHc145_MA35", "MHc145_MA92", "MHc145_MA140",
        "MHc160_MA50", "MHc160_MA85", "MHc160_MA98", "MHc160_MA120", "MHc160_MA135", "MHc160_MA155"
    ]
elif args.method == "ParticleNet":
    MASSPOINTs = [
        "MHc100_MA95", "MHc130_MA90", "MHc160_MA85", "MHc115_MA87", "MHc145_MA92", "MHc160_MA98"
    ]
else:
    raise ValueError(f"Invalid method: {args.method}. Must be Baseline or ParticleNet")

# Reference cross-section and normalization constants
# Use 13 TeV ttbar cross-section for ALL eras (signal samples already scaled to 13 TeV reference)
REFERENCE_XSEC = 5.0  # fb
TTBAR_XEC_13TEV = 833.9e3  # fb
BR_TTBAR_TO_LEPTON = 2 * 0.5456  # 2 for charge conjugation, 0.5456 for non-hadronic decay of two W bosons


def parseAsymptoticLimit(masspoint, method, era):
    """Parse asymptotic limits from Combine ROOT output file."""
    base_dir = f"templates/{era}/Combined/{masspoint}/{method}/extended"
    root_file = f"{base_dir}/combine_output/asymptotic/higgsCombine.{masspoint}.{method}.extended.AsymptoticLimits.mH120.root"

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


if __name__ == "__main__":
    logger.info(f"Collecting limits for era={args.era}, method={args.method}")

    limits = {}
    failed_masspoints = []

    for masspoint in MASSPOINTs:
        try:
            limits[masspoint] = parseAsymptoticLimit(masspoint, args.method, args.era)
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
    outpath = f"results/json/limits.{args.era}.Asymptotic.{args.method}.json"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)

    with open(outpath, "w") as f:
        json.dump(limits, f, indent=4)

    logger.info(f"Saved limits to {outpath}")
