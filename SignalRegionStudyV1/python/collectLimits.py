#!/usr/bin/env python3
"""
Collect limits from Combine output files and write to JSON.

Usage:
    collectLimits.py --era 2018 --channel SR1E2Mu --method Baseline --binning uniform --limit-type Asymptotic
"""
import os
import sys
import json
import logging
import argparse
import ROOT

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Configuration constants
REFERENCE_XSEC = 5.0  # fb (signal cross-section used in templates)
TTBAR_XSEC_13TEV = 833.9e3  # fb
TTBAR_XSEC_13p6TEV = 923.6e3  # fb
BR_TTBAR_TO_LEPTON = 2 * 0.5456  # Branching ratio for ttbar -> lepton+jets

# Mass point definitions
BASELINE_MASSPOINTS = [
    "MHc70_MA15", "MHc70_MA40", "MHc70_MA65",
    "MHc100_MA15", "MHc100_MA60", "MHc100_MA95",
    "MHc130_MA15", "MHc130_MA55", "MHc130_MA90", "MHc130_MA125",
    "MHc160_MA15", "MHc160_MA85", "MHc160_MA120", "MHc160_MA155"
]

PARTICLENET_MASSPOINTS = [
    "MHc100_MA95", "MHc130_MA90", "MHc160_MA85"
]


def get_masspoints(method):
    """Return list of mass points for given method."""
    if method == "ParticleNet":
        return PARTICLENET_MASSPOINTS
    return BASELINE_MASSPOINTS


def get_ttbar_xsec(era):
    """Get ttbar cross-section for era."""
    run2_eras = ["2016preVFP", "2016postVFP", "2017", "2018", "FullRun2"]
    if era in run2_eras:
        return TTBAR_XSEC_13TEV
    return TTBAR_XSEC_13p6TEV


def parse_asymptotic_limit(root_file, era):
    """Extract limits from AsymptoticLimits output."""
    xsec_ref = get_ttbar_xsec(era)

    f = ROOT.TFile.Open(root_file)
    if not f or f.IsZombie():
        raise FileNotFoundError(f"Cannot open {root_file}")

    limit_tree = f.Get("limit")
    if not limit_tree:
        f.Close()
        raise ValueError(f"No 'limit' tree in {root_file}")

    limits = {}
    quantile_map = {
        0: "exp-2",
        1: "exp-1",
        2: "exp0",
        3: "exp+1",
        4: "exp+2",
        5: "obs"
    }

    for idx in range(limit_tree.GetEntries()):
        limit_tree.GetEntry(idx)
        if idx in quantile_map:
            r = limit_tree.limit
            # Convert r to branching ratio
            br = r * REFERENCE_XSEC / xsec_ref / BR_TTBAR_TO_LEPTON
            limits[quantile_map[idx]] = br

    f.Close()
    return limits


def parse_hybridnew_limit(output_dir, masspoint, method, binning, era):
    """Extract limits from HybridNew output files."""
    xsec_ref = get_ttbar_xsec(era)
    limits = {}

    quantile_files = {
        "exp-2": f"higgsCombine.{masspoint}.{method}.{binning}.exp0.025.HybridNew.mH120.root",
        "exp-1": f"higgsCombine.{masspoint}.{method}.{binning}.exp0.160.HybridNew.mH120.root",
        "exp0": f"higgsCombine.{masspoint}.{method}.{binning}.exp0.500.HybridNew.mH120.root",
        "exp+1": f"higgsCombine.{masspoint}.{method}.{binning}.exp0.840.HybridNew.mH120.root",
        "exp+2": f"higgsCombine.{masspoint}.{method}.{binning}.exp0.975.HybridNew.mH120.root",
        "obs": f"higgsCombine.{masspoint}.{method}.{binning}.obs.HybridNew.mH120.root"
    }

    for key, filename in quantile_files.items():
        filepath = os.path.join(output_dir, filename)
        if not os.path.exists(filepath):
            logging.warning(f"Missing file: {filepath}")
            continue

        f = ROOT.TFile.Open(filepath)
        if not f or f.IsZombie():
            continue

        limit_tree = f.Get("limit")
        if limit_tree and limit_tree.GetEntries() > 0:
            limit_tree.GetEntry(0)
            r = limit_tree.limit
            br = r * REFERENCE_XSEC / xsec_ref / BR_TTBAR_TO_LEPTON
            limits[key] = br

        f.Close()

    return limits


def main():
    parser = argparse.ArgumentParser(description="Collect limits from Combine output")
    parser.add_argument("--era", required=True, help="Data-taking period")
    parser.add_argument("--channel", required=True, help="Analysis channel")
    parser.add_argument("--method", required=True, help="Template method")
    parser.add_argument("--binning", default="uniform", help="Binning scheme")
    parser.add_argument("--limit-type", default="Asymptotic",
                        choices=["Asymptotic", "HybridNew"],
                        help="Limit calculation method")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get WORKDIR
    WORKDIR = os.getenv("WORKDIR")
    if not WORKDIR:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'")

    # Setup ROOT
    ROOT.gROOT.SetBatch(True)

    # Get mass points
    masspoints = get_masspoints(args.method)
    logging.info(f"Collecting limits for {len(masspoints)} mass points")

    # Collect limits
    limits = {}
    for masspoint in masspoints:
        template_dir = f"{WORKDIR}/SignalRegionStudyV1/templates/{args.era}/{args.channel}/{masspoint}/{args.method}/{args.binning}"

        if args.limit_type == "Asymptotic":
            output_dir = f"{template_dir}/combine_output/asymptotic"
            # Try different file name patterns
            patterns = [
                f"higgsCombine.{masspoint}.{args.method}.{args.binning}.AsymptoticLimits.mH120.root",
                f"higgsCombine.{masspoint}.{args.method}.AsymptoticLimits.mH120.root",
                "higgsCombineTest.AsymptoticLimits.mH120.root"
            ]

            root_file = None
            for pattern in patterns:
                candidate = os.path.join(output_dir, pattern)
                if os.path.exists(candidate):
                    root_file = candidate
                    break

            if not root_file:
                logging.warning(f"No output found for {masspoint}")
                continue

            try:
                limits[masspoint] = parse_asymptotic_limit(root_file, args.era)
                logging.debug(f"Collected {masspoint}: {limits[masspoint]}")
            except Exception as e:
                logging.warning(f"Failed to parse {masspoint}: {e}")
                continue

        elif args.limit_type == "HybridNew":
            output_dir = f"{template_dir}/combine_output/hybridnew"
            if not os.path.exists(output_dir):
                logging.warning(f"No output found for {masspoint}")
                continue

            try:
                result = parse_hybridnew_limit(output_dir, masspoint, args.method, args.binning, args.era)
                if result:
                    limits[masspoint] = result
                    logging.debug(f"Collected {masspoint}: {limits[masspoint]}")
            except Exception as e:
                logging.warning(f"Failed to parse {masspoint}: {e}")
                continue

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"{WORKDIR}/SignalRegionStudyV1/results/json/limits.{args.era}.{args.channel}.{args.limit_type}.{args.method}.{args.binning}.json"

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save limits
    with open(output_path, 'w') as f:
        json.dump(limits, f, indent=4)

    logging.info(f"Collected {len(limits)} mass points")
    logging.info(f"Saved to: {output_path}")

    # Print summary
    if limits:
        print("\nLimit Summary (Branching Ratio):")
        print("-" * 80)
        print(f"{'Masspoint':<20} {'exp-2':<12} {'exp-1':<12} {'exp0':<12} {'exp+1':<12} {'exp+2':<12}")
        print("-" * 80)
        for mp in sorted(limits.keys()):
            lim = limits[mp]
            print(f"{mp:<20} {lim.get('exp-2', 0):.2e} {lim.get('exp-1', 0):.2e} {lim.get('exp0', 0):.2e} {lim.get('exp+1', 0):.2e} {lim.get('exp+2', 0):.2e}")


if __name__ == "__main__":
    main()
