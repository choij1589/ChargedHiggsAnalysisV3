#!/usr/bin/env python3
"""
Summarize discrimination variable correctness for signal pair selection.
Extracts correctness metrics from all mass points and saves to JSON.
"""

import os
import sys
import argparse
import json
import ROOT
ROOT.gROOT.SetBatch(True)

WORKDIR = os.environ['WORKDIR']

# All discrimination variables (27 total)
DISCRIMINATION_VARIABLES = [
    "acoplanarity_correct",
    "scalarPtSum_correct",
    "ptAsymmetry_correct",
    "gammaFactor_smaller_correct",
    "gammaFactor_larger_correct",
    "gammaAcop_smaller_correct",
    "gammaAcop_larger_correct",
    "deltaR_pair_mu3rd_larger_correct",
    "deltaR_pair_mu3rd_smaller_correct",
    "deltaPhi_pair_mu3rd_larger_correct",
    "ptRatio_mu3rd_smaller_correct",
    "deltaR_nearestBjet_smaller_correct",
    "deltaR_nearestBjet_larger_correct",
    "deltaR_leadingNonBjet_smaller_correct",
    "deltaR_leadingNonBjet_larger_correct",
    "deltaPhi_pair_MET_smaller_correct",
    "deltaPhi_pair_MET_larger_correct",
    "MT_pair_MET_smaller_correct",
    "MT_pair_MET_larger_correct",
    "deltaPhi_muSS_MET_larger_correct",
    "deltaPhi_muSS_MET_smaller_correct",
    "MT_muSS_MET_larger_correct",
    "MT_muSS_MET_smaller_correct",
    "MT_asymmetry_smaller_correct",
    "MT_asymmetry_larger_correct",
    "mass_smaller_correct",
    "mass_larger_correct",
]


def get_available_samples(era):
    """Get list of available signal sample mass points for given era"""
    sample_dir = f"{WORKDIR}/SKNanoOutput/SignalKinematics/Run3Mu/{era}"

    if not os.path.exists(sample_dir):
        raise RuntimeError(f"Sample directory not found: {sample_dir}")

    samples = []
    for filename in os.listdir(sample_dir):
        if filename.startswith("TTToHcToWAToMuMu-") and filename.endswith(".root"):
            # Extract mass point (e.g., "MHc70_MA15")
            mass_point = filename.replace("TTToHcToWAToMuMu-", "").replace(".root", "")
            samples.append(mass_point)

    return sorted(samples)


def extract_correctness(file_path, channel, variable):
    """
    Extract correctness metrics from a discrimination histogram.

    Args:
        file_path: Path to ROOT file
        channel: Analysis channel (e.g., SR3Mu)
        variable: Discrimination variable name

    Returns:
        Dictionary with n_incorrect, n_correct, n_total, correctness
    """
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open file: {file_path}")

    hist_path = f"{channel}/Discrimination/{variable}"
    h = f.Get(hist_path)

    if not h:
        f.Close()
        raise RuntimeError(f"Histogram not found: {hist_path}")

    # Binary histogram: bin 1 = incorrect, bin 2 = correct
    n_incorrect = h.GetBinContent(1)
    n_correct = h.GetBinContent(2)
    n_total = n_incorrect + n_correct

    # Calculate correctness percentage
    if n_total > 0:
        correctness = (n_correct / n_total) * 100.0
    else:
        correctness = 0.0

    f.Close()

    return {
        "n_incorrect": round(n_incorrect, 2),
        "n_correct": round(n_correct, 2),
        "n_total": round(n_total, 2),
        "correctness": round(correctness, 2)
    }


def summarize_discrimination(era, channel):
    """
    Summarize discrimination correctness for all mass points.

    Args:
        era: Data-taking era
        channel: Analysis channel

    Returns:
        Dictionary with structure: {mass_point: {variable: metrics}}
    """
    mass_points = get_available_samples(era)
    print(f"Found {len(mass_points)} mass points for era {era}")

    summary = {}

    for mass_point in mass_points:
        print(f"Processing {mass_point}...")

        file_path = f"{WORKDIR}/SKNanoOutput/SignalKinematics/Run3Mu/{era}/TTToHcToWAToMuMu-{mass_point}.root"

        if not os.path.exists(file_path):
            print(f"  WARNING: File not found: {file_path}")
            continue

        summary[mass_point] = {}

        for variable in DISCRIMINATION_VARIABLES:
            try:
                metrics = extract_correctness(file_path, channel, variable)
                summary[mass_point][variable] = metrics
            except RuntimeError as e:
                print(f"  WARNING: {e}")
                summary[mass_point][variable] = {
                    "n_incorrect": 0.0,
                    "n_correct": 0.0,
                    "n_total": 0.0,
                    "correctness": 0.0
                }

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Summarize discrimination variable correctness"
    )
    parser.add_argument("--era", default="2017", type=str, help="Data era")
    parser.add_argument("--channel", default="SR3Mu", type=str, help="Analysis channel")
    parser.add_argument("--output", default=None, type=str,
                       help="Output JSON file path (default: results/{era}/{channel}/discrimination_summary.json)")

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        output_path = f"{WORKDIR}/SignalKinematics/results/{args.era}/{args.channel}/discrimination_summary.json"
    else:
        output_path = args.output

    print("=" * 60)
    print("Discrimination Variable Correctness Summary")
    print("=" * 60)
    print(f"Era: {args.era}")
    print(f"Channel: {args.channel}")
    print(f"Output: {output_path}")
    print("=" * 60)

    # Generate summary
    summary = summarize_discrimination(args.era, args.channel)

    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save to JSON with pretty printing
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print("=" * 60)
    print(f"Summary saved to: {output_path}")
    print(f"Total mass points: {len(summary)}")
    print(f"Variables per mass point: {len(DISCRIMINATION_VARIABLES)}")
    print("=" * 60)

    # Print some statistics
    print("\nTop 5 variables by average correctness:")
    print("-" * 60)

    # Calculate average correctness for each variable
    avg_correctness = {}
    for variable in DISCRIMINATION_VARIABLES:
        total_correct = 0.0
        count = 0
        for mass_point, data in summary.items():
            if variable in data and data[variable]["n_total"] > 0:
                total_correct += data[variable]["correctness"]
                count += 1
        if count > 0:
            avg_correctness[variable] = total_correct / count
        else:
            avg_correctness[variable] = 0.0

    # Sort by average correctness
    sorted_vars = sorted(avg_correctness.items(), key=lambda x: x[1], reverse=True)

    for i, (var, correctness) in enumerate(sorted_vars[:5], 1):
        print(f"{i}. {var:50s} {correctness:6.2f}%")

    print("\nBottom 5 variables by average correctness:")
    print("-" * 60)
    for i, (var, correctness) in enumerate(sorted_vars[-5:], 1):
        print(f"{i}. {var:50s} {correctness:6.2f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
