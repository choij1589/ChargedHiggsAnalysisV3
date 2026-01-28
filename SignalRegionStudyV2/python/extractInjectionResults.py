#!/usr/bin/env python3
"""
extractInjectionResults.py - Extract signal injection fit results from FitDiagnostics

Extracts (r, rLoErr, rHiErr, fit_status) from tree_fit_sb in FitDiagnostics output.
Filters on fit_status == 0 and warns if >10% of fits fail.

Supports two modes:
1. --input-dir: Scan flat directory for fitDiagnostics.recovery_*.root files (condor output)
2. Legacy: Infer path from era/channel/masspoint/method/binning

Usage:
    python extractInjectionResults.py --input-dir /path/to/injection/ --output results.json
"""
import argparse
import os
import re
import json
import glob
import ROOT

ROOT.gROOT.SetBatch(True)

PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
WORKDIR = os.path.dirname(PYTHON_DIR)


def get_injection_dir(era, channel, masspoint, method, binning):
    """Construct the injection output directory path."""
    return os.path.join(
        WORKDIR, "templates", era, channel, masspoint, method, binning,
        "combine_output", "injection"
    )


def extract_from_fitdiag(fitdiag_file):
    """
    Extract fit results from FitDiagnostics output.

    Returns list of dicts: {r, rLoErr, rHiErr, fit_status}
    """
    results = []

    f = ROOT.TFile.Open(fitdiag_file)
    if not f or f.IsZombie():
        print(f"WARNING: Cannot open {fitdiag_file}")
        return results

    tree = f.Get("tree_fit_sb")
    if not tree:
        print(f"WARNING: No 'tree_fit_sb' in {fitdiag_file}")
        f.Close()
        return results

    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        results.append({
            'r': float(tree.r),
            'rLoErr': float(tree.rLoErr),
            'rHiErr': float(tree.rHiErr),
            'fit_status': int(tree.fit_status)
        })

    f.Close()
    return results


def parse_r_value_from_filename(filename):
    """
    Extract (r_label, r_value) from fitDiagnostics filename.

    Expected patterns:
        fitDiagnostics.recovery_r0_s1.root        -> label="r0"
        fitDiagnostics.recovery_rM1_s2.root       -> label="rM1"
        fitDiagnostics.recovery_rMed_s3.root      -> label="rMed"
        fitDiagnostics.recovery_rP1_s4.root       -> label="rP1"
    """
    basename = os.path.basename(filename)
    match = re.match(r'fitDiagnostics\.recovery_(r\w+)_s\d+\.root', basename)
    if match:
        return match.group(1)
    return None


def read_r_values_mapping(input_dir):
    """
    Read r_values.txt and r_labels.txt to build label->value mapping.
    Falls back to parsing labels directly if files not found.
    """
    r_values_file = os.path.join(input_dir, "r_values.txt")
    r_labels_file = os.path.join(input_dir, "r_labels.txt")

    if os.path.exists(r_values_file) and os.path.exists(r_labels_file):
        with open(r_values_file) as f:
            values = [line.strip() for line in f if line.strip()]
        with open(r_labels_file) as f:
            labels = [line.strip() for line in f if line.strip()]

        if len(values) == len(labels):
            return dict(zip(labels, [float(v) for v in values]))

    # Fallback: check condor/ subdirectory
    condor_r_values = os.path.join(input_dir, "condor", "r_values.txt")
    condor_r_labels = os.path.join(input_dir, "condor", "r_labels.txt")

    if os.path.exists(condor_r_values) and os.path.exists(condor_r_labels):
        with open(condor_r_values) as f:
            values = [line.strip() for line in f if line.strip()]
        with open(condor_r_labels) as f:
            labels = [line.strip() for line in f if line.strip()]

        if len(values) == len(labels):
            return dict(zip(labels, [float(v) for v in values]))

    raise FileNotFoundError(
        f"Cannot find r_values.txt and r_labels.txt in {input_dir} or {input_dir}/condor/. "
        "These files are required to map labels to r values."
    )


def main():
    parser = argparse.ArgumentParser(description="Extract signal injection fit results")
    parser.add_argument("--input-dir", help="Directory containing fitDiagnostics.recovery_*.root files")
    parser.add_argument("--era", help="Data-taking era (for path inference)")
    parser.add_argument("--channel", default="Combined", help="Analysis channel")
    parser.add_argument("--masspoint", help="Signal mass point (for path inference)")
    parser.add_argument("--method", default="Baseline", help="Template method")
    parser.add_argument("--binning", default="extended", help="Binning scheme")
    parser.add_argument("--output", help="Output JSON path")
    args = parser.parse_args()

    # Determine input directory
    if args.input_dir:
        input_dir = args.input_dir
    elif args.era and args.masspoint:
        input_dir = get_injection_dir(
            args.era, args.channel, args.masspoint, args.method, args.binning
        )
    else:
        raise ValueError("Must provide either --input-dir or (--era and --masspoint)")

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Find all fitDiagnostics files
    fitdiag_files = glob.glob(os.path.join(input_dir, "fitDiagnostics.recovery_*.root"))

    # Also check condor/ subdirectory
    fitdiag_files += glob.glob(os.path.join(input_dir, "condor", "fitDiagnostics.recovery_*.root"))

    if not fitdiag_files:
        raise FileNotFoundError(f"No fitDiagnostics.recovery_*.root files found in {input_dir}")

    # Read r label -> value mapping
    label_to_rvalue = read_r_values_mapping(input_dir)

    # Group files by r_label
    files_by_label = {}
    for fpath in sorted(fitdiag_files):
        label = parse_r_value_from_filename(fpath)
        if label is None:
            print(f"WARNING: Cannot parse r label from {fpath}, skipping")
            continue
        files_by_label.setdefault(label, []).append(fpath)

    all_results = {}
    summary = {}

    print(f"Extracting results from {input_dir}")
    print("-" * 60)

    for label in sorted(files_by_label.keys(), key=lambda l: label_to_rvalue.get(l, 0)):
        if label not in label_to_rvalue:
            print(f"WARNING: No r value mapping for label '{label}', skipping")
            continue

        r_inj = label_to_rvalue[label]
        files = files_by_label[label]

        # Extract from all batch files
        results = []
        for fpath in files:
            results.extend(extract_from_fitdiag(fpath))

        # Filter on fit_status == 0
        n_total = len(results)
        good_results = [r for r in results if r['fit_status'] == 0]
        n_good = len(good_results)
        fail_rate = (n_total - n_good) / n_total if n_total > 0 else 0

        if fail_rate > 0.1:
            print(f"WARNING: r={r_inj} ({label}): {fail_rate*100:.1f}% of fits failed (>10% threshold)")

        all_results[str(r_inj)] = good_results
        summary[str(r_inj)] = {
            'n_total': n_total,
            'n_good': n_good,
            'fail_rate': fail_rate,
            'r_inj': r_inj,
            'label': label
        }

        print(f"  r={r_inj:.4f} ({label}): {n_good}/{n_total} good fits "
              f"({fail_rate*100:.1f}% failed, {len(files)} files)")

    if not all_results:
        raise RuntimeError("No valid results found")

    # Save results
    output_path = args.output or os.path.join(input_dir, "injection_results.json")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'results': all_results,
            'summary': summary
        }, f, indent=2)

    print("-" * 60)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
