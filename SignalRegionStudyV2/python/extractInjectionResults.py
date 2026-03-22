#!/usr/bin/env python3
"""
extractInjectionResults.py - Extract signal injection fit results from MultiDimFit

Extracts (r_fit, r_lo, r_hi) from the limit tree in MultiDimFit --algo singles output.
Each toy produces 3 entries: best-fit r, -1sigma r, +1sigma r.

Supports two modes:
1. --input-dir: Scan flat directory for higgsCombine.recovery_*.MultiDimFit.*.root files
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


def extract_from_multidimfit(mdf_file):
    """
    Extract fit results from MultiDimFit --algo singles output.

    The limit tree has 3 entries per toy:
    - Entry i*3:   best-fit r (r_fit)
    - Entry i*3+1: -1 sigma r value (r_lo = r_fit - err_down)
    - Entry i*3+2: +1 sigma r value (r_hi = r_fit + err_up)

    Returns list of dicts: {r_fit, r_lo, r_hi}
    """
    results = []

    f = ROOT.TFile.Open(mdf_file)
    if not f or f.IsZombie():
        print(f"WARNING: Cannot open {mdf_file}")
        return results

    tree = f.Get("limit")
    if not tree:
        print(f"WARNING: No 'limit' tree in {mdf_file}")
        f.Close()
        return results

    n_entries = tree.GetEntries()
    n_toys = n_entries // 3

    if n_entries % 3 != 0:
        print(f"WARNING: {mdf_file} has {n_entries} entries (not divisible by 3), "
              f"using first {n_toys * 3} entries")

    for i in range(n_toys):
        tree.GetEntry(i * 3)
        r_fit = float(tree.r)

        tree.GetEntry(i * 3 + 1)
        r_lo = float(tree.r)

        tree.GetEntry(i * 3 + 2)
        r_hi = float(tree.r)

        results.append({
            'r_fit': r_fit,
            'r_lo': r_lo,
            'r_hi': r_hi
        })

    f.Close()
    return results


def parse_r_value_from_filename(filename):
    """
    Extract r_label from MultiDimFit filename.

    Expected patterns:
        higgsCombine.recovery_r0_s1.MultiDimFit.mH120.1.root     -> label="r0"
        higgsCombine.recovery_rM1_s2.MultiDimFit.mH120.2.root    -> label="rM1"
        higgsCombine.recovery_rMed_s3.MultiDimFit.mH120.3.root   -> label="rMed"
        higgsCombine.recovery_rP1_s4.MultiDimFit.mH120.4.root    -> label="rP1"
    """
    basename = os.path.basename(filename)
    match = re.match(r'higgsCombine\.recovery_(r\w+)_s\d+\.MultiDimFit\..*\.root', basename)
    if match:
        return match.group(1)
    return None


def read_r_values_mapping(input_dir):
    """
    Read r_values.txt and r_labels.txt to build label->value mapping.
    Falls back to condor/ subdirectory if not found in input_dir.
    """
    for base_dir in [input_dir, os.path.join(input_dir, "condor")]:
        r_values_file = os.path.join(base_dir, "r_values.txt")
        r_labels_file = os.path.join(base_dir, "r_labels.txt")

        if os.path.exists(r_values_file) and os.path.exists(r_labels_file):
            with open(r_values_file) as fv:
                values = [line.strip() for line in fv if line.strip()]
            with open(r_labels_file) as fl:
                labels = [line.strip() for line in fl if line.strip()]

            if len(values) == len(labels):
                return dict(zip(labels, [float(v) for v in values]))

    raise FileNotFoundError(
        f"Cannot find r_values.txt and r_labels.txt in {input_dir} or {input_dir}/condor/. "
        "These files are required to map labels to r values."
    )


def read_fit_config(input_dir):
    """Read fit configuration (rMin, rMax) from fit_config.json."""
    for base_dir in [input_dir, os.path.join(input_dir, "condor")]:
        config_file = os.path.join(base_dir, "fit_config.json")
        if os.path.exists(config_file):
            with open(config_file) as f:
                return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(description="Extract signal injection fit results")
    parser.add_argument("--input-dir", help="Directory containing higgsCombine.recovery_*.MultiDimFit.*.root files")
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

    # Find all MultiDimFit output files
    mdf_files = glob.glob(os.path.join(input_dir, "higgsCombine.recovery_*.MultiDimFit.*.root"))
    mdf_files += glob.glob(os.path.join(input_dir, "condor", "higgsCombine.recovery_*.MultiDimFit.*.root"))

    if not mdf_files:
        raise FileNotFoundError(
            f"No higgsCombine.recovery_*.MultiDimFit.*.root files found in {input_dir}"
        )

    # Read r label -> value mapping
    label_to_rvalue = read_r_values_mapping(input_dir)

    # Read fit config (rMin, rMax) for embedding in output
    fit_config = read_fit_config(input_dir)

    # Group files by r_label
    files_by_label = {}
    for fpath in sorted(mdf_files):
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
            results.extend(extract_from_multidimfit(fpath))

        n_total = len(results)
        all_results[str(r_inj)] = results
        summary[str(r_inj)] = {
            'n_total': n_total,
            'r_inj': r_inj,
            'label': label
        }

        print(f"  r={r_inj:.4f} ({label}): {n_total} toys ({len(files)} files)")

    if not all_results:
        raise RuntimeError("No valid results found")

    # Save results
    output_path = args.output or os.path.join(input_dir, "injection_results.json")
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    output_data = {
        'results': all_results,
        'summary': summary
    }
    if fit_config:
        output_data['fit_config'] = fit_config

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print("-" * 60)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
