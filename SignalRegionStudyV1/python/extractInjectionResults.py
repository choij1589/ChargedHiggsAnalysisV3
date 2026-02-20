#!/usr/bin/env python3
"""
extractInjectionResults.py - Extract signal injection fit results from FitDiagnostics

Extracts (r, rLoErr, rHiErr, fit_status) from tree_fit_sb in FitDiagnostics output.
Filters on fit_status == 0 and warns if >10% of fits fail.

Usage:
    python extractInjectionResults.py --era Run2 --channel Combined --masspoint MHc130_MA90 \
                                      --method Baseline --binning extended
"""
import argparse
import os
import json
import glob
import ROOT

ROOT.gROOT.SetBatch(True)

# Path inference
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


def main():
    parser = argparse.ArgumentParser(description="Extract signal injection fit results")
    parser.add_argument("--era", required=True, help="Data-taking era")
    parser.add_argument("--channel", required=True, help="Analysis channel")
    parser.add_argument("--masspoint", required=True, help="Signal mass point")
    parser.add_argument("--method", default="Baseline", help="Template method")
    parser.add_argument("--binning", default="uniform", help="Binning scheme")
    parser.add_argument("--output", help="Output JSON path (auto-inferred if not provided)")
    args = parser.parse_args()

    injection_dir = get_injection_dir(
        args.era, args.channel, args.masspoint, args.method, args.binning
    )

    if not os.path.exists(injection_dir):
        raise FileNotFoundError(f"Injection directory not found: {injection_dir}")

    # Find all r* directories
    r_dirs = glob.glob(os.path.join(injection_dir, "r*"))

    all_results = {}
    summary = {}

    print(f"Extracting results from {injection_dir}")
    print("-" * 60)

    for r_dir in sorted(r_dirs):
        dirname = os.path.basename(r_dir)
        if not dirname.startswith("r"):
            continue

        try:
            r_inj = float(dirname[1:])
        except ValueError:
            continue

        # Find FitDiagnostics output
        fitdiag_files = glob.glob(os.path.join(r_dir, "fitDiagnostics.recovery_r*.root"))
        if not fitdiag_files:
            print(f"WARNING: No FitDiagnostics file for r={r_inj}")
            continue

        results = extract_from_fitdiag(fitdiag_files[0])

        # Filter on fit_status == 0
        n_total = len(results)
        good_results = [r for r in results if r['fit_status'] == 0]
        n_good = len(good_results)
        fail_rate = (n_total - n_good) / n_total if n_total > 0 else 0

        if fail_rate > 0.1:
            print(f"WARNING: r={r_inj}: {fail_rate*100:.1f}% of fits failed (>10% threshold)")

        all_results[str(r_inj)] = good_results
        summary[str(r_inj)] = {
            'n_total': n_total,
            'n_good': n_good,
            'fail_rate': fail_rate,
            'r_inj': r_inj
        }

        print(f"  r={r_inj}: {n_good}/{n_total} good fits ({fail_rate*100:.1f}% failed)")

    if not all_results:
        raise RuntimeError("No valid results found")

    # Save results
    output_path = args.output or os.path.join(injection_dir, "injection_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump({
            'results': all_results,
            'summary': summary
        }, f, indent=2)

    print("-" * 60)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
