#!/usr/bin/env python3
"""
Scan different core bin counts (15, 12, 9) to find statistically safe binning.

For each masspoint/era/channel, creates histograms with different numbers of
uniform core bins in [-5σ, +5σ] plus 4 tail bins at ±7σ/±10σ, then reports:
  - Number of empty total-background bins
  - Number of bins with >100% stat error (per process)
  - Max relative stat error

Usage:
    python3 python/scanBinning.py [--masspoint MHc100_MA15] [--era 2023] [--channel SR1E2Mu]
    python3 python/scanBinning.py --scan-all   # scan all problematic cases
"""
import os
import sys
import json
import argparse
import logging
from math import sqrt

import ROOT
import numpy as np

ROOT.gROOT.SetBatch(True)

# Suppress ROOT info messages
ROOT.gErrorIgnoreLevel = ROOT.kWarning


def parse_args():
    parser = argparse.ArgumentParser(description="Scan core bin counts for statistical safety")
    parser.add_argument("--era", type=str, help="Era to scan")
    parser.add_argument("--channel", type=str, help="Channel to scan")
    parser.add_argument("--masspoint", type=str, help="Mass point to scan")
    parser.add_argument("--scan-all", action="store_true", help="Scan all problematic cases")
    parser.add_argument("--core-bins", type=str, default="15,12,9,7,5",
                        help="Comma-separated list of core bin counts to test (default: 15,12,9,7,5)")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def get_extended_bins(mA, width, sigma, n_core_bins):
    """Generate extended bin edges with variable number of core bins."""
    voigt_width = sqrt(width**2 + sigma**2)

    # Uniform core bins in [-5σ, +5σ]
    uniform_sigma_fractions = np.linspace(-5, 5, n_core_bins + 1)

    # Tail bins at [-10, -7] and [+7, +10]
    extra_low = np.array([-10, -7])
    extra_high = np.array([7, 10])

    sigma_fractions = np.concatenate([extra_low, uniform_sigma_fractions, extra_high])
    bin_edges = mA + sigma_fractions * voigt_width

    return bin_edges, voigt_width


def fill_histogram(file_path, tree_name, bin_edges, mass_min, mass_max, hist_name, use_weight=True):
    """Fill histogram from preprocessed tree."""
    if not os.path.exists(file_path):
        return None

    rfile = ROOT.TFile.Open(file_path, "READ")
    tree = rfile.Get(tree_name)
    if not tree:
        rfile.Close()
        return None

    nbins = len(bin_edges) - 1
    bin_edges_arr = np.array(bin_edges, dtype='d')
    h = ROOT.TH1D(hist_name, "", nbins, bin_edges_arr)
    h.SetDirectory(0)

    for entry in range(tree.GetEntries()):
        tree.GetEntry(entry)
        mass = tree.mass
        if mass < mass_min or mass > mass_max:
            continue
        if use_weight:
            h.Fill(mass, tree.weight)
        else:
            h.Fill(mass)

    rfile.Close()
    return h


def analyze_binning(basedir, masspoint, mA, width, sigma, n_core_bins, sample_groups, era, channel):
    """Analyze a single binning configuration. Returns dict of results."""
    bin_edges, voigt_width = get_extended_bins(mA, width, sigma, n_core_bins)
    nbins = len(bin_edges) - 1
    mass_min = mA - 10 * voigt_width
    mass_max = mA + 10 * voigt_width

    # Fill signal histogram
    sig_path = f"{basedir}/{masspoint}.root"
    h_sig = fill_histogram(sig_path, "Central", bin_edges, mass_min, mass_max, "signal")

    if h_sig is None:
        return None

    # Fill background histograms
    bkg_hists = {}
    reserved = {"data", "nonprompt", "others"}
    bkg_categories = [k for k in sample_groups.keys() if k not in reserved]

    # nonprompt
    h_np = fill_histogram(f"{basedir}/nonprompt.root", "Central", bin_edges, mass_min, mass_max, "nonprompt")
    if h_np is not None:
        bkg_hists["nonprompt"] = h_np

    # each prompt background
    for cat in bkg_categories:
        process = "conversion" if cat == "conv" else cat
        h = fill_histogram(f"{basedir}/{process}.root", "Central", bin_edges, mass_min, mass_max, process)
        if h is not None:
            bkg_hists[process] = h

    # others
    h_others = fill_histogram(f"{basedir}/others.root", "Central", bin_edges, mass_min, mass_max, "others")
    if h_others is not None:
        bkg_hists["others"] = h_others

    # Compute total background
    h_total = None
    for name, h in bkg_hists.items():
        if h_total is None:
            h_total = h.Clone("total_bkg")
            h_total.SetDirectory(0)
        else:
            h_total.Add(h)

    if h_total is None:
        return None

    # Analyze per-bin statistics
    n_empty_bkg = 0
    n_bkg_over100 = 0
    n_sig_over100 = 0
    max_bkg_relerr = 0
    max_sig_relerr = 0
    problem_bins = []

    for i in range(1, nbins + 1):
        bc = h_total.GetBinContent(i)
        be = h_total.GetBinError(i)
        sc = h_sig.GetBinContent(i)
        se = h_sig.GetBinError(i)

        sigma_lo = (bin_edges[i-1] - mA) / voigt_width
        sigma_hi = (bin_edges[i] - mA) / voigt_width
        is_core = abs(sigma_lo) <= 5 and abs(sigma_hi) <= 5

        bkg_relerr = be / bc if bc > 0 else float('inf')
        sig_relerr = se / sc if sc > 0 else float('inf')

        flags = []
        if bc == 0:
            n_empty_bkg += 1
            flags.append("EMPTY_BKG")
        elif bkg_relerr > 1.0:
            n_bkg_over100 += 1
            flags.append(f"BKG>{bkg_relerr*100:.0f}%")

        if sc > 0 and sig_relerr > 1.0:
            n_sig_over100 += 1
            flags.append(f"SIG>{sig_relerr*100:.0f}%")

        if bc > 0:
            max_bkg_relerr = max(max_bkg_relerr, bkg_relerr)
        if sc > 0:
            max_sig_relerr = max(max_sig_relerr, sig_relerr)

        if flags:
            region = "CORE" if is_core else "TAIL"
            problem_bins.append({
                "bin": i, "sigma_lo": sigma_lo, "sigma_hi": sigma_hi,
                "region": region, "sig": sc, "bkg": bc, "bkg_err": be,
                "flags": flags
            })

    # Per-process empty bin count
    n_proc_empty = {}
    for name, h in bkg_hists.items():
        count = sum(1 for i in range(1, nbins+1) if h.GetBinContent(i) == 0)
        if count > 0:
            n_proc_empty[name] = count

    return {
        "n_core_bins": n_core_bins,
        "n_total_bins": nbins,
        "n_empty_bkg": n_empty_bkg,
        "n_bkg_over100": n_bkg_over100,
        "n_sig_over100": n_sig_over100,
        "max_bkg_relerr": max_bkg_relerr * 100,
        "max_sig_relerr": max_sig_relerr * 100,
        "problem_bins": problem_bins,
        "per_process_empty": n_proc_empty,
        "signal_integral": h_sig.Integral(),
        "bkg_integral": h_total.Integral(),
    }


def print_results(results, label):
    """Print results for one configuration."""
    r = results
    status = "PASS" if r["n_empty_bkg"] == 0 and r["n_bkg_over100"] == 0 else "FAIL"

    print(f"  Core={r['n_core_bins']:2d} ({r['n_total_bins']:2d} total) | "
          f"EmptyBkg={r['n_empty_bkg']} Bkg>100%={r['n_bkg_over100']} "
          f"Sig>100%={r['n_sig_over100']} | "
          f"MaxBkgErr={r['max_bkg_relerr']:>8.1f}% MaxSigErr={r['max_sig_relerr']:>6.1f}% | "
          f"S={r['signal_integral']:.4f} B={r['bkg_integral']:.4f} | {status}")

    if r["problem_bins"]:
        for pb in r["problem_bins"]:
            flags_str = ", ".join(pb["flags"])
            print(f"         bin {pb['bin']:2d} [{pb['sigma_lo']:+5.1f}σ,{pb['sigma_hi']:+5.1f}σ] "
                  f"{pb['region']:4s}: sig={pb['sig']:.4f} bkg={pb['bkg']:.4f}±{pb['bkg_err']:.4f} "
                  f"| {flags_str}")


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(levelname)s - %(message)s')

    workdir = os.getenv("WORKDIR")
    if not workdir:
        raise EnvironmentError("WORKDIR not set. Please run 'source setup.sh'")

    core_bin_counts = [int(x) for x in args.core_bins.split(",")]

    # Load sample groups
    with open(f"{workdir}/SignalRegionStudyV3/configs/samplegroups.json") as f:
        all_samplegroups = json.load(f)

    if args.scan_all:
        # Scan the most problematic cases identified earlier
        scan_cases = []
        # Low-MA points across eras
        low_ma_masspoints = ["MHc70_MA15", "MHc100_MA15", "MHc70_MA18", "MHc85_MA15",
                             "MHc115_MA15", "MHc130_MA15", "MHc145_MA15", "MHc160_MA15"]
        eras = ["2016preVFP", "2016postVFP", "2017", "2018",
                "2022", "2022EE", "2023", "2023BPix"]
        channels = ["SR1E2Mu", "SR3Mu"]

        for era in eras:
            for channel in channels:
                for mp in low_ma_masspoints:
                    basedir = f"{workdir}/SignalRegionStudyV3/samples/{era}/{channel}/{mp}"
                    sig_file = f"{basedir}/{mp}.root"
                    if os.path.exists(sig_file):
                        scan_cases.append((era, channel, mp))

        # Also add a few higher-MA cases for comparison
        higher_ma = ["MHc130_MA90", "MHc160_MA120", "MHc100_MA60"]
        for era in ["2022", "2023BPix"]:
            for channel in channels:
                for mp in higher_ma:
                    basedir = f"{workdir}/SignalRegionStudyV3/samples/{era}/{channel}/{mp}"
                    sig_file = f"{basedir}/{mp}.root"
                    if os.path.exists(sig_file):
                        scan_cases.append((era, channel, mp))
    else:
        if not all([args.era, args.channel, args.masspoint]):
            print("Error: provide --era, --channel, --masspoint or use --scan-all")
            sys.exit(1)
        scan_cases = [(args.era, args.channel, args.masspoint)]

    # Summary tracking
    summary = {n: {"pass": 0, "fail": 0} for n in core_bin_counts}

    for era, channel, masspoint in scan_cases:
        basedir = f"{workdir}/SignalRegionStudyV3/samples/{era}/{channel}/{masspoint}"

        # Load fit parameters
        # Try to load from existing templates first
        fit_path = f"{workdir}/SignalRegionStudyV3/templates/{era}/SR1E2Mu/{masspoint}/Baseline/extended/signal_fit.json"
        if not os.path.exists(fit_path):
            # For SR3Mu, also check SR1E2Mu
            fit_path = f"{workdir}/SignalRegionStudyV3/templates/{era}/{channel}/{masspoint}/Baseline/extended/signal_fit.json"
        if not os.path.exists(fit_path):
            logging.debug(f"No fit result for {era}/{channel}/{masspoint}, skipping")
            continue

        with open(fit_path) as ff:
            fit = json.load(ff)

        mA = fit["mass"]
        width_val = fit["width"]
        sigma_val = fit["sigma"]

        if era not in all_samplegroups or channel not in all_samplegroups[era]:
            continue

        sample_groups = all_samplegroups[era][channel]

        mA_label = masspoint.split("_")[1]
        print(f"\n{'='*80}")
        print(f"{era}/{channel}/{masspoint}  (mA={mA:.1f}, voigt={sqrt(width_val**2+sigma_val**2):.3f} GeV)")
        print(f"{'='*80}")

        for n_core in core_bin_counts:
            result = analyze_binning(basedir, masspoint, mA, width_val, sigma_val,
                                     n_core, sample_groups, era, channel)
            if result is None:
                print(f"  Core={n_core:2d}: could not build histograms")
                continue

            print_results(result, f"{era}/{channel}/{masspoint}")

            is_pass = result["n_empty_bkg"] == 0 and result["n_bkg_over100"] == 0
            summary[n_core]["pass" if is_pass else "fail"] += 1

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    for n_core in core_bin_counts:
        total = summary[n_core]["pass"] + summary[n_core]["fail"]
        if total == 0:
            continue
        pass_rate = summary[n_core]["pass"] / total * 100
        print(f"  Core={n_core:2d} bins: {summary[n_core]['pass']:3d} PASS / {summary[n_core]['fail']:3d} FAIL "
              f"({pass_rate:.1f}% pass rate)")


if __name__ == "__main__":
    main()
