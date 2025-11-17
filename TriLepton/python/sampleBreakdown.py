#!/usr/bin/env python3
"""
sampleBreakdown.py

Extract event counts and errors (statistical and systematic separately) for all samples.

Histogram selection:
- 1E2Mu SR/ZFake channels: 'pair/mass' by default, or 'pair_onZ/mass' with --onZ flag
- 3Mu SR/ZFake channels: 'pair_lowM/mass' by default, or 'pair_lowM_onZ/mass' with --onZ flag
- Control regions (ZG/WZ): 'ZCand/mass' always (--onZ flag not supported)

Usage:
    python python/sampleBreakdown.py --era Run2 --channel SR1E2Mu
    python python/sampleBreakdown.py --era 2017 --channel SR3Mu --exclude WZSF
    python python/sampleBreakdown.py --era 2016postVFP --channel SR1E2Mu --onZ
    python python/sampleBreakdown.py --era 2016postVFP --channel ZG1E2Mu
"""

import sys
import os
import json
import ctypes
from math import sqrt
import argparse
from ROOT import TFile, gROOT

gROOT.SetBatch(True)

# Add Common/Tools to path
WORKDIR = os.environ.get("WORKDIR", os.getcwd())
sys.path.insert(0, f"{WORKDIR}/Common/Tools")

from plotter import get_era_list, get_CoM_energy
from HistoUtils import load_histogram, sum_histograms, load_era_configs, get_sample_lists, merge_systematics


def load_signal_config():
    """Load signal mass points from configuration file."""
    config_path = f"{WORKDIR}/TriLepton/configs/signals.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Signal configuration not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config.get("signals", [])


def extract_stat_syst_errors(h_central, hSysts=None, rate_unc=0.0, rate_unc_name=None):
    """
    Extract statistical and systematic errors separately.

    Args:
        h_central: Central histogram (contains stat errors from Sumw2)
        hSysts: List of (name, h_up, h_down) systematic variation tuples (optional)
        rate_unc: Flat rate uncertainty to add (e.g., 0.30 for nonprompt)
        rate_unc_name: Name for the rate uncertainty (e.g., "nonprompt_rate")

    Returns:
        dict with events, stat_error, systematics (dict), syst_error, total_error
    """
    if h_central is None:
        return {
            "events": 0.0,
            "stat_error": 0.0,
            "systematics": {},
            "syst_error": 0.0,
            "total_error": 0.0
        }

    # Get integral and statistical error (includes under/overflow bins 0 to N+1)
    error_stat = ctypes.c_double(0.0)
    events = h_central.IntegralAndError(0, h_central.GetNbinsX() + 1, error_stat)
    stat_error = float(error_stat.value)

    # Dictionary to store individual systematic contributions
    systematics = {}
    syst_error_squared = 0.0

    # Add shape systematics using envelope method (bin-by-bin)
    if hSysts:
        # Initialize systematic error dict for each source
        syst_errors = {name: 0.0 for name, _, _ in hSysts}

        for bin in range(h_central.GetNcells()):
            central = h_central.GetBinContent(bin)

            for name, h_up, h_down in hSysts:
                systUp = abs(h_up.GetBinContent(bin) - central)
                systDown = abs(h_down.GetBinContent(bin) - central)
                envelope = max(systUp, systDown)
                # Accumulate bin-by-bin envelope in quadrature
                syst_errors[name] += envelope**2

        # Take square root and store in systematics dict
        for name, error_squared in syst_errors.items():
            systematics[name] = sqrt(error_squared)
            syst_error_squared += error_squared

    # Add flat rate uncertainty (applied to total events)
    if rate_unc > 0.0:
        rate_error = abs(events * rate_unc)
        if rate_unc_name:
            systematics[rate_unc_name] = rate_error
        syst_error_squared += rate_error**2

    syst_error = sqrt(syst_error_squared)
    total_error = sqrt(stat_error**2 + syst_error**2)

    return {
        "events": float(events),
        "stat_error": stat_error,
        "systematics": systematics,
        "syst_error": syst_error,
        "total_error": total_error
    }


def load_systematic_variations(era, sample, channel, histkey, systematics, analyzer, flag, debug=False):
    """Load systematic up/down variations for a sample

    Args:
        systematics: Dictionary mapping systematic names to [up_variation, down_variation] pairs
                    e.g., {"L1Prefire": ["L1Prefire_Up", "L1Prefire_Down"]}

    Returns:
        List of (name, h_up, h_down) tuples, or None if no systematics found
    """
    hSysts = []

    for syst, sources in systematics.items():
        syst_up, syst_down = tuple(sources)
        file_path = f"{WORKDIR}/SKNanoOutput/{analyzer}/{flag}_RunSyst/{era}/Skim_TriLep_{sample}.root"
        hist_path_up = f"{channel}/{syst_up}/{histkey}"
        hist_path_down = f"{channel}/{syst_down}/{histkey}"

        h_up = load_histogram(file_path, hist_path_up, era)
        h_down = load_histogram(file_path, hist_path_down, era)
        if h_up and h_down:
            hSysts.append((syst, h_up, h_down))
        elif debug:
            print(f"[DEBUG]     Missing {syst}: h_up={h_up is not None}, h_down={h_down is not None}")

    return hSysts if hSysts else None


def validate_era_systematics(era_systematics, era_list):
    """
    Validate that all eras in era_list have identical systematic sources.
    Raises ValueError if systematics differ across eras.

    Args:
        era_systematics: Dictionary mapping era to systematic dict
        era_list: List of eras to validate

    Raises:
        ValueError: If systematic sources differ across eras
    """
    if len(era_list) <= 1:
        return  # Nothing to validate

    # Get reference systematics from first era
    ref_era = era_list[0]
    if ref_era not in era_systematics:
        raise ValueError(f"Reference era {ref_era} not found in ERA_SYSTEMATICS")

    ref_systs = set(era_systematics[ref_era].keys())

    # Compare all other eras
    for era in era_list[1:]:
        if era not in era_systematics:
            raise ValueError(f"Era {era} not found in ERA_SYSTEMATICS")

        era_systs = set(era_systematics[era].keys())

        if era_systs != ref_systs:
            missing_in_era = ref_systs - era_systs
            extra_in_era = era_systs - ref_systs

            error_msg = f"Systematic sources differ between {ref_era} and {era}:\n"
            if missing_in_era:
                error_msg += f"  Missing in {era}: {sorted(missing_in_era)}\n"
            if extra_in_era:
                error_msg += f"  Extra in {era}: {sorted(extra_in_era)}\n"
            raise ValueError(error_msg)


def sum_sample_errors(error_dicts):
    """
    Sum errors from multiple samples in quadrature (for category totals).
    Assumes samples are independent.

    Args:
        error_dicts: List of dicts with events, stat_error, systematics, syst_error, total_error

    Returns:
        dict with merged events and errors, including per-systematic breakdown
    """
    total_events = sum(d["events"] for d in error_dicts)
    total_stat = sqrt(sum(d["stat_error"]**2 for d in error_dicts))

    # Merge systematic uncertainties by source
    # Collect all unique systematic names
    all_syst_names = set()
    for d in error_dicts:
        all_syst_names.update(d["systematics"].keys())

    # Sum each systematic in quadrature across samples
    merged_systematics = {}
    for syst_name in all_syst_names:
        syst_squared = sum(d["systematics"].get(syst_name, 0.0)**2 for d in error_dicts)
        merged_systematics[syst_name] = sqrt(syst_squared)

    # Calculate total systematic error
    total_syst = sqrt(sum(v**2 for v in merged_systematics.values()))
    total_error = sqrt(total_stat**2 + total_syst**2)

    return {
        "events": total_events,
        "stat_error": total_stat,
        "systematics": merged_systematics,
        "syst_error": total_syst,
        "total_error": total_error
    }


def get_conv_scale_factor(era_list, sample, channel):
    """
    Get conversion scale factor for ZG channels, applied to conversion samples only.
    Returns (scale, relative_uncertainty).

    The scale factor corrects for conversion lepton identification efficiency differences
    between data and MC in the ZG control region.

    For multi-era (Run2/Run3), loads individual era SFs and returns weighted average.
    Uncertainty is calculated from envelope of all systematic variations.
    """
    if not channel.startswith("ZG"):
        return 1.0, 0.0

    # Load conversion SFs for each era
    scales = []
    uncertainties = []

    for era in era_list:
        conv_sf_path = f"{WORKDIR}/TriLepton/results/{channel}/{era}/ConvSF.json"
        if not os.path.exists(conv_sf_path):
            print(f"[WARNING] Conversion SF not found: {conv_sf_path}")
            continue

        try:
            with open(conv_sf_path, 'r') as f:
                conv_data = json.load(f)

            # Find central value
            central_sf = None
            syst_values = []

            for correction in conv_data.get("corrections", []):
                name = correction.get("name", "")
                value = float(correction["data"]["expression"])

                if "Central" in name:
                    central_sf = value
                else:
                    # Collect all systematic variations for uncertainty calculation
                    syst_values.append(value)

            if central_sf is None:
                print(f"[WARNING] No central SF found in {conv_sf_path}")
                continue

            # Calculate uncertainty from envelope of systematics
            if syst_values:
                max_variation = max(abs(v - central_sf) for v in syst_values)
                rel_unc = max_variation / central_sf if central_sf != 0 else 0.0
            else:
                rel_unc = 0.0

            scales.append(central_sf)
            uncertainties.append(rel_unc)

        except Exception as e:
            print(f"[WARNING] Failed to load conversion SF from {conv_sf_path}: {e}")
            continue

    # Return weighted average (for now, simple average since we don't have luminosity weights here)
    # TODO: Implement luminosity-weighted average for Run2/Run3
    if scales:
        avg_scale = sum(scales) / len(scales)
        avg_unc = sqrt(sum(u**2 for u in uncertainties)) / len(uncertainties) if uncertainties else 0.0
        return avg_scale, avg_unc
    else:
        print(f"[WARNING] No conversion SFs loaded for {channel}, using scale=1.0")
        return 1.0, 0.0


def main():
    parser = argparse.ArgumentParser(description="Extract sample breakdown with stat/syst errors")
    parser.add_argument("--era", required=True, type=str,
                       help="Era (2016preVFP, 2016postVFP, 2017, 2018, Run2, 2022, 2022EE, 2023, 2023BPix, Run3)")
    parser.add_argument("--channel", required=True, type=str,
                       help="Channel (SR1E2Mu, SR3Mu, ZFake1E2Mu, ZFake3Mu, ZG1E2Mu, ZG3Mu)")
    parser.add_argument("--exclude", default=None, type=str,
                       help="Exclude systematics: WZSF, ConvSF, or Syst")
    parser.add_argument("--blind", action="store_true",
                       help="Blind data")
    parser.add_argument("--onZ", action="store_true",
                       help="Use pair_onZ/mass histogram (subset with Z mass window, only for SR/ZFake channels)")
    args = parser.parse_args()

    # Determine if this is a control region channel
    is_control_region = args.channel.startswith("ZG") or args.channel.startswith("WZ")

    # Deprecate --onZ flag for control regions
    if args.onZ and is_control_region:
        raise ValueError(f"--onZ flag is not supported for control region channels (ZG, WZ). "
                        f"Control regions use ZCand/mass by default.")

    # Histogram key selection:
    # - Control regions (ZG, WZ): ZCand/mass
    # - 1E2Mu SR/ZFake: pair/mass by default, or pair_onZ/mass with --onZ flag
    # - 3Mu SR/ZFake: pair_lowM/mass by default, or pair_lowM_onZ/mass with --onZ flag
    if is_control_region:
        HISTKEY = "ZCand/mass"
    elif "1E2Mu" in args.channel:
        HISTKEY = "pair_onZ/mass" if args.onZ else "pair/mass"
    elif "3Mu" in args.channel:
        HISTKEY = "pair_lowM_onZ/mass" if args.onZ else "pair_lowM/mass"
    else:
        raise ValueError(f"Cannot determine histogram key for channel: {args.channel}")

    # Check channel validity
    if args.channel not in ["SR1E2Mu", "SR3Mu", "ZFake1E2Mu", "ZFake3Mu", "ZG1E2Mu", "ZG3Mu", "WZ1E2Mu", "WZ3Mu"]:
        raise ValueError(f"Invalid channel: {args.channel}")

    # Extract channel flag (1E2Mu or 3Mu)
    if "1E2Mu" in args.channel:
        channel_flag = "1E2Mu"
    elif "3Mu" in args.channel:
        channel_flag = "3Mu"
    else:
        raise ValueError(f"Cannot extract channel flag from: {args.channel}")

    # Determine ANALYZER based on channel
    ANALYZER = ""
    if "SR" in args.channel or "ZFake" in args.channel:
        ANALYZER = "PromptSelector"
    elif "ZG" in args.channel or "WZ" in args.channel:
        ANALYZER = "CRPromptSelector"
    else:
        raise ValueError(f"Cannot determine ANALYZER for channel: {args.channel}")

    # Determine FLAG based on channel
    if "1E2Mu" in args.channel:
        FLAG = "Run1E2Mu"
    elif "3Mu" in args.channel:
        FLAG = "Run3Mu"
    else:
        raise ValueError(f"Cannot determine FLAG for channel: {args.channel}")

    # Get era list (handles Run2/Run3)
    era_list = get_era_list(args.era)

    # Create a channel args object for load_era_configs
    class ChannelArgs:
        def __init__(self, channel_flag):
            self.channel = channel_flag

    channel_args = ChannelArgs(channel_flag)

    # Load configurations using HistoUtils functions
    ERA_SAMPLES, ERA_SYSTEMATICS = load_era_configs(channel_args, era_list)
    DATAPERIODs, MC_CATEGORIES, MCList = get_sample_lists(ERA_SAMPLES, ["nonprompt", "conv", "ttX", "diboson", "others"])
    SYSTEMATICS = merge_systematics(ERA_SYSTEMATICS)

    # Validate that systematics are consistent across eras (unless systematics are excluded)
    if len(era_list) > 1 and not (args.exclude == "Syst"):
        validate_era_systematics(ERA_SYSTEMATICS, era_list)

    # Unpack MC categories
    nonprompt = MC_CATEGORIES["nonprompt"]

    # Determine WZ sample name based on era (Run2 vs Run3)
    if args.era in ["2016preVFP", "2016postVFP", "2017", "2018", "Run2"]:
        WZ_SAMPLES = ["WZTo3LNu_amcatnlo", "ZZTo4L_powheg"]
    elif args.era in ["2022", "2022EE", "2023", "2023BPix", "Run3"]:
        WZ_SAMPLES = ["WZTo3LNu_powheg", "ZZTo4L_powheg"]
    else:
        raise ValueError(f"Invalid era: {args.era}")

    # Initialize output structure
    output = {
        "era": args.era,
        "channel": args.channel,
        "histkey": HISTKEY,
        "systematic": f"No{args.exclude}" if args.exclude else "Central",
        "data": None,
        "samples": {},
        "categories": {},
        "total_background": None,
        "signals": {}
    }

    print(f"[INFO] Processing era={args.era}, channel={args.channel}, histkey={HISTKEY}")

    # ===== 1. Load and sum DATA histograms =====
    if not args.blind:
        print("[INFO] Loading data histograms...")
        era_data_hists = []
        # Dictionary to track individual data sample histograms across eras
        data_sample_hists = {}

        for era in era_list:
            if era not in ERA_SAMPLES:
                print(f"[WARNING] Era {era} not found in ERA_SAMPLES")
                continue
            for sample in ERA_SAMPLES[era]["data"]:
                file_path = f"{WORKDIR}/SKNanoOutput/{ANALYZER}/{FLAG}/{era}/Skim_TriLep_{sample}.root"
                hist_path = f"{args.channel}/Central/{HISTKEY}"
                h = load_histogram(file_path, hist_path, era)
                if h:
                    era_data_hists.append(h)
                    # Track individual sample histograms
                    if sample not in data_sample_hists:
                        data_sample_hists[sample] = []
                    data_sample_hists[sample].append(h)

        # Create merged data histogram
        if era_data_hists:
            data_hist = sum_histograms(era_data_hists, "data_total")
            output["data"] = extract_stat_syst_errors(data_hist)
            print(f"[INFO] Data: {output['data']['events']:.1f} ± {output['data']['stat_error']:.1f}")
        else:
            print("[WARNING] No data histograms found")

        # Add individual data samples to output
        for sample, hists in data_sample_hists.items():
            h_total = sum_histograms(hists, f"{sample}_total")
            output["samples"][sample] = extract_stat_syst_errors(h_total)
            print(f"[INFO]   {sample}: {output['samples'][sample]['events']:.2f} events")

    # ===== 2. Load and process NONPROMPT histograms =====
    print("[INFO] Loading nonprompt histograms...")
    nonprompt_hists = {}
    for sample in nonprompt:
        era_hists = []
        for era in era_list:
            if era not in ERA_SAMPLES:
                continue
            file_path = f"{WORKDIR}/SKNanoOutput/{ANALYZER.replace('Prompt', 'Matrix')}/{FLAG}/{era}/Skim_TriLep_{sample}.root"
            hist_path = f"{args.channel}/Central/{HISTKEY}"
            h = load_histogram(file_path, hist_path, era)
            if h:
                era_hists.append(h)

        if era_hists:
            h_total = sum_histograms(era_hists, f"{sample}_total")
            # Nonprompt has 30% flat systematic uncertainty
            # Use "nonprompt_" prefix to avoid overwriting data samples
            nonprompt_sample_name = f"nonprompt_{sample}"
            output["samples"][nonprompt_sample_name] = extract_stat_syst_errors(
                h_total, rate_unc=0.30, rate_unc_name="nonprompt_rate")
            nonprompt_hists[nonprompt_sample_name] = h_total
            print(f"[INFO]   {nonprompt_sample_name}: {output['samples'][nonprompt_sample_name]['events']:.2f} events")

    # ===== 3. Load and process MC histograms =====
    print("[INFO] Loading MC histograms...")
    mc_hists = {}
    for sample in MCList:
        era_hists = []
        # Dictionary to track systematic variations per source: {syst_name: {'up': [h_up_era1, ...], 'down': [h_down_era1, ...]}}
        syst_variations = {}

        for era in era_list:
            if era not in ERA_SAMPLES:
                continue
            # Load central histogram
            file_path = f"{WORKDIR}/SKNanoOutput/{ANALYZER}/{FLAG}_RunSyst/{era}/Skim_TriLep_{sample}.root"
            hist_path = f"{args.channel}/Central/{HISTKEY}"
            h = load_histogram(file_path, hist_path, era)
            if h:
                era_hists.append(h)

                # Load systematic variations (unless excluded)
                if not args.exclude or args.exclude != "Syst":
                    # Use era-specific systematics (validated to be consistent across eras)
                    era_systs = ERA_SYSTEMATICS.get(era, {})
                    if era_systs:
                        hSysts = load_systematic_variations(era, sample, args.channel,
                                                           HISTKEY, era_systs, ANALYZER, FLAG, False)
                        if hSysts:
                            # Store up/down variations for each systematic source
                            for syst_name, h_up, h_down in hSysts:
                                if syst_name not in syst_variations:
                                    syst_variations[syst_name] = {'up': [], 'down': []}
                                syst_variations[syst_name]['up'].append(h_up)
                                syst_variations[syst_name]['down'].append(h_down)

        if era_hists:
            h_total = sum_histograms(era_hists, f"{sample}_total")

            # Sum systematic variations across eras
            combined_systs = None
            if syst_variations:
                combined_systs = []
                for syst_name, variations in syst_variations.items():
                    h_up_total = sum_histograms(variations['up'], f"{sample}_{syst_name}_up")
                    h_down_total = sum_histograms(variations['down'], f"{sample}_{syst_name}_down")
                    if h_up_total and h_down_total:
                        combined_systs.append((syst_name, h_up_total, h_down_total))

            # Determine rate uncertainty
            rate_unc = 0.0
            rate_unc_name = None
            if sample in WZ_SAMPLES and not (args.exclude == "WZSF"):
                rate_unc = 0.20  # 20% WZ uncertainty
                rate_unc_name = "WZ_rate"

            # Apply conversion scale factor if needed (only for conversion samples in ZG channels)
            if args.channel.startswith("ZG") and not (args.exclude == "ConvSF") and sample in MC_CATEGORIES["conv"]:
                scale, rel_unc = get_conv_scale_factor(era_list, sample, args.channel)
                if scale != 1.0:
                    h_total.Scale(scale)
                    # Combine conversion uncertainty with other rate uncertainties
                    if rate_unc > 0.0:
                        rate_unc = sqrt(rate_unc**2 + rel_unc**2)
                    else:
                        rate_unc = rel_unc
                        rate_unc_name = "conv_rate"

            output["samples"][sample] = extract_stat_syst_errors(
                h_total, combined_systs, rate_unc, rate_unc_name)
            mc_hists[sample] = h_total
            print(f"[INFO]   {sample}: {output['samples'][sample]['events']:.2f} events")

    # ===== 4. Merge into categories =====
    print("[INFO] Merging samples into categories...")
    all_hists = {**nonprompt_hists, **mc_hists}
    for category in ["nonprompt", "conv", "ttX", "diboson", "others"]:
        # Handle nonprompt samples which have "nonprompt_" prefix
        if category == "nonprompt":
            cat_sample_names = [f"nonprompt_{s}" for s in MC_CATEGORIES[category] if f"nonprompt_{s}" in output["samples"]]
        else:
            cat_sample_names = [s for s in MC_CATEGORIES[category] if s in output["samples"]]

        if cat_sample_names:
            output["categories"][category] = sum_sample_errors(
                [output["samples"][s] for s in cat_sample_names])
            print(f"[INFO]   {category}: {output['categories'][category]['events']:.2f} events")

    # Calculate total background
    if output["categories"]:
        output["total_background"] = sum_sample_errors(list(output["categories"].values()))
        print(f"[INFO] Total background: {output['total_background']['events']:.2f} events")
    else:
        output["total_background"] = None

    # ===== 5. Load signal histograms (if SR channel) =====
    if args.channel.startswith("SR"):
        print("[INFO] Loading signal histograms...")
        SIGNALS = load_signal_config()
        print(f"[INFO] Found {len(SIGNALS)} signal mass points in configuration")

        for signal_mass in SIGNALS:
            era_signal_hists = []
            for era in era_list:
                if era not in ERA_SAMPLES:
                    continue
                signal_name = f"TTToHcToWAToMuMu-{signal_mass}"
                file_path = f"{WORKDIR}/SKNanoOutput/{ANALYZER}/{FLAG}_RunSyst/{era}/{signal_name}.root"
                hist_path = f"{args.channel}/Central/{HISTKEY}"
                h = load_histogram(file_path, hist_path, era)
                if h:
                    era_signal_hists.append(h)

            if era_signal_hists:
                h_signal = sum_histograms(era_signal_hists, signal_mass)
                output["signals"][signal_mass] = extract_stat_syst_errors(h_signal)
                print(f"[INFO]   {signal_mass}: {output['signals'][signal_mass]['events']:.2f} events")

    # ===== 6. Save to JSON =====
    syst_tag = f"No{args.exclude}" if args.exclude else "Central"
    onZ_tag = "_onZ" if args.onZ else ""
    json_dir = f"{WORKDIR}/TriLepton/results/{args.era}/{args.channel}"
    json_filename = f"sample_breakdown_{syst_tag}{onZ_tag}.json"
    json_path = f"{json_dir}/{json_filename}"

    os.makedirs(json_dir, exist_ok=True)

    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n[INFO] Sample breakdown saved to {json_path}")
    print(f"[INFO] Processed {len(output['samples'])} samples, "
          f"{len(output['categories'])} categories, {len(output['signals'])} signals")


if __name__ == "__main__":
    main()
