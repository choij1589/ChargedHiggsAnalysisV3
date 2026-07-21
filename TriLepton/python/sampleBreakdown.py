#!/usr/bin/env python3
"""
sampleBreakdown.py

Extract event counts and errors (statistical and systematic separately) for all samples.

Histogram selection:
- 1E2Mu SR/ZFake channels: 'pair/mass' by default, or 'pair_onZ/mass' with --onZ flag
- 3Mu SR/ZFake channels: 'pair_lowM/mass' by default, or 'pair_lowM_onZ/mass' with --onZ flag
- Control regions (ZG/WZ/TTZ): 'ZCand/mass' always (--onZ flag not supported)

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
from ROOT import gROOT

gROOT.SetBatch(True)

# Flat normalization uncertainty for the "others" category. These samples are absent
# from KFactors.json and otherwise carry no theory norm (see plot.py:apply_others_uncertainty).
OTHERS_XSEC_UNC = 0.50

# Add Common/Tools to path
WORKDIR = os.environ.get("WORKDIR", os.getcwd())
sys.path.insert(0, f"{WORKDIR}/Common/Tools")

from plotter import get_era_list
from HistoUtils import (load_histogram, sum_histograms, load_era_configs, get_sample_lists,
                        load_systematic_variations, CorrelatedTotalBuilder)
from utils import build_sknanoutput_path, scale_with_variations

CATEGORIES = ["nonprompt", "conv", "ttX", "diboson", "others"]


def load_signal_config():
    """Load signal mass points from configuration file."""
    config_path = f"{WORKDIR}/TriLepton/configs/signals.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Signal configuration not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    return config.get("signals", [])


def load_kfactors():
    """Load K-factors from configuration file."""
    kfactors_path = f"{WORKDIR}/Common/Data/KFactors.json"
    if not os.path.exists(kfactors_path):
        raise FileNotFoundError(f"K-factors configuration not found: {kfactors_path}")

    with open(kfactors_path, 'r') as f:
        return json.load(f)


def load_fake_norm():
    """Load per-era nonprompt normalization uncertainties."""
    fake_norm_path = f"{WORKDIR}/Common/Data/FakeNorm.json"
    if not os.path.exists(fake_norm_path):
        raise FileNotFoundError(f"FakeNorm configuration not found: {fake_norm_path}")

    with open(fake_norm_path, 'r') as f:
        return json.load(f)


def get_run_period(era):
    """Determine run period (Run2/Run3) from era."""
    if era in ["2016preVFP", "2016postVFP", "2017", "2018", "Run2"]:
        return "Run2"
    elif era in ["2022", "2022EE", "2023", "2023BPix", "Run3"]:
        return "Run3"
    else:
        raise ValueError(f"Cannot determine run period for era: {era}")


def get_kfactor_info(kfactors, sample, run_period):
    """Get K-factor and cross-section uncertainty for a sample.

    Args:
        kfactors: Dictionary loaded from KFactors.json
        sample: Sample name
        run_period: "Run2" or "Run3"

    Returns:
        tuple: (kfactor, xsec_rel_unc) where xsec_rel_unc is relative uncertainty (e.g., 0.075 for 7.5%)
               Returns (1.0, 0.0) if sample not in K-factors config
    """
    if run_period not in kfactors:
        return 1.0, 0.0

    run_kfactors = kfactors[run_period]
    if sample not in run_kfactors:
        return 1.0, 0.0

    kfactor = run_kfactors[sample].get("kFactor", 1.0)
    # xsecErr is stored as multiplicative factor (e.g., 1.075 means 7.5% uncertainty)
    xsec_err_factor = run_kfactors[sample].get("xsecErr", 1.0)
    xsec_rel_unc = xsec_err_factor - 1.0

    return kfactor, xsec_rel_unc


def add_systematic_error(systematics, name, error):
    """Add a systematic source, combining duplicate names in quadrature."""
    if name in systematics:
        systematics[name] = sqrt(systematics[name]**2 + error**2)
    else:
        systematics[name] = error


def extract_stat_syst_errors(h_central, hSysts=None, rate_unc=0.0, rate_unc_name=None,
                             rate_uncs=None):
    """
    Extract statistical and systematic errors separately.

    Args:
        h_central: Central histogram (contains stat errors from Sumw2)
        hSysts: List of (name, h_up, h_down) systematic variation tuples (optional)
        rate_unc: Flat rate uncertainty to add (e.g., 0.30 for nonprompt)
        rate_unc_name: Name for the rate uncertainty (e.g., "nonprompt_rate")
        rate_uncs: Optional list of (name, relative uncertainty) rate uncertainties

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
            add_systematic_error(systematics, name, sqrt(error_squared))
            syst_error_squared += error_squared

    # Add flat rate uncertainty (applied to total events)
    if rate_uncs is None:
        rate_uncs = []
        if rate_unc > 0.0:
            rate_uncs.append((rate_unc_name, rate_unc))

    for name, rel_unc in rate_uncs:
        if rel_unc <= 0.0:
            continue
        rate_error = abs(events * rel_unc)
        if name:
            add_systematic_error(systematics, name, rate_error)
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


def load_conv_sf_data():
    """Load conversion SF data from Common/Data/ConvSF.json."""
    conv_sf_path = f"{WORKDIR}/Common/Data/ConvSF.json"
    if not os.path.exists(conv_sf_path):
        raise FileNotFoundError(f"Conversion SF file not found: {conv_sf_path}")
    with open(conv_sf_path, 'r') as f:
        return json.load(f)


def get_fake_norm_uncertainty(fake_norm, flag, era):
    """Return the nonprompt rate uncertainty for one flag/era."""
    try:
        return fake_norm[flag][era]
    except KeyError:
        print(f"[WARNING] No FakeNorm entry for flag={flag}, era={era}; using 30% fallback")
        return 0.30


def get_conv_scale_factor(conv_sf_data, era, channel):
    """
    Get conversion scale factor for one era and final-state channel.

    Returns (scale, relative_uncertainty).

    Reads Common/Data/ConvSF.json (structure: {channel_flag: {era: {central, total, ...}}}).
    The 'total' field is already a relative fraction (e.g. 0.10 = 10%).
    TTZ2E1Mu has no dedicated ConvSF measurement; match plotting behavior with
    no central scale and a 20% rate uncertainty.
    """
    if "1E2Mu" in channel:
        channel_flag = "1E2Mu"
    elif "2E1Mu" in channel:
        return 1.0, 0.20
    elif "3Mu" in channel:
        channel_flag = "3Mu"
    else:
        raise ValueError(f"Cannot extract channel flag from: {channel}")

    era_data = conv_sf_data.get(channel_flag, {}).get(era)
    if era_data is None:
        raise KeyError(f"No ConvSF entry for channel_flag={channel_flag}, era={era} in ConvSF.json")

    return era_data["central"], era_data["total"]


def main():
    parser = argparse.ArgumentParser(description="Extract sample breakdown with stat/syst errors")
    parser.add_argument("--era", required=True, type=str,
                       help="Era (2016preVFP, 2016postVFP, 2017, 2018, Run2, 2022, 2022EE, 2023, 2023BPix, Run3)")
    parser.add_argument("--channel", required=True, type=str,
                       help="Channel (SR1E2Mu, SR3Mu, ZFake1E2Mu, ZFake3Mu, ZG1E2Mu, ZG3Mu, WZ1E2Mu, WZ3Mu, TTZ2E1Mu)")
    parser.add_argument("--exclude", default=None, type=str,
                       help="Exclude systematics: WZSF, ConvSF, or Syst")
    parser.add_argument("--blind", action="store_true",
                       help="Blind data")
    parser.add_argument("--onZ", action="store_true",
                       help="Use pair_onZ/mass histogram (subset with Z mass window, only for SR/ZFake channels)")
    args = parser.parse_args()

    # Determine if this is a control region channel
    is_control_region = (
        args.channel.startswith("ZG")
        or args.channel.startswith("WZ")
        or args.channel.startswith("TTZ")
    )

    # Deprecate --onZ flag for control regions
    if args.onZ and is_control_region:
        raise ValueError(f"--onZ flag is not supported for control region channels (ZG, WZ, TTZ). "
                        f"Control regions use ZCand/mass by default.")

    # Histogram key selection:
    # - Control regions (ZG, WZ, TTZ): ZCand/mass
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
    if args.channel not in ["SR1E2Mu", "SR3Mu", "ZFake1E2Mu", "ZFake3Mu", "ZG1E2Mu", "ZG3Mu", "WZ1E2Mu", "WZ3Mu", "TTZ2E1Mu"]:
        raise ValueError(f"Invalid channel: {args.channel}")

    # Extract channel flag (1E2Mu, 2E1Mu, or 3Mu)
    if "1E2Mu" in args.channel:
        channel_flag = "1E2Mu"
    elif "2E1Mu" in args.channel:
        channel_flag = "2E1Mu"
    elif "3Mu" in args.channel:
        channel_flag = "3Mu"
    else:
        raise ValueError(f"Cannot extract channel flag from: {args.channel}")

    # Determine FLAG based on channel
    if "1E2Mu" in args.channel:
        FLAG = "Run1E2Mu"
    elif "2E1Mu" in args.channel:
        FLAG = "Run2E1Mu"
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
    DATAPERIODs, MC_CATEGORIES, _ = get_sample_lists(ERA_SAMPLES, ["nonprompt", "conv", "ttX", "diboson", "others"])

    # Validate that systematics are consistent across eras (unless systematics are excluded)
    if len(era_list) > 1 and not (args.exclude == "Syst"):
        validate_era_systematics(ERA_SYSTEMATICS, era_list)

    # Unpack MC categories
    nonprompt = MC_CATEGORIES["nonprompt"]
    prompt_mc_categories = ["conv", "ttX", "diboson", "others"]
    prompt_mc_list = sum([MC_CATEGORIES[category] for category in prompt_mc_categories], [])

    # Determine WZ sample name based on era (Run2 vs Run3)
    run_period = get_run_period(args.era)
    if run_period == "Run2":
        WZ_SAMPLES = ["WZTo3LNu_amcatnlo", "ZZTo4L_powheg"]
    elif run_period == "Run3":
        WZ_SAMPLES = ["WZTo3LNu_powheg", "ZZTo4L_powheg"]

    # Map each sample to its category, for routing contributions into the builders
    SAMPLE_CATEGORY = {}
    for category in CATEGORIES:
        for sample in MC_CATEGORIES[category]:
            SAMPLE_CATEGORY[sample] = category

    # Load K-factors
    KFACTORS = load_kfactors()
    FAKENORM = load_fake_norm()
    CONV_SF_DATA = load_conv_sf_data() if args.exclude != "ConvSF" else {}

    # Category and total-background builders. Contributions are added per (era, sample)
    # so each named source stays one nuisance within an era; the background total is
    # built from every sample rather than by merging the category summaries, which
    # would re-introduce quadrature between categories for shared shape sources.
    category_builders = {category: CorrelatedTotalBuilder(category) for category in CATEGORIES}
    total_builder = CorrelatedTotalBuilder("total_background")

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
                file_path = build_sknanoutput_path(WORKDIR, args.channel, FLAG, era, sample)
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
    # The fake-rate normalization is one measurement per era, shared by every data
    # stream, so it is added as a correlated rate source.
    print("[INFO] Loading nonprompt histograms...")
    for sample in nonprompt:
        sample_builder = CorrelatedTotalBuilder(f"nonprompt_{sample}")
        for era in era_list:
            if era not in ERA_SAMPLES:
                continue
            file_path = build_sknanoutput_path(WORKDIR, args.channel, FLAG, era, sample, is_nonprompt=True)
            hist_path = f"{args.channel}/Central/{HISTKEY}"
            h = load_histogram(file_path, hist_path, era)
            if not h:
                continue

            rate_uncs = [("nonprompt_rate",
                          get_fake_norm_uncertainty(FAKENORM, FLAG, era), True)]
            for builder in (sample_builder, category_builders["nonprompt"], total_builder):
                builder.add(era, sample, h, rate_uncs=rate_uncs)

        if not sample_builder.is_empty():
            # Use "nonprompt_" prefix to avoid overwriting data samples
            nonprompt_sample_name = f"nonprompt_{sample}"
            output["samples"][nonprompt_sample_name] = sample_builder.summary()
            print(f"[INFO]   {nonprompt_sample_name}: {output['samples'][nonprompt_sample_name]['events']:.2f} events")

    # ===== 3. Load and process MC histograms =====
    print("[INFO] Loading MC histograms...")
    for sample in prompt_mc_list:
        sample_builder = CorrelatedTotalBuilder(sample)
        category = SAMPLE_CATEGORY[sample]
        kfactor, xsec_rel_unc = get_kfactor_info(KFACTORS, sample, run_period)

        for era in era_list:
            if era not in ERA_SAMPLES:
                continue
            # Load central histogram
            file_path = build_sknanoutput_path(WORKDIR, args.channel, FLAG, era, sample, run_syst=True)
            hist_path = f"{args.channel}/Central/{HISTKEY}"
            h = load_histogram(file_path, hist_path, era)
            if not h:
                continue

            # Load systematic variations (unless excluded)
            hSysts = {}
            if args.exclude != "Syst":
                # Use era-specific systematics (validated to be consistent across eras)
                era_systs = ERA_SYSTEMATICS.get(era, {})
                if era_systs:
                    hSysts = load_systematic_variations(file_path, args.channel, HISTKEY,
                                                       era_systs, era)

            scale_with_variations(h, hSysts, kfactor)

            rate_uncs = []
            # Per-process cross-section priors stay independent: a datacard writes
            # these as separate lnN per process, so they do not share a nuisance.
            if xsec_rel_unc > 0.0:
                rate_uncs.append((f"xsec_{sample}", xsec_rel_unc, False))

            # One WZNjSF measurement, shared by the samples it reweights
            if sample in WZ_SAMPLES and args.exclude != "WZSF":
                rate_uncs.append(("WZ_rate", 0.20, True))

            # One ConvSF measurement per era, shared by every conversion sample
            if args.exclude != "ConvSF" and sample in MC_CATEGORIES["conv"]:
                scale, conv_rel_unc = get_conv_scale_factor(CONV_SF_DATA, era, args.channel)
                scale_with_variations(h, hSysts, scale)
                rate_uncs.append(("conv_rate", conv_rel_unc, True))

            # Flat normalization for the "others" category. These are unrelated
            # processes absent from KFactors.json, so each carries its own prior.
            if sample in ERA_SAMPLES[era]["others"]:
                rate_uncs.append(("others_xsec", OTHERS_XSEC_UNC, False))

            for builder in (sample_builder, category_builders[category], total_builder):
                builder.add(era, sample, h, variations=hSysts, rate_uncs=rate_uncs)

        if not sample_builder.is_empty():
            output["samples"][sample] = sample_builder.summary()
            print(f"[INFO]   {sample}: {output['samples'][sample]['events']:.2f} events")

    # ===== 4. Summarize categories and the background total =====
    print("[INFO] Merging samples into categories...")
    for category in CATEGORIES:
        if category_builders[category].is_empty():
            continue
        output["categories"][category] = category_builders[category].summary()
        print(f"[INFO]   {category}: {output['categories'][category]['events']:.2f} events")

    if not total_builder.is_empty():
        output["total_background"] = total_builder.summary()
        print(f"[INFO] Total background: {output['total_background']['events']:.2f} events")

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
                # Signal files don't have "Skim_TriLep_" prefix, construct path directly
                file_path = f"{WORKDIR}/SKNanoOutput/PromptAnalyzer/{FLAG}_RunSyst_RunTheoryUnc/{era}/{signal_name}.root"
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
