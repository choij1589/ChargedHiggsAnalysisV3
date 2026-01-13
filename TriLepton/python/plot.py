#!/usr/bin/env python
import os
import argparse
import logging
import json
import ROOT
from math import pow, sqrt
from plotter import ComparisonCanvas, get_era_list, get_CoM_energy
from HistoUtils import (setup_missing_histogram_logging, load_histogram,
                        calculate_systematics, sum_histograms, load_era_configs,
                        merge_systematics, get_sample_lists)
import correctionlib
ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--histkey", required=True, type=str, help="histkey")
parser.add_argument("--exclude", default=None, type=str,
                    help="exclude reweighting (WZSF, ConvSF)")
parser.add_argument("--blind", default=False, action="store_true", help="blind data")
parser.add_argument("--signals", default=["MHc70_MA15", "MHc100_MA60", "MHc130_MA90", "MHc160_MA155"],
                    nargs="+", help="Signal mass points to overlay")
parser.add_argument("--signal-scale", default=10.0, type=float,
                    help="Scale factor for signal histograms")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

WORKDIR = os.environ['WORKDIR']

# Setup missing histogram logging
missing_logger = setup_missing_histogram_logging(args)

with open("configs/histkeys.json") as f:
    config = json.load(f)[args.histkey]

# Load K-factors
KFACTORS_PATH = f"{WORKDIR}/Common/Data/KFactors.json"
with open(KFACTORS_PATH) as f:
    KFACTORS = json.load(f)

config["era"] = args.era
config["CoM"] = get_CoM_energy(args.era)
config["rTitle"] = "Data / Pred"
config["maxDigits"] = 3
config["blind"] = args.blind  # Pass blind flag to ComparisonCanvas
config["overflow"] = True  # Accumulate overflow into last visible bin
if not args.blind:
    config["chi2_test"] = True
    config["normalize_chi2"] = False
#### Configurations
# Get era list for merging
era_list = get_era_list(args.era)

if args.era in ["2016preVFP", "2016postVFP", "2017", "2018", "Run2"]:
    RUN = "Run2"
    samplename_WZ = "WZTo3LNu_amcatnlo"
elif args.era in ["2022", "2022EE", "2023", "2023BPix", "Run3"]:
    RUN = "Run3"
    samplename_WZ = "WZTo3LNu_powheg"
else:
    raise ValueError(f"Invalid era: {args.era}")

## Check channel
if args.channel not in ["SR1E2Mu", "SR3Mu", "ZFake1E2Mu", "ZFake3Mu", "ZG1E2Mu", "ZG3Mu", "WZ1E2Mu", "WZ3Mu", "TTZ2E1Mu"]:
    raise ValueError(f"Invalid channel: {args.channel}")

ANALYZER = ""
if "SR" in args.channel or "ZFake" in args.channel or "TTZ" in args.channel:
    ANALYZER = "PromptSelector"
if "ZG" in args.channel or "WZ" in args.channel:
    ANALYZER = "CRPromptSelector"
config["channel"] = args.channel

if "1E2Mu" in args.channel:
    FLAG = "Run1E2Mu"
    channel_flag = "1E2Mu"
elif "2E1Mu" in args.channel:
    FLAG = "Run2E1Mu"
    channel_flag = "2E1Mu"
elif "3Mu" in args.channel:
    FLAG = "Run3Mu"
    channel_flag = "3Mu"
else:
    raise ValueError(f"Cannot determine FLAG for channel: {args.channel}")

# Create a modified args object for the common function
class ChannelArgs:
    def __init__(self, channel_flag):
        self.channel = channel_flag

channel_args = ChannelArgs(channel_flag)

# Load configurations
ERA_SAMPLES, ERA_SYSTEMATICS = load_era_configs(channel_args, era_list)
DATAPERIODs, MC_CATEGORIES, MCList = get_sample_lists(ERA_SAMPLES, ["nonprompt", "conv", "ttX", "diboson", "others"])
SYSTs = merge_systematics(ERA_SYSTEMATICS)

# Unpack MC categories for backward compatibility
nonprompt = MC_CATEGORIES["nonprompt"]
conv = MC_CATEGORIES["conv"]
ttX = MC_CATEGORIES["ttX"]
diboson = MC_CATEGORIES["diboson"]
others = MC_CATEGORIES["others"]

if args.exclude:
    OUTPUTPATH = f"{WORKDIR}/TriLepton/plots/{args.era}/{args.channel}/No{args.exclude}/{args.histkey.replace('/', '_')}.png"
else:
    OUTPUTPATH = f"{WORKDIR}/TriLepton/plots/{args.era}/{args.channel}/Central/{args.histkey.replace('/', '_')}.png"

os.makedirs(os.path.dirname(OUTPUTPATH), exist_ok=True)


def get_zg_channel_for_convsf(channel):
    """Map channel to corresponding ZG channel for ConvSF loading.
    Returns None if no ZG measurement exists for this channel."""
    if "1E2Mu" in channel:
        return "ZG1E2Mu"
    elif "3Mu" in channel:
        return "ZG3Mu"
    elif "2E1Mu" in channel:
        # No ZG2E1Mu measurement yet - will use default proxy
        return None
    else:
        return None

def load_conv_scale_factors(channel, era_list):
    """Load conversion scale factors for the channel.
    Maps non-ZG channels to their corresponding ZG channel for SF loading."""
    conv_sf = {}

    # Get the ZG channel for loading ConvSF
    zg_channel = get_zg_channel_for_convsf(channel)

    if zg_channel is None:
        # No ZG measurement for this channel - will use default proxy
        logging.info(f"No ZG measurement for channel {channel}, will use default ConvSF = 1.0 ± 20%")
        return conv_sf

    for era in era_list:
        sf_file = f"{WORKDIR}/TriLepton/results/{zg_channel}/{era}/ConvSF.json"

        if not os.path.exists(sf_file):
            logging.warning(f"Conversion SF file not found: {sf_file}")
            continue

        try:
            cset = correctionlib.CorrectionSet.from_file(sf_file)
            conv_sf[era] = {"cset": cset, "zg_channel": zg_channel}
        except Exception as e:
            logging.warning(f"Failed to load conversion SF from {sf_file}: {e}")

    return conv_sf

def apply_conv_scale_factor(hist, sample, era, conv_sf, channel, era_samples):
    """Apply conversion scale factor to conversion samples.
    If no ConvSF measurement is available, uses default SF = 1.0 with 20% uncertainty."""
    if args.exclude == "ConvSF":
        return hist

    # Check if this sample is in the conv list for this era
    # era_samples[era] already contains the channel-specific data
    is_conv_sample = sample in era_samples[era]["conv"]
    if not is_conv_sample:
        return hist

    # Case 1: No ConvSF available for this era - use default proxy (SF = 1.0 ± 20%)
    if not conv_sf or era not in conv_sf:
        # Apply default 20% rate uncertainty as proxy
        for bin in range(hist.GetNcells()):
            content = hist.GetBinContent(bin)
            stat_error = hist.GetBinError(bin)
            rate_error = content * 0.20  # 20% rate uncertainty
            # Combine statistical and rate uncertainties in quadrature
            total_error = sqrt(stat_error**2 + rate_error**2)
            hist.SetBinError(bin, total_error)
        return hist

    # Case 2: ConvSF measurement available - apply measured SF
    try:
        cset = conv_sf[era]["cset"]
        zg_channel = conv_sf[era]["zg_channel"]

        # Get central scale factor using the ZG channel name
        sf_name = f"ConvSF_{zg_channel}_{era}_Central"
        sf_central = cset[sf_name].evaluate()

        # Apply scale factor
        hist.Scale(sf_central)

        # Calculate systematic uncertainty
        # Get prompt and nonprompt uncertainties
        sf_prompt_up = cset[f"ConvSF_{zg_channel}_{era}_prompt_up"].evaluate()
        sf_prompt_down = cset[f"ConvSF_{zg_channel}_{era}_prompt_down"].evaluate()
        sf_nonprompt_up = cset[f"ConvSF_{zg_channel}_{era}_nonprompt_up"].evaluate()
        sf_nonprompt_down = cset[f"ConvSF_{zg_channel}_{era}_nonprompt_down"].evaluate()

        # Calculate relative uncertainties
        prompt_rel_unc = max(abs(sf_prompt_up - sf_central), abs(sf_central - sf_prompt_down)) / sf_central
        nonprompt_rel_unc = max(abs(sf_nonprompt_up - sf_central), abs(sf_central - sf_nonprompt_down)) / sf_central

        # Quadrature sum of prompt and nonprompt uncertainties
        total_rel_unc = sqrt(prompt_rel_unc**2 + nonprompt_rel_unc**2)

        # Apply rate uncertainty to all bins
        for bin in range(hist.GetNcells()):
            content = hist.GetBinContent(bin)
            stat_error = hist.GetBinError(bin)
            rate_error = content * total_rel_unc
            # Combine statistical and rate uncertainties in quadrature
            total_error = sqrt(stat_error**2 + rate_error**2)
            hist.SetBinError(bin, total_error)


    except Exception as e:
        logging.warning(f"Failed to apply conversion SF to {sample} in {era}: {e}")

    return hist

def apply_wz_uncertainty(hist, sample, args):
    """Apply 20% flat uncertainty to WZ samples when WZSF is applied"""
    # Only apply when WZSF is NOT excluded and sample is WZ
    if args.exclude == "WZSF":
        return hist
    
    # Check if this is a WZ sample
    is_wz_sample = "WZTo3LNu" in sample or "ZZTo4L" in sample
    if not is_wz_sample:
        return hist
    
    # Apply 20% rate uncertainty to all bins
    for bin in range(hist.GetNcells()):
        content = hist.GetBinContent(bin)
        stat_error = hist.GetBinError(bin)
        rate_error = content * 0.20  # 20% rate uncertainty
        # Combine statistical and rate uncertainties in quadrature
        total_error = sqrt(stat_error**2 + rate_error**2)
        hist.SetBinError(bin, total_error)

    return hist

def apply_kfactor(hist, sample, run):
    """Apply K-factor and theory uncertainty to sample if defined in KFactors.json

    The xsecErr in KFactors.json is a multiplicative factor (e.g., 1.075 means 7.5% uncertainty).
    This uncertainty is applied to all bins in quadrature with existing errors.
    """
    if run not in KFACTORS:
        return hist

    kfactors = KFACTORS[run]
    if sample not in kfactors:
        return hist

    kfactor = kfactors[sample]["kFactor"]
    hist.Scale(kfactor)
    logging.debug(f"Applied K-factor {kfactor} to {sample}")

    # Apply theory uncertainty if available
    if "xsecErr" in kfactors[sample]:
        xsec_err_factor = kfactors[sample]["xsecErr"]
        # Convert multiplicative factor to relative uncertainty (e.g., 1.075 -> 0.075)
        rel_unc = xsec_err_factor - 1.0

        for bin in range(hist.GetNcells()):
            content = hist.GetBinContent(bin)
            stat_error = hist.GetBinError(bin)
            theory_error = content * rel_unc
            # Combine statistical and theory uncertainties in quadrature
            total_error = sqrt(stat_error**2 + theory_error**2)
            hist.SetBinError(bin, total_error)
        logging.debug(f"Applied theory uncertainty {rel_unc*100:.1f}% to {sample}")

    return hist

#### Get Histograms

# Load conversion scale factors if not excluded
# ConvSF is applied to all channels (maps to corresponding ZG channel)
# For channels without ZG measurement (e.g., TTZ2E1Mu), uses default SF = 1.0 ± 20%
conv_sf = {}
if args.exclude != "ConvSF":
    conv_sf = load_conv_scale_factors(args.channel, era_list)

# Step 1: Load histograms from each era
era_data_hists = []
era_mc_hists = {sample: [] for sample in MCList}
era_nonprompt_hists = {sample: [] for sample in nonprompt}
eras_with_data = []
eras_without_data = []

for era in era_list:
    
    # Load data for this era
    era_data = []
    for sample in ERA_SAMPLES[era]["data"]:
        file_path = f"{WORKDIR}/SKNanoOutput/{ANALYZER}/{FLAG}/{era}/Skim_TriLep_{sample}.root"
        hist_path = f"{args.channel}/Central/{args.histkey}"
        
        h = load_histogram(file_path, hist_path, era, missing_logger)
        if h:
            era_data.append(h)
    
    # Sum data for this era and track which eras have data
    if era_data:
        era_data_sum = sum_histograms(era_data, f"data_{era}")
        era_data_hists.append(era_data_sum)
        eras_with_data.append(era)
    else:
        eras_without_data.append(era)

# Step 2: Sum data histograms across eras
data = sum_histograms(era_data_hists, "data_total")
if data:
    data.SetTitle("Data")

# Check if we have any valid data
if data is None:
    logging.error(f"No valid data histograms found for {args.histkey} in any of the eras: {era_list}")
    logging.error("Cannot proceed with plotting without data. Exiting...")
    exit(1)
else:
    # Report which eras contributed data
    if eras_without_data:
        logging.warning(f"Data for {args.histkey} completely missing in eras: {eras_without_data}")

# Load nonprompt samples from each era
HISTs = {}
for era in era_list:
    # Load nonprompt for this era
    for sample in ERA_SAMPLES[era]["nonprompt"]:
        file_path = f"{WORKDIR}/SKNanoOutput/{ANALYZER.replace('Prompt', 'Matrix')}/{FLAG}/{era}/Skim_TriLep_{sample}.root"
        hist_path = f"{args.channel}/Central/{args.histkey}"
        
        h = load_histogram(file_path, hist_path, era, missing_logger)
        if h:
            # Set 30% systematic uncertainty for nonprompt
            for bin in range(h.GetNcells()):
                h.SetBinError(bin, h.GetBinContent(bin) * 0.3)
            era_nonprompt_hists[sample].append(h)

    # Load MC for this era
    all_era_samples = ERA_SAMPLES[era]["conv"] + ERA_SAMPLES[era]["ttX"] + ERA_SAMPLES[era]["diboson"] + ERA_SAMPLES[era]["others"]
    for sample in all_era_samples:
        file_path = f"{WORKDIR}/SKNanoOutput/{ANALYZER}/{FLAG}_RunSyst/{era}/Skim_TriLep_{sample}.root"
        if args.exclude == "WZSF" and ("WZTo3LNu" in sample or "ZZTo4L" in sample):
            file_path = f"{WORKDIR}/SKNanoOutput/{ANALYZER}/{FLAG}_RunNoWZSF_RunSyst/{era}/Skim_TriLep_{sample}.root"
        
        hist_path = f"{args.channel}/Central/{args.histkey}"
        h = load_histogram(file_path, hist_path, era, missing_logger)
        if h:
            # Apply K-factor before systematics
            h = apply_kfactor(h, sample, RUN)
            h = calculate_systematics(h, ERA_SYSTEMATICS[era], file_path, args, era, missing_logger)
            # Apply WZ uncertainty when WZSF is applied
            h = apply_wz_uncertainty(h, sample, args)
            # Apply conversion scale factor
            h = apply_conv_scale_factor(h, sample, era, conv_sf, args.channel, ERA_SAMPLES)
            era_mc_hists[sample].append(h)

# Step 3: Sum histograms across eras

# Sum nonprompt samples
for sample in nonprompt:
    if era_nonprompt_hists[sample]:
        HISTs[sample] = sum_histograms(era_nonprompt_hists[sample], f"{sample}_total")

# Sum MC samples
for sample in MCList:
    if era_mc_hists[sample]:
        HISTs[sample] = sum_histograms(era_mc_hists[sample], f"{sample}_total")

# Check the final MC histograms
valid_mc_samples = 0
for sample in MCList + list(nonprompt):
    if sample in HISTs and HISTs[sample] is not None:
        valid_mc_samples += 1
    else:
        logging.debug(f"No histograms found for sample {sample}")

# Check if we have at least some MC samples
if valid_mc_samples == 0:
    logging.error("No valid MC histograms found for any sample!")
    logging.error("Cannot proceed with plotting without any MC. Exiting...")
    exit(1)
#### Merge backgrounds
def add_hist(name, hist, histDict):
    # content of dictionary should be initialized as "None"
    if not histDict[name]:
        histDict[name] = hist.Clone(name)
    else:
        histDict[name].Add(hist)
    
temp_dict = { "nonprompt": None, "conv": None, "ttX": None, "diboson": None, "others": None }
for sample in nonprompt:
    if not sample in HISTs.keys(): continue
    add_hist("nonprompt", HISTs[sample], temp_dict)
for sample in conv:
    if not sample in HISTs.keys(): continue
    add_hist("conv", HISTs[sample], temp_dict)
for sample in ttX:
    if not sample in HISTs.keys(): continue
    add_hist("ttX", HISTs[sample], temp_dict)
for sample in diboson:
    if not sample in HISTs.keys(): continue
    add_hist("diboson", HISTs[sample], temp_dict)
for sample in others:
    if not sample in HISTs.keys(): continue
    add_hist("others", HISTs[sample], temp_dict)

# filter out none histograms from temp_dict
BKGs = {name: hist for name, hist in temp_dict.items() if hist}
# Sort BKGs by hist.Integral()
BKGs = dict(sorted(BKGs.items(), key=lambda x: x[1].Integral(), reverse=True))

# Note: Blinding is now handled in ComparisonCanvas to ensure
# data and systematics are guaranteed to be identical

# Load signal histograms (only for SR1E2Mu and SR3Mu)
SIGNALs = {}
if args.channel in ["SR1E2Mu", "SR3Mu"]:
    for signal_mass in args.signals:
        signal_name = f"TTToHcToWAToMuMu-{signal_mass}"
        signal_hist = None

        for era in era_list:
            path = f"{WORKDIR}/SKNanoOutput/{ANALYZER}/{FLAG}_RunSyst/{era}/{signal_name}.root"
            if not os.path.exists(path):
                logging.debug(f"Signal file not found: {path}")
                continue

            f = ROOT.TFile.Open(path)
            h = f.Get(f"{args.channel}/Central/{args.histkey}")
            if not h:
                logging.debug(f"Signal histogram not found: {args.channel}/Central/{args.histkey} in {path}")
                f.Close()
                continue

            h.SetDirectory(0)
            h.Scale(args.signal_scale)

            if signal_hist is None:
                signal_hist = h.Clone(signal_mass)
                signal_hist.SetDirectory(0)
            else:
                signal_hist.Add(h)
            f.Close()

        if signal_hist:
            SIGNALs[signal_mass] = signal_hist
    # For ParticleNet score plots, keep only matching signal
    if "score" in args.histkey:
        # Extract mass point from histkey (e.g., "MHc160_MA155/score_diboson" -> "MHc160_MA155")
        mass_point = args.histkey.split("/")[0]
        # Keep only the matching signal
        SIGNALs = {k: v for k, v in SIGNALs.items() if k == mass_point}
        if not SIGNALs:
            logging.warning(f"ParticleNet score plot for {mass_point}, but no matching signal histogram found")

plotter = ComparisonCanvas(data, BKGs, config)
plotter.drawPadUp()
if SIGNALs:
    plotter.drawSignals(SIGNALs)
plotter.drawPadDown()
plotter.canv.SaveAs(OUTPUTPATH)
