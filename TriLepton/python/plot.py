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
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

WORKDIR = os.environ['WORKDIR']

# Setup missing histogram logging
missing_logger = setup_missing_histogram_logging(args)

with open("configs/histkeys.json") as f:
    config = json.load(f)[args.histkey]
config["era"] = args.era
config["CoM"] = get_CoM_energy(args.era)
config["rTitle"] = "Data / Pred"
config["maxDigits"] = 3
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
if args.channel not in ["SR1E2Mu", "SR3Mu", "ZFake1E2Mu", "ZFake3Mu", "ZG1E2Mu", "ZG3Mu", "WZ1E2Mu", "WZ3Mu"]:
    raise ValueError(f"Invalid channel: {args.channel}")

ANALYZER = ""
if "SR" in args.channel or "ZFake" in args.channel:
    ANALYZER = "PromptSelector"
if "ZG" in args.channel or "WZ" in args.channel:
    ANALYZER = "CRPromptSelector"

if "1E2Mu" in args.channel:
    FLAG = "Run1E2Mu"
    channel_flag = "1E2Mu"
if "3Mu" in args.channel:
    FLAG = "Run3Mu"
    channel_flag = "3Mu"

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


def load_conv_scale_factors(channel, era_list):
    """Load conversion scale factors for the channel"""
    conv_sf = {}
    
    # Only apply to ZG channels
    if "ZG" not in channel:
        return conv_sf
    
    # Get the ZG channel name for loading SF
    zg_channel = channel  # Channel is already ZG1E2Mu or ZG3Mu
    
    for era in era_list:
        sf_file = f"{WORKDIR}/TriLepton/results/{zg_channel}/{era}/ConvSF.json"
        
        if not os.path.exists(sf_file):
            logging.warning(f"Conversion SF file not found: {sf_file}")
            continue
            
        try:
            cset = correctionlib.CorrectionSet.from_file(sf_file)
            conv_sf[era] = cset
        except Exception as e:
            logging.warning(f"Failed to load conversion SF from {sf_file}: {e}")
    
    return conv_sf

def apply_conv_scale_factor(hist, sample, era, conv_sf, channel, era_samples):
    """Apply conversion scale factor to conversion samples"""
    if args.exclude == "ConvSF":
        return hist
    
    if not conv_sf or era not in conv_sf:
        return hist
    
    # Check if this sample is in the conv list for this era
    # era_samples[era] already contains the channel-specific data
    is_conv_sample = sample in era_samples[era]["conv"]
    if not is_conv_sample:
        return hist
    
    try:
        # Get central scale factor
        sf_name = f"ConvSF_{channel}_{era}_Central"
        sf_central = conv_sf[era][sf_name].evaluate()
        
        # Apply scale factor
        hist.Scale(sf_central)
        
        # Calculate systematic uncertainty
        # Get prompt and nonprompt uncertainties
        sf_prompt_up = conv_sf[era][f"ConvSF_{channel}_{era}_prompt_up"].evaluate()
        sf_prompt_down = conv_sf[era][f"ConvSF_{channel}_{era}_prompt_down"].evaluate()
        sf_nonprompt_up = conv_sf[era][f"ConvSF_{channel}_{era}_nonprompt_up"].evaluate()
        sf_nonprompt_down = conv_sf[era][f"ConvSF_{channel}_{era}_nonprompt_down"].evaluate()
        
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

#### Get Histograms

# Load conversion scale factors if not excluded
conv_sf = {}
if args.exclude != "ConvSF" and "ZG" in args.channel:
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

if args.blind:
    data.Scale(0.)

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
        logging.warning(f"No histograms found for sample {sample}")

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

#print(f"data: {data.Integral()}")
#prediction = 0.
#for name, hist in BKGs.items():
#    print(f"{name}: {hist.Integral()} ({hist.Integral()/data.Integral()*100:.1f}%)")
#    prediction += hist.Integral()
#print(f"prediction: {prediction}")

plotter = ComparisonCanvas(data, BKGs, config)
plotter.drawPadUp()
plotter.drawPadDown()
plotter.canv.SaveAs(OUTPUTPATH)
