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
ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--histkey", required=True, type=str, help="histkey")
parser.add_argument("--exclude", default=None, type=str, help="exclude weight")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

WORKDIR = os.environ['WORKDIR']

# Setup missing histogram logging
missing_logger = setup_missing_histogram_logging(args)

with open("configs/histkeys.json") as f:
    config = json.load(f)[args.histkey]
config["era"] = args.era
config["rTitle"] = "Data / Pred"
config["maxDigits"] = 3
config["CoM"] = get_CoM_energy(args.era)

#### Configurations
# Get era list for merging
era_list = get_era_list(args.era)

if args.era in ["2016preVFP", "2016postVFP", "2017", "2018", "Run2"]:
    RUN = "Run2"
elif args.era in ["2022", "2022EE", "2023", "2023BPix", "Run3"]:
    RUN = "Run3"
else:
    raise ValueError(f"Invalid era: {args.era}")

if args.channel == "DIMU":
    FLAG = "RunDiMu"
    config["channel"] = "DiMu"
elif args.channel == "EMU":
    FLAG = "RunEMu"
    config["channel"] = "EMu"
else:
    raise ValueError(f"Invalid channel: {args.channel}")

# Load configurations
ERA_SAMPLES, ERA_SYSTEMATICS = load_era_configs(args, era_list)
DATAPERIODs, MC_CATEGORIES, MCList = get_sample_lists(ERA_SAMPLES, ["W", "Z", "TT", "ST", "VV"])
SYSTs = merge_systematics(ERA_SYSTEMATICS)

# Unpack MC categories for backward compatibility
W, Z, TT, ST, VV = MC_CATEGORIES["W"], MC_CATEGORIES["Z"], MC_CATEGORIES["TT"], MC_CATEGORIES["ST"], MC_CATEGORIES["VV"]

if args.exclude:
    OUTPUTPATH = f"{WORKDIR}/DiLepton/plots/{args.era}/{args.channel}/No{args.exclude}/{args.histkey.replace('/', '_')}.png"
else:
    OUTPUTPATH = f"{WORKDIR}/DiLepton/plots/{args.era}/{args.channel}/Central/{args.histkey.replace('/', '_')}.png"

os.makedirs(os.path.dirname(OUTPUTPATH), exist_ok=True)


#### Get Histograms
# Step 1: Load histograms from each era
era_data_hists = []
era_mc_hists = {sample: [] for sample in MCList}
eras_with_data = []
eras_without_data = []

for era in era_list:
    # Load data for this era
    era_data = []
    for sample in ERA_SAMPLES[era]["data"]:
        file_path = f"{WORKDIR}/SKNanoOutput/DiLepton/{FLAG}_RunSyst/{era}/{sample}.root"
        # Data always uses Central path - systematics only apply to MC
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
    
    # Load MC for this era
    all_era_samples = sum([ERA_SAMPLES[era][cat] for cat in ["W", "Z", "TT", "ST", "VV"]], [])
    for sample in all_era_samples:
        file_path = f"{WORKDIR}/SKNanoOutput/DiLepton/{FLAG}_RunSyst/{era}/{sample}.root"
        hist_path = f"{args.channel}/Central/{args.histkey}"
        if args.exclude:
            hist_path = f"{args.channel}/{args.exclude}_NotImplemented/{args.histkey}"
        
        h = load_histogram(file_path, hist_path, era, missing_logger)
        if h:
            h = calculate_systematics(h, ERA_SYSTEMATICS[era], file_path, args, era, missing_logger)
            era_mc_hists[sample].append(h)

# Step 2: Sum histograms across eras

data = sum_histograms(era_data_hists, "data_total")
if data:
    data.SetTitle("Data")
HISTs = {}
for sample in MCList:
    if era_mc_hists[sample]:
        HISTs[sample] = sum_histograms(era_mc_hists[sample], f"{sample}_total")

# Check if we have any valid data (only after processing all eras)
if data is None:
    logging.error(f"No valid data histograms found for {args.histkey} in any of the eras: {era_list}")
    logging.error("Cannot proceed with plotting without data. Exiting...")
    exit(1)
else:
    # Report which eras contributed data
    if eras_without_data:
        logging.warning(f"Data for {args.histkey} completely missing in eras: {eras_without_data}")

# Check the final MC histograms
valid_mc_samples = 0
for sample in MCList:
    if sample in HISTs and HISTs[sample] is not None:
        valid_mc_samples += 1
    else:
        logging.warning(f"No histograms found for sample {sample}")

# Check if we have at least some MC samples
if valid_mc_samples == 0:
    logging.error("No valid MC histograms found for any sample!")
    logging.error("Cannot proceed with plotting without any MC. Exiting...")
    exit(1)

#### merge background
def add_hist(name, hist, histDict):
    # content of dictionary should be initialized as "None"
    if not histDict[name]:
        histDict[name] = hist.Clone(name)
    else:
        histDict[name].Add(hist)

temp_dict = { "W": None, "Z": None, "TT": None, "ST": None, "VV": None }
for sample in W:
    if not sample in HISTs.keys(): continue
    add_hist("W", HISTs[sample], temp_dict)
for sample in Z:
    if not sample in HISTs.keys(): continue
    add_hist("Z", HISTs[sample], temp_dict)
for sample in TT:
    if not sample in HISTs.keys(): continue
    add_hist("TT", HISTs[sample], temp_dict)
for sample in VV:
    if not sample in HISTs.keys(): continue
    add_hist("VV", HISTs[sample], temp_dict)
for sample in ST:
    if not sample in HISTs.keys(): continue
    add_hist("ST", HISTs[sample], temp_dict)

# filter out none historgrams from temp_dict
BKGs = {name: hist for name, hist in temp_dict.items() if hist}
# Sort BKGs by hist.Integral()
BKGs = dict(sorted(BKGs.items(), key=lambda x: x[1].Integral(), reverse=True))

plotter = ComparisonCanvas(data, BKGs, config)
plotter.drawPadUp()
plotter.drawPadDown()
plotter.canv.SaveAs(OUTPUTPATH)
