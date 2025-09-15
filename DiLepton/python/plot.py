#!/usr/bin/env python
import os
import argparse
import logging
import json
import ROOT
from math import pow, sqrt
from plotter import ComparisonCanvas, get_era_list, get_CoM_energy
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
elif args.channel == "EMU":
    FLAG = "RunEMu"
else:
    raise ValueError(f"Invalid channel: {args.channel}")

def load_era_configs():
    """Load sample groups and systematics for all relevant eras"""
    samplegroup_config = json.load(open("configs/samplegroup.json"))
    systematics_config = json.load(open("configs/systematics.json"))
    
    era_samples = {}
    era_systematics = {}
    
    for era in era_list:
        era_samples[era] = samplegroup_config[era][args.channel]
        era_systematics[era] = systematics_config[era][args.channel]
    
    return era_samples, era_systematics

def get_sample_lists(era_samples):
    """Extract and organize sample lists"""
    # All data samples (different names per era)
    data_samples = []
    for era in era_list:
        data_samples.extend(era_samples[era]["data"])
    
    # Unique MC samples by category
    mc_categories = {"W": set(), "Z": set(), "TT": set(), "ST": set(), "VV": set()}
    for era in era_list:
        for category in mc_categories:
            mc_categories[category].update(era_samples[era][category])
    
    # Convert to lists and create full MC list
    mc_lists = {cat: list(samples) for cat, samples in mc_categories.items()}
    all_mc_samples = sum(mc_lists.values(), [])
    
    return data_samples, mc_lists, all_mc_samples

def merge_systematics(era_systematics):
    """Merge systematics from all eras"""
    all_systs = []
    for era_systs in era_systematics.values():
        for syst in era_systs:
            if syst not in all_systs:
                all_systs.append(syst)
    return all_systs

# Load configurations
logging.debug(f"Processing {args.era} with histkey: {args.histkey}")
logging.debug(f"Era list: {era_list}")

ERA_SAMPLES, ERA_SYSTEMATICS = load_era_configs()
DATAPERIODs, MC_CATEGORIES, MCList = get_sample_lists(ERA_SAMPLES)
SYSTs = merge_systematics(ERA_SYSTEMATICS)

# Unpack MC categories for backward compatibility
W, Z, TT, ST, VV = MC_CATEGORIES["W"], MC_CATEGORIES["Z"], MC_CATEGORIES["TT"], MC_CATEGORIES["ST"], MC_CATEGORIES["VV"]

logging.debug(f"Eras: {era_list}")
logging.debug(f"MCList: {MCList}")
logging.debug(f"Data samples: {len(DATAPERIODs)}")

if args.exclude:
    OUTPUTPATH = f"{WORKDIR}/DiLepton/plots/{args.era}/{args.channel}/No{args.exclude}/{args.histkey.replace('/', '_')}.png"
else:
    OUTPUTPATH = f"{WORKDIR}/DiLepton/plots/{args.era}/{args.channel}/Central/{args.histkey.replace('/', '_')}.png"

os.makedirs(os.path.dirname(OUTPUTPATH), exist_ok=True)

def load_histogram(file_path, hist_path):
    """Load a single histogram from a ROOT file"""
    if not os.path.exists(file_path):
        logging.debug(f"File not found: {file_path}")
        return None
    
    try:
        f = ROOT.TFile.Open(file_path)
        if not f or f.IsZombie():
            if f: f.Close()
            logging.debug(f"Cannot open file: {file_path}")
            return None
        
        h = f.Get(hist_path)
        if h and h.GetEntries() >= 0:
            h.SetDirectory(0)
            f.Close()
            logging.debug(f"Successfully loaded {hist_path} from {os.path.basename(file_path)}")
            return h
        else:
            f.Close()
            logging.debug(f"Histogram {hist_path} not found in {os.path.basename(file_path)}")
            return None
                    
    except Exception as e:
        logging.debug(f"Error loading histogram: {e}")
        return None

def calculate_systematics(h, systematics, file_path, args):
    """Calculate systematic uncertainties for a histogram"""
    if args.exclude:
        return h
    
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        return h
    
    try:
        hSysts = []
        for syst, sources in systematics.items():
            syst_up, syst_down = tuple(sources)
            h_up = f.Get(f"{args.channel}/{syst_up}/{args.histkey}")
            h_down = f.Get(f"{args.channel}/{syst_down}/{args.histkey}")
            if h_up and h_down:
                h_up.SetDirectory(0)
                h_down.SetDirectory(0)
                hSysts.append((h_up, h_down))
        
        # Apply systematic uncertainties bin by bin
        for bin in range(h.GetNcells()):
            stat_unc = h.GetBinError(bin)
            envelops = []
            for h_up, h_down in hSysts:
                systUp = abs(h_up.GetBinContent(bin) - h.GetBinContent(bin))
                systDown = abs(h_down.GetBinContent(bin) - h.GetBinContent(bin))
                envelops.append(max(systUp, systDown))
            total_unc = sqrt(pow(stat_unc, 2) + sum([pow(x, 2) for x in envelops]))
            h.SetBinError(bin, total_unc)
    
    finally:
        f.Close()
    
    return h

def sum_histograms(hist_list, name):
    """Sum a list of histograms"""
    if not hist_list:
        return None
    
    total = hist_list[0].Clone(name)
    total.SetDirectory(0)
    for h in hist_list[1:]:
        total.Add(h)
    return total

#### Get Histograms
logging.debug("Loading histograms...")

# Step 1: Load histograms from each era
era_data_hists = []
era_mc_hists = {sample: [] for sample in MCList}
eras_with_data = []
eras_without_data = []

for era in era_list:
    logging.debug(f"Processing era: {era}")
    
    # Load data for this era
    era_data = []
    for sample in ERA_SAMPLES[era]["data"]:
        file_path = f"{WORKDIR}/SKNanoOutput/DiLepton/{FLAG}_RunSyst/{era}/{sample}.root"
        # Data always uses Central path - systematics only apply to MC
        hist_path = f"{args.channel}/Central/{args.histkey}"
        
        h = load_histogram(file_path, hist_path)
        if h:
            era_data.append(h)
            logging.debug(f"Loaded data {sample} from {era} (entries: {h.GetEntries()})")
        else:
            logging.debug(f"Histogram {args.histkey} not found in {sample} from {era}")
    
    # Sum data for this era and track which eras have data
    if era_data:
        era_data_sum = sum_histograms(era_data, f"data_{era}")
        era_data_hists.append(era_data_sum)
        eras_with_data.append(era)
        logging.debug(f"Loaded data for {args.histkey} from {len(era_data)}/{len(ERA_SAMPLES[era]['data'])} files in era {era}")
    else:
        eras_without_data.append(era)
        logging.debug(f"No data found for {args.histkey} in any files of era {era}")
    
    # Load MC for this era
    all_era_samples = sum([ERA_SAMPLES[era][cat] for cat in ["W", "Z", "TT", "ST", "VV"]], [])
    for sample in all_era_samples:
        file_path = f"{WORKDIR}/SKNanoOutput/DiLepton/{FLAG}_RunSyst/{era}/{sample}.root"
        hist_path = f"{args.channel}/Central/{args.histkey}"
        if args.exclude:
            hist_path = f"{args.channel}/{args.exclude}_NotImplemented/{args.histkey}"
        
        h = load_histogram(file_path, hist_path)
        if h:
            h = calculate_systematics(h, ERA_SYSTEMATICS[era], file_path, args)
            era_mc_hists[sample].append(h)
            logging.debug(f"Loaded MC {sample} from {era}")

# Step 2: Sum histograms across eras
logging.debug("Summing histograms across eras...")

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
    logging.debug(f"Successfully loaded data histogram with {data.GetEntries()} entries")
    # Report which eras contributed data
    if eras_without_data:
        logging.warning(f"Data for {args.histkey} completely missing in eras: {eras_without_data}")
    if len(eras_with_data) == len(era_list):
        logging.debug(f"Data loaded successfully from all eras: {eras_with_data}")
    else:
        logging.debug(f"Data loaded from {len(eras_with_data)}/{len(era_list)} eras: {eras_with_data}")

# Check the final MC histograms
valid_mc_samples = 0
for sample in MCList:
    if sample in HISTs and HISTs[sample] is not None:
        valid_mc_samples += 1
        logging.debug(f"Successfully loaded {sample} with {HISTs[sample].GetEntries()} entries")
    else:
        logging.warning(f"No histograms found for sample {sample}")

# Check if we have at least some MC samples
if valid_mc_samples == 0:
    logging.error("No valid MC histograms found for any sample!")
    logging.error("Cannot proceed with plotting without any MC. Exiting...")
    exit(1)
else:
    logging.debug(f"Successfully loaded {valid_mc_samples} out of {len(MCList)} MC samples")

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
logging.debug(f"BKGs: {BKGs}")

plotter = ComparisonCanvas(data, BKGs, config)
plotter.drawPadUp()
plotter.drawPadDown()
plotter.canv.SaveAs(OUTPUTPATH)
