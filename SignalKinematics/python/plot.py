#!/usr/bin/env python
import os
import argparse
import logging
import json
import ROOT
from plotter import KinematicCanvas, get_era_list, get_CoM_energy, PALETTE_LONG
from HistoUtils import setup_missing_histogram_logging, load_histogram, sum_histograms
ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel (SR3Mu or SR1E2Mu)")
parser.add_argument("--histkey", required=True, type=str, help="histogram key from histkeys.json")
parser.add_argument("--sample-list", default="default", choices=["default", "validation", "full"],
                    help="which sample list to use (default: default)")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

WORKDIR = os.environ['WORKDIR']

# Setup missing histogram logging
missing_logger = setup_missing_histogram_logging(args)

# Load histogram config
with open("configs/histkeys.json") as f:
    histkeys = json.load(f)

if args.histkey not in histkeys:
    logging.error(f"histkey '{args.histkey}' not found in configs/histkeys.json")
    exit(1)

config = histkeys[args.histkey]
config["era"] = args.era
config["CoM"] = get_CoM_energy(args.era)

# Determine run key (for samplelist.json lookup)
if args.era in ["2016preVFP", "2016postVFP", "2017", "2018", "Run2"]:
    RUN = "Run2"
elif args.era in ["2022", "2022EE", "2023", "2023BPix", "Run3"]:
    RUN = "Run3"
else:
    raise ValueError(f"Invalid era: {args.era}")

# Determine run flag (for file path)
if args.channel == "SR3Mu":
    FLAG = "Run3Mu"
elif args.channel == "SR1E2Mu":
    FLAG = "Run1E2Mu"
else:
    raise ValueError(f"Invalid channel: {args.channel}")

# Load sample list
with open("configs/samplelist.json") as f:
    samplelist_config = json.load(f)

mass_points = samplelist_config[RUN][args.sample_list]
if not mass_points:
    logging.error(f"No mass points in {RUN}/{args.sample_list} sample list")
    exit(1)

# Get list of individual eras to merge
era_list = get_era_list(args.era)

# Determine histogram path within ROOT file
if args.histkey.startswith("GenLevel/"):
    hist_path = args.histkey
else:
    hist_path = f"{args.channel}/Central/{args.histkey}"

# Load and merge histograms for each mass point
HISTs = {}
for mass_point in mass_points:
    era_hists = []
    for era in era_list:
        file_path = f"{WORKDIR}/SKNanoOutput/SignalKinematics/{FLAG}/{era}/TTToHcToWAToMuMu-{mass_point}.root"
        h = load_histogram(file_path, hist_path, era, missing_logger)
        if h:
            era_hists.append(h)

    if not era_hists:
        logging.debug(f"No data found for mass point {mass_point}, skipping")
        continue

    h_total = sum_histograms(era_hists, mass_point)
    if h_total:
        HISTs[mass_point] = h_total

if not HISTs:
    logging.warning(f"No histograms found for {args.histkey} in {args.era}/{args.channel}, skipping")
    exit(0)

logging.info(f"Loaded {len(HISTs)} mass points for {args.histkey}")

# Output path
if args.histkey.startswith("GenLevel/"):
    subdir = "GEN"
elif args.histkey.startswith("GenMatched/"):
    subdir = "RECO-GENMATCHED"
else:
    subdir = "RECO"
out_name = args.histkey.replace("/", "_")
OUTPUTPATH = f"plots/{args.era}/{args.channel}/{subdir}/{out_name}.png"
os.makedirs(os.path.dirname(OUTPUTPATH), exist_ok=True)

# Scale legend height to number of entries with a compact text size
n_hists = len(HISTs)
leg_text_size = 0.035
leg_row_height = 0.045
config["legend"] = (0.55, 0.89 - leg_row_height * n_hists, 0.99, 0.89, leg_text_size, 1)

# KinematicCanvas palette supports at most len(PALETTE_LONG) colors; truncate if needed
max_hists = len(PALETTE_LONG)
if len(HISTs) > max_hists:
    logging.warning(f"Too many mass points ({len(HISTs)}) for palette size ({max_hists}); truncating to first {max_hists}")
    HISTs = dict(list(HISTs.items())[:max_hists])

# Draw
plotter = KinematicCanvas(HISTs, config)
plotter.leg.SetTextFont(42)
plotter.drawPad()
plotter.canv.SaveAs(OUTPUTPATH)
logging.info(f"Saved: {OUTPUTPATH}")
