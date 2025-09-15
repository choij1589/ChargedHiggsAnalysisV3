#!/usr/bin/env python
import os
import argparse
import logging
import json
import ROOT
from plotter import KinematicCanvas
ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--object", required=True, type=str, help="object")
parser.add_argument("--region", required=True, type=str, help="region")
parser.add_argument("--histkey", required=True, type=str, help="histkey")
parser.add_argument("--ptbin", default=None, type=str, help="pt bin (e.g., pt15to20, pt70toInf)")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
WORKDIR = os.getenv("WORKDIR")

with open("configs/histkeys.json") as f:
    config = json.load(f)[args.histkey]
config["era"] = args.era
config["maxDigits"] = 3

if args.era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
    RUN = "Run2"
    config["CoM"] = 13
elif args.era in ["2022", "2022EE", "2023", "2023BPix"]:
    RUN = "Run3"
    config["CoM"] = 13.6
else:
    raise ValueError(f"Invalid era: {args.era}")
config["overflow"] = True
config["normalize"] = True

##### Get Histograms
HISTs = {}
lepTypes = ["prompt", "conv", "fromTau", "fromC", "fromB", "fromL", "fromPU"]

# Determine histogram path based on whether pt-binning is requested
if args.ptbin:
    histkey_path = f"{args.histkey}_{args.ptbin}"
    plot_suffix = f"_{args.ptbin}"
else:
    histkey_path = args.histkey
    plot_suffix = ""

f = ROOT.TFile(f"{WORKDIR}/LeptonIDTest/histograms/{args.object}.{args.era}.root")
for lepType in lepTypes:
    hist = f.Get(f"{args.region}/{lepType}/{histkey_path}")
    if hist:
        hist.SetDirectory(0)
        HISTs[lepType] = hist
    else:
        logging.warning(f"Histogram {args.region}/{lepType}/{histkey_path} not found")
f.Close()

##### Plot
if HISTs:  # Only plot if we have histograms
    canvas = KinematicCanvas(HISTs, config)
    canvas.drawPad()
    
    # Organize output path based on pt binning
    if args.ptbin:
        outpath = f"{WORKDIR}/LeptonIDTest/plots/{args.era}/{args.object}/idvar/{args.region}/{args.ptbin}/{args.histkey}.png"
    else:
        outpath = f"{WORKDIR}/LeptonIDTest/plots/{args.era}/{args.object}/idvar/{args.region}/{args.histkey}.png"
    
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    canvas.canv.SaveAs(outpath)
    print(f"Plot saved to {outpath}")
else:
    print("No histograms found to plot")
