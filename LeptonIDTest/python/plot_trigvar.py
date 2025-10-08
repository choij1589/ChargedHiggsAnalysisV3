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
parser.add_argument("--histkey", required=True, type=str, help="histkey")
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

#### Get Histograms
HISTs = {}
REGIONs = ["InnerBarrel", "OuterBarrel", "Endcap"]

f = ROOT.TFile(f"{WORKDIR}/LeptonIDTest/histograms/{args.object}.{args.era}.root")
for region in REGIONs:
    hist = f.Get(f"{region}/{args.histkey}")
    hist.SetDirectory(0)
    HISTs[region] = hist
f.Close()

##### Plot
canvas = KinematicCanvas(HISTs, config)
canvas.drawPad()
outpath = f"{WORKDIR}/LeptonIDTest/plots/{args.era}/{args.object}/trigvar/{args.histkey}.png"
os.makedirs(os.path.dirname(outpath), exist_ok=True)
canvas.canv.SaveAs(outpath)
