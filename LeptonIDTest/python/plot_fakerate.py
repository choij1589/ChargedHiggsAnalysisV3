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
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
WORKDIR = os.getenv("WORKDIR")

ptcorr_bins = []
if args.object == "muon":
    ptcorr_bins = [10., 15., 20., 30., 50., 70.]
elif args.object == "electron":
    ptcorr_bins = [10., 15., 20., 25., 35., 50., 70.]
else:
    raise ValueError(f"Invalid object: {args.object}")

config = {
    "xTitle": "p_{T}^{corr} [GeV]",
    "yTitle": "Fake rate",
    "xRange": [ptcorr_bins[0], ptcorr_bins[-1]],
    "yRange": [0.0, 1.0],
    "era": args.era,
    "maxDigits": 3,
    "overflow": False,
}

if args.era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
    RUN = "Run2"
    config["CoM"] = 13
elif args.era in ["2022", "2022EE", "2023", "2023BPix"]:
    RUN = "Run3"
    config["CoM"] = 13.6
else:
    raise ValueError(f"Invalid era: {args.era}")

##### Get Histograms
if args.object == "muon":
    lepTypes = ["fromC", "fromB"]
elif args.object == "electron":
    lepTypes = ["fromC", "fromB", "fromL"]
else:
    raise ValueError(f"Invalid object: {args.object}")
regions = ["InnerBarrel", "OuterBarrel", "Endcap"]

# Convert ptcorr_bins to array for ROOT
import array
ptcorr_bins_array = array.array('d', ptcorr_bins)

f = ROOT.TFile(f"{WORKDIR}/LeptonIDTest/histograms/{args.object}.{args.era}.root")

for region in regions:
    HISTs = {}
    for lepType in lepTypes:
        hist_loose = f.Get(f"{region}/{lepType}/passLooseID")
        hist_tight = f.Get(f"{region}/{lepType}/passTightID")
        hist_loose_rebinned = hist_loose.Rebin(len(ptcorr_bins)-1, f"{hist_loose.GetName()}_rebinned", ptcorr_bins_array)
        hist_tight_rebinned = hist_tight.Rebin(len(ptcorr_bins)-1, f"{hist_tight.GetName()}_rebinned", ptcorr_bins_array)
    
        # Calculate fake rate: tight / loose
        hist_fakerate = hist_tight_rebinned.Clone(f"fakerate_{lepType}_{region}")
        hist_fakerate.Divide(hist_loose_rebinned)
        hist_fakerate.SetDirectory(0)
    
        # Store in HISTs dictionary
        HISTs[lepType] = hist_fakerate
    canvas = KinematicCanvas(HISTs, config)
    canvas.drawPad()
    outpath = f"{WORKDIR}/LeptonIDTest/plots/{args.era}/{args.object}/fakerate/{region}.png"
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    canvas.canv.SaveAs(outpath)
    del canvas

f.Close()