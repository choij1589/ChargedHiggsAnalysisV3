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
parser.add_argument("--region", required=True, type=str, help="region (InnerBarrel, OuterBarrel, Endcap)")
parser.add_argument("--plotvar", required=True, type=str, help="plotting variable, registered in histkeys.json")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
WORKDIR = os.getenv("WORKDIR")

with open("configs/efficiency.json") as f:
    config = json.load(f)[args.plotvar]
config["era"] = args.era
config["maxDigits"] = 3
config["yTitle"] = "Efficiency"
config["overflow"] = False

if args.era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
    RUN = "Run2"
    config["CoM"] = 13
elif args.era in ["2022", "2022EE", "2023", "2023BPix"]:
    RUN = "Run3"
    config["CoM"] = 13.6
else:
    raise ValueError(f"Invalid era: {args.era}")

##### Get Histograms
HISTs = {}
lepTypes = ["prompt", "conv", "fromTau", "fromC", "fromB", "fromL"]
if args.object == "muon":
    histname = "miniIso_sip3d"
elif args.object == "electron":
    histname = "miniIso_sip3d_mvaNoIso"
else:
    raise ValueError(f"Invalid object: {args.object}")

f = ROOT.TFile(f"{WORKDIR}/LeptonIDTest/histograms/{args.object}.{args.era}.root")
for lepType in lepTypes:
    hist = f.Get(f"{args.region}/{lepType}/{histname}")
    if not hist:
        print(f"Warning: Could not find histogram {args.region}/{lepType}/{histname}")
        continue
        
    # Clone to detach from file
    hist = hist.Clone(f"{lepType}_{args.plotvar}")
    hist.SetDirectory(0)  # Detach from file
    
    # Project on the plotting variable
    # For muon, the plot name is miniIso_sip3d
    # For electron, the plot name is miniIso_sip3d_mvaNoIso
    if args.plotvar == "miniIso":
        proj_hist = hist.ProjectionX(f"{lepType}_{args.plotvar}")
    elif args.plotvar == "sip3d":
        proj_hist = hist.ProjectionY(f"{lepType}_{args.plotvar}")
    elif args.plotvar == "mvaNoIso":
        proj_hist = hist.ProjectionZ(f"{lepType}_{args.plotvar}")
    else:
        raise ValueError(f"Invalid plotvar: {args.plotvar}")

    # Normalize
    proj_hist.Scale(1.0/proj_hist.Integral())
    proj_hist.SetDirectory(0)  

    # Make CDF
    cdf = proj_hist.GetCumulative()
    cdf.SetDirectory(0)  
    
    # Convert CDF to efficiency based on variable type
    eff_hist = cdf.Clone(f"{lepType}_efficiency")
    eff_hist.SetDirectory(0)
    
    # Determine cut direction based on variable
    # For sip3d and miniIso: efficiency = CDF (smaller than cut)
    # For mvaNoIso: efficiency = 1 - CDF (greater than cut)
    if args.plotvar in ["sip3d", "miniIso"]:
        # "Smaller than" efficiency - use CDF directly
        for bin_idx in range(1, eff_hist.GetNbinsX() + 1):
            cdf_value = cdf.GetBinContent(bin_idx)
            efficiency = cdf_value
            eff_hist.SetBinContent(bin_idx, efficiency)
            eff_hist.SetBinError(bin_idx, 0)
    else:
        # "Greater than" efficiency - use 1 - CDF
        for bin_idx in range(1, eff_hist.GetNbinsX() + 1):
            cdf_value = cdf.GetBinContent(bin_idx)
            efficiency = 1.0 - cdf_value
            eff_hist.SetBinContent(bin_idx, efficiency)
            eff_hist.SetBinError(bin_idx, 0)
    eff_hist.SetName(f"{lepType}")
    eff_hist.SetTitle("")
    
    HISTs[lepType] = eff_hist

f.Close()

canvas = KinematicCanvas(HISTs, config)
canvas.drawPad()
outpath = f"{WORKDIR}/LeptonIDTest/plots/{args.era}/{args.object}/efficiency/{args.region}/{args.plotvar}.png"
os.makedirs(os.path.dirname(outpath), exist_ok=True)
canvas.canv.SaveAs(outpath)