#! /usr/bin/env python
import os
import argparse
import logging
import json
import ctypes
import ROOT
from plotter import KinematicCanvasWithRatio
ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--sample", required=True, type=str, help="MC sample name")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
WORKDIR = os.getenv("WORKDIR")
ERA = "2018"
HISTKEY = "electrons/1/pt"

config = {
    "xTitle": "p_{T}(e) [GeV]",
    "yTitle": "Events / 5 GeV",
    "rTitle": "Ratio",
    "xRange": [0., 150.],
    "rRange": [0.5, 1.5],
    "rebin": 5,
    "era": ERA,
    "CoM": 13.,
    "overflow": True,
    "maxDigits": 3,
    "legend": (0.54, 0.89 - 0.05 * 7, 0.89, 0.89, 0.04)
}

#### Get histograms
HISTs = {}
FLAGs = ["EMuTrigs", 
         "EMuTrigsWithSglElTrigs", 
         "EMuTrigsWithSglMuTrigs", 
         "EMuTrigsWithDblMuTrigs"]

# Dictionary to store integral values
integrals_data = {
    "sample": args.sample,
    "era": ERA,
    "histkey": HISTKEY,
    "integrals": {}
}

for flag in FLAGs:
    f = ROOT.TFile(f"{WORKDIR}/SKNanoOutput/TriggerStudy/Run{flag}/{ERA}/{args.sample}.root")
    hist = f.Get(f"SR1E2Mu/Central/{HISTKEY}")
    hist.SetDirectory(0)
    # For signal sample, apply filter efficiency
    if "MHc70_MA15" in args.sample:
        hist.Scale(0.6113)
    if "MHc100_MA60" in args.sample:
        hist.Scale(0.684)
    if "MHc130_MA90" in args.sample:
        hist.Scale(0.687)
    if "MHc160_MA155" in args.sample:
        hist.Scale(0.5187)

    HISTs[flag] = hist
    
    # Calculate integral including underflow and overflow bins with errors
    error = ctypes.c_double(0)
    integral_value = hist.IntegralAndError(0, hist.GetNbinsX()+1, error)
    
    # Calculate integral without overflow bins with errors
    error_no_overflow = ctypes.c_double(0)
    integral_no_overflow = hist.IntegralAndError(1, hist.GetNbinsX(), error_no_overflow)
    
    integrals_data["integrals"][flag] = {
        "total_integral": integral_value,
        "total_integral_error": error.value,
        "nbins": hist.GetNbinsX(),
        "underflow": hist.GetBinContent(0),
        "overflow": hist.GetBinContent(hist.GetNbinsX()+1),
        "integral_no_overflow": integral_no_overflow,
        "integral_no_overflow_error": error_no_overflow.value
    }
    
    f.Close()

#### Plot
canvas = KinematicCanvasWithRatio(HISTs, config)
canvas.drawPadUp()
canvas.drawPadDown()
outpath = f"{WORKDIR}/TriggerStrategy/plots/{args.sample}.png"
os.makedirs(os.path.dirname(outpath), exist_ok=True)
canvas.canv.SaveAs(outpath)

#### Save integrals to JSON
json_outpath = f"{WORKDIR}/TriggerStrategy/integrals/{args.sample}.json"
os.makedirs(os.path.dirname(json_outpath), exist_ok=True)
with open(json_outpath, 'w') as f:
    json.dump(integrals_data, f, indent=2)

logging.info(f"Plot saved to: {outpath}")
logging.info(f"Integrals saved to: {json_outpath}")
