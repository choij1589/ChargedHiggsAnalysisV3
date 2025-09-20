#!/usr/bin/env python
import os
import argparse
import logging
import json
import ROOT
from math import pow, sqrt
from plotter import ComparisonCanvas
ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--hlt", required=True, type=str, help="hlt")
parser.add_argument("--wp", required=True, type=str, help="wp")
parser.add_argument("--selection", default="Central", type=str, help="selection variations")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

WORKDIR = os.environ['WORKDIR']
if "El" in args.hlt:
    MEASURE = "electron"
elif "Mu" in args.hlt:
    MEASURE = "muon"
else:
    raise NameError(f"invalid hlt {args.hlt}")
SAMPLEGROUP = json.load(open("configs/samplegroup.json"))[args.era][MEASURE]
DATAPERIODs = SAMPLEGROUP["data"]
W = SAMPLEGROUP["W"]
Z = SAMPLEGROUP["Z"]
TT = SAMPLEGROUP["TT"]
VV = SAMPLEGROUP["VV"]
ST = SAMPLEGROUP["ST"]
MCList = W + Z + TT + VV + ST
SYSTs = json.load(open("configs/systematics.json"))[args.era][MEASURE]

if MEASURE == "muon":
    xTitle = "M(#mu#mu)"
elif MEASURE == "electron":
    xTitle = "M(#ee)"
else:
    raise NameError(f"invalid measure {MEASURE}")
config = {
    "xTitle": xTitle,
    "yTitle": "Events",
    "xRange": [75., 105.],
    "rRange": [0.0, 2.0],
    "logy": False,
    "era": args.era,
    "CoM": 13 if "201" in args.era else 13.6,
    "rTitle": "Data / Pred",
    "maxDigits": 3,
    "prescaled": True
}

## get histograms
data = None
for DATAPERIOD in DATAPERIODs:
    file_path = f"{WORKDIR}/SKNanoOutput/MeasFakeRateV2/{args.hlt}_RunSyst/{args.era}/{DATAPERIOD}.root"
    assert os.path.exists(file_path), f"{file_path} does not exist"
    f = ROOT.TFile.Open(file_path)
    try:
        h = f.Get(f"ZEnriched/{args.wp}/{args.selection}/ZCand/mass"); h.SetDirectory(0)
    except:
        logging.warning(f"No ZEnriched/{args.wp}/{args.selection}/ZCand/mass histogram for {DATAPERIOD}")
        continue
    f.Close()
    
    if data is None:
        data = h.Clone()
    else:
        data.Add(h)

HISTs = {}
for sample in MCList:
    file_path = f"{WORKDIR}/SKNanoOutput/MeasFakeRateV2/{args.hlt}_RunSyst/{args.era}/{sample}.root"
    assert os.path.exists(file_path), f"{file_path} does not exist"
    f = ROOT.TFile.Open(file_path)

    # Get central histogram
    try:
        h = f.Get(f"ZEnriched/{args.wp}/{args.selection}/ZCand/mass"); h.SetDirectory(0)
        hSysts = []
        for syst, sources in SYSTs.items():
            if not args.selection == "Central": continue
            systUp, systDown = sources
            h_up = f.Get(f"ZEnriched/{args.wp}/{systUp}/ZCand/mass"); h_up.SetDirectory(0)
            h_down = f.Get(f"ZEnriched/{args.wp}/{systDown}/ZCand/mass"); h_down.SetDirectory(0) 
            hSysts.append((h_up, h_down))
        f.Close()
    except:
        logging.warning(f"No ZEnriched/{args.wp}/Central/ZCand/mass histogram for {sample}")
        continue
    f.Close()

    # estimate total unc. bin by bin
    for bin in range(h.GetNcells()):
        stat_unc = h.GetBinError(bin)
        envelops = []
        for hset in hSysts:
            h_up, h_down = hset
            systUp = abs(h_up.GetBinContent(bin) - h.GetBinContent(bin))
            systDown = abs(h_up.GetBinContent(bin) - h.GetBinContent(bin))
            envelops.append(max(systUp, systDown))
            total_unc = pow(stat_unc, 2)
            for unc in envelops:
                total_unc += pow(unc, 2)
            total_unc = sqrt(total_unc)
            h.SetBinError(bin, total_unc)
    HISTs[sample] = h.Clone(sample)

# load scale from JSON file
scale_key = f"{args.hlt}_{args.wp}_{args.selection}"
json_path = f"{WORKDIR}/MeasFakeRateV2/results/{args.era}/JSON/{MEASURE}/prompt_scale.json"
with open(json_path, 'r') as f:
    scale_dict = json.load(f)
scale = scale_dict[scale_key]
logging.debug(f"MC histograms scaled to {scale}")
for hist in HISTs.values(): 
    hist.Scale(scale)

# merge backgrounds
def add_hist(name, hist, histDict):
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

# filter out none histograms from temp_dict
BKGs = {name: hist for name, hist in temp_dict.items() if hist}
# Sort BKGs by hist.Integral()
BKGs = dict(sorted(BKGs.items(), key=lambda x: x[1].Integral(), reverse=True))
logging.debug(f"BKGs: {BKGs}")

output_path = f"{WORKDIR}/MeasFakeRateV2/plots/{args.era}/{MEASURE}/{args.hlt}/ZEnriched/{args.selection}/Zmass_{args.wp}.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

plotter = ComparisonCanvas(data, BKGs, config)
plotter.drawPadUp()
plotter.drawPadDown()
plotter.canv.SaveAs(output_path)