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
#### Configurations
if args.era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
    RUN = "Run2"
    config["CoM"] = 13
elif args.era in ["2022", "2022EE", "2023", "2023BPix"]:
    RUN = "Run3"
    config["CoM"] = 13.6
else:
    raise ValueError(f"Invalid era: {args.era}")

if args.channel == "DIMU":
    FLAG = "RunDiMu"
elif args.channel == "EMU":
    FLAG = "RunEMu"
else:
    raise ValueError(f"Invalid channel: {args.channel}")

SAMPLEGROUP = json.load(open("configs/samplegroup.json"))[args.era][args.channel]
SYSTs = json.load(open("configs/systematics.json"))[args.era][args.channel]
logging.debug(f"SAMPLEGROUP: {SAMPLEGROUP}")
logging.debug(f"SYSTs: {SYSTs}")

DATAPERIODs = SAMPLEGROUP["data"]
W = SAMPLEGROUP["W"]
Z = SAMPLEGROUP["Z"]
TT = SAMPLEGROUP["TT"]
ST = SAMPLEGROUP["ST"]
VV = SAMPLEGROUP["VV"]
MCList = W + Z + TT + ST + VV

if args.exclude:
    OUTPUTPATH = f"{WORKDIR}/DiLepton/plots/{args.era}/{args.channel}/No{args.exclude}/{args.histkey.replace('/', '_')}.png"
else:
    OUTPUTPATH = f"{WORKDIR}/DiLepton/plots/{args.era}/{args.channel}/Central/{args.histkey.replace('/', '_')}.png"

os.makedirs(os.path.dirname(OUTPUTPATH), exist_ok=True)

#### Get Histograms
HISTs = {}

## DATA
## merge histograms from different periods
data = None
for sample in DATAPERIODs:
    file_path = f"{WORKDIR}/SKNanoOutput/DiLepton/{FLAG}_RunSyst/{args.era}/{sample}.root"
    logging.debug(f"file_path: {file_path}")
    assert os.path.exists(file_path), f"file {file_path} does not exist"
    f = ROOT.TFile.Open(file_path)
    h = f.Get(f"{args.channel}/Central/{args.histkey}"); h.SetDirectory(0); h.SetTitle("data")
    f.Close()
    if data is None:
        data = h.Clone("data")
    else:
        data.Add(h)

for sample in MCList:
    file_path = f"{WORKDIR}/SKNanoOutput/DiLepton/{FLAG}_RunSyst/{args.era}/{sample}.root"
    logging.debug(f"file_path: {file_path}")
    assert os.path.exists(file_path), f"file {file_path} does not exist"
    f = ROOT.TFile.Open(file_path)
    try:
        if args.exclude:
            h = f.Get(f"{args.channel}/{args.exclude}_NotImplemented/{args.histkey}"); h.SetDirectory(0)
        else:
            h = f.Get(f"{args.channel}/Central/{args.histkey}"); h.SetDirectory(0)
            #print(h.Integral(), h.Integral(0, h.GetNbinsX()+1))
    except Exception as e:
        logging.debug(e, sample, args.histkey)
        f.Close()
        continue

    # Get Systematic Histograms
    hSysts = []
    for syst, sources in SYSTs.items():
        if args.exclude: continue
        syst_up, syst_down = tuple(sources)
        try:
            h_up = f.Get(f"{args.channel}/{syst_up}/{args.histkey}"); h_up.SetDirectory(0)
            h_down = f.Get(f"{args.channel}/{syst_down}/{args.histkey}"); h_down.SetDirectory(0)
            hSysts.append((h_up, h_down))
        except Exception as e:
            logging.debug(e, sample, syst_up, syst_down)
            f.Close()
            continue
    f.Close()

    # estimate total unc. bin by bin
    for bin in range(h.GetNcells()):
        stat_unc = h.GetBinError(bin)
        envelops = []
        for h_up, h_down in hSysts:
            systUp = abs(h_up.GetBinContent(bin) - h.GetBinContent(bin))
            systDown = abs(h_down.GetBinContent(bin) - h.GetBinContent(bin))
            envelops.append(max(systUp, systDown))
        total_unc = sqrt(pow(stat_unc, 2) + sum([pow(x, 2) for x in envelops]))
        h.SetBinError(bin, total_unc)
    HISTs[sample] = h.Clone(sample)

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
