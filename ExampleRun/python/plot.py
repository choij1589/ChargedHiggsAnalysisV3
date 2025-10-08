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
parser.add_argument("--histkey", required=True, type=str, help="histkey")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

WORKDIR = os.environ['WORKDIR']

with open("configs/histkeys.json") as f:
    config = json.load(f)[args.histkey]
config["era"] = args.era

#### Configurations
if args.era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
    DATASTREAM = "SingleMuon"
    DY = ["DYJets"]
    TT = ["TTLL_powheg"]
    VV = ["WW_pythia", "WZ_pythia", "ZZ_pythia"]
    ST = ["SingleTop_sch_Lep", "SingleTop_tch_antitop_Incl", "SingleTop_tch_top_Incl", "SingleTop_tW_antitop_NoFullyHad", "SingleTop_tW_top_NoFullyHad"]
elif args.era in ["2022", "2022EE", "2023", "2023BPix"]:
    DATASTREAM = "Muon"
MCList = DY + TT + VV + ST

output_path = f"{WORKDIR}/ExampleRun/plots/{args.era}/{args.histkey.replace('/', '_')}.png"

#### Get Histograms
HISTs = {}
COLORs = {}

## DATA
## hadd data if there is no combined file
file_path = f"{WORKDIR}/SKNanoOutput/ExampleRun/{args.era}/{DATASTREAM}.root"
logging.debug(f"file_path: {file_path}")
if not os.path.exists(file_path):
    logging.info(os.listdir(f"{os.path.dirname(file_path)}"))
    logging.info(f"file {file_path} does not exist. hadding...")
    response = input("Do you want to continue? [y/n]: ").strip().lower()
    if response == "y":
        os.system(f"hadd -f {file_path} {os.path.dirname(file_path)}/{DATASTREAM}_*.root")
    elif response == "n":
        print("No data file to proceed plotting, exiting...")
        exit(1)
    else:
        raise ValueError("invalid response")

assert os.path.exists(file_path), f"file {file_path} does not exist"
f = ROOT.TFile.Open(file_path)
data = f.Get(args.histkey); data.SetDirectory(0)
f.Close()

for sample in MCList:
    file_path = f"{WORKDIR}/SKNanoOutput/ExampleRun/{args.era}/{sample}.root"
    logging.debug(f"file_path: {file_path}")
    assert os.path.exists(file_path), f"file {file_path} does not exist"
    f = ROOT.TFile.Open(file_path)
    h = f.Get(args.histkey); h.SetDirectory(0)
    f.Close()

    HISTs[sample] = h.Clone(sample)

#### merge background
def add_hist(name, hist, histDict):
    # content of dictionary should be initialized as "None"
    if not histDict[name]:
        histDict[name] = hist.Clone(name)
    else:
        histDict[name].Add(hist)

temp_dict = {}
temp_dict["DY"] = None
temp_dict["TT"] = None
temp_dict["VV"] = None
temp_dict["ST"] = None

for sample in DY:
    if not sample in HISTs.keys(): continue
    add_hist("DY", HISTs[sample], temp_dict)
for sample in TT:
    if not sample in HISTs.keys(): continue
    add_hist("TT", HISTs[sample], temp_dict)
for sample in VV:
    if not sample in HISTs.keys(): continue
    add_hist("VV", HISTs[sample], temp_dict)
for sample in ST:
    if not sample in HISTs.keys(): continue
    add_hist("ST", HISTs[sample], temp_dict)

#### remove none histograms
BKGs = {}
for key, value in temp_dict.items():
    if value is None: continue
    BKGs[key] = value

COLORs["data"] = ROOT.kBlack
COLORs["DY"] = ROOT.kGray
COLORs["TT"] = ROOT.kViolet
COLORs["VV"] = ROOT.kGreen
COLORs["ST"] = ROOT.kAzure

#### draw plots
c = ComparisonCanvas(config=config)
c.drawBackgrounds(BKGs, COLORs)
c.drawData(data)
c.drawRatio()
c.drawLegend()
c.finalize()

os.makedirs(os.path.dirname(output_path), exist_ok=True)
c.SaveAs(output_path)
