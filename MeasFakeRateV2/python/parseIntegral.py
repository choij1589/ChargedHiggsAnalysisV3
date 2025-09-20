#!/usr/bin/env python
import os
import logging
import argparse
import ROOT
import json
from ctypes import c_double
from itertools import product
from common import findbin

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--measure", required=True, type=str, help="electron / muon")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
if args.debug:
    logging.basicConfig(level=logging.DEBUG)

#### Settings
WORKDIR = os.environ['WORKDIR']
ptcorr_bins = []
eta_bins = []

# Determine if this is Run3
is_run3 = args.era in ["2022", "2022EE", "2023", "2023BPix"]

SYSTs = ["Central", "Stat"]
if args.measure == "electron":
    ptcorr_bins = [10., 15., 20., 25., 35., 50., 100.]
    if is_run3:
        eta_bins = [-2.5, -1.479, -0.8, 0., 0.8, 1.479, 2.5]
    else:
        eta_bins = [0., 0.8, 1.479, 2.5]  # abseta_bins for Run2
elif args.measure == "muon":
    ptcorr_bins = [10., 15., 20., 30., 50., 100.]
    if is_run3:
        eta_bins = [-2.4, -1.6, -0.9, 0., 0.9, 1.6, 2.4]
    else:
        eta_bins = [0., 0.9, 1.6, 2.4]  # abseta_bins for Run2
else:
    raise KeyError(f"Wrong measure {args.measure}")

SAMPLEGROUP = json.load(open("configs/samplegroup.json"))[args.era][args.measure]
PromptSystematics = json.load(open("configs/systematics.json"))[args.era][args.measure]
SelectionVariations = ["Central", "MotherJetPt_Up", "MotherJetPt_Down", "RequireHeavyTag"]

DATAPERIODs = SAMPLEGROUP["data"]
W = SAMPLEGROUP["W"]
Z = SAMPLEGROUP["Z"]
TT = SAMPLEGROUP["TT"]
ST = SAMPLEGROUP["ST"]
VV = SAMPLEGROUP["VV"]
MCList = W + Z + TT + ST + VV

def get_hist(sample, ptcorr, eta_value, wp, syst="Central"):
    prefix = findbin(ptcorr, eta_value, ptcorr_bins, eta_bins, is_run3)
    channel = ""
    if args.measure == "muon":
        if ptcorr < 30.: channel = "MeasFakeMu8"
        else:            channel = "MeasFakeMu17"
    elif args.measure == "electron":
        if ptcorr < 20.: channel = "MeasFakeEl8"
        elif ptcorr < 35.: channel = "MeasFakeEl12"
        else:            channel = "MeasFakeEl23"
    else:
        raise KeyError(f"Wrong measure {args.measure}")
    
    file_path = f"{WORKDIR}/SKNanoOutput/MeasFakeRateV2/{channel}_RunSyst/{args.era}/{sample}.root"
    logging.debug(f"file_path: {file_path}")
    
    assert os.path.exists(file_path), f"File not found: {file_path}"
    
    f = ROOT.TFile.Open(file_path)
    try:
        if syst == "Stat":
            h = f.Get(f"{prefix}/QCDEnriched/{wp}/Central/MT"); h.SetDirectory(0)
        else:
            h = f.Get(f"{prefix}/QCDEnriched/{wp}/{syst}/MT"); h.SetDirectory(0)
        f.Close()
        return h
    except:
        print(f"Cannot find {prefix}/QCDEnriched/{wp}/{syst}/MT for sample {sample}")
        return None
    
def collect_data(sample, wp, syst):
    data = []
    # For Run2, we need to iterate over abseta values but fill with center values
    # For Run3, we iterate over eta values directly
    if is_run3:
        eta_values = [0.5 * (eta_bins[i] + eta_bins[i+1]) for i in range(len(eta_bins)-1)]
    else:
        # For Run2, use center of abseta bins
        eta_values = [0.5 * (eta_bins[i] + eta_bins[i+1]) for i in range(len(eta_bins)-1)]

    ptcorr_values = [0.5 * (ptcorr_bins[i] + ptcorr_bins[i+1]) for i in range(len(ptcorr_bins)-1)]

    for ptcorr in ptcorr_values:
        for eta_value in eta_values:
            h = get_hist(sample, ptcorr, eta_value, wp, syst)
            if h is None:   # It means that no events are filled in the histogram
                data.append(0.)
                continue

            if syst == "Stat":
                err = c_double()
                rate = h.IntegralAndError(0, h.GetNbinsX()+1, err)
                data.append(err.value)
            else:
                rate = h.Integral(0, h.GetNbinsX()+1)
                data.append(rate)
    return data

if __name__ == "__main__":
    # Make DataFrame
    index_col = []
    # For Run2, we need to iterate over abseta values but fill with center values
    # For Run3, we iterate over eta values directly
    if is_run3:
        eta_values = [0.5 * (eta_bins[i] + eta_bins[i+1]) for i in range(len(eta_bins)-1)]
    else:
        # For Run2, use center of abseta bins
        eta_values = [0.5 * (eta_bins[i] + eta_bins[i+1]) for i in range(len(eta_bins)-1)]

    ptcorr_values = [0.5 * (ptcorr_bins[i] + ptcorr_bins[i+1]) for i in range(len(ptcorr_bins)-1)]

    for ptcorr in ptcorr_values:
        for eta_value in eta_values:
            index_col.append(findbin(ptcorr, eta_value, ptcorr_bins, eta_bins, is_run3))
        
    # Collect DATASTREAM
    for sample, wp in product(DATAPERIODs, ["loose", "tight"]):
        data_dict = {}
        for syst in SelectionVariations:
            data = collect_data(sample, wp, syst)
            data_dict[syst] = {index_col[i]: data[i] for i in range(len(data))}
        
        json_path = f"results/{args.era}/JSON/{args.measure}/{sample}_{wp}.json"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(data_dict, f, indent=2)
    
    # Collect MC
    for sample, wp in product(MCList, ["loose", "tight"]):
        data_dict = {}
        for syst in SelectionVariations:
            data = collect_data(sample, wp, syst)
            data_dict[syst] = {index_col[i]: data[i] for i in range(len(data))}
        for syst, source in PromptSystematics.items():
            syst_up, syst_down = tuple(source)
            data_up = collect_data(sample, wp, syst_up)
            data_down = collect_data(sample, wp, syst_down)
            data_dict[syst_up] = {index_col[i]: data_up[i] for i in range(len(data_up))}
            data_dict[syst_down] = {index_col[i]: data_down[i] for i in range(len(data_down))}
        
        json_path = f"results/{args.era}/JSON/{args.measure}/{sample}_{wp}.json"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(data_dict, f, indent=2)