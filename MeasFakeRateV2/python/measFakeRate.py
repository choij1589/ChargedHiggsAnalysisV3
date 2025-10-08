#!/usr/bin/env python
import os
import argparse
import logging
import json
import ROOT
from itertools import product
from array import array
from common import findbin

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--measure", required=True, type=str, help="electron / muon")
parser.add_argument("--isQCD", default=False, action="store_true", help="isQCD")
parser.add_argument("--debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
if args.debug:
    logging.basicConfig(level=logging.DEBUG)

#### Settings
WORKDIR = os.environ['WORKDIR']
HLTPATHs = []
ptcorr_bins = []
eta_bins = []

# Determine if this is Run3
is_run3 = args.era in ["2022", "2022EE", "2023", "2023BPix"]

if args.measure == "electron":
    ptcorr_bins = [10., 15., 20., 25., 35., 50., 100.]
    if is_run3:
        eta_bins = [-2.5, -1.479, -0.8, 0., 0.8, 1.479, 2.5]
    else:
        eta_bins = [0., 0.8, 1.479, 2.5]  # abseta_bins for Run2
    HLTPATHs = ["MeasFakeEl8", "MeasFakeEl12", "MeasFakeEl23"]
    QCD = ["QCD_EMEnriched", "QCD_bcToE"]
elif args.measure == "muon":
    ptcorr_bins = [10., 15., 20., 30., 50., 100.]
    if is_run3:
        eta_bins = [-2.4, -1.6, -0.9, 0., 0.9, 1.6, 2.4]
    else:
        eta_bins = [0., 0.9, 1.6, 2.4]  # abseta_bins for Run2
    HLTPATHs = ["MeasFakeMu8", "MeasFakeMu17"]
    QCD = ["QCD_MuEnriched"]
else:
    raise KeyError(f"Wrong measure {args.measure}")

SAMPLEGROUP = json.load(open("configs/samplegroup.json"))[args.era][args.measure]
#PromptSystematics = json.load(open("configs/systematics.json"))[args.era][args.measure]
SelectionVariations = ["Central", "PromptNorm_Up", "PromptNorm_Down", "MotherJetPt_Up", "MotherJetPt_Down", "RequireHeavyTag"]

DATAPERIODs = SAMPLEGROUP["data"]
W = SAMPLEGROUP["W"]
Z = SAMPLEGROUP["Z"]
TT = SAMPLEGROUP["TT"]
ST = SAMPLEGROUP["ST"]
VV = SAMPLEGROUP["VV"]
MCList = W + Z + TT + ST + VV

WPs = ["loose", "tight"]

#### first evaluate central scale for product(hlt, wp, syst)
def get_prompt_scale(hltpath, wp, syst):
    histkey = f"ZEnriched/{wp}/{syst}/ZCand/mass"
    if syst in ["PromptNorm_Up", "PromptNorm_Down"]:
        histkey = f"ZEnriched/{wp}/Central/ZCand/mass"
    if syst == "Stat": histkey = f"ZEnriched/{wp}/Central/ZCand/mass"
    
    rate_data = 0.
    for sample in DATAPERIODs:
        file_path = f"{WORKDIR}/SKNanoOutput/MeasFakeRateV2/{hltpath}_RunSyst/{args.era}/{sample}.root"
        assert os.path.exists(file_path), f"File not found: {file_path}"
        f = ROOT.TFile.Open(file_path)
        try:
            h = f.Get(histkey); h.SetDirectory(0)
            rate_data += h.Integral(0, h.GetNbinsX()+1)
            f.Close()
        except:
            logging.warning(f"Cannot find {histkey} for sample {sample}")
            f.Close()
            continue
    
    rate_mc = 0.
    for sample in MCList:
        file_path = f"{WORKDIR}/SKNanoOutput/MeasFakeRateV2/{hltpath}_RunSyst/{args.era}/{sample}.root"
        assert os.path.exists(file_path), f"File not found: {file_path}"
        f = ROOT.TFile.Open(file_path)
        try:
            h = f.Get(histkey); h.SetDirectory(0)
            rate_mc += h.Integral(0, h.GetNbinsX()+1)
            f.Close()
        except:
            logging.warning(f"Cannot find {histkey} for sample {sample}")
            f.Close()
            continue
    
    scale = rate_data / rate_mc
    
    if syst == "PromptNorm_Up": scale *= 1.15
    if syst == "PromptNorm_Down": scale *= 0.85
    
    return scale

def extract_fake_from_data(ptcorr, eta_value, wp, syst):
    prefix = findbin(ptcorr, eta_value, ptcorr_bins, eta_bins, is_run3)
        
    # get integral for data
    data = 0.
    for sample in DATAPERIODs:
        json_path = f"{WORKDIR}/MeasFakeRateV2/results/{args.era}/JSON/{args.measure}/{sample}_{wp}.json"
        with open(json_path, 'r') as f:
            data_dict = json.load(f)
        if syst in ["PromptNorm_Up", "PromptNorm_Down"]:
            data += data_dict["Central"][prefix]
        else:
            data += data_dict[syst][prefix]
    
    # get MCs
    prompt = 0.
    for sample in MCList:
        json_path = f"{WORKDIR}/MeasFakeRateV2/results/{args.era}/JSON/{args.measure}/{sample}_{wp}.json"
        with open(json_path, 'r') as f:
            data_dict = json.load(f)
        if syst in ["PromptNorm_Up", "PromptNorm_Down"]:
            prompt += data_dict["Central"][prefix]
        else:
            prompt += data_dict[syst][prefix]
        
    # get prompt scale
    if args.measure == "electron":
        if ptcorr < 20.: hltpath = "MeasFakeEl8"
        elif ptcorr < 35.: hltpath = "MeasFakeEl12"
        else:            hltpath = "MeasFakeEl23"
    elif args.measure == "muon":
        if ptcorr < 30.: hltpath = "MeasFakeMu8"
        else:            hltpath = "MeasFakeMu17"
    else:
        raise KeyError(f"Wrong measure {args.measure}")
    scale = get_prompt_scale(hltpath, wp, syst)
    logging.debug(prefix, data, prompt, scale, prompt*scale)
    
    return data - prompt*scale

def extract_fake_from_qcd(ptcorr, eta_value, wp, samplegroup):
    prefix = findbin(ptcorr, eta_value, ptcorr_bins, eta_bins, is_run3)

    # get integral from QCD
    if args.measure == "electron":
        if ptcorr < 20.: hltpath = "MeasFakeEl8"
        elif ptcorr < 35.: hltpath = "MeasFakeEl12"
        else:            hltpath = "MeasFakeEl23"
               
    elif args.measure == "muon":
        if ptcorr < 30.: hltpath = "MeasFakeMu8"
        else:            hltpath = "MeasFakeMu17"

    histkey = f"{prefix}/Inclusive/{wp}/Central/MT"
    integral = 0.
    for sample in SAMPLEGROUP[samplegroup]:
        file_path = f"{WORKDIR}/SKNanoOutput/MeasFakeRateV2/{hltpath}_RunSyst/{args.era}/{sample}.root"
        assert os.path.exists(file_path), f"File not found: {file_path}"
        f = ROOT.TFile.Open(file_path)
        try:
            h = f.Get(histkey); h.SetDirectory(0)
            integral += h.Integral()
            f.Close()
        except:
            logging.warning(f"Cannot find {histkey} for sample {sample}")
            f.Close()
            continue
    
    return integral

def get_fake_rate(isQCD=False, syst="Central"):
    # Use unique names to avoid ROOT warnings about replacing histograms
    h_loose = ROOT.TH2D(f"h_loose_{syst}", f"h_loose_{syst}", len(eta_bins)-1, array('d', eta_bins), len(ptcorr_bins)-1, array('d', ptcorr_bins))
    h_tight = ROOT.TH2D(f"h_tight_{syst}", f"h_tight_{syst}", len(eta_bins)-1, array('d', eta_bins), len(ptcorr_bins)-1, array('d', ptcorr_bins))
    h_loose.SetDirectory(0)
    h_tight.SetDirectory(0)

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
            if isQCD:
                loose = extract_fake_from_qcd(ptcorr, eta_value, "loose", syst)
                tight = extract_fake_from_qcd(ptcorr, eta_value, "tight", syst)
            else:
                loose = extract_fake_from_data(ptcorr, eta_value, "loose", syst)
                tight = extract_fake_from_data(ptcorr, eta_value, "tight", syst)
            h_loose.Fill(eta_value, ptcorr, loose)
            h_tight.Fill(eta_value, ptcorr, tight)

    fake_rate = h_tight.Clone(f"fake rate - ({syst})")
    fake_rate.Divide(h_loose)
    fake_rate.SetTitle(f"fake rate - ({syst})")
    fake_rate.SetDirectory(0)
    return fake_rate

if __name__ == "__main__":
    #### Collect scales
    # For prompt normalization, assign 15% variation
    scale_dict = {}
    for hltpath, wp, syst in product(HLTPATHs, WPs, SelectionVariations):
        scale_dict[f"{hltpath}_{wp}_{syst}"] = get_prompt_scale(hltpath, wp, syst)

    #### Save scales
    json_path = f"{WORKDIR}/MeasFakeRateV2/results/{args.era}/JSON/{args.measure}/prompt_scale.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(scale_dict, f, indent=2)
    
    #### Evaluate fake rate
    output_path = f"{WORKDIR}/MeasFakeRateV2/results/{args.era}/ROOT/{args.measure}/fakerate.root"
    if args.isQCD:
        output_path = output_path.replace("fakerate", "fakerate_qcd")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    out = ROOT.TFile.Open(output_path, "RECREATE")
    if args.isQCD:
        for sample in QCD:
            h = get_fake_rate(args.isQCD, sample)
            out.cd()
            h.Write()
    else:
        for syst in SelectionVariations:
            h = get_fake_rate(args.isQCD, syst)
            out.cd()
            h.Write()
    out.Close()