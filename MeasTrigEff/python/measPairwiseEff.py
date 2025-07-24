#!/usr/bin/env python
import os
import logging
import argparse
import json
import ROOT
import numpy as np

WORKDIR = os.environ['WORKDIR']
parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--filter", required=True, type=str, help="DblMuDZ / DblMuDZM / DblMuM / EMuDZ")
parser.add_argument("--debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
if args.debug: logging.basicConfig(level=logging.DEBUG)

if args.filter == "DblMuDZ":
    DATASTREAMs = {
        "2016preVFP": "DoubleMuon",
        "2016postVFP": "DoubleMuon",
        "2017": "DoubleMuon",
        "2018": "DoubleMuon",
        "2022": ["DoubleMuon", "Muon"],
        "2022EE": ["Muon"],
        "2023": ["Muon0_C_v1", "Muon0_C_v2", "Muon0_C_v3", "Muon0_C_v4", "Muon1_C_v1", "Muon1_C_v2", "Muon1_C_v3", "Muon1_C_v4"],
        "2023BPix": ["Muon"]
    }
    FLAG = "MeasDblMuPairwise"
    DENOM = "TrigEff_Iso"
    NUM = "TrigEff_IsoDZ"
elif args.filter == "DblMuDZM":
    DATASTREAMs = {
        "2016preVFP": "DoubleMuon",
        "2016postVFP": "DoubleMuon",
        "2017": "DoubleMuon",
        "2018": "DoubleMuon",
        "2022": ["DoubleMuon", "Muon"],
        "2022EE": ["Muon"],
        "2023": ["Muon0_C_v1", "Muon0_C_v2", "Muon0_C_v3", "Muon0_C_v4", "Muon1_C_v1", "Muon1_C_v2", "Muon1_C_v3", "Muon1_C_v4"],
        "2023BPix": ["Muon"]
    }
    FLAG = "MeasDblMuPairwise"
    DENOM = "TrigEff_Iso"
    NUM = "TrigEff_IsoDZM"
elif args.filter == "DblMuM":
    DATASTREAMs = {
        "2016preVFP": "DoubleMuon",
        "2016postVFP": "DoubleMuon",
        "2017": "DoubleMuon",
        "2018": "DoubleMuon",
        "2022": ["DoubleMuon", "Muon"],
        "2022EE": ["Muon"],
        "2023": ["Muon0_C_v1", "Muon0_C_v2", "Muon0_C_v3", "Muon0_C_v4", "Muon1_C_v1", "Muon1_C_v2", "Muon1_C_v3", "Muon1_C_v4"],
        "2023BPix": ["Muon"]
    }
    FLAG = "MeasDblMuPairwise"
    DENOM = "TrigEff_IsoDZ"
    NUM = "TrigEff_IsoDZM"
elif args.filter == "EMuDZ":
    DATASTREAMs = {
        "2016preVFP": "MuonEG",
        "2016postVFP": "MuonEG",
        "2017": "MuonEG",
        "2018": "MuonEG",
        "2022": ["MuonEG"],
        "2022EE": ["MuonEG"],
        "2023": ["MuonEG_C_v1", "MuonEG_C_v2", "MuonEG_C_v3", "MuonEG_C_v4"],
        "2023BPix": ["MuonEG"]
    }
    FLAG = "MeasEMuPairwise"
    DENOM = "TrigEff_EMuDZ"
    NUM = "TrigEff_EMuDZ"
else:
    raise ValueError(f"Invalid filter {args.filter}")

DATASTREAMs = DATASTREAMs[args.era]

# helper functions
def meas_efficiency(filter: ROOT.TString, syst: ROOT.TString, is_data: bool) -> ROOT.TH1D:
    h_denom_total = None
    h_num_total = None
    
    ## Get data histograms and add up
    if is_data:
        for rtdata in DATASTREAMs:
            f = ROOT.TFile.Open(f"{WORKDIR}/SKNanoOutput/MeasTrigEff/{FLAG}/{args.era}/{rtdata}.root")
            h_denom = f.Get(DENOM); h_denom.SetDirectory(0)
            h_num = f.Get(NUM); h_num.SetDirectory(0)
            f.Close()
            if h_denom_total is None:
                h_denom_total = h_denom.Clone(f"TrigEff_{FLAG}_DENOM/{syst}/fEta_Pt")
                h_num_total = h_num.Clone(f"TrigEff_{FLAG}_NUM/{syst}/fEta_Pt")
            else:
                h_denom_total.Add(h_denom)
                h_num_total.Add(h_num)
        h_denom_total.SetDirectory(0)
        h_num_total.SetDirectory(0)
    elif syst == "AltMC":
        for rtmc in ["DYJets", "DYJets10to50"]:
            f = ROOT.TFile.Open(f"{WORKDIR}/SKNanoOutput/MeasTrigEff/{FLAG}/{args.era}/{rtmc}.root")
            h_denom = f.Get(DENOM); h_denom.SetDirectory(0)
            h_num = f.Get(NUM); h_num.SetDirectory(0)
            f.Close()
            if h_denom_total is None:
                h_denom_total = h_denom.Clone(f"TrigEff_{FLAG}_DENOM/Central/fEta_Pt")
                h_num_total = h_num.Clone(f"TrigEff_{FLAG}_NUM/Central/fEta_Pt")
            else:
                h_denom_total.Add(h_denom)
                h_num_total.Add(h_num)
        h_denom_total.SetDirectory(0)
        h_num_total.SetDirectory(0)
    else:
        f = ROOT.TFile.Open(f"{WORKDIR}/SKNanoOutput/MeasTrigEff/{FLAG}/{args.era}/TTLL_powheg.root")
        h_denom = f.Get(DENOM); h_denom.SetDirectory(0)
        h_num = f.Get(NUM); h_num.SetDirectory(0)
        f.Close()
        h_denom_total = h_denom.Clone(f"TrigEff_{FLAG}_DENOM/Central/fEta_Pt")
        h_num_total = h_num.Clone(f"TrigEff_{FLAG}_NUM/Central/fEta_Pt")
        h_denom_total.SetDirectory(0)
        h_num_total.SetDirectory(0)
    
    if filter == "EMuDZ":
        num, err_num = h_num_total.GetBinContent(2), h_num_total.GetBinError(2)
        denom, err_denom = h_denom_total.GetBinContent(1), h_denom_total.GetBinError(1)
    else:
        err_num = np.array([0.])
        err_denom = np.array([0.])
        num = h_num_total.IntegralAndError(1, h_num_total.GetNbinsX(), err_num)
        denom = h_denom_total.IntegralAndError(1, h_denom_total.GetNbinsX(), err_denom)
        err_num, err_denom = err_num[0], err_denom[0]
    eff = num / denom
    err = eff*ROOT.TMath.Sqrt(ROOT.TMath.Power(err_num/num, 2) + ROOT.TMath.Power(err_denom/denom, 2))
    return (eff, err)

if __name__ == "__main__":
    # Calculate efficiencies
    data_eff = meas_efficiency(args.filter, "Central", True)
    mc_eff = meas_efficiency(args.filter, "Central", False)
    altmc_eff = meas_efficiency(args.filter, "AltMC", False)
    
    # Print results (original format)
    print(f"# Pairwise Filter, Data Eff, MC Eff, MC Eff - AltMC")
    print(args.filter, data_eff, mc_eff, altmc_eff, sep=", ")
    
    # Save results as JSON
    results = {
        "filter": args.filter,
        "era": args.era,
        "data_efficiency": {
            "value": data_eff[0],
            "error": data_eff[1]
        },
        "mc_efficiency": {
            "value": mc_eff[0],
            "error": mc_eff[1]
        },
        "altmc_efficiency": {
            "value": altmc_eff[0],
            "error": altmc_eff[1]
        }
    }
    
    # Create output directory
    output_dir = f"results/{args.era}/json"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON file
    output_file = f"{output_dir}/{args.filter}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")