#!/usr/bin/env python
import os
import logging
import argparse
import ROOT

WORKDIR = os.environ['WORKDIR']
parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--hltpath", required=True, type=str, help="Mu8El23 / Mu23El12")
parser.add_argument("--leg", required=True, type=str, help="muon / electron")
parser.add_argument("--debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
if args.debug: logging.basicConfig(level=logging.DEBUG)

if args.leg == "muon":
    DATASTREAMs = {
        "2016preVFP": ["SingleElectron_B", "SingleElectron_C", "SingleElectron_D", "SingleElectron_E", "SingleElectron_F"],
        "2016postVFP": ["SingleElectron_F", "SingleElectron_G", "SingleElectron_H"],
        "2017": ["SingleElectron_B", "SingleElectron_C", "SingleElectron_D", "SingleElectron_E", "SingleElectron_F"],
        "2018": ["EGamma_A", "EGamma_B", "EGamma_C", "EGamma_D"],
        "2022": ["EGamma_C", "EGamma_D"],
        "2022EE": ["EGamma_E", "EGamma_F", "EGamma_G"],
        "2023": ["EGamma0_C_v1", "EGamma0_C_v2", "EGamma0_C_v3", "EGamma0_C_v4", "EGamma1_C_v1", "EGamma1_C_v2", "EGamma1_C_v3", "EGamma1_C_v4"],
        "2023BPix": ["EGamma0_D_v1", "EGamma0_D_v2", "EGamma1_D_v1", "EGamma1_D_v2"]
    }
    FLAG = "MeasMuLegs"
    LEG = "MuLeg"
elif args.leg == "electron":
    DATASTREAMs = {
        "2016preVFP": ["SingleMuon_B", "SingleMuon_C", "SingleMuon_D", "SingleMuon_E", "SingleMuon_F"],
        "2016postVFP": ["SingleMuon_F", "SingleMuon_G", "SingleMuon_H"],
        "2017": ["SingleMuon_B", "SingleMuon_C", "SingleMuon_D", "SingleMuon_E", "SingleMuon_F"],
        "2018": ["SingleMuon_A", "SingleMuon_B", "SingleMuon_C", "SingleMuon_D"],
        "2022": ["SingleMuon_C", "Muon_C", "Muon_D"],
        "2022EE": ["Muon_E", "Muon_F", "Muon_G"],
        "2023": ["Muon0_C_v1", "Muon0_C_v2", "Muon0_C_v3", "Muon0_C_v4", "Muon1_C_v1", "Muon1_C_v2", "Muon1_C_v3", "Muon1_C_v4"],
        "2023BPix": ["Muon0_D_v1", "Muon0_D_v2", "Muon1_D_v1", "Muon1_D_v2"]
    }
    FLAG = "MeasElLegs"
    LEG = "ElLeg"
else:
    raise ValueError(f"Invalid leg {args.leg}")

DATASTREAMs = DATASTREAMs[args.era]
DYJets = ["DYJets", "DYJets10to50"]

# helper functions
def get_histograms(hltpath: ROOT.TString, syst: ROOT.TString, is_data: bool) -> (ROOT.TH2D, ROOT.TH2D):
    h_denom_total = None
    h_num_total = None
    
    ## Get data histograms and add up
    if is_data:
        for rtdata in DATASTREAMs:
            logging.debug(f"Processing {rtdata}")
            f = ROOT.TFile.Open(f"{WORKDIR}/SKNanoOutput/MeasTrigEff/{FLAG}/{args.era}/{rtdata}.root")
            print(f"denom: {f.Get(f'TrigEff_{hltpath}_{LEG}_DENOM/{syst}/fEta_Pt')}")
            print(f"num: {f.Get(f'TrigEff_{hltpath}_{LEG}_NUM/{syst}/fEta_Pt')}")
            h_denom = f.Get(f"TrigEff_{hltpath}_{LEG}_DENOM/{syst}/fEta_Pt"); h_denom.SetDirectory(0)
            h_num = f.Get(f"TrigEff_{hltpath}_{LEG}_NUM/{syst}/fEta_Pt"); h_num.SetDirectory(0)
            f.Close()
            if h_denom_total is None:
                h_denom_total = h_denom.Clone(f"TrigEff_{hltpath}_{LEG}_DENOM/{syst}/fEta_Pt")
                h_num_total = h_num.Clone(f"TrigEff_{hltpath}_{LEG}_NUM/{syst}/fEta_Pt")
            else:
                h_denom_total.Add(h_denom)
                h_num_total.Add(h_num)
        h_denom_total.SetDirectory(0)
        h_num_total.SetDirectory(0)
    elif syst == "AltMC":
        for rtdata in DYJets:
            logging.debug(f"Processing {rtdata}")
            f = ROOT.TFile.Open(f"{WORKDIR}/SKNanoOutput/MeasTrigEff/{FLAG}/{args.era}/{rtdata}.root")
            h_denom = f.Get(f"TrigEff_{hltpath}_{LEG}_DENOM/Central/fEta_Pt"); h_denom.SetDirectory(0)
            h_num = f.Get(f"TrigEff_{hltpath}_{LEG}_NUM/Central/fEta_Pt"); h_num.SetDirectory(0)
            f.Close()
            if h_denom_total is None:
                h_denom_total = h_denom.Clone(f"TrigEff_{hltpath}_{LEG}_DENOM/Central/fEta_Pt")
                h_num_total = h_num.Clone(f"TrigEff_{hltpath}_{LEG}_NUM/Central/fEta_Pt")
            else:
                h_denom_total.Add(h_denom)
                h_num_total.Add(h_num)
        h_denom_total.SetDirectory(0)
        h_num_total.SetDirectory(0)
    else:
        logging.debug(f"Processing TTLL_powheg")
        f = ROOT.TFile.Open(f"{WORKDIR}/SKNanoOutput/MeasTrigEff/{FLAG}/{args.era}/TTLL_powheg.root")
        h_denom = f.Get(f"TrigEff_{hltpath}_{LEG}_DENOM/{syst}/fEta_Pt"); h_denom.SetDirectory(0)
        h_num = f.Get(f"TrigEff_{hltpath}_{LEG}_NUM/{syst}/fEta_Pt"); h_num.SetDirectory(0)
        f.Close()
        h_denom_total = h_denom.Clone(f"TrigEff_{hltpath}_{LEG}_DENOM/{syst}/fEta_Pt")
        h_num_total = h_num.Clone(f"TrigEff_{hltpath}_{LEG}_NUM/{syst}/fEta_Pt")
        h_denom_total.SetDirectory(0)
        h_num_total.SetDirectory(0)
    
    return (h_num_total, h_denom_total)

def get_efficiency(histkey: ROOT.TString):
    hltpath, _, syst = tuple(histkey.split("_"))
    is_data = True if "Data" in histkey else False
    h_num, h_denom = get_histograms(hltpath, syst, is_data)
    h_eff = h_num.Clone(f"Eff_{histkey}"); h_eff.Reset()
    for bin in range(1, h_num.GetNcells()+1):
        num, err_num = h_num.GetBinContent(bin), h_num.GetBinError(bin)
        denom, err_denom = h_denom.GetBinContent(bin), h_denom.GetBinError(bin)
        if denom == 0 or num == 0:
            eff, err = 0., 0.
        else:
            eff = num / denom
            err = eff*ROOT.TMath.Sqrt(ROOT.TMath.Power(err_num/num, 2) + ROOT.TMath.Power(err_denom/denom, 2))
        h_eff.SetBinContent(bin, eff)
        h_eff.SetBinError(bin, err)
    h_eff.SetDirectory(0) 
    return h_eff

def main():
    # Get Efficiency Histograms
    h_eff_data_central = get_efficiency(f"{args.hltpath}_Data_Central")
    h_eff_data_alttag = get_efficiency(f"{args.hltpath}_Data_AltTag")
    h_eff_mc_central = get_efficiency(f"{args.hltpath}_MC_Central")
    h_eff_mc_altmc = get_efficiency(f"{args.hltpath}_MC_AltMC")
    h_eff_mc_alttag = get_efficiency(f"{args.hltpath}_MC_AltTag")

    
    # Calculate Systematic Uncertainties
    h_eff_data = h_eff_data_central.Clone(f"{args.hltpath}_Data"); h_eff_data.Reset()
    for bin in range(1, h_eff_data.GetNcells()+1):
        stat = h_eff_data_central.GetBinError(bin)
        diff_alttag = h_eff_data_alttag.GetBinContent(bin) - h_eff_data_central.GetBinContent(bin)
        total = ROOT.TMath.Sqrt(ROOT.TMath.Power(stat, 2) + ROOT.TMath.Power(diff_alttag, 2))
        h_eff_data.SetBinContent(bin, h_eff_data_central.GetBinContent(bin))
        h_eff_data.SetBinError(bin, total)
    
    h_eff_mc = h_eff_mc_central.Clone(f"{args.hltpath}_MC"); h_eff_mc.Reset()
    for bin in range(1, h_eff_mc.GetNcells()+1):
        stat = h_eff_mc_central.GetBinError(bin)
        diff_altmc = h_eff_mc_altmc.GetBinContent(bin) - h_eff_mc_central.GetBinContent(bin)
        diff_alttag = h_eff_mc_alttag.GetBinContent(bin) - h_eff_mc_central.GetBinContent(bin)
        total = ROOT.TMath.Sqrt(ROOT.TMath.Power(stat, 2) + ROOT.TMath.Power(diff_altmc, 2) + ROOT.TMath.Power(diff_alttag, 2))
        print(total)
        h_eff_mc.SetBinContent(bin, h_eff_mc_central.GetBinContent(bin))
        h_eff_mc.SetBinError(bin, total)
        
    # Save Histograms
    outdir = f"{WORKDIR}/MeasTrigEff/results/{args.era}/ROOT/{args.hltpath}_{args.leg}.root"
    os.makedirs(os.path.dirname(outdir), exist_ok=True)
    f = ROOT.TFile(outdir, "RECREATE")
    f.cd()
    h_eff_data_central.Write()
    h_eff_data_alttag.Write()
    h_eff_mc_central.Write()
    h_eff_mc_altmc.Write()
    h_eff_mc_alttag.Write()
    h_eff_data.Write()
    h_eff_mc.Write()
    f.Close()
    print(f"Results saved to {outdir}")

if __name__ == "__main__":
    main()
