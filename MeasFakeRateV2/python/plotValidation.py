#!/usr/bin/env python
import os
import argparse
import logging
import ROOT
import json
from itertools import product
from plotter import ComparisonCanvas
from common import findbin

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--hlt", required=True, type=str, help="hlt")
parser.add_argument("--wp", required=True, type=str, help="wp")
parser.add_argument("--region", required=True, type=str, help="region")
parser.add_argument("--selection", default="Central", type=str, help="selection")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

WORKDIR = os.environ['WORKDIR']

# Determine if this is Run3
is_run3 = args.era in ["2022", "2022EE", "2023", "2023BPix"]

if "El" in args.hlt:
    MEASURE = "electron"
    if args.hlt == "MeasFakeEl8":
        ptcorr_bins = [10., 15., 20., 25., 35., 50., 100.]
    elif args.hlt == "MeasFakeEl12":
        ptcorr_bins = [15., 20., 25., 35., 50., 100.]
    elif args.hlt == "MeasFakeEl23":
        ptcorr_bins = [25., 35., 50., 100.]
    else:
        raise ValueError(f"invalid hlt {args.hlt}")
    if is_run3:
        eta_bins = [-2.5, -1.479, -0.8, 0., 0.8, 1.479, 2.5]
    else:
        eta_bins = [0., 0.8, 1.479, 2.5]  # abseta_bins for Run2
elif "Mu" in args.hlt:
    MEASURE = "muon"
    if args.hlt == "MeasFakeMu8":
        ptcorr_bins = [10., 15., 20., 30., 50., 100.]
    elif args.hlt == "MeasFakeMu17":
        ptcorr_bins = [20., 30., 50., 100.]
    else:
        raise ValueError(f"invalid hlt {args.hlt}")
    if is_run3:
        eta_bins = [-2.4, -1.6, -0.9, 0., 0.9, 1.6, 2.4]
    else:
        eta_bins = [0., 0.9, 1.6, 2.4]  # abseta_bins for Run2
else:
    raise ValueError(f"invalid hlt {args.hlt}")

trigPathDict = {
    "MeasFakeMu8": "HLT_Mu8_TrkIsoVVL_v",
    "MeasFakeMu17": "HLT_Mu17_TrkIsoVVL_v",
    "MeasFakeEl8": "HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30_v",
    "MeasFakeEl12": "HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v",
    "MeasFakeEl23": "HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v"
}

SAMPLEGROUP = json.load(open("configs/samplegroup.json"))[args.era][MEASURE]

DATAPERIODs = SAMPLEGROUP["data"]
W = SAMPLEGROUP["W"]
Z = SAMPLEGROUP["Z"]
TT = SAMPLEGROUP["TT"]
VV = SAMPLEGROUP["VV"]
ST = SAMPLEGROUP["ST"]
MCList = W + Z + TT + VV + ST

QCD_EMEnriched = []
QCD_bcToE = []
QCD_MuEnriched = []
if MEASURE == "electron":
    QCD_EMEnriched = SAMPLEGROUP["QCD_EMEnriched"]
    QCD_bcToE = SAMPLEGROUP["QCD_bcToE"]
    MCList += QCD_EMEnriched + QCD_bcToE
elif MEASURE == "muon":
    QCD_MuEnriched = SAMPLEGROUP["QCD_MuEnriched"]
    MCList += QCD_MuEnriched
else:
    raise KeyError(f"Wrong measure {MEASURE}")

SYSTs = json.load(open("configs/systematics.json"))[args.era][MEASURE]

def add_hist(name, hist, histDict):
    if not histDict[name]:
        histDict[name] = hist.Clone(name)
    else:
        histDict[name].Add(hist)

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
        prefix = findbin(ptcorr, eta_value, ptcorr_bins, eta_bins, is_run3)
        HISTs = {}

        # data
        data = None
        for DATAPERIOD in DATAPERIODs:
            file_path = f"{WORKDIR}/SKNanoOutput/MeasFakeRateV2/{args.hlt}_RunSyst/{args.era}/{DATAPERIOD}.root"
            assert os.path.exists(file_path), f"{file_path} does not exist"
            f = ROOT.TFile.Open(file_path)
            try:
                h = f.Get(f"{prefix}/{args.region}/{args.wp}/{args.selection}/MT")
                h.SetDirectory(0)
            except:
                logging.warning(f"Cannot find {prefix}/{args.region}/{args.wp}/{args.selection}/MT for DATAPERIOD {DATAPERIOD}")
                f.Close()
                continue
            f.Close()
            if data is None:
                data = h.Clone()
            else:
                data.Add(h)

        for sample in MCList:
            file_path = f"{WORKDIR}/SKNanoOutput/MeasFakeRateV2/{args.hlt}_RunSyst/{args.era}/{sample}.root"
            assert os.path.exists(file_path), f"{file_path} does not exist"
            f = ROOT.TFile.Open(file_path)
            try:
                h = f.Get(f"{prefix}/{args.region}/{args.wp}/{args.selection}/MT")
                h.SetDirectory(0)
            except:
                logging.warning(f"Cannot find {prefix}/{args.region}/{args.wp}/{args.selection}/MT for sample {sample}")
                f.Close()
                continue

            hSysts = []
            for syst, sources in SYSTs.items():
                if not args.selection == "Central": continue
                systUp, systDown = sources
                try:
                    h_up = f.Get(f"{prefix}/{args.region}/{args.wp}/{systUp}/MT")
                    h_up.SetDirectory(0)
                    h_down = f.Get(f"{prefix}/{args.region}/{args.wp}/{systDown}/MT")
                    h_down.SetDirectory(0)
                except:
                    logging.warning(f"Cannot find {prefix}/{args.region}/{args.wp}/{systUp}/MT or {prefix}/{args.region}/{args.wp}/{systDown}/MT for sample {sample}")
                    f.Close()
                    continue
                hSysts.append((h_up, h_down))
            f.Close()

            # estimate total unc. bin by bin
            for bin in range(h.GetNcells()):
                stat_unc = h.GetBinError(bin)
                envelops = []
                for hset in hSysts:
                    if len(hset) == 2:
                        h_up, h_down = hset
                        envelops.append(h_up.GetBinContent(bin) - h_down.GetBinContent(bin))
                    else:
                        # only one systematic source
                        h_syst = hset
                        envelops.append(h_syst.GetBinContent(bin))
                total_unc = ROOT.TMath.Power(stat_unc, 2)
                for unc in envelops:
                    total_unc += ROOT.TMath.Power(unc, 2)
                total_unc = ROOT.TMath.Sqrt(total_unc)
                h.SetBinError(bin, total_unc)
            HISTs[sample] = h.Clone(sample)

        # now scale MC histograms
        # get normalization factors
        scale_key = f"{args.hlt}_{args.wp}_{args.selection}"
        json_path = f"{WORKDIR}/MeasFakeRateV2/results/{args.era}/JSON/{MEASURE}/prompt_scale.json"
        with open(json_path, 'r') as f:
            scale_dict = json.load(f)
        scale = scale_dict[scale_key]
        logging.debug(f"MC histograms scaled to {scale}")
        for hist in HISTs.values():
            hist.Scale(scale)

        # merge backgrounds
        temp_dict = { "W": None, "Z": None, "TT": None, "ST": None, "VV": None, "QCD_EMEnriched": None, "QCD_bcToE": None, "QCD_MuEnriched": None }
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
        for sample in QCD_EMEnriched:
            if not sample in HISTs.keys(): continue
            add_hist("QCD_EMEnriched", HISTs[sample], temp_dict)
        for sample in QCD_bcToE:
            if not sample in HISTs.keys(): continue
            add_hist("QCD_bcToE", HISTs[sample], temp_dict)
        for sample in QCD_MuEnriched:
            if not sample in HISTs.keys(): continue
            add_hist("QCD_MuEnriched", HISTs[sample], temp_dict)
    
        # filter out none histograms from temp_dict
        BKGs = {name: hist for name, hist in temp_dict.items() if hist}
        # Sort BKGs by hist.Integral()
        BKGs = dict(sorted(BKGs.items(), key=lambda x: x[1].Integral(), reverse=True))
        logging.debug(f"BKGs: {BKGs}")

        # plot
        config = {"era": args.era,
                  "xTitle": "M_{T}",
                  "yTitle": "Events / 5 GeV",
                  "xRange": [0., 200.],
                  "rRange": [0.0, 2.0],
                  "rebin": 5}

        output_path = f"{WORKDIR}/MeasFakeRateV2/plots/{args.era}/{MEASURE}/{args.hlt}/{args.region}/{args.selection}/{prefix}_{args.wp}_MT.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plotter = ComparisonCanvas(data, BKGs, config)
        plotter.drawPadUp()
        plotter.drawPadDown()
        plotter.canv.SaveAs(output_path)