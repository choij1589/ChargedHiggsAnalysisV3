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
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

WORKDIR = os.environ['WORKDIR']
if "El" in args.hlt:
    MEASURE = "electron"
    QCD = ["QCD_EMEnriched", "QCD_bcToE"]
    if args.hlt == "MeasFakeEl8":
        ptcorr_bins = [10., 15., 20., 25., 35., 50., 100.]
    elif args.hlt == "MeasFakeEl12":
        ptcorr_bins = [15., 20., 25., 35., 50., 100.]
    elif args.hlt == "MeasFakeEl23":
        ptcorr_bins = [25., 35., 50., 100.]
    else:
        raise ValueError(f"invalid hlt {args.hlt}")
    abseta_bins = [0., 0.8, 1.479, 2.5]
elif "Mu" in args.hlt:
    MEASURE = "muon"
    if args.hlt == "MeasFakeMu8":
        ptcorr_bins = [10., 15., 20., 30., 50., 100.]
    elif args.hlt == "MeasFakeMu17":
        ptcorr_bins = [20., 30., 50., 100.]
    else:
        raise ValueError(f"invalid hlt {args.hlt}")
    abseta_bins = [0., 0.9, 1.6, 2.4]
    QCD = ["QCD_MuEnriched"]
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

SYSTs = json.load(open("configs/systematics.json"))[args.era][MEASURE]

def add_hist(name, hist, histDict):
    if not histDict[name]:
        histDict[name] = hist.Clone(name)
    else:
        histDict[name].Add(hist)

for ptcorr, abseta in product(ptcorr_bins[:-1], abseta_bins[:-1]):
    prefix = findbin(ptcorr, abseta, ptcorr_bins, abseta_bins)
    HISTs = {}
    
    # data
    data = None
    for DATAPERIOD in DATAPERIODs:
        file_path = f"{WORKDIR}/SKNanoOutput/MeasFakeRate/{args.hlt}_RunSyst/{args.era}/{DATAPERIOD}.root"
        try:
            assert os.path.exists(file_path)
        except:
            raise FileNotFoundError(f"{file_path} does not exist")
        f = ROOT.TFile.Open(file_path)
        h = f.Get(f"ZEnriched/{args.wp}/{args.selection}/ZCand/mass"); h.SetDirectory(0)
        f.Close()
        if data is None:
            data = h.Clone()
        else:
            data.Add(h)

    for sample in MCList+QCD:
        file_path = f"{WORKDIR}/SKNanoOutput/MeasFakeRate/{args.hlt}_RunSyst/{args.era}/{sample}.root"
        assert os.path.exists(file_path), f"{file_path} does not exist"
        f = ROOT.TFile.Open(file_path)
        h = f.Get(f"ZEnriched/{args.wp}/{args.selection}/ZCand/mass"); h.SetDirectory(0)
        hSysts = []
        for syst, sources in SYSTs.items():
            systUp, systDown = sources
            h_up = f.Get(f"ZEnriched/{args.wp}/{systUp}/ZCand/mass"); h_up.SetDirectory(0)
            h_down = f.Get(f"ZEnriched/{args.wp}/{systDown}/ZCand/mass"); h_down.SetDirectory(0) 
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
    json_path = f"{WORKDIR}/MeasFakeRate/results/{args.era}/JSON/{MEASURE}/prompt_scale.json"
    with open(json_path, 'r') as f:
        scale_dict = json.load(f)
    scale = scale_dict[scale_key]
    logging.debug(f"MC histograms scaled to {scale}")
    for hist in HISTs.values(): 
        hist.Scale(scale)

    # merge backgrounds
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

    # plot
    config = {"era": args.era,
              "xTitle": "M_{T}",
              "yTitle": "Events / 5 GeV",
              "xRange": [0., 200.],
              "rebin": 5}

    output_path = f"{WORKDIR}/MeasFakeRate/results/{args.era}/plots/{MEASURE}/{args.region}/{args.syst}/{prefix}_{args.hlt}_{args.wp}_MT.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plotter = ComparisonCanvas(data, BKGs, config)
    plotter.drawPadUp()
    plotter.drawPadDown()
    plotter.canv.SaveAs(output_path)