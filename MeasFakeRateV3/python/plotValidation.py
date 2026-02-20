#!/usr/bin/env python
import os
import argparse
import logging
import ROOT
import json
from itertools import product
from plotter import ComparisonCanvas, get_era_list, get_CoM_energy
import cmsstyle as CMS
from common import findbin

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--hlt", required=True, type=str, help="hlt")
parser.add_argument("--wp", required=True, type=str, help="wp")
parser.add_argument("--region", required=True, type=str, help="region")
parser.add_argument("--selection", default="Central", type=str, help="selection")
parser.add_argument("--histkey", default="MT", type=str, help="histogram key (MT, pt, eta, scEta, MET)")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

HIST_CONFIGS = json.load(open("configs/histkeys.json"))

WORKDIR = os.environ['WORKDIR']
if "El" in args.hlt:
    MEASURE = "electron"
    if args.hlt == "MeasFakeEl8":
        ptcorr_bins = [15., 17., 20., 25., 35., 50., 100., 200.]
    elif args.hlt == "MeasFakeEl12":
        ptcorr_bins = [17., 20., 25., 35., 50., 100., 200.]
    elif args.hlt == "MeasFakeEl23":
        ptcorr_bins = [25., 35., 50., 100., 200.]
    else:
        raise ValueError(f"invalid hlt {args.hlt}")
    abseta_bins = [0., 0.8, 1.479, 2.5]
elif "Mu" in args.hlt:
    MEASURE = "muon"
    if args.hlt == "MeasFakeMu8":
        ptcorr_bins = [10., 12., 14., 17., 20., 30., 50., 100., 200.]
    elif args.hlt == "MeasFakeMu17":
        ptcorr_bins = [20., 30., 50., 100., 200.]
    else:
        raise ValueError(f"invalid hlt {args.hlt}")
    abseta_bins = [0., 0.9, 1.6, 2.4]
else:
    raise ValueError(f"invalid hlt {args.hlt}")

# Histograms stored in Inclusive directory (not per ptcorr bin)
INCLUSIVE_HISTKEYS = ["pt", "eta", "scEta", "MET", "MT"]

# ZEnriched histkeys - use directly without MEASURE prefix
ZENRICHED_HISTKEYS = ["muons/1/pt", "muons/1/eta", "muons/2/pt", "muons/2/eta",
                      "ZCand/mass", "ZCand/pt", "ZCand/eta"]

# Map user-friendly histkey to actual histogram name (MEASURE-specific)
# electrons: eta -> scEta (no eta histogram for electrons)
# muons: eta -> eta
HISTKEY_MAP = {
    "electron": {"eta": "scEta"},
    "muon": {},
}

# Build histogram path based on histkey type and region
IS_ZENRICHED = (args.region == "ZEnriched")

if IS_ZENRICHED:
    # ZEnriched uses histkeys directly (muons/1/pt, ZCand/mass, etc.)
    IS_INCLUSIVE = True  # ZEnriched doesn't have per-bin structure
    HISTKEY = args.histkey
elif args.histkey in INCLUSIVE_HISTKEYS:
    IS_INCLUSIVE = True
    if args.histkey in ["MT", "MET"]:
        HISTKEY = args.histkey
    else:
        measure_map = HISTKEY_MAP.get(MEASURE, {})
        actual_histkey = measure_map.get(args.histkey, args.histkey)
        HISTKEY = f"{MEASURE}/{actual_histkey}"
else:
    IS_INCLUSIVE = False
    HISTKEY = args.histkey

trigPathDict = {
    "MeasFakeMu8": "HLT_Mu8_TrkIsoVVL_v",
    "MeasFakeMu17": "HLT_Mu17_TrkIsoVVL_v",
    "MeasFakeEl8": "HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30_v",
    "MeasFakeEl12": "HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30_v",
    "MeasFakeEl23": "HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30_v"
}

shortHLTDict = {
    "MeasFakeMu8": "HLT_Mu8_TrkIsoVVL",
    "MeasFakeMu17": "HLT_Mu17_TrkIsoVVL",
    "MeasFakeEl8": "HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30",
    "MeasFakeEl12": "HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30",
    "MeasFakeEl23": "HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30"
}

# Handle merged eras (Run2/Run3)
era_list = get_era_list(args.era)
logging.info(f"Processing {args.era} with eras: {era_list}")

# Load sample groups and systematics for all eras
SAMPLEGROUPS_JSON = json.load(open("configs/samplegroup.json"))
SYSTEMATICS_JSON = json.load(open("configs/systematics.json"))

# Collect data periods and MC samples from all eras
# Format: list of (era, sample) tuples
DATAPERIODs = []  # list of (era, data_period) tuples
W = []            # list of (era, sample) tuples
Z = []
TT = []
VV = []
ST = []
QCD_EMEnriched = []
QCD_bcToE = []
QCD_MuEnriched = []

for era in era_list:
    era_samplegroup = SAMPLEGROUPS_JSON[era][MEASURE]
    DATAPERIODs.extend([(era, d) for d in era_samplegroup["data"]])
    W.extend([(era, s) for s in era_samplegroup["W"]])
    Z.extend([(era, s) for s in era_samplegroup["Z"]])
    TT.extend([(era, s) for s in era_samplegroup["TT"]])
    VV.extend([(era, s) for s in era_samplegroup["VV"]])
    ST.extend([(era, s) for s in era_samplegroup["ST"]])
    if MEASURE == "electron":
        QCD_EMEnriched.extend([(era, s) for s in era_samplegroup["QCD_EMEnriched"]])
        QCD_bcToE.extend([(era, s) for s in era_samplegroup["QCD_bcToE"]])
    elif MEASURE == "muon":
        QCD_MuEnriched.extend([(era, s) for s in era_samplegroup["QCD_MuEnriched"]])

MCList = W + Z + TT + VV + ST
if MEASURE == "electron":
    MCList += QCD_EMEnriched + QCD_bcToE
elif MEASURE == "muon":
    MCList += QCD_MuEnriched
else:
    raise KeyError(f"Wrong measure {MEASURE}")

# Collect systematics from all eras (use first era's systematics as reference)
SYSTs = SYSTEMATICS_JSON[era_list[0]][MEASURE]

def add_hist(name, hist, histDict):
    if not histDict[name]:
        histDict[name] = hist.Clone(name)
    else:
        histDict[name].Add(hist)

# Use MeasFakeRateV3 for both muon and electron
analyzer = "MeasFakeRateV3"

def get_histogram_path(prefix, is_inclusive):
    """Build histogram path based on whether it's inclusive or per-bin"""
    if is_inclusive:
        return f"{args.region}/{args.wp}/{args.selection}/{HISTKEY}"
    else:
        return f"{prefix}/{args.region}/{args.wp}/{args.selection}/{HISTKEY}"

def get_syst_path(prefix, syst, is_inclusive):
    """Build systematic histogram path"""
    if is_inclusive:
        return f"{args.region}/{args.wp}/{syst}/{HISTKEY}"
    else:
        return f"{prefix}/{args.region}/{args.wp}/{syst}/{HISTKEY}"

def process_bin(prefix):
    """Process a single bin (or inclusive if prefix is None)"""
    HISTs = {}  # key: (era, sample) tuple
    is_inclusive = (prefix is None)

    # data - now using (era, data_period) tuples
    data = None
    for era, dataperiod in DATAPERIODs:
        file_path = f"{WORKDIR}/SKNanoOutput/{analyzer}/{args.hlt}_RunSyst/{era}/{dataperiod}.root"
        if not os.path.exists(file_path):
            logging.warning(f"{file_path} does not exist")
            continue
        f = ROOT.TFile.Open(file_path)
        hist_path = get_histogram_path(prefix, is_inclusive)
        try:
            h = f.Get(hist_path)
            h.SetDirectory(0)
        except:
            logging.warning(f"Cannot find {hist_path} for {era}/{dataperiod}")
            f.Close()
            continue
        f.Close()
        if data is None:
            data = h.Clone()
        else:
            data.Add(h)

    # MC - now using (era, sample) tuples
    for era, sample in MCList:
        file_path = f"{WORKDIR}/SKNanoOutput/{analyzer}/{args.hlt}_RunSyst/{era}/{sample}.root"
        if not os.path.exists(file_path):
            logging.warning(f"{file_path} does not exist")
            continue
        f = ROOT.TFile.Open(file_path)
        hist_path = get_histogram_path(prefix, is_inclusive)
        try:
            h = f.Get(hist_path)
            h.SetDirectory(0)
        except:
            logging.warning(f"Cannot find {hist_path} for {era}/{sample}")
            f.Close()
            continue

        # Get systematics for this era
        era_systs = SYSTEMATICS_JSON.get(era, {}).get(MEASURE, {})
        hSysts = []
        for syst, sources in era_systs.items():
            if not args.selection == "Central": continue
            systUp, systDown = sources
            try:
                h_up = f.Get(get_syst_path(prefix, systUp, is_inclusive))
                h_up.SetDirectory(0)
                h_down = f.Get(get_syst_path(prefix, systDown, is_inclusive))
                h_down.SetDirectory(0)
            except:
                logging.warning(f"Cannot find syst {systUp}/{systDown} for {era}/{sample}")
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
                    h_syst = hset
                    envelops.append(h_syst.GetBinContent(bin))
            total_unc = ROOT.TMath.Power(stat_unc, 2)
            for unc in envelops:
                total_unc += ROOT.TMath.Power(unc, 2)
            total_unc = ROOT.TMath.Sqrt(total_unc)
            h.SetBinError(bin, total_unc)
        HISTs[(era, sample)] = h.Clone(f"{era}_{sample}")

    # now scale MC histograms (per-era prompt scale)
    scale_key = f"{args.hlt}_{args.wp}_{args.selection}"
    for (era, sample), hist in HISTs.items():
        json_path = f"{WORKDIR}/MeasFakeRateV3/results/{era}/JSON/{MEASURE}/prompt_scale.json"
        with open(json_path, 'r') as f:
            scale_dict = json.load(f)
        scale = scale_dict[scale_key]
        logging.debug(f"Scaling {era}/{sample} by {scale}")
        hist.Scale(scale)

    # merge backgrounds - W, Z, TT, etc. are now lists of (era, sample) tuples
    temp_dict = { "W": None, "Z": None, "TT": None, "ST": None, "VV": None, "QCD_EMEnriched": None, "QCD_bcToE": None, "QCD_MuEnriched": None }
    for era_sample in W:
        if era_sample not in HISTs.keys(): continue
        add_hist("W", HISTs[era_sample], temp_dict)
    for era_sample in Z:
        if era_sample not in HISTs.keys(): continue
        add_hist("Z", HISTs[era_sample], temp_dict)
    for era_sample in TT:
        if era_sample not in HISTs.keys(): continue
        add_hist("TT", HISTs[era_sample], temp_dict)
    for era_sample in VV:
        if era_sample not in HISTs.keys(): continue
        add_hist("VV", HISTs[era_sample], temp_dict)
    for era_sample in ST:
        if era_sample not in HISTs.keys(): continue
        add_hist("ST", HISTs[era_sample], temp_dict)
    for era_sample in QCD_EMEnriched:
        if era_sample not in HISTs.keys(): continue
        add_hist("QCD_EMEnriched", HISTs[era_sample], temp_dict)
    for era_sample in QCD_bcToE:
        if era_sample not in HISTs.keys(): continue
        add_hist("QCD_bcToE", HISTs[era_sample], temp_dict)
    for era_sample in QCD_MuEnriched:
        if era_sample not in HISTs.keys(): continue
        add_hist("QCD_MuEnriched", HISTs[era_sample], temp_dict)

    # filter out none histograms from temp_dict
    BKGs = {name: hist for name, hist in temp_dict.items() if hist}
    BKGs = dict(sorted(BKGs.items(), key=lambda x: x[1].Integral(), reverse=True))
    logging.debug(f"BKGs: {BKGs}")

    return data, BKGs

# Main logic
if IS_INCLUSIVE:
    # For inclusive histkeys (pt, eta, scEta, MET), process once
    data, BKGs = process_bin(None)

    if data is None:
        raise RuntimeError(f"No data histogram found for {args.region}/{args.wp}/{args.selection}/{HISTKEY}")
    if not BKGs:
        raise RuntimeError(f"No MC histograms found for {args.region}/{args.wp}/{args.selection}/{HISTKEY}")

    # plot
    config = HIST_CONFIGS[HISTKEY].copy()
    config["era"] = args.era
    config["CoM"] = get_CoM_energy(args.era)
    config["prescaled"] = True  # triggers are prescaled, don't show lumi
    config["rRange"] = [0.0, 2.0]

    histkey_safe = args.histkey.replace("/", "_")
    output_path = f"{WORKDIR}/MeasFakeRateV3/plots/{args.era}/{MEASURE}/{args.hlt}/{args.region}/{args.selection}/{args.wp}_{histkey_safe}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plotter = ComparisonCanvas(data, BKGs, config)
    plotter.drawPadUp()

    # Add HLT + WP info text
    plotter.canv.cd(1)
    CMS.drawText(shortHLTDict[args.hlt], posX=0.20, posY=0.70, font=42, align=0, size=0.03)
    CMS.drawText(f"{args.wp} ID", posX=0.20, posY=0.65, font=42, align=0, size=0.03)

    plotter.drawPadDown()
    plotter.canv.SaveAs(output_path)
else:
    # For per-bin histkeys (MT), iterate over ptcorr/abseta bins
    for ptcorr, abseta in product(ptcorr_bins[:-1], abseta_bins[:-1]):
        prefix = findbin(ptcorr, abseta, ptcorr_bins, abseta_bins)
        data, BKGs = process_bin(prefix)

        if data is None:
            logging.warning(f"No data histogram found for {prefix}, skipping")
            continue
        if not BKGs:
            logging.warning(f"No MC histograms found for {prefix}, skipping")
            continue

        # plot
        config = HIST_CONFIGS[HISTKEY].copy()
        config["era"] = args.era
        config["CoM"] = get_CoM_energy(args.era)
        config["prescaled"] = True  # triggers are prescaled, don't show lumi
        config["rRange"] = [0.0, 2.0]

        output_path = f"{WORKDIR}/MeasFakeRateV3/plots/{args.era}/{MEASURE}/{args.hlt}/{args.region}/{args.selection}/{prefix}_{args.wp}_{args.histkey}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plotter = ComparisonCanvas(data, BKGs, config)
        plotter.drawPadUp()

        # Add HLT + WP info text
        plotter.canv.cd(1)
        CMS.drawText(shortHLTDict[args.hlt], posX=0.20, posY=0.70, font=42, align=0, size=0.03)
        CMS.drawText(f"{args.wp} ID", posX=0.20, posY=0.65, font=42, align=0, size=0.03)

        plotter.drawPadDown()
        plotter.canv.SaveAs(output_path)
