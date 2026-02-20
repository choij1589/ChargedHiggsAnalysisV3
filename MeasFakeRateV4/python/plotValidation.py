#!/usr/bin/env python
import os
import argparse
import logging
import ROOT
import json
from itertools import product
from plotter import ComparisonCanvas, PALETTE_LONG, get_era_list, get_CoM_energy
import cmsstyle as CMS
from common import findbin

# Fixed color assignment per background
BKG_COLORS = {
    "W":        PALETTE_LONG[0],
    "Z":        PALETTE_LONG[1],
    "TT":       PALETTE_LONG[2],
    "ST":       PALETTE_LONG[3],
    "VV":       PALETTE_LONG[4],
    "QCD(e)":  PALETTE_LONG[5],
    "QCD(b)":  PALETTE_LONG[6],
    "QCD(#mu)": PALETTE_LONG[7],
}

ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--hlt", required=True, type=str, help="trigger prefix (El8/El12/El23/Mu8/Mu17)")
parser.add_argument("--wp", required=True, type=str, help="wp")
parser.add_argument("--region", required=True, type=str, help="region (Inclusive/ZEnriched)")
parser.add_argument("--selection", default="Central", type=str, help="selection")
parser.add_argument("--histkey", default="MT", type=str, help="histogram key (MT, pt, eta, scEta, MET, ZCand/mass, ...)")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

HIST_CONFIGS = json.load(open("configs/histkeys.json"))

WORKDIR = os.environ['WORKDIR']
if "El" in args.hlt:
    MEASURE = "electron"
    if args.hlt == "El8":
        ptcorr_bins = [15., 17., 20., 25., 35., 50., 100., 200.]
    elif args.hlt == "El12":
        ptcorr_bins = [17., 20., 25., 35., 50., 100., 200.]
    elif args.hlt == "El23":
        ptcorr_bins = [25., 35., 50., 100., 200.]
    else:
        raise ValueError(f"invalid hlt {args.hlt}")
    abseta_bins = [0., 0.8, 1.479, 2.5]
elif "Mu" in args.hlt:
    MEASURE = "muon"
    if args.hlt == "Mu8":
        ptcorr_bins = [10., 12., 14., 17., 20., 30., 50., 100., 200.]
    elif args.hlt == "Mu17":
        ptcorr_bins = [20., 30., 50., 100., 200.]
    else:
        raise ValueError(f"invalid hlt {args.hlt}")
    abseta_bins = [0., 0.9, 1.6, 2.4]
else:
    raise ValueError(f"invalid hlt {args.hlt}")

# Histograms stored in Inclusive directory (not per ptcorr bin)
INCLUSIVE_HISTKEYS = ["pt", "eta", "scEta", "MET", "MT"]

# Map user-friendly histkey to actual histogram name
HISTKEY_MAP = {
    "electron": {"eta": "scEta"},
    "muon": {},
}

# Build histogram path based on histkey type and region
IS_ZENRICHED = (args.region == "ZEnriched")

if IS_ZENRICHED:
    IS_INCLUSIVE = True
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

shortHLTDict = {
    "Mu8": "HLT_Mu8_TrkIsoVVL",
    "Mu17": "HLT_Mu17_TrkIsoVVL",
    "El8": "HLT_Ele8_CaloIdL_TrackIdL_IsoVL_PFJet30",
    "El12": "HLT_Ele12_CaloIdL_TrackIdL_IsoVL_PFJet30",
    "El23": "HLT_Ele23_CaloIdL_TrackIdL_IsoVL_PFJet30"
}

# Handle merged eras (Run2/Run3)
era_list = get_era_list(args.era)
logging.info(f"Processing {args.era} with eras: {era_list}")

# Load sample groups for all eras
SAMPLEGROUPS_JSON = json.load(open("configs/samplegroup.json"))

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
        QCD_EMEnriched.extend([(era, s) for s in era_samplegroup.get("QCD_EMEnriched", [])])
        QCD_bcToE.extend([(era, s) for s in era_samplegroup.get("QCD_bcToE", [])])
    elif MEASURE == "muon":
        QCD_MuEnriched.extend([(era, s) for s in era_samplegroup.get("QCD_MuEnriched", [])])

PromptMCList = W + Z + TT + VV + ST
QCD = QCD_EMEnriched + QCD_bcToE + QCD_MuEnriched

# V4 consolidated lepton type file
lepton_type = "MeasFakeEl" if MEASURE == "electron" else "MeasFakeMu"

def add_hist(name, hist, histDict):
    if not histDict[name]:
        histDict[name] = hist.Clone(name)
    else:
        histDict[name].Add(hist)

def get_histogram_path(prefix, is_inclusive):
    """Build histogram path for V4 structure: {region}/{trigPrefix}/{wp}/{syst}/{histkey}"""
    if is_inclusive:
        return f"{args.region}/{args.hlt}/{args.wp}/{args.selection}/{HISTKEY}"
    else:
        return f"{prefix}/{args.region}/{args.hlt}/{args.wp}/{args.selection}/{HISTKEY}"

def process_bin(prefix):
    """Process a single bin (or inclusive if prefix is None)"""
    HISTs = {}  # key: (era, sample) tuple
    is_inclusive = (prefix is None)

    # data - now using (era, data_period) tuples
    data = None
    for era, dataperiod in DATAPERIODs:
        file_path = f"{WORKDIR}/SKNanoOutput/MeasFakeRateV4/{lepton_type}_RunSyst/{era}/{dataperiod}.root"
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
    if data is not None:
        data.SetTitle("Data")

    # Prompt MC (from _RunSyst files, stat errors only) - now using (era, sample) tuples
    for era, sample in PromptMCList:
        file_path = f"{WORKDIR}/SKNanoOutput/MeasFakeRateV4/{lepton_type}_RunSyst/{era}/{sample}.root"
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
        f.Close()
        HISTs[(era, sample)] = h.Clone(f"{era}_{sample}")

    # QCD MC (from non-_RunSyst files, no systematics) - now using (era, sample) tuples
    for era, sample in QCD:
        file_path = f"{WORKDIR}/SKNanoOutput/MeasFakeRateV4/{lepton_type}/{era}/{sample}.root"
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
        f.Close()
        HISTs[(era, sample)] = h.Clone(f"{era}_{sample}")

    # scale all MC histograms using per-era prompt_scale
    scale_key = f"{args.hlt}_{args.wp}_{args.selection}"
    for (era, sample), hist in HISTs.items():
        json_path = f"{WORKDIR}/MeasFakeRateV4/results/{era}/JSON/{MEASURE}/prompt_scale.json"
        with open(json_path, 'r') as f:
            scale_dict = json.load(f)
        scale = scale_dict[scale_key]
        logging.debug(f"Scaling {era}/{sample} by {scale}")
        hist.Scale(scale)

    # merge backgrounds - W, Z, TT, etc. are now lists of (era, sample) tuples
    temp_dict = { "W": None, "Z": None, "TT": None, "ST": None, "VV": None,
                  "QCD(e)": None, "QCD(b)": None, "QCD(#mu)": None }
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
        add_hist("QCD(e)", HISTs[era_sample], temp_dict)
    for era_sample in QCD_bcToE:
        if era_sample not in HISTs.keys(): continue
        add_hist("QCD(b)", HISTs[era_sample], temp_dict)
    for era_sample in QCD_MuEnriched:
        if era_sample not in HISTs.keys(): continue
        add_hist("QCD(#mu)", HISTs[era_sample], temp_dict)

    # filter out none histograms from temp_dict
    BKGs = {name: hist for name, hist in temp_dict.items() if hist}
    BKGs = dict(sorted(BKGs.items(), key=lambda x: x[1].Integral(), reverse=True))
    logging.debug(f"BKGs: {BKGs}")

    return data, BKGs

# Main logic
if IS_INCLUSIVE:
    data, BKGs = process_bin(None)

    if data is None:
        raise RuntimeError(f"No data histogram found for {args.region}/{args.hlt}/{args.wp}/{args.selection}/{HISTKEY}")
    if not BKGs:
        raise RuntimeError(f"No MC histograms found for {args.region}/{args.hlt}/{args.wp}/{args.selection}/{HISTKEY}")

    config = HIST_CONFIGS[HISTKEY].copy()
    config["era"] = args.era
    config["CoM"] = get_CoM_energy(args.era)
    config["prescaled"] = True
    config["rRange"] = [0.0, 2.0]
    config["systSrc"] = "Stat"
    nBKGs = len(BKGs)
    config["colors"] = [BKG_COLORS[name] for name in BKGs.keys()]
    config["legend"] = (0.75, 0.87 - 0.04 * (nBKGs + 2), 0.95, 0.87)

    histkey_safe = args.histkey.replace("/", "_")
    output_path = f"{WORKDIR}/MeasFakeRateV4/plots/{args.era}/{MEASURE}/{args.hlt}/{args.region}/{args.selection}/{args.wp}_{histkey_safe}.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plotter = ComparisonCanvas(data, BKGs, config)
    plotter.drawPadUp()

    plotter.canv.cd(1)
    CMS.drawText(shortHLTDict.get(args.hlt, args.hlt), posX=0.18, posY=0.72, font=42, align=0, size=0.04)
    CMS.drawText(f"{args.wp} ID", posX=0.18, posY=0.68, font=42, align=0, size=0.04)

    plotter.drawPadDown()
    plotter.canv.SaveAs(output_path)
else:
    for ptcorr, abseta in product(ptcorr_bins[:-1], abseta_bins[:-1]):
        prefix = findbin(ptcorr, abseta, ptcorr_bins, abseta_bins)
        data, BKGs = process_bin(prefix)

        if data is None:
            logging.warning(f"No data histogram found for {prefix}, skipping")
            continue
        if not BKGs:
            logging.warning(f"No MC histograms found for {prefix}, skipping")
            continue

        config = HIST_CONFIGS[HISTKEY].copy()
        config["era"] = args.era
        config["CoM"] = get_CoM_energy(args.era)
        config["prescaled"] = True
        config["rRange"] = [0.0, 2.0]
        config["systSrc"] = "Stat"
        nBKGs = len(BKGs)
        config["colors"] = [BKG_COLORS[name] for name in BKGs.keys()]
        config["legend"] = (0.75, 0.87 - 0.04 * (nBKGs + 2), 0.95, 0.87)

        output_path = f"{WORKDIR}/MeasFakeRateV4/plots/{args.era}/{MEASURE}/{args.hlt}/{args.region}/{args.selection}/{prefix}_{args.wp}_{args.histkey}.png"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        plotter = ComparisonCanvas(data, BKGs, config)
        plotter.drawPadUp()

        plotter.canv.cd(1)
        CMS.drawText(shortHLTDict.get(args.hlt, args.hlt), posX=0.20, posY=0.70, font=42, align=0, size=0.03)
        CMS.drawText(f"{args.wp} ID", posX=0.20, posY=0.65, font=42, align=0, size=0.03)

        plotter.drawPadDown()
        plotter.canv.SaveAs(output_path)
