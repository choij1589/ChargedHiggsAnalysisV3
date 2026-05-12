#!/usr/bin/env python
"""Overlay central vs private (test_*) signal samples for one mass point.

Output preserves absolute yields (no unit-area normalization) so that
cross-section normalization can be cross-checked alongside shape.
"""
import os
import re
import argparse
import logging
import json
from collections import OrderedDict
from types import SimpleNamespace
import ROOT
import cmsstyle as CMS
from plotter import KinematicCanvasWithRatio, get_CoM_energy, get_era_list
from HistoUtils import setup_missing_histogram_logging, load_histogram, sum_histograms
ROOT.gROOT.SetBatch(True)

INPUT_BASE = "/home/choij/Sync/workspace/SKNanoOutput_V1/SignalKinematics"

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str,
                    help="single era (e.g. 2018) or combined Run2 / Run3")
parser.add_argument("--channel", required=True, type=str,
                    choices=["SR3Mu", "SR1E2Mu", "Combined", "Inclusive"])
parser.add_argument("--mass-point", required=True, type=str,
                    help="e.g. MHc100_MA60 (must have a matching test_*.root)")
parser.add_argument("--histkey", required=True, type=str)
parser.add_argument("--debug", default=False, action="store_true")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

with open("configs/histkeys.json") as f:
    histkeys = json.load(f)

if args.histkey not in histkeys:
    logging.error(f"histkey '{args.histkey}' not found in configs/histkeys.json")
    exit(1)

# Logger writes to logs/validation/{era}/{channel}/{mass}/{histkey}.log
log_args = SimpleNamespace(
    era=f"validation/{args.era}",
    channel=f"{args.channel}/{args.mass_point}",
    histkey=args.histkey,
    debug=args.debug,
)
missing_logger = setup_missing_histogram_logging(log_args)

if args.channel == "Combined":
    channel_flags = [("SR1E2Mu", "Run1E2Mu"), ("SR3Mu", "Run3Mu")]
    channel_text = "e#mu#mu + #mu#mu#mu"
elif args.channel == "Inclusive":
    # InclusiveGen is filled before SR selection; gen content is identical
    # between Run3Mu and Run1E2Mu runs (same events). Load from Run3Mu only
    # to avoid double-counting. Absolute yields use Run3Mu trigger lumi.
    channel_flags = [("Inclusive", "Run3Mu")]
    channel_text = ""
else:
    flag = "Run3Mu" if args.channel == "SR3Mu" else "Run1E2Mu"
    channel_flags = [(args.channel, flag)]
    channel_text = args.channel

sample = f"TTToHcToWAToMuMu-{args.mass_point}"
is_gen_key = args.histkey.startswith(("GenLevel/", "InclusiveGen/"))
subdir = "GEN" if is_gen_key else "RECO"

# Load (and sum, for combined eras / channels) central and private histograms.
era_list = get_era_list(args.era)
central_parts, private_parts = [], []
for era in era_list:
    for ch, flag in channel_flags:
        if is_gen_key:
            hist_path = args.histkey
        else:
            hist_path = f"{ch}/Central/{args.histkey}"
        central_path = f"{INPUT_BASE}/{flag}/{era}/{sample}.root"
        private_path = f"{INPUT_BASE}/{flag}/{era}/test_{sample}.root"
        h_c = load_histogram(central_path, hist_path, era, missing_logger)
        h_p = load_histogram(private_path, hist_path, era, missing_logger)
        if h_c is not None:
            central_parts.append(h_c)
        if h_p is not None:
            private_parts.append(h_p)

expected = len(era_list) * len(channel_flags)
if not central_parts or not private_parts:
    logging.warning(
        f"Skipping {args.era}/{args.channel}/{args.mass_point}/{args.histkey}: "
        f"central={len(central_parts)}/{expected} "
        f"private={len(private_parts)}/{expected}"
    )
    exit(0)

h_central = sum_histograms(central_parts, f"{sample}_central")
h_private = sum_histograms(private_parts, f"{sample}_private")

# Pretty-print mass point: "MHc100_MA60" -> "M_{H^{+}}=100 GeV, M_{A}=60 GeV"
m = re.match(r"MHc(\d+)_MA(\d+)", args.mass_point)
if m:
    mass_label = f"(m_{{H^{{+}}}}, m_{{A}}) = ({m.group(1)}, {m.group(2)}) GeV"
else:
    mass_label = args.mass_point

config = dict(histkeys[args.histkey])
config["era"] = args.era
config["CoM"] = get_CoM_energy(args.era)
config["channel"] = channel_text
config["channelFont"] = 62
config["channelSize"] = 0.04
config["overflow"] = True
config["normalize"] = False  # keep absolute yields to validate xsec scaling
config["yTitle"] = "Events"
config["rTitle"] = "private / central"
config["rRange"] = [0.5, 1.5]
config["legendTextSize"] = 0.06
config["legend"] = (0.6, 0.60, 0.95, 0.85)
config["iPos"] = 11

hists = OrderedDict([
    ("central", h_central),
    ("private", h_private),
])

OUTPUTPATH = (
    f"plots/validation/{args.era}/{args.channel}/{args.mass_point}/{subdir}/"
    f"{args.histkey.replace('/', '_')}.png"
)
os.makedirs(os.path.dirname(OUTPUTPATH), exist_ok=True)

plotter = KinematicCanvasWithRatio(hists, config)
plotter.drawPadUp()
plotter.canv.cd(1)
CMS.drawText(mass_label, posX=0.2, posY=0.63, font=42, align=0, size=0.04)
CMS.drawText(subdir, posX=0.2, posY=0.57, font=61, align=0, size=0.04)
plotter.drawPadDown()
plotter.canv.SaveAs(OUTPUTPATH)
logging.info(f"Saved: {OUTPUTPATH}")
