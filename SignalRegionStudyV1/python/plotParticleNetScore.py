#!/usr/bin/env python3
"""
Plot ParticleNet score distributions for signal and backgrounds.

This script creates diagnostic plots showing the Modified ParticleNet Score
distribution for signal vs stacked backgrounds after mass window selection.

Usage:
    plotParticleNetScore.py --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --binning uniform
"""
import os
import sys
import logging
import argparse
import json
import ROOT
import numpy as np

# Argument parsing
parser = argparse.ArgumentParser(description="Plot ParticleNet score distributions for signal and backgrounds")
parser.add_argument("--era", required=True, type=str, help="Data-taking period (2016preVFP, 2017, 2018, 2022, etc.)")
parser.add_argument("--channel", required=True, type=str, help="Analysis channel (SR1E2Mu, SR3Mu)")
parser.add_argument("--masspoint", required=True, type=str, help="Signal mass point (e.g., MHc130_MA90)")
parser.add_argument("--binning", default="uniform", choices=["uniform", "extended"],
                    help="Binning method: 'uniform' (15 bins, default) or 'extended' (15 bins + tails)")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args = parser.parse_args()

# Setup logging
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                    format='%(levelname)s - %(message)s')

# Path setup
WORKDIR = os.getenv("WORKDIR")
if not WORKDIR:
    raise EnvironmentError("WORKDIR environment variable not set. Please run 'source setup.sh'")

# Add path to Common/Tools for plotter imports
sys.path.insert(0, f"{WORKDIR}/Common/Tools")
from plotter import ComparisonCanvas, get_CoM_energy
import cmsstyle as CMS

# Input from samples directory (ParticleNet scores are in preprocessed samples)
BASEDIR = f"{WORKDIR}/SignalRegionStudyV1/samples/{args.era}/{args.channel}/{args.masspoint}"
# Output to ParticleNet template directory
OUTDIR = f"{WORKDIR}/SignalRegionStudyV1/templates/{args.era}/{args.channel}/{args.masspoint}/ParticleNet/{args.binning}"

logging.info(f"Input directory: {BASEDIR}")
logging.info(f"Output directory: {OUTDIR}")

# Load mass window from binning.json
binning_path = f"{OUTDIR}/binning.json"
if not os.path.exists(binning_path):
    raise FileNotFoundError(
        f"Binning results not found: {binning_path}\n"
        f"Please run makeBinnedTemplates.py first:\n"
        f"  makeBinnedTemplates.py --era {args.era} --channel {args.channel} --masspoint {args.masspoint} --method ParticleNet --binning {args.binning}"
    )

with open(binning_path, 'r') as f:
    binning_params = json.load(f)

MASS_MIN = binning_params["mass_min"]
MASS_MAX = binning_params["mass_max"]

logging.info(f"Mass window from binning: [{MASS_MIN:.2f}, {MASS_MAX:.2f}] GeV")

# Load background weights
weights_json_path = f"{OUTDIR}/background_weights.json"
BG_WEIGHTS = None
if os.path.exists(weights_json_path):
    with open(weights_json_path, 'r') as f:
        weights_data = json.load(f)
        BG_WEIGHTS = weights_data.get("weights", None)
        logging.info(f"Loaded background weights: {BG_WEIGHTS}")
else:
    logging.warning(f"Background weights not found: {weights_json_path}")
    logging.warning(f"Using unweighted ParticleNet score (equal priors)")

# Load process list for dynamic background categories
process_list_path = f"{OUTDIR}/process_list.json"
if not os.path.exists(process_list_path):
    raise FileNotFoundError(
        f"Process list not found: {process_list_path}\n"
        f"Please run makeBinnedTemplates.py first"
    )

with open(process_list_path, 'r') as f:
    process_list = json.load(f)

SEPARATE_PROCESSES = process_list.get("separate_processes", ["nonprompt"])
MERGED_TO_OTHERS = process_list.get("merged_to_others", [])

logging.info(f"Separate processes: {SEPARATE_PROCESSES}")
logging.info(f"Merged to others: {MERGED_TO_OTHERS}")

# Setup ROOT for batch mode
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# Color palette for dynamic background assignment
COLOR_PALETTE = [
    ROOT.TColor.GetColor("#5790fc"),
    ROOT.TColor.GetColor("#f89c20"),
    ROOT.TColor.GetColor("#e42536"),
    ROOT.TColor.GetColor("#964a8b"),
    ROOT.TColor.GetColor("#9c9ca1"),
    ROOT.TColor.GetColor("#7a21dd")
]

# Special colors for known categories
KNOWN_COLORS = {
    "nonprompt": ROOT.kAzure + 2,
    "others": ROOT.kGray + 1,
}


def get_color(process, idx):
    """Get color for process - known colors first, then palette for others."""
    if process in KNOWN_COLORS:
        return KNOWN_COLORS[process]
    return COLOR_PALETTE[idx % len(COLOR_PALETTE)]


def loadScores(process, masspoint):
    """
    Load ParticleNet scores from a process sample.

    Args:
        process: Process name
        masspoint: Signal mass point

    Returns:
        Tuple of (scores, weights) as numpy arrays
    """
    file_path = f"{BASEDIR}/{process}.root"

    if not os.path.exists(file_path):
        logging.warning(f"Sample file not found: {file_path}")
        return np.array([]), np.array([])

    # Open file and get tree
    rfile = ROOT.TFile.Open(file_path, "READ")
    tree = rfile.Get("Central")

    if not tree:
        logging.warning(f"Central tree not found in {file_path}")
        rfile.Close()
        return np.array([]), np.array([])

    # Branch names
    score_sig = f"score_{masspoint}_signal"
    score_nonprompt = f"score_{masspoint}_nonprompt"
    score_diboson = f"score_{masspoint}_diboson"
    score_ttZ = f"score_{masspoint}_ttZ"

    # Check if ParticleNet scores exist
    branches = [b.GetName() for b in tree.GetListOfBranches()]
    if score_sig not in branches:
        logging.warning(f"ParticleNet scores not found in {file_path}. This is expected for untrained mass points.")
        rfile.Close()
        return np.array([]), np.array([])

    # Load data
    scores_list = []
    weights_list = []

    for entry in range(tree.GetEntries()):
        tree.GetEntry(entry)

        mass = tree.mass

        # Apply mass window cut
        if not (MASS_MIN <= mass <= MASS_MAX):
            continue

        s0 = getattr(tree, score_sig)
        s1 = getattr(tree, score_nonprompt)
        s2 = getattr(tree, score_diboson)
        s3 = getattr(tree, score_ttZ)
        weight = getattr(tree, "weight")

        # Calculate ParticleNet likelihood ratio with cross-section weights
        if BG_WEIGHTS:
            w1 = BG_WEIGHTS.get("nonprompt", 1.0)
            w2 = BG_WEIGHTS.get("diboson", 1.0)
            w3 = BG_WEIGHTS.get("ttX", 1.0)
            score_denom = s0 + w1*s1 + w2*s2 + w3*s3
        else:
            # Use unweighted likelihood ratio (equal priors)
            score_denom = s0 + s1 + s2 + s3

        if score_denom > 0:
            score_PN = s0 / score_denom
        else:
            score_PN = 0.0

        scores_list.append(score_PN)
        weights_list.append(weight)

    rfile.Close()

    return np.array(scores_list), np.array(weights_list)


def createHistogram(process, masspoint, name, color):
    """
    Create histogram of ParticleNet scores for a process.

    Args:
        process: Process name
        masspoint: Signal mass point
        name: Histogram display name
        color: ROOT color

    Returns:
        TH1D histogram
    """
    scores, weights = loadScores(process, masspoint)

    if len(scores) == 0:
        # Return empty histogram
        hist = ROOT.TH1D(process, name, 50, 0., 1.)
        hist.SetLineColor(color)
        hist.SetFillColor(color)
        return hist

    # Create histogram
    hist = ROOT.TH1D(process, name, 50, 0., 1.)
    hist.SetLineColor(color)
    hist.SetFillColor(color)

    for score, weight in zip(scores, weights):
        hist.Fill(score, weight)

    return hist


# Main execution
if __name__ == "__main__":
    logging.info(f"Plotting Modified ParticleNet scores for {args.masspoint}, {args.era}, {args.channel}")

    # Create histograms for backgrounds with dynamic colors
    bkg_hists = {}
    color_idx = 0

    for process in SEPARATE_PROCESSES:
        color = get_color(process, color_idx)
        if process not in KNOWN_COLORS:
            color_idx += 1
        display_name = process.capitalize()
        hist = createHistogram(process, args.masspoint, display_name, color)
        if hist.Integral() > 0:
            bkg_hists[process] = hist
            logging.info(f"  {process}: {hist.Integral():.2f} events")

    # Always include "others"
    hist_others = createHistogram("others", args.masspoint, "Others", get_color("others", 0))
    if hist_others.Integral() > 0:
        bkg_hists["others"] = hist_others
        logging.info(f"  others: {hist_others.Integral():.2f} events")

    # Create signal histogram (not stacked)
    hist_signal = createHistogram(args.masspoint, args.masspoint, f"Signal ({args.masspoint})", ROOT.kRed + 1)

    # Check if we have any data
    if len(bkg_hists) == 0 and hist_signal.Integral() == 0:
        logging.error("No ParticleNet scores found! This masspoint may not be in the training region (80 < mA < 100).")
        logging.error("ParticleNet scores are only available for trained mass points.")
        sys.exit(1)

    # Create "data_obs" histogram (sum of backgrounds for blind analysis)
    data_obs = ROOT.TH1D("data_obs", "data_obs", 50, 0., 1.)
    for hist in bkg_hists.values():
        data_obs.Add(hist)

    # Configuration for ComparisonCanvas
    config = {
        "era": args.era,
        "CoM": get_CoM_energy(args.era),
        "channel": f"{args.channel}",
        "xTitle": "Modified ParticleNet Score",
        "yTitle": "Events",
        "rTitle": "Data / Pred",
        "maxDigits": 3,
    }

    # Create plot
    plotter = ComparisonCanvas(data_obs, bkg_hists, config)
    plotter.drawPadUp()

    # Draw signal on upper pad
    plotter.canv.cd(1)
    hist_signal.Scale(6.0)  # ~30 fb
    hist_signal.SetLineColor(ROOT.kBlack)
    hist_signal.SetLineWidth(2)
    hist_signal.SetLineStyle(1)  # Solid line
    hist_signal.SetFillStyle(0)  # No fill
    hist_signal.Draw("HIST SAME")

    # Add signal to legend
    current_pad = ROOT.gPad
    primitives = current_pad.GetListOfPrimitives()
    for obj in primitives:
        if obj.InheritsFrom("TLegend"):
            obj.AddEntry(hist_signal, f"signal (30fb)", "l")
            break

    plotter.drawPadDown()

    # Draw additional text
    plotter.canv.cd()
    CMS.drawText(f"{args.masspoint}", posX=0.2, posY=0.76, font=61, align=0, size=0.03)

    # Save
    output_path = f"{OUTDIR}/score_distribution.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plotter.canv.SaveAs(output_path)

    logging.info(f"Modified ParticleNet score plot saved: {output_path}")
    logging.info(f"Signal yield: {hist_signal.Integral():.2f}")
    logging.info(f"Background yield: {data_obs.Integral():.2f}")
    logging.info(f"S/B ratio: {hist_signal.Integral()/data_obs.Integral() if data_obs.Integral() > 0 else 0:.4f}")
