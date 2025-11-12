#!/usr/bin/env python
import os
import argparse
import json
import ROOT
from plotter import KinematicCanvas, get_CoM_energy
ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel (DIMU or EMU)")
args = parser.parse_args()

WORKDIR = os.environ['WORKDIR']

# Load sample configuration
with open("configs/samplegroup.json") as f:
    sample_config = json.load(f)

# Determine flag based on channel
if args.channel == "DIMU":
    FLAG = "RunDiMu_RunSyst"
elif args.channel == "EMU":
    FLAG = "RunEMu_RunSyst"
else:
    raise ValueError(f"Invalid channel: {args.channel}")

# Get samples for this era and channel
if args.era not in sample_config:
    raise ValueError(f"Invalid era: {args.era}")
if args.channel not in sample_config[args.era]:
    raise ValueError(f"Invalid channel: {args.channel}")

era_samples = sample_config[args.era][args.channel]

# Define output path
OUTPUTPATH = f"{WORKDIR}/DiLepton/plots/{args.era}/{args.channel}/weights/btagSF.png"
os.makedirs(os.path.dirname(OUTPUTPATH), exist_ok=True)

# Histogram path in ROOT file
hist_path = f"{args.channel}/Central/weights/btagSF"

def handle_overflow_underflow(hist):
    """
    Handle overflow and underflow bins by adding them to the first and last visible bins.
    Also prints warning if significant events are found in over/underflow.

    Args:
        hist: TH1 histogram
    """
    nbins = hist.GetNbinsX()

    # Check underflow (bin 0)
    underflow = hist.GetBinContent(0)
    underflow_err = hist.GetBinError(0)

    # Check overflow (bin nbins+1)
    overflow = hist.GetBinContent(nbins + 1)
    overflow_err = hist.GetBinError(nbins + 1)

    total = hist.Integral(0, nbins + 1)

    # Print warnings if significant events in over/underflow
    if underflow > 0:
        print(f"  Underflow: {underflow:.2f} events ({100*underflow/total:.2f}%)")
        # Add underflow to first bin
        first_bin_content = hist.GetBinContent(1)
        first_bin_err = hist.GetBinError(1)
        hist.SetBinContent(1, first_bin_content + underflow)
        hist.SetBinError(1, ROOT.TMath.Sqrt(first_bin_err**2 + underflow_err**2))

    if overflow > 0:
        print(f"  Overflow: {overflow:.2f} events ({100*overflow/total:.2f}%)")
        # Add overflow to last bin
        last_bin_content = hist.GetBinContent(nbins)
        last_bin_err = hist.GetBinError(nbins)
        hist.SetBinContent(nbins, last_bin_content + overflow)
        hist.SetBinError(nbins, ROOT.TMath.Sqrt(last_bin_err**2 + overflow_err**2))

# Load histograms for each background category
def load_and_sum_histograms(samples, category_name):
    """Load histograms for a list of samples and sum them"""
    hists = []
    for sample in samples:
        file_path = f"{WORKDIR}/SKNanoOutput/DiLepton/{FLAG}/{args.era}/{sample}.root"
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            continue

        f = ROOT.TFile.Open(file_path)
        if not f or f.IsZombie():
            print(f"Warning: Cannot open file: {file_path}")
            continue

        h = f.Get(hist_path)
        if not h:
            print(f"Warning: Histogram {hist_path} not found in {file_path}")
            f.Close()
            continue

        # Clone and detach from file
        h_clone = h.Clone(f"{category_name}_{sample}")
        h_clone.SetDirectory(0)
        hists.append(h_clone)
        f.Close()

    if not hists:
        return None

    # Sum all histograms
    h_sum = hists[0].Clone(category_name)
    for h in hists[1:]:
        h_sum.Add(h)

    h_sum.SetTitle(category_name)
    return h_sum

# Load background categories
background_hists = {}

# W jets
if "W" in era_samples and era_samples["W"]:
    h = load_and_sum_histograms(era_samples["W"], "W")
    if h:
        background_hists["W"] = h

# Z jets
if "Z" in era_samples and era_samples["Z"]:
    h = load_and_sum_histograms(era_samples["Z"], "Z")
    if h:
        background_hists["Z"] = h

# Top quark (TT)
if "TT" in era_samples and era_samples["TT"]:
    h = load_and_sum_histograms(era_samples["TT"], "TT")
    if h:
        background_hists["TT"] = h

# Single top (ST)
if "ST" in era_samples and era_samples["ST"]:
    h = load_and_sum_histograms(era_samples["ST"], "ST")
    if h:
        background_hists["ST"] = h

# Diboson (VV)
if "VV" in era_samples and era_samples["VV"]:
    h = load_and_sum_histograms(era_samples["VV"], "VV")
    if h:
        background_hists["VV"] = h

# Check if we have any valid histograms
if not background_hists:
    print("Error: No valid histograms found!")
    exit(1)

print(f"Loaded histograms for: {list(background_hists.keys())}")

# Handle overflow and underflow for each histogram
print("\nChecking overflow/underflow:")
for name, hist in background_hists.items():
    print(f"{name}:")
    handle_overflow_underflow(hist)

# Configuration for plotting
config = {
    "era": args.era,
    "CoM": get_CoM_energy(args.era),
    "xTitle": "b-tagging Scale Factor",
    "yTitle": "Events",
    "xRange": [0.5, 2.0],
    "yRange": None,  # Auto-calculate
    "channel": args.channel,
    "logy": False,
    "normalize": False,
    "legend": [0.65, 0.89 - 0.05 * (len(background_hists) + 1), 0.99, 0.89]
}

# Create canvas and plot
canvas = KinematicCanvas(background_hists, config)
canvas.drawPad()
canvas.leg.Draw()
canvas.canv.SaveAs(OUTPUTPATH)
print(f"\nSaved plot to: {OUTPUTPATH}")
