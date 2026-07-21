#!/usr/bin/env python
import os
import argparse
import logging
import json
import re
import ROOT
from plotter import ComparisonCanvas, get_era_list, get_CoM_energy
from plotter import PALETTE_LONG as PALETTE
from HistoUtils import (setup_missing_histogram_logging, load_histogram,
                        load_systematic_variations, sum_histograms, load_era_configs,
                        get_sample_lists, clip_negative_bins, CorrelatedTotalBuilder)
from utils import build_sknanoutput_path, scale_with_variations
ROOT.gROOT.SetBatch(True)

# Flat normalization uncertainty for the "others" category, which is absent from
# KFactors.json and otherwise carries no theory norm.
OTHERS_XSEC_UNC = 0.50

# Fixed color mapping for backgrounds (consistent across all plots)
BKG_COLORS = {
    "nonprompt": PALETTE[0],
    "diboson": PALETTE[1],
    "ttX": PALETTE[2],
    "conv": PALETTE[3],
    "others": PALETTE[4],
}

# Preferred background order for stack plots (bottom to top)
# Legend will display in reverse order (top to bottom)
BKG_ORDER = ["others", "conv", "diboson", "ttX", "nonprompt"]

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str,
                    help="era (single era, Run2, Run3, or All for Run2+Run3)")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--histkey", required=True, type=str, help="histkey")
parser.add_argument("--exclude", default=None, type=str,
                    help="exclude reweighting (WZSF, ConvSF)")
parser.add_argument("--blind", default=False, action="store_true", help="blind data")
parser.add_argument("--signals", default=["MHc70_MA15", "MHc100_MA60", "MHc130_MA90", "MHc160_MA155"],
                    nargs="+", help="Signal mass points to overlay")
parser.add_argument("--signal-scale", default=2.0, type=float,
                    help="Scale factor for signal histograms")
parser.add_argument("--signal-colors", default=None, nargs="+",
                    help="signal line colors as ROOT color codes or hex strings")
parser.add_argument("--signal-line-width", default=2, type=int,
                    help="signal line width")
parser.add_argument("--signal-fill", default=False, action="store_true",
                    help="fill signal histograms under the line")
parser.add_argument("--signal-fill-alpha", default=0.20, type=float,
                    help="signal fill alpha when --signal-fill is used")
parser.add_argument("--noHEMVeto", default=False, action="store_true",
                    help="use NoHEMVeto samples (2018 only, SR1E2Mu/ZFake1E2Mu/TTZ2E1Mu)")
parser.add_argument("--output-root", default=None, type=str,
                    help="base output directory (default: $WORKDIR/TriLepton/plots)")
parser.add_argument("--output-format", default="png", choices=["png", "pdf"],
                    help="output file format")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

WORKDIR = os.environ['WORKDIR']

# Setup missing histogram logging
missing_logger = setup_missing_histogram_logging(args)

with open("configs/histkeys.json") as f:
    config = json.load(f)[args.histkey]

# Load K-factors
KFACTORS_PATH = f"{WORKDIR}/Common/Data/KFactors.json"
with open(KFACTORS_PATH) as f:
    KFACTORS = json.load(f)

with open(f"{WORKDIR}/Common/Data/ConvSF.json") as f:
    CONV_SF_DATA = json.load(f)

with open(f"{WORKDIR}/Common/Data/FakeNorm.json") as f:
    FAKENORM = json.load(f)


RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]


def get_plot_era_list(era):
    if era == "All":
        return RUN2_ERAS + RUN3_ERAS
    return get_era_list(era)


def get_run_period(era):
    if era in RUN2_ERAS or era == "Run2":
        return "Run2"
    if era in RUN3_ERAS or era == "Run3":
        return "Run3"
    raise ValueError(f"Invalid era: {era}")


def get_plot_com_energy(era):
    if era == "All":
        return "13/13.6"
    return get_CoM_energy(era)


config["era"] = args.era
config["CoM"] = get_plot_com_energy(args.era)
config["rTitle"] = "Data / Pred"
config["maxDigits"] = 3
config["blind"] = args.blind  # Pass blind flag to ComparisonCanvas
config["overflow"] = True  # Accumulate overflow into last visible bin
config["iPos"] = 0
config["legend"] = (0.72, 0.55, 0.99, 0.89)
config["legendTextSize"] = 0.038
config["signalLegend"] = (0.32, 0.63, 0.73, 0.87)
config["signalLegendTextSize"] = 0.034
config["signalLineWidth"] = args.signal_line_width
config["signalFill"] = args.signal_fill
config["signalFillAlpha"] = args.signal_fill_alpha
if args.signal_colors:
    config["signalColors"] = [
        ROOT.TColor.GetColor(color) if color.startswith("#") else int(color)
        for color in args.signal_colors
    ]
if args.era == "All":
    config["run_label"] = "Run 2+3, 200 fb^{#minus1}"
if not args.blind:
    config["chi2_test"] = True
    config["normalize_chi2"] = False
else:
    config["no_ratio"] = True  # Ratio pad is meaningless when data is blinded
#### Configurations
# Get era list for merging
era_list = get_plot_era_list(args.era)

if args.era != "All":
    get_run_period(args.era)
elif not era_list:
    raise ValueError(f"Invalid era: {args.era}")

## Check channel
if args.channel not in ["SR1E2Mu", "SR3Mu", "ZFake1E2Mu", "ZFake3Mu", "ZG1E2Mu", "ZG3Mu", "WZ1E2Mu", "WZ3Mu", "TTZ2E1Mu"]:
    raise ValueError(f"Invalid channel: {args.channel}")

if args.channel in ["SR1E2Mu", "SR3Mu"]:
    config["chi2_test"] = False

if args.noHEMVeto:
    if args.era != "2018":
        raise ValueError("--noHEMVeto only valid for era 2018")
    if args.channel not in ["SR1E2Mu", "ZFake1E2Mu", "TTZ2E1Mu"]:
        raise ValueError(f"--noHEMVeto not supported for channel {args.channel}")

CHANNEL_LABELS = {
    "SR1E2Mu":    ("SR", "e#mu#mu"),
    "SR3Mu":      ("SR", "#mu#mu#mu"),
    "ZFake1E2Mu": ("Z+nonprompt CR",  "e#mu#mu"),
    "ZFake3Mu":   ("Z+nonprompt CR",  "#mu#mu#mu"),
    "ZG1E2Mu":    ("Z+#gamma CR", "e#mu#mu"),
    "ZG3Mu":      ("Z+#gamma CR", "#mu#mu#mu"),
    "WZ1E2Mu":    ("WZ CR",     "e#mu#mu"),
    "WZ3Mu":      ("WZ CR",     "#mu#mu#mu"),
    "TTZ2E1Mu":   ("TTZ CR",    "ee#mu"),
}
config["channel"], config["region"] = CHANNEL_LABELS[args.channel]
config["channelPosY"] = 0.75
config["channelPosX"] = 0.22
if "1E2Mu" in args.channel:
    FLAG = "Run1E2Mu"
    channel_flag = "1E2Mu"
elif "2E1Mu" in args.channel:
    FLAG = "Run2E1Mu"
    channel_flag = "2E1Mu"
elif "3Mu" in args.channel:
    FLAG = "Run3Mu"
    channel_flag = "3Mu"
else:
    raise ValueError(f"Cannot determine FLAG for channel: {args.channel}")

# Create a modified args object for the common function
class ChannelArgs:
    def __init__(self, channel_flag):
        self.channel = channel_flag

channel_args = ChannelArgs(channel_flag)


def format_signal_label(signal_mass):
    match = re.fullmatch(r"MHc(\d+)_MA(\d+)", signal_mass)
    if not match:
        return signal_mass
    mhc, ma = match.groups()
    return f"(m_{{H^{{+}}}}, m_{{A}}) = ({mhc}, {ma}) GeV"

# Load configurations
ERA_SAMPLES, ERA_SYSTEMATICS = load_era_configs(channel_args, era_list)
DATAPERIODs, MC_CATEGORIES, MCList = get_sample_lists(ERA_SAMPLES, ["nonprompt", "conv", "ttX", "diboson", "others"])

OUTPUTROOT = args.output_root or f"{WORKDIR}/TriLepton/plots"
output_name = f"{args.histkey.replace('/', '_')}.{args.output_format}"
if args.exclude:
    OUTPUTPATH = f"{OUTPUTROOT}/{args.era}/{args.channel}/No{args.exclude}/{output_name}"
elif args.noHEMVeto:
    OUTPUTPATH = f"{OUTPUTROOT}/{args.era}/{args.channel}/NoHEMVeto/{output_name}"
else:
    OUTPUTPATH = f"{OUTPUTROOT}/{args.era}/{args.channel}/Central/{output_name}"

os.makedirs(os.path.dirname(OUTPUTPATH), exist_ok=True)


# Rate uncertainties are reported to CorrelatedTotalBuilder rather than folded into
# bin errors here, so that sources shared between processes stay correlated. Bin
# errors must therefore remain pure statistical.

def get_conv_scale_factor(sample, era, era_samples):
    """Return (scale, relative uncertainty) for a conversion sample.

    Uses Common/Data/ConvSF.json. Falls back to SF=1.0 +- 20% for 2E1Mu, which has
    no dedicated measurement.
    """
    if args.exclude == "ConvSF" or sample not in era_samples[era]["conv"]:
        return 1.0, 0.0

    if channel_flag == "2E1Mu":
        return 1.0, 0.20

    era_data = CONV_SF_DATA.get(channel_flag, {}).get(era)
    if era_data is None:
        logging.warning(f"No ConvSF for {channel_flag}/{era}, using default SF=1.0 ± 20%")
        return 1.0, 0.20

    return era_data["central"], era_data["total"]

def get_kfactor_info(sample, run):
    """Return (K-factor, relative cross-section uncertainty) from KFactors.json.

    xsecErr is stored as a multiplicative factor (1.075 means 7.5%).
    """
    if run not in KFACTORS or sample not in KFACTORS[run]:
        return 1.0, 0.0

    entry = KFACTORS[run][sample]
    return entry["kFactor"], entry.get("xsecErr", 1.0) - 1.0

def get_mc_rate_uncertainties(sample, era, era_samples, xsec_rel_unc, conv_rel_unc):
    """Rate uncertainties for one (era, sample): (name, value, corr_samples, corr_eras).

    The two axes are independent. Theory priors (cross sections, the flat "others"
    normalization) are the same number every year, so they correlate across eras but
    not across unrelated processes -- a datacard writes them as separate per-process
    lnN. Measured normalizations (ConvSF, WZNjSF) are shared by every sample they
    scale; ConvSF is re-measured per era, while the WZNjSF prior is not.
    """
    rate_uncs = []

    if xsec_rel_unc > 0.0:
        rate_uncs.append((f"xsec_{sample}", xsec_rel_unc, False, True))

    if conv_rel_unc > 0.0:
        rate_uncs.append(("conv_rate", conv_rel_unc, True, False))

    # "others" samples are unrelated processes absent from KFactors.json, so each
    # carries its own prior rather than a shared one.
    if sample in era_samples[era]["others"]:
        rate_uncs.append(("others_xsec", OTHERS_XSEC_UNC, False, True))

    return rate_uncs

#### Get Histograms

# Step 1: Load histograms from each era
era_data_hists = []
era_mc_hists = {sample: [] for sample in MCList}
era_nonprompt_hists = {sample: [] for sample in MC_CATEGORIES["nonprompt"]}
eras_with_data = []
eras_without_data = []

for era in era_list:

    # Load data for this era
    era_data = []
    for sample in ERA_SAMPLES[era]["data"]:
        file_path = build_sknanoutput_path(WORKDIR, args.channel, FLAG, era, sample,
                                           no_hem_veto=args.noHEMVeto)
        hist_path = f"{args.channel}/Central/{args.histkey}"

        h = load_histogram(file_path, hist_path, era, missing_logger)
        if h:
            clip_negative_bins(h)
            era_data.append(h)
    
    # Sum data for this era and track which eras have data
    if era_data:
        era_data_sum = sum_histograms(era_data, f"data_{era}")
        era_data_hists.append(era_data_sum)
        eras_with_data.append(era)
    else:
        eras_without_data.append(era)

# Step 2: Sum data histograms across eras
data = sum_histograms(era_data_hists, "data_total")
if data:
    data.SetTitle("Data")

# Check if we have any valid data
if data is None:
    logging.error(f"No valid data histograms found for {args.histkey} in any of the eras: {era_list}")
    logging.error("Cannot proceed with plotting without data. Exiting...")
    exit(1)
else:
    # Report which eras contributed data
    if eras_without_data:
        logging.warning(f"Data for {args.histkey} completely missing in eras: {eras_without_data}")

# Load nonprompt samples from each era.
# Contributions are accumulated per (era, sample) so that each named source stays
# one nuisance within an era; the builder supplies the uncertainty band below.
HISTs = {}
total_builder = CorrelatedTotalBuilder("total_background")

for era in era_list:
    # Load nonprompt for this era
    for sample in ERA_SAMPLES[era]["nonprompt"]:
        file_path = build_sknanoutput_path(WORKDIR, args.channel, FLAG, era, sample,
                                           is_nonprompt=True, no_hem_veto=args.noHEMVeto)
        hist_path = f"{args.channel}/Central/{args.histkey}"

        h = load_histogram(file_path, hist_path, era, missing_logger)
        if h:
            clip_negative_bins(h)
            # Per-era nonprompt normalization from FakeNorm.json (fallback 30%). One
            # fake-rate measurement per era, shared by every data stream, so it is
            # correlated -- and added alongside the statistical error rather than
            # overwriting it.
            np_unc = FAKENORM.get(FLAG, {}).get(era, 0.30)
            total_builder.add(era, sample, h,
                              rate_uncs=[("nonprompt_rate", np_unc, True, False)])
            era_nonprompt_hists[sample].append(h)

    # Load MC for this era
    all_era_samples = ERA_SAMPLES[era]["conv"] + ERA_SAMPLES[era]["ttX"] + ERA_SAMPLES[era]["diboson"] + ERA_SAMPLES[era]["others"]
    for sample in all_era_samples:
        use_no_wzsf = args.exclude == "WZSF" and "WZTo3LNu" in sample
        file_path = build_sknanoutput_path(WORKDIR, args.channel, FLAG, era, sample,
                                           run_syst=True, no_wzsf=use_no_wzsf,
                                           no_hem_veto=args.noHEMVeto)

        hist_path = f"{args.channel}/Central/{args.histkey}"
        h = load_histogram(file_path, hist_path, era, missing_logger)
        if h:
            clip_negative_bins(h)
            variations = load_systematic_variations(file_path, args.channel, args.histkey,
                                                    ERA_SYSTEMATICS[era], era,
                                                    missing_logger, clip=True)

            # Scale central and variations together, K-factor before ConvSF
            kfactor, xsec_rel_unc = get_kfactor_info(sample, get_run_period(era))
            scale_with_variations(h, variations, kfactor)
            conv_scale, conv_rel_unc = get_conv_scale_factor(sample, era, ERA_SAMPLES)
            scale_with_variations(h, variations, conv_scale)

            rate_uncs = get_mc_rate_uncertainties(sample, era, ERA_SAMPLES,
                                                  xsec_rel_unc, conv_rel_unc)
            total_builder.add(era, sample, h, variations=variations, rate_uncs=rate_uncs)
            era_mc_hists[sample].append(h)

# Step 3: Sum histograms across eras

# Sum nonprompt samples
for sample in MC_CATEGORIES["nonprompt"]:
    if era_nonprompt_hists[sample]:
        HISTs[sample] = sum_histograms(era_nonprompt_hists[sample], f"{sample}_total")

# Sum MC samples
for sample in MCList:
    if era_mc_hists[sample]:
        HISTs[sample] = sum_histograms(era_mc_hists[sample], f"{sample}_total")

# Check the final MC histograms
valid_mc_samples = 0
for sample in MCList + list(MC_CATEGORIES["nonprompt"]):
    if sample in HISTs and HISTs[sample] is not None:
        valid_mc_samples += 1
    else:
        logging.debug(f"No histograms found for sample {sample}")

# Check if we have at least some MC samples
if valid_mc_samples == 0:
    logging.error("No valid MC histograms found for any sample!")
    logging.error("Cannot proceed with plotting without any MC. Exiting...")
    exit(1)
#### Merge backgrounds by category
temp_dict = {cat: None for cat in BKG_ORDER}
for category, samples in MC_CATEGORIES.items():
    for sample in samples:
        if sample not in HISTs:
            continue
        if temp_dict[category] is None:
            temp_dict[category] = HISTs[sample].Clone(category)
        else:
            temp_dict[category].Add(HISTs[sample])

# Build BKGs in fixed order for consistent stacking and colors
BKGs = {}
for bkg_name in BKG_ORDER:
    if bkg_name in temp_dict and temp_dict[bkg_name] is not None:
        BKGs[bkg_name] = temp_dict[bkg_name]

# Build colors list in same order as BKGs
config["colors"] = [BKG_COLORS[bkg] for bkg in BKGs.keys()]

# Note: Blinding is now handled in ComparisonCanvas to ensure
# data and systematics are guaranteed to be identical

# Load signal histograms (only for SR1E2Mu and SR3Mu)
SIGNALs = {}
if args.channel in ["SR1E2Mu", "SR3Mu"]:
    for signal_mass in args.signals:
        signal_name = f"TTToHcToWAToMuMu-{signal_mass}"
        signal_hist = None

        for era in era_list:
            # Signal files don't have "Skim_TriLep_" prefix, construct path directly
            path = f"{WORKDIR}/SKNanoOutput/PromptAnalyzer/{FLAG}_RunSyst_RunTheoryUnc/{era}/{signal_name}.root"
            if not os.path.exists(path):
                logging.debug(f"Signal file not found: {path}")
                continue

            f = ROOT.TFile.Open(path)
            h = f.Get(f"{args.channel}/Central/{args.histkey}")
            if not h:
                logging.debug(f"Signal histogram not found: {args.channel}/Central/{args.histkey} in {path}")
                f.Close()
                continue

            h.SetDirectory(0)
            h.Scale(args.signal_scale)

            if signal_hist is None:
                signal_hist = h.Clone(signal_mass)
                signal_hist.SetDirectory(0)
            else:
                signal_hist.Add(h)
            f.Close()

        if signal_hist:
            SIGNALs[format_signal_label(signal_mass)] = signal_hist
    # For ParticleNet score plots, keep only matching signal
    if "score" in args.histkey:
        # Extract mass point from histkey (e.g., "MHc160_MA155/score_diboson" -> "MHc160_MA155")
        mass_point = args.histkey.split("/")[0]
        # Keep only the matching signal
        SIGNALs = {k: v for k, v in SIGNALs.items() if k == format_signal_label(mass_point)}
        if not SIGNALs:
            logging.warning(f"ParticleNet score plot for {mass_point}, but no matching signal histogram found")

plotter = ComparisonCanvas(data, BKGs, config, total_syst=total_builder.total_hist())
plotter.drawPadUp()
if SIGNALs:
    plotter.drawSignals(SIGNALs)
plotter.drawPadDown()
plotter.canv.SaveAs(OUTPUTPATH)
