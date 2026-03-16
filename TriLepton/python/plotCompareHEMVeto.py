#!/usr/bin/env python
"""Compare total background with and without HEM veto (2018 only).

Produces a two-panel plot: upper panel overlays total background histograms
(HEM veto vs no HEM veto); lower panel shows the ratio (no HEM / HEM).
"""
import os
import logging
import argparse
import json
import ROOT
from math import sqrt
from plotter import KinematicCanvasWithRatio, get_CoM_energy
from HistoUtils import (setup_missing_histogram_logging, load_histogram,
                        calculate_systematics, sum_histograms, load_era_configs,
                        get_sample_lists)
from utils import build_sknanoutput_path, apply_rate_uncertainty
import correctionlib

ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--channel", required=True, type=str,
                    choices=["SR1E2Mu", "ZFake1E2Mu", "TTZ2E1Mu"],
                    help="analysis channel")
parser.add_argument("--histkey", required=True, type=str, help="histogram key")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

WORKDIR = os.environ['WORKDIR']
ERA = "2018"
ERA_LIST = [ERA]
RUN = "Run2"

# Determine FLAG from channel
if "1E2Mu" in args.channel:
    FLAG = "Run1E2Mu"
    channel_flag = "1E2Mu"
elif "2E1Mu" in args.channel:
    FLAG = "Run2E1Mu"
    channel_flag = "2E1Mu"
else:
    raise ValueError(f"Cannot determine FLAG for channel: {args.channel}")


class ChannelArgs:
    def __init__(self, cf):
        self.channel = cf


channel_args = ChannelArgs(channel_flag)
ERA_SAMPLES, ERA_SYSTEMATICS = load_era_configs(channel_args, ERA_LIST)
_, MC_CATEGORIES, MCList = get_sample_lists(ERA_SAMPLES, ["nonprompt", "conv", "ttX", "diboson", "others"])

# Load K-factors
with open(f"{WORKDIR}/Common/Data/KFactors.json") as f:
    KFACTORS = json.load(f)

# Load histogram config for axis labels
with open("configs/histkeys.json") as f:
    hk_config = json.load(f)[args.histkey]

missing_logger = logging.getLogger("missing")


# Proxy args object for calculate_systematics (needs .channel, .histkey, .exclude)
class SystArgs:
    channel = args.channel
    histkey = args.histkey
    exclude = None


syst_args = SystArgs()


def _apply_kfactor(hist, sample):
    if RUN not in KFACTORS:
        return hist
    kfactors = KFACTORS[RUN]
    if sample not in kfactors:
        return hist
    hist.Scale(kfactors[sample]["kFactor"])
    if "xsecErr" in kfactors[sample]:
        apply_rate_uncertainty(hist, kfactors[sample]["xsecErr"] - 1.0)
    return hist


def _apply_wz_uncertainty(hist, sample):
    if "WZTo3LNu" in sample or "ZZTo4L" in sample:
        apply_rate_uncertainty(hist, 0.20)
    return hist


def _load_conv_sfs():
    if "1E2Mu" in args.channel:
        zg_channel = "ZG1E2Mu"
    elif "3Mu" in args.channel:
        zg_channel = "ZG3Mu"
    else:
        return {}
    sf_file = f"{WORKDIR}/TriLepton/results/{zg_channel}/{ERA}/ConvSF.json"
    if not os.path.exists(sf_file):
        return {}
    try:
        cset = correctionlib.CorrectionSet.from_file(sf_file)
        return {ERA: {"cset": cset, "zg_channel": zg_channel}}
    except Exception as e:
        logging.warning(f"Failed to load ConvSF: {e}")
        return {}


def _apply_conv_sf(hist, sample, era, conv_sf):
    if sample not in ERA_SAMPLES[era]["conv"]:
        return hist
    if not conv_sf or era not in conv_sf:
        apply_rate_uncertainty(hist, 0.20)
        return hist
    try:
        cset = conv_sf[era]["cset"]
        zg_channel = conv_sf[era]["zg_channel"]
        sf_central = cset[f"ConvSF_{zg_channel}_{era}_Central"].evaluate()
        hist.Scale(sf_central)
        sf_pu = cset[f"ConvSF_{zg_channel}_{era}_prompt_up"].evaluate()
        sf_pd = cset[f"ConvSF_{zg_channel}_{era}_prompt_down"].evaluate()
        sf_nu = cset[f"ConvSF_{zg_channel}_{era}_nonprompt_up"].evaluate()
        sf_nd = cset[f"ConvSF_{zg_channel}_{era}_nonprompt_down"].evaluate()
        p_rel = max(abs(sf_pu - sf_central), abs(sf_central - sf_pd)) / sf_central
        n_rel = max(abs(sf_nu - sf_central), abs(sf_central - sf_nd)) / sf_central
        apply_rate_uncertainty(hist, sqrt(p_rel**2 + n_rel**2))
    except Exception as e:
        logging.warning(f"Failed to apply ConvSF to {sample}: {e}")
    return hist


def load_total_bkg(no_hem_veto=False):
    """Load and return summed total background histogram."""
    conv_sf = _load_conv_sfs()
    era_mc_hists = {sample: [] for sample in MCList}
    era_nonprompt_hists = {sample: [] for sample in MC_CATEGORIES["nonprompt"]}

    for era in ERA_LIST:
        # Load nonprompt
        for sample in ERA_SAMPLES[era]["nonprompt"]:
            file_path = build_sknanoutput_path(WORKDIR, args.channel, FLAG, era, sample,
                                               is_nonprompt=True, no_hem_veto=no_hem_veto)
            hist_path = f"{args.channel}/Central/{args.histkey}"
            h = load_histogram(file_path, hist_path, era, missing_logger)
            if h:
                for bin_idx in range(h.GetNcells()):
                    h.SetBinError(bin_idx, h.GetBinContent(bin_idx) * 0.3)
                era_nonprompt_hists[sample].append(h)

        # Load MC (conv, ttX, diboson, others)
        all_samples = (ERA_SAMPLES[era]["conv"] + ERA_SAMPLES[era]["ttX"] +
                       ERA_SAMPLES[era]["diboson"] + ERA_SAMPLES[era]["others"])
        for sample in all_samples:
            file_path = build_sknanoutput_path(WORKDIR, args.channel, FLAG, era, sample,
                                               run_syst=True, no_hem_veto=no_hem_veto)
            hist_path = f"{args.channel}/Central/{args.histkey}"
            h = load_histogram(file_path, hist_path, era, missing_logger)
            if h:
                h = _apply_kfactor(h, sample)
                h = calculate_systematics(h, ERA_SYSTEMATICS[era], file_path, syst_args, era, missing_logger)
                h = _apply_wz_uncertainty(h, sample)
                h = _apply_conv_sf(h, sample, era, conv_sf)
                era_mc_hists[sample].append(h)

    all_bkg = []
    for sample in MC_CATEGORIES["nonprompt"]:
        if era_nonprompt_hists[sample]:
            h = sum_histograms(era_nonprompt_hists[sample], f"{sample}_total")
            if h:
                all_bkg.append(h)
    for sample in MCList:
        if era_mc_hists[sample]:
            h = sum_histograms(era_mc_hists[sample], f"{sample}_total")
            if h:
                all_bkg.append(h)

    return sum_histograms(all_bkg, "total_bkg")


# Load both configurations
h_hem = load_total_bkg(no_hem_veto=False)
h_nohem = load_total_bkg(no_hem_veto=True)

if h_hem is None or h_nohem is None:
    logging.error("Failed to load total background histograms")
    exit(1)

# Compute totals including underflow and overflow
def _integral_with_flow(h):
    return h.Integral(0, h.GetNbinsX() + 1)

integral_hem = _integral_with_flow(h_hem)
integral_nohem = _integral_with_flow(h_nohem)
logging.info(f"HEM veto integral: {integral_hem:.1f}")
logging.info(f"No HEM veto integral: {integral_nohem:.1f}")

# KinematicCanvasWithRatio uses first key as the ratio reference
hists = {
    f"HEM veto ({integral_hem:.1f})": h_hem,
    f"No HEM veto ({integral_nohem:.1f})": h_nohem,
}

config = {**hk_config,
          "era": ERA,
          "CoM": get_CoM_energy(ERA),
          "rTitle": "No HEM / HEM",
          "channel": args.channel,
          "overflow": True,
          "legend": [0.6, 0.6, 0.85, 0.89],
          }

plotter = KinematicCanvasWithRatio(hists, config)
plotter.drawPadUp()
plotter.drawPadDown()

output_path = (f"{WORKDIR}/TriLepton/plots/{ERA}/{args.channel}/compareHEMVeto/"
               f"{args.histkey.replace('/', '_')}.png")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plotter.canv.SaveAs(output_path)
logging.info(f"Saved: {output_path}")
