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
                        sum_histograms, load_era_configs,
                        get_sample_lists, clip_negative_bins)
from utils import build_sknanoutput_path
ROOT.gROOT.SetBatch(True)

BKG_COLORS = {
    "nonprompt": PALETTE[0],
    "diboson": PALETTE[1],
    "ttX": PALETTE[2],
    "conv": PALETTE[3],
    "others": PALETTE[4],
}

BKG_ORDER = ["others", "conv", "diboson", "ttX", "nonprompt"]

RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]

SUPPORTED_HISTKEYS = {
    "SR1E2Mu": [
        "os_mumu/mu_charge_sum",
        "os_mumu/pair/mass",
        "njet_ge2/n_jets",
        "njet_ge2/pair/mass",
        "nbjet_ge1/n_bjets",
        "nbjet_ge1/pair/mass",
        "baseline/jets/size",
        "baseline/bjets/size",
        "baseline/pair/mass",
    ],
    "SR3Mu": [
        "charge_abs1/charge_sum",
        "charge_abs1/pair_lowM/mass",
        "charge_abs1/pair_highM/mass",
        "njet_ge2/n_jets",
        "njet_ge2/pair_lowM/mass",
        "njet_ge2/pair_highM/mass",
        "nbjet_ge1/n_bjets",
        "nbjet_ge1/pair_lowM/mass",
        "nbjet_ge1/pair_highM/mass",
        "baseline/jets/size",
        "baseline/bjets/size",
        "baseline/pair_lowM/mass",
        "baseline/pair_highM/mass",
    ],
}

CHANNEL_LABELS = {
    "SR1E2Mu": ("SR", "e#mu#mu"),
    "SR3Mu": ("SR", "#mu#mu#mu"),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--era", required=True, type=str,
                        help="era (single era, Run2, Run3, or All for Run2+Run3)")
    parser.add_argument("--channel", required=True, type=str, help="channel")
    parser.add_argument("--histkey", required=True, type=str,
                        help="N-1 histkey without NMinusOne prefix")
    parser.add_argument("--exclude", default=None, type=str, choices=["WZSF", "ConvSF"],
                        help="exclude central reweighting (WZSF or ConvSF)")
    parser.add_argument("--blind", default=False, action="store_true", help="blind data")
    parser.add_argument("--signals",
                        default=["MHc70_MA15", "MHc100_MA60", "MHc130_MA90", "MHc160_MA155"],
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
    parser.add_argument("--output-root", default=None, type=str,
                        help="base output directory (default: $WORKDIR/TriLepton/plots)")
    parser.add_argument("--output-format", default="png", choices=["png", "pdf"],
                        help="output file format")
    parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
    return parser.parse_args()


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


def get_channel_flags(channel):
    if channel == "SR1E2Mu":
        return "Run1E2Mu", "1E2Mu"
    if channel == "SR3Mu":
        return "Run3Mu", "3Mu"
    raise ValueError(f"N-1 plotting is only supported for SR1E2Mu and SR3Mu, got {channel}")


def format_signal_label(signal_mass):
    match = re.fullmatch(r"MHc(\d+)_MA(\d+)", signal_mass)
    if not match:
        return signal_mass
    mhc, ma = match.groups()
    return f"(m_{{H^{{+}}}}, m_{{A}}) = ({mhc}, {ma}) GeV"


class ChannelArgs:
    def __init__(self, channel_flag):
        self.channel = channel_flag


def apply_kfactor(hist, sample, run, kfactors):
    if run not in kfactors:
        return hist
    if sample not in kfactors[run]:
        return hist

    kfactor = kfactors[run][sample]["kFactor"]
    hist.Scale(kfactor)
    logging.debug(f"Applied K-factor {kfactor} to {sample}")
    return hist


def apply_conv_scale_factor(hist, sample, era, era_samples, channel_flag, conv_sf_data, exclude=None):
    if exclude == "ConvSF":
        return hist
    if sample not in era_samples[era]["conv"]:
        return hist

    era_data = conv_sf_data.get(channel_flag, {}).get(era)
    if era_data is None:
        logging.warning(f"No ConvSF for {channel_flag}/{era}; leaving conversion sample unscaled")
        return hist

    hist.Scale(era_data["central"])
    logging.debug(f"Applied ConvSF {era_data['central']} to {sample} in {channel_flag}/{era}")
    return hist


def build_config(args, hist_configs):
    config = dict(hist_configs[args.histkey])
    config["era"] = args.era
    config["CoM"] = get_plot_com_energy(args.era)
    config["rTitle"] = "Data / Pred"
    config["maxDigits"] = 3
    config["blind"] = args.blind
    config["overflow"] = True
    config["iPos"] = 0
    config["legend"] = (0.72, 0.55, 0.99, 0.89)
    config["legendTextSize"] = 0.038
    config["systSrc"] = "Stat"
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
    config["chi2_test"] = False
    config["channel"], config["region"] = CHANNEL_LABELS[args.channel]
    config["channelPosY"] = 0.75
    config["channelPosX"] = 0.22
    if args.era == "All":
        config["run_label"] = "Run 2+3, 200 fb^{#minus1}"
    if args.blind:
        config["no_ratio"] = True
    return config


def build_output_path(args, workdir):
    output_root = args.output_root or f"{workdir}/TriLepton/plots"
    output_name = f"{args.histkey.replace('/', '_')}.{args.output_format}"
    subdir = "NMinusOne"
    if args.exclude:
        subdir = f"NMinusOne_No{args.exclude}"
    return f"{output_root}/{args.era}/{args.channel}/{subdir}/{output_name}"


def get_root_hist_path(channel, histkey):
    if histkey.startswith("baseline/"):
        return f"{channel}/Central/{histkey.removeprefix('baseline/')}"
    return f"{channel}/Central/NMinusOne/{histkey}"


def load_plot_objects(args, workdir, era_list, flag, channel_flag, era_samples,
                      mc_categories, mc_list, kfactors, conv_sf_data, missing_logger):
    hist_path = get_root_hist_path(args.channel, args.histkey)
    era_data_hists = []
    era_mc_hists = {sample: [] for sample in mc_list}
    era_nonprompt_hists = {sample: [] for sample in mc_categories["nonprompt"]}
    eras_without_data = []

    for era in era_list:
        era_data = []
        for sample in era_samples[era]["data"]:
            file_path = build_sknanoutput_path(workdir, args.channel, flag, era, sample)
            hist = load_histogram(file_path, hist_path, era, missing_logger)
            if hist:
                clip_negative_bins(hist)
                era_data.append(hist)
        if era_data:
            era_data_hists.append(sum_histograms(era_data, f"data_{era}"))
        else:
            eras_without_data.append(era)

    data = sum_histograms(era_data_hists, "data_total")
    if data:
        data.SetTitle("Data")
    else:
        logging.error(f"No valid data histograms found for {hist_path} in eras: {era_list}")
        raise SystemExit(1)

    if eras_without_data:
        logging.warning(f"Data for {hist_path} completely missing in eras: {eras_without_data}")

    for era in era_list:
        for sample in era_samples[era]["nonprompt"]:
            file_path = build_sknanoutput_path(workdir, args.channel, flag, era, sample,
                                               is_nonprompt=True)
            hist = load_histogram(file_path, hist_path, era, missing_logger)
            if hist:
                clip_negative_bins(hist)
                era_nonprompt_hists[sample].append(hist)

        all_era_samples = (
            era_samples[era]["conv"]
            + era_samples[era]["ttX"]
            + era_samples[era]["diboson"]
            + era_samples[era]["others"]
        )
        for sample in all_era_samples:
            use_no_wzsf = args.exclude == "WZSF" and "WZTo3LNu" in sample
            file_path = build_sknanoutput_path(workdir, args.channel, flag, era, sample,
                                               run_syst=True, no_wzsf=use_no_wzsf)
            hist = load_histogram(file_path, hist_path, era, missing_logger)
            if hist:
                clip_negative_bins(hist)
                hist = apply_kfactor(hist, sample, get_run_period(era), kfactors)
                hist = apply_conv_scale_factor(hist, sample, era, era_samples,
                                               channel_flag, conv_sf_data, args.exclude)
                era_mc_hists[sample].append(hist)

    hists = {}
    for sample in mc_categories["nonprompt"]:
        if era_nonprompt_hists[sample]:
            hists[sample] = sum_histograms(era_nonprompt_hists[sample], f"{sample}_total")

    for sample in mc_list:
        if era_mc_hists[sample]:
            hists[sample] = sum_histograms(era_mc_hists[sample], f"{sample}_total")

    valid_mc_samples = sum(1 for sample in mc_list + list(mc_categories["nonprompt"])
                           if sample in hists and hists[sample] is not None)
    if valid_mc_samples == 0:
        logging.error("No valid MC histograms found for any N-1 sample")
        raise SystemExit(1)

    merged_by_category = {category: None for category in BKG_ORDER}
    for category, samples in mc_categories.items():
        for sample in samples:
            if sample not in hists:
                continue
            if merged_by_category[category] is None:
                merged_by_category[category] = hists[sample].Clone(category)
                merged_by_category[category].SetDirectory(0)
            else:
                merged_by_category[category].Add(hists[sample])

    backgrounds = {}
    for bkg_name in BKG_ORDER:
        if merged_by_category[bkg_name] is not None:
            backgrounds[bkg_name] = merged_by_category[bkg_name]

    return data, backgrounds


def load_signals(args, workdir, era_list, flag, missing_logger):
    hist_path = get_root_hist_path(args.channel, args.histkey)
    signals = {}

    for signal_mass in args.signals:
        signal_name = f"TTToHcToWAToMuMu-{signal_mass}"
        signal_hist = None

        for era in era_list:
            file_path = (
                f"{workdir}/SKNanoOutput/PromptAnalyzer/"
                f"{flag}_RunSyst_RunTheoryUnc/{era}/{signal_name}.root"
            )
            hist = load_histogram(file_path, hist_path, era, missing_logger)
            if hist:
                clip_negative_bins(hist)
                hist.Scale(args.signal_scale)
                if signal_hist is None:
                    signal_hist = hist.Clone(signal_mass)
                    signal_hist.SetDirectory(0)
                else:
                    signal_hist.Add(hist)

        if signal_hist:
            signals[format_signal_label(signal_mass)] = signal_hist

    return signals


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    if args.channel not in SUPPORTED_HISTKEYS:
        raise SystemExit(f"N-1 plotting is only supported for {sorted(SUPPORTED_HISTKEYS.keys())}")
    if args.histkey not in SUPPORTED_HISTKEYS[args.channel]:
        raise SystemExit(f"Invalid N-1 histkey {args.histkey} for {args.channel}. "
                         f"Valid choices: {SUPPORTED_HISTKEYS[args.channel]}")

    workdir = os.environ["WORKDIR"]
    missing_logger = setup_missing_histogram_logging(args)

    with open("configs/histkeys.nminusone.json") as f:
        hist_configs = json.load(f)
    with open(f"{workdir}/Common/Data/KFactors.json") as f:
        kfactors = json.load(f)
    with open(f"{workdir}/Common/Data/ConvSF.json") as f:
        conv_sf_data = json.load(f)

    config = build_config(args, hist_configs)
    era_list = get_plot_era_list(args.era)
    if args.era != "All":
        get_run_period(args.era)
    elif not era_list:
        raise ValueError(f"Invalid era: {args.era}")

    flag, channel_flag = get_channel_flags(args.channel)
    channel_args = ChannelArgs(channel_flag)
    era_samples, _ = load_era_configs(channel_args, era_list)
    _, mc_categories, mc_list = get_sample_lists(
        era_samples, ["nonprompt", "conv", "ttX", "diboson", "others"]
    )

    output_path = build_output_path(args, workdir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    data, backgrounds = load_plot_objects(
        args, workdir, era_list, flag, channel_flag, era_samples, mc_categories,
        mc_list, kfactors, conv_sf_data, missing_logger
    )
    signals = load_signals(args, workdir, era_list, flag, missing_logger)
    config["colors"] = [BKG_COLORS[bkg] for bkg in backgrounds.keys()]

    plotter = ComparisonCanvas(data, backgrounds, config)
    plotter.drawPadUp()
    if signals:
        plotter.drawSignals(signals)
    plotter.drawPadDown()
    plotter.canv.SaveAs(output_path)


if __name__ == "__main__":
    main()
