#!/usr/bin/env python
import json
import logging
import os
import re
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import ROOT

from plotter import ComparisonCanvas, PALETTE, PALETTE_LONG, get_era_list, get_CoM_energy
from HistoUtils import (
    setup_missing_histogram_logging,
    load_histogram,
    calculate_systematics,
    sum_histograms,
    load_era_configs,
    get_sample_lists,
    clip_negative_bins,
)
from utils import build_sknanoutput_path, apply_rate_uncertainty


RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]

BKG_COLORS = {
    "nonprompt": PALETTE_LONG[0],
    "diboson": PALETTE_LONG[1],
    "ttX": PALETTE_LONG[2],
    "conv": PALETTE_LONG[3],
    "others": PALETTE_LONG[4],
}
BKG_ORDER = ["others", "conv", "diboson", "ttX", "nonprompt"]

CHANNEL_LABELS = {
    "SR1E2Mu": ("SR", "e#mu#mu"),
    "SR3Mu": ("SR", "#mu#mu#mu"),
}


@dataclass
class PaperPlotOptions:
    workdir: str
    output_root: Path
    hist_configs: dict
    signals: list[str]
    signal_scale: float = 2.0
    blind: bool = False
    debug: bool = False
    adaptive_binning: bool = False
    adaptive_min_bkg: float = 10.0
    adaptive_max_width: float = 10.0
    adaptive_base_width: float = 2.0
    signal_colors: list[str] = field(default_factory=lambda: ["#5790fc", "#f89c20", "#964a8b", "#e42536"])


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
    if "1E2Mu" in channel:
        return "Run1E2Mu", "1E2Mu"
    if "2E1Mu" in channel:
        return "Run2E1Mu", "2E1Mu"
    if "3Mu" in channel:
        return "Run3Mu", "3Mu"
    raise ValueError(f"Cannot determine FLAG for channel: {channel}")


def format_signal_label(signal_mass):
    match = re.fullmatch(r"MHc(\d+)_MA(\d+)", signal_mass)
    if not match:
        return signal_mass
    mhc, ma = match.groups()
    return f"(m_{{H^{{+}}}}, m_{{A}}) = ({mhc}, {ma}) GeV"


def load_common_data(workdir):
    with open(f"{workdir}/Common/Data/KFactors.json") as f:
        kfactors = json.load(f)
    with open(f"{workdir}/Common/Data/ConvSF.json") as f:
        conv_sf = json.load(f)
    with open(f"{workdir}/Common/Data/FakeNorm.json") as f:
        fake_norm = json.load(f)
    return kfactors, conv_sf, fake_norm


def apply_kfactor(hist, sample, run, kfactors):
    if run not in kfactors or sample not in kfactors[run]:
        return hist

    sample_kfactor = kfactors[run][sample]["kFactor"]
    hist.Scale(sample_kfactor)

    if "xsecErr" in kfactors[run][sample]:
        apply_rate_uncertainty(hist, kfactors[run][sample]["xsecErr"] - 1.0)

    return hist


def apply_conv_scale_factor(hist, sample, era, era_samples, channel_flag, conv_sf_data, exclude=None):
    if exclude == "ConvSF" or sample not in era_samples[era]["conv"]:
        return hist

    if channel_flag == "2E1Mu":
        apply_rate_uncertainty(hist, 0.20)
        return hist

    era_data = conv_sf_data.get(channel_flag, {}).get(era)
    if era_data is None:
        logging.warning(f"No ConvSF for {channel_flag}/{era}, using default SF=1.0 ± 20%")
        apply_rate_uncertainty(hist, 0.20)
        return hist

    hist.Scale(era_data["central"])
    apply_rate_uncertainty(hist, era_data["total"])
    return hist


def apply_others_uncertainty(hist, sample, era, era_samples):
    if sample in era_samples[era]["others"]:
        apply_rate_uncertainty(hist, 0.50)
    return hist


def build_config(histkey, channel, options):
    config = dict(options.hist_configs[histkey])
    config["era"] = "All"
    config["CoM"] = get_plot_com_energy("All")
    config["rTitle"] = "Data / Pred"
    config["maxDigits"] = 3
    config["blind"] = options.blind
    config["overflow"] = True
    config["iPos"] = 0
    config["legend"] = (0.72, 0.55, 0.99, 0.89)
    config["legendTextSize"] = 0.038
    config["signalLegend"] = (0.32, 0.63, 0.73, 0.87)
    config["signalLegendTextSize"] = 0.034
    config["signalLineWidth"] = 3
    config["signalFill"] = True
    config["signalFillAlpha"] = 0.18
    config["signalColors"] = [ROOT.TColor.GetColor(color) for color in options.signal_colors]
    config["run_label"] = "Run 2+3, 200 fb^{#minus1}"
    config["chi2_test"] = False
    if options.blind:
        config["no_ratio"] = True

    config["channel"], config["region"] = CHANNEL_LABELS[channel]
    config["channelPosY"] = 0.75
    config["channelPosX"] = 0.22
    return config


def build_output_path(channel, histkey, options):
    return options.output_root / "All" / channel / "Central" / f"{histkey.replace('/', '_')}.pdf"


def build_base_edges(xmin, xmax, base_width):
    edges = []
    value = xmin
    while value < xmax:
        edges.append(round(value, 6))
        value += base_width
    if not edges or edges[-1] != xmax:
        edges.append(round(xmax, 6))
    return edges


def build_adaptive_edges(total_bkg, xmin, xmax, min_bkg, max_width, base_width):
    base_edges = build_base_edges(xmin, xmax, base_width)
    base_hist = total_bkg.Rebin(len(base_edges) - 1, f"{total_bkg.GetName()}_adaptive_base", array("d", base_edges))

    adaptive_edges = [base_edges[0]]
    bin_start = base_edges[0]
    content_sum = 0.0

    for ibin in range(1, base_hist.GetNbinsX() + 1):
        content_sum += base_hist.GetBinContent(ibin)
        high_edge = base_hist.GetXaxis().GetBinUpEdge(ibin)
        width = high_edge - bin_start
        is_last = ibin == base_hist.GetNbinsX()

        if content_sum >= min_bkg or width >= max_width or is_last:
            adaptive_edges.append(round(high_edge, 6))
            bin_start = high_edge
            content_sum = 0.0

    if len(adaptive_edges) >= 3:
        last_content = 0.0
        last_low = adaptive_edges[-2]
        last_high = adaptive_edges[-1]
        for ibin in range(1, base_hist.GetNbinsX() + 1):
            low_edge = base_hist.GetXaxis().GetBinLowEdge(ibin)
            high_edge = base_hist.GetXaxis().GetBinUpEdge(ibin)
            if low_edge >= last_low and high_edge <= last_high:
                last_content += base_hist.GetBinContent(ibin)

        merged_width = last_high - adaptive_edges[-3]
        if last_content < min_bkg and merged_width <= max_width:
            adaptive_edges.pop(-2)

    return adaptive_edges


def load_plot_objects(channel, histkey, config, options):
    workdir = options.workdir
    era_list = get_plot_era_list("All")
    flag, channel_flag = get_channel_flags(channel)
    kfactors, conv_sf_data, fake_norm = load_common_data(workdir)

    args = SimpleNamespace(
        era="All",
        channel=channel,
        histkey=histkey,
        exclude=None,
        noHEMVeto=False,
        blind=options.blind,
        debug=options.debug,
    )
    missing_logger = setup_missing_histogram_logging(args)

    channel_args = SimpleNamespace(channel=channel_flag)
    era_samples, era_systematics = load_era_configs(channel_args, era_list)
    _, mc_categories, mc_list = get_sample_lists(era_samples, ["nonprompt", "conv", "ttX", "diboson", "others"])

    era_data_hists = []
    era_mc_hists = {sample: [] for sample in mc_list}
    era_nonprompt_hists = {sample: [] for sample in mc_categories["nonprompt"]}

    for era in era_list:
        era_data = []
        for sample in era_samples[era]["data"]:
            file_path = build_sknanoutput_path(workdir, channel, flag, era, sample)
            hist_path = f"{channel}/Central/{histkey}"
            hist = load_histogram(file_path, hist_path, era, missing_logger)
            if hist:
                clip_negative_bins(hist)
                era_data.append(hist)
        if era_data:
            era_data_hists.append(sum_histograms(era_data, f"data_{era}"))

        for sample in era_samples[era]["nonprompt"]:
            file_path = build_sknanoutput_path(workdir, channel, flag, era, sample, is_nonprompt=True)
            hist_path = f"{channel}/Central/{histkey}"
            hist = load_histogram(file_path, hist_path, era, missing_logger)
            if hist:
                clip_negative_bins(hist)
                np_unc = fake_norm.get(flag, {}).get(era, 0.30)
                for bin_idx in range(hist.GetNcells()):
                    hist.SetBinError(bin_idx, hist.GetBinContent(bin_idx) * np_unc)
                era_nonprompt_hists[sample].append(hist)

        all_era_samples = (
            era_samples[era]["conv"]
            + era_samples[era]["ttX"]
            + era_samples[era]["diboson"]
            + era_samples[era]["others"]
        )
        for sample in all_era_samples:
            file_path = build_sknanoutput_path(workdir, channel, flag, era, sample, run_syst=True)
            hist_path = f"{channel}/Central/{histkey}"
            hist = load_histogram(file_path, hist_path, era, missing_logger)
            if hist:
                clip_negative_bins(hist)
                hist = apply_kfactor(hist, sample, get_run_period(era), kfactors)
                hist = calculate_systematics(hist, era_systematics[era], file_path, args, era, missing_logger)
                hist = apply_conv_scale_factor(hist, sample, era, era_samples, channel_flag, conv_sf_data)
                hist = apply_others_uncertainty(hist, sample, era, era_samples)
                era_mc_hists[sample].append(hist)

    data = sum_histograms(era_data_hists, "data_total")
    if data is None:
        raise RuntimeError(f"No valid data histograms found for {histkey} in {channel}")
    data.SetTitle("Data")

    hists = {}
    for sample in mc_categories["nonprompt"]:
        if era_nonprompt_hists[sample]:
            hists[sample] = sum_histograms(era_nonprompt_hists[sample], f"{sample}_total")
    for sample in mc_list:
        if era_mc_hists[sample]:
            hists[sample] = sum_histograms(era_mc_hists[sample], f"{sample}_total")

    if not hists:
        raise RuntimeError(f"No valid MC histograms found for {histkey} in {channel}")

    merged_categories = {cat: None for cat in BKG_ORDER}
    for category, samples in mc_categories.items():
        for sample in samples:
            if sample not in hists:
                continue
            if merged_categories[category] is None:
                merged_categories[category] = hists[sample].Clone(category)
            else:
                merged_categories[category].Add(hists[sample])

    bkgs = {
        name: merged_categories[name]
        for name in BKG_ORDER
        if merged_categories.get(name) is not None
    }
    config["colors"] = [BKG_COLORS[name] for name in bkgs.keys()]

    return data, bkgs, load_signals(channel, histkey, flag, era_list, options)


def load_signals(channel, histkey, flag, era_list, options):
    signals = {}
    if channel not in ["SR1E2Mu", "SR3Mu"]:
        return signals

    for signal_mass in options.signals:
        signal_name = f"TTToHcToWAToMuMu-{signal_mass}"
        signal_hist = None

        for era in era_list:
            path = f"{options.workdir}/SKNanoOutput/PromptAnalyzer/{flag}_RunSyst_RunTheoryUnc/{era}/{signal_name}.root"
            if not os.path.exists(path):
                continue
            root_file = ROOT.TFile.Open(path)
            hist = root_file.Get(f"{channel}/Central/{histkey}")
            if not hist:
                root_file.Close()
                continue
            hist.SetDirectory(0)
            hist.Scale(options.signal_scale)
            if signal_hist is None:
                signal_hist = hist.Clone(signal_mass)
                signal_hist.SetDirectory(0)
            else:
                signal_hist.Add(hist)
            root_file.Close()

        if signal_hist:
            signals[format_signal_label(signal_mass)] = signal_hist

    return signals


def apply_adaptive_binning(config, bkgs, options):
    if not options.adaptive_binning:
        return None

    total_bkg = None
    for hist in bkgs.values():
        if total_bkg is None:
            total_bkg = hist.Clone("adaptive_total_bkg")
            total_bkg.SetDirectory(0)
        else:
            total_bkg.Add(hist)

    xmin, xmax = config["xRange"][0], config["xRange"][-1]
    edges = build_adaptive_edges(
        total_bkg,
        xmin,
        xmax,
        options.adaptive_min_bkg,
        options.adaptive_max_width,
        options.adaptive_base_width,
    )
    config["xRange"] = edges
    config.pop("rebin", None)
    return edges


def render_paper_plot(channel, histkey, options):
    config = build_config(histkey, channel, options)
    data, bkgs, signals = load_plot_objects(channel, histkey, config, options)
    adaptive_edges = apply_adaptive_binning(config, bkgs, options)

    output_path = build_output_path(channel, histkey, options)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = ComparisonCanvas(data, bkgs, config)
    plotter.drawPadUp()
    if signals:
        plotter.drawSignals(signals)
    plotter.drawPadDown()
    plotter.canv.SaveAs(str(output_path))

    return output_path, adaptive_edges
