#!/usr/bin/env python
import json
import logging
import os
import re
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import cmsstyle as CMS
import ROOT

from plotter import (ComparisonCanvas, PALETTE, PALETTE_LONG, EnergyInfo,
                     LumiInfoExact, get_era_list)
from HistoUtils import (
    setup_missing_histogram_logging,
    load_histogram,
    load_systematic_variations,
    sum_histograms,
    load_era_configs,
    get_sample_lists,
    clip_negative_bins,
    CorrelatedTotalBuilder,
)
from utils import build_sknanoutput_path, scale_with_variations


RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]

# Flat normalization uncertainty for the "others" category, absent from KFactors.json
OTHERS_XSEC_UNC = 0.50

BKG_COLORS = {
    "nonprompt": PALETTE_LONG[0],
    "diboson": PALETTE_LONG[1],
    "ttX": PALETTE_LONG[2],
    "conv": PALETTE_LONG[3],
    "others": PALETTE_LONG[4],
}
BKG_ORDER = ["others", "conv", "diboson", "ttX", "nonprompt"]
# Legend labels for the paper figures. The internal category names stay as they
# are everywhere else (sample groups, yield tables); only what the reader sees
# is typeset.
BKG_LABELS = {
    "nonprompt": "Nonprompt",
    "diboson": "Diboson",
    "ttX": "t#bar{t}X",
    "conv": "Conversions",
    "others": "Others",
}

SIGNAL_LINE_WIDTH = 3
SIGNAL_FILL_ALPHA = 0.18
SIGNAL_FILL_STYLE = 1001
DATA_LABEL = "Data"
SYST_LABEL = "Stat.+Syst."
# Uncertainty band style, mirrored from ComparisonCanvas.drawPadUp so the shared
# legend panel shows the same hatching as the plots.
SYST_FILL_STYLE = 3004
SYST_FILL_COLOR = 12

# Standalone legend panel: the plots publish no legend of their own, so the 2x2
# paper figures spend their fourth panel on this one. Geometry is in NDC of a
# canvas the same size as a plot panel.
LEGEND_PANEL_ROW_SPACING = 1.55  # row pitch in units of the text size
# Two side-by-side columns: signals on the left, backgrounds on the right. The
# mass-point labels are far longer than the process names, so the columns are
# unequal and each sets its own symbol width (the TLegend margin) to leave the
# text as much room as it needs.
LEGEND_PANEL_LEFT = 0.03
LEGEND_PANEL_COLUMN_GAP = 0.01
LEGEND_PANEL_BKG_WIDTH = 0.33
LEGEND_PANEL_BKG_MARGIN = 0.30
LEGEND_PANEL_SIGNAL_WIDTH = 0.62
LEGEND_PANEL_SIGNAL_MARGIN = 0.16

CHANNEL_LABELS = {
    "SR1E2Mu": ("SR", "e#mu#mu"),
    "SR3Mu": ("SR", "#mu#mu#mu"),
    "ZFake1E2Mu": ("Z+nonprompt CR", "e#mu#mu"),
    "ZFake3Mu": ("Z+nonprompt CR", "#mu#mu#mu"),
    "ZG1E2Mu": ("Z+#gamma CR", "e#mu#mu"),
    "ZG3Mu": ("Z+#gamma CR", "#mu#mu#mu"),
    "WZ1E2Mu": ("WZ CR", "e#mu#mu"),
    "WZ3Mu": ("WZ CR", "#mu#mu#mu"),
    "TTZ2E1Mu": ("t#bar{t}Z CR", "ee#mu"),
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
    # Legends are identical in every paper plot, so they are dropped from the
    # plots and published once via render_paper_legend().
    draw_legends: bool = False
    legend_panel_text_size: float = 0.040
    y_headroom: float = 1.5


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


# Rate uncertainties are reported to CorrelatedTotalBuilder rather than folded into
# bin errors, so that sources shared between processes stay correlated. Bin errors
# must therefore remain pure statistical.

def get_kfactor_info(sample, run, kfactors):
    """Return (K-factor, relative cross-section uncertainty) from KFactors.json."""
    if run not in kfactors or sample not in kfactors[run]:
        return 1.0, 0.0

    entry = kfactors[run][sample]
    return entry["kFactor"], entry.get("xsecErr", 1.0) - 1.0


def get_conv_scale_factor(sample, era, era_samples, channel_flag, conv_sf_data, exclude=None):
    """Return (scale, relative uncertainty) for a conversion sample."""
    if exclude == "ConvSF" or sample not in era_samples[era]["conv"]:
        return 1.0, 0.0

    if channel_flag == "2E1Mu":
        return 1.0, 0.20

    era_data = conv_sf_data.get(channel_flag, {}).get(era)
    if era_data is None:
        logging.warning(f"No ConvSF for {channel_flag}/{era}, using default SF=1.0 ± 20%")
        return 1.0, 0.20

    return era_data["central"], era_data["total"]


def get_mc_rate_uncertainties(sample, era, era_samples, xsec_rel_unc, conv_rel_unc, exclude=None):
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

    if sample in era_samples[era]["others"]:
        rate_uncs.append(("others_xsec", OTHERS_XSEC_UNC, False, True))

    return rate_uncs


def build_config(histkey, channel, options):
    config = dict(options.hist_configs[histkey])
    config["era"] = "All"
    # Per-energy luminosities, CMS style for multi-energy combinations:
    # "137.6 fb^-1 (13 TeV) + 62.4 fb^-1 (13.6 TeV)". cmsstyle appends the
    # CoM in parentheses, so the Run3 energy is carried by "CoM" and the
    # Run2 term is baked into "run_label".
    config["CoM"] = f"{EnergyInfo['Run3']:g} TeV"
    config["run_label"] = (f"{LumiInfoExact['Run2']:g} fb^{{#minus1}} ({EnergyInfo['Run2']:g} TeV) + "
                           f"{LumiInfoExact['Run3']:g} fb^{{#minus1}}")
    config["rTitle"] = "Data / Pred."
    # No in-plot legend to clear, so the stack can use the vertical space.
    config["yHeadroom"] = options.y_headroom
    # Signal regions are statistics-limited over most of the mass range, so the
    # ratio needs the wider window; the control regions stay on the tight one
    # shared with the SignalRegionStudyV3 paper plots.
    config["rRange"] = [0.0, 2.0] if channel.startswith("SR") else [0.5, 1.5]
    config["maxDigits"] = 3
    config["blind"] = options.blind
    config["overflow"] = True
    config["iPos"] = 11
    config["legend"] = (0.72, 0.55, 0.99, 0.89)
    config["legendTextSize"] = 0.038
    config["signalLegend"] = (0.32, 0.63, 0.73, 0.87)
    config["signalLegendTextSize"] = 0.034
    # Both legends live in the shared panel unless the caller asks for the
    # self-contained version of the plots.
    config["drawLegend"] = options.draw_legends
    config["drawSignalLegend"] = options.draw_legends
    config["signalLineWidth"] = SIGNAL_LINE_WIDTH
    config["signalFill"] = True
    config["signalFillAlpha"] = SIGNAL_FILL_ALPHA
    config["signalFillStyle"] = SIGNAL_FILL_STYLE
    config["signalColors"] = [ROOT.TColor.GetColor(color) for color in options.signal_colors]
    config["systSrc"] = SYST_LABEL
    config["chi2_test"] = False
    if histkey == "ZCand/mass":
        config["xRange"] = [81, 101]
        config["yTitle"] = "Events"
        if channel == "TTZ2E1Mu":
            config["xTitle"] = "m(e^{+}e^{-}) [GeV]"
        config["overflow"] = False
    if options.blind:
        config["no_ratio"] = True

    config["channel"], config["region"] = CHANNEL_LABELS[channel]
    # Same block position as the SignalRegionStudyV3 paper panels.
    config["channelPosY"] = 0.72
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

    # Accumulated per (era, sample) so each named source stays one nuisance within
    # an era; supplies the uncertainty band instead of stacking per-sample errors.
    total_builder = CorrelatedTotalBuilder("total_background")

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
                # One fake-rate measurement per era, shared by every data stream, so
                # correlated -- and added alongside the statistical error rather than
                # overwriting it.
                np_unc = fake_norm.get(flag, {}).get(era, 0.30)
                total_builder.add(era, sample, hist,
                                  rate_uncs=[("nonprompt_rate", np_unc, True, False)])
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
                variations = load_systematic_variations(file_path, channel, histkey,
                                                        era_systematics[era], era,
                                                        missing_logger, clip=True)

                # Scale central and variations together, K-factor before ConvSF
                kfactor, xsec_rel_unc = get_kfactor_info(sample, get_run_period(era), kfactors)
                scale_with_variations(hist, variations, kfactor)
                conv_scale, conv_rel_unc = get_conv_scale_factor(
                    sample, era, era_samples, channel_flag, conv_sf_data)
                scale_with_variations(hist, variations, conv_scale)

                rate_uncs = get_mc_rate_uncertainties(sample, era, era_samples,
                                                      xsec_rel_unc, conv_rel_unc)
                total_builder.add(era, sample, hist, variations=variations,
                                  rate_uncs=rate_uncs)
                era_mc_hists[sample].append(hist)

    data = sum_histograms(era_data_hists, "data_total")
    if data is None:
        raise RuntimeError(f"No valid data histograms found for {histkey} in {channel}")
    data.SetTitle(DATA_LABEL)

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

    # Keyed by legend label, since ComparisonCanvas labels the stack with the
    # dict keys; colors are collected alongside so they stay aligned with the
    # categories that actually survived.
    bkgs = {}
    config["colors"] = []
    for name in BKG_ORDER:
        if merged_categories.get(name) is None:
            continue
        bkgs[BKG_LABELS[name]] = merged_categories[name]
        config["colors"].append(BKG_COLORS[name])

    return (data, bkgs, load_signals(channel, histkey, flag, era_list, options),
            total_builder.total_hist())


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


def apply_adaptive_binning(config, bkgs, options, histkey):
    if not options.adaptive_binning:
        return None
    if histkey == "ZCand/mass":
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


def build_legend_output_path(options, with_signals=True):
    """Signal regions and control regions get their own panel, since only the
    former overlay signals."""
    name = "legend.pdf" if with_signals else "legend_nosignal.pdf"
    return options.output_root / "All" / name


def paper_panel_size(options):
    """Pixel size of one paper plot, so the legend panel matches it exactly.

    Taken from a throwaway canvas built the same way ComparisonCanvas builds
    its own, rather than from hardcoded numbers that would silently drift if
    cmsstyle changed its reference dimensions.
    """
    if options.blind:
        probe = CMS.cmsCanvas("panel_size_probe", 0., 1., 0., 1., "", "",
                              square=True, iPos=11, extraSpace=0)
    else:
        probe = CMS.cmsDiCanvas("panel_size_probe", 0., 1., 0., 1., 0., 1., "", "", "",
                                square=True, iPos=11, extraSpace=0)
    size = (probe.GetWindowWidth(), probe.GetWindowHeight())
    probe.Close()
    return size


def build_legend_proxies(options, prefix="legend"):
    """Dummy objects styled like the drawn ones, plus their legend entries.

    Returns (background_entries, signal_entries, proxies); the caller must keep
    `proxies` alive until the canvas is written, since TLegend does not own them.
    The prefix keeps the ROOT names unique when several panels are built in one
    process.
    """
    proxies = []

    def new_proxy(suffix):
        hist = ROOT.TH1F(f"{prefix}_{suffix}", "", 1, 0., 1.)
        hist.SetDirectory(0)
        hist.SetStats(0)
        proxies.append(hist)
        return hist

    data = new_proxy("data")
    data.SetMarkerStyle(ROOT.kFullCircle)
    data.SetMarkerSize(1.0)
    data.SetMarkerColor(ROOT.kBlack)
    data.SetLineColor(ROOT.kBlack)
    bkg_entries = [(data, DATA_LABEL, "PE")]

    # Same order as the stacked legend: top of the stack listed first.
    for name in reversed(BKG_ORDER):
        proxy = new_proxy(name)
        proxy.SetFillColor(BKG_COLORS[name])
        proxy.SetLineColor(BKG_COLORS[name])
        proxy.SetFillStyle(1001)
        bkg_entries.append((proxy, BKG_LABELS[name], "F"))

    syst = new_proxy("syst")
    syst.SetFillStyle(SYST_FILL_STYLE)
    syst.SetFillColor(SYST_FILL_COLOR)
    syst.SetLineWidth(0)
    syst.SetMarkerSize(0)
    bkg_entries.append((syst, SYST_LABEL, " FE2"))

    signal_entries = []
    for idx, signal_mass in enumerate(options.signals):
        color = ROOT.TColor.GetColor(options.signal_colors[idx % len(options.signal_colors)])
        proxy = new_proxy(f"signal_{signal_mass}")
        proxy.SetLineColor(color)
        proxy.SetLineWidth(SIGNAL_LINE_WIDTH)
        proxy.SetMarkerSize(0)
        # Signals are drawn as a translucent fill with a solid outline, so the
        # legend shows the same filled box rather than a bare line.
        proxy.SetFillColorAlpha(color, SIGNAL_FILL_ALPHA)
        proxy.SetFillStyle(SIGNAL_FILL_STYLE)
        signal_entries.append((proxy, format_signal_label(signal_mass), "F"))

    return bkg_entries, signal_entries, proxies


def render_paper_legend(options, with_signals=True):
    """Write the shared legend as its own panel, sized like a paper plot."""
    width, height = paper_panel_size(options)
    name = "paper_legend" if with_signals else "paper_legend_nosignal"
    canvas = ROOT.TCanvas(name, name, 50, 50, width, height)
    canvas.SetFillColor(0)
    canvas.SetBorderMode(0)
    canvas.SetFrameFillStyle(0)
    canvas.SetFrameBorderMode(0)
    canvas.cd()

    bkg_entries, signal_entries, proxies = build_legend_proxies(options, prefix=name)
    if not with_signals:
        signal_entries = []
    text_size = options.legend_panel_text_size
    row = text_size * LEGEND_PANEL_ROW_SPACING

    # Each block is centred on the panel's midline, which leaves the shorter
    # signal column sitting in the middle of the background one.
    def block_y(n_entries):
        height = row * n_entries
        return 0.5 - 0.5 * height, 0.5 + 0.5 * height

    if signal_entries:
        signal_x1 = LEGEND_PANEL_LEFT
        bkg_x1 = signal_x1 + LEGEND_PANEL_SIGNAL_WIDTH + LEGEND_PANEL_COLUMN_GAP
    else:
        # Nothing to sit beside, so the single column is centred instead.
        bkg_x1 = 0.5 * (1.0 - LEGEND_PANEL_BKG_WIDTH)

    legends = []
    if signal_entries:
        y1, y2 = block_y(len(signal_entries))
        signal_leg = CMS.cmsLeg(signal_x1, y1, signal_x1 + LEGEND_PANEL_SIGNAL_WIDTH, y2,
                                textSize=text_size)
        signal_leg.SetMargin(LEGEND_PANEL_SIGNAL_MARGIN)
        CMS.addToLegend(signal_leg, *signal_entries)
        legends.append(signal_leg)

    y1, y2 = block_y(len(bkg_entries))
    bkg_leg = CMS.cmsLeg(bkg_x1, y1, bkg_x1 + LEGEND_PANEL_BKG_WIDTH, y2, textSize=text_size)
    bkg_leg.SetMargin(LEGEND_PANEL_BKG_MARGIN)
    CMS.addToLegend(bkg_leg, *bkg_entries)
    legends.append(bkg_leg)

    canvas.Update()
    output_path = build_legend_output_path(options, with_signals)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(str(output_path))
    canvas.Close()
    del proxies, legends
    return output_path


def render_paper_plot(channel, histkey, options):
    config = build_config(histkey, channel, options)
    data, bkgs, signals, total_syst = load_plot_objects(channel, histkey, config, options)
    adaptive_edges = apply_adaptive_binning(config, bkgs, options, histkey)

    output_path = build_output_path(channel, histkey, options)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = ComparisonCanvas(data, bkgs, config, total_syst=total_syst)
    plotter.drawPadUp()
    if signals:
        plotter.drawSignals(signals)
    plotter.drawPadDown()
    plotter.canv.SaveAs(str(output_path))

    return output_path, adaptive_edges
