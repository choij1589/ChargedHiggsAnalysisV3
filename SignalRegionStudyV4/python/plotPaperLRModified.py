#!/usr/bin/env python3
"""Produce paper-style ParticleNet LR_modified plots from cached score hists."""

import argparse
import logging
import os
import re
import sys
from array import array
from math import sqrt
from pathlib import Path

import ROOT


MODULE_DIR = Path(__file__).resolve().parents[1]
WORKDIR = Path(os.environ.get("WORKDIR", MODULE_DIR.parent))

sys.path.insert(0, str(WORKDIR / "Common" / "Tools"))
sys.path.insert(0, str(MODULE_DIR / "python"))
from plotter import (ComparisonCanvas, EnergyInfo,  # noqa: E402
                     LumiInfo, PALETTE_LONG)
from HistoUtils import calculate_chi2  # noqa: E402
import cmsstyle as CMS  # noqa: E402
import srspaths  # noqa: E402


ROOT.gROOT.SetBatch(True)

SCORE_KEY = "LR_modified"
# Template signal source of the score caches. interp-signal is the V4
# production arm; main() overrides this from --signal-source, and the two
# sibling paper scripts read it so all three stay on one source.
SIGNAL_SOURCE = "interp-signal"
DEFAULT_MASSPOINTS = ("MHc160_MA85", "MHc130_MA90", "MHc100_MA95")
REGIONS = ("SR", "TTZCR")
ORIGINAL_BKGS = ("nonprompt", "WZ", "ZZ", "ttW", "ttZ", "ttH", "tZq", "conversion", "others")
GROUP_MAP = {
    "nonprompt": ("nonprompt",),
    "conv": ("conversion",),
    "diboson": ("WZ", "ZZ"),
    "ttX": ("ttZ", "tZq", "ttH", "ttW"),
    "others": ("others",),
}
BKG_ORDER = ("others", "conv", "diboson", "ttX", "nonprompt")
BKG_COLORS = {
    "nonprompt": PALETTE_LONG[0],
    "diboson": PALETTE_LONG[1],
    "ttX": PALETTE_LONG[2],
    "conv": PALETTE_LONG[3],
    "others": PALETTE_LONG[4],
}
# Legend labels for the paper figures. The internal group names stay as they are
# everywhere else (template processes, yield tables); only what the reader sees
# is typeset. Matches TriLepton/python/paper_plotting.py.
BKG_LABELS = {
    "nonprompt": "Nonprompt",
    "diboson": "Diboson",
    "ttX": "t#bar{t}X",
    "conv": "Conversions",
    "others": "Others",
}
DATA_LABEL = "Data"
SYST_LABEL = "Stat.+Syst."
RATIO_LABEL = "Data / Pred."
# Uncertainty band style, mirrored from ComparisonCanvas.drawPadUp so the shared
# legend panel shows the same hatching as the plots.
SYST_FILL_STYLE = 3004
SYST_FILL_COLOR = 12

BASE_WIDTH = 0.04
ADAPTIVE_MIN_BKG = 10.0
ADAPTIVE_MAX_WIDTH = 0.20
SIGNAL_SCALE = 6.0
SIGNAL_COLOR = ROOT.kBlack
SIGNAL_LINE_WIDTH = 3
# No fill: at 18% alpha the black shading was indistinguishable from the
# Conversions grey (#94a4a2) of the CMS colourblind-safe palette. A bare
# heavy line also reads correctly as an overlay rather than a stack member,
# and leaves the palette untouched. Legend swatch follows as a line.
SIGNAL_FILL_ALPHA = 0.0
SIGNAL_LEGEND_OPT = "L"
SIGNAL_LABEL = "Signal"

# ParticleNet working point drawn on the SR panels. The production WP is
# eps_B = 20%, but the frozen values in fits/pnet/MHc*/threshold_wp.json are
# per (channel, run period) -- FOUR different cuts are summed into this one
# All/Combined panel (e.g. MHc130_MA90: 0.5712 to 0.6221) -- so no stored
# number is the line for this figure. It is recomputed on the panel's own
# total background instead: the score above which EFF_B_TARGET of the plotted
# background lies. That is the same quantity measPnetThresholds.py defines --
# identical process list, and the cached hists already carry the per-category
# mass window and bg_weights -- evaluated on the union of the four categories
# rather than on one of them.
EFF_B_TARGET = 0.20
# Style follows plotParticleNetScore.draw_threshold_overlay, so the line means
# the same thing in the diagnostic score plots and in the paper panel.
THRESHOLD_COLOR = ROOT.kRed + 1
THRESHOLD_LINE_STYLE = 7
THRESHOLD_LINE_WIDTH = 3
THRESHOLD_LABEL = f"#varepsilon_{{B}} = {EFF_B_TARGET:.0%} cut"

# CMS block. cmsstyle hardcodes the in-frame offsets of "CMS"/"Preliminary" at
# 3.5% of the frame from its top-left corner, which at this panel size puts the
# text on the axis ticks; supplying cmsPosX/cmsPosY hands the block to
# BaseCanvas._configure_cms_label() instead. Same placement and size as the
# TriLepton paper plots (TriLepton/docs/PaperPlotting.md), so the two figure
# families carry an identical CMS block.
CMS_LABEL_POS = (0.20, 0.865)
CMS_LABEL_SIZE = 0.070
# Channel/region caption, directly below the CMS block.
CHANNEL_POS = (0.20, 0.665)
CHANNEL_SIZE = 0.063
# cmsDiCanvas' 0.015 upper-pad bottom margin is smaller than half that pad's
# scaled y-label, so the "0" at the axis origin is cut in half. Widening the
# margin opens a white strip between the frames, so the label is dropped
# instead. Shared by all three paper scripts.
HIDE_ORIGIN_Y_LABEL = True

# In-plot legend, in the top-right corner. Sitting 0.04 below the frame top
# rather than against it: the eps_B entry added a fifth row, and the block read
# top-heavy pushed all the way up. The bottom edge still clears the tallest
# drawn point by ~0.05 of the frame -- Y_HEADROOM puts that point at 1/1.9 =
# 0.53 of the frame, and this box starts above it -- so nothing is covered.
LEGEND_BOX = (0.43, 0.54, 0.97, 0.86)
LEGEND_TEXT_SIZE = 0.028
LEGEND_COLUMNS = 2

# Headroom above the stack for the in-frame CMS block, the channel text and
# the legend beneath them.
Y_HEADROOM = 1.9
# Mass point moves to the top right, where the in-plot legend used to sit.
MASS_LABEL_POS = (0.90, 0.80)
MASS_LABEL_SIZE = 0.052
# Nudge in PDF points, converted against the panel below so the shift stays the
# same physical distance whatever the pad geometry.
MASS_LABEL_OFFSET_PT = (-20.0, -6.0)
# CropBox ROOT writes for this canvas; used only to size the nudge above.
PANEL_SIZE_PT = (526.0, 567.0)

# --private-work variant (TTZ CR only): non-approved data figure. The CMS logo
# is replaced by "Private work (CMS data)" in the extra-text style (font 52,
# 0.76 of the logo size), on one line, with the channel caption moved up
# directly beneath it. The legend keeps its text size but its top drops below
# the label, with tighter rows and columns so its bottom edge stays near the
# paper layout's.
PRIVATE_WORK_TEXT = "Private work (CMS data)"
PRIVATE_WORK_SIZE = 0.76 * CMS_LABEL_SIZE
PRIVATE_WORK_CHANNEL_POS = (CHANNEL_POS[0], 0.735)
PRIVATE_WORK_LEGEND_BOX = (0.55, 0.53, 0.95, 0.79)
# Mass point and data/prediction chi2, left-aligned under the channel caption,
# in the column the narrowed legend leaves free.
PRIVATE_WORK_INFO_SIZE = 0.036
PRIVATE_WORK_INFO_STEP = 0.055
PRIVATE_WORK_SUFFIX = "_pw"

# Standalone legend panel geometry, in NDC of a canvas the size of a plot panel.
LEGEND_KEY = "legend"
LEGEND_PANEL_ROW_SPACING = 1.55  # row pitch in units of the text size
LEGEND_PANEL_WIDTH = 0.42
LEGEND_PANEL_MARGIN = 0.26
LEGEND_PANEL_TEXT_SIZE = 0.040


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create Run2+Run3 paper PDF plots for ParticleNet LR_modified."
    )
    parser.add_argument(
        "--masspoint",
        choices=("all", *DEFAULT_MASSPOINTS),
        default="all",
        help="mass point to plot, or all three",
    )
    parser.add_argument(
        "--region",
        choices=("all", LEGEND_KEY, *REGIONS),
        default="all",
        help=f"plot region to produce ('{LEGEND_KEY}' = shared legend panels only)",
    )
    parser.add_argument(
        "--standalone-legend",
        action="store_true",
        dest="standalone_legend",
        help="publish the legend as its own panel instead of drawing it in "
             "each plot (the pre-2026-08-18 layout)",
    )
    parser.add_argument(
        "--output-root",
        default="results/plots/paper",
        help="base output directory, relative to SignalRegionStudyV4 unless absolute",
    )
    parser.add_argument(
        "--signal-source", dest="signal_source", default="interp-signal",
        choices=("mc-signal", "interp-signal"),
        help="score-cache source; interp-signal is the V4 production arm",
    )
    parser.add_argument(
        "--base-width",
        type=float,
        default=BASE_WIDTH,
        help=("uniform bin width of the starting grid, before adaptive merging "
              "(default: %(default)s). A non-default value writes into a "
              "bin<width> subdirectory, e.g. --base-width 0.02 -> SR/bin0p02/"),
    )
    parser.add_argument(
        "--private-work",
        action="store_true",
        dest="private_work",
        help=("TTZ CR only: label 'Private work (CMS data)' and write "
              "LR_modified_<mp>_pw.pdf"),
    )
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    args = parser.parse_args()
    if args.private_work and args.region not in ("all", "TTZCR"):
        parser.error("--private-work applies to the TTZ CR only (--region TTZCR or all)")
    return args


def base_width_tag(base_width):
    """Output subdirectory for a base binning; empty string keeps the default location."""
    if abs(base_width - BASE_WIDTH) < 1e-9:
        return ""
    return "bin" + f"{base_width:g}".replace(".", "p")


def resolve_output_root(path):
    output_root = Path(path)
    if not output_root.is_absolute():
        output_root = MODULE_DIR / output_root
    return output_root


def cache_path(region, masspoint, signal_source=None):
    """The score-histogram cache plotParticleNetScore.py leaves behind.

    V4 layout: templates/{mp}/{method}/{source}/{era}/{channel}/scores/
    {score_region}/histograms.root, built through srspaths rather than by
    hand (path construction lives in srspaths.py and scripts/env.sh only).
    """
    if region == "SR":
        channel = "Combined"
        score_region = "Combined"
    elif region == "TTZCR":
        channel = "SR3Mu"
        score_region = "TTZ2E1Mu"
    else:
        raise ValueError(f"Unsupported region: {region}")

    return Path(srspaths.template_dir(
        masspoint, "ParticleNet", "All", channel,
        source=signal_source or SIGNAL_SOURCE
    )) / "scores" / score_region / "histograms.root"


def output_path(output_root, region, masspoint, subdir="", suffix=""):
    directory = output_root / region
    if subdir:
        directory = directory / subdir
    return directory / f"{SCORE_KEY}_{masspoint}{suffix}.pdf"


def format_signal_label(masspoint):
    """(m_H+, m_A) = (130, 90) GeV. Accepts V4 p-notation (MA87p5 -> 87.5)."""
    match = re.fullmatch(r"MHc(\d+)_MA(\d+(?:p\d+)?)", masspoint)
    if not match:
        return masspoint
    mhc, ma = srspaths.masspoint_mhc_ma(masspoint)
    return f"(m_{{H^{{+}}}}, m_{{A}}) = ({mhc:g}, {ma:g}) GeV"


def load_score_histograms(region, masspoint):
    path = cache_path(region, masspoint)
    if not path.exists():
        raise FileNotFoundError(f"Missing cached score histograms: {path}")

    root_file = ROOT.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie():
        raise RuntimeError(f"Failed to open cached score histograms: {path}")

    directory = root_file.Get(SCORE_KEY)
    if not directory:
        root_file.Close()
        raise RuntimeError(f"Directory {SCORE_KEY} not found in {path}")

    hists = {}
    for key in directory.GetListOfKeys():
        obj = key.ReadObj()
        if obj and obj.InheritsFrom("TH1"):
            hist = obj.Clone(key.GetName())
            hist.SetDirectory(0)
            hists[key.GetName()] = hist

    root_file.Close()
    return hists


def clone_sum(hists, names, out_name):
    total = None
    for name in names:
        hist = hists.get(name)
        if hist is None:
            continue
        if total is None:
            total = hist.Clone(out_name)
            total.SetDirectory(0)
        else:
            total.Add(hist)
    return total


def rebin_with_edges(hist, edges, name):
    bins = array("d", edges)
    rebinned = hist.Rebin(len(edges) - 1, name, bins)
    rebinned.SetDirectory(0)
    return rebinned


def build_base_edges(base_width=BASE_WIDTH):
    if base_width <= 0.0:
        raise ValueError(f"base width must be positive, got {base_width}")
    n_bins = int(round(1.0 / base_width))
    if abs(n_bins * base_width - 1.0) > 1e-9:
        raise ValueError(f"base width {base_width} does not divide the [0, 1] score range evenly")
    return [round(i * base_width, 6) for i in range(n_bins + 1)]


def build_adaptive_edges(total_bkg, base_width=BASE_WIDTH):
    base_edges = build_base_edges(base_width)
    base_hist = rebin_with_edges(total_bkg, base_edges, f"{total_bkg.GetName()}_base")

    adaptive_edges = [base_edges[0]]
    bin_start = base_edges[0]
    content_sum = 0.0

    for ibin in range(1, base_hist.GetNbinsX() + 1):
        content_sum += base_hist.GetBinContent(ibin)
        high_edge = base_hist.GetXaxis().GetBinUpEdge(ibin)
        width = high_edge - bin_start
        is_last = ibin == base_hist.GetNbinsX()

        if content_sum >= ADAPTIVE_MIN_BKG or width >= ADAPTIVE_MAX_WIDTH or is_last:
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
        if last_content < ADAPTIVE_MIN_BKG and merged_width <= ADAPTIVE_MAX_WIDTH:
            adaptive_edges.pop(-2)

    return adaptive_edges


def parse_systematic_hist_name(hist_name):
    if hist_name.endswith("Up"):
        direction = "Up"
        base = hist_name[:-2]
    elif hist_name.endswith("Down"):
        direction = "Down"
        base = hist_name[:-4]
    else:
        return None

    for process in ORIGINAL_BKGS:
        prefix = f"{process}_"
        if base.startswith(prefix):
            return process, base[len(prefix):], direction
    return None


def central_backgrounds(hists, edges):
    grouped = {}
    for group in BKG_ORDER:
        source_hist = clone_sum(hists, GROUP_MAP[group], group)
        if source_hist is None:
            continue
        grouped[group] = rebin_with_edges(source_hist, edges, group)
    return grouped


def background_efficiency_threshold(total_bkg, eff=EFF_B_TARGET):
    """Score above which a fraction `eff` of `total_bkg` lies.

    Evaluated on the cache's own fine binning, BEFORE the adaptive rebinning,
    and interpolated linearly inside the crossing bin -- the working point is a
    property of the selection, not of the display binning, so the line must not
    snap to a drawn bin edge.
    """
    if not 0.0 < eff < 1.0:
        raise ValueError(f"background efficiency must be in (0, 1), got {eff}")

    n_bins = total_bkg.GetNbinsX()
    integral = total_bkg.Integral(1, n_bins)
    if integral <= 0.0:
        raise RuntimeError("total background is empty; cannot locate a working point")

    target = eff * integral
    running = 0.0
    axis = total_bkg.GetXaxis()
    for ibin in range(n_bins, 0, -1):
        content = total_bkg.GetBinContent(ibin)
        if running + content >= target:
            low, up = axis.GetBinLowEdge(ibin), axis.GetBinUpEdge(ibin)
            # Fraction of this bin that still has to be given away to the
            # high side; content > 0 is guaranteed by the test above.
            return up - (target - running) / content * (up - low)
        running += content
    raise RuntimeError(f"no score retains {eff:.0%} of the background")


def total_background(hists):
    total = clone_sum(hists, ORIGINAL_BKGS, "total_bkg")
    if total is None:
        raise RuntimeError("No background histograms found in cache")
    return total


def build_rebinned_originals(hists, edges):
    originals = {}
    for process in ORIGINAL_BKGS:
        if process in hists:
            originals[process] = rebin_with_edges(hists[process], edges, process)
    return originals


def collect_systematic_names(hists):
    systematics = {}
    for name in hists:
        parsed = parse_systematic_hist_name(name)
        if parsed is None:
            continue
        process, syst_name, direction = parsed
        systematics.setdefault(syst_name, {}).setdefault(direction, set()).add(process)
    return systematics


def make_total_variation(hists, central_originals, edges, syst_name, direction):
    total = None
    for process in ORIGINAL_BKGS:
        if process not in central_originals:
            continue
        varied_name = f"{process}_{syst_name}{direction}"
        if varied_name in hists:
            source = rebin_with_edges(hists[varied_name], edges, varied_name)
        else:
            source = central_originals[process]
        if total is None:
            total = source.Clone(f"total_{syst_name}{direction}")
            total.SetDirectory(0)
        else:
            total.Add(source)
    return total


def apply_total_uncertainty(grouped_bkgs, hists, edges):
    central_originals = build_rebinned_originals(hists, edges)
    total_central = None
    for hist in central_originals.values():
        if total_central is None:
            total_central = hist.Clone("total_central_rebinned")
            total_central.SetDirectory(0)
        else:
            total_central.Add(hist)
    if total_central is None:
        raise RuntimeError("No central background histograms available after rebinning")

    systematics = collect_systematic_names(hists)
    total_variations = {}
    for syst_name in systematics:
        up = make_total_variation(hists, central_originals, edges, syst_name, "Up")
        down = make_total_variation(hists, central_originals, edges, syst_name, "Down")
        if up is not None and down is not None:
            total_variations[syst_name] = (up, down)

    total_errors = []
    for ibin in range(1, total_central.GetNbinsX() + 1):
        stat_err2 = total_central.GetBinError(ibin) ** 2
        syst_err2 = 0.0
        central = total_central.GetBinContent(ibin)

        for up, down in total_variations.values():
            max_dev = max(
                abs(up.GetBinContent(ibin) - central),
                abs(down.GetBinContent(ibin) - central),
            )
            syst_err2 += max_dev ** 2

        total_errors.append(sqrt(stat_err2 + syst_err2))

    for ibin, total_err in enumerate(total_errors, start=1):
        total_content = sum(hist.GetBinContent(ibin) for hist in grouped_bkgs.values())
        if total_content <= 0.0 or total_err <= 0.0:
            for hist in grouped_bkgs.values():
                hist.SetBinError(ibin, 0.0)
            continue

        # ComparisonCanvas sums background errors in quadrature. Assign the
        # full total uncertainty to the dominant group in each bin so the
        # rendered summed band exactly matches the precomputed total.
        dominant = max(grouped_bkgs.values(), key=lambda hist: hist.GetBinContent(ibin))
        for hist in grouped_bkgs.values():
            hist.SetBinError(ibin, total_err if hist is dominant else 0.0)


def build_plot_objects(region, masspoint, base_width=BASE_WIDTH):
    hists = load_score_histograms(region, masspoint)
    required_hists = [*ORIGINAL_BKGS, "data_obs"]
    if region == "SR":
        required_hists.append(masspoint)
    for required in required_hists:
        if required not in hists:
            raise RuntimeError(f"{required} histogram missing for {region}/{masspoint}")

    total_bkg = total_background(hists)
    edges = build_adaptive_edges(total_bkg, base_width)
    # SR only: the TTZ CR has no working point of its own (loadScores does not
    # even apply the mass window there), so plotParticleNetScore draws no line
    # on it either.
    threshold = (background_efficiency_threshold(total_bkg)
                 if region == "SR" else None)

    data = rebin_with_edges(hists["data_obs"], edges, "data_obs")
    data.SetTitle("Data")

    bkgs = central_backgrounds(hists, edges)
    apply_total_uncertainty(bkgs, hists, edges)
    bkgs = {BKG_LABELS[group]: hist for group, hist in bkgs.items()}
    for label, hist in bkgs.items():
        hist.SetTitle(label)

    signals = {}
    if region == "SR":
        signal = rebin_with_edges(hists[masspoint], edges, masspoint)
        signal.Scale(SIGNAL_SCALE)
        signal.SetTitle("signal")
        signals["signal"] = signal

    return data, bkgs, signals, edges, threshold


def stack_maximum(bkgs):
    total = None
    for hist in bkgs.values():
        if total is None:
            total = hist.Clone("y_scale_total")
            total.SetDirectory(0)
        else:
            total.Add(hist)
    return total.GetMaximum() if total is not None else 0.0


def data_maximum(data):
    """Tallest data point including its error bar."""
    return max((data.GetBinContent(i) + data.GetBinError(i)
                for i in range(1, data.GetNbinsX() + 1)), default=0.0)


def build_y_range(data, bkgs, signals):
    """Explicit y range so the scale accounts for data as well as the stack.

    ComparisonCanvas sizes the axis from the background stack (and any signals)
    alone, so a data point above the stack would otherwise run off the top.
    """
    y_max = max(stack_maximum(bkgs), data_maximum(data),
                *(hist.GetMaximum() for hist in signals.values()) if signals else (0.0,))
    return [0.0, (y_max if y_max > 0 else 1.0) * Y_HEADROOM]


def build_config(region, edges, draw_legend=False, y_range=None, private_work=False):
    if region == "SR":
        channel_label = "SR"
        region_label = "e#mu#mu + #mu#mu#mu"
    elif region == "TTZCR":
        channel_label = "t#bar{t}Z CR"
        region_label = "ee#mu"
    else:
        raise ValueError(f"Unsupported region: {region}")

    return {
        "era": "All",
        # Per-energy luminosities, CMS style for multi-energy combinations:
        # "138 fb^-1 (13 TeV) + 62 fb^-1 (13.6 TeV)". cmsstyle appends the
        # CoM in parentheses, so the Run3 energy is carried by "CoM" and the
        # Run2 term is baked into "run_label".
        "CoM": f"{EnergyInfo['Run3']:g} TeV",
        "run_label": (f"{LumiInfo['Run2']:g} fb^{{#minus1}} ({EnergyInfo['Run2']:g} TeV) + "
                      f"{LumiInfo['Run3']:g} fb^{{#minus1}}"),
        "xTitle": "Modified LR Score",
        "yTitle": "Events",
        "rTitle": RATIO_LABEL,
        "systSrc": SYST_LABEL,
        # Histograms are already adaptively rebinned before construction.
        # Keeping xRange to endpoints avoids a second variable Rebin call.
        "xRange": [edges[0], edges[-1]],
        "rRange": [0.5, 1.5],
        # No in-plot legend to clear, so the stack can use the vertical space.
        "yHeadroom": Y_HEADROOM,
        "yRange": y_range,
        "maxDigits": 3,
        "overflow": False,
        "iPos": 11,
        # The figure places the CMS block itself; see CMS_LABEL_POS.
        "cmsPosX": CMS_LABEL_POS[0],
        "cmsPosY": CMS_LABEL_POS[1],
        "cmsLabelSize": CMS_LABEL_SIZE,
        # Published paper figures drop "Preliminary"; only "CMS" is drawn.
        "extraText": "",
        "hideOriginYLabel": HIDE_ORIGIN_Y_LABEL,
        # Two columns in the top-right corner, clearing the in-frame CMS
        # block on the left. The signal entry carries the mass point, so it
        # is the widest cell and takes the right column's second-to-last row,
        # with the working-point line under it.
        "legend": PRIVATE_WORK_LEGEND_BOX if private_work else LEGEND_BOX,
        "legendTextSize": LEGEND_TEXT_SIZE,
        "legendColumns": LEGEND_COLUMNS,
        # In-plot by default; --standalone-legend publishes it as its own panel.
        "drawLegend": draw_legend,
        "colors": [BKG_COLORS[name] for name in BKG_ORDER],
        "signalLineWidth": 3,
        "signalFill": False,
        "signalColors": [SIGNAL_COLOR],
        "channel": channel_label,
        "region": region_label,
        # Directly below the self-placed CMS block.
        "channelPosX": (PRIVATE_WORK_CHANNEL_POS if private_work else CHANNEL_POS)[0],
        "channelPosY": (PRIVATE_WORK_CHANNEL_POS if private_work else CHANNEL_POS)[1],
        "channelSize": CHANNEL_SIZE,
        "chi2_test": False,
        "normalize_chi2": False,
        # Private work: ComparisonCanvas draws no CMS text; the label is
        # drawn by draw_private_work_text.
        **({"cmsText": "", "extraText": ""} if private_work else {}),
    }


def draw_integrated_signal(plotter, signals, masspoint, draw_legend=False):
    plotter.update_y_scale(signals)
    plotter.canv.cd(1)
    plotter.signals = {}

    for name, hist in signals.items():
        signal_hist = hist.Clone(f"signal_{name}")
        signal_hist.SetDirectory(0)
        signal_hist.SetStats(0)
        signal_hist.SetLineColor(SIGNAL_COLOR)
        signal_hist.SetLineWidth(SIGNAL_LINE_WIDTH)
        signal_hist.SetFillStyle(0)
        signal_hist.Draw("HIST SAME")
        plotter.signals[name] = signal_hist
        plotter.leg.AddEntry(signal_hist, format_signal_label(masspoint),
                             SIGNAL_LEGEND_OPT)

    if draw_legend:
        plotter.leg.Draw()
    plotter.canv.cd(1).RedrawAxis()


def pad_legend_to_last_column(plotter):
    """Blank out cells so the NEXT entry lands in the legend's last column.

    TLegend fills row-major, so with the eight standard entries (data, five
    background groups, Stat.+Syst., signal) filling four full rows, a ninth
    would open a new row on the LEFT. The working-point line belongs on the
    right, under the signal it qualifies, so the left cell of that row is
    filled with an empty entry first. Written against GetNColumns() rather
    than the literal 2 so it still holds if the legend is ever re-laid out.
    """
    columns = plotter.leg.GetNColumns()
    if columns <= 1:
        return
    while plotter.leg.GetListOfPrimitives().GetSize() % columns != columns - 1:
        # A null object with no draw option renders as an empty cell.
        plotter.leg.AddEntry(0, "", "")


def draw_threshold_overlay(plotter, threshold):
    """Vertical eps_B = 20% marker in both pads, plus its legend entry.

    Mirrors plotParticleNetScore.draw_threshold_overlay, including the stop at
    half the axis maximum: at Y_HEADROOM = 1.9 that is just above the tallest
    stack point, which keeps the line clear of the in-plot legend it would
    otherwise cross (the legend spans x = 0.43 to 0.97, and the working point
    sits inside that range).
    """
    if threshold is None:
        return

    lines = []
    plotter.canv.cd(1)
    frame = CMS.GetCmsCanvasHist(plotter.canv.cd(1))
    line_up = ROOT.TLine(threshold, frame.GetMinimum(),
                         threshold, 0.5 * frame.GetMaximum())
    line_up.SetLineColor(THRESHOLD_COLOR)
    line_up.SetLineStyle(THRESHOLD_LINE_STYLE)
    line_up.SetLineWidth(THRESHOLD_LINE_WIDTH)
    line_up.Draw("SAME")
    lines.append(line_up)
    pad_legend_to_last_column(plotter)
    plotter.leg.AddEntry(line_up, THRESHOLD_LABEL, "L")
    plotter.canv.cd(1).RedrawAxis()

    plotter.canv.cd(2)
    rmin, rmax = plotter.config.get("rRange", [0.5, 1.5])
    line_down = ROOT.TLine(threshold, rmin, threshold, rmax)
    line_down.SetLineColor(THRESHOLD_COLOR)
    line_down.SetLineStyle(THRESHOLD_LINE_STYLE)
    line_down.SetLineWidth(THRESHOLD_LINE_WIDTH)
    line_down.Draw("SAME")
    lines.append(line_down)
    plotter.canv.cd(2).RedrawAxis()

    plotter._threshold_lines = lines  # keep alive until the canvas is written


def draw_private_work_label(plotter, masspoint):
    """Private-work label in place of the CMS block; mass point and chi2/ndf
    below the channel caption.

    chi2 is the same data vs. prediction test ComparisonCanvas runs with
    chi2_test (HistoUtils.calculate_chi2 on data and the Stat.+Syst. total),
    drawn here so its size and position follow this column.
    """
    plotter.canv.cd(1)
    CMS.drawText(PRIVATE_WORK_TEXT, posX=CMS_LABEL_POS[0], posY=CMS_LABEL_POS[1],
                 font=52, align=13, size=PRIVATE_WORK_SIZE)

    chi2, ndf, p_value = calculate_chi2(plotter.incl, plotter.systematics)
    if ndf <= 0:
        raise RuntimeError(f"chi2 test has no degrees of freedom for {masspoint}")
    # Baseline of the channel caption's second line ("region"), then one row per line.
    y_text = PRIVATE_WORK_CHANNEL_POS[1] - CHANNEL_SIZE
    for text in (format_signal_label(masspoint),
                 f"#chi^{{2}}/ndf = {chi2 / ndf:.2f} (p = {p_value:.2f})"):
        y_text -= PRIVATE_WORK_INFO_STEP
        CMS.drawText(text, posX=PRIVATE_WORK_CHANNEL_POS[0], posY=y_text,
                     font=42, align=11, size=PRIVATE_WORK_INFO_SIZE)
    logging.info("chi2/ndf for TTZCR/%s: %.2f/%d = %.2f (p = %.3f)",
                 masspoint, chi2, ndf, chi2 / ndf, p_value)
    plotter.canv.cd(1).RedrawAxis()


def offset_ndc_by_points(pad, x_ndc, y_ndc, dx_pt, dy_pt):
    """Nudge a pad-NDC position by an offset given in PDF points."""
    pad_width_pt = PANEL_SIZE_PT[0] * pad.GetAbsWNDC()
    pad_height_pt = PANEL_SIZE_PT[1] * pad.GetAbsHNDC()
    return x_ndc + dx_pt / pad_width_pt, y_ndc + dy_pt / pad_height_pt


def legend_output_path(output_root, with_signal=True):
    """Signal and control regions get their own panel, since only the former
    overlays a signal."""
    name = "legend.pdf" if with_signal else "legend_nosignal.pdf"
    return output_root / name


def paper_panel_size():
    """Pixel size of one paper plot, so the legend panel matches it exactly.

    Taken from a throwaway canvas built the way ComparisonCanvas builds its own,
    rather than from hardcoded numbers that would silently drift if cmsstyle
    changed its reference dimensions.
    """
    probe = CMS.cmsDiCanvas("panel_size_probe", 0., 1., 0., 1., 0., 1., "", "", "",
                            square=True, iPos=11, extraSpace=0)
    size = (probe.GetWindowWidth(), probe.GetWindowHeight())
    probe.Close()
    return size


def build_legend_proxies(with_signal=True, prefix=LEGEND_KEY):
    """Dummy objects styled like the drawn ones, plus their legend entries.

    Returns (entries, proxies); the caller must keep `proxies` alive until the
    canvas is written, since TLegend does not own them. The prefix keeps the
    ROOT names unique when several panels are built in one process.
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
    # The adaptive bins have unequal widths, so the data points keep their
    # horizontal bars; "PLE" draws the same cross in the legend. Matches
    # ComparisonCanvas.data_legend_option for the in-plot legend.
    entries = [(data, DATA_LABEL, "PLE")]

    # Same order as the in-plot legend: top of the stack listed first.
    for name in reversed(BKG_ORDER):
        proxy = new_proxy(name)
        proxy.SetFillColor(BKG_COLORS[name])
        proxy.SetLineColor(BKG_COLORS[name])
        proxy.SetFillStyle(1001)
        entries.append((proxy, BKG_LABELS[name], "F"))

    syst = new_proxy("syst")
    syst.SetFillStyle(SYST_FILL_STYLE)
    syst.SetFillColor(SYST_FILL_COLOR)
    syst.SetLineWidth(0)
    syst.SetMarkerSize(0)
    entries.append((syst, SYST_LABEL, " FE2"))

    if with_signal:
        signal = new_proxy("signal")
        signal.SetLineColor(SIGNAL_COLOR)
        signal.SetLineWidth(SIGNAL_LINE_WIDTH)
        signal.SetFillStyle(0)
        signal.SetFillStyle(1001)
        signal.SetMarkerSize(0)
        entries.append((signal, SIGNAL_LABEL, SIGNAL_LEGEND_OPT))

        # SR only, matching draw_threshold_overlay: the CR panels carry no
        # working-point line, so the no-signal panel must not advertise one.
        cut = new_proxy("threshold")
        cut.SetLineColor(THRESHOLD_COLOR)
        cut.SetLineStyle(THRESHOLD_LINE_STYLE)
        cut.SetLineWidth(THRESHOLD_LINE_WIDTH)
        cut.SetFillStyle(0)
        cut.SetMarkerSize(0)
        entries.append((cut, THRESHOLD_LABEL, "L"))

    return entries, proxies


def render_paper_legend(output_root, with_signal=True):
    """Write the shared legend as its own panel, sized like a paper plot."""
    width, height = paper_panel_size()
    name = "paper_legend" if with_signal else "paper_legend_nosignal"
    canvas = ROOT.TCanvas(name, name, 50, 50, width, height)
    canvas.SetFillColor(0)
    canvas.SetBorderMode(0)
    canvas.SetFrameFillStyle(0)
    canvas.SetFrameBorderMode(0)
    canvas.cd()

    entries, proxies = build_legend_proxies(with_signal=with_signal, prefix=name)
    row = LEGEND_PANEL_TEXT_SIZE * LEGEND_PANEL_ROW_SPACING

    # A single column, centred on the panel in both directions.
    block = row * len(entries)
    x1 = 0.5 * (1.0 - LEGEND_PANEL_WIDTH)
    legend = CMS.cmsLeg(x1, 0.5 - 0.5 * block, x1 + LEGEND_PANEL_WIDTH, 0.5 + 0.5 * block,
                        textSize=LEGEND_PANEL_TEXT_SIZE)
    legend.SetMargin(LEGEND_PANEL_MARGIN)
    CMS.addToLegend(legend, *entries)

    canvas.Update()
    out_path = legend_output_path(output_root, with_signal)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(str(out_path))
    canvas.Close()
    del proxies, legend
    return out_path


def draw_masspoint(region, masspoint, output_root, base_width=BASE_WIDTH, draw_legend=False,
                   private_work=False):
    if private_work and region != "TTZCR":
        raise ValueError(f"--private-work is TTZ CR only, got {region}")
    data, bkgs, signals, edges, threshold = build_plot_objects(
        region, masspoint, base_width)
    config = build_config(region, edges, draw_legend=draw_legend,
                          y_range=build_y_range(data, bkgs, signals),
                          private_work=private_work)

    out_path = output_path(output_root, region, masspoint, base_width_tag(base_width),
                           PRIVATE_WORK_SUFFIX if private_work else "")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = ComparisonCanvas(data, bkgs, config)
    plotter.drawPadUp()
    if signals:
        draw_integrated_signal(plotter, signals, masspoint, draw_legend=draw_legend)
    plotter.drawPadDown()
    # After the signal, so the legend reads ... Stat.+Syst., signal, cut; and
    # after drawPadDown, which is what creates the ratio pad the line needs.
    draw_threshold_overlay(plotter, threshold)
    if private_work:
        draw_private_work_label(plotter, masspoint)
    plotter.canv.SaveAs(str(out_path))
    return out_path, edges, threshold


def main():
    global SIGNAL_SOURCE
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s - %(message)s")
    SIGNAL_SOURCE = args.signal_source

    selected = DEFAULT_MASSPOINTS if args.masspoint == "all" else (args.masspoint,)
    if args.region == "all":
        selected_regions = ("TTZCR",) if args.private_work else REGIONS
    elif args.region == LEGEND_KEY:
        selected_regions = ()
    else:
        selected_regions = (args.region,)
    output_root = resolve_output_root(args.output_root)

    # Published once each: SR panels overlay a signal, the TTZ CR panels do not.
    if args.region in ("all", LEGEND_KEY) and args.standalone_legend and not args.private_work:
        for with_signal in (True, False):
            logging.info("Wrote %s", render_paper_legend(output_root, with_signal))

    for region in selected_regions:
        for masspoint in selected:
            out_path, edges, threshold = draw_masspoint(
                region, masspoint, output_root, args.base_width,
                draw_legend=not args.standalone_legend, private_work=args.private_work)
            logging.info("Wrote %s", out_path)
            logging.info(
                "Adaptive edges for %s/%s (%d bins): %s",
                region, masspoint, len(edges) - 1, edges
            )
            if threshold is not None:
                logging.info("eps_B = %.0f%% working point for %s/%s: %.4f",
                             100 * EFF_B_TARGET, region, masspoint, threshold)


if __name__ == "__main__":
    main()
