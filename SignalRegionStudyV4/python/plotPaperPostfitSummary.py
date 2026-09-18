#!/usr/bin/env python3
"""Produce paper-style b-only postfit mA summary plots from cached hists."""

import argparse
import ctypes
import logging
import os
import sys
from array import array
from pathlib import Path
from types import SimpleNamespace

import ROOT

import cmsstyle as CMS
import plotPostfitSummary as summary
# Paper wording is defined once in the LR_modified script; reuse it so the two
# figure sets cannot drift apart. These plots keep their legend in-panel, so
# only the labels are shared, not the standalone-legend machinery.
from plotPaperLRModified import (BKG_LABELS, CHANNEL_POS, CHANNEL_SIZE,  # noqa: E402
                                 CMS_LABEL_POS, CMS_LABEL_SIZE, DATA_LABEL,
                                 HIDE_ORIGIN_Y_LABEL,
                                 SIGNAL_SOURCE,
                                 LEGEND_KEY, MASS_LABEL_OFFSET_PT,
                                 MASS_LABEL_POS, MASS_LABEL_SIZE, RATIO_LABEL,
                                 SYST_LABEL, Y_HEADROOM, offset_ndc_by_points,
                                 render_paper_legend)


MODULE_DIR = Path(__file__).resolve().parents[1]
WORKDIR = Path(os.environ.get("WORKDIR", MODULE_DIR.parent))

sys.path.insert(0, str(WORKDIR / "Common" / "Tools"))
from plotter import (ComparisonCanvas, EnergyInfo,  # noqa: E402
                     LumiInfo, PALETTE_LONG, reanchor_lumi_header)


ROOT.gROOT.SetBatch(True)

MHC = 160
METHOD = "ParticleNet"
ERA = "All"
DEFAULT_CHANNELS = ("SR1E2Mu", "SR3Mu", "Combined")
FIT_TYPE = "b"
BIN_WIDTH = 1.0
REGIONS = (
    ("mA_lt85", "m_{A} < 85 GeV", None, 85.0, None, 85.0),
    ("mA_85to95", "85 #leq m_{A} #leq 95 GeV", 85.0, 95.0, 78.0, 104.0),
    ("mA_gt95", "m_{A} > 95 GeV", 95.0, None, 95.0, None),
)

BKG_ORDER = ("others", "conv", "diboson", "ttX", "nonprompt")
BKG_COLORS = {
    "nonprompt": PALETTE_LONG[0],
    "diboson": PALETTE_LONG[1],
    "ttX": PALETTE_LONG[2],
    "conv": PALETTE_LONG[3],
    "others": PALETTE_LONG[4],
}
# ComparisonCanvas labels the stack with the dict keys, so the merged
# backgrounds are keyed by legend label and the colours looked up by it too.
LABEL_COLORS = {BKG_LABELS[group]: BKG_COLORS[group] for group in BKG_ORDER}
BKG_GROUPS = {
    "others": ("others",),
    "conv": ("conv", "conversion"),
    "diboson": ("diboson", "WZ", "ZZ"),
    "ttX": ("ttX", "ttZ", "tZq", "ttH", "ttW"),
    "nonprompt": ("nonprompt",),
}
CHANNEL_LABELS = {
    "SR1E2Mu": ("SR", "e#mu#mu"),
    "SR3Mu": ("SR", "#mu#mu#mu"),
    "Combined": ("SR", "e#mu#mu + #mu#mu#mu"),
}

# The legend is published once as its own panel, so the mA range and the fit
# stage take over the top-right corner it used to occupy. Position, size and
# nudge are the mass-point label's from plotPaperLRModified.py, so the two
# figure sets carry their annotation in exactly the same place.
# Left-hand text block, below the two channel lines drawn at channelPosY. Kept
# relative to CHANNEL_POS/CHANNEL_SIZE so it follows the shared caption block
# instead of having to be re-tuned whenever that moves: the region line sits at
# channelPosY - channelSize, and this clears it by one more caption row.
REGION_LABEL_POS = (CHANNEL_POS[0], CHANNEL_POS[1] - 2 * CHANNEL_SIZE - 0.007)
REGION_LABEL_SIZE = 0.035
STAGE_LABEL_GAP = 0.05  # drop from the mA range line to the fit-stage line
FIT_STAGE_LABEL = "B-only Post-fit"

# --- Full-range panel -------------------------------------------------------
# One stitched b-only spectrum over the whole mA reach, following
# results/plots/postfit_summary/mHc160/ParticleNet/
# postfit_summary.mHc160.All.{channel}.ParticleNet.postfit_b.unblind.pdf and
# redrawn in the paper style. This is the published postfit figure; the three
# mA windows (REGIONS, --mode regions) are kept for diagnostics.
FULL_RANGE_STEM = "postfit_b_only"
# cmsDiCanvas is near-square, and 15-160 GeV of 1 GeV bins is unreadable in it,
# so the panel is published 16:9.
FULL_RANGE_CANVAS_SIZE = (1600, 900)
# The legend keeps its height but moves into the right-hand third: on a 16:9
# frame LEGEND_BOX would span half the width. Its left edge is not fixed --
# place_full_range_legend() pushes it clear of the on-Z handover line, which is
# drawn over the full frame height and would otherwise run through the box.
FULL_RANGE_LEGEND = (0.70, 0.54, 0.985, 0.86)
FULL_RANGE_LEGEND_CLEARANCE = 0.035
# TLegend centres each entry's text within its row rather than hanging it off
# the box's Y2NDC, so the "Data"/"Nonprompt" row sits below the box's own top
# edge. Measured against the rendered PDF text bounding boxes (pdftotext
# -bbox): with the caption's top at FULL_RANGE_LEGEND[3], "SR" sits 6.61 pt
# above "Data"'s glyph top; converting with the same px/pt scale used for
# the caption's earlier point-nudge (0.00913 NDC <-> 2.126 pt) gives this drop.
FULL_RANGE_LEGEND_TEXT_TOP_DROP = 0.00913 / 2.126 * 6.61
# Label placement on this panel is derived from the pad margins rather than
# fixed, because resize_canvas() rescales those for 16:9. Insets are measured
# from the frame's top-left corner.
FULL_RANGE_STAGE_GAP = 0.022   # blank below the caption's second row
# Bigger than the windows' REGION_LABEL_SIZE: this panel is the published one
# and the stage line is the only thing naming the fit, so it is sized between
# the mA-window labels and the CHANNEL_SIZE caption above it.
FULL_RANGE_STAGE_SIZE = 0.045
FULL_RANGE_RRANGE = [0.0, 3.0]
# Display range of the reference, [12, mHc]: the stitched edges run to 178 GeV
# because the topmost seed's fit window does, but mA <= mHc - 5 is the model's
# reach and anything past mHc is filled by nearest-owner extrapolation.
FULL_RANGE_XRANGE = (12.0, float(MHC))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create paper b-only postfit summary PDFs for mHc=160."
    )
    parser.add_argument("--output-root", default="results/plots/paper/Postfit")
    parser.add_argument("--mode", choices=("full-range", "regions"),
                        default="full-range",
                        help="'full-range' is the published panel: one stitched "
                             "spectrum over the whole mA reach. 'regions' is the "
                             "older split into the three mA windows, kept for "
                             "diagnostics (default: %(default)s)")
    parser.add_argument("--channels", nargs="+", choices=DEFAULT_CHANNELS,
                        default=None,
                        help="default: Combined for --mode full-range (the "
                             "published e-mu-mu + mu-mu-mu panel), all three "
                             "for --mode regions")
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="allow refilling fine-mass hists if a cache is missing")
    parser.add_argument("--standalone-legend", action="store_true",
                        dest="standalone_legend",
                        help="publish the legend as its own panel instead of "
                             f"drawing it in each plot (the '{LEGEND_KEY}' "
                             "panel; pre-2026-08-18 layout)")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_output_root(path):
    output_root = Path(path)
    if not output_root.is_absolute():
        output_root = MODULE_DIR / output_root
    return output_root


def loader_args(channel, plot_only, debug):
    return SimpleNamespace(
        mhc=[MHC],
        methods=[METHOD],
        eras=[ERA],
        channels=[channel],
        fit_channel="Combined",
        # V4 replaces V3's binning/unblind flags: the only binning is the
        # adaptive one (no name in the path) and blinding is a method
        # segment, so what selects the templates is the signal source.
        signal_source=SIGNAL_SOURCE,
        nuisance="fallback_lnn",
        fit_type=FIT_TYPE,
        bin_width=BIN_WIDTH,
        output_dir="",
        signal_line="none",
        signal_mas=[],
        wide_mhc=[],
        wide_factor=1.0,
        signal_region_style=False,
        plot_only=plot_only,
        debug=debug,
        blind=False,
    )


def load_channel_results(channel, plot_only, debug):
    args = loader_args(channel, plot_only, debug)
    sources = summary.discover_masspoint_sources(args, ERA, METHOD, MHC)
    if not sources:
        raise RuntimeError(f"No fitDiagnostics found for mHc={MHC}, {ERA}, {channel}, {METHOD}")

    results = []
    for source in sources:
        results.append(
            summary.load_one_masspoint(
                args,
                ERA,
                source["source_method"],
                channel,
                source["masspoint"],
                [FIT_TYPE],
            )
        )

    return results


def region_contains(ma, low, high):
    if low is None:
        return ma < high
    if high is None:
        return ma > low
    return low <= ma <= high


def build_region_content(results, low, high):
    region_results = [
        result for result in results
        if region_contains(result["ma"], low, high)
    ]
    if not region_results:
        raise RuntimeError(f"No cached mA results found for region bounds ({low}, {high})")

    edges = summary.build_edges(region_results, BIN_WIDTH)
    _prefit, postfit, data = build_stitched_content_fill_gaps(region_results, FIT_TYPE, edges)
    data.SetTitle(DATA_LABEL)
    return data, merge_backgrounds(postfit), edges, region_results


def nearest_owner_index(x, results):
    idx = summary.owner_index(x, results)
    if idx is not None:
        return idx
    return min(
        range(len(results)),
        key=lambda i: min(
            abs(x - results[i]["mass_min"]),
            abs(x - results[i]["mass_max"]),
        ),
    )


def stitch_histograms_fill_gaps(hist_list, results, edges, name):
    out = ROOT.TH1D(name, "", len(edges) - 1, array("d", edges))
    out.SetDirectory(0)
    for ibin in range(1, out.GetNbinsX() + 1):
        x = out.GetBinCenter(ibin)
        idx = nearest_owner_index(x, results)
        src = hist_list[idx]
        if src is None:
            continue
        src_bin = src.FindBin(x)
        if src_bin < 1 or src_bin > src.GetNbinsX():
            continue
        out.SetBinContent(ibin, src.GetBinContent(src_bin))
        out.SetBinError(ibin, src.GetBinError(src_bin))
    return out


def build_stitched_content_fill_gaps(results, fit_type, edges):
    ordered_bkgs = summary.all_backgrounds(results)
    pre_bkgs = {}
    post_bkgs = {}
    for bkg in ordered_bkgs:
        pre_list = [item["per_fit"][fit_type]["pre_bkgs"].get(bkg) for item in results]
        post_list = [item["per_fit"][fit_type]["post_bkgs"].get(bkg) for item in results]
        if any(hist is not None for hist in pre_list):
            pre_bkgs[bkg] = stitch_histograms_fill_gaps(pre_list, results, edges, f"{bkg}_pre")
        if any(hist is not None for hist in post_list):
            post_bkgs[bkg] = stitch_histograms_fill_gaps(post_list, results, edges, f"{bkg}_post_{fit_type}")

    data = stitch_histograms_fill_gaps(
        [item["per_fit"][fit_type]["data"] for item in results],
        results,
        edges,
        "data",
    )
    data.SetTitle("data")
    return pre_bkgs, post_bkgs, data


def clone_or_add(total, hist, name):
    if total is None:
        total = hist.Clone(name)
        total.SetDirectory(0)
    else:
        total.Add(hist)
    return total


def merge_backgrounds(backgrounds):
    merged = {}
    for group in BKG_ORDER:
        sources = (group,) if group in backgrounds else BKG_GROUPS[group]
        total = None
        for source in sources:
            hist = backgrounds.get(source)
            if hist is None:
                continue
            total = clone_or_add(total, hist, group)
        if total is not None and total.Integral() > 0:
            label = BKG_LABELS[group]
            total.SetTitle(label)
            merged[label] = total
    if not merged:
        raise RuntimeError("No postfit backgrounds available after paper grouping")
    return merged


def total_background(backgrounds):
    total = None
    for hist in backgrounds.values():
        total = clone_or_add(total, hist, "visible_total_bkg")
    return total


def visible_maximum(hist, x_min, x_max):
    max_value = 0.0
    for ibin in range(1, hist.GetNbinsX() + 1):
        x = hist.GetXaxis().GetBinCenter(ibin)
        if x_min <= x <= x_max:
            max_value = max(max_value, hist.GetBinContent(ibin))
    return max_value


def visible_y_range(data, backgrounds, x_min, x_max):
    total = total_background(backgrounds)
    max_value = max(
        visible_maximum(total, x_min, x_max),
        visible_maximum(data, x_min, x_max),
    )
    if max_value <= 0.0:
        max_value = 1.0
    # Headroom for the in-frame CMS block, the caption block and the legend.
    return [0.0, max_value * Y_HEADROOM]


def build_config(channel, edges, data, backgrounds, display_low, display_high,
                 draw_legend=False):
    channel_label, region_label = CHANNEL_LABELS[channel]
    x_min = edges[0] if display_low is None else display_low
    x_max = edges[-1] if display_high is None else display_high
    y_range = visible_y_range(data, backgrounds, x_min, x_max)
    return {
        "era": "All",
        # Per-energy luminosities, CMS style for multi-energy combinations:
        # "138 fb^-1 (13 TeV) + 62 fb^-1 (13.6 TeV)". cmsstyle appends the
        # CoM in parentheses, so the Run3 energy is carried by "CoM" and the
        # Run2 term is baked into "run_label".
        "CoM": f"{EnergyInfo['Run3']:g} TeV",
        "run_label": (f"{LumiInfo['Run2']:g} fb^{{#minus1}} ({EnergyInfo['Run2']:g} TeV) + "
                      f"{LumiInfo['Run3']:g} fb^{{#minus1}}"),
        "xTitle": "m(#mu^{+}, #mu^{-}) [GeV]",
        "yTitle": f"Events / {BIN_WIDTH:g} GeV",
        "rTitle": RATIO_LABEL,
        "systSrc": SYST_LABEL,
        "xRange": [x_min, x_max],
        "yRange": y_range,
        "rRange": [0.0, 3.5],
        "maxDigits": 3,
        "overflow": False,
        "iPos": 11,
        # The figure places the CMS block itself; see plotPaperLRModified.
        "cmsPosX": CMS_LABEL_POS[0],
        "cmsPosY": CMS_LABEL_POS[1],
        "cmsLabelSize": CMS_LABEL_SIZE,
        # Published paper figures drop "Preliminary"; only "CMS" is drawn.
        "extraText": "",
        "hideOriginYLabel": HIDE_ORIGIN_Y_LABEL,
        # Two columns: data + 5 background groups + Stat.+Syst. fill four rows.
        # The right edge stops short of the frame so the longest label
        # ("Nonprompt") clears the right-hand axis ticks.
        # Two columns in the top-right corner, matching the LR_modified
        # panels. Data + 5 background groups + Stat.+Syst. fill four rows.
        # Same vertical placement as the LR_modified panels' LEGEND_BOX: air
        # above the box rather than pressed against the frame top. The left
        # edge stays at 0.46 -- these entries are shorter than the LR ones.
        "legend": (0.46, 0.54, 0.97, 0.86),
        "legendTextSize": 0.030,
        "legendColumns": 2,
        # In-plot by default; --standalone-legend publishes it as its own panel.
        "drawLegend": draw_legend,
        "colors": [LABEL_COLORS[name] for name in backgrounds.keys()],
        "channel": channel_label,
        "region": region_label,
        # Directly below the self-placed CMS block.
        "channelPosX": CHANNEL_POS[0],
        "channelPosY": CHANNEL_POS[1],
        "channelSize": CHANNEL_SIZE,
        "chi2_test": False,
        "normalize_chi2": False,
    }


def ndc_text_width(pad, latex):
    """Rendered width of an NDC TLatex, in pad NDC.

    TLatex::GetXsize reports axis units, which differ per panel because each mA
    region spans a different mass range. The bounding box is in pixels, so it
    converts cleanly and gives equal widths for equal-length labels.
    """
    width_px, height_px = ctypes.c_uint(0), ctypes.c_uint(0)
    latex.GetBoundingBox(width_px, height_px)
    pad_width_px = pad.GetWw() * pad.GetAbsWNDC()
    return width_px.value / pad_width_px if pad_width_px else 0.0


def draw_region_label(plotter, label, pos=None, align=11, size=None):
    """mA range and fit stage, left-aligned under the channel block.

    The in-plot legend owns the top-right corner these used to sit in, so
    they join the left-hand stack (CMS / Preliminary / SR / final state),
    which is where the non-paper postfit summaries put the same two lines.

    `label` may be None -- the full-range panel covers the whole scan, so it
    has no mA window to name, and the mHc is already carried by the caption of
    the figure. The fit stage then moves up into the freed line.
    """
    plotter.canv.cd(1)

    pos = REGION_LABEL_POS if pos is None else pos
    size = REGION_LABEL_SIZE if size is None else size
    labels = []
    stage_y = pos[1]
    if label is not None:
        region = ROOT.TLatex()
        region.SetNDC(True)
        region.SetTextFont(42)
        region.SetTextSize(size)
        region.SetTextAlign(align)
        region.DrawLatex(*pos, label)
        labels.append(region)
        stage_y -= STAGE_LABEL_GAP

    stage = ROOT.TLatex()
    stage.SetNDC(True)
    stage.SetTextFont(62)
    stage.SetTextSize(size)
    stage.SetTextAlign(align)
    stage.DrawLatex(pos[0], stage_y, FIT_STAGE_LABEL)
    labels.append(stage)

    plotter._paper_region_labels = labels


def output_path(output_root, channel, region_name):
    return output_root / f"postfit_b_mHc{MHC}_{channel}_{region_name}.pdf"


def full_range_output_path(output_root, channel):
    """Combined carries no filename token, as everywhere else in the module."""
    token = "" if channel == "Combined" else f"_{channel}"
    return output_root / f"{FULL_RANGE_STEM}{token}.pdf"


def build_full_range_content(results):
    """Every seed stitched into one spectrum over the full mA reach."""
    edges = summary.build_edges(results, BIN_WIDTH)
    _prefit, postfit, data = build_stitched_content_fill_gaps(results, FIT_TYPE, edges)
    data.SetTitle(DATA_LABEL)
    return data, merge_backgrounds(postfit), edges


def resize_canvas(plotter, size):
    """Reshape the canvas, and undo what that does to the y-axis spacing.

    ROOT measures y-axis label and title offsets against the pad WIDTH, so
    making the canvas proportionally wider pushes the numbers and the axis
    title away from the axis by exactly that factor -- on a 16:9 frame that is
    nearly 2x the gap the square panels have. Dividing both offsets by the
    stretch puts them back. Must run before drawPadUp: the pads size
    themselves from the canvas, and everything placed later is in NDC.
    """
    width, height = size
    stretch = ((width / height)
               / (plotter.canv.GetWindowWidth() / plotter.canv.GetWindowHeight()))
    upper = plotter.canv.cd(1)
    # Margins as the SQUARE paper panels have them, kept so the CMS block can be
    # given the same offsets from the frame corner that those panels use.
    plotter._paper_resize = {"stretch": stretch,
                             "left": upper.GetLeftMargin(),
                             "top": upper.GetTopMargin()}
    plotter.canv.SetCanvasSize(width, height)
    plotter.canv.SetWindowSize(width, height)
    for pad_index in (1, 2):
        pad = plotter.canv.cd(pad_index)
        # Same story as the offsets: the side margins are fractions of the pad
        # WIDTH, so cmsstyle's 0.15/0.05 become half the panel of empty paper
        # once the canvas is 16:9. Dividing them by the stretch keeps the same
        # physical room for the y title and labels and gives the rest to the
        # spectrum.
        pad.SetLeftMargin(pad.GetLeftMargin() / stretch)
        pad.SetRightMargin(pad.GetRightMargin() / stretch)
        axis = CMS.GetCmsCanvasHist(pad).GetYaxis()
        axis.SetLabelOffset(axis.GetLabelOffset() / stretch)
        axis.SetTitleOffset(axis.GetTitleOffset() / stretch)
    # cmsstyle's CMS_lumi() already ran inside cmsDiCanvas and right-aligned the
    # luminosity header to the margin it saw then, so after the rescale it stops
    # short of the frame.
    reanchor_lumi_header(plotter.canv.cd(1), width, height)
    plotter.canv.Modified()
    plotter.canv.Update()


def place_full_range_labels(plotter, x_range, boundaries):
    """CMS block into the frame's top-left corner, caption over the below-Z arm.

    Both are anchored on the live pad margins, which resize_canvas() has just
    rescaled, so the block sits against the corner at any canvas shape. The
    caption is centred on the region LEFT of the first handover guide -- the
    below-Z part of the spectrum it describes -- rather than at a fixed x.
    """
    pad = plotter.canv.cd(1)
    left, right, top = (pad.GetLeftMargin(), pad.GetRightMargin(),
                        pad.GetTopMargin())
    span = 1.0 - left - right
    frame_top = 1.0 - top
    resize = plotter._paper_resize

    # Same offsets from the frame corner that CMS_LABEL_POS gives the square
    # paper panels. The x offset is a fraction of WIDTH, so it is divided by
    # the stretch to stay the same physical inset; the y offset is not.
    inset_x = (CMS_LABEL_POS[0] - resize["left"]) / resize["stretch"]
    inset_y = (1.0 - resize["top"]) - CMS_LABEL_POS[1]
    plotter._cms_label["posX"] = left + inset_x
    plotter._cms_label["posY"] = frame_top - inset_y

    # SR / final state / fit stage: one left-aligned block over the below-Z
    # part of the spectrum. Its top matches the top of the legend's own
    # "Data"/"Nonprompt" row text, not the legend box's outer edge, so the
    # caption and the first legend row read as one horizontal line.
    x_min, x_max = x_range
    first_guide = min(boundaries) if boundaries else x_max
    guide_ndc = left + (first_guide - x_min) / (x_max - x_min) * span
    caption_x = 0.5 * (left + guide_ndc)
    caption_top = FULL_RANGE_LEGEND[3] - FULL_RANGE_LEGEND_TEXT_TOP_DROP
    plotter.config["channelPosX"] = caption_x
    plotter.config["channelPosY"] = caption_top
    plotter.config["channelAlign"] = 13   # left, top -- all three lines flush
    # The caption is two rows of CHANNEL_SIZE hanging off channelPosY, so the
    # fit-stage line has to start below both of them, not below the first.
    return (caption_x, caption_top - 2 * CHANNEL_SIZE - FULL_RANGE_STAGE_GAP)


def place_full_range_legend(plotter, boundaries, x_range):
    """Push the legend clear of the rightmost arm-handover guide.

    The guides span the full frame height, so a legend starting at or before
    the upper on-Z boundary gets a dashed line drawn straight through it. The
    box is anchored off the guide rather than at a fixed NDC so it stays clear
    if the ParticleNet window, the display range or the margins ever move.
    """
    pad = plotter.canv.cd(1)
    left, right = pad.GetLeftMargin(), pad.GetRightMargin()
    x_min, x_max = x_range
    x1 = FULL_RANGE_LEGEND[0]
    if boundaries:
        guide_ndc = (left + (max(boundaries) - x_min) / (x_max - x_min)
                     * (1.0 - left - right))
        x1 = max(x1, guide_ndc + FULL_RANGE_LEGEND_CLEARANCE)
    plotter.leg.SetX1NDC(x1)
    plotter.leg.SetX2NDC(FULL_RANGE_LEGEND[2])
    plotter.leg.SetY1NDC(FULL_RANGE_LEGEND[1])
    plotter.leg.SetY2NDC(FULL_RANGE_LEGEND[3])


def hide_top_ratio_label(plotter):
    """Drop the ratio axis' topmost label.

    Same defect as the upper pad's origin label and the same reason: cmsstyle
    gives the ratio pad a ZERO top margin, so a label sitting on the top of its
    range is cut in half by the pad edge -- and with rRange [0, 3] the ladder
    ends exactly there. Widening the margin instead would open a gap between
    the two frames.
    """
    axis = CMS.GetCmsCanvasHist(plotter.canv.cd(2)).GetYaxis()
    axis.ChangeLabel(-1, -1, 0.0)


def draw_full_range(output_root, channel, plot_only, debug, draw_legend=True):
    results = load_channel_results(channel, plot_only, debug)
    logging.info("%s: loaded %d cached masspoints", channel, len(results))

    data, backgrounds, edges = build_full_range_content(results)
    config = build_config(channel, edges, data, backgrounds,
                          FULL_RANGE_XRANGE[0], FULL_RANGE_XRANGE[1],
                          draw_legend=draw_legend)
    config["rRange"] = FULL_RANGE_RRANGE
    config["legend"] = FULL_RANGE_LEGEND
    logging.info("%s/full-range: x [%s, %s], y [%s, %s], backgrounds=%s",
                 channel, edges[0], edges[-1],
                 config["yRange"][0], config["yRange"][1],
                 ", ".join(backgrounds.keys()))

    out_path = full_range_output_path(output_root, channel)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = ComparisonCanvas(data.Clone(f"data_{channel}_full"), {
        name: hist.Clone(f"{name}_{channel}_full")
        for name, hist in backgrounds.items()
    }, config)
    resize_canvas(plotter, FULL_RANGE_CANVAS_SIZE)
    # The two dashed verticals mark where the panel hands over between the
    # Baseline and ParticleNet arms -- the only ownership change worth drawing,
    # as ownership_boundaries() explains. Per-seed edges are deliberately not.
    intervals = summary.collect_ownership(results, edges)
    boundaries = summary.ownership_boundaries(intervals, FULL_RANGE_XRANGE)
    stage_pos = place_full_range_labels(plotter, FULL_RANGE_XRANGE, boundaries)

    plotter.drawPadUp()
    # No mA window to name on the full-range panel, and no mHc line: the
    # b-only spectrum does not depend on the signal hypothesis being scanned.
    draw_region_label(plotter, None, pos=stage_pos, align=13,
                      size=FULL_RANGE_STAGE_SIZE)
    plotter.drawPadDown()
    hide_top_ratio_label(plotter)
    summary.draw_ownership_guides(plotter.canv, intervals, FULL_RANGE_XRANGE)
    place_full_range_legend(plotter, boundaries, FULL_RANGE_XRANGE)
    plotter.canv.SaveAs(str(out_path))
    logging.info("Wrote %s", out_path)


def draw_channel(output_root, channel, plot_only, debug, draw_legend=False):
    results = load_channel_results(channel, plot_only, debug)
    logging.info(
        "%s: loaded %d cached masspoints: %s",
        channel,
        len(results),
        ", ".join(result["masspoint"] for result in results),
    )

    for region_name, label, low, high, display_low, display_high in REGIONS:
        data, backgrounds, edges, region_results = build_region_content(results, low, high)
        x_min = edges[0] if display_low is None else display_low
        x_max = edges[-1] if display_high is None else display_high
        logging.info(
            "%s/%s: %d masspoints, cache x-range [%s, %s], display x-range [%s, %s], backgrounds=%s",
            channel,
            region_name,
            len(region_results),
            edges[0],
            edges[-1],
            x_min,
            x_max,
            ", ".join(backgrounds.keys()),
        )
        config = build_config(channel, edges, data, backgrounds, display_low, display_high,
                              draw_legend=draw_legend)
        logging.info(
            "%s/%s: display y-range [%s, %s]",
            channel,
            region_name,
            config["yRange"][0],
            config["yRange"][1],
        )
        out_path = output_path(output_root, channel, region_name)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        plotter = ComparisonCanvas(data.Clone(f"data_{channel}_{region_name}"), {
            name: hist.Clone(f"{name}_{channel}_{region_name}")
            for name, hist in backgrounds.items()
        }, config)
        plotter.drawPadUp()
        draw_region_label(plotter, label)
        plotter.drawPadDown()
        plotter.canv.SaveAs(str(out_path))
        logging.info("Wrote %s", out_path)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s - %(message)s",
    )
    output_root = resolve_output_root(args.output_root)
    plot_only = not args.rebuild_cache

    # These panels carry no signal overlay, so the no-signal variant is the one
    # they share. Published once, like the LR_modified figures.
    if args.standalone_legend:
        logging.info("Wrote %s", render_paper_legend(output_root, with_signal=False))

    channels = args.channels
    if channels is None:
        channels = ["Combined"] if args.mode == "full-range" else list(DEFAULT_CHANNELS)

    for channel in channels:
        if args.mode == "full-range":
            draw_full_range(output_root, channel, plot_only, args.debug,
                            draw_legend=not args.standalone_legend)
        else:
            draw_channel(output_root, channel, plot_only, args.debug,
                         draw_legend=not args.standalone_legend)


if __name__ == "__main__":
    main()
