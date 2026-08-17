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

import plotPostfitSummary as summary
# Paper wording is defined once in the LR_modified script; reuse it so the two
# figure sets cannot drift apart. These plots keep their legend in-panel, so
# only the labels are shared, not the standalone-legend machinery.
from plotPaperLRModified import (BKG_LABELS, DATA_LABEL,  # noqa: E402
                                 SIGNAL_SOURCE,
                                 LEGEND_KEY, MASS_LABEL_OFFSET_PT,
                                 MASS_LABEL_POS, MASS_LABEL_SIZE, RATIO_LABEL,
                                 SYST_LABEL, Y_HEADROOM, offset_ndc_by_points,
                                 render_paper_legend)


MODULE_DIR = Path(__file__).resolve().parents[1]
WORKDIR = Path(os.environ.get("WORKDIR", MODULE_DIR.parent))

sys.path.insert(0, str(WORKDIR / "Common" / "Tools"))
from plotter import (ComparisonCanvas, EnergyInfo,  # noqa: E402
                     LumiInfoExact, PALETTE_LONG)


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
# Left-hand text block, below the two channel lines drawn at channelPosY.
REGION_LABEL_POS = (0.22, 0.60)
REGION_LABEL_SIZE = 0.035
STAGE_LABEL_GAP = 0.05  # drop from the mA range line to the fit-stage line
FIT_STAGE_LABEL = "B-only Post-fit"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create paper b-only postfit summary PDFs for mHc=160."
    )
    parser.add_argument("--output-root", default="results/plots/paper/Postfit")
    parser.add_argument("--channels", nargs="+", choices=DEFAULT_CHANNELS, default=list(DEFAULT_CHANNELS))
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
    # No in-plot legend to clear, so the stack can use the vertical space.
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
        # "138 fb^-1 (13 TeV) + 62.4 fb^-1 (13.6 TeV)". cmsstyle appends the
        # CoM in parentheses, so the Run3 energy is carried by "CoM" and the
        # Run2 term is baked into "run_label".
        "CoM": f"{EnergyInfo['Run3']:g} TeV",
        "run_label": (f"{LumiInfoExact['Run2']:g} fb^{{#minus1}} ({EnergyInfo['Run2']:g} TeV) + "
                      f"{LumiInfoExact['Run3']:g} fb^{{#minus1}}"),
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
        # Two columns: data + 5 background groups + Stat.+Syst. fill four rows.
        # The right edge stops short of the frame so the longest label
        # ("Nonprompt") clears the right-hand axis ticks.
        # Two columns in the top-right corner, matching the LR_modified
        # panels. Data + 5 background groups + Stat.+Syst. fill four rows.
        "legend": (0.46, 0.58, 0.97, 0.90),
        "legendTextSize": 0.030,
        "legendColumns": 2,
        # In-plot by default; --standalone-legend publishes it as its own panel.
        "drawLegend": draw_legend,
        "colors": [LABEL_COLORS[name] for name in backgrounds.keys()],
        "channel": channel_label,
        "region": region_label,
        "channelPosX": 0.22,
        # iPos=11 puts "CMS"/"Preliminary" inside the frame, so the channel
        # block starts lower to clear them.
        "channelPosY": 0.72,
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


def draw_region_label(plotter, label):
    """mA range and fit stage, left-aligned under the channel block.

    The in-plot legend owns the top-right corner these used to sit in, so
    they join the left-hand stack (CMS / Preliminary / SR / final state),
    which is where the non-paper postfit summaries put the same two lines.
    """
    plotter.canv.cd(1)

    region = ROOT.TLatex()
    region.SetNDC(True)
    region.SetTextFont(42)
    region.SetTextSize(REGION_LABEL_SIZE)
    region.SetTextAlign(11)
    region.DrawLatex(*REGION_LABEL_POS, label)

    stage = ROOT.TLatex()
    stage.SetNDC(True)
    stage.SetTextFont(62)
    stage.SetTextSize(REGION_LABEL_SIZE)
    stage.SetTextAlign(11)
    stage.DrawLatex(REGION_LABEL_POS[0],
                    REGION_LABEL_POS[1] - STAGE_LABEL_GAP, FIT_STAGE_LABEL)

    plotter._paper_region_labels = [region, stage]


def output_path(output_root, channel, region_name):
    return output_root / f"postfit_b_mHc{MHC}_{channel}_{region_name}.pdf"


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

    for channel in args.channels:
        draw_channel(output_root, channel, plot_only, args.debug,
                     draw_legend=not args.standalone_legend)


if __name__ == "__main__":
    main()
