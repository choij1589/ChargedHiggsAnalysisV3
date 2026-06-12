#!/usr/bin/env python3
"""Produce paper-style b-only postfit mA summary plots from cached hists."""

import argparse
import logging
import os
import sys
from array import array
from pathlib import Path
from types import SimpleNamespace

import ROOT

import plotPostfitSummary as summary


MODULE_DIR = Path(__file__).resolve().parents[1]
WORKDIR = Path(os.environ.get("WORKDIR", MODULE_DIR.parent))

sys.path.insert(0, str(WORKDIR / "Common" / "Tools"))
from plotter import ComparisonCanvas, PALETTE_LONG  # noqa: E402


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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create paper b-only postfit summary PDFs for mHc=160."
    )
    parser.add_argument("--output-root", default="results/paper/Postfit")
    parser.add_argument("--channels", nargs="+", choices=DEFAULT_CHANNELS, default=list(DEFAULT_CHANNELS))
    parser.add_argument("--rebuild-cache", action="store_true",
                        help="allow refilling fine-mass hists if a cache is missing")
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
        binning="extended",
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
        unblind=True,
        partial_unblind=False,
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
    data.SetTitle("Data")
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
            total.SetTitle(group)
            merged[group] = total
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
    return [0.0, max_value * 2.0]


def build_config(channel, edges, data, backgrounds, display_low, display_high):
    channel_label, region_label = CHANNEL_LABELS[channel]
    x_min = edges[0] if display_low is None else display_low
    x_max = edges[-1] if display_high is None else display_high
    y_range = visible_y_range(data, backgrounds, x_min, x_max)
    return {
        "era": "All",
        "CoM": "13/13.6",
        "run_label": "Run 2+3, 200 fb^{#minus1}",
        "xTitle": "M(#mu^{+}#mu^{-}) [GeV]",
        "yTitle": f"Events / {BIN_WIDTH:g} GeV",
        "rTitle": "Data / Pred",
        "xRange": [x_min, x_max],
        "yRange": y_range,
        "rRange": [0.0, 5.0],
        "maxDigits": 3,
        "overflow": False,
        "iPos": 0,
        "legend": (0.72, 0.55, 0.99, 0.89),
        "legendTextSize": 0.038,
        "legendColumns": 1,
        "colors": [BKG_COLORS[name] for name in backgrounds.keys()],
        "channel": channel_label,
        "region": region_label,
        "channelPosX": 0.22,
        "channelPosY": 0.75,
        "chi2_test": False,
        "normalize_chi2": False,
    }


def draw_region_label(plotter, label):
    plotter.canv.cd(1)
    labels = []
    region = ROOT.TLatex()
    region.SetNDC(True)
    region.SetTextFont(42)
    region.SetTextSize(0.045)
    region.DrawLatex(0.22, 0.65, label)
    labels.append(region)

    fit_label = ROOT.TLatex()
    fit_label.SetNDC(True)
    fit_label.SetTextFont(42)
    fit_label.SetTextSize(0.04)
    fit_label.DrawLatex(0.22, 0.60, "B-only Post-fit")
    labels.append(fit_label)
    plotter._paper_region_labels = labels


def output_path(output_root, channel, region_name):
    return output_root / f"postfit_b_mHc{MHC}_{channel}_{region_name}.pdf"


def draw_channel(output_root, channel, plot_only, debug):
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
        config = build_config(channel, edges, data, backgrounds, display_low, display_high)
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

    for channel in args.channels:
        draw_channel(output_root, channel, plot_only, args.debug)


if __name__ == "__main__":
    main()
