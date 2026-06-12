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
from plotter import ComparisonCanvas, PALETTE_LONG  # noqa: E402


ROOT.gROOT.SetBatch(True)

SCORE_KEY = "LR_modified"
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

BASE_WIDTH = 0.02
ADAPTIVE_MIN_BKG = 10.0
ADAPTIVE_MAX_WIDTH = 0.20
SIGNAL_SCALE = 6.0
SIGNAL_COLOR = ROOT.kBlack


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
        choices=("all", *REGIONS),
        default="all",
        help="plot region to produce",
    )
    parser.add_argument(
        "--output-root",
        default="results/paper",
        help="base output directory, relative to SignalRegionStudyV3 unless absolute",
    )
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    return parser.parse_args()


def resolve_output_root(path):
    output_root = Path(path)
    if not output_root.is_absolute():
        output_root = MODULE_DIR / output_root
    return output_root


def cache_path(region, masspoint):
    if region == "SR":
        channel = "Combined"
        score_region = "Combined"
    elif region == "TTZCR":
        channel = "SR3Mu"
        score_region = "TTZ2E1Mu"
    else:
        raise ValueError(f"Unsupported region: {region}")

    return (
        MODULE_DIR / "templates" / "All" / channel / masspoint / "ParticleNet"
        / "extended_unblind" / "scores" / score_region / "histograms.root"
    )


def output_path(output_root, region, masspoint):
    return output_root / region / f"{SCORE_KEY}_{masspoint}.pdf"


def format_signal_label(masspoint):
    match = re.fullmatch(r"MHc(\d+)_MA(\d+)", masspoint)
    if not match:
        return masspoint
    mhc, ma = match.groups()
    return f"(m_{{H^{{+}}}}, m_{{A}}) = ({mhc}, {ma}) GeV"


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


def build_base_edges():
    n_bins = int(round(1.0 / BASE_WIDTH))
    return [round(i * BASE_WIDTH, 6) for i in range(n_bins + 1)]


def build_adaptive_edges(total_bkg):
    base_edges = build_base_edges()
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


def build_plot_objects(region, masspoint):
    hists = load_score_histograms(region, masspoint)
    required_hists = [*ORIGINAL_BKGS, "data_obs"]
    if region == "SR":
        required_hists.append(masspoint)
    for required in required_hists:
        if required not in hists:
            raise RuntimeError(f"{required} histogram missing for {region}/{masspoint}")

    total_bkg = total_background(hists)
    edges = build_adaptive_edges(total_bkg)

    data = rebin_with_edges(hists["data_obs"], edges, "data_obs")
    data.SetTitle("Data")

    bkgs = central_backgrounds(hists, edges)
    apply_total_uncertainty(bkgs, hists, edges)
    for group, hist in bkgs.items():
        hist.SetTitle(group)

    signals = {}
    if region == "SR":
        signal = rebin_with_edges(hists[masspoint], edges, masspoint)
        signal.Scale(SIGNAL_SCALE)
        signal.SetTitle("signal")
        signals["signal"] = signal

    return data, bkgs, signals, edges


def build_config(region, edges):
    if region == "SR":
        channel_label = "SR"
        region_label = "e#mu#mu + #mu#mu#mu"
    elif region == "TTZCR":
        channel_label = "TTZ CR"
        region_label = "ee#mu"
    else:
        raise ValueError(f"Unsupported region: {region}")

    return {
        "era": "All",
        "CoM": "13/13.6",
        "run_label": "Run 2+3, 200 fb^{#minus1}",
        "xTitle": "Modified LR Score",
        "yTitle": "Events",
        "rTitle": "Data / Pred",
        # Histograms are already adaptively rebinned before construction.
        # Keeping xRange to endpoints avoids a second variable Rebin call.
        "xRange": [edges[0], edges[-1]],
        "rRange": [0.0, 2.0],
        "maxDigits": 3,
        "overflow": False,
        "iPos": 0,
        "legend": (0.72, 0.55, 0.99, 0.89),
        "legendTextSize": 0.038,
        "legendColumns": 1,
        "colors": [BKG_COLORS[name] for name in BKG_ORDER],
        "signalLineWidth": 3,
        "signalFill": True,
        "signalFillAlpha": 0.18,
        "signalColors": [SIGNAL_COLOR],
        "channel": channel_label,
        "region": region_label,
        "channelPosY": 0.75,
        "channelPosX": 0.22,
        "chi2_test": False,
        "normalize_chi2": False,
    }


def draw_integrated_signal(plotter, signals):
    plotter.update_y_scale(signals)
    plotter.canv.cd(1)
    plotter.signals = {}

    for name, hist in signals.items():
        signal_hist = hist.Clone(f"signal_{name}")
        signal_hist.SetDirectory(0)
        signal_hist.SetStats(0)
        signal_hist.SetLineColor(SIGNAL_COLOR)
        signal_hist.SetLineWidth(3)
        signal_hist.SetFillColorAlpha(SIGNAL_COLOR, 0.18)
        signal_hist.SetFillStyle(1001)
        signal_hist.Draw("HIST SAME")
        plotter.signals[name] = signal_hist
        plotter.leg.AddEntry(signal_hist, name, "L")

    plotter.leg.Draw()
    plotter.canv.cd(1).RedrawAxis()


def draw_masspoint(region, masspoint, output_root):
    data, bkgs, signals, edges = build_plot_objects(region, masspoint)
    config = build_config(region, edges)

    out_path = output_path(output_root, region, masspoint)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = ComparisonCanvas(data, bkgs, config)
    plotter.drawPadUp()
    if signals:
        draw_integrated_signal(plotter, signals)
    plotter.canv.cd(1)
    plotter.mass_label = ROOT.TLatex()
    plotter.mass_label.SetNDC(True)
    plotter.mass_label.SetTextFont(42)
    plotter.mass_label.SetTextSize(0.045)
    plotter.mass_label.DrawLatex(0.22, 0.65, format_signal_label(masspoint))
    plotter.drawPadDown()
    plotter.canv.SaveAs(str(out_path))
    return out_path, edges


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s - %(message)s")

    selected = DEFAULT_MASSPOINTS if args.masspoint == "all" else (args.masspoint,)
    selected_regions = REGIONS if args.region == "all" else (args.region,)
    output_root = resolve_output_root(args.output_root)

    for region in selected_regions:
        for masspoint in selected:
            out_path, edges = draw_masspoint(region, masspoint, output_root)
            logging.info("Wrote %s", out_path)
            logging.info(
                "Adaptive edges for %s/%s (%d bins): %s",
                region, masspoint, len(edges) - 1, edges
            )


if __name__ == "__main__":
    main()
