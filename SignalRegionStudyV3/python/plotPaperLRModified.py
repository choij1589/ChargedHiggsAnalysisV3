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
from plotter import (ComparisonCanvas, EnergyInfo,  # noqa: E402
                     LumiInfoExact, PALETTE_LONG)
import cmsstyle as CMS  # noqa: E402


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
SIGNAL_FILL_ALPHA = 0.18
SIGNAL_LABEL = "Signal"

# The legend is identical in every panel, so it is dropped from the plots and
# published once as its own panel, letting the stack use the freed height.
Y_HEADROOM = 1.7
# Mass point moves to the top right, where the in-plot legend used to sit.
MASS_LABEL_POS = (0.90, 0.80)
MASS_LABEL_SIZE = 0.052
# Nudge in PDF points, converted against the panel below so the shift stays the
# same physical distance whatever the pad geometry.
MASS_LABEL_OFFSET_PT = (-20.0, -6.0)
# CropBox ROOT writes for this canvas; used only to size the nudge above.
PANEL_SIZE_PT = (526.0, 567.0)

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
        "--keep-legends",
        action="store_true",
        dest="keep_legends",
        help="draw the legend inside each plot instead of only in the legend panel",
    )
    parser.add_argument(
        "--output-root",
        default="results/paper",
        help="base output directory, relative to SignalRegionStudyV3 unless absolute",
    )
    parser.add_argument(
        "--base-width",
        type=float,
        default=BASE_WIDTH,
        help=("uniform bin width of the starting grid, before adaptive merging "
              "(default: %(default)s). A non-default value writes into a "
              "bin<width> subdirectory, e.g. --base-width 0.02 -> SR/bin0p02/"),
    )
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    return parser.parse_args()


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


def output_path(output_root, region, masspoint, subdir=""):
    directory = output_root / region
    if subdir:
        directory = directory / subdir
    return directory / f"{SCORE_KEY}_{masspoint}.pdf"


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


def build_config(region, edges, draw_legend=False, y_range=None):
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
        # "138 fb^-1 (13 TeV) + 62.4 fb^-1 (13.6 TeV)". cmsstyle appends the
        # CoM in parentheses, so the Run3 energy is carried by "CoM" and the
        # Run2 term is baked into "run_label".
        "CoM": f"{EnergyInfo['Run3']:g} TeV",
        "run_label": (f"{LumiInfoExact['Run2']:g} fb^{{#minus1}} ({EnergyInfo['Run2']:g} TeV) + "
                      f"{LumiInfoExact['Run3']:g} fb^{{#minus1}}"),
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
        "legend": (0.72, 0.55, 0.99, 0.89),
        "legendTextSize": 0.038,
        "legendColumns": 1,
        # The legend lives in the shared panel unless the caller asks for it.
        "drawLegend": draw_legend,
        "colors": [BKG_COLORS[name] for name in BKG_ORDER],
        "signalLineWidth": 3,
        "signalFill": True,
        "signalFillAlpha": 0.18,
        "signalColors": [SIGNAL_COLOR],
        "channel": channel_label,
        "region": region_label,
        # iPos=11 puts "CMS"/"Preliminary" inside the frame, so the channel
        # block starts lower to clear them.
        "channelPosY": 0.72,
        "channelPosX": 0.22,
        "chi2_test": False,
        "normalize_chi2": False,
    }


def draw_integrated_signal(plotter, signals, draw_legend=False):
    plotter.update_y_scale(signals)
    plotter.canv.cd(1)
    plotter.signals = {}

    for name, hist in signals.items():
        signal_hist = hist.Clone(f"signal_{name}")
        signal_hist.SetDirectory(0)
        signal_hist.SetStats(0)
        signal_hist.SetLineColor(SIGNAL_COLOR)
        signal_hist.SetLineWidth(SIGNAL_LINE_WIDTH)
        signal_hist.SetFillColorAlpha(SIGNAL_COLOR, SIGNAL_FILL_ALPHA)
        signal_hist.SetFillStyle(1001)
        signal_hist.Draw("HIST SAME")
        plotter.signals[name] = signal_hist
        plotter.leg.AddEntry(signal_hist, SIGNAL_LABEL, "F")

    if draw_legend:
        plotter.leg.Draw()
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
    entries = [(data, DATA_LABEL, "PE")]

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
        signal.SetFillColorAlpha(SIGNAL_COLOR, SIGNAL_FILL_ALPHA)
        signal.SetFillStyle(1001)
        signal.SetMarkerSize(0)
        entries.append((signal, SIGNAL_LABEL, "F"))

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


def draw_masspoint(region, masspoint, output_root, base_width=BASE_WIDTH, draw_legend=False):
    data, bkgs, signals, edges = build_plot_objects(region, masspoint, base_width)
    config = build_config(region, edges, draw_legend=draw_legend,
                          y_range=build_y_range(data, bkgs, signals))

    out_path = output_path(output_root, region, masspoint, base_width_tag(base_width))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plotter = ComparisonCanvas(data, bkgs, config)
    plotter.drawPadUp()
    if signals:
        draw_integrated_signal(plotter, signals, draw_legend=draw_legend)
    pad = plotter.canv.cd(1)
    # Right-aligned in the top corner the legend used to occupy.
    plotter.mass_label = ROOT.TLatex()
    plotter.mass_label.SetNDC(True)
    plotter.mass_label.SetTextFont(42)
    plotter.mass_label.SetTextSize(MASS_LABEL_SIZE)
    plotter.mass_label.SetTextAlign(31)
    label_x, label_y = offset_ndc_by_points(pad, *MASS_LABEL_POS, *MASS_LABEL_OFFSET_PT)
    plotter.mass_label.DrawLatex(label_x, label_y, format_signal_label(masspoint))
    plotter.drawPadDown()
    plotter.canv.SaveAs(str(out_path))
    return out_path, edges


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format="%(levelname)s - %(message)s")

    selected = DEFAULT_MASSPOINTS if args.masspoint == "all" else (args.masspoint,)
    if args.region == "all":
        selected_regions = REGIONS
    elif args.region == LEGEND_KEY:
        selected_regions = ()
    else:
        selected_regions = (args.region,)
    output_root = resolve_output_root(args.output_root)

    # Published once each: SR panels overlay a signal, the TTZ CR panels do not.
    if args.region in ("all", LEGEND_KEY) and not args.keep_legends:
        for with_signal in (True, False):
            logging.info("Wrote %s", render_paper_legend(output_root, with_signal))

    for region in selected_regions:
        for masspoint in selected:
            out_path, edges = draw_masspoint(region, masspoint, output_root,
                                             args.base_width, draw_legend=args.keep_legends)
            logging.info("Wrote %s", out_path)
            logging.info(
                "Adaptive edges for %s/%s (%d bins): %s",
                region, masspoint, len(edges) - 1, edges
            )


if __name__ == "__main__":
    main()
