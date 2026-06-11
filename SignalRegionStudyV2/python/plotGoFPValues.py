#!/usr/bin/env python3
"""Plot background-only GoF p-values vs mA for each mHc."""

import argparse
import json
import os
import re
import sys
from array import array
from pathlib import Path

import ROOT
import cmsstyle as CMS

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_TOOLS = REPO_ROOT.parent / "Common" / "Tools"
if str(COMMON_TOOLS) not in sys.path:
    sys.path.insert(0, str(COMMON_TOOLS))

from plotter import LumiInfo, PALETTE  # noqa: E402

ROOT.gROOT.SetBatch(ROOT.kTRUE)

VALID_ERAS = ("Run2", "Run3", "All")
VALID_METHODS = ("Baseline", "ParticleNet")
VALID_CHANNELS = ("Combined", "SR1E2Mu", "SR3Mu")
MASSPOINT_RE = re.compile(r"^MHc(?P<mhc>\d+)_MA(?P<ma>\d+)$")

ERA_COLORS = {
    "Run2": PALETTE[0],
    "Run3": PALETTE[1],
    "All": ROOT.kBlack,
}

PARTICLENET_COLORS = {
    "Run2": PALETTE[3],
    "Run3": PALETTE[4],
    "All": PALETTE[5],
}

ERA_LABELS = {
    "Run2": "Run2",
    "Run3": "Run3",
    "All": "Run2+Run3",
}

METHOD_STYLES = {
    "Baseline": {
        "marker": ROOT.kFullCircle,
        "label": "Baseline",
    },
    "ParticleNet": {
        "marker": ROOT.kFullSquare,
        "label": "ParticleNet",
    },
}

CHANNEL_STYLES = {
    "Combined": {
        "color": ROOT.kBlack,
        "marker": ROOT.kFullCircle,
        "label": "e#mu#mu+#mu#mu#mu",
    },
    "SR1E2Mu": {
        "color": PALETTE[0],
        "marker": ROOT.kFullSquare,
        "label": "e#mu#mu",
    },
    "SR3Mu": {
        "color": PALETTE[1],
        "marker": ROOT.kFullTriangleUp,
        "label": "#mu#mu#mu",
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot b-only saturated GoF p-values vs mA for fixed mHc values."
    )
    parser.add_argument("--mhc", nargs="+", type=int, default=[70, 160],
                        help="mHc values to plot (default: 70 160)")
    parser.add_argument("--eras", nargs="+", choices=VALID_ERAS, default=list(VALID_ERAS),
                        help="Era combinations to overlay")
    parser.add_argument("--methods", nargs="+", choices=VALID_METHODS, default=list(VALID_METHODS),
                        help="Methods to overlay")
    parser.add_argument("--channel", choices=VALID_CHANNELS, default="Combined",
                        help="Single channel directory to read")
    parser.add_argument("--channels", nargs="+", choices=VALID_CHANNELS,
                        help="Channels to overlay; overrides --channel when set")
    parser.add_argument("--fit-name", default="120.0",
                        help="Top-level GoF JSON key to read (default: 120.0)")
    parser.add_argument("--template-root", default="templates",
                        help="Template output root directory")
    parser.add_argument("--suffix", default="extended_unblind",
                        help="Template suffix containing GoF outputs")
    parser.add_argument("--output-dir", default="results/plots/gof_pvalues",
                        help="Directory for output plots")
    parser.add_argument("--y-min", type=float, default=1e-3,
                        help="Minimum y-axis value")
    parser.add_argument("--y-max", type=float, default=100.0,
                        help="Maximum y-axis value")
    return parser.parse_args()


def get_lumi_label():
    run2_lumi = LumiInfo["Run2"]
    run3_lumi = LumiInfo["Run3"]
    return f"Run 2+3, {run2_lumi}+{run3_lumi} fb^{{#minus1}}"


def read_gof(path, fit_name):
    with open(path, "r") as handle:
        payload = json.load(handle)

    if fit_name in payload:
        result = payload[fit_name]
    elif len(payload) == 1:
        result = next(iter(payload.values()))
    else:
        raise KeyError(f"{path}: no '{fit_name}' entry in GoF JSON")

    p_value = float(result["p"])
    toy_values = result.get("toy", [])
    n_toys = len(toy_values)

    plot_value = p_value
    if plot_value <= 0.0:
        plot_value = 0.5 / n_toys if n_toys > 0 else 1e-6

    return p_value, plot_value, n_toys


def selected_channels(args):
    return args.channels if args.channels else [args.channel]


def collect_points(args, mhc, era, channel, method):
    channel_dir = Path(args.template_root) / era / channel
    points = []

    for masspoint_dir in sorted(channel_dir.glob(f"MHc{mhc}_MA*")):
        match = MASSPOINT_RE.match(masspoint_dir.name)
        if not match:
            continue

        ma = int(match.group("ma"))
        gof_path = (
            masspoint_dir
            / method
            / args.suffix
            / "combine_output"
            / "gof"
            / "gof.json"
        )
        if not gof_path.exists():
            continue

        try:
            p_value, plot_value, n_toys = read_gof(gof_path, args.fit_name)
        except (KeyError, ValueError, json.JSONDecodeError) as exc:
            print(f"Warning: skipped {gof_path}: {exc}")
            continue

        points.append({
            "ma": ma,
            "p": p_value,
            "plot_p": plot_value,
            "n_toys": n_toys,
            "path": gof_path,
        })

    return sorted(points, key=lambda point: point["ma"])


def make_graph(points, era, channel, method, overlay_mode):
    graph = ROOT.TGraph(
        len(points),
        array("d", [point["ma"] for point in points]),
        array("d", [point["plot_p"] for point in points]),
    )
    if overlay_mode == "channels":
        color = CHANNEL_STYLES[channel]["color"]
        marker = CHANNEL_STYLES[channel]["marker"]
        line_style = ROOT.kSolid
    else:
        color = PARTICLENET_COLORS[era] if method == "ParticleNet" else ERA_COLORS[era]
        marker = METHOD_STYLES[method]["marker"]
        line_style = ROOT.kSolid if method == "ParticleNet" or era == "All" else ROOT.kDashed

    graph.SetLineColor(color)
    graph.SetMarkerColor(color)
    graph.SetLineWidth(2)
    graph.SetLineStyle(line_style)
    graph.SetMarkerStyle(marker)
    graph.SetMarkerSize(1.0)
    return graph


def series_label(item, overlay_mode):
    method_label = METHOD_STYLES[item["method"]]["label"]
    channel_label = CHANNEL_STYLES[item["channel"]]["label"]
    era_label = ERA_LABELS[item["era"]]
    if overlay_mode == "channels":
        return channel_label if len(item["all_methods"]) == 1 else f"{channel_label} {method_label}"
    return (
        f"{era_label} {method_label}"
        if len(item["all_channels"]) == 1
        else f"{era_label} {channel_label} {method_label}"
    )


def draw_plot(args, mhc, series):
    all_ma = [point["ma"] for item in series for point in item["points"]]
    if not all_ma:
        print(f"Warning: no GoF points found for mHc={mhc}")
        return []

    xmin = max(0.0, min(all_ma) - 5.0)
    xmax = max(all_ma) + 5.0

    CMS.SetExtraText("Preliminary")
    CMS.ResetAdditionalInfo()
    CMS.SetLumi(None, run=get_lumi_label())
    CMS.SetEnergy(0, unit="13/13.6 TeV")

    canvas_name = f"gof_pvalues_mHc{mhc}"
    canvas = CMS.cmsCanvas(
        canvas_name,
        xmin,
        xmax,
        args.y_min,
        args.y_max,
        "m_{A} [GeV]",
        "B-only GoF p-value",
        square=True,
        iPos=11,
        extraSpace=0.01,
    )
    canvas.SetLogy(1)
    canvas.cd()

    line_005 = ROOT.TLine(xmin, 0.05, xmax, 0.05)
    line_005.SetLineColor(ROOT.kGray + 2)
    line_005.SetLineStyle(ROOT.kDotted)
    line_005.SetLineWidth(2)
    line_005.Draw("same")

    graphs = []
    overlay_mode = "channels" if args.channels else "eras"

    for item in series:
        graph = make_graph(
            item["points"],
            item["era"],
            item["channel"],
            item["method"],
            overlay_mode,
        )
        graphs.append((graph, item))
        CMS.cmsObjectDraw(graph, "LP same")

    text = ROOT.TLatex()
    text.SetNDC()
    text.SetTextFont(42)
    text.SetTextSize(0.046)
    text.DrawLatex(0.19, 0.76, f"m_{{H^{{+}}}} = {mhc} GeV")
    if overlay_mode != "channels":
        text.SetTextSize(0.036)
        text.DrawLatex(0.19, 0.71, args.channel)

    n_entries = len(graphs) + 1
    leg_y1 = max(0.56, 0.90 - 0.045 * n_entries)
    legend = CMS.cmsLeg(0.57, leg_y1, 0.95, 0.90, textSize=0.032)
    for graph, item in graphs:
        legend.AddEntry(graph, series_label(item, overlay_mode), "lp")
    legend.AddEntry(line_005, "p = 0.05", "l")

    canvas.RedrawAxis()

    os.makedirs(args.output_dir, exist_ok=True)
    channels = selected_channels(args)
    channels_tag = "".join(channels)
    methods_tag = "".join(args.methods)
    eras_tag = "".join(args.eras)
    output_base = (
        Path(args.output_dir)
        / f"pvalue.mHc{mhc}.{channels_tag}.{eras_tag}.{methods_tag}.unblind"
    )

    saved = []
    for ext in ("png", "pdf"):
        output_path = f"{output_base}.{ext}"
        canvas.SaveAs(output_path)
        saved.append(output_path)

    return saved


def main():
    args = parse_args()
    saved_paths = []
    channels = selected_channels(args)
    if args.channels and len(args.eras) != 1:
        raise ValueError("--channels overlay expects exactly one era; use e.g. --eras All")

    for mhc in args.mhc:
        series = []
        for era in args.eras:
            for channel in channels:
                for method in args.methods:
                    points = collect_points(args, mhc, era, channel, method)
                    if not points:
                        print(
                            f"Warning: no points for mHc={mhc}, era={era}, "
                            f"channel={channel}, method={method}"
                        )
                        continue
                    series.append({
                        "era": era,
                        "channel": channel,
                        "method": method,
                        "all_channels": channels,
                        "all_methods": args.methods,
                        "points": points,
                    })

        saved_paths.extend(draw_plot(args, mhc, series))

    for output_path in saved_paths:
        print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
