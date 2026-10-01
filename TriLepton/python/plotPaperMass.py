#!/usr/bin/env python
import argparse
import json
import os
import sys
from pathlib import Path


TRILEPTON_DIR = Path(__file__).resolve().parents[1]
WORKDIR = os.environ.get("WORKDIR")
if not WORKDIR:
    raise RuntimeError("WORKDIR is not set")

sys.path.insert(0, str(TRILEPTON_DIR / "python"))
sys.path.insert(0, str(Path(WORKDIR) / "Common" / "Tools"))

import ROOT

from paper_plotting import (SIGNAL_COLORS, PaperPlotOptions,
                            build_legend_output_path, build_output_path,
                            render_paper_legend, render_paper_plot)


ROOT.gROOT.SetBatch(True)

PAPER_PLOTS = {
    "sr1-pair": ("SR1E2Mu", "pair/mass"),
    "sr3-low": ("SR3Mu", "pair_lowM/mass"),
    "sr3-high": ("SR3Mu", "pair_highM/mass"),
    "zfake1-zcand": ("ZFake1E2Mu", "ZCand/mass"),
    "zfake3-zcand": ("ZFake3Mu", "ZCand/mass"),
    "zg1-zcand": ("ZG1E2Mu", "ZCand/mass"),
    "zg3-zcand": ("ZG3Mu", "ZCand/mass"),
    "wz1-zcand": ("WZ1E2Mu", "ZCand/mass"),
    "wz3-zcand": ("WZ3Mu", "ZCand/mass"),
    "ttz-zcand": ("TTZ2E1Mu", "ZCand/mass"),
}

# Thesis (private-work) figures: the signal-region masses only, one per run.
PRIVATE_WORK_PLOTS = ["sr1-pair", "sr3-low", "sr3-high"]
PRIVATE_WORK_ERAS = ["Run2", "Run3"]

DEFAULT_SIGNALS = ["MHc70_MA15", "MHc100_MA60", "MHc130_MA90", "MHc160_MA155"]

# Each plot carries its own legend, and the same legend is published separately
# as its own panel for layouts that drop the in-plot ones (slides, --no-legends).
LEGEND_KEY = "legend"


def selected_plots(selection, private_work=False):
    if selection == "all":
        keys = PRIVATE_WORK_PLOTS if private_work else PAPER_PLOTS
        return [(key, PAPER_PLOTS[key]) for key in keys]
    if selection == LEGEND_KEY:
        return []
    return [(selection, PAPER_PLOTS[selection])]


def main():
    os.chdir(TRILEPTON_DIR)

    parser = argparse.ArgumentParser(
        description="Produce isolated Run 2+3 paper PDF plots for selected mass distributions."
    )
    parser.add_argument("--plot", choices=["all", LEGEND_KEY, *PAPER_PLOTS.keys()], default="all",
                        help="which paper plot to produce ('legend' = shared legend panel only)")
    parser.add_argument("--output-root", default=None,
                        help="base output directory (default: $WORKDIR/TriLepton/plots/Paper)")
    parser.add_argument("--signals", default=DEFAULT_SIGNALS, nargs="+",
                        help="signal mass points to overlay")
    parser.add_argument("--signal-scale", default=2.0, type=float,
                        help="scale factor for signal histograms")
    parser.add_argument("--blind", action="store_true", help="blind data")
    parser.add_argument("--no-legends", dest="legends", action="store_false",
                        help="drop the in-plot legends and publish them as a standalone panel instead")
    parser.add_argument("--legend-text-size", default=0.040, type=float,
                        help="text size of the standalone legend panel")
    parser.add_argument("--y-headroom", default=None, type=float,
                        help="linear-scale y-axis multiplier above the tallest bin "
                             "(default: 1.35, or 1.6 with --private-work)")
    parser.add_argument("--adaptive-binning", action="store_true",
                        help="merge 2 GeV base bins using expected background only")
    parser.add_argument("--adaptive-min-bkg", default=10.0, type=float,
                        help="minimum expected background target per adaptive bin")
    parser.add_argument("--adaptive-max-width", default=10.0, type=float,
                        help="maximum adaptive bin width in GeV")
    parser.add_argument("--adaptive-base-width", default=2.0, type=float,
                        help="starting bin width in GeV for adaptive binning")
    parser.add_argument("--private-work", dest="private_work", action="store_true",
                        help="thesis version: 'Private work (CMS data)' label, SR masses split "
                             "into Run2 and Run3, written to plots/PW/{Run}/{channel}/")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="print selected plots and output paths without rendering")
    args = parser.parse_args()
    if args.private_work and args.plot not in ("all", *PRIVATE_WORK_PLOTS):
        parser.error(f"--private-work only makes {', '.join(PRIVATE_WORK_PLOTS)}")

    default_root = "PW" if args.private_work else "Paper"
    if args.y_headroom is None:
        # The private-work label and the caption under it reach further down
        # the frame than the CMS mark, so the thesis plots get more headroom.
        args.y_headroom = 1.6 if args.private_work else 1.35
    output_root = Path(args.output_root or Path(WORKDIR) / "TriLepton" / "plots" / default_root)
    with open(TRILEPTON_DIR / "configs" / "histkeys.json") as f:
        hist_configs = json.load(f)

    options = PaperPlotOptions(
        workdir=WORKDIR,
        output_root=output_root,
        hist_configs=hist_configs,
        signals=args.signals,
        signal_scale=args.signal_scale,
        blind=args.blind,
        debug=args.debug,
        adaptive_binning=args.adaptive_binning,
        adaptive_min_bkg=args.adaptive_min_bkg,
        adaptive_max_width=args.adaptive_max_width,
        adaptive_base_width=args.adaptive_base_width,
        signal_colors=SIGNAL_COLORS,
        draw_legends=args.legends,
        legend_panel_text_size=args.legend_text_size,
        y_headroom=args.y_headroom,
    )

    if args.private_work:
        for era in PRIVATE_WORK_ERAS:
            options.era = era
            options.private_work = True
            render_selected(args, options)
        return

    if args.plot in ("all", LEGEND_KEY):
        # Two variants: signal regions overlay signals, control regions do not.
        for with_signals in (True, False):
            if args.dry_run:
                print(f"{LEGEND_KEY} (signals={with_signals}) -> "
                      f"{build_legend_output_path(options, with_signals)}")
            else:
                print(f"Wrote {render_paper_legend(options, with_signals)}")

    render_selected(args, options)


def render_selected(args, options):
    for key, (channel, histkey) in selected_plots(args.plot, options.private_work):
        output_path = build_output_path(channel, histkey, options)
        if args.dry_run:
            print(f"{key}: {channel} {histkey} -> {output_path}")
            continue

        rendered_path, adaptive_edges = render_paper_plot(channel, histkey, options)
        print(f"Wrote {rendered_path}")
        if adaptive_edges:
            print(f"Adaptive edges ({len(adaptive_edges) - 1} bins): {adaptive_edges}")


if __name__ == "__main__":
    main()
