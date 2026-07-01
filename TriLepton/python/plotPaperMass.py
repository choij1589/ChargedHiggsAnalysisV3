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

from paper_plotting import PaperPlotOptions, render_paper_plot


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

DEFAULT_SIGNALS = ["MHc70_MA15", "MHc100_MA60", "MHc130_MA90", "MHc160_MA155"]
PAPER_SIGNAL_COLORS = ["#5790fc", "#f89c20", "#964a8b", "#e42536"]


def selected_plots(selection):
    if selection == "all":
        return PAPER_PLOTS.items()
    return [(selection, PAPER_PLOTS[selection])]


def main():
    os.chdir(TRILEPTON_DIR)

    parser = argparse.ArgumentParser(
        description="Produce isolated Run 2+3 paper PDF plots for selected mass distributions."
    )
    parser.add_argument("--plot", choices=["all", *PAPER_PLOTS.keys()], default="all",
                        help="which paper mass plot to produce")
    parser.add_argument("--output-root", default=None,
                        help="base output directory (default: $WORKDIR/TriLepton/plots/Paper)")
    parser.add_argument("--signals", default=DEFAULT_SIGNALS, nargs="+",
                        help="signal mass points to overlay")
    parser.add_argument("--signal-scale", default=2.0, type=float,
                        help="scale factor for signal histograms")
    parser.add_argument("--blind", action="store_true", help="blind data")
    parser.add_argument("--adaptive-binning", action="store_true",
                        help="merge 2 GeV base bins using expected background only")
    parser.add_argument("--adaptive-min-bkg", default=10.0, type=float,
                        help="minimum expected background target per adaptive bin")
    parser.add_argument("--adaptive-max-width", default=10.0, type=float,
                        help="maximum adaptive bin width in GeV")
    parser.add_argument("--adaptive-base-width", default=2.0, type=float,
                        help="starting bin width in GeV for adaptive binning")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="print selected plots and output paths without rendering")
    args = parser.parse_args()

    output_root = Path(args.output_root or Path(WORKDIR) / "TriLepton" / "plots" / "Paper")
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
        signal_colors=PAPER_SIGNAL_COLORS,
    )

    for key, (channel, histkey) in selected_plots(args.plot):
        output_path = output_root / "All" / channel / "Central" / f"{histkey.replace('/', '_')}.pdf"
        if args.dry_run:
            print(f"{key}: {channel} {histkey} -> {output_path}")
            continue

        rendered_path, adaptive_edges = render_paper_plot(channel, histkey, options)
        print(f"Wrote {rendered_path}")
        if adaptive_edges:
            print(f"Adaptive edges ({len(adaptive_edges) - 1} bins): {adaptive_edges}")


if __name__ == "__main__":
    main()
