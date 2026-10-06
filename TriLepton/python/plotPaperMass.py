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
    "ttz-zcand": ("TTZ2E1Mu", "ZCand/mass"),
}
# The Z+gamma and WZ control-region masses are thesis-only: the paper and PAS do
# not show them.
PRIVATE_WORK_MASS_PLOTS = {
    "zg1-zcand": ("ZG1E2Mu", "ZCand/mass"),
    "zg3-zcand": ("ZG3Mu", "ZCand/mass"),
    "wz1-zcand": ("WZ1E2Mu", "ZCand/mass"),
    "wz3-zcand": ("WZ3Mu", "ZCand/mass"),
}
# Thesis-only kinematics, drawn with the same style as the masses. Leptons are
# stored per flavour, so eμμ's three leptons are e1, mu1, mu2.
COMMON_KINEMATICS = ["jets/1/pt", "jets/1/eta", "jets/2/pt", "jets/2/eta",
                     "bjets/1/pt", "bjets/1/eta", "jets/size", "bjets/size", "METv/pt"]
PRIVATE_WORK_KINEMATICS = {
    "sr1": ("SR1E2Mu", ["electrons/1/pt", "electrons/1/eta", "muons/1/pt", "muons/1/eta",
                        "muons/2/pt", "muons/2/eta", *COMMON_KINEMATICS]),
    "sr3": ("SR3Mu", ["muons/1/pt", "muons/1/eta", "muons/2/pt", "muons/2/eta",
                      "muons/3/pt", "muons/3/eta", *COMMON_KINEMATICS]),
}
# Control regions: the paper Z-candidate mass plus the conversion lepton (Z+gamma),
# the fake-lepton candidate (Z+nonprompt), the leading jet and the jet
# multiplicity (WZ), or the leptons, the leading jet, N_j, N_b and pTmiss (ttZ).
PRIVATE_WORK_CR_KINEMATICS = {
    "ttz": ("TTZ2E1Mu", ["muons/1/pt", "electrons/1/pt", "jets/1/pt", "jets/1/eta",
                         "jets/1/phi", "jets/size", "bjets/size", "METv/pt"]),
    "zfake1": ("ZFake1E2Mu", ["nonprompt/pt", "nonprompt/eta"]),
    "zfake3": ("ZFake3Mu", ["nonprompt/pt", "nonprompt/eta"]),
    "zg1": ("ZG1E2Mu", ["convLep/pt", "convLep/eta"]),
    "zg3": ("ZG3Mu", ["convLep/pt", "convLep/eta"]),
    "wz1": ("WZ1E2Mu", ["jets/1/pt", "jets/size"]),
    "wz3": ("WZ3Mu", ["jets/1/pt", "jets/size"]),
}
# Keys like "sr1-jets-1-pt", "zg1-convLep-pt".
PRIVATE_WORK_EXTRA_PLOTS = {
    f"{prefix}-{histkey.replace('/', '-')}": (channel, histkey)
    for kinematics in (PRIVATE_WORK_KINEMATICS, PRIVATE_WORK_CR_KINEMATICS)
    for prefix, (channel, histkeys) in kinematics.items()
    for histkey in histkeys
}
# Every plot that only the thesis makes.
PRIVATE_WORK_ONLY_PLOTS = {**PRIVATE_WORK_MASS_PLOTS, **PRIVATE_WORK_EXTRA_PLOTS}

# Thesis (private-work) figures, per run and combined: the signal-region masses
# plus kinematics, and the Z+nonprompt, Z+gamma and WZ control regions.
PRIVATE_WORK_PLOTS = ["sr1-pair", "sr3-low", "sr3-high", "zfake1-zcand", "zfake3-zcand",
                      "zg1-zcand", "zg3-zcand", "wz1-zcand", "wz3-zcand", "ttz-zcand",
                      *PRIVATE_WORK_EXTRA_PLOTS]
# Channels drawn both after (default) and before the ConvSF normalization, or
# the WZNjetsSF reweighting. WZNjetsSF is Run3 only, so the Run2 "before" plot
# would be the default one again and is not drawn.
CONV_SF_CHANNELS = ("ZG1E2Mu", "ZG3Mu")
WZ_SF_CHANNELS = ("WZ1E2Mu", "WZ3Mu")
WZ_SF_ERAS = ("Run3", "All")
PRIVATE_WORK_ERAS = ["Run2", "Run3", "All"]
# Preliminary (CMS Preliminary) versions of the paper masses, for the PAS.
PRELIMINARY_PLOTS = list(PAPER_PLOTS)

DEFAULT_SIGNALS = ["MHc70_MA15", "MHc100_MA60", "MHc130_MA90", "MHc160_MA155"]

# Each plot carries its own legend, and the same legend is published separately
# as its own panel for layouts that drop the in-plot ones (slides, --no-legends).
LEGEND_KEY = "legend"


def selected_plots(selection, private_work=False, preliminary=False):
    plots = {**PAPER_PLOTS, **PRIVATE_WORK_ONLY_PLOTS}
    if selection == "all":
        if private_work:
            keys = PRIVATE_WORK_PLOTS
        elif preliminary:
            keys = PRELIMINARY_PLOTS
        else:
            keys = PAPER_PLOTS
        return [(key, plots[key]) for key in keys]
    if selection == LEGEND_KEY:
        return []
    return [(selection, plots[selection])]


def main():
    os.chdir(TRILEPTON_DIR)

    parser = argparse.ArgumentParser(
        description="Produce isolated Run 2+3 paper PDF plots for selected mass distributions."
    )
    parser.add_argument("--plot", choices=["all", LEGEND_KEY, *PAPER_PLOTS.keys(),
                                                *PRIVATE_WORK_ONLY_PLOTS.keys()], default="all",
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
    label_group = parser.add_mutually_exclusive_group()
    label_group.add_argument("--private-work", dest="private_work", action="store_true",
                        help="thesis version: 'Private work (CMS data)' label, SR masses split "
                             "+ kinematics, the Z+nonprompt CRs (mass, l_fake pT/eta), the Z+gamma CRs (mass, l_conv pT/eta, "
                             "after and before ConvSF -> NoConvSF/), the WZ CRs (mass, "
                             "leading-jet pT, N_j; before WZNjetsSF -> NoWZSF/ for Run3 and All) "
                             "the ttZ CR (mass, mu/e1/j1 pT, j1 eta/phi, N_j, N_b, pTmiss) "
                             "for Run2, Run3 and "
                             "combined (All), written to plots/PW/{Run2,Run3,All}/{channel}/")
    label_group.add_argument("--preliminary", action="store_true",
                             help="paper layout with 'CMS Preliminary', signal- and control-region masses, "
                                  "written next to the paper PDF as <histkey>_preliminary.pdf")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="print selected plots and output paths without rendering")
    args = parser.parse_args()
    if args.private_work and args.plot not in ("all", *PRIVATE_WORK_PLOTS):
        parser.error(f"--private-work only makes {', '.join(PRIVATE_WORK_PLOTS)}")
    if args.preliminary and args.plot not in ("all", *PRELIMINARY_PLOTS):
        parser.error(f"--preliminary only makes {', '.join(PRELIMINARY_PLOTS)}")
    if args.preliminary and not args.legends:
        parser.error("--preliminary draws its legends in the plots; drop --no-legends")
    if not args.private_work and args.plot in PRIVATE_WORK_ONLY_PLOTS:
        parser.error(f"--plot {args.plot} is a thesis plot; add --private-work")

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
        preliminary=args.preliminary,
    )

    if args.private_work:
        for era in PRIVATE_WORK_ERAS:
            options.era = era
            options.private_work = True
            render_selected(args, options)
        return

    if args.plot in ("all", LEGEND_KEY) and not args.preliminary:
        # Two variants: signal regions overlay signals, control regions do not.
        for with_signals in (True, False):
            if args.dry_run:
                print(f"{LEGEND_KEY} (signals={with_signals}) -> "
                      f"{build_legend_output_path(options, with_signals)}")
            else:
                print(f"Wrote {render_paper_legend(options, with_signals)}")

    render_selected(args, options)


def render_selected(args, options):
    for key, (channel, histkey) in selected_plots(args.plot, options.private_work,
                                                  options.preliminary):
        # (conv_sf, wz_sf): the default, then the one reweighting switched off.
        variants = [(True, True)]
        if options.private_work and channel in CONV_SF_CHANNELS:
            variants.append((False, True))
        if options.private_work and channel in WZ_SF_CHANNELS and options.era in WZ_SF_ERAS:
            variants.append((True, False))
        for conv_sf, wz_sf in variants:
            options.conv_sf, options.wz_sf = conv_sf, wz_sf
            output_path = build_output_path(channel, histkey, options)
            if args.dry_run:
                print(f"{key}: {channel} {histkey} -> {output_path}")
                continue

            rendered_path, adaptive_edges = render_paper_plot(channel, histkey, options)
            print(f"Wrote {rendered_path}")
            if adaptive_edges:
                print(f"Adaptive edges ({len(adaptive_edges) - 1} bins): {adaptive_edges}")
        options.conv_sf = options.wz_sf = True


if __name__ == "__main__":
    main()
