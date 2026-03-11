#!/usr/bin/env python3
"""
Plot post-fit distributions from FitDiagnostics output.

Reads fitDiagnostics.root and creates:
  - postfit_b_{channel}.png: Data vs post-fit B-only background stack + signal overlay
  - prefit_vs_postfit_{channel}.png: Pre-fit vs post-fit total_background overlay

Usage:
    python3 plotPostfit.py --era All --channel Combined --masspoint MHc160_MA50 \
        --method Baseline --binning extended [--unblind] [--fit-type b]
"""
import os
import sys
import json
import logging
import argparse
import ROOT

# Argument parsing
parser = argparse.ArgumentParser(description="Plot post-fit distributions from FitDiagnostics")
parser.add_argument("--era", required=True, type=str, help="Data-taking period")
parser.add_argument("--channel", required=True, type=str, help="Analysis channel (SR1E2Mu, SR3Mu, Combined)")
parser.add_argument("--masspoint", required=True, type=str, help="Signal mass point (e.g., MHc130_MA90)")
parser.add_argument("--method", required=True, type=str, help="Template method (Baseline, ParticleNet)")
parser.add_argument("--binning", default="extended", choices=["uniform", "extended"],
                    help="Binning method: 'extended' (19 bins, default) or 'uniform' (15 bins)")
parser.add_argument("--unblind", action="store_true",
                    help="Use unblind templates")
parser.add_argument("--partial-unblind", action="store_true", dest="partial_unblind",
                    help="Use partial-unblind templates")
parser.add_argument("--fit-type", default="b", choices=["b", "s"],
                    help="Fit type: 'b' for B-only (default) or 's' for S+B")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args = parser.parse_args()

# Validate unblind options
if args.unblind and args.partial_unblind:
    raise ValueError("--unblind and --partial-unblind are mutually exclusive")

# Setup logging
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                    format='%(levelname)s - %(message)s')

# Path setup
WORKDIR = os.getenv("WORKDIR")
if not WORKDIR:
    raise EnvironmentError("WORKDIR environment variable not set. Please run 'source setup.sh'")

sys.path.insert(0, f"{WORKDIR}/Common/Tools")
from plotter import KinematicCanvas, ComparisonCanvas, get_CoM_energy
from plotter import PALETTE_LONG as PALETTE
import cmsstyle as CMS

# Fixed color mapping for backgrounds (consistent with checkTemplates.py)
BKG_COLORS = {
    "nonprompt": PALETTE[0],
    "WZ": PALETTE[1],
    "ZZ": PALETTE[2],
    "ttW": PALETTE[3],
    "ttZ": PALETTE[4],
    "ttH": PALETTE[5],
    "tZq": PALETTE[6],
    "others": PALETTE[7],
    "conversion": PALETTE[8]
}

# Preferred background order for stack plots (bottom to top)
BKG_ORDER = ["others", "conversion", "WZ", "ZZ", "ttW", "ttH", "tZq", "ttZ", "nonprompt"]

# Setup ROOT for batch mode
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

# Determine binning suffix and paths
binning_suffix = args.binning
if args.unblind:
    binning_suffix = f"{args.binning}_unblind"
elif args.partial_unblind:
    binning_suffix = f"{args.binning}_partial_unblind"

TEMPLATE_DIR = f"{WORKDIR}/SignalRegionStudyV2/templates/{args.era}/{args.channel}/{args.masspoint}/{args.method}/{binning_suffix}"
FITDIAG_DIR = f"{TEMPLATE_DIR}/combine_output/fitdiag"
OUTPUT_DIR = f"{FITDIAG_DIR}/plots"

# Locate fitDiagnostics file
fitdiag_filename = f"fitDiagnostics.{args.masspoint}.{args.method}.{binning_suffix}.root"
fitdiag_path = f"{FITDIAG_DIR}/{fitdiag_filename}"

if not os.path.exists(fitdiag_path):
    raise FileNotFoundError(f"fitDiagnostics file not found: {fitdiag_path}")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def graph_to_hist(graph, template_hist):
    """Convert TGraphAsymmErrors (data) to TH1D using a template histogram for binning."""
    nbins = template_hist.GetNbinsX()
    hist = template_hist.Clone(f"{graph.GetName()}_hist")
    hist.SetDirectory(0)
    hist.Reset()

    for i in range(graph.GetN()):
        x = graph.GetPointX(i)
        y = graph.GetPointY(i)
        err_up = graph.GetErrorYhigh(i)
        err_down = graph.GetErrorYlow(i)
        ibin = hist.FindBin(x)
        if 1 <= ibin <= nbins:
            hist.SetBinContent(ibin, y)
            hist.SetBinError(ibin, max(err_up, err_down))

    return hist


def discover_channels(f):
    """Auto-discover channels from shapes_prefit/ in the fitDiagnostics file."""
    prefit_dir = f.Get("shapes_prefit")
    if not prefit_dir:
        raise RuntimeError("shapes_prefit not found in fitDiagnostics file")

    channels = []
    for key in prefit_dir.GetListOfKeys():
        obj = prefit_dir.Get(key.GetName())
        if obj and obj.InheritsFrom("TDirectory"):
            channels.append(key.GetName())

    if not channels:
        raise RuntimeError("No channels found in shapes_prefit/")

    logging.info(f"Discovered channels: {channels}")
    return channels


def discover_processes(fit_dir, channel):
    """Discover background processes in a fit channel directory."""
    ch_dir = fit_dir.Get(channel)
    if not ch_dir:
        return []

    skip = {"total", "total_background", "total_signal", "total_covar", "data"}
    processes = []
    for key in ch_dir.GetListOfKeys():
        name = key.GetName()
        if name in skip:
            continue
        obj = ch_dir.Get(name)
        if obj and obj.InheritsFrom("TH1"):
            processes.append(name)

    return processes


def make_postfit_plot(f, channel, fit_type):
    """Create post-fit data vs background stack plot."""
    fit_dir_name = f"shapes_fit_{fit_type}"
    fit_dir = f.Get(fit_dir_name)
    if not fit_dir:
        logging.warning(f"{fit_dir_name} not found, skipping postfit plot for {channel}")
        return

    ch_dir = fit_dir.Get(channel)
    if not ch_dir:
        logging.warning(f"{fit_dir_name}/{channel} not found, skipping")
        return

    # Get total_background as template for binning
    total_bkg = ch_dir.Get("total_background")
    if not total_bkg:
        logging.warning(f"total_background not found in {fit_dir_name}/{channel}")
        return
    total_bkg.SetDirectory(0)

    # Get data (TGraphAsymmErrors)
    data_graph = ch_dir.Get("data")
    if not data_graph:
        logging.warning(f"data not found in {fit_dir_name}/{channel}")
        return

    data_hist = graph_to_hist(data_graph, total_bkg)
    data_hist.SetTitle("data_obs")

    # Get signal histogram
    signal_hist = ch_dir.Get(args.masspoint)
    has_signal = signal_hist is not None
    if has_signal:
        signal_hist.SetDirectory(0)

    # Discover and load background processes
    processes = discover_processes(fit_dir, channel)
    # Separate signal from backgrounds
    bkg_processes = [p for p in processes if p != args.masspoint]

    # Order backgrounds
    ordered_bkgs = []
    for bkg in BKG_ORDER:
        if bkg in bkg_processes:
            ordered_bkgs.append(bkg)
    for bkg in bkg_processes:
        if bkg not in ordered_bkgs:
            ordered_bkgs.append(bkg)

    bkg_hists = {}
    for bkg in ordered_bkgs:
        hist = ch_dir.Get(bkg)
        if hist and hist.Integral() > 0:
            h = hist.Clone(f"{bkg}_postfit")
            h.SetDirectory(0)
            bkg_hists[bkg] = h

    if not bkg_hists:
        logging.warning(f"No background histograms found for {channel}")
        return

    # Build colors
    colors = []
    for bkg in bkg_hists.keys():
        colors.append(BKG_COLORS.get(bkg, ROOT.kGray))

    fit_label = "B-only" if fit_type == "b" else "S+B"

    config = {
        "era": args.era,
        "CoM": get_CoM_energy(args.era),
        "channel": channel,
        "xTitle": "M(#mu^{+}#mu^{-}) [GeV]",
        "yTitle": "Events",
        "rTitle": "Data / Pred",
        "rRange": [0, 2.5],
        "maxDigits": 3,
        "systSrc": f"Post-fit ({fit_label})",
        "colors": colors
    }

    plotter = ComparisonCanvas(data_hist, bkg_hists, config)
    plotter.drawPadUp()

    # Draw signal overlay
    if has_signal:
        plotter.canv.cd(1)
        signal_hist.SetLineColor(ROOT.kBlack)
        signal_hist.SetLineWidth(2)
        signal_hist.SetLineStyle(1)
        signal_hist.Draw("HIST SAME")

        # Add signal to legend
        current_pad = ROOT.gPad
        for obj in current_pad.GetListOfPrimitives():
            if obj.InheritsFrom("TLegend"):
                obj.AddEntry(signal_hist, f"Signal ({signal_hist.Integral():.1f})", "l")
                break

    plotter.drawPadDown()

    # Add text
    plotter.canv.cd()
    CMS.drawText(f"{args.masspoint} ({args.method})", posX=0.2, posY=0.76, font=61, align=0, size=0.03)
    CMS.drawText(f"Post-fit {fit_label}", posX=0.2, posY=0.72, font=42, align=0, size=0.025)

    output_path = f"{OUTPUT_DIR}/postfit_{fit_type}_{channel}.png"
    plotter.canv.SaveAs(output_path)
    logging.info(f"Saved: {output_path}")

    # Print yield summary
    logging.info(f"  {channel} post-fit ({fit_label}):")
    logging.info(f"    total_background: {total_bkg.Integral():.2f}")
    logging.info(f"    data: {data_hist.Integral():.0f}")
    if has_signal:
        logging.info(f"    signal: {signal_hist.Integral():.2f}")


def make_prefit_vs_postfit_plot(f, channel, fit_type):
    """Create pre-fit vs post-fit total_background overlay."""
    prefit_dir = f.Get("shapes_prefit")
    postfit_dir = f.Get(f"shapes_fit_{fit_type}")

    if not prefit_dir or not postfit_dir:
        logging.warning(f"Missing prefit or postfit directories for {channel}")
        return

    prefit_ch = prefit_dir.Get(channel)
    postfit_ch = postfit_dir.Get(channel)

    if not prefit_ch or not postfit_ch:
        logging.warning(f"Channel {channel} not found in prefit or postfit directories")
        return

    prefit_total = prefit_ch.Get("total_background")
    postfit_total = postfit_ch.Get("total_background")

    if not prefit_total or not postfit_total:
        logging.warning(f"total_background not found for {channel}")
        return

    prefit_clone = prefit_total.Clone("prefit_total")
    prefit_clone.SetDirectory(0)
    postfit_clone = postfit_total.Clone("postfit_total")
    postfit_clone.SetDirectory(0)

    prefit_int = prefit_clone.Integral()
    postfit_int = postfit_clone.Integral()

    fit_label = "B-only" if fit_type == "b" else "S+B"

    xmin = prefit_clone.GetXaxis().GetXmin()
    xmax = prefit_clone.GetXaxis().GetXmax()

    config = {
        "era": args.era,
        "CoM": get_CoM_energy(args.era),
        "channel": channel,
        "xTitle": "M(#mu^{+}#mu^{-}) [GeV]",
        "yTitle": "Events",
        "maxDigits": 3,
        "xRange": [xmin, xmax]
    }

    hists = {
        f"Pre-fit ({prefit_int:.1f})": prefit_clone,
        f"Post-fit {fit_label} ({postfit_int:.1f})": postfit_clone
    }

    plotter = KinematicCanvas(hists, config)
    plotter.drawPad()

    plotter.canv.cd()
    CMS.drawText(f"{args.masspoint} ({args.method})", posX=0.2, posY=0.65, font=61, align=0, size=0.04)
    ratio = postfit_int / prefit_int if prefit_int > 0 else 0
    CMS.drawText(f"Post/Pre = {ratio:.3f}", posX=0.2, posY=0.60, font=42, align=0, size=0.03)

    output_path = f"{OUTPUT_DIR}/prefit_vs_postfit_{channel}.png"
    plotter.canv.SaveAs(output_path)
    logging.info(f"Saved: {output_path}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    logging.info(f"Post-fit plotting")
    logging.info(f"  Era: {args.era}")
    logging.info(f"  Channel: {args.channel}")
    logging.info(f"  Masspoint: {args.masspoint}")
    logging.info(f"  Method: {args.method}")
    logging.info(f"  Fit type: {args.fit_type}")
    logging.info(f"  Input: {fitdiag_path}")

    f = ROOT.TFile.Open(fitdiag_path, "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Failed to open: {fitdiag_path}")

    # Discover channels
    channels = discover_channels(f)

    for channel in channels:
        logging.info(f"Processing channel: {channel}")
        make_postfit_plot(f, channel, args.fit_type)
        make_prefit_vs_postfit_plot(f, channel, args.fit_type)

    f.Close()

    logging.info(f"Output directory: {OUTPUT_DIR}")
    logging.info("Done.")
