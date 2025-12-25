#!/usr/bin/env python3
"""
Validate histogram templates for HiggsCombine.

This script performs validation checks on generated templates and creates
diagnostic plots for visual inspection.

Features:
- Dynamic loading of shape systematics from configs/systematics.{era}.json
- Dynamic loading of background processes from process_list.json
- Background stack plot with signal overlay
- Signal vs background comparison
- Systematic variation plots for each source

Usage:
    checkTemplates.py --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --method Baseline --binning uniform
"""
import os
import sys
import json
import logging
import argparse
import ROOT
from math import sqrt

# Argument parsing
parser = argparse.ArgumentParser(description="Validate histogram templates for HiggsCombine")
parser.add_argument("--era", required=True, type=str, help="Data-taking period (2016preVFP, 2017, 2018, 2022, etc.)")
parser.add_argument("--channel", required=True, type=str, help="Analysis channel (SR1E2Mu, SR3Mu)")
parser.add_argument("--masspoint", required=True, type=str, help="Signal mass point (e.g., MHc130_MA90)")
parser.add_argument("--method", required=True, type=str, help="Template method (Baseline, ParticleNet, etc.)")
parser.add_argument("--binning", default="uniform", choices=["uniform", "sigma"],
                    help="Binning method: 'uniform' (15 bins, default) or 'sigma' (non-uniform)")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args = parser.parse_args()

# Setup logging
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                    format='%(levelname)s - %(message)s')

# Path setup
WORKDIR = os.getenv("WORKDIR")
if not WORKDIR:
    raise EnvironmentError("WORKDIR environment variable not set. Please run 'source setup.sh'")

# Add path to Common/Tools for plotter imports
sys.path.insert(0, f"{WORKDIR}/Common/Tools")
from plotter import KinematicCanvas, ComparisonCanvas, get_CoM_energy
import cmsstyle as CMS

TEMPLATE_DIR = f"{WORKDIR}/SignalRegionStudyV1/templates/{args.era}/{args.channel}/{args.masspoint}/{args.method}/{args.binning}"
VALIDATION_DIR = f"{TEMPLATE_DIR}/validation"

# Create validation output directory
os.makedirs(VALIDATION_DIR, exist_ok=True)

# Setup ROOT for batch mode
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)


# =============================================================================
# Configuration Loading
# =============================================================================

def load_systematics_config():
    """Load shape systematics from era-specific config."""
    config_path = f"{WORKDIR}/SignalRegionStudyV1/configs/systematics.{args.era}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Systematics config not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    if args.channel not in config:
        raise ValueError(f"Channel '{args.channel}' not found in {config_path}")

    return config[args.channel]


def load_process_list():
    """Load dynamic process list from template output."""
    process_list_path = f"{TEMPLATE_DIR}/process_list.json"
    if not os.path.exists(process_list_path):
        raise FileNotFoundError(f"Process list not found: {process_list_path}")

    with open(process_list_path) as f:
        return json.load(f)


def get_shape_systematics(syst_config):
    """
    Extract shape systematics that have Up/Down variations in shapes.root.

    Returns dict: {syst_name: {"group": [...], "source": "preprocessed"|"valued"}}
    """
    shape_systs = {}

    for syst_name, props in syst_config.items():
        if props.get("type") != "shape":
            continue

        # Only include systematics that produce Up/Down histograms
        source = props.get("source", "")
        if source in ["preprocessed", "valued"]:
            shape_systs[syst_name] = {
                "group": props.get("group", []),
                "source": source
            }

    return shape_systs


def get_lnN_systematics(syst_config):
    """Extract lnN (rate) systematics."""
    lnN_systs = {}

    for syst_name, props in syst_config.items():
        if props.get("type") != "lnN":
            continue

        lnN_systs[syst_name] = {
            "value": props.get("value", 1.0),
            "group": props.get("group", [])
        }

    return lnN_systs


# =============================================================================
# Histogram Validation Functions
# =============================================================================

def validate_histogram(hist, hist_name, min_entries=0):
    """
    Validate basic histogram properties.

    Returns:
        Tuple of (is_valid, issues_list)
    """
    issues = []

    if not hist:
        issues.append(f"Histogram {hist_name} not found")
        return False, issues

    # Check 1: Integral > 0
    integral = hist.Integral()
    if integral <= 0:
        issues.append(f"{hist_name}: Zero or negative integral ({integral:.4f})")

    # Check 2: No negative bins
    for ibin in range(1, hist.GetNbinsX() + 1):
        content = hist.GetBinContent(ibin)
        if content < 0:
            issues.append(f"{hist_name}: Negative bin content at bin {ibin} ({content:.4f})")

    # Check 3: Minimum entries
    if hist.GetEntries() < min_entries:
        issues.append(f"{hist_name}: Low statistics ({hist.GetEntries()} entries < {min_entries})")

    return len(issues) == 0, issues


def check_systematic_variation(nominal_hist, syst_hist_up, syst_hist_down, process, syst_name):
    """Check systematic variation is reasonable. Returns list of warnings."""
    warnings = []

    nominal_integral = nominal_hist.Integral()
    if nominal_integral <= 0:
        return warnings

    # Check Up variation
    if syst_hist_up:
        up_integral = syst_hist_up.Integral()
        up_ratio = up_integral / nominal_integral

        if up_ratio < 0.5 or up_ratio > 2.0:
            warnings.append(f"{process}_{syst_name}Up: Large variation (ratio={up_ratio:.2f})")

    # Check Down variation
    if syst_hist_down:
        down_integral = syst_hist_down.Integral()
        down_ratio = down_integral / nominal_integral

        if down_ratio < 0.5 or down_ratio > 2.0:
            warnings.append(f"{process}_{syst_name}Down: Large variation (ratio={down_ratio:.2f})")

    # Check symmetry (only if both variations exist)
    if syst_hist_up and syst_hist_down:
        up_integral = syst_hist_up.Integral()
        down_integral = syst_hist_down.Integral()
        up_ratio = up_integral / nominal_integral
        down_ratio = down_integral / nominal_integral

        asymmetry = abs((up_ratio - 1) + (down_ratio - 1)) / 2
        if asymmetry > 0.5:  # More than 50% asymmetry
            warnings.append(f"{process}_{syst_name}: Highly asymmetric (Up: {up_ratio:.2f}, Down: {down_ratio:.2f})")

    return warnings


# =============================================================================
# Systematic Error Calculation
# =============================================================================

def calculate_systematic_error(shapes_file, process, ibin, shape_systs):
    """
    Calculate systematic uncertainty for a specific process and bin.

    Args:
        shapes_file: TFile containing histograms
        process: Process name
        ibin: Bin number
        shape_systs: Dictionary of shape systematics

    Returns:
        Systematic error (absolute)
    """
    nominal = shapes_file.Get(process)
    if not nominal:
        return 0.0

    nominal_content = nominal.GetBinContent(ibin)
    if nominal_content <= 0:
        return 0.0

    syst_errors_sq = []

    for syst_name, props in shape_systs.items():
        # Check if this systematic applies to this process
        group = props["group"]
        process_key = "signal" if process == args.masspoint else process

        if process_key not in group:
            continue

        # Get Up/Down histograms
        up_hist = shapes_file.Get(f"{process}_{syst_name}Up")
        down_hist = shapes_file.Get(f"{process}_{syst_name}Down")

        if up_hist and down_hist:
            up_content = up_hist.GetBinContent(ibin)
            down_content = down_hist.GetBinContent(ibin)

            # Envelope: max deviation from nominal
            up_dev = abs(up_content - nominal_content)
            down_dev = abs(down_content - nominal_content)
            max_dev = max(up_dev, down_dev)
            syst_errors_sq.append(max_dev ** 2)

    if syst_errors_sq:
        return sqrt(sum(syst_errors_sq))
    return 0.0


# =============================================================================
# Plotting Functions
# =============================================================================

def make_background_stack(shapes_file, backgrounds, shape_systs):
    """
    Create background stack plot with signal overlay and systematic uncertainties.
    """
    logging.info("Creating background stack plot...")

    # Get data_obs
    data_obs = shapes_file.Get("data_obs")
    if not data_obs:
        logging.warning("data_obs not found, skipping background stack plot")
        return

    incl = data_obs.Clone("data_obs_clone")
    incl.SetTitle("data_obs")

    # Get signal histogram
    signal_hist = shapes_file.Get(args.masspoint)
    if not signal_hist:
        logging.warning("Signal histogram not found, skipping background stack plot")
        return
    signal = signal_hist.Clone(f"{args.masspoint}_clone")

    # Get background histograms with systematic uncertainties
    bkg_hists = {}
    total_bkg_yield = 0

    for bkg in backgrounds:
        hist = shapes_file.Get(bkg)
        if hist and hist.Integral() > 0:
            hist_clone = hist.Clone(f"{bkg}_clone")

            # Add systematic uncertainties to each bin
            for ibin in range(1, hist_clone.GetNbinsX() + 1):
                stat_error = hist_clone.GetBinError(ibin)
                syst_error = calculate_systematic_error(shapes_file, bkg, ibin, shape_systs)
                total_error = sqrt(stat_error ** 2 + syst_error ** 2)
                hist_clone.SetBinError(ibin, total_error)

            bkg_hists[bkg] = hist_clone
            total_bkg_yield += hist_clone.Integral()

    if not bkg_hists:
        logging.warning("No background histograms found, skipping plot")
        return

    # Configuration for ComparisonCanvas
    config = {
        "era": args.era,
        "CoM": get_CoM_energy(args.era),
        "channel": f"{args.channel}",
        "xTitle": "M(#mu^{+}#mu^{-}) [GeV]",
        "yTitle": "Events",
        "rTitle": "Data / Pred",
        "maxDigits": 3,
        "systSrc": "Stat+Syst"
    }

    # Create plot
    plotter = ComparisonCanvas(incl, bkg_hists, config)
    plotter.drawPadUp()

    # Draw signal on upper pad
    plotter.canv.cd(1)
    signal.SetLineColor(ROOT.kBlack)
    signal.SetLineWidth(2)
    signal.SetLineStyle(1)
    signal.Draw("HIST SAME")

    # Add signal to legend
    current_pad = ROOT.gPad
    primitives = current_pad.GetListOfPrimitives()
    for obj in primitives:
        if obj.InheritsFrom("TLegend"):
            obj.AddEntry(signal, f"Signal ({signal.Integral():.1f})", "l")
            break

    plotter.drawPadDown()

    # Draw additional text
    plotter.canv.cd()
    CMS.drawText(f"{args.masspoint} ({args.method})", posX=0.2, posY=0.76, font=61, align=0, size=0.03)

    # Save
    output_path = f"{VALIDATION_DIR}/background_stack.png"
    plotter.canv.SaveAs(output_path)

    logging.info(f"  Saved: {output_path}")
    logging.info(f"  Background yield: {total_bkg_yield:.2f}")
    logging.info(f"  Signal yield: {signal.Integral():.2f}")


def make_signal_vs_background(shapes_file):
    """Create signal vs background comparison plot."""
    logging.info("Creating signal vs background plot...")

    signal_hist = shapes_file.Get(args.masspoint)
    data_obs = shapes_file.Get("data_obs")

    if not signal_hist or not data_obs:
        logging.warning("Signal or background histogram not found, skipping comparison plot")
        return

    signal = signal_hist.Clone("signal_clone")
    background = data_obs.Clone("background_clone")

    s = signal.Integral()
    b = background.Integral()
    sb_ratio = s / b if b > 0 else 0

    xmin = signal.GetXaxis().GetXmin()
    xmax = signal.GetXaxis().GetXmax()

    config = {
        "era": args.era,
        "CoM": get_CoM_energy(args.era),
        "channel": f"{args.channel}",
        "xTitle": "M(#mu^{+}#mu^{-}) [GeV]",
        "yTitle": "Events",
        "maxDigits": 3,
        "xRange": [xmin, xmax]
    }

    hists = {
        f"Signal ({s:.1f} events)": signal,
        f"Background ({b:.1f} events)": background
    }

    plotter = KinematicCanvas(hists, config)
    plotter.drawPad()

    plotter.canv.cd()
    CMS.drawText(f"{args.masspoint}, S/B={sb_ratio:.3f}", posX=0.2, posY=0.65, font=61, align=0, size=0.05)

    output_path = f"{VALIDATION_DIR}/signal_vs_background.png"
    plotter.canv.SaveAs(output_path)

    logging.info(f"  Saved: {output_path}")


def make_systematic_plot(shapes_file, process, syst_name):
    """
    Create systematic variation plot for a specific process and systematic.

    Returns True if plot was created, False otherwise.
    """
    central = shapes_file.Get(process)
    hist_up = shapes_file.Get(f"{process}_{syst_name}Up")
    hist_down = shapes_file.Get(f"{process}_{syst_name}Down")

    if not central or not hist_up or not hist_down:
        return False

    central_clone = central.Clone(f"{syst_name}_central")
    up_clone = hist_up.Clone(f"{syst_name}_up")
    down_clone = hist_down.Clone(f"{syst_name}_down")

    central_int = central_clone.Integral()
    if central_int <= 0:
        return False

    up_int = up_clone.Integral()
    down_int = down_clone.Integral()

    up_ratio = up_int / central_int
    down_ratio = down_int / central_int

    xmin = central_clone.GetXaxis().GetXmin()
    xmax = central_clone.GetXaxis().GetXmax()

    config = {
        "era": args.era,
        "CoM": get_CoM_energy(args.era),
        "channel": f"{args.channel}, {process}",
        "xTitle": "M(#mu^{+}#mu^{-}) [GeV]",
        "yTitle": "Events",
        "maxDigits": 3,
        "xRange": [xmin, xmax]
    }

    hists = {
        "Central": central_clone,
        f"Up (+{(up_ratio - 1) * 100:.1f}%)": up_clone,
        f"Down ({(down_ratio - 1) * 100:.1f}%)": down_clone
    }

    plotter = KinematicCanvas(hists, config)
    plotter.drawPad()

    plotter.canv.cd()
    CMS.drawText(syst_name, posX=0.2, posY=0.65, font=61, align=0, size=0.04)

    output_path = f"{VALIDATION_DIR}/systematic_{process}_{syst_name}.png"
    plotter.canv.SaveAs(output_path)

    return True


def make_systematic_stack_plot(shapes_file, syst_name, processes, group):
    """
    Create stacked plot showing Central/Up/Down variations for a systematic.

    Stacks ALL processes, but only varies the ones the systematic applies to.
    Non-applicable processes use their Central histogram for all three curves.

    Args:
        shapes_file: ROOT TFile with histograms
        syst_name: Systematic name
        processes: List of all processes (signal + backgrounds)
        group: List of process categories this systematic applies to

    Returns True if plot was created, False otherwise.
    """
    # Build stacked histograms for Central, Up, Down
    stack_central = None
    stack_up = None
    stack_down = None

    applicable_processes = []

    for process in processes:
        # Skip signal - only stack backgrounds
        if process == args.masspoint:
            continue

        central = shapes_file.Get(process)
        if not central:
            continue

        is_applicable = process in group

        if is_applicable:
            hist_up = shapes_file.Get(f"{process}_{syst_name}Up")
            hist_down = shapes_file.Get(f"{process}_{syst_name}Down")

            if not hist_up or not hist_down:
                # Systematic should apply but histograms missing - use central
                hist_up = central
                hist_down = central
            else:
                applicable_processes.append(process)
        else:
            # Systematic doesn't apply - use central for all
            hist_up = central
            hist_down = central

        if stack_central is None:
            stack_central = central.Clone(f"stack_central_{syst_name}")
            stack_up = hist_up.Clone(f"stack_up_{syst_name}")
            stack_down = hist_down.Clone(f"stack_down_{syst_name}")
        else:
            stack_central.Add(central)
            stack_up.Add(hist_up)
            stack_down.Add(hist_down)

    if stack_central is None or stack_central.Integral() <= 0:
        return False

    central_int = stack_central.Integral()
    up_int = stack_up.Integral()
    down_int = stack_down.Integral()

    up_ratio = up_int / central_int
    down_ratio = down_int / central_int

    xmin = stack_central.GetXaxis().GetXmin()
    xmax = stack_central.GetXaxis().GetXmax()

    config = {
        "era": args.era,
        "CoM": get_CoM_energy(args.era),
        "channel": f"{args.channel}",
        "xTitle": "M(#mu^{+}#mu^{-}) [GeV]",
        "yTitle": "Events",
        "maxDigits": 3,
        "xRange": [xmin, xmax]
    }

    hists = {
        f"Central ({central_int:.1f})": stack_central,
        f"Up (+{(up_ratio - 1) * 100:.1f}%)": stack_up,
        f"Down ({(down_ratio - 1) * 100:.1f}%)": stack_down
    }

    plotter = KinematicCanvas(hists, config)
    plotter.drawPad()

    plotter.canv.cd()
    CMS.drawText(syst_name, posX=0.2, posY=0.65, font=61, align=0, size=0.04)
    CMS.drawText(f"Varies: {', '.join(applicable_processes)}", posX=0.2, posY=0.60, font=42, align=0, size=0.025)

    output_path = f"{VALIDATION_DIR}/systematic_stacked_{syst_name}.png"
    plotter.canv.SaveAs(output_path)

    return True


def make_all_systematic_plots(shapes_file, processes, shape_systs):
    """Create systematic variation plots for all processes and systematics."""
    logging.info("Creating systematic variation plots...")

    # Individual process plots
    individual_plots = 0
    for syst_name, props in shape_systs.items():
        group = props["group"]

        for process in processes:
            process_key = "signal" if process == args.masspoint else process
            if process_key not in group:
                continue

            if make_systematic_plot(shapes_file, process, syst_name):
                individual_plots += 1
                logging.debug(f"  Created: systematic_{process}_{syst_name}.png")

    logging.info(f"  Created {individual_plots} individual systematic plots")

    # Stacked summary plots
    stacked_plots = 0
    for syst_name, props in shape_systs.items():
        group = props["group"]

        if make_systematic_stack_plot(shapes_file, syst_name, processes, group):
            stacked_plots += 1
            logging.debug(f"  Created: systematic_stacked_{syst_name}.png")

    logging.info(f"  Created {stacked_plots} stacked systematic plots")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    logging.info(f"Starting template validation")
    logging.info(f"  Era: {args.era}")
    logging.info(f"  Channel: {args.channel}")
    logging.info(f"  Masspoint: {args.masspoint}")
    logging.info(f"  Method: {args.method}")

    # Load configurations
    syst_config = load_systematics_config()
    process_list = load_process_list()

    shape_systs = get_shape_systematics(syst_config)
    lnN_systs = get_lnN_systematics(syst_config)

    logging.info(f"Loaded {len(shape_systs)} shape systematics")
    logging.info(f"Loaded {len(lnN_systs)} lnN systematics")

    # Build process lists
    separate_processes = process_list.get("separate_processes", ["nonprompt"])
    backgrounds = separate_processes + ["others"]
    all_processes = [args.masspoint] + backgrounds

    logging.info(f"Processes: {all_processes}")

    # Open shapes file
    shapes_path = f"{TEMPLATE_DIR}/shapes.root"
    if not os.path.exists(shapes_path):
        raise FileNotFoundError(f"Shapes file not found: {shapes_path}")

    shapes_file = ROOT.TFile.Open(shapes_path, "READ")

    # ========================================
    # Validation 1: Basic Histogram Checks
    # ========================================
    logging.info("=" * 60)
    logging.info("Validating histogram integrity...")

    all_issues = []
    all_warnings = []

    # Check data_obs
    data_obs = shapes_file.Get("data_obs")
    valid, issues = validate_histogram(data_obs, "data_obs", min_entries=1)
    all_issues.extend(issues)

    # Check signal
    signal_hist = shapes_file.Get(args.masspoint)
    valid, issues = validate_histogram(signal_hist, args.masspoint, min_entries=10)
    all_issues.extend(issues)

    # Check backgrounds
    for bkg in backgrounds:
        hist = shapes_file.Get(bkg)
        valid, issues = validate_histogram(hist, bkg, min_entries=1)
        all_issues.extend(issues)

    # ========================================
    # Validation 2: Systematic Coverage
    # ========================================
    logging.info("Validating systematic variations...")

    for syst_name, props in shape_systs.items():
        group = props["group"]

        for process in all_processes:
            process_key = "signal" if process == args.masspoint else process
            if process_key not in group:
                continue

            # Check Up variation
            hist_up = shapes_file.Get(f"{process}_{syst_name}Up")
            if not hist_up:
                all_issues.append(f"Missing: {process}_{syst_name}Up")

            # Check Down variation
            hist_down = shapes_file.Get(f"{process}_{syst_name}Down")
            if not hist_down:
                all_issues.append(f"Missing: {process}_{syst_name}Down")

            # Check variation magnitude
            nominal = shapes_file.Get(process)
            if nominal and hist_up and hist_down:
                warnings = check_systematic_variation(nominal, hist_up, hist_down, process, syst_name)
                all_warnings.extend(warnings)

    # ========================================
    # Validation 3: Generate Diagnostic Plots
    # ========================================
    logging.info("=" * 60)
    logging.info("Generating diagnostic plots...")

    make_background_stack(shapes_file, backgrounds, shape_systs)
    make_signal_vs_background(shapes_file)
    make_all_systematic_plots(shapes_file, all_processes, shape_systs)

    shapes_file.Close()

    # ========================================
    # Print Summary
    # ========================================
    logging.info("=" * 60)
    logging.info("Validation Summary")
    logging.info("=" * 60)

    # Print yields
    shapes_file = ROOT.TFile.Open(shapes_path, "READ")
    logging.info("Process Yields:")
    for process in all_processes:
        hist = shapes_file.Get(process)
        if hist:
            logging.info(f"  {process:<20s}: {hist.Integral():>10.4f} events")
    data_obs = shapes_file.Get("data_obs")
    if data_obs:
        logging.info(f"  {'data_obs':<20s}: {data_obs.Integral():>10.4f} events")

    # S/B ratio
    signal = shapes_file.Get(args.masspoint)
    if signal and data_obs and data_obs.Integral() > 0:
        sb_ratio = signal.Integral() / data_obs.Integral()
        logging.info(f"S/B ratio: {sb_ratio:.4f}")

    shapes_file.Close()

    # Print issues and warnings
    logging.info("-" * 60)
    logging.info(f"Critical issues: {len(all_issues)}")
    for issue in all_issues[:10]:
        logging.error(f"  {issue}")
    if len(all_issues) > 10:
        logging.error(f"  ... and {len(all_issues) - 10} more")

    logging.info(f"Warnings: {len(all_warnings)}")
    for warning in all_warnings[:10]:
        logging.warning(f"  {warning}")
    if len(all_warnings) > 10:
        logging.warning(f"  ... and {len(all_warnings) - 10} more")

    logging.info("=" * 60)
    if len(all_issues) == 0:
        logging.info("VALIDATION: PASSED")
    else:
        logging.error("VALIDATION: FAILED")
    logging.info(f"Output directory: {VALIDATION_DIR}")
    logging.info("=" * 60)
