#!/usr/bin/env python3
import os
import sys
import json
import logging
import argparse
import ROOT
from math import sqrt

# Add path to Common/Tools for plotter imports
WORKDIR_TEMP = os.getenv("WORKDIR")
if WORKDIR_TEMP:
    sys.path.insert(0, f"{WORKDIR_TEMP}/Common/Tools")
    from plotter import KinematicCanvas, ComparisonCanvas, get_CoM_energy, PALETTE
    import cmsstyle as CMS

# Argument parsing
parser = argparse.ArgumentParser(description="Validate histogram templates for HiggsCombine")
parser.add_argument("--era", required=True, type=str, help="Data-taking period (2016preVFP, 2017, 2018, 2022, etc.)")
parser.add_argument("--channel", required=True, type=str, help="Analysis channel (SR1E2Mu, SR3Mu)")
parser.add_argument("--masspoint", required=True, type=str, help="Signal mass point (e.g., MHc130_MA90)")
parser.add_argument("--method", required=True, type=str, help="Template method (Baseline, ParticleNet, etc.)")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args = parser.parse_args()

# Setup logging
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

# Path setup
WORKDIR = os.getenv("WORKDIR")
if not WORKDIR:
    raise EnvironmentError("WORKDIR environment variable not set. Please run 'source setup.sh'")

TEMPLATE_DIR = f"{WORKDIR}/SignalRegionStudy/templates/{args.era}/{args.channel}/{args.masspoint}/Shape/{args.method}"
VALIDATION_DIR = f"{TEMPLATE_DIR}/validation"

# Create validation output directory
os.makedirs(VALIDATION_DIR, exist_ok=True)

# Determine run period
if args.era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
    RUN = "Run2"
elif args.era in ["2022", "2022EE", "2023", "2023BPix"]:
    RUN = "Run3"
else:
    raise ValueError(f"Invalid era: {args.era}")

# Era suffix mapping for uncorrelated systematics
def get_era_suffix(era):
    """Get era suffix for systematic naming"""
    era_map = {
        "2016preVFP": "_16a", "2016postVFP": "_16b",
        "2017": "_17", "2018": "_18",
        "2022": "_22", "2022EE": "_22EE",
        "2023": "_23", "2023BPix": "_23BPix"
    }
    return era_map.get(era, "")

ERA_SUFFIX = get_era_suffix(args.era)

# Load systematics configuration
json_systematics = json.load(open(f"{WORKDIR}/SignalRegionStudy/configs/systematics.json"))

# Load experimental systematics (preprocessed shape systematics)
channel_config = json_systematics[RUN][args.channel]
prompt_systematics = {k: v.get("variations", [])
                      for k, v in channel_config.get("experimental", {}).items()
                      if v.get("source") == "preprocessed"}

# Track which systematics are uncorrelated (need era suffix in histogram names)
# Include both experimental and datadriven systematics
uncorrelated_systematics = set()

# Add uncorrelated experimental systematics
for k, v in channel_config.get("experimental", {}).items():
    if v.get("correlation") == "uncorrelated" and v.get("source") == "preprocessed":
        uncorrelated_systematics.add(k)

# Add uncorrelated datadriven systematics
for k, v in channel_config.get("datadriven", {}).items():
    if v.get("correlation") == "uncorrelated" and v.get("type") == "shape":
        uncorrelated_systematics.add(k)

# Process and background lists
PROCESSES = [args.masspoint, "nonprompt", "conversion", "diboson", "ttX", "others"]
BACKGROUNDS = ["nonprompt", "conversion", "diboson", "ttX", "others"]

# Validation counters
validation_issues = []
warnings = []

# Setup ROOT for batch mode
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)


def get_histogram_name(process, syst_name, variation):
    """
    Build the correct histogram name for a systematic variation.

    Args:
        process: Process name (e.g., "MHc130_MA90", "nonprompt")
        syst_name: Systematic base name (e.g., "L1Prefire", "PileupReweight")
        variation: "Up" or "Down"

    Returns:
        Histogram name as stored in shapes.root
    """
    # For uncorrelated systematics, add era suffix
    if syst_name in uncorrelated_systematics:
        return f"{process}_{syst_name}{ERA_SUFFIX}{variation}"

    # For correlated systematics, no era suffix
    return f"{process}_{syst_name}{variation}"


def get_lnN_error_fraction(process_name):
    """
    Calculate combined lnN fractional uncertainty for a process.

    Args:
        process_name: Process name (nonprompt, conversion, diboson, ttX, others)

    Returns:
        Fractional error (e.g., 0.20 for 20%)
    """
    lnN_errors_sq = []

    # Search all systematic categories for lnN systematics
    for category in channel_config.values():
        if not isinstance(category, dict):
            continue

        for syst_name, syst_props in category.items():
            # Check if this is an lnN systematic
            if syst_props.get("type") != "lnN":
                continue

            # Check if it applies to this process
            applies_to = syst_props.get("applies_to", [])
            if process_name not in applies_to:
                continue

            # Extract fractional error from value
            value = syst_props.get("value", 1.0)
            frac_error = abs(value - 1.0)
            lnN_errors_sq.append(frac_error**2)

    # Return quadrature sum
    if lnN_errors_sq:
        return sqrt(sum(lnN_errors_sq))
    return 0.0


def validate_histogram(hist, hist_name, min_entries=0):
    """
    Validate basic histogram properties.

    Args:
        hist: TH1D histogram
        hist_name: Histogram name for error reporting
        min_entries: Minimum required entries (default: 0)

    Returns:
        True if valid, False otherwise
    """
    if not hist:
        validation_issues.append(f"Histogram {hist_name} not found")
        return False

    # Check 1: Integral > 0
    integral = hist.Integral()
    if integral <= 0:
        validation_issues.append(f"{hist_name}: Zero or negative integral ({integral:.4f})")
        return False

    # Check 2: No negative bins
    for ibin in range(1, hist.GetNbinsX() + 1):
        content = hist.GetBinContent(ibin)
        if content < 0:
            validation_issues.append(f"{hist_name}: Negative bin content at bin {ibin} ({content:.4f})")
            return False

    # Check 3: Minimum entries
    if hist.GetEntries() < min_entries:
        warnings.append(f"{hist_name}: Low statistics ({hist.GetEntries()} entries)")

    return True


def check_systematic_variation(nominal_hist, syst_hist_up, syst_hist_down, process, syst_name):
    """
    Check systematic variation is reasonable.

    Args:
        nominal_hist: Central histogram
        syst_hist_up: Up variation
        syst_hist_down: Down variation
        process: Process name
        syst_name: Systematic name
    """
    nominal_integral = nominal_hist.Integral()
    if nominal_integral <= 0:
        return

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
            warnings.append(f"{process}_{syst_name}: Highly asymmetric variations (Up: {up_ratio:.2f}, Down: {down_ratio:.2f})")


def calculate_systematic_error(shapes_file, bkg_name, ibin):
    """
    Calculate systematic uncertainty for a specific background and bin using envelope method.

    Args:
        shapes_file: TFile containing histograms
        bkg_name: Background process name
        ibin: Bin number

    Returns:
        Systematic error (absolute)
    """
    # Get nominal histogram
    nominal = shapes_file.Get(bkg_name)
    if not nominal:
        return 0.0

    nominal_content = nominal.GetBinContent(ibin)
    if nominal_content <= 0:
        return 0.0

    # Calculate envelope for each systematic
    syst_errors_sq = []

    # For nonprompt, only use Nonprompt systematic
    if bkg_name == "nonprompt":
        up_hist = shapes_file.Get(get_histogram_name(bkg_name, "Nonprompt", "Up"))
        down_hist = shapes_file.Get(get_histogram_name(bkg_name, "Nonprompt", "Down"))

        if up_hist and down_hist:
            up_content = up_hist.GetBinContent(ibin)
            down_content = down_hist.GetBinContent(ibin)

            # Envelope: max deviation from nominal
            up_dev = abs(up_content - nominal_content)
            down_dev = abs(down_content - nominal_content)
            max_dev = max(up_dev, down_dev)
            syst_errors_sq.append(max_dev**2)

    # For prompt backgrounds, use all prompt systematics
    else:
        for syst_name in prompt_systematics.keys():
            up_hist = shapes_file.Get(get_histogram_name(bkg_name, syst_name, "Up"))
            down_hist = shapes_file.Get(get_histogram_name(bkg_name, syst_name, "Down"))

            if up_hist and down_hist:
                up_content = up_hist.GetBinContent(ibin)
                down_content = down_hist.GetBinContent(ibin)

                # Envelope: max deviation from nominal
                up_dev = abs(up_content - nominal_content)
                down_dev = abs(down_content - nominal_content)
                max_dev = max(up_dev, down_dev)
                syst_errors_sq.append(max_dev**2)

    # Add lnN rate uncertainties (flat rate systematics from config)
    lnN_frac = get_lnN_error_fraction(bkg_name)
    if lnN_frac > 0:
        lnN_abs = nominal_content * lnN_frac
        syst_errors_sq.append(lnN_abs**2)

    # Return quadrature sum of all systematics
    if syst_errors_sq:
        return sqrt(sum(syst_errors_sq))
    return 0.0


def make_background_stack(shapes_file):
    """
    Create background stack plot using ComparisonCanvas with signal overlay and systematic uncertainties.

    Args:
        shapes_file: TFile containing histograms
    """
    logging.info("Creating background stack plot with systematic uncertainties")

    # Get data_obs
    data_obs = shapes_file.Get("data_obs")
    if not data_obs:
        logging.warning("data_obs not found, skipping background stack plot")
        return

    # Clone to avoid modifying original
    incl = data_obs.Clone("data_obs_clone")
    incl.SetTitle("data_obs")

    # Get signal histogram
    signal_hist = shapes_file.Get(args.masspoint)
    if not signal_hist:
        logging.warning("Signal histogram not found, skipping background stack plot")
        return
    signal = signal_hist.Clone(f"{args.masspoint}_clone")

    # Get background histograms and add systematic uncertainties to each
    bkg_hists = {}
    total_bkg_yield = 0

    logging.debug("Calculating systematic uncertainties for each background")
    for bkg in BACKGROUNDS:
        hist = shapes_file.Get(bkg)
        if hist:
            hist_clone = hist.Clone(f"{bkg}_clone")

            # Add systematic uncertainties to each bin
            for ibin in range(1, hist_clone.GetNbinsX() + 1):
                stat_error = hist_clone.GetBinError(ibin)
                syst_error = calculate_systematic_error(shapes_file, bkg, ibin)

                # Total error: statistical and systematic in quadrature
                total_error = sqrt(stat_error**2 + syst_error**2)
                hist_clone.SetBinError(ibin, total_error)

            bkg_hists[bkg] = hist_clone
            total_bkg_yield += hist_clone.Integral()

    if not bkg_hists:
        logging.warning("No background histograms found, skipping plot")
        return

    # Calculate total systematic uncertainty for logging
    total_stat_error_sq = 0.0
    total_syst_error_sq = 0.0
    for bkg in BACKGROUNDS:
        nominal = shapes_file.Get(bkg)
        if nominal:
            for ibin in range(1, nominal.GetNbinsX() + 1):
                total_stat_error_sq += nominal.GetBinError(ibin)**2
                syst_err = calculate_systematic_error(shapes_file, bkg, ibin)
                total_syst_error_sq += syst_err**2

    total_stat = sqrt(total_stat_error_sq)
    total_syst = sqrt(total_syst_error_sq)
    logging.info(f"Background yield: {total_bkg_yield:.2f} ± {total_stat:.2f} (stat) ± {total_syst:.2f} (syst)")

    # Configuration for ComparisonCanvas
    config = {
        "era": args.era,
        "CoM": get_CoM_energy(args.era),
        "channel": f"{args.channel}",
        "xTitle": "Reconstructed A mass [GeV]",
        "yTitle": "Events",
        "rTitle": "Data / Pred",
        "maxDigits": 3,
        "systSrc": "Stat+Syst"  # Label for systematic source
    }

    # Create plot
    plotter = ComparisonCanvas(incl, bkg_hists, config)
    plotter.drawPadUp()

    # Draw signal on upper pad
    plotter.canv.cd(1)
    signal.SetLineColor(ROOT.kBlack)
    signal.SetLineWidth(2)
    signal.SetLineStyle(1)  # Solid line
    signal.Draw("HIST SAME")

    # Add signal to legend (find legend in current pad)
    current_pad = ROOT.gPad
    primitives = current_pad.GetListOfPrimitives()
    for obj in primitives:
        if obj.InheritsFrom("TLegend"):
            obj.AddEntry(signal, f"Signal ({signal.Integral():.1f})", "l")
            break

    plotter.drawPadDown()

    # Draw second line text manually (ROOT doesn't support \n in TLatex)
    plotter.canv.cd()
    CMS.drawText(f"{args.masspoint} ({args.method})", posX=0.2, posY=0.76, font=61, align=0, size=0.03)

    # Save
    output_path = f"{VALIDATION_DIR}/background_stack.png"
    plotter.canv.SaveAs(output_path)

    logging.info(f"Background stack saved: {output_path}")
    logging.info(f"Total background yield: {total_bkg_yield:.2f}")
    logging.info(f"Signal yield: {signal.Integral():.2f}")


def make_signal_vs_background(shapes_file):
    """
    Create signal vs background comparison plot using KinematicCanvas.

    Args:
        shapes_file: TFile containing histograms
    """
    logging.info("Creating signal vs background plot")

    # Get histograms
    signal_hist = shapes_file.Get(args.masspoint)
    data_obs = shapes_file.Get("data_obs")

    if not signal_hist or not data_obs:
        logging.warning("Signal or background histogram not found, skipping comparison plot")
        return

    # Clone to avoid modifying originals
    signal = signal_hist.Clone("signal_clone")
    background = data_obs.Clone("background_clone")

    # Calculate S/B ratio
    s = signal.Integral()
    b = background.Integral()
    sb_ratio = s / b if b > 0 else 0

    # Get x-axis range from histogram binning
    xmin = signal.GetXaxis().GetXmin()
    xmax = signal.GetXaxis().GetXmax()

    # Configuration for KinematicCanvas
    config = {
        "era": args.era,
        "CoM": get_CoM_energy(args.era),
        "channel": f"{args.channel}",
        "xTitle": "Reconstructed A mass [GeV]",
        "yTitle": "Events",
        "maxDigits": 3,
        "xRange": [xmin, xmax]
    }

    # Create plot with labeled histograms
    hists = {
        f"Signal ({s:.1f} events)": signal,
        f"Background ({b:.1f} events)": background
    }

    plotter = KinematicCanvas(hists, config)
    plotter.drawPad()

    # Draw additional channel info manually (ROOT doesn't support \n in TLatex)
    plotter.canv.cd()
    CMS.drawText(args.masspoint + f", S/B={sb_ratio:.3f}", posX=0.2, posY=0.65, font=61, align=0, size=0.05)

    # Save
    output_path = f"{VALIDATION_DIR}/signal_vs_background.png"
    plotter.canv.SaveAs(output_path)

    logging.info(f"Signal vs background plot saved: {output_path}")


def make_systematic_variations(shapes_file):
    """
    Create systematic uncertainty comparison plots using KinematicCanvas.
    Creates one publication-quality plot for each systematic.

    Args:
        shapes_file: TFile containing histograms
    """
    logging.info("Creating systematic variation plots")

    # Plot all systematics
    plot_systematics = list(prompt_systematics.keys())

    plots_created = 0
    for syst_name in plot_systematics:
        # Get signal histograms
        central = shapes_file.Get(args.masspoint)
        up_var = get_histogram_name(args.masspoint, syst_name, "Up")
        down_var = get_histogram_name(args.masspoint, syst_name, "Down")

        hist_up = shapes_file.Get(up_var)
        hist_down = shapes_file.Get(down_var)

        if not central or not hist_up or not hist_down:
            logging.debug(f"Skipping {syst_name} plot (histograms not found)")
            continue

        # Clone to avoid modifying originals
        central_clone = central.Clone(f"{syst_name}_central")
        up_clone = hist_up.Clone(f"{syst_name}_up")
        down_clone = hist_down.Clone(f"{syst_name}_down")

        # Calculate variation magnitudes
        central_int = central_clone.Integral()
        up_int = up_clone.Integral()
        down_int = down_clone.Integral()

        up_ratio = (up_int / central_int if central_int > 0 else 1.0)
        down_ratio = (down_int / central_int if central_int > 0 else 1.0)

        # Get x-axis range from histogram binning
        xmin = central_clone.GetXaxis().GetXmin()
        xmax = central_clone.GetXaxis().GetXmax()

        # Configuration for KinematicCanvas
        config = {
            "era": args.era,
            "CoM": get_CoM_energy(args.era),
            "channel": f"{args.channel}, {args.masspoint}",
            "xTitle": "Reconstructed A mass [GeV]",
            "yTitle": "Events",
            "maxDigits": 3,
            "xRange": [xmin, xmax]
        }

        # Create plot with variations
        hists = {
            "Central": central_clone,
            f"Up (+{(up_ratio-1)*100:.1f}%)": up_clone,
            f"Down ({(down_ratio-1)*100:.1f}%)": down_clone
        }

        plotter = KinematicCanvas(hists, config)
        plotter.drawPad()

        # Draw systematic name on second line (ROOT doesn't support \n in TLatex)
        plotter.canv.cd()
        CMS.drawText(syst_name, posX=0.2, posY=0.65, font=61, align=0, size=0.05)

        # Save individual plot
        output_path = f"{VALIDATION_DIR}/systematic_{syst_name}.png"
        plotter.canv.SaveAs(output_path)
        plots_created += 1

        logging.debug(f"Saved {syst_name} systematic plot: {output_path}")

    logging.info(f"Created {plots_created} systematic variation plots")


# Main execution
if __name__ == "__main__":
    logging.info(f"Starting template validation for {args.masspoint}, {args.era}, {args.channel}, {args.method}")

    # Open shapes file
    shapes_path = f"{TEMPLATE_DIR}/shapes.root"
    if not os.path.exists(shapes_path):
        raise FileNotFoundError(f"Shapes file not found: {shapes_path}")

    shapes_file = ROOT.TFile.Open(shapes_path, "READ")

    # ========================================
    # Validation 1: Basic Histogram Checks
    # ========================================
    logging.info("Validating histogram integrity...")

    # Check data_obs
    data_obs = shapes_file.Get("data_obs")
    validate_histogram(data_obs, "data_obs", min_entries=1)

    # Check signal central
    signal_hist = shapes_file.Get(args.masspoint)
    validate_histogram(signal_hist, args.masspoint, min_entries=10)

    # Check backgrounds
    for bkg in BACKGROUNDS:
        hist = shapes_file.Get(bkg)
        validate_histogram(hist, bkg, min_entries=1)

    # ========================================
    # Validation 2: Systematic Coverage
    # ========================================
    logging.info("Validating systematic variations...")

    # Check signal systematics
    for syst_name in prompt_systematics.keys():
        # Check Up variation
        hist_name_up = get_histogram_name(args.masspoint, syst_name, "Up")
        hist_up = shapes_file.Get(hist_name_up)

        if not hist_up:
            validation_issues.append(f"Missing systematic: {hist_name_up}")
        else:
            validate_histogram(hist_up, hist_name_up, min_entries=1)

        # Check Down variation
        hist_name_down = get_histogram_name(args.masspoint, syst_name, "Down")
        hist_down = shapes_file.Get(hist_name_down)

        if not hist_down:
            validation_issues.append(f"Missing systematic: {hist_name_down}")
        else:
            validate_histogram(hist_down, hist_name_down, min_entries=1)

        # Check variation magnitude if both exist
        if hist_up and hist_down:
            check_systematic_variation(signal_hist, hist_up, hist_down, args.masspoint, syst_name)

    # Check nonprompt systematics
    hist_nonprompt_central = shapes_file.Get("nonprompt")
    hist_nonprompt_up = shapes_file.Get(get_histogram_name("nonprompt", "Nonprompt", "Up"))
    hist_nonprompt_down = shapes_file.Get(get_histogram_name("nonprompt", "Nonprompt", "Down"))

    if not hist_nonprompt_up:
        validation_issues.append(f"Missing systematic: {get_histogram_name('nonprompt', 'Nonprompt', 'Up')}")
    if not hist_nonprompt_down:
        validation_issues.append(f"Missing systematic: {get_histogram_name('nonprompt', 'Nonprompt', 'Down')}")

    if hist_nonprompt_up and hist_nonprompt_down:
        check_systematic_variation(hist_nonprompt_central, hist_nonprompt_up,
                                  hist_nonprompt_down, "nonprompt", "Nonprompt")

    # Check prompt background systematics
    for bkg in ["conversion", "diboson", "ttX", "others"]:
        bkg_central = shapes_file.Get(bkg)

        for syst_name in prompt_systematics.keys():
            # Check Up variation
            hist_name_up = get_histogram_name(bkg, syst_name, "Up")
            hist_up = shapes_file.Get(hist_name_up)

            if not hist_up:
                validation_issues.append(f"Missing systematic: {hist_name_up}")
            else:
                validate_histogram(hist_up, hist_name_up, min_entries=0)

            # Check Down variation
            hist_name_down = get_histogram_name(bkg, syst_name, "Down")
            hist_down = shapes_file.Get(hist_name_down)

            if not hist_down:
                validation_issues.append(f"Missing systematic: {hist_name_down}")
            else:
                validate_histogram(hist_down, hist_name_down, min_entries=0)

    # ========================================
    # Validation 3: Generate Diagnostic Plots
    # ========================================
    logging.info("Generating diagnostic plots...")

    make_background_stack(shapes_file)
    make_signal_vs_background(shapes_file)
    make_systematic_variations(shapes_file)

    # ========================================
    # Validation Report
    # ========================================
    report_path = f"{VALIDATION_DIR}/validation_report.txt"
    with open(report_path, "w") as report:
        report.write("=" * 80 + "\n")
        report.write(f"Template Validation Report\n")
        report.write("=" * 80 + "\n")
        report.write(f"Era:        {args.era}\n")
        report.write(f"Channel:    {args.channel}\n")
        report.write(f"Masspoint:  {args.masspoint}\n")
        report.write(f"Method:     {args.method}\n")
        report.write(f"Input:      {shapes_path}\n")
        report.write("=" * 80 + "\n\n")

        # Process yields
        report.write("Process Yields:\n")
        report.write("-" * 80 + "\n")
        for process in PROCESSES:
            hist = shapes_file.Get(process)
            if hist:
                report.write(f"  {process:<20s}: {hist.Integral():>10.4f} events\n")
        data_obs = shapes_file.Get("data_obs")
        if data_obs:
            report.write(f"  {'data_obs':<20s}: {data_obs.Integral():>10.4f} events\n")
        report.write("\n")

        # S/B ratio
        if signal_hist and data_obs:
            sb_ratio = signal_hist.Integral() / data_obs.Integral() if data_obs.Integral() > 0 else 0
            report.write(f"Signal/Background ratio: {sb_ratio:.4f}\n\n")

        # Validation issues
        report.write("Validation Issues:\n")
        report.write("-" * 80 + "\n")
        if len(validation_issues) == 0:
            report.write("  ✓ No critical issues found\n")
        else:
            for issue in validation_issues:
                report.write(f"  ✗ {issue}\n")
        report.write("\n")

        # Warnings
        report.write("Warnings:\n")
        report.write("-" * 80 + "\n")
        if len(warnings) == 0:
            report.write("  ✓ No warnings\n")
        else:
            for warning in warnings:
                report.write(f"  ⚠ {warning}\n")
        report.write("\n")

        # Histogram count
        total_hists = len([key.GetName() for key in shapes_file.GetListOfKeys()])
        report.write(f"Total histograms: {total_hists}\n\n")

        # Output files
        report.write("Output Files:\n")
        report.write("-" * 80 + "\n")
        report.write(f"  Background stack:       {VALIDATION_DIR}/background_stack.png\n")
        report.write(f"  Signal vs background:   {VALIDATION_DIR}/signal_vs_background.png\n")

        # List individual systematic plots
        import glob
        syst_plots = glob.glob(f"{VALIDATION_DIR}/systematic_*.png")
        if syst_plots:
            report.write(f"  Systematic variations:  {len(syst_plots)} plots\n")
            for syst_plot in sorted(syst_plots):
                plot_name = os.path.basename(syst_plot)
                report.write(f"    - {plot_name}\n")
        else:
            report.write(f"  Systematic variations:  No plots generated\n")

        report.write(f"  Validation report:      {report_path}\n")
        report.write("\n")

        # Summary
        report.write("=" * 80 + "\n")
        if len(validation_issues) == 0:
            report.write("VALIDATION: PASSED\n")
        else:
            report.write("VALIDATION: FAILED\n")
        report.write("=" * 80 + "\n")

    shapes_file.Close()

    # Print summary to console
    logging.info("=" * 60)
    logging.info("Validation complete!")
    logging.info(f"Report: {report_path}")
    logging.info("=" * 60)
    logging.info(f"Critical issues: {len(validation_issues)}")
    logging.info(f"Warnings: {len(warnings)}")

    if len(validation_issues) == 0:
        logging.info("✓ Validation PASSED")
    else:
        logging.error("✗ Validation FAILED")
        for issue in validation_issues[:5]:  # Print first 5 issues
            logging.error(f"  - {issue}")
        if len(validation_issues) > 5:
            logging.error(f"  ... and {len(validation_issues) - 5} more issues")

    logging.info("=" * 60)
