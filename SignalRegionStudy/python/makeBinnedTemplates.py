#!/usr/bin/env python3
import os
import shutil
import logging
import argparse
import json
import ROOT
from math import sqrt

# Argument parsing
parser = argparse.ArgumentParser(description="Generate binned histogram templates for HiggsCombine")
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

BASEDIR = f"{WORKDIR}/SignalRegionStudy/samples/{args.era}/{args.channel}/{args.masspoint}/{args.method}"
OUTDIR = f"{WORKDIR}/SignalRegionStudy/templates/{args.era}/{args.channel}/{args.masspoint}/Shape/{args.method}"

logging.info(f"Input directory: {BASEDIR}")
logging.info(f"Output directory: {OUTDIR}")

# Load SignalRegionStudy library
lib_path = f"{WORKDIR}/SignalRegionStudy/lib/libSignalRegionStudy.so"
if not os.path.exists(lib_path):
    raise FileNotFoundError(f"SignalRegionStudy library not found: {lib_path}. Please run './scripts/build.sh'")
ROOT.gSystem.Load(lib_path)

# Load JSON configurations
json_systematics = json.load(open(f"{WORKDIR}/SignalRegionStudy/configs/systematics.json"))

# Determine run period
if args.era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
    RUN = "Run2"
elif args.era in ["2022", "2022EE", "2023", "2023BPix"]:
    RUN = "Run3"
else:
    raise ValueError(f"Invalid era: {args.era}")

# Load experimental systematics (preprocessed prompt systematics)
channel_config = json_systematics[RUN][args.channel]
prompt_systematics = {k: v.get("variations", [])
                      for k, v in channel_config.get("experimental", {}).items()
                      if v.get("source") == "preprocessed"}
logging.info(f"Loaded {len(prompt_systematics)} experimental systematic categories for {RUN} {args.channel}")


# Helper function: A mass fitting
def getFitResult(input_path, output_path, mA_nominal):
    """
    Fit A mass distribution using AmassFitter C++ class.

    Args:
        input_path: Path to input ROOT file with signal
        output_path: Path to save fit results
        mA_nominal: Nominal A mass value

    Returns:
        Tuple of (fitted_mA, width, sigma)
    """
    logging.info(f"Fitting A mass distribution with nominal mA = {mA_nominal} GeV")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    fitter = ROOT.AmassFitter(input_path, output_path)
    fitter.fitMass(mA_nominal, mA_nominal - 20., mA_nominal + 20.)
    fitter.saveCanvas(f"{OUTDIR}/fit_result.png")

    # Extract fit parameters
    mA = fitter.getRooMA().getVal()
    width = fitter.getRooWidth().getVal()
    sigma = fitter.getRooSigma().getVal()
    fitter.Close()

    logging.info(f"Fit results: mA = {mA:.2f} GeV, Γ = {width:.3f} GeV, σ = {sigma:.3f} GeV")

    return mA, width, sigma


def loadFitResult(fit_result_path):
    """
    Load fit parameters from an existing fit_result.root file.

    This is used for SR3Mu channel to load parameters from SR1E2Mu,
    since SR3Mu has fake A mass candidates that make direct fitting unreliable.

    Args:
        fit_result_path: Path to existing fit_result.root file

    Returns:
        Tuple of (fitted_mA, width, sigma)
    """
    logging.info(f"Loading fit parameters from: {fit_result_path}")

    if not os.path.exists(fit_result_path):
        raise FileNotFoundError(f"Fit result file not found: {fit_result_path}")

    # Open the fit result file
    fit_file = ROOT.TFile.Open(fit_result_path, "READ")

    # Get the RooFitResult object
    fit_result = fit_file.Get("fitresult_model_data")
    if not fit_result:
        # Try alternative name
        fit_result = fit_file.Get("fit_result")
        if not fit_result:
            fit_file.Close()
            raise RuntimeError("RooFitResult not found in fit_result.root")

    # Extract fitted parameters
    params = fit_result.floatParsFinal()
    mA = params.find("mA").getVal()
    width = params.find("width").getVal()
    sigma = params.find("sigma").getVal()

    fit_file.Close()

    logging.info(f"Loaded fit results: mA = {mA:.2f} GeV, Γ = {width:.3f} GeV, σ = {sigma:.3f} GeV")

    return mA, width, sigma


# Helper function: Ensure positive integral
def ensurePositiveIntegral(hist, min_integral=1e-10):
    """
    Ensure histogram has positive integral for normalization.
    If integral is negative or zero, set a minimal positive value.

    Args:
        hist: TH1D histogram
        min_integral: Minimum integral value to set

    Returns:
        True if histogram was modified, False otherwise
    """
    integral = hist.Integral()
    if integral <= 0:
        logging.warning(f"  Histogram {hist.GetName()} has non-positive integral: {integral:.4e}")

        # Find the central bin
        central_bin = hist.GetNbinsX() // 2 + 1

        # Set a small positive value in the central bin
        logging.warning(f"  Setting bin {central_bin} to {min_integral} to ensure positive normalization")
        hist.SetBinContent(central_bin, min_integral)
        hist.SetBinError(central_bin, min_integral)

        return True
    return False


# Helper function: Create histogram
def getHist(process, mA, width, sigma, syst="Central"):
    """
    Create histogram from preprocessed tree using RDataFrame.
    Always uses 'mass' branch (already set in preprocessing).

    Args:
        process: Process name (e.g., "MHc130_MA90", "nonprompt", "diboson")
        mA: Fitted A mass
        width: Breit-Wigner width
        sigma: Gaussian resolution
        syst: Systematic variation name (e.g., "Central", "MuonIDSF_Up")

    Returns:
        TH1D histogram with negative bins fixed
    """
    # File and tree names
    file_path = f"{BASEDIR}/{process}.root"
    tree_name = syst
    # Combine expects no underscore before Up/Down, so remove it
    syst_formatted = syst.replace("_Up", "Up").replace("_Down", "Down")
    hist_name = process if syst == "Central" else f"{process}_{syst_formatted}"

    logging.debug(f"Creating histogram: {hist_name} from tree {tree_name}")

    # Check file existence
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample file not found: {file_path}")

    # Calculate binning: mA ± 5×√(Γ² + σ²)
    mass_range = 5 * sqrt(width**2 + sigma**2)
    nbins = 15
    mass_min = mA - mass_range
    mass_max = mA + mass_range
    hist_range = (nbins, mass_min, mass_max)

    # Check if tree exists
    test_file = ROOT.TFile.Open(file_path)
    tree = test_file.Get(tree_name)
    if not tree:
        test_file.Close()
        raise RuntimeError(f"Tree '{tree_name}' not found in {file_path}")
    test_file.Close()

    # Create RDataFrame and fill histogram
    rdf = ROOT.RDataFrame(tree_name, file_path)
    hist = rdf.Histo1D((hist_name, "", *hist_range), "mass", "weight")

    # Detach from file
    hist_result = hist.GetValue()
    hist_result.SetDirectory(0)

    # Ensure positive integral for normalization
    was_modified = ensurePositiveIntegral(hist_result)

    # Log final state
    if was_modified:
        logging.info(f"Histogram {hist_name} modified to ensure positive integral: {hist_result.Integral():.4e}")
    else:
        logging.debug(f"Histogram {hist_name}: {hist_result.GetEntries()} entries, integral = {hist_result.Integral():.4f}")

    return hist_result


# Main execution
if __name__ == "__main__":
    logging.info(f"Starting template generation for {args.masspoint}, {args.era}, {args.channel}, {args.method}")

    # Create output directory
    if os.path.exists(OUTDIR):
        logging.info(f"Removing existing output directory: {OUTDIR}")
        shutil.rmtree(OUTDIR)
    os.makedirs(OUTDIR, exist_ok=True)

    # Extract nominal mA from masspoint name (e.g., "MHc130_MA90" -> 90.0)
    mA_nominal = float(args.masspoint.split("_")[1].replace("MA", ""))

    # Determine whether to fit or load parameters
    if args.channel == "SR3Mu":
        # SR3Mu has fake A mass candidates, so use SR1E2Mu fit results
        sr1e2mu_fit_path = f"{WORKDIR}/SignalRegionStudy/templates/{args.era}/SR1E2Mu/{args.masspoint}/Shape/{args.method}/fit_result.root"

        if not os.path.exists(sr1e2mu_fit_path):
            raise FileNotFoundError(
                f"SR1E2Mu fit results not found: {sr1e2mu_fit_path}\n"
                f"Please run makeBinnedTemplates.py for SR1E2Mu first:\n"
                f"  makeBinnedTemplates.py --era {args.era} --channel SR1E2Mu --masspoint {args.masspoint} --method {args.method}"
            )

        mA, width, sigma = loadFitResult(sr1e2mu_fit_path)

        # Still save a marker file for SR3Mu (contains no actual fit)
        marker_file = ROOT.TFile(f"{OUTDIR}/fit_result.root", "RECREATE")
        marker_file.WriteObject(ROOT.TNamed("source", f"SR1E2Mu/{args.era}/{args.masspoint}"), "fit_source")
        marker_file.Close()

    else:
        # SR1E2Mu: Perform direct fit
        signal_path = f"{BASEDIR}/{args.masspoint}.root"
        mA, width, sigma = getFitResult(signal_path, f"{OUTDIR}/fit_result.root", mA_nominal)

    # Calculate mass window
    mass_range = 5 * sqrt(width**2 + sigma**2)
    logging.info(f"Mass window: [{mA - mass_range:.2f}, {mA + mass_range:.2f}] GeV")
    logging.info(f"Bin width: {2*mass_range/15:.3f} GeV")

    # Create output ROOT file
    output_file = ROOT.TFile(f"{OUTDIR}/shapes.root", "RECREATE")

    # Initialize data_obs histogram (will be sum of all backgrounds)
    nbins = 15
    mass_min = mA - mass_range
    mass_max = mA + mass_range
    data_obs = ROOT.TH1D("data_obs", "data_obs", nbins, mass_min, mass_max)
    data_obs.SetDirectory(0)

    # ========================================
    # Process Signal
    # ========================================
    logging.info(f"Processing signal: {args.masspoint}")

    # Central histogram
    hist_signal_central = getHist(args.masspoint, mA, width, sigma, "Central")
    output_file.cd()
    hist_signal_central.Write()

    # Prompt systematics
    for syst_name, variations in prompt_systematics.items():
        logging.debug(f"  Processing signal systematic: {syst_name}")
        for var in variations:
            hist = getHist(args.masspoint, mA, width, sigma, var)
            output_file.cd()
            hist.Write()

            # Check systematic variation magnitude
            ratio = hist.Integral() / hist_signal_central.Integral() if hist_signal_central.Integral() > 0 else 0
            if ratio < 0.5 or ratio > 2.0:
                logging.warning(f"Large systematic variation for {args.masspoint}_{var}: ratio = {ratio:.2f}")

    logging.info(f"Signal templates created: {args.masspoint} (integral = {hist_signal_central.Integral():.4f})")

    # ========================================
    # Process Nonprompt Background
    # ========================================
    logging.info("Processing nonprompt background")

    # Central
    hist_nonprompt = getHist("nonprompt", mA, width, sigma, "Central")
    data_obs.Add(hist_nonprompt)
    output_file.cd()
    hist_nonprompt.Write()

    # Nonprompt-specific systematics (weight variations)
    hist_nonprompt_up = getHist("nonprompt", mA, width, sigma, "Nonprompt_Up")
    hist_nonprompt_down = getHist("nonprompt", mA, width, sigma, "Nonprompt_Down")
    output_file.cd()
    hist_nonprompt_up.Write()
    hist_nonprompt_down.Write()

    logging.info(f"Nonprompt templates created (integral = {hist_nonprompt.Integral():.4f})")

    # ========================================
    # Process Conversion Background
    # ========================================
    logging.info("Processing conversion background")

    # Central
    hist_conversion = getHist("conversion", mA, width, sigma, "Central")
    data_obs.Add(hist_conversion)
    output_file.cd()
    hist_conversion.Write()

    # Prompt systematics
    for syst_name, variations in prompt_systematics.items():
        logging.debug(f"  Processing conversion systematic: {syst_name}")
        for var in variations:
            hist = getHist("conversion", mA, width, sigma, var)
            output_file.cd()
            hist.Write()

    logging.info(f"Conversion templates created (integral = {hist_conversion.Integral():.4f})")

    # ========================================
    # Process Diboson Background
    # ========================================
    logging.info("Processing diboson background")

    # Central
    hist_diboson = getHist("diboson", mA, width, sigma, "Central")
    data_obs.Add(hist_diboson)
    output_file.cd()
    hist_diboson.Write()

    # Prompt systematics
    for syst_name, variations in prompt_systematics.items():
        logging.debug(f"  Processing diboson systematic: {syst_name}")
        for var in variations:
            hist = getHist("diboson", mA, width, sigma, var)
            output_file.cd()
            hist.Write()

    logging.info(f"Diboson templates created (integral = {hist_diboson.Integral():.4f})")

    # ========================================
    # Process ttX Background
    # ========================================
    logging.info("Processing ttX background")

    # Central
    hist_ttX = getHist("ttX", mA, width, sigma, "Central")
    data_obs.Add(hist_ttX)
    output_file.cd()
    hist_ttX.Write()

    # Prompt systematics
    for syst_name, variations in prompt_systematics.items():
        logging.debug(f"  Processing ttX systematic: {syst_name}")
        for var in variations:
            hist = getHist("ttX", mA, width, sigma, var)
            output_file.cd()
            hist.Write()

    logging.info(f"ttX templates created (integral = {hist_ttX.Integral():.4f})")

    # ========================================
    # Process Others Background
    # ========================================
    logging.info("Processing others background")

    # Central
    hist_others = getHist("others", mA, width, sigma, "Central")
    data_obs.Add(hist_others)
    output_file.cd()
    hist_others.Write()

    # Prompt systematics
    for syst_name, variations in prompt_systematics.items():
        logging.debug(f"  Processing others systematic: {syst_name}")
        for var in variations:
            hist = getHist("others", mA, width, sigma, var)
            output_file.cd()
            hist.Write()

    logging.info(f"Others templates created (integral = {hist_others.Integral():.4f})")

    # ========================================
    # Write data_obs
    # ========================================
    logging.info(f"Writing data_obs (sum of all backgrounds, integral = {data_obs.Integral():.4f})")
    output_file.cd()
    data_obs.Write()

    # Close output file
    output_file.Close()

    # Summary
    logging.info("=" * 60)
    logging.info("Template generation complete!")
    logging.info(f"Output file: {OUTDIR}/shapes.root")
    logging.info("=" * 60)
    logging.info("Process yields:")
    logging.info(f"  Signal ({args.masspoint}):  {hist_signal_central.Integral():>10.4f}")
    logging.info(f"  Nonprompt:                   {hist_nonprompt.Integral():>10.4f}")
    logging.info(f"  Conversion:                  {hist_conversion.Integral():>10.4f}")
    logging.info(f"  Diboson:                     {hist_diboson.Integral():>10.4f}")
    logging.info(f"  ttX:                         {hist_ttX.Integral():>10.4f}")
    logging.info(f"  Others:                      {hist_others.Integral():>10.4f}")
    logging.info(f"  Total background:            {data_obs.Integral():>10.4f}")
    logging.info(f"  S/B ratio:                   {hist_signal_central.Integral()/data_obs.Integral() if data_obs.Integral() > 0 else 0:>10.4f}")
    logging.info("=" * 60)
