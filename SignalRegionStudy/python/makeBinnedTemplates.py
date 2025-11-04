#!/usr/bin/env python3
import os
import shutil
import logging
import argparse
import json
import ROOT
import numpy as np
import pandas as pd
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

# Always read samples from Baseline directory (preprocessing is method-independent)
# But write templates to method-specific directory
BASEDIR = f"{WORKDIR}/SignalRegionStudy/samples/{args.era}/{args.channel}/{args.masspoint}/Baseline"
OUTDIR = f"{WORKDIR}/SignalRegionStudy/templates/{args.era}/{args.channel}/{args.masspoint}/Shape/{args.method}"

logging.info(f"Input directory: {BASEDIR}")
logging.info(f"Output directory: {OUTDIR}")
logging.info(f"Method: {args.method}")

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


# =============================================================================
# Helper Functions
# =============================================================================

def save_json(data, path):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def get_mass_window(mA, width, sigma):
    """Calculate mass window [mA ± 5σ_voigt]. Returns (mass_min, mass_max)."""
    mass_range = 5 * sqrt(width**2 + sigma**2)
    return mA - mass_range, mA + mass_range


def getBackgroundWeights(mA, width, sigma):
    """Calculate normalized background class weights for ParticleNet (corrects training imbalance)."""
    logging.info("Calculating background cross-section weights:")

    mass_min, mass_max = get_mass_window(mA, width, sigma)

    weights = {}
    for process in ["nonprompt", "diboson", "ttX"]:
        file_path = f"{BASEDIR}/{process}.root"
        if not os.path.exists(file_path):
            logging.warning(f"  Cannot calculate weight for {process}: file not found")
            weights[process] = 1.0 / 3.0  # Default to equal weights
            continue

        rfile = ROOT.TFile.Open(file_path, "READ")
        tree = rfile.Get("Central")
        if not tree:
            logging.warning(f"  Cannot calculate weight for {process}: Central tree not found")
            rfile.Close()
            weights[process] = 1.0 / 3.0
            continue

        # Calculate total weighted yield in mass window
        total_weight = 0.0
        for entry in range(tree.GetEntries()):
            tree.GetEntry(entry)
            if mass_min <= tree.mass <= mass_max:
                total_weight += tree.weight

        weights[process] = total_weight
        rfile.Close()
        logging.info(f"  {process}: {total_weight:.4f}")

    # Normalize to sum = 1.0
    total = sum(weights.values())
    if total > 0:
        weights = {k: v/total for k, v in weights.items()}
    else:
        logging.warning("Total background weight is zero! Using equal weights.")
        weights = {k: 1.0/3.0 for k in weights.keys()}

    logging.info("Background weights (normalized to sum=1):")
    for k, v in weights.items():
        logging.info(f"  {k}: {v:.4f}")

    save_json({
        "weights": weights,
        "yields": {k: v * total for k, v in weights.items()},
        "total_yield": float(total),
        "mass_window": [float(mass_min), float(mass_max)]
    }, f"{OUTDIR}/background_weights.json")

    return weights
def getFitResult(input_path, output_path, mA_nominal):
    """Fit A mass distribution using AmassFitter. Returns (mA, width, sigma)."""
    logging.info(f"Fitting A mass with nominal mA = {mA_nominal} GeV")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    fitter = ROOT.AmassFitter(input_path, output_path)
    fitter.fitMass(mA_nominal, mA_nominal - 20., mA_nominal + 20.)
    fitter.saveCanvas(f"{OUTDIR}/signal_fit.png")

    mA = fitter.getRooMA().getVal()
    width = fitter.getRooWidth().getVal()
    sigma = fitter.getRooSigma().getVal()
    fitter.Close()

    mass_min, mass_max = get_mass_window(mA, width, sigma)

    save_json({
        "mass": float(mA),
        "width": float(width),
        "sigma": float(sigma),
        "mass_min": float(mass_min),
        "mass_max": float(mass_max),
        "mass_range": float(mass_max - mass_min),
        "nbins": 15
    }, f"{OUTDIR}/signal_fit.json")

    logging.info(f"Fit: mA={mA:.2f}, Γ={width:.3f}, σ={sigma:.3f} GeV")
    return mA, width, sigma


def loadFitResult(fit_result_path):
    """Load fit parameters from SR1E2Mu for SR3Mu channel. Returns (mA, width, sigma)."""
    if not os.path.exists(fit_result_path):
        raise FileNotFoundError(f"Fit result file not found: {fit_result_path}")

    fit_file = ROOT.TFile.Open(fit_result_path, "READ")
    fit_result = fit_file.Get("fitresult_model_data") or fit_file.Get("fit_result")
    if not fit_result:
        fit_file.Close()
        raise RuntimeError("RooFitResult not found in fit_result.root")

    params = fit_result.floatParsFinal()
    mA = params.find("mA").getVal()
    width = params.find("width").getVal()
    sigma = params.find("sigma").getVal()
    fit_file.Close()

    mass_min, mass_max = get_mass_window(mA, width, sigma)

    save_json({
        "mass": float(mA),
        "width": float(width),
        "sigma": float(sigma),
        "mass_min": float(mass_min),
        "mass_max": float(mass_max),
        "mass_range": float(mass_max - mass_min),
        "nbins": 15,
        "source": "SR1E2Mu"
    }, f"{OUTDIR}/signal_fit.json")

    logging.info(f"Loaded: mA={mA:.2f}, Γ={width:.3f}, σ={sigma:.3f} GeV")
    return mA, width, sigma


def calculateFixedBins(mA, width, sigma):
    """Generate 15 uniform bins in [mA ± 5σ_voigt]."""
    voigt_width = sqrt(width**2 + sigma**2)
    mass_min, mass_max = get_mass_window(mA, width, sigma)
    nbins = 15
    bin_edges = np.linspace(mass_min, mass_max, nbins + 1)

    logging.info(f"Uniform binning: {nbins} bins, σ_voigt={voigt_width:.3f} GeV, range=[{mass_min:.2f}, {mass_max:.2f}] GeV")
    return bin_edges


def validateBackgroundStatistics(bin_edges, mA, width, sigma, basedir,
                                  method, masspoint, best_threshold=-999.,
                                  bg_weights=None, min_total_events=1):
    """
    Validate statistical quality of each background process.

    Simplified version: Only checks if total event yield >= min_total_events.

    Args:
        bin_edges: Fixed bin edges (15 bins in mass space)
        mA, width, sigma: Signal parameters
        basedir: Sample directory
        method, masspoint: Template method
        best_threshold, bg_weights: ParticleNet parameters
        min_total_events: Minimum total events to keep separate (default: 1)

    Returns:
        Dictionary with validation results per process
    """
    logging.info("Validating background statistics...")

    voigt_width = sqrt(width**2 + sigma**2)
    mass_min = mA - 5 * voigt_width
    mass_max = mA + 5 * voigt_width
    nbins = len(bin_edges) - 1

    results = {}

    for process in ["conversion", "diboson", "ttX"]:
        logging.info(f"  Validating {process}...")

        # Fill histogram for this process
        file_path = f"{basedir}/{process}.root"
        if not os.path.exists(file_path):
            logging.warning(f"    File not found, will merge to others")
            results[process] = {
                "total_events": 0,
                "decision": "merge",
                "reason": "file not found"
            }
            continue

        # Create histogram using same logic as getHist
        rdf = ROOT.RDataFrame("Central", file_path)
        rdf = rdf.Filter(f"mass >= {mass_min} && mass <= {mass_max}")

        # Apply ParticleNet cut if applicable
        if best_threshold > -999.:
            # Check if ParticleNet scores exist
            test_file = ROOT.TFile.Open(file_path, "READ")
            tree = test_file.Get("Central")
            if tree:
                branches = [b.GetName() for b in tree.GetListOfBranches()]
                test_file.Close()

                score_sig = f"score_{masspoint}_signal"
                if score_sig in branches:
                    score_nonprompt = f"score_{masspoint}_nonprompt"
                    score_diboson_score = f"score_{masspoint}_diboson"
                    score_ttZ = f"score_{masspoint}_ttZ"

                    # Define score formula
                    if bg_weights:
                        w1 = bg_weights.get("nonprompt", 1.0)
                        w2 = bg_weights.get("diboson", 1.0)
                        w3 = bg_weights.get("ttX", 1.0)
                        score_formula = f"({score_sig}) / ({score_sig} + {w1}*{score_nonprompt} + {w2}*{score_diboson_score} + {w3}*{score_ttZ})"
                    else:
                        score_formula = f"({score_sig}) / ({score_sig} + {score_nonprompt} + {score_diboson_score} + {score_ttZ})"

                    rdf = rdf.Define("score_PN", score_formula)
                    rdf = rdf.Filter(f"score_PN >= {best_threshold}")
            else:
                test_file.Close()

        # Get total weighted events - now using mass directly
        bin_edges_vector = ROOT.std.vector['double'](bin_edges)
        hist = rdf.Histo1D(("temp", "", nbins, bin_edges_vector.data()),
                          "mass", "weight")
        hist_result = hist.GetValue()

        total_events = hist_result.Integral()

        # Decision logic: keep if total_events >= min_total_events
        if total_events < min_total_events:
            decision = "merge"
            reason = f"total events ({total_events:.1f}) < {min_total_events}"
        else:
            decision = "keep"
            reason = "passes statistical requirements"

        results[process] = {
            "total_events": total_events,
            "decision": decision,
            "reason": reason
        }

        logging.info(f"    Total events: {total_events:.1f}")
        logging.info(f"    Decision: {decision.upper()} ({reason})")

    return results


def determineProcessList(validation_results):
    """
    Determine final process list based on validation.

    Args:
        validation_results: Output from validateBackgroundStatistics()

    Returns:
        Dictionary with separate_processes and merged_to_others lists
    """
    separate_processes = ["nonprompt"]  # Always separate
    merged_to_others = []

    for process in ["conversion", "diboson", "ttX"]:
        if validation_results[process]["decision"] == "keep":
            separate_processes.append(process)
            logging.info(f"  {process}: keeping as separate process")
        else:
            merged_to_others.append(process)
            logging.info(f"  {process}: merging to others ({validation_results[process]['reason']})")

    return {
        "separate_processes": separate_processes,
        "merged_to_others": merged_to_others,
        "validation_results": validation_results
    }


def getHistMerged(process_list, bin_edges, mA, width, sigma, syst="Central",
                  threshold=-999., bg_weights=None):
    """
    Create merged histogram from multiple processes.

    Args:
        process_list: List of process names to merge
        bin_edges, mA, width, sigma, syst, threshold, bg_weights: Same as getHist

    Returns:
        TH1D histogram (sum of all processes)
    """
    if len(process_list) == 0:
        raise ValueError("process_list cannot be empty")

    # Get first histogram
    hist_merged = getHist(process_list[0], bin_edges, mA, width, sigma,
                         syst, threshold, bg_weights)

    # Add remaining processes
    for process in process_list[1:]:
        hist_add = getHist(process, bin_edges, mA, width, sigma,
                          syst, threshold, bg_weights)
        hist_merged.Add(hist_add)

    return hist_merged


# Helper function: Ensure positive integral
def ensurePositiveIntegral(hist, min_integral=1e-10):
    """
    Ensure histogram has positive integral for normalization.

    This function performs two checks:
    1. Ensures all bins are non-negative (sets negative bins to zero)
    2. Ensures total integral is positive (adds minimal value if needed)

    This prevents "Bogus norm" errors in Combine that occur when:
    - Individual bins are negative
    - Systematic variations have negative normalization ratios

    Args:
        hist: TH1D histogram
        min_integral: Minimum integral value to set

    Returns:
        True if histogram was modified, False otherwise
    """
    modified = False

    # Step 1: Fix negative bins
    for i in range(1, hist.GetNbinsX() + 1):
        content = hist.GetBinContent(i)
        if content < 0:
            logging.warning(f"  Histogram {hist.GetName()}, bin {i}: negative content {content:.4e}, setting to 0")
            hist.SetBinContent(i, 0.0)
            hist.SetBinError(i, 0.0)
            modified = True

    # Step 2: Ensure positive total integral
    integral = hist.Integral()
    if integral <= 0:
        logging.warning(f"  Histogram {hist.GetName()} has non-positive integral: {integral:.4e}")

        # Find the central bin
        central_bin = hist.GetNbinsX() // 2 + 1

        # Set a small positive value in the central bin
        logging.warning(f"  Setting bin {central_bin} to {min_integral} to ensure positive normalization")
        hist.SetBinContent(central_bin, min_integral)
        hist.SetBinError(central_bin, min_integral)

        modified = True

    return modified


# Helper function: Load dataset for ParticleNet optimization
def loadDataset(process, masspoint, mA, width, sigma, bg_weights=None):
    """
    Load events with ParticleNet scores from preprocessed samples.

    Args:
        process: Process name
        masspoint: Signal mass point
        mA: Fitted A mass
        width: Breit-Wigner width
        sigma: Gaussian resolution
        bg_weights: Dictionary of background weights (nonprompt, diboson, ttX).
                   If provided, uses weighted likelihood ratio. If None, uses unweighted.

    Returns:
        Tuple of (scores, weights, labels)
        - scores: ParticleNet likelihood ratios (cross-section weighted if bg_weights provided)
        - weights: Event weights
        - labels: 1 for signal, 0 for background
    """
    file_path = f"{BASEDIR}/{process}.root"

    if not os.path.exists(file_path):
        logging.warning(f"Sample file not found for optimization: {file_path}")
        return np.array([]), np.array([]), np.array([])

    # Open file and get tree
    rfile = ROOT.TFile.Open(file_path, "READ")
    tree = rfile.Get("Central")

    if not tree:
        logging.warning(f"Central tree not found in {file_path}")
        rfile.Close()
        return np.array([]), np.array([]), np.array([])

    # Apply mass window cut
    mass_range = 5 * sqrt(width**2 + sigma**2)
    mass_min = mA - mass_range
    mass_max = mA + mass_range

    # Branch names
    score_sig = f"score_{masspoint}_signal"
    score_nonprompt = f"score_{masspoint}_nonprompt"
    score_diboson = f"score_{masspoint}_diboson"
    score_ttZ = f"score_{masspoint}_ttZ"

    # Check if ParticleNet scores exist
    branches = [b.GetName() for b in tree.GetListOfBranches()]
    if score_sig not in branches:
        logging.warning(f"ParticleNet scores not found in {file_path}. This is expected for untrained mass points.")
        rfile.Close()
        return np.array([]), np.array([]), np.array([])

    # Load data
    scores_list = []
    weights_list = []
    labels_list = []

    for entry in range(tree.GetEntries()):
        tree.GetEntry(entry)

        mass = tree.mass
        s0 = getattr(tree, score_sig)
        s1 = getattr(tree, score_nonprompt)
        s2 = getattr(tree, score_diboson)
        s3 = getattr(tree, score_ttZ)
        weight = tree.weight

        # Apply mass window cut
        if not (mass_min <= mass <= mass_max):
            continue

        # Calculate ParticleNet likelihood ratio
        if bg_weights:
            # Use cross-section weighted likelihood ratio
            w1 = bg_weights.get("nonprompt", 1.0)
            w2 = bg_weights.get("diboson", 1.0)
            w3 = bg_weights.get("ttX", 1.0)
            score_denom = s0 + w1*s1 + w2*s2 + w3*s3
        else:
            # Use unweighted likelihood ratio (original method)
            score_denom = s0 + s1 + s2 + s3

        if score_denom > 0:
            score_PN = s0 / score_denom
        else:
            score_PN = 0.0

        scores_list.append(score_PN)
        weights_list.append(weight)
        labels_list.append(1 if process == masspoint else 0)

    rfile.Close()

    return np.array(scores_list), np.array(weights_list), np.array(labels_list)


# Helper function: Evaluate sensitivity
def evalSensitivity(y_true, y_pred, weights, threshold=0.):
    """
    Calculate significance Z using Asimov formula.

    Args:
        y_true: True labels (1=signal, 0=background)
        y_pred: Predicted scores
        weights: Event weights
        threshold: Score threshold

    Returns:
        Significance Z
    """
    signal_mask = (y_true == 1) & (y_pred > threshold)
    background_mask = (y_true == 0) & (y_pred > threshold)

    S = np.sum(weights[signal_mask])
    B = np.sum(weights[background_mask])

    if B <= 0:
        return 0.0

    # Asimov significance: Z = sqrt(2((S+B)ln(1+S/B)-S))
    return np.sqrt(2 * ((S + B) * np.log(1 + S / B) - S))


# Helper function: Optimize threshold
def getOptimizedThreshold(scores_sig, weights_sig, scores_bkg, weights_bkg):
    """
    Find optimal ParticleNet score threshold to maximize sensitivity.

    Args:
        scores_sig: Signal scores
        weights_sig: Signal weights
        scores_bkg: Background scores
        weights_bkg: Background weights

    Returns:
        Tuple of (best_threshold, initial_sensitivity, max_sensitivity)
    """
    # Combine signal and background
    y_pred = np.concatenate([scores_sig, scores_bkg])
    y_true = np.concatenate([np.ones(len(scores_sig)), np.zeros(len(scores_bkg))])
    weights = np.concatenate([weights_sig, weights_bkg])

    # Scan thresholds
    thresholds = np.linspace(0, 1, 101)
    sensitivities = [evalSensitivity(y_true, y_pred, weights, threshold) for threshold in thresholds]

    best_idx = np.argmax(sensitivities)
    best_threshold = thresholds[best_idx]
    initial_sensitivity = sensitivities[0]  # No cut
    max_sensitivity = sensitivities[best_idx]

    logging.info(f"Threshold optimization:")
    logging.info(f"  Best threshold: {best_threshold:.3f}")
    logging.info(f"  Initial sensitivity (no cut): {initial_sensitivity:.3f}")
    logging.info(f"  Max sensitivity: {max_sensitivity:.3f}")
    logging.info(f"  Improvement: {(max_sensitivity/initial_sensitivity-1)*100:.2f}%")

    return best_threshold, initial_sensitivity, max_sensitivity


# Helper function: Plot score distribution
def plotScoreDistribution(scores_sig, weights_sig, scores_bkg, weights_bkg, best_threshold, improvement):
    """
    Create diagnostic plot showing ParticleNet score distribution.

    Args:
        scores_sig: Signal scores
        weights_sig: Signal weights
        scores_bkg: Background scores
        weights_bkg: Background weights
        best_threshold: Optimized threshold
        improvement: Sensitivity improvement factor
    """
    # Create histograms
    h_sig = ROOT.TH1D("signal", "", 100, 0., 1.)
    h_bkg = ROOT.TH1D("background", "", 100, 0., 1.)

    for score, weight in zip(scores_sig, weights_sig):
        h_sig.Fill(score, weight)

    for score, weight in zip(scores_bkg, weights_bkg):
        h_bkg.Fill(score, weight)

    # Normalize
    if h_sig.Integral() > 0:
        h_sig.Scale(1. / h_sig.Integral())
    if h_bkg.Integral() > 0:
        h_bkg.Scale(1. / h_bkg.Integral())

    h_sig.SetStats(0)
    h_bkg.SetStats(0)

    # Style
    h_sig.SetLineColor(ROOT.kRed)
    h_sig.SetLineWidth(2)
    h_bkg.SetLineColor(ROOT.kGray + 2)
    h_bkg.SetLineWidth(2)
    h_sig.GetXaxis().SetTitle("ParticleNet Score")
    h_sig.GetYaxis().SetTitle("A.U.")
    h_sig.GetYaxis().SetRangeUser(0., 2. * max(h_sig.GetMaximum(), h_bkg.GetMaximum()))

    # Threshold line
    line = ROOT.TLine(best_threshold, 0., best_threshold, 0.5 * h_sig.GetMaximum())
    line.SetLineColor(ROOT.kBlack)
    line.SetLineWidth(3)
    line.SetLineStyle(2)

    # Legend
    legend = ROOT.TLegend(0.65, 0.7, 0.88, 0.88)
    legend.AddEntry(h_sig, "Signal", "l")
    legend.AddEntry(h_bkg, "Background", "l")
    legend.AddEntry(line, f"Threshold: {best_threshold:.3f}", "l")

    # Canvas
    canvas = ROOT.TCanvas("c_score", "c_score", 800, 600)
    canvas.cd()
    h_sig.Draw("HIST")
    h_bkg.Draw("HIST SAME")
    line.Draw("SAME")
    legend.Draw()

    # Labels
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(61)
    latex.SetTextSize(0.05)
    latex.DrawLatex(0.12, 0.92, "CMS")

    latex.SetTextFont(52)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.22, 0.92, "Preliminary")

    latex.SetTextFont(42)
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.12, 0.83, f"Channel: {args.channel}")
    latex.DrawLatex(0.12, 0.78, f"Mass point: {args.masspoint}")
    latex.DrawLatex(0.12, 0.73, f"Improvement: {improvement*100:.2f}%")

    # Save
    output_path = f"{OUTDIR}/score_optimization.png"
    canvas.SaveAs(output_path)
    logging.info(f"Score optimization plot saved: {output_path}")


# Helper function: Create histogram
def getHist(process, bin_edges, mA, width, sigma, syst="Central", threshold=-999., bg_weights=None):
    """
    Create histogram from preprocessed tree using RDataFrame.
    Uses 'mass' branch as the discrimination variable.

    Args:
        process: Process name (e.g., "MHc130_MA90", "nonprompt", "diboson")
        bin_edges: Numpy array of bin edges in mass space (GeV)
        mA: Fitted A mass
        width: Breit-Wigner width
        sigma: Gaussian resolution
        syst: Systematic variation name (e.g., "Central", "MuonIDSF_Up")
        threshold: ParticleNet score threshold for event selection (default: -999, no cut)
        bg_weights: Dictionary of background weights (nonprompt, diboson, ttX).
                   If provided, uses weighted likelihood ratio. If None, uses unweighted.

    Returns:
        TH1D histogram with mass as the x-axis variable
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

    # Use provided bin edges (in mass space)
    nbins = len(bin_edges) - 1

    # Mass window: [mA - 5σ, mA + 5σ] where σ is Voigt width
    voigt_width = sqrt(width**2 + sigma**2)
    mass_min = mA - 5 * voigt_width
    mass_max = mA + 5 * voigt_width

    # Create ROOT vector for histogram binning
    bin_edges_vector = ROOT.std.vector['double'](bin_edges)

    # Check if tree exists and get branch list
    test_file = ROOT.TFile.Open(file_path)
    tree = test_file.Get(tree_name)
    if not tree:
        test_file.Close()
        raise RuntimeError(f"Tree '{tree_name}' not found in {file_path}")

    # Get list of branches
    branches = [b.GetName() for b in tree.GetListOfBranches()]
    test_file.Close()

    # Create RDataFrame
    rdf = ROOT.RDataFrame(tree_name, file_path)

    # Apply mass window cut (always applied for all methods)
    rdf = rdf.Filter(f"mass >= {mass_min} && mass <= {mass_max}")
    logging.debug(f"  Applied mass window cut: [{mass_min:.2f}, {mass_max:.2f}] GeV")

    # Apply ParticleNet score cut if threshold is provided
    if threshold > -999.:
        # Define ParticleNet likelihood ratio
        score_sig = f"score_{args.masspoint}_signal"
        score_nonprompt = f"score_{args.masspoint}_nonprompt"
        score_diboson = f"score_{args.masspoint}_diboson"
        score_ttZ = f"score_{args.masspoint}_ttZ"

        # Check if scores exist (only for trained mass points)
        if score_sig in branches:
            # Define score formula (weighted or unweighted)
            if bg_weights:
                # Use cross-section weighted likelihood ratio
                w1 = bg_weights.get("nonprompt", 1.0)
                w2 = bg_weights.get("diboson", 1.0)
                w3 = bg_weights.get("ttX", 1.0)
                score_formula = f"({score_sig}) / ({score_sig} + {w1}*{score_nonprompt} + {w2}*{score_diboson} + {w3}*{score_ttZ})"
            else:
                # Use unweighted likelihood ratio (original method)
                score_formula = f"({score_sig}) / ({score_sig} + {score_nonprompt} + {score_diboson} + {score_ttZ})"

            rdf = rdf.Define("score_PN", score_formula)
            rdf = rdf.Filter(f"score_PN >= {threshold}")
            logging.debug(f"  Applied ParticleNet cut: score_PN >= {threshold:.3f}")
        else:
            logging.warning(f"  ParticleNet scores not found in {file_path}, skipping cut")

    # Fill histogram using mass as the variable
    hist = rdf.Histo1D((hist_name, "", nbins, bin_edges_vector.data()), "mass", "weight")

    # Detach from file
    hist_result = hist.GetValue()
    hist_result.SetDirectory(0)

    # Note: Do NOT call ensurePositiveIntegral here!
    # Individual histograms may have negative bins due to weights/systematics,
    # but we fix them AFTER merging (for "others") or before writing (for all processes)

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

    # ========================================
    # Calculate Fixed Binning
    # ========================================
    logging.info("=" * 60)
    logging.info("Calculating uniform mass binning...")
    logging.info("=" * 60)

    bin_edges = calculateFixedBins(mA, width, sigma)

    # Save binning to JSON
    voigt_width = sqrt(width**2 + sigma**2)
    binning_json = {
        "nbins": len(bin_edges) - 1,
        "bin_edges": bin_edges.tolist(),
        "method": "UniformMass",
        "voigt_width": float(voigt_width),
        "mass_min": float(mA - 5 * voigt_width),
        "mass_max": float(mA + 5 * voigt_width),
        "bin_width": float((bin_edges[-1] - bin_edges[0]) / (len(bin_edges) - 1))
    }
    json_path = f"{OUTDIR}/binning.json"
    with open(json_path, 'w') as f:
        json.dump(binning_json, f, indent=2)
    logging.info(f"Binning saved to: {json_path}")

    # Update signal_fit.json with nbins
    signal_fit_json_path = f"{OUTDIR}/signal_fit.json"
    with open(signal_fit_json_path, 'r') as f:
        fit_params = json.load(f)
    fit_params["nbins"] = len(bin_edges) - 1
    with open(signal_fit_json_path, 'w') as f:
        json.dump(fit_params, f, indent=2)
    logging.info(f"Updated signal_fit.json with nbins = {len(bin_edges) - 1}")

    logging.info("=" * 60)

    # ========================================
    # Calculate Background Weights (if applicable)
    # ========================================
    bg_weights = None
    if args.method == "ParticleNet":
        bg_weights = getBackgroundWeights(mA, width, sigma)

    # ========================================
    # ParticleNet Optimization (if applicable)
    # ========================================
    best_threshold = -999.

    if args.method == "ParticleNet":
        logging.info("=" * 60)
        logging.info("ParticleNet score optimization")
        logging.info("=" * 60)

        # Load signal dataset
        scores_sig, weights_sig, labels_sig = loadDataset(args.masspoint, args.masspoint, mA, width, sigma, bg_weights)

        if len(scores_sig) == 0:
            logging.warning("ParticleNet scores not found! This is expected for untrained mass points (mA < 80 or mA > 100).")
            logging.warning("Proceeding without ParticleNet cuts (equivalent to Baseline method).")
        else:
            # Load background datasets
            scores_bkg_list = []
            weights_bkg_list = []

            for process in ["nonprompt", "conversion", "diboson", "ttX", "others"]:
                scores, weights, labels = loadDataset(process, args.masspoint, mA, width, sigma, bg_weights)
                if len(scores) > 0:
                    scores_bkg_list.append(scores)
                    weights_bkg_list.append(weights)

            if len(scores_bkg_list) == 0:
                logging.warning("No background samples loaded for optimization!")
                logging.warning("Proceeding without ParticleNet cuts.")
            else:
                # Merge all backgrounds
                scores_bkg = np.concatenate(scores_bkg_list)
                weights_bkg = np.concatenate(weights_bkg_list)

                # Optimize threshold
                best_threshold, initial_sensitivity, max_sensitivity = getOptimizedThreshold(
                    scores_sig, weights_sig, scores_bkg, weights_bkg
                )

                # Save threshold to CSV
                improvement = (max_sensitivity / initial_sensitivity - 1) if initial_sensitivity > 0 else 0
                threshold_df = pd.DataFrame({
                    "masspoint": [args.masspoint],
                    "threshold": [best_threshold],
                    "initial_sensitivity": [initial_sensitivity],
                    "max_sensitivity": [max_sensitivity],
                    "improvement": [improvement]
                })
                threshold_df.to_csv(f"{OUTDIR}/threshold.csv", index=False)
                logging.info(f"Threshold saved to: {OUTDIR}/threshold.csv")

                # Create diagnostic plot
                plotScoreDistribution(scores_sig, weights_sig, scores_bkg, weights_bkg, best_threshold, improvement)

        logging.info("=" * 60)

    # ========================================
    # Validate Background Statistics
    # ========================================
    logging.info("=" * 60)
    logging.info("Validating background statistics...")
    logging.info("=" * 60)

    validation_results = validateBackgroundStatistics(
        bin_edges, mA, width, sigma, BASEDIR,
        args.method, args.masspoint, best_threshold, bg_weights,
        min_total_events=1
    )

    # Save validation results to JSON
    validation_json_path = f"{OUTDIR}/background_validation.json"
    with open(validation_json_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        validation_json = {}
        for process, result in validation_results.items():
            validation_json[process] = {
                "total_events": float(result["total_events"]),
                "decision": result["decision"],
                "reason": result["reason"]
            }
        json.dump(validation_json, f, indent=2)
    logging.info(f"Validation results saved to: {validation_json_path}")

    # Determine final process list
    logging.info("Determining final process list...")
    process_config = determineProcessList(validation_results)
    separate_processes = process_config["separate_processes"]
    merged_to_others = process_config["merged_to_others"]

    # Save process list to JSON
    process_list_json = {
        "separate_processes": separate_processes,
        "merged_to_others": merged_to_others,
        "description": "Processes kept separate vs merged into 'others' based on statistical validation"
    }
    process_list_json_path = f"{OUTDIR}/process_list.json"
    with open(process_list_json_path, 'w') as f:
        json.dump(process_list_json, f, indent=2)
    logging.info(f"Process list saved to: {process_list_json_path}")

    logging.info("Final process configuration:")
    logging.info(f"  Separate processes: {separate_processes}")
    logging.info(f"  Merged to others: {merged_to_others}")
    logging.info("=" * 60)

    # Create output ROOT file
    output_file = ROOT.TFile(f"{OUTDIR}/shapes.root", "RECREATE")

    # Initialize data_obs histogram (will be sum of all backgrounds) with variable-width bins
    nbins = len(bin_edges) - 1
    bin_edges_vector = ROOT.std.vector['double'](bin_edges)
    data_obs = ROOT.TH1D("data_obs", "data_obs", nbins, bin_edges_vector.data())
    data_obs.SetDirectory(0)

    # ========================================
    # Process Signal
    # ========================================
    logging.info(f"Processing signal: {args.masspoint}")

    # Central histogram
    hist_signal_central = getHist(args.masspoint, bin_edges, mA, width, sigma, "Central", best_threshold, bg_weights)

    # Ensure positive integral before writing
    ensurePositiveIntegral(hist_signal_central)

    output_file.cd()
    hist_signal_central.Write()

    # Prompt systematics
    for syst_name, variations in prompt_systematics.items():
        logging.debug(f"  Processing signal systematic: {syst_name}")
        for var in variations:
            hist = getHist(args.masspoint, bin_edges, mA, width, sigma, var, best_threshold, bg_weights)

            # Ensure positive integral for systematic variations
            ensurePositiveIntegral(hist)

            output_file.cd()
            hist.Write()

            # Check systematic variation magnitude
            ratio = hist.Integral() / hist_signal_central.Integral() if hist_signal_central.Integral() > 0 else 0
            if ratio < 0.5 or ratio > 2.0:
                logging.warning(f"Large systematic variation for {args.masspoint}_{var}: ratio = {ratio:.2f}")

    logging.info(f"Signal templates created: {args.masspoint} (integral = {hist_signal_central.Integral():.4f})")

    # ========================================
    # Process Background Templates (Dynamic)
    # ========================================
    # Store histograms for summary
    background_hists = {}

    # Process separate backgrounds (always includes "nonprompt", may include conversion/diboson/ttX)
    for process in separate_processes:
        logging.info(f"Processing {process} background (separate template)")

        # Central histogram
        hist_central = getHist(process, bin_edges, mA, width, sigma, "Central", best_threshold, bg_weights)

        # Ensure positive integral before writing
        ensurePositiveIntegral(hist_central)

        data_obs.Add(hist_central)
        output_file.cd()
        hist_central.Write()
        background_hists[process] = hist_central

        # Systematics
        if process == "nonprompt":
            # Nonprompt-specific systematics (weight variations)
            hist_up = getHist(process, bin_edges, mA, width, sigma, "Nonprompt_Up", best_threshold, bg_weights)
            hist_down = getHist(process, bin_edges, mA, width, sigma, "Nonprompt_Down", best_threshold, bg_weights)

            # Ensure positive integral for systematic variations
            ensurePositiveIntegral(hist_up)
            ensurePositiveIntegral(hist_down)

            output_file.cd()
            hist_up.Write()
            hist_down.Write()
            logging.info(f"  Nonprompt templates created (integral = {hist_central.Integral():.4f})")
        else:
            # Prompt systematics for conversion, diboson, ttX
            for syst_name, variations in prompt_systematics.items():
                logging.debug(f"  Processing {process} systematic: {syst_name}")
                for var in variations:
                    hist = getHist(process, bin_edges, mA, width, sigma, var, best_threshold, bg_weights)

                    # Ensure positive integral for systematic variations
                    ensurePositiveIntegral(hist)

                    output_file.cd()
                    hist.Write()
            logging.info(f"  {process} templates created (integral = {hist_central.Integral():.4f})")

    # Process "others" background (merge poor-statistics processes + original "others")
    logging.info("Processing others background (merged template)")

    # Build list of processes to merge: "others" first to ensure correct histogram naming
    # (getHistMerged uses the first process name for the output histogram)
    others_process_list = ["others"] + merged_to_others
    logging.info(f"  Merging processes: {others_process_list}")

    # Central histogram
    hist_others = getHistMerged(others_process_list, bin_edges, mA, width, sigma,
                                "Central", best_threshold, bg_weights)

    # Ensure positive integral after merging (critical for merged histograms!)
    ensurePositiveIntegral(hist_others)

    data_obs.Add(hist_others)
    output_file.cd()
    hist_others.Write()
    background_hists["others"] = hist_others

    # Prompt systematics
    for syst_name, variations in prompt_systematics.items():
        logging.debug(f"  Processing others systematic: {syst_name}")
        for var in variations:
            hist = getHistMerged(others_process_list, bin_edges, mA, width, sigma,
                                var, best_threshold, bg_weights)

            # Ensure positive integral for merged systematic variations
            # This is critical because merging can result in negative integrals
            # even when individual histograms are positive
            ensurePositiveIntegral(hist)

            output_file.cd()
            hist.Write()

    logging.info(f"  Others templates created (integral = {hist_others.Integral():.4f})")

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

    # Print separate background processes
    for process in separate_processes:
        logging.info(f"  {process.capitalize():23s} {background_hists[process].Integral():>10.4f}")

    # Print merged "others" background
    logging.info(f"  {'Others':23s} {background_hists['others'].Integral():>10.4f}")
    if merged_to_others:
        logging.info(f"    (merged: {', '.join(merged_to_others)})")

    logging.info(f"  Total background:            {data_obs.Integral():>10.4f}")
    logging.info(f"  S/B ratio:                   {hist_signal_central.Integral()/data_obs.Integral() if data_obs.Integral() > 0 else 0:>10.4f}")
    logging.info("=" * 60)
