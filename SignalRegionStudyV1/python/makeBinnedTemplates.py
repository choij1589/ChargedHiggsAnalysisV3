#!/usr/bin/env python3
"""
Generate binned histogram templates for HiggsCombine from preprocessed ROOT files.

This script creates shape-based templates with systematic variations for limit setting.
Supports two binning methods:
- uniform (default): 15 uniform bins in [mA - 5*sigma_voigt, mA + 5*sigma_voigt]
- extended: Same 15 uniform bins as uniform, plus extended bins at [-10, -7, -5] and [+5, +7, +10] * sigma_voigt

Features:
- A mass fitting to extract signal parameters
- PDF and QCDScale envelope creation from multi-variation systematics
- Auto-merging of backgrounds with insufficient statistics

Usage:
    python makeBinnedTemplates.py --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --method Baseline
    python makeBinnedTemplates.py --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --method Baseline --binning extended
"""
import os
import re
import shutil
import logging
import argparse
import json
import ROOT
import numpy as np
from math import sqrt


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate binned histogram templates for HiggsCombine")
    parser.add_argument("--era", required=True, type=str, help="Data-taking period (2016preVFP, 2017, 2018, 2022, etc.)")
    parser.add_argument("--channel", required=True, type=str, help="Analysis channel (SR1E2Mu, SR3Mu)")
    parser.add_argument("--masspoint", required=True, type=str, help="Signal mass point (e.g., MHc130_MA90)")
    parser.add_argument("--method", required=True, type=str, help="Template method (Baseline, ParticleNet, etc.)")
    parser.add_argument("--binning", default="uniform", choices=["uniform", "extended"],
                        help="Binning method: 'uniform' (15 bins, default) or 'extended' (15 bins + tails)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def save_json(data, path):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_config(workdir, era, channel):
    """Load systematics and sample group configurations."""
    # Load systematics config
    config_path = f"{workdir}/SignalRegionStudyV1/configs/systematics.{era}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Systematics config not found: {config_path}")

    with open(config_path) as f:
        json_systematics = json.load(f)

    if channel not in json_systematics:
        raise ValueError(f"Channel '{channel}' not found in {config_path}")

    # Load sample groups config
    samplegroups_path = f"{workdir}/SignalRegionStudyV1/configs/samplegroups.json"
    if not os.path.exists(samplegroups_path):
        raise FileNotFoundError(f"Sample groups config not found: {samplegroups_path}")

    with open(samplegroups_path) as f:
        json_samplegroups = json.load(f)

    if era not in json_samplegroups:
        raise ValueError(f"Era '{era}' not found in {samplegroups_path}")
    if channel not in json_samplegroups[era]:
        raise ValueError(f"Channel '{channel}' not found for era '{era}'")

    return {
        'systematics': json_systematics[channel],
        'samples': json_samplegroups[era][channel],
        'aliases': json_samplegroups.get("aliases", {})
    }


def parse_variations(variation_spec):
    """
    Parse variation specification strings from config.

    Supports:
    - ["Var_Up", "Var_Down"] - simple list
    - ["Scale_{0..8}//5,7"] - range with exclusions
    - ["pdf_{00..99}"] - range with zero-padded numbers
    """
    if not isinstance(variation_spec, list):
        return []

    if len(variation_spec) == 1 and '{' in variation_spec[0]:
        return _expand_range_pattern(variation_spec[0])
    return variation_spec


def _expand_range_pattern(pattern):
    """Expand range pattern like 'Scale_{0..8}//5,7' or 'pdf_{00..99}'."""
    exclusions = set()
    if '//' in pattern:
        pattern, excl_str = pattern.split('//')
        exclusions = set(int(x) for x in excl_str.split(','))

    match = re.search(r'\{(\d+)\.\.(\d+)\}', pattern)
    if not match:
        return [pattern]

    start, end = int(match.group(1)), int(match.group(2))
    start_str = match.group(1)
    pad_width = len(start_str) if start_str.startswith('0') and len(start_str) > 1 else 0

    prefix, suffix = pattern[:match.start()], pattern[match.end():]
    return [
        f"{prefix}{str(i).zfill(pad_width) if pad_width else str(i)}{suffix}"
        for i in range(start, end + 1) if i not in exclusions
    ]


def categorize_systematics(config):
    """
    Categorize systematics from config into processing groups.

    Returns dict with keys:
    - preprocessed_shape: list of (syst_name, [variations], group)
    - valued_shape: list of (syst_name, value, group)
    - multi_variation: list of (syst_name, [variations], group) - for PDF/Scale
    - valued_lnN: list of (syst_name, value, group)
    """
    result = {'preprocessed_shape': [], 'valued_shape': [], 'multi_variation': [], 'valued_lnN': []}

    for syst_name, syst_config in config.items():
        source = syst_config.get('source')
        syst_type = syst_config.get('type')
        group = syst_config.get('group', [])

        if source == 'preprocessed' and syst_type == 'shape':
            variations = parse_variations(syst_config.get('variations', []))
            if len(variations) > 2:
                result['multi_variation'].append((syst_name, variations, group))
            elif len(variations) == 2:
                result['preprocessed_shape'].append((syst_name, variations, group))
            else:
                logging.warning(f"Unexpected variation count for {syst_name}: {variations}")

        elif source == 'valued' and syst_type == 'shape':
            result['valued_shape'].append((syst_name, syst_config.get('value'), group))

        elif source == 'valued' and syst_type == 'lnN':
            result['valued_lnN'].append((syst_name, syst_config.get('value'), group))

    return result


def get_output_tree_name(syst_name, variation):
    """Get output tree name from systematic name and variation."""
    if variation.endswith("_Up") or variation.endswith("Up"):
        return f"{syst_name}_Up"
    elif variation.endswith("_Down") or variation.endswith("Down"):
        return f"{syst_name}_Down"
    return variation


# =============================================================================
# Binning Functions
# =============================================================================

def get_mass_window(mA, width, sigma, binning="uniform"):
    """Calculate mass window based on binning type. Returns (mass_min, mass_max)."""
    voigt_width = sqrt(width**2 + sigma**2)
    if binning == "extended":
        return mA - 10 * voigt_width, mA + 10 * voigt_width
    else:  # uniform (default)
        return mA - 5 * voigt_width, mA + 5 * voigt_width


def calculateFixedBins(mA, width, sigma):
    """Generate 15 uniform bins in [mA - 5*sigma_voigt, mA + 5*sigma_voigt]."""
    voigt_width = sqrt(width**2 + sigma**2)
    mass_min = mA - 5 * voigt_width
    mass_max = mA + 5 * voigt_width
    nbins = 15
    bin_edges = np.linspace(mass_min, mass_max, nbins + 1)

    logging.info(f"Uniform binning: {nbins} bins, sigma_voigt={voigt_width:.3f} GeV, range=[{mass_min:.2f}, {mass_max:.2f}] GeV")
    return bin_edges


def calculateExtendedBins(mA, width, sigma):
    """Generate extended bin edges: [-10, -7, -5]*sigma + 15 uniform bins in [-5, +5]*sigma + [+5, +7, +10]*sigma."""
    voigt_width = sqrt(width**2 + sigma**2)

    # Use same 15 uniform bins as calculateFixedBins in [-5σ, +5σ]
    uniform_sigma_fractions = np.linspace(-5, 5, 16)  # 16 edges for 15 bins

    # Add extra bins at [-10, -7] on low side and [+7, +10] on high side
    extra_low = np.array([-10, -7])
    extra_high = np.array([7, 10])

    sigma_fractions = np.concatenate([extra_low, uniform_sigma_fractions, extra_high])
    bin_edges = mA + sigma_fractions * voigt_width

    logging.info(f"Extended binning: {len(bin_edges) - 1} bins, sigma_voigt={voigt_width:.3f} GeV")
    logging.info(f"  Bin edges (in sigma): {sigma_fractions.tolist()}")
    logging.info(f"  Bin edges (in GeV): [{bin_edges[0]:.2f}, ..., {bin_edges[-1]:.2f}]")
    return bin_edges


# =============================================================================
# A Mass Fitting Functions
# =============================================================================

def getFitResult(input_path, output_path, mA_nominal, outdir):
    """Fit A mass distribution using AmassFitter. Returns (mA, width, sigma)."""
    logging.info(f"Fitting A mass with nominal mA = {mA_nominal} GeV")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    fitter = ROOT.AmassFitter(input_path, output_path)
    fitter.fitMass(mA_nominal, mA_nominal - 20., mA_nominal + 20.)
    fitter.saveCanvas(f"{outdir}/signal_fit.png")

    mA = fitter.getRooMA().getVal()
    width = fitter.getRooWidth().getVal()
    sigma = fitter.getRooSigma().getVal()
    fitter.Close()

    logging.info(f"Fit result: mA={mA:.2f}, width={width:.3f}, sigma={sigma:.3f} GeV")
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

    logging.info(f"Loaded fit: mA={mA:.2f}, width={width:.3f}, sigma={sigma:.3f} GeV (from SR1E2Mu)")
    return mA, width, sigma


# =============================================================================
# Histogram Creation Functions
# =============================================================================

def getHist(basedir, process, bin_edges, mA, width, sigma, binning, syst="Central",
            threshold=-999., bg_weights=None, masspoint=None):
    """
    Create histogram from preprocessed tree using RDataFrame.

    Args:
        basedir: Sample directory path
        process: Process name (e.g., "MHc130_MA90", "nonprompt", "diboson")
        bin_edges: Numpy array of bin edges in mass space (GeV)
        mA, width, sigma: Signal parameters
        binning: Binning method ("uniform" or "sigma")
        syst: Systematic variation name (e.g., "Central", "MuonIDSF_Up")
        threshold: ParticleNet score threshold for event selection
        bg_weights: Dictionary of background weights
        masspoint: Signal mass point name

    Returns:
        TH1D histogram
    """
    file_path = f"{basedir}/{process}.root"
    tree_name = syst

    # Combine expects no underscore before Up/Down
    syst_formatted = syst.replace("_Up", "Up").replace("_Down", "Down")
    hist_name = process if syst == "Central" else f"{process}_{syst_formatted}"

    logging.debug(f"Creating histogram: {hist_name} from tree {tree_name}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample file not found: {file_path}")

    nbins = len(bin_edges) - 1
    mass_min, mass_max = get_mass_window(mA, width, sigma, binning)

    # Create ROOT vector for histogram binning
    bin_edges_vector = ROOT.std.vector['double'](bin_edges)

    # Check if tree exists and get branch list
    test_file = ROOT.TFile.Open(file_path)
    tree = test_file.Get(tree_name)
    if not tree:
        test_file.Close()
        raise RuntimeError(f"Tree '{tree_name}' not found in {file_path}")

    branches = [b.GetName() for b in tree.GetListOfBranches()]
    test_file.Close()

    # Create RDataFrame
    rdf = ROOT.RDataFrame(tree_name, file_path)

    # Apply mass window cut
    rdf = rdf.Filter(f"mass >= {mass_min} && mass <= {mass_max}")
    logging.debug(f"  Applied mass window cut: [{mass_min:.2f}, {mass_max:.2f}] GeV")

    # Apply ParticleNet score cut if threshold is provided
    if threshold > -999. and masspoint:
        score_sig = f"score_{masspoint}_signal"

        if score_sig in branches:
            score_nonprompt = f"score_{masspoint}_nonprompt"
            score_diboson = f"score_{masspoint}_diboson"
            score_ttZ = f"score_{masspoint}_ttZ"

            if bg_weights:
                w1 = bg_weights.get("nonprompt", 1.0)
                w2 = bg_weights.get("diboson", 1.0)
                w3 = bg_weights.get("ttX", 1.0)
                score_formula = f"({score_sig}) / ({score_sig} + {w1}*{score_nonprompt} + {w2}*{score_diboson} + {w3}*{score_ttZ})"
            else:
                score_formula = f"({score_sig}) / ({score_sig} + {score_nonprompt} + {score_diboson} + {score_ttZ})"

            rdf = rdf.Define("score_PN", score_formula)
            rdf = rdf.Filter(f"score_PN >= {threshold}")
            logging.debug(f"  Applied ParticleNet cut: score_PN >= {threshold:.3f}")
        else:
            # ParticleNet method requires score branches in all trees
            # Raise error instead of silently skipping the LR cut
            raise RuntimeError(
                f"ParticleNet score branches not found in {file_path}/{tree_name}\n"
                f"  Expected branch: {score_sig}\n"
                f"  This is required for method=ParticleNet to ensure PDF/Scale envelopes are calculated after LR cuts.\n"
                f"  Ensure that preprocess.py copies score branches to all systematic trees."
            )

    # Fill histogram
    hist = rdf.Histo1D((hist_name, "", nbins, bin_edges_vector.data()), "mass", "weight")

    hist_result = hist.GetValue()
    hist_result.SetDirectory(0)

    logging.debug(f"Histogram {hist_name}: {hist_result.GetEntries()} entries, integral = {hist_result.Integral():.4f}")

    return hist_result


def getHistMerged(basedir, process_list, bin_edges, mA, width, sigma, binning,
                  syst="Central", threshold=-999., bg_weights=None, masspoint=None):
    """Create merged histogram from multiple processes."""
    if len(process_list) == 0:
        raise ValueError("process_list cannot be empty")

    # Get first histogram
    hist_merged = getHist(basedir, process_list[0], bin_edges, mA, width, sigma, binning,
                          syst, threshold, bg_weights, masspoint)

    # Add remaining processes
    for process in process_list[1:]:
        try:
            hist_add = getHist(basedir, process, bin_edges, mA, width, sigma, binning,
                              syst, threshold, bg_weights, masspoint)
            hist_merged.Add(hist_add)
        except (FileNotFoundError, RuntimeError) as e:
            logging.warning(f"  Skipping {process} in merge: {e}")

    return hist_merged


def createEnvelopeHists(basedir, process, bin_edges, mA, width, sigma, binning,
                        variations, syst_name, threshold=-999., bg_weights=None, masspoint=None):
    """
    Create up/down envelope histograms from multiple variations.

    Args:
        basedir: Sample directory path
        process: Process name
        bin_edges: Bin edges array
        mA, width, sigma: Signal parameters
        binning: Binning method
        variations: List of variation tree names (e.g., ["PDF_0", "PDF_1", ...])
        syst_name: Output systematic name (e.g., "pdf_00", "QCDscale_BSMsignal")
        threshold, bg_weights, masspoint: Optional ParticleNet parameters

    Returns:
        Tuple of (hist_up, hist_down) envelope histograms
    """
    logging.debug(f"Creating envelope for {process}_{syst_name} from {len(variations)} variations")

    # Collect all variation histograms
    variation_hists = []
    for var in variations:
        try:
            hist = getHist(basedir, process, bin_edges, mA, width, sigma, binning,
                          var, threshold, bg_weights, masspoint)
            variation_hists.append(hist)
        except (FileNotFoundError, RuntimeError) as e:
            logging.warning(f"  Skipping variation {var}: {e}")

    if not variation_hists:
        raise RuntimeError(f"No variation histograms found for {process}_{syst_name}")

    # Get central histogram for reference shape
    central_hist = getHist(basedir, process, bin_edges, mA, width, sigma, binning,
                           "Central", threshold, bg_weights, masspoint)

    nbins = central_hist.GetNbinsX()

    # Create envelope histograms (same naming convention: {process}_{syst_name}Up/Down)
    hist_up = central_hist.Clone(f"{process}_{syst_name}Up")
    hist_down = central_hist.Clone(f"{process}_{syst_name}Down")
    hist_up.SetDirectory(0)
    hist_down.SetDirectory(0)

    # Calculate bin-by-bin envelope
    for i in range(1, nbins + 1):
        bin_values = [h.GetBinContent(i) for h in variation_hists]

        if bin_values:
            bin_max = max(bin_values)
            bin_min = min(bin_values)
            # Propagate errors as RMS of variations
            bin_err = np.std(bin_values) if len(bin_values) > 1 else 0.0

            hist_up.SetBinContent(i, bin_max)
            hist_up.SetBinError(i, bin_err)
            hist_down.SetBinContent(i, bin_min)
            hist_down.SetBinError(i, bin_err)

    logging.debug(f"  Envelope {syst_name}: up_integral={hist_up.Integral():.4f}, down_integral={hist_down.Integral():.4f}")

    return hist_up, hist_down


def ensurePositiveIntegral(hist, min_integral=1e-10):
    """
    Ensure histogram has positive integral for normalization.

    This prevents "Bogus norm" errors in Combine.
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
        central_bin = hist.GetNbinsX() // 2 + 1
        logging.warning(f"  Setting bin {central_bin} to {min_integral} to ensure positive normalization")
        hist.SetBinContent(central_bin, min_integral)
        hist.SetBinError(central_bin, min_integral)
        modified = True

    return modified


def calculate_weight_scale(value, direction):
    """
    Calculate weight scale for valued+shape systematics.

    For value >= 1: up = value, down = 2 - value
    For value < 1: up = 1 + value, down = 1 - value
    """
    if value >= 1.0:
        return value if direction == 'up' else 2.0 - value
    return 1.0 + value if direction == 'up' else 1.0 - value


def createScaledHist(central_hist, process, syst_name, value, direction):
    """
    Create a scaled histogram for valued+shape systematics.

    Args:
        central_hist: Central histogram to scale
        process: Process name
        syst_name: Systematic name
        value: Systematic value from config
        direction: 'up' or 'down'

    Returns:
        TH1D histogram scaled by the systematic value
    """
    scale = calculate_weight_scale(value, direction)

    # Create clone with appropriate name (same convention as preprocessed: {process}_{syst_name}Up/Down)
    suffix = "Up" if direction == "up" else "Down"
    hist_name = f"{process}_{syst_name}{suffix}"

    hist = central_hist.Clone(hist_name)
    hist.SetDirectory(0)
    hist.Scale(scale)

    logging.debug(f"  Created scaled histogram {hist_name}: scale={scale:.4f}, integral={hist.Integral():.4f}")

    return hist


# =============================================================================
# Background Validation Functions
# =============================================================================

def validateBackgroundStatistics(basedir, bin_edges, mA, width, sigma, binning,
                                  background_categories, masspoint, threshold=-999.,
                                  bg_weights=None, min_total_events=1):
    """
    Validate statistical quality of each background process.

    Args:
        background_categories: List of background category names from config
                              (excludes 'data', 'nonprompt', 'others')
    """
    logging.info("Validating background statistics...")

    mass_min, mass_max = get_mass_window(mA, width, sigma, binning)
    nbins = len(bin_edges) - 1
    results = {}

    for category in background_categories:
        # Map category name to output file name
        process = "conversion" if category == "conv" else category
        logging.info(f"  Validating {process}...")

        file_path = f"{basedir}/{process}.root"
        if not os.path.exists(file_path):
            logging.warning(f"    File not found, will merge to others")
            results[process] = {
                "total_events": 0,
                "decision": "merge",
                "reason": "file not found"
            }
            continue

        try:
            rdf = ROOT.RDataFrame("Central", file_path)
            rdf = rdf.Filter(f"mass >= {mass_min} && mass <= {mass_max}")

            if threshold > -999.:
                test_file = ROOT.TFile.Open(file_path, "READ")
                tree = test_file.Get("Central")
                if tree:
                    branches = [b.GetName() for b in tree.GetListOfBranches()]
                    test_file.Close()

                    score_sig = f"score_{masspoint}_signal"
                    if score_sig in branches:
                        score_nonprompt = f"score_{masspoint}_nonprompt"
                        score_diboson = f"score_{masspoint}_diboson"
                        score_ttZ = f"score_{masspoint}_ttZ"

                        if bg_weights:
                            w1 = bg_weights.get("nonprompt", 1.0)
                            w2 = bg_weights.get("diboson", 1.0)
                            w3 = bg_weights.get("ttX", 1.0)
                            score_formula = f"({score_sig}) / ({score_sig} + {w1}*{score_nonprompt} + {w2}*{score_diboson} + {w3}*{score_ttZ})"
                        else:
                            score_formula = f"({score_sig}) / ({score_sig} + {score_nonprompt} + {score_diboson} + {score_ttZ})"

                        rdf = rdf.Define("score_PN", score_formula)
                        rdf = rdf.Filter(f"score_PN >= {threshold}")
                else:
                    test_file.Close()

            bin_edges_vector = ROOT.std.vector['double'](bin_edges)
            hist = rdf.Histo1D(("temp", "", nbins, bin_edges_vector.data()), "mass", "weight")
            total_events = hist.GetValue().Integral()

        except Exception as e:
            logging.warning(f"    Error processing {process}: {e}")
            total_events = 0

        if total_events < min_total_events:
            decision = "merge"
            reason = f"total events ({total_events:.2f}) < {min_total_events}"
        else:
            decision = "keep"
            reason = "passes statistical requirements"

        results[process] = {
            "total_events": total_events,
            "decision": decision,
            "reason": reason
        }

        logging.info(f"    Total events: {total_events:.2f}")
        logging.info(f"    Decision: {decision.upper()} ({reason})")

    return results


def determineProcessList(validation_results, background_categories):
    """Determine final process list based on validation.

    Args:
        validation_results: Dict from validateBackgroundStatistics()
        background_categories: List of background category names from config
    """
    separate_processes = ["nonprompt"]  # Always separate
    merged_to_others = []

    for category in background_categories:
        # Map category name to output file name
        process = "conversion" if category == "conv" else category

        if process in validation_results and validation_results[process]["decision"] == "keep":
            separate_processes.append(process)
            logging.info(f"  {process}: keeping as separate process")
        else:
            merged_to_others.append(process)
            reason = validation_results.get(process, {}).get("reason", "not validated")
            logging.info(f"  {process}: merging to others ({reason})")

    return {
        "separate_processes": separate_processes,
        "merged_to_others": merged_to_others,
        "validation_results": validation_results
    }


# =============================================================================
# ParticleNet Optimization Functions
# =============================================================================

# Mapping from config categories to ParticleNet classes
# ParticleNet has 3 background classes: nonprompt, diboson, ttX
PARTICLENET_CLASS_MAPPING = {
    "nonprompt": ["nonprompt"],
    "diboson": ["diboson", "WZ", "ZZ"],
    "ttX": ["ttX", "ttW", "ttZ", "ttH", "tZq"],
}


def getBackgroundWeights(basedir, mA, width, sigma, binning, outdir):
    """
    Calculate normalized background class weights for ParticleNet.

    Maps config background categories to 3 ParticleNet classes:
    - nonprompt
    - diboson (includes WZ, ZZ)
    - ttX (includes ttW, ttZ, ttH, tZq)
    """
    logging.info("Calculating background cross-section weights:")

    mass_min, mass_max = get_mass_window(mA, width, sigma, binning)

    weights = {}
    for pn_class, possible_categories in PARTICLENET_CLASS_MAPPING.items():
        total_weight = 0.0
        found_any = False

        for category in possible_categories:
            # Map category to file name
            process = "conversion" if category == "conv" else category
            file_path = f"{basedir}/{process}.root"
            if not os.path.exists(file_path):
                continue

            rfile = ROOT.TFile.Open(file_path, "READ")
            tree = rfile.Get("Central")
            if not tree:
                rfile.Close()
                continue

            found_any = True
            for entry in range(tree.GetEntries()):
                tree.GetEntry(entry)
                if mass_min <= tree.mass <= mass_max:
                    total_weight += tree.weight

            rfile.Close()
            logging.info(f"  {pn_class} ({process}): {total_weight:.4f}")

        if not found_any:
            logging.warning(f"  No files found for ParticleNet class '{pn_class}', using default weight")
            weights[pn_class] = 1.0 / 3.0
        else:
            weights[pn_class] = total_weight

    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}
    else:
        logging.warning("Total background weight is zero! Using equal weights.")
        weights = {k: 1.0 / 3.0 for k in weights.keys()}

    logging.info("Background weights (normalized to sum=1):")
    for k, v in weights.items():
        logging.info(f"  {k}: {v:.4f}")

    save_json({
        "weights": weights,
        "yields": {k: v * total for k, v in weights.items()},
        "total_yield": float(total),
        "mass_window": [float(mass_min), float(mass_max)]
    }, f"{outdir}/background_weights.json")

    return weights


def loadDataset(basedir, process, masspoint, mA, width, sigma, binning, bg_weights=None):
    """Load events with ParticleNet scores from preprocessed samples."""
    file_path = f"{basedir}/{process}.root"
    if not os.path.exists(file_path):
        logging.warning(f"Sample file not found for optimization: {file_path}")
        return np.array([]), np.array([]), np.array([])

    rfile = ROOT.TFile.Open(file_path, "READ")
    tree = rfile.Get("Central")

    if not tree:
        logging.warning(f"Central tree not found in {file_path}")
        rfile.Close()
        return np.array([]), np.array([]), np.array([])

    mass_min, mass_max = get_mass_window(mA, width, sigma, binning)

    score_sig = f"score_{masspoint}_signal"
    score_nonprompt = f"score_{masspoint}_nonprompt"
    score_diboson = f"score_{masspoint}_diboson"
    score_ttZ = f"score_{masspoint}_ttZ"

    branches = [b.GetName() for b in tree.GetListOfBranches()]
    if score_sig not in branches:
        logging.warning(f"ParticleNet scores not found in {file_path}")
        rfile.Close()
        return np.array([]), np.array([]), np.array([])

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

        if not (mass_min <= mass <= mass_max):
            continue

        if bg_weights:
            w1 = bg_weights.get("nonprompt", 1.0)
            w2 = bg_weights.get("diboson", 1.0)
            w3 = bg_weights.get("ttX", 1.0)
            score_denom = s0 + w1 * s1 + w2 * s2 + w3 * s3
        else:
            score_denom = s0 + s1 + s2 + s3

        score_PN = s0 / score_denom if score_denom > 0 else 0.0

        scores_list.append(score_PN)
        weights_list.append(weight)
        labels_list.append(1 if process == masspoint else 0)

    rfile.Close()

    return np.array(scores_list), np.array(weights_list), np.array(labels_list)


def evalSensitivity(y_true, y_pred, weights, threshold=0.):
    """Calculate significance Z using Asimov formula."""
    signal_mask = (y_true == 1) & (y_pred > threshold)
    background_mask = (y_true == 0) & (y_pred > threshold)

    S = np.sum(weights[signal_mask])
    B = np.sum(weights[background_mask])

    if B <= 0:
        return 0.0

    return np.sqrt(2 * ((S + B) * np.log(1 + S / B) - S))


def getOptimizedThreshold(scores_sig, weights_sig, scores_bkg, weights_bkg):
    """Find optimal ParticleNet score threshold to maximize sensitivity."""
    y_pred = np.concatenate([scores_sig, scores_bkg])
    y_true = np.concatenate([np.ones(len(scores_sig)), np.zeros(len(scores_bkg))])
    weights = np.concatenate([weights_sig, weights_bkg])

    thresholds = np.linspace(0, 1, 101)
    sensitivities = [evalSensitivity(y_true, y_pred, weights, threshold) for threshold in thresholds]

    best_idx = np.argmax(sensitivities)
    best_threshold = thresholds[best_idx]
    initial_sensitivity = sensitivities[0]
    max_sensitivity = sensitivities[best_idx]

    logging.info(f"Threshold optimization:")
    logging.info(f"  Best threshold: {best_threshold:.3f}")
    logging.info(f"  Initial sensitivity (no cut): {initial_sensitivity:.3f}")
    logging.info(f"  Max sensitivity: {max_sensitivity:.3f}")
    if initial_sensitivity > 0:
        logging.info(f"  Improvement: {(max_sensitivity / initial_sensitivity - 1) * 100:.2f}%")

    return best_threshold, initial_sensitivity, max_sensitivity


# =============================================================================
# Main Execution
# =============================================================================

def main():
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(levelname)s - %(message)s')

    workdir = os.getenv("WORKDIR")
    if not workdir:
        raise EnvironmentError("WORKDIR environment variable not set. Please run 'source setup.sh'")

    # Paths
    basedir = f"{workdir}/SignalRegionStudyV1/samples/{args.era}/{args.channel}/{args.masspoint}"
    outdir = f"{workdir}/SignalRegionStudyV1/templates/{args.era}/{args.channel}/{args.masspoint}/{args.method}/{args.binning}"

    logging.info(f"Starting template generation")
    logging.info(f"  Mass point: {args.masspoint}")
    logging.info(f"  Era: {args.era}")
    logging.info(f"  Channel: {args.channel}")
    logging.info(f"  Method: {args.method}")
    logging.info(f"  Binning: {args.binning}")
    logging.info(f"Input directory: {basedir}")
    logging.info(f"Output directory: {outdir}")

    # Load SignalRegionStudy library
    lib_path = f"{workdir}/SignalRegionStudy/lib/libSignalRegionStudy.so"
    if not os.path.exists(lib_path):
        raise FileNotFoundError(f"SignalRegionStudy library not found: {lib_path}. Please run './scripts/build.sh'")
    ROOT.gSystem.Load(lib_path)

    # Load configurations
    config = load_config(workdir, args.era, args.channel)
    syst_categories = categorize_systematics(config['systematics'])

    logging.info(f"Found {len(syst_categories['preprocessed_shape'])} preprocessed shape systematics")
    logging.info(f"Found {len(syst_categories['valued_shape'])} valued shape systematics (in preprocess)")
    logging.info(f"Found {len(syst_categories['multi_variation'])} multi-variation systematics (PDF/Scale)")

    # Create output directory
    if os.path.exists(outdir):
        logging.info(f"Removing existing output directory: {outdir}")
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    # Extract nominal mA from masspoint name
    mA_nominal = float(args.masspoint.split("_")[1].replace("MA", ""))

    # ========================================
    # A Mass Fitting
    # ========================================
    logging.info("=" * 60)
    logging.info("A Mass Fitting")
    logging.info("=" * 60)

    if args.channel == "SR3Mu":
        sr1e2mu_fit_path = f"{workdir}/SignalRegionStudyV1/templates/{args.era}/SR1E2Mu/{args.masspoint}/{args.method}/{args.binning}/fit_result.root"

        if not os.path.exists(sr1e2mu_fit_path):
            raise FileNotFoundError(
                f"SR1E2Mu fit results not found: {sr1e2mu_fit_path}\n"
                f"Please run makeBinnedTemplates.py for SR1E2Mu first"
            )

        mA, width, sigma = loadFitResult(sr1e2mu_fit_path)

        # Save marker file for SR3Mu
        marker_file = ROOT.TFile(f"{outdir}/fit_result.root", "RECREATE")
        marker_file.WriteObject(ROOT.TNamed("source", f"SR1E2Mu/{args.era}/{args.masspoint}"), "fit_source")
        marker_file.Close()

        save_json({
            "mass": float(mA),
            "width": float(width),
            "sigma": float(sigma),
            "source": "SR1E2Mu"
        }, f"{outdir}/signal_fit.json")

    else:
        # SR1E2Mu: Perform direct fit
        signal_path = f"{basedir}/{args.masspoint}.root"
        mA, width, sigma = getFitResult(signal_path, f"{outdir}/fit_result.root", mA_nominal, outdir)

        save_json({
            "mass": float(mA),
            "width": float(width),
            "sigma": float(sigma)
        }, f"{outdir}/signal_fit.json")

    # ========================================
    # Calculate Binning
    # ========================================
    logging.info("=" * 60)
    logging.info(f"Calculating {args.binning} binning...")
    logging.info("=" * 60)

    if args.binning == "extended":
        bin_edges = calculateExtendedBins(mA, width, sigma)
        binning_method = "ExtendedBins"
    else:
        bin_edges = calculateFixedBins(mA, width, sigma)
        binning_method = "UniformMass"

    voigt_width = sqrt(width**2 + sigma**2)
    mass_min, mass_max = get_mass_window(mA, width, sigma, args.binning)

    save_json({
        "nbins": len(bin_edges) - 1,
        "bin_edges": bin_edges.tolist(),
        "method": binning_method,
        "voigt_width": float(voigt_width),
        "mass_min": float(mass_min),
        "mass_max": float(mass_max),
        "binning_type": args.binning
    }, f"{outdir}/binning.json")

    # ========================================
    # Extract Background Categories
    # ========================================
    # Extract background categories from config (exclude reserved keys)
    reserved_keys = {"data", "nonprompt", "others"}
    background_categories = [k for k in config['samples'].keys() if k not in reserved_keys]
    logging.info(f"Background categories from config: {background_categories}")

    # ========================================
    # Background Weights (ParticleNet only)
    # ========================================
    bg_weights = None
    best_threshold = -999.

    if args.method == "ParticleNet":
        logging.info("=" * 60)
        logging.info("ParticleNet optimization")
        logging.info("=" * 60)

        bg_weights = getBackgroundWeights(basedir, mA, width, sigma, args.binning, outdir)

        # Load signal dataset
        scores_sig, weights_sig, _ = loadDataset(basedir, args.masspoint, args.masspoint, mA, width, sigma, args.binning, bg_weights)

        if len(scores_sig) == 0:
            logging.warning("ParticleNet scores not found! Proceeding without cuts.")
        else:
            # Load background datasets from all background categories + nonprompt + others
            scores_bkg_list = []
            weights_bkg_list = []

            # Build list of all background processes to load
            bkg_processes = ["nonprompt"] + [("conversion" if c == "conv" else c) for c in background_categories] + ["others"]
            logging.info(f"Loading backgrounds for optimization: {bkg_processes}")

            for process in bkg_processes:
                scores, weights, _ = loadDataset(basedir, process, args.masspoint, mA, width, sigma, args.binning, bg_weights)
                if len(scores) > 0:
                    scores_bkg_list.append(scores)
                    weights_bkg_list.append(weights)

            if len(scores_bkg_list) > 0:
                scores_bkg = np.concatenate(scores_bkg_list)
                weights_bkg = np.concatenate(weights_bkg_list)

                best_threshold, initial_sensitivity, max_sensitivity = getOptimizedThreshold(
                    scores_sig, weights_sig, scores_bkg, weights_bkg
                )

                improvement = (max_sensitivity / initial_sensitivity - 1) if initial_sensitivity > 0 else 0
                save_json({
                    "masspoint": args.masspoint,
                    "threshold": float(best_threshold),
                    "initial_sensitivity": float(initial_sensitivity),
                    "max_sensitivity": float(max_sensitivity),
                    "improvement": float(improvement)
                }, f"{outdir}/threshold.json")

    # ========================================
    # Validate Background Statistics
    # ========================================
    logging.info("=" * 60)
    logging.info("Validating background statistics...")
    logging.info("=" * 60)

    validation_results = validateBackgroundStatistics(
        basedir, bin_edges, mA, width, sigma, args.binning,
        background_categories, args.masspoint, best_threshold, bg_weights,
        min_total_events=1
    )

    save_json({
        process: {
            "total_events": float(result["total_events"]),
            "decision": result["decision"],
            "reason": result["reason"]
        }
        for process, result in validation_results.items()
    }, f"{outdir}/background_validation.json")

    # Determine final process list
    logging.info("Determining final process list...")
    process_config = determineProcessList(validation_results, background_categories)
    separate_processes = process_config["separate_processes"]
    merged_to_others = process_config["merged_to_others"]

    save_json({
        "separate_processes": separate_processes,
        "merged_to_others": merged_to_others,
        "description": "Processes kept separate vs merged into 'others' based on statistical validation"
    }, f"{outdir}/process_list.json")

    logging.info(f"Final process configuration:")
    logging.info(f"  Separate processes: {separate_processes}")
    logging.info(f"  Merged to others: {merged_to_others}")

    # ========================================
    # Create Output ROOT File
    # ========================================
    logging.info("=" * 60)
    logging.info("Creating histogram templates...")
    logging.info("=" * 60)

    output_file = ROOT.TFile(f"{outdir}/shapes.root", "RECREATE")

    # Initialize data_obs histogram
    nbins = len(bin_edges) - 1
    bin_edges_vector = ROOT.std.vector['double'](bin_edges)
    data_obs = ROOT.TH1D("data_obs", "data_obs", nbins, bin_edges_vector.data())
    data_obs.SetDirectory(0)

    background_hists = {}

    # ========================================
    # Process Signal
    # ========================================
    logging.info(f"Processing signal: {args.masspoint}")

    # Central histogram
    hist_signal_central = getHist(basedir, args.masspoint, bin_edges, mA, width, sigma, args.binning,
                                   "Central", best_threshold, bg_weights, args.masspoint)
    ensurePositiveIntegral(hist_signal_central)
    output_file.cd()
    hist_signal_central.Write()

    # Preprocessed shape systematics (2 variations each)
    for syst_name, variations, group in syst_categories['preprocessed_shape']:
        if "signal" not in group:
            continue
        logging.debug(f"  Processing signal systematic: {syst_name}")
        for var in variations:
            output_tree = get_output_tree_name(syst_name, var)
            try:
                hist = getHist(basedir, args.masspoint, bin_edges, mA, width, sigma, args.binning,
                              output_tree, best_threshold, bg_weights, args.masspoint)
                ensurePositiveIntegral(hist)
                output_file.cd()
                hist.Write()
            except (FileNotFoundError, RuntimeError) as e:
                logging.warning(f"    Skipping {syst_name}/{var}: {e}")

    # Valued shape systematics (created by scaling Central histogram)
    for syst_name, value, group in syst_categories['valued_shape']:
        if "signal" not in group:
            continue
        logging.debug(f"  Processing signal valued systematic: {syst_name} (value={value})")
        for direction in ["up", "down"]:
            hist = createScaledHist(hist_signal_central, args.masspoint, syst_name, value, direction)
            ensurePositiveIntegral(hist)
            output_file.cd()
            hist.Write()

    # Multi-variation systematics (PDF/Scale envelopes)
    for syst_name, variations, group in syst_categories['multi_variation']:
        if "signal" not in group:
            continue
        logging.info(f"  Creating envelope for signal: {syst_name}")

        # Map variation names to tree names (preprocess uses different naming)
        tree_variations = []
        for var in variations:
            if var.startswith("pdf_"):
                num = int(var.replace("pdf_", ""))
                tree_variations.append(f"PDF_{num}")
            elif var.startswith("Scale_"):
                tree_variations.append(var)
            else:
                tree_variations.append(var)

        try:
            hist_up, hist_down = createEnvelopeHists(
                basedir, args.masspoint, bin_edges, mA, width, sigma, args.binning,
                tree_variations, syst_name, best_threshold, bg_weights, args.masspoint
            )
            ensurePositiveIntegral(hist_up)
            ensurePositiveIntegral(hist_down)
            output_file.cd()
            hist_up.Write()
            hist_down.Write()
        except RuntimeError as e:
            logging.warning(f"    Skipping envelope {syst_name}: {e}")

    logging.info(f"Signal templates created: {args.masspoint} (integral = {hist_signal_central.Integral():.4f})")

    # ========================================
    # Process Backgrounds (Separate)
    # ========================================
    for process in separate_processes:
        logging.info(f"Processing {process} background (separate template)")

        # Central histogram
        hist_central = getHist(basedir, process, bin_edges, mA, width, sigma, args.binning,
                               "Central", best_threshold, bg_weights, args.masspoint)
        ensurePositiveIntegral(hist_central)
        data_obs.Add(hist_central)
        output_file.cd()
        hist_central.Write()
        background_hists[process] = hist_central

        # Determine which systematics apply to this process
        if process == "nonprompt":
            # Nonprompt-specific valued_shape systematics (scaled from Central)
            for syst_name, value, grp in syst_categories['valued_shape']:
                if "nonprompt" not in grp:
                    continue
                for direction in ["up", "down"]:
                    hist = createScaledHist(hist_central, process, syst_name, value, direction)
                    ensurePositiveIntegral(hist)
                    output_file.cd()
                    hist.Write()
        else:
            # Prompt systematics - use process name directly
            # Note: systematics config uses "conversion" in groups, not "conv"
            for syst_name, variations, group in syst_categories['preprocessed_shape']:
                if process not in group:
                    continue
                for var in variations:
                    output_tree = get_output_tree_name(syst_name, var)
                    try:
                        hist = getHist(basedir, process, bin_edges, mA, width, sigma, args.binning,
                                      output_tree, best_threshold, bg_weights, args.masspoint)
                        ensurePositiveIntegral(hist)
                        output_file.cd()
                        hist.Write()
                    except (FileNotFoundError, RuntimeError) as e:
                        logging.warning(f"    Skipping {process}/{syst_name}/{var}: {e}")

            # Valued shape systematics (created by scaling Central histogram)
            for syst_name, value, group in syst_categories['valued_shape']:
                if process not in group:
                    continue
                for direction in ["up", "down"]:
                    hist = createScaledHist(hist_central, process, syst_name, value, direction)
                    ensurePositiveIntegral(hist)
                    output_file.cd()
                    hist.Write()

        logging.info(f"  {process} templates created (integral = {hist_central.Integral():.4f})")

    # ========================================
    # Process "others" Background (Merged)
    # ========================================
    logging.info("Processing others background (merged template)")

    others_process_list = ["others"] + merged_to_others
    logging.info(f"  Merging processes: {others_process_list}")

    # Central histogram
    hist_others = getHistMerged(basedir, others_process_list, bin_edges, mA, width, sigma, args.binning,
                                "Central", best_threshold, bg_weights, args.masspoint)
    ensurePositiveIntegral(hist_others)
    data_obs.Add(hist_others)
    output_file.cd()
    hist_others.Write()
    background_hists["others"] = hist_others

    # Prompt systematics for others
    # Build list of process names that could be in "others" (use actual process names for group matching)
    others_process_names = [("conversion" if c == "conv" else c) for c in background_categories] + ["others"]
    for syst_name, variations, group in syst_categories['preprocessed_shape']:
        # Apply to all prompt backgrounds in "others"
        applicable = any(proc in group for proc in others_process_names)
        if not applicable:
            continue

        for var in variations:
            output_tree = get_output_tree_name(syst_name, var)
            try:
                hist = getHistMerged(basedir, others_process_list, bin_edges, mA, width, sigma, args.binning,
                                    output_tree, best_threshold, bg_weights, args.masspoint)
                ensurePositiveIntegral(hist)
                output_file.cd()
                hist.Write()
            except (FileNotFoundError, RuntimeError) as e:
                logging.warning(f"    Skipping others/{syst_name}/{var}: {e}")

    # Valued shape systematics (created by scaling merged Central histogram)
    for syst_name, value, group in syst_categories['valued_shape']:
        applicable = any(proc in group for proc in others_process_names)
        if not applicable:
            continue

        for direction in ["up", "down"]:
            hist = createScaledHist(hist_others, "others", syst_name, value, direction)
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

    output_file.Close()

    # ========================================
    # Summary
    # ========================================
    logging.info("=" * 60)
    logging.info("Template generation complete!")
    logging.info(f"Output file: {outdir}/shapes.root")
    logging.info("=" * 60)
    logging.info("Process yields:")
    logging.info(f"  Signal ({args.masspoint}):  {hist_signal_central.Integral():>10.4f}")

    for process in separate_processes:
        logging.info(f"  {process.capitalize():23s} {background_hists[process].Integral():>10.4f}")

    logging.info(f"  {'Others':23s} {background_hists['others'].Integral():>10.4f}")
    if merged_to_others:
        logging.info(f"    (merged: {', '.join(merged_to_others)})")

    logging.info(f"  Total background:            {data_obs.Integral():>10.4f}")
    if data_obs.Integral() > 0:
        logging.info(f"  S/B ratio:                   {hist_signal_central.Integral() / data_obs.Integral():>10.4f}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
