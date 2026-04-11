#!/usr/bin/env python3
"""
Generate binned histogram templates for HiggsCombine.

Usage:
    python makeBinnedTemplates.py --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --method Baseline
"""
import os
import shutil
import logging
import argparse
import json
from collections import OrderedDict
import ROOT
import numpy as np
from math import sqrt

from template_utils import (
    save_json, parse_variations, get_output_tree_name, combine_suffix_from_tree,
    ensure_positive_integral, build_particlenet_score, create_filtered_rdf,
    create_scaled_hist, is_run3_era, is_signal_scaled_from_run2,
    get_run2_tree_name_for_run3_syst, categorize_systematics,
    calculate_adaptive_bins, check_binning_quality, apply_syst_driven_merging,
    iter_shape_directions
)
from plotter import FitCanvasWithRatio

# Bins with total-bkg syst envelope / nominal above this threshold are merged
# into a neighbour. Addresses a stat-review concern about sudden-dip bins with
# very large relative systematic uncertainties underestimating backgrounds.
SYST_MERGE_THRESHOLD = 2.0

# Signal scaling factor for partial-unblind mode
# When using --partial-unblind, signal is scaled by this factor
# The resulting limit on r should be interpreted as limit on (PARTIAL_UNBLIND_SIGNAL_SCALE × σ)
PARTIAL_UNBLIND_SIGNAL_SCALE = 50

# Floor value for empty/problematic bins when adaptive binning is exhausted
BIN_FLOOR_VALUE = 1e-6


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate binned histogram templates for HiggsCombine")
    parser.add_argument("--era", required=True, type=str, help="Data-taking period (2016preVFP, 2017, 2018, 2022, etc.)")
    parser.add_argument("--channel", required=True, type=str, help="Analysis channel (SR1E2Mu, SR3Mu)")
    parser.add_argument("--masspoint", required=True, type=str, help="Signal mass point (e.g., MHc130_MA90)")
    parser.add_argument("--method", required=True, type=str, help="Template method (Baseline, ParticleNet, etc.)")
    parser.add_argument("--binning", default="extended", choices=["uniform", "extended"],
                        help="Binning method: 'extended' (19 bins, default) or 'uniform' (15 bins)")
    parser.add_argument("--unblind", action="store_true",
                        help="Use real data for data_obs instead of MC sum")
    parser.add_argument("--partial-unblind", action="store_true", dest="partial_unblind",
                        help="Unblind low LR region (score < 0.3). Requires --method ParticleNet")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def load_config(workdir, era, channel):
    """Load systematics and sample group configurations."""
    # Load systematics config
    config_path = f"{workdir}/SignalRegionStudyV2/configs/systematics.{era}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Systematics config not found: {config_path}")

    with open(config_path) as f:
        json_systematics = json.load(f)

    if channel not in json_systematics:
        raise ValueError(f"Channel '{channel}' not found in {config_path}")

    # Load sample groups config
    samplegroups_path = f"{workdir}/SignalRegionStudyV2/configs/samplegroups.json"
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


# =============================================================================
# A Mass Fitting Functions (Double Crystal Ball)
# =============================================================================

def getFitResultDCB(input_path, mA_nominal, outdir, era, masspoint, channel=""):
    """
    Fit A mass distribution using Double Crystal Ball (RooCrystalBall).

    Strategy: unbinned DCB pre-fit (mA +/- mA/3) → get peak scale →
              unbinned DCB narrow fit (fitted_mA +/- 10*vw).

    Args:
        input_path: Path to signal ROOT file with Central tree
        mA_nominal: Nominal A mass value
        outdir: Output directory for fit plot and JSON
        era: Data-taking era (for CMS style labels on fit plot)
        masspoint: Mass point name (for plot label)
        channel: Channel name (for plot label)

    Returns:
        dict with keys: x0, sigmaL, sigmaR, alphaL, nL, alphaR, nR, sigma_eff
    """
    logging.info(f"Fitting A mass with DCB, nominal mA = {mA_nominal} GeV")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

    f = ROOT.TFile.Open(input_path)
    tree = f.Get("Central")

    # Step 1: unbinned DCB pre-fit to get peak position and scale
    wide_lo = max(mA_nominal - mA_nominal / 3.0, 12.0)
    wide_hi = mA_nominal + mA_nominal / 3.0

    mass_w = ROOT.RooRealVar("mass", "mass", wide_lo, wide_hi)
    weight_w = ROOT.RooRealVar("weight", "weight", -10, 10)
    ds_wide = ROOT.RooDataSet("ds_wide", "", tree, ROOT.RooArgSet(mass_w, weight_w),
                               f"mass >= {wide_lo} && mass <= {wide_hi}", "weight")

    pre_x0 = ROOT.RooRealVar("pre_x0", "x0", mA_nominal, wide_lo, wide_hi)
    pre_sL = ROOT.RooRealVar("pre_sL", "sL", 1.0, 0.01, 10.0)
    pre_sR = ROOT.RooRealVar("pre_sR", "sR", 1.0, 0.01, 10.0)
    pre_aL = ROOT.RooRealVar("pre_aL", "aL", 1.5, 0.5, 10.0)
    pre_nL = ROOT.RooRealVar("pre_nL", "nL", 2.0, 0.1, 50.0)
    pre_aR = ROOT.RooRealVar("pre_aR", "aR", 1.5, 0.5, 10.0)
    pre_nR = ROOT.RooRealVar("pre_nR", "nR", 2.0, 0.1, 50.0)
    pre_dcb = ROOT.RooCrystalBall("pre_dcb", "", mass_w, pre_x0,
                                   pre_sL, pre_sR, pre_aL, pre_nL, pre_aR, pre_nR)
    pre_dcb.fitTo(ds_wide, ROOT.RooFit.SumW2Error(True),
                  ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

    fitted_mA = pre_x0.getVal()
    vw = sqrt(0.5 * (pre_sL.getVal()**2 + pre_sR.getVal()**2))

    # Step 2: unbinned DCB narrow fit
    fit_lo = max(fitted_mA - 10.0 * vw, 12.0)
    fit_hi = fitted_mA + 10.0 * vw

    mass_n = ROOT.RooRealVar("mass", "mass", fit_lo, fit_hi)
    weight_n = ROOT.RooRealVar("weight", "weight", -10, 10)
    ds_narrow = ROOT.RooDataSet("ds_narrow", "", tree, ROOT.RooArgSet(mass_n, weight_n),
                                 f"mass >= {fit_lo} && mass <= {fit_hi}", "weight")

    dcb_x0 = ROOT.RooRealVar("dcb_x0", "x0", fitted_mA, fit_lo, fit_hi)
    dcb_sL = ROOT.RooRealVar("dcb_sL", "sigmaL", 0.8 * vw, 0.01 * vw, 3.0 * vw)
    dcb_sR = ROOT.RooRealVar("dcb_sR", "sigmaR", 0.8 * vw, 0.01 * vw, 3.0 * vw)
    dcb_aL = ROOT.RooRealVar("dcb_aL", "alphaL", 1.5, 0.5, 10.0)
    dcb_nL = ROOT.RooRealVar("dcb_nL", "nL", 2.0, 0.1, 50.0)
    dcb_aR = ROOT.RooRealVar("dcb_aR", "alphaR", 1.5, 0.5, 10.0)
    dcb_nR = ROOT.RooRealVar("dcb_nR", "nR", 2.0, 0.1, 50.0)
    dcb = ROOT.RooCrystalBall("dcb", "", mass_n, dcb_x0,
                               dcb_sL, dcb_sR, dcb_aL, dcb_nL, dcb_aR, dcb_nR)
    dcb.fitTo(ds_narrow, ROOT.RooFit.SumW2Error(True),
              ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

    f.Close()

    sigma_eff = sqrt(0.5 * (dcb_sL.getVal()**2 + dcb_sR.getVal()**2))

    # Save fit plot (binned histogram for visualization, 20 uniform bins)
    nbins = 100
    rdf = ROOT.RDataFrame("Central", input_path)
    rdf = rdf.Filter(f"mass >= {fit_lo} && mass <= {fit_hi}")
    hist = rdf.Histo1D(
        ROOT.RDF.TH1DModel("h_fit", "", nbins, fit_lo, fit_hi), "mass", "weight"
    ).GetValue().Clone("h_fit_c")
    hist.SetDirectory(0)

    mass_plot = ROOT.RooRealVar("mass_plot", "mass", fit_lo, fit_hi)
    roo_data = ROOT.RooDataHist("data_plot", "", ROOT.RooArgList(mass_plot), hist)

    # Reconstruct DCB on plot variable with fixed parameters
    p_x0 = ROOT.RooRealVar("p_x0", "x0", dcb_x0.getVal())
    p_sL = ROOT.RooRealVar("p_sL", "sL", dcb_sL.getVal())
    p_sR = ROOT.RooRealVar("p_sR", "sR", dcb_sR.getVal())
    p_aL = ROOT.RooRealVar("p_aL", "aL", dcb_aL.getVal())
    p_nL = ROOT.RooRealVar("p_nL", "nL", dcb_nL.getVal())
    p_aR = ROOT.RooRealVar("p_aR", "aR", dcb_aR.getVal())
    p_nR = ROOT.RooRealVar("p_nR", "nR", dcb_nR.getVal())
    for v in [p_x0, p_sL, p_sR, p_aL, p_nL, p_aR, p_nR]:
        v.setConstant(True)
    dcb_plot = ROOT.RooCrystalBall("dcb_plot", "", mass_plot, p_x0,
                                    p_sL, p_sR, p_aL, p_nL, p_aR, p_nR)

    fit_models = OrderedDict([("DCB", (dcb_plot, 7, ROOT.kSolid))])
    config = {
        "era": era, "xTitle": "m_{A} [GeV]", "yTitle": "Events",
        "rTitle": "Fit / MC", "rRange": [0.5, 1.5],
        "channel": channel, "masspoint": masspoint,
        "channelPosX": 0.2, "channelPosY": 0.74,
        "channelFont": 61, "channelSize": 0.04,
        "masspointPosX": 0.2, "masspointPosY": 0.69,
        "masspointFont": 61, "masspointSize": 0.04,
        "legend": [0.60, 0.65, 0.90, 0.78],
        "legendTextSize": 0.03, "iPos": 0, "maxDigits": 3,
        "colors": [ROOT.kRed],
    }
    canvas = FitCanvasWithRatio(roo_data, mass_plot, hist, fit_models, config)
    canvas.drawPadUp()
    canvas.drawPadDown()
    canvas.drawMasspoint()
    canvas.canv.SaveAs(f"{outdir}/signal_fit.png")

    result = {
        "x0": float(dcb_x0.getVal()),
        "sigmaL": float(dcb_sL.getVal()),
        "sigmaR": float(dcb_sR.getVal()),
        "alphaL": float(dcb_aL.getVal()),
        "nL": float(dcb_nL.getVal()),
        "alphaR": float(dcb_aR.getVal()),
        "nR": float(dcb_nR.getVal()),
        "sigma_eff": float(sigma_eff),
    }

    logging.info(f"DCB fit result: x0={result['x0']:.2f}, "
                 f"sigmaL={result['sigmaL']:.3f}, sigmaR={result['sigmaR']:.3f}, "
                 f"sigma_eff={sigma_eff:.3f} GeV")

    return result


# =============================================================================
# Histogram Creation Functions
# =============================================================================

def getHist(basedir, process, bin_edges, mass_min, mass_max, syst="Central",
            threshold=-999., upper_threshold=None, bg_weights=None, masspoint=None):
    """Create histogram from preprocessed tree using RDataFrame."""
    file_path = f"{basedir}/{process}.root"
    tree_name = syst

    # Combine expects no underscore before Up/Down
    syst_formatted = syst.replace("_Up", "Up").replace("_Down", "Down")
    hist_name = process if syst == "Central" else f"{process}_{syst_formatted}"

    logging.debug(f"Creating histogram: {hist_name} from tree {tree_name}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample file not found: {file_path}")

    nbins = len(bin_edges) - 1

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
    if (threshold > -999. or upper_threshold is not None) and masspoint:
        score_sig = f"score_{masspoint}_signal"
        if score_sig in branches:
            score_formula = build_particlenet_score(masspoint, bg_weights)
            rdf = rdf.Define("score_PN", score_formula)
            if upper_threshold is not None:
                rdf = rdf.Filter(f"score_PN < {upper_threshold}")
                logging.debug(f"  Applied ParticleNet cut: score_PN < {upper_threshold:.3f}")
            elif threshold > -999.:
                rdf = rdf.Filter(f"score_PN >= {threshold}")
                logging.debug(f"  Applied ParticleNet cut: score_PN >= {threshold:.3f}")
        else:
            raise RuntimeError(
                f"ParticleNet score branches not found in {file_path}/{tree_name}\n"
                f"  Expected branch: {score_sig}"
            )

    # Fill histogram
    hist = rdf.Histo1D((hist_name, "", nbins, bin_edges_vector.data()), "mass", "weight")

    hist_result = hist.GetValue()
    hist_result.SetDirectory(0)

    logging.debug(f"Histogram {hist_name}: {hist_result.GetEntries()} entries, integral = {hist_result.Integral():.4f}")

    return hist_result


def getHistMerged(basedir, process_list, bin_edges, mass_min, mass_max,
                  syst="Central", threshold=-999., upper_threshold=None, bg_weights=None, masspoint=None):
    """Create merged histogram from multiple processes."""
    if len(process_list) == 0:
        raise ValueError("process_list cannot be empty")

    # Get first histogram
    hist_merged = getHist(basedir, process_list[0], bin_edges, mass_min, mass_max,
                          syst, threshold, upper_threshold, bg_weights, masspoint)

    # Add remaining processes
    for process in process_list[1:]:
        try:
            hist_add = getHist(basedir, process, bin_edges, mass_min, mass_max,
                              syst, threshold, upper_threshold, bg_weights, masspoint)
            hist_merged.Add(hist_add)
        except (FileNotFoundError, RuntimeError) as e:
            logging.warning(f"  Skipping {process} in merge: {e}")

    return hist_merged


def createEnvelopeHists(basedir, process, bin_edges, mass_min, mass_max,
                        variations, syst_name, threshold=-999., upper_threshold=None,
                        bg_weights=None, masspoint=None):
    """Create up/down envelope histograms from multiple variations."""
    logging.debug(f"Creating envelope for {process}_{syst_name} from {len(variations)} variations")

    # Collect all variation histograms
    variation_hists = []
    for var in variations:
        try:
            hist = getHist(basedir, process, bin_edges, mass_min, mass_max,
                          var, threshold, upper_threshold, bg_weights, masspoint)
            variation_hists.append(hist)
        except (FileNotFoundError, RuntimeError) as e:
            logging.warning(f"  Skipping variation {var}: {e}")

    if not variation_hists:
        raise RuntimeError(f"No variation histograms found for {process}_{syst_name}")

    # Get central histogram for reference shape
    central_hist = getHist(basedir, process, bin_edges, mass_min, mass_max,
                           "Central", threshold, upper_threshold, bg_weights, masspoint)

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


def getDataHist(basedir, bin_edges, mass_min, mass_max,
                threshold=-999., upper_threshold=None, bg_weights=None, masspoint=None):
    """Create data histogram from data.root file."""
    file_path = f"{basedir}/data.root"
    hist_name = "data_obs"

    logging.debug(f"Creating data histogram from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    nbins = len(bin_edges) - 1
    bin_edges_vector = ROOT.std.vector['double'](bin_edges)

    # Check if tree exists and get branch list
    test_file = ROOT.TFile.Open(file_path)
    tree = test_file.Get("Central")
    if not tree:
        test_file.Close()
        raise RuntimeError(f"Tree 'Central' not found in {file_path}")

    branches = [b.GetName() for b in tree.GetListOfBranches()]
    test_file.Close()

    # Create RDataFrame
    rdf = ROOT.RDataFrame("Central", file_path)

    # Apply mass window cut
    rdf = rdf.Filter(f"mass >= {mass_min} && mass <= {mass_max}")
    logging.debug(f"  Applied mass window cut: [{mass_min:.2f}, {mass_max:.2f}] GeV")

    # Apply ParticleNet score cut if threshold or upper_threshold is provided
    if (threshold > -999. or upper_threshold is not None) and masspoint:
        score_sig = f"score_{masspoint}_signal"
        if score_sig in branches:
            score_formula = build_particlenet_score(masspoint, bg_weights)
            rdf = rdf.Define("score_PN", score_formula)
            if upper_threshold is not None:
                rdf = rdf.Filter(f"score_PN < {upper_threshold}")
                logging.debug(f"  Applied ParticleNet cut: score_PN < {upper_threshold:.3f}")
            elif threshold > -999.:
                rdf = rdf.Filter(f"score_PN >= {threshold}")
                logging.debug(f"  Applied ParticleNet cut: score_PN >= {threshold:.3f}")
        else:
            raise RuntimeError(
                f"ParticleNet score branches not found in {file_path}/Central\n"
                f"  Expected branch: {score_sig}"
            )

    # Fill histogram (data uses weight=1)
    hist = rdf.Histo1D((hist_name, "", nbins, bin_edges_vector.data()), "mass")

    hist_result = hist.GetValue()
    hist_result.SetDirectory(0)

    logging.debug(f"Data histogram: {hist_result.GetEntries()} entries, integral = {hist_result.Integral():.4f}")

    return hist_result


# =============================================================================
# Background Validation Functions
# =============================================================================

def validateBackgroundStatistics(basedir, bin_edges, mass_min, mass_max,
                                  background_categories, masspoint, threshold=-999.,
                                  upper_threshold=None, bg_weights=None, min_total_events=1):
    """Validate statistical quality of each background process."""
    logging.info("Validating background statistics...")
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

            if threshold > -999. or upper_threshold is not None:
                test_file = ROOT.TFile.Open(file_path, "READ")
                tree = test_file.Get("Central")
                if tree:
                    branches = [b.GetName() for b in tree.GetListOfBranches()]
                    test_file.Close()
                    score_sig = f"score_{masspoint}_signal"
                    if score_sig in branches:
                        score_formula = build_particlenet_score(masspoint, bg_weights)
                        rdf = rdf.Define("score_PN", score_formula)
                        if upper_threshold is not None:
                            rdf = rdf.Filter(f"score_PN < {upper_threshold}")
                        elif threshold > -999.:
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
    """Determine final process list based on validation."""
    # Nonprompt is always kept separate because its transfer-factor uncertainty
    # is attached to its own column. Conversion can be merged into `others`:
    # ConvSF is baked in as a per-sample K-factor at preprocess time, and the
    # CMS_B2G25013_Norm_conversion_* lnN naturally drops out when conversion
    # no longer has its own column (group filter in printDatacard.py).
    always_separate = ["nonprompt"]
    separate_processes = ["nonprompt"]  # nonprompt is always first
    merged_to_others = []

    for category in background_categories:
        # Map category name to output file name
        process = "conversion" if category == "conv" else category

        if process in always_separate:
            if process not in separate_processes:
                separate_processes.append(process)
            logging.info(f"  {process}: always kept separate (dedicated normalization)")
        elif process in validation_results and validation_results[process]["decision"] == "keep":
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


def getBackgroundWeights(basedir, mass_min, mass_max, outdir):
    """Calculate normalized background class weights for ParticleNet."""
    logging.info("Calculating background cross-section weights:")

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


def loadDataset(basedir, process, masspoint, mass_min, mass_max, bg_weights=None):
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

    # Validate unblind options
    if args.unblind and args.partial_unblind:
        raise ValueError("--unblind and --partial-unblind are mutually exclusive")
    if args.partial_unblind and args.method != "ParticleNet":
        raise ValueError("--partial-unblind requires --method ParticleNet")

    workdir = os.getenv("WORKDIR")
    if not workdir:
        raise EnvironmentError("WORKDIR environment variable not set. Please run 'source setup.sh'")

    # Paths
    basedir = f"{workdir}/SignalRegionStudyV2/samples/{args.era}/{args.channel}/{args.masspoint}"
    binning_suffix = args.binning
    if args.unblind:
        binning_suffix = f"{args.binning}_unblind"
    elif args.partial_unblind:
        binning_suffix = f"{args.binning}_partial_unblind"
    outdir = f"{workdir}/SignalRegionStudyV2/templates/{args.era}/{args.channel}/{args.masspoint}/{args.method}/{binning_suffix}"

    logging.info(f"Starting template generation")
    logging.info(f"  Mass point: {args.masspoint}")
    logging.info(f"  Era: {args.era}")
    logging.info(f"  Channel: {args.channel}")
    logging.info(f"  Method: {args.method}")
    logging.info(f"  Binning: {args.binning}")
    logging.info(f"Input directory: {basedir}")
    logging.info(f"Output directory: {outdir}")

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
    # A Mass Fitting (Double Crystal Ball)
    # ========================================
    logging.info("=" * 60)
    logging.info("A Mass Fitting (DCB)")
    logging.info("=" * 60)

    signal_path = f"{basedir}/{args.masspoint}.root"
    fit_result = getFitResultDCB(signal_path, mA_nominal, outdir, args.era, args.masspoint, args.channel)
    save_json(fit_result, f"{outdir}/signal_fit.json")

    x0 = fit_result["x0"]
    sigma_eff = fit_result["sigma_eff"]
    mass_min = max(x0 - 10.0 * sigma_eff, 12.0)
    mass_max = x0 + 10.0 * sigma_eff

    # ========================================
    # Initial Binning (will be refined after background validation)
    # ========================================
    logging.info("=" * 60)
    logging.info(f"Initial binning with sigma_eff = {sigma_eff:.3f} GeV")
    logging.info("=" * 60)

    # Initial bin edges — will be refined by adaptive binning loop below
    bin_edges = calculate_adaptive_bins(x0, sigma_eff, 15)
    logging.info(f"Initial binning: {len(bin_edges)-1} bins (15 core + 2 sideband)")

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
    upper_threshold = None

    # For partial-unblind, use upper threshold of 0.3 (no lower cut)
    if args.partial_unblind:
        upper_threshold = 0.3
        logging.info(f"Partial-unblind mode: using upper_threshold = {upper_threshold}")
        logging.info(f"Partial-unblind mode: signal will be scaled by {PARTIAL_UNBLIND_SIGNAL_SCALE}x")

    if args.method == "ParticleNet":
        logging.info("=" * 60)
        logging.info("ParticleNet optimization")
        logging.info("=" * 60)

        bg_weights = getBackgroundWeights(basedir, mass_min, mass_max, outdir)

        # Skip threshold optimization for partial-unblind (using upper_threshold instead)
        if args.partial_unblind:
            logging.info("Skipping threshold optimization for partial-unblind mode")
            # Save partial-unblind configuration for downstream tools
            save_json({
                "masspoint": args.masspoint,
                "upper_threshold": float(upper_threshold),
                "signal_scale_factor": int(PARTIAL_UNBLIND_SIGNAL_SCALE),
                "description": f"Signal scaled by {PARTIAL_UNBLIND_SIGNAL_SCALE}x. Limit on r = limit on {PARTIAL_UNBLIND_SIGNAL_SCALE}x original cross-section."
            }, f"{outdir}/threshold.json")
        else:
            # Load signal dataset
            scores_sig, weights_sig, _ = loadDataset(basedir, args.masspoint, args.masspoint, mass_min, mass_max, bg_weights)

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
                    scores, weights, _ = loadDataset(basedir, process, args.masspoint, mass_min, mass_max, bg_weights)
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
        basedir, bin_edges, mass_min, mass_max,
        background_categories, args.masspoint, best_threshold, upper_threshold,
        bg_weights, min_total_events=1
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
    # Adaptive Binning Loop
    # ========================================
    logging.info("=" * 60)
    logging.info("Adaptive binning optimization...")
    logging.info("=" * 60)

    apply_floor = False
    n_core_final = 15
    others_process_list = ["others"] + merged_to_others

    for n_core in [15, 13, 11, 9, 7, 5]:
        candidate_edges = calculate_adaptive_bins(x0, sigma_eff, n_core)
        logging.info(f"Testing {n_core} core bins ({n_core + 2} total)...")

        # Fill central backgrounds with candidate binning
        test_hists = {}
        for process in separate_processes:
            try:
                h = getHist(basedir, process, candidate_edges, mass_min, mass_max,
                            "Central", best_threshold, upper_threshold, bg_weights, args.masspoint)
                ensure_positive_integral(h)
                test_hists[process] = h
            except (FileNotFoundError, RuntimeError):
                pass

        # Others (merged)
        try:
            h_others = getHistMerged(basedir, others_process_list, candidate_edges, mass_min, mass_max,
                                     "Central", best_threshold, upper_threshold, bg_weights, args.masspoint)
            ensure_positive_integral(h_others)
            test_hists["others"] = h_others
        except (FileNotFoundError, RuntimeError):
            pass

        ok, diagnostics = check_binning_quality(test_hists)

        if ok:
            bin_edges = candidate_edges
            n_core_final = n_core
            logging.info(f"  PASS: {n_core} core bins accepted")
            break
        else:
            logging.info(f"  FAIL: {len(diagnostics)} issues")
            for d in diagnostics[:5]:
                logging.info(f"    {d}")
            if len(diagnostics) > 5:
                logging.info(f"    ... and {len(diagnostics) - 5} more")
    else:
        # 5 core bins still failed — keep 5 bins and apply floor
        bin_edges = candidate_edges
        n_core_final = 5
        apply_floor = True
        logging.warning(f"All bin counts failed. Keeping 5 core bins with floor applied.")

    logging.info(f"Final binning: {n_core_final} core + 2 sideband = {len(bin_edges)-1} total bins")

    # ========================================
    # Build histogram templates in memory
    # ========================================
    logging.info("=" * 60)
    logging.info("Building histogram templates...")
    logging.info("=" * 60)

    # templates["data_obs"] -> TH1; templates[<process>] -> {"nominal"|"<syst>Up"|"<syst>Down": TH1}
    templates = {}

    # Initialize data_obs histogram
    nbins = len(bin_edges) - 1
    bin_edges_vector = ROOT.std.vector['double'](bin_edges)

    if args.unblind or args.partial_unblind:
        # Use real data for data_obs
        logging.info("Using real data for data_obs")
        data_obs = getDataHist(basedir, bin_edges, mass_min, mass_max,
                               best_threshold, upper_threshold, bg_weights, args.masspoint)
    else:
        # Initialize empty histogram to sum backgrounds
        data_obs = ROOT.TH1D("data_obs", "data_obs", nbins, bin_edges_vector.data())
        data_obs.SetDirectory(0)

    background_hists = {}

    # ========================================
    # Process Signal
    # ========================================
    logging.info(f"Processing signal: {args.masspoint}")

    # Detect if Run3 signal is scaled from Run2 (2018)
    signal_file_path = f"{basedir}/{args.masspoint}.root"
    signal_scaled_from_run2 = is_signal_scaled_from_run2(signal_file_path, args.era)
    if signal_scaled_from_run2:
        logging.info(f"  Signal detected as scaled from Run2 (2018) - will remap systematic tree names")

    signal_map = {}
    templates[args.masspoint] = signal_map

    # Central histogram
    hist_signal_central = getHist(basedir, args.masspoint, bin_edges, mass_min, mass_max,
                                   "Central", best_threshold, upper_threshold, bg_weights, args.masspoint)
    # Scale signal for partial-unblind mode
    if args.partial_unblind:
        hist_signal_central.Scale(PARTIAL_UNBLIND_SIGNAL_SCALE)
    ensure_positive_integral(hist_signal_central)
    signal_map["nominal"] = hist_signal_central

    # Preprocessed shape systematics (Up/Down pairs, list or dict form)
    for syst_name, variations, group in syst_categories['preprocessed_shape']:
        if "signal" not in group:
            continue
        logging.debug(f"  Processing signal systematic: {syst_name}")
        for direction in iter_shape_directions(variations):
            combine_suffix = f"{syst_name}{direction}"

            if signal_scaled_from_run2:
                read_tree = get_run2_tree_name_for_run3_syst(syst_name, direction, args.era)
                logging.debug(f"    Remapped: {combine_suffix} -> {read_tree}")
            else:
                read_tree = f"{syst_name}_{direction}"

            try:
                hist = getHist(basedir, args.masspoint, bin_edges, mass_min, mass_max,
                              read_tree, best_threshold, upper_threshold, bg_weights, args.masspoint)
                hist.SetName(f"{args.masspoint}_{combine_suffix}")
                if args.partial_unblind:
                    hist.Scale(PARTIAL_UNBLIND_SIGNAL_SCALE)
                ensure_positive_integral(hist)
                signal_map[combine_suffix] = hist
            except (FileNotFoundError, RuntimeError) as e:
                logging.warning(f"    Skipping {syst_name}/{direction}: {e}")

    # Valued shape systematics (created by scaling Central histogram)
    # Note: hist_signal_central is already scaled for partial-unblind, so the scaled
    # histograms created from it will also have the correct scaling applied
    for syst_name, value, group in syst_categories['valued_shape']:
        if "signal" not in group:
            continue
        logging.debug(f"  Processing signal valued systematic: {syst_name} (value={value})")
        for direction in ["up", "down"]:
            hist = create_scaled_hist(hist_signal_central, args.masspoint, syst_name, value, direction)
            ensure_positive_integral(hist)
            suffix = f"{syst_name}Up" if direction == "up" else f"{syst_name}Down"
            signal_map[suffix] = hist

    # Multi-variation systematics (PDF envelopes; QCD scale moved to preprocessed_shape)
    for syst_name, variations, group in syst_categories['multi_variation']:
        if "signal" not in group:
            continue
        logging.info(f"  Creating envelope for signal: {syst_name}")

        # Map variation names to preprocessed tree names
        tree_variations = []
        for var in variations:
            if var.startswith("pdf_"):
                num = int(var.replace("pdf_", ""))
                tree_variations.append(f"PDF_{num}")
            else:
                tree_variations.append(var)

        try:
            hist_up, hist_down = createEnvelopeHists(
                basedir, args.masspoint, bin_edges, mass_min, mass_max,
                tree_variations, syst_name, best_threshold, upper_threshold, bg_weights, args.masspoint
            )
            if args.partial_unblind:
                hist_up.Scale(PARTIAL_UNBLIND_SIGNAL_SCALE)
                hist_down.Scale(PARTIAL_UNBLIND_SIGNAL_SCALE)
            ensure_positive_integral(hist_up)
            ensure_positive_integral(hist_down)
            signal_map[f"{syst_name}Up"] = hist_up
            signal_map[f"{syst_name}Down"] = hist_down
        except RuntimeError as e:
            logging.warning(f"    Skipping envelope {syst_name}: {e}")

    logging.info(f"Signal templates created: {args.masspoint} (integral = {hist_signal_central.Integral():.4f})")

    # ========================================
    # Process Backgrounds (Separate)
    # ========================================
    for process in separate_processes:
        logging.info(f"Processing {process} background (separate template)")
        proc_map = {}
        templates[process] = proc_map

        # Central histogram
        hist_central = getHist(basedir, process, bin_edges, mass_min, mass_max,
                               "Central", best_threshold, upper_threshold, bg_weights, args.masspoint)
        ensure_positive_integral(hist_central)
        if not (args.unblind or args.partial_unblind):
            data_obs.Add(hist_central)
        proc_map["nominal"] = hist_central
        background_hists[process] = hist_central

        # Determine which systematics apply to this process
        if process == "nonprompt":
            # Nonprompt-specific valued_shape systematics (scaled from Central)
            for syst_name, value, grp in syst_categories['valued_shape']:
                if "nonprompt" not in grp:
                    continue
                for direction in ["up", "down"]:
                    hist = create_scaled_hist(hist_central, process, syst_name, value, direction)
                    ensure_positive_integral(hist)
                    suffix = f"{syst_name}Up" if direction == "up" else f"{syst_name}Down"
                    proc_map[suffix] = hist
        else:
            # Prompt systematics - use process name directly
            # Note: systematics config uses "conversion" in groups, not "conv"
            for syst_name, variations, group in syst_categories['preprocessed_shape']:
                if process not in group:
                    continue
                for var in variations:
                    output_tree = get_output_tree_name(syst_name, var)
                    try:
                        hist = getHist(basedir, process, bin_edges, mass_min, mass_max,
                                      output_tree, best_threshold, upper_threshold, bg_weights, args.masspoint)
                        ensure_positive_integral(hist)
                        variation_suffix = combine_suffix_from_tree(output_tree)
                        hist.SetName(f"{process}_{variation_suffix}")
                        proc_map[variation_suffix] = hist
                    except (FileNotFoundError, RuntimeError) as e:
                        logging.warning(f"    Skipping {process}/{syst_name}/{var}: {e}")

            # Valued shape systematics (created by scaling Central histogram)
            for syst_name, value, group in syst_categories['valued_shape']:
                if process not in group:
                    continue
                for direction in ["up", "down"]:
                    hist = create_scaled_hist(hist_central, process, syst_name, value, direction)
                    ensure_positive_integral(hist)
                    suffix = f"{syst_name}Up" if direction == "up" else f"{syst_name}Down"
                    proc_map[suffix] = hist

        logging.info(f"  {process} templates created (integral = {hist_central.Integral():.4f})")

    # ========================================
    # Process "others" Background (Merged)
    # ========================================
    logging.info("Processing others background (merged template)")
    logging.info(f"  Merging processes: {others_process_list}")

    others_map = {}
    templates["others"] = others_map

    # Central histogram
    hist_others = getHistMerged(basedir, others_process_list, bin_edges, mass_min, mass_max,
                                "Central", best_threshold, upper_threshold, bg_weights, args.masspoint)
    ensure_positive_integral(hist_others)
    if not (args.unblind or args.partial_unblind):
        data_obs.Add(hist_others)
    others_map["nominal"] = hist_others
    background_hists["others"] = hist_others

    # Prompt systematics for others.
    # Only apply a systematic to the merged `others` template if its group
    # explicitly contains "others". Per-process normalization systs (e.g.
    # Norm_WZ with group=["WZ"]) are intentionally skipped — they would fail
    # to build on the merged template anyway, and Norm_others (50% lnN)
    # covers the merged bucket's normalization uncertainty.
    for syst_name, variations, group in syst_categories['preprocessed_shape']:
        if "others" not in group:
            continue

        for var in variations:
            output_tree = get_output_tree_name(syst_name, var)
            try:
                hist = getHistMerged(basedir, others_process_list, bin_edges, mass_min, mass_max,
                                    output_tree, best_threshold, upper_threshold, bg_weights, args.masspoint)
                ensure_positive_integral(hist)
                variation_suffix = combine_suffix_from_tree(output_tree)
                hist.SetName(f"others_{variation_suffix}")
                others_map[variation_suffix] = hist
            except (FileNotFoundError, RuntimeError) as e:
                logging.warning(f"    Skipping others/{syst_name}/{var}: {e}")

    # Valued shape systematics (created by scaling merged Central histogram)
    for syst_name, value, group in syst_categories['valued_shape']:
        if "others" not in group:
            continue

        for direction in ["up", "down"]:
            hist = create_scaled_hist(hist_others, "others", syst_name, value, direction)
            ensure_positive_integral(hist)
            suffix = f"{syst_name}Up" if direction == "up" else f"{syst_name}Down"
            others_map[suffix] = hist

    logging.info(f"  Others templates created (integral = {hist_others.Integral():.4f})")

    templates["data_obs"] = data_obs

    # ========================================
    # Pre-merge: apply floor if adaptive binning exhausted its options
    # ========================================
    if apply_floor:
        logging.warning("Applying bin floor to empty/problematic bins (nominal + syst variations)")
        for process in separate_processes + ["others"]:
            proc_map = templates.get(process, {})
            patched = 0
            for h in proc_map.values():
                if not h:
                    continue
                modified = False
                for i in range(1, h.GetNbinsX() + 1):
                    if h.GetBinContent(i) <= 0:
                        h.SetBinContent(i, BIN_FLOOR_VALUE)
                        h.SetBinError(i, BIN_FLOOR_VALUE)  # 100% error
                        modified = True
                    elif h.GetBinError(i) / h.GetBinContent(i) > 1.0:
                        h.SetBinError(i, h.GetBinContent(i))  # cap at 100%
                        modified = True
                if modified:
                    patched += 1
            if patched > 0:
                logging.warning(f"  Patched {process}: {patched} histogram(s)")

    # ========================================
    # Post-binning syst-driven bin merging
    # ========================================
    logging.info("=" * 60)
    logging.info("Post-binning syst-driven bin merging...")
    logging.info("=" * 60)

    bkg_process_list = list(separate_processes) + ["others"]
    pre_merge_nbins = len(bin_edges) - 1
    bin_edges, templates, n_syst_merges = apply_syst_driven_merging(
        bin_edges, templates, bkg_process_list,
        max_rel_syst=SYST_MERGE_THRESHOLD, logger=logging)
    post_merge_nbins = len(bin_edges) - 1
    if n_syst_merges > 0:
        logging.warning(
            f"Syst-merge: {pre_merge_nbins} bins -> {post_merge_nbins} bins "
            f"({n_syst_merges} merges applied, threshold rel syst = "
            f"{SYST_MERGE_THRESHOLD:.2f})"
        )

    # Refresh references used by the summary log after rebinning.
    hist_signal_central = templates[args.masspoint]["nominal"]
    for process in separate_processes + ["others"]:
        background_hists[process] = templates[process]["nominal"]
    data_obs = templates["data_obs"]

    # ========================================
    # Persist binning.json with final edges
    # ========================================
    save_json({
        "nbins": len(bin_edges) - 1,
        "bin_edges": [float(e) for e in bin_edges],
        "method": "AdaptiveExtendedBins",
        "sigma_eff": float(sigma_eff),
        "mass_min": float(mass_min),
        "mass_max": float(mass_max),
        "binning_type": args.binning,
        "n_core_bins": n_core_final,
        "fit_model": "dcb",
        "floor_applied": apply_floor,
        "syst_merge_applied": n_syst_merges > 0,
        "n_bins_merged": int(n_syst_merges),
        "syst_merge_threshold": SYST_MERGE_THRESHOLD,
    }, f"{outdir}/binning.json")

    # ========================================
    # Write all templates to shapes.root
    # ========================================
    logging.info("=" * 60)
    logging.info("Writing shapes.root...")
    logging.info("=" * 60)
    output_file = ROOT.TFile(f"{outdir}/shapes.root", "RECREATE")
    output_file.cd()

    # data_obs
    data_source = (
        "real data" + (" (score < 0.3)" if args.partial_unblind else "")
        if (args.unblind or args.partial_unblind)
        else "sum of all backgrounds"
    )
    logging.info(f"  data_obs ({data_source}, integral = {data_obs.Integral():.4f})")
    data_obs.Write("data_obs")

    process_order = [args.masspoint] + list(separate_processes) + ["others"]
    for process in process_order:
        proc_map = templates.get(process, {})
        nominal = proc_map.get("nominal")
        if nominal is None:
            continue
        nominal.Write(process)
        for key, hist in proc_map.items():
            if key == "nominal":
                continue
            hist.Write(f"{process}_{key}")

    output_file.Close()

    # ========================================
    # Summary
    # ========================================
    logging.info("=" * 60)
    logging.info("Template generation complete!")
    logging.info(f"Output file: {outdir}/shapes.root")
    logging.info("=" * 60)
    logging.info("Process yields:")
    signal_label = f"Signal ({args.masspoint})"
    if args.partial_unblind:
        signal_label += f" [x{PARTIAL_UNBLIND_SIGNAL_SCALE}]"
    logging.info(f"  {signal_label}:  {hist_signal_central.Integral():>10.4f}")

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
