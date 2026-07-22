#!/usr/bin/env python3
"""
Generate binned histogram templates for HiggsCombine.

Usage:
    python makeBinnedTemplates.py --era Run2 --channel Combined --masspoint MHc130_MA90 --method Baseline
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
    save_json,
    ensure_positive_integral, cap_stat_errors, build_particlenet_score,
    create_scaled_hist, categorize_systematics,
    calculate_adaptive_bins, check_binning_quality, apply_syst_driven_merging,
    iter_shape_directions
)
from run_period_utils import (
    PHYSICS_PROCESS_ORDER,
    category_name,
    component_name,
    resolve_channels,
    resolve_run_periods,
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

MIN_CORE_BINS = 5


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate binned histogram templates for HiggsCombine")
    parser.add_argument("--era", required=True, type=str, help="Run-period target: Run2, Run3, or All")
    parser.add_argument("--channel", required=True, type=str, help="Analysis channel (SR1E2Mu, SR3Mu, Combined)")
    parser.add_argument("--masspoint", required=True, type=str, help="Signal mass point (e.g., MHc130_MA90)")
    parser.add_argument("--method", required=True, type=str, help="Template method (Baseline, ParticleNet, etc.)")
    parser.add_argument("--binning", default="extended",
                        choices=["extended", "uniform"],
                        help=("Binning method: 'extended' uses adaptive coarser "
                              "binning down to 5 core bins; 'uniform' keeps the "
                              "fixed candidate set"))
    parser.add_argument("--unblind", action="store_true",
                        help="Use real data for data_obs instead of MC sum")
    parser.add_argument("--partial-unblind", action="store_true", dest="partial_unblind",
                        help="Unblind low LR region (score < 0.3). Requires --method ParticleNet")
    parser.add_argument("--nuisance", default="fallback_lnn",
                        choices=["fallback_lnn", "preserve_shape"],
                        help="Low-stat nuisance handling mode used to choose the output suffix")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

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
    """Validate statistical quality of each background process.

    Returns a dict of {process: {total_events, status, reason}} where status is
    one of: 'present', 'low_stat', 'empty', 'missing_file'.

    All processes except missing-file cases are kept as separate columns
    regardless of yield.
    """
    logging.info("Validating background statistics...")
    nbins = len(bin_edges) - 1
    results = {}

    for category in background_categories:
        # Map category name to output file name
        process = "conversion" if category == "conv" else category
        logging.info(f"  Validating {process}...")

        file_path = f"{basedir}/{process}.root"
        if not os.path.exists(file_path):
            logging.error(f"    {process}: sample file not found — will be dropped from this era/channel")
            results[process] = {
                "total_events": 0,
                "status": "missing_file",
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

        if total_events <= 0:
            status = "empty"
            reason = f"total events ({total_events:.2f}) <= 0"
        elif total_events < min_total_events:
            status = "low_stat"
            reason = f"total events ({total_events:.2f}) < {min_total_events} (low statistics)"
            logging.warning(f"    {process}: {reason} — kept as separate column; "
                            f"low-stat handling (floor + shape→lnN fallback) will apply")
        else:
            status = "present"
            reason = "passes statistical requirements"

        results[process] = {
            "total_events": total_events,
            "status": status,
            "reason": reason
        }

        logging.info(f"    Total events: {total_events:.2f}, status: {status}")

    return results


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
# Run-period component template construction
# =============================================================================

def make_binning_suffix(args):
    suffix = args.binning
    if args.unblind:
        suffix = f"{args.binning}_unblind"
    elif args.partial_unblind:
        suffix = f"{args.binning}_partial_unblind"
    if args.nuisance == "preserve_shape":
        suffix = f"{suffix}_preserve_shape"
    return suffix


def sample_basedir(workdir, era, channel, masspoint):
    return f"{workdir}/SignalRegionStudyV3/samples/{era}/{channel}/{masspoint}"


def load_systematics_block(workdir, era, channel):
    config_path = f"{workdir}/SignalRegionStudyV3/configs/systematics.{era}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Systematics config not found: {config_path}")
    with open(config_path) as f:
        config = json.load(f)
    if channel not in config:
        raise ValueError(f"Channel '{channel}' not found in {config_path}")
    return config[channel]


def load_sample_block(workdir, era, channel):
    samplegroups_path = f"{workdir}/SignalRegionStudyV3/configs/samplegroups.json"
    if not os.path.exists(samplegroups_path):
        raise FileNotFoundError(f"Sample groups config not found: {samplegroups_path}")
    with open(samplegroups_path) as f:
        samplegroups = json.load(f)
    if era not in samplegroups:
        raise ValueError(f"Era '{era}' not found in {samplegroups_path}")
    if channel not in samplegroups[era]:
        raise ValueError(f"Channel '{channel}' not found for era '{era}'")
    return samplegroups[era][channel]


def fit_dcb(chain, mA_nominal):
    """Two-stage Double Crystal Ball fit on a ``TChain('Central')`` holding
    ``mass`` and ``weight`` branches.

    Returns the fitted parameters plus the narrow fit window (``fit_lo``,
    ``fit_hi``). This is the pure-fit core shared by ``getFitResultDCBMulti``
    (which adds diagnostic plotting) and the standalone signal-efficiency tool;
    it has no plotting side effects.
    """
    if chain.GetEntries() <= 0:
        raise RuntimeError("No signal entries found for DCB fit")

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

    wide_lo = max(mA_nominal - mA_nominal / 3.0, 12.0)
    wide_hi = mA_nominal + mA_nominal / 3.0

    mass_w = ROOT.RooRealVar("mass", "mass", wide_lo, wide_hi)
    weight_w = ROOT.RooRealVar("weight", "weight", -10, 10)
    ds_wide = ROOT.RooDataSet(
        "ds_wide", "", ROOT.RooArgSet(mass_w, weight_w),
        ROOT.RooFit.Import(chain),
        ROOT.RooFit.Cut(f"mass >= {wide_lo} && mass <= {wide_hi}"),
        ROOT.RooFit.WeightVar("weight"),
    )

    pre_x0 = ROOT.RooRealVar("pre_x0", "x0", mA_nominal, wide_lo, wide_hi)
    pre_sL = ROOT.RooRealVar("pre_sL", "sL", 1.0, 0.01, 10.0)
    pre_sR = ROOT.RooRealVar("pre_sR", "sR", 1.0, 0.01, 10.0)
    pre_aL = ROOT.RooRealVar("pre_aL", "aL", 1.5, 0.5, 10.0)
    pre_nL = ROOT.RooRealVar("pre_nL", "nL", 2.0, 0.1, 50.0)
    pre_aR = ROOT.RooRealVar("pre_aR", "aR", 1.5, 0.5, 10.0)
    pre_nR = ROOT.RooRealVar("pre_nR", "nR", 2.0, 0.1, 50.0)
    pre_dcb = ROOT.RooCrystalBall(
        "pre_dcb", "", mass_w, pre_x0,
        pre_sL, pre_sR, pre_aL, pre_nL, pre_aR, pre_nR
    )
    pre_dcb.fitTo(ds_wide, ROOT.RooFit.SumW2Error(True),
                  ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

    fitted_mA = pre_x0.getVal()
    vw = sqrt(0.5 * (pre_sL.getVal()**2 + pre_sR.getVal()**2))

    fit_lo = max(fitted_mA - 10.0 * vw, 12.0)
    fit_hi = fitted_mA + 10.0 * vw

    mass_n = ROOT.RooRealVar("mass", "mass", fit_lo, fit_hi)
    weight_n = ROOT.RooRealVar("weight", "weight", -10, 10)
    ds_narrow = ROOT.RooDataSet(
        "ds_narrow", "", ROOT.RooArgSet(mass_n, weight_n),
        ROOT.RooFit.Import(chain),
        ROOT.RooFit.Cut(f"mass >= {fit_lo} && mass <= {fit_hi}"),
        ROOT.RooFit.WeightVar("weight"),
    )

    dcb_x0 = ROOT.RooRealVar("dcb_x0", "x0", fitted_mA, fit_lo, fit_hi)
    dcb_sL = ROOT.RooRealVar("dcb_sL", "sigmaL", 0.8 * vw, 0.01 * vw, 3.0 * vw)
    dcb_sR = ROOT.RooRealVar("dcb_sR", "sigmaR", 0.8 * vw, 0.01 * vw, 3.0 * vw)
    dcb_aL = ROOT.RooRealVar("dcb_aL", "alphaL", 1.5, 0.5, 10.0)
    dcb_nL = ROOT.RooRealVar("dcb_nL", "nL", 2.0, 0.1, 50.0)
    dcb_aR = ROOT.RooRealVar("dcb_aR", "alphaR", 1.5, 0.5, 10.0)
    dcb_nR = ROOT.RooRealVar("dcb_nR", "nR", 2.0, 0.1, 50.0)
    dcb = ROOT.RooCrystalBall(
        "dcb", "", mass_n, dcb_x0,
        dcb_sL, dcb_sR, dcb_aL, dcb_nL, dcb_aR, dcb_nR
    )
    dcb.fitTo(ds_narrow, ROOT.RooFit.SumW2Error(True),
              ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

    sigma_eff = sqrt(0.5 * (dcb_sL.getVal()**2 + dcb_sR.getVal()**2))

    return {
        "x0": float(dcb_x0.getVal()),
        "sigmaL": float(dcb_sL.getVal()),
        "sigmaR": float(dcb_sR.getVal()),
        "alphaL": float(dcb_aL.getVal()),
        "nL": float(dcb_nL.getVal()),
        "alphaR": float(dcb_aR.getVal()),
        "nR": float(dcb_nR.getVal()),
        "sigma_eff": float(sigma_eff),
        "fit_lo": float(fit_lo),
        "fit_hi": float(fit_hi),
    }


def getFitResultDCBMulti(input_paths, mA_nominal, outdir, era, masspoint, channel=""):
    """Fit the summed/appended signal trees for one merged Run-period category."""
    if not input_paths:
        raise FileNotFoundError(f"No signal files provided for {era}/{channel}/{masspoint}")
    missing = [path for path in input_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing signal file(s): " + ", ".join(missing))

    logging.info(f"Fitting category signal with DCB, nominal mA = {mA_nominal} GeV")

    chain = ROOT.TChain("Central")
    for path in input_paths:
        chain.Add(path)
    if chain.GetEntries() <= 0:
        raise RuntimeError(f"No signal entries found for {era}/{channel}/{masspoint}")

    fit = fit_dcb(chain, mA_nominal)
    fit_lo = fit["fit_lo"]
    fit_hi = fit["fit_hi"]

    hist_name = f"h_fit_{era}_{channel}".replace("-", "_")
    hist = ROOT.TH1D(hist_name, "", 100, fit_lo, fit_hi)
    hist.Sumw2()
    chain.Draw(f"mass>>{hist_name}", f"weight*(mass >= {fit_lo} && mass <= {fit_hi})", "goff")
    hist.SetDirectory(0)

    mass_plot = ROOT.RooRealVar("mass_plot", "mass", fit_lo, fit_hi)
    roo_data = ROOT.RooDataHist("data_plot", "", ROOT.RooArgList(mass_plot), hist)

    p_x0 = ROOT.RooRealVar("p_x0", "x0", fit["x0"])
    p_sL = ROOT.RooRealVar("p_sL", "sL", fit["sigmaL"])
    p_sR = ROOT.RooRealVar("p_sR", "sR", fit["sigmaR"])
    p_aL = ROOT.RooRealVar("p_aL", "aL", fit["alphaL"])
    p_nL = ROOT.RooRealVar("p_nL", "nL", fit["nL"])
    p_aR = ROOT.RooRealVar("p_aR", "aR", fit["alphaR"])
    p_nR = ROOT.RooRealVar("p_nR", "nR", fit["nR"])
    for var in [p_x0, p_sL, p_sR, p_aL, p_nL, p_aR, p_nR]:
        var.setConstant(True)
    dcb_plot = ROOT.RooCrystalBall(
        "dcb_plot", "", mass_plot, p_x0,
        p_sL, p_sR, p_aL, p_nL, p_aR, p_nR
    )

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
    canvas.canv.SaveAs(f"{outdir}/signal_fit.{category_name(channel, era)}.png")

    result = {
        "x0": fit["x0"],
        "sigmaL": fit["sigmaL"],
        "sigmaR": fit["sigmaR"],
        "alphaL": fit["alphaL"],
        "nL": fit["nL"],
        "alphaR": fit["alphaR"],
        "nR": fit["nR"],
        "sigma_eff": fit["sigma_eff"],
    }
    logging.info(
        "Category DCB fit result: x0=%.2f, sigma_eff=%.3f GeV",
        result["x0"], result["sigma_eff"]
    )
    return result


def getCategoryBackgroundWeights(basedirs, mass_min, mass_max, outdir, category):
    """Calculate ParticleNet background weights over all suberas in one category."""
    logging.info("Calculating category background weights for %s", category)
    weights = {}
    for pn_class, possible_categories in PARTICLENET_CLASS_MAPPING.items():
        total_weight = 0.0
        found_any = False
        for process in possible_categories:
            for basedir in basedirs:
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
        weights[pn_class] = total_weight if found_any else 1.0 / 3.0

    total = sum(weights.values())
    if total > 0:
        weights = {key: val / total for key, val in weights.items()}
    else:
        weights = {key: 1.0 / 3.0 for key in weights}

    save_json({
        "category": category,
        "weights": weights,
        "mass_window": [float(mass_min), float(mass_max)],
    }, f"{outdir}/background_weights.{category}.json")
    return weights


def optimizeCategoryParticleNetThreshold(basedirs, masspoint, mass_min, mass_max, bg_weights, outdir, category):
    sig_scores = []
    sig_weights = []
    for basedir in basedirs:
        scores, weights, _ = loadDataset(basedir, masspoint, masspoint, mass_min, mass_max, bg_weights)
        if len(scores) > 0:
            sig_scores.append(scores)
            sig_weights.append(weights)
    if not sig_scores:
        logging.warning("ParticleNet signal scores not found for %s; no threshold applied", category)
        return -999., None

    bkg_processes = ["nonprompt", "WZ", "ZZ", "ttW", "ttZ", "ttH", "tZq", "conversion", "others"]
    bkg_scores = []
    bkg_weights = []
    for basedir in basedirs:
        for process in bkg_processes:
            scores, weights, _ = loadDataset(basedir, process, masspoint, mass_min, mass_max, bg_weights)
            if len(scores) > 0:
                bkg_scores.append(scores)
                bkg_weights.append(weights)

    if not bkg_scores:
        logging.warning("ParticleNet background scores not found for %s; no threshold applied", category)
        return -999., None

    best_threshold, initial_sensitivity, max_sensitivity = getOptimizedThreshold(
        np.concatenate(sig_scores),
        np.concatenate(sig_weights),
        np.concatenate(bkg_scores),
        np.concatenate(bkg_weights),
    )
    payload = {
        "category": category,
        "masspoint": masspoint,
        "threshold": float(best_threshold),
        "initial_sensitivity": float(initial_sensitivity),
        "max_sensitivity": float(max_sensitivity),
        "improvement": float(max_sensitivity / initial_sensitivity - 1) if initial_sensitivity > 0 else 0.0,
    }
    save_json(payload, f"{outdir}/threshold.{category}.json")
    return best_threshold, payload


def category_background_processes(workdir, suberas, channel):
    """Return stable background process names and per-subera validation metadata."""
    processes = []
    by_subera = {}
    for subera in suberas:
        samples = load_sample_block(workdir, subera, channel)
        reserved = {"data"}
        subera_processes = []
        for key in PHYSICS_PROCESS_ORDER:
            if key == "signal" or key in reserved:
                continue
            if key in samples:
                subera_processes.append(key)
                if key not in processes:
                    processes.append(key)
        by_subera[subera] = subera_processes
    return processes, by_subera


def hist_exists(path, tree="Central"):
    if not os.path.exists(path):
        return False
    f = ROOT.TFile.Open(path, "READ")
    ok = bool(f and not f.IsZombie() and f.Get(tree))
    if f:
        f.Close()
    return ok


def build_component_shape_templates(basedir, base_process, component, bin_edges,
                                    mass_min, mass_max, syst_categories,
                                    threshold, upper_threshold, bg_weights,
                                    masspoint, floor_mode):
    proc_map = {}
    central = getHist(
        basedir, base_process, bin_edges, mass_min, mass_max,
        "Central", threshold, upper_threshold, bg_weights, masspoint
    )
    central.SetName(component)
    ensure_positive_integral(central, floor_mode=floor_mode)
    cap_stat_errors(central)
    proc_map["nominal"] = central

    if base_process == "nonprompt":
        for syst_name, value, group in syst_categories["valued_shape"]:
            if "nonprompt" not in group:
                continue
            for direction in ["up", "down"]:
                hist = create_scaled_hist(central, component, syst_name, value, direction)
                ensure_positive_integral(hist, floor_mode=floor_mode)
                cap_stat_errors(hist)
                suffix = f"{syst_name}Up" if direction == "up" else f"{syst_name}Down"
                proc_map[suffix] = hist
        return proc_map

    for syst_name, variations, group in syst_categories["preprocessed_shape"]:
        if base_process not in group:
            continue
        for direction in iter_shape_directions(variations):
            read_tree = f"{syst_name}_{direction}"
            combine_suffix = f"{syst_name}{direction}"
            try:
                hist = getHist(
                    basedir, base_process, bin_edges, mass_min, mass_max,
                    read_tree, threshold, upper_threshold, bg_weights, masspoint
                )
                hist.SetName(f"{component}_{combine_suffix}")
                ensure_positive_integral(hist, floor_mode=floor_mode)
                cap_stat_errors(hist)
                proc_map[combine_suffix] = hist
            except (FileNotFoundError, RuntimeError) as exc:
                logging.warning("    Skipping %s/%s/%s: %s", component, syst_name, direction, exc)

    for syst_name, value, group in syst_categories["valued_shape"]:
        if base_process not in group:
            continue
        for direction in ["up", "down"]:
            hist = create_scaled_hist(central, component, syst_name, value, direction)
            ensure_positive_integral(hist, floor_mode=floor_mode)
            cap_stat_errors(hist)
            suffix = f"{syst_name}Up" if direction == "up" else f"{syst_name}Down"
            proc_map[suffix] = hist

    return proc_map


def build_signal_component_templates(basedir, component, bin_edges, mass_min, mass_max,
                                     syst_categories, threshold, upper_threshold,
                                     bg_weights, masspoint, partial_unblind):
    proc_map = {}
    central = getHist(
        basedir, masspoint, bin_edges, mass_min, mass_max,
        "Central", threshold, upper_threshold, bg_weights, masspoint
    )
    central.SetName(component)
    if partial_unblind:
        central.Scale(PARTIAL_UNBLIND_SIGNAL_SCALE)
    ensure_positive_integral(central)
    cap_stat_errors(central)
    proc_map["nominal"] = central

    for syst_name, variations, group in syst_categories["preprocessed_shape"]:
        if "signal" not in group:
            continue
        for direction in iter_shape_directions(variations):
            read_tree = f"{syst_name}_{direction}"
            combine_suffix = f"{syst_name}{direction}"
            try:
                hist = getHist(
                    basedir, masspoint, bin_edges, mass_min, mass_max,
                    read_tree, threshold, upper_threshold, bg_weights, masspoint
                )
                hist.SetName(f"{component}_{combine_suffix}")
                if partial_unblind:
                    hist.Scale(PARTIAL_UNBLIND_SIGNAL_SCALE)
                ensure_positive_integral(hist)
                cap_stat_errors(hist)
                proc_map[combine_suffix] = hist
            except (FileNotFoundError, RuntimeError) as exc:
                logging.warning("    Skipping %s/%s/%s: %s", component, syst_name, direction, exc)

    for syst_name, value, group in syst_categories["valued_shape"]:
        if "signal" not in group:
            continue
        for direction in ["up", "down"]:
            hist = create_scaled_hist(central, component, syst_name, value, direction)
            ensure_positive_integral(hist)
            cap_stat_errors(hist)
            suffix = f"{syst_name}Up" if direction == "up" else f"{syst_name}Down"
            proc_map[suffix] = hist

    for syst_name, variations, group in syst_categories["multi_variation"]:
        if "signal" not in group:
            continue
        tree_variations = []
        for var in variations:
            if var.startswith("pdf_"):
                tree_variations.append(f"PDF_{int(var.replace('pdf_', ''))}")
            else:
                tree_variations.append(var)
        try:
            hist_up, hist_down = createEnvelopeHists(
                basedir, masspoint, bin_edges, mass_min, mass_max,
                tree_variations, syst_name, threshold, upper_threshold, bg_weights, masspoint
            )
            hist_up.SetName(f"{component}_{syst_name}Up")
            hist_down.SetName(f"{component}_{syst_name}Down")
            if partial_unblind:
                hist_up.Scale(PARTIAL_UNBLIND_SIGNAL_SCALE)
                hist_down.Scale(PARTIAL_UNBLIND_SIGNAL_SCALE)
            ensure_positive_integral(hist_up)
            ensure_positive_integral(hist_down)
            cap_stat_errors(hist_up)
            cap_stat_errors(hist_down)
            proc_map[f"{syst_name}Up"] = hist_up
            proc_map[f"{syst_name}Down"] = hist_down
        except RuntimeError as exc:
            logging.warning("    Skipping envelope %s/%s: %s", component, syst_name, exc)

    return proc_map


def write_run_period_shapes(outdir, categories):
    output_file = ROOT.TFile(f"{outdir}/shapes.root", "RECREATE")
    for cat_name, cat_payload in categories.items():
        cat_dir = output_file.mkdir(cat_name)
        cat_dir.cd()
        cat_payload["templates"]["data_obs"].Write("data_obs")
        for process in cat_payload["process_order"]:
            proc_map = cat_payload["templates"].get(process, {})
            nominal = proc_map.get("nominal")
            if nominal is None:
                continue
            nominal.Write(process)
            for key, hist in proc_map.items():
                if key == "nominal":
                    continue
                hist.Write(f"{process}_{key}")
        output_file.cd()
    output_file.Close()


def build_run_period_templates(args, workdir):
    """Build merged Run-period categories with subera component processes."""
    binning_suffix = make_binning_suffix(args)
    outdir = f"{workdir}/SignalRegionStudyV3/templates/{args.era}/{args.channel}/{args.masspoint}/{args.method}/{binning_suffix}"
    periods = resolve_run_periods(args.era)
    channels = resolve_channels(args.channel)
    mA_nominal = float(args.masspoint.split("_")[1].replace("MA", ""))

    logging.info("Starting Run-period component template generation")
    logging.info("  Era request: %s", args.era)
    logging.info("  Channel request: %s", args.channel)
    logging.info("  Mass point: %s", args.masspoint)
    logging.info("  Output directory: %s", outdir)

    if os.path.exists(outdir):
        logging.info("Removing existing output directory: %s", outdir)
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    category_outputs = OrderedDict()
    categories_json = OrderedDict()
    binning_json = {
        "construction": "run_period_components",
        "binning_type": args.binning,
        "min_core_bins": MIN_CORE_BINS,
        "categories": OrderedDict(),
    }
    process_components = []
    background_validation = OrderedDict()
    threshold_summary = {"construction": "run_period_components", "categories": OrderedDict()}
    background_weight_summary = {"construction": "run_period_components", "categories": OrderedDict()}

    for period, suberas in periods:
        for channel in channels:
            cat = category_name(channel, period)
            logging.info("=" * 60)
            logging.info("Building category %s", cat)
            logging.info("=" * 60)

            basedirs = [sample_basedir(workdir, subera, channel, args.masspoint) for subera in suberas]
            signal_paths = [f"{basedir}/{args.masspoint}.root" for basedir in basedirs]
            fit_result = getFitResultDCBMulti(signal_paths, mA_nominal, outdir, period, args.masspoint, channel)
            x0 = fit_result["x0"]
            sigma_eff = fit_result["sigma_eff"]
            mass_min = max(x0 - 10.0 * sigma_eff, 12.0)
            mass_max = x0 + 10.0 * sigma_eff

            bg_weights = None
            best_threshold = -999.
            upper_threshold = None
            if args.partial_unblind:
                upper_threshold = 0.3
                threshold_summary["categories"][cat] = {
                    "upper_threshold": float(upper_threshold),
                    "signal_scale_factor": int(PARTIAL_UNBLIND_SIGNAL_SCALE),
                }
            elif args.method == "ParticleNet":
                bg_weights = getCategoryBackgroundWeights(basedirs, mass_min, mass_max, outdir, cat)
                best_threshold, threshold_payload = optimizeCategoryParticleNetThreshold(
                    basedirs, args.masspoint, mass_min, mass_max, bg_weights, outdir, cat
                )
                if threshold_payload:
                    threshold_summary["categories"][cat] = threshold_payload
                background_weight_summary["categories"][cat] = {"weights": bg_weights}

            all_bg_processes, bg_by_subera = category_background_processes(workdir, suberas, channel)
            background_validation[cat] = OrderedDict()
            active_by_subera = OrderedDict()
            for subera, processes in bg_by_subera.items():
                basedir = sample_basedir(workdir, subera, channel, args.masspoint)
                validation = validateBackgroundStatistics(
                    basedir, calculate_adaptive_bins(x0, sigma_eff, 15), mass_min, mass_max,
                    [p if p != "conversion" else "conv" for p in processes if p not in {"nonprompt", "others"}],
                    args.masspoint, best_threshold, upper_threshold, bg_weights, min_total_events=1
                )
                # Add nonprompt/others explicitly; validateBackgroundStatistics only sees MC categories.
                for proc in ["nonprompt", "others"]:
                    if proc not in processes:
                        continue
                    file_path = f"{basedir}/{proc}.root"
                    status = "present" if hist_exists(file_path) else "missing_file"
                    validation[proc] = {
                        "total_events": 0.0,
                        "status": status,
                        "reason": "checked by file presence for component construction",
                    }
                background_validation[cat][subera] = validation
                active_by_subera[subera] = [
                    proc for proc in processes
                    if validation.get(proc, {}).get("status") != "missing_file"
                ]

            if args.binning == "extended":
                core_bin_candidates = list(range(15, MIN_CORE_BINS - 1, -1))
            else:
                core_bin_candidates = [15, 13, 11, 9, 7, 5]

            apply_floor = False
            n_core_final = core_bin_candidates[-1]
            bin_edges = calculate_adaptive_bins(x0, sigma_eff, n_core_final)
            for n_core in core_bin_candidates:
                candidate_edges = calculate_adaptive_bins(x0, sigma_eff, n_core)
                logging.info("Testing %s with %d core bins", cat, n_core)
                test_hists = {}
                for subera in suberas:
                    basedir = sample_basedir(workdir, subera, channel, args.masspoint)
                    for proc in active_by_subera[subera]:
                        try:
                            comp = component_name(proc, subera)
                            h = getHist(
                                basedir, proc, candidate_edges, mass_min, mass_max,
                                "Central", best_threshold, upper_threshold, bg_weights, args.masspoint
                            )
                            test_hists[comp] = h
                        except (FileNotFoundError, RuntimeError):
                            continue
                ok, diagnostics = check_binning_quality(test_hists)
                if ok:
                    bin_edges = candidate_edges
                    n_core_final = n_core
                    logging.info("  PASS: %d core bins accepted for %s", n_core, cat)
                    break
                logging.info("  FAIL: %d issues", len(diagnostics))
                for diag in diagnostics[:5]:
                    logging.info("    %s", diag)
            else:
                apply_floor = True
                logging.warning("All candidates failed for %s. Keeping %d core bins with floor handling.",
                                cat, n_core_final)

            nbins = len(bin_edges) - 1
            bin_edges_vector = ROOT.std.vector["double"](bin_edges)
            templates = {}
            process_order = []
            category_processes = []

            if args.unblind or args.partial_unblind:
                data_obs = None
                for subera in suberas:
                    basedir = sample_basedir(workdir, subera, channel, args.masspoint)
                    h_data = getDataHist(
                        basedir, bin_edges, mass_min, mass_max,
                        best_threshold, upper_threshold, bg_weights, args.masspoint
                    )
                    if data_obs is None:
                        data_obs = h_data.Clone("data_obs")
                        data_obs.SetDirectory(0)
                    else:
                        data_obs.Add(h_data)
            else:
                data_obs = ROOT.TH1D("data_obs", "data_obs", nbins, bin_edges_vector.data())
                data_obs.SetDirectory(0)

            for subera in suberas:
                basedir = sample_basedir(workdir, subera, channel, args.masspoint)
                syst_config = load_systematics_block(workdir, subera, channel)
                syst_categories = categorize_systematics(syst_config)

                sig_comp = component_name("signal", subera, is_signal=True)
                logging.info("Processing %s", sig_comp)
                sig_map = build_signal_component_templates(
                    basedir, sig_comp, bin_edges, mass_min, mass_max,
                    syst_categories, best_threshold, upper_threshold,
                    bg_weights, args.masspoint, args.partial_unblind
                )
                templates[sig_comp] = sig_map
                process_order.append(sig_comp)
                category_processes.append({
                    "name": sig_comp,
                    "base_process": "signal",
                    "physics_group": "signal",
                    "subera": subera,
                    "is_signal": True,
                })

                for proc in active_by_subera[subera]:
                    comp = component_name(proc, subera)
                    logging.info("Processing %s", comp)
                    floor_mode = "floor" if proc == "others" else "zero"
                    proc_map = build_component_shape_templates(
                        basedir, proc, comp, bin_edges, mass_min, mass_max,
                        syst_categories, best_threshold, upper_threshold,
                        bg_weights, args.masspoint, floor_mode
                    )
                    templates[comp] = proc_map
                    process_order.append(comp)
                    category_processes.append({
                        "name": comp,
                        "base_process": proc,
                        "physics_group": proc,
                        "subera": subera,
                        "is_signal": False,
                    })
                    if not (args.unblind or args.partial_unblind):
                        data_obs.Add(proc_map["nominal"])

            templates["data_obs"] = data_obs

            if apply_floor:
                logging.warning("Applying floor/zero fallback after failed binning scan for %s", cat)
                for proc_name, proc_map in templates.items():
                    if proc_name == "data_obs":
                        continue
                    base = proc_name.rsplit("_", 1)[0]
                    mode = "floor" if base in {"signal", "others"} else "zero"
                    for hist in proc_map.values():
                        ensure_positive_integral(hist, floor_mode=mode)
                        cap_stat_errors(hist)

            bkg_process_list = [p["name"] for p in category_processes if not p["is_signal"]]
            pre_merge_nbins = len(bin_edges) - 1
            bin_edges, templates, n_syst_merges = apply_syst_driven_merging(
                bin_edges, templates, bkg_process_list,
                max_rel_syst=SYST_MERGE_THRESHOLD, logger=logging
            )
            if n_syst_merges:
                logging.warning(
                    "%s syst-merge: %d bins -> %d bins",
                    cat, pre_merge_nbins, len(bin_edges) - 1
                )

            category_outputs[cat] = {
                "templates": templates,
                "process_order": process_order,
                "processes": category_processes,
            }
            categories_json[cat] = {
                "category": cat,
                "channel": channel,
                "run_period": period,
                "suberas": suberas,
                "processes": category_processes,
            }
            binning_json["categories"][cat] = {
                "nbins": len(bin_edges) - 1,
                "bin_edges": [float(e) for e in bin_edges],
                "method": "AdaptiveExtendedBins",
                "sigma_eff": float(sigma_eff),
                "mass_min": float(mass_min),
                "mass_max": float(mass_max),
                "n_core_bins": int(n_core_final),
                "fit_model": "dcb",
                "fit_result": fit_result,
                "floor_applied": bool(apply_floor),
                "syst_merge_applied": n_syst_merges > 0,
                "n_bins_merged": int(n_syst_merges),
                "syst_merge_threshold": SYST_MERGE_THRESHOLD,
            }

            for proc_meta in category_processes:
                process_components.append({"category": cat, **proc_meta})

            logging.info("Finished %s: data_obs=%.4f, processes=%d",
                         cat, templates["data_obs"].Integral(), len(process_order))

    write_run_period_shapes(outdir, category_outputs)

    save_json({"construction": "run_period_components", "categories": categories_json},
              f"{outdir}/categories.json")
    save_json({
        "construction": "run_period_components",
        "separate_processes": [p["name"] for p in process_components if not p["is_signal"]],
        "signal_processes": [p["name"] for p in process_components if p["is_signal"]],
        "components": process_components,
        "physics_groups": {
            group: [
                p["name"] for p in process_components
                if p["physics_group"] == group
            ]
            for group in PHYSICS_PROCESS_ORDER
        },
        "merged_to_others": [],
        "description": "Run-period categories with subera component processes",
    }, f"{outdir}/process_list.json")
    save_json(binning_json, f"{outdir}/binning.json")
    save_json(background_validation, f"{outdir}/background_validation.json")
    if threshold_summary["categories"]:
        save_json(threshold_summary, f"{outdir}/threshold.json")
    if background_weight_summary["categories"]:
        save_json(background_weight_summary, f"{outdir}/background_weights.json")

    logging.info("Run-period component template generation complete: %s/shapes.root", outdir)


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

    build_run_period_templates(args, workdir)

if __name__ == "__main__":
    main()
