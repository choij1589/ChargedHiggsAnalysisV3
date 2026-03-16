#!/usr/bin/env python3
"""
Plot ParticleNet score distributions for signal and backgrounds.

This script creates diagnostic plots showing the Modified ParticleNet Score
distribution for signal vs stacked backgrounds after mass window selection.
Always uses template binning mass window and generates validation plots.

Workflow:
    1. Create Histograms (200 bins) - central + shape systematics
    2. Rebin Histograms - apply analysis-specific binning
    3. Calculate Bin-by-Bin Errors - envelope method for total uncertainty
    4. Draw Plot - visualization with ComparisonCanvas

Usage:
    plotParticleNetScore.py --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --binning extended
"""
import os
import sys
import logging
import argparse
import json
import ROOT
import numpy as np
from math import sqrt

from template_utils import (
    categorize_systematics, parse_variations, get_output_tree_name, calculate_weight_scale,
    is_signal_scaled_from_run2, get_run2_tree_name_for_run3_syst
)

# Argument parsing
parser = argparse.ArgumentParser(description="Plot ParticleNet score distributions for signal and backgrounds")
parser.add_argument("--era", required=True, type=str, help="Data-taking period (2016preVFP, 2017, 2018, 2022, etc.)")
parser.add_argument("--channel", required=True, type=str, help="Analysis channel (SR1E2Mu, SR3Mu)")
parser.add_argument("--masspoint", required=True, type=str, help="Signal mass point (e.g., MHc130_MA90)")
parser.add_argument("--binning", default="extended", choices=["uniform", "extended"],
                    help="Binning method: 'extended' (19 bins, default) or 'uniform' (15 bins)")
parser.add_argument("--unblind", action="store_true",
                    help="Show real data distribution")
parser.add_argument("--partial-unblind", action="store_true", dest="partial_unblind",
                    help="Show real data only for score < 0.3")
parser.add_argument("--skip-histogram", action="store_true", dest="skip_histogram",
                    help="Load from existing histograms.root instead of reprocessing")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args = parser.parse_args()

# Validate unblind options
if args.unblind and args.partial_unblind:
    raise ValueError("--unblind and --partial-unblind are mutually exclusive")

# Validate channel (TTZ2E1Mu is included automatically in validation plots)
if args.channel not in ["SR1E2Mu", "SR3Mu"]:
    raise ValueError(f"Invalid channel: {args.channel}. Must be SR1E2Mu or SR3Mu. "
                     "TTZ2E1Mu control region is automatically included in validation plots.")

# Setup logging
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                    format='%(levelname)s - %(message)s')

# Path setup
WORKDIR = os.getenv("WORKDIR")
if not WORKDIR:
    raise EnvironmentError("WORKDIR environment variable not set. Please run 'source setup.sh'")

# Add path to Common/Tools for plotter imports
sys.path.insert(0, f"{WORKDIR}/Common/Tools")
from plotter import ComparisonCanvas, get_CoM_energy, get_era_list, LumiInfo
from plotter import PALETTE_LONG as PALETTE
import cmsstyle as CMS

# Default binning for stored histograms (fine granularity for era combination)
DEFAULT_NBINS = 200
DEFAULT_XMIN = 0.0
DEFAULT_XMAX = 1.0

# Expand combined eras
era_list = get_era_list(args.era)
is_combined_era = len(era_list) > 1

if is_combined_era:
    logging.info(f"Combined era mode: {args.era} -> {era_list}")

# Fixed color mapping for backgrounds (consistent across all plots)
# Uses PALETTE colors from plotter.py for consistency
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
# Legend will display in reverse order (top to bottom) to match visual stacking
BKG_ORDER = ["others", "conversion", "WZ", "ZZ", "ttW", "ttH", "tZq", "ttZ", "nonprompt"]

# Determine binning suffix (with unblind suffix if applicable)
binning_suffix = args.binning
if args.unblind:
    binning_suffix = f"{args.binning}_unblind"
elif args.partial_unblind:
    binning_suffix = f"{args.binning}_partial_unblind"

# For combined eras, we use first era's config as reference
reference_era = era_list[0]

# Input from samples directory (ParticleNet scores are in preprocessed samples)
# For combined eras, BASEDIR points to reference era (used for config loading)
BASEDIR = f"{WORKDIR}/SignalRegionStudyV2/samples/{reference_era}/{args.channel}/{args.masspoint}"
# Config directory (always from base binning, not unblind variants)
CONFIGDIR = f"{WORKDIR}/SignalRegionStudyV2/templates/{reference_era}/{args.channel}/{args.masspoint}/ParticleNet/{args.binning}"
# Output to ParticleNet template directory (with unblind suffix if applicable)
OUTDIR = f"{WORKDIR}/SignalRegionStudyV2/templates/{args.era}/{args.channel}/{args.masspoint}/ParticleNet/{binning_suffix}"

# Always generate validation plots
VALIDATION_OUTDIR = f"{OUTDIR}/scores"

if is_combined_era:
    logging.info(f"Combined era mode: loading from per-era histograms")
    logging.info(f"Reference era for config: {reference_era}")
else:
    logging.info(f"Input directory: {BASEDIR}")
logging.info(f"Output directory: {OUTDIR}")

# Load histogram configuration
histkeys_path = f"{WORKDIR}/SignalRegionStudyV2/configs/histkeys.json"
with open(histkeys_path) as f:
    HISTKEYS_CONFIG = json.load(f)
logging.info(f"Loaded {len(HISTKEYS_CONFIG)} histogram configurations from histkeys.json")

# Load mass window from binning.json or use full event selection
binning_path = f"{CONFIGDIR}/binning.json"
if not os.path.exists(binning_path):
    raise FileNotFoundError(
        f"Binning results not found: {binning_path}\n"
        f"Please run makeBinnedTemplates.py first:\n"
        f"  makeBinnedTemplates.py --era {args.era} --channel {args.channel} --masspoint {args.masspoint} --method ParticleNet --binning {args.binning}"
    )

with open(binning_path, 'r') as f:
    binning_params = json.load(f)

# Always use template binning for signal regions
MASS_MIN = binning_params["mass_min"]
MASS_MAX = binning_params["mass_max"]
logging.info(f"Mass window from binning: [{MASS_MIN:.2f}, {MASS_MAX:.2f}] GeV")

# Load background weights
weights_json_path = f"{CONFIGDIR}/background_weights.json"
BG_WEIGHTS = None
if os.path.exists(weights_json_path):
    with open(weights_json_path, 'r') as f:
        weights_data = json.load(f)
        BG_WEIGHTS = weights_data.get("weights", None)
        logging.info(f"Loaded background weights: {BG_WEIGHTS}")
else:
    logging.warning(f"Background weights not found: {weights_json_path}")
    logging.warning(f"Using unweighted ParticleNet score (equal priors)")

# Load systematics configuration (only for single era mode - combined eras use pre-stored histograms)
if is_combined_era:
    logging.info("Skipping systematics loading for combined era (using pre-stored histograms)")
    syst_categories = {'preprocessed_shape': [], 'valued_shape': [], 'multi_variation': [], 'valued_lnN': []}
else:
    syst_config_path = f"{WORKDIR}/SignalRegionStudyV2/configs/systematics.{args.era}.json"
    if os.path.exists(syst_config_path):
        with open(syst_config_path) as f:
            syst_config_full = json.load(f)
        syst_config = syst_config_full.get(args.channel, {})
        syst_categories = categorize_systematics(syst_config)
        logging.info(f"Loaded {len(syst_categories['preprocessed_shape'])} preprocessed shape systematics")
        logging.info(f"Loaded {len(syst_categories['valued_shape'])} valued shape systematics")
        logging.info(f"Loaded {len(syst_categories['valued_lnN'])} valued lnN systematics")
    else:
        logging.warning(f"Systematics config not found: {syst_config_path}")
        syst_categories = {'preprocessed_shape': [], 'valued_shape': [], 'multi_variation': [], 'valued_lnN': []}

# Setup ROOT for batch mode
ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)


# =============================================================================
# Helper Functions
# =============================================================================

def get_color(process):
    """Get color for process from BKG_COLORS."""
    if process in BKG_COLORS:
        return BKG_COLORS[process]
    # Fallback for unknown processes
    return ROOT.kGray


def get_rebin_factor(plot_key):
    """Get rebin factor from HISTKEYS_CONFIG for a given plot key."""
    return HISTKEYS_CONFIG[plot_key].get("rebin", 4)


def build_colors_list(bkg_hists):
    """Build colors list from BKG_COLORS for background histograms."""
    colors = []
    for bkg in bkg_hists.keys():
        if bkg in BKG_COLORS:
            colors.append(BKG_COLORS[bkg])
        else:
            colors.append(ROOT.kGray)
    return colors


def should_skip_plot(bkg_hists, signal_hist):
    """Check if plot should be skipped due to empty data."""
    return len(bkg_hists) == 0 and (signal_hist is None or signal_hist.Integral() == 0)


def style_signal_histogram(hist):
    """Apply signal styling: black line, width 2, no fill."""
    hist.SetLineColor(ROOT.kBlack)
    hist.SetLineWidth(2)
    hist.SetLineStyle(1)
    hist.SetFillStyle(0)


def create_filled_histogram(name, title, values, weights, nbins=DEFAULT_NBINS,
                            xmin=DEFAULT_XMIN, xmax=DEFAULT_XMAX):
    """Create and fill a histogram from arrays."""
    hist = ROOT.TH1D(name, title, nbins, xmin, xmax)
    for val, wt in zip(values, weights):
        hist.Fill(val, wt)
    hist.SetDirectory(0)
    return hist


def process_in_group(process, group, masspoint):
    """Check if process belongs to a systematic group."""
    if process in group:
        return True
    if process == masspoint and "signal" in group:
        return True
    return False


def sum_histograms(hist_list, name):
    """Sum list of histograms into one."""
    if not hist_list:
        return None

    result = hist_list[0].Clone(name)
    result.SetDirectory(0)

    for h in hist_list[1:]:
        result.Add(h)

    return result


# =============================================================================
# Step 1: Create Histograms (200 bins)
# =============================================================================

def load_histograms_from_file(hist_file_path, score_type):
    """
    Load all histograms for a score type from per-era ROOT file.

    Args:
        hist_file_path: Path to the histograms.root file
        score_type: Score type directory name (e.g., 'LR_modified')

    Returns:
        Dict of {process_name: TH1D histogram}
    """
    if not os.path.exists(hist_file_path):
        raise FileNotFoundError(f"Histogram file not found: {hist_file_path}")

    rfile = ROOT.TFile.Open(hist_file_path, "READ")
    if not rfile or rfile.IsZombie():
        raise RuntimeError(f"Failed to open: {hist_file_path}")

    # Navigate to score type directory
    if not rfile.cd(score_type):
        rfile.Close()
        raise RuntimeError(f"Score type directory '{score_type}' not found in {hist_file_path}")

    histograms = {}
    for key in ROOT.gDirectory.GetListOfKeys():
        name = key.GetName()
        hist = key.ReadObj()
        if hist and hist.InheritsFrom("TH1"):
            hist.SetDirectory(0)
            histograms[name] = hist

    rfile.Close()
    return histograms


def loadScores(process, masspoint, sample_dir=None, return_raw=False,
               mass_min=None, mass_max=None, sample_channel=None, syst="Central"):
    """
    Load ParticleNet scores from a process sample.

    Args:
        process: Process name
        masspoint: Signal mass point
        sample_dir: Base directory for samples (default: BASEDIR)
        return_raw: If True, return dict with all raw scores + LRs
        mass_min: Override mass window minimum (default: use global MASS_MIN)
        mass_max: Override mass window maximum (default: use global MASS_MAX)
        sample_channel: Channel for mass selection logic (default: use args.channel)
        syst: Systematic variation tree name (default: "Central")

    Returns:
        If return_raw=False: (scores, weights) - modified LR only (backward compatible)
        If return_raw=True: dict with keys:
            'raw_signal', 'raw_nonprompt', 'raw_diboson', 'raw_ttZ',
            'LR_nonprompt', 'LR_diboson', 'LR_ttZ', 'LR_modified',
            'weights'
    """
    if sample_dir is None:
        sample_dir = BASEDIR
    if mass_min is None:
        mass_min = MASS_MIN
    if mass_max is None:
        mass_max = MASS_MAX
    if sample_channel is None:
        sample_channel = args.channel

    file_path = f"{sample_dir}/{process}.root"

    empty_result = {
        'raw_signal': np.array([]), 'raw_nonprompt': np.array([]),
        'raw_diboson': np.array([]), 'raw_ttZ': np.array([]),
        'LR_nonprompt': np.array([]), 'LR_diboson': np.array([]),
        'LR_ttZ': np.array([]), 'LR_modified': np.array([]),
        'weights': np.array([])
    }

    if not os.path.exists(file_path):
        logging.warning(f"Sample file not found: {file_path}")
        if return_raw:
            return empty_result
        return np.array([]), np.array([])

    # Open file and get tree
    rfile = ROOT.TFile.Open(file_path, "READ")
    tree = rfile.Get(syst)

    if not tree:
        if syst != "Central":
            logging.debug(f"Tree '{syst}' not found in {file_path}, returning empty")
        else:
            logging.warning(f"Central tree not found in {file_path}")
        rfile.Close()
        if return_raw:
            return empty_result
        return np.array([]), np.array([])

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
        if return_raw:
            return empty_result
        return np.array([]), np.array([])

    # Load data
    scores_list = []
    weights_list = []

    # For return_raw mode
    raw_signal_list = []
    raw_nonprompt_list = []
    raw_diboson_list = []
    raw_ttZ_list = []
    LR_nonprompt_list = []
    LR_diboson_list = []
    LR_ttZ_list = []

    for entry in range(tree.GetEntries()):
        tree.GetEntry(entry)

        # Handle mass selection based on channel
        # TTZ2E1Mu is a control region without resonance - skip mass window cut
        if sample_channel == "TTZ2E1Mu":
            pass  # No mass window cut for control region
        elif sample_channel == "SR3Mu":
            mass1 = getattr(tree, "mass1")
            mass2 = getattr(tree, "mass2")
            # Check if either mass is in the window
            if not (mass_min <= mass1 <= mass_max or mass_min <= mass2 <= mass_max):
                continue
        else:
            # SR1E2Mu uses single mass variable
            mass = tree.mass
            # Apply mass window cut
            if not (mass_min <= mass <= mass_max):
                continue

        s0 = getattr(tree, score_sig)
        s1 = getattr(tree, score_nonprompt)
        s2 = getattr(tree, score_diboson)
        s3 = getattr(tree, score_ttZ)
        weight = getattr(tree, "weight")

        # Calculate ParticleNet likelihood ratio with cross-section weights
        if BG_WEIGHTS:
            w1 = BG_WEIGHTS.get("nonprompt", 1.0)
            w2 = BG_WEIGHTS.get("diboson", 1.0)
            w3 = BG_WEIGHTS.get("ttX", 1.0)
            score_denom = s0 + w1*s1 + w2*s2 + w3*s3
        else:
            # Use unweighted likelihood ratio (equal priors)
            score_denom = s0 + s1 + s2 + s3

        if score_denom > 0:
            score_PN = s0 / score_denom
        else:
            score_PN = 0.0

        scores_list.append(score_PN)
        weights_list.append(weight)

        # Calculate additional quantities if requested
        if return_raw:
            raw_signal_list.append(s0)
            raw_nonprompt_list.append(s1)
            raw_diboson_list.append(s2)
            raw_ttZ_list.append(s3)

            # Individual LRs
            LR_np = s0 / (s0 + s1) if (s0 + s1) > 0 else 0.0
            LR_db = s0 / (s0 + s2) if (s0 + s2) > 0 else 0.0
            LR_ttZ = s0 / (s0 + s3) if (s0 + s3) > 0 else 0.0

            LR_nonprompt_list.append(LR_np)
            LR_diboson_list.append(LR_db)
            LR_ttZ_list.append(LR_ttZ)

    rfile.Close()

    if return_raw:
        return {
            'raw_signal': np.array(raw_signal_list),
            'raw_nonprompt': np.array(raw_nonprompt_list),
            'raw_diboson': np.array(raw_diboson_list),
            'raw_ttZ': np.array(raw_ttZ_list),
            'LR_nonprompt': np.array(LR_nonprompt_list),
            'LR_diboson': np.array(LR_diboson_list),
            'LR_ttZ': np.array(LR_ttZ_list),
            'LR_modified': np.array(scores_list),
            'weights': np.array(weights_list)
        }

    return np.array(scores_list), np.array(weights_list)


def load_preprocessed_syst_scores(process, masspoint, sample_dir, mass_min, mass_max,
                                   sample_channel, syst_categories, era=None):
    """
    Load preprocessed shape systematic variation scores for a process.

    Loads all systematic scores ONCE per process (not per score_type) for efficiency.

    Args:
        process: Process name
        masspoint: Signal mass point
        sample_dir: Sample directory
        mass_min, mass_max: Mass window
        sample_channel: Channel for mass selection
        syst_categories: Categorized systematics
        era: Data-taking era (needed for Run2/Run3 tree name mapping)

    Returns:
        Dict of {syst_name: {'up': scores_dict, 'down': scores_dict}}
        where scores_dict has keys like 'LR_modified', 'weights', etc.
    """
    syst_scores = {}

    # Check if this is signal and if it's scaled from Run2 (has Run2 tree names)
    is_signal = (process == masspoint)
    signal_scaled_from_run2 = False
    if is_signal and era:
        signal_file = f"{sample_dir}/{process}.root"
        if os.path.exists(signal_file):
            signal_scaled_from_run2 = is_signal_scaled_from_run2(signal_file, era)
            if signal_scaled_from_run2:
                logging.debug(f"Signal {process} is scaled from Run2, will remap tree names")

    for syst_name, variations, group in syst_categories['preprocessed_shape']:
        if not process_in_group(process, group, masspoint):
            continue

        up_var = None
        down_var = None
        for var in variations:
            if var.endswith("Up") or var.endswith("_Up"):
                up_var = var
            elif var.endswith("Down") or var.endswith("_Down"):
                down_var = var

        if up_var and down_var:
            # Get output tree names (Run3-style)
            up_tree = get_output_tree_name(syst_name, up_var)
            down_tree = get_output_tree_name(syst_name, down_var)

            # For Run2-scaled signal, map Run3 names to Run2 tree names
            if is_signal and signal_scaled_from_run2 and era:
                up_tree = get_run2_tree_name_for_run3_syst(syst_name, "Up", era)
                down_tree = get_run2_tree_name_for_run3_syst(syst_name, "Down", era)
                logging.debug(f"  Remapped signal trees: {syst_name} -> {up_tree}, {down_tree}")

            # Load scores for up variation (all score types at once)
            up_scores = loadScores(
                process, masspoint,
                sample_dir=sample_dir, return_raw=True,
                mass_min=mass_min, mass_max=mass_max,
                sample_channel=sample_channel, syst=up_tree
            )

            # Load scores for down variation (all score types at once)
            down_scores = loadScores(
                process, masspoint,
                sample_dir=sample_dir, return_raw=True,
                mass_min=mass_min, mass_max=mass_max,
                sample_channel=sample_channel, syst=down_tree
            )

            # Store if data exists
            if len(up_scores['weights']) > 0 and len(down_scores['weights']) > 0:
                syst_scores[syst_name] = {'up': up_scores, 'down': down_scores}

    return syst_scores


def create_syst_histograms_from_scores(process, syst_scores, score_type):
    """
    Create systematic histograms for a specific score_type from pre-loaded scores.

    Args:
        process: Process name
        syst_scores: Dict from load_preprocessed_syst_scores
        score_type: Score type (e.g., 'LR_modified')

    Returns:
        Dict of {syst_name: {'up': TH1D, 'down': TH1D}}
    """
    syst_hists = {}

    for syst_name, scores in syst_scores.items():
        up_scores = scores['up']
        down_scores = scores['down']

        hist_up = create_filled_histogram(
            f"{process}_{syst_name}Up_{score_type}", "",
            up_scores[score_type], up_scores['weights']
        )
        hist_down = create_filled_histogram(
            f"{process}_{syst_name}Down_{score_type}", "",
            down_scores[score_type], down_scores['weights']
        )
        syst_hists[syst_name] = {'up': hist_up, 'down': hist_down}

    return syst_hists


def create_valued_syst_hists(central_hist, process, masspoint, syst_categories):
    """
    Create valued shape/lnN systematic variation histograms by scaling central.

    Args:
        central_hist: Central histogram (200 bins)
        process: Process name
        masspoint: Signal mass point
        syst_categories: Categorized systematics

    Returns:
        Dict of {syst_name: {'up': TH1D, 'down': TH1D}}
    """
    syst_hists = {}

    for syst_name, value, group in syst_categories['valued_shape'] + syst_categories['valued_lnN']:
        if not process_in_group(process, group, masspoint):
            continue

        scale_up = calculate_weight_scale(value, 'up')
        scale_down = calculate_weight_scale(value, 'down')

        hist_up = central_hist.Clone(f"{process}_{syst_name}Up")
        hist_down = central_hist.Clone(f"{process}_{syst_name}Down")
        hist_up.Scale(scale_up)
        hist_down.Scale(scale_down)
        hist_up.SetDirectory(0)
        hist_down.SetDirectory(0)

        syst_hists[syst_name] = {'up': hist_up, 'down': hist_down}

    return syst_hists


def create_histograms(era, channel, masspoint, sample_channel, syst_categories,
                      mass_min, mass_max, ordered_bkgs, show_data):
    """
    Step 1: Create 200-bin histograms for all processes and ALL systematic variations.

    Central histograms have statistical errors only. Systematic errors are computed
    from Up/Down variations at plot time (or when rebinning with --skip-histogram).

    Args:
        era: Data-taking era
        channel: Analysis channel
        masspoint: Signal mass point
        sample_channel: Channel for loading samples
        syst_categories: Categorized systematics
        mass_min, mass_max: Mass window
        ordered_bkgs: Ordered list of background process names
        show_data: Whether to show real data

    Returns:
        Tuple of (stored_hists, all_syst_hists)
        - stored_hists: Dict[score_type, Dict[hist_name, TH1D]] - central histograms (stat errors only)
        - all_syst_hists: Dict[score_type, Dict[process, Dict[syst_name, {'up': TH1D, 'down': TH1D}]]]
    """
    sample_dir = f"{WORKDIR}/SignalRegionStudyV2/samples/{era}/{sample_channel}/{masspoint}"

    # Check if samples exist
    if not os.path.exists(sample_dir):
        raise FileNotFoundError(f"Samples not found for {sample_channel} at {sample_dir}")

    # Load all scores with raw values for each process
    all_process_scores = {}

    # Load signal (only for SR channels, not control region)
    if sample_channel != "TTZ2E1Mu":
        signal_scores = loadScores(
            masspoint, masspoint,
            sample_dir=sample_dir, return_raw=True,
            mass_min=mass_min, mass_max=mass_max,
            sample_channel=sample_channel
        )
        all_process_scores[masspoint] = signal_scores

    # Load backgrounds
    for process in ordered_bkgs:
        scores_data = loadScores(
            process, masspoint,
            sample_dir=sample_dir, return_raw=True,
            mass_min=mass_min, mass_max=mass_max,
            sample_channel=sample_channel
        )
        if len(scores_data['weights']) > 0:
            all_process_scores[process] = scores_data

    # Load data
    data_scores = loadScores(
        "data", masspoint,
        sample_dir=sample_dir, return_raw=True,
        mass_min=mass_min, mass_max=mass_max,
        sample_channel=sample_channel
    )

    # Load preprocessed systematic scores ONCE per process (outside score_type loop)
    # This is much more efficient than loading per score_type
    all_process_syst_scores = {}  # {process: {syst_name: {'up': scores, 'down': scores}}}

    for process in ordered_bkgs:
        if process not in all_process_scores:
            continue
        syst_scores = load_preprocessed_syst_scores(
            process, masspoint, sample_dir, mass_min, mass_max,
            sample_channel, syst_categories, era
        )
        if syst_scores:
            all_process_syst_scores[process] = syst_scores

    # Load signal systematic scores (only for SR channels)
    if sample_channel != "TTZ2E1Mu" and masspoint in all_process_scores:
        signal_syst_scores = load_preprocessed_syst_scores(
            masspoint, masspoint, sample_dir, mass_min, mass_max,
            sample_channel, syst_categories, era
        )
        if signal_syst_scores:
            all_process_syst_scores[masspoint] = signal_syst_scores

    # Prepare histogram storage
    stored_hists = {}  # {score_type: {process: hist, ...}}
    all_syst_hists = {}  # {score_type: {process: {syst_name: {'up': hist, 'down': hist}}}}

    # Build plot types from histkeys config
    plot_types = list(HISTKEYS_CONFIG.keys())

    for score_type in plot_types:
        stored_hists[score_type] = {}
        all_syst_hists[score_type] = {}

        # Create histograms for backgrounds (200 bins for storage)
        for process in ordered_bkgs:
            if process not in all_process_scores:
                continue

            scores_data = all_process_scores[process]
            values = scores_data[score_type]
            weights = scores_data['weights']

            if len(values) == 0:
                continue

            # Create 200-bin histogram for storage (stat errors only)
            hist_200 = create_filled_histogram(
                f"{process}_{score_type}", process.capitalize(), values, weights
            )
            stored_hists[score_type][process] = hist_200

            # Create preprocessed shape systematic histograms from pre-loaded scores
            if process in all_process_syst_scores:
                preproc_syst_hists = create_syst_histograms_from_scores(
                    process, all_process_syst_scores[process], score_type
                )
            else:
                preproc_syst_hists = {}

            # Create valued shape/lnN systematic histograms (scale central)
            valued_syst_hists = create_valued_syst_hists(
                hist_200, process, masspoint, syst_categories
            )

            # Combine all systematics for this process
            if preproc_syst_hists or valued_syst_hists:
                all_syst_hists[score_type][process] = {**preproc_syst_hists, **valued_syst_hists}

        # Create signal histogram (only for SR channels)
        if sample_channel != "TTZ2E1Mu" and masspoint in all_process_scores:
            signal_data = all_process_scores[masspoint]
            if len(signal_data['weights']) > 0:
                hist_signal_200 = create_filled_histogram(
                    f"{masspoint}_{score_type}",
                    f"Signal ({masspoint})",
                    signal_data[score_type], signal_data['weights']
                )
                stored_hists[score_type][masspoint] = hist_signal_200

                # Create signal preprocessed systematic histograms from pre-loaded scores
                if masspoint in all_process_syst_scores:
                    signal_preproc_systs = create_syst_histograms_from_scores(
                        masspoint, all_process_syst_scores[masspoint], score_type
                    )
                else:
                    signal_preproc_systs = {}

                # Create signal valued systematic histograms
                signal_valued_systs = create_valued_syst_hists(
                    hist_signal_200, masspoint, masspoint, syst_categories
                )

                if signal_preproc_systs or signal_valued_systs:
                    all_syst_hists[score_type][masspoint] = {**signal_preproc_systs, **signal_valued_systs}

        # Create data_obs histogram (200 bins for storage)
        data_obs_200 = ROOT.TH1D(f"data_obs_{score_type}", "Data",
                                  DEFAULT_NBINS, DEFAULT_XMIN, DEFAULT_XMAX)
        if show_data:
            if len(data_scores['weights']) > 0:
                for val, wt in zip(data_scores[score_type], data_scores['weights']):
                    data_obs_200.Fill(val, wt)

            if args.partial_unblind and sample_channel != "TTZ2E1Mu":
                # Zero out bins with score >= 0.3 (only for LR plots, only for SR)
                if score_type.startswith("LR"):
                    threshold_bin = int(0.3 * DEFAULT_NBINS / (DEFAULT_XMAX - DEFAULT_XMIN)) + 1
                    for ibin in range(threshold_bin, data_obs_200.GetNbinsX() + 1):
                        data_obs_200.SetBinContent(ibin, 0)
                        data_obs_200.SetBinError(ibin, 0)
        else:
            # Blinded: sum of backgrounds
            for proc, hist in stored_hists[score_type].items():
                if proc != masspoint:
                    data_obs_200.Add(hist)

        data_obs_200.SetDirectory(0)
        stored_hists[score_type]["data_obs"] = data_obs_200

    return stored_hists, all_syst_hists


# =============================================================================
# Step 2: Rebin Histograms
# =============================================================================

def rebin_histograms(stored_hists, score_type):
    """
    Step 2: Rebin all histograms according to config.

    Args:
        stored_hists: Dict from create_histograms() for a specific score_type
        score_type: The score type being processed

    Returns:
        rebinned_hists: Dict with same structure, but rebinned
    """
    rebin_factor = get_rebin_factor(score_type)
    rebinned = {}

    for hist_name, hist in stored_hists.items():
        rebinned_hist = hist.Clone(f"{hist_name}_rebinned")
        rebinned_hist.Rebin(rebin_factor)
        rebinned_hist.SetDirectory(0)
        rebinned[hist_name] = rebinned_hist

    return rebinned


def rebin_syst_histograms(syst_hists, score_type, rebin_factor):
    """
    Rebin systematic variation histograms.

    Args:
        syst_hists: Dict of {process: {syst_name: {'up': TH1D, 'down': TH1D}}}
        score_type: Score type
        rebin_factor: Rebin factor to apply

    Returns:
        rebinned_syst_hists with same structure
    """
    rebinned = {}

    for process, proc_systs in syst_hists.items():
        rebinned[process] = {}
        for syst_name, variations in proc_systs.items():
            up_rebinned = variations['up'].Clone(f"{process}_{syst_name}Up_rebinned")
            down_rebinned = variations['down'].Clone(f"{process}_{syst_name}Down_rebinned")
            up_rebinned.Rebin(rebin_factor)
            down_rebinned.Rebin(rebin_factor)
            up_rebinned.SetDirectory(0)
            down_rebinned.SetDirectory(0)
            rebinned[process][syst_name] = {'up': up_rebinned, 'down': down_rebinned}

    return rebinned


def extract_syst_hists(all_hists, central_processes):
    """
    Extract systematic histograms from loaded ROOT file.

    Args:
        all_hists: {hist_name: TH1D} from ROOT file
        central_processes: List of central process names

    Returns:
        {process: {syst_name: {'up': TH1D, 'down': TH1D}}}
    """
    syst_hists = {}

    for hist_name, hist in all_hists.items():
        if 'Up' not in hist_name and 'Down' not in hist_name:
            continue

        # Parse: "{process}_{syst_name}Up" or "{process}_{syst_name}Down"
        for proc in central_processes:
            if hist_name.startswith(f"{proc}_"):
                suffix = hist_name[len(proc)+1:]
                if suffix.endswith("Up"):
                    syst_name = suffix[:-2]
                    direction = 'up'
                elif suffix.endswith("Down"):
                    syst_name = suffix[:-4]
                    direction = 'down'
                else:
                    continue

                if proc not in syst_hists:
                    syst_hists[proc] = {}
                if syst_name not in syst_hists[proc]:
                    syst_hists[proc][syst_name] = {}
                syst_hists[proc][syst_name][direction] = hist
                break

    return syst_hists


# =============================================================================
# Step 3: Calculate Bin-by-Bin Errors
# =============================================================================

def calculate_errors(rebinned_hists, syst_categories, masspoint, all_syst_hists=None,
                     rebin_factor=4, debug_bin=None):
    """
    Step 3: Calculate bin-by-bin systematic errors for each background process.

    Uses envelope method for shapes + valued systematics.

    Args:
        rebinned_hists: Dict of rebinned histograms {process: TH1D}
        syst_categories: Categorized systematics
        masspoint: Signal mass point
        all_syst_hists: Dict of {process: {syst_name: {'up': TH1D, 'down': TH1D}}}
        rebin_factor: Rebin factor applied
        debug_bin: If set, print detailed breakdown for this bin

    Returns:
        hists_with_errors: Dict of {process: TH1D with total errors}
    """
    result = {}

    # Get all central process names (excluding data_obs and systematics)
    central_processes = [p for p in rebinned_hists.keys()
                        if p != 'data_obs' and 'Up' not in p and 'Down' not in p]

    # First, collect all unique systematic names across all background processes
    all_syst_names = set()
    if all_syst_hists:
        for process, proc_systs in all_syst_hists.items():
            if process == masspoint:
                continue  # Skip signal for background error calculation
            all_syst_names.update(proc_systs.keys())

    # Also add valued systematics
    for syst_name, value, group in syst_categories['valued_shape']:
        all_syst_names.add(syst_name)
    for syst_name, value, group in syst_categories['valued_lnN']:
        all_syst_names.add(syst_name)

    # Calculate background sum for proper error distribution
    bkg_sum = None
    bkg_processes = [p for p in central_processes if p != masspoint]
    for process in bkg_processes:
        if process in rebinned_hists:
            if bkg_sum is None:
                bkg_sum = rebinned_hists[process].Clone("bkg_sum_for_errors")
                bkg_sum.SetDirectory(0)
            else:
                bkg_sum.Add(rebinned_hists[process])

    if bkg_sum is None:
        return {p: rebinned_hists[p].Clone() for p in central_processes}

    # Build combined systematic histograms for background sum
    combined_syst_hists = {}

    for syst_name in all_syst_names:
        combined_syst_hists[syst_name] = {'up': None, 'down': None}

        for process in bkg_processes:
            if process not in rebinned_hists:
                continue

            central_hist = rebinned_hists[process]

            # Check if this process has preprocessed shape for this systematic
            if all_syst_hists and process in all_syst_hists and syst_name in all_syst_hists[process]:
                # Use the preprocessed systematic variation (already rebinned)
                up_hist = all_syst_hists[process][syst_name]['up']
                down_hist = all_syst_hists[process][syst_name]['down']
            else:
                # Check for valued systematics
                is_valued_syst = False
                for valued_syst_name, value, group in syst_categories['valued_shape'] + syst_categories['valued_lnN']:
                    if valued_syst_name == syst_name and process_in_group(process, group, masspoint):
                        is_valued_syst = True
                        scale_up = calculate_weight_scale(value, 'up')
                        scale_down = calculate_weight_scale(value, 'down')

                        up_hist = central_hist.Clone(f"{process}_{syst_name}Up_temp")
                        down_hist = central_hist.Clone(f"{process}_{syst_name}Down_temp")
                        up_hist.Scale(scale_up)
                        down_hist.Scale(scale_down)
                        up_hist.SetDirectory(0)
                        down_hist.SetDirectory(0)
                        break

                if not is_valued_syst:
                    # Process doesn't have this systematic - use central
                    up_hist = central_hist
                    down_hist = central_hist

            # Add to combined histograms
            if combined_syst_hists[syst_name]['up'] is None:
                combined_syst_hists[syst_name]['up'] = up_hist.Clone(f"bkg_{syst_name}Up")
                combined_syst_hists[syst_name]['down'] = down_hist.Clone(f"bkg_{syst_name}Down")
                combined_syst_hists[syst_name]['up'].SetDirectory(0)
                combined_syst_hists[syst_name]['down'].SetDirectory(0)
            else:
                combined_syst_hists[syst_name]['up'].Add(up_hist)
                combined_syst_hists[syst_name]['down'].Add(down_hist)

    # Calculate systematic errors using envelope method on bkg_sum
    nbins = bkg_sum.GetNbinsX()
    syst_errors = [0.0] * (nbins + 2)

    for ibin in range(1, nbins + 1):
        central_val = bkg_sum.GetBinContent(ibin)
        if central_val <= 0:
            continue

        errors_sq = []
        debug_info = []

        for syst_name, variations in combined_syst_hists.items():
            up_hist = variations.get('up')
            down_hist = variations.get('down')

            if up_hist and down_hist:
                up_val = up_hist.GetBinContent(ibin)
                down_val = down_hist.GetBinContent(ibin)
                # Envelope: max deviation from central
                max_dev = max(abs(up_val - central_val), abs(down_val - central_val))
                errors_sq.append(max_dev ** 2)

                if debug_bin and ibin == debug_bin:
                    rel_err = max_dev / central_val * 100 if central_val > 0 else 0
                    debug_info.append((syst_name, max_dev, rel_err))

        if errors_sq:
            syst_errors[ibin] = sqrt(sum(errors_sq))

        if debug_bin and ibin == debug_bin and debug_info:
            logging.info(f"  Bin {ibin}: central={central_val:.3f}, total_syst={syst_errors[ibin]:.3f} ({syst_errors[ibin]/central_val*100:.1f}%)")
            debug_info.sort(key=lambda x: x[1], reverse=True)
            for syst_name, dev, rel in debug_info[:5]:
                logging.info(f"    {syst_name}: {dev:.3f} ({rel:.1f}%)")

    # Apply errors to each background histogram proportionally
    for process in central_processes:
        if process not in rebinned_hists:
            continue

        hist = rebinned_hists[process].Clone(f"{process}_with_errors")
        hist.SetDirectory(0)

        if process != masspoint:  # Apply systematic errors only to backgrounds
            for ibin in range(1, hist.GetNbinsX() + 1):
                stat_err = hist.GetBinError(ibin)
                total_content = bkg_sum.GetBinContent(ibin)
                if total_content > 0:
                    frac = hist.GetBinContent(ibin) / total_content
                    syst_err = syst_errors[ibin] * frac if ibin < len(syst_errors) else 0.0
                else:
                    syst_err = 0.0
                total_err = sqrt(stat_err**2 + syst_err**2)
                hist.SetBinError(ibin, total_err)

        result[process] = hist

    return result


# =============================================================================
# Step 4: Parse to ComparisonCanvas
# =============================================================================

def build_canvas_config(era, region_label, x_title, plot_key, colors, com_energy, enable_chi2=False):
    """Build configuration dict for ComparisonCanvas."""
    config = {
        "era": era,
        "CoM": com_energy,
        "channel": region_label,
        "xTitle": x_title,
        "yTitle": "Events",
        "rTitle": "Data / Pred",
        "rRange": HISTKEYS_CONFIG[plot_key].get("rRange", [0, 5.]),
        "maxDigits": 3,
        "colors": colors
    }
    # Add chi2 test when enabled (full unblind or control region)
    if enable_chi2:
        config["chi2_test"] = True
        config["normalize_chi2"] = False
        config["chi2_posY"] = 0.58  # Lower position to avoid overlap with masspoint label
    return config


def draw_signal_overlay(plotter, signal_hist, scale=6.0):
    """Draw scaled signal overlay on upper pad and add to legend."""
    plotter.canv.cd(1)
    signal_hist.Scale(scale)
    signal_hist.Draw("HIST SAME")

    # Add signal to legend
    current_pad = ROOT.gPad
    primitives = current_pad.GetListOfPrimitives()
    for obj in primitives:
        if obj.InheritsFrom("TLegend"):
            obj.AddEntry(signal_hist, f"signal (x{int(scale)})", "l")
            break


def draw_plot(data_hist, bkg_hists_with_errors, signal_hist, config, output_path, masspoint):
    """
    Step 4: Create comparison plot using ComparisonCanvas.

    Args:
        data_hist: Data histogram
        bkg_hists_with_errors: Dict of background histograms with total errors
        signal_hist: Signal histogram (or None)
        config: Canvas configuration
        output_path: Output file path
        masspoint: Mass point label
    """
    # Create plot
    plotter = ComparisonCanvas(data_hist, bkg_hists_with_errors, config)
    plotter.drawPadUp()

    # Draw signal on upper pad (only for SR channels)
    if signal_hist is not None and signal_hist.Integral() > 0:
        draw_signal_overlay(plotter, signal_hist)

    # Draw lower pad
    plotter.drawPadDown()

    # Draw additional text
    plotter.canv.cd()
    CMS.drawText(f"{masspoint}", posX=0.2, posY=0.76, font=61, align=0, size=0.03)

    # Save
    plotter.canv.SaveAs(output_path)
    logging.info(f"  Saved: {output_path}")


def save_histograms_to_root(histfile_path, stored_hists, all_syst_hists=None):
    """
    Save histograms to ROOT file with ALL systematic variations.

    Args:
        histfile_path: Path to output ROOT file
        stored_hists: Dict of {score_type: {process: TH1D, ...}} - central histograms (stat errors only)
        all_syst_hists: Dict of {score_type: {process: {syst_name: {'up': TH1D, 'down': TH1D}}}}
    """
    histfile = ROOT.TFile.Open(histfile_path, "RECREATE")

    for score_type, hists in stored_hists.items():
        histfile.mkdir(score_type)
        histfile.cd(score_type)

        # Save central histograms (stat errors only)
        for proc_name, hist in hists.items():
            # Rename histogram for saving
            hist.SetName(proc_name)
            hist.Write()

        # Save ALL systematic variations for this score type
        if all_syst_hists and score_type in all_syst_hists:
            for proc_name, proc_systs in all_syst_hists[score_type].items():
                if proc_name == "data_obs":
                    continue
                for syst_name, variations in proc_systs.items():
                    histfile.cd(score_type)
                    up_hist = variations['up'].Clone(f"{proc_name}_{syst_name}Up")
                    down_hist = variations['down'].Clone(f"{proc_name}_{syst_name}Down")
                    up_hist.Write()
                    down_hist.Write()

    histfile.Close()
    logging.info(f"  Saved histograms: {histfile_path}")


# =============================================================================
# Combined Era Processing
# =============================================================================

def process_combined_era(region_label, sample_channel, show_data, region_outdir, ordered_bkgs):
    """
    Process combined era by loading and summing per-era histograms.

    Following TriLepton pattern: sum histograms across eras using TH1::Add
    which adds errors in quadrature.
    """
    logging.info(f"Processing region: {region_label} (combined era mode: {args.era})")

    stored_hists = {}  # {score_type: {process: hist, ...}}
    plot_types = list(HISTKEYS_CONFIG.keys())

    for plot_key in plot_types:
        logging.info(f"  Creating {plot_key} plot for {region_label}...")

        stored_hists[plot_key] = {}

        # Collect histograms from each era
        per_era_hists = {}  # {era: {hist_name: TH1D}}

        for era in era_list:
            hist_file = f"{WORKDIR}/SignalRegionStudyV2/templates/{era}/{args.channel}/{args.masspoint}/ParticleNet/{binning_suffix}/scores/{region_label}/histograms.root"

            if not os.path.exists(hist_file):
                logging.warning(f"Missing histogram file for {era}: {hist_file}")
                continue

            try:
                era_histograms = load_histograms_from_file(hist_file, plot_key)
                per_era_hists[era] = era_histograms
            except Exception as e:
                logging.warning(f"Failed to load {plot_key} from {era}: {e}")
                continue

        # Helper functions for decorrelated systematics
        def get_decorrelated_era(hist_name):
            """Return the era suffix if this is a decorrelated systematic, else None."""
            for era in era_list:
                if f"_{era}Up" in hist_name or f"_{era}Down" in hist_name:
                    return era
            return None

        def parse_syst_hist_name(hist_name):
            """Parse 'process_systUp' or 'process_systDown' into (process, syst, direction)."""
            if hist_name.endswith("Up"):
                direction = "Up"
                base = hist_name[:-2]
            elif hist_name.endswith("Down"):
                direction = "Down"
                base = hist_name[:-4]
            else:
                return None, None, None

            for proc in list(ordered_bkgs) + [args.masspoint]:
                if base.startswith(proc + "_"):
                    syst = base[len(proc) + 1:]
                    return proc, syst, direction
            return None, None, None

        # Collect all unique histogram names and categorize them
        all_hist_names = set()
        for era, hists in per_era_hists.items():
            all_hist_names.update(hists.keys())

        # Separate into central/correlated (sum directly) and decorrelated (special handling)
        central_and_correlated = []
        decorrelated_systs = {}  # {(process, syst, direction): source_era}

        for name in all_hist_names:
            decor_era = get_decorrelated_era(name)
            if decor_era is not None:
                proc, syst, direction = parse_syst_hist_name(name)
                if proc and syst and direction:
                    decorrelated_systs[(proc, syst, direction)] = decor_era
            else:
                central_and_correlated.append(name)

        # Combine central and correlated histograms by simple summation
        for name in central_and_correlated:
            hlist = []
            for era in era_list:
                if era in per_era_hists and name in per_era_hists[era]:
                    hlist.append(per_era_hists[era][name])
            if hlist:
                combined = sum_histograms(hlist, f"{name}_{args.era}")
                stored_hists[plot_key][name] = combined

        # Combine decorrelated systematics: varied_era + sum(central_other_eras)
        for (proc, syst, direction), source_era in decorrelated_systs.items():
            hist_name = f"{proc}_{syst}{direction}"

            if source_era not in per_era_hists or hist_name not in per_era_hists[source_era]:
                logging.debug(f"  Decorrelated syst {hist_name} not found in {source_era}")
                continue

            varied_hist = per_era_hists[source_era][hist_name].Clone(f"{hist_name}_{args.era}")
            varied_hist.SetDirectory(0)

            # Add central histograms from other eras
            for other_era in era_list:
                if other_era == source_era:
                    continue
                if other_era in per_era_hists and proc in per_era_hists[other_era]:
                    varied_hist.Add(per_era_hists[other_era][proc])

            stored_hists[plot_key][hist_name] = varied_hist

        logging.debug(f"  Combined {len(central_and_correlated)} central/correlated, "
                      f"{len(decorrelated_systs)} decorrelated systematics")

    return stored_hists


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    logging.info(f"Plotting Modified ParticleNet scores for {args.masspoint}, {args.era}, {args.channel}")

    # Helper function to scan for available processes
    def get_background_processes():
        """Get list of available background processes."""
        if is_combined_era:
            # For combined eras, get process list from first era's histogram file
            first_era_hist_file = f"{WORKDIR}/SignalRegionStudyV2/templates/{reference_era}/{args.channel}/{args.masspoint}/ParticleNet/{binning_suffix}/scores/{args.channel}/histograms.root"
            if not os.path.exists(first_era_hist_file):
                raise FileNotFoundError(
                    f"Per-era histogram file not found: {first_era_hist_file}\n"
                    f"Please run plotParticleNetScore.py for individual eras first"
                )
            first_score_type = list(HISTKEYS_CONFIG.keys())[0]
            first_era_hists = load_histograms_from_file(first_era_hist_file, first_score_type)
            sample_files = list(first_era_hists.keys())
            processes = [p for p in sample_files
                         if p != 'data_obs' and p != args.masspoint
                         and 'Up' not in p and 'Down' not in p]
            logging.info(f"Found background processes from {reference_era} histogram: {processes}")
        else:
            # For single era, scan samples directory
            if not os.path.exists(BASEDIR):
                raise FileNotFoundError(
                    f"Samples directory not found: {BASEDIR}\n"
                    f"Please run preprocessing first"
                )
            sample_files = [f.replace('.root', '') for f in os.listdir(BASEDIR) if f.endswith('.root')]
            processes = [p for p in sample_files if p != 'data' and p != args.masspoint]
            logging.info(f"Found background processes: {processes}")
        return processes

    BACKGROUND_PROCESSES = get_background_processes()

    # Order backgrounds using BKG_ORDER, then append any remaining
    ordered_bkgs = []
    for bkg in BKG_ORDER:
        if bkg in BACKGROUND_PROCESSES:
            ordered_bkgs.append(bkg)
    # Add any remaining backgrounds not in BKG_ORDER
    for bkg in BACKGROUND_PROCESSES:
        if bkg not in ordered_bkgs:
            ordered_bkgs.append(bkg)

    # Define regions to plot
    regions = [
        (args.channel, args.channel, args.unblind or args.partial_unblind),
        ("TTZ2E1Mu", "TTZ2E1Mu", True),  # Control region always unblinded
    ]

    # Build plot types from histkeys config
    plot_types = [(key, cfg["xTitle"]) for key, cfg in HISTKEYS_CONFIG.items()]

    for sample_channel, region_label, show_data in regions:
        # Output directory for this region
        region_outdir = f"{OUTDIR}/scores/{region_label}"
        os.makedirs(region_outdir, exist_ok=True)

        # =================================================================
        # Combined era mode: Load from per-era ROOT files and sum
        # =================================================================
        if is_combined_era:
            stored_hists = process_combined_era(region_label, sample_channel, show_data,
                                                 region_outdir, ordered_bkgs)

            # Generate plots for each score type
            for plot_key, x_title in plot_types:
                if plot_key not in stored_hists:
                    continue

                # Step 2: Rebin histograms
                rebin_factor = get_rebin_factor(plot_key)
                rebinned_hists = rebin_histograms(stored_hists[plot_key], plot_key)

                # Step 3: Calculate errors from combined syst histograms
                central_processes_for_syst = [p for p in ordered_bkgs if p in rebinned_hists]
                syst_hists_combined = extract_syst_hists(stored_hists[plot_key], central_processes_for_syst)
                rebinned_syst_hists = rebin_syst_histograms(syst_hists_combined, plot_key, rebin_factor)
                hists_with_errors = calculate_errors(
                    rebinned_hists, syst_categories, args.masspoint,
                    rebinned_syst_hists, rebin_factor
                )

                # Build background histograms for plotting
                bkg_hists = {}
                for proc in ordered_bkgs:
                    if proc not in hists_with_errors:
                        continue
                    color = get_color(proc)
                    hist = hists_with_errors[proc].Clone(f"{proc}_{plot_key}_{region_label}")
                    hist.SetLineColor(color)
                    hist.SetFillColor(color)
                    if hist.Integral() > 0:
                        bkg_hists[proc] = hist

                # Signal histogram
                if sample_channel != "TTZ2E1Mu" and args.masspoint in hists_with_errors:
                    signal_hist = hists_with_errors[args.masspoint].Clone(f"signal_{plot_key}_{region_label}")
                    if signal_hist.Integral() > 0:
                        style_signal_histogram(signal_hist)
                    else:
                        signal_hist = None
                else:
                    signal_hist = None

                # Data histogram
                data_hist = rebinned_hists.get('data_obs')
                if data_hist is None:
                    logging.warning(f"  No data_obs histogram for {plot_key}, skipping")
                    continue

                # Skip if no backgrounds
                if should_skip_plot(bkg_hists, signal_hist):
                    logging.warning(f"  Skipping {plot_key} for {region_label}: no data")
                    continue

                # Build colors list and configuration
                colors = build_colors_list(bkg_hists)
                enable_chi2 = args.unblind or sample_channel == "TTZ2E1Mu"
                config = build_canvas_config(
                    args.era, region_label, x_title, plot_key, colors,
                    get_CoM_energy(reference_era),
                    enable_chi2=enable_chi2
                )

                # Step 4: Draw plot
                draw_plot(data_hist, bkg_hists, signal_hist, config,
                         f"{region_outdir}/{plot_key}.png", args.masspoint)

            # Save combined histograms to ROOT file
            save_histograms_to_root(f"{region_outdir}/histograms.root", stored_hists)
            logging.info(f"Validation plots for {region_label} saved to: {region_outdir}")
            continue

        # =================================================================
        # Single era mode with --skip-histogram: Load from existing histograms
        # and recalculate errors from rebinned systematics
        # =================================================================
        if args.skip_histogram:
            hist_file = f"{region_outdir}/histograms.root"
            if not os.path.exists(hist_file):
                raise FileNotFoundError(
                    f"Histogram file not found: {hist_file}\n"
                    f"Please run without --skip-histogram first to generate histograms"
                )

            logging.info(f"Processing region: {region_label} (loading from existing histograms)")

            for plot_key, x_title in plot_types:
                logging.info(f"  Creating {plot_key} plot for {region_label} (from existing histograms)...")

                # Load all histograms including systematics from file
                all_hists = load_histograms_from_file(hist_file, plot_key)
                if not all_hists:
                    logging.warning(f"  No histograms found for {plot_key} in {hist_file}, skipping")
                    continue

                # Separate central and systematic histograms
                central_processes = [p for p in ordered_bkgs if p in all_hists]
                if args.masspoint in all_hists:
                    central_processes.append(args.masspoint)

                central_hists = {k: v for k, v in all_hists.items()
                                 if 'Up' not in k and 'Down' not in k}

                # Extract systematic histograms from loaded file
                syst_hists = extract_syst_hists(all_hists, central_processes)

                # Step 2: Rebin ALL histograms (central and systematics)
                rebin_factor = get_rebin_factor(plot_key)
                rebinned_hists = rebin_histograms(central_hists, plot_key)
                rebinned_syst_hists = rebin_syst_histograms(syst_hists, plot_key, rebin_factor)

                # Step 3: Calculate errors from rebinned systematics
                debug_bin = 10 if args.debug else None
                hists_with_errors = calculate_errors(
                    rebinned_hists, syst_categories, args.masspoint,
                    rebinned_syst_hists, rebin_factor, debug_bin
                )

                # Build background histograms for plotting (with errors applied)
                bkg_hists = {}
                for proc in ordered_bkgs:
                    if proc not in hists_with_errors:
                        continue
                    color = get_color(proc)
                    hist = hists_with_errors[proc].Clone(f"{proc}_{plot_key}_{region_label}")
                    hist.SetLineColor(color)
                    hist.SetFillColor(color)
                    if hist.Integral() > 0:
                        bkg_hists[proc] = hist

                # Signal histogram
                if sample_channel != "TTZ2E1Mu" and args.masspoint in hists_with_errors:
                    signal_hist = hists_with_errors[args.masspoint].Clone(f"signal_{plot_key}_{region_label}")
                    if signal_hist.Integral() > 0:
                        style_signal_histogram(signal_hist)
                    else:
                        signal_hist = None
                else:
                    signal_hist = None

                # Data histogram (from rebinned central, not hists_with_errors)
                data_hist = rebinned_hists.get('data_obs')
                if data_hist is None:
                    logging.warning(f"  No data_obs histogram for {plot_key}, skipping")
                    continue

                # Skip if no backgrounds
                if should_skip_plot(bkg_hists, signal_hist):
                    logging.warning(f"  Skipping {plot_key} for {region_label}: no data")
                    continue

                # Build colors list and configuration
                colors = build_colors_list(bkg_hists)
                enable_chi2 = args.unblind or sample_channel == "TTZ2E1Mu"
                config = build_canvas_config(
                    args.era, region_label, x_title, plot_key, colors,
                    get_CoM_energy(args.era),
                    enable_chi2=enable_chi2
                )

                # Step 4: Draw plot
                draw_plot(data_hist, bkg_hists, signal_hist, config,
                         f"{region_outdir}/{plot_key}.png", args.masspoint)

            logging.info(f"Validation plots for {region_label} saved to: {region_outdir}")
            continue

        # =================================================================
        # Single era mode: Process from raw samples
        # =================================================================
        sample_dir = f"{WORKDIR}/SignalRegionStudyV2/samples/{args.era}/{sample_channel}/{args.masspoint}"

        if not os.path.exists(sample_dir):
            logging.warning(f"Samples not found for {sample_channel} at {sample_dir}, skipping")
            continue

        # Use template binning mass window for SR, no mass cut for control region
        region_mass_min = MASS_MIN
        region_mass_max = MASS_MAX

        # Log region info
        if sample_channel == "TTZ2E1Mu":
            logging.info(f"Processing region: {region_label} (data: shown, no mass cut for control region)")
        else:
            logging.info(f"Processing region: {region_label} (data: {'shown' if show_data else 'blinded'}, mass: [{region_mass_min:.2f}, {region_mass_max:.2f}] GeV)")

        # Step 1: Create histograms
        stored_hists, all_syst_hists = create_histograms(
            args.era, args.channel, args.masspoint, sample_channel, syst_categories,
            region_mass_min, region_mass_max, ordered_bkgs, show_data
        )

        # Generate plots for each score type
        for plot_key, x_title in plot_types:
            logging.info(f"  Creating {plot_key} plot for {region_label}...")

            if plot_key not in stored_hists:
                continue

            # Step 2: Rebin histograms
            rebin_factor = get_rebin_factor(plot_key)
            rebinned_hists = rebin_histograms(stored_hists[plot_key], plot_key)

            # Rebin systematic histograms (now per-score_type)
            score_type_syst_hists = all_syst_hists.get(plot_key, {})
            rebinned_syst_hists = rebin_syst_histograms(score_type_syst_hists, plot_key, rebin_factor)

            # Step 3: Calculate errors
            debug_bin = 10 if args.debug else None
            hists_with_errors = calculate_errors(
                rebinned_hists, syst_categories, args.masspoint,
                rebinned_syst_hists, rebin_factor, debug_bin
            )

            # Build background histograms for plotting (with errors applied)
            bkg_hists = {}
            for proc in ordered_bkgs:
                if proc not in hists_with_errors:
                    continue
                color = get_color(proc)
                hist = hists_with_errors[proc].Clone(f"{proc}_{plot_key}_{region_label}")
                hist.SetLineColor(color)
                hist.SetFillColor(color)
                if hist.Integral() > 0:
                    bkg_hists[proc] = hist

            # Signal histogram
            if sample_channel != "TTZ2E1Mu" and args.masspoint in hists_with_errors:
                signal_hist = hists_with_errors[args.masspoint].Clone(f"signal_{plot_key}_{region_label}")
                if signal_hist.Integral() > 0:
                    style_signal_histogram(signal_hist)
                else:
                    signal_hist = None
            else:
                signal_hist = None

            # Data histogram (from rebinned, not hists_with_errors)
            data_hist = rebinned_hists.get('data_obs')
            if data_hist is None:
                logging.warning(f"  No data_obs histogram for {plot_key}, skipping")
                continue

            # Skip if no backgrounds
            if should_skip_plot(bkg_hists, signal_hist):
                logging.warning(f"  Skipping {plot_key} for {region_label}: no data")
                continue

            # Build colors list and configuration
            colors = build_colors_list(bkg_hists)
            enable_chi2 = args.unblind or sample_channel == "TTZ2E1Mu"
            config = build_canvas_config(
                args.era, region_label, x_title, plot_key, colors,
                get_CoM_energy(args.era),
                enable_chi2=enable_chi2
            )

            # Step 4: Draw plot
            draw_plot(data_hist, bkg_hists, signal_hist, config,
                     f"{region_outdir}/{plot_key}.png", args.masspoint)

        # Save histograms to ROOT file (central histograms have stat errors only,
        # systematic variations stored separately for proper error recalculation)
        save_histograms_to_root(f"{region_outdir}/histograms.root", stored_hists, all_syst_hists)
        logging.info(f"Validation plots for {region_label} saved to: {region_outdir}")
