#!/usr/bin/env python
"""
Multi-class classification visualization script for ParticleNetMD training results.

This script generates visualizations matching visualizeGAIteration.py output format,
plus mass decorrelation studies specific to ParticleNetMD.

Outputs (matching visualizeGAIteration.py):
- ks_test_heatmap.png - 4x4 KS p-value matrix for overfitting detection
- score_distributions_grid.png - 4x4 train/test score comparison
- roc_curves.png - Combined ROC curves via ROCCurveCalculator
- confusion_matrix.png - Single confusion matrix with Train(Test) format
- training_curves.png - Loss and accuracy curves (ROOT-based)
- kolmogorov.json + kolmogorov.root - KS test results

Additional (Mass Studies):
- score_vs_mass_{class}.png - 2D score vs mass per class
- mass_profile_vs_score.png - Mean mass vs score
- mass_sculpting.png - Mass distribution at score cuts

Usage:
    python visualizeMultiClass.py --signal MHc130_MA90 --channel Combined --pilot
"""

import os
import sys
import argparse
import json
import numpy as np
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

import ROOT
ROOT.gROOT.SetBatch(True)

from ROCCurveCalculator import ROCCurveCalculator
import logging

# Try to import CMS style
try:
    import cmsstyle as CMS
    HAS_CMS_STYLE = True
except ImportError:
    print("Warning: cmsstyle not available, using default ROOT style")
    HAS_CMS_STYLE = False

# Color palette consistent with Common/Tools/plotter.py
PALETTE = [
    ROOT.TColor.GetColor("#5790fc"),  # Blue - Signal
    ROOT.TColor.GetColor("#f89c20"),  # Orange - Nonprompt
    ROOT.TColor.GetColor("#e42536"),  # Red - Diboson
    ROOT.TColor.GetColor("#964a8b"),  # Purple - TTX
]

CLASS_NAMES = ['signal', 'nonprompt', 'diboson', 'ttX']
CLASS_DISPLAY_NAMES = ['Signal', 'Nonprompt', 'Diboson', 'TTX']
SCORE_NAMES = ['signal_score', 'nonprompt_score', 'diboson_score', 'ttX_score']

logging.basicConfig(level=logging.INFO)


def setup_cms_style():
    """Setup CMS style for ROOT plots."""
    if HAS_CMS_STYLE:
        CMS.setCMSStyle()
        CMS.SetEnergy(13)
        CMS.SetLumi(-1, run="Run2")
        CMS.SetExtraText("Simulation Preliminary")
    else:
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetPadLeftMargin(0.12)
        ROOT.gStyle.SetPadBottomMargin(0.12)


# =============================================================================
# File Discovery and Data Loading
# =============================================================================

def find_multiclass_results(signal, channel, fold=None, pilot=False):
    """Find multi-class classification results files for ParticleNetMD.

    Args:
        signal: Signal point name (e.g., "MHc130_MA90")
        channel: Analysis channel (e.g., "Combined")
        fold: Test fold number. If None, automatically detect from most recent results.
        pilot: Whether to use pilot mode results

    Returns:
        ga_json_file: GA-compatible JSON file path (for training history)
        root_file: ROOT file path with predictions tree
        result_dir: Result directory path
        detected_fold: The fold number used (useful when auto-detected)
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.join(script_dir, "..", "results")
    base_path = os.path.abspath(base_path)

    signal_full = f"TTToHcToWAToMuMu-{signal}"
    signal_base_dir = os.path.join(base_path, channel, "multiclass", signal_full)

    if pilot:
        result_dir = os.path.join(signal_base_dir, "pilot")
        detected_fold = None
    elif fold is not None:
        result_dir = os.path.join(signal_base_dir, f"fold-{fold}")
        detected_fold = fold
    else:
        # Auto-detect: search all fold-* directories for the most recent JSON
        if not os.path.exists(signal_base_dir):
            raise FileNotFoundError(f"Signal directory not found: {signal_base_dir}")

        all_candidates = []
        for subdir in os.listdir(signal_base_dir):
            if subdir.startswith("fold-"):
                subdir_path = os.path.join(signal_base_dir, subdir)
                if os.path.isdir(subdir_path):
                    for f in os.listdir(subdir_path):
                        if f.endswith('.json') and not f.endswith('_performance.json') and not f.endswith('_model_info.json'):
                            full_path = os.path.join(subdir_path, f)
                            all_candidates.append((full_path, os.path.getmtime(full_path), subdir_path))

        if not all_candidates:
            raise FileNotFoundError(f"No GA-compatible JSON files found in any fold directory under: {signal_base_dir}")

        # Sort by modification time (newest first)
        all_candidates.sort(key=lambda x: x[1], reverse=True)
        ga_json_file = all_candidates[0][0]
        result_dir = all_candidates[0][2]

        # Extract fold from the JSON file's test_folds
        with open(ga_json_file, 'r') as f:
            config_data = json.load(f)
        test_folds = config_data.get('hyperparameters', {}).get('test_folds', [4])
        detected_fold = test_folds[0] if test_folds else 4

        logging.info(f"Auto-detected fold {detected_fold} from most recent results")

        # Find ROOT file
        trees_dir = os.path.join(result_dir, "trees")
        root_candidates = []
        if os.path.exists(trees_dir):
            for f in os.listdir(trees_dir):
                if f.endswith('.root'):
                    full_path = os.path.join(trees_dir, f)
                    root_candidates.append((full_path, os.path.getmtime(full_path)))

        if not root_candidates:
            raise FileNotFoundError(f"ROOT file not found in: {trees_dir}")

        root_candidates.sort(key=lambda x: x[1], reverse=True)
        root_file = root_candidates[0][0]

        logging.info(f"Found GA JSON: {ga_json_file}")
        logging.info(f"Found ROOT file: {root_file}")

        return ga_json_file, root_file, result_dir, detected_fold

    if not os.path.exists(result_dir):
        raise FileNotFoundError(f"Result directory not found: {result_dir}")

    # Find GA-compatible JSON file (pick the most recently modified one)
    ga_json_candidates = []
    for f in os.listdir(result_dir):
        if f.endswith('.json') and not f.endswith('_performance.json') and not f.endswith('_model_info.json'):
            full_path = os.path.join(result_dir, f)
            ga_json_candidates.append((full_path, os.path.getmtime(full_path)))

    if not ga_json_candidates:
        raise FileNotFoundError(f"GA-compatible JSON file not found in: {result_dir}")

    # Sort by modification time (newest first) and pick the most recent
    ga_json_candidates.sort(key=lambda x: x[1], reverse=True)
    ga_json_file = ga_json_candidates[0][0]

    # Find ROOT file in trees/ subdirectory (pick the most recently modified one)
    trees_dir = os.path.join(result_dir, "trees")
    root_candidates = []
    if os.path.exists(trees_dir):
        for f in os.listdir(trees_dir):
            if f.endswith('.root'):
                full_path = os.path.join(trees_dir, f)
                root_candidates.append((full_path, os.path.getmtime(full_path)))

    if not root_candidates:
        raise FileNotFoundError(f"ROOT file not found in: {trees_dir}")

    # Sort by modification time (newest first) and pick the most recent
    root_candidates.sort(key=lambda x: x[1], reverse=True)
    root_file = root_candidates[0][0]

    logging.info(f"Found GA JSON: {ga_json_file}")
    logging.info(f"Found ROOT file: {root_file}")

    return ga_json_file, root_file, result_dir, detected_fold


def load_training_data(ga_json_file):
    """Load training performance data from GA-compatible JSON file."""
    with open(ga_json_file, 'r') as f:
        data = json.load(f)
    return data


def load_multiclass_predictions_from_root(root_file):
    """Load multiclass predictions from ROOT tree file for ParticleNetMD.

    Returns:
        Tuple of (y_true_train, y_scores_train, weights_train, mass1_train, mass2_train, has_bjet_train,
                  y_true_test, y_scores_test, weights_test, mass1_test, mass2_test, has_bjet_test,
                  class_names)
    """
    if not os.path.exists(root_file):
        raise FileNotFoundError(f"ROOT file not found: {root_file}")

    f = ROOT.TFile.Open(root_file)
    if not f or f.IsZombie():
        raise RuntimeError(f"Failed to open ROOT file: {root_file}")

    tree = f.Get("Events")
    if not tree:
        raise RuntimeError(f"Tree 'Events' not found in {root_file}")

    n_entries = tree.GetEntries()
    logging.info(f"Loading {n_entries} entries from ROOT file...")

    # Prepare arrays
    score_signal = np.zeros(n_entries)
    score_nonprompt = np.zeros(n_entries)
    score_diboson = np.zeros(n_entries)
    score_ttX = np.zeros(n_entries)
    y_true = np.zeros(n_entries, dtype=np.int32)
    train_mask = np.zeros(n_entries, dtype=bool)
    test_mask = np.zeros(n_entries, dtype=bool)
    sample_weights = np.zeros(n_entries)
    mass1 = np.zeros(n_entries)
    mass2 = np.zeros(n_entries)
    has_bjet = np.zeros(n_entries)

    # Read tree entries
    for i, event in enumerate(tree):
        score_signal[i] = event.score_signal
        score_nonprompt[i] = event.score_nonprompt
        score_diboson[i] = event.score_diboson
        score_ttX[i] = event.score_ttX
        y_true[i] = event.true_label
        train_mask[i] = event.train_mask
        test_mask[i] = event.test_mask
        sample_weights[i] = event.weight
        mass1[i] = event.mass1
        mass2[i] = event.mass2
        has_bjet[i] = event.has_bjet

    f.Close()

    # Stack scores into proper shape (n_samples, n_classes)
    y_scores = np.column_stack([score_signal, score_nonprompt, score_diboson, score_ttX])

    # Split into train and test sets
    y_true_train = y_true[train_mask]
    y_scores_train = y_scores[train_mask]
    weights_train = sample_weights[train_mask]
    mass1_train = mass1[train_mask]
    mass2_train = mass2[train_mask]
    has_bjet_train = has_bjet[train_mask]

    y_true_test = y_true[test_mask]
    y_scores_test = y_scores[test_mask]
    weights_test = sample_weights[test_mask]
    mass1_test = mass1[test_mask]
    mass2_test = mass2[test_mask]
    has_bjet_test = has_bjet[test_mask]

    logging.info(f"Loaded {len(y_true_train)} train + {len(y_true_test)} test samples from ROOT file")
    logging.info(f"Physics weight range: {sample_weights.min():.6f} to {sample_weights.max():.6f}")

    valid_mass1 = mass1[mass1 > 0]
    if len(valid_mass1) > 0:
        logging.info(f"Mass1 range: {valid_mass1.min():.1f} to {valid_mass1.max():.1f} GeV")

    # Print class distribution for test set
    for i, class_name in enumerate(CLASS_DISPLAY_NAMES):
        count_test = np.sum(y_true_test == i)
        weight_test = np.sum(weights_test[y_true_test == i])
        logging.info(f"  {class_name} (test): {count_test} events, total weight: {weight_test:.3f}")

    return (y_true_train, y_scores_train, weights_train, mass1_train, mass2_train, has_bjet_train,
            y_true_test, y_scores_test, weights_test, mass1_test, mass2_test, has_bjet_test,
            CLASS_DISPLAY_NAMES)


# =============================================================================
# Histogram Utilities for KS Tests (from visualizeGAIteration.py)
# =============================================================================

def create_weighted_histogram(scores: np.ndarray, weights: np.ndarray, name: str,
                              title: str, nbins: int = 50) -> ROOT.TH1D:
    """Create weighted histogram from scores."""
    hist = ROOT.TH1D(name, title, nbins, 0.0, 1.0)
    hist.SetDirectory(0)

    for score, weight in zip(scores, weights):
        hist.Fill(float(score), float(weight))

    # Normalize to unit area
    if hist.Integral() > 0:
        hist.Scale(1.0 / hist.Integral())

    return hist


def identify_problematic_bins(hist: ROOT.TH1D, threshold: float = 1e-6) -> List[int]:
    """Identify bins with negative content or very small absolute content."""
    problematic_bins = []
    nbins = hist.GetNbinsX()

    for i in range(1, nbins + 1):
        content = hist.GetBinContent(i)
        if content < 0 or abs(content) < threshold:
            problematic_bins.append(i)

    return problematic_bins


def create_merge_groups(hist_train: ROOT.TH1D, hist_test: ROOT.TH1D,
                       threshold: float = 1e-6) -> List[Tuple[int, int]]:
    """Create groups of consecutive bins to merge based on problematic bins."""
    prob_train = set(identify_problematic_bins(hist_train, threshold))
    prob_test = set(identify_problematic_bins(hist_test, threshold))
    problematic_bins = sorted(prob_train | prob_test)

    if not problematic_bins:
        return []

    nbins = hist_train.GetNbinsX()
    merge_groups = []
    i = 0

    while i < len(problematic_bins):
        group_start = problematic_bins[i]
        group_end = group_start

        cum_content_train = hist_train.GetBinContent(group_start)
        cum_abs_train = abs(cum_content_train)
        cum_content_test = hist_test.GetBinContent(group_start)
        cum_abs_test = abs(cum_content_test)

        while (cum_abs_train < threshold or cum_content_train < 0 or
               cum_abs_test < threshold or cum_content_test < 0):

            next_bin = group_end + 1
            if next_bin > nbins:
                break

            group_end = next_bin
            cum_content_train += hist_train.GetBinContent(next_bin)
            cum_abs_train += abs(hist_train.GetBinContent(next_bin))
            cum_content_test += hist_test.GetBinContent(next_bin)
            cum_abs_test += abs(hist_test.GetBinContent(next_bin))

            if i + 1 < len(problematic_bins) and problematic_bins[i + 1] == next_bin:
                i += 1

        if group_end > group_start:
            merge_groups.append((group_start, group_end))

        i += 1

    return merge_groups


def apply_bin_merging(hist_train: ROOT.TH1D, hist_test: ROOT.TH1D,
                     merge_groups: List[Tuple[int, int]]) -> Tuple[ROOT.TH1D, ROOT.TH1D]:
    """Apply bin merging to both histograms using the same merge groups."""
    if not merge_groups:
        h_train_new = hist_train.Clone(f"{hist_train.GetName()}_merged")
        h_test_new = hist_test.Clone(f"{hist_test.GetName()}_merged")
        h_train_new.SetDirectory(0)
        h_test_new.SetDirectory(0)
        return h_train_new, h_test_new

    nbins_old = hist_train.GetNbinsX()
    xmin = hist_train.GetXaxis().GetXmin()
    xmax = hist_train.GetXaxis().GetXmax()

    merged_bins = set()
    for start, end in merge_groups:
        for b in range(start, end + 1):
            merged_bins.add(b)

    bin_edges = [xmin]
    bin_width = (xmax - xmin) / nbins_old

    old_bin = 1
    while old_bin <= nbins_old:
        in_merge_group = False
        for start, end in merge_groups:
            if old_bin == start:
                current_edge = xmin + end * bin_width
                bin_edges.append(current_edge)
                old_bin = end + 1
                in_merge_group = True
                break

        if not in_merge_group:
            if old_bin not in merged_bins:
                current_edge = xmin + old_bin * bin_width
                bin_edges.append(current_edge)
            old_bin += 1

    nbins_new = len(bin_edges) - 1
    bin_edges_array = np.array(bin_edges, dtype=np.float64)

    h_train_new = ROOT.TH1D(f"{hist_train.GetName()}_merged", hist_train.GetTitle(),
                            nbins_new, bin_edges_array)
    h_test_new = ROOT.TH1D(f"{hist_test.GetName()}_merged", hist_test.GetTitle(),
                           nbins_new, bin_edges_array)
    h_train_new.SetDirectory(0)
    h_test_new.SetDirectory(0)

    new_bin = 1
    old_bin = 1

    while old_bin <= nbins_old:
        is_merged = False
        for start, end in merge_groups:
            if old_bin == start:
                content_train = 0.0
                error2_train = 0.0
                content_test = 0.0
                error2_test = 0.0

                for b in range(start, end + 1):
                    content_train += hist_train.GetBinContent(b)
                    error2_train += hist_train.GetBinError(b) ** 2
                    content_test += hist_test.GetBinContent(b)
                    error2_test += hist_test.GetBinError(b) ** 2

                h_train_new.SetBinContent(new_bin, content_train)
                h_train_new.SetBinError(new_bin, np.sqrt(error2_train))
                h_test_new.SetBinContent(new_bin, content_test)
                h_test_new.SetBinError(new_bin, np.sqrt(error2_test))

                new_bin += 1
                old_bin = end + 1
                is_merged = True
                break

        if not is_merged:
            if old_bin not in merged_bins:
                h_train_new.SetBinContent(new_bin, hist_train.GetBinContent(old_bin))
                h_train_new.SetBinError(new_bin, hist_train.GetBinError(old_bin))
                h_test_new.SetBinContent(new_bin, hist_test.GetBinContent(old_bin))
                h_test_new.SetBinError(new_bin, hist_test.GetBinError(old_bin))
                new_bin += 1
            old_bin += 1

    return h_train_new, h_test_new


def merge_histograms_iteratively(hist_train: ROOT.TH1D, hist_test: ROOT.TH1D,
                                 threshold: float = 1e-6, max_iterations: int = 100) -> Tuple[ROOT.TH1D, ROOT.TH1D, int]:
    """Iteratively merge bins in both histograms until no problematic bins remain."""
    h_train = hist_train.Clone(f"{hist_train.GetName()}_itermerge")
    h_test = hist_test.Clone(f"{hist_test.GetName()}_itermerge")
    h_train.SetDirectory(0)
    h_test.SetDirectory(0)

    iteration = 0

    while iteration < max_iterations:
        prob_train = identify_problematic_bins(h_train, threshold)
        prob_test = identify_problematic_bins(h_test, threshold)

        if not prob_train and not prob_test:
            break

        merge_groups = create_merge_groups(h_train, h_test, threshold)

        if not merge_groups:
            break

        h_train, h_test = apply_bin_merging(h_train, h_test, merge_groups)
        iteration += 1

    if iteration >= max_iterations:
        logging.warning(f"Bin merging reached max iterations ({max_iterations}) for {hist_train.GetName()}")

    return h_train, h_test, iteration


# =============================================================================
# KS Test Functions (from visualizeGAIteration.py)
# =============================================================================

def perform_ks_tests_comprehensive(y_true_train: np.ndarray, y_scores_train: np.ndarray, weights_train: np.ndarray,
                                   y_true_test: np.ndarray, y_scores_test: np.ndarray, weights_test: np.ndarray,
                                   p_threshold: float = 0.05,
                                   bin_merge_threshold: float = 1e-6,
                                   bin_merge_max_iterations: int = 100) -> Tuple[Dict, Dict]:
    """Perform comprehensive KS tests: all 4 output scores for each of 4 true classes."""
    num_classes = 4
    ks_results = {}
    histograms = {}

    for true_class in range(num_classes):
        train_mask = (y_true_train == true_class)
        test_mask = (y_true_test == true_class)

        train_scores_class = y_scores_train[train_mask]
        test_scores_class = y_scores_test[test_mask]
        train_weights_class = weights_train[train_mask]
        test_weights_class = weights_test[test_mask]

        for score_idx in range(num_classes):
            train_scores = train_scores_class[:, score_idx]
            test_scores = test_scores_class[:, score_idx]

            key = f"{CLASS_NAMES[true_class]}_{SCORE_NAMES[score_idx]}"

            h_train = create_weighted_histogram(
                train_scores, train_weights_class,
                f"h_train_{key}", f"Train: {CLASS_DISPLAY_NAMES[true_class]} - {SCORE_NAMES[score_idx]}"
            )
            h_test = create_weighted_histogram(
                test_scores, test_weights_class,
                f"h_test_{key}", f"Test: {CLASS_DISPLAY_NAMES[true_class]} - {SCORE_NAMES[score_idx]}"
            )

            h_train_merged, h_test_merged, n_merge_iterations = merge_histograms_iteratively(
                h_train, h_test, threshold=bin_merge_threshold, max_iterations=bin_merge_max_iterations
            )

            p_value = h_train_merged.KolmogorovTest(h_test_merged, option="X")

            ks_results[key] = {
                'true_class': CLASS_NAMES[true_class],
                'score_type': SCORE_NAMES[score_idx],
                'p_value': float(p_value),
                'is_overfitted': bool(p_value < p_threshold),
                'n_train': int(train_mask.sum()),
                'n_test': int(test_mask.sum()),
                'n_merge_iterations': int(n_merge_iterations),
                'n_bins_original': int(h_train.GetNbinsX()),
                'n_bins_merged': int(h_train_merged.GetNbinsX())
            }

            histograms[key] = {
                'train': h_train_merged,
                'test': h_test_merged
            }

    return ks_results, histograms


def save_ks_results(ks_results: Dict, histograms: Dict, output_dir: str):
    """Save KS test results to JSON and ROOT files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON results
    json_path = os.path.join(output_dir, "kolmogorov.json")
    with open(json_path, 'w') as f:
        json.dump(ks_results, f, indent=2)

    # Save ROOT file with histograms
    root_path = os.path.join(output_dir, "kolmogorov.root")
    root_file = ROOT.TFile(root_path, "RECREATE")

    for key, hists in histograms.items():
        hists['train'].Write(f"h_train_{key}")
        hists['test'].Write(f"h_test_{key}")

    root_file.Close()

    any_overfitted = any(result['is_overfitted'] for result in ks_results.values())
    min_p_value = min(result['p_value'] for result in ks_results.values())

    logging.info(f"KS results saved to: {json_path}")
    logging.info(f"KS histograms saved to: {root_path}")

    return any_overfitted, min_p_value


# =============================================================================
# Plotting Functions (from visualizeGAIteration.py)
# =============================================================================

def plot_ks_test_heatmap(ks_results: Dict, hyperparams: Dict, output_path: str):
    """Plot heatmap of p-values for all 16 KS tests."""
    p_matrix = np.zeros((4, 4))

    for true_class_idx in range(4):
        for score_idx in range(4):
            key = f"{CLASS_NAMES[true_class_idx]}_{SCORE_NAMES[score_idx]}"
            p_matrix[true_class_idx, score_idx] = ks_results[key]['p_value']

    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(p_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels([s.replace('_score', '') for s in SCORE_NAMES])
    ax.set_yticklabels([f"True {name}" for name in CLASS_DISPLAY_NAMES])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(4):
        for j in range(4):
            p_val = p_matrix[i, j]
            color = 'white' if p_val < 0.3 else 'black'
            ax.text(j, i, f'{p_val:.3f}', ha="center", va="center", color=color, fontsize=10)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('KS Test p-value', rotation=270, labelpad=20)

    title = "KS Test p-values (Overfitting Detection)\n"
    title += f"({hyperparams.get('num_hidden', 'N/A')} nodes, {hyperparams.get('optimizer', 'N/A')}, "
    title += f"lr={hyperparams.get('initial_lr', 'N/A')}, wd={hyperparams.get('weight_decay', 'N/A')})"
    ax.set_title(title, fontsize=12, pad=20)

    ax.set_xlabel('Output Score Type', fontsize=11)
    ax.set_ylabel('True Class', fontsize=11)

    plt.tight_layout()
    output_file = os.path.join(output_path, 'ks_test_heatmap.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"KS test heatmap saved to: {output_file}")


def plot_score_distributions_grid_bjet(y_true_test: np.ndarray, y_scores_test: np.ndarray,
                                        weights_test: np.ndarray, has_bjet_test: np.ndarray,
                                        output_path: str):
    """Plot 4x4 grid of score distributions split by has_bjet (0 vs 1).

    For each (true_class, score_type) cell, overlays:
    - Blue: events with has_bjet=0 (no b-jets)
    - Red: events with has_bjet=1 (has b-jets)
    """
    setup_cms_style()

    canvas = ROOT.TCanvas("c_grid", "Score Distributions", 2400, 2400)
    canvas.Divide(4, 4, 0.001, 0.001)

    # Pre-compute masks for has_bjet split
    no_bjet_mask = (has_bjet_test == 0)
    has_bjet_mask = (has_bjet_test == 1)

    # Log has_bjet distribution per class
    logging.info("has_bjet distribution per class:")
    for i, name in enumerate(CLASS_DISPLAY_NAMES):
        class_mask = (y_true_test == i)
        n_no_bjet = np.sum(class_mask & no_bjet_mask)
        n_has_bjet = np.sum(class_mask & has_bjet_mask)
        logging.info(f"  {name}: no_bjet={n_no_bjet}, has_bjet={n_has_bjet}")

    # Keep all ROOT objects alive to prevent garbage collection before canvas.SaveAs
    keep_alive = []

    pad_idx = 1
    for true_class_idx in range(4):
        for score_idx in range(4):
            canvas.cd(pad_idx)
            ROOT.gStyle.SetOptTitle(1)
            ROOT.gPad.SetLeftMargin(0.15)
            ROOT.gPad.SetRightMargin(0.05)
            ROOT.gPad.SetTopMargin(0.10)
            ROOT.gPad.SetBottomMargin(0.12)

            # Get events for this true class
            class_mask = (y_true_test == true_class_idx)

            # Create histograms for no b-jet and has b-jet
            h_name_no = f"h_no_bjet_{true_class_idx}_{score_idx}"
            h_name_has = f"h_has_bjet_{true_class_idx}_{score_idx}"

            h_no_bjet = ROOT.TH1F(h_name_no, f"True: {CLASS_DISPLAY_NAMES[true_class_idx]}", 50, 0, 1)
            h_has_bjet = ROOT.TH1F(h_name_has, "", 50, 0, 1)
            h_no_bjet.SetDirectory(0)
            h_has_bjet.SetDirectory(0)

            # Fill histograms
            combined_no = class_mask & no_bjet_mask
            combined_has = class_mask & has_bjet_mask

            scores_col = y_scores_test[:, score_idx]

            for score, w in zip(scores_col[combined_no], weights_test[combined_no]):
                h_no_bjet.Fill(score, abs(w))
            for score, w in zip(scores_col[combined_has], weights_test[combined_has]):
                h_has_bjet.Fill(score, abs(w))

            # Normalize
            if h_no_bjet.Integral() > 0:
                h_no_bjet.Scale(1.0 / h_no_bjet.Integral())
            if h_has_bjet.Integral() > 0:
                h_has_bjet.Scale(1.0 / h_has_bjet.Integral())

            # Style
            h_no_bjet.SetLineColor(ROOT.kBlue)
            h_no_bjet.SetLineWidth(2)
            h_no_bjet.SetLineStyle(1)

            h_has_bjet.SetLineColor(ROOT.kRed)
            h_has_bjet.SetLineWidth(2)
            h_has_bjet.SetLineStyle(1)

            # Determine which histogram to draw first (the one with more entries)
            # This ensures proper frame creation even when one category is empty
            max_val = max(h_no_bjet.GetMaximum(), h_has_bjet.GetMaximum())
            if max_val == 0:
                max_val = 0.1  # Default range when both are empty

            # Determine draw order: histogram with entries draws first to create frame
            if h_has_bjet.GetEntries() > h_no_bjet.GetEntries():
                h_first, h_second = h_has_bjet, h_no_bjet
            else:
                h_first, h_second = h_no_bjet, h_has_bjet

            h_first.SetMaximum(max_val * 1.3)
            h_first.SetMinimum(0)
            h_first.SetXTitle(SCORE_NAMES[score_idx].replace('_', ' '))
            h_first.SetYTitle("Normalized")
            h_first.GetXaxis().SetTitleSize(0.05)
            h_first.GetYaxis().SetTitleSize(0.05)
            h_first.GetXaxis().SetLabelSize(0.04)
            h_first.GetYaxis().SetLabelSize(0.04)

            # Set title on first histogram
            h_first.SetTitle(f"True: {CLASS_DISPLAY_NAMES[true_class_idx]}")

            h_first.Draw("HIST")
            h_second.Draw("HIST SAME")

            legend = ROOT.TLegend(0.55, 0.75, 0.9, 0.88)
            legend.SetTextSize(0.035)
            legend.SetBorderSize(0)
            legend.SetFillStyle(0)
            legend.AddEntry(h_no_bjet, "No B-jets", "L")
            legend.AddEntry(h_has_bjet, "Has B-jets", "L")
            legend.Draw("SAME")

            # Keep references to prevent garbage collection
            keep_alive.extend([h_no_bjet, h_has_bjet, legend])

            ROOT.gPad.Update()
            pad_idx += 1

    canvas.Update()
    output_file = os.path.join(output_path, 'score_distributions_grid_bjet.png')
    canvas.SaveAs(output_file)
    canvas.Close()

    logging.info(f"Score distributions grid (bjet split) saved to: {output_file}")


def plot_score_distributions_grid_train_test(y_true_train: np.ndarray, y_scores_train: np.ndarray,
                                              weights_train: np.ndarray,
                                              y_true_test: np.ndarray, y_scores_test: np.ndarray,
                                              weights_test: np.ndarray,
                                              output_path: str):
    """Plot 4x4 grid of score distributions comparing train vs test.

    For each (true_class, score_type) cell, overlays:
    - Blue: train events
    - Red: test events
    """
    setup_cms_style()

    canvas = ROOT.TCanvas("c_grid_tt", "Score Distributions (Train/Test)", 2400, 2400)
    canvas.Divide(4, 4, 0.001, 0.001)

    # Keep all ROOT objects alive to prevent garbage collection before canvas.SaveAs
    keep_alive = []

    pad_idx = 1
    for true_class_idx in range(4):
        for score_idx in range(4):
            canvas.cd(pad_idx)
            ROOT.gStyle.SetOptTitle(1)
            ROOT.gPad.SetLeftMargin(0.15)
            ROOT.gPad.SetRightMargin(0.05)
            ROOT.gPad.SetTopMargin(0.10)
            ROOT.gPad.SetBottomMargin(0.12)

            # Get events for this true class
            train_class_mask = (y_true_train == true_class_idx)
            test_class_mask = (y_true_test == true_class_idx)

            # Create histograms for train and test
            h_name_train = f"h_train_{true_class_idx}_{score_idx}"
            h_name_test = f"h_test_{true_class_idx}_{score_idx}"

            h_train = ROOT.TH1F(h_name_train, f"True: {CLASS_DISPLAY_NAMES[true_class_idx]}", 50, 0, 1)
            h_test = ROOT.TH1F(h_name_test, "", 50, 0, 1)
            h_train.SetDirectory(0)
            h_test.SetDirectory(0)

            # Fill histograms
            scores_train = y_scores_train[:, score_idx]
            scores_test = y_scores_test[:, score_idx]

            for score, w in zip(scores_train[train_class_mask], weights_train[train_class_mask]):
                h_train.Fill(score, abs(w))
            for score, w in zip(scores_test[test_class_mask], weights_test[test_class_mask]):
                h_test.Fill(score, abs(w))

            # Normalize
            if h_train.Integral() > 0:
                h_train.Scale(1.0 / h_train.Integral())
            if h_test.Integral() > 0:
                h_test.Scale(1.0 / h_test.Integral())

            # Style
            h_train.SetLineColor(ROOT.kBlue)
            h_train.SetLineWidth(2)
            h_train.SetLineStyle(1)

            h_test.SetLineColor(ROOT.kRed)
            h_test.SetLineWidth(2)
            h_test.SetLineStyle(1)

            # Determine y-axis range
            max_val = max(h_train.GetMaximum(), h_test.GetMaximum())
            if max_val == 0:
                max_val = 0.1

            # Draw order: histogram with more entries first
            if h_test.GetEntries() > h_train.GetEntries():
                h_first, h_second = h_test, h_train
            else:
                h_first, h_second = h_train, h_test

            h_first.SetMaximum(max_val * 1.3)
            h_first.SetMinimum(0)
            h_first.SetXTitle(SCORE_NAMES[score_idx].replace('_', ' '))
            h_first.SetYTitle("Normalized")
            h_first.GetXaxis().SetTitleSize(0.05)
            h_first.GetYaxis().SetTitleSize(0.05)
            h_first.GetXaxis().SetLabelSize(0.04)
            h_first.GetYaxis().SetLabelSize(0.04)

            h_first.SetTitle(f"True: {CLASS_DISPLAY_NAMES[true_class_idx]}")

            h_first.Draw("HIST")
            h_second.Draw("HIST SAME")

            legend = ROOT.TLegend(0.55, 0.75, 0.9, 0.88)
            legend.SetTextSize(0.035)
            legend.SetBorderSize(0)
            legend.SetFillStyle(0)
            legend.AddEntry(h_train, "Train", "L")
            legend.AddEntry(h_test, "Test", "L")
            legend.Draw("SAME")

            # Keep references to prevent garbage collection
            keep_alive.extend([h_train, h_test, legend])

            ROOT.gPad.Update()
            pad_idx += 1

    canvas.Update()
    output_file = os.path.join(output_path, 'score_distributions_grid_train_test.png')
    canvas.SaveAs(output_file)
    canvas.Close()

    logging.info(f"Score distributions grid (train/test) saved to: {output_file}")


def plot_roc_curves(y_true_train: np.ndarray, y_scores_train: np.ndarray, weights_train: np.ndarray,
                   y_true_test: np.ndarray, y_scores_test: np.ndarray, weights_test: np.ndarray,
                   output_path: str):
    """Plot ROC curves using ROCCurveCalculator."""
    calculator = ROCCurveCalculator()
    output_file = os.path.join(output_path, 'roc_curves.png')
    calculator.plot_multiclass_rocs(
        y_true_train, y_scores_train, weights_train,
        y_true_test, y_scores_test, weights_test,
        output_file, model_idx=0, class_names=CLASS_DISPLAY_NAMES
    )
    logging.info(f"ROC curves saved to: {output_file}")


def plot_confusion_matrices(y_true_train: np.ndarray, y_scores_train: np.ndarray, weights_train: np.ndarray,
                           y_true_test: np.ndarray, y_scores_test: np.ndarray, weights_test: np.ndarray,
                           output_path: str):
    """Plot confusion matrix with train(test) format."""
    y_pred_train = np.argmax(y_scores_train, axis=1)
    y_pred_test = np.argmax(y_scores_test, axis=1)

    # Compute weighted confusion matrices
    cm_train = np.zeros((4, 4))
    cm_test = np.zeros((4, 4))

    for true_label in range(4):
        mask_train = (y_true_train == true_label)
        if mask_train.sum() > 0:
            for pred_label in range(4):
                pred_mask = (y_pred_train == pred_label)
                cm_train[true_label, pred_label] = weights_train[mask_train & pred_mask].sum()

        mask_test = (y_true_test == true_label)
        if mask_test.sum() > 0:
            for pred_label in range(4):
                pred_mask = (y_pred_test == pred_label)
                cm_test[true_label, pred_label] = weights_test[mask_test & pred_mask].sum()

    # Normalize by row (true class)
    cm_train_normalized = cm_train / (cm_train.sum(axis=1, keepdims=True) + 1e-10)
    cm_test_normalized = cm_test / (cm_test.sum(axis=1, keepdims=True) + 1e-10)

    fig, ax = plt.subplots(figsize=(12, 10))

    im = ax.imshow(cm_train_normalized, cmap='Blues', vmin=0, vmax=1)

    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels(CLASS_DISPLAY_NAMES)
    ax.set_yticklabels(CLASS_DISPLAY_NAMES)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Text annotations with train(test) format
    for i in range(4):
        for j in range(4):
            val_train = cm_train_normalized[i, j]
            val_test = cm_test_normalized[i, j]
            color = 'white' if val_train > 0.5 else 'black'
            text_str = f'{val_train:.3f}\n({val_test:.3f})'
            ax.text(j, i, text_str, ha="center", va="center",
                   color=color, fontsize=11, fontweight='bold')

    ax.set_xlabel('Predicted Class', fontsize=13)
    ax.set_ylabel('True Class', fontsize=13)
    ax.set_title('Confusion Matrix [Train (Test)]', fontsize=14, pad=15)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fraction (Train)', rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    output_file = os.path.join(output_path, 'confusion_matrix.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Confusion matrix saved to: {output_file}")


def plot_training_curves(config_data: Dict, output_path: str):
    """Plot training curves from JSON history using ROOT.

    Shows 3 panels: CE Loss, DisCo term, and Accuracy.
    """
    if 'epoch_history' not in config_data:
        logging.warning("No epoch history found - skipping training curves")
        return

    history = config_data['epoch_history']
    epochs = history.get('epoch', [])

    if not epochs:
        return

    # Check if decomposed losses are available
    has_decomposed = 'train_ce_loss' in history and 'train_disco_term' in history

    if not has_decomposed:
        logging.warning("Decomposed losses (CE, DisCo) not found - using aggregated loss")

    best_epoch = config_data.get('training_summary', {}).get('best_epoch', -1)

    setup_cms_style()

    canvas = ROOT.TCanvas("c_training", "Training Curves", 2100, 600)
    canvas.Divide(3, 1, 0.01, 0.01)

    n_epochs = len(epochs)

    # ===== CE LOSS PLOT =====
    pad1 = canvas.cd(1)
    pad1.SetLeftMargin(0.12)
    pad1.SetRightMargin(0.05)
    pad1.SetTopMargin(0.08)
    pad1.SetBottomMargin(0.12)
    pad1.SetGrid()

    gr_train_ce = ROOT.TGraph(n_epochs)
    gr_valid_ce = ROOT.TGraph(n_epochs)

    if has_decomposed:
        for i, (epoch, train_ce, valid_ce) in enumerate(zip(epochs, history['train_ce_loss'], history['valid_ce_loss'])):
            gr_train_ce.SetPoint(i, epoch, train_ce)
            gr_valid_ce.SetPoint(i, epoch, valid_ce)
        ce_train_vals = history['train_ce_loss']
        ce_valid_vals = history['valid_ce_loss']
    else:
        # Fallback to aggregated loss
        for i, (epoch, train_loss, valid_loss) in enumerate(zip(epochs, history['train_loss'], history['valid_loss'])):
            gr_train_ce.SetPoint(i, epoch, train_loss)
            gr_valid_ce.SetPoint(i, epoch, valid_loss)
        ce_train_vals = history['train_loss']
        ce_valid_vals = history['valid_loss']

    gr_train_ce.SetLineColor(ROOT.kBlue)
    gr_train_ce.SetLineWidth(2)
    gr_train_ce.SetMarkerColor(ROOT.kBlue)
    gr_train_ce.SetMarkerStyle(20)
    gr_train_ce.SetMarkerSize(0.8)

    gr_valid_ce.SetLineColor(ROOT.kRed)
    gr_valid_ce.SetLineWidth(2)
    gr_valid_ce.SetMarkerColor(ROOT.kRed)
    gr_valid_ce.SetMarkerStyle(21)
    gr_valid_ce.SetMarkerSize(0.8)

    min_ce = min(min(ce_train_vals), min(ce_valid_vals))
    max_ce = max(max(ce_train_vals), max(ce_valid_vals))
    ce_range = max_ce - min_ce
    y_min_ce = max(0, min_ce - 0.1 * ce_range)
    y_max_ce = max_ce + 0.1 * ce_range

    frame1 = pad1.DrawFrame(min(epochs), y_min_ce, max(epochs), y_max_ce)
    frame1.SetTitle("Cross-Entropy Loss")
    frame1.GetXaxis().SetTitle("Epoch")
    frame1.GetYaxis().SetTitle("CE Loss")
    frame1.GetXaxis().SetTitleSize(0.05)
    frame1.GetYaxis().SetTitleSize(0.05)

    gr_train_ce.Draw("LP SAME")
    gr_valid_ce.Draw("LP SAME")

    if best_epoch >= 0:
        line_ce = ROOT.TLine(best_epoch, y_min_ce, best_epoch, y_max_ce)
        line_ce.SetLineColor(ROOT.kGreen+2)
        line_ce.SetLineWidth(2)
        line_ce.SetLineStyle(2)
        line_ce.Draw()

    legend1 = ROOT.TLegend(0.55, 0.65, 0.90, 0.88)
    legend1.SetBorderSize(0)
    legend1.SetFillStyle(0)
    legend1.SetTextSize(0.04)
    legend1.AddEntry(gr_train_ce, "Train", "LP")
    legend1.AddEntry(gr_valid_ce, "Valid", "LP")
    if best_epoch >= 0:
        legend1.AddEntry(line_ce, f"Best ({best_epoch})", "L")
    legend1.Draw()

    pad1.Update()

    # ===== DISCO TERM PLOT =====
    pad2 = canvas.cd(2)
    pad2.SetLeftMargin(0.12)
    pad2.SetRightMargin(0.05)
    pad2.SetTopMargin(0.08)
    pad2.SetBottomMargin(0.12)
    pad2.SetGrid()

    gr_train_disco = ROOT.TGraph(n_epochs)
    gr_valid_disco = ROOT.TGraph(n_epochs)

    if has_decomposed:
        for i, (epoch, train_disco, valid_disco) in enumerate(zip(epochs, history['train_disco_term'], history['valid_disco_term'])):
            gr_train_disco.SetPoint(i, epoch, train_disco)
            gr_valid_disco.SetPoint(i, epoch, valid_disco)
        disco_train_vals = history['train_disco_term']
        disco_valid_vals = history['valid_disco_term']
    else:
        # No DisCo data - show zeros
        for i, epoch in enumerate(epochs):
            gr_train_disco.SetPoint(i, epoch, 0)
            gr_valid_disco.SetPoint(i, epoch, 0)
        disco_train_vals = [0] * n_epochs
        disco_valid_vals = [0] * n_epochs

    gr_train_disco.SetLineColor(ROOT.kBlue)
    gr_train_disco.SetLineWidth(2)
    gr_train_disco.SetMarkerColor(ROOT.kBlue)
    gr_train_disco.SetMarkerStyle(20)
    gr_train_disco.SetMarkerSize(0.8)

    gr_valid_disco.SetLineColor(ROOT.kRed)
    gr_valid_disco.SetLineWidth(2)
    gr_valid_disco.SetMarkerColor(ROOT.kRed)
    gr_valid_disco.SetMarkerStyle(21)
    gr_valid_disco.SetMarkerSize(0.8)

    min_disco = min(min(disco_train_vals), min(disco_valid_vals))
    max_disco = max(max(disco_train_vals), max(disco_valid_vals))
    disco_range = max_disco - min_disco if max_disco > min_disco else 0.1
    y_min_disco = max(0, min_disco - 0.1 * disco_range)
    y_max_disco = max_disco + 0.1 * disco_range

    frame2 = pad2.DrawFrame(min(epochs), y_min_disco, max(epochs), y_max_disco)
    frame2.SetTitle("DisCo Term")
    frame2.GetXaxis().SetTitle("Epoch")
    frame2.GetYaxis().SetTitle("DisCo")
    frame2.GetXaxis().SetTitleSize(0.05)
    frame2.GetYaxis().SetTitleSize(0.05)

    gr_train_disco.Draw("LP SAME")
    gr_valid_disco.Draw("LP SAME")

    if best_epoch >= 0:
        line_disco = ROOT.TLine(best_epoch, y_min_disco, best_epoch, y_max_disco)
        line_disco.SetLineColor(ROOT.kGreen+2)
        line_disco.SetLineWidth(2)
        line_disco.SetLineStyle(2)
        line_disco.Draw()

    legend2 = ROOT.TLegend(0.55, 0.65, 0.90, 0.88)
    legend2.SetBorderSize(0)
    legend2.SetFillStyle(0)
    legend2.SetTextSize(0.04)
    legend2.AddEntry(gr_train_disco, "Train", "LP")
    legend2.AddEntry(gr_valid_disco, "Valid", "LP")
    if best_epoch >= 0:
        legend2.AddEntry(line_disco, f"Best ({best_epoch})", "L")
    legend2.Draw()

    pad2.Update()

    # ===== ACCURACY PLOT =====
    pad3 = canvas.cd(3)
    pad3.SetLeftMargin(0.12)
    pad3.SetRightMargin(0.05)
    pad3.SetTopMargin(0.08)
    pad3.SetBottomMargin(0.12)
    pad3.SetGrid()

    gr_train_acc = ROOT.TGraph(n_epochs)
    gr_valid_acc = ROOT.TGraph(n_epochs)

    for i, (epoch, train_acc, valid_acc) in enumerate(zip(epochs, history['train_acc'], history['valid_acc'])):
        gr_train_acc.SetPoint(i, epoch, train_acc)
        gr_valid_acc.SetPoint(i, epoch, valid_acc)

    gr_train_acc.SetLineColor(ROOT.kBlue)
    gr_train_acc.SetLineWidth(2)
    gr_train_acc.SetMarkerColor(ROOT.kBlue)
    gr_train_acc.SetMarkerStyle(20)
    gr_train_acc.SetMarkerSize(0.8)

    gr_valid_acc.SetLineColor(ROOT.kRed)
    gr_valid_acc.SetLineWidth(2)
    gr_valid_acc.SetMarkerColor(ROOT.kRed)
    gr_valid_acc.SetMarkerStyle(21)
    gr_valid_acc.SetMarkerSize(0.8)

    min_acc = min(min(history['train_acc']), min(history['valid_acc']))
    max_acc = max(max(history['train_acc']), max(history['valid_acc']))
    acc_range = max_acc - min_acc
    y_min_acc = max(0, min_acc - 0.1 * acc_range)
    y_max_acc = min(1.0, max_acc + 0.1 * acc_range)

    frame3 = pad3.DrawFrame(min(epochs), y_min_acc, max(epochs), y_max_acc)
    frame3.SetTitle("Accuracy")
    frame3.GetXaxis().SetTitle("Epoch")
    frame3.GetYaxis().SetTitle("Accuracy")
    frame3.GetXaxis().SetTitleSize(0.05)
    frame3.GetYaxis().SetTitleSize(0.05)

    gr_train_acc.Draw("LP SAME")
    gr_valid_acc.Draw("LP SAME")

    if best_epoch >= 0:
        line_acc = ROOT.TLine(best_epoch, y_min_acc, best_epoch, y_max_acc)
        line_acc.SetLineColor(ROOT.kGreen+2)
        line_acc.SetLineWidth(2)
        line_acc.SetLineStyle(2)
        line_acc.Draw()

    legend3 = ROOT.TLegend(0.55, 0.20, 0.90, 0.43)
    legend3.SetBorderSize(0)
    legend3.SetFillStyle(0)
    legend3.SetTextSize(0.04)
    legend3.AddEntry(gr_train_acc, "Train", "LP")
    legend3.AddEntry(gr_valid_acc, "Valid", "LP")
    if best_epoch >= 0:
        legend3.AddEntry(line_acc, f"Best ({best_epoch})", "L")
    legend3.Draw()

    pad3.Update()

    output_file = os.path.join(output_path, 'training_curves.png')
    canvas.SaveAs(output_file)
    canvas.Close()

    logging.info(f"Training curves saved to: {output_file}")


# =============================================================================
# Mass Decorrelation Visualization Functions (ParticleNetMD specific)
# =============================================================================

def plot_score_vs_mass_2d(mass1, y_scores, y_true, weights, class_names, output_path):
    """Plot 2D histogram of signal score vs di-muon mass for each true class."""
    setup_cms_style()

    signal_scores = y_scores[:, 0]
    valid_mask = mass1 > 0

    for class_idx, class_name in enumerate(class_names):
        class_mask = (y_true == class_idx) & valid_mask

        if np.sum(class_mask) < 10:
            logging.warning(f"Not enough events for {class_name} in score vs mass 2D plot")
            continue

        h2d = ROOT.TH2F(f"h2d_score_mass_{class_idx}",
                        f"{class_name}: Signal Score vs Mass;Signal Score;m_{{#mu#mu}} [GeV]",
                        50, 0, 1, 50, 0, 200)
        h2d.SetDirectory(0)

        for score, mass, w in zip(signal_scores[class_mask], mass1[class_mask], weights[class_mask]):
            h2d.Fill(score, mass, abs(w))

        canvas = ROOT.TCanvas(f"c_2d_{class_idx}", "", 800, 700)
        canvas.SetRightMargin(0.15)
        canvas.SetLeftMargin(0.12)
        canvas.SetBottomMargin(0.12)

        h2d.Draw("COLZ")

        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.04)
        latex.DrawLatex(0.15, 0.85, f"True: {class_name}")

        try:
            if HAS_CMS_STYLE:
                CMS.CMS_lumi(canvas, "", 0)
        except:
            pass

        output_file = os.path.join(output_path, f'score_vs_mass_{class_name.lower()}.png')
        canvas.SaveAs(output_file)
        canvas.Close()

        logging.info(f"Score vs mass 2D plot for {class_name} saved to: {output_file}")


def plot_mass_profile_vs_score(mass1, y_scores, weights, output_path):
    """Plot mean mass as function of signal score (profile plot)."""
    setup_cms_style()

    signal_scores = y_scores[:, 0]
    valid_mask = mass1 > 0

    h_profile = ROOT.TProfile("h_profile", "Mean Mass vs Signal Score;Signal Score;<m_{#mu#mu}> [GeV]",
                               20, 0, 1, 0, 200)
    h_profile.SetDirectory(0)

    for score, mass, w in zip(signal_scores[valid_mask], mass1[valid_mask], weights[valid_mask]):
        h_profile.Fill(score, mass, abs(w))

    mean_mass = np.average(mass1[valid_mask], weights=np.abs(weights[valid_mask]))

    canvas = ROOT.TCanvas("c_profile", "", 800, 600)
    canvas.SetLeftMargin(0.12)
    canvas.SetBottomMargin(0.12)
    canvas.SetGrid()

    h_profile.SetLineWidth(2)
    h_profile.SetLineColor(ROOT.kBlue)
    h_profile.SetMarkerStyle(20)
    h_profile.SetMarkerSize(1.2)
    h_profile.SetMarkerColor(ROOT.kBlue)
    h_profile.Draw("E")

    line = ROOT.TLine(0, mean_mass, 1, mean_mass)
    line.SetLineStyle(2)
    line.SetLineColor(ROOT.kRed)
    line.SetLineWidth(2)
    line.Draw()

    legend = ROOT.TLegend(0.55, 0.75, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.AddEntry(h_profile, "Mean mass vs score", "PE")
    legend.AddEntry(line, f"Overall mean: {mean_mass:.1f} GeV", "L")
    legend.Draw()

    try:
        if HAS_CMS_STYLE:
            CMS.CMS_lumi(canvas, "", 0)
    except:
        pass

    output_file = os.path.join(output_path, 'mass_profile_vs_score.png')
    canvas.SaveAs(output_file)
    canvas.Close()

    logging.info(f"Mass profile plot saved to: {output_file}")


def plot_mass_sculpting(mass1, y_scores, weights, output_path):
    """Compare mass distribution at high vs low signal score."""
    setup_cms_style()

    signal_scores = y_scores[:, 0]
    valid_mask = mass1 > 0

    high_score_mask = (signal_scores > 0.7) & valid_mask
    mid_score_mask = (signal_scores > 0.3) & (signal_scores <= 0.7) & valid_mask
    low_score_mask = (signal_scores <= 0.3) & valid_mask

    h_high = ROOT.TH1F("h_mass_high", "High score (>0.7);m_{#mu#mu} [GeV];Normalized Events", 40, 0, 200)
    h_mid = ROOT.TH1F("h_mass_mid", "Mid score (0.3-0.7);m_{#mu#mu} [GeV];Normalized Events", 40, 0, 200)
    h_low = ROOT.TH1F("h_mass_low", "Low score (<0.3);m_{#mu#mu} [GeV];Normalized Events", 40, 0, 200)

    for h in [h_high, h_mid, h_low]:
        h.SetDirectory(0)

    for m, w in zip(mass1[high_score_mask], weights[high_score_mask]):
        h_high.Fill(m, abs(w))
    for m, w in zip(mass1[mid_score_mask], weights[mid_score_mask]):
        h_mid.Fill(m, abs(w))
    for m, w in zip(mass1[low_score_mask], weights[low_score_mask]):
        h_low.Fill(m, abs(w))

    for h in [h_high, h_mid, h_low]:
        if h.Integral() > 0:
            h.Scale(1.0 / h.Integral())

    canvas = ROOT.TCanvas("c_sculpt", "", 800, 600)
    canvas.SetLeftMargin(0.12)
    canvas.SetBottomMargin(0.12)

    h_high.SetLineColor(ROOT.kRed)
    h_mid.SetLineColor(ROOT.kGreen+2)
    h_low.SetLineColor(ROOT.kBlue)
    for h in [h_high, h_mid, h_low]:
        h.SetLineWidth(2)

    max_y = max(h_high.GetMaximum(), h_mid.GetMaximum(), h_low.GetMaximum()) * 1.3
    h_high.SetMaximum(max_y)
    h_high.SetMinimum(0)

    h_high.Draw("HIST")
    h_mid.Draw("HIST SAME")
    h_low.Draw("HIST SAME")

    legend = ROOT.TLegend(0.55, 0.70, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.AddEntry(h_high, f"Score > 0.7 (N={np.sum(high_score_mask)})", "L")
    legend.AddEntry(h_mid, f"0.3 < Score #leq 0.7 (N={np.sum(mid_score_mask)})", "L")
    legend.AddEntry(h_low, f"Score #leq 0.3 (N={np.sum(low_score_mask)})", "L")
    legend.Draw()

    try:
        if HAS_CMS_STYLE:
            CMS.CMS_lumi(canvas, "", 0)
    except:
        pass

    output_file = os.path.join(output_path, 'mass_sculpting.png')
    canvas.SaveAs(output_file)
    canvas.Close()

    logging.info(f"Mass sculpting plot saved to: {output_file}")


# =============================================================================
# Main Function
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Visualize multi-class ParticleNetMD classification results')
    parser.add_argument('--signal', required=True,
                        help='Signal point (e.g., MHc130_MA90)')
    parser.add_argument('--channel', default='Combined',
                        help='Analysis channel (default: Combined)')
    parser.add_argument('--fold', type=int, default=None,
                        help='Cross-validation fold (default: auto-detect from most recent results)')
    parser.add_argument('--pilot', action='store_true',
                        help='Use pilot dataset results')
    parser.add_argument('--output',
                        help='Output directory (default: auto-generated)')
    parser.add_argument('--p-threshold', type=float, default=0.05,
                        help='p-value threshold for overfitting detection')

    args = parser.parse_args()

    try:
        # Find and load training results (do this first to get detected fold)
        logging.info(f"Loading results for multi-class {args.signal}")
        ga_json_file, root_file, result_dir, detected_fold = find_multiclass_results(
            args.signal, args.channel, args.fold, args.pilot)

        # Use detected fold for output directory if not specified
        fold_for_output = detected_fold if args.fold is None else args.fold

        # Create output directory
        if args.output is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            base_output = os.path.join(script_dir, "..", "plots", args.channel, "multiclass")
            base_output = os.path.abspath(base_output)
            if args.pilot:
                args.output = os.path.join(base_output, args.signal, "pilot")
            else:
                args.output = os.path.join(base_output, args.signal, f"fold-{fold_for_output}")

        os.makedirs(args.output, exist_ok=True)

        logging.info("="*60)
        logging.info("ParticleNetMD Multi-class Visualization")
        logging.info("="*60)
        logging.info(f"Signal: {args.signal}")
        logging.info(f"Channel: {args.channel}")
        logging.info(f"Fold: {fold_for_output}")
        logging.info(f"Pilot mode: {args.pilot}")
        logging.info(f"Output: {args.output}")
        logging.info("="*60)

        config_data = load_training_data(ga_json_file)
        hyperparams = config_data.get('hyperparameters', {})

        # Load prediction data with mass variables and b-jet flags
        logging.info("Loading prediction data from ROOT file...")
        (y_true_train, y_scores_train, weights_train, mass1_train, mass2_train, has_bjet_train,
         y_true_test, y_scores_test, weights_test, mass1_test, mass2_test, has_bjet_test,
         class_names) = load_multiclass_predictions_from_root(root_file)

        # =====================================================================
        # Standard Plots (matching visualizeGAIteration.py)
        # =====================================================================

        # Training curves
        logging.info("Generating training curves...")
        plot_training_curves(config_data, args.output)

        # KS tests for overfitting detection
        logging.info("Performing KS tests (16 tests)...")
        ks_results, histograms = perform_ks_tests_comprehensive(
            y_true_train, y_scores_train, weights_train,
            y_true_test, y_scores_test, weights_test,
            p_threshold=args.p_threshold
        )

        is_overfitted, min_p_value = save_ks_results(ks_results, histograms, args.output)
        logging.info(f"Overfitting status: {'OVERFITTED' if is_overfitted else 'OK'}")
        logging.info(f"Min p-value: {min_p_value:.4f}")
        logging.info(f"Failed tests: {sum(1 for r in ks_results.values() if r['is_overfitted'])}/16")

        # KS test heatmap
        logging.info("Generating KS test heatmap...")
        plot_ks_test_heatmap(ks_results, hyperparams, args.output)

        # Score distributions grid (train/test comparison)
        logging.info("Generating score distributions grid (train/test)...")
        plot_score_distributions_grid_train_test(
            y_true_train, y_scores_train, weights_train,
            y_true_test, y_scores_test, weights_test,
            args.output)

        # Score distributions grid (split by has_bjet)
        logging.info("Generating score distributions grid (bjet split)...")
        plot_score_distributions_grid_bjet(y_true_test, y_scores_test, weights_test, has_bjet_test, args.output)

        # ROC curves
        logging.info("Generating ROC curves...")
        plot_roc_curves(y_true_train, y_scores_train, weights_train,
                       y_true_test, y_scores_test, weights_test,
                       args.output)

        # Confusion matrix
        logging.info("Generating confusion matrix...")
        plot_confusion_matrices(y_true_train, y_scores_train, weights_train,
                               y_true_test, y_scores_test, weights_test,
                               args.output)

        # =====================================================================
        # Mass Decorrelation Plots (ParticleNetMD specific)
        # =====================================================================
        logging.info("Generating mass decorrelation plots...")

        logging.info("  - Score vs mass 2D histograms...")
        plot_score_vs_mass_2d(mass1_test, y_scores_test, y_true_test, weights_test,
                              class_names, args.output)

        logging.info("  - Mass profile vs score...")
        plot_mass_profile_vs_score(mass1_test, y_scores_test, weights_test, args.output)

        logging.info("  - Mass sculpting comparison...")
        plot_mass_sculpting(mass1_test, y_scores_test, weights_test, args.output)

        # =====================================================================
        # Summary
        # =====================================================================
        logging.info("="*60)
        logging.info("VISUALIZATION COMPLETE")
        logging.info("="*60)
        logging.info(f"All plots saved to: {args.output}")
        logging.info("")
        logging.info("Generated files:")
        logging.info("  - training_curves.png")
        logging.info("  - ks_test_heatmap.png")
        logging.info("  - score_distributions_grid_train_test.png")
        logging.info("  - score_distributions_grid_bjet.png")
        logging.info("  - roc_curves.png")
        logging.info("  - confusion_matrix.png")
        logging.info("  - kolmogorov.json, kolmogorov.root")
        logging.info("  - score_vs_mass_*.png (4 files)")
        logging.info("  - mass_profile_vs_score.png")
        logging.info("  - mass_sculpting.png")

    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise


if __name__ == "__main__":
    main()
