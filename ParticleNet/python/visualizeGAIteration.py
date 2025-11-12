#!/usr/bin/env python3
"""
Comprehensive visualization and overfitting analysis for GA iteration results.

This script evaluates all models from a GA iteration, performs KS tests on all
output scores for each class, and generates comprehensive visualizations.

For each model:
- Performs 16 KS tests (4 true classes × 4 output scores)
- Generates score distribution plots
- Creates ROC curves and confusion matrices
- Saves overfitting diagnostic results

Usage:
    python visualizeGAIteration.py --signal MHc130_MA90 --channel Run1E2Mu --iteration 3 --device cuda:0 --pilot
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import ROOT
ROOT.gROOT.SetBatch(True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from GAConfig import load_ga_config
from DynamicDatasetLoader import DynamicDatasetLoader
from MultiClassModels import create_multiclass_model
from Preprocess import GraphDataset
from ROCCurveCalculator import ROCCurveCalculator

# Try to import CMS style
try:
    import cmsstyle as CMS
    HAS_CMS_STYLE = True
except ImportError:
    print("Warning: cmsstyle not available, using default ROOT style")
    HAS_CMS_STYLE = False

# Color palette (consistent with plotter.py)
PALETTE = [
    ROOT.TColor.GetColor("#5790fc"),  # Blue - Signal
    ROOT.TColor.GetColor("#f89c20"),  # Orange - Nonprompt
    ROOT.TColor.GetColor("#e42536"),  # Red - Diboson
    ROOT.TColor.GetColor("#964a8b"),  # Purple - TTX
]

CLASS_NAMES = ['signal', 'nonprompt', 'diboson', 'ttX']
CLASS_DISPLAY_NAMES = ['Signal', 'Nonprompt', 'Diboson', 'TTX']
SCORE_NAMES = ['signal_score', 'nonprompt_score', 'diboson_score', 'ttX_score']


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


def load_model_config(config_path: str) -> Dict:
    """Load model configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return config_data


def load_model_checkpoint(model_path: str, config_path: str, device: str) -> torch.nn.Module:
    """Load model from checkpoint with proper architecture."""
    # Load hyperparameters
    config_data = load_model_config(config_path)
    hyperparams = config_data['hyperparameters']

    num_hidden = hyperparams['num_hidden']
    num_classes = hyperparams.get('num_classes', 4)
    num_node_features = hyperparams.get('num_node_features', 9)
    num_graph_features = hyperparams.get('num_graph_features', 4)
    dropout_p = hyperparams.get('dropout_p', 0.4)
    model_type = hyperparams.get('model_type', 'OptimizedParticleNet')

    # Create model
    model = create_multiclass_model(
        model_type=model_type,
        num_node_features=num_node_features,
        num_graph_features=num_graph_features,
        num_classes=num_classes,
        num_hidden=num_hidden,
        dropout_p=dropout_p
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_state = checkpoint['model_state_dict'] if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(model_state)
    model.eval()

    return model, hyperparams


def evaluate_model(model: torch.nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate model on dataset.

    Returns:
        y_true: True labels (n_samples,)
        y_scores: Predicted scores (n_samples, n_classes)
        weights: Event weights (n_samples,)
    """
    model.eval()
    y_true_list, y_scores_list, weights_list = [], [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.graphInput, data.batch)
            scores = F.softmax(out, dim=1)

            y_true_list.append(data.y)
            y_scores_list.append(scores)
            weights_list.append(data.weight)

    # Concatenate on GPU, then move to CPU
    y_true = torch.cat(y_true_list).cpu().numpy()
    y_scores = torch.cat(y_scores_list).cpu().numpy()
    weights = torch.cat(weights_list).cpu().numpy()

    return y_true, y_scores, weights


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
    """
    Identify bins with negative content or very small absolute content.

    Args:
        hist: ROOT histogram to analyze
        threshold: Minimum absolute content threshold

    Returns:
        List of bin indices (1-indexed) that are problematic
    """
    problematic_bins = []
    nbins = hist.GetNbinsX()

    for i in range(1, nbins + 1):  # ROOT bins are 1-indexed
        content = hist.GetBinContent(i)
        if content < 0 or abs(content) < threshold:
            problematic_bins.append(i)

    return problematic_bins


def create_merge_groups(hist_train: ROOT.TH1D, hist_test: ROOT.TH1D,
                       threshold: float = 1e-6) -> List[Tuple[int, int]]:
    """
    Create groups of consecutive bins to merge based on problematic bins in both histograms.

    Algorithm:
    1. Identify problematic bins in both train and test histograms
    2. Create union of problematic bin indices
    3. Find consecutive problematic bins
    4. Extend merge groups until cumulative content meets criteria:
       - sum(abs(content)) >= threshold
       - sum(content) >= 0 (net positive)

    Args:
        hist_train: Training histogram
        hist_test: Test histogram
        threshold: Minimum content threshold

    Returns:
        List of (start_bin, end_bin) tuples for bins to merge
    """
    # Get problematic bins from both histograms
    prob_train = set(identify_problematic_bins(hist_train, threshold))
    prob_test = set(identify_problematic_bins(hist_test, threshold))
    problematic_bins = sorted(prob_train | prob_test)  # Union

    if not problematic_bins:
        return []

    nbins = hist_train.GetNbinsX()
    merge_groups = []
    i = 0

    while i < len(problematic_bins):
        group_start = problematic_bins[i]
        group_end = group_start

        # Calculate cumulative content from both histograms
        cum_content_train = hist_train.GetBinContent(group_start)
        cum_abs_train = abs(cum_content_train)
        cum_content_test = hist_test.GetBinContent(group_start)
        cum_abs_test = abs(cum_content_test)

        # Extend group while consecutive and criteria not met
        while (cum_abs_train < threshold or cum_content_train < 0 or
               cum_abs_test < threshold or cum_content_test < 0):

            # Try to extend to next bin
            next_bin = group_end + 1
            if next_bin > nbins:
                break  # Reached end of histogram

            # Add next bin to group
            group_end = next_bin
            cum_content_train += hist_train.GetBinContent(next_bin)
            cum_abs_train += abs(hist_train.GetBinContent(next_bin))
            cum_content_test += hist_test.GetBinContent(next_bin)
            cum_abs_test += abs(hist_test.GetBinContent(next_bin))

            # Check if next_bin was in problematic list - if so, skip it in outer loop
            if i + 1 < len(problematic_bins) and problematic_bins[i + 1] == next_bin:
                i += 1

        # Only create merge group if we're actually merging multiple bins
        if group_end > group_start:
            merge_groups.append((group_start, group_end))

        i += 1

    return merge_groups


def apply_bin_merging(hist_train: ROOT.TH1D, hist_test: ROOT.TH1D,
                     merge_groups: List[Tuple[int, int]]) -> Tuple[ROOT.TH1D, ROOT.TH1D]:
    """
    Apply bin merging to both histograms using the same merge groups.

    Creates new histograms with merged bins, maintaining synchronization between
    train and test histograms.

    Args:
        hist_train: Training histogram
        hist_test: Test histogram
        merge_groups: List of (start_bin, end_bin) tuples

    Returns:
        Tuple of (merged_train_hist, merged_test_hist)
    """
    if not merge_groups:
        # No merging needed - return clones
        h_train_new = hist_train.Clone(f"{hist_train.GetName()}_merged")
        h_test_new = hist_test.Clone(f"{hist_test.GetName()}_merged")
        h_train_new.SetDirectory(0)
        h_test_new.SetDirectory(0)
        return h_train_new, h_test_new

    nbins_old = hist_train.GetNbinsX()
    xmin = hist_train.GetXaxis().GetXmin()
    xmax = hist_train.GetXaxis().GetXmax()

    # Create mapping: old_bin -> new_bin
    # and track which bins are merged
    merged_bins = set()
    for start, end in merge_groups:
        for b in range(start, end + 1):
            merged_bins.add(b)

    # Build new bin edges
    bin_edges = [xmin]
    current_edge = xmin
    bin_width = (xmax - xmin) / nbins_old

    old_bin = 1
    while old_bin <= nbins_old:
        # Check if this bin is start of a merge group
        in_merge_group = False
        for start, end in merge_groups:
            if old_bin == start:
                # This is start of merge group - skip to end
                current_edge = xmin + end * bin_width
                bin_edges.append(current_edge)
                old_bin = end + 1
                in_merge_group = True
                break

        if not in_merge_group:
            # Regular bin (not being merged)
            if old_bin not in merged_bins:
                current_edge = xmin + old_bin * bin_width
                bin_edges.append(current_edge)
            old_bin += 1

    # Create new histograms with variable binning
    nbins_new = len(bin_edges) - 1
    bin_edges_array = np.array(bin_edges, dtype=np.float64)

    h_train_new = ROOT.TH1D(
        f"{hist_train.GetName()}_merged",
        hist_train.GetTitle(),
        nbins_new,
        bin_edges_array
    )
    h_test_new = ROOT.TH1D(
        f"{hist_test.GetName()}_merged",
        hist_test.GetTitle(),
        nbins_new,
        bin_edges_array
    )
    h_train_new.SetDirectory(0)
    h_test_new.SetDirectory(0)

    # Fill new histograms
    new_bin = 1
    old_bin = 1

    while old_bin <= nbins_old:
        # Check if this bin is start of a merge group
        is_merged = False
        for start, end in merge_groups:
            if old_bin == start:
                # Merge bins from start to end
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
            # Regular bin - copy as-is
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
    """
    Iteratively merge bins in both histograms until no problematic bins remain.

    Applies adaptive bin merging to handle negative weights and low-statistics bins.
    The same merging scheme is applied to both train and test histograms to maintain
    synchronized binning for KS tests.

    Args:
        hist_train: Training histogram
        hist_test: Test histogram
        threshold: Minimum bin content threshold (default: 1e-6)
        max_iterations: Maximum number of merge iterations (default: 100)

    Returns:
        Tuple of (merged_train_hist, merged_test_hist, n_iterations)
    """
    h_train = hist_train.Clone(f"{hist_train.GetName()}_itermerge")
    h_test = hist_test.Clone(f"{hist_test.GetName()}_itermerge")
    h_train.SetDirectory(0)
    h_test.SetDirectory(0)

    iteration = 0

    while iteration < max_iterations:
        # Check if any problematic bins remain
        prob_train = identify_problematic_bins(h_train, threshold)
        prob_test = identify_problematic_bins(h_test, threshold)

        if not prob_train and not prob_test:
            break  # Stable - no more problematic bins

        # Create merge groups
        merge_groups = create_merge_groups(h_train, h_test, threshold)

        if not merge_groups:
            break  # No merging possible

        # Apply merging
        h_train, h_test = apply_bin_merging(h_train, h_test, merge_groups)
        iteration += 1

    # Warn if max iterations reached
    if iteration >= max_iterations:
        print(f"Warning: Bin merging reached max iterations ({max_iterations}) for {hist_train.GetName()}")

    return h_train, h_test, iteration


def perform_ks_tests_comprehensive(y_true_train: np.ndarray, y_scores_train: np.ndarray, weights_train: np.ndarray,
                                   y_true_test: np.ndarray, y_scores_test: np.ndarray, weights_test: np.ndarray,
                                   p_threshold: float = 0.05,
                                   bin_merge_threshold: float = 1e-6,
                                   bin_merge_max_iterations: int = 100) -> Tuple[Dict, Dict]:
    """
    Perform comprehensive KS tests: all 4 output scores for each of 4 true classes.

    Args:
        y_true_train: True labels for training set
        y_scores_train: Predicted scores for training set
        weights_train: Event weights for training set
        y_true_test: True labels for test set
        y_scores_test: Predicted scores for test set
        weights_test: Event weights for test set
        p_threshold: p-value threshold for overfitting detection
        bin_merge_threshold: Minimum bin content threshold for adaptive bin merging
        bin_merge_max_iterations: Maximum iterations for iterative bin merging

    Returns:
        ks_results: Dict with p-values for all 16 tests
        histograms: Dict with all train/test histogram pairs
    """
    num_classes = 4
    ks_results = {}
    histograms = {}

    for true_class in range(num_classes):
        # Select events of this true class
        train_mask = (y_true_train == true_class)
        test_mask = (y_true_test == true_class)

        train_scores_class = y_scores_train[train_mask]
        test_scores_class = y_scores_test[test_mask]
        train_weights_class = weights_train[train_mask]
        test_weights_class = weights_test[test_mask]

        for score_idx in range(num_classes):
            # Extract scores for this output
            train_scores = train_scores_class[:, score_idx]
            test_scores = test_scores_class[:, score_idx]

            # Create histograms
            key = f"{CLASS_NAMES[true_class]}_{SCORE_NAMES[score_idx]}"

            h_train = create_weighted_histogram(
                train_scores, train_weights_class,
                f"h_train_{key}", f"Train: {CLASS_DISPLAY_NAMES[true_class]} - {SCORE_NAMES[score_idx]}"
            )
            h_test = create_weighted_histogram(
                test_scores, test_weights_class,
                f"h_test_{key}", f"Test: {CLASS_DISPLAY_NAMES[true_class]} - {SCORE_NAMES[score_idx]}"
            )

            # Apply adaptive bin merging to handle negative weights and low-statistics bins
            h_train_merged, h_test_merged, n_merge_iterations = merge_histograms_iteratively(
                h_train, h_test, threshold=bin_merge_threshold, max_iterations=bin_merge_max_iterations
            )

            # Perform KS test on merged histograms
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


def save_ks_results(ks_results: Dict, histograms: Dict, output_dir: str, model_idx: int):
    """Save KS test results to JSON and ROOT files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON results
    json_path = os.path.join(output_dir, f"model{model_idx}_kolmogorov.json")
    with open(json_path, 'w') as f:
        json.dump(ks_results, f, indent=2)

    # Save ROOT file with histograms
    root_path = os.path.join(output_dir, f"model{model_idx}_kolmogorov.root")
    root_file = ROOT.TFile(root_path, "RECREATE")

    for key, hists in histograms.items():
        # Write with explicit clean names (histograms may have _merged or _itermerge suffixes)
        hists['train'].Write(f"h_train_{key}")
        hists['test'].Write(f"h_test_{key}")

    root_file.Close()

    # Save overall result
    any_overfitted = any(result['is_overfitted'] for result in ks_results.values())
    min_p_value = min(result['p_value'] for result in ks_results.values())

    result_path = os.path.join(output_dir, f"model{model_idx}_result.json")
    with open(result_path, 'w') as f:
        json.dump({
            'model_idx': model_idx,
            'is_overfitted': any_overfitted,
            'min_p_value': min_p_value,
            'n_failed_tests': sum(1 for r in ks_results.values() if r['is_overfitted']),
            'total_tests': len(ks_results)
        }, f, indent=2)

    return any_overfitted, min_p_value


def plot_ks_test_heatmap(ks_results: Dict, output_path: str, model_idx: int, hyperparams: Dict):
    """Plot heatmap of p-values for all 16 KS tests."""
    # Create 4x4 matrix of p-values
    p_matrix = np.zeros((4, 4))

    for true_class_idx in range(4):
        for score_idx in range(4):
            key = f"{CLASS_NAMES[true_class_idx]}_{SCORE_NAMES[score_idx]}"
            p_matrix[true_class_idx, score_idx] = ks_results[key]['p_value']

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap
    im = ax.imshow(p_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    # Set ticks and labels
    ax.set_xticks(np.arange(4))
    ax.set_yticks(np.arange(4))
    ax.set_xticklabels([s.replace('_score', '') for s in SCORE_NAMES])
    ax.set_yticklabels([f"True {name}" for name in CLASS_DISPLAY_NAMES])

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    for i in range(4):
        for j in range(4):
            p_val = p_matrix[i, j]
            color = 'white' if p_val < 0.3 else 'black'
            text = ax.text(j, i, f'{p_val:.3f}', ha="center", va="center", color=color, fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('KS Test p-value', rotation=270, labelpad=20)

    # Title with hyperparameters
    title = f"Model {model_idx} - KS Test p-values\n"
    title += f"({hyperparams['num_hidden']} nodes, {hyperparams['optimizer']}, "
    title += f"lr={hyperparams['initial_lr']}, wd={hyperparams['weight_decay']})"
    ax.set_title(title, fontsize=12, pad=20)

    ax.set_xlabel('Output Score Type', fontsize=11)
    ax.set_ylabel('True Class', fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_score_distributions_grid(histograms: Dict, output_path: str, model_idx: int):
    """Plot 4x4 grid of score distributions (train vs test overlay)."""
    setup_cms_style()

    # Create canvas
    canvas = ROOT.TCanvas(f"c_grid_model{model_idx}", "Score Distributions", 2400, 2400)
    canvas.Divide(4, 4, 0.001, 0.001)

    pad_idx = 1
    for true_class_idx in range(4):
        for score_idx in range(4):
            canvas.cd(pad_idx)
            ROOT.gStyle.SetOptTitle(1)  # Enable title display
            ROOT.gPad.SetLeftMargin(0.15)
            ROOT.gPad.SetRightMargin(0.05)
            ROOT.gPad.SetTopMargin(0.10)
            ROOT.gPad.SetBottomMargin(0.12)

            key = f"{CLASS_NAMES[true_class_idx]}_{SCORE_NAMES[score_idx]}"
            h_train = histograms[key]['train']
            h_test = histograms[key]['test']

            # Style histograms
            h_train.SetLineColor(ROOT.kBlue)
            h_train.SetLineWidth(2)
            h_train.SetLineStyle(1)
            h_train.SetTitle(f"True: {CLASS_DISPLAY_NAMES[true_class_idx]}")

            h_test.SetLineColor(ROOT.kRed)
            h_test.SetLineWidth(2)
            h_test.SetLineStyle(2)

            # Get max for y-axis
            max_val = max(h_train.GetMaximum(), h_test.GetMaximum())
            h_train.SetMaximum(max_val * 1.3)

            # Labels
            h_train.SetXTitle(SCORE_NAMES[score_idx].replace('_', ' '))
            h_train.SetYTitle("Normalized")
            h_train.GetXaxis().SetTitleSize(0.05)
            h_train.GetYaxis().SetTitleSize(0.05)
            h_train.GetXaxis().SetLabelSize(0.04)
            h_train.GetYaxis().SetLabelSize(0.04)

            # Draw
            h_train.Draw("HIST")
            h_test.Draw("HIST SAME")

            # Add legend for Train/Test
            legend = ROOT.TLegend(0.6, 0.75, 0.9, 0.88)
            legend.SetTextSize(0.04)
            legend.SetBorderSize(0)
            legend.SetFillStyle(0)
            legend.AddEntry(h_train, "Train", "L")
            legend.AddEntry(h_test, "Test", "L")
            legend.Draw("SAME")

            pad_idx += 1

    canvas.SaveAs(output_path)
    canvas.Close()


def plot_roc_curves(y_true_train: np.ndarray, y_scores_train: np.ndarray, weights_train: np.ndarray,
                   y_true_test: np.ndarray, y_scores_test: np.ndarray, weights_test: np.ndarray,
                   output_path: str, model_idx: int):
    """
    Plot ROC curves for signal vs each background class using ROOT.

    This function uses the ROCCurveCalculator class which properly handles
    negative weights from NLO MC generators. Plots both train and test ROC curves
    with their respective AUC values.

    Args:
        y_true_train: True class labels for training set
        y_scores_train: Predicted class probabilities for training set
        weights_train: Event weights for training set (can be negative)
        y_true_test: True class labels for test set
        y_scores_test: Predicted class probabilities for test set
        weights_test: Event weights for test set (can be negative)
        output_path: Path to save the plot
        model_idx: Model index for title
    """
    calculator = ROCCurveCalculator()
    calculator.plot_multiclass_rocs(
        y_true_train, y_scores_train, weights_train,
        y_true_test, y_scores_test, weights_test,
        output_path, model_idx, CLASS_DISPLAY_NAMES
    )


def plot_confusion_matrices(y_true_train: np.ndarray, y_scores_train: np.ndarray, weights_train: np.ndarray,
                           y_true_test: np.ndarray, y_scores_test: np.ndarray, weights_test: np.ndarray,
                           output_path: str, model_idx: int):
    """Plot confusion matrix with train(test) format."""
    # Get predictions
    y_pred_train = np.argmax(y_scores_train, axis=1)
    y_pred_test = np.argmax(y_scores_test, axis=1)

    # Compute confusion matrices (weighted)
    cm_train = np.zeros((4, 4))
    cm_test = np.zeros((4, 4))

    for true_label in range(4):
        # Train
        mask_train = (y_true_train == true_label)
        if mask_train.sum() > 0:
            for pred_label in range(4):
                pred_mask = (y_pred_train == pred_label)
                cm_train[true_label, pred_label] = weights_train[mask_train & pred_mask].sum()

        # Test
        mask_test = (y_true_test == true_label)
        if mask_test.sum() > 0:
            for pred_label in range(4):
                pred_mask = (y_pred_test == pred_label)
                cm_test[true_label, pred_label] = weights_test[mask_test & pred_mask].sum()

    # Normalize by row (true class)
    cm_train_normalized = cm_train / (cm_train.sum(axis=1, keepdims=True) + 1e-10)
    cm_test_normalized = cm_test / (cm_test.sum(axis=1, keepdims=True) + 1e-10)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    # Use train values for color mapping
    im = ax.imshow(cm_train_normalized, cmap='Blues', vmin=0, vmax=1)

    # Ticks and labels
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

            # Color based on train value
            color = 'white' if val_train > 0.5 else 'black'

            # Format: train(test)
            text_str = f'{val_train:.3f}\n({val_test:.3f})'

            ax.text(j, i, text_str, ha="center", va="center",
                   color=color, fontsize=11, fontweight='bold')

    ax.set_xlabel('Predicted Class', fontsize=13)
    ax.set_ylabel('True Class', fontsize=13)
    ax.set_title(f'Model {model_idx} - Confusion Matrix [Train (Test)]', fontsize=14, pad=15)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fraction (Train)', rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(config_data: Dict, output_path: str, model_idx: int):
    """Plot training curves from JSON history using ROOT with cmsstyle."""
    if 'epoch_history' not in config_data:
        print(f"Warning: No epoch history found for model {model_idx}")
        return

    history = config_data['epoch_history']
    epochs = history.get('epoch', [])

    if not epochs:
        return

    # Get best epoch from training summary
    best_epoch = config_data.get('training_summary', {}).get('best_epoch', -1)

    # Setup CMS style
    setup_cms_style()

    # Create canvas with two pads
    canvas = ROOT.TCanvas(f"c_training_model{model_idx}", "Training Curves", 1400, 600)
    canvas.Divide(2, 1, 0.01, 0.01)

    # ===== LOSS PLOT =====
    pad1 = canvas.cd(1)
    pad1.SetLeftMargin(0.12)
    pad1.SetRightMargin(0.05)
    pad1.SetTopMargin(0.08)
    pad1.SetBottomMargin(0.12)
    pad1.SetGrid()

    # Create TGraphs for loss
    n_epochs = len(epochs)
    gr_train_loss = ROOT.TGraph(n_epochs)
    gr_valid_loss = ROOT.TGraph(n_epochs)

    for i, (epoch, train_loss, valid_loss) in enumerate(zip(epochs, history['train_loss'], history['valid_loss'])):
        gr_train_loss.SetPoint(i, epoch, train_loss)
        gr_valid_loss.SetPoint(i, epoch, valid_loss)

    # Style for loss graphs
    gr_train_loss.SetLineColor(ROOT.kBlue)
    gr_train_loss.SetLineWidth(2)
    gr_train_loss.SetMarkerColor(ROOT.kBlue)
    gr_train_loss.SetMarkerStyle(20)
    gr_train_loss.SetMarkerSize(0.8)

    gr_valid_loss.SetLineColor(ROOT.kRed)
    gr_valid_loss.SetLineWidth(2)
    gr_valid_loss.SetMarkerColor(ROOT.kRed)
    gr_valid_loss.SetMarkerStyle(21)
    gr_valid_loss.SetMarkerSize(0.8)

    # Find y-axis range for loss
    min_loss = min(min(history['train_loss']), min(history['valid_loss']))
    max_loss = max(max(history['train_loss']), max(history['valid_loss']))
    loss_range = max_loss - min_loss
    y_min_loss = max(0, min_loss - 0.1 * loss_range)
    y_max_loss = max_loss + 0.1 * loss_range

    # Create frame for loss plot
    frame1 = pad1.DrawFrame(min(epochs), y_min_loss, max(epochs), y_max_loss)
    frame1.SetTitle("Training and Validation Loss")
    frame1.GetXaxis().SetTitle("Epoch")
    frame1.GetYaxis().SetTitle("Loss")
    frame1.GetXaxis().SetTitleSize(0.05)
    frame1.GetYaxis().SetTitleSize(0.05)
    frame1.GetXaxis().SetLabelSize(0.04)
    frame1.GetYaxis().SetLabelSize(0.04)

    # Draw loss graphs
    gr_train_loss.Draw("LP SAME")
    gr_valid_loss.Draw("LP SAME")

    # Add vertical line at best epoch
    if best_epoch >= 0:
        line_loss = ROOT.TLine(best_epoch, y_min_loss, best_epoch, y_max_loss)
        line_loss.SetLineColor(ROOT.kGreen+2)
        line_loss.SetLineWidth(2)
        line_loss.SetLineStyle(2)
        line_loss.Draw()

    # Legend for loss plot
    legend1 = ROOT.TLegend(0.55, 0.65, 0.90, 0.88)
    legend1.SetBorderSize(0)
    legend1.SetFillStyle(0)
    legend1.SetTextSize(0.04)
    legend1.AddEntry(gr_train_loss, "Train Loss", "LP")
    legend1.AddEntry(gr_valid_loss, "Valid Loss", "LP")
    if best_epoch >= 0:
        legend1.AddEntry(line_loss, f"Best Epoch ({best_epoch})", "L")
    legend1.Draw()

    pad1.Update()

    # ===== ACCURACY PLOT =====
    pad2 = canvas.cd(2)
    pad2.SetLeftMargin(0.12)
    pad2.SetRightMargin(0.05)
    pad2.SetTopMargin(0.08)
    pad2.SetBottomMargin(0.12)
    pad2.SetGrid()

    # Create TGraphs for accuracy
    gr_train_acc = ROOT.TGraph(n_epochs)
    gr_valid_acc = ROOT.TGraph(n_epochs)

    for i, (epoch, train_acc, valid_acc) in enumerate(zip(epochs, history['train_acc'], history['valid_acc'])):
        gr_train_acc.SetPoint(i, epoch, train_acc)
        gr_valid_acc.SetPoint(i, epoch, valid_acc)

    # Style for accuracy graphs
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

    # Find y-axis range for accuracy
    min_acc = min(min(history['train_acc']), min(history['valid_acc']))
    max_acc = max(max(history['train_acc']), max(history['valid_acc']))
    acc_range = max_acc - min_acc
    y_min_acc = max(0, min_acc - 0.1 * acc_range)
    y_max_acc = min(1.0, max_acc + 0.1 * acc_range)

    # Create frame for accuracy plot
    frame2 = pad2.DrawFrame(min(epochs), y_min_acc, max(epochs), y_max_acc)
    frame2.SetTitle("Training and Validation Accuracy")
    frame2.GetXaxis().SetTitle("Epoch")
    frame2.GetYaxis().SetTitle("Accuracy")
    frame2.GetXaxis().SetTitleSize(0.05)
    frame2.GetYaxis().SetTitleSize(0.05)
    frame2.GetXaxis().SetLabelSize(0.04)
    frame2.GetYaxis().SetLabelSize(0.04)

    # Draw accuracy graphs
    gr_train_acc.Draw("LP SAME")
    gr_valid_acc.Draw("LP SAME")

    # Add vertical line at best epoch
    if best_epoch >= 0:
        line_acc = ROOT.TLine(best_epoch, y_min_acc, best_epoch, y_max_acc)
        line_acc.SetLineColor(ROOT.kGreen+2)
        line_acc.SetLineWidth(2)
        line_acc.SetLineStyle(2)
        line_acc.Draw()

    # Legend for accuracy plot
    legend2 = ROOT.TLegend(0.55, 0.20, 0.90, 0.43)
    legend2.SetBorderSize(0)
    legend2.SetFillStyle(0)
    legend2.SetTextSize(0.04)
    legend2.AddEntry(gr_train_acc, "Train Accuracy", "LP")
    legend2.AddEntry(gr_valid_acc, "Valid Accuracy", "LP")
    if best_epoch >= 0:
        legend2.AddEntry(line_acc, f"Best Epoch ({best_epoch})", "L")
    legend2.Draw()

    pad2.Update()

    # Add overall title
    canvas.cd()
    title_text = ROOT.TLatex()
    title_text.SetNDC()
    title_text.SetTextSize(0.035)
    title_text.SetTextAlign(22)
    title_text.DrawLatex(0.5, 0.97, f"Model {model_idx} - Training History")

    # Save canvas
    canvas.SaveAs(output_path)
    canvas.Close()


def load_datasets(config, signal: str, channel: str, pilot: bool, device: str):
    """Load train and test datasets."""
    print("Loading datasets...")

    # Get configuration
    dataset_config = config.get_dataset_config()
    signal_full = f"TTToHcToWAToMuMu-{signal}"
    background_groups = config.get_background_groups()
    train_folds = config.get_training_parameters()['train_folds']  # [0, 1, 2]
    test_folds = config.get_overfitting_config()['test_folds']  # [4]
    batch_size = config.get_training_parameters()['batch_size']
    use_bjets = dataset_config['use_bjets']
    background_prefix = dataset_config.get('background_prefix', 'Skim_TriLep_')

    # Add prefix to background sample names (like trainMultiClassForGA.py does)
    background_groups_full = {
        group_name: [background_prefix + sample for sample in samples]
        for group_name, samples in background_groups.items()
    }

    print(f"  Signal: {signal_full}")
    print(f"  Channel: {channel}")
    print(f"  Train folds: {train_folds}")
    print(f"  Test folds: {test_folds}")
    print(f"  Pilot mode: {pilot}")
    print(f"  Use bjets: {use_bjets}")

    # Setup dataset root
    WORKDIR = os.getenv('WORKDIR', os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    dataset_root = f"{WORKDIR}/ParticleNet/dataset"

    loader = DynamicDatasetLoader(dataset_root=dataset_root, separate_bjets=use_bjets)

    # Get max events per fold parameter and balance_weights flag from config
    max_events_per_fold = config.get_training_parameters().get('max_events_per_fold_per_class', None)
    balance_weights = config.get_training_parameters().get('balance_weights', True)

    if max_events_per_fold:
        print(f"  Train event subsampling: max {max_events_per_fold} events per fold per class")
    print(f"  Test event subsampling: None (using full test set)")
    print(f"  Weight balancing: {balance_weights}")

    # Load datasets using the same method as training
    train_data = loader.load_multiclass_with_subsampling(
        signal_sample=signal_full,
        background_groups=background_groups_full,
        channel=channel,
        fold_list=train_folds,
        pilot=pilot,
        max_events_per_fold=max_events_per_fold,
        balance_weights=balance_weights,
        random_state=42
    )
    # Load test data WITHOUT subsampling for proper evaluation
    test_data = loader.load_multiclass_with_subsampling(
        signal_sample=signal_full,
        background_groups=background_groups_full,
        channel=channel,
        fold_list=test_folds,
        pilot=pilot,
        max_events_per_fold=None,  # Use full test set for proper evaluation
        balance_weights=balance_weights,
        random_state=42
    )

    print(f"  Train samples: {len(train_data)}")
    print(f"  Test samples: {len(test_data)}")

    # Create dataloaders
    train_dataset = GraphDataset(train_data)
    test_dataset = GraphDataset(test_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def check_existing_results(model_idx: int, output_dirs: Dict) -> bool:
    """Check if evaluation results already exist for this model."""
    result_file = os.path.join(output_dirs['overfitting'], f"model{model_idx}_result.json")
    kolmogorov_json = os.path.join(output_dirs['overfitting'], f"model{model_idx}_kolmogorov.json")
    kolmogorov_root = os.path.join(output_dirs['overfitting'], f"model{model_idx}_kolmogorov.root")

    return all(os.path.exists(f) for f in [result_file, kolmogorov_json, kolmogorov_root])


def load_histograms_from_file(model_idx: int, output_dirs: Dict) -> Dict:
    """Load histograms from ROOT file for plot regeneration."""
    root_file_path = os.path.join(output_dirs['overfitting'], f"model{model_idx}_kolmogorov.root")

    if not os.path.exists(root_file_path):
        raise FileNotFoundError(f"ROOT file not found: {root_file_path}")

    root_file = ROOT.TFile.Open(root_file_path, "READ")
    histograms = {}

    # Load all 16 histograms (4 true classes × 4 scores)
    for true_class_idx in range(4):
        for score_idx in range(4):
            key = f"{CLASS_NAMES[true_class_idx]}_{SCORE_NAMES[score_idx]}"

            h_train_name = f"h_train_{key}"
            h_test_name = f"h_test_{key}"

            h_train = root_file.Get(h_train_name)
            h_test = root_file.Get(h_test_name)

            if h_train and h_test:
                # Clone histograms and detach from file
                h_train_clone = h_train.Clone(f"{h_train_name}_clone")
                h_test_clone = h_test.Clone(f"{h_test_name}_clone")
                h_train_clone.SetDirectory(0)
                h_test_clone.SetDirectory(0)

                histograms[key] = {
                    'train': h_train_clone,
                    'test': h_test_clone
                }

    root_file.Close()
    return histograms


def load_existing_results(model_idx: int, config_path: str, output_dirs: Dict) -> Dict:
    """Load existing evaluation results without re-evaluating."""
    print(f"  Loading existing results from disk...")

    # Load result summary
    result_file = os.path.join(output_dirs['overfitting'], f"model{model_idx}_result.json")
    with open(result_file, 'r') as f:
        result_data = json.load(f)

    # Load KS test results
    ks_json = os.path.join(output_dirs['overfitting'], f"model{model_idx}_kolmogorov.json")
    with open(ks_json, 'r') as f:
        ks_results = json.load(f)

    # Load hyperparameters
    config_data = load_model_config(config_path)
    hyperparams = config_data['hyperparameters']

    return {
        'model_idx': model_idx,
        'hyperparams': hyperparams,
        'is_overfitted': result_data['is_overfitted'],
        'min_p_value': result_data['min_p_value'],
        'ks_results': ks_results
    }


def process_model(model_idx: int, model_path: str, config_path: str,
                 train_loader: DataLoader, test_loader: DataLoader,
                 device: str, output_dirs: Dict, p_threshold: float,
                 bin_merge_threshold: float = 1e-6,
                 bin_merge_max_iterations: int = 100,
                 skip_eval: bool = False):
    """Process a single model: evaluate (or load existing), perform KS tests, generate plots."""
    print(f"\n{'='*60}")
    print(f"Processing Model {model_idx}")
    print(f"{'='*60}")

    # Check if we can skip evaluation
    if skip_eval and check_existing_results(model_idx, output_dirs):
        # Load existing results
        result = load_existing_results(model_idx, config_path, output_dirs)
        print(f"  Overfitting status: {'OVERFITTED' if result['is_overfitted'] else 'OK'}")
        print(f"  Min p-value: {result['min_p_value']:.4f}")
        failed_tests = sum(1 for r in result['ks_results'].values() if r['is_overfitted'])
        print(f"  Failed tests: {failed_tests}/16")

        # Load histograms from ROOT file for plot regeneration
        print("  Loading histograms from ROOT file...")
        histograms = load_histograms_from_file(model_idx, output_dirs)

        # Regenerate plots with loaded histograms
        plots_dir = output_dirs['plots']
        hyperparams = result['hyperparams']
        ks_results = result['ks_results']

        print("  Generating KS test heatmap...")
        plot_ks_test_heatmap(
            ks_results,
            os.path.join(plots_dir, f"model{model_idx}_ks_test_heatmap.png"),
            model_idx, hyperparams
        )

        print("  Generating score distribution grid...")
        plot_score_distributions_grid(
            histograms,
            os.path.join(plots_dir, f"model{model_idx}_score_distributions_grid.png"),
            model_idx
        )

        print("  Generating training curves...")
        config_data = load_model_config(config_path)
        plot_training_curves(
            config_data,
            os.path.join(plots_dir, f"model{model_idx}_training_curves.png"),
            model_idx
        )

        # Load predictions for ROC curves and confusion matrices
        predictions_path = os.path.join(output_dirs['overfitting'], f"model{model_idx}_predictions.npz")
        if os.path.exists(predictions_path):
            print("  Loading predictions from file...")
            predictions = np.load(predictions_path)
            y_true_train = predictions['y_true_train']
            y_scores_train = predictions['y_scores_train']
            weights_train = predictions['weights_train']
            y_true_test = predictions['y_true_test']
            y_scores_test = predictions['y_scores_test']
            weights_test = predictions['weights_test']

            print("  Generating ROC curves...")
            plot_roc_curves(
                y_true_train, y_scores_train, weights_train,
                y_true_test, y_scores_test, weights_test,
                os.path.join(plots_dir, f"model{model_idx}_roc_curves.png"),
                model_idx
            )

            print("  Generating confusion matrix...")
            plot_confusion_matrices(
                y_true_train, y_scores_train, weights_train,
                y_true_test, y_scores_test, weights_test,
                os.path.join(plots_dir, f"model{model_idx}_confusion_matrix.png"),
                model_idx
            )
        else:
            print(f"  Warning: Predictions file not found: {predictions_path}")
            print("  Skipping ROC curves and confusion matrix generation")

        return result

    # If skip_eval but results don't exist, warn and continue
    if skip_eval:
        print(f"  Warning: --skip-eval specified but results not found, performing evaluation")

    # Load model
    print(f"  Loading model from {model_path}")
    model, hyperparams = load_model_checkpoint(model_path, config_path, device)

    print(f"  Hyperparameters:")
    print(f"    Hidden nodes: {hyperparams['num_hidden']}")
    print(f"    Optimizer: {hyperparams['optimizer']}")
    print(f"    Learning rate: {hyperparams['initial_lr']}")
    print(f"    Weight decay: {hyperparams['weight_decay']}")
    print(f"    Scheduler: {hyperparams['scheduler']}")

    # Evaluate on train and test
    print("  Evaluating on train set...")
    y_true_train, y_scores_train, weights_train = evaluate_model(model, train_loader, device)

    print("  Evaluating on test set...")
    y_true_test, y_scores_test, weights_test = evaluate_model(model, test_loader, device)

    # Save predictions for later use with --skip-eval
    print("  Saving predictions...")
    predictions_path = os.path.join(output_dirs['overfitting'], f"model{model_idx}_predictions.npz")
    np.savez(predictions_path,
             y_true_train=y_true_train, y_scores_train=y_scores_train, weights_train=weights_train,
             y_true_test=y_true_test, y_scores_test=y_scores_test, weights_test=weights_test)

    # Perform comprehensive KS tests
    print("  Performing KS tests (16 tests)...")
    ks_results, histograms = perform_ks_tests_comprehensive(
        y_true_train, y_scores_train, weights_train,
        y_true_test, y_scores_test, weights_test,
        p_threshold,
        bin_merge_threshold,
        bin_merge_max_iterations
    )

    # Save KS results
    print("  Saving KS test results...")
    is_overfitted, min_p_value = save_ks_results(
        ks_results, histograms, output_dirs['overfitting'], model_idx
    )

    print(f"  Overfitting status: {'OVERFITTED' if is_overfitted else 'OK'}")
    print(f"  Min p-value: {min_p_value:.4f}")
    print(f"  Failed tests: {sum(1 for r in ks_results.values() if r['is_overfitted'])}/16")

    # Generate plots
    plots_dir = output_dirs['plots']

    print("  Generating KS test heatmap...")
    plot_ks_test_heatmap(
        ks_results,
        os.path.join(plots_dir, f"model{model_idx}_ks_test_heatmap.png"),
        model_idx, hyperparams
    )

    print("  Generating score distribution grid...")
    plot_score_distributions_grid(
        histograms,
        os.path.join(plots_dir, f"model{model_idx}_score_distributions_grid.png"),
        model_idx
    )

    print("  Generating ROC curves...")
    plot_roc_curves(
        y_true_train, y_scores_train, weights_train,
        y_true_test, y_scores_test, weights_test,
        os.path.join(plots_dir, f"model{model_idx}_roc_curves.png"),
        model_idx
    )

    print("  Generating confusion matrix...")
    plot_confusion_matrices(
        y_true_train, y_scores_train, weights_train,
        y_true_test, y_scores_test, weights_test,
        os.path.join(plots_dir, f"model{model_idx}_confusion_matrix.png"),
        model_idx
    )

    print("  Generating training curves...")
    config_data = load_model_config(config_path)
    plot_training_curves(
        config_data,
        os.path.join(plots_dir, f"model{model_idx}_training_curves.png"),
        model_idx
    )

    return {
        'model_idx': model_idx,
        'hyperparams': hyperparams,
        'is_overfitted': is_overfitted,
        'min_p_value': min_p_value,
        'ks_results': ks_results
    }


def create_summary_plots(all_results: List[Dict], model_info_df: Dict, output_dir: str):
    """Create summary plots across all models."""
    print("\nCreating summary plots...")

    # Extract data
    model_indices = [r['model_idx'] for r in all_results]
    min_p_values = [r['min_p_value'] for r in all_results]
    is_overfitted = [r['is_overfitted'] for r in all_results]

    # Get fitness scores from model_info
    fitness_scores = [model_info_df[idx]['fitness'] for idx in model_indices]

    # Plot 1: Fitness vs worst p-value
    fig, ax = plt.subplots(figsize=(12, 8))

    colors = ['red' if ovf else 'green' for ovf in is_overfitted]
    scatter = ax.scatter(fitness_scores, min_p_values, c=colors, s=100, alpha=0.6, edgecolors='black')

    # Add model labels
    for idx, fit, pval in zip(model_indices, fitness_scores, min_p_values):
        ax.annotate(f'{idx}', (fit, pval), fontsize=8, ha='center', va='center')

    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=1.5, label='p-value threshold (0.05)')
    ax.set_xlabel('Fitness Score (Validation Loss)', fontsize=13)
    ax.set_ylabel('Minimum KS Test p-value', fontsize=13)
    ax.set_title('Model Performance: Fitness vs Overfitting', fontsize=14, pad=15)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add custom legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', alpha=0.6, edgecolor='black', label='Not Overfitted'),
        Patch(facecolor='red', alpha=0.6, edgecolor='black', label='Overfitted')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_fitness_vs_ks_worst.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Plot 2: Overall KS test heatmap (all models)
    fig, ax = plt.subplots(figsize=(16, 20))

    # Create matrix: rows=models, columns=16 tests
    n_models = len(all_results)
    p_matrix = np.zeros((n_models, 16))

    test_labels = []
    for true_class_idx in range(4):
        for score_idx in range(4):
            test_labels.append(f"{CLASS_NAMES[true_class_idx][:3]}/{SCORE_NAMES[score_idx][:3]}")

    for i, result in enumerate(all_results):
        ks_results = result['ks_results']
        j = 0
        for true_class_idx in range(4):
            for score_idx in range(4):
                key = f"{CLASS_NAMES[true_class_idx]}_{SCORE_NAMES[score_idx]}"
                p_matrix[i, j] = ks_results[key]['p_value']
                j += 1

    im = ax.imshow(p_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')

    ax.set_xticks(np.arange(16))
    ax.set_yticks(np.arange(n_models))
    ax.set_xticklabels(test_labels, fontsize=8)
    ax.set_yticklabels([f"Model {idx}" for idx in model_indices], fontsize=8)

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

    ax.set_xlabel('KS Test (True Class / Score Type)', fontsize=12)
    ax.set_ylabel('Model', fontsize=12)
    ax.set_title('All Models - KS Test p-values Heatmap', fontsize=14, pad=15)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('p-value', rotation=270, labelpad=20, fontsize=12)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_ks_heatmap_all_models.png'), dpi=300, bbox_inches='tight')
    plt.close()

    print("  Summary plots created successfully")


def save_summary_text(all_results: List[Dict], model_info_df: Dict, output_path: str):
    """Save summary to text file."""
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("GA ITERATION OVERFITTING ANALYSIS SUMMARY\n")
        f.write("="*80 + "\n\n")

        f.write(f"Total models evaluated: {len(all_results)}\n")
        f.write(f"Models overfitted: {sum(1 for r in all_results if r['is_overfitted'])}\n")
        f.write(f"Models OK: {sum(1 for r in all_results if not r['is_overfitted'])}\n\n")

        f.write("-"*80 + "\n")
        f.write(f"{'Model':<8} {'Fitness':<12} {'Min p-val':<12} {'Overfitted':<12} {'Optimizer':<12} {'Nodes':<8}\n")
        f.write("-"*80 + "\n")

        for result in sorted(all_results, key=lambda x: x['min_p_value'], reverse=True):
            idx = result['model_idx']
            fitness = model_info_df[idx]['fitness']
            min_pval = result['min_p_value']
            ovf = "YES" if result['is_overfitted'] else "NO"
            optimizer = result['hyperparams']['optimizer']
            nodes = result['hyperparams']['num_hidden']

            f.write(f"{idx:<8} {fitness:<12.6f} {min_pval:<12.6f} {ovf:<12} {optimizer:<12} {nodes:<8}\n")

        f.write("-"*80 + "\n")

    print(f"  Summary saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize GA iteration results with comprehensive overfitting analysis")
    parser.add_argument("--signal", type=str, required=True, help="Signal name (e.g., MHc130_MA90)")
    parser.add_argument("--channel", type=str, required=True, help="Channel (e.g., Run1E2Mu)")
    parser.add_argument("--iteration", type=int, required=True, help="GA iteration number")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--pilot", action="store_true", help="Use pilot datasets")
    parser.add_argument("--skip-eval", action="store_true",
                       help="Skip evaluation and reuse existing histograms")
    parser.add_argument("--p-threshold", type=float, default=0.05, help="p-value threshold for overfitting")
    parser.add_argument("--model-indices", type=str, default=None,
                       help="Comma-separated model indices to process (default: all)")
    parser.add_argument("--input", type=str, default="GAOptim_bjets",
                       help="Input directory path (default: GAOptim_bjets)")

    args = parser.parse_args()

    print("="*80)
    print("GA ITERATION VISUALIZATION AND OVERFITTING ANALYSIS")
    print("="*80)
    print(f"Signal: {args.signal}")
    print(f"Channel: {args.channel}")
    print(f"Iteration: {args.iteration}")
    print(f"Device: {args.device}")
    print(f"Pilot mode: {args.pilot}")
    print(f"p-value threshold: {args.p_threshold}")
    print(f"Input directory: {args.input}")
    print("="*80)

    # Load configuration
    config = load_ga_config()

    # Get bin merge parameters from config
    overfitting_config = config.get_overfitting_config()
    bin_merge_threshold = overfitting_config.get('bin_merge_threshold', 1e-6)
    bin_merge_max_iterations = overfitting_config.get('bin_merge_max_iterations', 100)

    print(f"Bin merge threshold: {bin_merge_threshold}")
    print(f"Bin merge max iterations: {bin_merge_max_iterations}")

    # Setup paths
    signal_full = f"TTToHcToWAToMuMu-{args.signal}"
    base_dir = f"{args.input}/{args.channel}/multiclass/{signal_full}/GA-iter{args.iteration}"

    models_dir = os.path.join(base_dir, "models")
    json_dir = os.path.join(base_dir, "json")
    overfitting_dir = os.path.join(base_dir, "overfitting_diagnostics")
    plots_dir = os.path.join(base_dir, "plots")

    os.makedirs(overfitting_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    output_dirs = {
        'overfitting': overfitting_dir,
        'plots': plots_dir
    }

    # Load model_info.csv
    import csv
    model_info_path = os.path.join(json_dir, "model_info.csv")
    model_info_df = {}

    with open(model_info_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row['model']
            model_idx = int(model_name.replace('model', ''))
            fitness = float(row['fitness'])
            model_info_df[model_idx] = {'fitness': fitness}

    # Determine which models to process
    if args.model_indices:
        model_indices = [int(x) for x in args.model_indices.split(',')]
    else:
        model_indices = sorted(model_info_df.keys())

    print(f"\nProcessing {len(model_indices)} models: {model_indices}")

    # Load datasets (skip if all results exist and --skip-eval is set)
    train_loader, test_loader = None, None
    if args.skip_eval:
        # Check if all models have existing results
        all_exist = all(
            check_existing_results(idx, output_dirs)
            for idx in model_indices
        )
        if all_exist:
            print("\n--skip-eval: All results exist, skipping dataset loading")
        else:
            print("\n--skip-eval: Some results missing, loading datasets anyway")
            train_loader, test_loader = load_datasets(config, args.signal, args.channel, args.pilot, args.device)
    else:
        train_loader, test_loader = load_datasets(config, args.signal, args.channel, args.pilot, args.device)

    # Process each model
    all_results = []

    for model_idx in model_indices:
        model_path = os.path.join(models_dir, f"model{model_idx}.pt")
        config_path = os.path.join(json_dir, f"model{model_idx}.json")

        if not os.path.exists(model_path):
            print(f"Warning: Model {model_idx} not found, skipping")
            continue

        try:
            result = process_model(
                model_idx, model_path, config_path,
                train_loader, test_loader,
                args.device, output_dirs, args.p_threshold,
                bin_merge_threshold, bin_merge_max_iterations,
                skip_eval=args.skip_eval
            )
            all_results.append(result)
        except Exception as e:
            print(f"Error processing model {model_idx}: {e}")
            import traceback
            traceback.print_exc()

    # Create summary
    print("\n" + "="*80)
    print("CREATING SUMMARY")
    print("="*80)

    create_summary_plots(all_results, model_info_df, plots_dir)
    save_summary_text(all_results, model_info_df, os.path.join(overfitting_dir, "overfitting_summary.txt"))

    # Save summary JSON
    summary_json_path = os.path.join(overfitting_dir, "overfitting_summary.json")
    with open(summary_json_path, 'w') as f:
        json.dump({
            'total_models': len(all_results),
            'overfitted_count': sum(1 for r in all_results if r['is_overfitted']),
            'p_threshold': args.p_threshold,
            'results': [
                {
                    'model_idx': r['model_idx'],
                    'fitness': model_info_df[r['model_idx']]['fitness'],
                    'is_overfitted': r['is_overfitted'],
                    'min_p_value': r['min_p_value']
                }
                for r in all_results
            ]
        }, f, indent=2)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to:")
    print(f"  Overfitting diagnostics: {overfitting_dir}")
    print(f"  Plots: {plots_dir}")
    print(f"\nSummary:")
    print(f"  Total models: {len(all_results)}")
    print(f"  Overfitted: {sum(1 for r in all_results if r['is_overfitted'])}")
    print(f"  OK: {sum(1 for r in all_results if not r['is_overfitted'])}")


if __name__ == "__main__":
    main()
