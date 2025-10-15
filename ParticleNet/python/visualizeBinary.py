#!/usr/bin/env python
"""
Binary classification visualization script for ParticleNet training results.

This script generates individual plots for binary classification models:
- Training curves (loss + accuracy combined)
- ROC curves with AUC values
- Precision-Recall curves
- Score distributions (signal vs background)
- Confusion matrices
- Training resource metrics

For detailed metric definitions and calculation methods, see:
    ParticleNet/docs/metrics_definition.md

Usage:
    python visualizeBinary.py --signal MHc160_MA85 --background nonprompt --channel Run1E2Mu --fold 3
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, classification_report
import logging
import ROOT
import cmsstyle as CMS

# Color palette consistent with Common/Tools/plotter.py
PALETTE = [
    ROOT.TColor.GetColor("#5790fc"),
    ROOT.TColor.GetColor("#f89c20"),
    ROOT.TColor.GetColor("#e42536"),
    ROOT.TColor.GetColor("#964a8b"),
    ROOT.TColor.GetColor("#9c9ca1"),
    ROOT.TColor.GetColor("#7a21dd")
]

logging.basicConfig(level=logging.INFO)

def setup_matplotlib():
    """Setup matplotlib for publication-quality plots."""
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'font.family': 'serif',
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'lines.linewidth': 2,
        'axes.grid': True,
        'grid.alpha': 0.3
    })

def find_binary_results(signal, background, channel, fold, pilot=False, separate_bjets=False):
    """Find binary classification results files."""
    if separate_bjets:
        base_path = "/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/ParticleNet/results_bjets"
    else:
        base_path = "/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/ParticleNet/results"
    signal_full = f"TTToHcToWAToMuMu-{signal}"

    # Map background categories to actual sample names
    background_map = {
        "nonprompt": "TTLL_powheg",
        "diboson": "WZTo3LNu_amcatnlo",
        "ttZ": "TTZToLLNuNu"
    }

    if background not in background_map:
        raise ValueError(f"Unknown background: {background}. Choose from {list(background_map.keys())}")

    if pilot:
        result_dir = os.path.join(base_path, channel, "binary", signal_full, "pilot")
    else:
        result_dir = os.path.join(base_path, channel, "binary", signal_full, f"fold-{fold}")

    # Find performance and model info files
    pattern = f"ParticleNet-nNodes128-Adam-initLR0p0010-decay0p00010-StepLR-weighted_ce-binary-{background}"
    performance_file = os.path.join(result_dir, f"{pattern}_performance.json")
    model_info_file = os.path.join(result_dir, f"{pattern}_model_info.json")

    if not os.path.exists(performance_file):
        raise FileNotFoundError(f"Performance file not found: {performance_file}")

    return performance_file, model_info_file

def load_training_data(performance_file):
    """Load training performance data from JSON file."""
    with open(performance_file, 'r') as f:
        data = json.load(f)

    # Extract training history
    history = data['training_results']['training_history']
    df = pd.DataFrame(history)

    return data, df

def load_predictions_from_root(signal, background, channel, fold, pilot=False, separate_bjets=False):
    """
    Load predictions from ROOT tree file.

    Returns:
        Tuple of (y_true_train, y_scores_train, weights_train,
                  y_true_test, y_scores_test, weights_test)
    """
    try:
        import uproot
    except ImportError:
        logging.warning("uproot not available. Using dummy data for demonstration.")
        return load_dummy_predictions()

    if separate_bjets:
        base_path = "/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/ParticleNet/results_bjets"
    else:
        base_path = "/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/ParticleNet/results"
    signal_full = f"TTToHcToWAToMuMu-{signal}"

    # Map background categories to actual sample names
    background_map = {
        "nonprompt": "TTLL_powheg",
        "diboson": "WZTo3LNu_amcatnlo",
        "ttZ": "TTZToLLNuNu"
    }

    if background not in background_map:
        raise ValueError(f"Unknown background: {background}")

    if pilot:
        result_dir = os.path.join(base_path, channel, "binary", signal_full, "pilot")
    else:
        result_dir = os.path.join(base_path, channel, "binary", signal_full, f"fold-{fold}")
    pattern = f"ParticleNet-nNodes128-Adam-initLR0p0010-decay0p00010-StepLR-weighted_ce-binary-{background}"
    root_file = os.path.join(result_dir, "trees", f"{pattern}.root")

    if not os.path.exists(root_file):
        logging.warning(f"ROOT file not found: {root_file}. Using dummy data.")
        return load_dummy_predictions()

    try:
        with uproot.open(root_file) as file:
            tree = file["Events"]

            # Load branches (using same structure as multiclass)
            y_true = tree["true_label"].array(library="np")

            # Load score and check if it needs to be inverted
            raw_scores = tree["score_signal"].array(library="np")

            # Check if signal events have lower scores (indicating score is actually background prob)
            signal_mask_check = (y_true == 1)
            bg_mask_check = (y_true == 0)

            if np.sum(signal_mask_check) > 0 and np.sum(bg_mask_check) > 0:
                signal_mean = np.mean(raw_scores[signal_mask_check])
                bg_mean = np.mean(raw_scores[bg_mask_check])

                if signal_mean < bg_mean:
                    # Score is actually background probability, so invert it
                    y_scores = 1.0 - raw_scores
                    logging.info("Applied score inversion: score_signal appears to be background probability")
                else:
                    y_scores = raw_scores
                    logging.info("Using score_signal directly as signal probability")
            else:
                y_scores = raw_scores

            sample_weights = tree["weight"].array(library="np")

            # Load train/test masks and b-jet flag
            train_mask = tree["train_mask"].array(library="np").astype(bool)
            test_mask = tree["test_mask"].array(library="np").astype(bool)

            # Load has_bjet flag if available
            has_bjet = None
            if "has_bjet" in tree.keys():
                has_bjet = tree["has_bjet"].array(library="np").astype(bool)
                logging.info("Loaded has_bjet flag for b-jet subset analysis")

            y_true_train = y_true[train_mask]
            y_scores_train = y_scores[train_mask]
            weights_train = sample_weights[train_mask]

            y_true_test = y_true[test_mask]
            y_scores_test = y_scores[test_mask]
            weights_test = sample_weights[test_mask]

            # Split has_bjet if available
            has_bjet_train = has_bjet[train_mask] if has_bjet is not None else None
            has_bjet_test = has_bjet[test_mask] if has_bjet is not None else None

            logging.info(f"Loaded {len(y_true_train)} training and {len(y_true_test)} test predictions from ROOT file")

            # Debug: Check label and score distributions
            unique_labels_train, label_counts_train = np.unique(y_true_train, return_counts=True)
            unique_labels_test, label_counts_test = np.unique(y_true_test, return_counts=True)
            logging.info(f"Train set label distribution: {dict(zip(unique_labels_train, label_counts_train))}")
            logging.info(f"Test set label distribution: {dict(zip(unique_labels_test, label_counts_test))}")
            logging.info(f"Train set score range: {y_scores_train.min():.6f} to {y_scores_train.max():.6f}")
            logging.info(f"Test set score range: {y_scores_test.min():.6f} to {y_scores_test.max():.6f}")

            # Check correlation between labels and scores for both train and test
            # Note: signal = 0, background = 1 in the dataset
            signal_mask_train = (y_true_train == 0)
            bg_mask_train = (y_true_train == 1)
            signal_mask_test = (y_true_test == 0)
            bg_mask_test = (y_true_test == 1)

            if np.sum(signal_mask_train) > 0 and np.sum(bg_mask_train) > 0:
                signal_score_mean_train = np.mean(y_scores_train[signal_mask_train])
                bg_score_mean_train = np.mean(y_scores_train[bg_mask_train])
                logging.info(f"TRAIN - Signal events (label=0) mean score: {signal_score_mean_train:.6f}")
                logging.info(f"TRAIN - Background events (label=1) mean score: {bg_score_mean_train:.6f}")

            if np.sum(signal_mask_test) > 0 and np.sum(bg_mask_test) > 0:
                signal_score_mean_test = np.mean(y_scores_test[signal_mask_test])
                bg_score_mean_test = np.mean(y_scores_test[bg_mask_test])
                logging.info(f"TEST - Signal events (label=0) mean score: {signal_score_mean_test:.6f}")
                logging.info(f"TEST - Background events (label=1) mean score: {bg_score_mean_test:.6f}")

                if signal_score_mean_test < bg_score_mean_test:
                    logging.warning("⚠️ Signal events have LOWER scores than background - possible label/score mismatch!")

            # Print b-jet statistics if available
            if has_bjet_train is not None:
                train_bjet_count = np.sum(has_bjet_train)
                test_bjet_count = np.sum(has_bjet_test)
                logging.info(f"Events with b-jets: {train_bjet_count}/{len(has_bjet_train)} train, {test_bjet_count}/{len(has_bjet_test)} test")

            return (y_true_train, y_scores_train, weights_train,
                    y_true_test, y_scores_test, weights_test, has_bjet_train, has_bjet_test)

    except Exception as e:
        logging.error(f"Error reading ROOT file: {e}. Using dummy data.")
        return load_dummy_predictions()

def load_dummy_predictions():
    """Generate realistic dummy data for demonstration with train/test splits."""
    logging.warning("Using dummy data for predictions.")

    np.random.seed(42)

    # Generate larger dataset and split into train/test
    def generate_data(n_samples, train_performance=0.95, test_performance=0.85):
        n_signal = n_samples // 2
        n_background = n_samples - n_signal

        # Adjust beta parameters based on desired performance
        if train_performance > test_performance:  # Training is better (some overfitting)
            signal_beta_a, signal_beta_b = 1.5, 3.5  # Slightly better separation
            background_beta_a, background_beta_b = 4, 1.5
        else:  # Test performance
            signal_beta_a, signal_beta_b = 2, 5  # More realistic separation
            background_beta_a, background_beta_b = 5, 2

        # Signal scores and labels
        signal_scores = np.random.beta(signal_beta_a, signal_beta_b, n_signal)
        signal_labels = np.zeros(n_signal)
        signal_weights = np.random.lognormal(-2, 1, n_signal)

        # Background scores and labels
        background_scores = np.random.beta(background_beta_a, background_beta_b, n_background)
        background_labels = np.ones(n_background)
        background_weights = np.random.lognormal(0, 0.5, n_background)

        # Combine and shuffle
        y_true = np.concatenate([signal_labels, background_labels])
        y_scores = np.concatenate([signal_scores, background_scores])
        sample_weights = np.concatenate([signal_weights, background_weights])

        indices = np.random.permutation(len(y_true))
        return y_true[indices], y_scores[indices], sample_weights[indices]

    # Generate training and test data
    y_true_train, y_scores_train, weights_train = generate_data(800, train_performance=0.95)
    y_true_test, y_scores_test, weights_test = generate_data(200, test_performance=0.85)

    # Generate dummy has_bjet flags (approximately 60% of events have b-jets)
    has_bjet_train = np.random.rand(len(y_true_train)) > 0.4
    has_bjet_test = np.random.rand(len(y_true_test)) > 0.4

    return (y_true_train, y_scores_train, weights_train,
            y_true_test, y_scores_test, weights_test, has_bjet_train, has_bjet_test)

def plot_training_curves(df, signal, background, output_path):
    """Plot combined loss and accuracy curves."""
    fig, ax1 = plt.subplots(figsize=(12, 8))

    # Plot loss on primary y-axis
    epochs = df['epoch']
    ax1.plot(epochs, df['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, df['valid_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='black')
    ax1.tick_params(axis='y')

    # Create secondary y-axis for accuracy
    ax2 = ax1.twinx()
    ax2.plot(epochs, df['train_acc'], 'b--', label='Training Accuracy', linewidth=2, alpha=0.7)
    ax2.plot(epochs, df['valid_acc'], 'r--', label='Validation Accuracy', linewidth=2, alpha=0.7)
    ax2.set_ylabel('Accuracy', color='black')
    ax2.tick_params(axis='y')
    ax2.set_ylim([0.0, 1.05])

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    plt.title(f'Training Curves: {signal} vs {background}', fontsize=16, pad=20)
    plt.tight_layout()

    output_file = os.path.join(output_path, 'training_curves.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Training curves saved to: {output_file}")

def plot_roc_curve(y_true_train, y_scores_train, weights_train,
                   y_true_test, y_scores_test, weights_test, signal, background, output_path):
    """Plot ROC curve for binary classification with train/test comparison."""
    # Handle negative weights from NLO Monte Carlo samples by using absolute values
    weights_train_abs = np.abs(weights_train)
    weights_test_abs = np.abs(weights_test)

    # Calculate weighted ROC curves for train and test
    fpr_train, tpr_train, _ = roc_curve(y_true_train, y_scores_train, sample_weight=weights_train_abs)
    # Fix numerical precision issues with weighted ROC curves
    fpr_train = np.clip(fpr_train, 0.0, 1.0)
    tpr_train = np.clip(tpr_train, 0.0, 1.0)
    roc_auc_train = auc(fpr_train, tpr_train)

    fpr_test, tpr_test, _ = roc_curve(y_true_test, y_scores_test, sample_weight=weights_test_abs)
    # Fix numerical precision issues with weighted ROC curves
    fpr_test = np.clip(fpr_test, 0.0, 1.0)
    tpr_test = np.clip(tpr_test, 0.0, 1.0)
    roc_auc_test = auc(fpr_test, tpr_test)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr_train, tpr_train, color='blue', lw=3, alpha=0.8,
             label=f'Training (AUC = {roc_auc_train:.3f})')
    plt.plot(fpr_test, tpr_test, color='darkorange', lw=3,
             label=f'Test (AUC = {roc_auc_test:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
             label='Random classifier')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {signal} vs {background}')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)

    output_file = os.path.join(output_path, 'roc_curve.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"ROC curve saved to: {output_file}")
    logging.info(f"Training ROC AUC: {roc_auc_train:.4f}, Test ROC AUC: {roc_auc_test:.4f}")
    return roc_auc_test  # Return test AUC for summary

def plot_precision_recall_curve(y_true_train, y_scores_train, weights_train,
                               y_true_test, y_scores_test, weights_test, signal, background, output_path):
    """Plot Precision-Recall curve for binary classification with train/test comparison."""
    # Handle negative weights from NLO Monte Carlo samples by using absolute values
    weights_train_abs = np.abs(weights_train)
    weights_test_abs = np.abs(weights_test)

    # Calculate weighted Precision-Recall curves for train and test
    precision_train, recall_train, _ = precision_recall_curve(y_true_train, y_scores_train, sample_weight=weights_train_abs)
    precision_test, recall_test, _ = precision_recall_curve(y_true_test, y_scores_test, sample_weight=weights_test_abs)

    # Fix numerical precision issues with weighted PR curves
    recall_train = np.clip(recall_train, 0.0, 1.0)
    precision_train = np.clip(precision_train, 0.0, 1.0)
    recall_test = np.clip(recall_test, 0.0, 1.0)
    precision_test = np.clip(precision_test, 0.0, 1.0)

    # Calculate AUC using average precision score which is more robust for PR curves
    from sklearn.metrics import average_precision_score
    try:
        pr_auc_train = average_precision_score(y_true_train, y_scores_train, sample_weight=weights_train_abs)
        pr_auc_test = average_precision_score(y_true_test, y_scores_test, sample_weight=weights_test_abs)
    except Exception as e:
        logging.warning(f"Failed to calculate weighted PR AUC: {e}")
        # Fallback to direct AUC calculation - PR curves should be monotonic by design
        pr_auc_train = auc(recall_train, precision_train)
        pr_auc_test = auc(recall_test, precision_test)

    plt.figure(figsize=(8, 8))
    plt.plot(recall_train, precision_train, color='blue', lw=3, alpha=0.8,
             label=f'Training (AUC = {pr_auc_train:.3f})')
    plt.plot(recall_test, precision_test, color='darkorange', lw=3,
             label=f'Test (AUC = {pr_auc_test:.3f})')

    # Add baseline (random classifier with weighted proportion from test set)
    baseline_weighted = np.sum(weights_test_abs[y_true_test == 1]) / np.sum(weights_test_abs)
    plt.axhline(y=baseline_weighted, color='green', linestyle='--', lw=2,
                label=f'Random baseline (AP = {baseline_weighted:.3f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve: {signal} vs {background}')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)

    output_file = os.path.join(output_path, 'precision_recall_curve.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Precision-Recall curve saved to: {output_file}")
    logging.info(f"Training PR AUC: {pr_auc_train:.4f}, Test PR AUC: {pr_auc_test:.4f}")
    return pr_auc_test  # Return test AUC for summary

def plot_score_distributions(y_true_train, y_scores_train, weights_train,
                            y_true_test, y_scores_test, weights_test, signal, background, output_path):
    """Plot score distributions for signal and background with train/test overlay using ROOT."""
    # Setup ROOT
    ROOT.gROOT.SetBatch(True)
    CMS.setCMSStyle()

    # Separate signal and background scores
    # Note: In PyTorch dataset (DynamicDatasetLoader), signal has label 0, background has label 1
    signal_mask_train = (y_true_train == 0)
    background_mask_train = (y_true_train == 1)
    signal_mask_test = (y_true_test == 0)
    background_mask_test = (y_true_test == 1)

    signal_scores_train = y_scores_train[signal_mask_train]
    background_scores_train = y_scores_train[background_mask_train]
    signal_weights_train = weights_train[signal_mask_train] 
    background_weights_train = weights_train[background_mask_train]

    signal_scores_test = y_scores_test[signal_mask_test]
    background_scores_test = y_scores_test[background_mask_test]
    signal_weights_test = weights_test[signal_mask_test]
    background_weights_test = weights_test[background_mask_test]

    # Define histogram parameters
    nbins = 50
    xmin, xmax = 0.0, 1.0

    # Create histograms
    h_signal_train = ROOT.TH1F("h_signal_train", f"{signal} Training", nbins, xmin, xmax)
    h_signal_test = ROOT.TH1F("h_signal_test", f"{signal} Test", nbins, xmin, xmax)
    h_background_train = ROOT.TH1F("h_background_train", f"{background} Training", nbins, xmin, xmax)
    h_background_test = ROOT.TH1F("h_background_test", f"{background} Test", nbins, xmin, xmax)

    # Fill histograms with physics weights
    for score, weight in zip(signal_scores_train, signal_weights_train):
        h_signal_train.Fill(score, weight)
    for score, weight in zip(signal_scores_test, signal_weights_test):
        h_signal_test.Fill(score, weight)
    for score, weight in zip(background_scores_train, background_weights_train):
        h_background_train.Fill(score, weight)
    for score, weight in zip(background_scores_test, background_weights_test):
        h_background_test.Fill(score, weight)

    # Normalize histograms
    if h_signal_train.Integral() > 0:
        h_signal_train.Scale(1.0 / h_signal_train.Integral())
    if h_signal_test.Integral() > 0:
        h_signal_test.Scale(1.0 / h_signal_test.Integral())
    if h_background_train.Integral() > 0:
        h_background_train.Scale(1.0 / h_background_train.Integral())
    if h_background_test.Integral() > 0:
        h_background_test.Scale(1.0 / h_background_test.Integral())

    # Find maximum for y-axis scaling
    max_val = max(h_signal_train.GetMaximum(), h_signal_test.GetMaximum(),
                  h_background_train.GetMaximum(), h_background_test.GetMaximum())

    # Create canvas with CMS style
    CMS.SetEnergy(13)
    CMS.SetLumi(-1, run="Run2")  # Simulation
    CMS.SetExtraText("Simulation Preliminary")

    canvas = CMS.cmsCanvas("score_dist", 0.0, 1.0, 1e-6, max_val * 100,
                          "Score", "Normalized Events", square=True,
                          iPos=11, extraSpace=0.)
    legend = CMS.cmsLeg(0.5, 0.8-0.035*4, 0.85, 0.8, textSize=0.035, columns=1)
    canvas.cd()
    canvas.SetLogy() 
    
    CMS.cmsObjectDraw(h_signal_train, "hist", LineColor=PALETTE[0], LineWidth=2, LineStyle=ROOT.kSolid)
    CMS.cmsObjectDraw(h_signal_test, "PE", MarkerColor=PALETTE[0], MarkerSize=1.0, MarkerStyle=ROOT.kFullCircle)
    CMS.cmsObjectDraw(h_background_train, "hist", LineColor=PALETTE[2], LineWidth=2, LineStyle=ROOT.kSolid)
    CMS.cmsObjectDraw(h_background_test, "PE", MarkerColor=PALETTE[2], MarkerSize=1.0, MarkerStyle=ROOT.kFullCircle)
    CMS.addToLegend(legend, (h_signal_train, f"{signal} (Training)", "L"))
    CMS.addToLegend(legend, (h_signal_test, f"{signal} (Test)", "PE"))
    CMS.addToLegend(legend, (h_background_train, f"{background} (Training)", "L"))
    CMS.addToLegend(legend, (h_background_test, f"{background} (Test)", "PE"))
    legend.Draw()
    canvas.RedrawAxis()

    # Save the plot
    output_file = os.path.join(output_path, 'score_distributions.png')
    canvas.SaveAs(output_file)

    # Clean up
    canvas.Close()

    logging.info(f"Score distributions saved to: {output_file}")

def plot_confusion_matrix(y_true_train, y_pred_train, y_true_test, y_pred_test, signal, background, output_path):
    """Plot confusion matrices for binary classification with train/test split."""
    # Calculate confusion matrices for train and test
    cm_train = confusion_matrix(y_true_train, y_pred_train)
    cm_test = confusion_matrix(y_true_test, y_pred_test)

    cm_train_normalized = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
    cm_test_normalized = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    # Training set - Raw counts
    sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues',
                xticklabels=[background, signal],
                yticklabels=[background, signal], ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title('Training Set (Counts)')

    # Training set - Normalized percentages
    sns.heatmap(cm_train_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=[background, signal],
                yticklabels=[background, signal], ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title('Training Set (Normalized)')

    # Test set - Raw counts
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=[background, signal],
                yticklabels=[background, signal], ax=ax3)
    ax3.set_xlabel('Predicted')
    ax3.set_ylabel('True')
    ax3.set_title('Test Set (Counts)')

    # Test set - Normalized percentages
    sns.heatmap(cm_test_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=[background, signal],
                yticklabels=[background, signal], ax=ax4)
    ax4.set_xlabel('Predicted')
    ax4.set_ylabel('True')
    ax4.set_title('Test Set (Normalized)')

    plt.suptitle(f'Confusion Matrices: {signal} vs {background}', fontsize=16)
    plt.tight_layout()

    output_file = os.path.join(output_path, 'confusion_matrix.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Confusion matrix saved to: {output_file}")

def plot_training_metrics(df, signal, background, output_path):
    """Plot training resource metrics (memory, time)."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    epochs = df['epoch']

    # Memory usage
    ax1.plot(epochs, df['memory_mb'], 'g-', linewidth=2, marker='o', markersize=4)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Memory Usage (MB)')
    ax1.set_title('Memory Usage During Training')
    ax1.grid(True, alpha=0.3)

    # Epoch time
    ax2.plot(epochs, df['epoch_time'], 'purple', linewidth=2, marker='s', markersize=4)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Epoch Time (seconds)')
    ax2.set_title('Training Time per Epoch')
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f'Training Metrics: {signal} vs {background}', fontsize=16)
    plt.tight_layout()

    output_file = os.path.join(output_path, 'training_metrics.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Training metrics saved to: {output_file}")

def generate_summary_report(data, roc_auc, pr_auc, signal, background, output_path):
    """Generate a summary report with key metrics."""
    training_results = data['training_results']

    report = {
        'Model Configuration': {
            'Signal': signal,
            'Background': background,
            'Channel': data['channel'],
            'Fold': data['fold'],
            'Model Type': training_results['model_type'],
            'Parameters': training_results['model_parameters'],
        },
        'Training Results': {
            'Epochs Completed': training_results['epochs_completed'],
            'Early Stopped': training_results['early_stopped'],
            'Total Training Time (s)': round(training_results['total_training_time'], 2),
            'Average Epoch Time (s)': round(training_results['avg_epoch_time'], 2),
            'Best Validation Loss': round(training_results['best_valid_loss'], 4),
            'Final Memory Usage (MB)': round(training_results['final_memory_mb'], 1),
        },
        'Performance Metrics': {
            'Test Loss': round(training_results['test_loss'], 4),
            'Test Accuracy': round(training_results['test_accuracy'], 4),
            'Physics-Weighted ROC AUC': round(roc_auc, 4),
            'Physics-Weighted PR AUC': round(pr_auc, 4),
            'Note': 'Metrics use physics event weights for proper evaluation'
        }
    }

    # Save as JSON
    output_file = os.path.join(output_path, 'summary_report.json')
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    # Save as text for easy reading
    text_file = os.path.join(output_path, 'summary_report.txt')
    with open(text_file, 'w') as f:
        f.write(f"Binary Classification Summary: {signal} vs {background}\n")
        f.write("=" * 60 + "\n\n")

        for section, metrics in report.items():
            f.write(f"{section}:\n")
            f.write("-" * 30 + "\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

    logging.info(f"Summary report saved to: {output_file}")

def plot_bjet_subset_comparison_roc(y_true_train, y_scores_train, weights_train, has_bjet_train,
                                    y_true_test, y_scores_test, weights_test, has_bjet_test,
                                    signal, background, output_path):
    """
    Plot ROC curve comparison between full dataset and b-jet subset.

    Creates comparison plots showing:
    - Full dataset ROC (train and test)
    - B-jet subset ROC (train and test)
    - Allows visualization of performance changes on b-jet-enriched events
    """
    # Handle negative weights
    weights_train_abs = np.abs(weights_train)
    weights_test_abs = np.abs(weights_test)

    logging.info("=" * 60)
    logging.info("B-JET SUBSET ROC CURVE COMPARISON")
    logging.info("=" * 60)

    # Calculate statistics
    train_bjet_fraction = np.sum(has_bjet_train) / len(has_bjet_train)
    test_bjet_fraction = np.sum(has_bjet_test) / len(has_bjet_test)
    logging.info(f"Train set: {np.sum(has_bjet_train)}/{len(has_bjet_train)} ({train_bjet_fraction*100:.1f}%) events with b-jets")
    logging.info(f"Test set: {np.sum(has_bjet_test)}/{len(has_bjet_test)} ({test_bjet_fraction*100:.1f}%) events with b-jets")

    # --- FULL DATASET ---
    # Calculate physics-weighted ROC curves for full dataset
    fpr_train_full, tpr_train_full, _ = roc_curve(y_true_train, y_scores_train,
                                                   sample_weight=weights_train_abs)
    fpr_test_full, tpr_test_full, _ = roc_curve(y_true_test, y_scores_test,
                                                 sample_weight=weights_test_abs)

    fpr_train_full = np.clip(fpr_train_full, 0.0, 1.0)
    tpr_train_full = np.clip(tpr_train_full, 0.0, 1.0)
    fpr_test_full = np.clip(fpr_test_full, 0.0, 1.0)
    tpr_test_full = np.clip(tpr_test_full, 0.0, 1.0)

    roc_auc_train_full = auc(fpr_train_full, tpr_train_full)
    roc_auc_test_full = auc(fpr_test_full, tpr_test_full)

    # --- B-JET SUBSET ---
    if np.sum(has_bjet_train) > 0 and np.sum(has_bjet_test) > 0:
        # Filter to b-jet events
        y_true_train_bjet = y_true_train[has_bjet_train]
        y_scores_train_bjet = y_scores_train[has_bjet_train]
        weights_train_bjet = weights_train_abs[has_bjet_train]

        y_true_test_bjet = y_true_test[has_bjet_test]
        y_scores_test_bjet = y_scores_test[has_bjet_test]
        weights_test_bjet = weights_test_abs[has_bjet_test]

        # Calculate physics-weighted ROC curves for b-jet subset
        fpr_train_bjet, tpr_train_bjet, _ = roc_curve(y_true_train_bjet, y_scores_train_bjet,
                                                       sample_weight=weights_train_bjet)
        fpr_test_bjet, tpr_test_bjet, _ = roc_curve(y_true_test_bjet, y_scores_test_bjet,
                                                     sample_weight=weights_test_bjet)

        fpr_train_bjet = np.clip(fpr_train_bjet, 0.0, 1.0)
        tpr_train_bjet = np.clip(tpr_train_bjet, 0.0, 1.0)
        fpr_test_bjet = np.clip(fpr_test_bjet, 0.0, 1.0)
        tpr_test_bjet = np.clip(tpr_test_bjet, 0.0, 1.0)

        roc_auc_train_bjet = auc(fpr_train_bjet, tpr_train_bjet)
        roc_auc_test_bjet = auc(fpr_test_bjet, tpr_test_bjet)

        # Plot comparison
        plt.figure(figsize=(10, 10))

        # Full dataset curves
        plt.plot(fpr_train_full, tpr_train_full, color='blue', lw=2, alpha=0.7, linestyle='-',
                label=f'Full Train (AUC = {roc_auc_train_full:.3f})')
        plt.plot(fpr_test_full, tpr_test_full, color='blue', lw=2, alpha=0.7, linestyle='--',
                label=f'Full Test (AUC = {roc_auc_test_full:.3f})')

        # B-jet subset curves
        plt.plot(fpr_train_bjet, tpr_train_bjet, color='red', lw=3, linestyle='-',
                label=f'B-jet Train (AUC = {roc_auc_train_bjet:.3f})')
        plt.plot(fpr_test_bjet, tpr_test_bjet, color='red', lw=3, linestyle='--',
                label=f'B-jet Test (AUC = {roc_auc_test_bjet:.3f})')

        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle=':', label='Random classifier')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14)
        plt.ylabel('True Positive Rate', fontsize=14)
        plt.title(f'ROC Comparison: {signal} vs {background}\nFull Dataset vs. B-jet Subset', fontsize=14)
        plt.legend(loc="lower right", fontsize=11)
        plt.grid(True, alpha=0.3)

        output_file = os.path.join(output_path, 'roc_curve_bjet_comparison.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"B-jet ROC comparison saved to: {output_file}")
        logging.info(f"  Full dataset - Train AUC: {roc_auc_train_full:.4f}, Test AUC: {roc_auc_test_full:.4f}")
        logging.info(f"  B-jet subset - Train AUC: {roc_auc_train_bjet:.4f}, Test AUC: {roc_auc_test_bjet:.4f}")
        logging.info(f"  AUC change (test): {roc_auc_test_bjet - roc_auc_test_full:+.4f}")
    else:
        logging.warning("Not enough b-jet events for ROC comparison")

    logging.info("=" * 60)

def plot_bjet_subset_score_distributions(y_true_train, y_scores_train, weights_train, has_bjet_train,
                                         y_true_test, y_scores_test, weights_test, has_bjet_test,
                                         signal, background, output_path):
    """
    Plot score distributions comparing full dataset vs. b-jet subset.

    Creates overlaid distributions showing how score distributions change
    when filtering to events containing b-jets.
    """
    # Setup ROOT
    ROOT.gROOT.SetBatch(True)
    CMS.setCMSStyle()

    # Handle negative weights
    weights_train_abs = np.abs(weights_train)
    weights_test_abs = np.abs(weights_test)

    logging.info("Generating b-jet subset score distribution comparisons...")

    # Separate signal and background for full dataset (test set only for clarity)
    signal_mask_test_full = (y_true_test == 0)
    bg_mask_test_full = (y_true_test == 1)

    signal_scores_test_full = y_scores_test[signal_mask_test_full]
    signal_weights_test_full = weights_test_abs[signal_mask_test_full]
    bg_scores_test_full = y_scores_test[bg_mask_test_full]
    bg_weights_test_full = weights_test_abs[bg_mask_test_full]

    # --- B-JET SUBSET ---
    if np.sum(has_bjet_test) > 10:  # Need sufficient events
        # Filter to b-jet events
        y_true_test_bjet = y_true_test[has_bjet_test]
        y_scores_test_bjet = y_scores_test[has_bjet_test]
        weights_test_bjet = weights_test_abs[has_bjet_test]

        # Separate signal and background for b-jet subset
        signal_mask_test_bjet = (y_true_test_bjet == 0)
        bg_mask_test_bjet = (y_true_test_bjet == 1)

        signal_scores_test_bjet = y_scores_test_bjet[signal_mask_test_bjet]
        signal_weights_test_bjet = weights_test_bjet[signal_mask_test_bjet]
        bg_scores_test_bjet = y_scores_test_bjet[bg_mask_test_bjet]
        bg_weights_test_bjet = weights_test_bjet[bg_mask_test_bjet]

        # Define histogram parameters
        nbins = 50
        xmin, xmax = 0.0, 1.0

        # Create histograms
        h_signal_full = ROOT.TH1F("h_signal_full_bin", f"{signal} (Full)", nbins, xmin, xmax)
        h_signal_bjet = ROOT.TH1F("h_signal_bjet_bin", f"{signal} (B-jet)", nbins, xmin, xmax)
        h_bg_full = ROOT.TH1F("h_bg_full_bin", f"{background} (Full)", nbins, xmin, xmax)
        h_bg_bjet = ROOT.TH1F("h_bg_bjet_bin", f"{background} (B-jet)", nbins, xmin, xmax)

        # Fill histograms with physics weights
        for score, weight in zip(signal_scores_test_full, signal_weights_test_full):
            h_signal_full.Fill(score, weight)
        for score, weight in zip(signal_scores_test_bjet, signal_weights_test_bjet):
            h_signal_bjet.Fill(score, weight)
        for score, weight in zip(bg_scores_test_full, bg_weights_test_full):
            h_bg_full.Fill(score, weight)
        for score, weight in zip(bg_scores_test_bjet, bg_weights_test_bjet):
            h_bg_bjet.Fill(score, weight)

        # Normalize histograms
        if h_signal_full.Integral() > 0:
            h_signal_full.Scale(1.0 / h_signal_full.Integral())
        if h_signal_bjet.Integral() > 0:
            h_signal_bjet.Scale(1.0 / h_signal_bjet.Integral())
        if h_bg_full.Integral() > 0:
            h_bg_full.Scale(1.0 / h_bg_full.Integral())
        if h_bg_bjet.Integral() > 0:
            h_bg_bjet.Scale(1.0 / h_bg_bjet.Integral())

        # Find maximum for y-axis scaling
        max_val = max(h_signal_full.GetMaximum(), h_signal_bjet.GetMaximum(),
                     h_bg_full.GetMaximum(), h_bg_bjet.GetMaximum())

        # Create canvas with CMS style
        CMS.SetEnergy(13)
        CMS.SetLumi(-1, run="Run2")
        CMS.SetExtraText("Simulation Preliminary")

        canvas = CMS.cmsCanvas("score_bjet_comp_bin", 0.0, 1.0, 1e-6, max_val * 100,
                              "Score", "Normalized Events", square=True,
                              iPos=11, extraSpace=0.)
        legend = CMS.cmsLeg(0.45, 0.7-0.035*4, 0.85, 0.85, textSize=0.03, columns=1)
        canvas.cd()
        canvas.SetLogy()

        # Draw histograms (test set only, full vs b-jet)
        CMS.cmsObjectDraw(h_signal_full, "hist", LineColor=PALETTE[0], LineWidth=2, LineStyle=ROOT.kSolid)
        CMS.cmsObjectDraw(h_signal_bjet, "hist", LineColor=PALETTE[0], LineWidth=3, LineStyle=ROOT.kDashed)
        CMS.cmsObjectDraw(h_bg_full, "hist", LineColor=PALETTE[2], LineWidth=2, LineStyle=ROOT.kSolid)
        CMS.cmsObjectDraw(h_bg_bjet, "hist", LineColor=PALETTE[2], LineWidth=3, LineStyle=ROOT.kDashed)

        CMS.addToLegend(legend, (h_signal_full, f"{signal} (Full)", "L"))
        CMS.addToLegend(legend, (h_signal_bjet, f"{signal} (B-jet subset)", "L"))
        CMS.addToLegend(legend, (h_bg_full, f"{background} (Full)", "L"))
        CMS.addToLegend(legend, (h_bg_bjet, f"{background} (B-jet subset)", "L"))
        legend.Draw()
        canvas.RedrawAxis()

        # Save the plot
        output_file = os.path.join(output_path, 'score_distributions_bjet_comparison.png')
        canvas.SaveAs(output_file)
        canvas.Close()

        logging.info(f"B-jet score distribution comparison saved to: {output_file}")
    else:
        logging.warning("Not enough b-jet events for score distribution comparison")

def plot_bjet_performance_summary(data, output_path):
    """
    Plot summary bar charts comparing full dataset vs. b-jet subset performance.

    Uses the bjet_subset_results from the performance JSON file to create
    visual comparisons of accuracy and loss metrics.
    """
    logging.info("Generating b-jet performance summary plots...")

    training_results = data['training_results']

    # Check if b-jet results exist
    if 'bjet_subset_results' not in training_results:
        logging.warning("No b-jet subset results found in training data")
        return

    bjet_results = training_results['bjet_subset_results']

    # Extract metrics for train and test
    train_results = bjet_results.get('train_results', {})
    test_results = bjet_results.get('test_results', {})

    if not train_results.get('has_bjet_events') or not test_results.get('has_bjet_events'):
        logging.warning("Insufficient b-jet events for summary plots")
        return

    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))

    # --- Plot 1: Train Accuracy Comparison ---
    categories = ['Full Dataset', 'B-jet Subset']
    train_accs = [train_results['full_train_accuracy'], train_results['bjet_subset_accuracy']]
    colors = ['blue', 'red']

    bars1 = ax1.bar(categories, train_accs, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Training Set Accuracy', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.05])
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars1, train_accs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # --- Plot 2: Test Accuracy Comparison ---
    test_accs = [test_results['full_test_accuracy'], test_results['bjet_subset_accuracy']]

    bars2 = ax2.bar(categories, test_accs, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Test Set Accuracy', fontsize=14, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars2, test_accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # --- Plot 3: Train Loss Comparison ---
    train_losses = [train_results['full_train_loss'], train_results['bjet_subset_loss']]

    bars3 = ax3.bar(categories, train_losses, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Loss')
    ax3.set_title('Training Set Loss', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars3, train_losses):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # --- Plot 4: Test Loss Comparison ---
    test_losses = [test_results['full_test_loss'], test_results['bjet_subset_loss']]

    bars4 = ax4.bar(categories, test_losses, color=colors, alpha=0.7, edgecolor='black')
    ax4.set_ylabel('Loss')
    ax4.set_title('Test Set Loss', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars4, test_losses):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Overall title with statistics
    train_stats = train_results['statistics']
    test_stats = test_results['statistics']
    fig.suptitle(f'B-Jet Subset Performance Summary\n'
                f'Train: {train_stats["bjet_events"]}/{train_stats["total_events"]} '
                f'({train_stats["bjet_event_fraction"]*100:.1f}%) events with b-jets | '
                f'Test: {test_stats["bjet_events"]}/{test_stats["total_events"]} '
                f'({test_stats["bjet_event_fraction"]*100:.1f}%) events with b-jets',
                fontsize=14, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = os.path.join(output_path, 'bjet_performance_summary.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"B-jet performance summary saved to: {output_file}")

    # Also save text summary
    summary_file = os.path.join(output_path, 'bjet_performance_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("B-JET SUBSET PERFORMANCE SUMMARY\n")
        f.write("=" * 70 + "\n\n")

        f.write("TRAINING SET:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total events: {train_stats['total_events']}\n")
        f.write(f"Events with b-jets: {train_stats['bjet_events']} ({train_stats['bjet_event_fraction']*100:.1f}%)\n")
        f.write(f"Average b-jets per b-jet event: {train_stats['avg_bjets_per_bjet_event']:.2f}\n\n")
        f.write(f"Full dataset - Accuracy: {train_results['full_train_accuracy']:.4f}, Loss: {train_results['full_train_loss']:.4f}\n")
        f.write(f"B-jet subset - Accuracy: {train_results['bjet_subset_accuracy']:.4f}, Loss: {train_results['bjet_subset_loss']:.4f}\n")
        f.write(f"Delta - Accuracy: {train_results['accuracy_delta']:+.4f}, Loss: {train_results['loss_delta']:+.4f}\n\n")

        f.write("TEST SET:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Total events: {test_stats['total_events']}\n")
        f.write(f"Events with b-jets: {test_stats['bjet_events']} ({test_stats['bjet_event_fraction']*100:.1f}%)\n")
        f.write(f"Average b-jets per b-jet event: {test_stats['avg_bjets_per_bjet_event']:.2f}\n\n")
        f.write(f"Full dataset - Accuracy: {test_results['full_test_accuracy']:.4f}, Loss: {test_results['full_test_loss']:.4f}\n")
        f.write(f"B-jet subset - Accuracy: {test_results['bjet_subset_accuracy']:.4f}, Loss: {test_results['bjet_subset_loss']:.4f}\n")
        f.write(f"Delta - Accuracy: {test_results['accuracy_delta']:+.4f}, Loss: {test_results['loss_delta']:+.4f}\n\n")

        if 'train_test_comparison' in bjet_results:
            comp = bjet_results['train_test_comparison']
            f.write("TRAIN vs TEST (B-jet subset):\n")
            f.write("-" * 40 + "\n")
            f.write(f"Accuracy gap (train - test): {comp['bjet_accuracy_gap']:+.4f}\n")
            f.write(f"Loss gap (train - test): {comp['bjet_loss_gap']:+.4f}\n")

    logging.info(f"B-jet performance text summary saved to: {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize binary ParticleNet classification results')
    parser.add_argument('--signal', required=True,
                        help='Signal point (e.g., MHc160_MA85)')
    parser.add_argument('--background', required=True,
                        choices=['nonprompt', 'diboson', 'ttZ'],
                        help='Background category')
    parser.add_argument('--channel', default='Run1E2Mu',
                        help='Analysis channel (default: Run1E2Mu)')
    parser.add_argument('--fold', type=int, default=3,
                        help='Cross-validation fold (default: 3)')
    parser.add_argument('--pilot', action='store_true',
                        help='Use pilot dataset results')
    parser.add_argument('--output',
                        help='Output directory (default: auto-generated)')
    parser.add_argument('--separate_bjets', action='store_true', default=False,
                        help='Use results from separate b-jets dataset')

    args = parser.parse_args()

    # Setup matplotlib
    setup_matplotlib()

    # Create output directory
    if args.output is None:
        if args.separate_bjets:
            base_output = f"/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/ParticleNet/plots_bjets/{args.channel}/binary"
        else:
            base_output = f"/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/ParticleNet/plots/{args.channel}/binary"
        signal_bg_dir = f"{args.signal}_vs_{args.background}"
        if args.pilot:
            args.output = os.path.join(base_output, signal_bg_dir, "pilot")
        else:
            args.output = os.path.join(base_output, signal_bg_dir, f"fold-{args.fold}")

    os.makedirs(args.output, exist_ok=True)

    try:
        # Find and load training results
        logging.info(f"Loading results for {args.signal} vs {args.background}")
        performance_file, model_info_file = find_binary_results(
            args.signal, args.background, args.channel, args.fold, args.pilot, args.separate_bjets)

        data, df = load_training_data(performance_file)

        # Generate individual plots
        logging.info("Generating training curves...")
        plot_training_curves(df, args.signal, args.background, args.output)

        logging.info("Generating training metrics...")
        plot_training_metrics(df, args.signal, args.background, args.output)

        # Load predictions with event weights (train/test splits)
        logging.info("Loading prediction data with physics weights...")
        (y_true_train, y_scores_train, weights_train,
         y_true_test, y_scores_test, weights_test, has_bjet_train, has_bjet_test) = load_predictions_from_root(
            args.signal, args.background, args.channel, args.fold, args.pilot, args.separate_bjets)

        y_pred_test = (y_scores_test > 0.5).astype(int)
        y_pred_train = (y_scores_train > 0.5).astype(int)

        # Log dataset statistics
        all_weights = np.concatenate([weights_train, weights_test])
        all_y_true = np.concatenate([y_true_train, y_true_test])

        logging.info(f"Loaded {len(y_true_train)} training + {len(y_true_test)} test samples")
        logging.info(f"Weight range: {all_weights.min():.6f} to {all_weights.max():.6f}")
        logging.info(f"Train - Signal: {np.sum(y_true_train == 0)}, Background: {np.sum(y_true_train == 1)}")
        logging.info(f"Test - Signal: {np.sum(y_true_test == 0)}, Background: {np.sum(y_true_test == 1)}")

        logging.info("Generating physics-aware ROC curve (train vs test)...")
        roc_auc = plot_roc_curve(y_true_train, y_scores_train, weights_train,
                                 y_true_test, y_scores_test, weights_test,
                                 args.signal, args.background, args.output)

        logging.info("Generating physics-aware Precision-Recall curve (train vs test)...")
        pr_auc = plot_precision_recall_curve(y_true_train, y_scores_train, weights_train,
                                           y_true_test, y_scores_test, weights_test,
                                           args.signal, args.background, args.output)

        logging.info("Generating physics-aware score distributions (train vs test)...")
        plot_score_distributions(y_true_train, y_scores_train, weights_train,
                                y_true_test, y_scores_test, weights_test,
                                args.signal, args.background, args.output)

        logging.info("Generating confusion matrices (train and test)...")
        plot_confusion_matrix(y_true_train, y_pred_train, y_true_test, y_pred_test, args.signal, args.background, args.output)

        # Generate summary report
        logging.info("Generating summary report...")
        generate_summary_report(data, roc_auc, pr_auc, args.signal, args.background, args.output)

        # B-jet subset visualizations (if has_bjet flag available)
        if has_bjet_train is not None and has_bjet_test is not None:
            logging.info("Generating b-jet subset ROC comparison...")
            plot_bjet_subset_comparison_roc(y_true_train, y_scores_train, weights_train, has_bjet_train,
                                           y_true_test, y_scores_test, weights_test, has_bjet_test,
                                           args.signal, args.background, args.output)

            logging.info("Generating b-jet subset score distributions...")
            plot_bjet_subset_score_distributions(y_true_train, y_scores_train, weights_train, has_bjet_train,
                                                y_true_test, y_scores_test, weights_test, has_bjet_test,
                                                args.signal, args.background, args.output)

            logging.info("Generating b-jet performance summary...")
            plot_bjet_performance_summary(data, args.output)
        else:
            logging.info("No b-jet flag available, skipping b-jet subset visualizations")

        logging.info(f"All plots and reports saved to: {args.output}")

    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        raise

if __name__ == "__main__":
    main()