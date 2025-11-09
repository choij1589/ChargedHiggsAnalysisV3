#!/usr/bin/env python
"""
Multi-class classification visualization script for ParticleNet training results.

This script generates individual plots for multi-class classification models:
- Training curves (loss + accuracy combined)
- Per-class ROC curves (using likelihood ratio discriminant)
- Per-class confusion matrices
- Per-class score distributions (multi-class overlay)
- Binary direct score distributions (Signal vs. each Background)
- Likelihood ratio score distributions (for direct comparison with binary models)
- Class performance metrics

For detailed metric definitions and calculation methods, see:
    ParticleNet/docs/metrics_definition.md

Usage:
    python visualizeMultiClass.py --signal MHc160_MA85 --channel Run1E2Mu --fold 3
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from ROCCurveCalculator import ROCCurveCalculator
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

def find_multiclass_results(signal, channel, fold, pilot=False, separate_bjets=False):
    """Find multi-class classification results files."""
    if separate_bjets:
        base_path = "/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/ParticleNet/results_bjets"
    else:
        base_path = "/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/ParticleNet/results"
    signal_full = f"TTToHcToWAToMuMu-{signal}"

    if pilot:
        result_dir = os.path.join(base_path, channel, "multiclass", signal_full, "pilot")
    else:
        result_dir = os.path.join(base_path, channel, "multiclass", signal_full, f"fold-{fold}")

    # Find performance and model info files
    pattern = f"ParticleNet-nNodes128-Adam-initLR0p0010-decay0p00010-StepLR-weighted_ce-3bg"
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

    # Extract class information
    class_names = ['Signal'] + [bg.replace('_', ' ').title() for bg in data['background_info']['backgrounds']]

    return data, df, class_names

def load_multiclass_predictions_from_root(signal, channel, fold, pilot=False, separate_bjets=False):
    """
    Load multiclass predictions from ROOT tree file.

    Returns:
        y_true: array of true class labels
        y_scores: array of prediction probabilities for each class
        class_names: list of class names
        sample_weights: array of physics event weights
    """
    try:
        import uproot
    except ImportError:
        logging.warning("uproot not available. Using dummy data for demonstration.")
        return load_dummy_multiclass_predictions()

    if separate_bjets:
        base_path = "/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/ParticleNet/results_bjets"
    else:
        base_path = "/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/ParticleNet/results"
    signal_full = f"TTToHcToWAToMuMu-{signal}"

    if pilot:
        result_dir = os.path.join(base_path, channel, "multiclass", signal_full, "pilot")
    else:
        result_dir = os.path.join(base_path, channel, "multiclass", signal_full, f"fold-{fold}")

    # Look for multiclass model files (3bg, 4bg, etc.)
    pattern = f"ParticleNet-nNodes128-Adam-initLR0p0010-decay0p00010-StepLR-weighted_ce-3bg"
    root_file = os.path.join(result_dir, "trees", f"{pattern}.root")

    if not os.path.exists(root_file):
        logging.warning(f"ROOT file not found: {root_file}. Using dummy data.")
        return load_dummy_multiclass_predictions()

    try:
        with uproot.open(root_file) as file:
            tree = file["Events"]

            # Load true labels and physics weights
            y_true = tree["true_label"].array(library="np")
            sample_weights = tree["weight"].array(library="np")

            # Load individual class scores and stack them
            score_signal = tree["score_signal"].array(library="np")
            score_ttll = tree["score_ttll"].array(library="np")  # Nonprompt
            score_wz = tree["score_wz"].array(library="np")     # Diboson
            score_ttz = tree["score_ttz"].array(library="np")   # TTZ

            # Stack scores into proper shape (n_samples, n_classes)
            y_scores = np.column_stack([score_signal, score_ttll, score_wz, score_ttz])

            # Load train/test masks and b-jet flag
            train_mask = tree["train_mask"].array(library="np").astype(bool)
            test_mask = tree["test_mask"].array(library="np").astype(bool)

            # Load has_bjet flag if available
            has_bjet = None
            if "has_bjet" in tree.keys():
                has_bjet = tree["has_bjet"].array(library="np").astype(bool)
                logging.info("Loaded has_bjet flag for b-jet subset analysis")

            # Split into train and test sets
            y_true_train = y_true[train_mask]
            y_scores_train = y_scores[train_mask]
            weights_train = sample_weights[train_mask]

            y_true_test = y_true[test_mask]
            y_scores_test = y_scores[test_mask]
            weights_test = sample_weights[test_mask]

            # Split has_bjet if available
            has_bjet_train = has_bjet[train_mask] if has_bjet is not None else None
            has_bjet_test = has_bjet[test_mask] if has_bjet is not None else None

            # Define class names based on the model
            class_names = ['Signal', 'Nonprompt', 'Diboson', 'TTZ']

            logging.info(f"Loaded {len(y_true_train)} train + {len(y_true_test)} test samples from ROOT file")
            logging.info(f"Classes: {len(class_names)} ({', '.join(class_names)})")
            logging.info(f"Physics weight range: {sample_weights.min():.6f} to {sample_weights.max():.6f}")

            # Print class distribution for test set
            for i, class_name in enumerate(class_names):
                count_test = np.sum(y_true_test == i)
                weight_test = np.sum(weights_test[y_true_test == i])
                logging.info(f"  {class_name} (test): {count_test} events, total weight: {weight_test:.3f}")

            # Print b-jet statistics if available
            if has_bjet_train is not None:
                train_bjet_count = np.sum(has_bjet_train)
                test_bjet_count = np.sum(has_bjet_test)
                logging.info(f"Events with b-jets: {train_bjet_count}/{len(has_bjet_train)} train, {test_bjet_count}/{len(has_bjet_test)} test")

            return y_true_train, y_scores_train, weights_train, y_true_test, y_scores_test, weights_test, class_names, has_bjet_train, has_bjet_test

    except Exception as e:
        logging.error(f"Error reading ROOT file {root_file}: {e}")
        logging.warning("Falling back to dummy data")
        return load_dummy_multiclass_predictions()

def load_dummy_multiclass_predictions():
    """Generate dummy multiclass data for demonstration."""
    logging.warning("Using dummy data for predictions. Implement real data loading for production use.")

    np.random.seed(42)
    n_train = 800
    n_test = 200
    n_classes = 4

    # Generate dummy data with realistic physics weights
    y_true_train = np.random.randint(0, n_classes, n_train)
    y_scores_train = np.random.dirichlet(np.ones(n_classes), n_train)

    y_true_test = np.random.randint(0, n_classes, n_test)
    y_scores_test = np.random.dirichlet(np.ones(n_classes), n_test)

    class_names = ['Signal', 'Nonprompt', 'Diboson', 'TTZ']

    # Generate realistic physics weights based on class
    def generate_weights(y_true):
        weights = np.ones(len(y_true))
        for i in range(len(y_true)):
            if y_true[i] == 0:  # Signal
                weights[i] = np.random.lognormal(-2, 1)  # Small weights, rare process
            elif y_true[i] == 1:  # Nonprompt
                weights[i] = np.random.lognormal(0.5, 0.8)  # Medium weights
            elif y_true[i] == 2:  # Diboson
                weights[i] = np.random.lognormal(-0.5, 0.6)  # Medium-small weights
            else:  # TTZ
                weights[i] = np.random.lognormal(-1, 0.7)  # Small-medium weights
        return weights

    weights_train = generate_weights(y_true_train)
    weights_test = generate_weights(y_true_test)

    # Generate dummy has_bjet flags (approximately 60% of events have b-jets)
    has_bjet_train = np.random.rand(n_train) > 0.4
    has_bjet_test = np.random.rand(n_test) > 0.4

    return y_true_train, y_scores_train, weights_train, y_true_test, y_scores_test, weights_test, class_names, has_bjet_train, has_bjet_test

def plot_training_curves(df, signal, output_path):
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

    plt.title(f'Training Curves: Multi-class {signal}', fontsize=16, pad=20)
    plt.tight_layout()

    output_file = os.path.join(output_path, 'training_curves.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Training curves saved to: {output_file}")

def plot_per_class_roc_curves(y_true_train, y_scores_train, weights_train,
                              y_true_test, y_scores_test, weights_test,
                              class_names, signal, output_path):
    """
    Plot individual ROC curves for Signal vs. each background type using likelihood ratio.

    This uses likelihood ratio scoring for optimal binary discrimination:
    - For each background, creates a Signal vs. Background binary problem
    - Uses likelihood ratio: score = P(signal) / (P(signal) + P(background))
    - Optimal discriminant under Neyman-Pearson lemma
    - Utilizes full multi-class probability information for fair comparison

    Uses ROOT and ROCCurveCalculator for proper negative weight handling.
    """
    ROOT.gROOT.SetBatch(True)
    CMS.setCMSStyle()

    # Initialize ROC calculator
    calculator = ROCCurveCalculator()

    n_classes = len(class_names)

    # Color palette consistent with other scripts
    PALETTE = [
        ROOT.TColor.GetColor("#5790fc"),  # Blue
        ROOT.TColor.GetColor("#f89c20"),  # Orange
        ROOT.TColor.GetColor("#e42536"),  # Red
        ROOT.TColor.GetColor("#964a8b"),  # Purple
    ]

    roc_aucs = {}

    # Plot Signal vs. each background separately (like binary classification)
    for bg_idx in range(1, n_classes):
        bg_name = class_names[bg_idx]

        # Filter to only Signal (label=0) vs. this background (label=bg_idx)
        train_mask = (y_true_train == 0) | (y_true_train == bg_idx)
        test_mask = (y_true_test == 0) | (y_true_test == bg_idx)

        # Create binary labels: Signal=1, Background=0
        y_binary_train = (y_true_train[train_mask] == 0).astype(int)
        y_binary_test = (y_true_test[test_mask] == 0).astype(int)

        # Calculate likelihood ratio: P(signal) / (P(signal) + P(background))
        signal_prob_train = y_scores_train[train_mask, 0]
        bg_prob_train = y_scores_train[train_mask, bg_idx]
        signal_scores_train = signal_prob_train / (signal_prob_train + bg_prob_train + 1e-10)

        signal_prob_test = y_scores_test[test_mask, 0]
        bg_prob_test = y_scores_test[test_mask, bg_idx]
        signal_scores_test = signal_prob_test / (signal_prob_test + bg_prob_test + 1e-10)

        # Get filtered weights (keep signed weights for proper handling)
        weights_train_filtered = weights_train[train_mask]
        weights_test_filtered = weights_test[test_mask]

        # Calculate ROC curves with proper negative weight handling
        fpr_train, tpr_train, auc_train = calculator.calculate_roc_curve(
            y_binary_train, signal_scores_train, weights_train_filtered
        )
        fpr_test, tpr_test, auc_test = calculator.calculate_roc_curve(
            y_binary_test, signal_scores_test, weights_test_filtered
        )

        # Store test AUC with "Signal vs. Background" naming for summary
        roc_aucs[f"Signal vs. {bg_name}"] = auc_test

        # Create ROOT canvas
        canvas = ROOT.TCanvas(f"c_roc_{bg_idx}", "ROC Curve", 800, 800)
        canvas.SetLeftMargin(0.13)
        canvas.SetRightMargin(0.05)
        canvas.SetTopMargin(0.08)
        canvas.SetBottomMargin(0.12)
        canvas.SetGrid()

        # Create frame
        frame = canvas.DrawFrame(0, 0, 1, 1)
        frame.SetTitle(f"Signal vs. {bg_name} ({signal})")
        frame.GetXaxis().SetTitle("False Positive Rate")
        frame.GetYaxis().SetTitle("True Positive Rate")
        frame.GetXaxis().SetTitleSize(0.045)
        frame.GetYaxis().SetTitleSize(0.045)
        frame.GetXaxis().SetLabelSize(0.04)
        frame.GetYaxis().SetLabelSize(0.04)

        # Create TGraphs for train and test
        graph_train = ROOT.TGraph(len(fpr_train))
        for i in range(len(fpr_train)):
            graph_train.SetPoint(i, fpr_train[i], tpr_train[i])
        graph_train.SetLineColor(PALETTE[bg_idx % len(PALETTE)])
        graph_train.SetLineWidth(3)
        graph_train.SetLineStyle(1)  # Solid

        graph_test = ROOT.TGraph(len(fpr_test))
        for i in range(len(fpr_test)):
            graph_test.SetPoint(i, fpr_test[i], tpr_test[i])
        graph_test.SetLineColor(PALETTE[bg_idx % len(PALETTE)])
        graph_test.SetLineWidth(3)
        graph_test.SetLineStyle(2)  # Dashed

        # Create diagonal line (random classifier)
        graph_diag = ROOT.TGraph(2)
        graph_diag.SetPoint(0, 0, 0)
        graph_diag.SetPoint(1, 1, 1)
        graph_diag.SetLineColor(ROOT.kGray+2)
        graph_diag.SetLineWidth(2)
        graph_diag.SetLineStyle(3)  # Dotted

        # Draw graphs
        graph_diag.Draw("L SAME")
        graph_train.Draw("L SAME")
        graph_test.Draw("L SAME")

        # Create legend
        legend = ROOT.TLegend(0.50, 0.15, 0.90, 0.35)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.035)
        legend.AddEntry(graph_train, f"Training (AUC = {auc_train:.3f})", "L")
        legend.AddEntry(graph_test, f"Test (AUC = {auc_test:.3f})", "L")
        legend.AddEntry(graph_diag, "Random classifier", "L")
        legend.Draw()

        # Add CMS label
        CMS.cmsText = "CMS"
        CMS.extraText = "Preliminary"
        CMS.cmsTextSize = 0.65
        CMS.extraTextSize = 0.55
        try:
            CMS.CMS_lumi(canvas, "", 0)
        except:
            pass  # Skip if CMS_lumi fails

        # Add subtitle for likelihood ratio
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.03)
        latex.SetTextAlign(22)
        latex.DrawLatex(0.5, 0.92, "Likelihood Ratio Discriminant")

        # Save canvas
        output_file = os.path.join(output_path,
                                  f'signal_vs_{bg_name.lower().replace(" ", "_")}_roc_curve.png')
        canvas.SaveAs(output_file)
        canvas.Close()

        logging.info(f"ROC curve for Signal vs. {bg_name} saved to: {output_file}")
        logging.info(f"  Training AUC: {auc_train:.4f}, Test AUC: {auc_test:.4f}")

    return roc_aucs

def plot_per_class_confusion_matrices(y_true_train, y_pred_train, y_true_test, y_pred_test, class_names, signal, output_path):
    """Plot individual confusion matrices focusing on each class with train/test split."""
    n_classes = len(class_names)

    # Overall confusion matrices for train and test
    cm_train = confusion_matrix(y_true_train, y_pred_train)
    cm_test = confusion_matrix(y_true_test, y_pred_test)
    cm_train_normalized = cm_train.astype('float') / cm_train.sum(axis=1)[:, np.newaxis]
    cm_test_normalized = cm_test.astype('float') / cm_test.sum(axis=1)[:, np.newaxis]

    # Plot overall confusion matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm_train_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')
    ax1.set_title(f'Training Set: {signal}')

    sns.heatmap(cm_test_normalized, annot=True, fmt='.2%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')
    ax2.set_title(f'Test Set: {signal}')

    plt.suptitle('Overall Confusion Matrices', fontsize=16)
    plt.tight_layout()

    output_file = os.path.join(output_path, 'overall_confusion_matrix.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Overall confusion matrix saved to: {output_file}")

    # Individual class performance matrices with train/test split
    for i, class_name in enumerate(class_names):
        # Create binary classification matrices for this class vs all others
        y_true_train_binary = (y_true_train == i).astype(int)
        y_pred_train_binary = (y_pred_train == i).astype(int)
        y_true_test_binary = (y_true_test == i).astype(int)
        y_pred_test_binary = (y_pred_test == i).astype(int)

        cm_train_binary = confusion_matrix(y_true_train_binary, y_pred_train_binary)
        cm_test_binary = confusion_matrix(y_true_test_binary, y_pred_test_binary)
        cm_train_binary_norm = cm_train_binary.astype('float') / cm_train_binary.sum(axis=1)[:, np.newaxis]
        cm_test_binary_norm = cm_test_binary.astype('float') / cm_test_binary.sum(axis=1)[:, np.newaxis]

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        # Training - Raw counts
        sns.heatmap(cm_train_binary, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Other', class_name],
                    yticklabels=['Other', class_name], ax=ax1)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('True')
        ax1.set_title(f'{class_name} vs Others - Training (Counts)')

        # Training - Normalized
        sns.heatmap(cm_train_binary_norm, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=['Other', class_name],
                    yticklabels=['Other', class_name], ax=ax2)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('True')
        ax2.set_title(f'{class_name} vs Others - Training (Normalized)')

        # Test - Raw counts
        sns.heatmap(cm_test_binary, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Other', class_name],
                    yticklabels=['Other', class_name], ax=ax3)
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('True')
        ax3.set_title(f'{class_name} vs Others - Test (Counts)')

        # Test - Normalized
        sns.heatmap(cm_test_binary_norm, annot=True, fmt='.2%', cmap='Blues',
                    xticklabels=['Other', class_name],
                    yticklabels=['Other', class_name], ax=ax4)
        ax4.set_xlabel('Predicted')
        ax4.set_ylabel('True')
        ax4.set_title(f'{class_name} vs Others - Test (Normalized)')

        plt.suptitle(f'Binary Classification: {class_name}', fontsize=16)
        plt.tight_layout()

        output_file = os.path.join(output_path, f'{class_name.lower().replace(" ", "_")}_confusion_matrix.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"Confusion matrix for {class_name} saved to: {output_file}")

def plot_per_class_score_distributions(y_true_train, y_scores_train, weights_train,
                                      y_true_test, y_scores_test, weights_test,
                                      class_names, signal, output_path):
    """Plot score distributions for each class with train/test overlay using ROOT."""
    n_classes = len(class_names)

    # Setup ROOT
    ROOT.gROOT.SetBatch(True)
    CMS.setCMSStyle()

    for i, class_name in enumerate(class_names):
        # Define histogram parameters
        nbins = 50
        xmin, xmax = 0.0, 1.0

        histograms = []
        max_val = 0.0

        # Create histograms for each true class (both train and test)
        for j, true_class in enumerate(class_names):
            # Training histograms
            mask_train = (y_true_train == j)
            if np.sum(mask_train) > 0:
                h_train = ROOT.TH1F(f"h_{true_class}_train_{i}", f"True {true_class} Training", nbins, xmin, xmax)
                scores_train = y_scores_train[mask_train, i]
                weights_class_train = weights_train[mask_train]

                for score, weight in zip(scores_train, weights_class_train):
                    h_train.Fill(score, weight)

                if h_train.Integral() > 0:
                    h_train.Scale(1.0 / h_train.Integral())


                histograms.append((h_train, f"{true_class} (Training)", "L"))
                max_val = max(max_val, h_train.GetMaximum())

            # Test histograms
            mask_test = (y_true_test == j)
            if np.sum(mask_test) > 0:
                h_test = ROOT.TH1F(f"h_{true_class}_test_{i}", f"True {true_class} Test", nbins, xmin, xmax)
                scores_test = y_scores_test[mask_test, i]
                weights_class_test = weights_test[mask_test]

                for score, weight in zip(scores_test, weights_class_test):
                    h_test.Fill(score, weight)

                if h_test.Integral() > 0:
                    h_test.Scale(1.0 / h_test.Integral())

                # Set style
                color_idx = j % len(PALETTE)
                h_test.SetLineColor(PALETTE[color_idx])
                h_test.SetLineWidth(3 if i == j else 2)  # Thicker line for target class
                h_test.SetLineStyle(ROOT.kDashed)  # Dashed for test
                h_test.SetStats(0)

                histograms.append((h_test, f"{true_class} (Test)", "PE"))
                max_val = max(max_val, h_test.GetMaximum())
                
        # Create canvas with CMS style
        CMS.SetEnergy(13)
        CMS.SetLumi(-1, run="Run2")  # Default Run2 luminosity
        CMS.SetExtraText("Simulation Preliminary")

        canvas = CMS.cmsCanvas(f"score_dist_{class_name}", 0.0, 1.0, 1e-4, max_val*100,
                              f"{class_name} Score", "Normalized Events")
        canvas.cd()
        canvas.SetLogy()
        
        legend = CMS.cmsLeg(0.5, 0.8-0.035*8, 0.85, 0.8, textSize=0.035, columns=1)
        
        for idx, (hist, label, style) in enumerate(histograms):
            # Determine color based on class pair (train/test share same color)
            color_idx = idx // 2  # Integer division to pair train/test
            
            if "(Training)" in label:
                # Draw train histograms as lines only
                CMS.cmsObjectDraw(hist, "hist", LineColor=PALETTE[color_idx], LineWidth=2, LineStyle=ROOT.kSolid)
            elif "(Test)" in label:
                # Draw test histograms as points with error bars only
                CMS.cmsObjectDraw(hist, "PE", MarkerColor=PALETTE[color_idx], MarkerSize=1.0, MarkerStyle=ROOT.kFullCircle)
            
            CMS.addToLegend(legend, (hist, label, style))
        legend.Draw()
        canvas.RedrawAxis()

        # Save the plot
        output_file = os.path.join(output_path, f'{class_name.lower().replace(" ", "_")}_score_distribution.png')
        canvas.SaveAs(output_file)

        # Clean up
        canvas.Close()

        logging.info(f"Score distribution for {class_name} saved to: {output_file}")

def plot_direct_score_distributions_binary(y_true_train, y_scores_train, weights_train,
                                          y_true_test, y_scores_test, weights_test,
                                          class_names, signal, output_path):
    """
    Plot direct score distributions for Signal vs. each background in binary format.

    This complements plot_per_class_score_distributions() by showing direct model outputs
    in a binary comparison format (Signal vs. each Background separately), allowing
    direct comparison with likelihood ratio plots.

    For each background, creates two plots:
    - P(signal) distribution for true Signal vs. true Background
    - P(background) distribution for true Signal vs. true Background

    This shows how the raw probability outputs separate signal from background,
    before the likelihood ratio transformation.
    """
    n_classes = len(class_names)

    # Setup ROOT
    ROOT.gROOT.SetBatch(True)
    CMS.setCMSStyle()

    # Handle negative weights
    weights_train_abs = np.abs(weights_train)
    weights_test_abs = np.abs(weights_test)

    # Plot Signal vs. each background separately
    for bg_idx in range(1, n_classes):
        bg_name = class_names[bg_idx]

        # Filter to only Signal (label=0) vs. this background (label=bg_idx)
        train_mask = (y_true_train == 0) | (y_true_train == bg_idx)
        test_mask = (y_true_test == 0) | (y_true_test == bg_idx)

        # Get filtered data
        signal_prob_train = y_scores_train[train_mask, 0]  # P(signal)
        bg_prob_train = y_scores_train[train_mask, bg_idx]  # P(background)
        weights_train_filtered = weights_train_abs[train_mask]

        signal_prob_test = y_scores_test[test_mask, 0]
        bg_prob_test = y_scores_test[test_mask, bg_idx]
        weights_test_filtered = weights_test_abs[test_mask]

        # Separate by true label within the filtered dataset
        signal_mask_train = (y_true_train[train_mask] == 0)
        bg_mask_train = (y_true_train[train_mask] == bg_idx)
        signal_mask_test = (y_true_test[test_mask] == 0)
        bg_mask_test = (y_true_test[test_mask] == bg_idx)

        # --- Plot 1: P(signal) distribution ---
        # Extract P(signal) scores for true signal and true background
        psignal_signal_train = signal_prob_train[signal_mask_train]
        psignal_signal_weights_train = weights_train_filtered[signal_mask_train]
        psignal_bg_train = signal_prob_train[bg_mask_train]
        psignal_bg_weights_train = weights_train_filtered[bg_mask_train]

        psignal_signal_test = signal_prob_test[signal_mask_test]
        psignal_signal_weights_test = weights_test_filtered[signal_mask_test]
        psignal_bg_test = signal_prob_test[bg_mask_test]
        psignal_bg_weights_test = weights_test_filtered[bg_mask_test]

        # Define histogram parameters
        nbins = 50
        xmin, xmax = 0.0, 1.0

        # Create histograms for P(signal)
        h_psignal_signal_train = ROOT.TH1F(f"h_psignal_signal_train_{bg_idx}",
                                           "Signal Training", nbins, xmin, xmax)
        h_psignal_signal_test = ROOT.TH1F(f"h_psignal_signal_test_{bg_idx}",
                                          "Signal Test", nbins, xmin, xmax)
        h_psignal_bg_train = ROOT.TH1F(f"h_psignal_bg_train_{bg_idx}",
                                       f"{bg_name} Training", nbins, xmin, xmax)
        h_psignal_bg_test = ROOT.TH1F(f"h_psignal_bg_test_{bg_idx}",
                                      f"{bg_name} Test", nbins, xmin, xmax)

        # Fill histograms with physics weights
        for score, weight in zip(psignal_signal_train, psignal_signal_weights_train):
            h_psignal_signal_train.Fill(score, weight)
        for score, weight in zip(psignal_signal_test, psignal_signal_weights_test):
            h_psignal_signal_test.Fill(score, weight)
        for score, weight in zip(psignal_bg_train, psignal_bg_weights_train):
            h_psignal_bg_train.Fill(score, weight)
        for score, weight in zip(psignal_bg_test, psignal_bg_weights_test):
            h_psignal_bg_test.Fill(score, weight)

        # Normalize histograms
        if h_psignal_signal_train.Integral() > 0:
            h_psignal_signal_train.Scale(1.0 / h_psignal_signal_train.Integral())
        if h_psignal_signal_test.Integral() > 0:
            h_psignal_signal_test.Scale(1.0 / h_psignal_signal_test.Integral())
        if h_psignal_bg_train.Integral() > 0:
            h_psignal_bg_train.Scale(1.0 / h_psignal_bg_train.Integral())
        if h_psignal_bg_test.Integral() > 0:
            h_psignal_bg_test.Scale(1.0 / h_psignal_bg_test.Integral())

        # Find maximum for y-axis scaling
        max_val = max(h_psignal_signal_train.GetMaximum(), h_psignal_signal_test.GetMaximum(),
                     h_psignal_bg_train.GetMaximum(), h_psignal_bg_test.GetMaximum())

        # Create canvas with CMS style
        CMS.SetEnergy(13)
        CMS.SetLumi(-1, run="Run2")
        CMS.SetExtraText("Simulation Preliminary")

        canvas = CMS.cmsCanvas(f"psignal_dist_{bg_idx}", 0.0, 1.0, 1e-6, max_val * 100,
                              "P(Signal)", "Normalized Events", square=True,
                              iPos=11, extraSpace=0.)
        legend = CMS.cmsLeg(0.5, 0.8-0.035*4, 0.85, 0.8, textSize=0.035, columns=1)
        canvas.cd()
        canvas.SetLogy()

        # Draw histograms with consistent styling
        CMS.cmsObjectDraw(h_psignal_signal_train, "hist", LineColor=PALETTE[0],
                         LineWidth=2, LineStyle=ROOT.kSolid)
        CMS.cmsObjectDraw(h_psignal_signal_test, "PE", MarkerColor=PALETTE[0],
                         MarkerSize=1.0, MarkerStyle=ROOT.kFullCircle)
        CMS.cmsObjectDraw(h_psignal_bg_train, "hist", LineColor=PALETTE[2],
                         LineWidth=2, LineStyle=ROOT.kSolid)
        CMS.cmsObjectDraw(h_psignal_bg_test, "PE", MarkerColor=PALETTE[2],
                         MarkerSize=1.0, MarkerStyle=ROOT.kFullCircle)

        CMS.addToLegend(legend, (h_psignal_signal_train, "Signal (Training)", "L"))
        CMS.addToLegend(legend, (h_psignal_signal_test, "Signal (Test)", "PE"))
        CMS.addToLegend(legend, (h_psignal_bg_train, f"{bg_name} (Training)", "L"))
        CMS.addToLegend(legend, (h_psignal_bg_test, f"{bg_name} (Test)", "PE"))
        legend.Draw()
        canvas.RedrawAxis()

        # Save the plot
        output_file = os.path.join(output_path,
                                  f'signal_vs_{bg_name.lower().replace(" ", "_")}_psignal_distribution.png')
        canvas.SaveAs(output_file)
        canvas.Close()

        logging.info(f"P(Signal) distribution for Signal vs. {bg_name} saved to: {output_file}")

        # --- Plot 2: P(background) distribution ---
        # Extract P(background) scores for true signal and true background
        pbg_signal_train = bg_prob_train[signal_mask_train]
        pbg_signal_weights_train = weights_train_filtered[signal_mask_train]
        pbg_bg_train = bg_prob_train[bg_mask_train]
        pbg_bg_weights_train = weights_train_filtered[bg_mask_train]

        pbg_signal_test = bg_prob_test[signal_mask_test]
        pbg_signal_weights_test = weights_test_filtered[signal_mask_test]
        pbg_bg_test = bg_prob_test[bg_mask_test]
        pbg_bg_weights_test = weights_test_filtered[bg_mask_test]

        # Create histograms for P(background)
        h_pbg_signal_train = ROOT.TH1F(f"h_pbg_signal_train_{bg_idx}",
                                       "Signal Training", nbins, xmin, xmax)
        h_pbg_signal_test = ROOT.TH1F(f"h_pbg_signal_test_{bg_idx}",
                                      "Signal Test", nbins, xmin, xmax)
        h_pbg_bg_train = ROOT.TH1F(f"h_pbg_bg_train_{bg_idx}",
                                   f"{bg_name} Training", nbins, xmin, xmax)
        h_pbg_bg_test = ROOT.TH1F(f"h_pbg_bg_test_{bg_idx}",
                                  f"{bg_name} Test", nbins, xmin, xmax)

        # Fill histograms with physics weights
        for score, weight in zip(pbg_signal_train, pbg_signal_weights_train):
            h_pbg_signal_train.Fill(score, weight)
        for score, weight in zip(pbg_signal_test, pbg_signal_weights_test):
            h_pbg_signal_test.Fill(score, weight)
        for score, weight in zip(pbg_bg_train, pbg_bg_weights_train):
            h_pbg_bg_train.Fill(score, weight)
        for score, weight in zip(pbg_bg_test, pbg_bg_weights_test):
            h_pbg_bg_test.Fill(score, weight)

        # Normalize histograms
        if h_pbg_signal_train.Integral() > 0:
            h_pbg_signal_train.Scale(1.0 / h_pbg_signal_train.Integral())
        if h_pbg_signal_test.Integral() > 0:
            h_pbg_signal_test.Scale(1.0 / h_pbg_signal_test.Integral())
        if h_pbg_bg_train.Integral() > 0:
            h_pbg_bg_train.Scale(1.0 / h_pbg_bg_train.Integral())
        if h_pbg_bg_test.Integral() > 0:
            h_pbg_bg_test.Scale(1.0 / h_pbg_bg_test.Integral())

        # Find maximum for y-axis scaling
        max_val = max(h_pbg_signal_train.GetMaximum(), h_pbg_signal_test.GetMaximum(),
                     h_pbg_bg_train.GetMaximum(), h_pbg_bg_test.GetMaximum())

        # Create canvas with CMS style
        canvas = CMS.cmsCanvas(f"pbg_dist_{bg_idx}", 0.0, 1.0, 1e-6, max_val * 100,
                              f"P({bg_name})", "Normalized Events", square=True,
                              iPos=11, extraSpace=0.)
        legend = CMS.cmsLeg(0.5, 0.8-0.035*4, 0.85, 0.8, textSize=0.035, columns=1)
        canvas.cd()
        canvas.SetLogy()

        # Draw histograms with consistent styling
        CMS.cmsObjectDraw(h_pbg_signal_train, "hist", LineColor=PALETTE[0],
                         LineWidth=2, LineStyle=ROOT.kSolid)
        CMS.cmsObjectDraw(h_pbg_signal_test, "PE", MarkerColor=PALETTE[0],
                         MarkerSize=1.0, MarkerStyle=ROOT.kFullCircle)
        CMS.cmsObjectDraw(h_pbg_bg_train, "hist", LineColor=PALETTE[2],
                         LineWidth=2, LineStyle=ROOT.kSolid)
        CMS.cmsObjectDraw(h_pbg_bg_test, "PE", MarkerColor=PALETTE[2],
                         MarkerSize=1.0, MarkerStyle=ROOT.kFullCircle)

        CMS.addToLegend(legend, (h_pbg_signal_train, "Signal (Training)", "L"))
        CMS.addToLegend(legend, (h_pbg_signal_test, "Signal (Test)", "PE"))
        CMS.addToLegend(legend, (h_pbg_bg_train, f"{bg_name} (Training)", "L"))
        CMS.addToLegend(legend, (h_pbg_bg_test, f"{bg_name} (Test)", "PE"))
        legend.Draw()
        canvas.RedrawAxis()

        # Save the plot
        output_file = os.path.join(output_path,
                                  f'signal_vs_{bg_name.lower().replace(" ", "_")}_p{bg_name.lower().replace(" ", "")}_distribution.png')
        canvas.SaveAs(output_file)
        canvas.Close()

        logging.info(f"P({bg_name}) distribution for Signal vs. {bg_name} saved to: {output_file}")

def plot_likelihood_ratio_score_distributions(y_true_train, y_scores_train, weights_train,
                                              y_true_test, y_scores_test, weights_test,
                                              class_names, signal, output_path):
    """
    Plot likelihood ratio score distributions for Signal vs. each background.

    Creates binary score distributions using likelihood ratio:
    score = P(signal) / (P(signal) + P(background))

    This shows how well the multi-class model's likelihood ratio discriminant
    separates signal from each individual background type.
    """
    n_classes = len(class_names)

    # Setup ROOT
    ROOT.gROOT.SetBatch(True)
    CMS.setCMSStyle()

    # Handle negative weights
    weights_train_abs = np.abs(weights_train)
    weights_test_abs = np.abs(weights_test)

    # Plot Signal vs. each background separately
    for bg_idx in range(1, n_classes):
        bg_name = class_names[bg_idx]

        # Filter to only Signal (label=0) vs. this background (label=bg_idx)
        train_mask = (y_true_train == 0) | (y_true_train == bg_idx)
        test_mask = (y_true_test == 0) | (y_true_test == bg_idx)

        # Calculate likelihood ratio scores
        signal_prob_train = y_scores_train[train_mask, 0]
        bg_prob_train = y_scores_train[train_mask, bg_idx]
        lr_scores_train = signal_prob_train / (signal_prob_train + bg_prob_train + 1e-10)

        signal_prob_test = y_scores_test[test_mask, 0]
        bg_prob_test = y_scores_test[test_mask, bg_idx]
        lr_scores_test = signal_prob_test / (signal_prob_test + bg_prob_test + 1e-10)

        # Get filtered weights
        weights_train_filtered = weights_train_abs[train_mask]
        weights_test_filtered = weights_test_abs[test_mask]

        # Separate by true label within the filtered dataset
        signal_mask_train = (y_true_train[train_mask] == 0)
        bg_mask_train = (y_true_train[train_mask] == bg_idx)
        signal_mask_test = (y_true_test[test_mask] == 0)
        bg_mask_test = (y_true_test[test_mask] == bg_idx)

        # Extract scores and weights for signal and background
        signal_scores_train = lr_scores_train[signal_mask_train]
        signal_weights_train = weights_train_filtered[signal_mask_train]
        bg_scores_train = lr_scores_train[bg_mask_train]
        bg_weights_train = weights_train_filtered[bg_mask_train]

        signal_scores_test = lr_scores_test[signal_mask_test]
        signal_weights_test = weights_test_filtered[signal_mask_test]
        bg_scores_test = lr_scores_test[bg_mask_test]
        bg_weights_test = weights_test_filtered[bg_mask_test]

        # Define histogram parameters
        nbins = 50
        xmin, xmax = 0.0, 1.0

        # Create histograms
        h_signal_train = ROOT.TH1F(f"h_signal_train_{bg_idx}", f"Signal Training", nbins, xmin, xmax)
        h_signal_test = ROOT.TH1F(f"h_signal_test_{bg_idx}", f"Signal Test", nbins, xmin, xmax)
        h_bg_train = ROOT.TH1F(f"h_bg_train_{bg_idx}", f"{bg_name} Training", nbins, xmin, xmax)
        h_bg_test = ROOT.TH1F(f"h_bg_test_{bg_idx}", f"{bg_name} Test", nbins, xmin, xmax)

        # Fill histograms with physics weights
        for score, weight in zip(signal_scores_train, signal_weights_train):
            h_signal_train.Fill(score, weight)
        for score, weight in zip(signal_scores_test, signal_weights_test):
            h_signal_test.Fill(score, weight)
        for score, weight in zip(bg_scores_train, bg_weights_train):
            h_bg_train.Fill(score, weight)
        for score, weight in zip(bg_scores_test, bg_weights_test):
            h_bg_test.Fill(score, weight)

        # Normalize histograms
        if h_signal_train.Integral() > 0:
            h_signal_train.Scale(1.0 / h_signal_train.Integral())
        if h_signal_test.Integral() > 0:
            h_signal_test.Scale(1.0 / h_signal_test.Integral())
        if h_bg_train.Integral() > 0:
            h_bg_train.Scale(1.0 / h_bg_train.Integral())
        if h_bg_test.Integral() > 0:
            h_bg_test.Scale(1.0 / h_bg_test.Integral())

        # Find maximum for y-axis scaling
        max_val = max(h_signal_train.GetMaximum(), h_signal_test.GetMaximum(),
                     h_bg_train.GetMaximum(), h_bg_test.GetMaximum())

        # Create canvas with CMS style
        CMS.SetEnergy(13)
        CMS.SetLumi(-1, run="Run2")
        CMS.SetExtraText("Simulation Preliminary")

        canvas = CMS.cmsCanvas(f"lr_score_dist_{bg_idx}", 0.0, 1.0, 1e-6, max_val * 100,
                              "Likelihood Ratio Score", "Normalized Events", square=True,
                              iPos=11, extraSpace=0.)
        legend = CMS.cmsLeg(0.5, 0.8-0.035*4, 0.85, 0.8, textSize=0.035, columns=1)
        canvas.cd()
        canvas.SetLogy()

        # Draw histograms with consistent styling (blue for signal, red for background)
        CMS.cmsObjectDraw(h_signal_train, "hist", LineColor=PALETTE[0], LineWidth=2, LineStyle=ROOT.kSolid)
        CMS.cmsObjectDraw(h_signal_test, "PE", MarkerColor=PALETTE[0], MarkerSize=1.0, MarkerStyle=ROOT.kFullCircle)
        CMS.cmsObjectDraw(h_bg_train, "hist", LineColor=PALETTE[2], LineWidth=2, LineStyle=ROOT.kSolid)
        CMS.cmsObjectDraw(h_bg_test, "PE", MarkerColor=PALETTE[2], MarkerSize=1.0, MarkerStyle=ROOT.kFullCircle)

        CMS.addToLegend(legend, (h_signal_train, "Signal (Training)", "L"))
        CMS.addToLegend(legend, (h_signal_test, "Signal (Test)", "PE"))
        CMS.addToLegend(legend, (h_bg_train, f"{bg_name} (Training)", "L"))
        CMS.addToLegend(legend, (h_bg_test, f"{bg_name} (Test)", "PE"))
        legend.Draw()
        canvas.RedrawAxis()

        # Save the plot
        output_file = os.path.join(output_path,
                                  f'signal_vs_{bg_name.lower().replace(" ", "_")}_lr_score_distribution.png')
        canvas.SaveAs(output_file)

        # Clean up
        canvas.Close()

        logging.info(f"Likelihood ratio score distribution for Signal vs. {bg_name} saved to: {output_file}")

def plot_bjet_subset_comparison_roc(y_true_train, y_scores_train, weights_train, has_bjet_train,
                                    y_true_test, y_scores_test, weights_test, has_bjet_test,
                                    class_names, signal, output_path):
    """
    Plot ROC curve comparison between full dataset and b-jet subset for Signal vs. each background.

    Creates comparison plots showing:
    - Full dataset ROC (train and test)
    - B-jet subset ROC (train and test)
    - Allows visualization of performance changes on b-jet-enriched events
    """
    n_classes = len(class_names)

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

    # Plot Signal vs. each background separately
    for bg_idx in range(1, n_classes):
        bg_name = class_names[bg_idx]

        # --- FULL DATASET ---
        # Filter to only Signal (label=0) vs. this background (label=bg_idx)
        train_mask_full = (y_true_train == 0) | (y_true_train == bg_idx)
        test_mask_full = (y_true_test == 0) | (y_true_test == bg_idx)

        # Create binary labels: Signal=1, Background=0
        y_binary_train_full = (y_true_train[train_mask_full] == 0).astype(int)
        y_binary_test_full = (y_true_test[test_mask_full] == 0).astype(int)

        # Calculate likelihood ratio scores
        signal_prob_train_full = y_scores_train[train_mask_full, 0]
        bg_prob_train_full = y_scores_train[train_mask_full, bg_idx]
        signal_scores_train_full = signal_prob_train_full / (signal_prob_train_full + bg_prob_train_full + 1e-10)

        signal_prob_test_full = y_scores_test[test_mask_full, 0]
        bg_prob_test_full = y_scores_test[test_mask_full, bg_idx]
        signal_scores_test_full = signal_prob_test_full / (signal_prob_test_full + bg_prob_test_full + 1e-10)

        weights_train_full = weights_train_abs[train_mask_full]
        weights_test_full = weights_test_abs[test_mask_full]

        # Calculate physics-weighted ROC curves for full dataset
        fpr_train_full, tpr_train_full, _ = roc_curve(y_binary_train_full, signal_scores_train_full,
                                                       sample_weight=weights_train_full)
        fpr_test_full, tpr_test_full, _ = roc_curve(y_binary_test_full, signal_scores_test_full,
                                                     sample_weight=weights_test_full)

        fpr_train_full = np.clip(fpr_train_full, 0.0, 1.0)
        tpr_train_full = np.clip(tpr_train_full, 0.0, 1.0)
        fpr_test_full = np.clip(fpr_test_full, 0.0, 1.0)
        tpr_test_full = np.clip(tpr_test_full, 0.0, 1.0)

        try:
            roc_auc_train_full = roc_auc_score(y_binary_train_full, signal_scores_train_full,
                                              sample_weight=weights_train_full)
            roc_auc_test_full = roc_auc_score(y_binary_test_full, signal_scores_test_full,
                                             sample_weight=weights_test_full)
        except Exception as e:
            logging.warning(f"Failed to calculate weighted ROC AUC for full dataset: {e}")
            roc_auc_train_full = auc(fpr_train_full, tpr_train_full)
            roc_auc_test_full = auc(fpr_test_full, tpr_test_full)

        # --- B-JET SUBSET ---
        # Apply b-jet filter on top of signal vs background filter
        train_mask_bjet = train_mask_full.copy()
        train_mask_bjet[train_mask_full] = has_bjet_train[train_mask_full]
        test_mask_bjet = test_mask_full.copy()
        test_mask_bjet[test_mask_full] = has_bjet_test[test_mask_full]

        if np.sum(train_mask_bjet) > 0 and np.sum(test_mask_bjet) > 0:
            # Create binary labels for b-jet subset
            y_binary_train_bjet = (y_true_train[train_mask_bjet] == 0).astype(int)
            y_binary_test_bjet = (y_true_test[test_mask_bjet] == 0).astype(int)

            # Calculate likelihood ratio scores for b-jet subset
            signal_prob_train_bjet = y_scores_train[train_mask_bjet, 0]
            bg_prob_train_bjet = y_scores_train[train_mask_bjet, bg_idx]
            signal_scores_train_bjet = signal_prob_train_bjet / (signal_prob_train_bjet + bg_prob_train_bjet + 1e-10)

            signal_prob_test_bjet = y_scores_test[test_mask_bjet, 0]
            bg_prob_test_bjet = y_scores_test[test_mask_bjet, bg_idx]
            signal_scores_test_bjet = signal_prob_test_bjet / (signal_prob_test_bjet + bg_prob_test_bjet + 1e-10)

            weights_train_bjet = weights_train_abs[train_mask_bjet]
            weights_test_bjet = weights_test_abs[test_mask_bjet]

            # Calculate physics-weighted ROC curves for b-jet subset
            fpr_train_bjet, tpr_train_bjet, _ = roc_curve(y_binary_train_bjet, signal_scores_train_bjet,
                                                           sample_weight=weights_train_bjet)
            fpr_test_bjet, tpr_test_bjet, _ = roc_curve(y_binary_test_bjet, signal_scores_test_bjet,
                                                         sample_weight=weights_test_bjet)

            fpr_train_bjet = np.clip(fpr_train_bjet, 0.0, 1.0)
            tpr_train_bjet = np.clip(tpr_train_bjet, 0.0, 1.0)
            fpr_test_bjet = np.clip(fpr_test_bjet, 0.0, 1.0)
            tpr_test_bjet = np.clip(tpr_test_bjet, 0.0, 1.0)

            try:
                roc_auc_train_bjet = roc_auc_score(y_binary_train_bjet, signal_scores_train_bjet,
                                                  sample_weight=weights_train_bjet)
                roc_auc_test_bjet = roc_auc_score(y_binary_test_bjet, signal_scores_test_bjet,
                                                 sample_weight=weights_test_bjet)
            except Exception as e:
                logging.warning(f"Failed to calculate weighted ROC AUC for b-jet subset: {e}")
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
            plt.title(f'ROC Comparison: Signal vs. {bg_name} ({signal})\nFull Dataset vs. B-jet Subset', fontsize=14)
            plt.legend(loc="lower right", fontsize=11)
            plt.grid(True, alpha=0.3)

            output_file = os.path.join(output_path,
                                      f'signal_vs_{bg_name.lower().replace(" ", "_")}_roc_bjet_comparison.png')
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()

            logging.info(f"B-jet ROC comparison for Signal vs. {bg_name} saved to: {output_file}")
            logging.info(f"  Full dataset - Train AUC: {roc_auc_train_full:.4f}, Test AUC: {roc_auc_test_full:.4f}")
            logging.info(f"  B-jet subset - Train AUC: {roc_auc_train_bjet:.4f}, Test AUC: {roc_auc_test_bjet:.4f}")
            logging.info(f"  AUC change (test): {roc_auc_test_bjet - roc_auc_test_full:+.4f}")
        else:
            logging.warning(f"Not enough b-jet events for Signal vs. {bg_name} comparison")

    logging.info("=" * 60)

def plot_bjet_subset_score_distributions(y_true_train, y_scores_train, weights_train, has_bjet_train,
                                         y_true_test, y_scores_test, weights_test, has_bjet_test,
                                         class_names, signal, output_path):
    """
    Plot likelihood ratio score distributions comparing full dataset vs. b-jet subset.

    Creates overlaid distributions showing how score distributions change
    when filtering to events containing b-jets.
    """
    n_classes = len(class_names)

    # Setup ROOT
    ROOT.gROOT.SetBatch(True)
    CMS.setCMSStyle()

    # Handle negative weights
    weights_train_abs = np.abs(weights_train)
    weights_test_abs = np.abs(weights_test)

    logging.info("Generating b-jet subset score distribution comparisons...")

    # Plot Signal vs. each background separately
    for bg_idx in range(1, n_classes):
        bg_name = class_names[bg_idx]

        # --- FULL DATASET ---
        # Filter to only Signal (label=0) vs. this background (label=bg_idx)
        train_mask_full = (y_true_train == 0) | (y_true_train == bg_idx)
        test_mask_full = (y_true_test == 0) | (y_true_test == bg_idx)

        # Calculate likelihood ratio scores for full dataset
        signal_prob_train_full = y_scores_train[train_mask_full, 0]
        bg_prob_train_full = y_scores_train[train_mask_full, bg_idx]
        lr_scores_train_full = signal_prob_train_full / (signal_prob_train_full + bg_prob_train_full + 1e-10)

        signal_prob_test_full = y_scores_test[test_mask_full, 0]
        bg_prob_test_full = y_scores_test[test_mask_full, bg_idx]
        lr_scores_test_full = signal_prob_test_full / (signal_prob_test_full + bg_prob_test_full + 1e-10)

        # Separate by true label
        signal_mask_train_full = (y_true_train[train_mask_full] == 0)
        bg_mask_train_full = (y_true_train[train_mask_full] == bg_idx)
        signal_mask_test_full = (y_true_test[test_mask_full] == 0)
        bg_mask_test_full = (y_true_test[test_mask_full] == bg_idx)

        # Get scores and weights for signal and background (test set only for clarity)
        signal_scores_test_full = lr_scores_test_full[signal_mask_test_full]
        signal_weights_test_full = weights_test_abs[test_mask_full][signal_mask_test_full]
        bg_scores_test_full = lr_scores_test_full[bg_mask_test_full]
        bg_weights_test_full = weights_test_abs[test_mask_full][bg_mask_test_full]

        # --- B-JET SUBSET ---
        # Apply b-jet filter
        train_mask_bjet = train_mask_full.copy()
        train_mask_bjet[train_mask_full] = has_bjet_train[train_mask_full]
        test_mask_bjet = test_mask_full.copy()
        test_mask_bjet[test_mask_full] = has_bjet_test[test_mask_full]

        if np.sum(test_mask_bjet) > 10:  # Need sufficient events
            # Calculate likelihood ratio scores for b-jet subset
            signal_prob_test_bjet = y_scores_test[test_mask_bjet, 0]
            bg_prob_test_bjet = y_scores_test[test_mask_bjet, bg_idx]
            lr_scores_test_bjet = signal_prob_test_bjet / (signal_prob_test_bjet + bg_prob_test_bjet + 1e-10)

            # Separate by true label
            signal_mask_test_bjet = (y_true_test[test_mask_bjet] == 0)
            bg_mask_test_bjet = (y_true_test[test_mask_bjet] == bg_idx)

            # Get scores and weights for b-jet subset
            signal_scores_test_bjet = lr_scores_test_bjet[signal_mask_test_bjet]
            signal_weights_test_bjet = weights_test_abs[test_mask_bjet][signal_mask_test_bjet]
            bg_scores_test_bjet = lr_scores_test_bjet[bg_mask_test_bjet]
            bg_weights_test_bjet = weights_test_abs[test_mask_bjet][bg_mask_test_bjet]

            # Define histogram parameters
            nbins = 50
            xmin, xmax = 0.0, 1.0

            # Create histograms
            h_signal_full = ROOT.TH1F(f"h_signal_full_{bg_idx}", "Signal (Full)", nbins, xmin, xmax)
            h_signal_bjet = ROOT.TH1F(f"h_signal_bjet_{bg_idx}", "Signal (B-jet)", nbins, xmin, xmax)
            h_bg_full = ROOT.TH1F(f"h_bg_full_{bg_idx}", f"{bg_name} (Full)", nbins, xmin, xmax)
            h_bg_bjet = ROOT.TH1F(f"h_bg_bjet_{bg_idx}", f"{bg_name} (B-jet)", nbins, xmin, xmax)

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

            canvas = CMS.cmsCanvas(f"lr_score_bjet_comp_{bg_idx}", 0.0, 1.0, 1e-6, max_val * 100,
                                  "Likelihood Ratio Score", "Normalized Events", square=True,
                                  iPos=11, extraSpace=0.)
            legend = CMS.cmsLeg(0.45, 0.7-0.035*4, 0.85, 0.85, textSize=0.03, columns=1)
            canvas.cd()
            canvas.SetLogy()

            # Draw histograms (test set only, full vs b-jet)
            CMS.cmsObjectDraw(h_signal_full, "hist", LineColor=PALETTE[0], LineWidth=2, LineStyle=ROOT.kSolid)
            CMS.cmsObjectDraw(h_signal_bjet, "hist", LineColor=PALETTE[0], LineWidth=3, LineStyle=ROOT.kDashed)
            CMS.cmsObjectDraw(h_bg_full, "hist", LineColor=PALETTE[2], LineWidth=2, LineStyle=ROOT.kSolid)
            CMS.cmsObjectDraw(h_bg_bjet, "hist", LineColor=PALETTE[2], LineWidth=3, LineStyle=ROOT.kDashed)

            CMS.addToLegend(legend, (h_signal_full, "Signal (Full)", "L"))
            CMS.addToLegend(legend, (h_signal_bjet, "Signal (B-jet subset)", "L"))
            CMS.addToLegend(legend, (h_bg_full, f"{bg_name} (Full)", "L"))
            CMS.addToLegend(legend, (h_bg_bjet, f"{bg_name} (B-jet subset)", "L"))
            legend.Draw()
            canvas.RedrawAxis()

            # Save the plot
            output_file = os.path.join(output_path,
                                      f'signal_vs_{bg_name.lower().replace(" ", "_")}_lr_score_bjet_comparison.png')
            canvas.SaveAs(output_file)
            canvas.Close()

            logging.info(f"B-jet score distribution comparison for Signal vs. {bg_name} saved to: {output_file}")
        else:
            logging.warning(f"Not enough b-jet events for Signal vs. {bg_name} score distribution")

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

def plot_class_performance_metrics(y_true, y_pred, class_names, sample_weights, signal, output_path):
    """Plot individual performance metrics for each class."""
    # Calculate physics-weighted classification report
    report = classification_report(y_true, y_pred, target_names=class_names,
                                 sample_weight=sample_weights, output_dict=True)

    # Extract metrics for each class
    metrics = ['precision', 'recall', 'f1-score']
    class_metrics = {metric: [] for metric in metrics}

    for class_name in class_names:
        for metric in metrics:
            class_metrics[metric].append(report[class_name][metric])

    # Plot each metric separately
    for metric in metrics:
        plt.figure(figsize=(10, 6))

        bars = plt.bar(class_names, class_metrics[metric],
                      color=['red', 'blue', 'green', 'orange'][:len(class_names)],
                      alpha=0.7, edgecolor='black')

        plt.xlabel('Class')
        plt.ylabel(metric.replace('-', ' ').title())
        plt.title(f'{metric.replace("-", " ").title()} by Class: {signal} (Physics-Weighted)')
        plt.ylim([0, 1.1])
        plt.xticks(rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, class_metrics[metric]):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')

        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()

        output_file = os.path.join(output_path, f'{metric.replace("-", "_")}_by_class.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

        logging.info(f"{metric} plot saved to: {output_file}")

    # Save classification report
    report_file = os.path.join(output_path, 'classification_report.json')
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)

    # Save as readable text
    text_file = os.path.join(output_path, 'classification_report.txt')
    with open(text_file, 'w') as f:
        f.write(f"Multi-class Classification Report: {signal} (Physics-Weighted)\n")
        f.write("=" * 70 + "\n\n")

        for class_name in class_names:
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {report[class_name]['precision']:.4f}\n")
            f.write(f"  Recall:    {report[class_name]['recall']:.4f}\n")
            f.write(f"  F1-score:  {report[class_name]['f1-score']:.4f}\n")
            f.write(f"  Support:   {report[class_name]['support']}\n\n")

        f.write("Overall:\n")
        f.write(f"  Accuracy:     {report['accuracy']:.4f}\n")
        f.write(f"  Macro avg:    {report['macro avg']['f1-score']:.4f}\n")
        f.write(f"  Weighted avg: {report['weighted avg']['f1-score']:.4f}\n")

def generate_summary_report(data, roc_aucs, signal, output_path):
    """Generate a summary report with key metrics."""
    training_results = data['training_results']

    report = {
        'Model Configuration': {
            'Signal': signal,
            'Channel': data['channel'],
            'Fold': data['fold'],
            'Model Type': training_results['model_type'],
            'Number of Classes': training_results['num_classes'],
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
            'Physics-Weighted ROC AUCs': {class_name: round(auc, 4) for class_name, auc in roc_aucs.items()},
            'Note': 'Metrics use physics event weights for proper evaluation'
        }
    }

    # Save as JSON
    output_file = os.path.join(output_path, 'summary_report.json')
    with open(output_file, 'w') as f:
        json.dump(report, f, indent=2)

    # Save as text
    text_file = os.path.join(output_path, 'summary_report.txt')
    with open(text_file, 'w') as f:
        f.write(f"Multi-class Classification Summary: {signal} (Physics-Weighted)\n")
        f.write("=" * 70 + "\n\n")

        for section, metrics in report.items():
            f.write(f"{section}:\n")
            f.write("-" * 30 + "\n")
            if section == 'Performance Metrics' and 'Physics-Weighted ROC AUCs' in metrics:
                for key, value in metrics.items():
                    if key == 'Physics-Weighted ROC AUCs':
                        f.write(f"  {key}:\n")
                        for class_name, auc in value.items():
                            f.write(f"    {class_name}: {auc}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            else:
                for key, value in metrics.items():
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

    logging.info(f"Summary report saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Visualize multi-class ParticleNet classification results')
    parser.add_argument('--signal', required=True,
                        help='Signal point (e.g., MHc160_MA85)')
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
            base_output = f"/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/ParticleNet/plots_bjets/{args.channel}/multiclass"
        else:
            base_output = f"/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/ParticleNet/plots/{args.channel}/multiclass"
        if args.pilot:
            args.output = os.path.join(base_output, args.signal, "pilot")
        else:
            args.output = os.path.join(base_output, args.signal, f"fold-{args.fold}")

    os.makedirs(args.output, exist_ok=True)

    try:
        # Find and load training results
        logging.info(f"Loading results for multi-class {args.signal}")
        performance_file, model_info_file = find_multiclass_results(
            args.signal, args.channel, args.fold, args.pilot, args.separate_bjets)

        data, df, class_names = load_training_data(performance_file)

        # Generate training curves
        logging.info("Generating training curves...")
        plot_training_curves(df, args.signal, args.output)

        # Load prediction data
        logging.info("Loading prediction data...")
        (y_true_train, y_scores_train, weights_train,
         y_true_test, y_scores_test, weights_test, class_names, has_bjet_train, has_bjet_test) = load_multiclass_predictions_from_root(
            args.signal, args.channel, args.fold, args.pilot, args.separate_bjets)

        # Use test data for evaluation (like binary visualization)
        y_pred_test = np.argmax(y_scores_test, axis=1)
        y_pred_train = np.argmax(y_scores_train, axis=1)

        # Generate individual plots with train/test comparison
        logging.info("Generating per-class ROC curves...")
        roc_aucs = plot_per_class_roc_curves(y_true_train, y_scores_train, weights_train,
                                            y_true_test, y_scores_test, weights_test,
                                            class_names, args.signal, args.output)

        logging.info("Generating per-class confusion matrices...")
        plot_per_class_confusion_matrices(y_true_train, y_pred_train, y_true_test, y_pred_test, class_names, args.signal, args.output)

        logging.info("Generating per-class score distributions...")
        plot_per_class_score_distributions(y_true_train, y_scores_train, weights_train,
                                          y_true_test, y_scores_test, weights_test,
                                          class_names, args.signal, args.output)

        logging.info("Generating binary direct score distributions...")
        plot_direct_score_distributions_binary(y_true_train, y_scores_train, weights_train,
                                              y_true_test, y_scores_test, weights_test,
                                              class_names, args.signal, args.output)

        logging.info("Generating likelihood ratio score distributions...")
        plot_likelihood_ratio_score_distributions(y_true_train, y_scores_train, weights_train,
                                                 y_true_test, y_scores_test, weights_test,
                                                 class_names, args.signal, args.output)

        logging.info("Generating class performance metrics...")
        plot_class_performance_metrics(y_true_test, y_pred_test, class_names, weights_test, args.signal, args.output)

        # Generate summary report
        logging.info("Generating summary report...")
        generate_summary_report(data, roc_aucs, args.signal, args.output)

        # B-jet subset visualizations (if has_bjet flag available)
        if has_bjet_train is not None and has_bjet_test is not None:
            logging.info("Generating b-jet subset ROC comparisons...")
            plot_bjet_subset_comparison_roc(y_true_train, y_scores_train, weights_train, has_bjet_train,
                                           y_true_test, y_scores_test, weights_test, has_bjet_test,
                                           class_names, args.signal, args.output)

            logging.info("Generating b-jet subset score distributions...")
            plot_bjet_subset_score_distributions(y_true_train, y_scores_train, weights_train, has_bjet_train,
                                                y_true_test, y_scores_test, weights_test, has_bjet_test,
                                                class_names, args.signal, args.output)

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