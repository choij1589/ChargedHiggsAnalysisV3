#!/usr/bin/env python3
"""
ROC Curve Calculator with Negative Weight Handling

This module provides a manual ROC curve calculation that properly handles
negative event weights (from NLO MC generators) which cause issues with
sklearn's roc_curve function.

The algorithm:
1. Sorts events by discriminant score
2. Accumulates signed weights for TPR/FPR calculation
3. Handles edge cases and ensures monotonicity
4. Uses ROOT/cmsstyle for visualization

Usage:
    from ROCCurveCalculator import ROCCurveCalculator

    calculator = ROCCurveCalculator()
    calculator.plot_multiclass_rocs(
        y_true, y_scores, weights, output_path, model_idx, class_names
    )
"""

import os
import numpy as np
from typing import Tuple, List, Dict
import ROOT

try:
    import cmsstyle as CMS
    CMS.setCMSStyle()
    HAS_CMS_STYLE = True
except ImportError:
    print("Warning: cmsstyle not available, using default ROOT style")
    HAS_CMS_STYLE = False

# Color palette (consistent with plotter.py and visualizeGAIteration.py)
PALETTE = [
    ROOT.TColor.GetColor("#5790fc"),  # Blue - Signal
    ROOT.TColor.GetColor("#f89c20"),  # Orange - Nonprompt
    ROOT.TColor.GetColor("#e42536"),  # Red - Diboson
    ROOT.TColor.GetColor("#964a8b"),  # Purple - ttX
    ROOT.TColor.GetColor("#9c9ca1"),  # Gray
    ROOT.TColor.GetColor("#7a21dd")   # Violet
]


class ROCCurveCalculator:
    """
    Manual ROC curve calculator handling negative weights.

    This class provides methods to calculate ROC curves from predictions
    and event weights, properly handling negative weights that arise from
    NLO MC generators. It uses ROOT for visualization with CMS style.
    """

    def __init__(self):
        """Initialize the ROC curve calculator."""
        ROOT.gROOT.SetBatch(True)
        if HAS_CMS_STYLE:
            CMS.setCMSStyle()

    def calculate_roc_curve(self, y_true: np.ndarray, scores: np.ndarray,
                           weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Calculate ROC curve with proper handling of negative weights.

        Algorithm:
        1. Sort events by score (descending)
        2. For each threshold, accumulate signed weights:
           - True positives: sum(weight) where (y_true == 1) and (score >= threshold)
           - False positives: sum(weight) where (y_true == 0) and (score >= threshold)
        3. Normalize by total positive/negative weights
        4. Handle edge cases for monotonicity

        Args:
            y_true: True binary labels (1 for signal, 0 for background)
            scores: Discriminant scores (higher = more signal-like)
            weights: Event weights (can be negative)

        Returns:
            fpr: False positive rate array
            tpr: True positive rate array
            auc: Area under the ROC curve

        Note:
            Negative weights are handled by:
            - Separating positive and negative weight contributions
            - Normalizing by sum of absolute weights in each class
            - Ensuring FPR/TPR remain in [0,1] through proper normalization
        """
        # Input validation
        if len(y_true) != len(scores) or len(y_true) != len(weights):
            raise ValueError("Input arrays must have same length")

        if len(y_true) == 0:
            return np.array([0, 1]), np.array([0, 1]), 0.5

        # Convert to numpy arrays
        y_true = np.asarray(y_true, dtype=int)
        scores = np.asarray(scores, dtype=float)
        weights = np.asarray(weights, dtype=float)

        # Sort by score (descending - higher score = more signal-like)
        sorted_indices = np.argsort(-scores)
        y_true_sorted = y_true[sorted_indices]
        weights_sorted = weights[sorted_indices]

        # Separate signal (1) and background (0) events
        is_signal = (y_true_sorted == 1)
        is_background = (y_true_sorted == 0)

        # Calculate total weights for normalization
        # Use sum of absolute weights to handle negative weights properly
        total_signal_weight = np.sum(np.abs(weights_sorted[is_signal]))
        total_background_weight = np.sum(np.abs(weights_sorted[is_background]))

        # Handle edge case: no signal or background events
        if total_signal_weight == 0 or total_background_weight == 0:
            return np.array([0, 1]), np.array([0, 1]), 0.5

        # Calculate cumulative sums
        # For each threshold (moving down sorted list), count how many signal/bg events pass
        cum_signal_weight = np.cumsum(weights_sorted * is_signal)
        cum_background_weight = np.cumsum(weights_sorted * is_background)

        # Calculate TPR and FPR
        # TPR = (true positives) / (total positives)
        # FPR = (false positives) / (total negatives)
        tpr = cum_signal_weight / total_signal_weight
        fpr = cum_background_weight / total_background_weight

        # Add endpoints (0,0) and (1,1)
        tpr = np.concatenate([[0], tpr, [1]])
        fpr = np.concatenate([[0], fpr, [1]])

        # Ensure monotonicity (may be violated due to negative weights)
        # FPR and TPR should be non-decreasing
        tpr = np.maximum.accumulate(tpr)
        fpr = np.maximum.accumulate(fpr)

        # Clip to [0, 1] range
        tpr = np.clip(tpr, 0, 1)
        fpr = np.clip(fpr, 0, 1)

        # Calculate AUC using trapezoidal rule
        auc = self.compute_auc(fpr, tpr)

        return fpr, tpr, auc

    def compute_auc(self, fpr: np.ndarray, tpr: np.ndarray) -> float:
        """
        Compute area under ROC curve using trapezoidal integration.

        Args:
            fpr: False positive rate array (x-axis)
            tpr: True positive rate array (y-axis)

        Returns:
            Area under the curve
        """
        # Sort by fpr to ensure proper integration
        sorted_indices = np.argsort(fpr)
        fpr_sorted = fpr[sorted_indices]
        tpr_sorted = tpr[sorted_indices]

        # Trapezoidal integration
        auc = np.trapz(tpr_sorted, fpr_sorted)

        return float(np.clip(auc, 0, 1))

    def plot_binary_roc(self, y_true: np.ndarray, scores: np.ndarray,
                       weights: np.ndarray, title: str = "",
                       color: int = None) -> Tuple[ROOT.TGraph, ROOT.TGraph, float]:
        """
        Create ROOT TGraph objects for a binary ROC curve.

        Args:
            y_true: True binary labels
            scores: Discriminant scores
            weights: Event weights
            title: Legend title for this curve
            color: ROOT color code (default: blue from PALETTE)

        Returns:
            roc_graph: TGraph of the ROC curve
            diag_graph: TGraph of diagonal (random classifier)
            auc: Area under the curve
        """
        if color is None:
            color = PALETTE[0]

        # Calculate ROC curve
        fpr, tpr, auc = self.calculate_roc_curve(y_true, scores, weights)

        # Create TGraph for ROC curve
        n_points = len(fpr)
        roc_graph = ROOT.TGraph(n_points)
        roc_graph.SetName(f"roc_{title.replace(' ', '_')}")

        for i in range(n_points):
            roc_graph.SetPoint(i, fpr[i], tpr[i])

        # Style the curve
        roc_graph.SetLineColor(color)
        roc_graph.SetLineWidth(2)
        roc_graph.SetMarkerColor(color)
        roc_graph.SetMarkerStyle(0)

        # Create diagonal line (random classifier)
        diag_graph = ROOT.TGraph(2)
        diag_graph.SetPoint(0, 0, 0)
        diag_graph.SetPoint(1, 1, 1)
        diag_graph.SetLineColor(ROOT.kGray+2)
        diag_graph.SetLineWidth(1)
        diag_graph.SetLineStyle(2)  # Dashed

        return roc_graph, diag_graph, auc

    def plot_multiclass_rocs(self, y_true_train: np.ndarray, y_scores_train: np.ndarray,
                            weights_train: np.ndarray,
                            y_true_test: np.ndarray, y_scores_test: np.ndarray,
                            weights_test: np.ndarray,
                            output_path: str,
                            model_idx: int = 0,
                            class_names: List[str] = None,
                            signal_class: int = 0) -> None:
        """
        Plot ROC curves for multi-class classification (one-vs-rest) with train and test.

        Creates a grid of ROC curves:
        - Signal vs each background class using likelihood ratio
        - Shows both train and test ROC curves with their AUC values

        Args:
            y_true_train: True class labels for training set (0=signal, 1=nonprompt, 2=diboson, 3=ttX)
            y_scores_train: Predicted class probabilities for training set (n_samples, n_classes)
            weights_train: Event weights for training set
            y_true_test: True class labels for test set
            y_scores_test: Predicted class probabilities for test set
            weights_test: Event weights for test set
            output_path: Path to save the plot
            model_idx: Model index for title
            class_names: List of class display names
            signal_class: Index of the signal class (default: 0)
        """
        # Default class names (using lowercase with hyphens as requested)
        if class_names is None:
            class_names = ["signal", "nonprompt", "diboson", "tt+X"]

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Number of background classes
        n_classes = y_scores_train.shape[1]
        n_backgrounds = n_classes - 1

        # Create canvas with subpads
        canvas = ROOT.TCanvas("c_roc", "ROC Curves", 1800, 600)
        canvas.Divide(3, 1, 0.01, 0.01)

        # Store graphs to prevent deletion
        graphs = []
        legends = []

        for idx, bg_class in enumerate([1, 2, 3]):  # Background classes
            pad = canvas.cd(idx + 1)
            pad.SetLeftMargin(0.13)
            pad.SetRightMargin(0.05)
            pad.SetTopMargin(0.08)
            pad.SetBottomMargin(0.12)
            pad.SetGrid()

            # Use color specific to this background class
            bg_color = PALETTE[bg_class]

            # ===== TRAIN ROC =====
            # Create binary classification: signal vs this background
            mask_train = (y_true_train == signal_class) | (y_true_train == bg_class)
            y_true_binary_train = (y_true_train[mask_train] == signal_class).astype(int)
            weights_binary_train = weights_train[mask_train]

            # Likelihood ratio score: P(signal) / [P(signal) + P(background)]
            signal_scores_train = y_scores_train[mask_train, signal_class]
            bg_scores_train = y_scores_train[mask_train, bg_class]
            lr_scores_train = signal_scores_train / (signal_scores_train + bg_scores_train + 1e-10)

            # Calculate train ROC
            roc_graph_train, _, auc_train = self.plot_binary_roc(
                y_true_binary_train, lr_scores_train, weights_binary_train,
                title=f"Signal_vs_{class_names[bg_class]}_train",
                color=bg_color
            )
            roc_graph_train.SetLineStyle(1)  # Solid line for train
            roc_graph_train.SetLineWidth(3)  # Thicker for visibility

            # ===== TEST ROC =====
            mask_test = (y_true_test == signal_class) | (y_true_test == bg_class)
            y_true_binary_test = (y_true_test[mask_test] == signal_class).astype(int)
            weights_binary_test = weights_test[mask_test]

            signal_scores_test = y_scores_test[mask_test, signal_class]
            bg_scores_test = y_scores_test[mask_test, bg_class]
            lr_scores_test = signal_scores_test / (signal_scores_test + bg_scores_test + 1e-10)

            # Calculate test ROC (use same color as train)
            roc_graph_test, diag_graph, auc_test = self.plot_binary_roc(
                y_true_binary_test, lr_scores_test, weights_binary_test,
                title=f"Signal_vs_{class_names[bg_class]}_test",
                color=bg_color
            )
            roc_graph_test.SetLineStyle(2)  # Dashed line for test
            roc_graph_test.SetLineWidth(3)  # Thicker for visibility

            graphs.append((roc_graph_train, roc_graph_test, diag_graph))

            # Create frame (capitalize for title display)
            bg_title = class_names[bg_class]
            if bg_title == "tt+X":
                bg_title = "tt+X"  # Keep as-is
            else:
                bg_title = bg_title.capitalize()

            frame = pad.DrawFrame(0, 0, 1, 1)
            frame.SetTitle(f"Signal vs {bg_title}")
            frame.GetXaxis().SetTitle("False Positive Rate")
            frame.GetYaxis().SetTitle("True Positive Rate")
            frame.GetXaxis().SetTitleSize(0.05)
            frame.GetYaxis().SetTitleSize(0.05)
            frame.GetXaxis().SetLabelSize(0.04)
            frame.GetYaxis().SetLabelSize(0.04)
            frame.GetXaxis().SetTitleOffset(1.0)
            frame.GetYaxis().SetTitleOffset(1.2)

            # Draw graphs
            diag_graph.Draw("L SAME")
            roc_graph_train.Draw("L SAME")
            roc_graph_test.Draw("L SAME")

            # Create legend with train and test AUC values
            legend = ROOT.TLegend(0.45, 0.15, 0.90, 0.35)
            legend.SetBorderSize(0)
            legend.SetFillStyle(0)
            legend.SetTextSize(0.035)
            legend.AddEntry(roc_graph_train, f"Train: AUC = {auc_train:.4f}", "L")
            legend.AddEntry(roc_graph_test, f"Test: AUC = {auc_test:.4f}", "L")
            legend.AddEntry(diag_graph, "Random", "L")
            legend.Draw()
            legends.append(legend)

            # Add CMS label if available
            if HAS_CMS_STYLE:
                CMS.cmsText = "CMS"
                CMS.extraText = "Preliminary"
                CMS.cmsTextSize = 0.65
                CMS.extraTextSize = 0.55
                CMS.relPosX = 0.10
                CMS.relPosY = -0.07
                try:
                    CMS.CMS_lumi(pad, "", 0)
                except:
                    pass  # Skip if CMS_lumi fails

            pad.Update()

        # Add title to canvas
        canvas.cd()
        title_text = ROOT.TLatex()
        title_text.SetNDC()
        title_text.SetTextSize(0.035)
        title_text.SetTextAlign(22)
        title_text.DrawLatex(0.5, 0.97, f"Model {model_idx} - ROC Curves")

        # Save canvas
        canvas.SaveAs(output_path)

        # Clean up
        canvas.Close()

    def plot_multiclass_ovr(self, y_true: np.ndarray, y_scores: np.ndarray,
                           weights: np.ndarray, output_path: str,
                           class_names: List[str] = None) -> None:
        """
        Plot one-vs-rest ROC curves for all classes on a single plot.

        Args:
            y_true: True class labels
            y_scores: Predicted class probabilities
            weights: Event weights
            output_path: Path to save the plot
            class_names: List of class display names
        """
        if class_names is None:
            class_names = ["Signal", "Nonprompt", "Diboson", "ttX"]

        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Create canvas
        canvas = ROOT.TCanvas("c_roc_ovr", "ROC Curves (One-vs-Rest)", 800, 800)
        canvas.SetLeftMargin(0.13)
        canvas.SetRightMargin(0.05)
        canvas.SetTopMargin(0.08)
        canvas.SetBottomMargin(0.12)
        canvas.SetGrid()

        # Create frame
        frame = canvas.DrawFrame(0, 0, 1, 1)
        frame.SetTitle("Multi-class ROC Curves (One-vs-Rest)")
        frame.GetXaxis().SetTitle("False Positive Rate")
        frame.GetYaxis().SetTitle("True Positive Rate")
        frame.GetXaxis().SetTitleSize(0.045)
        frame.GetYaxis().SetTitleSize(0.045)

        # Store graphs
        graphs = []
        legend = ROOT.TLegend(0.50, 0.15, 0.90, 0.40)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.035)

        # Plot ROC for each class (one-vs-rest)
        n_classes = y_scores.shape[1]
        for class_idx in range(n_classes):
            # Binary classification: this class vs all others
            y_true_binary = (y_true == class_idx).astype(int)
            scores_binary = y_scores[:, class_idx]

            # Plot ROC
            roc_graph, _, auc = self.plot_binary_roc(
                y_true_binary, scores_binary, weights,
                title=f"{class_names[class_idx]}_OvR",
                color=PALETTE[class_idx % len(PALETTE)]
            )

            graphs.append(roc_graph)
            roc_graph.Draw("L SAME")
            legend.AddEntry(roc_graph, f"{class_names[class_idx]}: AUC={auc:.3f}", "L")

        # Draw diagonal
        diag = ROOT.TGraph(2)
        diag.SetPoint(0, 0, 0)
        diag.SetPoint(1, 1, 1)
        diag.SetLineColor(ROOT.kGray+2)
        diag.SetLineWidth(1)
        diag.SetLineStyle(2)
        diag.Draw("L SAME")
        legend.AddEntry(diag, "Random", "L")

        legend.Draw()

        # Add CMS label if available
        if HAS_CMS_STYLE:
            try:
                CMS.CMS_lumi(canvas, "", 0)
            except:
                pass

        canvas.Update()
        canvas.SaveAs(output_path)
        canvas.Close()
