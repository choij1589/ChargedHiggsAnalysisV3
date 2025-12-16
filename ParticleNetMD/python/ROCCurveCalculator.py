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

        # Drop intermediate collinear points to reduce curve complexity
        # This makes dashed lines visible and improves rendering performance
        fpr, tpr = self.drop_intermediate(fpr, tpr)

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

    def drop_intermediate(self, fpr: np.ndarray, tpr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Drop intermediate points that lie on a straight line.

        This reduces the number of points in the ROC curve while preserving
        the exact curve shape and AUC. It keeps only the "corner" points where
        the curve changes direction, matching sklearn's drop_intermediate=True behavior.

        Args:
            fpr: False positive rate array
            tpr: True positive rate array

        Returns:
            fpr_reduced: FPR array with intermediate points removed
            tpr_reduced: TPR array with intermediate points removed
        """
        if len(fpr) <= 2:
            return fpr, tpr

        # Always keep first and last points
        keep_mask = np.zeros(len(fpr), dtype=bool)
        keep_mask[0] = True
        keep_mask[-1] = True

        # For each intermediate point, check if it's collinear with neighbors
        for i in range(1, len(fpr) - 1):
            # Calculate slopes: (y2-y1)/(x2-x1)
            # Point is NOT collinear if slopes differ
            dx1 = fpr[i] - fpr[i-1]
            dy1 = tpr[i] - tpr[i-1]
            dx2 = fpr[i+1] - fpr[i]
            dy2 = tpr[i+1] - tpr[i]

            # Use cross product to check collinearity (avoids division by zero)
            # Points are collinear if cross product is zero
            cross_product = dy1 * dx2 - dy2 * dx1

            # Keep point if it's NOT collinear (cross product non-zero)
            # Use small epsilon for numerical stability
            if abs(cross_product) > 1e-10:
                keep_mask[i] = True

        return fpr[keep_mask], tpr[keep_mask]

    def get_lighter_color(self, color_idx: int, lightness: float = 0.5) -> int:
        """
        Create a lighter version of a color from the PALETTE.

        Args:
            color_idx: ROOT color index from PALETTE
            lightness: Amount to lighten (0.0 = original, 1.0 = white)
                      Default 0.5 creates a nice pastel shade

        Returns:
            New ROOT color index for the lighter color
        """
        # Get RGB components from the original color
        color = ROOT.gROOT.GetColor(color_idx)
        if color is None:
            return color_idx

        r, g, b = color.GetRed(), color.GetGreen(), color.GetBlue()

        # Lighten by interpolating towards white (1.0, 1.0, 1.0)
        r_light = r + (1.0 - r) * lightness
        g_light = g + (1.0 - g) * lightness
        b_light = b + (1.0 - b) * lightness

        # Create and return new color
        return ROOT.TColor.GetColor(r_light, g_light, b_light)

    def plot_multiclass_rocs(self, y_true_train: np.ndarray, y_scores_train: np.ndarray,
                            weights_train: np.ndarray,
                            y_true_test: np.ndarray, y_scores_test: np.ndarray,
                            weights_test: np.ndarray,
                            output_path: str,
                            model_idx: int = 0,
                            class_names: List[str] = None,
                            signal_class: int = 0) -> None:
        """
        Plot ROC curves for multi-class classification on a single canvas.

        Creates consolidated ROC curves showing signal efficiency vs background rejection:
        - Signal vs each background class using likelihood ratio
        - Shows both train and test ROC curves with their AUC values
        - All curves on single plot with unified legend

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
        # Default class names
        if class_names is None:
            class_names = ["Signal", "Nonprompt", "Diboson", "TTX"]

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Configure CMS style
        if HAS_CMS_STYLE:
            CMS.SetEnergy(13)
            CMS.SetLumi(-1, run="Run2")
            CMS.SetExtraText("Simulation Preliminary")

        # Create single CMS canvas
        canvas = CMS.cmsCanvas(
            "",
            0, 1,  # x-axis: signal efficiency from 0 to 1
            0, 1,  # y-axis: background rejection from 0 to 1
            "Signal Efficiency",
            "Background Rejection",
            square=True,
            iPos=0,
            extraSpace=0.
        )
        canvas.SetGrid()

        # Store graphs to prevent deletion
        all_graphs = []

        # Create legend
        legend = CMS.cmsLeg(0.20, 0.15, 0.65, 0.45, textSize=0.030, columns=1)

        # Draw diagonal reference line (from (0,1) to (1,0))
        diag_graph = ROOT.TGraph(2)
        diag_graph.SetPoint(0, 0, 1)
        diag_graph.SetPoint(1, 1, 0)
        CMS.cmsObjectDraw(diag_graph, "L", LineColor=ROOT.kGray+2, LineWidth=1, LineStyle=2)
        all_graphs.append(diag_graph)

        # Process each background class
        for bg_class in [1, 2, 3]:  # Background classes
            # Use color specific to this background class
            bg_color = PALETTE[bg_class]
            bg_color_light = self.get_lighter_color(bg_color, lightness=0.5)  # Lighter shade for train
            bg_name = class_names[bg_class]

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
            fpr_train, tpr_train, auc_train = self.calculate_roc_curve(
                y_true_binary_train, lr_scores_train, weights_binary_train
            )

            # Create TGraph with transformed axes: (signal_eff, 1 - bg_eff) = (tpr, 1-fpr)
            n_points = len(fpr_train)
            roc_graph_train = ROOT.TGraph(n_points)
            for i in range(n_points):
                roc_graph_train.SetPoint(i, tpr_train[i], 1 - fpr_train[i])

            # Train: lighter color, solid line
            CMS.cmsObjectDraw(roc_graph_train, "L", LineColor=bg_color_light, LineWidth=2, LineStyle=ROOT.kSolid)
            all_graphs.append(roc_graph_train)

            # ===== TEST ROC =====
            mask_test = (y_true_test == signal_class) | (y_true_test == bg_class)
            y_true_binary_test = (y_true_test[mask_test] == signal_class).astype(int)
            weights_binary_test = weights_test[mask_test]

            signal_scores_test = y_scores_test[mask_test, signal_class]
            bg_scores_test = y_scores_test[mask_test, bg_class]
            lr_scores_test = signal_scores_test / (signal_scores_test + bg_scores_test + 1e-10)

            # Calculate test ROC
            fpr_test, tpr_test, auc_test = self.calculate_roc_curve(
                y_true_binary_test, lr_scores_test, weights_binary_test
            )

            # Create TGraph with transformed axes
            n_points = len(fpr_test)
            roc_graph_test = ROOT.TGraph(n_points)
            for i in range(n_points):
                roc_graph_test.SetPoint(i, tpr_test[i], 1 - fpr_test[i])

            # Test: full color, solid line
            CMS.cmsObjectDraw(roc_graph_test, "L", LineColor=bg_color, LineWidth=2, LineStyle=ROOT.kSolid)
            all_graphs.append(roc_graph_test)

            # Add to legend
            CMS.addToLegend(legend, (roc_graph_train, f"Signal vs {bg_name} (Train): AUC = {auc_train:.4f}", "L"))
            CMS.addToLegend(legend, (roc_graph_test, f"Signal vs {bg_name} (Test): AUC = {auc_test:.4f}", "L"))

        # Add diagonal to legend
        legend.Draw()
        canvas.RedrawAxis()

        canvas.Update()

        # Save canvas
        canvas.SaveAs(output_path)

        # Clean up
        canvas.Close()
