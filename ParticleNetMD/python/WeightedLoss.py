#!/usr/bin/env python
"""
Weighted loss functions for physics-aware multi-class training.

Implements weighted cross-entropy, focal loss, and DisCo (Distance Correlation)
loss variants that incorporate event weights for proper statistical modeling.

Mass-Decorrelated ParticleNet adds DisCo regularization to decorrelate
classifier output from di-muon mass.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

class WeightedCrossEntropyLoss(nn.Module):
    """
    Cross-entropy loss weighted by per-event physics weights.

    Uses the weight stored in each Data object (weight = genWeight*puWeight*prefireWeight)
    to properly weight the contribution of each event to the loss function.
    """

    def __init__(self, reduction='mean'):
        """
        Initialize weighted cross-entropy loss.

        Args:
            reduction: Specifies reduction to apply to output: 'none' | 'mean' | 'sum'
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        self.reduction = reduction

    def forward(self, input, target, weight):
        """
        Forward pass of weighted cross-entropy loss.

        Args:
            input: Predictions (N, C) where N = batch size, C = number of classes
            target: True labels (N,) with class indices
            weight: Per-event weights (N,) from Data.weight

        Returns:
            Weighted cross-entropy loss
        """
        # Compute standard cross-entropy loss per sample (no reduction)
        ce_loss = F.cross_entropy(input, target, reduction='none')

        # Apply event weights
        weighted_loss = ce_loss * weight

        # Apply reduction
        if self.reduction == 'none':
            return weighted_loss
        elif self.reduction == 'mean':
            # Weighted average: sum of weighted losses / sum of weights
            return weighted_loss.sum() / weight.sum().clamp(min=1e-8)
        elif self.reduction == 'sum':
            return weighted_loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class SampleNormalizedWeightedLoss(nn.Module):
    """
    Weighted cross-entropy with per-sample class normalization.

    Ensures each sample class (signal, TTLL_powheg, WZTo3LNu, TTZToLLNuNu)
    contributes equally to the loss by normalizing weights within each class.
    This is applied on top of the weight normalization in DynamicDatasetLoader.
    """

    def __init__(self, num_classes=4, reduction='mean'):
        """
        Initialize sample-normalized weighted loss.

        Args:
            num_classes: Number of classes (4 for signal + 3 backgrounds)
            reduction: Specifies reduction to apply to output
        """
        super(SampleNormalizedWeightedLoss, self).__init__()
        self.num_classes = num_classes
        self.reduction = reduction
        self.base_loss = WeightedCrossEntropyLoss(reduction='none')

    def forward(self, input, target, weight):
        """
        Forward pass with per-class weight normalization.

        Args:
            input: Predictions (N, C)
            target: True labels (N,)
            weight: Per-event weights (N,)

        Returns:
            Sample-normalized weighted loss
        """
        # Compute base weighted loss (no reduction)
        weighted_loss = self.base_loss(input, target, weight)

        if self.reduction == 'none':
            return weighted_loss

        # Compute per-class normalization
        total_loss = 0.0
        total_weight = 0.0

        for class_idx in range(self.num_classes):
            # Get mask for this class
            class_mask = (target == class_idx)

            if class_mask.sum() > 0:  # Only process if class has samples
                class_loss = weighted_loss[class_mask]
                class_weight = weight[class_mask]

                # Compute weighted average for this class
                class_avg_loss = class_loss.sum() / class_weight.sum().clamp(min=1e-8)

                # Weight by total class weight for final averaging
                class_total_weight = class_weight.sum()
                total_loss += class_avg_loss * class_total_weight
                total_weight += class_total_weight

        if self.reduction == 'mean':
            return total_loss / total_weight.clamp(min=1e-8)
        elif self.reduction == 'sum':
            return total_loss
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


class WeightedFocalLoss(nn.Module):
    """
    Focal loss with event weighting for handling class imbalance.

    Focal loss helps focus training on hard examples by down-weighting
    easy examples. Combined with event weights for physics-aware training.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)
    """

    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Initialize weighted focal loss.

        Args:
            alpha: Weighting factor for rare class (scalar or tensor of size C)
            gamma: Focusing parameter (higher gamma -> more focus on hard examples)
            reduction: Specifies reduction to apply to output
        """
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target, weight):
        """
        Forward pass of weighted focal loss.

        Args:
            input: Predictions (N, C)
            target: True labels (N,)
            weight: Per-event weights (N,)

        Returns:
            Weighted focal loss
        """
        # Compute cross-entropy loss
        ce_loss = F.cross_entropy(input, target, reduction='none')

        # Compute p_t (probability of true class)
        p = torch.softmax(input, dim=1)
        p_t = p.gather(1, target.unsqueeze(1)).squeeze(1)

        # Apply focal weight: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting if specified
        if self.alpha is not None:
            if isinstance(self.alpha, (float, int)):
                alpha_t = self.alpha
            else:
                alpha_t = self.alpha.gather(0, target)
            focal_weight = alpha_t * focal_weight

        # Combine focal weighting with event weights
        focal_loss = focal_weight * ce_loss * weight

        # Apply reduction
        if self.reduction == 'none':
            return focal_loss
        elif self.reduction == 'mean':
            return focal_loss.sum() / weight.sum().clamp(min=1e-8)
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            raise ValueError(f"Invalid reduction mode: {self.reduction}")


def distance_correlation(X, Y, weights=None):
    """
    Compute weighted distance correlation between X and Y.

    Distance correlation measures statistical dependence between random variables.
    DisCo(X,Y) = 0 means X and Y are independent.
    DisCo(X,Y) = 1 means X and Y are fully dependent.

    Args:
        X: Tensor (N,) - e.g., classifier scores
        Y: Tensor (N,) - e.g., mass values
        weights: Tensor (N,) - event weights (optional)

    Returns:
        Distance correlation value (scalar tensor)
    """
    n = X.shape[0]
    if n < 2:
        return torch.tensor(0.0, device=X.device)

    if weights is None:
        weights = torch.ones(n, device=X.device)

    # Normalize weights to sum to 1
    weights = weights / weights.sum().clamp(min=1e-8)

    # Compute pairwise distance matrices: a_ij = |X_i - X_j|
    a = torch.abs(X.unsqueeze(0) - X.unsqueeze(1))
    b = torch.abs(Y.unsqueeze(0) - Y.unsqueeze(1))

    # Weighted outer product for double-centering
    w_outer = weights.unsqueeze(0) * weights.unsqueeze(1)

    # Double-center distance matrices (weighted version)
    # A_ij = a_ij - E[a_i.] - E[a_.j] + E[a_..]
    a_row = (a * weights.unsqueeze(0)).sum(dim=1, keepdim=True)
    a_col = (a * weights.unsqueeze(1)).sum(dim=0, keepdim=True)
    a_grand = (a * w_outer).sum()
    A = a - a_row - a_col + a_grand

    b_row = (b * weights.unsqueeze(0)).sum(dim=1, keepdim=True)
    b_col = (b * weights.unsqueeze(1)).sum(dim=0, keepdim=True)
    b_grand = (b * w_outer).sum()
    B = b - b_row - b_col + b_grand

    # Distance covariance squared: dCov^2(X,Y) = E[A * B]
    dcov_sq = (A * B * w_outer).sum()

    # Distance variances squared
    dvar_X_sq = (A * A * w_outer).sum()
    dvar_Y_sq = (B * B * w_outer).sum()

    # Distance correlation
    if dvar_X_sq > 1e-10 and dvar_Y_sq > 1e-10:
        # Use abs() to handle numerical issues with negative dcov_sq
        dcor = torch.sqrt(torch.abs(dcov_sq) / torch.sqrt(dvar_X_sq * dvar_Y_sq))
    else:
        dcor = torch.tensor(0.0, device=X.device)

    return dcor


class DiScoWeightedLoss(nn.Module):
    """
    Weighted cross-entropy with Distance Correlation (DisCo) regularization.

    L_total = L_CE + disco_lambda * (DisCo(score, mass1) + DisCo(score, mass2))

    This encourages the classifier to be independent of the di-muon mass,
    preventing mass sculpting in background estimation.

    Reference: Kasieczka et al. "DisCo: Learning-based classification
    discriminatively constrained on decorrelation" (2020)
    """

    def __init__(self, disco_lambda=0.1, reduction='mean'):
        """
        Initialize DisCo weighted loss.

        Args:
            disco_lambda: Weight for DisCo regularization term
            reduction: Specifies reduction to apply to output
        """
        super(DiScoWeightedLoss, self).__init__()
        self.disco_lambda = disco_lambda
        self.reduction = reduction
        self.base_loss = WeightedCrossEntropyLoss(reduction='none')

        # Store decomposed losses for tracking
        self.last_ce_loss = 0.0
        self.last_disco_term = 0.0
        self.last_disco1 = 0.0
        self.last_disco2 = 0.0

    def forward(self, logits, target, weight, mass1, mass2):
        """
        Forward pass with DisCo regularization.

        Args:
            logits: Model output (N, C)
            target: True labels (N,)
            weight: Per-event weights (N,)
            mass1: First OS muon pair mass (N,)
            mass2: Second OS muon pair mass (N,), -1 for 1e2mu events

        Returns:
            Combined loss value (CE + DisCo regularization)
        """
        # Classification loss
        ce_loss = self.base_loss(logits, target, weight)

        # Get signal score for decorrelation
        probs = F.softmax(logits, dim=1)
        scores = probs[:, 0]  # Signal class probability

        # DisCo penalty for mass1 (always valid)
        valid1 = mass1 > 0
        if valid1.sum() > 1:
            disco1 = distance_correlation(
                scores[valid1],
                mass1[valid1],
                weight[valid1]
            )
        else:
            disco1 = torch.tensor(0.0, device=logits.device)

        # DisCo penalty for mass2 (only for 3mu events)
        valid2 = mass2 > 0
        if valid2.sum() > 1:
            disco2 = distance_correlation(
                scores[valid2],
                mass2[valid2],
                weight[valid2]
            )
        else:
            disco2 = torch.tensor(0.0, device=logits.device)

        # Combined DisCo term
        disco_term = disco1 + disco2

        # Apply reduction
        if self.reduction == 'mean':
            ce_mean = ce_loss.sum() / weight.sum().clamp(min=1e-8)
            total_loss = ce_mean + self.disco_lambda * disco_term

            # Store decomposed losses for tracking
            self.last_ce_loss = ce_mean.item()
            self.last_disco_term = disco_term.item()
            self.last_disco1 = disco1.item() if isinstance(disco1, torch.Tensor) else disco1
            self.last_disco2 = disco2.item() if isinstance(disco2, torch.Tensor) else disco2

            return total_loss
        elif self.reduction == 'sum':
            total_loss = ce_loss.sum() + self.disco_lambda * disco_term * weight.sum()

            # Store decomposed losses
            self.last_ce_loss = ce_loss.sum().item()
            self.last_disco_term = disco_term.item()
            self.last_disco1 = disco1.item() if isinstance(disco1, torch.Tensor) else disco1
            self.last_disco2 = disco2.item() if isinstance(disco2, torch.Tensor) else disco2

            return total_loss
        else:
            # Store decomposed losses (for 'none' reduction)
            self.last_ce_loss = ce_loss.mean().item()
            self.last_disco_term = disco_term.item()
            self.last_disco1 = disco1.item() if isinstance(disco1, torch.Tensor) else disco1
            self.last_disco2 = disco2.item() if isinstance(disco2, torch.Tensor) else disco2

            return ce_loss + self.disco_lambda * disco_term

    def get_decomposed_losses(self):
        """
        Get the decomposed loss components from the last forward pass.

        Returns:
            dict: Dictionary containing:
                - 'ce_loss': Cross-entropy loss component
                - 'disco_term': Total DisCo regularization term (before lambda weighting)
                - 'disco1': DisCo term for mass1
                - 'disco2': DisCo term for mass2
                - 'disco_weighted': DisCo term after lambda weighting
        """
        return {
            'ce_loss': self.last_ce_loss,
            'disco_term': self.last_disco_term,
            'disco1': self.last_disco1,
            'disco2': self.last_disco2,
            'disco_weighted': self.last_disco_term * self.disco_lambda
        }


def create_loss_function(loss_type='weighted_ce', num_classes=4, **kwargs):
    """
    Factory function to create loss functions for multi-class training.

    Args:
        loss_type: Type of loss function
                  - 'weighted_ce': WeightedCrossEntropyLoss
                  - 'sample_normalized': SampleNormalizedWeightedLoss
                  - 'focal': WeightedFocalLoss
                  - 'disco': DiScoWeightedLoss (mass-decorrelated)
        num_classes: Number of classes
        **kwargs: Additional arguments passed to loss function
                  For 'disco': disco_lambda (default 0.1)

    Returns:
        Initialized loss function
    """
    if loss_type == 'weighted_ce':
        return WeightedCrossEntropyLoss(**kwargs)
    elif loss_type == 'sample_normalized':
        return SampleNormalizedWeightedLoss(num_classes=num_classes, **kwargs)
    elif loss_type == 'focal':
        return WeightedFocalLoss(**kwargs)
    elif loss_type == 'disco':
        return DiScoWeightedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Example usage and testing
if __name__ == "__main__":
    import torch

    # Test data
    batch_size = 10
    num_classes = 4

    # Create dummy predictions and targets
    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    weights = torch.abs(torch.randn(batch_size)) + 0.1  # Positive weights

    # Create dummy mass values for DisCo test
    mass1 = torch.abs(torch.randn(batch_size)) * 50 + 20  # 20-70 GeV range
    mass2 = torch.abs(torch.randn(batch_size)) * 50 + 20
    mass2[batch_size//2:] = -1  # Half are 1e2mu events (no mass2)

    print("Testing weighted loss functions...")
    print(f"Batch size: {batch_size}, Classes: {num_classes}")
    print(f"Targets: {targets}")
    print(f"Weights: {weights}")

    # Test WeightedCrossEntropyLoss
    print("\n1. WeightedCrossEntropyLoss:")
    loss_fn = WeightedCrossEntropyLoss()
    loss = loss_fn(predictions, targets, weights)
    print(f"   Loss: {loss.item():.4f}")

    # Compare with unweighted loss
    unweighted_loss = F.cross_entropy(predictions, targets)
    print(f"   Unweighted CE: {unweighted_loss.item():.4f}")

    # Test SampleNormalizedWeightedLoss
    print("\n2. SampleNormalizedWeightedLoss:")
    loss_fn = SampleNormalizedWeightedLoss(num_classes=num_classes)
    loss = loss_fn(predictions, targets, weights)
    print(f"   Loss: {loss.item():.4f}")

    # Test WeightedFocalLoss
    print("\n3. WeightedFocalLoss:")
    loss_fn = WeightedFocalLoss(gamma=2.0)
    loss = loss_fn(predictions, targets, weights)
    print(f"   Loss: {loss.item():.4f}")

    # Test DiScoWeightedLoss
    print("\n4. DiScoWeightedLoss:")
    loss_fn = DiScoWeightedLoss(disco_lambda=0.1)
    loss = loss_fn(predictions, targets, weights, mass1, mass2)
    print(f"   Loss: {loss.item():.4f}")

    # Test distance correlation directly
    print("\n5. Distance Correlation test:")
    scores = F.softmax(predictions, dim=1)[:, 0]
    dcor = distance_correlation(scores, mass1, weights)
    print(f"   DisCo(score, mass1): {dcor.item():.4f}")

    # Test factory function
    print("\n6. Factory function test:")
    for loss_type in ['weighted_ce', 'sample_normalized', 'focal', 'disco']:
        loss_fn = create_loss_function(loss_type, num_classes=num_classes)
        if loss_type == 'disco':
            loss = loss_fn(predictions, targets, weights, mass1, mass2)
        else:
            loss = loss_fn(predictions, targets, weights)
        print(f"   {loss_type}: {loss.item():.4f}")

    print("\nAll tests completed successfully!")