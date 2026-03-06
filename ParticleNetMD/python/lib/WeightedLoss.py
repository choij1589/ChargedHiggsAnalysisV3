#!/usr/bin/env python
"""
Weighted loss functions for physics-aware multi-class training.

Implements weighted cross-entropy and DisCo (Distance Correlation) loss variants
that incorporate event weights for proper statistical modeling.

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
            # Weighted average: sum of weighted losses / sum of |weights|
            # Use abs(weight) for normalization to handle MC negative weights properly
            # Negative weights contribute negatively to loss, but normalization uses effective sample size
            return weighted_loss.sum() / torch.abs(weight).sum().clamp(min=1e-8)
        elif self.reduction == 'sum':
            return weighted_loss.sum()
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
            disco_lambda: Weight for mass DisCo regularization term
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

        # Combined DisCo term for mass decorrelation
        disco_term = disco1 + disco2

        # Apply reduction
        if self.reduction == 'mean':
            # Use abs(weight) for normalization to handle MC negative weights
            ce_mean = ce_loss.sum() / torch.abs(weight).sum().clamp(min=1e-8)
            total_loss = ce_mean + self.disco_lambda * disco_term

            # Store decomposed losses for tracking
            self.last_ce_loss = ce_mean.item()
            self.last_disco_term = disco_term.item()
            self.last_disco1 = disco1.item() if isinstance(disco1, torch.Tensor) else disco1
            self.last_disco2 = disco2.item() if isinstance(disco2, torch.Tensor) else disco2

            return total_loss
        elif self.reduction == 'sum':
            # Use abs(weight) for proper scaling of DisCo term
            total_loss = ce_loss.sum() + self.disco_lambda * disco_term * torch.abs(weight).sum()

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
                  - 'disco': DiScoWeightedLoss (mass-decorrelated)
        num_classes: Number of classes (unused, kept for API compatibility)
        **kwargs: Additional arguments passed to loss function
                  For 'disco': disco_lambda (default 0.1)

    Returns:
        Initialized loss function
    """
    if loss_type == 'weighted_ce':
        return WeightedCrossEntropyLoss(**kwargs)
    elif loss_type == 'disco':
        return DiScoWeightedLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")