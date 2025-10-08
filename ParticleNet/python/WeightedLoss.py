#!/usr/bin/env python
"""
Weighted loss functions for physics-aware multi-class training.

Implements weighted cross-entropy and focal loss variants that incorporate
event weights (genWeight × puWeight × prefireWeight) for proper statistical
modeling of physics processes.
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


def create_loss_function(loss_type='weighted_ce', num_classes=4, **kwargs):
    """
    Factory function to create loss functions for multi-class training.

    Args:
        loss_type: Type of loss function
                  - 'weighted_ce': WeightedCrossEntropyLoss
                  - 'sample_normalized': SampleNormalizedWeightedLoss
                  - 'focal': WeightedFocalLoss
        num_classes: Number of classes
        **kwargs: Additional arguments passed to loss function

    Returns:
        Initialized loss function
    """
    if loss_type == 'weighted_ce':
        return WeightedCrossEntropyLoss(**kwargs)
    elif loss_type == 'sample_normalized':
        return SampleNormalizedWeightedLoss(num_classes=num_classes, **kwargs)
    elif loss_type == 'focal':
        return WeightedFocalLoss(**kwargs)
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

    # Test factory function
    print("\n4. Factory function test:")
    for loss_type in ['weighted_ce', 'sample_normalized', 'focal']:
        loss_fn = create_loss_function(loss_type, num_classes=num_classes)
        loss = loss_fn(predictions, targets, weights)
        print(f"   {loss_type}: {loss.item():.4f}")

    print("\nAll tests completed successfully!")