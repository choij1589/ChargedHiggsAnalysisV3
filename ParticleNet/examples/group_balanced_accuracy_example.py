#!/usr/bin/env python
"""
Example demonstrating group-balanced accuracy calculation.

This example shows the difference between standard accuracy and group-balanced accuracy
when dealing with grouped backgrounds with different sample sizes.
"""

import torch

def calculate_group_balanced_accuracy(predictions, labels, weights, use_groups, num_classes):
    """
    Calculate group-balanced accuracy.

    For grouped backgrounds:
    - Within each group, accuracy is weighted by sample weights
    - Between groups, each group contributes equally (1/num_classes)

    Args:
        predictions: predicted class labels (tensor)
        labels: true class labels (tensor)
        weights: sample weights (tensor)
        use_groups: whether using grouped backgrounds
        num_classes: total number of classes

    Returns:
        group_balanced_accuracy: float
    """
    if not use_groups:
        # Fall back to standard weighted accuracy for individual backgrounds
        correct_mask = (predictions == labels).float()
        weighted_correct = (correct_mask * weights).sum()
        total_weights = weights.sum()
        return (weighted_correct / total_weights).item() if total_weights > 0 else 0.0

    # Group-balanced accuracy calculation
    class_accuracies = []

    for class_idx in range(num_classes):
        class_mask = (labels == class_idx)
        if class_mask.sum() == 0:
            continue  # Skip classes with no samples

        class_predictions = predictions[class_mask]
        class_labels = labels[class_mask]
        class_weights = weights[class_mask]

        # Within-class weighted accuracy
        correct_mask = (class_predictions == class_labels).float()
        weighted_correct = (correct_mask * class_weights).sum()
        total_class_weight = class_weights.sum()

        if total_class_weight > 0:
            class_accuracy = (weighted_correct / total_class_weight).item()
            class_accuracies.append(class_accuracy)

    # Return mean accuracy across all classes (equal weight per group)
    return sum(class_accuracies) / len(class_accuracies) if class_accuracies else 0.0

def create_example_data():
    """Create example predictions and labels for demonstration."""

    # Example: 4 classes (signal + 3 background groups)
    # Class 0: Signal (100 samples)
    # Class 1: Nonprompt backgrounds (1000 samples) - large group
    # Class 2: Diboson backgrounds (200 samples) - medium group
    # Class 3: ttX backgrounds (50 samples) - small group

    predictions = []
    labels = []
    weights = []

    # Signal class (class 0): 90% accuracy
    signal_size = 100
    signal_correct = int(0.9 * signal_size)
    predictions.extend([0] * signal_correct + [1] * (signal_size - signal_correct))  # Most wrong predictions go to class 1
    labels.extend([0] * signal_size)
    weights.extend([2.5] * signal_size)  # Signal has higher weight

    # Nonprompt backgrounds (class 1): 85% accuracy, large group
    nonprompt_size = 1000
    nonprompt_correct = int(0.85 * nonprompt_size)
    predictions.extend([1] * nonprompt_correct + [0] * (nonprompt_size - nonprompt_correct))  # Wrong predictions mostly go to signal
    labels.extend([1] * nonprompt_size)
    weights.extend([1.0] * nonprompt_size)  # Standard weight

    # Diboson backgrounds (class 2): 75% accuracy, medium group
    diboson_size = 200
    diboson_correct = int(0.75 * diboson_size)
    predictions.extend([2] * diboson_correct + [1] * (diboson_size - diboson_correct))  # Wrong predictions go to nonprompt
    labels.extend([2] * diboson_size)
    weights.extend([1.8] * diboson_size)  # Higher weight due to lower cross-section

    # ttX backgrounds (class 3): 95% accuracy, small group
    ttx_size = 50
    ttx_correct = int(0.95 * ttx_size)
    predictions.extend([3] * ttx_correct + [2] * (ttx_size - ttx_correct))  # Wrong predictions go to diboson
    labels.extend([3] * ttx_size)
    weights.extend([8.5] * ttx_size)  # Much higher weight due to very low cross-section

    return (torch.tensor(predictions, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(weights, dtype=torch.float))

def calculate_standard_accuracy(predictions, labels, weights):
    """Calculate standard weighted accuracy for comparison."""
    correct_mask = (predictions == labels).float()
    weighted_correct = (correct_mask * weights).sum()
    total_weights = weights.sum()
    return (weighted_correct / total_weights).item() if total_weights > 0 else 0.0

def main():
    """Demonstrate group-balanced vs standard accuracy calculation."""

    print("=== Group-Balanced Accuracy Example ===\n")

    # Create example data
    predictions, labels, weights = create_example_data()

    print("Dataset composition:")
    for class_idx in range(4):
        class_mask = (labels == class_idx)
        class_count = class_mask.sum().item()
        class_weight = weights[class_mask].sum().item()
        class_predictions = predictions[class_mask]
        correct_count = (class_predictions == class_idx).sum().item()
        class_acc = correct_count / class_count if class_count > 0 else 0.0

        class_names = ["Signal", "Nonprompt", "Diboson", "ttX"]
        print(f"  {class_names[class_idx]:>9}: {class_count:4d} samples, weight: {class_weight:6.1f}, accuracy: {class_acc*100:5.1f}%")

    print(f"\nTotal samples: {len(predictions)}")
    print(f"Total weight: {weights.sum().item():.1f}")

    # Calculate different accuracy metrics
    standard_acc = calculate_standard_accuracy(predictions, labels, weights)

    # Group-balanced accuracy (individual backgrounds mode - should be same as standard)
    group_balanced_individual = calculate_group_balanced_accuracy(
        predictions, labels, weights, use_groups=False, num_classes=4
    )

    # Group-balanced accuracy (grouped backgrounds mode)
    group_balanced_grouped = calculate_group_balanced_accuracy(
        predictions, labels, weights, use_groups=True, num_classes=4
    )

    print(f"\n=== Accuracy Comparison ===")
    print(f"Standard weighted accuracy:        {standard_acc*100:6.2f}%")
    print(f"Group-balanced (individual mode):  {group_balanced_individual*100:6.2f}%")
    print(f"Group-balanced (grouped mode):     {group_balanced_grouped*100:6.2f}%")

    print(f"\n=== Explanation ===")
    print(f"Standard accuracy is dominated by the large nonprompt group (1000 samples)")
    print(f"Group-balanced accuracy treats each group equally:")
    print(f"  - Signal accuracy:     90.0% (weight 1/4 = 25%)")
    print(f"  - Nonprompt accuracy:  85.0% (weight 1/4 = 25%)")
    print(f"  - Diboson accuracy:    75.0% (weight 1/4 = 25%)")
    print(f"  - ttX accuracy:        95.0% (weight 1/4 = 25%)")
    print(f"  - Group-balanced mean: {(90.0 + 85.0 + 75.0 + 95.0)/4:.1f}%")

    print(f"\nThis prevents the large nonprompt group from dominating the accuracy metric")
    print(f"and ensures each physics process contributes equally to the evaluation.")

if __name__ == "__main__":
    main()