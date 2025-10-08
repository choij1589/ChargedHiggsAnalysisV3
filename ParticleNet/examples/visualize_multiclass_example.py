#!/usr/bin/env python
"""
Example of how to use the multi-class visualization tools.

This shows how to generate plots after training completes.
"""

import os
import sys

# Add python directory to path
sys.path.append(os.path.join(os.environ['WORKDIR'], 'ParticleNet/python'))
from visualizeMultiClass import *

# Example paths (adjust based on your training results)
root_file = "results/multiclass/Run1E2Mu/TTToHcToWAToMuMu-MHc130_MA100/fold-0/trees/ParticleNet-0-Adam-StepLR-weighted_ce.root"
csv_file = "results/multiclass/Run1E2Mu/TTToHcToWAToMuMu-MHc130_MA100/fold-0/CSV/ParticleNet-0-Adam-StepLR-weighted_ce.csv"
output_dir = "results/multiclass/Run1E2Mu/TTToHcToWAToMuMu-MHc130_MA100/fold-0/plots"

# Create output directory
os.makedirs(output_dir, exist_ok=True)

# Load data
print("Loading ROOT data...")
data = load_root_data(root_file)

# Generate individual plots
for subset in ['test']:  # Can also do 'train' and 'valid'
    mask = data['masks'][subset]
    
    print(f"Generating {subset} set plots...")
    
    # 1. Multi-class ROC curves
    print("  - ROC curves")
    auc_values = plot_multiclass_roc(data['scores'], data['labels'], mask, subset, output_dir)
    print(f"    AUC values: {auc_values}")
    
    # 2. Confusion matrix
    print("  - Confusion matrix")
    cm, report = plot_confusion_matrix(data['scores'], data['labels'], mask, subset, output_dir)
    
    # 3. Score distributions
    print("  - Score distributions")
    plot_score_distributions(data['scores'], data['labels'], mask, subset, output_dir)

# 4. Training history
print("Plotting training history...")
plot_training_history(csv_file, output_dir)

# 5. Summary plot
print("Creating summary plot...")
create_summary_plot(data, output_dir)

print(f"\nAll plots saved to: {output_dir}")

# Print some quick statistics
test_mask = data['masks']['test']
test_scores = data['scores'][test_mask]
test_labels = data['labels'][test_mask]
test_predictions = np.argmax(test_scores, axis=1)

overall_acc = np.mean(test_predictions == test_labels)
print(f"\nTest set overall accuracy: {overall_acc:.3f}")

# Per-class accuracy
for i, class_name in enumerate(CLASS_NAMES):
    class_mask = test_labels == i
    if np.sum(class_mask) > 0:
        class_acc = np.mean(test_predictions[class_mask] == i)
        print(f"{class_name} accuracy: {class_acc:.3f} (n={np.sum(class_mask)})")

# You can also use the shell script for convenience:
# ./scripts/visualizeResults.sh Run1E2Mu TTToHcToWAToMuMu-MHc130_MA100 0
