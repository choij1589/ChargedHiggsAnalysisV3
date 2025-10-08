#!/usr/bin/env python
"""
Comparison visualization script for ParticleNet binary vs multi-class methodologies.

This script generates comparison plots to analyze the performance differences
between binary classification and multi-class classification approaches:
- Performance comparison across signal points
- Background separation effectiveness
- Training efficiency comparison
- Methodology recommendation analysis

Usage:
    python visualizeComparison.py --channel Run1E2Mu --fold 3
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path

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

def find_all_results(channel, fold):
    """Find all available training results for comparison."""
    base_path = Path("/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/ParticleNet/results")

    signal_points = ["MHc160_MA85", "MHc130_MA90", "MHc100_MA95"]
    backgrounds = ["nonprompt", "diboson", "ttZ"]

    binary_results = {}
    multiclass_results = {}

    # Find binary results
    binary_path = base_path / "binary" / channel
    for signal_dir in binary_path.glob("TTToHcToWAToMuMu-*"):
        signal = signal_dir.name.replace("TTToHcToWAToMuMu-", "")
        if signal in signal_points:
            binary_results[signal] = {}
            pilot_dir = signal_dir / "pilot"

            for bg in backgrounds:
                pattern = f"ParticleNet-nNodes128-Adam-initLR0p0010-decay0p00010-StepLR-weighted_ce-binary-{bg}_performance.json"
                perf_file = pilot_dir / pattern
                if perf_file.exists():
                    binary_results[signal][bg] = str(perf_file)

    # Find multiclass results
    multiclass_path = base_path / "multiclass" / channel
    for signal_dir in multiclass_path.glob("TTToHcToWAToMuMu-*"):
        signal = signal_dir.name.replace("TTToHcToWAToMuMu-", "")
        if signal in signal_points:
            pilot_dir = signal_dir / "pilot"
            pattern = f"ParticleNet-nNodes128-Adam-initLR0p0010-decay0p00010-StepLR-weighted_ce-3bg_performance.json"
            perf_file = pilot_dir / pattern
            if perf_file.exists():
                multiclass_results[signal] = str(perf_file)

    return binary_results, multiclass_results, signal_points, backgrounds

def load_performance_data(results_dict, is_binary=True):
    """Load performance data from JSON files."""
    performance_data = {}

    if is_binary:
        for signal, bg_dict in results_dict.items():
            performance_data[signal] = {}
            for bg, file_path in bg_dict.items():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                performance_data[signal][bg] = data['training_results']
    else:
        for signal, file_path in results_dict.items():
            with open(file_path, 'r') as f:
                data = json.load(f)
            performance_data[signal] = data['training_results']

    return performance_data

def plot_performance_comparison(binary_data, multiclass_data, signal_points, backgrounds, output_path):
    """Plot performance comparison between binary and multi-class approaches."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    # Prepare data for plotting
    signal_labels = [sp.replace('_', ' ') for sp in signal_points]

    # 1. Test Accuracy Comparison
    binary_accuracies = []
    multiclass_accuracies = []

    for signal in signal_points:
        # Binary: average accuracy across all backgrounds
        if signal in binary_data:
            bg_accs = [binary_data[signal][bg]['test_accuracy'] for bg in backgrounds if bg in binary_data[signal]]
            binary_accuracies.append(np.mean(bg_accs) if bg_accs else 0)
        else:
            binary_accuracies.append(0)

        # Multi-class: direct accuracy
        if signal in multiclass_data:
            multiclass_accuracies.append(multiclass_data[signal]['test_accuracy'])
        else:
            multiclass_accuracies.append(0)

    x = np.arange(len(signal_labels))
    width = 0.35

    ax1.bar(x - width/2, binary_accuracies, width, label='Binary (Average)', color='skyblue', alpha=0.8)
    ax1.bar(x + width/2, multiclass_accuracies, width, label='Multi-class', color='orange', alpha=0.8)
    ax1.set_xlabel('Signal Point')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(signal_labels, rotation=45)
    ax1.legend()
    ax1.set_ylim([0, 1.1])

    # Add value labels
    for i, (binary_acc, multi_acc) in enumerate(zip(binary_accuracies, multiclass_accuracies)):
        ax1.text(i - width/2, binary_acc + 0.01, f'{binary_acc:.3f}', ha='center', va='bottom', fontsize=10)
        ax1.text(i + width/2, multi_acc + 0.01, f'{multi_acc:.3f}', ha='center', va='bottom', fontsize=10)

    # 2. Training Time Comparison
    binary_times = []
    multiclass_times = []

    for signal in signal_points:
        # Binary: sum of training times for all backgrounds
        if signal in binary_data:
            bg_times = [binary_data[signal][bg]['total_training_time'] for bg in backgrounds if bg in binary_data[signal]]
            binary_times.append(sum(bg_times) if bg_times else 0)
        else:
            binary_times.append(0)

        # Multi-class: direct training time
        if signal in multiclass_data:
            multiclass_times.append(multiclass_data[signal]['total_training_time'])
        else:
            multiclass_times.append(0)

    ax2.bar(x - width/2, np.array(binary_times) / 60, width, label='Binary (Total)', color='lightcoral', alpha=0.8)
    ax2.bar(x + width/2, np.array(multiclass_times) / 60, width, label='Multi-class', color='lightgreen', alpha=0.8)
    ax2.set_xlabel('Signal Point')
    ax2.set_ylabel('Training Time (minutes)')
    ax2.set_title('Training Time Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(signal_labels, rotation=45)
    ax2.legend()

    # Add value labels
    for i, (binary_time, multi_time) in enumerate(zip(binary_times, multiclass_times)):
        ax2.text(i - width/2, binary_time/60 + 0.5, f'{binary_time/60:.1f}', ha='center', va='bottom', fontsize=10)
        ax2.text(i + width/2, multi_time/60 + 0.5, f'{multi_time/60:.1f}', ha='center', va='bottom', fontsize=10)

    # 3. Model Parameters Comparison
    binary_params = []
    multiclass_params = []

    for signal in signal_points:
        # Binary: parameters from first available model (should be same for all)
        if signal in binary_data:
            bg_params = [binary_data[signal][bg]['model_parameters'] for bg in backgrounds if bg in binary_data[signal]]
            binary_params.append(bg_params[0] if bg_params else 0)
        else:
            binary_params.append(0)

        # Multi-class: direct parameters
        if signal in multiclass_data:
            multiclass_params.append(multiclass_data[signal]['model_parameters'])
        else:
            multiclass_params.append(0)

    ax3.bar(x - width/2, np.array(binary_params) / 1000, width, label='Binary (per model)', color='gold', alpha=0.8)
    ax3.bar(x + width/2, np.array(multiclass_params) / 1000, width, label='Multi-class', color='purple', alpha=0.8)
    ax3.set_xlabel('Signal Point')
    ax3.set_ylabel('Model Parameters (thousands)')
    ax3.set_title('Model Complexity Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(signal_labels, rotation=45)
    ax3.legend()

    # 4. Convergence Speed (Epochs to convergence)
    binary_epochs = []
    multiclass_epochs = []

    for signal in signal_points:
        # Binary: average epochs across backgrounds
        if signal in binary_data:
            bg_epochs = [binary_data[signal][bg]['epochs_completed'] for bg in backgrounds if bg in binary_data[signal]]
            binary_epochs.append(np.mean(bg_epochs) if bg_epochs else 0)
        else:
            binary_epochs.append(0)

        # Multi-class: direct epochs
        if signal in multiclass_data:
            multiclass_epochs.append(multiclass_data[signal]['epochs_completed'])
        else:
            multiclass_epochs.append(0)

    ax4.bar(x - width/2, binary_epochs, width, label='Binary (Average)', color='cyan', alpha=0.8)
    ax4.bar(x + width/2, multiclass_epochs, width, label='Multi-class', color='magenta', alpha=0.8)
    ax4.set_xlabel('Signal Point')
    ax4.set_ylabel('Epochs to Convergence')
    ax4.set_title('Convergence Speed Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(signal_labels, rotation=45)
    ax4.legend()

    plt.suptitle('Binary vs Multi-class Classification Performance Comparison', fontsize=18)
    plt.tight_layout()

    output_file = os.path.join(output_path, 'binary_vs_multiclass_performance.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Performance comparison saved to: {output_file}")

def plot_signal_point_analysis(binary_data, multiclass_data, signal_points, backgrounds, output_path):
    """Plot performance trends across different signal points."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Extract signal masses for x-axis
    mhc_values = [int(sp.split('_')[0].replace('MHc', '')) for sp in signal_points]
    ma_values = [int(sp.split('_')[1].replace('MA', '')) for sp in signal_points]

    # 1. Accuracy vs Signal Mass
    binary_accs = []
    multiclass_accs = []

    for signal in signal_points:
        # Binary: average accuracy
        if signal in binary_data:
            bg_accs = [binary_data[signal][bg]['test_accuracy'] for bg in backgrounds if bg in binary_data[signal]]
            binary_accs.append(np.mean(bg_accs) if bg_accs else 0)
        else:
            binary_accs.append(0)

        # Multi-class: direct accuracy
        if signal in multiclass_data:
            multiclass_accs.append(multiclass_data[signal]['test_accuracy'])
        else:
            multiclass_accs.append(0)

    # Plot against MHc (charged Higgs mass)
    ax1.plot(mhc_values, binary_accs, 'o-', label='Binary Classification', linewidth=2, markersize=8, color='blue')
    ax1.plot(mhc_values, multiclass_accs, 's-', label='Multi-class Classification', linewidth=2, markersize=8, color='red')
    ax1.set_xlabel('Charged Higgs Mass (GeV)')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Performance vs Charged Higgs Mass')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add point labels
    for i, (mhc, binary_acc, multi_acc) in enumerate(zip(mhc_values, binary_accs, multiclass_accs)):
        ax1.annotate(f'{signal_points[i].replace("_", "\\n")}',
                    xy=(mhc, binary_acc), xytext=(5, 5),
                    textcoords='offset points', fontsize=9, ha='left')

    # 2. Training Efficiency (Accuracy per training time)
    binary_efficiency = []
    multiclass_efficiency = []

    for i, signal in enumerate(signal_points):
        # Binary: accuracy / total training time
        if signal in binary_data:
            bg_times = [binary_data[signal][bg]['total_training_time'] for bg in backgrounds if bg in binary_data[signal]]
            total_time = sum(bg_times) if bg_times else 1
            binary_efficiency.append(binary_accs[i] / (total_time / 3600))  # per hour
        else:
            binary_efficiency.append(0)

        # Multi-class: accuracy / training time
        if signal in multiclass_data:
            time_hours = multiclass_data[signal]['total_training_time'] / 3600
            multiclass_efficiency.append(multiclass_accs[i] / time_hours)
        else:
            multiclass_efficiency.append(0)

    ax2.plot(mhc_values, binary_efficiency, 'o-', label='Binary Classification', linewidth=2, markersize=8, color='green')
    ax2.plot(mhc_values, multiclass_efficiency, 's-', label='Multi-class Classification', linewidth=2, markersize=8, color='orange')
    ax2.set_xlabel('Charged Higgs Mass (GeV)')
    ax2.set_ylabel('Training Efficiency (Accuracy / Hour)')
    ax2.set_title('Training Efficiency vs Charged Higgs Mass')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file = os.path.join(output_path, 'signal_point_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Signal point analysis saved to: {output_file}")

def plot_background_separation_analysis(binary_data, signal_points, backgrounds, output_path):
    """Plot background separation effectiveness for binary classifiers."""
    # Create a heatmap showing accuracy for each signal-background combination
    accuracy_matrix = np.zeros((len(signal_points), len(backgrounds)))

    for i, signal in enumerate(signal_points):
        for j, bg in enumerate(backgrounds):
            if signal in binary_data and bg in binary_data[signal]:
                accuracy_matrix[i, j] = binary_data[signal][bg]['test_accuracy']

    plt.figure(figsize=(10, 8))

    # Create heatmap
    sns.heatmap(accuracy_matrix,
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r',
                xticklabels=[bg.replace('_', ' ').title() for bg in backgrounds],
                yticklabels=[sp.replace('_', ' ') for sp in signal_points],
                cbar_kws={'label': 'Test Accuracy'})

    plt.xlabel('Background Category')
    plt.ylabel('Signal Point')
    plt.title('Binary Classification Accuracy Matrix\n(Signal vs Background Separation)')
    plt.tight_layout()

    output_file = os.path.join(output_path, 'background_separation_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Background separation analysis saved to: {output_file}")

def plot_training_efficiency_comparison(binary_data, multiclass_data, signal_points, backgrounds, output_path):
    """Plot detailed training efficiency comparison."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    signal_labels = [sp.replace('_', ' ') for sp in signal_points]

    # 1. Memory Usage Comparison
    binary_memory = []
    multiclass_memory = []

    for signal in signal_points:
        if signal in binary_data:
            bg_memory = [binary_data[signal][bg]['final_memory_mb'] for bg in backgrounds if bg in binary_data[signal]]
            binary_memory.append(np.mean(bg_memory) if bg_memory else 0)
        else:
            binary_memory.append(0)

        if signal in multiclass_data:
            multiclass_memory.append(multiclass_data[signal]['final_memory_mb'])
        else:
            multiclass_memory.append(0)

    x = np.arange(len(signal_labels))
    width = 0.35

    ax1.bar(x - width/2, binary_memory, width, label='Binary (Average)', color='lightblue', alpha=0.8)
    ax1.bar(x + width/2, multiclass_memory, width, label='Multi-class', color='lightcoral', alpha=0.8)
    ax1.set_xlabel('Signal Point')
    ax1.set_ylabel('Memory Usage (MB)')
    ax1.set_title('Memory Usage Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(signal_labels, rotation=45)
    ax1.legend()

    # 2. Average Epoch Time
    binary_epoch_time = []
    multiclass_epoch_time = []

    for signal in signal_points:
        if signal in binary_data:
            bg_times = [binary_data[signal][bg]['avg_epoch_time'] for bg in backgrounds if bg in binary_data[signal]]
            binary_epoch_time.append(np.mean(bg_times) if bg_times else 0)
        else:
            binary_epoch_time.append(0)

        if signal in multiclass_data:
            multiclass_epoch_time.append(multiclass_data[signal]['avg_epoch_time'])
        else:
            multiclass_epoch_time.append(0)

    ax2.bar(x - width/2, binary_epoch_time, width, label='Binary (Average)', color='gold', alpha=0.8)
    ax2.bar(x + width/2, multiclass_epoch_time, width, label='Multi-class', color='purple', alpha=0.8)
    ax2.set_xlabel('Signal Point')
    ax2.set_ylabel('Average Epoch Time (seconds)')
    ax2.set_title('Training Speed Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(signal_labels, rotation=45)
    ax2.legend()

    # 3. Best Validation Loss
    binary_val_loss = []
    multiclass_val_loss = []

    for signal in signal_points:
        if signal in binary_data:
            bg_losses = [binary_data[signal][bg]['best_valid_loss'] for bg in backgrounds if bg in binary_data[signal]]
            binary_val_loss.append(np.mean(bg_losses) if bg_losses else 0)
        else:
            binary_val_loss.append(0)

        if signal in multiclass_data:
            multiclass_val_loss.append(multiclass_data[signal]['best_valid_loss'])
        else:
            multiclass_val_loss.append(0)

    ax3.bar(x - width/2, binary_val_loss, width, label='Binary (Average)', color='green', alpha=0.8)
    ax3.bar(x + width/2, multiclass_val_loss, width, label='Multi-class', color='orange', alpha=0.8)
    ax3.set_xlabel('Signal Point')
    ax3.set_ylabel('Best Validation Loss')
    ax3.set_title('Final Model Quality Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(signal_labels, rotation=45)
    ax3.legend()

    # 4. Resource Efficiency (Accuracy per memory per time)
    binary_efficiency = []
    multiclass_efficiency = []

    for i, signal in enumerate(signal_points):
        # Calculate efficiency metric: accuracy / (memory_gb * time_hours)
        if signal in binary_data:
            bg_accs = [binary_data[signal][bg]['test_accuracy'] for bg in backgrounds if bg in binary_data[signal]]
            avg_acc = np.mean(bg_accs) if bg_accs else 0
            memory_gb = binary_memory[i] / 1024
            time_hours = sum([binary_data[signal][bg]['total_training_time'] for bg in backgrounds if bg in binary_data[signal]]) / 3600
            efficiency = avg_acc / (memory_gb * time_hours) if (memory_gb * time_hours) > 0 else 0
            binary_efficiency.append(efficiency)
        else:
            binary_efficiency.append(0)

        if signal in multiclass_data:
            acc = multiclass_data[signal]['test_accuracy']
            memory_gb = multiclass_memory[i] / 1024
            time_hours = multiclass_data[signal]['total_training_time'] / 3600
            efficiency = acc / (memory_gb * time_hours) if (memory_gb * time_hours) > 0 else 0
            multiclass_efficiency.append(efficiency)
        else:
            multiclass_efficiency.append(0)

    ax4.bar(x - width/2, binary_efficiency, width, label='Binary', color='cyan', alpha=0.8)
    ax4.bar(x + width/2, multiclass_efficiency, width, label='Multi-class', color='magenta', alpha=0.8)
    ax4.set_xlabel('Signal Point')
    ax4.set_ylabel('Resource Efficiency (Acc / GB / Hour)')
    ax4.set_title('Overall Resource Efficiency')
    ax4.set_xticks(x)
    ax4.set_xticklabels(signal_labels, rotation=45)
    ax4.legend()

    plt.suptitle('Training Efficiency Analysis', fontsize=18)
    plt.tight_layout()

    output_file = os.path.join(output_path, 'training_efficiency_comparison.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    logging.info(f"Training efficiency comparison saved to: {output_file}")

def generate_comparison_summary(binary_data, multiclass_data, signal_points, backgrounds, output_path):
    """Generate a comprehensive comparison summary report."""
    summary = {
        'Methodology Comparison Summary': {
            'Signal Points Analyzed': signal_points,
            'Backgrounds Analyzed': backgrounds,
            'Total Binary Models': sum(len(bg_dict) for bg_dict in binary_data.values()),
            'Total Multi-class Models': len(multiclass_data)
        },
        'Performance Analysis': {},
        'Efficiency Analysis': {},
        'Resource Usage': {},
        'Recommendations': {}
    }

    # Performance analysis
    for signal in signal_points:
        summary['Performance Analysis'][signal] = {}

        # Binary performance
        if signal in binary_data:
            bg_accs = [binary_data[signal][bg]['test_accuracy'] for bg in backgrounds if bg in binary_data[signal]]
            summary['Performance Analysis'][signal]['Binary_Average_Accuracy'] = np.mean(bg_accs) if bg_accs else 0
            summary['Performance Analysis'][signal]['Binary_Best_Accuracy'] = max(bg_accs) if bg_accs else 0
            summary['Performance Analysis'][signal]['Binary_Worst_Accuracy'] = min(bg_accs) if bg_accs else 0

        # Multi-class performance
        if signal in multiclass_data:
            summary['Performance Analysis'][signal]['Multiclass_Accuracy'] = multiclass_data[signal]['test_accuracy']

    # Overall recommendations
    binary_avg_acc = np.mean([np.mean([binary_data[s][bg]['test_accuracy'] for bg in backgrounds if bg in binary_data[s]])
                             for s in signal_points if s in binary_data])
    multiclass_avg_acc = np.mean([multiclass_data[s]['test_accuracy'] for s in signal_points if s in multiclass_data])

    binary_total_time = sum([sum([binary_data[s][bg]['total_training_time'] for bg in backgrounds if bg in binary_data[s]])
                            for s in signal_points if s in binary_data])
    multiclass_total_time = sum([multiclass_data[s]['total_training_time'] for s in signal_points if s in multiclass_data])

    summary['Recommendations'] = {
        'Best_Performance_Method': 'Multi-class' if multiclass_avg_acc > binary_avg_acc else 'Binary',
        'Most_Efficient_Method': 'Multi-class' if multiclass_total_time < binary_total_time else 'Binary',
        'Performance_Difference': abs(multiclass_avg_acc - binary_avg_acc),
        'Time_Savings_Ratio': binary_total_time / multiclass_total_time if multiclass_total_time > 0 else 0,
        'Summary': f"Multi-class shows {'better' if multiclass_avg_acc > binary_avg_acc else 'worse'} average performance "
                  f"with {binary_total_time/multiclass_total_time:.1f}x {'less' if multiclass_total_time < binary_total_time else 'more'} training time"
    }

    # Save summary
    with open(os.path.join(output_path, 'comparison_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # Save readable text report
    with open(os.path.join(output_path, 'comparison_summary.txt'), 'w') as f:
        f.write("ParticleNet Methodology Comparison Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write("Performance Comparison:\n")
        f.write(f"  Binary Classification (Average): {binary_avg_acc:.4f}\n")
        f.write(f"  Multi-class Classification:     {multiclass_avg_acc:.4f}\n")
        f.write(f"  Performance Difference:          {abs(multiclass_avg_acc - binary_avg_acc):.4f}\n\n")

        f.write("Training Time Comparison:\n")
        f.write(f"  Binary Classification (Total):   {binary_total_time/3600:.2f} hours\n")
        f.write(f"  Multi-class Classification:      {multiclass_total_time/3600:.2f} hours\n")
        f.write(f"  Time Savings Ratio:              {binary_total_time/multiclass_total_time:.1f}x\n\n")

        f.write("Recommendations:\n")
        f.write(f"  Best Performance Method:         {summary['Recommendations']['Best_Performance_Method']}\n")
        f.write(f"  Most Efficient Method:           {summary['Recommendations']['Most_Efficient_Method']}\n")
        f.write(f"  Overall Assessment:              {summary['Recommendations']['Summary']}\n")

    logging.info(f"Comparison summary saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Compare binary vs multi-class ParticleNet methodologies')
    parser.add_argument('--channel', default='Run1E2Mu',
                        help='Analysis channel (default: Run1E2Mu)')
    parser.add_argument('--fold', type=int, default=3,
                        help='Cross-validation fold (default: 3)')
    parser.add_argument('--output',
                        help='Output directory (default: auto-generated)')

    args = parser.parse_args()

    # Setup matplotlib
    setup_matplotlib()

    # Create output directory
    if args.output is None:
        base_output = "/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/ParticleNet/plots/comparison"
        args.output = base_output

    os.makedirs(args.output, exist_ok=True)

    try:
        # Find all available results
        logging.info("Searching for training results...")
        binary_results, multiclass_results, signal_points, backgrounds = find_all_results(args.channel, args.fold)

        logging.info(f"Found binary results for {len(binary_results)} signal points")
        logging.info(f"Found multiclass results for {len(multiclass_results)} signal points")

        # Load performance data
        logging.info("Loading performance data...")
        binary_data = load_performance_data(binary_results, is_binary=True)
        multiclass_data = load_performance_data(multiclass_results, is_binary=False)

        # Generate comparison plots
        logging.info("Generating performance comparison...")
        plot_performance_comparison(binary_data, multiclass_data, signal_points, backgrounds, args.output)

        logging.info("Generating signal point analysis...")
        plot_signal_point_analysis(binary_data, multiclass_data, signal_points, backgrounds, args.output)

        logging.info("Generating background separation analysis...")
        plot_background_separation_analysis(binary_data, signal_points, backgrounds, args.output)

        logging.info("Generating training efficiency comparison...")
        plot_training_efficiency_comparison(binary_data, multiclass_data, signal_points, backgrounds, args.output)

        # Generate summary report
        logging.info("Generating comparison summary...")
        generate_comparison_summary(binary_data, multiclass_data, signal_points, backgrounds, args.output)

        logging.info(f"All comparison plots and reports saved to: {args.output}")

    except Exception as e:
        logging.error(f"Error during comparison analysis: {e}")
        raise

if __name__ == "__main__":
    main()