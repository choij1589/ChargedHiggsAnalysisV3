#!/usr/bin/env python3
"""
Summarize GA optimization loss evolution across iterations.

This script reads the training results from all GA iterations and computes
statistics on train/validation loss to track optimization progress.

For each iteration:
- Reads all model JSON files
- Computes mean train/valid loss across population
- Tracks best model performance
- Generates summary plots and JSON output

Usage:
    python summarizeGALoss.py --signal MHc130_MA100 --channel Run1E2Mu
    python summarizeGALoss.py --signal MHc130_MA100 --channel Run1E2Mu --input GAOptim_bjets_maxepoch50
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import ROOT
ROOT.gROOT.SetBatch(True)

# Try to import CMS style
try:
    import cmsstyle as CMS
    HAS_CMS_STYLE = True
except ImportError:
    print("Warning: cmsstyle not available, using default ROOT style")
    HAS_CMS_STYLE = False


def setup_cms_style():
    """Setup CMS style for ROOT plots."""
    if HAS_CMS_STYLE:
        CMS.setCMSStyle()
        CMS.SetEnergy(13)
        CMS.SetLumi(-1, run="Run2")
        CMS.SetExtraText("Simulation Preliminary")
    else:
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetPadLeftMargin(0.12)
        ROOT.gStyle.SetPadBottomMargin(0.12)
        ROOT.gStyle.SetPadRightMargin(0.05)
        ROOT.gStyle.SetPadTopMargin(0.08)
        ROOT.gStyle.SetTitleSize(0.05, "XYZ")
        ROOT.gStyle.SetLabelSize(0.04, "XYZ")


def read_model_json(json_path: str) -> Optional[Dict]:
    """Read a model JSON file and return the data."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Warning: Failed to read {json_path}: {e}")
        return None


def get_iteration_statistics(iteration_dir: Path) -> Optional[Dict]:
    """
    Compute statistics for all models in a GA iteration.

    Args:
        iteration_dir: Path to GA-iter{N} directory

    Returns:
        Dictionary with iteration statistics or None if failed
    """
    json_dir = iteration_dir / "json"

    if not json_dir.exists():
        print(f"Warning: JSON directory not found: {json_dir}")
        return None

    # Read model_info.csv to get list of models
    model_info_path = json_dir / "model_info.csv"
    if not model_info_path.exists():
        print(f"Warning: model_info.csv not found: {model_info_path}")
        return None

    # Parse model list
    model_indices = []
    with open(model_info_path, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:  # Skip header
            if line.strip():
                model_name = line.split(',')[0].strip()
                if model_name.startswith('model'):
                    idx = int(model_name.replace('model', ''))
                    model_indices.append(idx)

    if not model_indices:
        print(f"Warning: No models found in {model_info_path}")
        return None

    # Read all model JSONs
    train_losses = []
    valid_losses = []
    best_train_losses = []
    best_valid_losses = []

    for idx in model_indices:
        json_path = json_dir / f"model{idx}.json"
        if not json_path.exists():
            print(f"Warning: Model JSON not found: {json_path}")
            continue

        model_data = read_model_json(str(json_path))
        if model_data is None:
            continue

        # Extract best losses from training_summary
        summary = model_data.get('training_summary', {})
        best_train = summary.get('best_train_loss')
        best_valid = summary.get('best_valid_loss')

        if best_train is not None and best_valid is not None:
            best_train_losses.append(best_train)
            best_valid_losses.append(best_valid)

    # Check if we have any valid data
    if not best_train_losses or not best_valid_losses:
        print(f"Warning: No valid loss data found in {iteration_dir}")
        return None

    # Compute statistics
    stats = {
        'num_models': len(best_train_losses),
        'mean_train_loss': float(np.mean(best_train_losses)),
        'mean_valid_loss': float(np.mean(best_valid_losses)),
        'std_train_loss': float(np.std(best_train_losses)),
        'std_valid_loss': float(np.std(best_valid_losses)),
        'min_train_loss': float(np.min(best_train_losses)),
        'min_valid_loss': float(np.min(best_valid_losses)),
        'max_train_loss': float(np.max(best_train_losses)),
        'max_valid_loss': float(np.max(best_valid_losses)),
        'median_train_loss': float(np.median(best_train_losses)),
        'median_valid_loss': float(np.median(best_valid_losses))
    }

    return stats


def create_loss_evolution_plot(summary_data: Dict, output_path: str):
    """
    Create ROOT plot showing loss evolution across GA iterations.

    Args:
        summary_data: Dictionary with iteration statistics
        output_path: Path to save the plot
    """
    iterations_data = summary_data['iterations']
    if not iterations_data:
        print("Warning: No iteration data to plot")
        return

    n_iters = len(iterations_data)

    # Extract data for plotting
    iter_nums = []
    mean_train = []
    mean_valid = []
    std_train = []
    std_valid = []
    best_train = []
    best_valid = []

    for iter_data in iterations_data:
        iter_nums.append(iter_data['iteration'])
        mean_train.append(iter_data['mean_train_loss'])
        mean_valid.append(iter_data['mean_valid_loss'])
        std_train.append(iter_data['std_train_loss'])
        std_valid.append(iter_data['std_valid_loss'])
        best_train.append(iter_data['min_train_loss'])
        best_valid.append(iter_data['min_valid_loss'])

    # Create TGraphs for mean loss
    gr_mean_train = ROOT.TGraph(n_iters)
    gr_mean_valid = ROOT.TGraph(n_iters)

    for i in range(n_iters):
        gr_mean_train.SetPoint(i, iter_nums[i], mean_train[i])
        gr_mean_valid.SetPoint(i, iter_nums[i], mean_valid[i])

    # Create TGraphErrors for error bands
    gr_err_train = ROOT.TGraphErrors(n_iters)
    gr_err_valid = ROOT.TGraphErrors(n_iters)

    for i in range(n_iters):
        gr_err_train.SetPoint(i, iter_nums[i], mean_train[i])
        gr_err_train.SetPointError(i, 0, std_train[i])
        gr_err_valid.SetPoint(i, iter_nums[i], mean_valid[i])
        gr_err_valid.SetPointError(i, 0, std_valid[i])

    # Create TGraphs for best loss
    gr_best_train = ROOT.TGraph(n_iters)
    gr_best_valid = ROOT.TGraph(n_iters)

    for i in range(n_iters):
        gr_best_train.SetPoint(i, iter_nums[i], best_train[i])
        gr_best_valid.SetPoint(i, iter_nums[i], best_valid[i])

    # Fixed y-axis range
    y_min = 0.7
    y_max = 1.3

    # X-axis range
    x_min = min(iter_nums) - 0.2
    x_max = max(iter_nums) + 0.2

    # Create canvas with CMS styling
    canvas = CMS.cmsCanvas("c_ga_loss",
                           x_min, x_max,
                           y_min, y_max,
                           "GA Iteration",
                           "Loss",
                           square=True,
                           iPos=0,
                           extraSpace=0.01)

    # Draw error bars first (background)
    COLOR_TRAIN = ROOT.TColor.GetColor("#5790fc")
    COLOR_TEST = ROOT.TColor.GetColor("#f89c20")
    CMS.cmsObjectDraw(gr_err_train, "PE", LineColor=COLOR_TRAIN, LineWidth=2,
                      MarkerStyle=1, MarkerSize=0)  # Invisible markers, only error bars
    CMS.cmsObjectDraw(gr_err_valid, "PE", LineColor=COLOR_TEST, LineWidth=2,
                      MarkerStyle=1, MarkerSize=0)  # Invisible markers, only error bars

    # Draw mean lines
    CMS.cmsObjectDraw(gr_mean_train, "LP", LineColor=COLOR_TRAIN, LineWidth=2,
                      MarkerColor=COLOR_TRAIN, MarkerStyle=20, MarkerSize=1.2)
    CMS.cmsObjectDraw(gr_mean_valid, "LP", LineColor=COLOR_TEST, LineWidth=2,
                      MarkerColor=COLOR_TEST, MarkerStyle=20, MarkerSize=1.2)

    # Draw best points (markers only)
    CMS.cmsObjectDraw(gr_best_train, "P", MarkerColor=COLOR_TRAIN,
                      MarkerStyle=29, MarkerSize=2.0)  # Filled star
    CMS.cmsObjectDraw(gr_best_valid, "P", MarkerColor=COLOR_TEST,
                      MarkerStyle=29, MarkerSize=2.0)  # Filled star

    # Legend
    legend = ROOT.TLegend(0.60, 0.65, 0.90, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.035)
    legend.AddEntry(gr_mean_train, "Mean Train Loss", "LPE")
    legend.AddEntry(gr_mean_valid, "Mean Valid Loss", "LPE")
    legend.AddEntry(gr_best_train, "Best Train Loss", "P")
    legend.AddEntry(gr_best_valid, "Best Valid Loss", "P")
    legend.Draw()

    canvas.Update()

    # Save canvas
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.SaveAs(output_path)
    canvas.Close()

    print(f"Saved plot to: {output_path}")


def summarize_ga_loss(signal: str, channel: str, input_dir: str = "GAOptim_bjets") -> Dict:
    """
    Summarize GA optimization loss evolution.

    Args:
        signal: Signal name (e.g., MHc130_MA100)
        channel: Channel name (e.g., Run1E2Mu)
        input_dir: Input directory containing GA optimization results

    Returns:
        Dictionary with summary statistics
    """
    # Construct base path
    signal_full = f"TTToHcToWAToMuMu-{signal}"
    base_path = Path(input_dir) / channel / "multiclass" / signal_full

    if not base_path.exists():
        raise FileNotFoundError(f"Signal directory not found: {base_path}")

    # Find all GA iterations
    iteration_dirs = sorted(base_path.glob("GA-iter*"))

    if not iteration_dirs:
        raise FileNotFoundError(f"No GA iterations found in: {base_path}")

    print(f"Found {len(iteration_dirs)} GA iterations")

    # Collect statistics for each iteration
    iterations_data = []

    for iter_dir in iteration_dirs:
        # Extract iteration number
        iter_name = iter_dir.name
        iter_num = int(iter_name.replace("GA-iter", ""))

        print(f"Processing {iter_name}...")

        stats = get_iteration_statistics(iter_dir)
        if stats is not None:
            stats['iteration'] = iter_num
            iterations_data.append(stats)
        else:
            print(f"Warning: Failed to get statistics for {iter_name}")

    # Sort by iteration number
    iterations_data = sorted(iterations_data, key=lambda x: x['iteration'])

    # Create summary
    summary = {
        'signal': signal,
        'channel': channel,
        'input_dir': input_dir,
        'num_iterations': len(iterations_data),
        'iterations': iterations_data
    }

    return summary


def save_summary_json(summary: Dict, output_path: str):
    """Save summary data to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Summarize GA optimization loss evolution across iterations'
    )
    parser.add_argument('--signal', type=str, required=True,
                        help='Signal name (e.g., MHc130_MA100)')
    parser.add_argument('--channel', type=str, required=True,
                        help='Channel name (e.g., Run1E2Mu, Run3Mu, Combined)')
    parser.add_argument('--input', type=str, default='GAOptim_bjets',
                        help='Input directory (default: GAOptim_bjets)')

    args = parser.parse_args()

    # Setup CMS style
    setup_cms_style()

    print("="*70)
    print("GA Loss Summarization")
    print("="*70)
    print(f"Signal      : {args.signal}")
    print(f"Channel     : {args.channel}")
    print(f"Input dir   : {args.input}")
    print("="*70)
    print()

    try:
        # Compute summary statistics
        summary = summarize_ga_loss(args.signal, args.channel, args.input)

        # Print summary
        print()
        print("="*70)
        print("Summary Statistics")
        print("="*70)
        for iter_data in summary['iterations']:
            print(f"Iteration {iter_data['iteration']}:")
            print(f"  Models: {iter_data['num_models']}")
            print(f"  Mean Train Loss: {iter_data['mean_train_loss']:.4f} ± {iter_data['std_train_loss']:.4f}")
            print(f"  Mean Valid Loss: {iter_data['mean_valid_loss']:.4f} ± {iter_data['std_valid_loss']:.4f}")
            print(f"  Best Train Loss: {iter_data['min_train_loss']:.4f}")
            print(f"  Best Valid Loss: {iter_data['min_valid_loss']:.4f}")
            print()
        print("="*70)

        # Save JSON summary
        signal_full = f"TTToHcToWAToMuMu-{args.signal}"
        base_path = Path(args.input) / args.channel / "multiclass" / signal_full
        json_output = base_path / "ga_loss_summary.json"
        save_summary_json(summary, str(json_output))

        # Create plot
        plot_output = base_path / "ga_loss_evolution.png"
        create_loss_evolution_plot(summary, str(plot_output))

        print()
        print("="*70)
        print("Summarization completed successfully!")
        print("="*70)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
