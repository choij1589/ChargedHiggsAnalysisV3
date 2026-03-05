#!/usr/bin/env python3
"""
Summarize GA optimization loss evolution across iterations for ParticleNetMD.

This script reads the training results from all GA iterations and computes
statistics on train/validation loss (including DisCo loss decomposition)
to track optimization progress.

For each iteration:
- Reads all model JSON files
- Computes mean train/valid loss across population
- Tracks DisCo loss decomposition (CE loss + DisCo term)
- Tracks best model performance
- Generates summary plots and JSON output
- Copies best model from last iteration to best_model/

Usage:
    python summarizeGALoss.py --signal MHc130_MA100 --channel Combined
    python summarizeGALoss.py --signal MHc130_MA100 --channel Combined --pilot
    python summarizeGALoss.py --signal MHc130_MA100 --channel Run1E2Mu --input GAOptim_custom
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import json
import argparse
import numpy as np
import shutil
from pathlib import Path
from typing import Dict, Optional

import ROOT
ROOT.gROOT.SetBatch(True)

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

    Extracts total loss, CE loss, DisCo term, and accuracy from each model's
    training summary (best epoch values).

    Args:
        iteration_dir: Path to GA-iter{N} directory

    Returns:
        Dictionary with iteration statistics or None if failed
    """
    json_dir = iteration_dir / "json"

    if not json_dir.exists():
        print(f"Warning: JSON directory not found: {json_dir}")
        return None

    model_info_path = json_dir / "model_info.csv"
    if not model_info_path.exists():
        print(f"Warning: model_info.csv not found: {model_info_path}")
        return None

    # Parse model list from CSV
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

    # Collect per-model metrics at best epoch
    best_train_losses = []
    best_valid_losses = []
    best_train_ce = []
    best_valid_ce = []
    best_train_disco = []
    best_valid_disco = []
    best_train_accs = []
    best_valid_accs = []
    model_data_list = []  # (idx, valid_loss)

    for idx in model_indices:
        json_path = json_dir / f"model{idx}.json"
        if not json_path.exists():
            print(f"Warning: Model JSON not found: {json_path}")
            continue

        model_data = read_model_json(str(json_path))
        if model_data is None:
            continue

        summary = model_data.get('training_summary', {})
        best_train = summary.get('best_train_loss')
        best_valid = summary.get('best_valid_loss')

        if best_train is None or best_valid is None:
            continue

        best_train_losses.append(best_train)
        best_valid_losses.append(best_valid)
        model_data_list.append((idx, best_valid))

        # Accuracy
        best_train_accs.append(summary.get('best_train_acc', float('nan')))
        best_valid_accs.append(summary.get('best_valid_acc', float('nan')))

        # DisCo decomposition: extract CE and DisCo at best epoch
        epoch_history = model_data.get('epoch_history', {})
        best_epoch = summary.get('best_epoch')
        if best_epoch is not None and 'train_ce_loss' in epoch_history:
            try:
                best_train_ce.append(epoch_history['train_ce_loss'][best_epoch])
                best_valid_ce.append(epoch_history['valid_ce_loss'][best_epoch])
                best_train_disco.append(epoch_history['train_disco_term'][best_epoch])
                best_valid_disco.append(epoch_history['valid_disco_term'][best_epoch])
            except (IndexError, KeyError):
                pass

    if not best_train_losses:
        print(f"Warning: No valid loss data found in {iteration_dir}")
        return None

    best_model_idx, best_model_valid_loss = min(model_data_list, key=lambda x: x[1])

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
        'median_valid_loss': float(np.median(best_valid_losses)),
        'mean_train_acc': float(np.nanmean(best_train_accs)),
        'mean_valid_acc': float(np.nanmean(best_valid_accs)),
        'best_model_idx': best_model_idx,
        'best_model_valid_loss': best_model_valid_loss,
    }

    # DisCo decomposition stats (only if available)
    if best_train_ce:
        stats['mean_train_ce_loss'] = float(np.mean(best_train_ce))
        stats['mean_valid_ce_loss'] = float(np.mean(best_valid_ce))
        stats['mean_train_disco_term'] = float(np.mean(best_train_disco))
        stats['mean_valid_disco_term'] = float(np.mean(best_valid_disco))
        stats['std_train_ce_loss'] = float(np.std(best_train_ce))
        stats['std_valid_ce_loss'] = float(np.std(best_valid_ce))
        stats['std_train_disco_term'] = float(np.std(best_train_disco))
        stats['std_valid_disco_term'] = float(np.std(best_valid_disco))

    return stats


def create_loss_evolution_plot(summary_data: Dict, output_path: str):
    """
    Create ROOT plot showing loss evolution across GA iterations.

    Shows mean ± std for total loss (train/valid) and best loss per iteration.
    """
    iterations_data = summary_data['iterations']
    if not iterations_data:
        print("Warning: No iteration data to plot")
        return

    n_iters = len(iterations_data)

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

    # Create TGraphs
    gr_mean_train = ROOT.TGraph(n_iters)
    gr_mean_valid = ROOT.TGraph(n_iters)
    gr_err_train = ROOT.TGraphErrors(n_iters)
    gr_err_valid = ROOT.TGraphErrors(n_iters)
    gr_best_train = ROOT.TGraph(n_iters)
    gr_best_valid = ROOT.TGraph(n_iters)

    for i in range(n_iters):
        gr_mean_train.SetPoint(i, iter_nums[i], mean_train[i])
        gr_mean_valid.SetPoint(i, iter_nums[i], mean_valid[i])
        gr_err_train.SetPoint(i, iter_nums[i], mean_train[i])
        gr_err_train.SetPointError(i, 0, std_train[i])
        gr_err_valid.SetPoint(i, iter_nums[i], mean_valid[i])
        gr_err_valid.SetPointError(i, 0, std_valid[i])
        gr_best_train.SetPoint(i, iter_nums[i], best_train[i])
        gr_best_valid.SetPoint(i, iter_nums[i], best_valid[i])

    # Y-axis range
    all_vals = mean_train + mean_valid + best_train + best_valid
    all_errs = std_train + std_valid
    y_min = max(0, min(all_vals) - max(all_errs) - 0.05)
    y_max = max(all_vals) + max(all_errs) + 0.05

    x_min = min(iter_nums) - 0.3
    x_max = max(iter_nums) + 0.3

    COLOR_TRAIN = ROOT.TColor.GetColor("#5790fc")
    COLOR_VALID = ROOT.TColor.GetColor("#f89c20")

    canvas = CMS.cmsCanvas("c_ga_loss",
                           x_min, x_max,
                           y_min, y_max,
                           "GA Iteration",
                           "Loss (CE + #lambda #upoint DisCo)",
                           square=True,
                           iPos=0,
                           extraSpace=0.01)

    CMS.cmsObjectDraw(gr_err_train, "PE", LineColor=COLOR_TRAIN, LineWidth=2,
                      MarkerStyle=1, MarkerSize=0)
    CMS.cmsObjectDraw(gr_err_valid, "PE", LineColor=COLOR_VALID, LineWidth=2,
                      MarkerStyle=1, MarkerSize=0)
    CMS.cmsObjectDraw(gr_mean_train, "LP", LineColor=COLOR_TRAIN, LineWidth=2,
                      MarkerColor=COLOR_TRAIN, MarkerStyle=20, MarkerSize=1.2)
    CMS.cmsObjectDraw(gr_mean_valid, "LP", LineColor=COLOR_VALID, LineWidth=2,
                      MarkerColor=COLOR_VALID, MarkerStyle=20, MarkerSize=1.2)
    CMS.cmsObjectDraw(gr_best_train, "P", MarkerColor=COLOR_TRAIN,
                      MarkerStyle=29, MarkerSize=2.0)
    CMS.cmsObjectDraw(gr_best_valid, "P", MarkerColor=COLOR_VALID,
                      MarkerStyle=29, MarkerSize=2.0)

    legend = ROOT.TLegend(0.55, 0.65, 0.90, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.035)
    legend.AddEntry(gr_mean_train, "Mean Train Loss", "LPE")
    legend.AddEntry(gr_mean_valid, "Mean Valid Loss", "LPE")
    legend.AddEntry(gr_best_train, "Best Train Loss", "P")
    legend.AddEntry(gr_best_valid, "Best Valid Loss", "P")
    legend.Draw()

    canvas.Update()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.SaveAs(output_path)
    canvas.Close()

    print(f"Saved plot to: {output_path}")


def create_disco_decomposition_plot(summary_data: Dict, output_path: str):
    """
    Create ROOT plot showing DisCo loss decomposition across GA iterations.

    Shows mean CE loss and mean DisCo term separately for train/valid.
    """
    iterations_data = summary_data['iterations']
    if not iterations_data:
        return

    # Check if DisCo decomposition data is available
    if 'mean_train_ce_loss' not in iterations_data[0]:
        print("No DisCo decomposition data available, skipping decomposition plot")
        return

    n_iters = len(iterations_data)

    iter_nums = []
    mean_train_ce = []
    mean_valid_ce = []
    mean_train_disco = []
    mean_valid_disco = []

    for iter_data in iterations_data:
        iter_nums.append(iter_data['iteration'])
        mean_train_ce.append(iter_data['mean_train_ce_loss'])
        mean_valid_ce.append(iter_data['mean_valid_ce_loss'])
        mean_train_disco.append(iter_data['mean_train_disco_term'])
        mean_valid_disco.append(iter_data['mean_valid_disco_term'])

    # Create TGraphs
    gr_train_ce = ROOT.TGraph(n_iters)
    gr_valid_ce = ROOT.TGraph(n_iters)
    gr_train_disco = ROOT.TGraph(n_iters)
    gr_valid_disco = ROOT.TGraph(n_iters)

    for i in range(n_iters):
        gr_train_ce.SetPoint(i, iter_nums[i], mean_train_ce[i])
        gr_valid_ce.SetPoint(i, iter_nums[i], mean_valid_ce[i])
        gr_train_disco.SetPoint(i, iter_nums[i], mean_train_disco[i])
        gr_valid_disco.SetPoint(i, iter_nums[i], mean_valid_disco[i])

    all_vals = mean_train_ce + mean_valid_ce + mean_train_disco + mean_valid_disco
    y_min = max(0, min(all_vals) - 0.05)
    y_max = max(all_vals) + 0.05

    x_min = min(iter_nums) - 0.3
    x_max = max(iter_nums) + 0.3

    COLOR_TRAIN_CE = ROOT.TColor.GetColor("#5790fc")
    COLOR_VALID_CE = ROOT.TColor.GetColor("#f89c20")
    COLOR_TRAIN_DISCO = ROOT.TColor.GetColor("#e42536")
    COLOR_VALID_DISCO = ROOT.TColor.GetColor("#964a8b")

    canvas = CMS.cmsCanvas("c_ga_disco",
                           x_min, x_max,
                           y_min, y_max,
                           "GA Iteration",
                           "Loss Component",
                           square=True,
                           iPos=0,
                           extraSpace=0.01)

    CMS.cmsObjectDraw(gr_train_ce, "LP", LineColor=COLOR_TRAIN_CE, LineWidth=2,
                      MarkerColor=COLOR_TRAIN_CE, MarkerStyle=20, MarkerSize=1.2)
    CMS.cmsObjectDraw(gr_valid_ce, "LP", LineColor=COLOR_VALID_CE, LineWidth=2,
                      MarkerColor=COLOR_VALID_CE, MarkerStyle=20, MarkerSize=1.2)
    CMS.cmsObjectDraw(gr_train_disco, "LP", LineColor=COLOR_TRAIN_DISCO, LineWidth=2,
                      LineStyle=2,
                      MarkerColor=COLOR_TRAIN_DISCO, MarkerStyle=21, MarkerSize=1.0)
    CMS.cmsObjectDraw(gr_valid_disco, "LP", LineColor=COLOR_VALID_DISCO, LineWidth=2,
                      LineStyle=2,
                      MarkerColor=COLOR_VALID_DISCO, MarkerStyle=21, MarkerSize=1.0)

    legend = ROOT.TLegend(0.55, 0.65, 0.90, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.032)
    legend.AddEntry(gr_train_ce, "Train CE Loss", "LP")
    legend.AddEntry(gr_valid_ce, "Valid CE Loss", "LP")
    legend.AddEntry(gr_train_disco, "Train DisCo Term", "LP")
    legend.AddEntry(gr_valid_disco, "Valid DisCo Term", "LP")
    legend.Draw()

    canvas.Update()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.SaveAs(output_path)
    canvas.Close()

    print(f"Saved plot to: {output_path}")


def summarize_ga_loss(signal: str, channel: str, input_dir: str = "GAOptim",
                      pilot: bool = False, fold: int = None) -> Dict:
    """
    Summarize GA optimization loss evolution.

    Args:
        signal: Signal name (e.g., MHc130_MA100)
        channel: Channel name (e.g., Run1E2Mu, Run3Mu, Combined)
        input_dir: Input directory containing GA optimization results
        pilot: Whether to look in pilot/ subdirectory
        fold: Test fold number for non-pilot runs (derived from config if None)

    Returns:
        Dictionary with summary statistics
    """
    if fold is None:
        from GAConfig import load_ga_config
        _config = load_ga_config()
        test_folds = _config.get_overfitting_config().get('test_folds', [4])
        fold = test_folds[0]
    if pilot:
        fold_dir = "pilot"
    else:
        fold_dir = f"fold-{fold}"
    base_path = Path(input_dir) / channel / signal / fold_dir

    if not base_path.exists():
        raise FileNotFoundError(f"Signal directory not found: {base_path}")

    iteration_dirs = sorted(base_path.glob("GA-iter*"))

    if not iteration_dirs:
        raise FileNotFoundError(f"No GA iterations found in: {base_path}")

    print(f"Found {len(iteration_dirs)} GA iterations")

    iterations_data = []
    for iter_dir in iteration_dirs:
        iter_name = iter_dir.name
        iter_num = int(iter_name.replace("GA-iter", ""))

        print(f"Processing {iter_name}...")

        stats = get_iteration_statistics(iter_dir)
        if stats is not None:
            stats['iteration'] = iter_num
            iterations_data.append(stats)
        else:
            print(f"Warning: Failed to get statistics for {iter_name}")

    iterations_data = sorted(iterations_data, key=lambda x: x['iteration'])

    summary = {
        'signal': signal,
        'channel': channel,
        'input_dir': input_dir,
        'pilot': pilot,
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


def copy_best_model(base_path: Path, last_iteration_dir: Path, best_model_idx: int):
    """
    Copy the best model from the last iteration to best_model/ subdirectory.

    Args:
        base_path: Base path for the signal/channel
        last_iteration_dir: Path to the last GA iteration directory
        best_model_idx: Index of the best model to copy
    """
    best_model_dir = base_path / "best_model"
    os.makedirs(best_model_dir, exist_ok=True)

    # Source files
    model_pt_src = last_iteration_dir / "models" / f"model{best_model_idx}.pt"
    model_json_src = last_iteration_dir / "json" / f"model{best_model_idx}.json"

    # Destination files (renamed)
    model_pt_dst = best_model_dir / "model.pt"
    model_json_dst = best_model_dir / "model_info.json"

    if not model_pt_src.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_pt_src}")
    if not model_json_src.exists():
        raise FileNotFoundError(f"Model JSON not found: {model_json_src}")

    shutil.copy2(model_pt_src, model_pt_dst)
    shutil.copy2(model_json_src, model_json_dst)

    # Copy validation plots
    plots_dir = last_iteration_dir / "plots"
    copied_plots = []

    if plots_dir.exists():
        plot_types = [
            "confusion_matrix",
            "ks_test_heatmap",
            "roc_curves",
            "score_distributions_grid",
            "training_curves",
            "mass_sculpting",
            "mass_profile_vs_score",
        ]

        for plot_type in plot_types:
            plot_src = plots_dir / f"model{best_model_idx}_{plot_type}.png"
            if plot_src.exists():
                plot_dst = best_model_dir / f"{plot_type}.png"
                shutil.copy2(plot_src, plot_dst)
                copied_plots.append(plot_dst.name)

    print()
    print("="*70)
    print("Best Model Identification")
    print("="*70)
    print(f"Best model from last iteration: model{best_model_idx}")
    print(f"Copied to: {best_model_dir}")
    print()
    print("Model files:")
    print(f"  - {model_pt_dst.name}")
    print(f"  - {model_json_dst.name}")
    if copied_plots:
        print()
        print(f"Validation plots ({len(copied_plots)}):")
        for plot in copied_plots:
            print(f"  - {plot}")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description='Summarize GA optimization loss evolution across iterations (ParticleNetMD)'
    )
    parser.add_argument('--signal', type=str, required=True,
                        help='Signal name (e.g., MHc130_MA100)')
    parser.add_argument('--channel', type=str, required=True,
                        help='Channel name (e.g., Run1E2Mu, Run3Mu, Combined)')
    parser.add_argument('--input', type=str, default='GAOptim',
                        help='Input directory (default: GAOptim)')
    parser.add_argument('--pilot', action='store_true',
                        help='Use pilot directory layout')

    args = parser.parse_args()

    setup_cms_style()

    # Derive fold from config
    from GAConfig import load_ga_config
    _config = load_ga_config()
    test_folds = _config.get_overfitting_config().get('test_folds', [4])
    fold = test_folds[0]

    print("="*70)
    print("GA Loss Summarization (ParticleNetMD)")
    print("="*70)
    print(f"Signal      : {args.signal}")
    print(f"Channel     : {args.channel}")
    print(f"Input dir   : {args.input}")
    print(f"Pilot mode  : {args.pilot}")
    print(f"Fold        : {fold} (from config)")
    print("="*70)
    print()

    try:
        summary = summarize_ga_loss(args.signal, args.channel, args.input, args.pilot, fold)

        # Print summary
        has_disco = 'mean_train_ce_loss' in summary['iterations'][0] if summary['iterations'] else False

        print()
        print("="*70)
        print("Summary Statistics")
        print("="*70)
        for i, iter_data in enumerate(summary['iterations']):
            is_last = (i == len(summary['iterations']) - 1)
            print(f"Iteration {iter_data['iteration']}:")
            print(f"  Models: {iter_data['num_models']}")
            print(f"  Mean Train Loss: {iter_data['mean_train_loss']:.4f} ± {iter_data['std_train_loss']:.4f}")
            print(f"  Mean Valid Loss: {iter_data['mean_valid_loss']:.4f} ± {iter_data['std_valid_loss']:.4f}")
            if has_disco:
                print(f"    CE component:    train={iter_data['mean_train_ce_loss']:.4f}  valid={iter_data['mean_valid_ce_loss']:.4f}")
                print(f"    DisCo component: train={iter_data['mean_train_disco_term']:.4f}  valid={iter_data['mean_valid_disco_term']:.4f}")
            print(f"  Mean Train Acc:  {iter_data['mean_train_acc']:.4f}")
            print(f"  Mean Valid Acc:  {iter_data['mean_valid_acc']:.4f}")
            print(f"  Best Train Loss: {iter_data['min_train_loss']:.4f}")
            print(f"  Best Valid Loss: {iter_data['min_valid_loss']:.4f}")
            if is_last:
                print(f"  Best Model: model{iter_data['best_model_idx']} (valid_loss={iter_data['best_model_valid_loss']:.4f})")
            print()
        print("="*70)

        # Resolve base_path for output
        fold_dir = "pilot" if args.pilot else f"fold-{fold}"
        base_path = Path(args.input) / args.channel / args.signal / fold_dir

        # Save JSON summary
        json_output = base_path / "ga_loss_summary.json"
        save_summary_json(summary, str(json_output))

        # Create total loss evolution plot
        plot_output = base_path / "ga_loss_evolution.png"
        create_loss_evolution_plot(summary, str(plot_output))

        # Create DisCo decomposition plot
        disco_plot_output = base_path / "ga_disco_decomposition.png"
        create_disco_decomposition_plot(summary, str(disco_plot_output))

        # Copy best model from last iteration
        if summary['iterations']:
            last_iter_data = summary['iterations'][-1]
            last_iter_num = last_iter_data['iteration']
            best_model_idx = last_iter_data['best_model_idx']

            last_iteration_dir = base_path / f"GA-iter{last_iter_num}"
            copy_best_model(base_path, last_iteration_dir, best_model_idx)

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
