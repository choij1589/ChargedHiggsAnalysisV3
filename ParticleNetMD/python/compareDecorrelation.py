#!/usr/bin/env python3
"""
Compare decorrelation vs performance across the lambda sweep.

Loads one ROOT tree per lambda (from LambdaSweep/{channel}/{signal}/fold-4/trees/),
computes per-lambda metrics (AUC, DisCo), and produces 4 output plots + summary.json.

Usage:
    python python/compareDecorrelation.py --signal MHc130_MA90 --channel Combined
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ROOT

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

from ROCCurveCalculator import ROCCurveCalculator
import torch
from WeightedLoss import distance_correlation



def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Compare decorrelation vs performance across lambda sweep"
    )
    parser.add_argument("--signal", required=True, type=str,
                        help="Signal name without prefix (e.g., MHc130_MA90)")
    parser.add_argument("--channel", required=True, type=str,
                        help="Channel (e.g., Combined, Run1E2Mu)")
    parser.add_argument("--results-dir", default="LambdaSweep", type=str,
                        help="Results directory relative to ParticleNetMD/ (default: LambdaSweep)")
    parser.add_argument("--pilot", action="store_true", default=False,
                        help="Use pilot results only (no full fold-4 results)")
    args = parser.parse_args()
    return args.signal, args.channel, args.results_dir, args.pilot


def parse_lambda_from_model_name(model_name: str) -> float:
    """Extract lambda value from model name (e.g., '...discoL0p05-...' → 0.05)."""
    match = re.search(r"discoL([0-9p]+)(?:-|$)", model_name)
    if match is None:
        raise ValueError(f"Cannot parse lambda from model name: {model_name}")
    lam_str = match.group(1).replace('p', '.')
    return float(lam_str)


def find_root_files(signal: str, channel: str, results_dir: str, workdir: str,
                    pilot: bool = False) -> list:
    """
    Glob for discoL*.root tree files for this signal × channel.
    Returns list of (lambda, tree_path) sorted by lambda.
    If pilot=True, only pilot/ results are used.
    Otherwise fold-4/ is preferred; pilot/ used as fallback.
    """
    base = os.path.join(workdir, "ParticleNetMD", results_dir, channel, signal)

    if pilot:
        patterns = [os.path.join(base, "pilot", "trees", "*.root")]
    else:
        patterns = [
            os.path.join(base, "fold-4", "trees", "*.root"),
            os.path.join(base, "pilot",  "trees", "*.root"),
        ]

    # Collect (lambda, path)
    found = {}
    for pattern in patterns:
        is_pilot = "pilot" in pattern
        for path in glob.glob(pattern):
            model_name = os.path.splitext(os.path.basename(path))[0]
            try:
                lam = parse_lambda_from_model_name(model_name)
            except ValueError:
                continue
            # Prefer fold-4 over pilot for the same lambda
            if lam not in found or (not is_pilot and "pilot" in found[lam]):
                found[lam] = path

    if not found:
        search_dirs = f"{base}/pilot/trees/" if pilot else f"{base}/fold-4/trees/ and {base}/pilot/trees/"
        raise RuntimeError(
            f"No discoL*.root files found for {signal}.\n"
            f"Searched in: {search_dirs}\n"
            "Run the lambda sweep first: bash scripts/runLambdaSweep.sh"
        )

    return sorted(found.items(), key=lambda x: x[0])


def load_tree_to_arrays(tree_path: str) -> dict:
    """Load ROOT tree branches into numpy arrays via RDataFrame (memory efficient)."""
    rdf = ROOT.RDataFrame("Events", tree_path)
    cols = rdf.AsNumpy(["score_signal", "true_label", "weight",
                        "mass1", "mass2", "test_mask"])
    arrays = {k: np.asarray(v) for k, v in cols.items()}
    arrays["test_mask"] = arrays["test_mask"].astype(bool)
    arrays["true_label"] = arrays["true_label"].astype(int)
    return arrays


def compute_auc(arrays: dict) -> float:
    """Signal-vs-rest AUC on test-set events."""
    mask = arrays["test_mask"]
    y_true = (arrays["true_label"][mask] == 0).astype(int)
    scores  = arrays["score_signal"][mask]
    weights = arrays["weight"][mask]

    calc = ROCCurveCalculator()
    _, _, auc = calc.calculate_roc_curve(y_true, scores, weights)
    return auc


def compute_disco_metrics(arrays: dict, max_disco_events: int = 5000) -> tuple:
    """
    Post-hoc DisCo(score_signal, mass1/2) on test-set events.
    Returns (disco_mass1, disco_mass2).
    Subsamples to max_disco_events before computing (DisCo is O(n^2) in memory).
    """
    mask = arrays["test_mask"]
    scores_np  = arrays["score_signal"][mask]
    mass1_np   = arrays["mass1"][mask]
    mass2_np   = arrays["mass2"][mask]
    weights_np = arrays["weight"][mask]

    # DisCo allocates an n×n pairwise matrix — subsample to keep memory bounded
    n = len(scores_np)
    if n > max_disco_events:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(n, max_disco_events, replace=False)
        scores_np  = scores_np[idx]
        mass1_np   = mass1_np[idx]
        mass2_np   = mass2_np[idx]
        weights_np = weights_np[idx]

    scores  = torch.tensor(scores_np,  dtype=torch.float32)
    weights = torch.tensor(weights_np, dtype=torch.float32)

    valid1 = mass1_np > 0
    if valid1.sum() > 1:
        d1 = distance_correlation(
            scores[valid1],
            torch.tensor(mass1_np[valid1], dtype=torch.float32),
            weights[valid1]
        ).item()
    else:
        d1 = 0.0

    valid2 = mass2_np > 0
    if valid2.sum() > 1:
        d2 = distance_correlation(
            scores[valid2],
            torch.tensor(mass2_np[valid2], dtype=torch.float32),
            weights[valid2]
        ).item()
    else:
        d2 = 0.0

    return d1, d2


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _symlog_xticks(ax, lambdas):
    """Use symlog x-scale that looks good for lambda lists like [0, 0.01, ..., 0.5]."""
    ax.set_xscale("symlog", linthresh=0.005)
    ax.set_xticks(lambdas)
    ax.set_xticklabels([str(l) for l in lambdas], rotation=30, ha='right')


def plot_performance_vs_lambda(lambdas, aucs, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lambdas, aucs, marker='o', linewidth=2, color='steelblue')
    _symlog_xticks(ax, lambdas)
    ax.set_xlabel(r"$\lambda$ (DisCo weight)", fontsize=13)
    ax.set_ylabel("AUC (signal vs rest)", fontsize=13)
    ax.set_title("Classifier Performance vs DisCo Strength", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=max(0.5, min(aucs) - 0.05))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_disco_vs_lambda(lambdas, disco1, disco2, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lambdas, disco1, marker='o', linewidth=2,
            label=r"DisCo(score, $m_1$)", color='steelblue')
    ax.plot(lambdas, disco2, marker='s', linewidth=2, linestyle='--',
            label=r"DisCo(score, $m_2$)", color='darkorange')
    _symlog_xticks(ax, lambdas)
    ax.set_xlabel(r"$\lambda$ (DisCo weight)", fontsize=13)
    ax.set_ylabel("Distance Correlation", fontsize=13)
    ax.set_title(r"Post-hoc DisCo vs $\lambda$", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0.0)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_tradeoff(lambdas, aucs, disco1, path):
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(disco1, aucs, c=range(len(lambdas)),
                    cmap='viridis', s=90, zorder=5)
    for lam, d, a in zip(lambdas, disco1, aucs):
        ax.annotate(f"λ={lam}", (d, a),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)
    cbar = fig.colorbar(sc, ax=ax, ticks=range(len(lambdas)))
    cbar.ax.set_yticklabels([str(l) for l in lambdas])
    cbar.set_label(r"$\lambda$", fontsize=12)
    ax.set_xlabel(r"DisCo(score, $m_1$)", fontsize=13)
    ax.set_ylabel("AUC (signal vs rest)", fontsize=13)
    ax.set_title("Performance–Decorrelation Tradeoff", fontsize=14)
    ax.grid(True, alpha=0.3)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mass_sculpting(lambda_arrays: list, path: str):
    """
    For each lambda: show background mass1 at 3 score cuts vs no-cut baseline.
    One subplot per lambda.
    """
    score_cuts  = [0.3, 0.5, 0.7]
    cut_colors  = ['steelblue', 'darkorange', 'crimson']
    cut_labels  = [f"score > {c}" for c in score_cuts]

    n     = len(lambda_arrays)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = np.asarray(axes).flatten()

    for idx, (lam, arrays) in enumerate(lambda_arrays):
        ax = axes[idx]
        mask = arrays["test_mask"]
        bg   = mask & (arrays["true_label"] != 0)   # background events only

        mass1   = arrays["mass1"][bg]
        scores  = arrays["score_signal"][bg]
        weights = np.abs(arrays["weight"][bg])       # abs for density plots

        valid = mass1 > 0
        if valid.sum() < 2:
            ax.set_title(f"λ={lam} (no data)")
            continue

        m_max = max(120.0, float(np.percentile(mass1[valid], 99)))
        bins  = np.linspace(0.0, m_max, 40)

        # Baseline: no score cut
        ax.hist(mass1[valid], bins=bins, weights=weights[valid],
                histtype='step', linewidth=2, color='black',
                label='no cut', density=True)

        # Score cuts
        for cut, color, label in zip(score_cuts, cut_colors, cut_labels):
            sel = valid & (scores > cut)
            if sel.sum() > 1:
                ax.hist(mass1[sel], bins=bins, weights=weights[sel],
                        histtype='step', linewidth=1.5, color=color,
                        label=label, density=True)

        ax.set_title(f"λ={lam}", fontsize=11)
        ax.set_xlabel(r"$m_1$ [GeV]", fontsize=10)
        ax.set_ylabel("a.u.", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Background Mass Sculpting at Score Cuts", fontsize=14)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    signal, channel, results_dir, pilot = parse_arguments()

    try:
        workdir = os.environ["WORKDIR"]
    except KeyError:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh' first.")

    lambda_paths = find_root_files(signal, channel, results_dir, workdir, pilot=pilot)
    print(f"Found {len(lambda_paths)} lambda values: {[lam for lam, _ in lambda_paths]}")

    fold_subdir = "pilot" if pilot else "fold-4"
    out_dir = os.path.join(workdir, "ParticleNetMD", results_dir, channel, signal,
                           fold_subdir, "comparison")
    os.makedirs(out_dir, exist_ok=True)

    ROOT.gROOT.SetBatch(True)
    ROOT.DisableImplicitMT()

    lambdas        = []
    aucs           = []
    disco1_list    = []
    disco2_list    = []
    lambda_arrays  = []

    for lam, tree_path in lambda_paths:
        print(f"  λ={lam}: {os.path.basename(tree_path)}")
        arrays = load_tree_to_arrays(tree_path)
        auc    = compute_auc(arrays)
        d1, d2 = compute_disco_metrics(arrays)

        lambdas.append(lam)
        aucs.append(auc)
        disco1_list.append(d1)
        disco2_list.append(d2)

        lambda_arrays.append((lam, arrays))
        del arrays

        print(f"    AUC={auc:.4f}  DisCo(m1)={d1:.4f}  DisCo(m2)={d2:.4f}")

    # Plots
    plot_performance_vs_lambda(
        lambdas, aucs,
        os.path.join(out_dir, "performance_vs_lambda.png")
    )
    plot_disco_vs_lambda(
        lambdas, disco1_list, disco2_list,
        os.path.join(out_dir, "disco_vs_lambda.png")
    )
    plot_tradeoff(
        lambdas, aucs, disco1_list,
        os.path.join(out_dir, "tradeoff.png")
    )
    plot_mass_sculpting(
        lambda_arrays,
        os.path.join(out_dir, "mass_sculpting.png")
    )

    # Summary JSON
    summary = [
        {"lambda": lam, "auc": auc, "disco_mass1": d1, "disco_mass2": d2}
        for lam, auc, d1, d2 in zip(lambdas, aucs, disco1_list, disco2_list)
    ]
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\nOutputs → {out_dir}")
    for fname in ("performance_vs_lambda.png", "disco_vs_lambda.png",
                  "tradeoff.png", "mass_sculpting.png", "summary.json"):
        print(f"  {fname}")


if __name__ == "__main__":
    main()
