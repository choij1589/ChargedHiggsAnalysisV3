#!/usr/bin/env python3
"""Plot signal-only mass sculpting from saved DNN comparison predictions."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

REPO_DIR = Path(__file__).resolve().parents[1]
SIGNALS = ["MHc100_MA95", "MHc130_MA90", "MHc160_MA85"]
MODEL_KEYS = {
    "DNN": ("dnn_lr", "tab:blue"),
    "ParticleNet": ("pn_lr", "tab:orange"),
}
REGIONS = [
    ("No cut", None, "black"),
    ("LR < 0.3", lambda lr: lr < 0.3, "tab:blue"),
    ("0.3 < LR < 0.7", lambda lr: (lr > 0.3) & (lr < 0.7), "tab:orange"),
    ("LR > 0.7", lambda lr: lr > 0.7, "tab:red"),
]


def normalized_hist(values: np.ndarray, weights: np.ndarray,
                    bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, weights=np.abs(weights))
    total = hist.sum()
    if total > 0:
        hist = hist / total
    return hist, edges


def load_split(out_dir: Path, split: str) -> Dict[str, np.ndarray]:
    path = out_dir / f"predictions_{split}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def plot_model_signal_shapes(train: Dict[str, np.ndarray], test: Dict[str, np.ndarray],
                             model_name: str, lr_key: str, out_dir: Path) -> None:
    bins = np.linspace(60.0, 120.0, 31)
    centers = 0.5 * (bins[:-1] + bins[1:])

    for mass_name in ["mass1", "mass2"]:
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 7.2), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1], "hspace": 0.05})
        ax, rax = axes
        refs: Dict[str, np.ndarray] = {}

        for split_name, table, linestyle in [
            ("train", train, "--"),
            ("test", test, "-"),
        ]:
            mass = table[mass_name]
            lr = table[lr_key]
            valid = (table["y"] == 0) & (mass > 0)
            if valid.sum() == 0 or lr.size == 0:
                continue

            for label, selector, color in REGIONS:
                mask = valid if selector is None else valid & selector(lr)
                if mask.sum() == 0:
                    continue
                hist, _edges = normalized_hist(mass[mask], table["weight"][mask], bins)
                ax.step(
                    centers,
                    hist,
                    where="mid",
                    color=color,
                    linestyle=linestyle,
                    linewidth=2,
                    label=f"{label} {split_name} (N={int(mask.sum())})",
                )
                if label == "No cut":
                    refs[split_name] = hist
                elif split_name in refs:
                    denom = np.where(refs[split_name] > 0, refs[split_name], np.nan)
                    ratio = hist / denom
                    rax.step(
                        centers,
                        ratio,
                        where="mid",
                        color=color,
                        linestyle=linestyle,
                        linewidth=2,
                    )

        ax.set_ylabel("Normalized")
        ax.set_title(f"{model_name}: signal {mass_name}")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=2)
        rax.axhline(1.0, color="black", linestyle=":", linewidth=1.5)
        rax.set_ylim(0.4, 1.8)
        rax.set_ylabel("Region / No cut")
        rax.set_xlabel(f"{mass_name} [GeV]")
        rax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(out_dir / f"signal_mass_shape_{model_name}_{mass_name}.png", dpi=160)
        plt.close(fig)


def plot_model_overlay_test(test: Dict[str, np.ndarray], out_dir: Path) -> None:
    bins = np.linspace(60.0, 120.0, 31)
    centers = 0.5 * (bins[:-1] + bins[1:])

    for mass_name in ["mass1", "mass2"]:
        fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.0), sharex=True, sharey=True)
        axes = axes.reshape(-1)
        for ax, (region_label, selector, _region_color) in zip(axes, REGIONS):
            for model_name, (lr_key, color) in MODEL_KEYS.items():
                lr = test[lr_key]
                if lr.size == 0:
                    continue
                mass = test[mass_name]
                valid = (test["y"] == 0) & (mass > 0)
                mask = valid if selector is None else valid & selector(lr)
                if mask.sum() == 0:
                    continue
                hist, _edges = normalized_hist(mass[mask], test["weight"][mask], bins)
                ax.step(
                    centers,
                    hist,
                    where="mid",
                    color=color,
                    linewidth=2,
                    label=f"{model_name} (N={int(mask.sum())})",
                )
            ax.set_title(region_label)
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
        for ax in axes[::2]:
            ax.set_ylabel("Normalized")
        for ax in axes[-2:]:
            ax.set_xlabel(f"{mass_name} [GeV]")
        fig.suptitle(f"Signal {mass_name}: DNN vs ParticleNet test overlays", y=0.98)
        fig.tight_layout()
        fig.savefig(out_dir / f"signal_mass_shape_DNN_vs_ParticleNet_test_{mass_name}.png", dpi=160)
        plt.close(fig)


def plot_signal(signal: str, channel: str) -> List[Path]:
    out_dir = REPO_DIR / "DNN" / channel / signal / "fold-4"
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    train = load_split(out_dir, "train")
    test = load_split(out_dir, "test")

    written = []
    for model_name, (lr_key, _color) in MODEL_KEYS.items():
        if lr_key not in train or lr_key not in test or train[lr_key].size == 0 or test[lr_key].size == 0:
            continue
        plot_model_signal_shapes(train, test, model_name, lr_key, plots_dir)
        written.extend([
            plots_dir / f"signal_mass_shape_{model_name}_mass1.png",
            plots_dir / f"signal_mass_shape_{model_name}_mass2.png",
        ])
    if all(key in test and test[key].size > 0 for key, _color in MODEL_KEYS.values()):
        plot_model_overlay_test(test, plots_dir)
        written.extend([
            plots_dir / "signal_mass_shape_DNN_vs_ParticleNet_test_mass1.png",
            plots_dir / "signal_mass_shape_DNN_vs_ParticleNet_test_mass2.png",
        ])
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signal", choices=SIGNALS, action="append",
                        help="Signal mass point. Can be passed multiple times. Defaults to all available.")
    parser.add_argument("--channel", default="Combined")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    signals: Iterable[str] = args.signal or SIGNALS
    for signal in signals:
        out_dir = REPO_DIR / "DNN" / args.channel / signal / "fold-4"
        if not (out_dir / "predictions_train.npz").exists() or not (out_dir / "predictions_test.npz").exists():
            print(f"Skipping {signal}: predictions are not ready under {out_dir}", flush=True)
            continue
        written = plot_signal(signal, args.channel)
        print(f"{signal}: wrote {len(written)} signal mass-shape plots", flush=True)
        for path in written:
            print(f"  {path}", flush=True)


if __name__ == "__main__":
    main()
