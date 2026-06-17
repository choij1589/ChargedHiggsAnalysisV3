#!/usr/bin/env python3
"""
Train and compare a high-capacity tabular DNN baseline against ParticleNet.

The DNN uses the same tabular inputs produced for the BDT comparison, so the
comparison isolates the classifier family while keeping event selection, folds,
weights, labels, masses, and diagnostics aligned with the ParticleNetMD workflow.

Example:
    python python/trainDNN.py --signal MHc160_MA85 --device cuda
    python python/trainDNN.py --all --device cuda
    python python/trainDNN.py --signal MHc160_MA85 --pilot --max-epochs 2
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

import trainBDT as bdt
from SglConfig import load_sgl_config
from TrainingUtilities import (
    calculate_group_balanced_accuracy,
    create_optimizer,
    create_scheduler,
)
from WeightedLoss import create_loss_function


SIGNALS = ["MHc160_MA85", "MHc130_MA90", "MHc100_MA95"]
OUTPUT_BASE = bdt.PARTICLENETMD_DIR / "DNN"
CLASS_NAMES = bdt.CLASS_NAMES


class TabularDNN(nn.Module):
    """Regularized MLP with the same call site shape as a tabular model."""

    def __init__(self, input_dim: int, hidden_layers: Sequence[int],
                 num_classes: int = 4, dropout_p: float = 0.4):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout_p),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_layers = list(hidden_layers)
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MassAgnosticLossAdapter(nn.Module):
    """Adapt CE-only losses to the DNN training loop's mass-aware call site."""

    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss
        self.last_ce_loss = 0.0

    def forward(self, logits, target, weight, mass1=None, mass2=None):
        loss = self.base_loss(logits, target, weight)
        self.last_ce_loss = float(loss.detach().cpu())
        return loss

    def get_decomposed_losses(self):
        return {
            "ce_loss": self.last_ce_loss,
            "disco_term": 0.0,
            "disco1": 0.0,
            "disco2": 0.0,
            "disco_weighted": 0.0,
        }


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def parse_hidden_layers(value: Optional[str], n_nodes: int) -> List[int]:
    if value:
        layers = [int(item) for item in value.split(",") if item.strip()]
        if not layers or any(width <= 0 for width in layers):
            raise ValueError("--hidden-layers must contain positive integers")
        return layers

    base_width = max(1024, 4 * int(n_nodes))
    return [base_width, base_width, max(512, 2 * int(n_nodes)), max(256, int(n_nodes))]


def fit_preprocessor(train: Dict[str, np.ndarray]) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ]).fit(train["X"])


def transform_table(table: Dict[str, np.ndarray], preprocessor: Pipeline) -> np.ndarray:
    return preprocessor.transform(table["X"]).astype(np.float32)


def load_npz_table(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as arrays:
        return {key: arrays[key] for key in arrays.files}


def resolve_table_cache_dir(args, signal: str, out_dir: Path) -> Optional[Path]:
    if args.table_cache_dir:
        path = Path(args.table_cache_dir)
        if not path.is_absolute():
            path = bdt.PARTICLENETMD_DIR / path
        return path

    if args.no_bdt_table_cache or args.rebuild_cache:
        return None

    bdt_cache_dir = bdt.PARTICLENETMD_DIR / "BDT" / "Combined" / signal / "fold-4" / "tables"
    return bdt_cache_dir if bdt_cache_dir.exists() else None


def load_active_feature_names(args, table_cache_dir: Optional[Path]) -> List[str]:
    if getattr(args, "feature_names", None):
        path = Path(args.feature_names)
        if not path.is_absolute():
            path = bdt.PARTICLENETMD_DIR / path
        return bdt.load_feature_names(path)
    if table_cache_dir is not None:
        return bdt.load_feature_names(table_cache_dir / "feature_names.json")
    return list(bdt.FEATURE_NAMES)


def build_or_load_dnn_table(config, signal: str, split: str, fold_list: Sequence[int],
                            max_events_per_fold: Optional[int], workers: int,
                            out_dir: Path, rebuild_cache: bool,
                            source_cache_dir: Optional[Path],
                            pilot_events_per_class: Optional[int] = None) -> Dict[str, np.ndarray]:
    if source_cache_dir is not None and not rebuild_cache:
        source_path = source_cache_dir / f"{split}_{bdt.cache_suffix(max_events_per_fold, pilot_events_per_class)}.npz"
        if source_path.exists():
            print(f"  Loading {split} table cache from {source_path}", flush=True)
            return load_npz_table(source_path)
        print(f"  Table cache not found at {source_path}; rebuilding from graph datasets.", flush=True)

    return bdt.build_or_load_table(
        config, signal, split, fold_list, max_events_per_fold, workers, out_dir, rebuild_cache,
        pilot_events_per_class=pilot_events_per_class,
    )


def make_loader(table: Dict[str, np.ndarray], x_values: np.ndarray,
                batch_size: int, shuffle: bool, device: str) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(x_values.astype(np.float32)),
        torch.from_numpy(table["y"].astype(np.int64)),
        torch.from_numpy(table["weight"].astype(np.float32)),
        torch.from_numpy(table["mass1"].astype(np.float32)),
        torch.from_numpy(table["mass2"].astype(np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=device.startswith("cuda"),
    )


def run_epoch(model: nn.Module, loader: DataLoader, loss_fn, device: torch.device,
              optimizer=None, scheduler=None, scheduler_type: str = "ExponentialLR") -> Tuple[float, float, Dict[str, float]]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: List[float] = []
    ce_losses: List[float] = []
    disco_terms: List[float] = []
    preds: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    weights_seen: List[torch.Tensor] = []

    for x, y, weight, mass1, mass2 in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        weight = weight.to(device, non_blocking=True)
        mass1 = mass1.to(device, non_blocking=True)
        mass2 = mass2.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            logits = model(x)
            loss = loss_fn(logits, y, weight, mass1, mass2)
            if is_train:
                loss.backward()
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        pred = logits.argmax(dim=1)
        preds.append(pred.detach().cpu())
        labels.append(y.detach().cpu())
        weights_seen.append(weight.detach().cpu())
        if hasattr(loss_fn, "get_decomposed_losses"):
            decomposed = loss_fn.get_decomposed_losses()
            ce_losses.append(float(decomposed["ce_loss"]))
            disco_terms.append(float(decomposed["disco_term"]))

    total_loss = float(np.mean(losses)) if losses else float("inf")
    if is_train and scheduler is not None and hasattr(scheduler, "step"):
        if scheduler_type == "ReduceLROnPlateau":
            scheduler.step(total_loss)
        else:
            scheduler.step()

    accuracy = calculate_group_balanced_accuracy(
        torch.cat(preds),
        torch.cat(labels),
        torch.cat(weights_seen),
        use_groups=True,
        num_classes=len(CLASS_NAMES),
    )
    decomposed = {
        "ce_loss": float(np.mean(ce_losses)) if ce_losses else 0.0,
        "disco_term": float(np.mean(disco_terms)) if disco_terms else 0.0,
    }
    return total_loss, accuracy, decomposed


def train_dnn(train: Dict[str, np.ndarray], valid: Dict[str, np.ndarray],
              train_x: np.ndarray, valid_x: np.ndarray,
              args, config, hidden_layers: Sequence[int]) -> Tuple[TabularDNN, Dict[str, object]]:
    device = torch.device(args.device)
    model = TabularDNN(
        input_dim=train_x.shape[1],
        hidden_layers=hidden_layers,
        num_classes=len(CLASS_NAMES),
        dropout_p=float(config.get_training_parameters()["dropout_p"]),
    ).to(device)

    optim_config = config.get_optimization_config()
    train_params = config.get_training_parameters()
    disco_params = config.config.get("disco_parameters", {})
    loss_type = args.loss_type or train_params.get("loss_type", "disco")
    if loss_type == "disco":
        loss_fn = create_loss_function(
            "disco",
            num_classes=len(CLASS_NAMES),
            disco_lambda=float(args.disco_lambda if args.disco_lambda is not None else disco_params.get("disco_lambda", 0.05)),
        )
    else:
        loss_fn = MassAgnosticLossAdapter(
            create_loss_function(loss_type, num_classes=len(CLASS_NAMES))
        )
    optimizer = create_optimizer(
        optim_config["optimizer"],
        model.parameters(),
        float(optim_config["initLR"]),
        float(optim_config["weight_decay"]),
    )
    scheduler = create_scheduler(optim_config["scheduler"], optimizer, float(optim_config["initLR"]))

    train_loader = make_loader(train, train_x, args.batch_size, True, args.device)
    valid_loader = make_loader(valid, valid_x, args.batch_size, False, args.device)

    max_epochs = args.max_epochs if args.max_epochs is not None else int(train_params["max_epochs"])
    patience = args.early_stopping_patience if args.early_stopping_patience is not None else int(train_params["early_stopping_patience"])
    min_delta = float(args.early_stopping_min_delta)

    best_valid_loss = float("inf")
    best_epoch = -1
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0
    history: List[Dict[str, float]] = []
    start = time.time()

    for epoch in range(max_epochs):
        epoch_start = time.time()
        train_loss, train_acc, train_dec = run_epoch(
            model, train_loader, loss_fn, device,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_type=optim_config["scheduler"],
        )
        valid_loss, valid_acc, valid_dec = run_epoch(model, valid_loader, loss_fn, device)
        current_lr = float(optimizer.param_groups[0]["lr"])
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "valid_loss": valid_loss,
            "valid_acc": valid_acc,
            "train_ce_loss": train_dec["ce_loss"],
            "train_disco_term": train_dec["disco_term"],
            "valid_ce_loss": valid_dec["ce_loss"],
            "valid_disco_term": valid_dec["disco_term"],
            "learning_rate": current_lr,
            "epoch_time": time.time() - epoch_start,
        }
        history.append(entry)
        print(
            f"    epoch {epoch:03d}: train_loss={train_loss:.5f} valid_loss={valid_loss:.5f} "
            f"train_acc={train_acc:.4f} valid_acc={valid_acc:.4f} "
            f"ce={train_dec['ce_loss']:.5f} disco={train_dec['disco_term']:.5f}",
            flush=True,
        )

        if valid_loss < best_valid_loss - min_delta:
            best_valid_loss = valid_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"    early stopping at epoch {epoch}; best epoch {best_epoch}", flush=True)
                break

    model.load_state_dict(best_state)
    summary = {
        "best_epoch": best_epoch,
        "best_valid_loss": best_valid_loss,
        "epochs_completed": len(history),
        "total_training_time": time.time() - start,
        "training_history": history,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
    }
    return model, summary


def predict_scores(model: nn.Module, table: Dict[str, np.ndarray], x_values: np.ndarray,
                   batch_size: int, device_name: str) -> np.ndarray:
    device = torch.device(device_name)
    loader = make_loader(table, x_values, batch_size, False, device_name)
    model.eval()
    scores: List[np.ndarray] = []
    with torch.no_grad():
        for x, _y, _w, _m1, _m2 in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            scores.append(F.softmax(logits, dim=1).cpu().numpy())
    return np.concatenate(scores, axis=0)


def save_pickle_or_joblib(obj, path: Path) -> None:
    try:
        import joblib
        joblib.dump(obj, path)
    except Exception:
        with open(path, "wb") as handle:
            pickle.dump(obj, handle)


def model_root_color(model_name: str) -> int:
    if not bdt.HAS_ROOT_CMSSTYLE:
        return 1
    color_index = {"DNN": 0, "ParticleNet": 1}.get(model_name, 0)
    return bdt.palette_root_color(color_index)


def lr_root_color(model_name: str, class_idx: int) -> int:
    if not bdt.HAS_ROOT_CMSSTYLE:
        return 1
    offset = 0 if model_name == "DNN" else 4
    return bdt.palette_root_color(offset + class_idx)


def plot_roc_comparison(train: Dict[str, np.ndarray], test: Dict[str, np.ndarray],
                        dnn_train_scores: np.ndarray, dnn_test_scores: np.ndarray,
                        pn_train_scores: Optional[np.ndarray],
                        pn_test_scores: Optional[np.ndarray],
                        out_path: Path) -> Dict[str, object]:
    if bdt.HAS_ROOT_CMSSTYLE:
        return plot_roc_comparison_root(
            train, test, dnn_train_scores, dnn_test_scores,
            pn_train_scores, pn_test_scores, out_path,
        )

    roc = bdt.ROCCurveCalculator()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    summary: Dict[str, object] = {}
    curves = [
        ("DNN", "train", train, dnn_train_scores, "tab:blue", "--"),
        ("DNN", "test", test, dnn_test_scores, "tab:blue", "-"),
        ("ParticleNet", "train", train, pn_train_scores, "tab:orange", "--"),
        ("ParticleNet", "test", test, pn_test_scores, "tab:orange", "-"),
    ]

    for ax, bg_class in zip(axes, [1, 2, 3]):
        bg_name = CLASS_NAMES[bg_class]
        for model_name, split_name, table, scores, color, linestyle in curves:
            if scores is None:
                continue
            mask = (table["y"] == 0) | (table["y"] == bg_class)
            y_bin = (table["y"][mask] == 0).astype(int)
            lr = bdt.binary_lr(scores[mask], bg_class)
            fpr, tpr, auc = roc.calculate_roc_curve(y_bin, lr, table["weight"][mask])
            ax.plot(
                tpr, 1.0 - fpr,
                label=f"{model_name} {split_name} AUC={auc:.3f}",
                color=color,
                linestyle=linestyle,
                linewidth=2,
            )
            summary.setdefault(bg_name, {}).setdefault(model_name, {})[split_name] = float(auc)
        ax.set_title(f"Signal vs {bg_name}")
        ax.set_xlabel("Signal efficiency")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Background rejection")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return summary


def plot_roc_comparison_root(train: Dict[str, np.ndarray], test: Dict[str, np.ndarray],
                             dnn_train_scores: np.ndarray, dnn_test_scores: np.ndarray,
                             pn_train_scores: Optional[np.ndarray],
                             pn_test_scores: Optional[np.ndarray],
                             out_path: Path) -> Dict[str, object]:
    ROOT = bdt.ROOT
    CMS = bdt.CMS
    bdt.setup_root_cms_style()
    roc = bdt.ROCCurveCalculator()
    summary: Dict[str, object] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = CMS.cmsCanvas("", 0.0, 1.0, 0.0, 1.0, "Signal Efficiency",
                           "Background Efficiency", square=True, iPos=11, extraSpace=0.0)
    canvas.SetGrid()
    legend = CMS.cmsLeg(0.18, 0.55, 0.82, 0.78, textSize=0.024, columns=1)
    keepalive = []
    diag_graph = ROOT.TGraph(2)
    diag_graph.SetPoint(0, 0.0, 0.0)
    diag_graph.SetPoint(1, 1.0, 1.0)
    CMS.cmsObjectDraw(diag_graph, "L", LineColor=ROOT.kGray + 2,
                      LineWidth=1, LineStyle=ROOT.kDashed)
    keepalive.append(diag_graph)

    for model_name, train_scores, test_scores in [
        ("DNN", dnn_train_scores, dnn_test_scores),
        ("ParticleNet", pn_train_scores, pn_test_scores),
    ]:
        if train_scores is None or test_scores is None:
            continue
        color = model_root_color(model_name)
        for bg_class in [1, 2, 3]:
            bg_name = CLASS_NAMES[bg_class]
            style = bdt.ROC_BG_LINE_STYLE[bg_class]
            train_mask = (train["y"] == 0) | (train["y"] == bg_class)
            test_mask = (test["y"] == 0) | (test["y"] == bg_class)
            y_train = (train["y"][train_mask] == 0).astype(int)
            y_test = (test["y"][test_mask] == 0).astype(int)
            lr_train = bdt.binary_lr(train_scores[train_mask], bg_class)
            lr_test = bdt.binary_lr(test_scores[test_mask], bg_class)
            fpr_train, tpr_train, auc_train = roc.calculate_roc_curve(
                y_train, lr_train, train["weight"][train_mask]
            )
            fpr_test, tpr_test, auc_test = roc.calculate_roc_curve(
                y_test, lr_test, test["weight"][test_mask]
            )
            train_graph = bdt.make_roc_graph(tpr_train, fpr_train)
            test_graph = bdt.make_roc_graph(tpr_test, fpr_test)
            CMS.cmsObjectDraw(train_graph, "L", LineColor=color, LineWidth=1, LineStyle=style)
            CMS.cmsObjectDraw(test_graph, "L", LineColor=color, LineWidth=3 if model_name == "DNN" else 2, LineStyle=style)
            keepalive.extend([train_graph, test_graph])
            CMS.addToLegend(
                legend,
                (test_graph, f"{model_name} vs {bg_name}: AUC = {auc_test:.4f} ({auc_train:.4f})", "L"),
            )
            summary.setdefault(bg_name, {}).setdefault(model_name, {})["train"] = float(auc_train)
            summary.setdefault(bg_name, {}).setdefault(model_name, {})["test"] = float(auc_test)

    legend.Draw()
    CMS.drawText("color: model, style: background, thick: test",
                 posX=0.20, posY=0.48, font=42, align=0, size=0.026)
    canvas.RedrawAxis()
    canvas._keepalive = keepalive
    bdt.save_root_canvas(canvas, out_path)
    return summary


def plot_lr_distributions(train: Dict[str, np.ndarray], test: Dict[str, np.ndarray],
                          dnn_train_lr: np.ndarray, dnn_test_lr: np.ndarray,
                          pn_train_lr: Optional[np.ndarray],
                          pn_test_lr: Optional[np.ndarray],
                          out_dir: Path) -> None:
    if bdt.HAS_ROOT_CMSSTYLE:
        plot_lr_distributions_root(train, test, dnn_train_lr, dnn_test_lr, pn_train_lr, pn_test_lr, out_dir)
        return

    bins = np.linspace(0.0, 1.0, 31)
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [("DNN", dnn_train_lr, dnn_test_lr, "--", "-")]
    if pn_train_lr is not None and pn_test_lr is not None:
        models.append(("ParticleNet", pn_train_lr, pn_test_lr, ":", "-"))

    for class_idx, class_name in enumerate(CLASS_NAMES):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for model_name, train_lr, test_lr, train_style, test_style in models:
            color = "tab:blue" if model_name == "DNN" else "tab:orange"
            for split_name, table, values, linestyle in [
                ("train", train, train_lr, train_style),
                ("test", test, test_lr, test_style),
            ]:
                mask = table["y"] == class_idx
                if mask.sum() == 0:
                    continue
                hist, edges = bdt.normalized_hist(values[mask], table["weight"][mask], bins)
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax.step(centers, hist, where="mid", linewidth=2,
                        label=f"{model_name} {split_name}", color=color, linestyle=linestyle)
        ax.set_xlabel("LR_modified")
        ax.set_ylabel("Normalized")
        ax.set_title(class_name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"lr_{class_name}.png", dpi=150)
        plt.close(fig)


def plot_lr_distributions_root(train: Dict[str, np.ndarray], test: Dict[str, np.ndarray],
                               dnn_train_lr: np.ndarray, dnn_test_lr: np.ndarray,
                               pn_train_lr: Optional[np.ndarray],
                               pn_test_lr: Optional[np.ndarray],
                               out_dir: Path) -> None:
    ROOT = bdt.ROOT
    CMS = bdt.CMS
    bdt.setup_root_cms_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    models = [("DNN", dnn_train_lr, dnn_test_lr)]
    if pn_train_lr is not None and pn_test_lr is not None:
        models.append(("ParticleNet", pn_train_lr, pn_test_lr))

    for class_idx, class_name in enumerate(CLASS_NAMES):
        hists = []
        for model_name, train_lr, test_lr in models:
            color = lr_root_color(model_name, class_idx)
            train_style = ROOT.kDashed if model_name == "DNN" else ROOT.kDotted
            for split_name, table, lr_values, style in [
                ("train", train, train_lr, train_style),
                ("test", test, test_lr, ROOT.kSolid),
            ]:
                mask = table["y"] == class_idx
                if mask.sum() == 0:
                    continue
                hist = bdt.make_root_hist(
                    f"h_lr_{model_name}_{class_name}_{split_name}",
                    lr_values[mask],
                    table["weight"][mask],
                    30,
                    0.0,
                    1.0,
                )
                bdt.normalize_root_hist(hist)
                hist.SetLineColor(color)
                hist.SetLineStyle(style)
                hist.SetLineWidth(2)
                hist.SetMarkerSize(0)
                label = "PN" if model_name == "ParticleNet" else model_name
                hists.append((hist, f"{label} {split_name}", color, style))
        if not hists:
            continue
        ymax = max(hist.GetMaximum() for hist, _label, _color, _style in hists)
        canvas = CMS.cmsCanvas("", 0.0, 1.0, 0.0, max(0.01, ymax * 1.65),
                               "LR_{modified}", "Normalized", square=True,
                               iPos=11, extraSpace=0.0)
        canvas.SetGrid()
        legend = CMS.cmsLeg(0.48, 0.62, 0.92, 0.84, textSize=0.030, columns=2)
        for hist, label, color, style in hists:
            CMS.cmsObjectDraw(hist, "hist", LineColor=color, LineWidth=2, LineStyle=style)
            CMS.cmsObjectDraw(hist, "E0 SAME", LineColor=color, LineWidth=2,
                              LineStyle=style, MarkerColor=color, MarkerSize=0)
            CMS.addToLegend(legend, (hist, label, "L"))
        legend.Draw()
        CMS.drawText(class_name, posX=0.20, posY=0.70, font=62, align=0, size=0.035)
        canvas.RedrawAxis()
        bdt.save_root_canvas(canvas, out_dir / f"lr_{class_name}.png")


def process_signal(args, signal: str) -> None:
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    set_random_seed(args.random_state)
    config = load_sgl_config(args.config)
    train_params = config.get_training_parameters()
    loss_type = args.loss_type or train_params.get("loss_type", "disco")
    print(f"\n=== DNN training: {signal} ({loss_type}) ===", flush=True)

    if args.pilot:
        train_folds = [train_params["train_folds"][0]]
        valid_folds = train_params["valid_folds"]
        test_folds = train_params["test_folds"]
        pilot_events_per_class = args.pilot_events_per_class
        train_cap = None
        valid_cap = None
        test_cap = None
        print(
            f"  PILOT MODE: train_folds={train_folds}, valid_folds={valid_folds}, "
            f"test_folds={test_folds}, events/class/split={pilot_events_per_class}",
            flush=True,
        )
    else:
        train_folds = train_params["train_folds"]
        valid_folds = train_params["valid_folds"]
        test_folds = train_params["test_folds"]
        pilot_events_per_class = None
        if args.max_events_per_class is not None:
            train_cap = args.max_events_per_class
            valid_cap = args.max_events_per_class
            test_cap = args.max_events_per_class if args.cap_test else None
        else:
            train_cap = train_params.get("max_events_per_fold_per_class")
            valid_cap = train_params.get("max_events_per_fold_per_class")
            test_cap = None

    output_base = bdt.resolve_repo_path(getattr(args, "output_base", None), OUTPUT_BASE)
    out_dir = output_base / "Combined" / signal / "fold-4"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    table_cache_dir = resolve_table_cache_dir(args, signal, out_dir)
    if table_cache_dir is not None and not args.rebuild_cache:
        print(f"  Table cache source: {table_cache_dir}", flush=True)

    feature_names = load_active_feature_names(args, table_cache_dir)
    bdt.validate_feature_names(feature_names)
    with open(out_dir / "feature_names.json", "w") as handle:
        json.dump(feature_names, handle, indent=2)

    train = build_or_load_dnn_table(config, signal, "train", train_folds,
                                    train_cap, args.workers, out_dir, args.rebuild_cache,
                                    table_cache_dir,
                                    pilot_events_per_class=pilot_events_per_class)
    valid = build_or_load_dnn_table(config, signal, "valid", valid_folds,
                                    valid_cap, args.workers, out_dir, args.rebuild_cache,
                                    table_cache_dir,
                                    pilot_events_per_class=pilot_events_per_class)
    test = build_or_load_dnn_table(config, signal, "test", test_folds,
                                   test_cap, args.workers, out_dir, args.rebuild_cache,
                                   table_cache_dir,
                                   pilot_events_per_class=pilot_events_per_class)
    for split_name, table in [("train", train), ("valid", valid), ("test", test)]:
        if table["X"].shape[1] != len(feature_names):
            raise RuntimeError(
                f"{split_name} table has {table['X'].shape[1]} columns but "
                f"feature list has {len(feature_names)} entries"
            )
    print(f"  Table sizes: train={len(train['y'])}, valid={len(valid['y'])}, test={len(test['y'])}", flush=True)

    preprocessor = fit_preprocessor(train)
    save_pickle_or_joblib(preprocessor, out_dir / "preprocessor.joblib")
    train_x = transform_table(train, preprocessor)
    valid_x = transform_table(valid, preprocessor)
    test_x = transform_table(test, preprocessor)

    hidden_layers = parse_hidden_layers(args.hidden_layers, config.get_model_config()["nNodes"])
    print(f"  DNN input_dim={train_x.shape[1]} hidden_layers={hidden_layers} device={args.device}", flush=True)
    model, training_summary = train_dnn(train, valid, train_x, valid_x, args, config, hidden_layers)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": int(train_x.shape[1]),
                "hidden_layers": list(hidden_layers),
                "num_classes": len(CLASS_NAMES),
                "dropout_p": float(config.get_training_parameters()["dropout_p"]),
            },
            "training_summary": training_summary,
        },
        out_dir / "model.pt",
    )

    dnn_train_scores = predict_scores(model, train, train_x, args.batch_size, args.device)
    dnn_valid_scores = predict_scores(model, valid, valid_x, args.batch_size, args.device)
    dnn_test_scores = predict_scores(model, test, test_x, args.batch_size, args.device)

    pn_train_scores = None
    pn_test_scores = None
    if not args.skip_pn:
        pn_train_scores = bdt.evaluate_particle_net(
            config, signal, train_folds, train_cap, args.workers, args.device,
            pilot_events_per_class=pilot_events_per_class, split_name="train",
        )
        if pn_train_scores is not None and len(pn_train_scores) != len(train["y"]):
            raise RuntimeError(f"ParticleNet/train length mismatch: {len(pn_train_scores)} != {len(train['y'])}")
        pn_test_scores = bdt.evaluate_particle_net(
            config, signal, test_folds, test_cap, args.workers, args.device,
            pilot_events_per_class=pilot_events_per_class, split_name="test",
        )
        if pn_test_scores is not None and len(pn_test_scores) != len(test["y"]):
            raise RuntimeError(f"ParticleNet/test length mismatch: {len(pn_test_scores)} != {len(test['y'])}")

    thresholds = bdt.load_thresholds(signal)
    dnn_train_lr = bdt.compute_lr_modified(dnn_train_scores, train["era"], train["channel_id"], thresholds)
    dnn_test_lr = bdt.compute_lr_modified(dnn_test_scores, test["era"], test["channel_id"], thresholds)
    pn_train_lr = (
        bdt.compute_lr_modified(pn_train_scores, train["era"], train["channel_id"], thresholds)
        if pn_train_scores is not None else None
    )
    pn_test_lr = (
        bdt.compute_lr_modified(pn_test_scores, test["era"], test["channel_id"], thresholds)
        if pn_test_scores is not None else None
    )

    _, train_aucs = bdt.average_signal_vs_bg_auc(train["y"], dnn_train_scores, train["weight"])
    _, valid_aucs = bdt.average_signal_vs_bg_auc(valid["y"], dnn_valid_scores, valid["weight"])
    test_avg_auc, test_aucs = bdt.average_signal_vs_bg_auc(test["y"], dnn_test_scores, test["weight"])
    roc_summary = plot_roc_comparison(
        train, test, dnn_train_scores, dnn_test_scores, pn_train_scores, pn_test_scores,
        plots_dir / "roc_dnn_vs_particlenet.png",
    )

    corr = {
        "train": {"DNN": bdt.mass_correlation_metrics(dnn_train_scores, train)},
        "test": {"DNN": bdt.mass_correlation_metrics(dnn_test_scores, test)},
    }
    if pn_train_scores is not None:
        corr["train"]["ParticleNet"] = bdt.mass_correlation_metrics(pn_train_scores, train)
    if pn_test_scores is not None:
        corr["test"]["ParticleNet"] = bdt.mass_correlation_metrics(pn_test_scores, test)
    bdt.plot_mass_correlation(corr, plots_dir / "score_mass_correlation.png")
    plot_lr_distributions(train, test, dnn_train_lr, dnn_test_lr, pn_train_lr, pn_test_lr, plots_dir)
    bdt.plot_mass_shapes_by_lr_train_test(train, test, dnn_train_lr, dnn_test_lr, "DNN", plots_dir)
    if pn_train_lr is not None and pn_test_lr is not None:
        bdt.plot_mass_shapes_by_lr_train_test(train, test, pn_train_lr, pn_test_lr, "ParticleNet", plots_dir)

    np.savez_compressed(
        out_dir / "predictions_train.npz",
        y=train["y"], weight=train["weight"], mass1=train["mass1"], mass2=train["mass2"],
        era=train["era"], channel_id=train["channel_id"],
        dnn_scores=dnn_train_scores, dnn_lr=dnn_train_lr,
        pn_scores=pn_train_scores if pn_train_scores is not None else np.empty((0, 4)),
        pn_lr=pn_train_lr if pn_train_lr is not None else np.empty((0,)),
    )
    np.savez_compressed(
        out_dir / "predictions_test.npz",
        y=test["y"], weight=test["weight"], mass1=test["mass1"], mass2=test["mass2"],
        era=test["era"], channel_id=test["channel_id"],
        dnn_scores=dnn_test_scores, dnn_lr=dnn_test_lr,
        pn_scores=pn_test_scores if pn_test_scores is not None else np.empty((0, 4)),
        pn_lr=pn_test_lr if pn_test_lr is not None else np.empty((0,)),
    )

    summary = {
        "signal": signal,
        "backend": "torch.TabularDNN",
        "channel": "Combined",
        "pilot_mode": bool(args.pilot),
        "folds": {"train": train_folds, "valid": valid_folds, "test": test_folds},
        "caps": {
            "train": train_cap,
            "valid": valid_cap,
            "test": test_cap,
            "pilot_events_per_class": pilot_events_per_class,
        },
        "n_events": {"train": int(len(train["y"])), "valid": int(len(valid["y"])), "test": int(len(test["y"]))},
        "dnn_config": {
            "input_features": int(len(feature_names)),
            "input_dim_after_preprocessing": int(train_x.shape[1]),
            "hidden_layers": list(hidden_layers),
            "dropout_p": float(config.get_training_parameters()["dropout_p"]),
            "batch_size": int(args.batch_size),
            "optimizer": config.get_optimization_config()["optimizer"],
            "scheduler": config.get_optimization_config()["scheduler"],
            "initLR": float(config.get_optimization_config()["initLR"]),
            "weight_decay": float(config.get_optimization_config()["weight_decay"]),
            "loss_type": loss_type,
            "disco_lambda": (
                float(args.disco_lambda if args.disco_lambda is not None else config.config.get("disco_parameters", {}).get("disco_lambda", 0.05))
                if loss_type == "disco" else 0.0
            ),
            "early_stopping_patience": int(args.early_stopping_patience if args.early_stopping_patience is not None else config.get_training_parameters()["early_stopping_patience"]),
        },
        "preprocessing": {
            "table_cache_source": str(table_cache_dir) if table_cache_dir is not None else str(out_dir / "tables"),
            "imputation": "median fit on train only",
            "missing_indicators": True,
            "scaling": "StandardScaler fit on train only",
        },
        "training_summary": training_summary,
        "dnn_auc": {
            "train": train_aucs,
            "valid": valid_aucs,
            "test": test_aucs,
            "test_average": test_avg_auc,
        },
        "roc_comparison": roc_summary,
        "mass_correlation": corr,
        "feature_policy": {
            "source": "saved PyG graphs via trainBDT feature extraction",
            "table_cache_source": str(table_cache_dir) if table_cache_dir is not None else str(out_dir / "tables"),
            "excluded": [
                "mass1",
                "mass2",
                "dimuon mass",
                "all invariant masses",
                "explicit channel flags",
                "explicit object counts",
                "HT",
                "ST",
            ],
        },
    }
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"  Best valid loss: {training_summary['best_valid_loss']:.5f}", flush=True)
    print(f"  Test avg AUC: {test_avg_auc:.4f}", flush=True)
    print(f"=== Done: {out_dir} ===", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signal", choices=SIGNALS, help="Signal mass point")
    parser.add_argument("--all", action="store_true", help="Run all configured comparison signals")
    parser.add_argument("--config", default=None, help="Path to SglConfig JSON")
    parser.add_argument("--workers", type=int, default=8, help="Parallel .pt loading workers")
    parser.add_argument("--device", default="cuda", help="Torch device, default cuda")
    parser.add_argument("--skip-pn", action="store_true", help="Skip ParticleNet inference/comparison")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild cached DNN tables")
    parser.add_argument("--table-cache-dir", default=None,
                        help="Directory containing train/valid/test table .npz files. "
                             "Relative paths are resolved under ParticleNetMD/.")
    parser.add_argument("--no-bdt-table-cache", action="store_true",
                        help="Do not auto-load matching BDT table caches when present")
    parser.add_argument("--feature-names", default=None,
                        help="JSON feature list for the active table cache")
    parser.add_argument("--output-base", default=None,
                        help="Output base directory. Relative paths are resolved under ParticleNetMD/.")
    parser.add_argument("--pilot", action="store_true", help="Run full workflow on a small sample")
    parser.add_argument("--pilot-events-per-class", type=int, default=250)
    parser.add_argument("--max-events-per-class", type=int, default=None)
    parser.add_argument("--cap-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--hidden-layers", default=None,
                        help="Comma-separated DNN widths. Default is a wide config derived from SglConfig nNodes.")
    parser.add_argument("--loss-type", choices=["weighted_ce", "disco"], default=None,
                        help="Training loss. Defaults to training_parameters.loss_type from config.")
    parser.add_argument("--disco-lambda", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.all:
        targets = SIGNALS
    elif args.signal:
        targets = [args.signal]
    else:
        targets = [SIGNALS[0]]
    for signal in targets:
        process_signal(args, signal)


if __name__ == "__main__":
    main()
