#!/usr/bin/env python3
"""
Train and compare a tabular BDT baseline against the GA-best ParticleNet model.

The BDT inputs are extracted from the saved PyG graph datasets so the event
selection, folds, augmentation choices, labels, and weights match the
ParticleNetMD workflow. Dimuon masses are kept only for post-training
decorrelation diagnostics and are never included as BDT features.

Example:
    python python/trainBDT.py --signal MHc160_MA85
    python python/trainBDT.py --signal MHc160_MA85 --max-events-per-class 2000 --skip-pn
    python python/trainBDT.py --all
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils import resample, shuffle
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

from SglConfig import load_sgl_config
from MultiClassModels import create_multiclass_model
from Preprocess import GraphDataset
from ROCCurveCalculator import ROCCurveCalculator
from WeightedLoss import distance_correlation


SIGNALS = ["MHc100_MA95", "MHc130_MA90", "MHc160_MA85"]
CLASS_NAMES = ["signal", "nonprompt", "diboson", "ttX"]
ERAS = ["2016preVFP", "2016postVFP", "2017", "2018",
        "2022", "2022EE", "2023", "2023BPix"]
ERA_TO_INDEX = {era: idx for idx, era in enumerate(ERAS)}
CHANNEL_ID_TO_NAME = {0: "Run1E2Mu", 1: "Run3Mu"}
CHANNEL_TO_THRESHOLD_KEY = {"Run1E2Mu": "SR1E2Mu", "Run3Mu": "SR3Mu"}

PARTICLENETMD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PARTICLENETMD_DIR.parent
DATASET_ROOT = PARTICLENETMD_DIR / "dataset" / "samples"
OUTPUT_BASE = PARTICLENETMD_DIR / "BDT"
THRESHOLDS_DIR = REPO_ROOT / "SignalRegionStudyV2" / "configs" / "thresholds"

COMMON_TOOLS_DIR = REPO_ROOT / "Common" / "Tools"
if COMMON_TOOLS_DIR.exists():
    sys.path.insert(0, str(COMMON_TOOLS_DIR))

try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
    import cmsstyle as CMS
    from plotter import KinematicCanvasWithRatio, PALETTE as PLOTTER_PALETTE, PALETTE_LONG as PLOTTER_PALETTE_LONG
    HAS_ROOT_CMSSTYLE = True
except ImportError:
    ROOT = None
    CMS = None
    KinematicCanvasWithRatio = None
    PLOTTER_PALETTE = []
    PLOTTER_PALETTE_LONG = []
    HAS_ROOT_CMSSTYLE = False

ROOT_CLASS_COLORS = ["#5790fc", "#f89c20", "#e42536", "#964a8b"]
ROOT_MODEL_LINE_WIDTH = {"BDT": 3, "ParticleNet": 2}
PLOTTER_PALETTE_HEX = [
    "#5790fc",
    "#f89c20",
    "#e42536",
    "#964a8b",
    "#9c9ca1",
    "#7a21dd",
    "#3f90da",
    "#ffa90e",
    "#bd1f01",
    "#94a4a2",
]
LR_COLOR_INDEX = {
    "BDT": [0, 1, 2, 3],
    "ParticleNet": [4, 5, 6, 7],
}
ROC_MODEL_COLOR_INDEX = {"BDT": 0, "ParticleNet": 1}
ROC_BG_LINE_STYLE = {
    1: ROOT.kSolid if ROOT is not None else "-",
    2: ROOT.kDashed if ROOT is not None else "--",
    3: ROOT.kDotted if ROOT is not None else ":",
}


def delta_phi(phi1: float, phi2: float) -> float:
    return math.atan2(math.sin(phi1 - phi2), math.cos(phi1 - phi2))


def delta_r(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    return math.hypot(eta1 - eta2, delta_phi(phi1, phi2))


def make_feature_names() -> List[str]:
    names: List[str] = []
    names.extend([f"era_{era}" for era in ERAS])

    for i in range(1, 4):
        names.extend([
            f"lep{i}_pt",
            f"lep{i}_eta",
            f"lep{i}_phi",
            f"lep{i}_charge",
            f"lep{i}_isMuon",
            f"lep{i}_isElectron",
        ])

    for i in range(1, 7):
        names.extend([
            f"jet{i}_pt",
            f"jet{i}_eta",
            f"jet{i}_phi",
            f"jet{i}_charge",
            f"jet{i}_btagged",
        ])

    names.extend(["ptmiss_proxy", "ptmiss_phi_proxy"])
    names.extend([f"mt_lep{i}_ptmiss" for i in range(1, 4)])
    names.extend([f"os_dimu{i}_pt" for i in range(1, 3)])
    names.extend(["dR_lep1_lep2", "dR_lep1_lep3", "dR_lep2_lep3"])
    names.extend([f"dR_lep{i}_jet{j}" for i in range(1, 4) for j in range(1, 7)])
    return names


FEATURE_NAMES = make_feature_names()


def infer_channel_id(x: torch.Tensor) -> int:
    n_mu = int((x[:, 5] > 0.5).sum().item())
    n_ele = int((x[:, 6] > 0.5).sum().item())
    if n_ele >= 1 and n_mu >= 2:
        return 0
    if n_mu >= 3:
        return 1
    return 0


def extract_features(data) -> Tuple[np.ndarray, Dict[str, object]]:
    x = data.x.detach().cpu().numpy().astype(np.float64)
    px = x[:, 1]
    py = x[:, 2]
    pz = x[:, 3]
    charge = x[:, 4]
    is_mu = x[:, 5] > 0.5
    is_ele = x[:, 6] > 0.5
    is_jet = x[:, 7] > 0.5
    is_bjet = x[:, 8] > 0.5

    pt = np.hypot(px, py)
    p = np.sqrt(px * px + py * py + pz * pz)
    eta = np.arctanh(np.clip(pz / (p + 1e-10), -1.0 + 1e-7, 1.0 - 1e-7))
    phi = np.arctan2(py, px)
    row = np.full(len(FEATURE_NAMES), np.nan, dtype=np.float32)
    pos = 0

    graph_input = getattr(data, "graphInput", None)
    if graph_input is not None:
        era_vec = graph_input.detach().cpu().numpy().reshape(-1)
        row[pos:pos + len(ERAS)] = era_vec[:len(ERAS)]
    else:
        era = getattr(data, "era", "")
        if era in ERA_TO_INDEX:
            row[pos + ERA_TO_INDEX[era]] = 1.0
    pos += len(ERAS)

    lep_indices = np.where(is_mu | is_ele)[0]
    lep_indices = lep_indices[np.argsort(-pt[lep_indices])]
    lep_slots: List[Optional[int]] = []
    for slot in range(3):
        idx = int(lep_indices[slot]) if slot < len(lep_indices) else None
        lep_slots.append(idx)
        if idx is not None:
            row[pos:pos + 6] = [
                pt[idx],
                eta[idx],
                phi[idx],
                charge[idx],
                float(is_mu[idx]),
                float(is_ele[idx]),
            ]
        pos += 6

    jet_indices = np.where(is_jet)[0]
    jet_indices = jet_indices[np.argsort(-pt[jet_indices])]
    jet_slots: List[Optional[int]] = []
    for slot in range(6):
        idx = int(jet_indices[slot]) if slot < len(jet_indices) else None
        jet_slots.append(idx)
        if idx is not None:
            row[pos:pos + 5] = [
                pt[idx],
                eta[idx],
                phi[idx],
                charge[idx],
                float(is_bjet[idx]),
            ]
        pos += 5

    ptmiss_px = -float(px.sum())
    ptmiss_py = -float(py.sum())
    ptmiss = math.hypot(ptmiss_px, ptmiss_py)
    ptmiss_phi = math.atan2(ptmiss_py, ptmiss_px)
    row[pos:pos + 2] = [ptmiss, ptmiss_phi]
    pos += 2

    for idx in lep_slots:
        if idx is not None:
            dphi = delta_phi(phi[idx], ptmiss_phi)
            row[pos] = math.sqrt(max(0.0, 2.0 * pt[idx] * ptmiss * (1.0 - math.cos(dphi))))
        pos += 1

    mu_indices = np.where(is_mu)[0]
    os_pair_pts: List[float] = []
    for i, idx1 in enumerate(mu_indices):
        for idx2 in mu_indices[i + 1:]:
            if charge[idx1] * charge[idx2] < 0:
                os_pair_pts.append(float(math.hypot(px[idx1] + px[idx2], py[idx1] + py[idx2])))
    os_pair_pts.sort(reverse=True)
    for slot in range(2):
        if slot < len(os_pair_pts):
            row[pos] = os_pair_pts[slot]
        pos += 1

    for a, b in [(0, 1), (0, 2), (1, 2)]:
        ia = lep_slots[a]
        ib = lep_slots[b]
        if ia is not None and ib is not None:
            row[pos] = delta_r(float(eta[ia]), float(phi[ia]), float(eta[ib]), float(phi[ib]))
        pos += 1

    for lep_idx in lep_slots:
        for jet_idx in jet_slots:
            if lep_idx is not None and jet_idx is not None:
                row[pos] = delta_r(
                    float(eta[lep_idx]),
                    float(phi[lep_idx]),
                    float(eta[jet_idx]),
                    float(phi[jet_idx]),
                )
            pos += 1

    if pos != len(FEATURE_NAMES):
        raise RuntimeError(f"Feature extraction length mismatch: {pos} != {len(FEATURE_NAMES)}")

    era = getattr(data, "era", "")
    if not era and graph_input is not None:
        idx = int(np.argmax(row[:len(ERAS)]))
        era = ERAS[idx] if row[idx] > 0.5 else ""

    meta = {
        "era": str(era),
        "channel_id": infer_channel_id(data.x.detach().cpu()),
        "mass1": float(data.mass1.item()) if hasattr(data, "mass1") else -1.0,
        "mass2": float(data.mass2.item()) if hasattr(data, "mass2") else -1.0,
    }
    return row, meta


def _load_pt_file(path: str):
    if not os.path.exists(path):
        return []
    dataset = torch.load(path, weights_only=False)
    return dataset.data_list if hasattr(dataset, "data_list") else []


def channel_from_dataset_path(path: str) -> str:
    name = Path(path).name
    if "_fold-" not in name:
        return ""
    return name.split("_fold-", 1)[0]


def file_specs(signal_full: str, bg_groups: Dict[str, List[str]],
               channel: str, fold_list: Sequence[int]) -> List[Tuple[str, int, int]]:
    sub_channels = ["Run1E2Mu", "Run3Mu"] if channel == "Combined" else [channel]
    specs: List[Tuple[str, int, int]] = []
    for fold in fold_list:
        for ch in sub_channels:
            specs.append((
                str(DATASET_ROOT / "signals" / signal_full / f"{ch}_fold-{fold}.pt"),
                0,
                fold,
            ))
    for group_idx, (_group_name, sample_list) in enumerate(bg_groups.items()):
        label = group_idx + 1
        for sample_name in sample_list:
            for fold in fold_list:
                for ch in sub_channels:
                    specs.append((
                        str(DATASET_ROOT / "backgrounds" / sample_name / f"{ch}_fold-{fold}.pt"),
                        label,
                        fold,
                    ))
    return specs


def apply_class_weight_balance(data_list: List[object]) -> None:
    class_w: Dict[int, float] = {}
    class_abs_w: Dict[int, float] = {}
    for data in data_list:
        label = int(data.y.item())
        weight = float(data.weight.item())
        class_w[label] = class_w.get(label, 0.0) + weight
        class_abs_w[label] = class_abs_w.get(label, 0.0) + abs(weight)

    positive_totals = [w for w in class_w.values() if w > 0]
    use_signed = len(positive_totals) == len(class_w)
    totals = class_w if use_signed else class_abs_w
    max_total = max(totals.values()) if totals else 1.0

    for data in data_list:
        label = int(data.y.item())
        denom = totals.get(label, 0.0)
        if denom > 0:
            data.weight = data.weight * (max_total / denom)


def load_split_data(config, signal: str, fold_list: Sequence[int],
                    max_events_per_fold: Optional[int], workers: int,
                    pilot_events_per_class: Optional[int] = None,
                    random_state: int = 42) -> List[object]:
    dataset_config = config.get_dataset_config()
    bg_config = config.get_background_config()

    signal_full = f"{dataset_config['signal_prefix']}{signal}"
    bg_prefix = dataset_config["background_prefix"]
    bg_groups = {
        group_name: [bg_prefix + sample for sample in samples]
        for group_name, samples in bg_config["background_groups"].items()
    }

    specs = file_specs(signal_full, bg_groups, "Combined", fold_list)
    by_label_fold: Dict[Tuple[int, int], List[object]] = {}

    if pilot_events_per_class is not None:
        label_totals = {label: 0 for label in range(len(bg_groups) + 1)}
        pilot_channels = ["Run1E2Mu", "Run3Mu"]
        per_channel_cap = int(math.ceil(pilot_events_per_class / len(pilot_channels)))
        label_channel_totals = {
            (label, ch): 0
            for label in range(len(bg_groups) + 1)
            for ch in pilot_channels
        }

        for path, label, fold in specs:
            if label_totals[label] >= pilot_events_per_class:
                continue
            ch = channel_from_dataset_path(path)
            if ch in pilot_channels and label_channel_totals[(label, ch)] >= per_channel_cap:
                continue

            data_list = _load_pt_file(path)
            remaining = pilot_events_per_class - label_totals[label]
            if ch in pilot_channels:
                remaining = min(remaining, per_channel_cap - label_channel_totals[(label, ch)])
            if len(data_list) > remaining:
                data_list = resample(
                    data_list,
                    n_samples=remaining,
                    replace=False,
                    random_state=random_state + label + fold,
                )
            by_label_fold.setdefault((label, fold), []).extend(data_list)
            label_totals[label] += len(data_list)
            if ch in pilot_channels:
                label_channel_totals[(label, ch)] += len(data_list)

            if all(count >= pilot_events_per_class for count in label_totals.values()):
                break
    else:
        rng = np.random.default_rng(random_state)
        seen_by_label_fold: Dict[Tuple[int, int], int] = {}
        for spec_idx, (path, label, fold) in enumerate(specs, start=1):
            data_list = _load_pt_file(path)
            key = (label, fold)
            selected = by_label_fold.setdefault(key, [])
            seen = seen_by_label_fold.get(key, 0)

            if max_events_per_fold is None:
                selected.extend(data_list)
                seen_by_label_fold[key] = seen + len(data_list)
            else:
                for data in data_list:
                    seen += 1
                    if len(selected) < max_events_per_fold:
                        selected.append(data)
                    else:
                        replace_idx = int(rng.integers(0, seen))
                        if replace_idx < max_events_per_fold:
                            selected[replace_idx] = data
                seen_by_label_fold[key] = seen

            if spec_idx % 10 == 0 or spec_idx == len(specs):
                kept = sum(len(items) for items in by_label_fold.values())
                seen_total = sum(seen_by_label_fold.values())
                print(
                    f"    loaded {spec_idx}/{len(specs)} files "
                    f"(seen={seen_total}, kept={kept})",
                    flush=True,
                )

    all_data: List[object] = []
    for (label, _fold), data_list in sorted(by_label_fold.items()):
        for data in data_list:
            data.y = torch.tensor(label, dtype=torch.long)
        if max_events_per_fold and len(data_list) > max_events_per_fold:
            data_list = resample(
                data_list,
                n_samples=max_events_per_fold,
                replace=False,
                random_state=random_state,
            )
        all_data.extend(data_list)

    apply_class_weight_balance(all_data)
    return list(shuffle(all_data, random_state=random_state))


def cache_suffix(max_events_per_fold: Optional[int],
                 pilot_events_per_class: Optional[int] = None) -> str:
    if pilot_events_per_class is not None:
        return f"pilot{pilot_events_per_class}"
    return f"cap{max_events_per_fold}" if max_events_per_fold else "full"


def table_path(out_dir: Path, split: str, max_events_per_fold: Optional[int],
               pilot_events_per_class: Optional[int] = None) -> Path:
    return out_dir / "tables" / f"{split}_{cache_suffix(max_events_per_fold, pilot_events_per_class)}.npz"


def build_or_load_table(config, signal: str, split: str, fold_list: Sequence[int],
                        max_events_per_fold: Optional[int], workers: int,
                        out_dir: Path, rebuild_cache: bool,
                        pilot_events_per_class: Optional[int] = None) -> Dict[str, np.ndarray]:
    path = table_path(out_dir, split, max_events_per_fold, pilot_events_per_class)
    if path.exists() and not rebuild_cache:
        with np.load(path, allow_pickle=True) as arrays:
            return {key: arrays[key] for key in arrays.files}

    print(f"  Loading {split} graph data for folds {list(fold_list)}...", flush=True)
    data_list = load_split_data(
        config,
        signal,
        fold_list,
        max_events_per_fold,
        workers,
        pilot_events_per_class=pilot_events_per_class,
    )
    print(f"  Extracting {split} BDT table from {len(data_list)} events...", flush=True)

    x_rows: List[np.ndarray] = []
    y: List[int] = []
    weights_signed: List[float] = []
    weights_fit: List[float] = []
    mass1: List[float] = []
    mass2: List[float] = []
    eras: List[str] = []
    channel_ids: List[int] = []

    for data in data_list:
        row, meta = extract_features(data)
        weight = float(data.weight.item())
        x_rows.append(row)
        y.append(int(data.y.item()))
        weights_signed.append(weight)
        weights_fit.append(abs(weight))
        mass1.append(float(meta["mass1"]))
        mass2.append(float(meta["mass2"]))
        eras.append(str(meta["era"]))
        channel_ids.append(int(meta["channel_id"]))

    arrays = {
        "X": np.vstack(x_rows).astype(np.float32),
        "y": np.asarray(y, dtype=np.int64),
        "weight": np.asarray(weights_signed, dtype=np.float64),
        "fit_weight": np.asarray(weights_fit, dtype=np.float64),
        "mass1": np.asarray(mass1, dtype=np.float32),
        "mass2": np.asarray(mass2, dtype=np.float32),
        "era": np.asarray(eras, dtype=object),
        "channel_id": np.asarray(channel_ids, dtype=np.int32),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return arrays


def proba_with_all_classes(model, X: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(X)
    out = np.zeros((X.shape[0], len(CLASS_NAMES)), dtype=np.float64)
    for idx, cls in enumerate(model.classes_):
        out[:, int(cls)] = raw[:, idx]
    return out


def binary_lr(scores: np.ndarray, bg_class: int) -> np.ndarray:
    sig = scores[:, 0]
    bg = scores[:, bg_class]
    return sig / (sig + bg + 1e-12)


def average_signal_vs_bg_auc(y_true: np.ndarray, scores: np.ndarray,
                             weights: np.ndarray) -> Tuple[float, Dict[str, float]]:
    roc = ROCCurveCalculator()
    aucs: Dict[str, float] = {}
    for bg_class in [1, 2, 3]:
        mask = (y_true == 0) | (y_true == bg_class)
        y_bin = (y_true[mask] == 0).astype(int)
        lr = binary_lr(scores[mask], bg_class)
        _, _, auc = roc.calculate_roc_curve(y_bin, lr, weights[mask])
        aucs[CLASS_NAMES[bg_class]] = float(auc)
    return float(np.mean(list(aucs.values()))), aucs


def train_bdt(train: Dict[str, np.ndarray], valid: Dict[str, np.ndarray],
              random_state: int) -> Tuple[HistGradientBoostingClassifier, Dict[str, object]]:
    grid = []
    for learning_rate in [0.03, 0.05, 0.08]:
        for max_leaf_nodes in [31, 63]:
            for l2_regularization in [0.0, 0.01]:
                grid.append({
                    "learning_rate": learning_rate,
                    "max_leaf_nodes": max_leaf_nodes,
                    "l2_regularization": l2_regularization,
                })

    best_model: Optional[HistGradientBoostingClassifier] = None
    best_record: Optional[Dict[str, object]] = None
    records: List[Dict[str, object]] = []

    for idx, params in enumerate(grid):
        print(f"  Grid {idx + 1}/{len(grid)}: {params}", flush=True)
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=random_state,
            **params,
        )
        model.fit(train["X"], train["y"], sample_weight=train["fit_weight"])
        valid_scores = proba_with_all_classes(model, valid["X"])
        avg_auc, aucs = average_signal_vs_bg_auc(valid["y"], valid_scores, valid["weight"])
        record = {
            "params": params,
            "n_iter": int(getattr(model, "n_iter_", 0)),
            "valid_avg_auc": avg_auc,
            "valid_auc_by_background": aucs,
        }
        records.append(record)
        if best_record is None or avg_auc > float(best_record["valid_avg_auc"]):
            best_model = model
            best_record = record

    if best_model is None or best_record is None:
        raise RuntimeError("BDT grid search did not produce a model")

    return best_model, {"best": best_record, "grid": records}


def save_model(model, path: Path) -> None:
    try:
        import joblib
        joblib.dump(model, path)
    except Exception:
        with open(path, "wb") as handle:
            pickle.dump(model, handle)


def load_particle_net_model(model_path: Path, info_path: Path, device: str):
    with open(info_path) as handle:
        info = json.load(handle)
    hp = info["hyperparameters"]
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint

    num_hidden = hp["num_hidden"]
    num_graph_features = hp.get("num_graph_features", 8)
    if "dense1.weight" in state:
        inferred = state["dense1.weight"].shape[1] - 3 * num_hidden
        if inferred != num_graph_features:
            num_graph_features = inferred

    model = create_multiclass_model(
        model_type=hp.get("model_type", "ParticleNet"),
        num_node_features=hp.get("num_node_features", 9),
        num_graph_features=num_graph_features,
        num_classes=hp.get("num_classes", 4),
        num_hidden=num_hidden,
        dropout_p=hp.get("dropout_p", 0.4),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, hp


def evaluate_particle_net(config, signal: str, fold_list: Sequence[int],
                          max_events_per_fold: Optional[int], workers: int,
                          device: str, batch_size: int = 512,
                          pilot_events_per_class: Optional[int] = None,
                          split_name: str = "split") -> Optional[np.ndarray]:
    model_dir = PARTICLENETMD_DIR / "GAOptim" / "Combined" / signal / "fold-4" / "best_model"
    model_path = model_dir / "model.pt"
    info_path = model_dir / "model_info.json"
    if not model_path.exists() or not info_path.exists():
        print(f"  ParticleNet reference missing under {model_dir}; skipping PN comparison.", flush=True)
        return None

    print(f"  Loading ParticleNet reference and matching {split_name} split...", flush=True)
    data_list = load_split_data(
        config,
        signal,
        fold_list,
        max_events_per_fold,
        workers,
        pilot_events_per_class=pilot_events_per_class,
    )
    model, hp = load_particle_net_model(model_path, info_path, device)
    loader = DataLoader(GraphDataset(data_list), batch_size=hp.get("batch_size", batch_size), shuffle=False)

    scores: List[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.graphInput, batch.batch)
            scores.append(F.softmax(logits, dim=1).cpu().numpy())

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return np.concatenate(scores, axis=0)


def load_thresholds(signal: str) -> Dict[str, object]:
    path = THRESHOLDS_DIR / f"{signal}.json"
    if not path.exists():
        print(f"  Threshold file not found: {path}; using unit LR weights.", flush=True)
        return {}
    with open(path) as handle:
        return json.load(handle)


def compute_lr_modified(scores: np.ndarray, eras: np.ndarray, channel_ids: np.ndarray,
                        thresholds: Dict[str, object]) -> np.ndarray:
    out = np.zeros(scores.shape[0], dtype=np.float64)
    for i in range(scores.shape[0]):
        era = str(eras[i])
        channel = CHANNEL_ID_TO_NAME.get(int(channel_ids[i]), "Run1E2Mu")
        weights = {"nonprompt": 1.0, "diboson": 1.0, "ttX": 1.0}
        if era in thresholds:
            key = CHANNEL_TO_THRESHOLD_KEY[channel]
            if key in thresholds[era]:
                weights.update(thresholds[era][key].get("weights", {}))

        denom = (
            scores[i, 0]
            + float(weights["nonprompt"]) * scores[i, 1]
            + float(weights["diboson"]) * scores[i, 2]
            + float(weights["ttX"]) * scores[i, 3]
        )
        out[i] = scores[i, 0] / denom if denom > 0 else 0.0
    return out


def compute_disco(score: np.ndarray, mass: np.ndarray, weights: np.ndarray,
                  max_events: int = 5000) -> float:
    valid = mass > 0
    if valid.sum() < 2:
        return 0.0
    score = score[valid]
    mass = mass[valid]
    weights = weights[valid]
    if score.size > max_events:
        rng = np.random.default_rng(42)
        idx = rng.choice(score.size, max_events, replace=False)
        score = score[idx]
        mass = mass[idx]
        weights = weights[idx]
    return float(distance_correlation(
        torch.tensor(score, dtype=torch.float32),
        torch.tensor(mass, dtype=torch.float32),
        torch.tensor(weights, dtype=torch.float32),
    ).item())


def weighted_pearson(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y) & (weights > 0)
    if valid.sum() < 2:
        return float("nan")
    x = x[valid]
    y = y[valid]
    w = weights[valid]
    w = w / np.sum(w)
    mx = np.sum(w * x)
    my = np.sum(w * y)
    cov = np.sum(w * (x - mx) * (y - my))
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(cov / math.sqrt(vx * vy))


def weighted_spearman(x: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return float("nan")
    xr = np.empty(valid.sum(), dtype=np.float64)
    yr = np.empty(valid.sum(), dtype=np.float64)
    xv = x[valid]
    yv = y[valid]
    xr[np.argsort(np.argsort(xv))] = np.arange(valid.sum(), dtype=np.float64)
    yr[np.argsort(np.argsort(yv))] = np.arange(valid.sum(), dtype=np.float64)
    return weighted_pearson(xr, yr, np.abs(weights[valid]))


def mass_correlation_metrics(scores: np.ndarray, table: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    p_sig = scores[:, 0]
    weights_abs = np.abs(table["weight"])
    metrics: Dict[str, Dict[str, float]] = {}
    for mass_name in ["mass1", "mass2"]:
        mass = table[mass_name]
        valid = mass > 0
        metrics[mass_name] = {
            "disco": compute_disco(p_sig, mass, table["weight"]),
            "pearson_absw": weighted_pearson(p_sig[valid], mass[valid], weights_abs[valid]),
            "spearman_absw": weighted_spearman(p_sig[valid], mass[valid], weights_abs[valid]),
            "n_valid": int(valid.sum()),
        }
    return metrics


def plot_roc_comparison(train: Dict[str, np.ndarray], test: Dict[str, np.ndarray],
                        bdt_train_scores: np.ndarray, bdt_test_scores: np.ndarray,
                        pn_train_scores: Optional[np.ndarray],
                        pn_test_scores: Optional[np.ndarray],
                        out_path: Path) -> Dict[str, object]:
    if HAS_ROOT_CMSSTYLE:
        return plot_roc_comparison_root(
            train, test, bdt_train_scores, bdt_test_scores, pn_train_scores, pn_test_scores, out_path
        )

    roc = ROCCurveCalculator()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    summary: Dict[str, object] = {}

    for ax, bg_class in zip(axes, [1, 2, 3]):
        bg_name = CLASS_NAMES[bg_class]

        curves = [
            ("BDT", "train", train, bdt_train_scores, "tab:blue", "--"),
            ("BDT", "test", test, bdt_test_scores, "tab:blue", "-"),
            ("ParticleNet", "train", train, pn_train_scores, "tab:orange", "--"),
            ("ParticleNet", "test", test, pn_test_scores, "tab:orange", "-"),
        ]
        for model_name, split_name, table, scores, color, linestyle in curves:
            if scores is None:
                continue
            y = table["y"]
            weights = table["weight"]
            mask = (y == 0) | (y == bg_class)
            y_bin = (y[mask] == 0).astype(int)
            lr = binary_lr(scores[mask], bg_class)
            fpr, tpr, auc = roc.calculate_roc_curve(y_bin, lr, weights[mask])
            ax.plot(
                tpr,
                1.0 - fpr,
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


def make_roc_graph(tpr: np.ndarray, fpr: np.ndarray, use_rejection: bool = False) -> "ROOT.TGraph":
    graph = ROOT.TGraph(len(fpr))
    y_values = 1.0 - fpr if use_rejection else fpr
    for idx, (x_value, y_value) in enumerate(zip(tpr, y_values)):
        graph.SetPoint(idx, float(x_value), float(y_value))
    return graph


def plot_roc_comparison_root(train: Dict[str, np.ndarray], test: Dict[str, np.ndarray],
                             bdt_train_scores: np.ndarray, bdt_test_scores: np.ndarray,
                             pn_train_scores: Optional[np.ndarray],
                             pn_test_scores: Optional[np.ndarray],
                             out_path: Path) -> Dict[str, object]:
    setup_root_cms_style()
    roc = ROCCurveCalculator()
    summary: Dict[str, object] = {}
    out_path.parent.mkdir(parents=True, exist_ok=True)

    canvas = CMS.cmsCanvas(
        "",
        0.0,
        1.0,
        0.0,
        1.0,
        "Signal Efficiency",
        "Background Efficiency",
        square=True,
        iPos=11,
        extraSpace=0.0,
    )
    canvas.SetGrid()
    legend = CMS.cmsLeg(0.18, 0.55, 0.82, 0.78, textSize=0.024, columns=1)

    keepalive = []
    diag_graph = ROOT.TGraph(2)
    diag_graph.SetPoint(0, 0.0, 0.0)
    diag_graph.SetPoint(1, 1.0, 1.0)
    CMS.cmsObjectDraw(diag_graph, "L", LineColor=ROOT.kGray + 2, LineWidth=1, LineStyle=ROOT.kDashed)
    keepalive.append(diag_graph)

    models = [
        ("BDT", bdt_train_scores, bdt_test_scores),
        ("ParticleNet", pn_train_scores, pn_test_scores),
    ]
    for model_idx, (model_name, train_scores, test_scores) in enumerate(models):
        if train_scores is None or test_scores is None:
            continue
        model_color = roc_model_root_color(model_name)
        for bg_class in [1, 2, 3]:
            bg_name = CLASS_NAMES[bg_class]
            bg_line_style = ROC_BG_LINE_STYLE[bg_class]

            y_train = train["y"]
            train_mask = (y_train == 0) | (y_train == bg_class)
            y_bin_train = (y_train[train_mask] == 0).astype(int)
            lr_train = binary_lr(train_scores[train_mask], bg_class)
            fpr_train, tpr_train, auc_train = roc.calculate_roc_curve(
                y_bin_train, lr_train, train["weight"][train_mask]
            )

            y_test = test["y"]
            test_mask = (y_test == 0) | (y_test == bg_class)
            y_bin_test = (y_test[test_mask] == 0).astype(int)
            lr_test = binary_lr(test_scores[test_mask], bg_class)
            fpr_test, tpr_test, auc_test = roc.calculate_roc_curve(
                y_bin_test, lr_test, test["weight"][test_mask]
            )

            train_graph = make_roc_graph(tpr_train, fpr_train)
            test_graph = make_roc_graph(tpr_test, fpr_test)
            CMS.cmsObjectDraw(
                train_graph,
                "L",
                LineColor=model_color,
                LineWidth=1,
                LineStyle=bg_line_style,
            )
            CMS.cmsObjectDraw(
                test_graph,
                "L",
                LineColor=model_color,
                LineWidth=ROOT_MODEL_LINE_WIDTH.get(model_name, 2),
                LineStyle=bg_line_style,
            )
            keepalive.extend([train_graph, test_graph])
            CMS.addToLegend(
                legend,
                (test_graph, f"{model_name} vs {bg_name}: AUC = {auc_test:.4f} ({auc_train:.4f})", "L"),
            )
            summary.setdefault(bg_name, {}).setdefault(model_name, {})["train"] = float(auc_train)
            summary.setdefault(bg_name, {}).setdefault(model_name, {})["test"] = float(auc_test)

    legend.Draw()
    CMS.drawText("color: model, style: background, thick: test", posX=0.20, posY=0.48, font=42, align=0, size=0.026)
    canvas.RedrawAxis()
    canvas._keepalive = keepalive
    save_root_canvas(canvas, out_path)
    return summary


def normalized_hist(values: np.ndarray, weights: np.ndarray,
                    bins: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    hist, edges = np.histogram(values, bins=bins, weights=weights)
    integral = hist.sum()
    if abs(integral) > 0:
        hist = hist / integral
    return hist, edges


def setup_root_cms_style() -> None:
    if not HAS_ROOT_CMSSTYLE:
        return
    CMS.setCMSStyle()
    CMS.SetExtraText("Simulation Preliminary")
    CMS.SetLumi(None, run="Run 2+3, 138+62 fb^{#minus1}")
    CMS.SetEnergy(0, unit="13/13.6 TeV")
    ROOT.gStyle.SetOptStat(0)


def root_color(hex_color: str) -> int:
    return int(ROOT.TColor.GetColor(hex_color))


def lr_color_index(model_name: str, class_idx: int) -> int:
    return LR_COLOR_INDEX[model_name][class_idx]


def lr_color_hex(model_name: str, class_idx: int) -> str:
    palette_idx = lr_color_index(model_name, class_idx)
    return PLOTTER_PALETTE_HEX[palette_idx % len(PLOTTER_PALETTE_HEX)]


def palette_root_color(palette_idx: int) -> int:
    palette = list(PLOTTER_PALETTE) + list(PLOTTER_PALETTE_LONG)
    if not palette:
        return root_color(PLOTTER_PALETTE_HEX[palette_idx % len(PLOTTER_PALETTE_HEX)])
    return int(palette[palette_idx % len(palette)])


def lr_root_color(model_name: str, class_idx: int) -> int:
    return palette_root_color(lr_color_index(model_name, class_idx))


def roc_model_root_color(model_name: str) -> int:
    return palette_root_color(ROC_MODEL_COLOR_INDEX[model_name])


def make_root_hist(name: str, values: np.ndarray, weights: np.ndarray,
                   nbins: int, xmin: float, xmax: float,
                   use_abs_weight: bool = False) -> "ROOT.TH1D":
    h = ROOT.TH1D(name, "", nbins, xmin, xmax)
    h.SetDirectory(0)
    h.Sumw2()
    finite = np.isfinite(values) & np.isfinite(weights)
    for value, weight in zip(values[finite], weights[finite]):
        fill_weight = abs(float(weight)) if use_abs_weight else float(weight)
        h.Fill(float(value), fill_weight)
    return h


def normalize_root_hist(h: "ROOT.TH1D") -> None:
    integral = h.Integral(0, h.GetNbinsX() + 1)
    if integral > 0:
        h.Scale(1.0 / integral)


def make_ratio_hist(numerator: "ROOT.TH1D", denominator: "ROOT.TH1D",
                    name: str) -> "ROOT.TH1D":
    ratio = numerator.Clone(name)
    ratio.SetDirectory(0)
    for ibin in range(1, ratio.GetNbinsX() + 1):
        denom = denominator.GetBinContent(ibin)
        if denom > 0:
            ratio.SetBinContent(ibin, numerator.GetBinContent(ibin) / denom)
            ratio.SetBinError(ibin, numerator.GetBinError(ibin) / denom)
        else:
            ratio.SetBinContent(ibin, 0.0)
            ratio.SetBinError(ibin, 0.0)
    return ratio


def save_root_canvas(canvas, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(str(output_path))
    canvas.Close()


def plot_lr_distributions(train: Dict[str, np.ndarray], test: Dict[str, np.ndarray],
                          bdt_train_lr: np.ndarray, bdt_test_lr: np.ndarray,
                          pn_train_lr: Optional[np.ndarray],
                          pn_test_lr: Optional[np.ndarray],
                          out_dir: Path) -> None:
    if HAS_ROOT_CMSSTYLE:
        plot_lr_distributions_root(
            train, test, bdt_train_lr, bdt_test_lr, pn_train_lr, pn_test_lr, out_dir
        )
        return

    bins = np.linspace(0.0, 1.0, 31)
    out_dir.mkdir(parents=True, exist_ok=True)
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        class_indices = [cls_idx]
        models = [("BDT", bdt_train_lr, bdt_test_lr, "--", "-")]
        if pn_train_lr is not None and pn_test_lr is not None:
            models.append(("ParticleNet", pn_train_lr, pn_test_lr, ":", "-"))
        for model_idx, (model_name, train_lr, test_lr, train_style, test_style) in enumerate(models):
            for class_idx in class_indices:
                color = lr_color_hex(model_name, class_idx)
                for split_name, table, values, linestyle in [
                    ("train", train, train_lr, train_style),
                    ("test", test, test_lr, test_style),
                ]:
                    mask = table["y"] == class_idx
                    if mask.sum() == 0:
                        continue
                    hist, edges = normalized_hist(values[mask], table["weight"][mask], bins)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                    ax.step(
                        centers,
                        hist,
                        where="mid",
                        linewidth=2,
                        label=f"{model_name} {CLASS_NAMES[class_idx]} {split_name}",
                        color=color,
                        linestyle=linestyle,
                    )
        ax.set_xlabel("LR_modified")
        ax.set_ylabel("Normalized")
        ax.set_title(cls_name)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"lr_{cls_name}.png", dpi=150)
        plt.close(fig)


def plot_lr_distributions_root(train: Dict[str, np.ndarray], test: Dict[str, np.ndarray],
                               bdt_train_lr: np.ndarray, bdt_test_lr: np.ndarray,
                               pn_train_lr: Optional[np.ndarray],
                               pn_test_lr: Optional[np.ndarray],
                               out_dir: Path) -> None:
    setup_root_cms_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    nbins = 30
    models = [("BDT", bdt_train_lr, bdt_test_lr)]
    if pn_train_lr is not None and pn_test_lr is not None:
        models.append(("ParticleNet", pn_train_lr, pn_test_lr))

    for target_idx, target_name in enumerate(CLASS_NAMES):
        class_indices = [target_idx]
        hists = []
        class_short = ["sig", "np", "VV", "ttX"]

        for model_name, train_lr, test_lr in models:
            for class_idx in class_indices:
                color = lr_root_color(model_name, class_idx)
                train_style = ROOT.kDashed if model_name == "BDT" else ROOT.kDotted
                for split_name, table, lr_values, style in [
                    ("train", train, train_lr, train_style),
                    ("test", test, test_lr, ROOT.kSolid),
                ]:
                    mask = table["y"] == class_idx
                    if mask.sum() == 0:
                        continue
                    hist = make_root_hist(
                        f"h_lr_{model_name}_{target_name}_{CLASS_NAMES[class_idx]}_{split_name}",
                        lr_values[mask],
                        table["weight"][mask],
                        nbins,
                        0.0,
                        1.0,
                    )
                    normalize_root_hist(hist)
                    hist.SetLineColor(color)
                    hist.SetLineStyle(style)
                    hist.SetLineWidth(2)
                    hist.SetMarkerSize(0)
                    model_label = "PN" if model_name == "ParticleNet" else model_name
                    hists.append((
                        hist,
                        f"{model_label} {split_name}",
                        color,
                        style,
                    ))

        if not hists:
            continue

        ymax = max(hist.GetMaximum() for hist, _label, _color, _style in hists)
        canvas = CMS.cmsCanvas(
            "",
            0.0,
            1.0,
            0.0,
            max(0.01, ymax * 1.65),
            "LR_{modified}",
            "Normalized",
            square=True,
            iPos=11,
            extraSpace=0.0,
        )
        canvas.SetGrid()
        legend = CMS.cmsLeg(0.48, 0.62, 0.92, 0.84, textSize=0.030, columns=2)
        for hist, label, color, style in hists:
            CMS.cmsObjectDraw(
                hist,
                "hist",
                LineColor=color,
                LineWidth=hist.GetLineWidth(),
                LineStyle=style,
            )
            CMS.cmsObjectDraw(
                hist,
                "E0 SAME",
                LineColor=color,
                LineWidth=hist.GetLineWidth(),
                LineStyle=style,
                MarkerColor=color,
                MarkerSize=0,
            )
            CMS.addToLegend(legend, (hist, label, "L"))
        legend.Draw()
        CMS.drawText(target_name, posX=0.20, posY=0.70, font=62, align=0, size=0.035)
        canvas.RedrawAxis()

        save_root_canvas(canvas, out_dir / f"lr_{target_name}.png")


def plot_mass_correlation(corr: Dict[str, Dict[str, Dict[str, Dict[str, float]]]],
                          out_path: Path) -> None:
    labels: List[str] = []
    for split_name in ["train", "test"]:
        for model_name in corr.get(split_name, {}):
            labels.append(f"{model_name} {split_name}")
    metrics = ["disco", "pearson_absw", "spearman_absw"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    for ax, mass_name in zip(axes, ["mass1", "mass2"]):
        x = np.arange(len(metrics))
        width = 0.8 / max(1, len(labels))
        for i, label in enumerate(labels):
            model_name, split_name = label.rsplit(" ", 1)
            vals = [corr[split_name][model_name][mass_name][metric] for metric in metrics]
            ax.bar(x + (i - (len(labels) - 1) / 2) * width, vals, width, label=label)
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=20, ha="right")
        ax.set_title(mass_name)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mass_shapes_by_lr(table: Dict[str, np.ndarray], lr_values: np.ndarray,
                           model_name: str, split_name: str, out_dir: Path) -> None:
    """Draw background mass shapes in LR_modified regions."""
    if HAS_ROOT_CMSSTYLE:
        plot_mass_shapes_by_lr_root(table, lr_values, model_name, split_name, out_dir)
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    selections = [
        ("No cut", None, "black", "-"),
        ("LR < 0.3", lr_values < 0.3, "tab:blue", "-"),
        ("0.3 < LR < 0.7", (lr_values > 0.3) & (lr_values < 0.7), "tab:orange", "-"),
        ("LR > 0.7", lr_values > 0.7, "tab:red", "-"),
    ]
    bins = np.linspace(60.0, 120.0, 31)
    background = table["y"] != 0
    weights = np.abs(table["weight"])

    for mass_name in ["mass1", "mass2"]:
        mass = table[mass_name]
        valid = background & (mass > 0)
        if valid.sum() == 0:
            continue

        fig, ax = plt.subplots(figsize=(6, 4.5))
        for label, lr_mask, color, linestyle in selections:
            mask = valid if lr_mask is None else valid & lr_mask
            if mask.sum() == 0:
                continue
            hist, edges = normalized_hist(mass[mask], weights[mask], bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            ax.step(
                centers,
                hist,
                where="mid",
                linewidth=2,
                color=color,
                linestyle=linestyle,
                label=f"{label} (N={int(mask.sum())})",
            )
        ax.set_xlabel(f"{mass_name} [GeV]")
        ax.set_ylabel("Normalized")
        ax.set_title(f"{model_name} {split_name}: background {mass_name}")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"mass_shape_{model_name}_{split_name}_{mass_name}.png", dpi=150)
        plt.close(fig)


def plot_mass_shapes_by_lr_root(table: Dict[str, np.ndarray], lr_values: np.ndarray,
                                model_name: str, split_name: str, out_dir: Path) -> None:
    """Draw background mass shapes in LR_modified regions with no-cut ratios."""
    setup_root_cms_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    selections = [
        ("No cut", None),
        ("LR < 0.3", lr_values < 0.3),
        ("0.3 < LR < 0.7", (lr_values > 0.3) & (lr_values < 0.7)),
        ("LR > 0.7", lr_values > 0.7),
    ]
    background = table["y"] != 0
    weights = table["weight"]

    for mass_name in ["mass1", "mass2"]:
        mass = table[mass_name]
        valid = background & (mass > 0)
        if valid.sum() == 0:
            continue

        hists = {}
        for label, lr_mask in selections:
            mask = valid if lr_mask is None else valid & lr_mask
            if mask.sum() == 0:
                continue
            hist = make_root_hist(
                f"h_mass_{model_name}_{split_name}_{mass_name}_{label.replace(' ', '_').replace('<', 'lt').replace('>', 'gt')}",
                mass[mask],
                weights[mask],
                30,
                60.0,
                120.0,
                use_abs_weight=True,
            )
            hists[f"{label} (N={int(mask.sum())})"] = hist

        if len(hists) <= 1:
            continue

        config = {
            "era": "Run2",
            "run_label": "Run 2+3, 138+62 fb^{#minus1}",
            "CoM": "13/13.6",
            "channel": f"{model_name} {split_name}",
            "region": f"Background {mass_name}",
            "xTitle": f"{mass_name} [GeV]",
            "yTitle": "Normalized",
            "rTitle": "Region / No cut",
            "rRange": [0.5, 1.5],
            "xRange": [60.0, 120.0],
            "normalize": True,
            "legend": [0.48, 0.64, 0.92, 0.88],
            "legendTextSize": 0.030,
            "channelPosX": 0.20,
            "channelPosY": 0.72,
            "channelSize": 0.035,
            "extraText": "Simulation Preliminary",
        }

        canvas = KinematicCanvasWithRatio(hists, config)
        canvas.drawPadUp()
        canvas.drawPadDown()
        save_root_canvas(canvas.canv, out_dir / f"mass_shape_{model_name}_{split_name}_{mass_name}.png")


def plot_mass_shapes_by_lr_train_test(train: Dict[str, np.ndarray], test: Dict[str, np.ndarray],
                                      train_lr: np.ndarray, test_lr: np.ndarray,
                                      model_name: str, out_dir: Path) -> None:
    """Draw train/test background mass sculpting plots with dCor annotations."""
    if not HAS_ROOT_CMSSTYLE:
        plot_mass_shapes_by_lr(train, train_lr, model_name, "train", out_dir)
        plot_mass_shapes_by_lr(test, test_lr, model_name, "test", out_dir)
        return

    setup_root_cms_style()
    out_dir.mkdir(parents=True, exist_ok=True)
    selections = [
        ("No cut", None, ROOT.kBlack),
        ("LR < 0.3", lambda lr: lr < 0.3, palette_root_color(0)),
        ("0.3 < LR < 0.7", lambda lr: (lr > 0.3) & (lr < 0.7), palette_root_color(1)),
        ("LR > 0.7", lambda lr: lr > 0.7, palette_root_color(2)),
    ]
    selection_order = [label for label, _selector, _color in selections]

    for mass_name in ["mass1", "mass2"]:
        split_inputs = [
            ("train", train, train_lr, ROOT.kDashed),
            ("test", test, test_lr, ROOT.kSolid),
        ]
        hists = []
        refs = {}
        dcors = {}

        for split_name, table, lr_values, line_style in split_inputs:
            mass = table[mass_name]
            valid = (table["y"] != 0) & (mass > 0)
            if valid.sum() == 0:
                continue

            dcors[split_name] = compute_disco(
                lr_values[valid],
                mass[valid],
                np.abs(table["weight"][valid]),
            )

            for label, selector, color in selections:
                mask = valid if selector is None else valid & selector(lr_values)
                if mask.sum() == 0:
                    continue
                hist = make_root_hist(
                    f"h_mass_{model_name}_{split_name}_{mass_name}_{label.replace(' ', '_').replace('<', 'lt').replace('>', 'gt')}",
                    mass[mask],
                    table["weight"][mask],
                    30,
                    60.0,
                    120.0,
                    use_abs_weight=True,
                )
                normalize_root_hist(hist)
                hist.SetLineColor(color)
                hist.SetLineStyle(line_style)
                hist.SetLineWidth(2)
                hist.SetMarkerSize(0)
                if label == "No cut":
                    refs[split_name] = hist
                hists.append((hist, f"{label} {split_name}", label, split_name, color, line_style))

        if not hists or not refs:
            continue

        hist_by_key = {(label, split_name): item for item in hists for label, split_name in [(item[2], item[3])]}
        hists = [
            hist_by_key[(label, split_name)]
            for label in selection_order
            for split_name in ["train", "test"]
            if (label, split_name) in hist_by_key
        ]

        ymax = max(hist.GetMaximum() for hist, *_rest in hists)
        CMS.SetLumi(None, run="")
        CMS.SetEnergy(0, unit="13/13.6 TeV")
        canvas = CMS.cmsDiCanvas(
            "",
            60.0,
            120.0,
            0.0,
            max(0.01, ymax * 1.75),
            0.4,
            1.8,
            f"{mass_name} [GeV]",
            "Normalized",
            "Region / No cut",
            square=True,
            iPos=0,
            extraSpace=0.0,
        )
        keepalive = []

        canvas.cd(1)
        legend = CMS.cmsLeg(0.40, 0.58, 0.94, 0.89, textSize=0.037, columns=2)
        for hist, legend_label, _label, _split_name, color, line_style in hists:
            CMS.cmsObjectDraw(
                hist,
                "hist",
                LineColor=color,
                LineWidth=2,
                LineStyle=line_style,
            )
            CMS.cmsObjectDraw(
                hist,
                "E0 SAME",
                LineColor=color,
                LineWidth=2,
                LineStyle=line_style,
                MarkerColor=color,
                MarkerSize=0,
            )
            CMS.addToLegend(legend, (hist, legend_label, "L"))
            keepalive.append(hist)
        legend.Draw()
        CMS.drawText(f"{model_name}", posX=0.18, posY=0.72, font=62, align=0, size=0.050)
        if "train" in dcors:
            CMS.drawText(f"dCor train = {dcors['train']:.4f}", posX=0.18, posY=0.56, font=42, align=0, size=0.050)
        if "test" in dcors:
            CMS.drawText(f"dCor test = {dcors['test']:.4f}", posX=0.18, posY=0.49, font=42, align=0, size=0.050)
        canvas.cd(1).RedrawAxis()

        canvas.cd(2)
        ref_line = ROOT.TLine(60.0, 1.0, 120.0, 1.0)
        ref_line.SetLineStyle(ROOT.kDotted)
        ref_line.SetLineColor(ROOT.kBlack)
        ref_line.SetLineWidth(2)
        ref_line.Draw()
        keepalive.append(ref_line)

        for hist, _legend_label, label, split_name, color, line_style in hists:
            if label == "No cut" or split_name not in refs:
                continue
            ratio = make_ratio_hist(hist, refs[split_name], f"{hist.GetName()}_ratio")
            CMS.cmsObjectDraw(
                ratio,
                "hist",
                LineColor=color,
                LineWidth=2,
                LineStyle=line_style,
            )
            CMS.cmsObjectDraw(
                ratio,
                "E0 SAME",
                LineColor=color,
                LineWidth=2,
                LineStyle=line_style,
                MarkerColor=color,
                MarkerSize=0,
            )
            keepalive.append(ratio)
        canvas.cd(2).RedrawAxis()
        canvas._keepalive = keepalive
        save_root_canvas(canvas, out_dir / f"mass_shape_{model_name}_{mass_name}.png")


def permutation_importance_auc(model, table: Dict[str, np.ndarray],
                               baseline_scores: np.ndarray,
                               max_events: int, random_state: int,
                               feature_names: Optional[Sequence[str]] = None) -> List[Tuple[str, float]]:
    rng = np.random.default_rng(random_state)
    n = table["X"].shape[0]
    idx = np.arange(n)
    if n > max_events:
        idx = rng.choice(n, max_events, replace=False)

    X = table["X"][idx].copy()
    y = table["y"][idx]
    weights = table["weight"][idx]
    base_avg, _ = average_signal_vs_bg_auc(y, baseline_scores[idx], weights)
    importances: List[Tuple[str, float]] = []

    names = list(feature_names) if feature_names is not None else FEATURE_NAMES
    for feat_idx, feat_name in enumerate(names):
        X_perm = X.copy()
        X_perm[:, feat_idx] = rng.permutation(X_perm[:, feat_idx])
        scores = proba_with_all_classes(model, X_perm)
        avg_auc, _ = average_signal_vs_bg_auc(y, scores, weights)
        importances.append((feat_name, float(base_avg - avg_auc)))

    importances.sort(key=lambda item: item[1], reverse=True)
    return importances


def save_importance(importances: List[Tuple[str, float]], out_dir: Path) -> None:
    with open(out_dir / "feature_importance.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "mean_auc_drop"])
        writer.writerows(importances)

    top = importances[:25]
    fig, ax = plt.subplots(figsize=(8, max(5, 0.25 * len(top))))
    names = [name for name, _value in reversed(top)]
    values = [value for _name, value in reversed(top)]
    ax.barh(names, values, color="tab:blue")
    ax.set_xlabel("Validation/test average AUC drop")
    ax.set_title("BDT permutation importance")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150)
    plt.close(fig)


def resolve_repo_path(path_text: Optional[str], default: Path) -> Path:
    if not path_text:
        return default
    path = Path(path_text)
    if not path.is_absolute():
        path = PARTICLENETMD_DIR / path
    return path


def load_feature_names(path: Optional[Path] = None) -> List[str]:
    if path is not None and path.exists():
        with open(path) as handle:
            return list(json.load(handle))
    return list(FEATURE_NAMES)


def validate_feature_names(feature_names: Optional[Sequence[str]] = None) -> None:
    names = list(feature_names) if feature_names is not None else FEATURE_NAMES
    banned_exact = {"mass1", "mass2", "HT", "ST", "nJets", "nBjets", "nLeptons"}
    banned_substrings = ["dimu_mass", "mass_dimu", "abs_eta", "min_", "max_", "sum_"]
    for name in names:
        if name in banned_exact:
            raise RuntimeError(f"Banned BDT feature present: {name}")
        if name.startswith("m_") or "_mass_" in name or name.endswith("_mass"):
            raise RuntimeError(f"Invariant-mass-like BDT feature present: {name}")
        for needle in banned_substrings:
            if needle in name:
                raise RuntimeError(f"Banned reducer/duplicate BDT feature present: {name}")


def load_npz_table(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as arrays:
        return {key: arrays[key] for key in arrays.files}


def build_or_load_bdt_table(config, signal: str, split: str, fold_list: Sequence[int],
                            max_events_per_fold: Optional[int], workers: int,
                            out_dir: Path, rebuild_cache: bool,
                            source_cache_dir: Optional[Path],
                            pilot_events_per_class: Optional[int] = None) -> Dict[str, np.ndarray]:
    if source_cache_dir is not None and not rebuild_cache:
        source_path = source_cache_dir / f"{split}_{cache_suffix(max_events_per_fold, pilot_events_per_class)}.npz"
        if source_path.exists():
            print(f"  Loading {split} table cache from {source_path}", flush=True)
            return load_npz_table(source_path)
        print(f"  Table cache not found at {source_path}; rebuilding from graph datasets.", flush=True)

    return build_or_load_table(
        config, signal, split, fold_list, max_events_per_fold, workers, out_dir, rebuild_cache,
        pilot_events_per_class=pilot_events_per_class,
    )


def process_signal(args, signal: str) -> None:
    print(f"\n=== BDT training: {signal} ===", flush=True)
    config = load_sgl_config(args.config)
    train_params = config.get_training_parameters()

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

    if (not args.pilot) and args.max_events_per_class is not None:
        train_cap = args.max_events_per_class
        valid_cap = args.max_events_per_class
        test_cap = args.max_events_per_class if args.cap_test else None
    elif not args.pilot:
        train_cap = train_params.get("max_events_per_fold_per_class")
        valid_cap = train_params.get("max_events_per_fold_per_class")
        test_cap = None

    output_base = resolve_repo_path(getattr(args, "output_base", None), OUTPUT_BASE)
    out_dir = output_base / "Combined" / signal / "fold-4"
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    table_cache_dir = resolve_repo_path(args.table_cache_dir, Path("")) if args.table_cache_dir else None
    feature_names = load_feature_names(
        resolve_repo_path(args.feature_names, Path("")) if args.feature_names else (
            table_cache_dir / "feature_names.json" if table_cache_dir is not None else None
        )
    )
    validate_feature_names(feature_names)

    with open(out_dir / "feature_names.json", "w") as handle:
        json.dump(feature_names, handle, indent=2)

    train = build_or_load_bdt_table(
        config, signal, "train", train_folds,
        train_cap, args.workers, out_dir, args.rebuild_cache,
        table_cache_dir,
        pilot_events_per_class=pilot_events_per_class,
    )
    valid = build_or_load_bdt_table(
        config, signal, "valid", valid_folds,
        valid_cap, args.workers, out_dir, args.rebuild_cache,
        table_cache_dir,
        pilot_events_per_class=pilot_events_per_class,
    )
    test = build_or_load_bdt_table(
        config, signal, "test", test_folds,
        test_cap, args.workers, out_dir, args.rebuild_cache,
        table_cache_dir,
        pilot_events_per_class=pilot_events_per_class,
    )
    for split_name, table in [("train", train), ("valid", valid), ("test", test)]:
        if table["X"].shape[1] != len(feature_names):
            raise RuntimeError(
                f"{split_name} table has {table['X'].shape[1]} columns but "
                f"feature list has {len(feature_names)} entries"
            )

    print(f"  Table sizes: train={len(train['y'])}, valid={len(valid['y'])}, test={len(test['y'])}", flush=True)
    model, grid_info = train_bdt(train, valid, args.random_state)
    save_model(model, out_dir / "model.joblib")

    bdt_train_scores = proba_with_all_classes(model, train["X"])
    bdt_valid_scores = proba_with_all_classes(model, valid["X"])
    bdt_test_scores = proba_with_all_classes(model, test["X"])
    pn_train_scores = None
    pn_test_scores = None
    if not args.skip_pn:
        pn_train_scores = evaluate_particle_net(
            config,
            signal,
            train_folds,
            train_cap,
            args.workers,
            args.device,
            pilot_events_per_class=pilot_events_per_class,
            split_name="train",
        )
        if pn_train_scores is not None and len(pn_train_scores) != len(train["y"]):
            raise RuntimeError(
                f"ParticleNet/train length mismatch: {len(pn_train_scores)} != {len(train['y'])}"
            )
        pn_test_scores = evaluate_particle_net(
            config,
            signal,
            test_folds,
            test_cap,
            args.workers,
            args.device,
            pilot_events_per_class=pilot_events_per_class,
            split_name="test",
        )
        if pn_test_scores is not None and len(pn_test_scores) != len(test["y"]):
            raise RuntimeError(
                f"ParticleNet/test length mismatch: {len(pn_test_scores)} != {len(test['y'])}"
            )

    thresholds = load_thresholds(signal)
    bdt_train_lr = compute_lr_modified(bdt_train_scores, train["era"], train["channel_id"], thresholds)
    bdt_test_lr = compute_lr_modified(bdt_test_scores, test["era"], test["channel_id"], thresholds)
    pn_train_lr = (
        compute_lr_modified(pn_train_scores, train["era"], train["channel_id"], thresholds)
        if pn_train_scores is not None else None
    )
    pn_test_lr = (
        compute_lr_modified(pn_test_scores, test["era"], test["channel_id"], thresholds)
        if pn_test_scores is not None else None
    )

    _, train_aucs = average_signal_vs_bg_auc(train["y"], bdt_train_scores, train["weight"])
    _, valid_aucs = average_signal_vs_bg_auc(valid["y"], bdt_valid_scores, valid["weight"])
    test_avg_auc, test_aucs = average_signal_vs_bg_auc(test["y"], bdt_test_scores, test["weight"])
    roc_summary = plot_roc_comparison(
        train,
        test,
        bdt_train_scores,
        bdt_test_scores,
        pn_train_scores,
        pn_test_scores,
        plots_dir / "roc_bdt_vs_particlenet.png",
    )

    corr = {
        "train": {"BDT": mass_correlation_metrics(bdt_train_scores, train)},
        "test": {"BDT": mass_correlation_metrics(bdt_test_scores, test)},
    }
    if pn_train_scores is not None:
        corr["train"]["ParticleNet"] = mass_correlation_metrics(pn_train_scores, train)
    if pn_test_scores is not None:
        corr["test"]["ParticleNet"] = mass_correlation_metrics(pn_test_scores, test)
    plot_mass_correlation(corr, plots_dir / "score_mass_correlation.png")
    plot_lr_distributions(
        train,
        test,
        bdt_train_lr,
        bdt_test_lr,
        pn_train_lr,
        pn_test_lr,
        plots_dir,
    )
    plot_mass_shapes_by_lr_train_test(train, test, bdt_train_lr, bdt_test_lr, "BDT", plots_dir)
    if pn_train_lr is not None and pn_test_lr is not None:
        plot_mass_shapes_by_lr_train_test(train, test, pn_train_lr, pn_test_lr, "ParticleNet", plots_dir)

    np.savez_compressed(
        out_dir / "predictions_train.npz",
        y=train["y"],
        weight=train["weight"],
        mass1=train["mass1"],
        mass2=train["mass2"],
        era=train["era"],
        channel_id=train["channel_id"],
        bdt_scores=bdt_train_scores,
        bdt_lr=bdt_train_lr,
        pn_scores=pn_train_scores if pn_train_scores is not None else np.empty((0, 4)),
        pn_lr=pn_train_lr if pn_train_lr is not None else np.empty((0,)),
    )

    np.savez_compressed(
        out_dir / "predictions_test.npz",
        y=test["y"],
        weight=test["weight"],
        mass1=test["mass1"],
        mass2=test["mass2"],
        era=test["era"],
        channel_id=test["channel_id"],
        bdt_scores=bdt_test_scores,
        bdt_lr=bdt_test_lr,
        pn_scores=pn_test_scores if pn_test_scores is not None else np.empty((0, 4)),
        pn_lr=pn_test_lr if pn_test_lr is not None else np.empty((0,)),
    )

    print("  Computing permutation importance...", flush=True)
    importances = permutation_importance_auc(
        model,
        test,
        bdt_test_scores,
        max_events=args.importance_events,
        random_state=args.random_state,
        feature_names=feature_names,
    )
    save_importance(importances, out_dir)

    summary = {
        "signal": signal,
        "backend": "sklearn.HistGradientBoostingClassifier",
        "channel": "Combined",
        "pilot_mode": bool(args.pilot),
        "folds": {
            "train": train_folds,
            "valid": valid_folds,
            "test": test_folds,
        },
        "caps": {
            "train": train_cap,
            "valid": valid_cap,
            "test": test_cap,
            "pilot_events_per_class": pilot_events_per_class,
        },
        "n_events": {
            "train": int(len(train["y"])),
            "valid": int(len(valid["y"])),
            "test": int(len(test["y"])),
        },
        "grid_search": grid_info,
        "bdt_auc": {
            "train": train_aucs,
            "valid": valid_aucs,
            "test": test_aucs,
            "test_average": test_avg_auc,
        },
        "roc_comparison": roc_summary,
        "mass_correlation": corr,
        "top_features": [
            {"feature": name, "mean_auc_drop": value}
            for name, value in importances[:25]
        ],
        "feature_policy": {
            "source": "external table cache" if table_cache_dir is not None else "saved PyG graphs",
            "table_cache_source": str(table_cache_dir) if table_cache_dir is not None else str(out_dir / "tables"),
            "input_features": int(len(feature_names)),
            "object_sorting": "leptons and jets sorted by descending pT",
            "os_dimuon_sorting": "opposite-sign muon candidates sorted by pair pT, not mass",
            "excluded": [
                "mass1",
                "mass2",
                "dimuon mass",
                "all invariant masses",
                "abs_eta duplicates",
                "simple reducer variables",
                "explicit channel flags",
                "explicit object counts",
                "HT",
                "ST",
            ],
        },
    }
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    print(f"  Best valid avg AUC: {grid_info['best']['valid_avg_auc']:.4f}", flush=True)
    print(f"  Test avg AUC: {test_avg_auc:.4f}", flush=True)
    print(f"=== Done: {out_dir} ===", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--signal", choices=SIGNALS, help="Signal mass point")
    parser.add_argument("--all", action="store_true", help="Run all configured comparison signals")
    parser.add_argument("--config", default=None, help="Path to SglConfig JSON")
    parser.add_argument("--workers", type=int, default=8, help="Parallel .pt loading workers")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device for ParticleNet inference")
    parser.add_argument("--skip-pn", action="store_true", help="Skip ParticleNet inference/comparison")
    parser.add_argument("--rebuild-cache", action="store_true", help="Rebuild cached BDT tables")
    parser.add_argument("--table-cache-dir", default=None,
                        help="Directory containing train/valid/test table .npz files. "
                             "Relative paths are resolved under ParticleNetMD/.")
    parser.add_argument("--feature-names", default=None,
                        help="JSON feature list for the active table cache")
    parser.add_argument("--output-base", default=None,
                        help="Output base directory. Relative paths are resolved under ParticleNetMD/.")
    parser.add_argument("--pilot", action="store_true",
                        help="Run full workflow on a small early-stop sample")
    parser.add_argument("--pilot-events-per-class", type=int, default=250,
                        help="Pilot cap per class per split, giving O(1000) events/split")
    parser.add_argument("--max-events-per-class", type=int, default=None,
                        help="Override train/valid cap per class/fold for a smoke test")
    parser.add_argument("--cap-test", action="store_true",
                        help="Also apply --max-events-per-class to test folds")
    parser.add_argument("--importance-events", type=int, default=20000,
                        help="Maximum test events for permutation importance")
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()
    if not args.all and not args.signal:
        parser.error("Provide --signal or --all")
    return args


def main() -> None:
    args = parse_args()
    targets = SIGNALS if args.all else [args.signal]
    for signal in targets:
        process_signal(args, signal)


if __name__ == "__main__":
    main()
