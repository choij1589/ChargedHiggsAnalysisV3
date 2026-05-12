#!/usr/bin/env python3
"""
computeTrainTestAUC.py — Train/test AUC table per (era, channel) for ParticleNetMD.

Reuses load_train_test, evaluate, compute_lr_modified from plotTrainTestLR.py to do
the same inference, then computes weighted AUC (signal vs all backgrounds, on
LR_modified) for train and test per (era, channel). Writes results to
TrainTestLR/{signal}/auc_summary.json and merges into the existing summary.json
if it exists. Also prints a console table.

Usage:
    python python/computeTrainTestAUC.py --signal MHc130_MA90 --device cuda:0 --workers 12
    python python/computeTrainTestAUC.py --all --device cuda:0 --workers 12
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

from SglConfig import load_sgl_config
from MultiClassModels import create_multiclass_model  # noqa: F401 (used via plotTrainTestLR)
from Preprocess import GraphDataset
from ROCCurveCalculator import ROCCurveCalculator

# Reuse helpers from plotTrainTestLR
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plotTrainTestLR import (
    SIGNALS, ERAS, CHANNEL_ID_TO_NAME, CHANNEL_TO_THRESHOLD_KEY,
    PARTICLENETMD_DIR, THRESHOLDS_DIR, OUTPUT_BASE,
    load_model, load_train_test, evaluate, compute_lr_modified,
)


def process_signal(signal: str, device: str, workers: int):
    print(f"\n=== AUC for {signal} (workers={workers}) ===", flush=True)

    config = load_sgl_config()

    model_dir = PARTICLENETMD_DIR / "GAOptim" / "Combined" / signal / "fold-4" / "best_model"
    model_path = model_dir / "model.pt"
    info_path = model_dir / "model_info.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    threshold_path = THRESHOLDS_DIR / f"{signal}.json"
    with open(threshold_path) as f:
        thresholds = json.load(f)

    model, hp = load_model(str(model_path), str(info_path), device)
    print(f"  Model loaded", flush=True)

    train_data, test_data = load_train_test(config, signal, channel="Combined", workers=workers)
    print(f"  Loaded train={len(train_data)}, test={len(test_data)} events", flush=True)

    batch_size = hp.get('batch_size', 512)
    train_loader = DataLoader(GraphDataset(train_data), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(GraphDataset(test_data), batch_size=batch_size, shuffle=False)

    print("  Inference on train...", flush=True)
    y_true_tr, y_sc_tr, w_tr, era_tr, ch_tr = evaluate(model, train_loader, device)
    print("  Inference on test...", flush=True)
    y_true_te, y_sc_te, w_te, era_te, ch_te = evaluate(model, test_loader, device)

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    out_dir = OUTPUT_BASE / signal
    out_dir.mkdir(parents=True, exist_ok=True)

    auc_table = {}
    roc = ROCCurveCalculator()

    print(f"\n  {'(era, channel)':25s}  train     test      delta")
    print(f"  {'-' * 25}  --------  --------  --------")
    for era in ERAS:
        if era not in thresholds:
            continue
        for ch_id, ch_name in CHANNEL_ID_TO_NAME.items():
            thr_key = CHANNEL_TO_THRESHOLD_KEY[ch_name]
            if thr_key not in thresholds[era]:
                continue
            cfg = thresholds[era][thr_key]

            mask_tr = (era_tr == era) & (ch_tr == ch_id)
            mask_te = (era_te == era) & (ch_te == ch_id)
            if mask_tr.sum() == 0 or mask_te.sum() == 0:
                continue

            lr_tr = compute_lr_modified(y_sc_tr[mask_tr], cfg['weights'])
            lr_te = compute_lr_modified(y_sc_te[mask_te], cfg['weights'])
            sig_tr = (y_true_tr[mask_tr] == 0).astype(int)
            sig_te = (y_true_te[mask_te] == 0).astype(int)

            _, _, auc_tr = roc.calculate_roc_curve(sig_tr, lr_tr, w_tr[mask_tr])
            _, _, auc_te = roc.calculate_roc_curve(sig_te, lr_te, w_te[mask_te])
            delta = auc_tr - auc_te

            key = f"{era}_{ch_name}"
            auc_table[key] = {
                "auc_train": float(auc_tr),
                "auc_test": float(auc_te),
                "delta_auc": float(delta),
                "threshold": float(cfg['threshold']),
            }
            print(f"  {key:25s}  {auc_tr:.4f}    {auc_te:.4f}    {delta:+.4f}", flush=True)

    # Save AUC-only summary
    auc_path = out_dir / "auc_summary.json"
    with open(auc_path, "w") as f:
        json.dump(auc_table, f, indent=2)
    print(f"  Saved {auc_path}")

    # Merge into existing summary.json if present
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        for key, auc in auc_table.items():
            if key in summary and isinstance(summary[key], dict):
                # Old format had per-class only at top level; new format nests under "classes"
                # Promote per-class entries under "classes" if needed, then attach AUCs.
                if "classes" not in summary[key]:
                    cls_keys = ("signal", "nonprompt", "diboson", "ttX")
                    classes_part = {k: summary[key].pop(k) for k in cls_keys if k in summary[key]}
                    summary[key]["classes"] = classes_part
                summary[key].update(auc)
            else:
                summary[key] = auc
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"  Merged into {summary_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signal", choices=SIGNALS)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    if args.all:
        targets = SIGNALS
    elif args.signal:
        targets = [args.signal]
    else:
        parser.error("Provide --signal or --all")

    for sig in targets:
        process_signal(sig, args.device, args.workers)


if __name__ == "__main__":
    main()
