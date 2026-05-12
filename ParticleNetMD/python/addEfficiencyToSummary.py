#!/usr/bin/env python3
"""
addEfficiencyToSummary.py — Inject AUC + selection-efficiency closure check
into existing TrainTestLR/{signal}/summary.json without re-plotting.

For each (signal, era, channel):
  - AUC of LR_modified for signal vs all backgrounds (train and test)
  - Per-class selection efficiency above SR threshold with weighted-binomial error

Usage:
    python python/addEfficiencyToSummary.py --signal MHc130_MA90 --device cuda:0 --workers 12
    python python/addEfficiencyToSummary.py --all --device cuda:0 --workers 12
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from SglConfig import load_sgl_config
from MultiClassModels import create_multiclass_model  # noqa: F401
from Preprocess import GraphDataset
from ROCCurveCalculator import ROCCurveCalculator

from plotTrainTestLR import (
    SIGNALS, ERAS, CHANNEL_ID_TO_NAME, CHANNEL_TO_THRESHOLD_KEY,
    CLASS_NAMES, PARTICLENETMD_DIR, THRESHOLDS_DIR, OUTPUT_BASE,
    load_model, load_train_test, evaluate,
    compute_lr_modified, compute_efficiency_above,
)


def process_signal(signal: str, device: str, workers: int):
    print(f"\n=== {signal} (workers={workers}) ===", flush=True)

    config = load_sgl_config()
    model_dir = PARTICLENETMD_DIR / "GAOptim" / "Combined" / signal / "fold-4" / "best_model"
    model_path = model_dir / "model.pt"
    info_path = model_dir / "model_info.json"
    threshold_path = THRESHOLDS_DIR / f"{signal}.json"

    with open(threshold_path) as f:
        thresholds = json.load(f)

    model, hp = load_model(str(model_path), str(info_path), device)
    print("  Model loaded", flush=True)

    train_data, test_data = load_train_test(config, signal, channel="Combined", workers=workers)
    print(f"  Loaded train={len(train_data)}, test={len(test_data)}", flush=True)

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
    summary_path = out_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
    else:
        summary = {}

    roc = ROCCurveCalculator()

    print(f"\n  {'(era, channel)':25s}  AUC train  AUC test  dAUC")
    print(f"  {'-' * 25}  ---------  --------  ------")
    for era in ERAS:
        if era not in thresholds:
            continue
        for ch_id, ch_name in CHANNEL_ID_TO_NAME.items():
            thr_key = CHANNEL_TO_THRESHOLD_KEY[ch_name]
            if thr_key not in thresholds[era]:
                continue
            cfg = thresholds[era][thr_key]
            threshold = float(cfg['threshold'])

            mask_tr = (era_tr == era) & (ch_tr == ch_id)
            mask_te = (era_te == era) & (ch_te == ch_id)
            if mask_tr.sum() == 0 or mask_te.sum() == 0:
                continue

            lr_tr = compute_lr_modified(y_sc_tr[mask_tr], cfg['weights'])
            lr_te = compute_lr_modified(y_sc_te[mask_te], cfg['weights'])
            yt_tr = y_true_tr[mask_tr]
            yt_te = y_true_te[mask_te]
            wt_tr = w_tr[mask_tr]
            wt_te = w_te[mask_te]

            sig_tr = (yt_tr == 0).astype(int)
            sig_te = (yt_te == 0).astype(int)
            _, _, auc_tr = roc.calculate_roc_curve(sig_tr, lr_tr, wt_tr)
            _, _, auc_te = roc.calculate_roc_curve(sig_te, lr_te, wt_te)

            key = f"{era}_{ch_name}"
            entry = summary.get(key, {})
            # Promote any old flat per-class entries under "classes"
            if "classes" not in entry:
                cls_keys = ("signal", "nonprompt", "diboson", "ttX")
                cls_part = {k: entry.pop(k) for k in cls_keys if k in entry}
                entry["classes"] = cls_part
            entry["auc_train"] = float(auc_tr)
            entry["auc_test"] = float(auc_te)
            entry["delta_auc"] = float(auc_tr - auc_te)
            entry["threshold"] = threshold

            for cls_idx, cls_name in enumerate(CLASS_NAMES):
                m_tr = (yt_tr == cls_idx)
                m_te = (yt_te == cls_idx)
                if m_tr.sum() == 0 or m_te.sum() == 0:
                    continue
                eff_tr = compute_efficiency_above(lr_tr[m_tr], wt_tr[m_tr], threshold)
                eff_te = compute_efficiency_above(lr_te[m_te], wt_te[m_te], threshold)
                cls_entry = entry["classes"].setdefault(cls_name, {})
                cls_entry["train_eff_above_thr"] = eff_tr
                cls_entry["test_eff_above_thr"] = eff_te

            summary[key] = entry
            print(f"  {key:25s}  {auc_tr:.4f}     {auc_te:.4f}    {auc_tr-auc_te:+.4f}", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  -> {summary_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
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
