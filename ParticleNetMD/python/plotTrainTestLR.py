#!/usr/bin/env python3
"""
plotTrainTestLR.py — Train/test LR_modified distributions per (era, channel, class).

For each signal point, loads the Combined-channel best ParticleNetMD model,
re-runs the SAME train/test split used during training, builds the analysis
discriminant LR_modified using per-(era, channel) background weights from
SignalRegionStudyV2/configs/thresholds/{signal}.json, and plots a train vs
test overlay (shape-normalized) with a test/train ratio panel — one PNG per
true class per (era, channel).

LR_modified formula:
    LR = s_sig / (s_sig + w_np * s_np + w_db * s_db + w_ttX * s_ttX)

Usage:
    python python/plotTrainTestLR.py --signal MHc130_MA90
    python python/plotTrainTestLR.py --all
"""

import os
import sys
import json
import argparse
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

import ROOT
ROOT.gROOT.SetBatch(True)

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

from SglConfig import load_sgl_config
from DynamicDatasetLoader import DynamicDatasetLoader
from MultiClassModels import create_multiclass_model
from Preprocess import GraphDataset
from ROCCurveCalculator import ROCCurveCalculator

# Common/Tools is added to PYTHONPATH by setup.sh
from plotter import KinematicCanvasWithRatio
import cmsstyle as CMS

SIGNALS = ["MHc100_MA95", "MHc130_MA90", "MHc160_MA85"]
ERAS = ["2016preVFP", "2016postVFP", "2017", "2018",
        "2022", "2022EE", "2023", "2023BPix"]
CHANNEL_ID_TO_NAME = {0: "Run1E2Mu", 1: "Run3Mu"}
CHANNEL_TO_THRESHOLD_KEY = {"Run1E2Mu": "SR1E2Mu", "Run3Mu": "SR3Mu"}
CLASS_NAMES = ["signal", "nonprompt", "diboson", "ttX"]
CLASS_DISPLAY = ["Signal", "Nonprompt", "Diboson", "ttX"]

PARTICLENETMD_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PARTICLENETMD_DIR.parent
THRESHOLDS_DIR = REPO_ROOT / "SignalRegionStudyV2" / "configs" / "thresholds"
OUTPUT_BASE = PARTICLENETMD_DIR / "TrainTestLR"


def load_model(model_path: str, info_path: str, device: str):
    with open(info_path) as f:
        info = json.load(f)
    hp = info["hyperparameters"]

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint

    num_hidden = hp["num_hidden"]
    num_node_features = hp.get("num_node_features", 9)
    num_classes = hp.get("num_classes", 4)
    dropout_p = hp.get("dropout_p", 0.4)
    model_type = hp.get("model_type", "ParticleNet")
    num_graph_features = hp.get("num_graph_features", 8)
    if "dense1.weight" in state:
        inferred = state["dense1.weight"].shape[1] - 3 * num_hidden
        if inferred != num_graph_features:
            num_graph_features = inferred

    model = create_multiclass_model(
        model_type=model_type,
        num_node_features=num_node_features,
        num_graph_features=num_graph_features,
        num_classes=num_classes,
        num_hidden=num_hidden,
        dropout_p=dropout_p,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, hp


def _load_pt_file(filepath: str):
    """Worker: load one .pt file and return its data list (or [] if missing)."""
    if not os.path.exists(filepath):
        return []
    dataset = torch.load(filepath, weights_only=False)
    return dataset.data_list if hasattr(dataset, 'data_list') else []


def _file_specs(signal_full: str, bg_groups: dict, channel: str, fold_list: list):
    """Build a list of (filepath, kind, group_label, fold) tuples for all needed .pt files.

    kind: 'signal' or 'bg'. group_label: 0 for signal, 1+ for backgrounds.
    """
    samples_root = PARTICLENETMD_DIR / "dataset" / "samples"
    sub_channels = ["Run1E2Mu", "Run3Mu"] if channel == "Combined" else [channel]
    specs = []
    for fold in fold_list:
        for ch in sub_channels:
            specs.append((str(samples_root / "signals" / signal_full / f"{ch}_fold-{fold}.pt"),
                          "signal", 0, fold))
    for grp_idx, (_grp_name, sample_list) in enumerate(bg_groups.items()):
        label = grp_idx + 1
        for sample_name in sample_list:
            for fold in fold_list:
                for ch in sub_channels:
                    specs.append((str(samples_root / "backgrounds" / sample_name / f"{ch}_fold-{fold}.pt"),
                                  "bg", label, fold))
    return specs


def load_train_test(config, signal: str, channel: str = "Combined", workers: int = 8):
    """Parallel-loaded equivalent of DynamicDatasetLoader.load_multiclass_with_subsampling.

    Loads all needed .pt files concurrently via ThreadPoolExecutor (torch.load releases
    the GIL during deserialization), then applies the same per-(group, fold) subsampling
    and per-class weight balancing as the framework function.
    """
    from sklearn.utils import resample, shuffle

    dataset_config = config.get_dataset_config()
    bg_config = config.get_background_config()
    train_params = config.get_training_parameters()

    signal_full = f"{dataset_config['signal_prefix']}{signal}"
    bg_prefix = dataset_config['background_prefix']
    bg_groups = {
        gn: [bg_prefix + s for s in samples]
        for gn, samples in bg_config['background_groups'].items()
    }

    train_folds = train_params['train_folds']
    test_folds = train_params['test_folds']
    max_events_per_fold = train_params.get('max_events_per_fold_per_class', None)
    balance_weights = train_params.get('balance_weights', True)

    def _load_split(fold_list, max_per_fold):
        specs = _file_specs(signal_full, bg_groups, channel, fold_list)
        # Parallel load
        paths = [s[0] for s in specs]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(_load_pt_file, paths))

        # Bin by (label, fold)
        by_lf = {}
        for (_, _kind, label, fold), data_list in zip(specs, results):
            by_lf.setdefault((label, fold), []).extend(data_list)

        # Subsample per (label, fold) and assign labels
        all_data = []
        for (label, _fold), dl in by_lf.items():
            for d in dl:
                d.y = torch.tensor(label, dtype=torch.long)
            if max_per_fold and len(dl) > max_per_fold:
                dl = resample(dl, n_samples=max_per_fold, replace=False, random_state=42)
            all_data.extend(dl)

        # Balance weights across classes
        if balance_weights:
            class_w = {}
            for d in all_data:
                lbl = d.y.item()
                class_w[lbl] = class_w.get(lbl, 0.0) + d.weight.item()
            if class_w:
                max_w = max(class_w.values())
                for d in all_data:
                    lbl = d.y.item()
                    if class_w[lbl] > 0:
                        d.weight = d.weight * (max_w / class_w[lbl])

        return shuffle(all_data, random_state=42)

    train_data = _load_split(train_folds, max_events_per_fold)
    test_data = _load_split(test_folds, None)
    return train_data, test_data


def _channel_id_from_x(x_cpu: torch.Tensor, batch_cpu: torch.Tensor, evt: int) -> int:
    """0 = Run1E2Mu (>=1 e + >=2 mu), 1 = Run3Mu (>=3 mu)."""
    node_mask = (batch_cpu == evt)
    nf = x_cpu[node_mask]
    n_mu = int((nf[:, 5] > 0.5).sum().item())
    n_ele = int((nf[:, 6] > 0.5).sum().item())
    if n_ele >= 1 and n_mu >= 2:
        return 0
    if n_mu >= 3:
        return 1
    return 0


def evaluate(model, loader, device):
    """Run inference. Returns y_true, y_scores, weights, eras, channel_ids."""
    y_true_list, y_scores_list, weights_list = [], [], []
    eras_list, ch_list = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.graphInput, data.batch)
            scores = F.softmax(out, dim=1)

            n_events = int(data.y.size(0))
            y_true_list.append(data.y.cpu().numpy())
            y_scores_list.append(scores.cpu().numpy())
            weights_list.append(data.weight.cpu().numpy())

            x_cpu = data.x.cpu()
            batch_cpu = data.batch.cpu()
            ch_arr = np.empty(n_events, dtype=np.int32)
            for i in range(n_events):
                ch_arr[i] = _channel_id_from_x(x_cpu, batch_cpu, i)
            ch_list.append(ch_arr)

            batch_eras = data.era if hasattr(data, 'era') else [""] * n_events
            eras_list.append(np.array(batch_eras, dtype=object))

    return (np.concatenate(y_true_list),
            np.concatenate(y_scores_list),
            np.concatenate(weights_list),
            np.concatenate(eras_list),
            np.concatenate(ch_list))


def compute_efficiency_above(scores: np.ndarray, weights: np.ndarray, threshold: float) -> dict:
    """Selection efficiency above threshold for weighted events.

    Returns:
        eff             — sum(w_pass) / sum(w_total)
        eff_err         — weighted-binomial error using N_eff = (sum_w)^2 / sum_w^2
        n_pass_unw      — unweighted count above threshold
        n_total_unw     — unweighted total
        sum_w_pass      — for downstream uses (e.g. yields)
        sum_w_total
    """
    n_total = int(len(scores))
    if n_total == 0:
        return {"eff": 0.0, "eff_err": 0.0, "n_pass_unw": 0, "n_total_unw": 0,
                "sum_w_pass": 0.0, "sum_w_total": 0.0}

    pass_mask = scores >= threshold
    sum_w_total = float(np.sum(weights))
    sum_w_pass = float(np.sum(weights[pass_mask]))
    sum_w2_total = float(np.sum(weights * weights))

    if sum_w_total <= 0 or sum_w2_total <= 0:
        return {"eff": 0.0, "eff_err": 0.0,
                "n_pass_unw": int(pass_mask.sum()), "n_total_unw": n_total,
                "sum_w_pass": sum_w_pass, "sum_w_total": sum_w_total}

    eff = sum_w_pass / sum_w_total
    n_eff = (sum_w_total * sum_w_total) / sum_w2_total
    if 0.0 < eff < 1.0 and n_eff > 0:
        eff_err = float(np.sqrt(eff * (1.0 - eff) / n_eff))
    else:
        eff_err = 0.0
    return {
        "eff": float(eff),
        "eff_err": eff_err,
        "n_pass_unw": int(pass_mask.sum()),
        "n_total_unw": n_total,
        "sum_w_pass": sum_w_pass,
        "sum_w_total": sum_w_total,
    }


def compute_lr_modified(scores: np.ndarray, weights: dict) -> np.ndarray:
    """LR_modified = s_sig / (s_sig + w_np*s_np + w_db*s_db + w_ttX*s_ttX)."""
    s_sig = scores[:, 0]
    s_np = scores[:, 1]
    s_db = scores[:, 2]
    s_ttx = scores[:, 3]
    w_np = float(weights['nonprompt'])
    w_db = float(weights['diboson'])
    w_ttx = float(weights['ttX'])
    denom = s_sig + w_np * s_np + w_db * s_db + w_ttx * s_ttx
    lr = np.zeros_like(s_sig)
    valid = denom > 0
    lr[valid] = s_sig[valid] / denom[valid]
    return lr


def make_hist(name: str, scores: np.ndarray, weights: np.ndarray, nbins: int = 30) -> ROOT.TH1D:
    h = ROOT.TH1D(name, "", nbins, 0.0, 1.0)
    h.SetDirectory(0)
    h.Sumw2()
    for s, w in zip(scores, weights):
        h.Fill(float(s), float(w))
    return h


def ks_pvalue(h_train: ROOT.TH1D, h_test: ROOT.TH1D) -> float:
    try:
        return float(h_train.KolmogorovTest(h_test))
    except Exception:
        return float('nan')


def plot_train_test(h_train: ROOT.TH1D, h_test: ROOT.TH1D,
                    threshold: float, era: str, channel: str, cls_display: str,
                    signal: str, output_path: str,
                    n_train_evt: int, n_test_evt: int) -> float:
    """Plot train vs test LR_modified with KinematicCanvasWithRatio (Common/Tools/plotter.py).

    Hist dict order matters: the FIRST entry is the ratio reference, so Train comes
    first to give a meaningful Test/Train ratio.
    """
    p_ks = ks_pvalue(h_train, h_test)

    h_train.SetTitle(f"Train (N={n_train_evt})")
    h_test.SetTitle(f"Test (N={n_test_evt})")

    config = {
        "era": era,
        "channel": f"{channel}, {cls_display}",
        "region": signal,
        "xTitle": "LR_{modified}",
        "yTitle": "Normalized",
        "rTitle": "Test / Train",
        "rRange": [0.5, 1.5],
        "xRange": [0.0, 1.0],
        "normalize": True,
        "legend": [0.55, 0.70, 0.93, 0.88],
        "legendTextSize": 0.04,
    }

    canvas = KinematicCanvasWithRatio({"Train": h_train, "Test": h_test}, config)
    canvas.drawPadUp()

    # Threshold line + KS p-value text on upper pad
    pad_up = canvas.canv.cd(1)
    if 0.0 < threshold < 1.0:
        ymin_pad = pad_up.GetUymin()
        ymax_pad = pad_up.GetUymax()
        thr_line = ROOT.TLine(threshold, ymin_pad, threshold, ymax_pad)
        thr_line.SetLineColor(ROOT.kRed + 1)
        thr_line.SetLineStyle(ROOT.kDashed)
        thr_line.SetLineWidth(2)
        thr_line.Draw()
        canvas._thr_line = thr_line  # keep alive

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.18, 0.55,
                    f"KS p = {p_ks:.3f}    SR thr = {threshold:.2f}")
    canvas._latex = latex

    canvas.drawPadDown()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.canv.SaveAs(output_path)
    canvas.canv.Close()
    return p_ks


def process_signal(signal: str, device: str, workers: int = 8):
    print(f"\n=== Processing {signal} (workers={workers}) ===", flush=True)

    config = load_sgl_config()

    model_dir = PARTICLENETMD_DIR / "GAOptim" / "Combined" / signal / "fold-4" / "best_model"
    model_path = model_dir / "model.pt"
    info_path = model_dir / "model_info.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not info_path.exists():
        raise FileNotFoundError(f"Model info not found: {info_path}")

    threshold_path = THRESHOLDS_DIR / f"{signal}.json"
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold JSON not found: {threshold_path}")
    with open(threshold_path) as f:
        thresholds = json.load(f)

    model, hp = load_model(str(model_path), str(info_path), device)
    print(f"  Model loaded: num_hidden={hp['num_hidden']}, dropout={hp.get('dropout_p')}")

    train_data, test_data = load_train_test(config, signal, channel="Combined", workers=workers)
    print(f"  Loaded train={len(train_data)}, test={len(test_data)} events", flush=True)

    batch_size = hp.get('batch_size', 512)
    train_loader = DataLoader(GraphDataset(train_data), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(GraphDataset(test_data), batch_size=batch_size, shuffle=False)

    print("  Inference on train set...", flush=True)
    y_true_tr, y_sc_tr, w_tr, era_tr, ch_tr = evaluate(model, train_loader, device)
    print("  Inference on test set...", flush=True)
    y_true_te, y_sc_te, w_te, era_te, ch_te = evaluate(model, test_loader, device)

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    out_dir = OUTPUT_BASE / signal
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for era in ERAS:
        if era not in thresholds:
            print(f"  WARN: era {era} not in threshold JSON, skipping")
            continue
        for ch_id, ch_name in CHANNEL_ID_TO_NAME.items():
            thr_key = CHANNEL_TO_THRESHOLD_KEY[ch_name]
            if thr_key not in thresholds[era]:
                print(f"  WARN: {era}/{thr_key} missing in threshold JSON")
                continue
            cfg = thresholds[era][thr_key]
            weights_dict = cfg['weights']
            threshold = float(cfg['threshold'])

            mask_tr = (era_tr == era) & (ch_tr == ch_id)
            mask_te = (era_te == era) & (ch_te == ch_id)
            if mask_tr.sum() == 0 or mask_te.sum() == 0:
                print(f"  {era}/{ch_name}: empty (train={int(mask_tr.sum())}, test={int(mask_te.sum())}), skipping")
                continue

            lr_tr = compute_lr_modified(y_sc_tr[mask_tr], weights_dict)
            lr_te = compute_lr_modified(y_sc_te[mask_te], weights_dict)
            yt_tr = y_true_tr[mask_tr]
            yt_te = y_true_te[mask_te]
            wt_tr = w_tr[mask_tr]
            wt_te = w_te[mask_te]

            ec_dir = out_dir / f"{era}_{ch_name}"
            ec_dir.mkdir(exist_ok=True)

            # AUC: signal (class 0) vs all backgrounds, on LR_modified
            roc = ROCCurveCalculator()
            sig_vs_bg_tr = (yt_tr == 0).astype(int)
            sig_vs_bg_te = (yt_te == 0).astype(int)
            _, _, auc_tr = roc.calculate_roc_curve(sig_vs_bg_tr, lr_tr, wt_tr)
            _, _, auc_te = roc.calculate_roc_curve(sig_vs_bg_te, lr_te, wt_te)

            era_ch_summary = {
                "auc_train": float(auc_tr),
                "auc_test": float(auc_te),
                "delta_auc": float(auc_tr - auc_te),
                "threshold": threshold,
                "classes": {},
            }
            for cls_idx, cls_name in enumerate(CLASS_NAMES):
                cls_mask_tr = (yt_tr == cls_idx)
                cls_mask_te = (yt_te == cls_idx)
                n_tr = int(cls_mask_tr.sum())
                n_te = int(cls_mask_te.sum())
                if n_tr == 0 or n_te == 0:
                    continue

                h_tr = make_hist(f"h_tr_{era}_{ch_name}_{cls_name}",
                                 lr_tr[cls_mask_tr], wt_tr[cls_mask_tr])
                h_te = make_hist(f"h_te_{era}_{ch_name}_{cls_name}",
                                 lr_te[cls_mask_te], wt_te[cls_mask_te])

                p_ks = plot_train_test(h_tr, h_te, threshold,
                                       era, ch_name, CLASS_DISPLAY[cls_idx], signal,
                                       str(ec_dir / f"{cls_name}.png"),
                                       n_tr, n_te)

                eff_tr = compute_efficiency_above(lr_tr[cls_mask_tr],
                                                  wt_tr[cls_mask_tr], threshold)
                eff_te = compute_efficiency_above(lr_te[cls_mask_te],
                                                  wt_te[cls_mask_te], threshold)
                era_ch_summary["classes"][cls_name] = {
                    "ks_p": float(p_ks),
                    "n_train_evt": n_tr,
                    "n_test_evt": n_te,
                    "train_eff_above_thr": eff_tr,
                    "test_eff_above_thr": eff_te,
                }

            summary[f"{era}_{ch_name}"] = era_ch_summary
            print(f"  {era}/{ch_name}: AUC train={era_ch_summary['auc_train']:.4f}  "
                  f"test={era_ch_summary['auc_test']:.4f}  "
                  f"d={era_ch_summary['delta_auc']:+.4f}", flush=True)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"=== {signal}: done -> {out_dir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signal", choices=SIGNALS,
                        help="Signal name (e.g., MHc130_MA90)")
    parser.add_argument("--all", action="store_true",
                        help="Process all signals")
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Torch device (default: cuda if available)")
    parser.add_argument("--workers", type=int, default=8,
                        help="Parallel .pt loaders (default: 8)")
    args = parser.parse_args()

    if args.all:
        targets = SIGNALS
    elif args.signal:
        targets = [args.signal]
    else:
        parser.error("Provide --signal or --all")

    for sig in targets:
        process_signal(sig, args.device, workers=args.workers)


if __name__ == "__main__":
    main()
