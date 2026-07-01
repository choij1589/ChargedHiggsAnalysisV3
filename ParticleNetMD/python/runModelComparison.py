#!/usr/bin/env python3
"""Run the masked OS-dimuon-pT BDT/DNN/ParticleNet comparison.

The comparison retrains the tabular BDT and DNN on a shared cache where
os_dimu1_pt and os_dimu2_pt are removed. ParticleNet is kept as the nominal
graph-model reference because its standard input has no explicit pair-pT column.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "lib"))

import trainBDT as bdt
from SglConfig import load_sgl_config


SIGNALS = ["MHc130_MA90", "MHc160_MA85", "MHc100_MA95"]
MASKED_COLUMNS = ["os_dimu1_pt", "os_dimu2_pt"]
COMPARISON_ROOT = bdt.PARTICLENETMD_DIR / "ModelComparison"
MODELS = ["BDT", "DNN", "DNN_MD", "ParticleNet", "ParticleNet_MD"]
DECORRELATION_MODELS = ["BDT", "DNN", "DNN_MD", "ParticleNet_best", "ParticleNet_best_MD"]
SPLITS = ["train", "test"]
PLOT_LINE_WIDTH = 2
MODEL_REGISTRY = {
    "BDT": {
        "source_dir": "BDT",
        "score_key": "bdt_scores",
        "lr_key": "bdt_lr",
        "color": "#5790fc",
        "display": "BDT",
        "tag": "BDT",
    },
    "DNN": {
        "source_dir": "DNN",
        "score_key": "dnn_scores",
        "lr_key": "dnn_lr",
        "color": "#e42536",
        "display": "DNN",
        "tag": "DNN",
    },
    "DNN_MD": {
        "source_dir": "DNN_MD",
        "score_key": "dnn_scores",
        "lr_key": "dnn_lr",
        "color": "#964a8b",
        "display": "DNN MD",
        "tag": "DNN_MD",
    },
    "ParticleNet": {
        "source_dir": "ParticleNet",
        "score_key": "pn_scores",
        "lr_key": "pn_lr",
        "color": "#f89c20",
        "display": "ParticleNet",
        "tag": "ParticleNet",
    },
    "ParticleNet_MD": {
        "source_dir": "ParticleNet_MD",
        "score_key": "pn_scores",
        "lr_key": "pn_lr",
        "color": "#7a21dd",
        "display": "ParticleNet MD",
        "tag": "ParticleNet_MD",
    },
    "ParticleNet_best": {
        "source_dir": "ParticleNet_512_256_256_lr5e-4_CyclicLR",
        "score_key": "pn_scores",
        "lr_key": "pn_lr",
        "color": "#f89c20",
        "display": "ParticleNet",
        "tag": "ParticleNet",
    },
    "ParticleNet_best_MD": {
        "source_dir": "ParticleNet_512_256_256_lr5e-4_CyclicLR_MD",
        "score_key": "pn_scores",
        "lr_key": "pn_lr",
        "color": "#7a21dd",
        "display": "ParticleNet MD",
        "tag": "ParticleNet_MD",
    },
}
MODEL_SCORE_KEYS = {key: cfg["score_key"] for key, cfg in MODEL_REGISTRY.items()}
MODEL_LR_KEYS = {key: cfg["lr_key"] for key, cfg in MODEL_REGISTRY.items()}
MODEL_ROOT_COLORS = {key: cfg["color"] for key, cfg in MODEL_REGISTRY.items()}
MODEL_DISPLAY = {key: cfg["display"] for key, cfg in MODEL_REGISTRY.items()}
TRAIN_LINE_STYLE = 7


def load_table(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as arrays:
        return {key: arrays[key] for key in arrays.files}


def save_table(path: Path, arrays: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def save_root_canvas_with_pdf(canvas, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(str(output_path))
    canvas.SaveAs(str(output_path.with_suffix(".pdf")))
    canvas.Close()


def split_settings(args, config) -> Tuple[Dict[str, Sequence[int]], Dict[str, Optional[int]], Optional[int]]:
    train_params = config.get_training_parameters()
    if args.pilot:
        folds = {
            "train": [train_params["train_folds"][0]],
            "valid": train_params["valid_folds"],
            "test": train_params["test_folds"],
        }
        caps = {"train": None, "valid": None, "test": None}
        return folds, caps, args.pilot_events_per_class

    folds = {
        "train": train_params["train_folds"],
        "valid": train_params["valid_folds"],
        "test": train_params["test_folds"],
    }
    if args.max_events_per_class is not None:
        caps = {
            "train": args.max_events_per_class,
            "valid": args.max_events_per_class,
            "test": args.max_events_per_class if args.cap_test else None,
        }
    else:
        cap = train_params.get("max_events_per_fold_per_class")
        caps = {"train": cap, "valid": cap, "test": None}
    return folds, caps, None


def maybe_build_source_table(args, config, signal: str, split: str, folds: Sequence[int],
                             cap: Optional[int], pilot_events_per_class: Optional[int],
                             source_dir: Path) -> Path:
    suffix = bdt.cache_suffix(cap, pilot_events_per_class)
    existing = bdt.PARTICLENETMD_DIR / "BDT" / "Combined" / signal / "fold-4" / "tables" / f"{split}_{suffix}.npz"
    if existing.exists() and not args.rebuild_dataset:
        return existing

    bdt.build_or_load_table(
        config,
        signal,
        split,
        folds,
        cap,
        args.workers,
        source_dir.parent,
        args.rebuild_dataset,
        pilot_events_per_class=pilot_events_per_class,
    )
    return source_dir / f"{split}_{suffix}.npz"


def ensure_masked_tables(args, signal: str) -> Path:
    config = load_sgl_config(args.config)
    folds, caps, pilot_events_per_class = split_settings(args, config)
    out_dir = COMPARISON_ROOT / "dataset" / signal / "fold-4"
    table_dir = out_dir / "tables"
    source_dir = out_dir / "source" / "tables"

    full_names = list(bdt.FEATURE_NAMES)
    drop_indices = [full_names.index(name) for name in MASKED_COLUMNS]
    keep_indices = [idx for idx in range(len(full_names)) if idx not in drop_indices]
    masked_names = [full_names[idx] for idx in keep_indices]

    table_dir.mkdir(parents=True, exist_ok=True)
    with open(table_dir / "feature_names.json", "w") as handle:
        json.dump(masked_names, handle, indent=2)

    manifest = {
        "signal": signal,
        "masked_columns": MASKED_COLUMNS,
        "drop_indices": drop_indices,
        "source_feature_count": len(full_names),
        "masked_feature_count": len(masked_names),
        "splits": {},
    }

    for split in ["train", "valid", "test"]:
        suffix = bdt.cache_suffix(caps[split], pilot_events_per_class)
        target = table_dir / f"{split}_{suffix}.npz"
        if target.exists() and not args.rebuild_dataset:
            arrays = load_table(target)
            source = target
        else:
            source = maybe_build_source_table(
                args, config, signal, split, folds[split], caps[split], pilot_events_per_class, source_dir
            )
            arrays = load_table(source)
            arrays = dict(arrays)
            arrays["X"] = arrays["X"][:, keep_indices].astype(np.float32)
            save_table(target, arrays)
        if arrays["X"].shape[1] != len(masked_names):
            raise RuntimeError(f"{target} has {arrays['X'].shape[1]} features, expected {len(masked_names)}")
        manifest["splits"][split] = {
            "source": str(source),
            "target": str(target),
            "events": int(len(arrays["y"])),
            "features": int(arrays["X"].shape[1]),
        }

    with open(out_dir / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    return table_dir


def run_command(cmd: Sequence[str]) -> None:
    print("  " + " ".join(cmd), flush=True)
    subprocess.run(list(cmd), check=True)


def load_json(path: Path) -> Dict[str, object]:
    with open(path) as handle:
        return json.load(handle)


def run_tabular_training(args, signal: str, table_dir: Path, model: str) -> None:
    if model not in {"BDT", "DNN", "DNN_MD"}:
        raise ValueError(f"Unsupported tabular model: {model}")
    script = "python/trainBDT.py" if model == "BDT" else "python/trainDNN.py"
    output_base = f"ModelComparison/{model}"
    cmd = [
        sys.executable,
        script,
        "--signal", signal,
        "--workers", str(args.workers),
        "--device", args.device,
        "--table-cache-dir", str(table_dir),
        "--feature-names", str(table_dir / "feature_names.json"),
        "--output-base", output_base,
        "--skip-pn",
    ]
    if args.config:
        cmd.extend(["--config", args.config])
    if args.pilot:
        cmd.extend(["--pilot", "--pilot-events-per-class", str(args.pilot_events_per_class)])
    if args.max_events_per_class is not None:
        cmd.extend(["--max-events-per-class", str(args.max_events_per_class)])
    if args.cap_test:
        cmd.append("--cap-test")
    if model in {"DNN", "DNN_MD"}:
        cmd.extend(["--loss-type", "disco" if model == "DNN_MD" else "weighted_ce"])
        if args.max_epochs is not None:
            cmd.extend(["--max-epochs", str(args.max_epochs)])
        if args.hidden_layers:
            cmd.extend(["--hidden-layers", args.hidden_layers])
        if model == "DNN_MD" and args.disco_lambda is not None:
            cmd.extend(["--disco-lambda", str(args.disco_lambda)])
    run_command(cmd)


def write_particlenet_weighted_ce_config(args, signal: str) -> Path:
    base_config = copy.deepcopy(load_sgl_config(args.config).config)
    ga_path = bdt.PARTICLENETMD_DIR / "GAOptim" / "Combined" / signal / "fold-4" / "ga_optimization_results.json"
    if not ga_path.exists():
        raise RuntimeError(f"GA best config is missing: {ga_path}")
    decoded = load_json(ga_path)["best_chromosome"]["decoded"]

    ga_config_path = bdt.PARTICLENETMD_DIR / "configs" / "GAConfig.json"
    ga_config = load_json(ga_config_path) if ga_config_path.exists() else {}
    ga_training = ga_config.get("training_parameters", {})

    train_params = base_config["training_parameters"]
    for key in ["max_epochs", "batch_size", "dropout_p", "early_stopping_patience", "train_folds", "valid_folds"]:
        if key in ga_training:
            train_params[key] = ga_training[key]
    train_params["loss_type"] = "weighted_ce"
    train_params["test_folds"] = train_params.get("test_folds", [4])

    base_config["model_config"]["nNodes"] = int(decoded["nNodes"])
    base_config["optimization_config"]["optimizer"] = decoded["optimizer"]
    base_config["optimization_config"]["initLR"] = float(decoded["initLR"])
    base_config["optimization_config"]["weight_decay"] = float(decoded["weight_decay"])
    base_config["optimization_config"]["scheduler"] = decoded["scheduler"]
    base_config["disco_parameters"]["disco_lambda"] = 0.0
    base_config["system_config"]["device"] = args.device
    base_config["output_config"]["results_dir"] = "ModelComparison/ParticleNet"
    base_config["description"] = f"ModelComparison no-decorrelation ParticleNet config for {signal}"

    out_dir = COMPARISON_ROOT / "configs" / "ParticleNet"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{signal}_weighted_ce_best_config.json"
    with open(out_path, "w") as handle:
        json.dump(base_config, handle, indent=2)
    return out_path


def run_particlenet_training(args, signal: str) -> None:
    config_path = write_particlenet_weighted_ce_config(args, signal)
    cmd = [
        sys.executable,
        "python/trainMultiClass.py",
        "--signal", signal,
        "--channel", "Combined",
        "--config", str(config_path),
    ]
    if args.pilot:
        cmd.append("--pilot")
    run_command(cmd)


def find_particlenet_weighted_ce_artifacts(signal: str) -> Tuple[Path, Path, Path]:
    out_dir = COMPARISON_ROOT / "ParticleNet" / "Combined" / signal / "fold-4"
    model_paths = sorted((out_dir / "models").glob("*-weighted_ce-*.pt"))
    if not model_paths:
        raise RuntimeError(f"No weighted-CE ParticleNet checkpoint found under {out_dir / 'models'}")
    model_path = model_paths[-1]
    model_name = model_path.stem
    json_path = out_dir / f"{model_name}.json"
    if not json_path.exists():
        raise RuntimeError(f"Missing ParticleNet weighted-CE metadata: {json_path}")
    return out_dir, model_path, json_path


def evaluate_particle_net_artifacts(config, signal: str, fold_list: Sequence[int],
                                    max_events_per_fold: Optional[int], workers: int,
                                    device: str, model_path: Path, info_path: Path,
                                    batch_size: int = 512,
                                    pilot_events_per_class: Optional[int] = None,
                                    split_name: str = "split") -> np.ndarray:
    print(f"  Loading ParticleNet artifact and matching {split_name} split...", flush=True)
    data_list = bdt.load_split_data(
        config,
        signal,
        fold_list,
        max_events_per_fold,
        workers,
        pilot_events_per_class=pilot_events_per_class,
    )
    model, hp = bdt.load_particle_net_model(model_path, info_path, device)
    loader = bdt.DataLoader(
        bdt.GraphDataset(data_list),
        batch_size=hp.get("batch_size", batch_size),
        shuffle=False,
    )

    scores: List[np.ndarray] = []
    with bdt.torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.graphInput, batch.batch)
            scores.append(bdt.F.softmax(logits, dim=1).cpu().numpy())

    del model
    if device.startswith("cuda"):
        bdt.torch.cuda.empty_cache()
    return np.concatenate(scores, axis=0)


def save_particlenet_reference(args, signal: str, table_dir: Path, model: str) -> None:
    config = load_sgl_config(args.config)
    folds, caps, pilot_events_per_class = split_settings(args, config)
    out_dir = COMPARISON_ROOT / model / "Combined" / signal / "fold-4"
    out_dir.mkdir(parents=True, exist_ok=True)

    thresholds = bdt.load_thresholds(signal)
    summary = {"signal": signal, "backend": model, "channel": "Combined"}
    if model == "ParticleNet":
        _train_dir, model_path, info_path = find_particlenet_weighted_ce_artifacts(signal)
        summary["source_model"] = str(model_path)
        summary["source_metadata"] = str(info_path)
    elif model == "ParticleNet_MD":
        model_path = bdt.PARTICLENETMD_DIR / "GAOptim" / "Combined" / signal / "fold-4" / "best_model" / "model.pt"
        info_path = bdt.PARTICLENETMD_DIR / "GAOptim" / "Combined" / signal / "fold-4" / "best_model" / "model_info.json"
        summary["source_model"] = str(model_path)
        summary["source_metadata"] = str(info_path)
    else:
        raise ValueError(f"Unsupported ParticleNet export model: {model}")

    for split in ["train", "test"]:
        table_path = table_dir / f"{split}_{bdt.cache_suffix(caps[split], pilot_events_per_class)}.npz"
        table = load_table(table_path)
        scores = evaluate_particle_net_artifacts(
            config,
            signal,
            folds[split],
            caps[split],
            args.workers,
            args.device,
            model_path,
            info_path,
            pilot_events_per_class=pilot_events_per_class,
            split_name=split,
        )
        if len(scores) != len(table["y"]):
            raise RuntimeError(f"{model}/{split} length mismatch: {len(scores)} != {len(table['y'])}")
        lr = bdt.compute_lr_modified(scores, table["era"], table["channel_id"], thresholds)
        np.savez_compressed(
            out_dir / f"predictions_{split}.npz",
            y=table["y"],
            weight=table["weight"],
            mass1=table["mass1"],
            mass2=table["mass2"],
            era=table["era"],
            channel_id=table["channel_id"],
            pn_scores=scores,
            pn_lr=lr,
        )
        avg_auc, aucs = bdt.average_signal_vs_bg_auc(table["y"], scores, table["weight"])
        summary[split] = {
            "auc": aucs,
            "average_auc": avg_auc,
            "mass_correlation": bdt.mass_correlation_metrics(scores, table),
        }
    summary["loss_type"] = "weighted_ce" if model == "ParticleNet" else "disco"
    summary["disco_lambda"] = 0.0 if model == "ParticleNet" else 0.1

    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)


def model_source_dir(model: str) -> str:
    return str(MODEL_REGISTRY[model]["source_dir"])


def model_output_tag(model: str) -> str:
    return str(MODEL_REGISTRY[model]["tag"])


def prediction_path(signal: str, model: str, split: str) -> Path:
    return COMPARISON_ROOT / model_source_dir(model) / "Combined" / signal / "fold-4" / f"predictions_{split}.npz"


def load_prediction(signal: str, model: str, split: str) -> Dict[str, np.ndarray]:
    path = prediction_path(signal, model, split)
    return load_table(path)


def load_available_predictions(signal: str, models: Sequence[str]) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
    preds: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for model in models:
        split_paths = {split: prediction_path(signal, model, split) for split in SPLITS}
        missing = [str(path) for path in split_paths.values() if not path.exists()]
        if missing:
            print(f"  Warning: skipping {MODEL_DISPLAY[model]} for {signal}; missing {', '.join(missing)}", flush=True)
            continue
        preds[model] = {split: load_table(path) for split, path in split_paths.items()}
    return preds


def model_scores(pred: Dict[str, np.ndarray], model: str) -> np.ndarray:
    return pred[MODEL_SCORE_KEYS[model]]


def model_lr(pred: Dict[str, np.ndarray], model: str) -> np.ndarray:
    return pred[MODEL_LR_KEYS[model]]


def require_root_cmsstyle() -> None:
    if not bdt.HAS_ROOT_CMSSTYLE:
        raise RuntimeError("ROOT/cmsstyle plotting is required for ModelComparison outputs")
    bdt.setup_root_cms_style()
    bdt.CMS.SetLumi(None, run="")
    bdt.ROOT.gStyle.SetLineStyleString(TRAIN_LINE_STYLE, "24 12")


def model_root_color(model: str) -> int:
    return bdt.root_color(MODEL_ROOT_COLORS[model])


def make_plot_roc_graph(tpr: np.ndarray, fpr: np.ndarray, max_points: int = 150):
    """Resample ROC curves by arc length so dashed TGraphs render uniformly."""
    finite = np.isfinite(tpr) & np.isfinite(fpr)
    tpr = tpr[finite]
    fpr = fpr[finite]
    if len(tpr) > max_points:
        dx = np.diff(tpr)
        dy = np.diff(fpr)
        arc = np.r_[0.0, np.cumsum(np.hypot(dx, dy))]
        if arc[-1] > 0:
            target = np.linspace(0.0, arc[-1], max_points)
            tpr = np.interp(target, arc, tpr)
            fpr = np.interp(target, arc, fpr)
    return bdt.make_roc_graph(tpr, fpr)


def background_root_color(bg_class: int) -> int:
    return bdt.palette_root_color(bg_class - 1)


def roc_background_label(bg_class: int) -> str:
    return {1: "NP", 2: "VV", 3: "ttX"}[bg_class]


def calculate_binary_roc(roc: object, pred: Dict[str, np.ndarray], model: str,
                         bg_class: int) -> Tuple[object, float]:
    scores = model_scores(pred, model)
    mask = (pred["y"] == 0) | (pred["y"] == bg_class)
    y_bin = (pred["y"][mask] == 0).astype(int)
    lr = bdt.binary_lr(scores[mask], bg_class)
    fpr, tpr, auc = roc.calculate_roc_curve(y_bin, lr, pred["weight"][mask])
    return make_plot_roc_graph(tpr, fpr), float(auc)


def draw_roc_frame(signal: str, subtitle: str):
    ROOT = bdt.ROOT
    CMS = bdt.CMS
    canvas = CMS.cmsCanvas(
        "",
        0.0,
        1.0,
        0.0,
        1.0,
        "signal efficiency",
        "Background Efficiency",
        square=True,
        iPos=0,
        extraSpace=0.0,
    )
    canvas.SetGrid()
    keepalive = []

    diag = ROOT.TGraph(2)
    diag.SetPoint(0, 0.0, 0.0)
    diag.SetPoint(1, 1.0, 1.0)
    CMS.cmsObjectDraw(diag, "L", LineColor=ROOT.kGray + 2, LineWidth=PLOT_LINE_WIDTH, LineStyle=ROOT.kDashed)
    keepalive.append(diag)
    CMS.drawText(signal, posX=0.20, posY=0.82, font=62, align=0, size=0.044)
    CMS.drawText(subtitle, posX=0.20, posY=0.75, font=42, align=0, size=0.040)
    return canvas, keepalive


def plot_model_train_test_rocs(signal: str, preds: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                               out_dir: Path) -> Dict[str, object]:
    require_root_cmsstyle()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {}
    roc = bdt.ROCCurveCalculator()
    ROOT = bdt.ROOT
    CMS = bdt.CMS

    for model, split_preds in preds.items():
        canvas, keepalive = draw_roc_frame(signal, MODEL_DISPLAY[model])
        legend = CMS.cmsLeg(0.20, 0.53, 0.78, 0.72, textSize=0.030, columns=2)
        for method, value in [("SetMargin", 0.14), ("SetColumnSeparation", 0.04), ("SetEntrySeparation", 0.01)]:
            if hasattr(legend, method):
                getattr(legend, method)(value)

        for bg_class in [1, 2, 3]:
            bg_name = bdt.CLASS_NAMES[bg_class]
            bg_label = roc_background_label(bg_class)
            color = background_root_color(bg_class)
            graph_by_split = {}
            auc_by_split = {}
            for split in ["train", "test"]:
                graph, auc = calculate_binary_roc(roc, split_preds[split], model, bg_class)
                CMS.cmsObjectDraw(
                    graph,
                    "C",
                    LineColor=color,
                    LineWidth=PLOT_LINE_WIDTH,
                    LineStyle=ROOT.kSolid if split == "test" else TRAIN_LINE_STYLE,
                )
                keepalive.append(graph)
                graph_by_split[split] = graph
                auc_by_split[split] = auc
            for split in ["train", "test"]:
                CMS.addToLegend(
                    legend,
                    (graph_by_split[split], f"{bg_label} {split} {auc_by_split[split]:.4f}", "L"),
                )
            summary.setdefault(model, {})[bg_name] = auc_by_split

        legend.Draw()
        canvas.RedrawAxis()
        canvas._keepalive = keepalive
        save_root_canvas_with_pdf(canvas, out_dir / f"roc_{model_output_tag(model)}_train_test.png")

    return summary


def plot_model_comparison_test_rocs(signal: str, preds: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                                    out_dir: Path) -> Dict[str, object]:
    require_root_cmsstyle()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, object] = {}
    roc = bdt.ROCCurveCalculator()
    CMS = bdt.CMS

    for bg_class in [1, 2, 3]:
        bg_name = bdt.CLASS_NAMES[bg_class]
        canvas, keepalive = draw_roc_frame(signal, f"signal vs {bg_name}")
        legend = CMS.cmsLeg(0.20, 0.53, 0.72, 0.72, textSize=0.030, columns=1)
        if hasattr(legend, "SetMargin"):
            legend.SetMargin(0.14)

        for model, split_preds in preds.items():
            graph, auc = calculate_binary_roc(roc, split_preds["test"], model, bg_class)
            CMS.cmsObjectDraw(
                graph,
                "C",
                LineColor=model_root_color(model),
                LineWidth=PLOT_LINE_WIDTH,
                LineStyle=bdt.ROOT.kSolid,
            )
            keepalive.append(graph)
            CMS.addToLegend(legend, (graph, f"{MODEL_DISPLAY[model]} {auc:.4f}", "L"))
            summary.setdefault(bg_name, {})[model] = {"test": auc}

        legend.Draw()
        canvas.RedrawAxis()
        canvas._keepalive = keepalive
        save_root_canvas_with_pdf(canvas, out_dir / f"roc_model_comparison_{bg_name}.png")

    return summary


def plot_roc_suite(signal: str, preds: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                   out_dir: Path) -> Dict[str, object]:
    return {
        "per_model": plot_model_train_test_rocs(signal, preds, out_dir),
        "model_comparison": plot_model_comparison_test_rocs(signal, preds, out_dir),
    }


def plot_lr_distributions(signal: str, preds: Dict[str, Dict[str, Dict[str, np.ndarray]]], out_dir: Path) -> None:
    require_root_cmsstyle()
    ROOT = bdt.ROOT
    CMS = bdt.CMS
    out_dir.mkdir(parents=True, exist_ok=True)

    for class_idx, class_name in enumerate(bdt.CLASS_NAMES):
        hists = []
        for model, split_preds in preds.items():
            for split in SPLITS:
                pred = split_preds[split]
                mask = pred["y"] == class_idx
                if mask.sum() == 0:
                    continue
                hist = bdt.make_root_hist(
                    f"h_lr_{signal}_{model}_{split}_{class_name}",
                    model_lr(pred, model)[mask],
                    pred["weight"][mask],
                    30,
                    0.0,
                    1.0,
                    use_abs_weight=True,
                )
                bdt.normalize_root_hist(hist)
                hist.SetLineColor(model_root_color(model))
                hist.SetLineWidth(PLOT_LINE_WIDTH)
                hist.SetLineStyle(ROOT.kSolid if split == "test" else TRAIN_LINE_STYLE)
                hist.SetMarkerSize(0)
                hists.append((model, split, hist))
        if not hists:
            continue

        ymax = max(hist.GetMaximum() for _model, _split, hist in hists)
        canvas = CMS.cmsCanvas(
            "",
            0.0,
            1.0,
            0.0,
            max(0.01, ymax * 1.55),
            "LR_{modified}",
            "Normalized",
            square=True,
            iPos=0,
            extraSpace=0.0,
        )
        canvas.SetGrid()
        legend = CMS.cmsLeg(0.52, 0.58, 0.92, 0.86, textSize=0.028, columns=2)
        keepalive = []
        for model, split, hist in hists:
            CMS.cmsObjectDraw(hist, "hist", LineColor=hist.GetLineColor(), LineWidth=hist.GetLineWidth())
            CMS.cmsObjectDraw(
                hist,
                "E0 SAME",
                LineColor=hist.GetLineColor(),
                LineWidth=hist.GetLineWidth(),
                MarkerColor=hist.GetLineColor(),
                MarkerSize=0,
            )
            CMS.addToLegend(legend, (hist, f"{MODEL_DISPLAY[model]} {split}", "L"))
            keepalive.append(hist)
        legend.Draw()
        CMS.drawText(signal, posX=0.20, posY=0.76, font=62, align=0, size=0.034)
        CMS.drawText(class_name, posX=0.20, posY=0.69, font=42, align=0, size=0.034)
        canvas.RedrawAxis()
        canvas._keepalive = keepalive
        bdt.save_root_canvas(canvas, out_dir / f"lr_three_models_{class_name}.png")


def plot_mass_sculpting(signal: str, preds: Dict[str, Dict[str, Dict[str, np.ndarray]]], out_dir: Path,
                        mass_name: str) -> None:
    require_root_cmsstyle()
    ROOT = bdt.ROOT
    CMS = bdt.CMS
    out_dir.mkdir(parents=True, exist_ok=True)

    region_defs = [
        ("low", "LR < 0.3", lambda lr: lr < 0.3, bdt.palette_root_color(0)),
        ("mid", "0.3 < LR < 0.7", lambda lr: (lr > 0.3) & (lr < 0.7), bdt.palette_root_color(1)),
        ("high", "LR > 0.7", lambda lr: lr > 0.7, bdt.palette_root_color(2)),
    ]

    class_selections = [("background", None)] + [(bdt.CLASS_NAMES[idx], idx) for idx in range(len(bdt.CLASS_NAMES))]

    for model in preds:
        for class_label, class_idx in class_selections:
            plot_mass_sculpting_one_class(signal, preds, out_dir, mass_name, model, class_label, class_idx, region_defs)


def plot_mass_sculpting_one_class(signal: str, preds: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                                  out_dir: Path, mass_name: str, model: str,
                                  class_label: str, class_idx: Optional[int],
                                  region_defs: Sequence[Tuple[str, str, object, int]]) -> None:
    ROOT = bdt.ROOT
    CMS = bdt.CMS
    if class_idx is None:
        class_selector = lambda pred: pred["y"] != 0
    else:
        class_selector = lambda pred: pred["y"] == class_idx

    def output_suffix() -> str:
        if class_idx is None:
            return f"mass_sculpting_{model_output_tag(model)}_{mass_name}"
        return f"mass_sculpting_{model_output_tag(model)}_{class_label}_{mass_name}"

    hists = []
    refs: Dict[str, object] = {}
    dcors: Dict[str, float] = {}

    for split in SPLITS:
        pred = preds[model][split]
        mass = pred[mass_name]
        lr = model_lr(pred, model)
        base = class_selector(pred) & (mass > 0)
        if base.sum() == 0:
            continue

        dcors[split] = bdt.compute_disco(
            lr[base],
            mass[base],
            np.abs(pred["weight"][base]),
        )

        ref_hist = bdt.make_root_hist(
            f"h_mass_{signal}_{model}_{class_label}_{split}_{mass_name}_nocut",
            mass[base],
            pred["weight"][base],
            30,
            60.0,
            120.0,
            use_abs_weight=True,
        )
        bdt.normalize_root_hist(ref_hist)
        ref_hist.SetLineColor(ROOT.kBlack)
        ref_hist.SetLineWidth(PLOT_LINE_WIDTH)
        ref_hist.SetLineStyle(ROOT.kSolid if split == "test" else TRAIN_LINE_STYLE)
        ref_hist.SetMarkerSize(0)
        refs[split] = ref_hist
        hists.append((split, "No cut", ref_hist))

        for suffix, label, selector, color in region_defs:
            mask = base & selector(lr)
            if mask.sum() == 0:
                continue
            hist = bdt.make_root_hist(
                f"h_mass_{signal}_{model}_{class_label}_{split}_{mass_name}_{suffix}",
                mass[mask],
                pred["weight"][mask],
                30,
                60.0,
                120.0,
                use_abs_weight=True,
            )
            bdt.normalize_root_hist(hist)
            hist.SetLineColor(color)
            hist.SetLineWidth(PLOT_LINE_WIDTH)
            hist.SetLineStyle(ROOT.kSolid if split == "test" else TRAIN_LINE_STYLE)
            hist.SetMarkerSize(0)
            hists.append((split, label, hist))

    if not hists or not refs:
        return

    ymax = max(hist.GetMaximum() for _split, _label, hist in hists)
    canvas = CMS.cmsDiCanvas(
        "",
        60.0,
        120.0,
        0.0,
        max(0.01, ymax * 2.0),
        0.0,
        2.0,
        f"{mass_name} [GeV]",
        "Normalized",
        "Region / No cut",
        square=True,
        iPos=0,
        extraSpace=0.0,
    )
    canvas.cd(2)
    ratio_frame = canvas.cd(2).GetPrimitive("hframe")
    if ratio_frame:
        ratio_frame.GetYaxis().CenterTitle()
        ratio_frame.GetYaxis().SetTitleSize(0.115)
        ratio_frame.GetYaxis().SetTitleOffset(0.58)
    canvas.cd(1)
    canvas.cd(1).SetGrid(0, 0)
    legend = CMS.cmsLeg(0.45, 0.55, 0.94, 0.88, textSize=0.034, columns=2)
    for method, value in [("SetMargin", 0.14), ("SetColumnSeparation", 0.03), ("SetEntrySeparation", 0.01)]:
        if hasattr(legend, method):
            getattr(legend, method)(value)
    keepalive = []

    canvas.cd(1)
    for split, label, hist in hists:
        CMS.cmsObjectDraw(
            hist,
            "hist",
            LineColor=hist.GetLineColor(),
            LineWidth=hist.GetLineWidth(),
            LineStyle=hist.GetLineStyle(),
        )
        CMS.cmsObjectDraw(
            hist,
            "E0 SAME",
            LineColor=hist.GetLineColor(),
            LineWidth=hist.GetLineWidth(),
            LineStyle=hist.GetLineStyle(),
            MarkerColor=hist.GetLineColor(),
            MarkerSize=0,
        )
        keepalive.append(hist)
    hist_by_legend_key = {(split, label): hist for split, label, hist in hists}
    legend_label_order = ["No cut"] + [label for _suffix, label, _selector, _color in region_defs]
    for label in legend_label_order:
        for split in SPLITS:
            hist = hist_by_legend_key.get((split, label))
            if hist is None:
                continue
            CMS.addToLegend(legend, (hist, f"{label} {split}", "L"))
    legend.Draw()
    CMS.drawText(signal, posX=0.20, posY=0.77, font=62, align=0, size=0.046)
    CMS.drawText(f"{MODEL_DISPLAY[model]} {class_label}", posX=0.20, posY=0.69, font=42, align=0, size=0.042)
    y_text = 0.61
    for split in SPLITS:
        if split in dcors:
            CMS.drawText(
                f"dCor {split} = {dcors[split]:.4f}",
                posX=0.20,
                posY=y_text,
                font=42,
                align=0,
                size=0.038,
            )
            y_text -= 0.065
    canvas.cd(1).RedrawAxis()

    canvas.cd(2)
    canvas.cd(2).SetGrid()
    ref_line = ROOT.TLine(60.0, 1.0, 120.0, 1.0)
    ref_line.SetLineStyle(ROOT.kDotted)
    ref_line.SetLineColor(ROOT.kBlack)
    ref_line.SetLineWidth(PLOT_LINE_WIDTH)
    ref_line.Draw()
    keepalive.append(ref_line)

    for split, label, hist in hists:
        if label == "No cut" or split not in refs:
            continue
        ratio = bdt.make_ratio_hist(hist, refs[split], f"{hist.GetName()}_ratio")
        CMS.cmsObjectDraw(
            ratio,
            "hist",
            LineColor=hist.GetLineColor(),
            LineWidth=hist.GetLineWidth(),
            LineStyle=hist.GetLineStyle(),
        )
        CMS.cmsObjectDraw(
            ratio,
            "E0 SAME",
            LineColor=hist.GetLineColor(),
            LineWidth=hist.GetLineWidth(),
            LineStyle=hist.GetLineStyle(),
            MarkerColor=hist.GetLineColor(),
            MarkerSize=0,
        )
        keepalive.append(ratio)
    canvas.cd(2).RedrawAxis()
    canvas._keepalive = keepalive
    save_root_canvas_with_pdf(canvas, out_dir / f"{output_suffix()}.png")


def plot_score_vs_mass_diagnostics(signal: str, preds: Dict[str, Dict[str, Dict[str, np.ndarray]]],
                                   out_dir: Path) -> None:
    require_root_cmsstyle()
    ROOT = bdt.ROOT
    CMS = bdt.CMS
    out_dir.mkdir(parents=True, exist_ok=True)

    for model, split_preds in preds.items():
        pred = split_preds["test"]
        signal_score = model_scores(pred, model)[:, 0]
        weights = np.abs(pred["weight"])

        for mass_name in ["mass1", "mass2"]:
            mass = pred[mass_name]
            valid_mass = mass > 0
            if valid_mass.sum() < 10:
                continue

            profile = ROOT.TProfile(
                f"h_profile_{signal}_{model}_{mass_name}",
                "",
                20,
                0.0,
                1.0,
                60.0,
                120.0,
            )
            profile.SetDirectory(0)
            finite = valid_mass & np.isfinite(signal_score) & np.isfinite(mass) & np.isfinite(weights)
            for score, mass_value, weight in zip(signal_score[finite], mass[finite], weights[finite]):
                profile.Fill(float(score), float(mass_value), float(weight))

            mean_mass = float(np.average(mass[finite], weights=weights[finite])) if finite.any() else 0.0
            canvas = CMS.cmsCanvas(
                "",
                0.0,
                1.0,
                60.0,
                120.0,
                "p_{sig}",
                f"<{mass_name}> [GeV]",
                square=True,
                iPos=0,
                extraSpace=0.0,
            )
            canvas.SetGrid()
            CMS.cmsObjectDraw(
                profile,
                "E",
                LineColor=model_root_color(model),
                MarkerColor=model_root_color(model),
                LineWidth=PLOT_LINE_WIDTH,
                MarkerStyle=20,
                MarkerSize=1.0,
            )
            mean_line = ROOT.TLine(0.0, mean_mass, 1.0, mean_mass)
            mean_line.SetLineStyle(ROOT.kDashed)
            mean_line.SetLineColor(ROOT.kGray + 2)
            mean_line.SetLineWidth(PLOT_LINE_WIDTH)
            mean_line.Draw()
            CMS.drawText(signal, posX=0.20, posY=0.76, font=62, align=0, size=0.034)
            CMS.drawText(f"{MODEL_DISPLAY[model]} {mass_name}", posX=0.20, posY=0.69, font=42, align=0, size=0.034)
            canvas.RedrawAxis()
            canvas._keepalive = [profile, mean_line]
            bdt.save_root_canvas(canvas, out_dir / f"mass_profile_{model_output_tag(model)}_{mass_name}.png")

            for class_idx, class_name in enumerate(bdt.CLASS_NAMES):
                class_mask = (pred["y"] == class_idx) & valid_mass
                if class_mask.sum() < 10:
                    continue
                hist = ROOT.TH2D(
                    f"h_score_mass_{signal}_{model}_{class_name}_{mass_name}",
                    "",
                    50,
                    0.0,
                    1.0,
                    50,
                    60.0,
                    120.0,
                )
                hist.SetDirectory(0)
                finite_class = class_mask & np.isfinite(signal_score) & np.isfinite(mass) & np.isfinite(weights)
                for score, mass_value, weight in zip(signal_score[finite_class], mass[finite_class], weights[finite_class]):
                    hist.Fill(float(score), float(mass_value), float(weight))

                canvas2 = CMS.cmsCanvas(
                    "",
                    0.0,
                    1.0,
                    60.0,
                    120.0,
                    "p_{sig}",
                    f"{mass_name} [GeV]",
                    square=True,
                    iPos=0,
                    extraSpace=0.0,
                )
                canvas2.SetRightMargin(0.15)
                CMS.cmsObjectDraw(hist, "COLZ")
                CMS.drawText(signal, posX=0.20, posY=0.76, font=62, align=0, size=0.034)
                CMS.drawText(
                    f"{MODEL_DISPLAY[model]} {class_name}",
                    posX=0.20,
                    posY=0.69,
                    font=42,
                    align=0,
                    size=0.034,
                )
                canvas2.RedrawAxis()
                canvas2._keepalive = [hist]
                bdt.save_root_canvas(
                    canvas2,
                    out_dir / f"score_vs_mass_{model_output_tag(model)}_{class_name}_{mass_name}.png",
                )

def peak_metrics(pred: Dict[str, np.ndarray], model: str, mass_name: str = "mass2") -> Dict[str, float]:
    mass = pred[mass_name]
    lr = model_lr(pred, model)
    w = np.abs(pred["weight"])
    region = (pred["y"] != 0) & (mass > 60) & (mass < 120)
    peak = region & (mass > 85) & (mass < 95)
    high = region & (lr > 0.7)
    peak_high = high & (mass > 85) & (mass < 95)
    all_w = float(w[region].sum())
    high_w = float(w[high].sum())
    return {
        "dcor_lr_mass2_60_120": bdt.compute_disco(lr[region], mass[region], w[region]) if region.sum() else 0.0,
        "high_lr_fraction_60_120": high_w / all_w if all_w > 0 else 0.0,
        "peak_fraction_all_60_120": float(w[peak].sum()) / all_w if all_w > 0 else 0.0,
        "peak_fraction_high_lr_60_120": float(w[peak_high].sum()) / high_w if high_w > 0 else 0.0,
    }


def plot_nominal_shift(signal: str, model: str, masked_pred: Dict[str, np.ndarray], out_dir: Path) -> Optional[Dict[str, float]]:
    nominal_path = bdt.PARTICLENETMD_DIR / model / "Combined" / signal / "fold-4" / "predictions_test.npz"
    if not nominal_path.exists():
        return None
    nominal = load_table(nominal_path)
    nominal_key = MODEL_LR_KEYS[model]
    if nominal_key not in nominal or len(nominal[nominal_key]) != len(masked_pred["y"]):
        return None
    if not np.array_equal(nominal["y"], masked_pred["y"]):
        return None

    require_root_cmsstyle()
    ROOT = bdt.ROOT
    CMS = bdt.CMS

    masked_lr = model_lr(masked_pred, model)
    delta = masked_lr - nominal[nominal_key]
    weights = np.abs(masked_pred["weight"])

    h_delta = bdt.make_root_hist(
        f"h_lr_shift_{signal}_{model}",
        delta,
        weights,
        60,
        -1.0,
        1.0,
        use_abs_weight=False,
    )
    h_delta.SetLineColor(model_root_color(model))
    h_delta.SetLineWidth(PLOT_LINE_WIDTH)
    h_delta.SetMarkerSize(0)
    canvas = CMS.cmsCanvas(
        "",
        -1.0,
        1.0,
        0.0,
        max(0.01, h_delta.GetMaximum() * 1.45),
        "Masked LR #minus nominal LR",
        "Weighted events",
        square=True,
        iPos=0,
        extraSpace=0.0,
    )
    canvas.SetGrid()
    CMS.cmsObjectDraw(h_delta, "hist", LineColor=h_delta.GetLineColor(), LineWidth=PLOT_LINE_WIDTH)
    CMS.drawText(f"{signal} {model}", posX=0.20, posY=0.76, font=62, align=0, size=0.034)
    canvas._keepalive = [h_delta]
    bdt.save_root_canvas(canvas, out_dir / f"lr_shift_{model}_delta.png")

    h2 = ROOT.TH2D(f"h_lr_shift2d_{signal}_{model}", "", 50, 0.0, 1.0, 50, 0.0, 1.0)
    h2.SetDirectory(0)
    finite = np.isfinite(nominal[nominal_key]) & np.isfinite(masked_lr) & np.isfinite(weights)
    for xval, yval, weight in zip(nominal[nominal_key][finite], masked_lr[finite], weights[finite]):
        h2.Fill(float(xval), float(yval), float(weight))
    canvas2 = CMS.cmsCanvas(
        "",
        0.0,
        1.0,
        0.0,
        1.0,
        "Nominal LR",
        "Masked LR",
        square=True,
        iPos=0,
        extraSpace=0.0,
    )
    canvas2.SetGrid()
    CMS.cmsObjectDraw(h2, "COLZ")
    diag = ROOT.TLine(0.0, 0.0, 1.0, 1.0)
    diag.SetLineStyle(ROOT.kDashed)
    diag.SetLineColor(ROOT.kGray + 2)
    diag.SetLineWidth(PLOT_LINE_WIDTH)
    diag.Draw()
    CMS.drawText(f"{signal} {model}", posX=0.20, posY=0.76, font=62, align=0, size=0.034)
    canvas2._keepalive = [h2, diag]
    bdt.save_root_canvas(canvas2, out_dir / f"lr_shift_{model}_scatter.png")
    return {
        "mean_delta": float(np.average(delta, weights=np.abs(masked_pred["weight"]))),
        "rms_delta": float(np.sqrt(np.average(delta * delta, weights=np.abs(masked_pred["weight"])))),
        "max_abs_delta": float(np.max(np.abs(delta))),
    }


def make_comparison_plots(signal: str) -> None:
    out_dir = COMPARISON_ROOT / "plots" / signal
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = {
        model: {split: load_prediction(signal, model, split) for split in SPLITS}
        for model in MODELS
    }

    roc_summary = plot_roc_suite(signal, preds, out_dir)
    plot_lr_distributions(signal, preds, out_dir)
    plot_mass_sculpting(signal, preds, out_dir, "mass1")
    plot_mass_sculpting(signal, preds, out_dir, "mass2")

    rows: List[Dict[str, object]] = []
    shift_summary: Dict[str, object] = {}
    for model, split_preds in preds.items():
        pred = split_preds["test"]
        scores = model_scores(pred, model)
        avg_auc, aucs = bdt.average_signal_vs_bg_auc(pred["y"], scores, pred["weight"])
        metrics = {
            "signal": signal,
            "model": model,
            "test_average_auc": avg_auc,
            "auc_nonprompt": aucs["nonprompt"],
            "auc_diboson": aucs["diboson"],
            "auc_ttX": aucs["ttX"],
            "dcor_psig_mass1": bdt.mass_correlation_metrics(scores, pred)["mass1"]["disco"],
            "dcor_psig_mass2": bdt.mass_correlation_metrics(scores, pred)["mass2"]["disco"],
        }
        metrics.update(peak_metrics(pred, model))
        rows.append(metrics)
        if model == "BDT":
            shift = plot_nominal_shift(signal, model, pred, out_dir)
            if shift is not None:
                shift_summary[model] = shift

    with open(out_dir / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_dir / "summary.json", "w") as handle:
        json.dump({"roc": roc_summary, "metrics": rows, "nominal_to_masked_shift": shift_summary}, handle, indent=2)


def make_decorrelation_plots(signal: str) -> None:
    out_dir = COMPARISON_ROOT / "plots" / signal / "decorrelation"
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = load_available_predictions(signal, DECORRELATION_MODELS)
    if not preds:
        print(f"  Warning: no cached predictions available for decorrelation plots in {signal}", flush=True)
        return

    roc_summary = plot_roc_suite(signal, preds, out_dir)
    plot_mass_sculpting(signal, preds, out_dir, "mass1")
    plot_mass_sculpting(signal, preds, out_dir, "mass2")
    plot_score_vs_mass_diagnostics(signal, preds, out_dir)

    rows: List[Dict[str, object]] = []
    for model, split_preds in preds.items():
        for split, pred in split_preds.items():
            scores = model_scores(pred, model)
            mass_corr = bdt.mass_correlation_metrics(scores, pred)
            rows.append({
                "signal": signal,
                "model": model,
                "display": MODEL_DISPLAY[model],
                "split": split,
                "dcor_psig_mass1": mass_corr["mass1"]["disco"],
                "dcor_psig_mass2": mass_corr["mass2"]["disco"],
                "dcor_lr_background_mass1": bdt.compute_disco(
                    model_lr(pred, model)[(pred["y"] != 0) & (pred["mass1"] > 0)],
                    pred["mass1"][(pred["y"] != 0) & (pred["mass1"] > 0)],
                    np.abs(pred["weight"][(pred["y"] != 0) & (pred["mass1"] > 0)]),
                ) if np.any((pred["y"] != 0) & (pred["mass1"] > 0)) else 0.0,
                "dcor_lr_background_mass2": bdt.compute_disco(
                    model_lr(pred, model)[(pred["y"] != 0) & (pred["mass2"] > 0)],
                    pred["mass2"][(pred["y"] != 0) & (pred["mass2"] > 0)],
                    np.abs(pred["weight"][(pred["y"] != 0) & (pred["mass2"] > 0)]),
                ) if np.any((pred["y"] != 0) & (pred["mass2"] > 0)) else 0.0,
            })

    if rows:
        with open(out_dir / "decorrelation_summary.csv", "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with open(out_dir / "decorrelation_summary.json", "w") as handle:
            json.dump({"roc": roc_summary, "decorrelation": rows}, handle, indent=2)
    print(f"=== Decorrelation plots saved: {out_dir} ===", flush=True)


def process_signal(args, signal: str) -> None:
    print(f"\n=== ModelComparison: {signal} ===", flush=True)
    if args.decorrelation_only:
        make_decorrelation_plots(signal)
        return
    table_dir = ensure_masked_tables(args, signal)
    if args.cache_only:
        print(f"=== Cache ready: {table_dir} ===", flush=True)
        return
    if not args.plots_only:
        if not args.pn_only:
            run_tabular_training(args, signal, table_dir, "BDT")
            run_tabular_training(args, signal, table_dir, "DNN")
            run_tabular_training(args, signal, table_dir, "DNN_MD")
            run_particlenet_training(args, signal)
        save_particlenet_reference(args, signal, table_dir, "ParticleNet")
        save_particlenet_reference(args, signal, table_dir, "ParticleNet_MD")
    make_comparison_plots(signal)
    print(f"=== Done: {COMPARISON_ROOT / 'plots' / signal} ===", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signal", choices=SIGNALS, help="Signal mass point")
    parser.add_argument("--all", action="store_true", help="Run all comparison signals")
    parser.add_argument("--config", default=None, help="Path to SglConfig JSON")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-events-per-class", type=int, default=250)
    parser.add_argument("--max-events-per-class", type=int, default=None)
    parser.add_argument("--cap-test", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None, help="DNN-only max epoch override")
    parser.add_argument("--hidden-layers", default=None, help="DNN-only hidden layer override")
    parser.add_argument("--disco-lambda", type=float, default=None, help="DNN-only DisCo lambda override")
    parser.add_argument("--rebuild-dataset", action="store_true", help="Rebuild masked table cache")
    parser.add_argument("--retrain", action="store_true",
                        help="Accepted for compatibility; tabular trainings always consume the masked cache")
    parser.add_argument("--plots-only", action="store_true", help="Only rebuild comparison plots from existing predictions")
    parser.add_argument("--decorrelation-only", action="store_true",
                        help="Only rebuild architecture decorrelation plots from existing predictions")
    parser.add_argument("--pn-only", action="store_true", help="Only refresh ParticleNet reference and plots")
    parser.add_argument("--cache-only", action="store_true", help="Only build/validate masked tabular caches")
    args = parser.parse_args()
    if not args.all and not args.signal:
        args.signal = SIGNALS[0]
    return args


def main() -> None:
    args = parse_args()
    targets = SIGNALS if args.all else [args.signal]
    for signal in targets:
        process_signal(args, signal)


if __name__ == "__main__":
    main()
