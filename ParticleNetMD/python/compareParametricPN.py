#!/usr/bin/env python3
"""Compare ParametricPN against per-mass GA-best ParticleNetMD baselines."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ROOT
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

try:
    import cmsstyle as CMS
    CMS.setCMSStyle()
    HAS_CMS_STYLE = True
except ImportError:
    HAS_CMS_STYLE = False

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))

from DynamicDatasetLoader import DynamicDatasetLoader
from MultiClassModels import create_multiclass_model
from Preprocess import GraphDataset
from ROCCurveCalculator import PALETTE, ROCCurveCalculator
from SglConfig import load_sgl_config
from WeightedLoss import distance_correlation


CLASS_NAMES = ["signal", "nonprompt", "diboson", "ttX"]
CLASS_DISPLAY = ["Signal", "Nonprompt", "Diboson", "ttX"]
MODEL_DISPLAY = {"parametric": "ParametricPN", "baseline": "ParticleNetMD"}
COLORS = {"parametric": "#1f77b4", "baseline": "#d62728"}
DEFAULT_DCOR_MAX_EVENTS = 3000
MODEL_LINE_STYLES = {"baseline": ROOT.kSolid, "parametric": ROOT.kDashed}
PLOT_LINE_WIDTH = 2
MASS_RANGE = (60.0, 120.0)
MASS_BINS = 30


def parse_ma_values(value: str) -> List[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def infer_num_graph_features(state_dict: Dict[str, torch.Tensor], num_hidden: int) -> int:
    if "dense1.weight" not in state_dict:
        return 8
    return int(state_dict["dense1.weight"].shape[1] - 3 * int(num_hidden))


def load_model(model_path: Path, info_path: Path, device: torch.device):
    with open(info_path) as handle:
        info = json.load(handle)

    hyper = info.get("hyperparameters", info)
    if "architecture" in info:
        architecture = info["architecture"]
        training_config = info.get("training_config", {})
        hyper = {
            "num_hidden": architecture.get("hidden_nodes", 256),
            "conv_channels": architecture.get("conv_channels"),
            "edge_dropout_p": architecture.get("edge_dropout_p", training_config.get("dropout_p", 0.4)),
            "dropout_p": architecture.get("dropout_p", training_config.get("dropout_p", 0.4)),
            "num_classes": info.get("num_classes", training_config.get("num_classes", 4)),
            "model_type": info.get("model_type", training_config.get("model_type", "ParticleNet")),
            "num_node_features": info.get("input_features", {}).get("node_features", 9),
            "num_graph_features": info.get("input_features", {}).get("graph_features", 8),
        }

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) else checkpoint
    num_hidden = hyper.get("num_hidden", hyper.get("hidden_nodes", 256))
    num_graph_features = infer_num_graph_features(state, num_hidden)

    model = create_multiclass_model(
        model_type=hyper.get("model_type", "ParticleNet"),
        num_node_features=hyper.get("num_node_features", 9),
        num_graph_features=num_graph_features,
        num_classes=hyper.get("num_classes", 4),
        num_hidden=num_hidden,
        dropout_p=hyper.get("dropout_p", 0.4),
        edge_dropout_p=hyper.get("edge_dropout_p", hyper.get("dropout_p", 0.4)),
        conv_channels=hyper.get("conv_channels"),
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def find_parametric_artifacts(base_dir: Path) -> Tuple[Path, Path]:
    model_paths = sorted((base_dir / "models").glob("*.pt"), key=lambda path: path.stat().st_mtime, reverse=True)
    if not model_paths:
        raise FileNotFoundError(f"No ParametricPN checkpoints found under {base_dir / 'models'}")
    model_path = model_paths[0]
    info_path = base_dir / f"{model_path.stem}.json"
    if not info_path.exists():
        model_info_path = base_dir / f"{model_path.stem}_model_info.json"
        if not model_info_path.exists():
            raise FileNotFoundError(f"No metadata JSON found for {model_path}")
        info_path = model_info_path
    return model_path, info_path


def clone_with_ma(data, ma_value: int, ma_center: float, ma_scale: float):
    cloned = data.clone()
    graph_input = cloned.graphInput
    if graph_input.dim() == 1:
        graph_input = graph_input.view(1, -1)
    ma_norm = (float(ma_value) - ma_center) / ma_scale
    cloned.graphInput = torch.cat(
        [graph_input, torch.tensor([[ma_norm]], dtype=graph_input.dtype)],
        dim=1,
    )
    cloned.param_mA = torch.tensor([float(ma_value)], dtype=torch.float)
    cloned.param_mA_norm = torch.tensor([float(ma_norm)], dtype=torch.float)
    return cloned


def get_background_groups_full(config_data: Dict) -> Dict[str, List[str]]:
    bg_config = config_data["background_config"]
    dataset_config = config_data["dataset_config"]
    prefix = dataset_config["background_prefix"]
    return {
        group: [prefix + sample for sample in samples]
        for group, samples in bg_config["background_groups"].items()
    }


def build_eval_data(args, config_data: Dict, ma_value: int, append_ma: bool):
    dataset_root = Path(os.environ["WORKDIR"]) / "ParticleNetMD" / "dataset"
    loader = DynamicDatasetLoader(str(dataset_root))
    signal_prefix = config_data["dataset_config"]["signal_prefix"]
    signal_sample = f"{signal_prefix}MHc{args.mhc}_MA{ma_value}"
    background_groups = get_background_groups_full(config_data)
    max_events = args.max_events_per_class
    if args.pilot and max_events is None:
        max_events = 2000

    data = loader.load_multiclass_with_subsampling(
        signal_sample=signal_sample,
        background_groups=background_groups,
        channel=args.channel,
        fold_list=[args.fold],
        max_events_per_fold=max_events,
        balance_weights=True,
        random_state=100 + ma_value,
    )
    if append_ma:
        param_cfg = config_data.get("parametric_config", {})
        ma_center = float(param_cfg.get("ma_center", 90.0))
        ma_scale = float(param_cfg.get("ma_scale", 10.0))
        return [clone_with_ma(item, ma_value, ma_center, ma_scale) for item in data]
    return [item.clone() for item in data]


def evaluate(model, data_list, batch_size: int, device: torch.device):
    loader = DataLoader(GraphDataset(data_list), batch_size=batch_size, shuffle=False)
    labels, scores, weights, mass1, mass2 = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch.x, batch.edge_index, batch.graphInput, batch.batch)
            probs = F.softmax(logits, dim=1)
            labels.append(batch.y.cpu().numpy())
            scores.append(probs.cpu().numpy())
            weights.append(batch.weight.cpu().numpy())
            mass1.append(batch.mass1.cpu().numpy().reshape(-1))
            mass2.append(batch.mass2.cpu().numpy().reshape(-1))
    return {
        "y": np.concatenate(labels),
        "scores": np.concatenate(scores),
        "weight": np.concatenate(weights),
        "mass1": np.concatenate(mass1),
        "mass2": np.concatenate(mass2),
    }


def lr_score(scores: np.ndarray, bg_idx: int) -> np.ndarray:
    return scores[:, 0] / (scores[:, 0] + scores[:, bg_idx] + 1e-10)


def weighted_hist(ax, values, weights, bins, label, color, linestyle="-"):
    hist, edges = np.histogram(values, bins=bins, weights=np.abs(weights))
    total = hist.sum()
    if total > 0:
        hist = hist / total
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax.step(centers, hist, where="mid", label=label, color=color, linestyle=linestyle, linewidth=1.8)


def setup_mass_sculpting_style():
    if not HAS_CMS_STYLE:
        raise RuntimeError("cmsstyle is required for GA-style mass sculpting plots")
    CMS.setCMSStyle()
    CMS.SetExtraText("Simulation Preliminary")
    CMS.SetLumi(None, run="Run 2+3,")
    CMS.SetEnergy(0, unit="13/13.6 TeV")


def root_color(hex_color: str) -> int:
    return ROOT.TColor.GetColor(hex_color)


def make_root_hist(name: str, values: np.ndarray, weights: np.ndarray) -> ROOT.TH1D:
    hist = ROOT.TH1D(name, "", MASS_BINS, MASS_RANGE[0], MASS_RANGE[1])
    hist.SetDirectory(0)
    hist.Sumw2()
    finite = np.isfinite(values) & np.isfinite(weights)
    for value, weight in zip(values[finite], weights[finite]):
        hist.Fill(float(value), abs(float(weight)))
    return hist


def normalize_root_hist(hist: ROOT.TH1D) -> None:
    integral = hist.Integral(0, hist.GetNbinsX() + 1)
    if integral > 0:
        hist.Scale(1.0 / integral)


def make_ratio_hist(numerator: ROOT.TH1D, denominator: ROOT.TH1D, name: str) -> ROOT.TH1D:
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


def unit_lr_score(scores: np.ndarray) -> np.ndarray:
    denom = np.sum(scores, axis=1)
    return np.divide(scores[:, 0], denom, out=np.zeros(scores.shape[0], dtype=np.float64), where=denom > 0)


def compute_dcor(
    values: np.ndarray,
    mass: np.ndarray,
    weights: np.ndarray,
    max_events: int = DEFAULT_DCOR_MAX_EVENTS,
    seed: int = 12345,
) -> float:
    valid_indices = np.flatnonzero(mass > 0)
    if valid_indices.size < 2:
        return 0.0
    if max_events and max_events > 0 and valid_indices.size > max_events:
        rng = np.random.default_rng(seed)
        valid_indices = rng.choice(valid_indices, size=max_events, replace=False)
    with torch.no_grad():
        return float(distance_correlation(
            torch.tensor(values[valid_indices], dtype=torch.float32),
            torch.tensor(mass[valid_indices], dtype=torch.float32),
            torch.tensor(np.abs(weights[valid_indices]), dtype=torch.float32),
        ).item())


def make_roc_plots(ma_value: int, predictions: Dict[str, Dict], out_dir: Path, summary_rows: List[Dict]):
    roc = ROCCurveCalculator()
    for bg_idx in [1, 2, 3]:
        fig, ax = plt.subplots(figsize=(6, 5))
        for model_key, pred in predictions.items():
            mask = (pred["y"] == 0) | (pred["y"] == bg_idx)
            y_binary = (pred["y"][mask] == 0).astype(int)
            lr = lr_score(pred["scores"][mask], bg_idx)
            fpr, tpr, auc = roc.calculate_roc_curve(y_binary, lr, pred["weight"][mask])
            ax.plot(tpr, fpr, label=f"{MODEL_DISPLAY[model_key]} AUC={auc:.4f}", color=COLORS[model_key])
            summary_rows.append({
                "ma": ma_value,
                "model": model_key,
                "background": CLASS_NAMES[bg_idx],
                "auc": auc,
            })
        ax.plot([0, 1], [0, 1], color="0.6", linestyle="--", linewidth=1)
        ax.set_xlabel("Signal efficiency")
        ax.set_ylabel("Background efficiency")
        ax.set_title(f"MHc{args_global.mhc}_MA{ma_value} signal vs {CLASS_DISPLAY[bg_idx]}")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"roc_MA{ma_value}_{CLASS_NAMES[bg_idx]}.png", dpi=160)
        plt.close(fig)


def make_test_roc_overlay(ma_value: int, predictions: Dict[str, Dict], out_dir: Path):
    roc = ROCCurveCalculator()
    output_file = out_dir / "roc_curves_test_overlay.png"

    ROOT.gROOT.SetBatch(True)
    if HAS_CMS_STYLE:
        CMS.SetExtraText("Simulation Preliminary")
        CMS.SetLumi(None, run="Run 2+3, 138+62 fb^{#minus1}")
        CMS.SetEnergy(0, unit="13/13.6 TeV")
        canvas = CMS.cmsCanvas(
            "",
            0, 1,
            0, 1,
            "Signal Efficiency",
            "Background Efficiency",
            square=True,
            iPos=11,
            extraSpace=0.,
        )
        legend = CMS.cmsLeg(0.19, 0.47, 0.82, 0.75, textSize=0.025, columns=1)
    else:
        canvas = ROOT.TCanvas("roc_overlay", "roc_overlay", 596, 572)
        frame = canvas.DrawFrame(0, 0, 1, 1)
        frame.GetXaxis().SetTitle("Signal Efficiency")
        frame.GetYaxis().SetTitle("Background Efficiency")
        legend = ROOT.TLegend(0.19, 0.47, 0.82, 0.75)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetTextSize(0.025)

    canvas.SetGrid()
    graphs = []

    diag_graph = ROOT.TGraph(2)
    diag_graph.SetPoint(0, 0, 0)
    diag_graph.SetPoint(1, 1, 1)
    if HAS_CMS_STYLE:
        CMS.cmsObjectDraw(diag_graph, "L", LineColor=ROOT.kGray + 2, LineWidth=1, LineStyle=2)
    else:
        diag_graph.SetLineColor(ROOT.kGray + 2)
        diag_graph.SetLineWidth(1)
        diag_graph.SetLineStyle(2)
        diag_graph.Draw("L")
    graphs.append(diag_graph)

    for bg_idx in [1, 2, 3]:
        bg_color = PALETTE[bg_idx]
        bg_name = CLASS_DISPLAY[bg_idx]
        for model_key in ["baseline", "parametric"]:
            pred = predictions[model_key]
            mask = (pred["y"] == 0) | (pred["y"] == bg_idx)
            y_binary = (pred["y"][mask] == 0).astype(int)
            lr = lr_score(pred["scores"][mask], bg_idx)
            fpr, tpr, auc = roc.calculate_roc_curve(y_binary, lr, pred["weight"][mask])

            graph = ROOT.TGraph(len(fpr))
            for idx in range(len(fpr)):
                graph.SetPoint(idx, tpr[idx], fpr[idx])

            line_style = MODEL_LINE_STYLES[model_key]
            if HAS_CMS_STYLE:
                CMS.cmsObjectDraw(graph, "L", LineColor=bg_color, LineWidth=2, LineStyle=line_style)
            else:
                graph.SetLineColor(bg_color)
                graph.SetLineWidth(2)
                graph.SetLineStyle(line_style)
                graph.Draw("L")
            graphs.append(graph)
            legend_label = f"{MODEL_DISPLAY[model_key]} vs {bg_name}: AUC = {auc:.4f}"
            if HAS_CMS_STYLE:
                CMS.addToLegend(legend, (graph, legend_label, "L"))
            else:
                legend.AddEntry(graph, legend_label, "L")

    legend.Draw()
    canvas.RedrawAxis()
    canvas.Update()
    canvas.SaveAs(str(output_file))
    canvas.Close()


def make_score_plots(ma_value: int, predictions: Dict[str, Dict], out_dir: Path):
    bins = np.linspace(0.0, 1.0, 41)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    for class_idx, ax in enumerate(axes.flat):
        for model_key, pred in predictions.items():
            mask = pred["y"] == class_idx
            weighted_hist(
                ax,
                pred["scores"][mask, 0],
                pred["weight"][mask],
                bins,
                MODEL_DISPLAY[model_key],
                COLORS[model_key],
            )
        ax.set_title(CLASS_DISPLAY[class_idx])
        ax.set_ylabel("Normalized")
        ax.grid(True, alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("Signal score")
    axes[0, 0].legend(frameon=False)
    fig.suptitle(f"MHc{args_global.mhc}_MA{ma_value} signal-score distributions")
    fig.tight_layout()
    fig.savefig(out_dir / f"score_signal_MA{ma_value}.png", dpi=160)
    plt.close(fig)


def make_mass_sculpting(ma_value: int, predictions: Dict[str, Dict], out_dir: Path, summary_rows: List[Dict], dcor_max_events: int):
    bins = np.linspace(60.0, 120.0, 31)
    regions = [
        ("low", "LR < 0.3", lambda lr: lr < 0.3, ":"),
        ("mid", "0.3 <= LR <= 0.7", lambda lr: (lr >= 0.3) & (lr <= 0.7), "--"),
        ("high", "LR > 0.7", lambda lr: lr > 0.7, "-"),
    ]
    for mass_name in ["mass1", "mass2"]:
        for bg_idx in [1, 2, 3]:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
            for ax, model_key in zip(axes, ["baseline", "parametric"]):
                pred = predictions[model_key]
                mass = pred[mass_name]
                lr = lr_score(pred["scores"], bg_idx)
                base = (pred["y"] == bg_idx) & (mass > 0)
                if base.sum() == 0:
                    continue
                weighted_hist(ax, mass[base], pred["weight"][base], bins, "No cut", "black")
                for suffix, label, selector, linestyle in regions:
                    mask = base & selector(lr)
                    if mask.sum() == 0:
                        continue
                    weighted_hist(ax, mass[mask], pred["weight"][mask], bins, label, COLORS[model_key], linestyle)
                seed = ma_value * 1000 + bg_idx * 100 + (0 if mass_name == "mass1" else 10) + (0 if model_key == "baseline" else 1)
                dcor = compute_dcor(lr[base], mass[base], pred["weight"][base], dcor_max_events, seed)
                summary_rows.append({
                    "ma": ma_value,
                    "model": model_key,
                    "background": CLASS_NAMES[bg_idx],
                    "mass": mass_name,
                    "dcor_lr_mass": dcor,
                })
                ax.set_title(f"{MODEL_DISPLAY[model_key]} {CLASS_DISPLAY[bg_idx]} dCor={dcor:.4f}")
                ax.set_xlabel(f"{mass_name} [GeV]")
                ax.grid(True, alpha=0.25)
            axes[0].set_ylabel("Normalized")
            axes[0].legend(frameon=False, fontsize=8)
            fig.suptitle(f"MHc{args_global.mhc}_MA{ma_value} {mass_name} sculpting")
            fig.tight_layout()
            fig.savefig(out_dir / f"mass_sculpting_MA{ma_value}_{CLASS_NAMES[bg_idx]}_{mass_name}.png", dpi=160)
            plt.close(fig)


def make_test_mass_sculpting_overlays(ma_value: int, predictions: Dict[str, Dict], out_dir: Path, dcor_max_events: int):
    setup_mass_sculpting_style()
    region_defs = [
        ("nocut", "No cut", None, ROOT.kBlack),
        ("low", "LR < 0.3", lambda lr: lr < 0.3, root_color("#5790fc")),
        ("mid", "0.3 < LR < 0.7", lambda lr: (lr > 0.3) & (lr < 0.7), root_color("#f89c20")),
        ("high", "LR > 0.7", lambda lr: lr > 0.7, root_color("#e42536")),
    ]
    selections = [("background", "background", None)] + [
        (CLASS_NAMES[idx], CLASS_DISPLAY[idx], idx) for idx in range(len(CLASS_NAMES))
    ]

    def selection_mask(labels: np.ndarray, class_idx):
        if class_idx is None:
            return labels != 0
        return labels == class_idx

    for mass_name in ["mass1", "mass2"]:
        for selection_name, selection_display, class_idx in selections:
            hists = []
            refs: Dict[str, ROOT.TH1D] = {}
            dcors: Dict[str, float] = {}

            for model_key in ["baseline", "parametric"]:
                pred = predictions[model_key]
                mass = pred[mass_name]
                lr = unit_lr_score(pred["scores"])
                base = selection_mask(pred["y"], class_idx) & (mass > 0)
                if base.sum() == 0:
                    continue

                seed = ma_value * 1000 + (99 if class_idx is None else class_idx * 100) + (0 if mass_name == "mass1" else 10) + (0 if model_key == "baseline" else 1)
                dcors[model_key] = compute_dcor(lr[base], mass[base], pred["weight"][base], dcor_max_events, seed)

                for region_key, region_label, selector, color in region_defs:
                    mask = base if selector is None else base & selector(lr)
                    if mask.sum() == 0:
                        continue
                    hist = make_root_hist(
                        f"h_msculpt_MA{ma_value}_{selection_name}_{mass_name}_{model_key}_{region_key}",
                        mass[mask],
                        pred["weight"][mask],
                    )
                    normalize_root_hist(hist)
                    hist.SetLineColor(color)
                    hist.SetLineWidth(PLOT_LINE_WIDTH)
                    hist.SetLineStyle(MODEL_LINE_STYLES[model_key])
                    hist.SetMarkerSize(0)
                    hists.append((model_key, region_key, region_label, hist))
                    if selector is None:
                        refs[model_key] = hist

            if not hists or not refs:
                print(f"  Warning: no mass sculpting entries for MA{ma_value}, {selection_name}, {mass_name}", flush=True)
                continue

            ymax = max(hist.GetMaximum() for _model_key, _region_key, _region_label, hist in hists)
            canvas = CMS.cmsDiCanvas(
                "",
                MASS_RANGE[0],
                MASS_RANGE[1],
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
            keepalive = []

            canvas.cd(1)
            canvas.cd(1).SetGrid(0, 0)
            legend = CMS.cmsLeg(0.38, 0.56, 0.95, 0.88, textSize=0.023, columns=2)
            hist_by_key = {}
            for model_key, region_key, region_label, hist in hists:
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
                hist_by_key[(model_key, region_key)] = hist

            for region_key, region_label, _selector, _color in region_defs:
                for model_key in ["baseline", "parametric"]:
                    hist = hist_by_key.get((model_key, region_key))
                    if hist is not None:
                        CMS.addToLegend(legend, (hist, f"{MODEL_DISPLAY[model_key]} {region_label}", "L"))
            legend.Draw()

            CMS.drawText(f"MA{ma_value} {selection_display}", posX=0.20, posY=0.77, font=62, align=0, size=0.040)
            y_text = 0.70
            for model_key in ["baseline", "parametric"]:
                if model_key in dcors:
                    CMS.drawText(
                        f"{MODEL_DISPLAY[model_key]} dCor = {dcors[model_key]:.4f}",
                        posX=0.20,
                        posY=y_text,
                        font=42,
                        align=0,
                        size=0.027,
                    )
                    y_text -= 0.050
            canvas.cd(1).RedrawAxis()

            canvas.cd(2)
            canvas.cd(2).SetGrid()
            ref_line = ROOT.TLine(MASS_RANGE[0], 1.0, MASS_RANGE[1], 1.0)
            ref_line.SetLineStyle(ROOT.kDotted)
            ref_line.SetLineColor(ROOT.kBlack)
            ref_line.SetLineWidth(PLOT_LINE_WIDTH)
            ref_line.Draw()
            keepalive.append(ref_line)

            for model_key, region_key, _region_label, hist in hists:
                if region_key == "nocut" or model_key not in refs:
                    continue
                ratio = make_ratio_hist(hist, refs[model_key], f"{hist.GetName()}_ratio")
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
            output_path = out_dir / f"mass_sculpting_test_overlay_MA{ma_value}_{selection_name}_{mass_name}.png"
            canvas.SaveAs(str(output_path))
            canvas.Close()


def write_summary(out_dir: Path, auc_rows: List[Dict], dcor_rows: List[Dict]):
    summary = {"auc": auc_rows, "dcor": dcor_rows}
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)

    with open(out_dir / "auc_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ma", "model", "background", "auc"])
        writer.writeheader()
        writer.writerows(auc_rows)

    with open(out_dir / "dcor_summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["ma", "model", "background", "mass", "dcor_lr_mass"])
        writer.writeheader()
        writer.writerows(dcor_rows)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Compare ParametricPN to plain ParticleNetMD")
    parser.add_argument("--mhc", type=int, default=130)
    parser.add_argument("--ma-values", type=parse_ma_values, default=[85, 90, 95])
    parser.add_argument("--channel", default="Combined", choices=["Run1E2Mu", "Run3Mu", "Combined"])
    parser.add_argument("--fold", type=int, default=4)
    parser.add_argument("--config", default="configs/ParametricPNConfig.json")
    parser.add_argument("--parametric-dir", default=None)
    parser.add_argument("--baseline-dir", default="GAOptim")
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-events-per-class", type=int, default=None)
    parser.add_argument("--dcor-max-events", type=int, default=DEFAULT_DCOR_MAX_EVENTS,
                        help="Maximum events used for each exact dCor calculation; plots still use the full selected sample.")
    parser.add_argument("--roc-overlay-only", action="store_true",
                        help="Only draw the test-only ParticleNetMD-vs-ParametricPN ROC overlay canvases.")
    parser.add_argument("--mass-sculpting-overlay-only", action="store_true",
                        help="Only draw the test-only ParticleNetMD-vs-ParametricPN mass-sculpting overlay canvases.")
    parser.add_argument("--pilot", action="store_true")
    return parser.parse_args()


def main():
    global args_global
    args = parse_arguments()
    args_global = args
    make_full_plot_set = not args.roc_overlay_only and not args.mass_sculpting_overlay_only
    make_roc_overlay = args.roc_overlay_only or make_full_plot_set
    make_mass_sculpting_overlay = args.mass_sculpting_overlay_only or make_full_plot_set

    workdir = Path(os.environ["WORKDIR"]) / "ParticleNetMD"
    config_data = load_sgl_config(args.config).config
    fold_dir = "pilot" if args.pilot else f"fold-{args.fold}"
    signal_label = f"MHc{args.mhc}_" + "_".join(f"MA{ma}" for ma in args.ma_values)

    parametric_dir = Path(args.parametric_dir) if args.parametric_dir else (
        workdir / "ParametricPN" / args.channel / signal_label / fold_dir
    )
    out_dir = Path(args.output) if args.output else parametric_dir / "comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() or not args.device.startswith("cuda") else "cpu")
    param_model_path, param_info_path = find_parametric_artifacts(parametric_dir)
    param_model = load_model(param_model_path, param_info_path, device)
    print(f"Loaded ParametricPN model: {param_model_path}", flush=True)
    print(f"Writing comparison outputs to: {out_dir}", flush=True)

    auc_rows = []
    dcor_rows = []
    for ma_value in args.ma_values:
        print(f"Processing MA{ma_value}", flush=True)
        baseline_dir = workdir / args.baseline_dir / args.channel / f"MHc{args.mhc}_MA{ma_value}" / f"fold-{args.fold}" / "best_model"
        baseline_model = load_model(baseline_dir / "model.pt", baseline_dir / "model_info.json", device)

        print(f"  Loading evaluation datasets for MA{ma_value}", flush=True)
        baseline_data = build_eval_data(args, config_data, ma_value, append_ma=False)
        param_data = build_eval_data(args, config_data, ma_value, append_ma=True)
        print(f"  Evaluating baseline and parametric models for MA{ma_value}", flush=True)
        predictions = {
            "baseline": evaluate(baseline_model, baseline_data, args.batch_size, device),
            "parametric": evaluate(param_model, param_data, args.batch_size, device),
        }

        ma_dir = out_dir / f"MA{ma_value}"
        ma_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Making plots for MA{ma_value}", flush=True)
        if make_roc_overlay:
            make_test_roc_overlay(ma_value, predictions, ma_dir)
        if make_mass_sculpting_overlay:
            make_test_mass_sculpting_overlays(ma_value, predictions, ma_dir, args.dcor_max_events)
        if make_full_plot_set:
            make_roc_plots(ma_value, predictions, ma_dir, auc_rows)
            make_score_plots(ma_value, predictions, ma_dir)
            make_mass_sculpting(ma_value, predictions, ma_dir, dcor_rows, args.dcor_max_events)
        print(f"  Finished MA{ma_value}", flush=True)

        del baseline_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if make_full_plot_set:
        write_summary(out_dir, auc_rows, dcor_rows)
    print(f"Comparison written to {out_dir}")


if __name__ == "__main__":
    args_global = None
    main()
