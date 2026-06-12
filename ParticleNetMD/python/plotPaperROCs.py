#!/usr/bin/env python3
"""
Create paper-style ParticleNet ROC plots from cached GA prediction files.

The script does not run model inference. It reads the best model index from
ga_loss_summary.json, loads the corresponding model*_predictions.npz cache,
and writes one PDF per mass point.
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)

try:
    import cmsstyle as CMS
    CMS.setCMSStyle()
    HAS_CMS_STYLE = True
except ImportError:
    CMS = None
    HAS_CMS_STYLE = False


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

from ROCCurveCalculator import PALETTE, ROCCurveCalculator  # noqa: E402


DEFAULT_SIGNALS = ["MHc100_MA95", "MHc130_MA90", "MHc160_MA85"]
CLASS_DISPLAY_NAMES = ["Signal", "Nonprompt", "Diboson", "ttX"]
BACKGROUND_CLASSES = [1, 2, 3]
SIGNAL_PATTERN = re.compile(r"^MHc(?P<hc>\d+)_MA(?P<a>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make paper ROC PDFs from cached ParticleNet predictions."
    )
    parser.add_argument("--input", default="GAOptim",
                        help="GA output directory (default: GAOptim)")
    parser.add_argument("--channel", default="Combined",
                        help="Training channel to read (default: Combined)")
    parser.add_argument("--fold", type=int, default=4,
                        help="Fold directory number to read (default: 4)")
    parser.add_argument("--output-dir", default="plots/paper/ROC",
                        help="Output directory for PDFs (default: plots/paper/ROC)")
    parser.add_argument("--signals", nargs="+", default=DEFAULT_SIGNALS,
                        help="Mass points to plot")
    return parser.parse_args()


def load_best_model(input_dir: Path, channel: str, signal: str, fold: int) -> Tuple[int, int, Path]:
    summary_path = input_dir / channel / signal / f"fold-{fold}" / "ga_loss_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing GA loss summary: {summary_path}")

    with summary_path.open() as handle:
        summary = json.load(handle)

    iterations = summary.get("iterations", [])
    if not iterations:
        raise ValueError(f"No iterations found in {summary_path}")

    last_iteration = max(iterations, key=lambda item: int(item["iteration"]))
    iteration = int(last_iteration["iteration"])
    model_idx = int(last_iteration["best_model_idx"])
    return iteration, model_idx, summary_path


def prediction_path(input_dir: Path, channel: str, signal: str, fold: int,
                    iteration: int, model_idx: int) -> Path:
    return (
        input_dir / channel / signal / f"fold-{fold}" / f"GA-iter{iteration}"
        / "overfitting_diagnostics" / f"model{model_idx}_predictions.npz"
    )


def configure_cms_style() -> None:
    if not HAS_CMS_STYLE:
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetPadLeftMargin(0.12)
        ROOT.gStyle.SetPadBottomMargin(0.12)
        return

    CMS.setCMSStyle()
    CMS.SetExtraText("Simulation Preliminary")
    CMS.SetLumi(None, run="Run 2+3,")
    CMS.SetEnergy(0, unit="13/13.6 TeV")


def make_graph(tpr: np.ndarray, fpr: np.ndarray) -> "ROOT.TGraph":
    graph = ROOT.TGraph(len(fpr))
    for idx, (signal_eff, background_eff) in enumerate(zip(tpr, fpr)):
        graph.SetPoint(idx, float(signal_eff), float(background_eff))
    return graph


def likelihood_ratio(scores: np.ndarray, bg_class: int) -> np.ndarray:
    signal_scores = scores[:, 0]
    background_scores = scores[:, bg_class]
    return signal_scores / (signal_scores + background_scores + 1e-10)


def calculate_curves(calculator: ROCCurveCalculator, predictions: Dict[str, np.ndarray],
                     bg_class: int, split: str) -> Tuple[np.ndarray, np.ndarray, float]:
    y_true = predictions[f"y_true_{split}"]
    scores = predictions[f"y_scores_{split}"]
    weights = predictions[f"weights_{split}"]

    mask = (y_true == 0) | (y_true == bg_class)
    y_binary = (y_true[mask] == 0).astype(int)
    lr_scores = likelihood_ratio(scores[mask], bg_class)
    return calculator.calculate_roc_curve(y_binary, lr_scores, weights[mask])


def draw_with_cms(obj, draw_option: str, **style_kwargs) -> None:
    if HAS_CMS_STYLE:
        CMS.cmsObjectDraw(obj, draw_option, **style_kwargs)
        return

    if "LineColor" in style_kwargs:
        obj.SetLineColor(style_kwargs["LineColor"])
    if "LineWidth" in style_kwargs:
        obj.SetLineWidth(style_kwargs["LineWidth"])
    if "LineStyle" in style_kwargs:
        obj.SetLineStyle(style_kwargs["LineStyle"])
    obj.Draw(draw_option)


def draw_legend(x1: float, y1: float, x2: float, y2: float, text_size: float):
    if HAS_CMS_STYLE:
        return CMS.cmsLeg(x1, y1, x2, y2, textSize=text_size, columns=1)

    legend = ROOT.TLegend(x1, y1, x2, y2)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(text_size)
    return legend


def mass_point_label(signal: str) -> str:
    match = SIGNAL_PATTERN.match(signal)
    if not match:
        raise ValueError(f"Cannot parse mass point from signal name: {signal}")

    mhc = match.group("hc")
    ma = match.group("a")
    return f"(m_{{H^{{+}}}}, m_{{A}}) = ({mhc}, {ma}) GeV"


def make_canvas() -> "ROOT.TCanvas":
    if HAS_CMS_STYLE:
        return CMS.cmsCanvas(
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

    canvas = ROOT.TCanvas("", "", 800, 800)
    frame = canvas.DrawFrame(0.0, 0.0, 1.0, 1.0)
    frame.GetXaxis().SetTitle("Signal Efficiency")
    frame.GetYaxis().SetTitle("Background Efficiency")
    return canvas


def draw_mass_point(signal: str, keepalive: List[object]) -> None:
    label = mass_point_label(signal)
    if HAS_CMS_STYLE:
        CMS.drawText(label, posX=0.20, posY=0.55, font=42, align=0, size=0.026)
        return

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.026)
    latex.DrawLatex(0.20, 0.55, label)
    keepalive.append(latex)


def plot_signal(signal: str, predictions: Dict[str, np.ndarray], output_path: Path) -> Dict[str, Dict[str, float]]:
    configure_cms_style()

    calculator = ROCCurveCalculator()
    canvas = make_canvas()
    canvas.SetGrid()

    legend = draw_legend(0.18, 0.62, 0.70, 0.77, 0.022)
    keepalive: List[object] = []
    summary: Dict[str, Dict[str, float]] = {}

    diagonal = ROOT.TGraph(2)
    diagonal.SetPoint(0, 0.0, 0.0)
    diagonal.SetPoint(1, 1.0, 1.0)
    draw_with_cms(
        diagonal,
        "L",
        LineColor=ROOT.kGray + 2,
        LineWidth=2,
        LineStyle=ROOT.kDashed,
    )
    keepalive.append(diagonal)

    for bg_class in BACKGROUND_CLASSES:
        bg_name = CLASS_DISPLAY_NAMES[bg_class]
        color = PALETTE[bg_class]

        fpr_train, tpr_train, auc_train = calculate_curves(
            calculator, predictions, bg_class, "train"
        )
        train_graph = make_graph(tpr_train, fpr_train)
        draw_with_cms(
            train_graph,
            "L",
            LineColor=color,
            LineWidth=3,
            LineStyle=ROOT.kDashed,
        )
        keepalive.append(train_graph)

        fpr_test, tpr_test, auc_test = calculate_curves(
            calculator, predictions, bg_class, "test"
        )
        test_graph = make_graph(tpr_test, fpr_test)
        draw_with_cms(
            test_graph,
            "L",
            LineColor=color,
            LineWidth=3,
            LineStyle=ROOT.kSolid,
        )
        keepalive.append(test_graph)

        legend.AddEntry(test_graph, f"{bg_name}: AUC = {auc_test:.4f}", "L")
        summary[bg_name] = {"train": float(auc_train), "test": float(auc_test)}

    legend.Draw()
    draw_mass_point(signal, keepalive)
    canvas.RedrawAxis()
    canvas.Update()
    canvas._keepalive = keepalive

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(str(output_path))
    canvas.Close()
    return summary


def load_predictions(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing cached predictions: {path}")

    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def format_summary(signal: str, iteration: int, model_idx: int,
                   output_path: Path, auc_summary: Dict[str, Dict[str, float]]) -> str:
    lines = [f"{signal}: GA-iter{iteration} model{model_idx} -> {output_path}"]
    for bg_name in CLASS_DISPLAY_NAMES[1:]:
        aucs = auc_summary[bg_name]
        lines.append(
            f"  Signal vs {bg_name}: test AUC {aucs['test']:.4f} "
            f"(train {aucs['train']:.4f})"
        )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input)
    output_dir = Path(args.output_dir)

    for signal in args.signals:
        iteration, model_idx, _summary_path = load_best_model(
            input_dir, args.channel, signal, args.fold
        )
        pred_path = prediction_path(
            input_dir, args.channel, signal, args.fold, iteration, model_idx
        )
        predictions = load_predictions(pred_path)
        output_path = output_dir / f"{signal}.pdf"
        auc_summary = plot_signal(signal, predictions, output_path)
        print(format_summary(signal, iteration, model_idx, output_path, auc_summary))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
