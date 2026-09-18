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
# Typeset names for the figures; the plain names above stay the keys everywhere
# else (summary dicts, stdout).
CLASS_LATEX_NAMES = ["Signal", "Nonprompt", "Diboson", "t#bar{t}X"]
BACKGROUND_CLASSES = [1, 2, 3]
SIGNAL_PATTERN = re.compile(r"^MHc(?P<hc>\d+)_MA(?P<a>\d+)$")

CURVE_LINE_WIDTH = 3
# The no-discrimination diagonal is a reference, not a result, so it stays the
# faintest thing on the canvas.
DIAGONAL_COLOR = ROOT.kGray

# Every figure carries its own key, so a mass point can be read on its own: the
# class colors with their test AUCs, then the line styles and the diagonal. The
# shared panel below is still published for layouts that drop the in-plot keys
# (slides, a multi-panel paper arrangement), the same scheme as the TriLepton
# paper figures.
LEGEND_PANEL_NAME = "legend.pdf"
LEGEND_PANEL_TEXT_SIZE = 0.055
LEGEND_PANEL_ROW_SPACING = 1.55  # row pitch in units of the text size
LEGEND_PANEL_LEFT = 0.04
LEGEND_PANEL_COLUMN_GAP = 0.01
LEGEND_PANEL_CLASS_WIDTH = 0.42
LEGEND_PANEL_CLASS_MARGIN = 0.24
LEGEND_PANEL_STYLE_WIDTH = 0.54
LEGEND_PANEL_STYLE_MARGIN = 0.20

# In-plot key. It sits in the upper-left triangle, which the curves never reach.
# The no-discrimination diagonal does cross that triangle, so the block is kept
# narrow enough that the diagonal stays to the right of the longest row -- the
# diagonal climbs faster than the rows descend, so the top of the block is the
# binding constraint on PLOT_LEGEND_WIDTH.
PLOT_LEGEND_LEFT = 0.19
PLOT_LEGEND_TOP = 0.715
PLOT_LEGEND_WIDTH = 0.44
PLOT_LEGEND_TEXT_SIZE = 0.034
PLOT_LEGEND_ROW_SPACING = 1.45  # row pitch in units of the text size
PLOT_LEGEND_MARGIN = 0.16

MASS_LABEL_POS = (0.20, 0.762)
MASS_LABEL_TEXT_SIZE = 0.040


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
    CMS.SetExtraText("Simulation")
    CMS.SetLumi(None, run="")
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


def paper_panel_size() -> Tuple[int, int]:
    """Pixel size of one ROC plot, so the legend panel matches it exactly.

    Read off a throwaway canvas built exactly like the real ones rather than
    hardcoded, which would drift if cmsstyle changed its reference dimensions.
    """
    probe = make_canvas()
    size = (probe.GetWindowWidth(), probe.GetWindowHeight())
    probe.Close()
    return size


def class_label(bg_class: int, auc_summary: Dict[str, float] = None) -> str:
    """Legend text for one background class, with its test AUC when known."""
    name = CLASS_LATEX_NAMES[bg_class]
    if auc_summary is None:
        return name
    return f"{name} (AUC = {auc_summary[CLASS_DISPLAY_NAMES[bg_class]]:.2f})"


def build_legend_proxies(prefix: str, auc_summary: Dict[str, float] = None
                         ) -> Tuple[List[Tuple], List[Tuple], List[object]]:
    """Dummy graphs styled like the drawn curves, plus their legend entries.

    Returns (class_entries, style_entries, proxies); the caller must keep
    `proxies` alive until the canvas is written, since TLegend does not own
    them. The prefix keeps the ROOT names unique across panels. `auc_summary`
    (test AUC per plain class name) is the only per-mass-point part of the key,
    so the shared panel builds its entries without it.
    """
    proxies: List[object] = []

    def new_proxy(color: int, line_style: int) -> "ROOT.TGraph":
        graph = ROOT.TGraph(2)
        graph.SetPoint(0, 0.0, 0.0)
        graph.SetPoint(1, 1.0, 1.0)
        graph.SetName(f"{prefix}_{len(proxies)}")
        graph.SetLineColor(color)
        graph.SetLineWidth(CURVE_LINE_WIDTH)
        graph.SetLineStyle(line_style)
        graph.SetMarkerSize(0)
        proxies.append(graph)
        return graph

    class_entries = [
        (new_proxy(PALETTE[bg_class], ROOT.kSolid), class_label(bg_class, auc_summary), "L")
        for bg_class in BACKGROUND_CLASSES
    ]

    # Line style carries train vs test for every class, so the key is drawn once
    # in black rather than repeated per color.
    style_entries = [
        (new_proxy(ROOT.kBlack, ROOT.kSolid), "Test", "L"),
        (new_proxy(ROOT.kBlack, ROOT.kDashed), "Train", "L"),
        (new_proxy(DIAGONAL_COLOR, ROOT.kDashed), "No discrimination", "L"),
    ]

    return class_entries, style_entries, proxies


def render_legend_panel(output_dir: Path) -> Path:
    """Write the shared key as its own panel, sized like a ROC plot."""
    width, height = paper_panel_size()
    canvas = ROOT.TCanvas("paper_roc_legend", "paper_roc_legend", 50, 50, width, height)
    canvas.SetFillColor(0)
    canvas.SetBorderMode(0)
    canvas.SetFrameFillStyle(0)
    canvas.SetFrameBorderMode(0)
    canvas.cd()

    class_entries, style_entries, proxies = build_legend_proxies("paper_roc_legend")
    row = LEGEND_PANEL_TEXT_SIZE * LEGEND_PANEL_ROW_SPACING

    # Both blocks are centred on the panel midline so unequal column lengths
    # stay visually balanced.
    def block_y(n_entries: int) -> Tuple[float, float]:
        block_height = row * n_entries
        return 0.5 - 0.5 * block_height, 0.5 + 0.5 * block_height

    class_x1 = LEGEND_PANEL_LEFT
    style_x1 = class_x1 + LEGEND_PANEL_CLASS_WIDTH + LEGEND_PANEL_COLUMN_GAP

    legends = []
    for x1, width_ndc, margin, entries in (
        (class_x1, LEGEND_PANEL_CLASS_WIDTH, LEGEND_PANEL_CLASS_MARGIN, class_entries),
        (style_x1, LEGEND_PANEL_STYLE_WIDTH, LEGEND_PANEL_STYLE_MARGIN, style_entries),
    ):
        y1, y2 = block_y(len(entries))
        legend = draw_legend(x1, y1, x1 + width_ndc, y2, LEGEND_PANEL_TEXT_SIZE)
        legend.SetMargin(margin)
        for entry in entries:
            legend.AddEntry(*entry)
        legend.Draw()
        legends.append(legend)

    canvas.Update()
    output_path = output_dir / LEGEND_PANEL_NAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.SaveAs(str(output_path))
    canvas.Close()
    del proxies, legends
    return output_path


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


def draw_mass_label(signal: str, keepalive: List[object]) -> None:
    """The mass point, drawn as the caption above the in-plot key."""
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(MASS_LABEL_TEXT_SIZE)
    latex.SetTextColor(ROOT.kBlack)
    latex.SetTextAlign(12)
    latex.DrawLatex(*MASS_LABEL_POS, mass_point_label(signal))
    keepalive.append(latex)


def draw_plot_legend(signal: str, auc_summary: Dict[str, float],
                     keepalive: List[object]) -> None:
    """The full key, in the plot: colors with their AUCs, then the line styles.

    Drawing the sample lines rather than coloring the text means the reader
    never has to match a hue to a curve by eye, and the block says everything
    the standalone panel says, so a single figure stands alone.
    """
    class_entries, style_entries, proxies = build_legend_proxies(
        f"paper_roc_{signal}", auc_summary
    )
    entries = class_entries + style_entries

    row = PLOT_LEGEND_TEXT_SIZE * PLOT_LEGEND_ROW_SPACING
    legend = draw_legend(
        PLOT_LEGEND_LEFT,
        PLOT_LEGEND_TOP - row * len(entries),
        PLOT_LEGEND_LEFT + PLOT_LEGEND_WIDTH,
        PLOT_LEGEND_TOP,
        PLOT_LEGEND_TEXT_SIZE,
    )
    legend.SetMargin(PLOT_LEGEND_MARGIN)
    for entry in entries:
        legend.AddEntry(*entry)
    legend.Draw()

    draw_mass_label(signal, keepalive)
    keepalive.extend(proxies)
    keepalive.append(legend)


def plot_signal(signal: str, predictions: Dict[str, np.ndarray], output_path: Path) -> Dict[str, Dict[str, float]]:
    configure_cms_style()

    calculator = ROCCurveCalculator()
    canvas = make_canvas()
    canvas.SetGrid()

    keepalive: List[object] = []
    summary: Dict[str, Dict[str, float]] = {}

    diagonal = ROOT.TGraph(2)
    diagonal.SetPoint(0, 0.0, 0.0)
    diagonal.SetPoint(1, 1.0, 1.0)
    draw_with_cms(
        diagonal,
        "L",
        LineColor=DIAGONAL_COLOR,
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
            LineWidth=CURVE_LINE_WIDTH,
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
            LineWidth=CURVE_LINE_WIDTH,
            LineStyle=ROOT.kSolid,
        )
        keepalive.append(test_graph)

        summary[bg_name] = {"train": float(auc_train), "test": float(auc_test)}

    draw_plot_legend(signal, {name: aucs["test"] for name, aucs in summary.items()}, keepalive)
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

    configure_cms_style()
    print(f"legend panel -> {render_legend_panel(output_dir)}")

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
