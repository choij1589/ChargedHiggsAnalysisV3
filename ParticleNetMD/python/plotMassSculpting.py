#!/usr/bin/env python3
"""
Create labelled ParticleNet mass-sculpting PDFs from cached GA prediction files.

Presentation counterpart of the per-class mass-sculpting diagnostics that
visualizeGAIteration.py writes (model*_mass_sculpting_<class>_<mass>.png, copied
to best_model/). Same selections and dCor values, with adaptive binning and a
simulation CMS label whose status is chosen on the command line
(--private-work, --preliminary, --work-in-progress);
written as PDFs into best_model/, the label mode in the file name.

The script does not run model inference. It reads the best model index from
ga_loss_summary.json, loads the corresponding model*_predictions.npz cache,
and writes one PDF per (signal, class, mass).
"""

import argparse
import os
import sys
from array import array
from pathlib import Path
from typing import Dict, List, NamedTuple, Tuple

import numpy as np
import ROOT

ROOT.gROOT.SetBatch(True)

import cmsstyle as CMS

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "lib"))

from plotPaperROCs import (  # noqa: E402
    CLASS_LATEX_NAMES,
    DEFAULT_SIGNALS,
    load_best_model,
    load_predictions,
    mass_point_label,
    prediction_path,
)
from visualizeGAIteration import (  # noqa: E402
    CLASS_NAMES,
    PLOT_LINE_WIDTH,
    TRAIN_LINE_STYLE,
    _compute_disco_np,
    _make_ratio_hist,
    _palette_root_color,
    _unit_lr,
)


MASS_NAMES = ["mass1", "mass2"]
MASS_RANGE = (60.0, 120.0)
SPLITS = ["train", "test"]

# Adaptive binning, same scheme as SignalRegionStudyV4/python/plotPaperLRModified.py:
# start from a uniform grid and close a bin once it is populated enough or has
# reached the maximum width; a sparse last bin is merged into its neighbour.
# The LR plot counts background yield in one histogram; here every drawn curve
# (each LR region, train and test) is a normalized shape, so a bin closes only
# once the sparsest of them reaches ADAPTIVE_MIN_NEFF effective entries
# (relative stat. error <= 1/sqrt(N_eff)). Edges are shared by all curves of a
# figure so the ratios stay bin-by-bin.
class LabelMode(NamedTuple):
    cms_text: str
    cms_font: int
    cms_size: float
    extra_text: str
    file_suffix: str


# CMS label per --<mode> flag; every mode marks the plot as simulation. Private
# work replaces the CMS logo with its own text at the logo position and carries
# no extra text; the other modes keep the logo at cmsstyle's defaults (font 61,
# size 0.75).
LABEL_MODES = {
    "preliminary": LabelMode("CMS", 61, 0.75, "Simulation Preliminary", "_preliminary"),
    "work-in-progress": LabelMode("CMS", 61, 0.75, "Simulation Work in progress", "_wip"),
    "private-work": LabelMode("Private work (CMS simulation)", 52, 0.75 * 0.76, "", "_pw"),
}

BASE_WIDTH = 2.0  # GeV
ADAPTIVE_MIN_NEFF = 10.0
ADAPTIVE_MAX_WIDTH = 12.0  # GeV

# Top header: three blocks on one row grid, their first rows aligned at
# HEADER_TOP. Left: class, dCor (train, test), mass point. Middle: split key
# (line style). Right: selection key (colour). Colour and line style are keyed
# separately, as in the ROC figures.
HEADER_TOP = 0.88
HEADER_ROW_PITCH = 0.065
TEXT_X = 0.20
STYLE_LEGEND_X = (0.53, 0.68)
REGION_LEGEND_X = (0.70, 0.95)
LEGEND_TEXT_SIZE = 0.036
REGION_LEGEND_MARGIN = 0.25
STYLE_LEGEND_MARGIN = 0.45

CLASS_LABEL_SIZE = 0.055
INFO_TEXT_SIZE = 0.046


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Make labelled mass-sculpting PDFs from cached ParticleNet predictions."
    )
    label_group = parser.add_mutually_exclusive_group(required=True)
    for mode, label in LABEL_MODES.items():
        label_group.add_argument(
            f"--{mode}", dest="label_mode", action="store_const", const=mode,
            help=f"label '{f'{label.cms_text} {label.extra_text}'.strip()}', "
                 f"files sculpting_<class>_<mass>{label.file_suffix}.pdf",
        )
    parser.add_argument("--input", default="GAOptim",
                        help="GA output directory (default: GAOptim)")
    parser.add_argument("--channel", default="Combined",
                        help="Training channel to read (default: Combined)")
    parser.add_argument("--fold", type=int, default=4,
                        help="Fold directory number to read (default: 4)")
    parser.add_argument("--signals", nargs="+", default=DEFAULT_SIGNALS,
                        help="Mass points to plot")
    return parser.parse_args()


def configure_cms_style(label: LabelMode) -> None:
    CMS.setCMSStyle()
    # SetExtraText blanks the CMS text whenever the extra text says "Private",
    # so the CMS text is set after it.
    CMS.SetExtraText(label.extra_text)
    CMS.SetCmsText(label.cms_text, font=label.cms_font, size=label.cms_size)
    CMS.SetLumi(None, run="")
    CMS.SetEnergy(0, unit="13/13.6 TeV")
    ROOT.gStyle.SetLineStyleString(TRAIN_LINE_STYLE, "24 12")


def output_name(class_label: str, mass_name: str, label: LabelMode) -> str:
    return f"sculpting_{class_label}_{mass_name}{label.file_suffix}.pdf"


def base_edges() -> np.ndarray:
    xmin, xmax = MASS_RANGE
    n_bins = int(round((xmax - xmin) / BASE_WIDTH))
    if abs(n_bins * BASE_WIDTH - (xmax - xmin)) > 1e-9:
        raise ValueError(f"base width {BASE_WIDTH} does not divide {MASS_RANGE} evenly")
    return np.round(np.linspace(xmin, xmax, n_bins + 1), 6)


def build_adaptive_edges(selections: List[Tuple[np.ndarray, np.ndarray]]) -> List[float]:
    """Merge the base grid until every selection's bin has ADAPTIVE_MIN_NEFF entries."""
    grid = base_edges()
    sumw = []
    sumw2 = []
    for values, weights in selections:
        sumw.append(np.histogram(values, bins=grid, weights=weights)[0])
        sumw2.append(np.histogram(values, bins=grid, weights=weights ** 2)[0])
    sumw = np.array(sumw)
    sumw2 = np.array(sumw2)

    def populated(low: int, high: int) -> bool:
        w = sumw[:, low:high].sum(axis=1)
        w2 = sumw2[:, low:high].sum(axis=1)
        neff = np.divide(w ** 2, w2, out=np.zeros_like(w), where=w2 > 0)
        return bool(np.all(neff >= ADAPTIVE_MIN_NEFF))

    edges_idx = [0]
    for ibin in range(1, len(grid)):
        width = grid[ibin] - grid[edges_idx[-1]]
        is_last = ibin == len(grid) - 1
        if populated(edges_idx[-1], ibin) or width >= ADAPTIVE_MAX_WIDTH or is_last:
            edges_idx.append(ibin)

    if len(edges_idx) >= 3:
        merged_width = grid[edges_idx[-1]] - grid[edges_idx[-3]]
        if not populated(edges_idx[-2], edges_idx[-1]) and merged_width <= ADAPTIVE_MAX_WIDTH:
            edges_idx.pop(-2)

    return [float(grid[i]) for i in edges_idx]


def make_density_hist(name: str, values: np.ndarray, weights: np.ndarray,
                      edges: List[float]) -> "ROOT.TH1D":
    """Unit-area shape, as the fraction per BASE_WIDTH.

    Dividing by the bin width keeps merged bins from standing out; scaling back
    to BASE_WIDTH keeps the unmerged bins at their plain normalized fraction.
    """
    hist = ROOT.TH1D(name, "", len(edges) - 1, array("d", edges))
    hist.SetDirectory(0)
    hist.Sumw2()
    for value, weight in zip(values, weights):
        hist.Fill(float(value), float(weight))
    integral = hist.Integral(0, hist.GetNbinsX() + 1)
    if integral > 0:
        hist.Scale(BASE_WIDTH / integral, "width")
    return hist


def plot_one_class(predictions: Dict[str, np.ndarray], signal: str, mass_name: str,
                   class_label: str, class_idx: int, output_path: Path,
                   label: LabelMode) -> None:
    """Mass shape of one class in LR regions, train and test, with ratio to no cut."""
    configure_cms_style(label)

    region_defs = [
        ("nocut", "No cut", lambda lr: np.ones(lr.shape, dtype=bool), ROOT.kBlack),
        ("low", "LR < 0.3", lambda lr: lr < 0.3, _palette_root_color(0)),
        ("mid", "0.3 < LR < 0.7", lambda lr: (lr > 0.3) & (lr < 0.7), _palette_root_color(1)),
        ("high", "LR > 0.7", lambda lr: lr > 0.7, _palette_root_color(2)),
    ]

    # (split, region suffix) -> (mass values, |weights>) of the drawn selection
    selections: Dict[Tuple[str, str], Tuple[np.ndarray, np.ndarray]] = {}
    dcors: Dict[str, float] = {}
    for split in SPLITS:
        y_true = predictions[f"y_true_{split}"]
        weights = predictions[f"weights_{split}"]
        mass = predictions[f"{mass_name}_{split}"]
        lr = _unit_lr(predictions[f"y_scores_{split}"])
        base = (y_true == class_idx) & (mass > 0) & np.isfinite(mass) & np.isfinite(weights)
        if base.sum() == 0:
            continue

        dcors[split] = _compute_disco_np(lr[base], mass[base], weights[base])
        for suffix, _label, selector, _color in region_defs:
            mask = base & selector(lr)
            if mask.sum() > 0:
                selections[(split, suffix)] = (mass[mask], np.abs(weights[mask]))

    if not dcors:
        raise ValueError(f"No {class_label} events with {mass_name} > 0 for {output_path}")

    edges = build_adaptive_edges(list(selections.values()))
    tag = f"{signal}_{class_label}_{mass_name}"
    hists = []
    refs: Dict[str, ROOT.TH1D] = {}
    for split in SPLITS:
        for suffix, label, _selector, color in region_defs:
            if (split, suffix) not in selections:
                continue
            values, weights = selections[(split, suffix)]
            hist = make_density_hist(f"h_{tag}_{split}_{suffix}", values, weights, edges)
            hist.SetLineColor(color)
            hist.SetLineWidth(PLOT_LINE_WIDTH)
            hist.SetLineStyle(ROOT.kSolid if split == "test" else TRAIN_LINE_STYLE)
            hist.SetMarkerSize(0)
            if suffix == "nocut":
                refs[split] = hist
            hists.append((split, suffix, hist))

    xmin, xmax = MASS_RANGE
    ymax = max(hist.GetMaximum() for _split, _suffix, hist in hists)
    canvas = CMS.cmsDiCanvas(
        "",
        xmin,
        xmax,
        0.0,
        max(1e-4, ymax * 2.0),
        0.0,
        2.0,
        f"{mass_name} [GeV]",
        "Normalized",
        "Region / No cut",
        square=True,
        iPos=0,
        extraSpace=0.0,
    )
    keepalive: List[object] = []

    ratio_frame = canvas.cd(2).GetPrimitive("hframe")
    if ratio_frame:
        ratio_frame.GetYaxis().CenterTitle()
        ratio_frame.GetYaxis().SetTitleSize(0.115)
        ratio_frame.GetYaxis().SetTitleOffset(0.58)

    canvas.cd(1)
    canvas.cd(1).SetGrid(0, 0)
    for _split, _suffix, hist in hists:
        draw_hist(hist)
        keepalive.append(hist)

    keepalive.extend(draw_legends(region_defs, tag))

    info_rows = [(CLASS_LATEX_NAMES[class_idx], 62, CLASS_LABEL_SIZE)]
    info_rows += [(f"dCor ({split}) = {100.0 * dcors[split]:.2f}%", 42, INFO_TEXT_SIZE)
                  for split in SPLITS if split in dcors]
    info_rows.append((mass_point_label(signal), 42, INFO_TEXT_SIZE))
    for row, (text, font, size) in enumerate(info_rows):
        CMS.drawText(text, posX=TEXT_X, posY=header_row_center(row), font=font,
                     align=12, size=size)
    canvas.cd(1).RedrawAxis()

    canvas.cd(2)
    canvas.cd(2).SetGrid()
    ref_line = ROOT.TLine(xmin, 1.0, xmax, 1.0)
    ref_line.SetLineStyle(ROOT.kDotted)
    ref_line.SetLineColor(ROOT.kBlack)
    ref_line.SetLineWidth(PLOT_LINE_WIDTH)
    ref_line.Draw()
    keepalive.append(ref_line)

    for split, suffix, hist in hists:
        if suffix == "nocut":
            continue
        ratio = _make_ratio_hist(hist, refs[split], f"{hist.GetName()}_ratio")
        draw_hist(ratio, style_from=hist)
        keepalive.append(ratio)
    canvas.cd(2).RedrawAxis()

    canvas._keepalive = keepalive
    os.makedirs(output_path.parent, exist_ok=True)
    canvas.SaveAs(str(output_path))
    canvas.Close()


def header_row_center(row: int) -> float:
    return HEADER_TOP - (row + 0.5) * HEADER_ROW_PITCH


def header_legend(x_range: Tuple[float, float], n_rows: int, margin: float):
    """A legend whose rows sit on the header grid."""
    legend = CMS.cmsLeg(x_range[0], HEADER_TOP - n_rows * HEADER_ROW_PITCH,
                        x_range[1], HEADER_TOP, textSize=LEGEND_TEXT_SIZE, columns=1)
    legend.SetMargin(margin)
    return legend


def draw_legends(region_defs, tag: str) -> List[object]:
    """Selection key in colour, and a black train/test line-style key beside it."""
    proxies: List[object] = []

    def proxy(color: int, line_style: int) -> "ROOT.TGraph":
        graph = ROOT.TGraph(2)
        graph.SetName(f"leg_{tag}_{len(proxies)}")
        graph.SetLineColor(color)
        graph.SetLineWidth(PLOT_LINE_WIDTH)
        graph.SetLineStyle(line_style)
        graph.SetMarkerSize(0)
        proxies.append(graph)
        return graph

    region_legend = header_legend(REGION_LEGEND_X, len(region_defs), REGION_LEGEND_MARGIN)
    for _suffix, label, _selector, color in region_defs:
        region_legend.AddEntry(proxy(color, ROOT.kSolid), label, "L")
    region_legend.Draw()

    style_legend = header_legend(STYLE_LEGEND_X, 2, STYLE_LEGEND_MARGIN)
    style_legend.AddEntry(proxy(ROOT.kBlack, ROOT.kSolid), "Test", "L")
    style_legend.AddEntry(proxy(ROOT.kBlack, TRAIN_LINE_STYLE), "Train", "L")
    style_legend.Draw()

    return proxies + [region_legend, style_legend]


def draw_hist(hist: "ROOT.TH1D", style_from: "ROOT.TH1D" = None) -> None:
    """Histogram line plus its error bars, styled like `style_from` (default: itself)."""
    ref = style_from or hist
    style = dict(
        LineColor=ref.GetLineColor(),
        LineWidth=ref.GetLineWidth(),
        LineStyle=ref.GetLineStyle(),
    )
    CMS.cmsObjectDraw(hist, "hist", **style)
    CMS.cmsObjectDraw(hist, "E0 SAME", MarkerColor=ref.GetLineColor(), MarkerSize=0, **style)


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input)
    label = LABEL_MODES[args.label_mode]

    for signal in args.signals:
        iteration, model_idx, _summary_path = load_best_model(
            input_dir, args.channel, signal, args.fold
        )
        predictions = load_predictions(prediction_path(
            input_dir, args.channel, signal, args.fold, iteration, model_idx
        ))
        best_model_dir = input_dir / args.channel / signal / f"fold-{args.fold}" / "best_model"
        print(f"{signal}: GA-iter{iteration} model{model_idx}")
        for mass_name in MASS_NAMES:
            for class_idx, class_label in enumerate(CLASS_NAMES):
                output_path = best_model_dir / output_name(class_label, mass_name, label)
                plot_one_class(predictions, signal, mass_name, class_label, class_idx,
                               output_path, label)
                print(f"  -> {output_path}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
