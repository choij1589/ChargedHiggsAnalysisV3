#!/usr/bin/env python3
"""Produce paper-style pre-fit / B-only / S+B post-fit mass template PDFs.

Reproduces the diagnostic PNGs written by ``plotPostfitMass.py`` into
``.../combine_output/fitdiag/plots_mass/{prefit,postfit_b,postfit_s}_mass_{Run2,Run3}_{SR1E2Mu,SR3Mu}.png``
as vector PDFs in the paper style shared with ``plotPaperLRModified.py`` and
``plotPaperPostfitSummary.py``.

Each of the four (era, channel) panels maps to exactly one Run-period category
(``SR1E2Mu_Run2``, ``SR3Mu_Run2``, ``SR1E2Mu_Run3``, ``SR3Mu_Run3``), so the
content comes straight from ``shapes.root`` plus the FitDiagnostics shape
folders -- no sample-tree refill and no fine-mass cache is involved.

Usage:
    python3 python/plotPaperTemplates.py --masspoint MHc130_MA90
    python3 python/plotPaperTemplates.py --masspoint all
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import ROOT

MODULE_DIR = Path(__file__).resolve().parents[1]
WORKDIR = Path(os.environ.get("WORKDIR", MODULE_DIR.parent))

sys.path.insert(0, str(WORKDIR / "Common" / "Tools"))
sys.path.insert(0, str(MODULE_DIR / "python"))
from plotter import (ComparisonCanvas, EnergyInfo,  # noqa: E402
                     LumiInfoExact, PALETTE_LONG, build_ratio_uncertainty_band)
import plotPostfitMass as pfm  # noqa: E402
import cmsstyle as CMS  # noqa: E402
# Paper wording is defined once in the LR_modified script; reuse it so the
# figure sets cannot drift apart. These panels keep their legend in-plot, so
# only the labels are shared, not the standalone-legend machinery.
from plotPaperLRModified import (BKG_LABELS, DATA_LABEL,  # noqa: E402
                                 RATIO_LABEL, SIGNAL_FILL_ALPHA,
                                 SIGNAL_LINE_WIDTH, SYST_LABEL)

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

FIT_ERA = "All"
FIT_CHANNEL = "Combined"
ERA_SCOPES = ("Run2", "Run3")
CHANNEL_SCOPES = ("SR1E2Mu", "SR3Mu")
FIT_STAGES = ("prefit", "b", "s")

# Paper background grouping, identical to plotPaperLRModified.py
GROUP_MAP = {
    "nonprompt": ("nonprompt",),
    "conv": ("conversion",),
    "diboson": ("WZ", "ZZ"),
    "ttX": ("ttZ", "tZq", "ttH", "ttW"),
    "others": ("others",),
}
BKG_ORDER = ("others", "conv", "diboson", "ttX", "nonprompt")
BKG_COLORS = {
    "nonprompt": PALETTE_LONG[0],
    "diboson": PALETTE_LONG[1],
    "ttX": PALETTE_LONG[2],
    "conv": PALETTE_LONG[3],
    "others": PALETTE_LONG[4],
}
# ComparisonCanvas labels the stack with the dict keys, so the merged
# backgrounds are keyed by legend label and the colours looked up by it too.
LABEL_COLORS = {BKG_LABELS[group]: BKG_COLORS[group] for group in BKG_ORDER}

CHANNEL_LABELS = {
    "SR1E2Mu": ("SR", "e#mu#mu"),
    "SR3Mu": ("SR", "#mu#mu#mu"),
}
# Wording matches plotPaperPostfitSummary.py so the paper's post-fit figures agree.
STAGE_LABELS = {"prefit": "Pre-fit", "b": "B-only Post-fit", "s": "S+B Post-fit"}
STAGE_FILE_TAGS = {"prefit": "prefit", "b": "postfit_b", "s": "postfit_s"}
# Stages that may overlay the signal template: pre-fit at its nominal r=1, and
# S+B at the fitted r -- but only when that fit returns a positive r, since a
# negative one has no curve that can be drawn above the baseline. B-only pins
# r=0. Pre-fit and S+B both quote sigma_sig; B-only has nothing to quote.
SIGNAL_STAGES = ("prefit", "s")
XSEC_QUOTE_STAGES = ("prefit", "s")

# Signal cross-section normalization. r=1 corresponds to REFERENCE_XSEC at
# 13 TeV (see collectLimits.py and the "normalization to 5 fb" in
# preprocess.py). The signal is produced in top-quark decays, so the Run3
# reference scales with the ttbar cross section from 13 to 13.6 TeV.
# ttbar values are NNLO+NNLL from
# https://twiki.cern.ch/twiki/bin/view/LHCPhysics/TtbarNNLO
REFERENCE_XSEC = 5.0  # fb, at 13 TeV
TTBAR_XSEC_13TEV = 833.9e3  # fb
TTBAR_XSEC_13P6TEV = 923.6e3  # fb
REFERENCE_XSEC_BY_ERA = {
    "Run2": REFERENCE_XSEC,
    "Run3": REFERENCE_XSEC * TTBAR_XSEC_13P6TEV / TTBAR_XSEC_13TEV,
}
SIGNAL_COLOR = ROOT.kBlack

# Main (background) legend, and the separate full-width line the signal entry
# gets on pre-fit plots so its mass label is not squeezed into one column.
# The right edge stops short of the frame so the longest label ("Nonprompt")
# clears the right-hand axis ticks.
LEGEND_BOX = (0.45, 0.65, 0.94, 0.89)
LEGEND_TEXT_SIZE = 0.038
SIGNAL_LEGEND_BOX = (0.45, 0.59, 0.94, 0.65)
SIGNAL_LEGEND_TEXT_SIZE = 0.038
# TLegend sizes the symbol column as a fraction of the whole box, so a
# full-width single-entry legend would draw a line twice as long as the
# two-column entries above it. Match them instead.
SIGNAL_LEGEND_MARGIN = 0.12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create paper pre-fit / B-only / S+B post-fit mass template PDFs.")
    parser.add_argument("--masspoint", default="MHc130_MA90",
                        help="mass point, or 'all' for every unblinded fitdiag")
    parser.add_argument("--method", default="ParticleNet")
    parser.add_argument("--binning", default="extended")
    parser.add_argument("--output-root", default="results/paper/templates",
                        help="base output directory, relative to SignalRegionStudyV3 "
                             "unless absolute")
    parser.add_argument("--ratio-max", type=float, default=3.5,
                        help="upper edge of the Data/Pred pad. Points and error bars "
                             "above this are clipped; pass 5.0 to contain every one "
                             "of the unblinded templates (default: 3.5)")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_output_root(path):
    output_root = Path(path)
    if not output_root.is_absolute():
        output_root = MODULE_DIR / output_root
    return output_root


def discover_masspoints(method, binning):
    """Mass points with a full-unblind All/Combined fitDiagnostics file."""
    suffix = f"{binning}_unblind"
    base = MODULE_DIR / "templates" / FIT_ERA / FIT_CHANNEL
    found = []
    for masspoint_dir in sorted(base.iterdir()):
        fitdiag = (masspoint_dir / method / suffix / "combine_output" / "fitdiag"
                   / f"fitDiagnostics.{masspoint_dir.name}.{method}.{suffix}.root")
        if fitdiag.exists():
            found.append(masspoint_dir.name)
    if not found:
        raise FileNotFoundError(
            f"No unblinded fitDiagnostics files found under {base} for method={method}")
    return found


def setup_module(masspoint, method, binning, debug):
    """Populate plotPostfitMass module state for this mass point."""
    pfm.entry_setup(SimpleNamespace(
        era=FIT_ERA,
        masspoint=masspoint,
        method=method,
        binning=binning,
        era_scope=None,
        channel_scope=None,
        fit_channel=FIT_CHANNEL,
        fit_type="b",
        unblind=True,
        partial_unblind=False,
        blind=False,
        nuisance="fallback_lnn",
        bin_width="auto",
        plot_only=False,
        debug=debug,
    ), make_output_dir=False)


def category_name(era_scope, channel_scope):
    return f"{channel_scope}_{era_scope}"


def read_best_fit_r(fitdiag):
    """Return (r, +err, -err) from the S+B fit result, or None if unavailable.

    The signal strength is a property of the whole fit, so it is the same on
    every (era, channel) panel drawn from one fitDiagnostics file.
    """
    tree = fitdiag.Get("tree_fit_sb")
    if not tree or tree.GetEntries() == 0:
        return None
    tree.GetEntry(0)
    return tree.r, abs(tree.rHiErr), abs(tree.rLoErr)


def format_xsec_quote(era_scope, fit_stage, best_fit_r):
    """TLatex string for the signal cross section on this panel.

    Pre-fit quotes the r=1 reference itself; S+B scales it by the fitted r and
    carries the fit uncertainty. The error collapses to "+/- e" when the two
    sides agree once rounded for display, otherwise the asymmetric
    super/subscript form is kept.
    """
    reference = REFERENCE_XSEC_BY_ERA[era_scope]
    if fit_stage == "prefit":
        return f"#sigma_{{sig}} = {reference:.2f} fb"
    if best_fit_r is None:
        return None
    r, hi, lo = best_fit_r
    central = f"{r * reference:.2f}".replace("-", "#minus")
    hi_text, lo_text = f"{hi * reference:.2f}", f"{lo * reference:.2f}"
    if hi_text == lo_text:
        return f"#sigma_{{sig}} = {central} #pm {hi_text} fb"
    return f"#sigma_{{sig}} = {central}^{{+{hi_text}}}_{{#minus{lo_text}}} fb"


def build_content(fitdiag, shapes_file, metadata, category, fit_stage):
    """Extract data, per-process-group backgrounds, signal and the fit
    uncertainty band for one Run-period category on its own template bins.

    Mirrors the single-category branch of
    ``plotPostfitMass.make_run_period_postfit_plots``.
    """
    edges = tuple(float(x) for x in metadata["binning"][category]["bin_edges"])
    source_edges_by_cat = {category: edges}

    cat_dir = shapes_file.Get(category)
    if not cat_dir:
        raise RuntimeError(f"Missing category '{category}' in {pfm.TEMPLATE_DIR}/shapes.root")
    cat_data = cat_dir.Get("data_obs")
    if not cat_data:
        raise RuntimeError(f"Missing data_obs for category '{category}'")
    data = pfm.clone_empty_like(edges, f"data_{category}_{fit_stage}")
    pfm.add_binwise(data, cat_data, edges)
    data.SetTitle(DATA_LABEL)

    process_names = {meta["name"] for meta in metadata["categories"][category]["processes"]}
    bkgs = {}
    total = pfm.clone_empty_like(edges, f"total_{category}_{fit_stage}")
    signal = pfm.clone_empty_like(edges, f"signal_{category}_{fit_stage}")
    for group, members in metadata["process_list"].get("physics_groups", {}).items():
        cat_members = [proc for proc in dict.fromkeys(members) if proc in process_names]
        if not cat_members:
            continue
        hist = pfm.sum_component_group(
            fitdiag, fit_stage, category, cat_members, edges,
            f"{group}_{category}_{fit_stage}", edges)
        if group == "signal":
            signal.Add(hist)
        elif hist.Integral() > 0:
            bkgs[group] = hist
            total.Add(hist)

    uncertainty = pfm.build_fitdiag_uncertainty_hist(
        fitdiag, fit_stage, edges, {category: total}, source_edges_by_cat,
        f"uncertainty_{category}_{fit_stage}")
    return edges, data, bkgs, signal, uncertainty


def clone_or_add(total, hist, name):
    if total is None:
        total = hist.Clone(name)
        total.SetDirectory(0)
    else:
        total.Add(hist)
    return total


def merge_backgrounds(bkgs, tag):
    """Merge the nine analysis processes into the five paper groups."""
    merged = {}
    for group in BKG_ORDER:
        total = None
        for source in GROUP_MAP[group]:
            hist = bkgs.get(source)
            if hist is None:
                continue
            total = clone_or_add(total, hist, f"{group}_{tag}")
        if total is not None and total.Integral() > 0:
            label = BKG_LABELS[group]
            total.SetTitle(label)
            merged[label] = total
    if not merged:
        raise RuntimeError(f"No backgrounds available after paper grouping for {tag}")
    return merged


def stack_maximum(bkgs):
    total = None
    for hist in bkgs.values():
        total = clone_or_add(total, hist, "y_scale_total")
    return total.GetMaximum() if total is not None else 0.0


def build_config(era_scope, channel_scope, edges, data, bkgs, signal,
                 show_signal, ratio_max):
    channel_label, region_label = CHANNEL_LABELS[channel_scope]
    y_max = max(stack_maximum(bkgs), data.GetMaximum(),
                signal.GetMaximum() if show_signal else 0.0)
    return {
        "era": era_scope,
        "CoM": EnergyInfo[era_scope],
        # Exact per-period sum (137.6, not the rounded 138) so every paper
        # figure quotes the same Run2 luminosity.
        "run_label": f"{LumiInfoExact[era_scope]:g} fb^{{#minus1}}",
        "xTitle": "m(#mu^{+}, #mu^{-}) [GeV]",
        "yTitle": "Events",
        "rTitle": RATIO_LABEL,
        "systSrc": SYST_LABEL,
        "xRange": [edges[0], edges[-1]],
        "yRange": [0.0, (y_max if y_max > 0 else 1.0) * 2.0],
        "rRange": [0.0, ratio_max],
        "maxDigits": 3,
        "overflow": False,
        # Single-energy header on these per-Run panels, so "CMS" stays above
        # the frame and the full frame height is available for the plot.
        "iPos": 0,
        # Two columns: data + 5 background groups + Stat+Syst fill four rows.
        # The signal gets its own full-width line below (see draw_panel), so the
        # long mass label never has to fit inside a half-width column.
        "legend": LEGEND_BOX,
        "legendTextSize": LEGEND_TEXT_SIZE,
        "legendColumns": 2,
        "colors": [LABEL_COLORS[name] for name in bkgs.keys()],
        "channel": channel_label,
        "region": region_label,
        "channelPosX": 0.22,
        "channelPosY": 0.78,
        "chi2_test": False,
        "normalize_chi2": False,
    }


def draw_panel(out_path, era_scope, channel_scope, fit_stage,
               edges, data, bkgs, signal, uncertainty, masspoint, ratio_max=5.0,
               best_fit_r=None):
    # A positive integral is exactly the r > 0 condition on the S+B panel, and
    # is always true pre-fit.
    show_signal = (fit_stage in SIGNAL_STAGES and signal is not None
                   and signal.Integral() > 0.0)
    config = build_config(era_scope, channel_scope, edges, data, bkgs, signal,
                          show_signal, ratio_max)
    plotter = ComparisonCanvas(data, bkgs, config)

    # FitDiagnostics stores the correlated total-background uncertainty; adopt it
    # for the stack band and rebuild the ratio band so both pads agree. The base
    # class builds ratio_band in __init__ from the quadrature sum of per-process
    # errors, which is narrower than the correlated band.
    pfm.apply_uncertainty_hist(plotter.systematics, uncertainty)
    plotter.ratio_band = build_ratio_uncertainty_band(
        plotter.systematics, f"ratio_uncertainty_{era_scope}_{channel_scope}_{fit_stage}")

    plotter.drawPadUp()

    if show_signal:
        plotter.canv.cd(1)
        # Shaded band under a heavy line, matching plotPaperLRModified.py.
        signal.SetLineColor(SIGNAL_COLOR)
        signal.SetLineWidth(SIGNAL_LINE_WIDTH)
        signal.SetLineStyle(ROOT.kSolid)
        signal.SetFillColorAlpha(SIGNAL_COLOR, SIGNAL_FILL_ALPHA)
        signal.SetFillStyle(1001)
        signal.Draw("HIST SAME")
        plotter.signal = signal  # keep alive
        signal_legend = CMS.cmsLeg(*SIGNAL_LEGEND_BOX,
                                   textSize=SIGNAL_LEGEND_TEXT_SIZE, columns=1)
        signal_legend.SetMargin(SIGNAL_LEGEND_MARGIN)
        signal_legend.AddEntry(signal, pfm.masspoint_label(masspoint), "F")
        signal_legend.Draw()
        plotter.signal_legend = signal_legend  # keep alive
        plotter.canv.cd(1).RedrawAxis()

    plotter.canv.cd(1)
    stage_label = ROOT.TLatex()
    stage_label.SetNDC(True)
    stage_label.SetTextFont(42)
    stage_label.SetTextSize(0.040)
    stage_label.DrawLatex(0.22, 0.66, STAGE_LABELS[fit_stage])
    plotter.stage_label = stage_label  # keep alive

    if fit_stage in XSEC_QUOTE_STAGES:
        quote = format_xsec_quote(era_scope, fit_stage, best_fit_r)
        if quote is not None:
            xsec_label = ROOT.TLatex()
            xsec_label.SetNDC(True)
            xsec_label.SetTextFont(42)
            xsec_label.SetTextSize(0.040)
            xsec_label.DrawLatex(0.22, 0.60, quote)
            plotter.xsec_label = xsec_label  # keep alive

    plotter.drawPadDown()
    plotter.canv.SaveAs(str(out_path))
    return plotter


def draw_masspoint(masspoint, method, binning, output_root, ratio_max, debug):
    setup_module(masspoint, method, binning, debug)
    metadata = pfm.load_run_period_metadata()
    if not metadata:
        raise FileNotFoundError(
            f"No categories.json under {pfm.TEMPLATE_DIR}; "
            f"{masspoint} is not a Run-period component workspace")

    out_dir = output_root / masspoint
    os.makedirs(out_dir, exist_ok=True)

    fitdiag = ROOT.TFile.Open(pfm.FITDIAG_PATH, "READ")
    if not fitdiag or fitdiag.IsZombie():
        raise RuntimeError(f"Failed to open {pfm.FITDIAG_PATH}")
    shapes_file = ROOT.TFile.Open(f"{pfm.TEMPLATE_DIR}/shapes.root", "READ")
    if not shapes_file or shapes_file.IsZombie():
        raise RuntimeError(f"Failed to open {pfm.TEMPLATE_DIR}/shapes.root")

    best_fit_r = read_best_fit_r(fitdiag)
    if best_fit_r is None:
        logging.warning("%s: no tree_fit_sb; S+B panels will not quote r", masspoint)
    else:
        logging.info("%s: best-fit r = %.3f +%.3f -%.3f",
                     masspoint, *best_fit_r)

    written = []
    try:
        for era_scope in ERA_SCOPES:
            for channel_scope in CHANNEL_SCOPES:
                category = category_name(era_scope, channel_scope)
                if category not in metadata["categories"]:
                    raise KeyError(
                        f"Category '{category}' missing from {pfm.TEMPLATE_DIR}/categories.json")
                for fit_stage in FIT_STAGES:
                    edges, data, bkgs, signal, uncertainty = build_content(
                        fitdiag, shapes_file, metadata, category, fit_stage)
                    merged = merge_backgrounds(bkgs, f"{category}_{fit_stage}")
                    out_path = (out_dir / f"{STAGE_FILE_TAGS[fit_stage]}_mass_"
                                          f"{era_scope}_{channel_scope}.pdf")
                    draw_panel(out_path, era_scope, channel_scope, fit_stage,
                               edges, data, merged, signal, uncertainty, masspoint,
                               ratio_max, best_fit_r)
                    logging.info(
                        "%s: data=%.0f  bkg=%.2f  signal=%.2f  (%d bins) -> %s",
                        f"{category}/{fit_stage}", data.Integral(),
                        sum(h.Integral() for h in merged.values()), signal.Integral(),
                        len(edges) - 1, out_path)
                    written.append(out_path)
    finally:
        shapes_file.Close()
        fitdiag.Close()
    return written


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s - %(message)s")
    output_root = resolve_output_root(args.output_root)

    masspoints = (discover_masspoints(args.method, args.binning)
                  if args.masspoint == "all" else [args.masspoint])
    logging.info("Mass points: %s", ", ".join(masspoints))

    total = 0
    for masspoint in masspoints:
        total += len(draw_masspoint(masspoint, args.method, args.binning,
                                    output_root, args.ratio_max, args.debug))
    logging.info("Done. Wrote %d PDFs under %s", total, output_root)


if __name__ == "__main__":
    main()
