#!/usr/bin/env python3
"""Plot the DCB signal mass resolution (sigma_eff) vs fitted peak (x0).

The Double Crystal Ball signal fits performed in makeBinnedTemplates.py are
already persisted per mass point under
    templates/All/Combined/{masspoint}/{method}/{binning_suffix}/binning.json
with, for each merged Run-period category, `fit_result.x0` and `sigma_eff`.

This script aggregates those values (no re-fitting) and, for each MHc value,
draws one CMS-style figure with x-axis = x0 and y-axis = sigma_eff, overlaying
the four categories SR1E2Mu_Run2, SR3Mu_Run2, SR1E2Mu_Run3, SR3Mu_Run3.
"""
import os
import re
import glob
import json
import argparse
from array import array

import ROOT
import cmsstyle as CMS
from plotter import PALETTE_LONG

ROOT.gROOT.SetBatch(ROOT.kTRUE)

# Load luminosity configuration (for the combined "All" header label).
_LUMI_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Common", "Data", "Luminosity.json")
with open(_LUMI_JSON_PATH, "r") as f:
    _LUMI_CONFIG = json.load(f)

# Categories overlaid on every plot, in draw order, with display labels.
_CATEGORIES = [
    ("SR1E2Mu_Run2", "e#mu#mu, Run2"),
    ("SR3Mu_Run2", "#mu#mu#mu, Run2"),
    ("SR1E2Mu_Run3", "e#mu#mu, Run3"),
    ("SR3Mu_Run3", "#mu#mu#mu, Run3"),
]
_MARKERS = [24, 25, 24, 25]  # open circle (SR1E2Mu) / square (SR3Mu); era by color
_MHC_VALUES = [70, 85, 100, 115, 130, 145, 160]


def get_all_energy_label():
    run2 = _LUMI_CONFIG["Run2"]["energy_TeV"]
    run3 = _LUMI_CONFIG["Run3"]["energy_TeV"]
    return f"{run2:g}/{run3:g} TeV"


def collect_fit_results(workdir, binning_suffix):
    """Return {mhc: {category: [(x0, sigma_eff), ...]}} sorted by x0.

    Reads x0/sigma_eff from the All/Combined binning.json of every mass point.
    The DCB fit values are shared across methods, so the full Baseline mass grid
    is used as the single source.
    """
    pattern = os.path.join(
        workdir, "SignalRegionStudyV3", "templates", "All", "Combined",
        "*", "Baseline", binning_suffix, "binning.json",
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No binning.json found for binning={binning_suffix}: {pattern}"
        )

    data = {mhc: {cat: [] for cat, _ in _CATEGORIES} for mhc in _MHC_VALUES}
    mp_re = re.compile(r"MHc(\d+)_MA(\d+)")
    for path in files:
        m = mp_re.search(path)
        if not m:
            print(f"WARNING: cannot parse mass point from {path}; skipping")
            continue
        mhc = int(m.group(1))
        if mhc not in data:
            print(f"WARNING: MHc{mhc} not in target list; skipping {path}")
            continue
        with open(path) as fh:
            categories = json.load(fh).get("categories", {})
        for cat, _ in _CATEGORIES:
            entry = categories.get(cat)
            if entry is None:
                print(f"WARNING: category {cat} missing in {path}; skipping")
                continue
            fit_result = entry.get("fit_result", {})
            x0 = fit_result.get("x0")
            sigma_eff = entry.get("sigma_eff", fit_result.get("sigma_eff"))
            if x0 is None or sigma_eff is None:
                print(f"WARNING: missing x0/sigma_eff for {cat} in {path}; skipping")
                continue
            data[mhc][cat].append((float(x0), float(sigma_eff)))

    for mhc in data:
        for cat in data[mhc]:
            data[mhc][cat].sort(key=lambda pt: pt[0])
    return data


def make_graph(points, color, marker):
    x = array('d', [p[0] for p in points])
    y = array('d', [p[1] for p in points])
    g = ROOT.TGraph(len(points), x, y)
    g.SetMarkerColor(color)
    g.SetLineColor(color)
    g.SetMarkerStyle(marker)
    g.SetMarkerSize(1.2)
    g.SetLineWidth(2)
    return g


def plot_mhc(mhc, per_cat, outdir):
    """Draw one resolution plot for a single MHc; return True if anything drawn."""
    all_points = [pt for pts in per_cat.values() for pt in pts]
    if not all_points:
        print(f"WARNING: no fit results for MHc{mhc}; skipping plot")
        return False

    x_vals = [pt[0] for pt in all_points]
    y_vals = [pt[1] for pt in all_points]
    x_min = 5.0
    x_max = max(x_vals) + 10.0
    y_max = 1.3 * max(y_vals)

    canv = CMS.cmsCanvas(
        f"res_MHc{mhc}", x_min, x_max, 0., y_max,
        "x_{0} [GeV]", "#sigma_{eff} [GeV]",
        square=True, iPos=0, extraSpace=0.01,
    )
    canv.cd()

    graphs = []  # keep refs alive
    leg = CMS.cmsLeg(0.20, 0.86 - 0.05 * len(_CATEGORIES), 0.50, 0.86, textSize=0.032)
    for idx, (cat, label) in enumerate(_CATEGORIES):
        points = per_cat.get(cat, [])
        if not points:
            continue
        color = PALETTE_LONG[idx]
        g = make_graph(points, color, _MARKERS[idx])
        CMS.cmsObjectDraw(g, "PL same")
        leg.AddEntry(g, label, "lp")
        graphs.append(g)
    canv.RedrawAxis()

    label = ROOT.TLatex()
    label.SetNDC(True)
    label.SetTextFont(42)
    label.SetTextSize(0.04)
    label.DrawLatex(0.20, 0.58, f"m_{{H^{{+}}}} = {mhc} GeV")

    os.makedirs(outdir, exist_ok=True)
    out_base = os.path.join(outdir, f"resolution.MHc{mhc}")
    canv.SaveAs(f"{out_base}.png")
    canv.SaveAs(f"{out_base}.pdf")
    print(f"Saved {out_base}.png ({len(graphs)} series)")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", type=int, default=None, choices=_MHC_VALUES,
                        help="Plot only this MHc value (default: all)")
    parser.add_argument("--binning", type=str, default="extended",
                        help="Binning base name (default: extended -> extended_unblind)")
    args = parser.parse_args()

    workdir = os.getenv("WORKDIR")
    if not workdir:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh' in SignalRegionStudyV3")

    binning_suffix = f"{args.binning}_unblind"
    data = collect_fit_results(workdir, binning_suffix)

    CMS.SetExtraText("Simulation Preliminary")
    CMS.ResetAdditionalInfo()
    CMS.SetLumi(None, run="")  # simulation only: no luminosity label
    CMS.SetEnergy(0, unit=get_all_energy_label())

    outdir = os.path.join(workdir, "SignalRegionStudyV3", "results", "plots", "resolution")
    targets = [args.mhc] if args.mhc is not None else _MHC_VALUES

    n_drawn = 0
    for mhc in targets:
        if plot_mhc(mhc, data[mhc], outdir):
            n_drawn += 1
    if n_drawn == 0:
        raise RuntimeError("No resolution plots produced; check template inputs")
    print(f"Done: {n_drawn} plot(s) under {outdir}")


if __name__ == "__main__":
    main()
