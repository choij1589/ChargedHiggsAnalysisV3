#!/usr/bin/env python
"""
Plot 2D electron supercluster eta-phi distributions before and after HEM veto.
Hardcoded to 2018 era, Run1E2Mu flag (only channel with electrons + HEM veto).
Produces 12 plots: 6 sample categories x {BeforeHEMVeto, AfterHEMVeto}.
"""
import os
import json
import logging
import argparse
import ROOT
import cmsstyle as CMS
from plotter import get_CoM_energy, LumiInfo
from utils import build_sknanoutput_path

ROOT.gROOT.SetBatch(True)
CMS.setCMSStyle()

ERA = "2018"
FLAG = "Run1E2Mu"
CHANNEL = "SR1E2Mu"
STAGES = ["BeforeHEMVeto", "AfterHEMVeto"]
CATEGORIES = ["data", "nonprompt", "conv", "ttX", "diboson", "others"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot 2D electron scEta-scPhi for HEM veto diagnostics (2018)"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


def load_histogram_2d(file_path, hist_path):
    """Load a single TH2F from a ROOT file. Returns None if not found."""
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        return None

    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        logging.warning(f"Cannot open file: {file_path}")
        return None

    hist = f.Get(hist_path)
    if not hist:
        logging.warning(f"Histogram {hist_path} not found in {file_path}")
        f.Close()
        return None

    hist.SetDirectory(0)
    f.Close()
    logging.debug(f"Loaded {hist_path} from {file_path}")
    return hist


def load_and_merge(workdir, samples, stage, category):
    """Sum TH2 across all samples in a category for the given stage."""
    hist_path = f"ElectronScEtaPhi/{stage}/scEta_scPhi"
    merged = None

    for sample in samples:
        is_nonprompt = (category == "nonprompt")
        is_mc = category not in ("data", "nonprompt")
        file_path = build_sknanoutput_path(
            workdir,
            channel=CHANNEL,
            flag=FLAG,
            era=ERA,
            sample=sample,
            is_nonprompt=is_nonprompt,
            run_syst=is_mc,
        )
        hist = load_histogram_2d(file_path, hist_path)
        if hist is None:
            continue
        if merged is None:
            merged = hist.Clone(f"{category}_{stage}")
        else:
            merged.Add(hist)

    if merged is None:
        logging.warning(f"No histograms found for category={category}, stage={stage}")
    return merged


def plot_2d(hist, output_path, category, stage):
    """Create 2D scEta-scPhi COLZ plot with CMS style."""
    canvas = ROOT.TCanvas("c", "", 900, 800)
    canvas.SetRightMargin(0.15)
    canvas.SetLeftMargin(0.1)
    canvas.SetBottomMargin(0.1)
    canvas.SetTopMargin(0.06)

    ROOT.gStyle.SetPalette(ROOT.kRainBow)
    ROOT.gStyle.SetOptStat(0)

    hist.GetXaxis().SetTitle("SC #eta")
    hist.GetYaxis().SetTitle("SC #phi")
    hist.GetZaxis().SetTitle("")
    hist.GetXaxis().SetTitleSize(0.06)
    hist.GetYaxis().SetTitleSize(0.06)
    hist.GetZaxis().SetTitleSize(0.06)
    hist.GetZaxis().SetLabelSize(0.035)
    hist.GetXaxis().SetTitleOffset(0.8)
    hist.GetYaxis().SetTitleOffset(0.8)
    hist.GetZaxis().SetTitleOffset(0.8)

    hist.GetXaxis().SetRangeUser(-2.5, 2.5)
    hist.GetYaxis().SetRangeUser(-3.2, 3.2)
    hist.Draw("COLZ")

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.15, 0.85, category)
    veto_label = "Before HEM Veto" if stage == "BeforeHEMVeto" else "After HEM Veto"
    latex.DrawLatex(0.15, 0.80, veto_label)

    CMS.SetEnergy(get_CoM_energy(ERA))
    CMS.SetLumi(LumiInfo[ERA], run=ERA)
    CMS.SetExtraText("Preliminary")
    CMS.CMS_lumi(canvas, 0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.SaveAs(output_path)
    logging.info(f"Saved: {output_path}")
    canvas.Close()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

    workdir = os.environ.get("WORKDIR")
    if not workdir:
        raise RuntimeError("WORKDIR not set. Run 'source setup.sh' first.")

    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "samplegroup.json")
    with open(config_path) as f:
        sample_config = json.load(f)

    samples_by_category = sample_config[ERA]["1E2Mu"]
    output_dir = os.path.join(os.path.dirname(__file__), "..", "plots", ERA, "SR1E2Mu", "ElectronScEtaPhi")

    for category in CATEGORIES:
        samples = samples_by_category[category]
        for stage in STAGES:
            hist = load_and_merge(workdir, samples, stage, category)
            if hist is None:
                logging.error(f"Skipping {category}_{stage}: no data")
                continue
            output_path = os.path.join(output_dir, f"{category}_{stage}.png")
            plot_2d(hist, output_path, category, stage)

    logging.info("Done.")


if __name__ == "__main__":
    main()
