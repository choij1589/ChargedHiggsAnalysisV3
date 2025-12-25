#!/usr/bin/env python3
"""
Generate Brazilian limit plots from collected limit JSON files.

Usage:
    plotLimits.py --era 2018 --channel SR1E2Mu --method Baseline --binning uniform
"""
import os
import sys
import json
import argparse
import logging
from array import array

import ROOT
ROOT.gROOT.SetBatch(True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Try to import CMS style
try:
    import cmsstyle as CMS
    HAS_CMSSTYLE = True
except ImportError:
    HAS_CMSSTYLE = False
    logging.warning("cmsstyle not found, using default ROOT style")

# Luminosity info (fb^-1)
LUMI_INFO = {
    "2016preVFP": 19.5,
    "2016postVFP": 16.8,
    "2017": 41.5,
    "2018": 59.8,
    "FullRun2": 138.0,
    "2022": 8.0,
    "2022EE": 27.0,
    "2023": 18.0,
    "2023BPix": 10.0,
    "FullRun3": 63.0,
}


def create_graphs(limits_dict):
    """Create TGraph objects from limits dictionary."""
    if not limits_dict:
        return None

    # Sort by mA
    mass_points = sorted(limits_dict.keys(), key=lambda mp: int(mp.split("_")[1][2:]))
    x = array('d', [int(mp.split("_")[1][2:]) for mp in mass_points])
    n = len(x)

    if n == 0:
        return None

    # Extract limit values (already in branching ratio units from collectLimits.py)
    limits = {}
    for key in ["obs", "exp0", "exp-1", "exp-2", "exp+1", "exp+2"]:
        limits[key] = array('d', [limits_dict[mp].get(key, 0) for mp in mass_points])

    # Create graphs
    g_obs = ROOT.TGraph(n, x, limits["obs"])
    g_obs.SetLineWidth(2)
    g_obs.SetLineColor(ROOT.kBlack)
    g_obs.SetMarkerStyle(20)
    g_obs.SetMarkerSize(0.8)
    g_obs.SetMarkerColor(ROOT.kBlack)

    g_exp = ROOT.TGraph(n, x, limits["exp0"])
    g_exp.SetLineWidth(2)
    g_exp.SetLineStyle(2)
    g_exp.SetLineColor(ROOT.kBlack)

    # 1-sigma band (green)
    g_exp1sigma = ROOT.TGraphAsymmErrors(n)
    g_exp1sigma.SetFillColor(ROOT.kGreen + 1)
    g_exp1sigma.SetFillStyle(1001)

    # 2-sigma band (yellow)
    g_exp2sigma = ROOT.TGraphAsymmErrors(n)
    g_exp2sigma.SetFillColor(ROOT.kOrange)
    g_exp2sigma.SetFillStyle(1001)

    for i in range(n):
        g_exp1sigma.SetPoint(i, x[i], limits["exp0"][i])
        g_exp2sigma.SetPoint(i, x[i], limits["exp0"][i])

        g_exp1sigma.SetPointError(i, 0, 0,
                                   limits["exp0"][i] - limits["exp-1"][i],
                                   limits["exp+1"][i] - limits["exp0"][i])
        g_exp2sigma.SetPointError(i, 0, 0,
                                   limits["exp0"][i] - limits["exp-2"][i],
                                   limits["exp+2"][i] - limits["exp0"][i])

    all_values = []
    for key in limits:
        all_values.extend(limits[key])

    return {
        'obs': g_obs,
        'exp': g_exp,
        'exp1sigma': g_exp1sigma,
        'exp2sigma': g_exp2sigma,
        'values': all_values,
        'x_min': min(x),
        'x_max': max(x)
    }


def setup_canvas(era, x_min=15, x_max=155, y_max=1e-5):
    """Create and configure canvas."""
    if HAS_CMSSTYLE:
        era_label = "Run2" if era == "FullRun2" else era
        CMS.SetExtraText("Preliminary")
        CMS.SetLumi(LUMI_INFO.get(era, 0), run=era_label)
        CMS.SetEnergy("13" if "20" in era and int(era[:4]) < 2022 else "13.6")

        canv = CMS.cmsCanvas("limit", x_min, x_max, 0, y_max,
                             "m_{A} [GeV]", "95% CL limit on B(t #rightarrow H^{#pm}b) #times B(H^{#pm} #rightarrow W^{#pm}A)",
                             square=True, iPos=11, extraSpace=0.01)
    else:
        canv = ROOT.TCanvas("limit", "Limit Plot", 800, 800)
        canv.SetLogy(True)
        canv.SetGrid()

        frame = canv.DrawFrame(x_min, 0, x_max, y_max)
        frame.GetXaxis().SetTitle("m_{A} [GeV]")
        frame.GetYaxis().SetTitle("95% CL limit on BR")

    return canv


def main():
    parser = argparse.ArgumentParser(description="Generate Brazilian limit plots")
    parser.add_argument("--era", required=True, help="Data-taking period")
    parser.add_argument("--channel", required=True, help="Analysis channel")
    parser.add_argument("--method", required=True, help="Template method")
    parser.add_argument("--binning", default="uniform", help="Binning scheme")
    parser.add_argument("--limit-type", default="Asymptotic",
                        choices=["Asymptotic", "HybridNew"],
                        help="Limit calculation method")
    parser.add_argument("--output", default=None, help="Output plot path")
    parser.add_argument("--logy", action="store_true", help="Use log scale for y-axis")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Get WORKDIR
    WORKDIR = os.getenv("WORKDIR")
    if not WORKDIR:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'")

    # Load limits
    json_path = f"{WORKDIR}/SignalRegionStudyV1/results/json/limits.{args.era}.{args.channel}.{args.limit_type}.{args.method}.{args.binning}.json"

    if not os.path.exists(json_path):
        logging.error(f"Limit file not found: {json_path}")
        sys.exit(1)

    with open(json_path) as f:
        limits = json.load(f)

    logging.info(f"Loaded {len(limits)} mass points from {json_path}")

    if not limits:
        logging.error("No limits found")
        sys.exit(1)

    # Create graphs
    graphs = create_graphs(limits)

    if not graphs:
        logging.error("Failed to create graphs")
        sys.exit(1)

    # Calculate y-axis range
    y_max = max(v for v in graphs['values'] if v > 0) * 2
    y_min = min(v for v in graphs['values'] if v > 0) * 0.5

    # Create canvas
    canv = ROOT.TCanvas("limit", "Limit Plot", 800, 800)
    canv.SetLeftMargin(0.15)
    canv.SetRightMargin(0.05)
    canv.SetTopMargin(0.08)
    canv.SetBottomMargin(0.12)

    if args.logy:
        canv.SetLogy(True)
        y_min = max(y_min, 1e-8)

    # Create frame
    x_min = max(10, graphs['x_min'] - 10)
    x_max = min(170, graphs['x_max'] + 10)

    frame = canv.DrawFrame(x_min, y_min if args.logy else 0, x_max, y_max)
    frame.GetXaxis().SetTitle("m_{A} [GeV]")
    frame.GetYaxis().SetTitle("95% CL limit on B(t #rightarrow H^{#pm}b) #times B(H^{#pm} #rightarrow W^{#pm}A)")
    frame.GetXaxis().SetTitleSize(0.045)
    frame.GetYaxis().SetTitleSize(0.045)
    frame.GetXaxis().SetLabelSize(0.04)
    frame.GetYaxis().SetLabelSize(0.04)
    frame.GetYaxis().SetTitleOffset(1.5)

    # Draw bands (2-sigma first, then 1-sigma)
    graphs['exp2sigma'].Draw("E3 same")
    graphs['exp1sigma'].Draw("E3 same")

    # Draw lines
    graphs['exp'].Draw("L same")
    graphs['obs'].Draw("LP same")

    # Legend
    leg = ROOT.TLegend(0.55, 0.70, 0.90, 0.90)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetTextSize(0.035)
    leg.AddEntry(graphs['obs'], "Observed", "lp")
    leg.AddEntry(graphs['exp'], "Expected", "l")
    leg.AddEntry(graphs['exp1sigma'], "Expected #pm1#sigma", "f")
    leg.AddEntry(graphs['exp2sigma'], "Expected #pm2#sigma", "f")
    leg.Draw()

    # Add CMS label
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(61)
    latex.SetTextSize(0.05)
    latex.DrawLatex(0.15, 0.93, "CMS")

    latex.SetTextFont(52)
    latex.SetTextSize(0.04)
    latex.DrawLatex(0.26, 0.93, "Preliminary")

    # Add lumi label
    latex.SetTextFont(42)
    latex.SetTextSize(0.035)
    lumi = LUMI_INFO.get(args.era, 0)
    energy = "13" if args.era in ["2016preVFP", "2016postVFP", "2017", "2018", "FullRun2"] else "13.6"
    latex.DrawLatex(0.65, 0.93, f"{lumi:.1f} fb^{{-1}} ({energy} TeV)")

    # Add era/channel info
    latex.SetTextSize(0.03)
    latex.DrawLatex(0.55, 0.65, f"{args.era} / {args.channel}")
    latex.DrawLatex(0.55, 0.60, f"{args.method} / {args.binning}")

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_dir = f"{WORKDIR}/SignalRegionStudyV1/results/plots"
        os.makedirs(output_dir, exist_ok=True)
        output_path = f"{output_dir}/limit.{args.era}.{args.channel}.{args.limit_type}.{args.method}.{args.binning}.png"

    # Save
    canv.SaveAs(output_path)
    logging.info(f"Saved: {output_path}")

    # Also save as PDF
    pdf_path = output_path.replace(".png", ".pdf")
    canv.SaveAs(pdf_path)
    logging.info(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
