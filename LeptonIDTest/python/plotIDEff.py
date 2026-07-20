#!/usr/bin/env python
import os
import sys
import math
import argparse
import ROOT
import cmsstyle as CMS

sys.path.append(os.path.join(os.environ["WORKDIR"], "Common", "Tools"))
from plotter import LumiInfo, PALETTE_LONG

ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--era", type=str, required=True, help="Era")
parser.add_argument("--object", type=str, required=True,
                    choices=["muon", "electron"], help="Object type")
args = parser.parse_args()
WORKDIR = os.getenv("WORKDIR")

# Sources to overlay (granular; 'unknown' intentionally excluded from plots)
sources = ["prompt", "conv", "fromTau", "fromB", "fromC", "fromL", "fromPU"]
source_labels = {
    "prompt": "prompt", "conv": "conv", "fromTau": "from #tau",
    "fromB": "from b", "fromC": "from c", "fromL": "from light", "fromPU": "from PU",
}
working_points = ["loose", "tight"]

# Electron crack bins to skip (signed scEta bins 3 & 8), mirrors plotEleIDEff.py
gap_bins = [3, 8] if args.object == "electron" else []

if args.era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
    energy = 13
else:
    energy = 13.6
CMS.SetEnergy(energy)
CMS.SetLumi(LumiInfo[args.era], run=args.era)
CMS.SetExtraText("Simulation")


def clean_nan_values(hist):
    """Replace NaN/Inf efficiency bins with 0 and NaN/Inf errors with 0."""
    for b in range(0, hist.GetNbinsX() + 2):
        content, error = hist.GetBinContent(b), hist.GetBinError(b)
        if math.isnan(content) or math.isinf(content):
            hist.SetBinContent(b, 0.0)
        if math.isnan(error) or math.isinf(error):
            hist.SetBinError(b, 0.0)


f = ROOT.TFile.Open(f"{WORKDIR}/LeptonIDTest/results/{args.era}/idEff_{args.object}.root")
if not f or f.IsZombie():
    raise RuntimeError(f"Cannot open idEff_{args.object}.root for {args.era}. Run measIDEff.py first.")

eff2d = {}
for wp in working_points:
    for src in sources:
        h = f.Get(f"eff_{src}_{wp}")
        h.SetDirectory(0)
        eff2d[(src, wp)] = h
f.Close()

# Axis structure (identical across sources/WPs)
ref = eff2d[(sources[0], working_points[0])]
n_eta_bins = ref.GetNbinsX()
n_pt_bins = ref.GetNbinsY()

output_dir = f"{WORKDIR}/LeptonIDTest/plots/{args.era}/{args.object}/idEff"
os.makedirs(output_dir, exist_ok=True)


def draw_overlay(projections, xmin, xmax, xtitle, header, save_name):
    """Overlay source efficiency curves on a single-pad CMS canvas."""
    canvas = CMS.cmsCanvas(save_name, xmin, xmax, 0.0, 1.1, xtitle, "Efficiency",
                           square=True, iPos=0, extraSpace=0)
    canvas.cd()
    leg = CMS.cmsLeg(0.60, 0.55, 0.92, 0.90, textSize=0.035, columns=2)
    for src, hist in projections:
        clean_nan_values(hist)
        color = PALETTE_LONG[sources.index(src)]
        CMS.cmsDraw(hist, "PE", mcolor=color, msize=1.0, lwidth=2)
        leg.AddEntry(hist, source_labels[src], "PE")
    leg.Draw()

    label = ROOT.TLatex()
    label.SetNDC()
    label.SetTextFont(42)
    label.SetTextSize(0.04)
    label.DrawLatex(0.18, 0.83, header)

    canvas.RedrawAxis()
    canvas.SaveAs(f"{output_dir}/{save_name}.png")


for wp in working_points:
    # One pT panel per eta range (x-axis = pT)
    for eta_bin in range(1, n_eta_bins + 1):
        if eta_bin in gap_bins:
            continue
        eta_low = ref.GetXaxis().GetBinLowEdge(eta_bin)
        eta_high = ref.GetXaxis().GetBinUpEdge(eta_bin)
        projections = []
        for src in sources:
            h = eff2d[(src, wp)].ProjectionY(f"py_{src}_{wp}_eta{eta_bin}", eta_bin, eta_bin)
            h.SetDirectory(0)
            projections.append((src, h))
        xmin = ref.GetYaxis().GetXmin()
        xmax = ref.GetYaxis().GetXmax()
        if args.object == "electron":
            header = f"{wp} ID, {eta_low:.3f} < #eta_{{SC}} < {eta_high:.3f}"
        else:
            header = f"{wp} ID, {eta_low:.1f} < |#eta| < {eta_high:.1f}"
        draw_overlay(projections, xmin, xmax, "p_{T} [GeV]", header,
                     f"eff_vs_pt_{wp}_etabin{eta_bin}")

    # One eta panel per pT range (x-axis = eta)
    for pt_bin in range(1, n_pt_bins + 1):
        pt_low = ref.GetYaxis().GetBinLowEdge(pt_bin)
        pt_high = ref.GetYaxis().GetBinUpEdge(pt_bin)
        projections = []
        for src in sources:
            h = eff2d[(src, wp)].ProjectionX(f"px_{src}_{wp}_pt{pt_bin}", pt_bin, pt_bin)
            h.SetDirectory(0)
            for gap_bin in gap_bins:
                h.SetBinContent(gap_bin, 0)
                h.SetBinError(gap_bin, 0)
            projections.append((src, h))
        xmin = ref.GetXaxis().GetXmin()
        xmax = ref.GetXaxis().GetXmax()
        xtitle = "#eta_{SC}" if args.object == "electron" else "|#eta|"
        header = f"{wp} ID, {pt_low:.0f} < p_{{T}} < {pt_high:.0f} GeV"
        draw_overlay(projections, xmin, xmax, xtitle, header,
                     f"eff_vs_eta_{wp}_ptbin{pt_bin}")

print(f"All plots saved to {output_dir}")
