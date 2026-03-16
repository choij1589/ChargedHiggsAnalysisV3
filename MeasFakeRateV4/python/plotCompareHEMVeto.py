#!/usr/bin/env python
import os
import logging
import argparse
import ROOT
import cmsstyle as CMS

ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

PALETTE = [
    ROOT.TColor.GetColor("#5790fc"),
    ROOT.TColor.GetColor("#f89c20"),
    ROOT.TColor.GetColor("#e42536"),
]

WORKDIR = os.environ['WORKDIR']
ERA = "2018"
MEASURE = "electron"
ptcorr_bins = [15., 17., 20., 25., 35., 50., 100., 200.]
abseta_bins = [0., 0.8, 1.479, 2.5]

## Load histograms
hem_path = f"{WORKDIR}/MeasFakeRateV4/results/{ERA}/ROOT/{MEASURE}/fakerate.root"
nohem_path = f"{WORKDIR}/MeasFakeRateV4/results/{ERA}/ROOT/{MEASURE}/noHEMVeto/fakerate.root"
assert os.path.exists(hem_path), f"File not found: {hem_path}"
assert os.path.exists(nohem_path), f"File not found: {nohem_path}"

f_hem = ROOT.TFile.Open(hem_path)
h_hem = f_hem.Get("fake rate - (Central)")
h_hem.SetDirectory(0)
f_hem.Close()

f_nohem = ROOT.TFile.Open(nohem_path)
h_nohem = f_nohem.Get("fake rate - (Central)")
h_nohem.SetDirectory(0)
f_nohem.Close()

## Project per eta bin
hem_proj = {}
nohem_proj = {}
ratio_proj = {}
for i in range(1, 4):
    label = f"eta{i}"
    hem_proj[label] = h_hem.ProjectionY(f"hem_{label}", i, i)
    hem_proj[label].SetDirectory(0)
    nohem_proj[label] = h_nohem.ProjectionY(f"nohem_{label}", i, i)
    nohem_proj[label].SetDirectory(0)

    r = nohem_proj[label].Clone(f"ratio_{label}")
    r.SetDirectory(0)
    r.Divide(hem_proj[label])
    ratio_proj[label] = r

## Style
eta_label = "|#eta_{SC}|"
eta_ranges = [
    f"{abseta_bins[0]} < {eta_label} < {abseta_bins[1]}",
    f"{abseta_bins[1]} < {eta_label} < {abseta_bins[2]}",
    f"{abseta_bins[2]} < {eta_label} < {abseta_bins[3]}",
]

CMS.SetEnergy(13)
CMS.SetLumi(-1, run=f"{ERA}, Prescaled")
CMS.SetExtraText("Preliminary")

canvas = CMS.cmsDiCanvas("", ptcorr_bins[0], 100.,
                         0., 1.,
                         0.5, 1.5,
                         "p_{T}^{corr}",
                         "fake rate (e)",
                         "no HEM / HEM",
                         square=False,
                         iPos=0,
                         extraSpace=0.015)

## Upper pad
canvas.cd(1)
legend = CMS.cmsLeg(0.50, 0.89 - 0.06*6, 0.92, 0.89, textSize=0.05, columns=1)

for i, (label, eta_range) in enumerate(zip(["eta1", "eta2", "eta3"], eta_ranges)):
    color = PALETTE[i]

    # HEM veto: full opacity
    hem_proj[label].SetTitle("")
    hem_proj[label].SetStats(0)
    CMS.cmsObjectDraw(hem_proj[label], "", LineColor=color, LineWidth=2, LineStyle=ROOT.kSolid)
    CMS.addToLegend(legend, (hem_proj[label], f"{eta_range} (HEM veto)", "lep"))

    # No HEM veto: semi-transparent (alpha=0.5)
    nohem_proj[label].SetTitle("")
    nohem_proj[label].SetStats(0)
    nohem_proj[label].SetLineColorAlpha(color, 0.5)
    nohem_proj[label].SetLineWidth(2)
    nohem_proj[label].SetLineStyle(ROOT.kSolid)
    nohem_proj[label].SetMarkerColorAlpha(color, 0.5)
    nohem_proj[label].Draw("same")
    CMS.addToLegend(legend, (nohem_proj[label], f"{eta_range} (no HEM veto)", "lep"))

canvas.cd(1).RedrawAxis()
legend.Draw("same")

## Ratio pad
canvas.cd(2)

ref_line = ROOT.TLine()
ref_line.SetLineStyle(ROOT.kDotted)
ref_line.SetLineColor(ROOT.kBlack)
ref_line.SetLineWidth(2)
ref_line.DrawLine(ptcorr_bins[0], 1.0, ptcorr_bins[-2], 1.0)

for i, label in enumerate(["eta1", "eta2", "eta3"]):
    ratio_proj[label].SetTitle("")
    ratio_proj[label].SetStats(0)
    ratio_proj[label].GetXaxis().SetRangeUser(ptcorr_bins[0], ptcorr_bins[-2])
    CMS.cmsObjectDraw(ratio_proj[label], "", LineColor=PALETTE[i], LineWidth=2, LineStyle=ROOT.kSolid)

canvas.cd(2).RedrawAxis()

output_path = f"{WORKDIR}/MeasFakeRateV4/plots/{ERA}/{MEASURE}/compareHEMVeto.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
canvas.SaveAs(output_path)
logging.info(f"Plot saved to {output_path}")
