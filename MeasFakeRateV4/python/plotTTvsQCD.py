#!/usr/bin/env python
import os
import logging
import argparse
import ROOT
import cmsstyle as CMS
ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--measure", required=True, type=str, help="electron / muon")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

PALETTE = [
    ROOT.TColor.GetColor("#5790fc"),
    ROOT.TColor.GetColor("#f89c20"),
    ROOT.TColor.GetColor("#e42536"),
    ROOT.TColor.GetColor("#964a8b"),
    ROOT.TColor.GetColor("#9c9ca1"),
    ROOT.TColor.GetColor("#7a21dd")
]

WORKDIR = os.environ['WORKDIR']

if args.measure == "muon":
    ptcorr_bins = [10., 12., 14., 17., 20., 30., 50., 100., 200.]
    abseta_bins = [0., 0.9, 1.6, 2.4]
    title = "fake rate (#mu)"
    qcd_name = "fake rate - (QCD_MuEnriched)"
elif args.measure == "electron":
    ptcorr_bins = [15., 17., 20., 25., 35., 50., 100., 200.]
    abseta_bins = [0., 0.8, 1.479, 2.5]
    title = "fake rate (e)"
    qcd_name = "fake rate - (QCD_EMEnriched)"
else:
    raise KeyError(f"Wrong measure {args.measure}")

eta_label = "|#eta_{SC}|" if args.measure == "electron" else "|#eta|"

## Load histograms
file_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/ROOT/{args.measure}/fakerate_MC.root"
assert os.path.exists(file_path), f"File not found: {file_path}"
f = ROOT.TFile.Open(file_path)
h_qcd = f.Get(qcd_name); h_qcd.SetDirectory(0)
h_tt  = f.Get("fake rate - (TT)"); h_tt.SetDirectory(0)
f.Close()

## Project per eta bin
qcd_proj = []
tt_proj  = []
ratios   = []
for eta_idx in range(1, 4):
    qcd_p = h_qcd.ProjectionY(f"qcd_eta{eta_idx}", eta_idx, eta_idx)
    tt_p  = h_tt.ProjectionY(f"tt_eta{eta_idx}",  eta_idx, eta_idx)
    for p in (qcd_p, tt_p):
        p.SetTitle("")
        p.SetStats(0)
    r = tt_p.Clone(f"ratio_eta{eta_idx}")
    r.Divide(qcd_p)
    qcd_proj.append(qcd_p)
    tt_proj.append(tt_p)
    ratios.append(r)

## Auto-scale y-axis to data
all_proj = qcd_proj + tt_proj
ymax = max(h.GetMaximum() for h in all_proj) * 1.4

## CMS style
CoM = 13 if "201" in args.era else 13.6
CMS.SetEnergy(CoM)
CMS.SetLumi(-1, run=f"{args.era}, Prescaled")
CMS.SetExtraText("Preliminary")

## Two-pad canvas
canvas = CMS.cmsDiCanvas("", ptcorr_bins[0], 100.,
                         0., ymax,
                         0.5, 1.5,
                         "p_{T}^{corr}",
                         title,
                         "TT / QCD",
                         square=False,
                         iPos=0,
                         extraSpace=0.01)

hdf = CMS.GetCmsCanvasHist(canvas.cd(1))
hdf.GetYaxis().SetMaxDigits(1)

legend = CMS.cmsLeg(0.38, 0.89 - 0.07*3, 0.92, 0.89, textSize=0.054, columns=2)

## Upper pad
canvas.cd(1)
for idx in range(3):
    CMS.cmsObjectDraw(qcd_proj[idx], "", LineColor=PALETTE[idx], LineWidth=2, LineStyle=ROOT.kSolid)
    CMS.cmsObjectDraw(tt_proj[idx],  "", LineColor=PALETTE[idx], LineWidth=2, LineStyle=ROOT.kDashed)
    eta_range = f"{abseta_bins[idx]} < {eta_label} < {abseta_bins[idx+1]}"
    CMS.addToLegend(legend, (qcd_proj[idx], f"QCD {eta_range}", "l"))
    CMS.addToLegend(legend, (tt_proj[idx],  f"TT {eta_range}",  "l"))
canvas.cd(1).RedrawAxis()
legend.Draw("same")

## Lower pad
canvas.cd(2)
ref_line = ROOT.TLine()
ref_line.SetLineStyle(ROOT.kDotted)
ref_line.SetLineColor(ROOT.kBlack)
ref_line.SetLineWidth(2)
ref_line.DrawLine(ptcorr_bins[0], 1.0, ptcorr_bins[-1], 1.0)
for idx in range(3):
    CMS.cmsObjectDraw(ratios[idx], "", LineColor=PALETTE[idx], LineWidth=2, LineStyle=ROOT.kSolid)
canvas.cd(2).RedrawAxis()

## Save
output_path = f"{WORKDIR}/MeasFakeRateV4/plots/{args.era}/{args.measure}/fakerate_TTvsQCD.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
canvas.SaveAs(output_path)
logging.info(f"Plot saved to {output_path}")
