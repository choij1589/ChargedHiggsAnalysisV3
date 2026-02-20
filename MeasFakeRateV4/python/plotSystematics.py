#!/usr/bin/env python
import os
import argparse
import logging
import ROOT
import cmsstyle as CMS
ROOT.gROOT.SetBatch(True)

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--measure", required=True, type=str, help="electron / muon")
parser.add_argument("--etabin", required=True, type=str, help="eta bin")
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
ptcorr_bins = []
abseta_bins = []
if args.measure == "muon":
    ptcorr_bins = [10., 12., 14., 17., 20., 30., 50., 100., 200.]
    abseta_bins = [0., 0.9, 1.6, 2.4]
elif args.measure == "electron":
    ptcorr_bins = [15., 17., 20., 25., 35., 50., 100., 200.]
    abseta_bins = [0., 0.8, 1.479, 2.5]
else:
    raise KeyError(f"Wrong measure {args.measure}")

eta_idx = -1
if args.etabin == "EB1":
    eta_idx = 1
elif args.etabin == "EB2":
    eta_idx = 2
elif args.etabin == "EE":
    eta_idx = 3
else:
    raise KeyError(f"Wrong eta bin {args.etabin}")

logging.debug(f"ptcorr_bins: {ptcorr_bins}")
logging.debug(f"abseta_bins: {abseta_bins}")
logging.debug(f"eta_idx: {eta_idx}")

## Prepare histograms
file_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/ROOT/{args.measure}/fakerate.root"
assert os.path.exists(file_path), f"File not found: {file_path}"
f = ROOT.TFile.Open(file_path)
h = f.Get("fake rate - (Central)"); h.SetDirectory(0)
h_prUp = f.Get("fake rate - (PromptNorm_Up)"); h_prUp.SetDirectory(0)
h_prDown = f.Get("fake rate - (PromptNorm_Down)"); h_prDown.SetDirectory(0)
h_jetUp = f.Get("fake rate - (MotherJetPt_Up)"); h_jetUp.SetDirectory(0)
h_jetDown = f.Get("fake rate - (MotherJetPt_Down)"); h_jetDown.SetDirectory(0)
h_btag = f.Get("fake rate - (RequireHeavyTag)"); h_btag.SetDirectory(0)
f.Close()

## Make projections
h_proj_central = h.ProjectionY("central", eta_idx, eta_idx)
h_proj_prUp = h_prUp.ProjectionY("prUp", eta_idx, eta_idx)
h_proj_prDown = h_prDown.ProjectionY("prDown", eta_idx, eta_idx)
h_proj_jetUp = h_jetUp.ProjectionY("jetUp", eta_idx, eta_idx)
h_proj_jetDown = h_jetDown.ProjectionY("jetDown", eta_idx, eta_idx)
h_proj_btag = h_btag.ProjectionY("btag", eta_idx, eta_idx)

## make ratio plots for each systematic sources
def make_ratio(h_syst, h_central, name):
    """Make ratio histogram (syst - central) / central.

    Error bars show the fractional statistical uncertainty of the systematic measurement.
    This properly reflects that:
    - For correlated systematics (e.g., PromptNorm): syst_err ~ central_err
    - For subset selections (e.g., RequireHeavyTag): syst_err > central_err due to fewer events
    """
    ratio = h_central.Clone(name)
    for bin in range(1, ratio.GetNbinsX()+1):
        central = h_central.GetBinContent(bin)
        syst = h_syst.GetBinContent(bin)
        syst_err = h_syst.GetBinError(bin)
        if central > 0:
            content = (syst - central) / central
            err = syst_err / central
        else:
            content = 0.
            err = 0.
        ratio.SetBinContent(bin, content)
        ratio.SetBinError(bin, err)
    return ratio

# promptNorm
ratio_prUp = make_ratio(h_proj_prUp, h_proj_central, "PromptNormUp")
ratio_prDown = make_ratio(h_proj_prDown, h_proj_central, "PromptNormDown")

# JetPt
ratio_jetUp = make_ratio(h_proj_jetUp, h_proj_central, "MotherJetPtUp")
ratio_jetDown = make_ratio(h_proj_jetDown, h_proj_central, "MotherJetPtDown")

# RequireHeavyTag
ratio_btag = make_ratio(h_proj_btag, h_proj_central, "RequireHeavyTag")

## Set CMS style and energy
CoM = 13 if "201" in args.era else 13.6
CMS.SetEnergy(CoM)
CMS.SetLumi(-1, run=f"{args.era}, Prescaled")
CMS.SetExtraText("Preliminary")

## Create canvas with CMS style
canvas = CMS.cmsCanvas("", ptcorr_bins[0], 100.,
                             -1.0, 1.0,
                             "p_{T}^{corr}",
                             "#Delta fr / fr",
                             square=False,
                             iPos=0,
                             extraSpace=0.01)
hdf = CMS.GetCmsCanvasHist(canvas)
hdf.GetYaxis().SetMaxDigits(1)
hdf.GetYaxis().CenterTitle()
hdf.GetYaxis().SetNdivisions(5)

# Create legend using CMS style
legend = CMS.cmsLeg(0.63, 0.89-0.05*3, 0.94, 0.89, textSize=0.04, columns=1)

## Draw
canvas.cd()

## Draw reference line at y = -0.5 and y = 0.5
ref_line = ROOT.TLine()
ref_line.SetLineStyle(ROOT.kDotted)
ref_line.SetLineColor(ROOT.kBlack)
ref_line.SetLineWidth(2)
ref_line.DrawLine(ptcorr_bins[0], -0.5, ptcorr_bins[-1], -0.5)
ref_line.DrawLine(ptcorr_bins[0], 0.5, ptcorr_bins[-1], 0.5)

# Set styles for systematic variations
def set_style(h, color, fill_style=3004):
    """Set histogram style for hatched error band and markers."""
    h.SetLineColor(color)
    h.SetLineWidth(2)
    h.SetMarkerColor(color)
    h.SetMarkerStyle(8)
    h.SetMarkerSize(1.5)
    h.SetFillColor(color)
    h.SetFillStyle(fill_style)  # 3004: 45 deg hatching, 3005: -45 deg, 3006: cross-hatch

set_style(ratio_prUp, PALETTE[0])
set_style(ratio_prDown, PALETTE[0])
set_style(ratio_jetUp, PALETTE[1])
set_style(ratio_jetDown, PALETTE[1])
set_style(ratio_btag, PALETTE[2])

# Draw all error bands first
ratio_prUp.Draw("e2 same")
ratio_prDown.Draw("e2 same")
ratio_jetUp.Draw("e2 same")
ratio_jetDown.Draw("e2 same")
ratio_btag.Draw("e2 same")

# Draw all markers on top (without error bars)
ratio_prUp.Draw("p same")
ratio_prDown.Draw("p same")
ratio_jetUp.Draw("p same")
ratio_jetDown.Draw("p same")
ratio_btag.Draw("p same")

# Add to legend
CMS.addToLegend(legend, (ratio_prUp, "PromptNorm", "fp"))
CMS.addToLegend(legend, (ratio_jetUp, "MotherJetPt", "fp"))
CMS.addToLegend(legend, (ratio_btag, "RequireHeavyTag", "fp"))

canvas.RedrawAxis()
legend.Draw("same")

# Add additional text for eta range
text = ROOT.TLatex()
text.SetTextSize(0.06)
text.SetTextFont(42)
eta_label = "|#eta_{SC}|" if args.measure == "electron" else "|#eta|"
text.DrawLatexNDC(0.2, 0.8, f"{abseta_bins[eta_idx-1]} < {eta_label} < {abseta_bins[eta_idx]}")

## Save the plot
output_path = f"{WORKDIR}/MeasFakeRateV4/plots/{args.era}/{args.measure}/systematics_{args.etabin}.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
canvas.SaveAs(output_path)
logging.info(f"Plot saved to {output_path}")
