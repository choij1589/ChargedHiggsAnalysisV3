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
eta_bins = []

# Determine if this is Run3
is_run3 = args.era in ["2022", "2022EE", "2023", "2023BPix"]

if args.measure == "muon":
    ptcorr_bins = [10., 15., 20., 30., 50., 100.]
    if is_run3:
        eta_bins = [-2.4, -1.6, -0.9, 0., 0.9, 1.6, 2.4]
    else:
        eta_bins = [0., 0.9, 1.6, 2.4]  # abseta_bins for Run2
elif args.measure == "electron":
    ptcorr_bins = [10., 15., 20., 25., 35., 50., 100.]
    if is_run3:
        eta_bins = [-2.5, -1.479, -0.8, 0., 0.8, 1.479, 2.5]
    else:
        eta_bins = [0., 0.8, 1.479, 2.5]  # abseta_bins for Run2
else:
    raise KeyError(f"Wrong measure {args.measure}")

eta_idx = -1
if is_run3:
    # Run3 eta bin mapping
    if args.etabin == "EEm":
        eta_idx = 1
    elif args.etabin == "EB2m":
        eta_idx = 2
    elif args.etabin == "EB1m":
        eta_idx = 3
    elif args.etabin == "EB1p":
        eta_idx = 4
    elif args.etabin == "EB2p":
        eta_idx = 5
    elif args.etabin == "EEp":
        eta_idx = 6
    else:
        raise KeyError(f"Wrong eta bin {args.etabin} for Run3")
else:
    # Run2 abseta bin mapping
    if args.etabin == "EB1":
        eta_idx = 1
    elif args.etabin == "EB2":
        eta_idx = 2
    elif args.etabin == "EE":
        eta_idx = 3
    else:
        raise KeyError(f"Wrong eta bin {args.etabin} for Run2")

logging.debug(f"ptcorr_bins: {ptcorr_bins}")
logging.debug(f"eta_bins: {eta_bins}")
logging.debug(f"is_run3: {is_run3}")
logging.debug(f"eta_idx: {eta_idx}")

## Prepare histograms
file_path = f"{WORKDIR}/MeasFakeRateV2/results/{args.era}/ROOT/{args.measure}/fakerate.root"
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
ratio_total = h_proj_central.Clone("total")
for bin in range(1, ratio_total.GetNbinsX()+1):
    error = ratio_total.GetBinError(bin) / ratio_total.GetBinContent(bin)
    ratio_total.SetBinContent(bin, 0.)
    ratio_total.SetBinError(bin, error)
    
# promptNorm
ratio_prUp = h_proj_central.Clone("PromptNormUp")
for bin in range(1, ratio_prUp.GetNbinsX()+1):
    content = (h_proj_prUp.GetBinContent(bin) - h_proj_central.GetBinContent(bin))/h_proj_central.GetBinContent(bin)
    ratio_prUp.SetBinContent(bin, content)

# promptNormDown
ratio_prDown = h_proj_central.Clone("PromptNormDown")
for bin in range(1, ratio_prDown.GetNbinsX()+1):
    content = (h_proj_prDown.GetBinContent(bin) - h_proj_central.GetBinContent(bin))/h_proj_central.GetBinContent(bin)
    ratio_prDown.SetBinContent(bin, content)
    
# JetPtUp
ratio_jetUp = h_proj_central.Clone("MotherJetPtUp")
for bin in range(1, ratio_jetUp.GetNbinsX()+1):
    content = (h_proj_jetUp.GetBinContent(bin) - h_proj_central.GetBinContent(bin))/h_proj_central.GetBinContent(bin)
    ratio_jetUp.SetBinContent(bin, content)
    
# JetPtDown
ratio_jetDown = h_proj_central.Clone("MotherJetPtDown")
for bin in range(1, ratio_jetDown.GetNbinsX()+1):
    content = (h_proj_jetDown.GetBinContent(bin) - h_proj_central.GetBinContent(bin))/h_proj_central.GetBinContent(bin)
    ratio_jetDown.SetBinContent(bin, content)
    
# RequireHeavyTag
ratio_btag = h_proj_central.Clone("RequireHeavyTag")
for bin in range(1, ratio_btag.GetNbinsX()+1):
    content = (h_proj_btag.GetBinContent(bin) - h_proj_central.GetBinContent(bin))/h_proj_central.GetBinContent(bin)
    ratio_btag.SetBinContent(bin, content)

## Set CMS style and energy
CoM = 13 if "201" in args.era else 13.6
CMS.SetEnergy(CoM)
CMS.SetLumi(-1, run=f"{args.era}, Prescaled")
CMS.SetExtraText("Preliminary")

## Create canvas with CMS style
canvas = CMS.cmsCanvas("", ptcorr_bins[0], 50., 
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

# Use CMS style drawing for systematic variations
CMS.cmsObjectDraw(ratio_total, "hist", LineColor=ROOT.kBlack, LineWidth=2)
CMS.cmsObjectDraw(ratio_prUp, "p&hist", MarkerStyle=8, MarkerSize=1.5, MarkerColor=PALETTE[0], LineColor=PALETTE[0], LineWidth=2)
CMS.cmsObjectDraw(ratio_prDown, "p&hist", MarkerStyle=8, MarkerSize=1.5, MarkerColor=PALETTE[0], LineColor=PALETTE[0], LineWidth=2)
CMS.cmsObjectDraw(ratio_jetUp, "p&hist", MarkerStyle=8, MarkerSize=1.5, MarkerColor=PALETTE[1], LineColor=PALETTE[1], LineWidth=2)
CMS.cmsObjectDraw(ratio_jetDown, "p&hist", MarkerStyle=8, MarkerSize=1.5, MarkerColor=PALETTE[1], LineColor=PALETTE[1], LineWidth=2)
CMS.cmsObjectDraw(ratio_btag, "p&hist", MarkerStyle=8, MarkerSize=1.5, MarkerColor=PALETTE[2], LineColor=PALETTE[2], LineWidth=2)

# Add to legend
CMS.addToLegend(legend, (ratio_prUp, "PromptNorm", "lep"))
CMS.addToLegend(legend, (ratio_jetUp, "MotherJetPt", "lep"))
CMS.addToLegend(legend, (ratio_btag, "RequireHeavyTag", "lep"))

canvas.RedrawAxis()
legend.Draw("same")

# Add additional text for eta range
text = ROOT.TLatex()
text.SetTextSize(0.06)
text.SetTextFont(42)
if is_run3:
    text.DrawLatexNDC(0.2, 0.8, f"{eta_bins[eta_idx-1]} < #eta < {eta_bins[eta_idx]}")
else:
    text.DrawLatexNDC(0.2, 0.8, f"{eta_bins[eta_idx-1]} < |#eta| < {eta_bins[eta_idx]}")

## Save the plot
output_path = f"{WORKDIR}/MeasFakeRateV2/plots/{args.era}/{args.measure}/systematics_{args.etabin}.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
canvas.SaveAs(output_path)
logging.info(f"Plot saved to {output_path}")