#!/usr/bin/env python
import os
import logging
import argparse
import ROOT
import cmsstyle as CMS

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--measure", required=True, type=str, help="electron / muon")
parser.add_argument("--isQCD", default=False, action="store_true", help="isQCD")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
if args.debug:
    logging.basicConfig(level=logging.DEBUG)

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
    title = "fake rate (#mu)"
elif args.measure == "electron":
    ptcorr_bins = [10., 15., 20., 25., 35., 50., 100.]
    if is_run3:
        eta_bins = [-2.5, -1.479, -0.8, 0., 0.8, 1.479, 2.5]
    else:
        eta_bins = [0., 0.8, 1.479, 2.5]  # abseta_bins for Run2
    title = "fake rate (e)"
else:
    raise KeyError(f"Wrong measure {args.measure}")
logging.debug(f"ptcorr_bins: {ptcorr_bins}")
logging.debug(f"eta_bins: {eta_bins}")
logging.debug(f"is_run3: {is_run3}")
logging.debug(f"title: {title}")

def setHistStyle(projections, rate_syst=0.3):
    for projection in projections.values():
        projection.SetTitle("")
        projection.SetStats(0)
        projection.GetXaxis().SetTitle("p_{T}^{corr}")
        projection.GetXaxis().SetRangeUser(10., 50.)
        projection.GetYaxis().SetRangeUser(0., 1.)
        projection.GetYaxis().SetTitle(title)
        for bin in range(0, projection.GetNbinsX()+1):
            projection.SetBinError(bin, projection.GetBinContent(bin)*rate_syst)

## Get fakerate histogram
if args.isQCD:
    file_path = f"{WORKDIR}/MeasFakeRateV2/results/{args.era}/ROOT/{args.measure}/fakerate_qcd.root"
else:
    file_path = f"{WORKDIR}/MeasFakeRateV2/results/{args.era}/ROOT/{args.measure}/fakerate.root"

assert os.path.exists(file_path), f"File not found: {file_path}"
f = ROOT.TFile(file_path)
if args.isQCD:
    h = f.Get("fake rate - (QCD_EMEnriched)") if args.measure == "electron" else f.Get("fake rate - (QCD_MuEnriched)")
else:
    h = f.Get("fake rate - (Central)")
h.SetDirectory(0)
f.Close()

## prepare projections
projections = {}
if is_run3:
    # Run3 has 6 eta bins
    projections["eta1"] = h.ProjectionY(f"eta{str(eta_bins[0])}to{str(eta_bins[1])}", 1, 1)  # EEm
    projections["eta2"] = h.ProjectionY(f"eta{str(eta_bins[1])}to{str(eta_bins[2])}", 2, 2)  # EB2m
    projections["eta3"] = h.ProjectionY(f"eta{str(eta_bins[2])}to{str(eta_bins[3])}", 3, 3)  # EB1m
    projections["eta4"] = h.ProjectionY(f"eta{str(eta_bins[3])}to{str(eta_bins[4])}", 4, 4)  # EB1p
    projections["eta5"] = h.ProjectionY(f"eta{str(eta_bins[4])}to{str(eta_bins[5])}", 5, 5)  # EB2p
    projections["eta6"] = h.ProjectionY(f"eta{str(eta_bins[5])}to{str(eta_bins[6])}", 6, 6)  # EEp
else:
    # Run2 has 3 abseta bins
    projections["eta1"] = h.ProjectionY(f"eta{str(eta_bins[0])}to{str(eta_bins[1])}", 1, 1)
    projections["eta2"] = h.ProjectionY(f"eta{str(eta_bins[1])}to{str(eta_bins[2])}", 2, 2)
    projections["eta3"] = h.ProjectionY(f"eta{str(eta_bins[2])}to{str(eta_bins[3])}", 3, 3)
setHistStyle(projections)

CoM = 13 if "201" in args.era else 13.6
CMS.SetEnergy(CoM)
CMS.SetLumi(-1,run=f"{args.era}, Prescaled")
CMS.SetExtraText("Preliminary")

canvas = CMS.cmsCanvas("", ptcorr_bins[0], 50., 
                             0., 1., 
                             "p_{T}^{corr}", 
                             title, 
                             square=True, 
                             iPos=11, 
                             extraSpace=0)
hdf = CMS.GetCmsCanvasHist(canvas)
hdf.GetYaxis().SetMaxDigits(1)

legend = CMS.cmsLeg(0.55, 0.89 - 0.05 * 7, 0.92, 0.85, textSize=0.04, columns=1)

canvas.cd()
if is_run3:
    # Run3: Draw all 6 eta bins
    CMS.cmsObjectDraw(projections["eta1"], "", LineColor=PALETTE[0], LineWidth=2, LineStyle=ROOT.kSolid)
    CMS.cmsObjectDraw(projections["eta2"], "", LineColor=PALETTE[1], LineWidth=2, LineStyle=ROOT.kSolid)
    CMS.cmsObjectDraw(projections["eta3"], "", LineColor=PALETTE[2], LineWidth=2, LineStyle=ROOT.kSolid)
    CMS.cmsObjectDraw(projections["eta4"], "", LineColor=PALETTE[3], LineWidth=2, LineStyle=ROOT.kSolid)
    CMS.cmsObjectDraw(projections["eta5"], "", LineColor=PALETTE[4], LineWidth=2, LineStyle=ROOT.kSolid)
    CMS.cmsObjectDraw(projections["eta6"], "", LineColor=PALETTE[5], LineWidth=2, LineStyle=ROOT.kSolid)
    CMS.addToLegend(legend, (projections["eta1"], f"{eta_bins[0]} < #eta < {eta_bins[1]}", "lep"))
    CMS.addToLegend(legend, (projections["eta2"], f"{eta_bins[1]} < #eta < {eta_bins[2]}", "lep"))
    CMS.addToLegend(legend, (projections["eta3"], f"{eta_bins[2]} < #eta < {eta_bins[3]}", "lep"))
    CMS.addToLegend(legend, (projections["eta4"], f"{eta_bins[3]} < #eta < {eta_bins[4]}", "lep"))
    CMS.addToLegend(legend, (projections["eta5"], f"{eta_bins[4]} < #eta < {eta_bins[5]}", "lep"))
    CMS.addToLegend(legend, (projections["eta6"], f"{eta_bins[5]} < #eta < {eta_bins[6]}", "lep"))
else:
    # Run2: Draw 3 abseta bins
    CMS.cmsObjectDraw(projections["eta1"], "", LineColor=PALETTE[0], LineWidth=2, LineStyle=ROOT.kSolid)
    CMS.cmsObjectDraw(projections["eta2"], "", LineColor=PALETTE[1], LineWidth=2, LineStyle=ROOT.kSolid)
    CMS.cmsObjectDraw(projections["eta3"], "", LineColor=PALETTE[2], LineWidth=2, LineStyle=ROOT.kSolid)
    CMS.addToLegend(legend, (projections["eta1"], f"{eta_bins[0]} < |#eta| < {eta_bins[1]}", "lep"))
    CMS.addToLegend(legend, (projections["eta2"], f"{eta_bins[1]} < |#eta| < {eta_bins[2]}", "lep"))
    CMS.addToLegend(legend, (projections["eta3"], f"{eta_bins[2]} < |#eta| < {eta_bins[3]}", "lep"))
canvas.RedrawAxis()
legend.Draw("same")

output_path = f"{WORKDIR}/MeasFakeRateV2/plots/{args.era}/{args.measure}/fakerate.png"
if args.isQCD:
    output_path = output_path.replace("fakerate", "fakerate_qcd")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
canvas.SaveAs(output_path)