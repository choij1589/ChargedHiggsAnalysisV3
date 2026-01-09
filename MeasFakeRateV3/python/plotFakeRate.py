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
abseta_bins = []
if args.measure == "muon":
    ptcorr_bins = [10., 12., 14., 17., 20., 30., 50., 100., 200.]
    abseta_bins = [0., 0.9, 1.6, 2.4]
    title = "fake rate (#mu)"
elif args.measure == "electron":
    ptcorr_bins = [15., 17., 20., 25., 35., 50., 100., 200.]
    abseta_bins = [0., 0.8, 1.479, 2.5]
    title = "fake rate (e)"
else:
    raise KeyError(f"Wrong measure {args.measure}")
logging.debug(f"ptcorr_bins: {ptcorr_bins}")
logging.debug(f"abseta_bins: {abseta_bins}")
logging.debug(f"title: {title}")

def setHistStyle(projections, rate_syst=0.3):
    for projection in projections.values():
        projection.SetTitle("")
        projection.SetStats(0)
        projection.GetXaxis().SetTitle("p_{T}^{corr}")
        projection.GetXaxis().SetRangeUser(10., 200.)
        projection.GetYaxis().SetRangeUser(0., 1.)
        projection.GetYaxis().SetTitle(title)
        for bin in range(0, projection.GetNbinsX()+1):
            projection.SetBinError(bin, projection.GetBinContent(bin)*rate_syst)

## Get fakerate histogram
if args.isQCD:
    file_path = f"{WORKDIR}/MeasFakeRateV3/results/{args.era}/ROOT/{args.measure}/fakerate_qcd.root"
else:
    file_path = f"{WORKDIR}/MeasFakeRateV3/results/{args.era}/ROOT/{args.measure}/fakerate.root"

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
projections["eta1"] = h.ProjectionY(f"eta{str(abseta_bins[0])}to{str(abseta_bins[1])}", 1, 1)
projections["eta2"] = h.ProjectionY(f"eta{str(abseta_bins[1])}to{str(abseta_bins[2])}", 2, 2)
projections["eta3"] = h.ProjectionY(f"eta{str(abseta_bins[2])}to{str(abseta_bins[3])}", 3, 3)
setHistStyle(projections)

CoM = 13 if "201" in args.era else 13.6
CMS.SetEnergy(CoM)
CMS.SetLumi(-1,run=f"{args.era}, Prescaled")
CMS.SetExtraText("Preliminary")

canvas = CMS.cmsCanvas("", ptcorr_bins[0], 100., 
                             0., 1., 
                             "p_{T}^{corr}", 
                             title, 
                             square=False, 
                             iPos=11, 
                             extraSpace=0.015)
hdf = CMS.GetCmsCanvasHist(canvas)
hdf.GetYaxis().SetMaxDigits(1)

legend = CMS.cmsLeg(0.6, 0.89 - 0.05*5, 0.92, 0.85, textSize=0.04, columns=1)

canvas.cd()
CMS.cmsObjectDraw(projections["eta1"], "", LineColor=PALETTE[0], LineWidth=2, LineStyle=ROOT.kSolid)
CMS.cmsObjectDraw(projections["eta2"], "", LineColor=PALETTE[1], LineWidth=2, LineStyle=ROOT.kSolid)
CMS.cmsObjectDraw(projections["eta3"], "", LineColor=PALETTE[2], LineWidth=2, LineStyle=ROOT.kSolid)
CMS.addToLegend(legend, (projections["eta1"], f"{abseta_bins[0]} < |#eta| < {abseta_bins[1]}", "lep"))
CMS.addToLegend(legend, (projections["eta2"], f"{abseta_bins[1]} < |#eta| < {abseta_bins[2]}", "lep"))
CMS.addToLegend(legend, (projections["eta3"], f"{abseta_bins[2]} < |#eta| < {abseta_bins[3]}", "lep"))
canvas.RedrawAxis()
legend.Draw("same")

output_path = f"{WORKDIR}/MeasFakeRateV3/plots/{args.era}/{args.measure}/fakerate.png"
if args.isQCD:
    output_path = output_path.replace("fakerate", "fakerate_qcd")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
canvas.SaveAs(output_path)
