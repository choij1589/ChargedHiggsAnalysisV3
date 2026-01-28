#!/usr/bin/env python
import os
import logging
import argparse
import ROOT
import cmsstyle as CMS

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--measure", required=True, type=str, help="electron / muon")
parser.add_argument("--isQCD", default=False, action="store_true", help="Plot from QCD samples")
parser.add_argument("--isTT", default=False, action="store_true", help="Plot from TTJJ_powheg")
parser.add_argument("--isMC", default=False, action="store_true", help="Plot from combined MC (QCD + TTJJ)")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

mc_flags = sum([args.isQCD, args.isTT, args.isMC])
if mc_flags > 1:
    raise ValueError("Only one of --isQCD, --isTT, --isMC can be specified")

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

def setHistStyle(projections):
    """Set histogram style, preserving statistical errors from the histogram."""
    for projection in projections.values():
        projection.SetTitle("")
        projection.SetStats(0)
        projection.GetXaxis().SetTitle("p_{T}^{corr}")
        projection.GetXaxis().SetRangeUser(ptcorr_bins[0], ptcorr_bins[-1])
        projection.GetYaxis().SetRangeUser(0., 1.)
        projection.GetYaxis().SetTitle(title)

def plot_fakerate(h, output_path):
    """Plot fake rate histogram and save to output_path."""
    ## prepare projections
    projections = {}
    projections["eta1"] = h.ProjectionY(f"eta{str(abseta_bins[0])}to{str(abseta_bins[1])}", 1, 1)
    projections["eta2"] = h.ProjectionY(f"eta{str(abseta_bins[1])}to{str(abseta_bins[2])}", 2, 2)
    projections["eta3"] = h.ProjectionY(f"eta{str(abseta_bins[2])}to{str(abseta_bins[3])}", 3, 3)
    setHistStyle(projections)

    CoM = 13 if "201" in args.era else 13.6
    CMS.SetEnergy(CoM)
    CMS.SetLumi(-1, run=f"{args.era}, Prescaled")
    CMS.SetExtraText("Preliminary")

    canvas = CMS.cmsCanvas("", ptcorr_bins[0], 100.,
                          0., 1.,
                          "p_{T}^{corr}",
                          title,
                          square=False,
                          iPos=0,
                          extraSpace=0.015)
    hdf = CMS.GetCmsCanvasHist(canvas)
    hdf.GetYaxis().SetMaxDigits(1)

    legend = CMS.cmsLeg(0.6, 0.89 - 0.05*3, 0.92, 0.89, textSize=0.04, columns=1)

    canvas.cd()
    eta_label = "|#eta_{SC}|" if args.measure == "electron" else "|#eta|"
    CMS.cmsObjectDraw(projections["eta1"], "", LineColor=PALETTE[0], LineWidth=2, LineStyle=ROOT.kSolid)
    CMS.cmsObjectDraw(projections["eta2"], "", LineColor=PALETTE[1], LineWidth=2, LineStyle=ROOT.kSolid)
    CMS.cmsObjectDraw(projections["eta3"], "", LineColor=PALETTE[2], LineWidth=2, LineStyle=ROOT.kSolid)
    CMS.addToLegend(legend, (projections["eta1"], f"{abseta_bins[0]} < {eta_label} < {abseta_bins[1]}", "lep"))
    CMS.addToLegend(legend, (projections["eta2"], f"{abseta_bins[1]} < {eta_label} < {abseta_bins[2]}", "lep"))
    CMS.addToLegend(legend, (projections["eta3"], f"{abseta_bins[2]} < {eta_label} < {abseta_bins[3]}", "lep"))
    canvas.RedrawAxis()
    legend.Draw("same")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    canvas.SaveAs(output_path)

## Get fakerate histogram
if args.isQCD:
    file_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/ROOT/{args.measure}/fakerate_qcd.root"
elif args.isTT:
    file_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/ROOT/{args.measure}/fakerate_TTJJ.root"
elif args.isMC:
    file_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/ROOT/{args.measure}/fakerate_MC.root"
else:
    file_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/ROOT/{args.measure}/fakerate.root"

assert os.path.exists(file_path), f"File not found: {file_path}"
f = ROOT.TFile(file_path)

# Determine what to plot
if args.isMC:
    # Plot QCD samples and TT separately
    base_output = f"{WORKDIR}/MeasFakeRateV4/plots/{args.era}/{args.measure}"
    os.makedirs(base_output, exist_ok=True)

    if args.measure == "electron":
        h_qcd_em = f.Get("fake rate - (QCD_EMEnriched)")
        h_qcd_em.SetDirectory(0)
        h_qcd_bc = f.Get("fake rate - (QCD_bcToE)")
        h_qcd_bc.SetDirectory(0)
        h_tt = f.Get("fake rate - (TT)")
        h_tt.SetDirectory(0)
        f.Close()

        plot_fakerate(h_qcd_em, f"{base_output}/fakerate_MC_QCD_EMEnriched.png")
        plot_fakerate(h_qcd_bc, f"{base_output}/fakerate_MC_QCD_bcToE.png")
        plot_fakerate(h_tt, f"{base_output}/fakerate_MC_TT.png")
    else:
        h_qcd = f.Get("fake rate - (QCD_MuEnriched)")
        h_qcd.SetDirectory(0)
        h_tt = f.Get("fake rate - (TT)")
        h_tt.SetDirectory(0)
        f.Close()

        plot_fakerate(h_qcd, f"{base_output}/fakerate_MC_QCD_MuEnriched.png")
        plot_fakerate(h_tt, f"{base_output}/fakerate_MC_TT.png")
elif args.isQCD:
    h = f.Get("fake rate - (QCD_EMEnriched)") if args.measure == "electron" else f.Get("fake rate - (QCD_MuEnriched)")
    h.SetDirectory(0)
    f.Close()
    output_path = f"{WORKDIR}/MeasFakeRateV4/plots/{args.era}/{args.measure}/fakerate_qcd.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_fakerate(h, output_path)
elif args.isTT:
    h = f.Get("fake rate - (TT)")
    h.SetDirectory(0)
    f.Close()
    output_path = f"{WORKDIR}/MeasFakeRateV4/plots/{args.era}/{args.measure}/fakerate_TTJJ.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_fakerate(h, output_path)
else:
    h = f.Get("fake rate - (Central)")
    h.SetDirectory(0)
    f.Close()
    output_path = f"{WORKDIR}/MeasFakeRateV4/plots/{args.era}/{args.measure}/fakerate.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plot_fakerate(h, output_path)
