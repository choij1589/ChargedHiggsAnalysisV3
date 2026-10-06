#!/usr/bin/env python
"""Paper, preliminary (PAS) and private-work (thesis) fake-rate figures.

Same figure as plotFakeRate.py's data-driven fake rate (one curve per |eta| bin
against pT^corr); only the label changes between the modes:
  (default)       "CMS"                      -> plots/Paper/fakerate_{measure}_{era}.pdf
  --preliminary   "CMS" + "Preliminary"      -> plots/Paper/fakerate_{measure}_{era}_preliminary.pdf
  --private-work  "Private work (CMS data)"  -> plots/PW/fakerate_{measure}_{era}.pdf
"""
import argparse
import os
import sys

import ROOT
import cmsstyle as CMS

WORKDIR = os.environ.get("WORKDIR")
if not WORKDIR:
    raise RuntimeError("WORKDIR is not set")
sys.path.insert(0, f"{WORKDIR}/Common/Tools")
sys.path.insert(0, f"{WORKDIR}/MeasFakeRateV4/python")

from common import RUN2_ERAS, RUN3_ERAS, get_run_period
from plotter import EnergyInfo

ROOT.gROOT.SetBatch(True)

ERAS = RUN2_ERAS + RUN3_ERAS
# The paper (and PAS) shows one representative era; the thesis shows them all.
PAPER_ERAS = ["2018"]
MEASURES = ["electron", "muon"]

PALETTE = [
    ROOT.TColor.GetColor("#5790fc"),
    ROOT.TColor.GetColor("#f89c20"),
    ROOT.TColor.GetColor("#e42536"),
]

BINNING = {
    "muon": {
        "ptcorr": [10., 12., 14., 17., 20., 30., 50., 100., 200.],
        "abseta": [0., 0.9, 1.6, 2.4],
        "title": "Misidentification rate (#mu)",
        "eta_label": "|#eta|",
    },
    "electron": {
        "ptcorr": [15., 17., 20., 25., 35., 50., 100., 200.],
        "abseta": [0., 0.8, 1.479, 2.5],
        "title": "Misidentification rate (e)",
        "eta_label": "|#eta_{SC}|",
    },
}
X_MAX = 100.

# cmsstyle hardcodes the in-frame offset of the CMS block at 3.5% of the frame
# from its top-left corner, which puts the text on the axis ticks, so the paper
# and preliminary figures place "CMS" (and "Preliminary" under it) themselves.
CMS_LABEL_POS = (0.145, 0.875)
CMS_LABEL_SIZE = 0.058
PRELIMINARY_TEXT = "Preliminary"
PRELIMINARY_SUFFIX = "_preliminary"
# Thesis figures: non-approved, so "Private work (CMS data)" replaces the CMS
# mark in the extra-text style, placed by cmsstyle itself at iPos=11.
PRIVATE_WORK_TEXT = "Private work (CMS data)"

# The three eta curves all sit below 0.6 while the axis runs to 1, so the whole
# top-right quadrant is free for the legend.
LEGEND = (0.50, 0.575, 0.95, 0.825)
LEGEND_TEXT_SIZE = 0.050

# One marker shape for every eta bin -- the colours already separate them.
MARKER_STYLE = ROOT.kFullCircle
MARKER_SIZE = 1.2
LINE_WIDTH = 3


def configure_cms_style(mode):
    """cmsstyle's CMS text is global state: set it per mode, before the canvas."""
    CMS.SetExtraText("")
    if mode == "private-work":
        CMS.SetCmsText(PRIVATE_WORK_TEXT, font=52, size=0.75 * 0.76)
    else:
        # Drawn by draw_cms_label(), not by cmsstyle's CMS_lumi().
        CMS.SetCmsText("")


def restore_cms_style():
    CMS.SetCmsText("CMS", font=61, size=0.75)
    CMS.SetExtraText("")


def draw_cms_label(mode):
    """Draw the CMS block at CMS_LABEL_POS, top-left aligned inside the frame."""
    if mode == "private-work":
        return
    CMS.drawText("CMS", posX=CMS_LABEL_POS[0], posY=CMS_LABEL_POS[1],
                 font=61, align=13, size=CMS_LABEL_SIZE)
    if mode == "preliminary":
        # Same relative offset and size ratio cmsstyle uses between the two.
        CMS.drawText(PRELIMINARY_TEXT, posX=CMS_LABEL_POS[0],
                     posY=CMS_LABEL_POS[1] - 1.2 * CMS_LABEL_SIZE,
                     font=52, align=13, size=0.76 * CMS_LABEL_SIZE)


def build_output_path(era, measure, mode):
    root = "PW" if mode == "private-work" else "Paper"
    suffix = PRELIMINARY_SUFFIX if mode == "preliminary" else ""
    return f"{WORKDIR}/MeasFakeRateV4/plots/{root}/fakerate_{measure}_{era}{suffix}.pdf"


def load_fakerate(era, measure):
    path = f"{WORKDIR}/MeasFakeRateV4/results/{era}/ROOT/{measure}/fakerate.root"
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    f = ROOT.TFile.Open(path)
    h = f.Get("fake rate - (Central)")
    if not h:
        f.Close()
        raise RuntimeError(f"'fake rate - (Central)' not found in {path}")
    h.SetDirectory(0)
    f.Close()
    return h


def plot_fakerate(era, measure, mode, output_path):
    binning = BINNING[measure]
    ptcorr_bins, abseta_bins = binning["ptcorr"], binning["abseta"]
    h = load_fakerate(era, measure)

    projections = []
    for idx in range(len(abseta_bins) - 1):
        projection = h.ProjectionY(f"{era}_{measure}_eta{idx}", idx + 1, idx + 1)
        projection.SetDirectory(0)
        projection.SetTitle("")
        projection.SetStats(0)
        projections.append(projection)

    CMS.SetEnergy(EnergyInfo[get_run_period(era)])
    CMS.SetLumi(-1, run=f"{era}, Prescaled")
    configure_cms_style(mode)
    canvas = CMS.cmsCanvas(f"c_{era}_{measure}", ptcorr_bins[0], X_MAX, 0., 1.,
                           "p_{T}^{corr} [GeV]", binning["title"],
                           square=False, iPos=11, extraSpace=0.015)
    CMS.GetCmsCanvasHist(canvas).GetYaxis().SetMaxDigits(1)

    legend = CMS.cmsLeg(*LEGEND, textSize=LEGEND_TEXT_SIZE, columns=1)
    canvas.cd()
    for idx, projection in enumerate(projections):
        CMS.cmsObjectDraw(projection, "",
                          LineColor=PALETTE[idx], LineWidth=LINE_WIDTH, LineStyle=ROOT.kSolid,
                          MarkerColor=PALETTE[idx], MarkerStyle=MARKER_STYLE,
                          MarkerSize=MARKER_SIZE)
        CMS.addToLegend(legend, (projection,
                                 f"{abseta_bins[idx]:g} < {binning['eta_label']} < "
                                 f"{abseta_bins[idx + 1]:g}",
                                 "lep"))
    canvas.RedrawAxis()
    legend.Draw("same")
    draw_cms_label(mode)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # A landscape canvas otherwise comes out as a rotated (/Rotate 90) page, which
    # some viewers and PDF tools place off-centre on the media box.
    canvas.Print(output_path, "pdf Portrait")
    canvas.Close()
    restore_cms_style()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era", nargs="+", choices=ERAS, default=None,
                        help=f"eras to plot (default: {', '.join(PAPER_ERAS)} for the paper "
                             "and --preliminary, every era for --private-work)")
    parser.add_argument("--measure", nargs="+", choices=MEASURES, default=MEASURES,
                        help="lepton flavours to plot")
    label_group = parser.add_mutually_exclusive_group()
    label_group.add_argument("--preliminary", action="store_true",
                             help="'CMS Preliminary' (PAS), written next to the paper PDF "
                                  "as fakerate_{measure}_{era}_preliminary.pdf")
    label_group.add_argument("--private-work", dest="private_work", action="store_true",
                             help="thesis version: 'Private work (CMS data)', written to "
                                  "plots/PW/fakerate_{measure}_{era}.pdf")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the output paths without rendering")
    args = parser.parse_args()

    if args.private_work:
        mode = "private-work"
    elif args.preliminary:
        mode = "preliminary"
    else:
        mode = "paper"

    eras = args.era or (ERAS if mode == "private-work" else PAPER_ERAS)
    for era in eras:
        for measure in args.measure:
            output_path = build_output_path(era, measure, mode)
            if args.dry_run:
                print(f"{era} {measure} -> {output_path}")
                continue
            plot_fakerate(era, measure, mode, output_path)
            print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
