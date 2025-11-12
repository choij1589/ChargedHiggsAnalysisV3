#!/usr/bin/env python

import os
import sys
import argparse
import json
import ROOT
ROOT.gROOT.SetBatch(True)
sys.path.insert(0, os.path.join(os.environ['WORKDIR'], 'Common', 'Tools'))
import cmsstyle as CMS
from plotter import BaseCanvas
from plotter import LumiInfo, PALETTE_LONG, get_CoM_energy

WORKDIR = os.environ['WORKDIR']


class KinematicCanvas(BaseCanvas):
    def __init__(self, hists_signals, hists_fakes, config, ref=None):
        super().__init__()
        self.config = config
        self.ref = ref
        self.hists_signals = hists_signals
        self.hists_fakes = hists_fakes

        # Select palette based on number of histograms
        self.palette = PALETTE_LONG

        # Apply binning to histograms
        self.hists_signals = self._apply_binning(self.hists_signals, config)
        self.hists_fakes = self._apply_binning(self.hists_fakes, config)

        # Apply overflow handling
        for hist in {**self.hists_signals, **self.hists_fakes}.values():
            self._set_overflow(hist, config)

        # Normalize histograms if requested
        if config.get("normalize", False):
            for hist in {**self.hists_signals, **self.hists_fakes}.values():
                self._normalize_histogram(hist)

        # Get axis ranges
        xmin, xmax = self._get_axis_range(config)
        ymin, ymax = self._get_y_range({**self.hists_signals, **self.hists_fakes}, config)

        # Configure CMS style
        lumiInfo, run = self._configure_cms_style(config)
        CMS.SetExtraText("Simulation Preliminary")

        # Create canvas
        self.canv = CMS.cmsCanvas("", xmin, xmax,
                                      ymin, ymax,
                                      config.get("xTitle", ""),
                                      config.get("yTitle", "Events"),
                                      square=True,
                                      iPos=11,
                                      extraSpace=0.)

        # Apply log scales BEFORE creating legend
        if config.get('logy', False):
            self.canv.SetLogy()
        if config.get('logx', False):
            self.canv.SetLogx()

        if config.get("maxDigits") is not None:
            hdf = CMS.GetCmsCanvasHist(self.canv)
            hdf.GetYaxis().SetMaxDigits(config["maxDigits"])

        # Create legend AFTER setting log scales
        self.leg_signal = CMS.cmsLeg(0.52, 0.89-0.05*len(self.hists_signals), 0.73, 0.89, textSize=0.02, columns=1)
        self.leg_fake = CMS.cmsLeg(0.73, 0.89-0.05*len(self.hists_fakes), 0.95, 0.89, textSize=0.02, columns=1)
    
    def drawPad(self):
        self.canv.cd()

        # Draw signal histograms
        for idx, (name, hist) in enumerate(self.hists_signals.items()):
            CMS.cmsObjectDraw(hist, "hist", LineColor=self.palette[idx], LineWidth=2, LineStyle=ROOT.kSolid)
            CMS.cmsObjectDraw(hist, "LE", LineColor=self.palette[idx], LineWidth=2, LineStyle=ROOT.kSolid, FillColor=ROOT.kWhite, MarkerSize=0)
            CMS.addToLegend(self.leg_signal, (hist, name, "LE"))

        # Draw fake histograms
        for idx, (name, hist) in enumerate(self.hists_fakes.items()):
            CMS.cmsObjectDraw(hist, "hist", LineColor=self.palette[len(self.hists_signals) + idx], LineWidth=2, LineStyle=ROOT.kSolid)
            CMS.cmsObjectDraw(hist, "LE", LineColor=self.palette[len(self.hists_signals) + idx], LineWidth=2, LineStyle=ROOT.kSolid, FillColor=ROOT.kWhite, MarkerSize=0)
            CMS.addToLegend(self.leg_fake, (hist, name, "LE"))

        # Draw channel text using base class method
        self._draw_channel_text(self.config)

        #self.leg.Draw()
        self.canv.RedrawAxis()


def load_histogram(era, channel, variable, mass_point):
    # We don't sum histograms, just load the histograms for specified mass point
    with ROOT.TFile.Open(f"{WORKDIR}/SKNanoOutput/SignalKinematics/Run3Mu/{era}/TTToHcToWAToMuMu-{mass_point}.root") as f:
        histSig = f.Get(f"{channel}/SignalPair/{variable}"); histSig.SetDirectory(0)
        histFake = f.Get(f"{channel}/FakePair/{variable}"); histFake.SetDirectory(0)
        return histSig, histFake
        
def get_signal_and_fake_hists(era, channel, variable, mass_points):
    """
    Plot SignalPair vs FakePair comparison for kinematic variable
    """
    workdir = os.environ['WORKDIR']

    # Load histograms
    hists_signals = {}
    hists_fakes = {}
    for mass_point in mass_points:
        h_sig, h_fake = load_histogram(era, channel, variable, mass_point)
        hists_signals[f"SP-{mass_point}"] = h_sig.Clone(f"SP-{mass_point}")
        hists_fakes[f"FP-{mass_point}"] = h_fake.Clone(f"FP-{mass_point}")
    return hists_signals, hists_fakes
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--era", required=True, type=str, help="Data era (e.g., 2017)")
    parser.add_argument("--channel", default="SR3Mu", type=str, help="Analysis channel")
    parser.add_argument("--variable", required=True, type=str, help="Variable to plot")
    parser.add_argument("--mass-points", default="MHc70_MA15,MHc100_MA60,MHc130_MA90,MHc160_MA155", type=str,
                       help="Comma-separated mass points to include")
    args = parser.parse_args()

    mass_points = args.mass_points.split(",")
    mass_points = [point.strip() for point in mass_points]
    
    # get config from configs/histkeys.json
    with open("configs/histkeys.json") as f:
        config = json.load(f)[args.variable]
    config["era"] = args.era
    config["maxDigits"] = 3
    config["CoM"] = get_CoM_energy(args.era)
     
    hists_signals, hists_fakes = get_signal_and_fake_hists(args.era, args.channel, args.variable, mass_points)

    canvas = KinematicCanvas(hists_signals, hists_fakes, config)
    canvas.drawPad()
    canvas.leg_signal.Draw()
    canvas.leg_fake.Draw()
    
    output_dir = f"plots/OSMuonSelection/{args.era}/{args.channel}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/{args.variable}.png"
    
    canvas.canv.SaveAs(output_path)
    
if __name__ == "__main__":
    sys.exit(main())