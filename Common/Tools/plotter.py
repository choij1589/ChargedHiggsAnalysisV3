import ROOT
import cmsstyle as CMS
from array import array

LumiInfo = {    # /fb
    "2016preVFP": 19.5,
    "2016postVFP": 16.8,
    "2017": 41.5,
    "2018": 59.8,
    "2022": 13.8,
    "2022EE": 20.9,
    "2023": 17.8,
    "2023BPix": 9.5,
}

PALETTE = [
    ROOT.TColor.GetColor("#5790fc"),
    ROOT.TColor.GetColor("#f89c20"),
    ROOT.TColor.GetColor("#e42536"),
    ROOT.TColor.GetColor("#964a8b"),
    ROOT.TColor.GetColor("#9c9ca1"),
    ROOT.TColor.GetColor("#7a21dd")
]
PALETTE_LONG = [
    ROOT.TColor.GetColor("#3f90da"),
    ROOT.TColor.GetColor("#ffa90e"),
    ROOT.TColor.GetColor("#bd1f01"),
    ROOT.TColor.GetColor("#94a4a2"),
    ROOT.TColor.GetColor("#832db6"),
    ROOT.TColor.GetColor("#a96b59"),
    ROOT.TColor.GetColor("#e76300"),
    ROOT.TColor.GetColor("#b9ac70"),
    ROOT.TColor.GetColor("#717581"),
    ROOT.TColor.GetColor("#92dadd")
]

class ComparisonCanvas():
    def __init__(self, incl, hists, config):
        super().__init__()
        self.config = config

        # store histogram and config
        self.incl = incl    # Inclusive histogram, will plot on top
        self.hists = hists  # Histograms to be plotted as a stack
        self.stack = ROOT.THStack("stack", "stack")

        # binning
        if "rebin" in config.keys():
            self.incl.Rebin(config["rebin"])
            for hist in self.hists.values():
                hist.Rebin(config["rebin"])

        # make necessary duplicates
        self.systematics = None
        for hist in self.hists.values():
            if self.systematics is None: self.systematics = hist.Clone("syst")
            else: self.systematics.Add(hist)

        self.ratio = self.incl.Clone("ratio")
        self.ratio.Divide(self.systematics)

        # axis range
        xmin, xmax = config["xRange"]
        if "yRange" not in config or config["yRange"] is None:
            ymin = 0.
            ymax = self.systematics.GetMaximum()*2
            if "logy" in config.keys() and config['logy']:
                ymin = self.systematics.GetMinimum()*0.5
                if not ymin > 0: ymin = 1e-3
                ymax = self.systematics.GetMaximum()*1e3
        else:
            ymin, ymax = config["yRange"]
        rmin, rmax = config["rRange"]

        # Default settings
        CMS.SetEnergy(13)
        CMS.SetLumi(LumiInfo[config["era"]])
        CMS.SetExtraText("Preliminary")
        self.canv = CMS.cmsDiCanvas("", xmin, xmax, ymin, ymax, rmin, rmax, config["xTitle"], config["yTitle"], config["rTitle"], square=True, iPos=11, extraSpace=0)
        if "maxDigits" in config.keys():
            hdf = CMS.GetcmsCanvasHist(self.canv.cd(1))
            hdf.GetYaxis().SetMaxDigits(config["maxDigits"])
        
        self.leg = CMS.cmsLeg(0.7, 0.89 - 0.05 * 7, 0.99, 0.89, textSize=0.04, columns=1)

    def drawPadUp(self):
        self.canv.cd(1)
        self.leg.AddEntry(self.incl, self.incl.GetTitle(), "PE")
        CMS.cmsDrawStack(self.stack, self.leg, self.hists)
        CMS.cmsDraw(self.systematics, "E2", fcolor=ROOT.kBlack, fstyle=3004, msize=0.)
        CMS.cmsDraw(self.incl, "PE", mcolor=ROOT.kBlack, msize=1.0)
        self.leg.AddEntry(self.systematics, "Stat+Syst", "FE2")

        self.canv.cd(1).RedrawAxis()

    def drawPadDown(self):
        self.canv.cd(2)
        
        # Draw reference line at y=1
        xmin, xmax = self.config["xRange"]
        ref_line = ROOT.TLine()
        ref_line.SetLineStyle(ROOT.kDotted)
        ref_line.SetLineColor(ROOT.kBlack) 
        ref_line.SetLineWidth(2)
        ref_line.DrawLine(xmin, 1.0, xmax, 1.0)

        CMS.cmsDraw(self.ratio, "PE2", fcolor=ROOT.kBlack, fstyle=3004, msize=1.0)
        
        self.canv.cd(2).RedrawAxis()


class KinematicCanvas():
    def __init__(self, hists, config, ref=None):
        super().__init__()
        self.config = config
        self.ref = ref
        self.hists = hists

        # binning
        if "rebin" in config.keys():
            for hist in self.hists.values():
                hist.Rebin(config["rebin"])
            if ref is not None:
                ref.Rebin(config["rebin"])
        elif len(config["xRange"]) > 2:
            # Create variable binning array from config
            bins = array('d', config["xRange"])
            # Rebin histograms with variable bin sizes
            self.ref = self.ref.Rebin(len(bins)-1, self.ref.GetName()+"_rebin", bins)
            for name, hist in self.hists.items():
                self.hists[name] = hist.Rebin(len(bins)-1, hist.GetName()+"_rebin", bins)
        else:
            pass

        # axis range
        xmin, xmax = config["xRange"]
        if config["overflow"]:
            if abs(xmin) == abs(xmax):
                # xRange is symmetric, no need to handle overflow
                pass
            else:
                # xRange is asymmetric, need to handle overflow
                for hist in self.hists.values():
                    self._set_overflow(hist)
                if ref is not None:
                    self._set_overflow(ref)
        
        # normalize histograms if requested
        if "normalize" in config and config["normalize"]:
            for hist in self.hists.values():
                integral = hist.Integral(0, hist.GetNbinsX()+1)
                if integral > 0:
                    hist.Scale(1.0/integral)
            if ref is not None:
                integral = ref.Integral(0, ref.GetNbinsX()+1)
                if integral > 0:
                    ref.Scale(1.0/integral)

        minimums = [hist.GetMinimum() for hist in self.hists.values()]
        maximums = [hist.GetMaximum() for hist in self.hists.values()]
        ymin, ymax = 0., max(maximums)*2
        if "logy" in config.keys() and config['logy']:
            ymin = min(minimums)*0.5
            if not ymin > 0: ymin = 1e-3
            ymax = max(maximums)*100
        if "yRange" in config.keys() and config["yRange"] is not None:
            ymin, ymax = config["yRange"]

        if len(self.hists) > 6:
            self.palette = PALETTE_LONG
        else:
            self.palette = PALETTE

        # Default settings
        CMS.SetEnergy(config["CoM"])
        CMS.SetLumi(LumiInfo[config["era"]])
        CMS.SetExtraText("Simulation Preliminary")
        self.canv = CMS.cmsCanvas("", xmin, xmax, ymin, ymax, config["xTitle"], config["yTitle"], square=True, iPos=11, extraSpace=0.)
        if "logy" in config.keys() and config['logy']:
            self.canv.SetLogy()
        if "maxDigits" in config.keys():
            hdf = CMS.GetcmsCanvasHist(self.canv)
            hdf.GetYaxis().SetMaxDigits(config["maxDigits"])
        
        self.leg = CMS.cmsLeg(0.7, 0.89 - 0.05 * 7, 0.99, 0.89, textSize=0.04, columns=1)

    def _set_overflow(self, hist):
        last_bin = hist.FindBin(self.config["xRange"][-1])-1
        overflow, overflow_err2 = 0., 0.
        for idx in range(last_bin, hist.GetNbinsX()+1):
            overflow += hist.GetBinContent(idx)
            overflow_err2 += hist.GetBinError(idx)**2
        hist.SetBinContent(last_bin, hist.GetBinContent(last_bin)+overflow)
        hist.SetBinError(last_bin, ROOT.TMath.Sqrt(hist.GetBinError(last_bin)**2 + overflow_err2))
    
    def drawPad(self):
        self.canv.cd()

        for idx, (name, hist) in enumerate(self.hists.items()):
            self.leg.AddEntry(hist, name, "LE")
            CMS.cmsDraw(hist, "hist", lcolor=self.palette[idx], lwidth=2, lstyle=ROOT.kSolid)
            CMS.cmsDraw(hist, "LE", lcolor=self.palette[idx], lwidth=2, lstyle=ROOT.kSolid, fcolor=ROOT.kWhite, msize=0)
        if self.ref is not None:
            self.leg.AddEntry(self.ref, "LEGACY", "PLE")
            CMS.cmsDraw(self.ref, "PLE", fcolor=ROOT.kWhite, lcolor=ROOT.kBlack, lwidth=2, mcolor=ROOT.kBlack, msize=1.)
        self.leg.Draw()
        self.canv.RedrawAxis()