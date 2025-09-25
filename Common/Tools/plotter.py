import ROOT
import cmsstyle as CMS
from array import array
CMS.setCMSStyle()

LumiInfo = {    # /fb
    "2016preVFP": 19.5,
    "2016postVFP": 16.8,
    "2017": 41.5,
    "2018": 59.8,
    "2022": 7.9,
    "2022EE": 26.7,
    "2023": 17.8,
    "2023BPix": 9.5,
    "Run2": 138,  # 19.5 + 16.8 + 41.5 + 59.8
    "Run3": 62,   # 7.9 + 26.7 + 17.8 + 9.5
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

def get_era_list(era):
    """Convert Run2/Run3 to list of individual eras"""
    if era == "Run2":
        return ["2016preVFP", "2016postVFP", "2017", "2018"]
    elif era == "Run3":
        return ["2022", "2022EE", "2023", "2023BPix"]
    else:
        return [era]

def get_datastream(era):
    """Get appropriate datastream name for era"""
    if era in ["2016preVFP", "2016postVFP", "2017", "2018", "Run2"]:
        return "SingleMuon"
    elif era in ["2022", "2022EE", "2023", "2023BPix", "Run3"]:
        return "Muon"
    else:
        raise ValueError(f"Unknown era: {era}")

def get_CoM_energy(era):
    """Get center-of-mass energy for era"""
    if era in ["2016preVFP", "2016postVFP", "2017", "2018", "Run2"]:
        return 13
    elif era in ["2022", "2022EE", "2023", "2023BPix", "Run3"]:
        return 13.6
    else:
        raise ValueError(f"Unknown era: {era}")

class ComparisonCanvas():
    def __init__(self, incl, hists, config):
        super().__init__()
        self.config = config

        # store histogram and config
        self.incl = incl    # Inclusive histogram, will plot on top
        self.hists = hists  # Histograms to be plotted as a stack
        self.palette = []
        if len(self.hists) > 6:
            self.palette = PALETTE_LONG[:len(self.hists)]
        else:
            self.palette = PALETTE[:len(self.hists)]
        
        # binning
        self.incl.Rebin(config.get("rebin", 1))
        for hist in self.hists.values():
            hist.Rebin(config.get("rebin", 1))

        # make necessary duplicates
        self.systematics = None
        for hist in self.hists.values():
            if self.systematics is None: 
                self.systematics = hist.Clone("syst")
            else: 
                self.systematics.Add(hist)

        # make ratio histogram
        self.ratio = self.incl.Clone("ratio")
        self.ratio.Divide(self.systematics)

        # axis range
        xmin, xmax = config.get("xRange", [self.systematics.GetXaxis().GetXmin(), self.systematics.GetXaxis().GetXmax()])
        if config.get("yRange") is None:
            ymin = 0.
            ymax = self.systematics.GetMaximum()*2
            if config.get('logy', False):
                ymin = self.systematics.GetMinimum()*0.5
                if not ymin > 0: ymin = 1e-3
                ymax = self.systematics.GetMaximum()*1e3
        else:
            ymin, ymax = config.get("yRange")
        rmin, rmax = config.get("rRange", [0.5, 1.5])

        # Override lumi info if prescaled
        if config.get("prescaled", False):
            lumiInfo = None
            run = f"{config['era']}, Prescaled"
        else:
            lumiInfo = LumiInfo[config["era"]]
            run = config["era"]

        # Default settings
        CMS.SetEnergy(config.get("CoM", 13))
        CMS.SetLumi(lumiInfo, run=run)
        CMS.SetExtraText("Preliminary")
        self.canv = CMS.cmsDiCanvas("", xmin, xmax, 
                                        ymin, ymax, 
                                        rmin, rmax, 
                                        config.get("xTitle", ""), 
                                        config.get("yTitle", "Events"), 
                                        config.get("rTitle", "Data / Pred"), 
                                        square=True, 
                                        iPos=11, 
                                        extraSpace=0)
        hdf = CMS.GetCmsCanvasHist(self.canv.cd(1))
        hdf.GetYaxis().SetMaxDigits(config.get("maxDigits", 3))

        if config.get("legend") is not None:
            self.leg = CMS.cmsLeg(*config["legend"])
        else:
            self.leg = CMS.cmsLeg(0.7, 0.89 - 0.05 * 7, 0.99, 0.89, textSize=0.04, columns=1)        

    def drawPadUp(self):
        self.canv.cd(1)
        self.hs = CMS.buildTHStack(list(self.hists.values()), self.palette, LineColor=-1, FillColor=-1)
        CMS.cmsObjectDraw(self.hs, "hist")
        CMS.cmsObjectDraw(self.systematics, "FE2", FillStyle=3004, LineWidth=0, FillColor=12, MarkerSize=0)
        CMS.cmsObjectDraw(self.incl, "PE", MarkerStyle=ROOT.kFullCircle, MarkerSize=1.0, MarkerColor=1)
        CMS.addToLegend(self.leg, (self.incl, self.incl.GetTitle(), "PE"))
        CMS.addToLegend(self.leg, *[(self.hists[name], name, "F") for name in self.hists.keys()])
        CMS.addToLegend(self.leg, (self.systematics, self.config.get("systSrc", "Stat+Syst"), " FE2"))
        
        if "channel" in self.config.keys():
            CMS.drawText(self.config['channel'], posX=0.2, posY=0.7, font=61, align=0, size=0.05)
            
        self.canv.cd(1).RedrawAxis()
        
    def drawSignals(self, signals):
        self.canv.cd(1)
        self.signals = signals
        self.sigleg = CMS.cmsLeg(0.38, 0.6, 0.6, 0.84, textSize=0.04, columns=1)
        for idx, (name, hist) in enumerate(self.signals.items()):
            hist.Rebin(self.config.get("rebin", 1))
            hist.SetStats(0)
            CMS.cmsObjectDraw(hist, "hist", LineColor=ROOT.TColor.GetColorDark(self.palette[idx]), LineWidth=2, LineStyle=ROOT.kSolid, MarkerSize=0)
            CMS.cmsObjectDraw(hist, "LE", LineColor=ROOT.TColor.GetColorDark(self.palette[idx]), LineWidth=2, LineStyle=ROOT.kSolid, FillColor=ROOT.kWhite, MarkerSize=0)
            CMS.addToLegend(self.sigleg, (hist, name, "LE"))
        self.canv.cd(1).RedrawAxis()        

    def drawPadDown(self):
        self.canv.cd(2)
        
        # Draw reference line at y=1
        xmin, xmax = self.config.get("xRange", [self.systematics.GetXaxis().GetXmin(), self.systematics.GetXaxis().GetXmax()])
        ref_line = ROOT.TLine()
        ref_line.SetLineStyle(ROOT.kDotted)
        ref_line.SetLineColor(ROOT.kBlack) 
        ref_line.SetLineWidth(2)
        ref_line.DrawLine(xmin, 1.0, xmax, 1.0)
        CMS.cmsObjectDraw(self.ratio, "FE2", FillStyle=3004, LineWidth=0, FillColor=12, MarkerSize=0)
        CMS.cmsObjectDraw(self.ratio, "PE", MarkerStyle=ROOT.kFullCircle, MarkerSize=1.0, MarkerColor=1)
        
        self.canv.cd(2).RedrawAxis()


class KinematicCanvas():
    def __init__(self, hists, config, ref=None):
        super().__init__()
        self.config = config
        self.ref = ref
        self.hists = hists
        self.palette = []
        if len(self.hists) > 6:
            self.palette = PALETTE_LONG[:len(self.hists)]
        else:
            self.palette = PALETTE[:len(self.hists)]

        # binning
        for hist in self.hists.values():
            hist.Rebin(config.get("rebin", 1))
        if ref is not None:
            ref.Rebin(config.get("rebin", 1))
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

        # Override lumi info if prescaled
        if "prescaled" in config.keys() and config["prescaled"]:
            lumiInfo = None
            run = f"{config['era']}, Prescaled"
        else:
            lumiInfo = LumiInfo[config["era"]]
            run = config["era"]

        # Default settings
        CMS.SetEnergy(config["CoM"])
        CMS.SetLumi(lumiInfo, run=run)
        CMS.SetExtraText("Simulation Preliminary")
        self.canv = CMS.cmsCanvas("", xmin, xmax, 
                                      ymin, ymax, 
                                      config["xTitle"], 
                                      config["yTitle"], 
                                      square=True, 
                                      iPos=11, 
                                      extraSpace=0.)
        if "logy" in config.keys() and config['logy']:
            self.canv.SetLogy()
        if "maxDigits" in config.keys():
            hdf = CMS.GetCmsCanvasHist(self.canv)
            hdf.GetYaxis().SetMaxDigits(config["maxDigits"])
        
        if "legend" in config.keys():
            self.leg = CMS.cmsLeg(*config["legend"])
        else:
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
            CMS.cmsObjectDraw(hist, "hist", LineColor=self.palette[idx], LineWidth=2, LineStyle=ROOT.kSolid)
            CMS.cmsObjectDraw(hist, "LE", LineColor=self.palette[idx], LineWidth=2, LineStyle=ROOT.kSolid, FillColor=ROOT.kWhite, MarkerSize=0)
            CMS.addToLegend(self.leg, (hist, name, "LE"))
        if self.ref is not None:
            CMS.cmsObjectDraw(self.ref, "PLE", FillColor=ROOT.kWhite, LineColor=ROOT.kBlack, LineWidth=2, MarkerColor=ROOT.kBlack, MarkerSize=1.0)
            CMS.addToLegend(self.leg, (self.ref, "LEGACY", "PLE"))
        #self.leg.Draw()
        self.canv.RedrawAxis()


class KinematicCanvasWithRatio():
    def __init__(self, hists, config, ref=None):
        super().__init__()
        self.config = config
        self.ref = ref
        self.hists = hists
        self.palette = []
        if len(self.hists) > 6:
            self.palette = PALETTE_LONG[:len(self.hists)]
        else:
            self.palette = PALETTE[:len(self.hists)]

        # Store the first histogram as reference for ratio calculation
        self.reference_hist = None
        first_key = list(self.hists.keys())[0]
        self.reference_hist = self.hists[first_key]

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
            if self.ref is not None:
                self.ref = self.ref.Rebin(len(bins)-1, self.ref.GetName()+"_rebin", bins)
            for name, hist in self.hists.items():
                self.hists[name] = hist.Rebin(len(bins)-1, hist.GetName()+"_rebin", bins)
            # Update reference after rebinning
            self.reference_hist = self.hists[first_key]
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

        # Ratio plot range
        rmin, rmax = config.get("rRange", [0.5, 1.5])

        # Override lumi info if prescaled
        if "prescaled" in config.keys() and config["prescaled"]:
            lumiInfo = None
            run = f"{config['era']}, Prescaled"
        else:
            lumiInfo = LumiInfo[config["era"]]
            run = config["era"]

        # Default settings
        CMS.SetEnergy(config["CoM"])
        CMS.SetLumi(lumiInfo, run=run)
        CMS.SetExtraText("Simulation Preliminary")
        self.canv = CMS.cmsDiCanvas("", xmin, xmax, 
                                        ymin, ymax, 
                                        rmin, rmax, 
                                        config["xTitle"], 
                                        config["yTitle"], 
                                        config.get("rTitle", "Ratio"), 
                                        square=True, 
                                        iPos=11, 
                                        extraSpace=0)
        if "logy" in config.keys() and config['logy']:
            self.canv.cd(1).SetLogy()
        if "maxDigits" in config.keys():
            hdf = CMS.GetCmsCanvasHist(self.canv.cd(1))
            hdf.GetYaxis().SetMaxDigits(config["maxDigits"])
        
        if "legend" in config.keys():
            self.leg = CMS.cmsLeg(*config["legend"])
        else:
            self.leg = CMS.cmsLeg(0.7, 0.89 - 0.05 * 7, 0.99, 0.89, textSize=0.04, columns=1)

        # Create ratio histograms
        self.ratio_hists = {}
        for name, hist in self.hists.items():
            ratio = hist.Clone(f"{name}_ratio")
            ratio.Divide(self.reference_hist)
            self.ratio_hists[name] = ratio

    def _set_overflow(self, hist):
        last_bin = hist.FindBin(self.config["xRange"][-1])-1
        overflow, overflow_err2 = 0., 0.
        for idx in range(last_bin, hist.GetNbinsX()+1):
            overflow += hist.GetBinContent(idx)
            overflow_err2 += hist.GetBinError(idx)**2
        hist.SetBinContent(last_bin, hist.GetBinContent(last_bin)+overflow)
        hist.SetBinError(last_bin, ROOT.TMath.Sqrt(hist.GetBinError(last_bin)**2 + overflow_err2))
    
    def drawPadUp(self):
        self.canv.cd(1)

        for idx, (name, hist) in enumerate(self.hists.items()):
            CMS.cmsObjectDraw(hist, "hist", LineColor=self.palette[idx], LineWidth=2, LineStyle=ROOT.kSolid)
            CMS.cmsObjectDraw(hist, "LE", LineColor=self.palette[idx], LineWidth=2, LineStyle=ROOT.kSolid, FillColor=ROOT.kWhite, MarkerSize=0)
            CMS.addToLegend(self.leg, (hist, name, "LE"))
        if self.ref is not None:
            CMS.cmsObjectDraw(self.ref, "PLE", FillColor=ROOT.kWhite, LineColor=ROOT.kBlack, LineWidth=2, MarkerColor=ROOT.kBlack, MarkerSize=1.0)
            CMS.addToLegend(self.leg, (self.ref, "LEGACY", "PLE"))

        self.leg.Draw()
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

        # Draw ratio histograms
        for idx, (name, ratio) in enumerate(self.ratio_hists.items()):
            CMS.cmsObjectDraw(ratio, "hist", LineColor=self.palette[idx], LineWidth=2, LineStyle=ROOT.kSolid)
            CMS.cmsObjectDraw(ratio, "LE", LineColor=self.palette[idx], LineWidth=2, LineStyle=ROOT.kSolid, FillColor=ROOT.kWhite, MarkerSize=0)
        
        self.canv.cd(2).RedrawAxis()