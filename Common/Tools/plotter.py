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

class BaseCanvas():
    """
    Base class for all canvas types with common configuration handling and utilities.

    This class provides:
    - Config management with defaults via config.get()
    - Palette selection (standard vs long)
    - Binning operations (fixed-width and variable)
    - Overflow handling
    - Axis range extraction
    - CMS style configuration
    - Legend creation
    - Channel text drawing
    - Normalization utilities
    """

    def __init__(self):
        """Base initialization - subclasses should call super().__init__()"""
        pass

    def _select_palette(self, num_hists):
        """
        Select appropriate color palette based on number of histograms.

        Args:
            num_hists (int): Number of histograms to display

        Returns:
            list: List of ROOT color codes
        """
        if num_hists > 6:
            return PALETTE_LONG[:num_hists]
        else:
            return PALETTE[:num_hists]

    def _apply_binning(self, hists, config):
        """
        Apply binning (fixed or variable) to histograms.

        Args:
            hists: Either a single histogram or dict of histograms
            config (dict): Configuration dictionary

        Returns:
            Same type as input (single hist or dict) with binning applied
        """
        is_dict = isinstance(hists, dict)
        hist_list = hists if not is_dict else hists.values()

        if "rebin" in config.keys():
            # Fixed-width rebinning
            for hist in (hist_list if is_dict else [hists]):
                hist.Rebin(config["rebin"])
            return hists
        elif len(config.get("xRange", [])) > 2:
            # Variable binning
            bins = array('d', config["xRange"])
            nbins = len(bins) - 1

            if is_dict:
                rebinned = {}
                for name, hist in hists.items():
                    rebinned[name] = hist.Rebin(nbins, hist.GetName()+"_rebin", bins)
                return rebinned
            else:
                return hists.Rebin(nbins, hists.GetName()+"_rebin", bins)
        else:
            # No rebinning
            return hists

    def _get_axis_range(self, config, hist=None):
        """
        Extract x-axis range from config, handling both simple and variable binning.

        Args:
            config (dict): Configuration dictionary
            hist (TH1, optional): Histogram to get default range from

        Returns:
            tuple: (xmin, xmax)
        """
        if hist is not None:
            default_range = [hist.GetXaxis().GetXmin(), hist.GetXaxis().GetXmax()]
        else:
            default_range = [0, 100]

        xRange = config.get("xRange", default_range)
        return xRange[0], xRange[-1]  # First and last elements

    def _set_overflow(self, hist, config):
        """
        Handle overflow bins by accumulating them into the last visible bin.
        Only applies to asymmetric ranges where abs(xmin) != abs(xmax).

        Args:
            hist (TH1): Histogram to process
            config (dict): Configuration dictionary with xRange
        """
        if not config.get("overflow", False):
            return

        xRange = config.get("xRange", [])
        if len(xRange) < 2:
            return

        xmin, xmax = xRange[0], xRange[-1]

        # Only apply overflow to asymmetric ranges
        if abs(xmin) == abs(xmax):
            return

        last_bin = hist.FindBin(xmax) - 1
        overflow, overflow_err2 = 0., 0.
        for idx in range(last_bin, hist.GetNbinsX()+1):
            overflow += hist.GetBinContent(idx)
            overflow_err2 += hist.GetBinError(idx)**2
        hist.SetBinContent(last_bin, hist.GetBinContent(last_bin) + overflow)
        hist.SetBinError(last_bin, ROOT.TMath.Sqrt(hist.GetBinError(last_bin)**2 + overflow_err2))

    def _configure_cms_style(self, config):
        """
        Configure CMS style parameters (energy, lumi, extra text).

        Args:
            config (dict): Configuration dictionary

        Returns:
            tuple: (lumiInfo, run) for canvas creation
        """
        if config.get("prescaled", False):
            lumiInfo = None
            run = f"{config['era']}, Prescaled"
        else:
            lumiInfo = LumiInfo[config["era"]]
            run = config["era"]

        CMS.SetEnergy(config.get("CoM", get_CoM_energy(config["era"])))
        CMS.SetLumi(lumiInfo, run=run)

        return lumiInfo, run

    def _create_legend(self, config):
        """
        Create CMS legend with default or custom parameters.

        Args:
            config (dict): Configuration dictionary

        Returns:
            TLegend: Configured legend object
        """
        if config.get("legend") is not None:
            return CMS.cmsLeg(*config["legend"])
        else:
            return CMS.cmsLeg(0.7, 0.89 - 0.05 * 7, 0.99, 0.89, textSize=0.04, columns=1)

    def _draw_channel_text(self, config):
        """
        Draw channel identification text on the canvas.

        Args:
            config (dict): Configuration dictionary
        """
        if "channel" not in config.keys():
            return

        posX = config.get("channelPosX", 0.2)
        posY = config.get("channelPosY", 0.7)
        font = config.get("channelFont", 61)
        align = config.get("channelAlign", 0)
        size = config.get("channelSize", 0.05)
        CMS.drawText(config['channel'], posX=posX, posY=posY, font=font, align=align, size=size)

    def _normalize_histogram(self, hist):
        """
        Normalize histogram to unit area (including overflow/underflow).

        Args:
            hist (TH1): Histogram to normalize
        """
        integral = hist.Integral(0, hist.GetNbinsX()+1)
        if integral > 0:
            hist.Scale(1.0/integral)

    def _get_y_range(self, hists, config):
        """
        Calculate y-axis range based on histogram content and config.

        Args:
            hists: List or dict of histograms
            config (dict): Configuration dictionary

        Returns:
            tuple: (ymin, ymax)
        """
        if config.get("yRange") is not None:
            return config["yRange"]

        hist_list = hists.values() if isinstance(hists, dict) else hists
        minimums = [h.GetMinimum() for h in hist_list]
        maximums = [h.GetMaximum() for h in hist_list]

        if config.get('logy', False):
            ymin = min(minimums) * 0.5
            if not ymin > 0:
                ymin = 1e-3
            ymax = max(maximums) * 100
        else:
            ymin = 0.
            ymax = max(maximums) * 2

        return ymin, ymax

class ComparisonCanvas(BaseCanvas):
    def __init__(self, incl, hists, config):
        super().__init__()
        self.config = config

        # Store histograms
        self.incl = incl    # Inclusive histogram, will plot on top
        self.hists = hists  # Histograms to be plotted as a stack

        # Select palette based on number of histograms
        self.palette = self._select_palette(len(self.hists))

        # Apply binning (fixed-width or variable)
        self.incl = self._apply_binning(self.incl, config)
        self.hists = self._apply_binning(self.hists, config)

        # Apply overflow handling if requested (NEW FEATURE for ComparisonCanvas)
        self._set_overflow(self.incl, config)
        for hist in self.hists.values():
            self._set_overflow(hist, config)

        # Create systematics histogram (sum of all backgrounds)
        self.systematics = None
        for hist in self.hists.values():
            if self.systematics is None:
                self.systematics = hist.Clone("syst")
            else:
                self.systematics.Add(hist)

        # Create ratio histogram
        self.ratio = self.incl.Clone("ratio")
        self.ratio.Divide(self.systematics)

        # Get axis ranges
        xmin, xmax = self._get_axis_range(config, self.systematics)

        # Calculate y-range with specific logic for ComparisonCanvas
        if config.get("yRange") is None:
            ymin = 0.
            ymax = self.systematics.GetMaximum() * 2
            if config.get('logy', False):
                ymin = self.systematics.GetMinimum() * 0.5
                if not ymin > 0:
                    ymin = 1e-3
                ymax = self.systematics.GetMaximum() * 1e3
        else:
            ymin, ymax = config.get("yRange")

        rmin, rmax = config.get("rRange", [0.5, 1.5])

        # Configure CMS style
        lumiInfo, run = self._configure_cms_style(config)
        CMS.SetExtraText("Preliminary")

        # Create canvas
        self.canv = CMS.cmsDiCanvas("", xmin, xmax,
                                        ymin, ymax,
                                        rmin, rmax,
                                        config.get("xTitle", ""),
                                        config.get("yTitle", "Events"),
                                        config.get("rTitle", "Data / Pred"),
                                        square=True,
                                        iPos=11,
                                        extraSpace=0)

        # Apply logarithmic x-axis if requested (BEFORE creating legend)
        if config.get('logx', False):
            self.canv.cd(1).SetLogx()
            self.canv.cd(2).SetLogx()

        hdf = CMS.GetCmsCanvasHist(self.canv.cd(1))
        hdf.GetYaxis().SetMaxDigits(config.get("maxDigits", 3))

        # Create legend AFTER setting log scales
        self.leg = self._create_legend(config)

    def update_y_scale(self, additional_hists=None):
        """
        Update y-axis scale to accommodate all histograms including optional additional ones.

        Args:
            additional_hists (list or dict, optional): Additional histograms to consider for scaling
        """
        # Skip if yRange is explicitly set in config
        if self.config.get("yRange") is not None:
            return

        # Find maximum among systematics (background stack)
        hist_max = self.systematics.GetMaximum()

        # Consider additional histograms (e.g., signals) if provided
        if additional_hists is not None:
            hist_list = additional_hists.values() if isinstance(additional_hists, dict) else additional_hists
            for hist in hist_list:
                hist_max = max(hist_max, hist.GetMaximum())

        # Calculate new y-axis range
        if self.config.get('logy', False):
            ymin = self.systematics.GetMinimum() * 0.5
            if not ymin > 0:
                ymin = 1e-3
            ymax = hist_max * 100
        else:
            ymin = 0.
            ymax = hist_max * 1.5

        # Update the canvas histogram's y-axis
        hdf = CMS.GetCmsCanvasHist(self.canv.cd(1))
        hdf.SetMinimum(ymin)
        hdf.SetMaximum(ymax)        

    def drawPadUp(self):
        self.canv.cd(1)
        self.hs = CMS.buildTHStack(list(self.hists.values()), self.palette, LineColor=-1, FillColor=-1)
        CMS.cmsObjectDraw(self.hs, "hist")
        CMS.cmsObjectDraw(self.systematics, "FE2", FillStyle=3004, LineWidth=0, FillColor=12, MarkerSize=0)
        CMS.cmsObjectDraw(self.incl, "PE", MarkerStyle=ROOT.kFullCircle, MarkerSize=1.0, MarkerColor=1)
        CMS.addToLegend(self.leg, (self.incl, self.incl.GetTitle(), "PE"))
        CMS.addToLegend(self.leg, *[(self.hists[name], name, "F") for name in self.hists.keys()])
        CMS.addToLegend(self.leg, (self.systematics, self.config.get("systSrc", "Stat+Syst"), " FE2"))

        # Draw channel text using base class method
        self._draw_channel_text(self.config)

        self.canv.cd(1).RedrawAxis()

    def drawSignals(self, signals):
        self.canv.cd(1)
        self.signals = {}
        self.sigleg = CMS.cmsLeg(0.38, 0.6, 0.6, 0.84, textSize=0.04, columns=1)

        # Process all signals
        for idx, (name, hist) in enumerate(signals.items()):
            # Clone with unique name to avoid ROOT histogram registry conflicts
            hist_clone = hist.Clone(f"signal_{name}")
            # Apply same binning as main histograms using base class method
            rebinned_hist = self._apply_binning(hist_clone, self.config)
            self.signals[name] = rebinned_hist

        # Update y-axis scale to accommodate signals
        self.update_y_scale(self.signals)

        # Now draw all signals
        for idx, (name, hist) in enumerate(self.signals.items()):
            hist.SetStats(0)
            CMS.cmsObjectDraw(hist, "hist", LineColor=ROOT.TColor.GetColorDark(self.palette[idx]), LineWidth=2, LineStyle=ROOT.kSolid, MarkerSize=0)
            CMS.cmsObjectDraw(hist, "LE", LineColor=ROOT.TColor.GetColorDark(self.palette[idx]), LineWidth=2, LineStyle=ROOT.kSolid, FillColor=ROOT.kWhite, MarkerSize=0)
            CMS.addToLegend(self.sigleg, (hist, name, "LE"))
        self.sigleg.Draw()
        self.canv.cd(1).RedrawAxis()

    def drawPadDown(self):
        self.canv.cd(2)

        # Draw reference line at y=1
        xmin, xmax = self._get_axis_range(self.config, self.systematics)
        ref_line = ROOT.TLine()
        ref_line.SetLineStyle(ROOT.kDotted)
        ref_line.SetLineColor(ROOT.kBlack)
        ref_line.SetLineWidth(2)
        ref_line.DrawLine(xmin, 1.0, xmax, 1.0)
        CMS.cmsObjectDraw(self.ratio, "FE2", FillStyle=3004, LineWidth=0, FillColor=12, MarkerSize=0)
        CMS.cmsObjectDraw(self.ratio, "PE", MarkerStyle=ROOT.kFullCircle, MarkerSize=1.0, MarkerColor=1)

        self.canv.cd(2).RedrawAxis()


class KinematicCanvas(BaseCanvas):
    def __init__(self, hists, config, ref=None):
        super().__init__()
        self.config = config
        self.ref = ref
        self.hists = hists

        # Select palette based on number of histograms
        self.palette = self._select_palette(len(self.hists))

        # Apply binning to histograms
        self.hists = self._apply_binning(self.hists, config)
        if ref is not None:
            self.ref = self._apply_binning(ref, config)

        # Apply overflow handling
        for hist in self.hists.values():
            self._set_overflow(hist, config)
        if ref is not None:
            self._set_overflow(ref, config)

        # Normalize histograms if requested
        if config.get("normalize", False):
            for hist in self.hists.values():
                self._normalize_histogram(hist)
            if ref is not None:
                self._normalize_histogram(ref)

        # Get axis ranges
        xmin, xmax = self._get_axis_range(config)
        ymin, ymax = self._get_y_range(self.hists, config)

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
        self.leg = self._create_legend(config)
    
    def drawPad(self):
        self.canv.cd()

        for idx, (name, hist) in enumerate(self.hists.items()):
            CMS.cmsObjectDraw(hist, "hist", LineColor=self.palette[idx], LineWidth=2, LineStyle=ROOT.kSolid)
            CMS.cmsObjectDraw(hist, "LE", LineColor=self.palette[idx], LineWidth=2, LineStyle=ROOT.kSolid, FillColor=ROOT.kWhite, MarkerSize=0)
            CMS.addToLegend(self.leg, (hist, name, "LE"))
        if self.ref is not None:
            CMS.cmsObjectDraw(self.ref, "PLE", FillColor=ROOT.kWhite, LineColor=ROOT.kBlack, LineWidth=2, MarkerColor=ROOT.kBlack, MarkerSize=1.0)
            CMS.addToLegend(self.leg, (self.ref, "LEGACY", "PLE"))

        # Draw channel text using base class method
        self._draw_channel_text(self.config)

        #self.leg.Draw()
        self.canv.RedrawAxis()


class KinematicCanvasWithRatio(BaseCanvas):
    def __init__(self, hists, config, ref=None):
        super().__init__()
        self.config = config
        self.ref = ref
        self.hists = hists

        # Select palette based on number of histograms
        self.palette = self._select_palette(len(self.hists))

        # Store the first histogram as reference for ratio calculation (before rebinning)
        first_key = list(self.hists.keys())[0]

        # Apply binning to histograms
        self.hists = self._apply_binning(self.hists, config)
        if ref is not None:
            self.ref = self._apply_binning(ref, config)

        # Update reference after rebinning
        self.reference_hist = self.hists[first_key]

        # Apply overflow handling
        for hist in self.hists.values():
            self._set_overflow(hist, config)
        if ref is not None:
            self._set_overflow(ref, config)

        # Normalize histograms if requested
        if config.get("normalize", False):
            for hist in self.hists.values():
                self._normalize_histogram(hist)
            if ref is not None:
                self._normalize_histogram(ref)

        # Get axis ranges
        xmin, xmax = self._get_axis_range(config)
        ymin, ymax = self._get_y_range(self.hists, config)
        rmin, rmax = config.get("rRange", [0.5, 1.5])

        # Configure CMS style
        lumiInfo, run = self._configure_cms_style(config)
        CMS.SetExtraText("Simulation Preliminary")

        # Create canvas
        self.canv = CMS.cmsDiCanvas("", xmin, xmax,
                                        ymin, ymax,
                                        rmin, rmax,
                                        config.get("xTitle", ""),
                                        config.get("yTitle", "Events"),
                                        config.get("rTitle", "Ratio"),
                                        square=True,
                                        iPos=11,
                                        extraSpace=0)

        # Apply log scales BEFORE creating legend
        if config.get('logy', False):
            self.canv.cd(1).SetLogy()
        if config.get('logx', False):
            self.canv.cd(1).SetLogx()
            self.canv.cd(2).SetLogx()

        if config.get("maxDigits") is not None:
            hdf = CMS.GetCmsCanvasHist(self.canv.cd(1))
            hdf.GetYaxis().SetMaxDigits(config["maxDigits"])

        # Create legend AFTER setting log scales
        self.leg = self._create_legend(config)

        # Create ratio histograms
        self.ratio_hists = {}
        for name, hist in self.hists.items():
            ratio = hist.Clone(f"{name}_ratio")
            ratio.Divide(self.reference_hist)
            self.ratio_hists[name] = ratio
    
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

        # Draw channel text using base class method
        self._draw_channel_text(self.config)

        self.canv.cd(1).RedrawAxis()

    def drawPadDown(self):
        self.canv.cd(2)

        # Draw reference line at y=1
        xmin, xmax = self._get_axis_range(self.config)
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