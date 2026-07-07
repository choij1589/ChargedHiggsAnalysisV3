#!/usr/bin/env python3
"""Validate Run-period component templates before Combine production."""

import argparse
import json
import os
import re
import shutil
import sys
from collections import OrderedDict

import ROOT
import cmsstyle as CMS

ROOT.gROOT.SetBatch(True)

WORKDIR_FOR_IMPORTS = os.getenv("WORKDIR")
if WORKDIR_FOR_IMPORTS:
    sys.path.insert(0, f"{WORKDIR_FOR_IMPORTS}/Common/Tools")
else:
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Common", "Tools")))
from plotter import ComparisonCanvas, KinematicCanvas, PALETTE, PALETTE_LONG, get_CoM_energy, _LUMI_CONFIG  # noqa: E402


GROUP_ORDER = ["others", "conversion", "WZ", "ZZ", "ttW", "ttH", "tZq", "ttZ", "nonprompt"]
GROUP_COLORS = {
    "nonprompt": PALETTE_LONG[0],
    "WZ": PALETTE_LONG[1],
    "ZZ": PALETTE_LONG[2],
    "ttW": PALETTE_LONG[3],
    "ttZ": PALETTE_LONG[4],
    "ttH": PALETTE_LONG[5],
    "tZq": PALETTE_LONG[6],
    "others": PALETTE_LONG[7],
    "conversion": PALETTE_LONG[8],
}
ERA_DECORRELATED_TOKENS = [
    "2016preVFP",
    "2016postVFP",
    "2017",
    "2018",
    "2022EE",
    "2022",
    "2023BPix",
    "2023",
]
RUN_PERIOD_ERAS = {
    "Run2": {"2016preVFP", "2016postVFP", "2017", "2018"},
    "Run3": {"2022", "2022EE", "2023", "2023BPix"},
}
POSTFIT_SUMMARY_LEGEND = [0.5, 0.62, 0.99, 0.89]
POSTFIT_SUMMARY_LEGEND_TEXT_SIZE = 0.035
SYST_UP_COLOR = PALETTE[5]
SYST_DOWN_COLOR = PALETTE[2]
SYST_BACKGROUND_ALPHA = 0.7


class ValidationComparisonCanvas(ComparisonCanvas):
    def _configure_cms_style(self, config):
        if config.get("era") == "All":
            run2_lumi = _LUMI_CONFIG["Run2"]["combined"]
            run3_lumi = _LUMI_CONFIG["Run3"]["combined"]
            CMS.SetLumi(None, run=f"Run 2+3, {run2_lumi}+{run3_lumi} fb^{{#minus1}}")
            CMS.SetEnergy(0, unit="13/13.6 TeV")
            return None, "Run 2+3"
        return super()._configure_cms_style(config)

    def drawPadUp(self):
        self._cd_main()
        self.hs = CMS.buildTHStack(list(self.hists.values()), self.palette, LineColor=-1, FillColor=-1)
        CMS.cmsObjectDraw(self.hs, "hist")
        CMS.cmsObjectDraw(self.systematics, "FE2", FillStyle=3004, LineWidth=0, FillColor=12, MarkerSize=0)

        signal = self.config.get("signalHist")
        if signal and signal.InheritsFrom("TH1") and signal.Integral() > 0:
            signal.SetTitle("Signal")
            signal.SetLineColor(ROOT.kBlack)
            signal.SetLineWidth(3)
            signal.SetLineStyle(ROOT.kSolid)
            signal.SetFillStyle(0)
            signal.SetMarkerSize(0)
            CMS.cmsObjectDraw(signal, "HIST SAME")
            self._signal_overlay = signal

        if not self.config.get("no_ratio", False):
            CMS.cmsObjectDraw(self.incl, "PE", MarkerStyle=ROOT.kFullCircle, MarkerSize=1.0, MarkerColor=1)

        entries = []
        if not self.config.get("no_ratio", False):
            entries.append((self.incl, self.incl.GetTitle(), "PE"))
        entries.extend((self.hists[name], name, "F") for name in reversed(list(self.hists.keys())))

        if signal and signal.InheritsFrom("TH1") and signal.Integral() > 0:
            entries.append((signal, "Signal", "L"))
        syst_entry = (self.systematics, self.config.get("systSrc", "Stat+Syst"), " FE2")
        entries.append(syst_entry)

        CMS.addToLegend(self.leg, *entries)
        self._draw_channel_text(self.config)
        self._cd_main().RedrawAxis()

    def drawPadDown(self):
        if self.config.get("no_ratio", False):
            return
        self.canv.cd(2)

        xmin, xmax = self._get_axis_range(self.config, self.systematics)
        ref_line = ROOT.TLine()
        ref_line.SetLineStyle(ROOT.kDotted)
        ref_line.SetLineColor(ROOT.kBlack)
        ref_line.SetLineWidth(2)
        ref_line.DrawLine(xmin, 1.0, xmax, 1.0)

        ratio_band = getattr(self, "ratio_band", self.ratio)
        CMS.cmsObjectDraw(ratio_band, "FE2", FillStyle=3004, LineWidth=0, FillColor=12, MarkerSize=0)
        CMS.cmsObjectDraw(self.ratio, "PE", MarkerStyle=ROOT.kFullCircle, MarkerSize=1.0, MarkerColor=1)
        self.canv.cd(2).RedrawAxis()


class ValidationKinematicCanvas(KinematicCanvas):
    def _configure_cms_style(self, config):
        if config.get("era") == "All":
            run2_lumi = _LUMI_CONFIG["Run2"]["combined"]
            run3_lumi = _LUMI_CONFIG["Run3"]["combined"]
            CMS.SetLumi(None, run=f"Run 2+3, {run2_lumi}+{run3_lumi} fb^{{#minus1}}")
            CMS.SetEnergy(0, unit="13/13.6 TeV")
            return None, "Run 2+3"
        return super()._configure_cms_style(config)

    def __init__(self, hists, config, ref=None):
        self.config = config
        self.ref = ref
        self.hists = hists

        self.palette = self._select_palette(len(self.hists), config)
        self.hists = self._apply_binning(self.hists, config)
        if ref is not None:
            self.ref = self._apply_binning(ref, config)

        for hist in self.hists.values():
            self._set_overflow(hist, config)
        if self.ref is not None:
            self._set_overflow(self.ref, config)

        if config.get("normalize", False):
            for hist in self.hists.values():
                self._normalize_histogram(hist)
            if self.ref is not None:
                self._normalize_histogram(self.ref)

        xmin, xmax = self._get_axis_range(config)
        ymin, ymax = self._get_y_range(self.hists, config)

        self._configure_cms_style(config)
        CMS.SetExtraText("Preliminary")

        self.canv = CMS.cmsCanvas(
            "", xmin, xmax, ymin, ymax,
            config.get("xTitle", ""),
            config.get("yTitle", "Events"),
            square=True,
            iPos=config.get("iPos", 11),
            extraSpace=0.,
        )

        if config.get("logy", False):
            self.canv.SetLogy()
        if config.get("logx", False):
            self.canv.SetLogx()

        if config.get("maxDigits") is not None:
            hdf = CMS.GetCmsCanvasHist(self.canv)
            hdf.GetYaxis().SetMaxDigits(config["maxDigits"])

        self.leg = self._create_legend(config)


class SystematicStackCanvas(ValidationComparisonCanvas):
    def drawPadUp(self):
        self._cd_main()
        self.hs = CMS.buildTHStack(list(self.hists.values()), self.palette, LineColor=-1, FillColor=-1)
        for idx, hist in enumerate(self.hists.values()):
            hist.SetFillColorAlpha(self.palette[idx], self.config.get("backgroundAlpha", SYST_BACKGROUND_ALPHA))
        CMS.cmsObjectDraw(self.hs, "hist")
        CMS.cmsObjectDraw(self.systematics, "FE2", FillStyle=3004, LineWidth=0, FillColor=12, MarkerSize=0)

        variation_entries = []
        for hist, title, color in self.config.get("variationHists", []):
            CMS.cmsObjectDraw(hist, "HIST SAME", LineColor=color, LineWidth=3, LineStyle=ROOT.kSolid, MarkerSize=0)
            variation_entries.append((hist, title, "L"))

        entries = []
        entries.extend((self.hists[name], name, "F") for name in reversed(list(self.hists.keys())))
        entries.append((self.systematics, self.config.get("systSrc", "Stat+Syst"), " FE2"))
        entries.extend(variation_entries)
        CMS.addToLegend(self.leg, *entries)
        self._draw_channel_text(self.config)
        self._cd_main().RedrawAxis()

    def drawPadDown(self):
        if self.config.get("no_ratio", False):
            return
        self.canv.cd(2)

        xmin, xmax = self._get_axis_range(self.config, self.systematics)
        ref_line = ROOT.TLine()
        ref_line.SetLineStyle(ROOT.kDotted)
        ref_line.SetLineColor(ROOT.kBlack)
        ref_line.SetLineWidth(2)
        ref_line.DrawLine(xmin, 1.0, xmax, 1.0)

        ratio_band = getattr(self, "ratio_band", self.ratio)
        CMS.cmsObjectDraw(ratio_band, "FE2", FillStyle=3004, LineWidth=0, FillColor=12, MarkerSize=0)
        for ratio, _, color in self.config.get("variationRatioHists", []):
            CMS.cmsObjectDraw(ratio, "HIST SAME", LineColor=color, LineWidth=3, LineStyle=ROOT.kSolid, MarkerSize=0)
        self.canv.cd(2).RedrawAxis()


class SystematicLineCanvas(ValidationKinematicCanvas):
    def __init__(self, hists, ratio_hists, ratio_band, config):
        self.ratio_hists = ratio_hists
        self.ratio_band = ratio_band
        super().__init__(hists, config)

        xmin, xmax = self._get_axis_range(config)
        ymin, ymax = self._get_y_range(self.hists, config)
        rmin, rmax = config.get("rRange", [0.5, 1.5])
        self._configure_cms_style(config)
        CMS.SetExtraText("Preliminary")
        self.canv = CMS.cmsDiCanvas(
            "", xmin, xmax, ymin, ymax, rmin, rmax,
            config.get("xTitle", ""),
            config.get("yTitle", "Events"),
            config.get("rTitle", "Var. / Nom."),
            square=True,
            iPos=config.get("iPos", 11),
            extraSpace=0,
        )
        if config.get("logy", False):
            self.canv.cd(1).SetLogy()
        if config.get("logx", False):
            self.canv.cd(1).SetLogx()
            self.canv.cd(2).SetLogx()
        if config.get("maxDigits") is not None:
            hdf = CMS.GetCmsCanvasHist(self.canv.cd(1))
            hdf.GetYaxis().SetMaxDigits(config["maxDigits"])
        self.leg = self._create_legend(config)

    def drawPad(self):
        self.canv.cd(1)
        for idx, (name, hist) in enumerate(self.hists.items()):
            color = self.palette[idx]
            CMS.cmsObjectDraw(hist, "hist", LineColor=color, LineWidth=2, LineStyle=ROOT.kSolid, MarkerSize=0)
            CMS.cmsObjectDraw(hist, "LE", LineColor=color, LineWidth=2, LineStyle=ROOT.kSolid, FillColor=ROOT.kWhite, MarkerSize=0)
            CMS.addToLegend(self.leg, (hist, name, "LE"))
        self._draw_channel_text(self.config)
        self.canv.cd(1).RedrawAxis()

        self.canv.cd(2)
        xmin, xmax = self._get_axis_range(self.config)
        ref_line = ROOT.TLine()
        ref_line.SetLineStyle(ROOT.kDotted)
        ref_line.SetLineColor(ROOT.kBlack)
        ref_line.SetLineWidth(2)
        ref_line.DrawLine(xmin, 1.0, xmax, 1.0)
        CMS.cmsObjectDraw(self.ratio_band, "FE2", FillStyle=3004, LineWidth=0, FillColor=12, MarkerSize=0)
        for idx, (name, hist) in enumerate(self.ratio_hists.items(), start=1):
            color = self.palette[idx]
            CMS.cmsObjectDraw(hist, "hist", LineColor=color, LineWidth=2, LineStyle=ROOT.kSolid, MarkerSize=0)
            CMS.cmsObjectDraw(hist, "LE", LineColor=color, LineWidth=2, LineStyle=ROOT.kSolid, FillColor=ROOT.kWhite, MarkerSize=0)
        self.canv.cd(2).RedrawAxis()


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Run-period component templates")
    parser.add_argument("--era", required=True)
    parser.add_argument("--channel", required=True)
    parser.add_argument("--masspoint", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--binning", default="extended")
    parser.add_argument("--unblind", action="store_true")
    parser.add_argument("--partial-unblind", action="store_true", dest="partial_unblind")
    parser.add_argument("--nuisance", default="fallback_lnn", choices=["fallback_lnn", "preserve_shape"])
    parser.add_argument("--max-systematic-plots", type=int, default=-1,
                        help="Maximum number of stacked systematic variation plots to write per validation run; negative means all")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Run validation checks without writing diagnostic plots")
    return parser.parse_args()


def binning_suffix(args):
    suffix = args.binning
    if args.unblind:
        suffix = f"{args.binning}_unblind"
    elif args.partial_unblind:
        suffix = f"{args.binning}_partial_unblind"
    if args.nuisance == "preserve_shape":
        suffix = f"{suffix}_preserve_shape"
    return suffix


def load_json(path):
    with open(path) as f:
        return json.load(f)


def hist_edges(hist):
    xaxis = hist.GetXaxis()
    return tuple(float(xaxis.GetBinLowEdge(i)) for i in range(1, hist.GetNbinsX() + 2))


def parse_datacard_shape_rows(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3 and parts[1] == "shape?":
                rows.append((parts[0], parts[2:]))
    return rows


def parse_datacard_nuisance_rows(path):
    if not os.path.exists(path):
        return {"bins": [], "processes": [], "rows": []}

    with open(path) as f:
        lines = [line.split() for line in f if line.strip() and not line.lstrip().startswith("#")]

    proc_line_idx = [i for i, parts in enumerate(lines) if parts and parts[0] == "process"]
    if len(proc_line_idx) < 2:
        return {"bins": [], "processes": [], "rows": []}

    proc_idx = proc_line_idx[0]
    bins = lines[proc_idx - 1][1:]
    processes = lines[proc_idx][1:]
    rows = []
    for parts in lines[proc_line_idx[1] + 2:]:
        if len(parts) != len(processes) + 2:
            continue
        if parts[1] not in {"lnN", "shape?"}:
            continue
        rows.append({
            "name": parts[0],
            "type": parts[1],
            "values": parts[2:],
        })

    return {"bins": bins, "processes": processes, "rows": rows}


def mkdir(path):
    os.makedirs(path, exist_ok=True)
    return path


def clone_hist(hist, name):
    clone = hist.Clone(name)
    clone.SetDirectory(0)
    return clone


def sum_hists(directory, names, out_name):
    total = None
    for name in names:
        hist = directory.Get(name)
        if not hist or not hist.InheritsFrom("TH1"):
            continue
        if total is None:
            total = clone_hist(hist, out_name)
        else:
            total.Add(hist)
    return total


def ordered_physics_groups(physics_groups):
    def unique_members(members):
        seen = set()
        unique = []
        for member in members:
            if member in seen:
                continue
            seen.add(member)
            unique.append(member)
        return unique

    ordered = OrderedDict()
    for group in GROUP_ORDER:
        if group in physics_groups:
            ordered[group] = unique_members(physics_groups[group])
    for group, members in physics_groups.items():
        if group not in ordered and group != "signal":
            ordered[group] = unique_members(members)
    return ordered


def save_canvas(canvas, output_base):
    canvas.SaveAs(f"{output_base}.png")


def plot_com_energy(era):
    if era == "All":
        return "13/13.6 TeV"
    return get_CoM_energy(era)


def plot_era_for_category(args, payload):
    if args.era == "All":
        return payload.get("period") or payload.get("run_period") or args.era
    return args.era


def lumi_header_text(era):
    if era == "All":
        run2_lumi = _LUMI_CONFIG["Run2"]["combined"]
        run3_lumi = _LUMI_CONFIG["Run3"]["combined"]
        return f"Run 2+3, {run2_lumi}+{run3_lumi} fb^{{#minus1}} (13/13.6 TeV)"
    if era == "Run2":
        return f"Run 2, {_LUMI_CONFIG['Run2']['combined']} fb^{{#minus1}} (13 TeV)"
    if era == "Run3":
        return f"Run 3, {_LUMI_CONFIG['Run3']['combined']} fb^{{#minus1}} (13.6 TeV)"
    for period in ("Run2", "Run3"):
        if era in _LUMI_CONFIG[period]:
            return (
                f"{era}, {_LUMI_CONFIG[period][era]} fb^{{#minus1}} "
                f"({_LUMI_CONFIG[period]['energy_TeV']} TeV)"
            )
    return era


def overdraw_lumi_header(canvas, era):
    if era != "All":
        return

    upper = canvas.cd(1)
    pad_y_lo = upper.GetYlowNDC()
    pad_y_hi = pad_y_lo + upper.GetHNDC()
    pad_x_lo = upper.GetXlowNDC()
    pad_x_hi = pad_x_lo + upper.GetWNDC()
    top = upper.GetTopMargin()
    right = upper.GetRightMargin()

    def sub_to_canvas(x_sub, y_sub):
        return (
            pad_x_lo + x_sub * (pad_x_hi - pad_x_lo),
            pad_y_lo + y_sub * (pad_y_hi - pad_y_lo),
        )

    text_x_sub = 1 - right
    text_y_sub = 1 - top + 0.2 * top
    text_x, text_y = sub_to_canvas(text_x_sub, text_y_sub)
    erase_x1, erase_y1 = sub_to_canvas(0.0, 1.0 - top + 0.001)
    erase_x2, erase_y2 = sub_to_canvas(1.0, 1.0)

    canvas.cd()
    erase = ROOT.TPave(erase_x1, erase_y1, erase_x2, erase_y2, 0, "brNDC")
    erase.SetFillColor(ROOT.kWhite)
    erase.SetBorderSize(0)
    erase.Draw()
    canvas._lumi_erase = erase

    text_size_sub = 0.6 * top
    text_size_canvas = text_size_sub * (pad_y_hi - pad_y_lo)
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextAlign(31)
    latex.SetTextSize(text_size_canvas)
    latex.DrawLatex(text_x, text_y, lumi_header_text(era))
    canvas._lumi_latex = latex


def plot_channel_label(channel):
    labels = {
        "SR1E2Mu": "e#mu#mu",
        "SR3Mu": "#mu#mu#mu",
        "Combined": "e#mu#mu+#mu#mu#mu",
        "TTZ2E1Mu": "ee#mu",
    }
    return labels.get(channel, channel)


def plot_region_label(channel):
    if channel == "TTZ2E1Mu":
        return "TTZ CR"
    return "Signal Region"


def plot_x_title(channel):
    if channel == "TTZ2E1Mu":
        return "m(e^{+}e^{-}) [GeV]"
    return "m(#mu#mu) [GeV]"


def plot_ipos(era):
    if era in {"Run2", "Run3"}:
        return 0
    return 11


def lnn_scale_factors(value):
    if value in {"-", "0"}:
        return None
    try:
        if "/" in value:
            down, up = value.split("/", 1)
            return float(up), float(down)
        scale = float(value)
        return scale, max(0.0, 2.0 - scale)
    except ValueError:
        return None


def nuisance_shifted_total(directory, category, payload, datacard_nuisances, row, direction):
    process_meta = {
        meta["name"]: meta
        for meta in payload["processes"]
        if not meta.get("is_signal", False)
    }
    total = None
    for cat_col, proc, value in zip(datacard_nuisances["bins"], datacard_nuisances["processes"], row["values"]):
        if cat_col != category or proc not in process_meta:
            continue
        nominal = directory.Get(proc)
        if not nominal or not nominal.InheritsFrom("TH1"):
            continue

        shifted = None
        if row["type"] == "shape?" and value == "1":
            shifted = directory.Get(f"{proc}_{row['name']}{direction}")
        elif value != "-":
            scales = lnn_scale_factors(value)
            if scales:
                shifted = clone_hist(nominal, f"{proc}_{row['name']}_{direction}_lnn")
                shifted.Scale(scales[0] if direction == "Up" else scales[1])

        if shifted is None or not shifted.InheritsFrom("TH1"):
            shifted = nominal

        if total is None:
            total = clone_hist(shifted, f"{category}_{row['name']}_{direction}_total")
        else:
            total.Add(shifted)

    return total


def apply_prefit_uncertainty_band(total, directory, category, payload, datacard_nuisances):
    for ibin in range(1, total.GetNbinsX() + 1):
        err2 = total.GetBinError(ibin) ** 2
        nominal = total.GetBinContent(ibin)
        for row in datacard_nuisances["rows"]:
            up = nuisance_shifted_total(directory, category, payload, datacard_nuisances, row, "Up")
            down = nuisance_shifted_total(directory, category, payload, datacard_nuisances, row, "Down")
            if not up or not down:
                continue
            up_shift = up.GetBinContent(ibin) - nominal
            down_shift = down.GetBinContent(ibin) - nominal
            err2 += max(abs(up_shift), abs(down_shift)) ** 2
        total.SetBinError(ibin, ROOT.TMath.Sqrt(err2))


def apply_ratio_uncertainty_band(plotter):
    if plotter.config.get("no_ratio", False):
        return
    ratio_band = plotter.systematics.Clone("ratio_uncertainty")
    ratio_band.SetDirectory(0)
    for ibin in range(1, ratio_band.GetNbinsX() + 1):
        nominal = plotter.systematics.GetBinContent(ibin)
        if nominal > 0:
            ratio_band.SetBinContent(ibin, 1.0)
            ratio_band.SetBinError(ibin, plotter.systematics.GetBinError(ibin) / nominal)
        else:
            ratio_band.SetBinContent(ibin, 0.0)
            ratio_band.SetBinError(ibin, 0.0)
    plotter.ratio_band = ratio_band


def ratio_uncertainty_band(hist, name):
    ratio = clone_hist(hist, name)
    for ibin in range(1, ratio.GetNbinsX() + 1):
        nominal = hist.GetBinContent(ibin)
        if nominal > 0:
            ratio.SetBinContent(ibin, 1.0)
            ratio.SetBinError(ibin, hist.GetBinError(ibin) / nominal)
        else:
            ratio.SetBinContent(ibin, 0.0)
            ratio.SetBinError(ibin, 0.0)
    return ratio


def ratio_hist(numerator, denominator, name):
    ratio = clone_hist(numerator, name)
    for ibin in range(1, ratio.GetNbinsX() + 1):
        denom = denominator.GetBinContent(ibin)
        if denom > 0:
            ratio.SetBinContent(ibin, numerator.GetBinContent(ibin) / denom)
            ratio.SetBinError(ibin, numerator.GetBinError(ibin) / denom)
        else:
            ratio.SetBinContent(ibin, 0.0)
            ratio.SetBinError(ibin, 0.0)
    return ratio


def ratio_y_range(ratios, ratio_band=None):
    ymin = 1.0
    ymax = 1.0
    hists = list(ratios)
    if ratio_band is not None:
        hists.append(ratio_band)
    for hist in hists:
        for ibin in range(1, hist.GetNbinsX() + 1):
            value = hist.GetBinContent(ibin)
            error = hist.GetBinError(ibin)
            if value <= 0 and error <= 0:
                continue
            ymin = min(ymin, value - error)
            ymax = max(ymax, value + error)
    span = max(ymax - ymin, 0.20)
    ymin = max(0.0, ymin - 0.20 * span)
    ymax = ymax + 0.20 * span
    return [ymin, ymax]


def category_signal_hist(directory, category, payload):
    signal_members = [
        meta["name"] for meta in payload.get("processes", [])
        if meta.get("is_signal", False) and not meta.get("dummy_signal", False)
    ]
    if not signal_members:
        return None
    return sum_hists(directory, signal_members, f"{category}_signal")


def validation_uses_real_data(args):
    return args.unblind or args.partial_unblind or args.method == "CR"


def stack_y_range(total_background, signal=None, data=None, include_data=False):
    """Use max(total background, signal, and data when real data is drawn)."""
    ymax = total_background.GetMaximum()
    if signal and signal.InheritsFrom("TH1"):
        ymax = max(ymax, signal.GetMaximum())
    if include_data and data and data.InheritsFrom("TH1"):
        ymax = max(ymax, data.GetMaximum())
    if ymax <= 0:
        ymax = 1.0
    return [0.0, ymax * 2.0]


def variation_y_range(nominal, up, down):
    ymax = max(nominal.GetMaximum(), up.GetMaximum(), down.GetMaximum())
    if ymax <= 0:
        ymax = 1.0
    return [0.0, ymax * 2.0]


def scoped_background_group_hists(directory, category, payload, physics_groups, scope):
    process_names = process_names_for_variation(payload, include_signal=False, scope=scope)
    group_hists = OrderedDict()
    for group, members in ordered_physics_groups(physics_groups).items():
        scoped_members = [member for member in members if member in process_names]
        hist = sum_hists(directory, scoped_members, f"{category}_{group}_{scope or 'all'}")
        if hist and hist.Integral() > 0:
            hist.SetTitle(group)
            group_hists[group] = hist
    return group_hists


def total_from_hists(hists, name):
    total = None
    for hist in hists:
        if total is None:
            total = clone_hist(hist, name)
        else:
            total.Add(hist)
    return total


def make_stack_plot(directory, category, payload, physics_groups, output_dir, args, datacard_nuisances):
    data = directory.Get("data_obs")
    if not data:
        return None

    group_hists = OrderedDict()
    for group, members in ordered_physics_groups(physics_groups).items():
        hist = sum_hists(directory, members, f"{category}_{group}")
        if hist and hist.Integral() > 0:
            group_hists[group] = hist

    if not group_hists:
        return None

    total = None
    colors = []
    for group, hist in group_hists.items():
        hist.SetTitle(f"{group} ({hist.Integral():.1f})")
        colors.append(GROUP_COLORS.get(group, ROOT.kGray + 1))
        if total is None:
            total = clone_hist(hist, f"{category}_total_bkg")
        else:
            total.Add(hist)

    signal = category_signal_hist(directory, category, payload)

    data_draw = clone_hist(data, f"{category}_data_obs_draw")
    data_draw.SetTitle("Data")
    plot_era = plot_era_for_category(args, payload)
    plot_channel = payload.get("channel", args.channel)

    config = {
        "era": plot_era,
        "CoM": plot_com_energy(plot_era),
        "iPos": plot_ipos(plot_era),
        "channel": plot_region_label(plot_channel),
        "region": plot_channel_label(plot_channel),
        "xTitle": plot_x_title(plot_channel),
        "yTitle": "Events",
        "yRange": stack_y_range(total, signal, data_draw, validation_uses_real_data(args)),
        "rTitle": "Data / Pred.",
        "rRange": [0.0, 2.0],
        "maxDigits": 3,
        "systSrc": "Stat+Syst",
        "signalHist": signal,
        "colors": colors,
        "legend": POSTFIT_SUMMARY_LEGEND,
        "legendColumns": 2,
        "legendTextSize": POSTFIT_SUMMARY_LEGEND_TEXT_SIZE,
        "systLegendSecondColumn": True,
    }
    plotter = ValidationComparisonCanvas(data_draw, group_hists, config)
    apply_prefit_uncertainty_band(plotter.systematics, directory, category, payload, datacard_nuisances)
    apply_ratio_uncertainty_band(plotter)
    plotter.drawPadUp()
    plotter.drawPadDown()
    overdraw_lumi_header(plotter.canv, plot_era)

    output_base = os.path.join(output_dir, "stack_background")
    save_canvas(plotter.canv, output_base)
    return {
        "category": category,
        "plot": f"{output_base}.png",
        "data_yield": data_draw.Integral(),
        "background_yield": total.Integral(),
        "signal_yield": signal.Integral() if signal else 0.0,
        "groups": {group: hist.Integral() for group, hist in group_hists.items()},
    }


def make_component_yield_plot(directory, category, payload, output_dir):
    entries = []
    for meta in payload["processes"]:
        if meta.get("is_signal", False):
            continue
        hist = directory.Get(meta["name"])
        if hist and hist.InheritsFrom("TH1"):
            entries.append((meta["name"], hist.Integral()))
    entries = [(name, value) for name, value in entries if value > 0]
    entries.sort(key=lambda item: item[1], reverse=True)
    if not entries:
        return None

    hist = ROOT.TH1F(f"h_component_yields_{category}", "", len(entries), 0, len(entries))
    hist.SetDirectory(0)
    for idx, (name, value) in enumerate(entries, start=1):
        hist.SetBinContent(idx, value)
        hist.GetXaxis().SetBinLabel(idx, name)

    canvas = ROOT.TCanvas(f"c_component_yields_{category}", "", max(900, 22 * len(entries)), 650)
    canvas.SetBottomMargin(0.34)
    canvas.SetLeftMargin(0.10)
    hist.SetFillColor(ROOT.TColor.GetColor("#4c78a8"))
    hist.SetLineColor(ROOT.kBlack)
    hist.GetYaxis().SetTitle("Yield")
    hist.GetXaxis().LabelsOption("v")
    hist.Draw("HIST")
    output_base = os.path.join(output_dir, "component_yields")
    save_canvas(canvas, output_base)
    return f"{output_base}.png"


def active_nuisance_value(value, nuisance_type="shape?"):
    if value in {"-", "0"}:
        return False
    if nuisance_type == "lnN":
        try:
            if "/" in value:
                down, up = value.split("/", 1)
                return float(down) != 1.0 or float(up) != 1.0
            return float(value) != 1.0
        except ValueError:
            return True
    return True


def shifted_hist_for_nuisance(directory, proc, syst_name, nuisance_type, value, direction):
    nominal = directory.Get(proc)
    if not nominal or not nominal.InheritsFrom("TH1"):
        return None
    if direction == "Nominal" or not active_nuisance_value(value, nuisance_type):
        return nominal

    if nuisance_type == "shape?" and value == "1":
        shifted = directory.Get(f"{proc}_{syst_name}{direction}")
        if shifted and shifted.InheritsFrom("TH1"):
            return shifted
        return nominal

    scales = lnn_scale_factors(value)
    if not scales:
        return nominal
    shifted = clone_hist(nominal, f"{proc}_{syst_name}_{direction}_numeric")
    shifted.Scale(scales[0] if direction == "Up" else scales[1])
    return shifted


def process_subera(process_name):
    for token in ERA_DECORRELATED_TOKENS:
        if process_name.endswith(f"_{token}"):
            return token
    return None


def process_matches_systematic_scope(process_name, payload, scope):
    if scope in ERA_DECORRELATED_TOKENS:
        return process_subera(process_name) == scope
    if scope in RUN_PERIOD_ERAS:
        subera = process_subera(process_name)
        if subera is not None:
            return subera in RUN_PERIOD_ERAS[scope]
        return (payload.get("period") or payload.get("run_period")) == scope
    return True


def process_names_for_variation(payload, include_signal=False, scope=None):
    return {
        meta["name"]
        for meta in payload["processes"]
        if bool(meta.get("is_signal", False)) == include_signal
        and not (include_signal and meta.get("dummy_signal", False))
        and process_matches_systematic_scope(meta["name"], payload, scope)
    }


def nuisance_has_category_targets(category, payload, datacard_nuisances, syst_name, include_signal=False, scope=None):
    targets = process_names_for_variation(payload, include_signal=include_signal, scope=scope)
    for row in datacard_nuisances["rows"]:
        if row["name"] != syst_name or row["type"] not in {"lnN", "shape?"}:
            continue
        for cat_col, proc, value in zip(datacard_nuisances["bins"], datacard_nuisances["processes"], row["values"]):
            if cat_col == category and proc in targets and active_nuisance_value(value, row["type"]):
                return True
    return False


def variation_target_kind(category, payload, datacard_nuisances, syst_name, scope=None):
    if nuisance_has_category_targets(category, payload, datacard_nuisances, syst_name, include_signal=False, scope=scope):
        return "background"
    if nuisance_has_category_targets(category, payload, datacard_nuisances, syst_name, include_signal=True, scope=scope):
        return "signal"
    return None


def variation_target_kinds(category, payload, datacard_nuisances, syst_name, scope=None):
    target_kinds = []
    if nuisance_has_category_targets(category, payload, datacard_nuisances, syst_name, include_signal=False, scope=scope):
        target_kinds.append("background")
    if nuisance_has_category_targets(category, payload, datacard_nuisances, syst_name, include_signal=True, scope=scope):
        target_kinds.append("signal")
    return target_kinds


def nuisance_row_value(datacard_nuisances, row, category, process_name):
    for cat_col, proc, value in zip(datacard_nuisances["bins"], datacard_nuisances["processes"], row["values"]):
        if cat_col == category and proc == process_name:
            return value
    return "-"


def varied_total_for_row(directory, category, payload, datacard_nuisances, row, direction, target_kind, scope=None):
    process_names = process_names_for_variation(payload, include_signal=(target_kind == "signal"), scope=scope)
    total = None
    for proc in process_names:
        value = nuisance_row_value(datacard_nuisances, row, category, proc)
        hist = shifted_hist_for_nuisance(directory, proc, row["name"], row["type"], value, direction)
        if not hist or not hist.InheritsFrom("TH1"):
            continue
        if total is None:
            total = clone_hist(hist, f"{category}_{row['name']}_{target_kind}_{direction}")
        else:
            total.Add(hist)
    return total


def varied_total_for_syst(directory, category, payload, datacard_nuisances, syst_name, direction, target_kind, scope=None):
    row = next(
        (candidate for candidate in datacard_nuisances["rows"]
         if candidate["name"] == syst_name and candidate["type"] in {"lnN", "shape?"}),
        None,
    )
    if row is None:
        return None
    return varied_total_for_row(directory, category, payload, datacard_nuisances, row, direction, target_kind, scope=scope)


def prefit_uncertainty_hist(directory, category, payload, datacard_nuisances, nominal, target_kind, scope=None):
    syst = clone_hist(nominal, f"{nominal.GetName()}_prefit_syst")
    row_variations = []
    for row in datacard_nuisances["rows"]:
        if row["type"] not in {"lnN", "shape?"}:
            continue
        up = varied_total_for_row(directory, category, payload, datacard_nuisances, row, "Up", target_kind, scope=scope)
        down = varied_total_for_row(directory, category, payload, datacard_nuisances, row, "Down", target_kind, scope=scope)
        if up and down:
            row_variations.append((up, down))
    for ibin in range(1, syst.GetNbinsX() + 1):
        err2 = nominal.GetBinError(ibin) ** 2
        nominal_value = nominal.GetBinContent(ibin)
        for up, down in row_variations:
            up_shift = up.GetBinContent(ibin) - nominal_value
            down_shift = down.GetBinContent(ibin) - nominal_value
            err2 += max(abs(up_shift), abs(down_shift)) ** 2
        syst.SetBinError(ibin, ROOT.TMath.Sqrt(err2))
    return syst


def active_systematics(datacard_nuisances):
    return list(OrderedDict(
        (row["name"], None) for row in datacard_nuisances["rows"]
        if row["type"] in {"lnN", "shape?"}
    ).keys())


def systematic_scope(syst_name):
    """Return subera, Run2/Run3, or correlated-run scope for a nuisance."""
    for token in ERA_DECORRELATED_TOKENS:
        if re.search(rf"(^|_){re.escape(token)}($|_)", syst_name):
            return token
    if re.search(r"(^|_)13TeV($|_)", syst_name):
        return "Run2"
    if re.search(r"(^|_)13p6TeV($|_)", syst_name):
        return "Run3"
    if "_uncorrelated_" in syst_name:
        suffix = syst_name.rsplit("_uncorrelated_", 1)[-1]
        if suffix in ERA_DECORRELATED_TOKENS:
            return suffix
        if suffix in RUN_PERIOD_ERAS:
            return suffix
        return "uncorrelated"
    return "run"


def systematic_output_scope(syst_name, payload, args):
    scope = systematic_scope(syst_name)
    if args.era == "All" and scope == "run":
        return "All"
    return scope


def systematic_active_for_category(syst_name, category, payload, datacard_nuisances):
    scope = systematic_scope(syst_name)
    if scope in ERA_DECORRELATED_TOKENS and scope not in payload.get("suberas", []):
        return False
    if scope in RUN_PERIOD_ERAS and (payload.get("period") or payload.get("run_period")) != scope:
        return False
    return variation_target_kind(category, payload, datacard_nuisances, syst_name, scope=scope) is not None


def systematic_plot_era(scope, args, payload):
    if scope in ERA_DECORRELATED_TOKENS or scope in RUN_PERIOD_ERAS:
        return scope
    return plot_era_for_category(args, payload)


def validation_category_dir_name(category, payload, args, n_categories):
    if args.era == "All":
        return payload.get("period", category)
    if n_categories == 1:
        return ""
    return category


def same_hist_binning(hist_a, hist_b):
    return hist_edges(hist_a) == hist_edges(hist_b)


def add_hist_to_sum(total, hist, name):
    if total is None:
        return clone_hist(hist, name)
    if not same_hist_binning(total, hist):
        return None
    total.Add(hist)
    return total


def make_aggregate_stack_plot(f, categories, physics_groups, output_dir, args, datacard_nuisances):
    data_total = None
    signal_total = None
    aggregate_groups = OrderedDict()
    per_category_syst = []

    for cat, payload in categories.items():
        directory = f.Get(cat)
        if not directory:
            continue
        data = directory.Get("data_obs")
        if data:
            data_total = add_hist_to_sum(data_total, data, "All_data_obs")
            if data_total is None:
                return None
        signal = category_signal_hist(directory, cat, payload)
        if signal:
            signal_total = add_hist_to_sum(signal_total, signal, "All_signal")
            if signal_total is None:
                return None

        category_total = None
        for group, members in ordered_physics_groups(physics_groups).items():
            hist = sum_hists(directory, members, f"{cat}_{group}_aggregate")
            if not hist or hist.Integral() <= 0:
                continue
            if group not in aggregate_groups:
                aggregate_groups[group] = clone_hist(hist, f"All_{group}")
            elif add_hist_to_sum(aggregate_groups[group], hist, f"All_{group}") is None:
                return None

            if category_total is None:
                category_total = clone_hist(hist, f"{cat}_total_for_all_band")
            else:
                category_total.Add(hist)

        if category_total:
            apply_prefit_uncertainty_band(category_total, directory, cat, payload, datacard_nuisances)
            per_category_syst.append(category_total)

    if not data_total or not aggregate_groups:
        return None

    colors = []
    total = None
    for group, hist in aggregate_groups.items():
        hist.SetTitle(f"{group} ({hist.Integral():.1f})")
        colors.append(GROUP_COLORS.get(group, ROOT.kGray + 1))
        if total is None:
            total = clone_hist(hist, "All_total_bkg")
        else:
            total.Add(hist)

    data_total.SetTitle("Data")
    config = {
        "era": args.era,
        "CoM": plot_com_energy(args.era),
        "iPos": plot_ipos(args.era),
        "channel": plot_region_label(args.channel),
        "region": plot_channel_label(args.channel),
        "xTitle": plot_x_title(args.channel),
        "yTitle": "Events",
        "yRange": stack_y_range(total, signal_total, data_total, validation_uses_real_data(args)),
        "rTitle": "Data / Pred.",
        "rRange": [0.0, 2.0],
        "maxDigits": 3,
        "systSrc": "Stat+Syst",
        "signalHist": signal_total,
        "colors": colors,
        "legend": POSTFIT_SUMMARY_LEGEND,
        "legendColumns": 2,
        "legendTextSize": POSTFIT_SUMMARY_LEGEND_TEXT_SIZE,
        "systLegendSecondColumn": True,
    }
    plotter = ValidationComparisonCanvas(data_total, aggregate_groups, config)
    for ibin in range(1, plotter.systematics.GetNbinsX() + 1):
        err2 = 0.0
        for cat_syst in per_category_syst:
            err2 += cat_syst.GetBinError(ibin) ** 2
        plotter.systematics.SetBinError(ibin, ROOT.TMath.Sqrt(err2))
    apply_ratio_uncertainty_band(plotter)
    plotter.drawPadUp()
    plotter.drawPadDown()
    overdraw_lumi_header(plotter.canv, "All")

    output_base = os.path.join(output_dir, "stack_background")
    save_canvas(plotter.canv, output_base)
    return {
        "category": "All",
        "plot": f"{output_base}.png",
        "data_yield": data_total.Integral(),
        "background_yield": total.Integral(),
        "signal_yield": signal_total.Integral() if signal_total else 0.0,
        "groups": {group: hist.Integral() for group, hist in aggregate_groups.items()},
    }


def make_aggregate_component_yield_plot(f, categories, output_dir):
    yields = OrderedDict()
    for cat, payload in categories.items():
        directory = f.Get(cat)
        if not directory:
            continue
        for meta in payload["processes"]:
            if meta.get("is_signal", False):
                continue
            hist = directory.Get(meta["name"])
            if hist and hist.InheritsFrom("TH1") and hist.Integral() > 0:
                yields.setdefault(meta["name"], 0.0)
                yields[meta["name"]] += hist.Integral()
    entries = [(name, value) for name, value in yields.items() if value > 0]
    if not entries:
        return None
    entries.sort(key=lambda item: item[1], reverse=True)

    hist = ROOT.TH1F("h_component_yields_All", "", len(entries), 0, len(entries))
    hist.SetDirectory(0)
    for idx, (name, value) in enumerate(entries, start=1):
        hist.SetBinContent(idx, value)
        hist.GetXaxis().SetBinLabel(idx, name)

    canvas = ROOT.TCanvas("c_component_yields_All", "", max(900, 22 * len(entries)), 650)
    canvas.SetBottomMargin(0.34)
    canvas.SetLeftMargin(0.10)
    hist.SetFillColor(ROOT.TColor.GetColor("#4c78a8"))
    hist.SetLineColor(ROOT.kBlack)
    hist.GetYaxis().SetTitle("Yield")
    hist.GetXaxis().LabelsOption("v")
    hist.Draw("HIST")
    output_base = os.path.join(output_dir, "component_yields")
    save_canvas(canvas, output_base)
    return f"{output_base}.png"


def make_systematic_background_plot(
    directory, category, payload, physics_groups, output_dir, args, datacard_nuisances, syst_name, scope, output_name
):
    group_hists = scoped_background_group_hists(directory, category, payload, physics_groups, scope)
    if not group_hists:
        return None

    nominal = total_from_hists(group_hists.values(), f"{category}_{syst_name}_background_nominal")
    up = varied_total_for_syst(
        directory, category, payload, datacard_nuisances, syst_name, "Up", "background", scope=scope
    )
    down = varied_total_for_syst(
        directory, category, payload, datacard_nuisances, syst_name, "Down", "background", scope=scope
    )
    if not nominal or not up or not down or nominal.Integral() <= 0:
        return None

    syst_band = prefit_uncertainty_hist(
        directory, category, payload, datacard_nuisances, nominal, "background", scope=scope
    )
    ratio_up = ratio_hist(up, nominal, f"{category}_{syst_name}_background_up_ratio")
    ratio_down = ratio_hist(down, nominal, f"{category}_{syst_name}_background_down_ratio")
    ratio_band = ratio_uncertainty_band(syst_band, f"{category}_{syst_name}_background_ratio_band")

    colors = [GROUP_COLORS.get(group, ROOT.kGray + 1) for group in group_hists]
    plot_era = systematic_plot_era(scope, args, payload)
    plot_channel = payload.get("channel", args.channel)
    up.SetTitle("Up")
    down.SetTitle("Down")
    config = {
        "era": plot_era,
        "CoM": plot_com_energy(plot_era),
        "iPos": 0,
        "channel": plot_region_label(plot_channel),
        "region": plot_channel_label(plot_channel),
        "xTitle": plot_x_title(plot_channel),
        "yTitle": "Events",
        "yRange": variation_y_range(syst_band, up, down),
        "rTitle": "Var. / Nom.",
        "rRange": [0.5, 1.5],
        "maxDigits": 3,
        "systSrc": "Stat+Syst",
        "colors": colors,
        "legend": POSTFIT_SUMMARY_LEGEND,
        "legendColumns": 2,
        "legendTextSize": POSTFIT_SUMMARY_LEGEND_TEXT_SIZE,
        "backgroundAlpha": SYST_BACKGROUND_ALPHA,
        "variationHists": [(up, up.GetTitle(), SYST_UP_COLOR), (down, down.GetTitle(), SYST_DOWN_COLOR)],
        "variationRatioHists": [
            (ratio_up, "Up / Nom.", SYST_UP_COLOR),
            (ratio_down, "Down / Nom.", SYST_DOWN_COLOR),
        ],
    }
    plotter = SystematicStackCanvas(nominal, group_hists, config)
    plotter.systematics = syst_band
    plotter.ratio_band = ratio_band
    plotter.drawPadUp()
    plotter.drawPadDown()
    plotter.canv.cd(1)
    CMS.drawText(syst_name, posX=0.20, posY=0.58, font=42, align=0, size=0.026)
    plotter.leg.Draw()
    overdraw_lumi_header(plotter.canv, plot_era)

    output_base = os.path.join(output_dir, output_name)
    save_canvas(plotter.canv, output_base)
    return f"{output_base}.png"


def make_systematic_signal_plot(
    directory, category, payload, output_dir, args, datacard_nuisances, syst_name, scope, output_name
):
    nominal = varied_total_for_syst(
        directory, category, payload, datacard_nuisances, syst_name, "Nominal", "signal", scope=scope
    )
    up = varied_total_for_syst(
        directory, category, payload, datacard_nuisances, syst_name, "Up", "signal", scope=scope
    )
    down = varied_total_for_syst(
        directory, category, payload, datacard_nuisances, syst_name, "Down", "signal", scope=scope
    )
    if not nominal or not up or not down or nominal.Integral() <= 0:
        return None

    syst_band = prefit_uncertainty_hist(
        directory, category, payload, datacard_nuisances, nominal, "signal", scope=scope
    )
    ratio_up = ratio_hist(up, nominal, f"{category}_{syst_name}_signal_up_ratio")
    ratio_down = ratio_hist(down, nominal, f"{category}_{syst_name}_signal_down_ratio")
    ratio_band = ratio_uncertainty_band(syst_band, f"{category}_{syst_name}_signal_ratio_band")

    hists = OrderedDict([
        ("Nominal", nominal),
        ("Up", up),
        ("Down", down),
    ])
    ratio_hists = OrderedDict([
        ("Up / Nom.", ratio_up),
        ("Down / Nom.", ratio_down),
    ])
    plot_era = systematic_plot_era(scope, args, payload)
    plot_channel = payload.get("channel", args.channel)
    config = {
        "era": plot_era,
        "CoM": plot_com_energy(plot_era),
        "iPos": 0,
        "channel": plot_region_label(plot_channel),
        "region": plot_channel_label(plot_channel),
        "xTitle": plot_x_title(plot_channel),
        "yTitle": "Events",
        "xRange": [nominal.GetXaxis().GetXmin(), nominal.GetXaxis().GetXmax()],
        "yRange": variation_y_range(nominal, up, down),
        "rTitle": "Var. / Nom.",
        "rRange": [0.5, 1.5],
        "maxDigits": 3,
        "colors": [ROOT.kBlack, SYST_UP_COLOR, SYST_DOWN_COLOR],
        "legend": POSTFIT_SUMMARY_LEGEND,
        "legendColumns": 1,
        "legendTextSize": POSTFIT_SUMMARY_LEGEND_TEXT_SIZE,
        "systSrc": "Stat+Syst",
    }
    plotter = SystematicLineCanvas(hists, ratio_hists, ratio_band, config)
    plotter.palette = config["colors"]
    plotter.drawPad()
    plotter.canv.cd(1)
    CMS.drawText(f"{syst_name} (signal)", posX=0.20, posY=0.58, font=42, align=0, size=0.026)
    plotter.leg.Draw()
    overdraw_lumi_header(plotter.canv, plot_era)

    output_base = os.path.join(output_dir, output_name)
    save_canvas(plotter.canv, output_base)
    return f"{output_base}.png"


def make_systematic_plots(f, categories, shape_rows, output_dir, args, datacard_nuisances, physics_groups):
    plots = {"run": [], "All": [], "Run2": [], "Run3": []}
    for token in ERA_DECORRELATED_TOKENS:
        plots[token] = []
    plots["uncorrelated"] = []
    syst_names = active_systematics(datacard_nuisances)
    if args.max_systematic_plots >= 0:
        syst_names = syst_names[:args.max_systematic_plots]
    omit_category_dir = len(categories) == 1 or args.era == "All"

    for cat, payload in categories.items():
        directory = f.Get(cat)
        if not directory:
            continue
        for syst_name in syst_names:
            if not systematic_active_for_category(syst_name, cat, payload, datacard_nuisances):
                continue
            scope = systematic_scope(syst_name)
            output_scope = systematic_output_scope(syst_name, payload, args)
            target_kinds = variation_target_kinds(cat, payload, datacard_nuisances, syst_name, scope=scope)
            if not target_kinds:
                continue
            scoped_output_dir = os.path.join(output_dir, output_scope)
            if not omit_category_dir:
                scoped_output_dir = os.path.join(scoped_output_dir, cat)
            mkdir(scoped_output_dir)
            output_name = f"syst_{syst_name}"
            if args.era == "All" and len(categories) > 1:
                output_name = f"syst_{cat}_{syst_name}"
            if "background" in target_kinds:
                plot_path = make_systematic_background_plot(
                    directory, cat, payload, physics_groups,
                    scoped_output_dir, args, datacard_nuisances, syst_name, scope, output_name
                )
                if plot_path:
                    plots.setdefault(output_scope, []).append(plot_path)
            if "signal" in target_kinds:
                plot_path = make_systematic_signal_plot(
                    directory, cat, payload, scoped_output_dir, args,
                    datacard_nuisances, syst_name, scope, f"{output_name}_signal"
                )
                if plot_path:
                    plots.setdefault(output_scope, []).append(plot_path)
    return plots


def main():
    args = parse_args()
    workdir = os.getenv("WORKDIR")
    if not workdir:
        raise EnvironmentError("WORKDIR environment variable not set. Please run 'source setup.sh'")

    suffix = binning_suffix(args)
    tdir = f"{workdir}/SignalRegionStudyV3/templates/{args.era}/{args.channel}/{args.masspoint}/{args.method}/{suffix}"
    categories_path = f"{tdir}/categories.json"
    process_path = f"{tdir}/process_list.json"
    binning_path = f"{tdir}/binning.json"
    shapes_path = f"{tdir}/shapes.root"
    datacard_path = f"{tdir}/datacard.txt"
    validation_dir = mkdir(f"{tdir}/validation")

    for path in [categories_path, process_path, binning_path, shapes_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)

    categories = load_json(categories_path)["categories"]
    categories_meta = load_json(categories_path)
    processes = load_json(process_path)
    binning = load_json(binning_path)["categories"]
    shape_rows = parse_datacard_shape_rows(datacard_path)
    datacard_nuisances = parse_datacard_nuisance_rows(datacard_path)

    f = ROOT.TFile.Open(shapes_path, "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Failed to open {shapes_path}")

    issues = []
    warnings = []

    for cat, payload in categories.items():
        directory = f.Get(cat)
        if not directory:
            issues.append(f"Missing category directory: {cat}")
            continue

        data_obs = directory.Get("data_obs")
        if not data_obs:
            issues.append(f"{cat}: missing data_obs")
            continue

        expected_edges = tuple(float(x) for x in binning[cat]["bin_edges"])
        if hist_edges(data_obs) != expected_edges:
            issues.append(f"{cat}: data_obs binning does not match binning.json")

        total_bkg = 0.0
        for meta in payload["processes"]:
            proc = meta["name"]
            hist = directory.Get(proc)
            if not hist:
                issues.append(f"{cat}: missing nominal {proc}")
                continue
            if hist_edges(hist) != expected_edges:
                issues.append(f"{cat}/{proc}: binning does not match category binning")
            if not meta.get("is_signal", False):
                total_bkg += hist.Integral()

            for key in directory.GetListOfKeys():
                name = key.GetName()
                if not name.startswith(proc + "_"):
                    continue
                obj = directory.Get(name)
                if obj and obj.InheritsFrom("TH1") and hist_edges(obj) != expected_edges:
                    issues.append(f"{cat}/{name}: variation binning does not match category binning")

        uses_real_data = args.method == "CR" or categories_meta.get("data_obs") == "real_data"
        if not (args.unblind or args.partial_unblind or uses_real_data):
            diff = abs(data_obs.Integral() - total_bkg)
            tol = max(1e-5, 1e-6 * max(1.0, total_bkg))
            if diff > tol:
                issues.append(f"{cat}: blinded data_obs {data_obs.Integral():.6f} != summed bkg {total_bkg:.6f}")

    # Check active shape? entries with value 1 have both Up/Down histograms.
    if shape_rows and datacard_path:
        with open(datacard_path) as fcard:
            lines = [line.split() for line in fcard if line.strip()]
        proc_line_idx = [i for i, p in enumerate(lines) if p and p[0] == "process"]
        if len(proc_line_idx) >= 2:
            bin_columns = lines[proc_line_idx[0] - 1][1:]
            proc_columns = lines[proc_line_idx[0]][1:]
            for syst_name, values in shape_rows:
                for cat, proc, value in zip(bin_columns, proc_columns, values):
                    if value != "1":
                        continue
                    directory = f.Get(cat)
                    if not directory:
                        continue
                    if not directory.Get(f"{proc}_{syst_name}Up") or not directory.Get(f"{proc}_{syst_name}Down"):
                        issues.append(f"{cat}/{proc}: shape?=1 but missing {syst_name} Up/Down")

    for group, members in processes.get("physics_groups", {}).items():
        if group != "signal" and not members:
            warnings.append(f"Physics group has no components: {group}")

    plot_summary = {}
    if not args.skip_plots:
        n_categories = len(categories)
        for cat, payload in categories.items():
            directory = f.Get(cat)
            if not directory:
                continue
            legacy_cat_plot_dir = os.path.join(validation_dir, cat)
            category_dir_name = validation_category_dir_name(cat, payload, args, n_categories)
            if not category_dir_name:
                if os.path.isdir(legacy_cat_plot_dir):
                    shutil.rmtree(legacy_cat_plot_dir)
                cat_plot_dir = validation_dir
            else:
                cat_plot_dir = os.path.join(validation_dir, category_dir_name)
                if os.path.isdir(cat_plot_dir):
                    shutil.rmtree(cat_plot_dir)
                if legacy_cat_plot_dir != cat_plot_dir and os.path.isdir(legacy_cat_plot_dir):
                    shutil.rmtree(legacy_cat_plot_dir)
            mkdir(cat_plot_dir)
            stack_info = make_stack_plot(
                directory, cat, payload, processes.get("physics_groups", {}),
                cat_plot_dir, args, datacard_nuisances
            )
            component_plot = make_component_yield_plot(directory, cat, payload, cat_plot_dir)
            plot_summary[category_dir_name or cat] = {
                "stack": stack_info,
                "component_yields": component_plot,
            }
        if args.era == "All" and n_categories > 1:
            all_plot_dir = os.path.join(validation_dir, "All")
            if os.path.isdir(all_plot_dir):
                shutil.rmtree(all_plot_dir)
            mkdir(all_plot_dir)
            aggregate_stack = make_aggregate_stack_plot(
                f, categories, processes.get("physics_groups", {}),
                all_plot_dir, args, datacard_nuisances
            )
            aggregate_component = make_aggregate_component_yield_plot(f, categories, all_plot_dir)
            if aggregate_stack is None:
                warnings.append("All aggregate stack plot skipped because category binnings are incompatible or inputs are missing")
            plot_summary["All"] = {
                "stack": aggregate_stack,
                "component_yields": aggregate_component,
            }
        syst_plot_dir = os.path.join(validation_dir, "systematics")
        if os.path.isdir(syst_plot_dir):
            shutil.rmtree(syst_plot_dir)
        mkdir(syst_plot_dir)
        plot_summary["systematics"] = make_systematic_plots(
            f, categories, shape_rows, syst_plot_dir, args,
            datacard_nuisances, processes.get("physics_groups", {})
        )

    summary = {
        "template_dir": tdir,
        "categories": list(categories.keys()),
        "issues": issues,
        "warnings": warnings,
        "plots": plot_summary,
    }
    with open(f"{validation_dir}/summary.json", "w") as fout:
        json.dump(summary, fout, indent=2)

    f.Close()

    print("=" * 60)
    print("Run-period template validation")
    print(f"Template dir: {tdir}")
    print(f"Categories: {', '.join(categories.keys())}")
    print(f"Issues: {len(issues)}")
    for issue in issues[:30]:
        print(f"ERROR: {issue}")
    if len(issues) > 30:
        print(f"ERROR: ... and {len(issues) - 30} more")
    print(f"Warnings: {len(warnings)}")
    for warning in warnings[:20]:
        print(f"WARNING: {warning}")
    print(f"Validation outputs: {validation_dir}")
    print("=" * 60)

    if issues:
        sys.exit(1)


if __name__ == "__main__":
    main()
