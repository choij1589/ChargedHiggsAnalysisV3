#!/usr/bin/env python3
"""
Plot post-fit distributions on the real physical mass axis, using a fine
uniform 1-GeV grid, with per-process histograms filled from the unbinned
preprocessed trees in `samples/` and then scaled by the FitDiagnostics
post/pre-fit ratio per coarse bin.

Aggregates multiple sub-channels (per-era/per-channel) from a combined
fitDiagnostics file. Each sub-channel's mass window and adaptive coarse bin
edges are taken from its `binning.json`. The union mass range defines the
fine uniform-bin grid (default bin width: auto = sigma_eff of widest sub-channel, snapped to 0.05 GeV).

Usage:
    python3 plotPostfitMass.py --era All --masspoint MHc130_MA90 \
        --method ParticleNet \
        --channel-scope Combined --fit-type b
"""
import os
import sys
import json
import math
import hashlib
import logging
import argparse
import bisect
from array import array
from dataclasses import dataclass
from typing import Tuple
import ROOT

def _build_parser():
    p = argparse.ArgumentParser(description="Real-mass unbinned post-fit plots")
    p.add_argument("--era", required=True, type=str,
                   help="Fit source: All, Run2, Run3, or per-era (e.g. 2018). "
                        "Selects which fitDiagnostics file to open.")
    p.add_argument("--masspoint", required=True, type=str)
    p.add_argument("--method", required=True, type=str,
                   help="Template method (Baseline, ParticleNet, CR, ...)")
    p.add_argument("--era-scope", default=None, dest="era_scope",
                   help="Filter plots to this era slice (e.g. 2018, Run2, All). "
                        "Default: iterate every era scope applicable to the fit.")
    p.add_argument("--channel-scope", default=None, type=str, dest="channel_scope",
                   help="Filter plots to this channel scope. Default: iterate all three "
                        "(SR1E2Mu/SR3Mu/Combined for SR; pass TTZ2E1Mu for CR).")
    p.add_argument("--fit-channel", default="Combined", type=str, dest="fit_channel",
                   help="Channel segment in the fitDiagnostics path. "
                        "SR uses 'Combined' (default); CR uses 'TTZ2E1Mu'.")
    p.add_argument("--fit-type", default="both", choices=["b", "s", "both"],
                   help="Which post-fit variant(s) to plot [default: both]")
    p.add_argument("--blind", action="store_true",
                   help="Asimov mode: data = sum of pre-fit backgrounds, "
                        "samples/.../data.root is never read; reads from the "
                        "{method}_blind template segment. Default is unblind.")
    p.add_argument("--bin-width", default="auto", type=str,
                   help="Fine-grid bin width in GeV, or 'auto' to derive from "
                        "sigma_eff of the widest sub-channel, snapped to 0.05 GeV (default: auto)")
    p.add_argument("--plot-only", action="store_true", dest="plot_only",
                   help="Skip tree reads; load cached fine-mass hists from "
                        "{fitdiag}/cached/ and only re-render plots.")
    p.add_argument("--debug", action="store_true")
    return p


# Module-level state populated by `entry_setup()` before any helper is called.
# This lets other scripts (e.g. plotCombinedMass.py) import this module and
# reuse the helpers without triggering argparse / filesystem work at import.
args = None
WORKDIR = os.getenv("WORKDIR")
if not WORKDIR:
    raise EnvironmentError("WORKDIR not set. Source setup.sh first.")

sys.path.insert(0, f"{WORKDIR}/Common/Tools")
sys.path.insert(0, f"{WORKDIR}/SignalRegionStudyV4/python")
from plotter import KinematicCanvas, ComparisonCanvas, get_CoM_energy
from plotter import PALETTE_LONG as PALETTE
import srspaths
from template_utils import build_particlenet_score
import cmsstyle as CMS

with open(f"{WORKDIR}/Common/Data/Luminosity.json", "r") as _lfh:
    _LUMI_CONFIG = json.load(_lfh)

TEMPLATE_DIR = None
FITDIAG_DIR = None
FITDIAG_PATH = None
OUTPUT_DIR = None
CACHE_DIR = None
CACHE_PATH = None


def _method_segment(parsed):
    return f"{parsed.method}_blind" if getattr(parsed, "blind", False) else parsed.method


def _compute_paths():
    """Derive TEMPLATE_DIR / FITDIAG_PATH / OUTPUT_DIR / CACHE_PATH from `args`.

    Callers that import this module (e.g. plotCombinedMass) set `args` to a
    SimpleNamespace and call this to refresh the derived paths between
    masspoints.
    """
    global TEMPLATE_DIR, FITDIAG_DIR, FITDIAG_PATH
    global OUTPUT_DIR, CACHE_DIR, CACHE_PATH
    fit_channel = getattr(args, "fit_channel", "Combined") or "Combined"
    method_segment = _method_segment(args)
    source = getattr(args, "signal_source", "mc-signal") or "mc-signal"
    TEMPLATE_DIR = (f"{WORKDIR}/SignalRegionStudyV4/templates/"
                    f"{args.masspoint}/{method_segment}/{source}/"
                    f"{args.era}/{fit_channel}")
    FITDIAG_DIR = f"{TEMPLATE_DIR}/combine_output/fitdiag"
    FITDIAG_PATH = (f"{FITDIAG_DIR}/fitDiagnostics."
                    f"{args.masspoint}.{method_segment}.root")
    OUTPUT_DIR = f"{FITDIAG_DIR}/plots_mass"
    CACHE_DIR = f"{FITDIAG_DIR}/cached"
    CACHE_PATH = f"{CACHE_DIR}/fine_hists_bw{args.bin_width}.root"


def entry_setup(parsed_args, *, require_fitdiag=True, make_output_dir=True):
    """Populate module-level `args` and derived paths.

    Parameters
    ----------
    parsed_args : argparse.Namespace or SimpleNamespace
        Must expose .era, .masspoint, .method,
        .blind, .bin_width, .debug, .plot_only, and optionally
        .channel_scope / .era_scope / .fit_type.
    require_fitdiag : bool
        If True, raise when the fitDiagnostics file is missing.
    make_output_dir : bool
        If True, create OUTPUT_DIR.
    """
    global args
    args = parsed_args

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(levelname)s - %(message)s')

    _compute_paths()
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)

    if require_fitdiag and not os.path.exists(FITDIAG_PATH):
        raise FileNotFoundError(f"fitDiagnostics file not found: {FITDIAG_PATH}")
    if make_output_dir:
        os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# Plotter subclasses — override CMS style for era='All'
# =============================================================================

class _AllEraCMSStyleMixin:
    """Show 'Run 2+3, L2+L3 fb^-1' + '13/13.6 TeV' for era='All'."""

    def _configure_cms_style(self, config):
        run2_lumi = _LUMI_CONFIG["Run2"]["combined"]
        run3_lumi = _LUMI_CONFIG["Run3"]["combined"]
        run2_energy = _LUMI_CONFIG["Run2"]["energy_TeV"]
        run3_energy = _LUMI_CONFIG["Run3"]["energy_TeV"]
        CMS.SetLumi(None, run=f"Run 2+3, {run2_lumi}+{run3_lumi} fb^{{#minus1}}")
        CMS.SetEnergy(0, unit=f"{run2_energy:g}/{run3_energy:g} TeV")
        return None, "Run 2+3"


class PostfitMassComparisonCanvas(ComparisonCanvas):
    """Comparison canvas with validation-style split ratio/error-band drawing."""

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

        ratio_band = getattr(self, "ratio_band", None)
        if ratio_band is None:
            ratio_band = build_ratio_uncertainty_band(self.systematics)
            self.ratio_band = ratio_band

        CMS.cmsObjectDraw(ratio_band, "FE2", FillStyle=3004, LineWidth=0, FillColor=12, MarkerSize=0)
        CMS.cmsObjectDraw(self.ratio, "PE", MarkerStyle=ROOT.kFullCircle, MarkerSize=1.0, MarkerColor=1)
        self.canv.cd(2).RedrawAxis()


class ComparisonCanvasAll(_AllEraCMSStyleMixin, PostfitMassComparisonCanvas):
    pass


class KinematicCanvasAll(_AllEraCMSStyleMixin, KinematicCanvas):
    pass


def _build_header_text(era_scope):
    """Compose the CMS lumi + CoM header string for a given era scope."""
    if era_scope == "All":
        l2 = _LUMI_CONFIG["Run2"]["combined"]
        l3 = _LUMI_CONFIG["Run3"]["combined"]
        e2 = _LUMI_CONFIG["Run2"]["energy_TeV"]
        e3 = _LUMI_CONFIG["Run3"]["energy_TeV"]
        return f"Run 2+3, {l2}+{l3} fb^{{#minus1}} ({e2:g}/{e3:g} TeV)"
    if era_scope == "Run2":
        l = _LUMI_CONFIG["Run2"]["combined"]
        return f"Run 2, {l} fb^{{#minus1}} (13 TeV)"
    if era_scope == "Run3":
        l = _LUMI_CONFIG["Run3"]["combined"]
        return f"Run 3, {l} fb^{{#minus1}} (13.6 TeV)"
    # Per-era
    for period in ("Run2", "Run3"):
        if era_scope in _LUMI_CONFIG[period]:
            l = _LUMI_CONFIG[period][era_scope]
            com = _LUMI_CONFIG[period]["energy_TeV"]
            return f"{era_scope}, {l} fb^{{#minus1}} ({com} TeV)"
    return era_scope


def _overdraw_lumi_header(canvas, target):
    """Workaround for a cmsstyle `cmsDiCanvas` rendering quirk:

    When TLatex is drawn on a sub-TPad (like cmsDiCanvas's upper pad), ROOT
    inserts extra whitespace before the `^{-1}` superscript and shifts the
    right-aligned lumi text inward from the plot-frame edge. Plain TLatex
    on the parent CANVAS renders correctly — so we wipe the original header
    strip and redraw it in canvas-NDC coordinates.

    Triggered only for era_scope='All' or channel_scope='Combined', i.e. the
    hero plots the user actually cares about.
    """
    if target.era_scope != "All" and target.channel_scope != "Combined":
        return

    upper = canvas.cd(1)
    pad_y_lo = upper.GetYlowNDC()
    pad_y_hi = pad_y_lo + upper.GetHNDC()
    pad_x_lo = upper.GetXlowNDC()
    pad_x_hi = pad_x_lo + upper.GetWNDC()

    t = upper.GetTopMargin()
    r = upper.GetRightMargin()

    # Map sub-pad NDC (where cmsstyle placed the text) to canvas NDC.
    def sub_to_canvas(x_sub, y_sub):
        return (pad_x_lo + x_sub * (pad_x_hi - pad_x_lo),
                pad_y_lo + y_sub * (pad_y_hi - pad_y_lo))

    text_x_sub = 1 - r
    text_y_sub = 1 - t + 0.2 * t
    text_x, text_y = sub_to_canvas(text_x_sub, text_y_sub)

    erase_x1, erase_y1 = sub_to_canvas(0.0, 1.0 - t + 0.001)
    erase_x2, erase_y2 = sub_to_canvas(1.0, 1.0)

    canvas.cd()
    erase = ROOT.TPave(erase_x1, erase_y1, erase_x2, erase_y2, 0, "brNDC")
    erase.SetFillColor(ROOT.kWhite)
    erase.SetBorderSize(0)
    erase.Draw()
    canvas._lumi_erase = erase  # keep alive

    text_size_sub = 0.6 * t
    text_size_canvas = text_size_sub * (pad_y_hi - pad_y_lo)

    lt = ROOT.TLatex()
    lt.SetNDC()
    lt.SetTextFont(42)
    lt.SetTextAlign(31)
    lt.SetTextSize(text_size_canvas)
    lt.DrawLatex(text_x, text_y, _build_header_text(target.era_scope))
    canvas._lumi_latex = lt  # keep alive


# =============================================================================
# Constants
# =============================================================================

BKG_COLORS = {
    "nonprompt": PALETTE[0],
    "WZ": PALETTE[1],
    "ZZ": PALETTE[2],
    "ttW": PALETTE[3],
    "ttZ": PALETTE[4],
    "ttH": PALETTE[5],
    "tZq": PALETTE[6],
    "others": PALETTE[7],
    "conversion": PALETTE[8],
}
BKG_ORDER = ["others", "conversion", "WZ", "ZZ", "ttW", "ttH", "tZq", "ttZ", "nonprompt"]

# Channel display labels
CHANNEL_LATEX = {
    "SR1E2Mu":  "e#mu#mu",   # 1 electron + OS dimuon (A->mumu)
    "SR3Mu":    "#mu#mu#mu", # 3 muons
    "TTZ2E1Mu": "ee#mu",     # OS dielectron (Z->ee) + 1 muon
}


def masspoint_label(masspoint):
    """Convert 'MHc130_MA90' -> '(m_{H^{+}}, m_{A}) = (130, 90) GeV'.

    For CR plots the masspoint is a placeholder; show 't#bar{t}+Z CR' instead.
    """
    if args is not None and getattr(args, "method", None) == "CR":
        return "t#bar{t}+Z CR"
    try:
        mhc, ma = masspoint.split("_")
        mhc_val = mhc.replace("MHc", "")
        ma_val = ma.replace("MA", "")
        return f"(m_{{H^{{+}}}}, m_{{A}}) = ({mhc_val}, {ma_val}) GeV"
    except Exception:
        return masspoint


# =============================================================================
# Sub-channel resolution
# =============================================================================

RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]
CHANNEL_SCOPES = ["SR1E2Mu", "SR3Mu", "Combined"]


@dataclass(frozen=True)
class PlotTarget:
    era_scope: str
    channel_scope: str
    xrange: Tuple[float, float]
    hist_edges: Tuple[float, ...]
    y_title: str


def default_channel_scopes():
    if args.method == "CR":
        return [getattr(args, "fit_channel", None) or "TTZ2E1Mu"]
    return CHANNEL_SCOPES


def parse_subchannel(subch, fallback_era):
    """Return (era, channel) from a fitdiag sub-channel name.

    Handles three name shapes:
      - "era2018_SR1E2Mu", "eraRun2_era2018_SR1E2Mu" -> ("2018", "SR1E2Mu")
      - "SR1E2Mu", "TTZ2E1Mu" (per-era fit)          -> (fallback_era, channel)
      - "era2018" (CR-only era combination, channel implicit from fit_channel)
                                                      -> ("2018", args.fit_channel)
    """
    parts = subch.split("_")
    # Standard form: SR/TTZ token present
    for i, part in enumerate(parts):
        if part.startswith("SR") or part.startswith("TTZ"):
            channel = part
            if i + 1 < len(parts) and parts[i + 1] in {"Run2", "Run3", "All"}:
                return parts[i + 1], channel
            if i > 0 and parts[i - 1].startswith("era"):
                return parts[i - 1][3:], channel
            return fallback_era, channel
    # CR-style era-only (single channel per era; channel is implicit).
    # For nested forms like "eraRun2_era2016postVFP" the innermost era is the
    # last "era<X>" token in the split.
    if subch.startswith("era"):
        last_era = parts[-1]
        if last_era.startswith("era"):
            last_era = last_era[3:]
        implicit_ch = getattr(args, "fit_channel", None) or "Combined"
        return last_era, implicit_ch
    raise ValueError(f"Cannot parse sub-channel: {subch}")


def keep_by_channel(subch, scope):
    """True if sub-channel `subch` belongs to the channel scope."""
    if scope == "Combined":
        return True
    if subch.startswith(scope + "_"):
        return True
    if subch.endswith("_" + scope) or subch == scope:
        return True
    # CR-style: era-only sub-channel (e.g. "era2018") implicitly belongs to fit_channel.
    if subch.startswith("era"):
        implicit_ch = getattr(args, "fit_channel", None) or "Combined"
        return scope == implicit_ch
    return False


def keep_by_era(subch, era_scope, fit_era):
    """True if sub-channel `subch` belongs to the era scope.

    Examples:
        eraRun2_era2018_SR1E2Mu, era_scope=Run2   -> True
        eraRun2_era2018_SR1E2Mu, era_scope=2018   -> True
        eraRun2_era2018_SR1E2Mu, era_scope=2017   -> False
        era2018_SR1E2Mu,         era_scope=Run2   -> True  (parent fit is Run2)
        SR1E2Mu,                 era_scope=2018   -> True  (per-era fit)
    """
    if era_scope == "All":
        return True
    sub_era, _ = parse_subchannel(subch, fit_era)
    if era_scope == "Run2":
        return sub_era in RUN2_ERAS
    if era_scope == "Run3":
        return sub_era in RUN3_ERAS
    return sub_era == era_scope


def applicable_era_scopes(fit_era):
    """Return era scopes we iterate for a given fit source."""
    if fit_era == "All":
        return ["All", "Run2", "Run3"] + RUN2_ERAS + RUN3_ERAS
    if fit_era == "Run2":
        return ["Run2"] + RUN2_ERAS
    if fit_era == "Run3":
        return ["Run3"] + RUN3_ERAS
    return [fit_era]


def discover_channels(f):
    prefit = f.Get("shapes_prefit")
    if not prefit:
        raise RuntimeError("shapes_prefit not found in fitDiagnostics file")
    out = []
    for key in prefit.GetListOfKeys():
        obj = prefit.Get(key.GetName())
        if obj and obj.InheritsFrom("TDirectory"):
            out.append(key.GetName())
    return out


def load_subchannel_config(era, channel):
    """Load per-(era, channel) binning, threshold, bg-weights, process list."""
    source = getattr(args, "signal_source", "mc-signal") or "mc-signal"
    tdir = (f"{WORKDIR}/SignalRegionStudyV4/templates/{args.masspoint}/"
            f"{_method_segment(args)}/{source}/{era}/{channel}")
    binning = json.load(open(f"{tdir}/binning.json"))
    plist = json.load(open(f"{tdir}/process_list.json"))
    category_key = f"{channel}_{era}"

    def category_payload(payload, filename):
        if "categories" not in payload:
            return payload
        categories = payload["categories"]
        if category_key in categories:
            return categories[category_key]
        if len(categories) == 1:
            return next(iter(categories.values()))
        raise KeyError(f"{filename}: no category '{category_key}' in {tdir}")

    binning_payload = category_payload(binning, "binning.json")
    threshold = None
    upper_threshold = None
    bg_weights = None
    if args.method == "ParticleNet":
        thr_path = f"{tdir}/threshold.json"
        bw_path = f"{tdir}/background_weights.json"
        if os.path.exists(thr_path):
            thr = json.load(open(thr_path))
            thr_payload = category_payload(thr, "threshold.json")
            threshold = thr_payload.get("threshold")
            upper_threshold = thr_payload.get("upper_threshold")
        if os.path.exists(bw_path):
            bw_payload = category_payload(json.load(open(bw_path)), "background_weights.json")
            bg_weights = bw_payload["weights"]
    return {
        "template_dir": tdir,
        "category_key": category_key,
        "era": era,
        "channel": channel,
        "mass_min": binning_payload["mass_min"],
        "mass_max": binning_payload["mass_max"],
        "bin_edges": binning_payload["bin_edges"],
        "sigma_eff": binning_payload.get("sigma_eff", 1.0),
        "threshold": threshold,
        "upper_threshold": upper_threshold,
        "bg_weights": bg_weights,
        "separate_processes": plist["separate_processes"],
        "merged_to_others": plist["merged_to_others"],
        "components": [
            item for item in plist.get("components", [])
            if item.get("category") == category_key
        ],
        "physics_groups": plist.get("physics_groups", {}),
    }


# =============================================================================
# Unbinned fill from preprocessed trees
# =============================================================================

def sample_path(era, channel, process):
    base_method = args.method[:-6] if args.method.endswith("_blind") else args.method
    return os.path.join(
        srspaths.sample_dir(era, channel, args.masspoint, base_method),
        f"{process}.root")


def build_filtered_rdf(sample_file, cfg):
    """RDataFrame on Central tree with mass window (+score cut if PN)."""
    if not os.path.exists(sample_file):
        return None, None
    test = ROOT.TFile.Open(sample_file)
    if not test or test.IsZombie():
        return None, None
    tree = test.Get("Central")
    if not tree or tree.GetEntries() == 0:
        test.Close()
        return None, None
    branches = [b.GetName() for b in tree.GetListOfBranches()]
    test.Close()

    rdf = ROOT.RDataFrame("Central", sample_file)
    rdf = rdf.Filter(f"mass >= {cfg['mass_min']} && mass <= {cfg['mass_max']}")
    if args.method == "ParticleNet":
        score_sig = f"score_{args.masspoint}_signal"
        if score_sig not in branches:
            return None, branches
        formula = build_particlenet_score(args.masspoint, cfg["bg_weights"])
        rdf = rdf.Define("score_PN", formula)
        if cfg["upper_threshold"] is not None:
            rdf = rdf.Filter(f"score_PN < {cfg['upper_threshold']}")
        elif cfg["threshold"] is not None:
            rdf = rdf.Filter(f"score_PN >= {cfg['threshold']}")
    return rdf, branches


def fill_fine_hist(era, channel, process, cfg, uniform_edges, name, is_data=False):
    """Fill a fine-bin TH1D from the Central tree of one sample file."""
    sp = sample_path(era, channel, process)
    rdf, _ = build_filtered_rdf(sp, cfg)
    nbins = len(uniform_edges) - 1
    edges_arr = array('d', uniform_edges)
    if rdf is None:
        h = ROOT.TH1D(name, "", nbins, edges_arr)
        h.SetDirectory(0)
        return h
    if is_data:
        result = rdf.Histo1D((name, "", nbins, edges_arr), "mass")
    else:
        result = rdf.Histo1D((name, "", nbins, edges_arr), "mass", "weight")
    hist = result.GetValue()
    hist.SetDirectory(0)
    return hist


# =============================================================================
# Post-fit scaling from fitDiagnostics per coarse bin
# =============================================================================

SCALE_FLOOR_FRAC = 1e-3  # bins below this fraction of the process total use
                         # the global post/pre scale instead of the per-bin one.


def get_coarse_scale(fitdiag_file, channel_key, proc_key, fit_type, n_coarse):
    """Return per-coarse-bin scales from shapes_prefit -> shapes_fit_{b|s}.

    For each coarse bin we normally use `post/pre` (per-bin post-fit pull).
    Bins with pre-fit content below SCALE_FLOOR_FRAC * total are treated as
    floored / meaningless — the per-bin ratio there can explode (e.g. pre=1e-9
    vs post=7e-4 giving 7e5) and pollute the tree-filled fine grid. For those
    we fall back to the overall process scale (post_total / pre_total), which
    carries the fit's real normalization pull.

    Returns a list of length n_coarse; 1.0 as a neutral fallback if either
    histogram is missing.
    """
    pre_h = fitdiag_file.Get(f"shapes_prefit/{channel_key}/{proc_key}")
    post_h = fitdiag_file.Get(f"shapes_fit_{fit_type}/{channel_key}/{proc_key}")
    scales = [1.0] * n_coarse
    if not pre_h or not post_h:
        return scales

    pre_total = pre_h.Integral()
    post_total = post_h.Integral()
    global_scale = post_total / pre_total if pre_total > 0 else 0.0
    min_meaningful = SCALE_FLOOR_FRAC * pre_total if pre_total > 0 else 0.0

    for i in range(n_coarse):
        pre = pre_h.GetBinContent(i + 1)
        post = post_h.GetBinContent(i + 1)
        if pre > min_meaningful:
            scales[i] = post / pre
        else:
            scales[i] = global_scale
    return scales


def apply_coarse_scale(fine_hist, scales, mass_edges):
    """Multiply fine bins by the post/pre scale of the coarse bin they fall in.

    Fine bins outside the coarse range are clamped to the nearest edge bin's
    scale rather than skipped, so the b-only r=0 suppression propagates to all
    fine bins (not just those strictly inside the coarse window).
    """
    n_coarse = len(mass_edges) - 1
    for j in range(1, fine_hist.GetNbinsX() + 1):
        x = fine_hist.GetBinCenter(j)
        idx = bisect.bisect_right(mass_edges, x) - 1
        if idx < 0:
            idx = 0
        elif idx >= n_coarse:
            idx = n_coarse - 1
        s = scales[idx]
        fine_hist.SetBinContent(j, fine_hist.GetBinContent(j) * s)
        fine_hist.SetBinError(j, fine_hist.GetBinError(j) * abs(s))


# =============================================================================
# Plot config helpers
# =============================================================================

def scope_label(scope):
    if scope in CHANNEL_LATEX:
        return CHANNEL_LATEX[scope]
    return f"{CHANNEL_LATEX['SR1E2Mu']} + {CHANNEL_LATEX['SR3Mu']}"


def make_canvas_config(era_scope, extra):
    """Build a canvas config dict for a given era scope.

    era_scope drives the CMS lumi label and CoM-energy string.
    """
    base = {}
    if era_scope == "All":
        base["era"] = "Run2"  # placeholder; mixin overrides CMS style
        base["CoM"] = _LUMI_CONFIG["Run3"]["energy_TeV"]
    else:
        base["era"] = era_scope
        base["CoM"] = get_CoM_energy(era_scope)
    base["legend"] = [0.65, 0.89 - 0.05 * 7, 0.94, 0.89]
    # Font 42 (Helvetica, TLatex-aware) so #mu renders as Greek mu; default 61
    # is precision-1 and would show "#mu" literally.
    base["channelFont"] = 42
    base.update(extra)
    return base


def select_comparison_cls(era_scope):
    return ComparisonCanvasAll if era_scope == "All" else PostfitMassComparisonCanvas


def select_kinematic_cls(era_scope):
    return KinematicCanvasAll if era_scope == "All" else KinematicCanvas


# =============================================================================
# Main
# =============================================================================

def build_uniform_edges(mass_lo, mass_hi, bin_width):
    lo = math.floor(mass_lo)
    hi = math.ceil(mass_hi)
    if hi <= lo:
        hi = lo + bin_width
    n = max(1, int(round((hi - lo) / bin_width)))
    return [lo + i * bin_width for i in range(n + 1)]


def is_atomic_template_target(era_scope, channel_scope):
    """Use original template bins only for single era x single SR channel."""
    return era_scope not in ("All", "Run2", "Run3") and channel_scope in ("SR1E2Mu", "SR3Mu")


def edge_cache_id(edges):
    payload = ",".join(f"{float(x):.8g}" for x in edges)
    return hashlib.md5(payload.encode("ascii")).hexdigest()[:12]


def target_y_title(target):
    return target.y_title


# =============================================================================
# Plot-drawing helpers
# =============================================================================

def build_ratio_uncertainty_band(prediction):
    """Return prediction uncertainty as a ratio-band centered at 1."""
    ratio_band = prediction.Clone(f"{prediction.GetName()}_ratio_uncertainty")
    ratio_band.SetDirectory(0)
    for ibin in range(1, ratio_band.GetNbinsX() + 1):
        nominal = prediction.GetBinContent(ibin)
        if nominal > 0:
            ratio_band.SetBinContent(ibin, 1.0)
            ratio_band.SetBinError(ibin, prediction.GetBinError(ibin) / nominal)
        else:
            ratio_band.SetBinContent(ibin, 0.0)
            ratio_band.SetBinError(ibin, 0.0)
    return ratio_band


def apply_uncertainty_hist(target, uncertainty):
    """Copy bin uncertainties from `uncertainty` onto `target`."""
    if uncertainty is None:
        return
    if target.GetNbinsX() != uncertainty.GetNbinsX():
        logging.warning("Cannot apply uncertainty band: incompatible bin counts")
        return
    for ibin in range(1, target.GetNbinsX() + 1):
        target.SetBinError(ibin, uncertainty.GetBinError(ibin))


def _blinding_label():
    # Full unblind (default): no extra label needed.
    return ""


def _make_stack(target, agg_data, agg_bkgs, agg_signal, label_top, out_path,
                uncertainty_hist=None):
    """Generic stack-plot builder shared by pre-fit and post-fit plots."""
    if not agg_bkgs:
        logging.warning(f"No backgrounds; skipping {out_path}")
        return
    agg_data.SetTitle("data")
    colors = [BKG_COLORS.get(b, ROOT.kGray) for b in agg_bkgs.keys()]

    # Compute stack total to derive y-max and trim empty edge bins.
    ref_h = next(iter(agg_bkgs.values()))
    stack_total = ref_h.Clone("_stack_total_tmp")
    stack_total.SetDirectory(0)
    for h in list(agg_bkgs.values())[1:]:
        stack_total.Add(h)
    stack_max = stack_total.GetMaximum()
    sig_max = (agg_signal.GetMaximum()
               if agg_signal is not None and agg_signal.Integral() > 0 else 0.0)
    data_max = agg_data.GetMaximum() if agg_data.Integral() > 0 else 0.0
    y_max = max(stack_max, sig_max, data_max) * 2

    x_lo, x_hi = target.xrange

    config = make_canvas_config(target.era_scope, {
        "channel": masspoint_label(args.masspoint),
        "channelPosY": 0.58,
        "channelSize": 0.04,
        "xTitle": "M(e^{+}e^{-}) [GeV]" if args.method == "CR" else "M(#mu^{+}#mu^{-}) [GeV]",
        "yTitle": target_y_title(target),
        "xRange": [x_lo, x_hi],
        "yRange": [0, y_max],
        "rTitle": "Data / Pred",
        "rRange": [0, 5],
        "maxDigits": 3,
        "systSrc": "Stat+Syst",
        "legend": [0.5, 0.62, 0.99, 0.89],
        "legendColumns": 2,
        "legendTextSize": 0.035,
        "colors": colors,
    })
    plotter = select_comparison_cls(target.era_scope)(agg_data, agg_bkgs, config)
    apply_uncertainty_hist(plotter.systematics, uncertainty_hist)
    plotter.drawPadUp()

    if agg_signal is not None:
        plotter.canv.cd(1)
        agg_signal.SetLineColor(ROOT.kBlack)
        agg_signal.SetLineWidth(2)
        agg_signal.SetLineStyle(1)
        # Draw curve only if positive integral; legend entry is always added
        # so post-fit yield is visible (r=0 B-only or negative S+B).
        if agg_signal.Integral() > 0:
            agg_signal.Draw("HIST SAME")
        for obj in ROOT.gPad.GetListOfPrimitives():
            if obj.InheritsFrom("TLegend"):
                obj.AddEntry(agg_signal, f"Signal ({agg_signal.Integral():.1f})", "l")
                break

    plotter.drawPadDown()
    plotter.canv.cd()
    CMS.drawText(scope_label(target.channel_scope),
                 posX=0.2, posY=0.8, font=42, align=0, size=0.04)
    blind = _blinding_label()
    line = label_top + (f" ({blind})" if blind else "")
    CMS.drawText(line, posX=0.2, posY=0.76, font=62, align=0, size=0.032)

    _overdraw_lumi_header(plotter.canv, target)

    plotter.canv.SaveAs(out_path)
    logging.info(f"Saved: {out_path}")
    sig_int = agg_signal.Integral() if agg_signal else 0.0
    logging.info(f"  total_bkg: {sum(h.Integral() for h in agg_bkgs.values()):.2f}  "
                 f"data: {agg_data.Integral():.0f}  signal: {sig_int:.2f}")


def make_postfit_stack(target, agg_data, post_bkgs, post_signal, fit_type,
                       uncertainty_hist=None):
    fit_label = "B-only" if fit_type == "b" else "S+B"
    out = f"{OUTPUT_DIR}/postfit_{fit_type}_mass_{target.era_scope}_{target.channel_scope}.png"
    _make_stack(target, agg_data, post_bkgs, post_signal,
                label_top=f"Post-fit {fit_label}",
                out_path=out,
                uncertainty_hist=uncertainty_hist)


def make_prefit_stack(target, agg_data, pre_bkgs, pre_signal,
                      uncertainty_hist=None):
    out = f"{OUTPUT_DIR}/prefit_mass_{target.era_scope}_{target.channel_scope}.png"
    _make_stack(target, agg_data, pre_bkgs, pre_signal,
                label_top="Pre-fit",
                out_path=out,
                uncertainty_hist=uncertainty_hist)


def make_prefit_vs_postfit(target, agg_total_pre, agg_total_post, fit_type):
    if agg_total_pre.Integral() <= 0:
        return
    pre_int = agg_total_pre.Integral()
    post_int = agg_total_post.Integral()
    fit_label = "B-only" if fit_type == "b" else "S+B"
    config = make_canvas_config(target.era_scope, {
        "channel": masspoint_label(args.masspoint),
        "channelPosY": 0.64,
        "channelSize": 0.035,
        "xTitle": "M(e^{+}e^{-}) [GeV]" if args.method == "CR" else "M(#mu^{+}#mu^{-}) [GeV]",
        "yTitle": target_y_title(target),
        "xRange": list(target.xrange),
        "maxDigits": 3,
    })
    hists = {
        f"Pre-fit ({pre_int:.1f})": agg_total_pre,
        f"Post-fit {fit_label} ({post_int:.1f})": agg_total_post,
    }
    plotter = select_kinematic_cls(target.era_scope)(hists, config)
    plotter.drawPad()
    plotter.canv.cd()
    CMS.drawText(scope_label(target.channel_scope),
                 posX=0.2, posY=0.60, font=42, align=0, size=0.03)
    blind = _blinding_label()
    post_tag = f"Post-fit {fit_label}" + (f" ({blind})" if blind else "")
    CMS.drawText(post_tag, posX=0.2, posY=0.56, font=62, align=0, size=0.03)
    ratio = post_int / pre_int if pre_int > 0 else 0
    CMS.drawText(f"Post/Pre = {ratio:.3f}", posX=0.2, posY=0.52,
                 font=42, align=0, size=0.03)

    out = f"{OUTPUT_DIR}/prefit_vs_postfit_{fit_type}_mass_{target.era_scope}_{target.channel_scope}.png"
    plotter.canv.SaveAs(out)
    logging.info(f"Saved: {out}")


def sum_total(hists, hist_edges, name):
    edges_arr = array('d', hist_edges)
    total = ROOT.TH1D(name, "", len(hist_edges) - 1, edges_arr)
    total.SetDirectory(0)
    for h in hists.values():
        total.Add(h)
    return total


def load_run_period_metadata():
    categories_path = f"{TEMPLATE_DIR}/categories.json"
    if not os.path.exists(categories_path):
        return None
    metadata = {
        "categories": json.load(open(categories_path))["categories"],
        "process_list": json.load(open(f"{TEMPLATE_DIR}/process_list.json")),
        "binning": json.load(open(f"{TEMPLATE_DIR}/binning.json"))["categories"],
    }
    threshold_path = f"{TEMPLATE_DIR}/threshold.json"
    weights_path = f"{TEMPLATE_DIR}/background_weights.json"
    metadata["threshold"] = (
        json.load(open(threshold_path)).get("categories", {})
        if os.path.exists(threshold_path) else {}
    )
    metadata["background_weights"] = (
        json.load(open(weights_path)).get("categories", {})
        if os.path.exists(weights_path) else {}
    )
    return metadata


def split_run_period_category(category):
    for suffix in ["_Run2", "_Run3"]:
        if category.endswith(suffix):
            return category[:-len(suffix)], suffix[1:]
    parts = category.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return category, None


def keep_run_period_category(category, era_scope, channel_scope):
    channel, period = split_run_period_category(category)
    if channel_scope != "Combined" and channel != channel_scope:
        return False
    if era_scope == "All":
        return True
    return period == era_scope


def clone_empty_like(edges, name):
    hist = ROOT.TH1D(name, "", len(edges) - 1, array("d", edges))
    hist.SetDirectory(0)
    return hist


def display_edges_for_run_period(era_scope, kept, binning):
    return tuple(float(x) for x in binning[kept[0]]["bin_edges"])


def should_refill_run_period_plot(era_scope, channel_scope, kept):
    """Use V2-style exact sample refilling whenever a plot combines categories."""
    return len(kept) > 1 or channel_scope == "Combined" or era_scope == "All"


def resolve_run_period_refill_edges(kept, binning):
    source_edges_by_cat = {
        cat: tuple(float(x) for x in binning[cat]["bin_edges"])
        for cat in kept
    }
    lo = min(edges[0] for edges in source_edges_by_cat.values())
    hi = max(edges[-1] for edges in source_edges_by_cat.values())
    if args.method == "CR":
        # Match the V2 TTZ CR display: integer-snapped 1 GeV bins. For ZWin
        # [81.2, 101.2], this gives 21 bins over [81, 102].
        bin_width = 1.0
    elif args.bin_width == "auto":
        widest = max(source_edges_by_cat.values(), key=lambda edges: edges[-1] - edges[0])
        bin_width = (widest[-1] - widest[0]) / 20.0
        logging.info(
            "  Auto refill bin width: widest range [%.3f, %.3f] / 20 -> %.4f GeV",
            widest[0], widest[-1], bin_width,
        )
    else:
        bin_width = float(args.bin_width)
    return tuple(build_uniform_edges(lo, hi, bin_width)), bin_width


def resolve_run_period_auto_bin_width(binning):
    """Match the V2 auto-bin convention: widest mass window in the fit / 20."""
    widest = max(
        binning.values(),
        key=lambda c: float(c.get("mass_max", c["bin_edges"][-1])) - float(c.get("mass_min", c["bin_edges"][0])),
    )
    mass_min = float(widest.get("mass_min", widest["bin_edges"][0]))
    mass_max = float(widest.get("mass_max", widest["bin_edges"][-1]))
    width = (mass_max - mass_min) / 20.0
    logging.info(
        "  Auto bin width: widest range [%.3f, %.3f] / 20 -> %.4f GeV",
        mass_min, mass_max, width,
    )
    return width


def get_fit_shape_hist(fitdiag_file, fit_type, category, process):
    folder = "shapes_prefit" if fit_type == "prefit" else f"shapes_fit_{fit_type}"
    return fitdiag_file.Get(f"{folder}/{category}/{process}")


def coarse_bin_index(edges, x):
    idx = bisect.bisect_right(edges, x) - 1
    if idx < 0:
        return 0
    if idx >= len(edges) - 1:
        return len(edges) - 2
    return idx


def build_fitdiag_uncertainty_hist(fitdiag_file, fit_type, target_edges,
                                   category_totals, source_edges_by_category,
                                   name):
    """Map FitDiagnostics total-background errors onto the plotted mass bins.

    The refilled mass plots use exact sample histograms on display bins that can
    differ from Combine's template bins. FitDiagnostics stores the post/pre-fit
    uncertainty on the original template-bin axis, so use the per-category
    total-background relative uncertainty in the matched coarse bin and apply it
    to that category's displayed background content. Multiple categories in one
    plot are added in quadrature for display.
    """
    out = clone_empty_like(target_edges, name)
    for category, category_total in category_totals.items():
        total_bkg = get_fit_shape_hist(
            fitdiag_file, fit_type, category, "total_background")
        if not total_bkg:
            logging.warning(
                "Missing total_background for %s/%s; keeping histogram errors",
                category, fit_type,
            )
            for ibin in range(1, out.GetNbinsX() + 1):
                out.SetBinContent(
                    ibin,
                    out.GetBinContent(ibin) + category_total.GetBinContent(ibin),
                )
                out.SetBinError(
                    ibin,
                    math.hypot(out.GetBinError(ibin), category_total.GetBinError(ibin)),
                )
            continue

        source_edges = source_edges_by_category[category]
        for ibin in range(1, out.GetNbinsX() + 1):
            content = category_total.GetBinContent(ibin)
            x = category_total.GetXaxis().GetBinCenter(ibin)
            coarse_idx = coarse_bin_index(source_edges, x) + 1
            nominal = total_bkg.GetBinContent(coarse_idx)
            rel_unc = total_bkg.GetBinError(coarse_idx) / nominal if nominal > 0 else 0.0
            out.SetBinContent(ibin, out.GetBinContent(ibin) + content)
            out.SetBinError(
                ibin,
                math.hypot(out.GetBinError(ibin), abs(content) * rel_unc),
            )
    return out


def add_binwise(target, source, source_edges=None):
    """Add a Combine shape to a physical-axis histogram by bin index.

    FitDiagnostics keeps shape histograms on a 0..N bin-number axis. The V3
    plotting target uses the original mass edges. In combined workspaces,
    Combine pads shorter categories to the maximum category bin count; the
    padded tail is outside ``source_edges`` and must be ignored.
    """
    if source_edges is not None:
        target_edges = [target.GetXaxis().GetBinLowEdge(1)]
        target_edges += [
            target.GetXaxis().GetBinUpEdge(ibin)
            for ibin in range(1, target.GetNbinsX() + 1)
        ]
        source_edges = tuple(float(x) for x in source_edges)
        target_edges = tuple(float(x) for x in target_edges)
        n_physical = min(source.GetNbinsX(), len(source_edges) - 1)

        if source_edges == target_edges:
            for ibin in range(1, min(n_physical, target.GetNbinsX()) + 1):
                val = target.GetBinContent(ibin) + source.GetBinContent(ibin)
                err = math.hypot(target.GetBinError(ibin), source.GetBinError(ibin))
                target.SetBinContent(ibin, val)
                target.SetBinError(ibin, err)
            return

        for src_bin in range(1, n_physical + 1):
            src_lo = source_edges[src_bin - 1]
            src_hi = source_edges[src_bin]
            width = src_hi - src_lo
            if width <= 0:
                continue
            val = source.GetBinContent(src_bin)
            err = source.GetBinError(src_bin)
            for dst_bin in range(1, target.GetNbinsX() + 1):
                dst_lo = target.GetXaxis().GetBinLowEdge(dst_bin)
                dst_hi = target.GetXaxis().GetBinUpEdge(dst_bin)
                overlap = max(0.0, min(src_hi, dst_hi) - max(src_lo, dst_lo))
                if overlap <= 0:
                    continue
                frac = overlap / width
                target.SetBinContent(
                    dst_bin, target.GetBinContent(dst_bin) + val * frac
                )
                target.SetBinError(
                    dst_bin, math.hypot(target.GetBinError(dst_bin), err * frac)
                )
        return

    if source.GetNbinsX() != target.GetNbinsX():
        target.Add(source)
        return
    for ibin in range(1, target.GetNbinsX() + 1):
        val = target.GetBinContent(ibin) + source.GetBinContent(ibin)
        err = math.hypot(target.GetBinError(ibin), source.GetBinError(ibin))
        target.SetBinContent(ibin, val)
        target.SetBinError(ibin, err)


def sum_component_group(fitdiag_file, fit_type, category, components, edges, out_name,
                        source_edges=None):
    out = clone_empty_like(edges, out_name)
    for process in dict.fromkeys(components):
        hist = get_fit_shape_hist(fitdiag_file, fit_type, category, process)
        if hist:
            add_binwise(out, hist, source_edges)
    return out


def component_channel(component, category_payload):
    return category_payload.get("channel", getattr(args, "fit_channel", "TTZ2E1Mu"))


def run_period_category_cfg(metadata, category):
    cat_binning = metadata["binning"][category]
    bin_edges = tuple(float(x) for x in cat_binning["bin_edges"])
    threshold_payload = metadata.get("threshold", {}).get(category, {})
    weights_payload = metadata.get("background_weights", {}).get(category, {})
    return {
        "mass_min": float(cat_binning.get("mass_min", bin_edges[0])),
        "mass_max": float(cat_binning.get("mass_max", bin_edges[-1])),
        "bin_edges": bin_edges,
        "threshold": threshold_payload.get("threshold"),
        "upper_threshold": threshold_payload.get("upper_threshold"),
        "bg_weights": weights_payload.get("weights"),
    }


def cached_run_period_fine(category, subera, channel, process, cfg, edges, name,
                           is_data=False):
    """Category-aware fine-hist cache for run-period component refills."""
    edge_id = edge_cache_id(edges)
    subch = f"{category}:{subera}:{channel}"
    key = (edge_id, subch, process, is_data)
    if key not in _FINE_CACHE:
        if args.plot_only:
            raise KeyError(
                f"Cached run-period hist not found in --plot-only mode: {key}. "
                f"Re-run without --plot-only to rebuild the cache.")
        _FINE_CACHE[key] = fill_fine_hist(
            subera, channel, process, cfg, edges, name, is_data=is_data)
    return _FINE_CACHE[key]


def fill_component_from_samples(component, category_payload, edges, cfg, name):
    if component.get("dummy_signal"):
        return clone_empty_like(edges, name)
    process = component["base_process"]
    if component.get("is_signal") or process == "signal":
        process = args.masspoint
    channel = component_channel(component, category_payload)
    return cached_run_period_fine(
        component["category"], component["subera"], channel, process,
        cfg, edges, name, is_data=False,
    )


def fill_data_from_subera(category, subera, channel, edges, cfg, name):
    return cached_run_period_fine(
        category, subera, channel, "data", cfg, edges, name, is_data=True)


def make_refilled_run_period_plot(fitdiag_file, metadata, kept, era_scope,
                                  channel_scope, fit_types):
    """Draw combined run-period/category views from exact subera sample trees.

    This mirrors the V2 postfit-mass workflow: refill each component from the
    preprocessed samples on a common display axis, then apply the corresponding
    FitDiagnostics post/pre scale in the component's original template bins.
    """
    categories = metadata["categories"]
    binning = metadata["binning"]
    source_edges_by_cat = {
        cat: tuple(float(x) for x in binning[cat]["bin_edges"])
        for cat in kept
    }
    cfg_by_cat = {
        cat: run_period_category_cfg(metadata, cat)
        for cat in kept
    }
    edges, bin_width = resolve_run_period_refill_edges(kept, binning)
    y_title = "Events / 1 GeV" if args.method == "CR" else f"Events / {round(bin_width, 2):.2g} GeV"
    target = PlotTarget(
        era_scope=era_scope,
        channel_scope=channel_scope,
        xrange=(edges[0], edges[-1]),
        hist_edges=edges,
        y_title=y_title,
    )

    components = []
    seen_data = set()
    for cat in kept:
        payload = categories[cat]
        for component in payload["processes"]:
            item = dict(component)
            item["category"] = cat
            item["channel"] = payload.get("channel", channel_scope)
            components.append(item)
        for subera in payload.get("suberas", []):
            seen_data.add((cat, subera, payload.get("channel", channel_scope)))

    bkg_groups = [g for g in BKG_ORDER if any(c.get("physics_group") == g for c in components)]
    pre_bkgs = {group: clone_empty_like(edges, f"{group}_{era_scope}_{channel_scope}_prefit")
                for group in bkg_groups}
    post_bkgs = {
        fit_type: {group: clone_empty_like(edges, f"{group}_{era_scope}_{channel_scope}_{fit_type}")
                   for group in bkg_groups}
        for fit_type in fit_types
    }
    pre_signal = clone_empty_like(edges, f"signal_{era_scope}_{channel_scope}_prefit")
    post_signal = {fit_type: clone_empty_like(edges, f"signal_{era_scope}_{channel_scope}_{fit_type}")
                   for fit_type in fit_types}
    pre_category_totals = {
        cat: clone_empty_like(edges, f"{cat}_{era_scope}_{channel_scope}_prefit_total")
        for cat in kept
    }
    post_category_totals = {
        fit_type: {
            cat: clone_empty_like(edges, f"{cat}_{era_scope}_{channel_scope}_{fit_type}_total")
            for cat in kept
        }
        for fit_type in fit_types
    }

    for component in components:
        group = component.get("physics_group")
        source_edges = source_edges_by_cat[component["category"]]
        cfg = cfg_by_cat[component["category"]]
        fine = fill_component_from_samples(
            component, categories[component["category"]], edges,
            cfg,
            f"{component['name']}_{era_scope}_{channel_scope}_refill"
        )
        if group == "signal":
            pre_signal.Add(fine)
        elif group in pre_bkgs:
            pre_bkgs[group].Add(fine)
            pre_category_totals[component["category"]].Add(fine)
        else:
            continue

        for fit_type in fit_types:
            post = fine.Clone(f"{component['name']}_{era_scope}_{channel_scope}_{fit_type}_refill")
            post.SetDirectory(0)
            scales = get_coarse_scale(
                fitdiag_file, component["category"], component["name"],
                fit_type, len(source_edges) - 1,
            )
            apply_coarse_scale(post, scales, source_edges)
            if group == "signal":
                post_signal[fit_type].Add(post)
            else:
                post_bkgs[fit_type][group].Add(post)
                post_category_totals[fit_type][component["category"]].Add(post)

    data = clone_empty_like(edges, f"data_{era_scope}_{channel_scope}")
    if args.method != "CR" and args.blind:
        for hist in pre_bkgs.values():
            data.Add(hist)
    else:
        for cat, subera, channel in sorted(seen_data):
            data.Add(fill_data_from_subera(
                cat, subera, channel, edges, cfg_by_cat[cat],
                f"data_{subera}_{channel}_{era_scope}_{channel_scope}_refill",
            ))
    data.SetTitle("data")

    pre_bkgs = {group: hist for group, hist in pre_bkgs.items() if hist.Integral() > 0}
    pre_uncertainty = build_fitdiag_uncertainty_hist(
        fitdiag_file, "prefit", edges, pre_category_totals, source_edges_by_cat,
        f"prefit_uncertainty_{era_scope}_{channel_scope}",
    )
    make_prefit_stack(target, data, pre_bkgs, pre_signal, pre_uncertainty)
    pre_total = sum_total(pre_bkgs, edges, f"prefit_total_{era_scope}_{channel_scope}")

    for fit_type in fit_types:
        post_group = {
            group: hist for group, hist in post_bkgs[fit_type].items()
            if hist.Integral() > 0
        }
        post_uncertainty = build_fitdiag_uncertainty_hist(
            fitdiag_file, fit_type, edges, post_category_totals[fit_type],
            source_edges_by_cat,
            f"postfit_uncertainty_{fit_type}_{era_scope}_{channel_scope}",
        )
        make_postfit_stack(
            target, data, post_group, post_signal[fit_type], fit_type,
            post_uncertainty,
        )
        post_total = sum_total(post_group, edges, f"postfit_total_{era_scope}_{channel_scope}_{fit_type}")
        make_prefit_vs_postfit(target, pre_total, post_total, fit_type)


def make_run_period_postfit_plots(fitdiag_file, fit_era, fit_types):
    metadata = load_run_period_metadata()
    if not metadata:
        return False

    logging.info("Detected run_period_components metadata; drawing grouped category plots")
    categories = metadata["categories"]
    process_list = metadata["process_list"]
    binning = metadata["binning"]
    shapes_file = ROOT.TFile.Open(f"{TEMPLATE_DIR}/shapes.root", "READ")
    if not shapes_file or shapes_file.IsZombie():
        raise RuntimeError(f"Failed to open {TEMPLATE_DIR}/shapes.root")

    era_scopes = [args.era_scope] if args.era_scope else applicable_era_scopes(fit_era)
    channel_scopes = [args.channel_scope] if args.channel_scope else default_channel_scopes()
    refill_targets = []
    for era_scope in era_scopes:
        for channel_scope in channel_scopes:
            kept = [
                cat for cat in categories
                if keep_run_period_category(cat, era_scope, channel_scope)
            ]
            if kept and should_refill_run_period_plot(era_scope, channel_scope, kept):
                refill_targets.append((era_scope, channel_scope, kept))

    if refill_targets and args.method != "CR":
        if args.bin_width == "auto":
            args.bin_width = resolve_run_period_auto_bin_width(binning)
        else:
            args.bin_width = float(args.bin_width)

    global CACHE_PATH
    CACHE_PATH = f"{CACHE_DIR}/mass_hists_run_period_bw{args.bin_width}.root"
    if refill_targets:
        if args.plot_only:
            if not load_cache_from_file(CACHE_PATH):
                raise FileNotFoundError(
                    f"--plot-only requires a cache file at {CACHE_PATH}. "
                    f"Run without --plot-only first to build the cache.")
        else:
            logging.info("  Run-period refill cache: %s", CACHE_PATH)

    for era_scope in era_scopes:
        for channel_scope in channel_scopes:
            kept = [
                cat for cat in categories
                if keep_run_period_category(cat, era_scope, channel_scope)
            ]
            if not kept:
                continue
            if should_refill_run_period_plot(era_scope, channel_scope, kept):
                make_refilled_run_period_plot(
                    fitdiag_file, metadata, kept, era_scope, channel_scope, fit_types
                )
                continue
            source_edges_by_cat = {
                cat: tuple(float(x) for x in binning[cat]["bin_edges"])
                for cat in kept
            }
            same_source_edges = all(source_edges_by_cat[cat] == source_edges_by_cat[kept[0]]
                                    for cat in kept)
            if not same_source_edges:
                logging.warning(
                    "Skipping grouped %s/%s plot: category binnings differ (%s)",
                    era_scope, channel_scope, ", ".join(kept),
                )
                continue
            edges = display_edges_for_run_period(era_scope, kept, binning)
            y_title = "Events / bin"

            target = PlotTarget(
                era_scope=era_scope,
                channel_scope=channel_scope,
                xrange=(edges[0], edges[-1]),
                hist_edges=edges,
                y_title=y_title,
            )
            data = clone_empty_like(edges, f"data_{era_scope}_{channel_scope}")
            groups_for_cat = {}
            bkg_union = []
            for cat in kept:
                cat_dir = shapes_file.Get(cat)
                if not cat_dir:
                    raise RuntimeError(f"Missing category '{cat}' in {TEMPLATE_DIR}/shapes.root")
                cat_data = cat_dir.Get("data_obs")
                if not cat_data:
                    raise RuntimeError(f"Missing data_obs for category '{cat}'")
                add_binwise(data, cat_data, source_edges_by_cat[cat])

                cat_groups = {}
                cat_process_names = {meta["name"] for meta in categories[cat]["processes"]}
                for group, members in process_list.get("physics_groups", {}).items():
                    cat_members = [
                        proc for proc in dict.fromkeys(members)
                        if proc in cat_process_names
                    ]
                    if cat_members:
                        cat_groups[group] = cat_members
                        if group in BKG_ORDER and group not in bkg_union:
                            bkg_union.append(group)
                groups_for_cat[cat] = cat_groups

            bkg_groups = [g for g in BKG_ORDER if g in bkg_union]
            pre_bkgs = {}
            pre_category_totals = {}
            for group in bkg_groups:
                hist = clone_empty_like(edges, f"{group}_{era_scope}_{channel_scope}_prefit")
                for cat in kept:
                    cat_hist = sum_component_group(
                        fitdiag_file, "prefit", cat, groups_for_cat[cat].get(group, []),
                        edges, f"{group}_{cat}_prefit", source_edges_by_cat[cat]
                    )
                    add_binwise(hist, cat_hist)
                    if cat not in pre_category_totals:
                        pre_category_totals[cat] = clone_empty_like(
                            edges, f"{cat}_{era_scope}_{channel_scope}_prefit_total")
                    add_binwise(pre_category_totals[cat], cat_hist)
                if hist.Integral() > 0:
                    pre_bkgs[group] = hist

            pre_signal = clone_empty_like(edges, f"signal_{era_scope}_{channel_scope}_prefit")
            for cat in kept:
                cat_signal = sum_component_group(
                    fitdiag_file, "prefit", cat, groups_for_cat[cat].get("signal", []),
                    edges, f"signal_{cat}_prefit", source_edges_by_cat[cat]
                )
                add_binwise(pre_signal, cat_signal)

            pre_uncertainty = build_fitdiag_uncertainty_hist(
                fitdiag_file, "prefit", edges, pre_category_totals,
                source_edges_by_cat,
                f"prefit_uncertainty_{era_scope}_{channel_scope}",
            )
            make_prefit_stack(target, data, pre_bkgs, pre_signal, pre_uncertainty)

            pre_total = sum_total(pre_bkgs, edges, f"prefit_total_{era_scope}_{channel_scope}")
            for fit_type in fit_types:
                post_bkgs = {}
                post_category_totals = {}
                for group in bkg_groups:
                    hist = clone_empty_like(edges, f"{group}_{era_scope}_{channel_scope}_{fit_type}")
                    for cat in kept:
                        cat_hist = sum_component_group(
                            fitdiag_file, fit_type, cat, groups_for_cat[cat].get(group, []),
                            edges, f"{group}_{cat}_{fit_type}", source_edges_by_cat[cat]
                        )
                        add_binwise(hist, cat_hist)
                        if cat not in post_category_totals:
                            post_category_totals[cat] = clone_empty_like(
                                edges, f"{cat}_{era_scope}_{channel_scope}_{fit_type}_total")
                        add_binwise(post_category_totals[cat], cat_hist)
                    if hist.Integral() > 0:
                        post_bkgs[group] = hist

                post_signal = clone_empty_like(edges, f"signal_{era_scope}_{channel_scope}_{fit_type}")
                for cat in kept:
                    cat_signal = sum_component_group(
                        fitdiag_file, fit_type, cat, groups_for_cat[cat].get("signal", []),
                        edges, f"signal_{cat}_{fit_type}", source_edges_by_cat[cat]
                    )
                    add_binwise(post_signal, cat_signal)

                post_uncertainty = build_fitdiag_uncertainty_hist(
                    fitdiag_file, fit_type, edges, post_category_totals,
                    source_edges_by_cat,
                    f"postfit_uncertainty_{fit_type}_{era_scope}_{channel_scope}",
                )
                make_postfit_stack(
                    target, data, post_bkgs, post_signal, fit_type,
                    post_uncertainty,
                )
                post_total = sum_total(
                    post_bkgs, edges, f"postfit_total_{era_scope}_{channel_scope}_{fit_type}"
                )
                make_prefit_vs_postfit(target, pre_total, post_total, fit_type)

    if refill_targets and not args.plot_only:
        save_cache_to_file(CACHE_PATH)

    shapes_file.Close()
    return True


# =============================================================================
# Main
# =============================================================================

_FINE_CACHE = {}  # (edge_id, subch, process, is_data) -> unscaled TH1D
_GLOBAL_EDGES = None  # fine-mass edges shared by every cached hist


def set_global_edges(edges):
    global _GLOBAL_EDGES
    _GLOBAL_EDGES = edges


def _cache_key_to_name(edge_id, subch, process, is_data):
    # Sub-channels have single underscores; use "__" as a safe separator.
    return f"{edge_id}__{subch}__{process}__{int(is_data)}"


def _cache_name_to_key(name):
    parts = name.split("__")
    if len(parts) != 4:
        return None
    edge_id, subch, process, flag = parts
    return (edge_id, subch, process, flag == "1")


def load_cache_from_file(path):
    """Populate _FINE_CACHE from a ROOT cache file. Returns True on success."""
    if not os.path.exists(path):
        return False
    f = ROOT.TFile.Open(path, "READ")
    if not f or f.IsZombie():
        return False
    n = 0
    for key in f.GetListOfKeys():
        name = key.GetName()
        k = _cache_name_to_key(name)
        if k is None:
            continue
        h = f.Get(name)
        if not h:
            continue
        h.SetDirectory(0)
        _FINE_CACHE[k] = h
        n += 1
    f.Close()
    logging.info(f"  Loaded {n} cached fine-mass hists from {path}")
    return n > 0


def save_cache_to_file(path):
    """Write _FINE_CACHE to a ROOT file (for later --plot-only runs)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    f = ROOT.TFile.Open(path, "RECREATE")
    for (edge_id, subch, process, is_data), h in _FINE_CACHE.items():
        name = _cache_key_to_name(edge_id, subch, process, is_data)
        h_clone = h.Clone(name)
        h_clone.Write()
    f.Close()
    logging.info(f"  Saved {len(_FINE_CACHE)} fine-mass hists to {path}")


def cached_fine(subch, process, cfg, hist_edges, is_data=False):
    """Fill a sub-channel's fine-mass hist once and cache it.

    Returns the cached TH1D directly (not a clone) — callers that need to
    mutate it must Clone() themselves. Using Add() against the returned
    hist is safe (it doesn't touch the source).
    Raises KeyError in --plot-only mode if the key is missing from cache.
    """
    edge_id = edge_cache_id(hist_edges)
    key = (edge_id, subch, process, is_data)
    if key not in _FINE_CACHE:
        if args.plot_only:
            raise KeyError(
                f"Cached hist not found in --plot-only mode: {key}. "
                f"Re-run without --plot-only to rebuild the cache.")
        era_i, ch_i = parse_subchannel(subch, args.era)
        _FINE_CACHE[key] = fill_fine_hist(
            era_i, ch_i, process, cfg, hist_edges,
            f"{process}_{subch}_{int(is_data)}_base",
            is_data=is_data)
    return _FINE_CACHE[key]


def build_process_aggregates_cached(fitdiag_file, kept, cfgs, ordered_bkgs, fit_type, hist_edges):
    """Pre/post backgrounds + signal + data aggregated for a subset of sub-channels.

    All sub-channel hists share `hist_edges`, so TH1::Add works natively.
    """
    edges_arr = array('d', hist_edges)
    n_uniform = len(hist_edges) - 1
    prefit = {bkg: ROOT.TH1D(f"{bkg}_pre_{id(kept)}", "", n_uniform, edges_arr)
              for bkg in ordered_bkgs}
    postfit = {bkg: ROOT.TH1D(f"{bkg}_post_{id(kept)}", "", n_uniform, edges_arr)
               for bkg in ordered_bkgs}
    for h in list(prefit.values()) + list(postfit.values()):
        h.SetDirectory(0)
    pre_category_totals = {
        subch: ROOT.TH1D(f"{subch}_pre_total_{id(kept)}", "", n_uniform, edges_arr)
        for subch in kept
    }
    post_category_totals = {
        subch: ROOT.TH1D(f"{subch}_post_total_{id(kept)}", "", n_uniform, edges_arr)
        for subch in kept
    }
    for h in list(pre_category_totals.values()) + list(post_category_totals.values()):
        h.SetDirectory(0)

    for subch in kept:
        cfg = cfgs[subch]
        n_coarse = len(cfg["bin_edges"]) - 1
        merged_set = set(cfg["merged_to_others"])

        # 'others' bucket = 'others' sample + all merged-to-others samples.
        # Clone on first hit because the first iteration takes the cached hist
        # directly, but subsequent `bucket.Add(fh)` would otherwise mutate it.
        bucket = None
        for proc in ["others"] + list(merged_set):
            fh = cached_fine(subch, proc, cfg, hist_edges)
            if bucket is None:
                bucket = fh.Clone(f"others_{subch}_bucket")
                bucket.SetDirectory(0)
            else:
                bucket.Add(fh)
        others_scales = get_coarse_scale(fitdiag_file, subch, "others",
                                         fit_type, n_coarse)

        for bkg in ordered_bkgs:
            if bkg == "others":
                if bucket is None or bucket.Integral() <= 0:
                    continue
                prefit[bkg].Add(bucket)
                pre_category_totals[subch].Add(bucket)
                post_clone = bucket.Clone(f"others_{subch}_post")
                post_clone.SetDirectory(0)
                apply_coarse_scale(post_clone, others_scales, cfg["bin_edges"])
                postfit[bkg].Add(post_clone)
                post_category_totals[subch].Add(post_clone)
                continue

            if bkg not in cfg["separate_processes"]:
                continue

            fh = cached_fine(subch, bkg, cfg, hist_edges)
            if fh.Integral() <= 0:
                continue
            prefit[bkg].Add(fh)
            pre_category_totals[subch].Add(fh)
            scales = get_coarse_scale(fitdiag_file, subch, bkg, fit_type, n_coarse)
            post_clone = fh.Clone(f"{bkg}_{subch}_post")
            post_clone.SetDirectory(0)
            apply_coarse_scale(post_clone, scales, cfg["bin_edges"])
            postfit[bkg].Add(post_clone)
            post_category_totals[subch].Add(post_clone)

    # Signal
    pre_signal = ROOT.TH1D(f"signal_pre_{id(kept)}", "", n_uniform, edges_arr)
    post_signal = ROOT.TH1D(f"signal_post_{id(kept)}", "", n_uniform, edges_arr)
    pre_signal.SetDirectory(0)
    post_signal.SetDirectory(0)
    for subch in kept:
        cfg = cfgs[subch]
        n_coarse = len(cfg["bin_edges"]) - 1
        fh = cached_fine(subch, args.masspoint, cfg, hist_edges)
        if fh.Integral() <= 0:
            continue
        pre_signal.Add(fh)
        scales = get_coarse_scale(fitdiag_file, subch, "signal", fit_type, n_coarse)
        post_clone = fh.Clone(f"signal_{subch}_post")
        post_clone.SetDirectory(0)
        apply_coarse_scale(post_clone, scales, cfg["bin_edges"])
        post_signal.Add(post_clone)

    prefit_bkgs = {k: v for k, v in prefit.items() if v.Integral() > 0}
    postfit_bkgs = {k: v for k, v in postfit.items() if v.Integral() > 0}

    data = ROOT.TH1D(f"data_{id(kept)}", "data", n_uniform, edges_arr)
    data.SetDirectory(0)
    data.SetTitle("data")
    if args.blind:
        # Asimov: data = sum of pre-fit backgrounds; samples/.../data.root
        # is never read.
        for h in prefit_bkgs.values():
            data.Add(h)
    else:
        for subch in kept:
            cfg = cfgs[subch]
            fh = cached_fine(subch, "data", cfg, hist_edges, is_data=True)
            data.Add(fh)

    source_edges_by_subch = {
        subch: tuple(float(x) for x in cfgs[subch]["bin_edges"])
        for subch in kept
    }
    pre_uncertainty = build_fitdiag_uncertainty_hist(
        fitdiag_file, "prefit", hist_edges, pre_category_totals,
        source_edges_by_subch, f"prefit_uncertainty_{id(kept)}",
    )
    post_uncertainty = build_fitdiag_uncertainty_hist(
        fitdiag_file, fit_type, hist_edges, post_category_totals,
        source_edges_by_subch, f"postfit_uncertainty_{fit_type}_{id(kept)}",
    )

    return (
        prefit_bkgs, postfit_bkgs, pre_signal, post_signal, data,
        pre_uncertainty, post_uncertainty,
    )


def process_plot_target(fitdiag_file, all_cfgs, era_scope, channel_scope,
                        fit_era, fit_types):
    """Generate plots for one (era_scope, channel_scope) pair.

    For each `fit_type in fit_types`, produces pre-fit stack (once, shared),
    post-fit stack, and pre/post overlay. Pre-fit aggregates are identical
    across fit types so we compute them once.
    """
    kept = [
        sc for sc in all_cfgs
        if keep_by_era(sc, era_scope, fit_era) and keep_by_channel(sc, channel_scope)
    ]
    if not kept:
        logging.debug(f"  skip {era_scope}/{channel_scope}: no sub-channels")
        return

    if is_atomic_template_target(era_scope, channel_scope) and len(kept) == 1:
        hist_edges = tuple(float(x) for x in all_cfgs[kept[0]]["bin_edges"])
        xrange = (hist_edges[0], hist_edges[-1])
        y_title = "Events / bin"
        binning_label = "template bins"
    else:
        # Use the widest sub-channel's range and sigma for the display range:
        # range and resolution are self-consistent, and the widest sub-channel
        # fully covers its own range.
        widest_sc = max(kept,
                        key=lambda sc: (all_cfgs[sc]["mass_max"]
                                        - all_cfgs[sc]["mass_min"]))
        mass_lo = all_cfgs[widest_sc]["mass_min"]
        mass_hi = all_cfgs[widest_sc]["mass_max"]
        xrange = (mass_lo, mass_hi)
        hist_edges = tuple(_GLOBAL_EDGES)
        y_title = f"Events / {round(args.bin_width, 2):.2g} GeV"
        binning_label = f"fine grid ({len(hist_edges) - 1} bins)"

    sub_cfgs = {sc: all_cfgs[sc] for sc in kept}
    bkg_union = []
    for sc in kept:
        for bkg in sub_cfgs[sc]["separate_processes"]:
            if bkg in BKG_ORDER and bkg not in bkg_union:
                bkg_union.append(bkg)
    if "others" not in bkg_union:
        bkg_union.append("others")
    ordered_bkgs = [b for b in BKG_ORDER if b in bkg_union]

    logging.info(f"  {era_scope}/{channel_scope}: {len(kept)} sub-channels, "
                 f"mass=[{xrange[0]:.2f}, {xrange[1]:.2f}] GeV, {binning_label}")

    target = PlotTarget(
        era_scope=era_scope,
        channel_scope=channel_scope,
        xrange=xrange,
        hist_edges=hist_edges,
        y_title=y_title,
    )

    prefit_drawn = False
    agg_pre_tot = None
    for ft in fit_types:
        pre_bkgs, post_bkgs, pre_signal, post_signal, agg_data, pre_uncertainty, post_uncertainty = \
            build_process_aggregates_cached(
                fitdiag_file, kept, sub_cfgs, ordered_bkgs, ft, target.hist_edges)

        if not prefit_drawn:
            make_prefit_stack(target, agg_data, pre_bkgs, pre_signal, pre_uncertainty)
            # Pre-fit total is fit-type-independent; build once.
            agg_pre_tot = sum_total(pre_bkgs, target.hist_edges, "prefit_total_mass")
            prefit_drawn = True

        make_postfit_stack(target, agg_data, post_bkgs, post_signal, ft, post_uncertainty)
        agg_post_tot = sum_total(post_bkgs, target.hist_edges, "postfit_total_mass")
        make_prefit_vs_postfit(target, agg_pre_tot, agg_post_tot, ft)


def main():
    fit_types = ["b", "s"] if args.fit_type == "both" else [args.fit_type]

    logging.info("Real-mass (unbinned) post-fit plotting")
    logging.info(f"  Fit source:     {args.era} (fitdiag: {FITDIAG_PATH})")
    logging.info(f"  Masspoint:      {args.masspoint}")
    logging.info(f"  Method:         {args.method}")
    logging.info(f"  Fit type:       {args.fit_type}")
    logging.info(f"  Bin width:      {args.bin_width} GeV (resolved after subchannel load)")
    logging.info(f"  Output dir:     {OUTPUT_DIR}")

    fit_era = args.era  # fit source (constant for this run)

    f = ROOT.TFile.Open(FITDIAG_PATH, "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Failed to open {FITDIAG_PATH}")

    if make_run_period_postfit_plots(f, fit_era, fit_types):
        f.Close()
        return

    all_subchannels = discover_channels(f)
    logging.info(f"  Sub-channels in fit: {all_subchannels}")

    # Load sub-channel configs once (shared across all plot targets).
    all_cfgs = {}
    for sc in all_subchannels:
        era_i, ch_i = parse_subchannel(sc, fit_era)
        all_cfgs[sc] = load_subchannel_config(era_i, ch_i)

    # Global fine grid = union of all sub-channel mass windows, snapped to
    # integer bin-width boundaries. All cached fine-mass hists share this.
    global_lo = min(cfg["mass_min"] for cfg in all_cfgs.values())
    global_hi = max(cfg["mass_max"] for cfg in all_cfgs.values())
    if args.bin_width == "auto":
        widest = max(all_cfgs.values(),
                     key=lambda c: c["mass_max"] - c["mass_min"])
        args.bin_width = (widest["mass_max"] - widest["mass_min"]) / 20
        logging.info(f"  Auto bin width: widest range "
                     f"[{widest['mass_min']:.3f}, {widest['mass_max']:.3f}] / 20 "
                     f"-> {args.bin_width:.4f} GeV")
    else:
        args.bin_width = float(args.bin_width)
    global CACHE_PATH
    CACHE_PATH = f"{CACHE_DIR}/mass_hists_v2_bw{args.bin_width:g}.root"
    set_global_edges(build_uniform_edges(global_lo, global_hi, args.bin_width))
    logging.info(f"  Global grid:    [{global_lo:.2f}, {global_hi:.2f}] GeV "
                 f"-> [{_GLOBAL_EDGES[0]:g}, {_GLOBAL_EDGES[-1]:g}] "
                 f"({len(_GLOBAL_EDGES) - 1} fine bins)")

    # Cache: load existing file if --plot-only, otherwise fill fresh.
    if args.plot_only:
        if not load_cache_from_file(CACHE_PATH):
            raise FileNotFoundError(
                f"--plot-only requires a cache file at {CACHE_PATH}. "
                f"Run without --plot-only first to build the cache.")

    era_scopes = ([args.era_scope]
                  if args.era_scope is not None
                  else applicable_era_scopes(fit_era))
    channel_scopes = ([args.channel_scope]
                      if args.channel_scope is not None
                      else default_channel_scopes())

    logging.info(f"  Era scopes:     {era_scopes}")
    logging.info(f"  Channel scopes: {channel_scopes}")
    logging.info(f"  Fit types:      {fit_types}")

    total = 0
    for era_scope in era_scopes:
        for scope in channel_scopes:
            process_plot_target(f, all_cfgs, era_scope, scope, fit_era, fit_types)
            total += 1

    f.Close()

    # Save cache for subsequent --plot-only invocations
    if not args.plot_only:
        save_cache_to_file(CACHE_PATH)

    logging.info(f"Done. Generated plots for {total} (era × channel) targets.")


if __name__ == "__main__":
    entry_setup(_build_parser().parse_args())
    main()
