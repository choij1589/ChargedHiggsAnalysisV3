#!/usr/bin/env python3
"""
Plot post-fit distributions on the real physical mass axis, using a fine
uniform 1-GeV grid, with per-process histograms filled from the unbinned
preprocessed trees in `samples/` and then scaled by the FitDiagnostics
post/pre-fit ratio per coarse bin.

Aggregates multiple sub-channels (per-era/per-channel) from a combined
fitDiagnostics file. Each sub-channel's mass window and adaptive coarse bin
edges are taken from its `binning.json`. The union mass range defines the
fine uniform-bin grid (default bin width: 1 GeV).

Usage:
    python3 plotPostfitMass.py --era All --masspoint MHc130_MA90 \
        --method ParticleNet --binning extended --partial-unblind \
        --channel-scope Combined --fit-type b
"""
import os
import sys
import json
import math
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
    p.add_argument("--method", required=True, type=str, choices=["Baseline", "ParticleNet"])
    p.add_argument("--binning", default="extended", choices=["uniform", "extended"])
    p.add_argument("--era-scope", default=None, dest="era_scope",
                   help="Filter plots to this era slice (e.g. 2018, Run2, All). "
                        "Default: iterate every era scope applicable to the fit.")
    p.add_argument("--channel-scope", default=None,
                   choices=["SR1E2Mu", "SR3Mu", "Combined"], dest="channel_scope",
                   help="Filter plots to this channel scope. Default: iterate all three.")
    p.add_argument("--fit-type", default="both", choices=["b", "s", "both"],
                   help="Which post-fit variant(s) to plot [default: both]")
    p.add_argument("--unblind", action="store_true")
    p.add_argument("--partial-unblind", action="store_true", dest="partial_unblind")
    p.add_argument("--blind", action="store_true",
                   help="Asimov mode: data = sum of pre-fit backgrounds; "
                        "samples/.../data.root is never read.")
    p.add_argument("--bin-width", default=1.0, type=float,
                   help="Fine-grid bin width in GeV (default 1.0)")
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
sys.path.insert(0, f"{WORKDIR}/SignalRegionStudyV2/python")
from plotter import KinematicCanvas, ComparisonCanvas, get_CoM_energy
from plotter import PALETTE_LONG as PALETTE
from template_utils import build_particlenet_score
import cmsstyle as CMS

with open(f"{WORKDIR}/Common/Data/Luminosity.json", "r") as _lfh:
    _LUMI_CONFIG = json.load(_lfh)

binning_suffix = None
TEMPLATE_DIR = None
FITDIAG_DIR = None
FITDIAG_PATH = None
OUTPUT_DIR = None
CACHE_DIR = None
CACHE_PATH = None


def _compute_binning_suffix(parsed):
    if parsed.unblind:
        return f"{parsed.binning}_unblind"
    if parsed.partial_unblind:
        return f"{parsed.binning}_partial_unblind"
    return parsed.binning


def _compute_paths():
    """Derive TEMPLATE_DIR / FITDIAG_PATH / OUTPUT_DIR / CACHE_PATH from `args`.

    Callers that import this module (e.g. plotCombinedMass) set `args` to a
    SimpleNamespace and call this to refresh the derived paths between
    masspoints.
    """
    global binning_suffix, TEMPLATE_DIR, FITDIAG_DIR, FITDIAG_PATH
    global OUTPUT_DIR, CACHE_DIR, CACHE_PATH
    binning_suffix = _compute_binning_suffix(args)
    TEMPLATE_DIR = (f"{WORKDIR}/SignalRegionStudyV2/templates/"
                    f"{args.era}/Combined/{args.masspoint}/{args.method}/{binning_suffix}")
    FITDIAG_DIR = f"{TEMPLATE_DIR}/combine_output/fitdiag"
    FITDIAG_PATH = (f"{FITDIAG_DIR}/fitDiagnostics."
                    f"{args.masspoint}.{args.method}.{binning_suffix}.root")
    OUTPUT_DIR = f"{FITDIAG_DIR}/plots_mass"
    CACHE_DIR = f"{FITDIAG_DIR}/cached"
    CACHE_PATH = f"{CACHE_DIR}/fine_hists_bw{args.bin_width:g}.root"


def entry_setup(parsed_args, *, require_fitdiag=True, make_output_dir=True):
    """Populate module-level `args` and derived paths.

    Parameters
    ----------
    parsed_args : argparse.Namespace or SimpleNamespace
        Must expose .era, .masspoint, .method, .binning, .unblind,
        .partial_unblind, .bin_width, .debug, .plot_only, and optionally
        .channel_scope / .era_scope / .fit_type.
    require_fitdiag : bool
        If True, raise when the fitDiagnostics file is missing.
    make_output_dir : bool
        If True, create OUTPUT_DIR.
    """
    global args
    args = parsed_args

    chosen = sum(bool(x) for x in (args.unblind, args.partial_unblind, args.blind))
    if chosen != 1:
        raise ValueError("Exactly one of --blind / --partial-unblind / --unblind is required")

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
        CMS.SetLumi(None, run=f"Run 2+3, {run2_lumi}+{run3_lumi} fb^{{#minus1}}")
        CMS.SetEnergy(0, unit="13/13.6 TeV")
        return None, "Run 2+3"


class ComparisonCanvasAll(_AllEraCMSStyleMixin, ComparisonCanvas):
    pass


class KinematicCanvasAll(_AllEraCMSStyleMixin, KinematicCanvas):
    pass


def _build_header_text(era_scope):
    """Compose the CMS lumi + CoM header string for a given era scope."""
    if era_scope == "All":
        l2 = _LUMI_CONFIG["Run2"]["combined"]
        l3 = _LUMI_CONFIG["Run3"]["combined"]
        return f"Run 2+3, {l2}+{l3} fb^{{#minus1}} (13/13.6 TeV)"
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
    "SR1E2Mu": "e#mu#mu",
    "SR3Mu": "#mu#mu#mu",
}


def masspoint_label(masspoint):
    """Convert 'MHc130_MA90' -> '(m_{H^{+}}, m_{A}) = (130, 90) GeV'."""
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


def parse_subchannel(subch, fallback_era):
    """Return (era, channel) from a fitdiag sub-channel name."""
    parts = subch.split("_")
    for i, part in enumerate(parts):
        if part.startswith("SR") or part.startswith("TTZ"):
            channel = part
            if i > 0 and parts[i - 1].startswith("era"):
                return parts[i - 1][3:], channel
            return fallback_era, channel
    raise ValueError(f"Cannot parse sub-channel: {subch}")


def keep_by_channel(subch, scope):
    """True if sub-channel `subch` belongs to the channel scope."""
    if scope == "Combined":
        return True
    return subch.endswith("_" + scope) or subch == scope


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
    tdir = f"{WORKDIR}/SignalRegionStudyV2/templates/{era}/{channel}/{args.masspoint}/{args.method}/{binning_suffix}"
    binning = json.load(open(f"{tdir}/binning.json"))
    plist = json.load(open(f"{tdir}/process_list.json"))
    threshold = None
    upper_threshold = None
    bg_weights = None
    if args.method == "ParticleNet":
        thr_path = f"{tdir}/threshold.json"
        bw_path = f"{tdir}/background_weights.json"
        if os.path.exists(thr_path):
            thr = json.load(open(thr_path))
            threshold = thr.get("threshold")
            upper_threshold = thr.get("upper_threshold")
        if os.path.exists(bw_path):
            bg_weights = json.load(open(bw_path))["weights"]
    return {
        "template_dir": tdir,
        "mass_min": binning["mass_min"],
        "mass_max": binning["mass_max"],
        "bin_edges": binning["bin_edges"],
        "threshold": threshold,
        "upper_threshold": upper_threshold,
        "bg_weights": bg_weights,
        "separate_processes": plist["separate_processes"],
        "merged_to_others": plist["merged_to_others"],
    }


# =============================================================================
# Unbinned fill from preprocessed trees
# =============================================================================

def sample_path(era, channel, process):
    return f"{WORKDIR}/SignalRegionStudyV2/samples/{era}/{channel}/{args.masspoint}/{process}.root"


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
    """Multiply fine bins by the post/pre scale of the coarse bin they fall in."""
    n_coarse = len(mass_edges) - 1
    for j in range(1, fine_hist.GetNbinsX() + 1):
        x = fine_hist.GetBinCenter(j)
        idx = bisect.bisect_right(mass_edges, x) - 1
        if idx < 0 or idx >= n_coarse:
            # Rescue: x exactly at the upper edge lands on n_coarse.
            if x == mass_edges[-1]:
                idx = n_coarse - 1
            else:
                continue
        s = scales[idx]
        fine_hist.SetBinContent(j, fine_hist.GetBinContent(j) * s)
        fine_hist.SetBinError(j, fine_hist.GetBinError(j) * abs(s))


# =============================================================================
# Plot config helpers
# =============================================================================

def scope_label(scope):
    if scope == "SR1E2Mu":
        return CHANNEL_LATEX["SR1E2Mu"]
    if scope == "SR3Mu":
        return CHANNEL_LATEX["SR3Mu"]
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
    return ComparisonCanvasAll if era_scope == "All" else ComparisonCanvas


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


# =============================================================================
# Plot-drawing helpers
# =============================================================================

def _blinding_label():
    if args.partial_unblind:
        return "Partial-Unblind"
    # Full unblind: no extra label needed.
    return ""


def _make_stack(target, agg_data, agg_bkgs, agg_signal, label_top, systSrc, out_path):
    """Generic stack-plot builder shared by pre-fit and post-fit plots."""
    if not agg_bkgs:
        logging.warning(f"No backgrounds; skipping {out_path}")
        return
    agg_data.SetTitle("data")
    colors = [BKG_COLORS.get(b, ROOT.kGray) for b in agg_bkgs.keys()]
    config = make_canvas_config(target.era_scope, {
        "channel": masspoint_label(args.masspoint),
        "channelPosY": 0.58,
        "channelSize": 0.04,
        "xTitle": "M(#mu^{+}#mu^{-}) [GeV]",
        "yTitle": f"Events / {args.bin_width:g} GeV",
        "xRange": list(target.xrange),
        "rTitle": "Data / Pred",
        "rRange": [0, 2.5],
        "maxDigits": 3,
        "systSrc": systSrc,
        "colors": colors,
    })
    plotter = select_comparison_cls(target.era_scope)(agg_data, agg_bkgs, config)
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


def make_postfit_stack(target, agg_data, post_bkgs, post_signal, fit_type):
    fit_label = "B-only" if fit_type == "b" else "S+B"
    out = f"{OUTPUT_DIR}/postfit_{fit_type}_mass_{target.era_scope}_{target.channel_scope}.png"
    _make_stack(target, agg_data, post_bkgs, post_signal,
                label_top=f"Post-fit {fit_label}",
                systSrc=f"Post-fit ({fit_label})",
                out_path=out)


def make_prefit_stack(target, agg_data, pre_bkgs, pre_signal):
    out = f"{OUTPUT_DIR}/prefit_mass_{target.era_scope}_{target.channel_scope}.png"
    _make_stack(target, agg_data, pre_bkgs, pre_signal,
                label_top="Pre-fit",
                systSrc="Pre-fit",
                out_path=out)


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
        "xTitle": "M(#mu^{+}#mu^{-}) [GeV]",
        "yTitle": f"Events / {args.bin_width:g} GeV",
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


def sum_total(hists, uniform_edges, name):
    edges_arr = array('d', uniform_edges)
    total = ROOT.TH1D(name, "", len(uniform_edges) - 1, edges_arr)
    total.SetDirectory(0)
    for h in hists.values():
        total.Add(h)
    return total


# =============================================================================
# Main
# =============================================================================

_FINE_CACHE = {}  # (subch, process, is_data) -> unscaled fine TH1D (shared grid)
_GLOBAL_EDGES = None  # fine-mass edges shared by every cached hist


def set_global_edges(edges):
    global _GLOBAL_EDGES
    _GLOBAL_EDGES = edges


def _cache_key_to_name(subch, process, is_data):
    # Sub-channels have single underscores; use "__" as a safe separator.
    return f"{subch}__{process}__{int(is_data)}"


def _cache_name_to_key(name):
    parts = name.split("__")
    if len(parts) != 3:
        return None
    subch, process, flag = parts
    return (subch, process, flag == "1")


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
    for (subch, process, is_data), h in _FINE_CACHE.items():
        name = _cache_key_to_name(subch, process, is_data)
        h_clone = h.Clone(name)
        h_clone.Write()
    f.Close()
    logging.info(f"  Saved {len(_FINE_CACHE)} fine-mass hists to {path}")


def cached_fine(subch, process, cfg, is_data=False):
    """Fill a sub-channel's fine-mass hist once and cache it.

    Returns the cached TH1D directly (not a clone) — callers that need to
    mutate it must Clone() themselves. Using Add() against the returned
    hist is safe (it doesn't touch the source).
    Raises KeyError in --plot-only mode if the key is missing from cache.
    """
    if _GLOBAL_EDGES is None:
        raise RuntimeError("set_global_edges() must be called first")
    key = (subch, process, is_data)
    if key not in _FINE_CACHE:
        if args.plot_only:
            raise KeyError(
                f"Cached hist not found in --plot-only mode: {key}. "
                f"Re-run without --plot-only to rebuild the cache.")
        era_i, ch_i = parse_subchannel(subch, args.era)
        _FINE_CACHE[key] = fill_fine_hist(
            era_i, ch_i, process, cfg, _GLOBAL_EDGES,
            f"{process}_{subch}_{int(is_data)}_base",
            is_data=is_data)
    return _FINE_CACHE[key]


def build_process_aggregates_cached(fitdiag_file, kept, cfgs, ordered_bkgs, fit_type):
    """Pre/post backgrounds + signal + data aggregated for a subset of sub-channels.

    All sub-channel hists share `_GLOBAL_EDGES`, so TH1::Add works natively.
    """
    edges_arr = array('d', _GLOBAL_EDGES)
    n_uniform = len(_GLOBAL_EDGES) - 1
    prefit = {bkg: ROOT.TH1D(f"{bkg}_pre_{id(kept)}", "", n_uniform, edges_arr)
              for bkg in ordered_bkgs}
    postfit = {bkg: ROOT.TH1D(f"{bkg}_post_{id(kept)}", "", n_uniform, edges_arr)
               for bkg in ordered_bkgs}
    for h in list(prefit.values()) + list(postfit.values()):
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
            fh = cached_fine(subch, proc, cfg)
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
                post_clone = bucket.Clone(f"others_{subch}_post")
                post_clone.SetDirectory(0)
                apply_coarse_scale(post_clone, others_scales, cfg["bin_edges"])
                postfit[bkg].Add(post_clone)
                continue

            if bkg not in cfg["separate_processes"]:
                continue

            fh = cached_fine(subch, bkg, cfg)
            if fh.Integral() <= 0:
                continue
            prefit[bkg].Add(fh)
            scales = get_coarse_scale(fitdiag_file, subch, bkg, fit_type, n_coarse)
            post_clone = fh.Clone(f"{bkg}_{subch}_post")
            post_clone.SetDirectory(0)
            apply_coarse_scale(post_clone, scales, cfg["bin_edges"])
            postfit[bkg].Add(post_clone)

    # Signal
    pre_signal = ROOT.TH1D(f"signal_pre_{id(kept)}", "", n_uniform, edges_arr)
    post_signal = ROOT.TH1D(f"signal_post_{id(kept)}", "", n_uniform, edges_arr)
    pre_signal.SetDirectory(0)
    post_signal.SetDirectory(0)
    for subch in kept:
        cfg = cfgs[subch]
        n_coarse = len(cfg["bin_edges"]) - 1
        fh = cached_fine(subch, args.masspoint, cfg)
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
            fh = cached_fine(subch, "data", cfg, is_data=True)
            data.Add(fh)

    return prefit_bkgs, postfit_bkgs, pre_signal, post_signal, data


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

    mass_lo = min(all_cfgs[sc]["mass_min"] for sc in kept)
    mass_hi = max(all_cfgs[sc]["mass_max"] for sc in kept)
    xrange = (math.floor(mass_lo), math.ceil(mass_hi))

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
                 f"mass=[{mass_lo:.2f}, {mass_hi:.2f}] GeV")

    target = PlotTarget(era_scope=era_scope, channel_scope=channel_scope, xrange=xrange)

    prefit_drawn = False
    agg_pre_tot = None
    for ft in fit_types:
        pre_bkgs, post_bkgs, pre_signal, post_signal, agg_data = \
            build_process_aggregates_cached(
                fitdiag_file, kept, sub_cfgs, ordered_bkgs, ft)

        if not prefit_drawn:
            make_prefit_stack(target, agg_data, pre_bkgs, pre_signal)
            # Pre-fit total is fit-type-independent; build once.
            agg_pre_tot = sum_total(pre_bkgs, _GLOBAL_EDGES, "prefit_total_mass")
            prefit_drawn = True

        make_postfit_stack(target, agg_data, post_bkgs, post_signal, ft)
        agg_post_tot = sum_total(post_bkgs, _GLOBAL_EDGES, "postfit_total_mass")
        make_prefit_vs_postfit(target, agg_pre_tot, agg_post_tot, ft)


def main():
    logging.info("Real-mass (unbinned) post-fit plotting")
    logging.info(f"  Fit source:     {args.era} (fitdiag: {FITDIAG_PATH})")
    logging.info(f"  Masspoint:      {args.masspoint}")
    logging.info(f"  Method:         {args.method}")
    logging.info(f"  Fit type:       {args.fit_type}")
    logging.info(f"  Bin width:      {args.bin_width} GeV")
    logging.info(f"  Output dir:     {OUTPUT_DIR}")

    fit_era = args.era  # fit source (constant for this run)

    f = ROOT.TFile.Open(FITDIAG_PATH, "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Failed to open {FITDIAG_PATH}")

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
                      else CHANNEL_SCOPES)
    fit_types = ["b", "s"] if args.fit_type == "both" else [args.fit_type]

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
