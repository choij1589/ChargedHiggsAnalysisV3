#!/usr/bin/env python3
"""Paper-style limit plots: the Brazilian-panel path of plotLimits.py, minus
"Preliminary" and --compare-mhc (paper figures never draw that overlay).

Kept as a separate script rather than a flag on plotLimits.py so the two
output trees never collide -- plotLimits.py's results/plots/{mode}/{era}/
stays the internal/preliminary version, this one writes under
results/plots/paper/Limits/, the same convention plotPaperLRModified.py,
plotPaperTemplates.py and plotPaperPostfitSummary.py use. Every remaining
option and layout constant matches plotLimits.py's Brazilian-panel path.
Differences from plotLimits.py: no "Preliminary"; always reads the
production interp-signal JSON (no --signal-source flag); the output name
drops the always-constant "Asymptotic"/source tokens; --panels-per-row
defaults to the production per-mHc convention (3 for mHc <= 100, 2 above)
instead of a flat 3; PDF only, no PNG; no --compare-mhc/--mhc-list/
--ratio-ref/--ratio-range (paper figures are per-mHc panels only).
"""
import os
import sys
from array import array
import argparse
import ROOT
import srspaths
import json
import cmsstyle as CMS
from plotter import (LumiInfo, EnergyInfo, get_CoM_energy,
                     configure_cms_label, draw_cms_label, restore_cms_label,
                     reanchor_lumi_header)
# Paper style is defined once in plotPaperLRModified.py and imported by every
# figure that has to match it, so the CMS block cannot drift between the limit
# curves and the paper panels.
from plotPaperLRModified import CMS_LABEL_POS, CMS_LABEL_SIZE

ROOT.gROOT.SetBatch(ROOT.kTRUE)

# Load luminosity configuration from JSON
_LUMI_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Common", "Data", "Luminosity.json")
with open(_LUMI_JSON_PATH, "r") as f:
    _LUMI_CONFIG = json.load(f)

# Curated baseline mass-point list — 1 MP per MA, mixed MHc to cover the full MA spectrum
# continuously (incl. on-Z points kept here intentionally for the ParticleNet baseline overlay).
_MASSPOINTS_JSON = os.path.join(os.path.dirname(__file__), "..", "configs", "masspoints.json")
with open(_MASSPOINTS_JSON) as f:
    _BASELINE_CURATED = set(json.load(f)["limits"])

parser = argparse.ArgumentParser()
parser.add_argument("--era", type=str, required=True,
                    help="2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix, Run2, Run3, All")
parser.add_argument("--channel", type=str, default="Combined",
                    choices=["Combined", "SR1E2Mu", "SR3Mu"],
                    help="Analysis channel (default: Combined)")
parser.add_argument("--method", type=str, default=None, help="Baseline, ParticleNet "
                    "(ignored, and not required, with --combine-mhc)")
parser.add_argument("--limit_type", type=str, default="Asymptotic",
                    choices=["Asymptotic"], help="Limit type (Asymptotic only in V4)")
parser.add_argument("--blind", action="store_true", help="Hide observed limit (for blinded results)")
parser.add_argument("--ymax", type=float, default=None,
                    help="Override the automatic y-axis maximum (2x max exp+2sigma) — "
                         "for synchronizing scales across method/mHc comparisons")
parser.add_argument("--stack_baseline", action="store_true", help="Show baseline expected limit on top (only for ParticleNet method)")
parser.add_argument("--mhc", type=int, default=None,
                    help="Plot only this MHc value. Baseline defaults to 160 when omitted; ParticleNet uses all trained points when omitted.")
parser.add_argument("--panels-per-row", dest="panels_per_row", type=int,
                    default=None, choices=[2, 3],
                    help="how many of these panels share a row in the paper. "
                         "Text prints at t*W/N/A, so keeping one physical text "
                         "size across rows needs the canvas aspect A to scale "
                         "with 1/N: 3 -> square 600x600, 2 -> 900x600. Both "
                         "then print at the same height, W/3. Default follows "
                         "the production panels (as plotLimitsCompareV3.py "
                         "does): 3 for mHc <= 100, 2 above.")
parser.add_argument("--mode", type=str, default="BR", choices=["BR", "xsec"],
                    help="Limit unit: BR (relative branching ratio, default) or xsec (sigma_sig = 2 sigma(ttbar) B_sig in fb)")
parser.add_argument("--combine-mhc", dest="combine_mhc", action="store_true",
                    help="DRAFT: one grid figure covering the canonical seven-mHc "
                         "paper set (CMS-HIG-20-012 Figure 5 style) -- a small "
                         "Brazilian-limit panel per mHc (Baseline at 70/85, "
                         "ParticleNet at 100/115/130/145/160), one shared CMS/lumi "
                         "header and one shared legend for the whole grid instead "
                         "of seven repeated ones. Ignores --method/--mhc/"
                         "--stack_baseline/--panels-per-row.")
args = parser.parse_args()

if not args.combine_mhc and args.method is None:
    parser.error("--method is required unless --combine-mhc is given")

# The seven paper panels (Baseline MHc70/85, ParticleNet MHc100/115/130/145/160)
# are laid out 3-square-then-2-wide-then-2-wide, matching plotLimitsCompareV3.py's
# own default. An omitted --mhc on ParticleNet (all trained points on one
# panel) has no single mHc column to size by, so it keeps the square default;
# an omitted --mhc on Baseline still resolves to mHc160 below, so it is sized
# as that column.
if args.panels_per_row is None:
    _mhc_for_sizing = args.mhc
    if _mhc_for_sizing is None and args.method == "Baseline":
        _mhc_for_sizing = 160
    args.panels_per_row = 3 if (_mhc_for_sizing is None or _mhc_for_sizing <= 100) else 2

# Validate era
VALID_ERAS = [
    "2016preVFP", "2016postVFP", "2017", "2018",
    "2022", "2022EE", "2023", "2023BPix",
    "Run2", "Run3", "All"
]
if args.era not in VALID_ERAS:
    raise ValueError(f"Invalid era: {args.era}. Must be one of {VALID_ERAS}")

# Extend LumiInfo for "All"
LumiInfo_extended = dict(LumiInfo)
LumiInfo_extended["All"] = _LUMI_CONFIG["All"]["combined"]


def create_graphs(limits_dict):
    """Create TGraph objects from limits dictionary."""
    def _ma(mp):
        return float(srspaths.parse_ma_token(mp.split("_")[1][2:]))

    mass_points = sorted(limits_dict.keys(), key=_ma)
    x = array('d', [_ma(mp) for mp in mass_points])
    n = len(x)

    # JSON values are already B_sig (converted in collectLimits.py); use directly
    limits = {key: array('d', [limits_dict[mp][key] for mp in mass_points])
              for key in ["obs", "exp0", "exp-1", "exp-2", "exp+1", "exp+2"]}

    # Create graphs
    g_obs = ROOT.TGraph(n, x, limits["obs"])
    g_obs.SetLineWidth(2)
    g_obs.SetMarkerStyle(20)
    g_obs.SetMarkerSize(0.8)

    g_exp = ROOT.TGraph(n, x, limits["exp0"])
    g_exp.SetLineWidth(2)
    g_exp.SetLineStyle(ROOT.kDashed)
    g_exp.SetLineColor(ROOT.kBlack)

    # Error bands
    g_exp1sigma = ROOT.TGraphAsymmErrors(n)
    g_exp2sigma = ROOT.TGraphAsymmErrors(n)
    for i in range(n):
        for g in [g_exp1sigma, g_exp2sigma]:
            g.SetPoint(i, x[i], limits["exp0"][i])
        g_exp1sigma.SetPointError(i, 0, 0, limits["exp0"][i] - limits["exp-1"][i], limits["exp+1"][i] - limits["exp0"][i])
        g_exp2sigma.SetPointError(i, 0, 0, limits["exp0"][i] - limits["exp-2"][i], limits["exp+2"][i] - limits["exp0"][i])

    return {'obs': g_obs, 'exp': g_exp, 'exp1sigma': g_exp1sigma, 'exp2sigma': g_exp2sigma,
            'values': [v for arr in limits.values() for v in arr]}


# The CMS 2016 (HEPData ins1735729) and ATLAS Run 2 (HEPData ins2654723) reference curves
# are no longer overlaid on the mH+ = 160 GeV BR plots; the paper figures show this result
# alone. The HEPData YAML files stay under results/yaml/ for reference.

# Setup CMS style. No "Preliminary" -- this is the published-paper variant of
# plotLimits.py; see configure_cms_label()/_CMS_LABEL_CONFIG below, which is
# what actually draws the in-frame block (this call only sets cmsstyle's own
# globals, which configure_cms_label() blanks again before every canvas).
CMS.SetExtraText("")
CMS.ResetAdditionalInfo()

# Luminosity header follows the paper convention shared with plotPaperPostfitSummary.py,
# plotPaperLRModified.py and plotPaperTemplates.py: rounded per-period luminosities from
# LumiInfo, each run period quoted with its own energy, and no "Run2,"-style prefix.
if args.era == "All":
    # cmsstyle renders "<cms_lumi> (<cms_energy>)", so only the Run3 energy can live in
    # SetEnergy; the whole Run2 term is baked into the run label.
    CMS.SetLumi(None, run=(f"{LumiInfo['Run2']:g} fb^{{#minus1}} ({EnergyInfo['Run2']:g} TeV) + "
                           f"{LumiInfo['Run3']:g} fb^{{#minus1}}"))
    CMS.SetEnergy(0, unit=f"{EnergyInfo['Run3']:g} TeV")
elif args.era in ("Run2", "Run3"):
    CMS.SetLumi(None, run=f"{LumiInfo[args.era]:g} fb^{{#minus1}}")
    CMS.SetEnergy(EnergyInfo[args.era])
else:
    # Individual era
    CMS.SetLumi(LumiInfo.get(args.era, LumiInfo_extended.get(args.era)), run=args.era)
    CMS.SetEnergy(get_CoM_energy(args.era))

if args.mode == "xsec":
    y_label_full = "95% CL upper limit on #sigma_{sig} [fb]"
    y_label_median = "95% CL median expected #sigma_{sig} [fb]"
else:
    y_label_full = "95% CL upper limit on #it{B}_{sig}"
    y_label_median = "95% CL median expected #it{B}_{sig}"

# Channel label drawn near MHc text on every plot.
_CHANNEL_LABELS = {
    "Combined": "e#mu#mu + #mu#mu#mu",
    "SR1E2Mu":  "e#mu#mu",
    "SR3Mu":    "#mu#mu#mu",
}
_channel_label_txt = _CHANNEL_LABELS[args.channel]

# iPos=11 keeps the luminosity on its own header line, so the split per-period
# string fits at full size. The CMS block itself is NOT left to cmsstyle, whose
# hardcoded 3.5%-of-frame offset puts it on the top-left ticks -- these figures
# place it exactly where the paper panels do (TriLepton/docs/PaperPlotting.md).
_iPos = 11
_CMS_LABEL_CONFIG = {
    "cmsPosX": CMS_LABEL_POS[0],
    "cmsPosY": CMS_LABEL_POS[1],
    "cmsLabelSize": CMS_LABEL_SIZE,
    # Published paper figures drop "Preliminary"; only "CMS" is drawn.
    "extraText": "",
}

# Information text: channel + mHc, stacked in two lines. The x position
# matches the CMS block (both share the frame's left inset) and the block's
# BOTTOM (the mHc line) matches the legend's bottom, so the two right-column
# and left-column blocks read as one row -- see _shape_panel() and
# _legend_text_bottom(). Both lines at one size, tighter than a paper caption:
# these are annotations on a limit curve, not a region tag, and the pair
# reads as a single block. There is deliberately NO bold "SR" tag -- this is
# a limit, not a region plot.
_INFO_SIZE_CHANNEL = 0.045
_INFO_SIZE_MHC = _INFO_SIZE_CHANNEL
_INFO_ROW = 0.050

# Legend: one column in the top-right, sized against the caption block on the
# left rather than cmsstyle's default -- at 0.035 it read small next to a 0.063
# final state. The row pitch tracks the text size, and the box is wider because
# "Expected #pm2#sigma" is the longest entry at this size.
# Panel shape. cmsCanvas(square=True) is 600x600 with margins L=0.155, R=0.05.
# A row of N panels across the text width W scales each by (W/N)/w_canvas, so a
# text of size t (a fraction of canvas HEIGHT) prints at t*W/(N*A) with
# A = w/h. Holding that fixed as N goes 3 -> 2 needs A * 3/N, i.e. 900x600 --
# which also makes both rows print at the same height, (W/N)/A = W/3.
_PANEL_HEIGHT = 600
_WIDTH_STRETCH = 3.0 / args.panels_per_row
_PANEL_WIDTH = int(round(_PANEL_HEIGHT * _WIDTH_STRETCH))

_LEGEND_TEXT_SIZE = 0.045
_LEGEND_ROW = 0.062
_LEGEND_X1 = 0.56
_LEGEND_X2 = 0.95
_LEGEND_Y2 = 0.90


# TLegend centres each entry's text within its row rather than hanging it
# off the box's Y1NDC, so the last row's glyphs ("Expected #pm2#sigma") sit
# above the box's bottom edge, not flush with it. Measured against the
# rendered PDF text bounding boxes (pdftotext -bbox) on the square panel:
# with the mHc line's bottom at the raw box edge, its lowest ink sat 12.63pt
# below "Expected #pm2#sigma"'s lowest ink, on a canvas whose visual height
# is 567pt -- 12.63 / 567 = 0.02228 of NDC. The gap is set by _LEGEND_ROW and
# _LEGEND_TEXT_SIZE alone, both NDC fractions of the pad height unaffected by
# --panels-per-row's width stretch, so this one constant applies to both
# panel shapes.
_LEGEND_LAST_ENTRY_GAP = 0.02228


def _legend_box_bottom(extra_entries=0):
    """NDC y of the legend box's bottom edge (its Y1NDC) -- the actual
    CMS.cmsLeg(...) coordinate, not where its last entry's text sits."""
    n_entries = (4 if not args.blind else 3) + extra_entries
    return _LEGEND_Y2 - _LEGEND_ROW * n_entries


def _legend_text_bottom(extra_entries=0):
    """NDC y of the legend's LAST ENTRY text (e.g. "Expected #pm2#sigma"),
    for anchoring the info block's last line to the text a reader actually
    sees rather than the box's own Y1NDC; see _LEGEND_LAST_ENTRY_GAP."""
    return _legend_box_bottom(extra_entries) + _LEGEND_LAST_ENTRY_GAP


def _shape_panel(canv, cms_state, legend_bottom):
    """Widen the canvas for a two-per-row layout, undoing every side effect.

    Margins and y-axis label/title offsets are fractions of pad WIDTH, so a
    1.5x wider canvas puts 1.5x the physical gap into each of them; dividing by
    the stretch keeps the frame, the axis title and the caption exactly where
    the square panels have them. Returns the position of the info block's two
    lines -- x matching the CMS block, and the bottom (mHc) line's y matching
    legend_bottom, the legend box's own bottom edge, so the two right- and
    left-hand blocks read as one row -- and the legend box's x-span.
    """
    info_bottom = legend_bottom
    info_top = info_bottom + _INFO_ROW

    if _WIDTH_STRETCH == 1.0:
        return ((cms_state["posX"], info_top), (cms_state["posX"], info_bottom),
                (_LEGEND_X1, _LEGEND_X2))

    left0, right0 = canv.GetLeftMargin(), canv.GetRightMargin()
    canv.SetCanvasSize(_PANEL_WIDTH, _PANEL_HEIGHT)
    canv.SetWindowSize(_PANEL_WIDTH, _PANEL_HEIGHT)
    left, right = left0 / _WIDTH_STRETCH, right0 / _WIDTH_STRETCH
    canv.SetLeftMargin(left)
    canv.SetRightMargin(right)

    axis = CMS.GetCmsCanvasHist(canv).GetYaxis()
    axis.SetLabelOffset(axis.GetLabelOffset() / _WIDTH_STRETCH)
    axis.SetTitleOffset(axis.GetTitleOffset() / _WIDTH_STRETCH)
    # CMS_lumi() ran while the canvas was still square, so its header is still
    # anchored to the old right margin.
    reanchor_lumi_header(canv, _PANEL_WIDTH, _PANEL_HEIGHT)

    cms_state["posX"] = left + (cms_state["posX"] - left0) / _WIDTH_STRETCH
    legend_x2 = 1.0 - right
    legend_x1 = legend_x2 - (_LEGEND_X2 - _LEGEND_X1) / _WIDTH_STRETCH

    canv.Modified()
    canv.Update()
    return ((cms_state["posX"], info_top),
            (cms_state["posX"], info_bottom),
            (legend_x1, legend_x2))


def _ymax_from(limits_dict):
    """Dynamic y-max: 2x the maximum exp+2sigma across all mass points."""
    return 2.0 * max(v["exp+2"] for v in limits_dict.values())


# Fixed y-axis ceiling for the seven Combined panels that make up the paper
# figure: Baseline at mHc <= 85 (ParticleNet does not reach below 100) and
# ParticleNet at mHc >= 100. ONE scale across all seven, so the rows can be read
# across; mHc 70/85 then sit low, which is the honest picture -- their mA range
# stops short of the Z peak, which is exactly why their limits are flat.
# Everything else keeps the dynamic scale: Combined Baseline at mHc >= 100
# reaches 16.5e-6 and the single channels 23.9e-6.
# A round ceiling per unit. These two happen to agree almost exactly: under the
# BR -> sigma_sig factor (2 x sigma(pp->ttbar)(13 TeV) = 1667.8 pb) 15e-6 is
# 25.017 fb, so the two versions of the figure are the same plot to 0.07% of
# height. Tallest drawn point on any paper panel is 11.1e-6 / 18.6 fb.
_FIXED_YMAX = {"BR": 15e-6, "xsec": 25.0}


def _is_paper_panel(mhc):
    """One of the seven Combined panels of the paper figure?

    The arm split IS the mHc split: ParticleNet covers mHc >= 100, Baseline the
    two below it.
    """
    if args.channel != "Combined" or mhc is None:
        return False
    return (args.method == "ParticleNet") == (mhc >= 100)


def _resolve_ymax(limits_dicts, mhc):
    """Fixed ceiling on the paper panels, dynamic everywhere else.

    Raises rather than clipping. A truncated +2sigma band at the Z peak is the
    feature the reader is looking at, and this module's convention is to fail on
    bad input rather than fall back silently -- so a re-collection that shifts a
    limit upward surfaces here instead of in a quietly cropped figure.
    """
    dicts = [d for d in limits_dicts if d]
    if args.ymax:
        return args.ymax
    if not _is_paper_panel(mhc):
        return max(_ymax_from(d) for d in dicts)

    ceiling = _FIXED_YMAX[args.mode]
    tallest = max(max(max(v["exp+2"], 0.0 if args.blind else v.get("obs", 0.0))
                      for v in d.values())
                  for d in dicts)
    if tallest > ceiling:
        raise ValueError(
            f"fixed y-max {ceiling:g} would clip {args.channel}/{args.method}/"
            f"MHc{mhc} ({args.mode} mode): tallest drawn point is {tallest:g}. "
            f"Pass --ymax to override deliberately.")
    return ceiling


_ch_suffix = "" if args.channel == "Combined" else f".{args.channel}"

def _filter_by_mhc(limits_dict, mhc_value):
    """Return only mass points whose MHc matches mhc_value."""
    prefix = f"MHc{mhc_value}_"
    return {mp: v for mp, v in limits_dict.items() if mp.startswith(prefix)}


_json_dir = f"results/json/{args.mode}/{args.era}"

# The paper figures only ever draw the production scan-grid collection, so
# unlike plotLimits.py this script has no --signal-source flag -- it always
# reads the file collectLimits.py writes for interp-signal, and (unlike
# _ch_suffix/_mode_suffix below) this token never appears in the OUTPUT name,
# since it no longer varies.
_SOURCE_INFIX = ".interp-signal"


# ---------------------------------------------------------------------------
# --combine-mhc: one grid figure, CMS-HIG-20-012 Figure 5 style
# (https://cms-results.web.cern.ch/cms-results/public-results/publications/
# HIG-20-012/CMS-HIG-20-012_Figure_005.png) -- a small Brazilian-limit panel
# per mHc, one shared CMS/lumi header and one shared legend for the whole
# grid, instead of the single-panel figures' seven repeated ones. DRAFT: a
# first pass at the grid, layout constants are placeholders for iteration.
# ---------------------------------------------------------------------------

# The canonical seven-panel paper set (see _is_paper_panel below): Baseline
# below mHc 100, ParticleNet at and above it. Ordered increasing top-to-
# bottom, so row 0 (top) is the NARROWEST panel and the last (bottom) row is
# the WIDEST -- see COMBINE_GRID_ROWS below.
COMBINE_MHC_VALUES = (70, 85, 100, 115, 130, 145, 160)
# One column: every row shares the same HEIGHT and the same GeV-per-NDC scale
# on the shared x (mA) axis, so row WIDTH is proportional to that row's own
# mA range (15 to mHc-5), left-aligned at COMBINE_X_MIN, instead of every row
# spanning the full column width.
COMBINE_X_MIN = 15.0

# NDC of the WHOLE canvas reserved for the column; the strips outside it
# carry the shared CMS/lumi header (top) and axis titles (left/bottom).
COMBINE_GRID_LEFT = 0.064
COMBINE_GRID_RIGHT = 0.99
COMBINE_GRID_TOP = 0.900
COMBINE_GRID_BOTTOM = 0.075
COMBINE_PAD_GAP = 0.004
# The two shared axis-direction arrows and the titles that sit beside them: the
# y arrow runs up the left edge between its title and the rows, the x arrow runs
# along the bottom with its title at the arrowhead (CMS-HIG-20-012 Fig. 5).
# Both sit just outside the grid -- they mark the direction of the axes, so a
# wide gap only reads as the figure having sloppy margins.
# The y title's baseline, not its centre line: rotated 90 deg its descenders
# (the "sig" subscript) fall towards +x, so the baseline has to clear the arrow
# by more than the gap alone suggests.
COMBINE_Y_TITLE_X = 0.036
COMBINE_Y_ARROW_X = 0.058
COMBINE_X_ARROW_Y = 0.055
COMBINE_ARROW_HEAD = 0.008
COMBINE_ARROW_WIDTH = 2
# Baseline of the CMS/luminosity header line, just above the top row.
COMBINE_HEADER_Y = 0.910
# Per-row pad margins, also in canvas NDC: left holds the y tick labels, right
# the overhang of each row's last x tick label.
COMBINE_MARGIN_L = 0.047
COMBINE_MARGIN_R = 0.018
# Vertical pad margins, as pad fractions (every row is the same height, so
# these need no canvas-NDC treatment). The bottom holds each row's own x tick
# labels; keeping it tight is what closes the gap between rows.
COMBINE_MARGIN_B = 0.28
COMBINE_MARGIN_T = 0.06

# Square page. Every text size below is NDC of the canvas HEIGHT, so what a
# reader actually sees is the size relative to the WIDTH -- multiply by H/W.
# At 1:1 the two coincide, which is why these are all ~1.43x the values the
# earlier 1400x2000 draft needed to print at the same physical size. Calibrated
# against CMS-HIG-20-012 Fig. 5, whose CMS label is 2.1% of the figure width
# and whose tick labels are 1.3%; these run larger, since this figure is one
# column wide and not three.
COMBINE_CANVAS_SIZE = (1800, 1800)
COMBINE_MHC_LABEL_SIZE = 0.245  # NDC of the PAD
COMBINE_AXIS_LABEL_SIZE = 0.198
COMBINE_LEGEND_TEXT_SIZE = 0.160  # NDC of the legend's OWN pad (see below)
COMBINE_LEGEND_MARGIN = 0.32      # of the legend width, for the line/box samples
COMBINE_LEGEND_TEXT_PAD = 0.008   # canvas NDC of slack past the widest label
COMBINE_CMS_SIZE = 0.047          # NDC of the CANVAS, from here down
COMBINE_LUMI_SIZE = 0.034
COMBINE_AXIS_TITLE_SIZE = 0.040
# The channel sits with the legend but reads as a caption on the figure, so it
# matches the axis titles rather than competing with them.
COMBINE_CHANNEL_SIZE = COMBINE_AXIS_TITLE_SIZE

# Linear y-axis (no log), so the band values are read off a plain scale. B_sig
# lives at O(1e-6), which a linear axis would otherwise label with a stray
# "x10^-6" exponent per panel; the graphs are scaled by COMBINE_Y_SCALE instead
# and the factor is stated once, in the shared y-axis title. xsec is already in
# fb and needs no rescaling.
COMBINE_Y_SCALE = 1.0 if args.mode == "xsec" else 1e6
COMBINE_Y_TITLE = (y_label_full if args.mode == "xsec"
                   else "95% CL upper limit on #it{B}_{sig} [10^{#minus6}]")

# Trimmed ceiling for the grid, in COMBINE_Y_SCALE units. The single-panel
# figures' 15e-6 was picked for a log axis; on a linear one it left the top
# third of every row empty. The tallest point drawn anywhere on the seven grid
# panels is 11.1 (the mHc100 ParticleNet Z peak) / 18.5 fb, so these keep
# healthy headroom -- _draw_*_panel raise rather than clip if a re-collection
# pushes past them.
#
# The ceiling must also sit ABOVE the topmost tick label, not on it: a label
# centred on y_max has half its height outside the frame, and with a pad top
# margin this tight ROOT clips it at the pad edge. 20.0 did exactly that to the
# xsec "20". Keeping the ceiling at ~1.2x the top label (12 over 10 for BR,
# 24 over 20 for xsec) leaves the label whole without costing frame height.
COMBINE_YMAX = {"BR": 12.0, "xsec": 24.0}


def _panel_method(mhc):
    """Which arm covers this mHc in the canonical seven-panel paper set."""
    return "ParticleNet" if mhc >= 100 else "Baseline"


def _rescale_graphs(graphs):
    """Scale a create_graphs() bundle's y values by COMBINE_Y_SCALE in place.

    Only the --combine-mhc figure does this; the single-panel figures keep the
    raw B_sig values on their log axis, where no exponent label appears.
    """
    if COMBINE_Y_SCALE == 1.0:
        return graphs
    for key in ('obs', 'exp', 'exp1sigma', 'exp2sigma'):
        g = graphs[key]
        for i in range(g.GetN()):
            g.SetPointY(i, g.GetPointY(i) * COMBINE_Y_SCALE)
        if isinstance(g, ROOT.TGraphAsymmErrors):
            for i in range(g.GetN()):
                g.SetPointEYlow(i, g.GetErrorYlow(i) * COMBINE_Y_SCALE)
                g.SetPointEYhigh(i, g.GetErrorYhigh(i) * COMBINE_Y_SCALE)
    return graphs


def _assert_fits(graphs, keys, y_max, mhc):
    """Fail fast if anything actually drawn would be clipped by the ceiling.

    Same policy as _resolve_ymax() on the single-panel figures: a band that
    runs off the top is the feature the reader is looking at, so it surfaces
    here rather than in a quietly cropped figure.
    """
    for key in keys:
        g = graphs[key]
        for i in range(g.GetN()):
            # TGraph (obs, exp) has no errors and returns -1 here.
            err = max(g.GetErrorYhigh(i), 0.0)
            top = g.GetPointY(i) + err
            if top > y_max:
                raise ValueError(
                    f"--combine-mhc y-max {y_max:g} would clip MHc{mhc} "
                    f"'{key}' at mA = {g.GetPointX(i):g} ({top:g}). Raise "
                    f"COMBINE_YMAX['{args.mode}'].")


def _load_baseline_limits():
    with open(f"{_json_dir}/limits.{args.era}{_ch_suffix}.{args.limit_type}"
              f".Baseline{_SOURCE_INFIX}.json") as f:
        return json.load(f)


def _draw_baseline_panel(pad, mhc, limits_all, y_min, y_max, keepalive):
    limits = _filter_by_mhc(limits_all, mhc)
    if not limits:
        raise RuntimeError(f"No mass points found for MHc{mhc} in JSON")
    graphs = _rescale_graphs(create_graphs(limits))
    _assert_fits(graphs, ('exp2sigma',) + (() if args.blind else ('obs',)), y_max, mhc)
    graphs['exp2sigma'].SetFillColor(ROOT.TColor.GetColor("#85D1FBff"))
    graphs['exp1sigma'].SetFillColor(ROOT.TColor.GetColor("#FFDF7Fff"))
    pad.cd()
    graphs['exp2sigma'].Draw("E3 same")
    graphs['exp1sigma'].Draw("E3 same")
    graphs['exp'].Draw("L same")
    if not args.blind:
        graphs['obs'].Draw("LP same")
    keepalive.append(graphs)
    return graphs


def _draw_particlenet_panel(pad, mhc, limits_baseline_all, limits_pnet_all, y_min, y_max, keepalive):
    def _ma_of(mp):
        return float(srspaths.parse_ma_token(mp.split("_MA")[1]))

    limits_pnet = _filter_by_mhc(limits_pnet_all, mhc)
    if not limits_pnet:
        raise RuntimeError(f"No ParticleNet mass points found for MHc{mhc} in JSON")
    limits_baseline = _filter_by_mhc(limits_baseline_all, mhc)

    pnet_mass = [_ma_of(mp) for mp in limits_pnet.keys()]
    pnet_min, pnet_max = min(pnet_mass), max(pnet_mass)

    limits_below = {mp: limits_baseline[mp] for mp in limits_baseline if _ma_of(mp) < pnet_min}
    limits_above = {mp: limits_baseline[mp] for mp in limits_baseline if _ma_of(mp) > pnet_max}

    def _boundary_anchor(target_ma):
        return {mp: v for mp, v in limits_baseline.items()
                if abs(_ma_of(mp) - target_ma) < 1e-9}

    limits_below = {**limits_below, **_boundary_anchor(pnet_min)}
    limits_above = {**limits_above, **_boundary_anchor(pnet_max)}
    limits_below_exp = limits_below if len(limits_below) >= 2 else {}
    limits_above_exp = limits_above if len(limits_above) >= 2 else {}

    graphs_pnet = create_graphs(limits_pnet)
    graphs_below = create_graphs(limits_below) if limits_below else None
    graphs_above = create_graphs(limits_above) if limits_above else None
    graphs_below_exp = create_graphs(limits_below_exp) if limits_below_exp else None
    graphs_above_exp = create_graphs(limits_above_exp) if limits_above_exp else None
    for g in (graphs_pnet, graphs_below, graphs_above,
              graphs_below_exp, graphs_above_exp):
        if g:
            _rescale_graphs(g)
    obs_keys = () if args.blind else ('obs',)
    for g in (graphs_pnet, graphs_below_exp, graphs_above_exp):
        if g:
            _assert_fits(g, ('exp2sigma',), y_max, mhc)
    for g in (graphs_pnet, graphs_below, graphs_above):
        if g:
            _assert_fits(g, obs_keys, y_max, mhc)

    pad.cd()
    for g in (graphs_pnet, graphs_below_exp, graphs_above_exp):
        if g:
            g['exp2sigma'].SetFillColor(ROOT.TColor.GetColor("#85D1FBff"))
            g['exp1sigma'].SetFillColor(ROOT.TColor.GetColor("#FFDF7Fff"))
            g['exp2sigma'].Draw("E3 same")
            g['exp1sigma'].Draw("E3 same")
    if graphs_below_exp:
        graphs_below_exp['exp'].Draw("L same")
    if graphs_above_exp:
        graphs_above_exp['exp'].Draw("L same")
    graphs_pnet['exp'].Draw("L same")
    if not args.blind:
        for g in (graphs_pnet, graphs_below, graphs_above):
            if g:
                g['obs'].Draw("LP same")

    # Real (non-log) user values: pad.GetUymin()/GetUymax() would return
    # log10 of the range on a log-scale pad, not the value DrawLine expects.
    line = ROOT.TLine()
    line.SetLineColor(ROOT.kBlack)
    line.SetLineStyle(2)
    line.SetLineWidth(1)
    line.DrawLine(pnet_min, y_min, pnet_min, y_max)
    line.DrawLine(pnet_max, y_min, pnet_max, y_max)

    keepalive.extend([graphs_pnet, graphs_below, graphs_above,
                      graphs_below_exp, graphs_above_exp, line])
    return graphs_pnet


def draw_combined_mhc_grid():
    """Build and save the combined grid figure; see the module banner above."""
    # Linear axis from zero, in COMBINE_Y_SCALE units (see COMBINE_Y_TITLE).
    y_max = COMBINE_YMAX[args.mode]
    y_min = 0.0

    limits_baseline_all = _load_baseline_limits()
    with open(f"{_json_dir}/limits.{args.era}{_ch_suffix}.{args.limit_type}"
              f".ParticleNet{_SOURCE_INFIX}.json") as f:
        limits_pnet_all = json.load(f)

    canv = ROOT.TCanvas("limit_combine", "limit_combine", *COMBINE_CANVAS_SIZE)
    canv.SetFillColor(0)
    canv.SetBorderMode(0)

    # Row i's width is its own (x_max - x_min) share of the WIDEST row's span
    # (mHc160, the last/bottom row), so 1 GeV is the same NDC (and pixel)
    # distance in every row, left-aligned at COMBINE_X_MIN -- a staircase
    # narrowing towards the top row, which is exactly what leaves the top
    # row's own top-right corner empty for the legend below.
    x_ranges = [(mhc, COMBINE_X_MIN, float(mhc - 5)) for mhc in COMBINE_MHC_VALUES]
    max_span = max(xmax - xmin for _, xmin, xmax in x_ranges)
    n_rows = len(x_ranges)
    row_h = (COMBINE_GRID_TOP - COMBINE_GRID_BOTTOM) / n_rows

    # Pad margins in CANVAS NDC, not pad fractions: the rows have different
    # widths, so one pad fraction would give each row a different absolute
    # margin -- the y labels would step in and the last x label would fall off
    # the narrow rows. The widest FRAME then gets whatever the column has left.
    frame_w_max = COMBINE_GRID_RIGHT - COMBINE_GRID_LEFT - COMBINE_MARGIN_L - COMBINE_MARGIN_R

    keepalive = []
    legend_graphs = None
    top_row_x2 = top_row_y1 = top_row_y2 = None
    for idx, (mhc, x_min, x_max) in enumerate(x_ranges):
        span = x_max - x_min
        x2 = (COMBINE_GRID_LEFT + COMBINE_MARGIN_L
              + (span / max_span) * frame_w_max + COMBINE_MARGIN_R)
        y2 = COMBINE_GRID_TOP - idx * row_h - COMBINE_PAD_GAP / 2
        y1 = COMBINE_GRID_TOP - (idx + 1) * row_h + COMBINE_PAD_GAP / 2
        pad_w = x2 - COMBINE_GRID_LEFT

        canv.cd()
        pad = ROOT.TPad(f"pad_mhc{mhc}", "", COMBINE_GRID_LEFT, y1, x2, y2)
        # Every row carries its own x-axis tick labels: the rows are not
        # visually contiguous (each one stops at its own mA ceiling), so a
        # single bottom-row scale would have to be read across a gap.
        pad.SetMargin(COMBINE_MARGIN_L / pad_w, COMBINE_MARGIN_R / pad_w,
                      COMBINE_MARGIN_B, COMBINE_MARGIN_T)
        pad.Draw()
        pad.cd()

        frame = pad.DrawFrame(x_min, y_min, x_max, y_max)
        frame.GetXaxis().SetLabelSize(COMBINE_AXIS_LABEL_SIZE)
        frame.GetYaxis().SetLabelSize(COMBINE_AXIS_LABEL_SIZE)
        frame.GetXaxis().SetLabelOffset(0.015)
        frame.GetXaxis().SetNdivisions(508)
        # Few, widely spaced y labels: the rows are short, so the 0/2/.../12
        # ladder collided with itself and with the mHc label.
        frame.GetYaxis().SetNdivisions(503)
        keepalive.append(frame)

        if _panel_method(mhc) == "Baseline":
            graphs = _draw_baseline_panel(
                pad, mhc, limits_baseline_all, y_min, y_max, keepalive)
        else:
            graphs = _draw_particlenet_panel(
                pad, mhc, limits_baseline_all, limits_pnet_all, y_min, y_max, keepalive)
        legend_graphs = legend_graphs or graphs
        pad.RedrawAxis()

        label = ROOT.TLatex()
        label.SetNDC(True)
        label.SetTextFont(42)
        label.SetTextSize(COMBINE_MHC_LABEL_SIZE)
        label.SetTextAlign(13)
        # Inset from the frame's top-left corner by a CANVAS-NDC amount, so the
        # label clears the y tick labels and the frame's own tick marks by the
        # same visible distance on every row, however wide that row is.
        label.DrawLatex((COMBINE_MARGIN_L + 0.030) / pad_w, 0.86,
                        f"m_{{H^{{+}}}} = {mhc} GeV")
        keepalive.append(label)

        if idx == 0:
            top_row_x2, top_row_y1, top_row_y2 = x2, y1, y2
        elif idx == 1:
            # The mHc85 frame: the legend runs down to its floor, and has to
            # clear this row's pad on the left as well as the top row's.
            second_row_x2 = x2
            second_frame_top = y2 - (row_h - COMBINE_PAD_GAP) * COMBINE_MARGIN_T
            second_frame_bottom = y1 + (row_h - COMBINE_PAD_GAP) * COMBINE_MARGIN_B
        if idx == n_rows - 1:
            # Where the mHc160 frame ends: the y arrow runs down to here.
            last_frame_bottom = y1 + (row_h - COMBINE_PAD_GAP) * COMBINE_MARGIN_B

    # Legend in the figure's top-right CORNER, registered to the panels rather
    # than to the canvas: its top edge is the mHc70 FRAME's top edge and its
    # right edge the widest (mHc160) frame's right edge, which is also where
    # the luminosity header ends. The staircase leaves that whole rectangle
    # empty. It runs down to the mHc70 frame's bottom edge, which leaves the
    # band beside the mHc85 row free for the channel label.
    canv.cd()
    header_left = COMBINE_GRID_LEFT + COMBINE_MARGIN_L
    header_right = COMBINE_GRID_RIGHT - COMBINE_MARGIN_R
    top_row_h = top_row_y2 - top_row_y1
    top_frame_top = top_row_y1 + top_row_h * (1.0 - COMBINE_MARGIN_T)
    # The legend's top edge is the mHc70 frame's top edge, and it runs down to
    # the mHc85 frame's FLOOR. The entry text size is capped by the slot height
    # (leg_h / 4), so spanning two rows rather than one is the only way to make
    # the entries bigger -- and the staircase leaves both those rows' right-hand
    # side empty, so nothing is in the way.
    leg_h = top_frame_top - second_frame_bottom

    entries = ([] if args.blind else [(legend_graphs['obs'], "Observed", "lp")]) + [
        (legend_graphs['exp'], "Expected", "l"),
        (legend_graphs['exp1sigma'], "Expected #pm1#sigma", "f"),
        (legend_graphs['exp2sigma'], "Expected #pm2#sigma", "f"),
    ]

    # TLegend packs its entries against its LEFT edge, so a pad that merely ends
    # at header_right leaves the labels stopping well short of it. Measure the
    # widest label at the size it will print (the legend's text size is a
    # fraction of ITS pad's height, hence the leg_h factor) and solve for the
    # pad left edge that puts that label's right end exactly on header_right:
    #   x1 + margin * (header_right - x1) + text_w = header_right.
    # TLatex::GetXsize() reports the width scaled by the canvas HEIGHT (the
    # same unit the NDC text size is in), not by its width, so on a portrait
    # canvas it under-reports the NDC width by exactly the aspect ratio.
    probe_size = COMBINE_LEGEND_TEXT_SIZE * leg_h
    aspect = COMBINE_CANVAS_SIZE[1] / COMBINE_CANVAS_SIZE[0]
    text_w = 0.0
    for _, txt, _ in entries:
        probe = ROOT.TLatex(0.0, 0.0, txt)
        probe.SetTextFont(42)
        probe.SetTextSize(probe_size)
        text_w = max(text_w, probe.GetXsize() * aspect)
    leg_w = (text_w + COMBINE_LEGEND_TEXT_PAD) / (1.0 - COMBINE_LEGEND_MARGIN)
    leg_x1 = max(header_right - leg_w,
                 max(top_row_x2, second_row_x2) + 2 * COMBINE_PAD_GAP)
    leg_w = header_right - leg_x1  # re-read, in case the clamp above bit

    leg_pad = ROOT.TPad("pad_combine_legend", "",
                        leg_x1, second_frame_bottom, header_right, top_frame_top)
    leg_pad.SetMargin(0.0, 0.0, 0.0, 0.0)
    leg_pad.Draw()
    leg_pad.cd()
    leg = ROOT.TLegend(0.0, 0.0, 1.0, 1.0)
    leg.SetNColumns(1)
    leg.SetBorderSize(0)
    leg.SetFillStyle(0)
    leg.SetMargin(COMBINE_LEGEND_MARGIN)
    leg.SetTextSize(COMBINE_LEGEND_TEXT_SIZE)
    for obj, txt, opt in entries:
        leg.AddEntry(obj, txt, opt)
    leg.Draw()
    keepalive.extend([leg, leg_pad])

    canv.cd()

    # Header/title sizes are NDC of the canvas HEIGHT -- see the COMBINE_*_SIZE
    # constants. CMS sits directly above the mHc70 row, flush with its FRAME's
    # left edge, and the luminosity flush with the legend's (and the widest
    # row's frame's) RIGHT edge, which is how CMS-HIG-20-012 Fig. 5 anchors
    # them -- not to the pad edges, which the y tick labels and the last x tick
    # label's overhang push outwards.
    cms_label = ROOT.TLatex()
    cms_label.SetNDC(True)
    cms_label.SetTextFont(61)
    cms_label.SetTextSize(COMBINE_CMS_SIZE)
    cms_label.SetTextAlign(11)
    cms_label.DrawLatex(header_left, COMBINE_HEADER_Y, "CMS")
    keepalive.append(cms_label)

    # The channel belongs with the legend, not with CMS: the top-right corner
    # has a whole empty band under the legend (beside the mHc85 row), and
    # crowding it against "CMS" only made the header look unbalanced. Left edge
    # flush with the legend's colour boxes, so the two read as one block:
    # TLegend centres each entry's symbol on x1 + margin/2 and draws it
    # 0.35*margin wide either side, so the boxes start at x1 + 0.15*margin.
    channel_label = ROOT.TLatex()
    channel_label.SetNDC(True)
    channel_label.SetTextFont(42)
    channel_label.SetTextSize(COMBINE_CHANNEL_SIZE)
    channel_label.SetTextAlign(13)
    channel_label.DrawLatex(leg_x1 + 0.15 * COMBINE_LEGEND_MARGIN * leg_w,
                            second_frame_bottom - 0.030,
                            _channel_label_txt)
    keepalive.append(channel_label)

    lumi_label = ROOT.TLatex()
    lumi_label.SetNDC(True)
    lumi_label.SetTextFont(42)
    lumi_label.SetTextSize(COMBINE_LUMI_SIZE)
    lumi_label.SetTextAlign(31)
    if args.era == "All":
        lumi_txt = (f"{LumiInfo['Run2']:g} fb^{{#minus1}} ({EnergyInfo['Run2']:g} TeV) + "
                    f"{LumiInfo['Run3']:g} fb^{{#minus1}} ({EnergyInfo['Run3']:g} TeV)")
    else:
        lumi_txt = f"{LumiInfo.get(args.era, LumiInfo_extended.get(args.era)):g} fb^{{#minus1}} ({get_CoM_energy(args.era):g} TeV)"
    lumi_label.DrawLatex(header_right, COMBINE_HEADER_Y, lumi_txt)
    keepalive.append(lumi_label)

    # Shared axis titles, each with the direction arrow CMS-HIG-20-012 Fig. 5
    # uses: no single panel owns the axis, so the arrow spans the whole grid and
    # says which way the quantity runs. A TArrow drawn on the canvas takes
    # canvas user coordinates, which for a pad that never got a frame are NDC.
    def _arrow(x1, y1, x2, y2):
        a = ROOT.TArrow(x1, y1, x2, y2, COMBINE_ARROW_HEAD, "|>")
        a.SetLineWidth(COMBINE_ARROW_WIDTH)
        a.SetLineColor(ROOT.kBlack)
        a.SetFillColor(ROOT.kBlack)
        a.SetAngle(35)
        a.Draw()
        keepalive.append(a)

    # Both arrows start and end on the FRAME edges, not the pad edges: the x
    # arrow spans the widest row's frame, the y arrow runs from the bottom
    # row's frame floor to the top row's frame ceiling. Anchoring them to the
    # pads instead let the y arrow overshoot by a whole bottom margin.
    _arrow(header_left, COMBINE_X_ARROW_Y, header_right, COMBINE_X_ARROW_Y)
    _arrow(COMBINE_Y_ARROW_X, last_frame_bottom,
           COMBINE_Y_ARROW_X, top_frame_top)

    # Titles sit at the arrowheads, the way the reference places them.
    x_title = ROOT.TLatex()
    x_title.SetNDC(True)
    x_title.SetTextFont(42)
    x_title.SetTextSize(COMBINE_AXIS_TITLE_SIZE)
    x_title.SetTextAlign(33)
    x_title.DrawLatex(header_right, COMBINE_X_ARROW_Y - 0.012, "m_{A} [GeV]")
    keepalive.append(x_title)

    y_title = ROOT.TLatex()
    y_title.SetNDC(True)
    y_title.SetTextFont(42)
    y_title.SetTextSize(COMBINE_AXIS_TITLE_SIZE)
    # Align 31 with the 90 deg rotation ends the string at the anchor, so the
    # title hangs from the top of the axis rather than sitting centred on it.
    y_title.SetTextAlign(31)
    y_title.SetTextAngle(90)
    y_title.DrawLatex(COMBINE_Y_TITLE_X, top_frame_top, COMBINE_Y_TITLE)
    keepalive.append(y_title)

    output_base = (f"results/plots/paper/Limits/{args.mode}/{args.era}/"
                  f"limit.{args.era}{_ch_suffix}.combineMHc")
    os.makedirs(os.path.dirname(output_base), exist_ok=True)
    canv.SaveAs(f"{output_base}.pdf")
    print(f"Created combined-mHc grid ({len(COMBINE_MHC_VALUES)} panels) -> {output_base}.pdf")
    canv._keepalive = keepalive


if args.combine_mhc:
    draw_combined_mhc_grid()
    sys.exit(0)

if args.method == "Baseline":
    with open(f"{_json_dir}/limits.{args.era}{_ch_suffix}.{args.limit_type}"
              f".Baseline{_SOURCE_INFIX}.json") as f:
        limits = json.load(f)

    mhc_value = args.mhc if args.mhc is not None else 160
    limits = _filter_by_mhc(limits, mhc_value)
    if not limits:
        raise RuntimeError(f"No mass points found for MHc{mhc_value} in JSON")
    graphs = create_graphs(limits)

    y_max = _resolve_ymax([limits], mhc_value)
    x_min = 15.0
    x_max = float(mhc_value - 5)
    _cms_state = configure_cms_label(_CMS_LABEL_CONFIG)
    canv = CMS.cmsCanvas("limit", x_min, x_max, 0., y_max,
                         "m_{A} [GeV]", y_label_full,
                         square=True, iPos=_iPos, extraSpace=0.01)
    restore_cms_label(_cms_state)
    _ch_pos, _mhc_pos, (_leg_x1, _leg_x2) = _shape_panel(canv, _cms_state, _legend_text_bottom())
    canv.cd()
    draw_cms_label(_cms_state)

    CMS.cmsObjectDraw(graphs['exp2sigma'], "E3", FillColor=ROOT.TColor.GetColor("#85D1FBff"))
    CMS.cmsObjectDraw(graphs['exp1sigma'], "E3 same", FillColor=ROOT.TColor.GetColor("#FFDF7Fff"))
    CMS.cmsObjectDraw(graphs['exp'], "L same")
    if not args.blind:
        CMS.cmsObjectDraw(graphs['obs'], "LP same")
    canv.RedrawAxis()

    mhc_label_txt = ROOT.TLatex()
    mhc_label_txt.SetNDC(True)
    mhc_label_txt.SetTextFont(42)
    mhc_label_txt.SetTextSize(_INFO_SIZE_CHANNEL)
    mhc_label_txt.DrawLatex(*_ch_pos, _channel_label_txt)
    mhc_label_txt.SetTextSize(_INFO_SIZE_MHC)
    mhc_label_txt.DrawLatex(*_mhc_pos, f"m_{{H^{{+}}}} = {mhc_value} GeV")

    leg = CMS.cmsLeg(_leg_x1, _legend_box_bottom(),
                     _leg_x2, _LEGEND_Y2, textSize=_LEGEND_TEXT_SIZE)
    if not args.blind:
        leg.AddEntry(graphs['obs'], "Observed", "lp")
    leg.AddEntry(graphs['exp'], "Expected", "l")
    leg.AddEntry(graphs['exp1sigma'], "Expected #pm1#sigma", "f")
    leg.AddEntry(graphs['exp2sigma'], "Expected #pm2#sigma", "f")

    print(f"Created Brazilian plot with {len(limits)} mass points (Baseline, MHc{mhc_value})")

elif args.method == "ParticleNet":
    # Load limits. Both files carry _SOURCE_INFIX: the ParticleNet reach
    # [82.5, 97.5] comes from the pnet scan, everything outside from the
    # Baseline interp scan (the two arms coexist).
    with open(f"{_json_dir}/limits.{args.era}{_ch_suffix}.{args.limit_type}.Baseline{_SOURCE_INFIX}.json") as f:
        limits_baseline = json.load(f)
    with open(f"{_json_dir}/limits.{args.era}{_ch_suffix}.{args.limit_type}.ParticleNet{_SOURCE_INFIX}.json") as f:
        limits_pnet = json.load(f)

    def _ma_of(mp):
        # p-notation-safe (MA87p5); plain ints parse identically.
        return float(srspaths.parse_ma_token(mp.split("_MA")[1]))

    if args.mhc is not None:
        limits_pnet = _filter_by_mhc(limits_pnet, args.mhc)
        if not limits_pnet:
            raise RuntimeError(f"No ParticleNet mass points found for MHc{args.mhc} in JSON")
        limits_baseline = _filter_by_mhc(limits_baseline, args.mhc)
    else:
        limits_baseline = {mp: v for mp, v in limits_baseline.items() if mp in _BASELINE_CURATED}

    # Split regions
    pnet_mass = [_ma_of(mp) for mp in limits_pnet.keys()]
    pnet_min, pnet_max = min(pnet_mass), max(pnet_mass)

    limits_below = {mp: limits_baseline[mp] for mp in limits_baseline if _ma_of(mp) < pnet_min}
    limits_above = {mp: limits_baseline[mp] for mp in limits_baseline if _ma_of(mp) > pnet_max}

    # Both the expected line/bands and the observed markers are continued onto the
    # ParticleNet window edge using the Baseline point sitting exactly at
    # m_A = pnet_min / pnet_max, so the Baseline and ParticleNet regions meet at the
    # boundary instead of leaving a hole. At the boundary mass the Baseline and
    # ParticleNet observed points are both drawn. Skipped when --mhc is omitted, where
    # the curated list can hold several entries at one m_A (e.g. two at m_A = 95).
    def _boundary_anchor(target_ma):
        if args.mhc is None:
            return {}
        return {mp: v for mp, v in limits_baseline.items()
                if abs(_ma_of(mp) - target_ma) < 1e-9}

    anchor_below = _boundary_anchor(pnet_min)
    anchor_above = _boundary_anchor(pnet_max)
    limits_below = {**limits_below, **anchor_below}
    limits_above = {**limits_above, **anchor_above}
    # A single point draws neither a line nor a band, so the expected graphs drop it;
    # it also stays out of the y_max scan (matters for MHc100, which has no "above"
    # region). The observed marker at that mass is still drawn from limits_below/above.
    limits_below_exp = limits_below if len(limits_below) >= 2 else {}
    limits_above_exp = limits_above if len(limits_above) >= 2 else {}

    # Create graphs. graphs_below/graphs_above feed the observed markers; the *_exp
    # graphs carry the expected line and bands.
    graphs_pnet = create_graphs(limits_pnet)
    graphs_below = create_graphs(limits_below) if limits_below else None
    graphs_above = create_graphs(limits_above) if limits_above else None
    graphs_below_exp = create_graphs(limits_below_exp) if limits_below_exp else None
    graphs_above_exp = create_graphs(limits_above_exp) if limits_above_exp else None

    y_max = _resolve_ymax([limits_pnet, limits_below_exp, limits_above_exp], args.mhc)
    x_max = float(args.mhc - 5) if args.mhc is not None else 155.0
    _cms_state = configure_cms_label(_CMS_LABEL_CONFIG)
    canv = CMS.cmsCanvas("limit", 15., x_max, 0., y_max,
                         "m_{A} [GeV]", y_label_full,
                         square=True, iPos=_iPos, extraSpace=0.01)
    restore_cms_label(_cms_state)
    _ch_pos, _mhc_pos, (_leg_x1, _leg_x2) = _shape_panel(
        canv, _cms_state, _legend_text_bottom(1 if args.stack_baseline else 0))
    canv.cd()
    draw_cms_label(_cms_state)

    # Draw all regions (bands and observed)
    for g in [graphs_pnet, graphs_below_exp, graphs_above_exp]:
        if g:
            CMS.cmsObjectDraw(g['exp2sigma'], "E3 same", FillColor=ROOT.TColor.GetColor("#85D1FBff"))
            CMS.cmsObjectDraw(g['exp1sigma'], "E3 same", FillColor=ROOT.TColor.GetColor("#FFDF7Fff"))

    # Optionally draw baseline for comparison — use baseline at the SAME (MHc, MA) as the PN
    # trained points so the overlay is apples-to-apples (PN spans varying MHc per point).
    if args.stack_baseline:
        limits_baseline_at_pnet = {mp: limits_baseline[mp] for mp in limits_pnet if mp in limits_baseline}
        if not limits_baseline_at_pnet:
            raise RuntimeError("Baseline JSON missing all PN trained mass points; cannot stack baseline.")
        graphs_baseline_at_pnet = create_graphs(limits_baseline_at_pnet)
        graphs_baseline_at_pnet['exp'].SetLineColor(ROOT.kRed+1)
        graphs_baseline_at_pnet['exp'].SetLineStyle(ROOT.kDashed)
        graphs_baseline_at_pnet['exp'].SetLineWidth(2)
        CMS.cmsObjectDraw(graphs_baseline_at_pnet['exp'], "L same")

    # Draw expected lines (baseline regions first, then ParticleNet on top)
    if graphs_below_exp:
        CMS.cmsObjectDraw(graphs_below_exp['exp'], "L same")
    if graphs_above_exp:
        CMS.cmsObjectDraw(graphs_above_exp['exp'], "L same")
    CMS.cmsObjectDraw(graphs_pnet['exp'], "L same")

    # Draw observed points
    if not args.blind:
        for g in [graphs_pnet, graphs_below, graphs_above]:
            if g:
                CMS.cmsObjectDraw(g['obs'], "LP same")

    # Draw vertical lines marking ParticleNet region
    line = ROOT.TLine()
    line.SetLineColor(ROOT.kBlack)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    separator_ymax = 0.63 * y_max
    line.DrawLine(pnet_min, 0, pnet_min, separator_ymax)
    line.DrawLine(pnet_max, 0, pnet_max, separator_ymax)

    if not args.blind:
        for g in [graphs_pnet, graphs_below, graphs_above]:
            if g:
                CMS.cmsObjectDraw(g['obs'], "LP same")
    canv.RedrawAxis()

    ch_label_pn = ROOT.TLatex()
    ch_label_pn.SetNDC(True)
    ch_label_pn.SetTextFont(42)
    ch_label_pn.SetTextSize(_INFO_SIZE_CHANNEL)
    ch_label_pn.DrawLatex(*_ch_pos, _channel_label_txt)
    if args.mhc is not None:
        ch_label_pn.SetTextSize(_INFO_SIZE_MHC)
        ch_label_pn.DrawLatex(*_mhc_pos, f"m_{{H^{{+}}}} = {args.mhc} GeV")

    # Legend
    leg = CMS.cmsLeg(_leg_x1, _legend_box_bottom(1 if args.stack_baseline else 0),
                         _leg_x2, _LEGEND_Y2, textSize=_LEGEND_TEXT_SIZE)
    if not args.blind:
        leg.AddEntry(graphs_pnet['obs'], "Observed", "lp")
    leg.AddEntry(graphs_pnet['exp'], "Expected", "l")
    leg.AddEntry(graphs_pnet['exp1sigma'], "Expected #pm1#sigma", "f")
    leg.AddEntry(graphs_pnet['exp2sigma'], "Expected #pm2#sigma", "f")
    if args.stack_baseline:
        leg.AddEntry(graphs_baseline_at_pnet['exp'], "w/o ParticleNet", "l")

    mhc_msg = f", MHc{args.mhc}" if args.mhc is not None else ""
    print(f"Created Brazilian plot with ParticleNet ({pnet_min}-{pnet_max} GeV{mhc_msg})")
    print(f"  ParticleNet: {len(limits_pnet)} mass points")
    if graphs_below:
        print(f"  Baseline (below): {len(limits_below)} mass points"
              f"{f' (incl. {len(anchor_below)} boundary anchor at MA{pnet_min})' if anchor_below else ''}")
    if graphs_above:
        print(f"  Baseline (above): {len(limits_above)} mass points"
              f"{f' (incl. {len(anchor_above)} boundary anchor at MA{pnet_max})' if anchor_above else ''}")
    if args.stack_baseline:
        print(f"  Baseline at PN points: {len(limits_baseline_at_pnet)} mass points overlay")

else:
    raise ValueError(f"Method {args.method} is not supported")

# Save outputs
_mode_suffix = (f".MHc{args.mhc}" if args.mhc is not None
               else f".MHc{160}" if args.method == "Baseline" else "")

# Under results/plots/paper/ -- the tree every plotPaper*.py script writes to
# (plotPaperLRModified.py, plotPaperTemplates.py, plotPaperPostfitSummary.py)
# -- rather than plotLimits.py's results/plots/{mode}/{era}/, so the two
# scripts' outputs never collide. "Asymptotic" and the source infix are
# dropped from the name (unlike the INPUT path above): both are fixed, so
# encoding them here would only be noise.
output_base = (f"results/plots/paper/Limits/{args.mode}/{args.era}/limit.{args.era}"
               f"{_ch_suffix}.{args.method}{_mode_suffix}")
os.makedirs(os.path.dirname(output_base), exist_ok=True)

# PDF only -- these are paper figures, not validation plots, so there is no
# use for the PNG plotLimits.py also writes.
canv.SaveAs(f"{output_base}.pdf")
