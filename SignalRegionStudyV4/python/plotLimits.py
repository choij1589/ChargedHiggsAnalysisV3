#!/usr/bin/env python3
import os
import math
from array import array
import argparse
import ROOT
import srspaths
import json
import cmsstyle as CMS
from plotter import (LumiInfo, EnergyInfo, get_CoM_energy, PALETTE_LONG,
                     configure_cms_label, draw_cms_label, restore_cms_label,
                     reanchor_lumi_header)
# Paper style is defined once in plotPaperLRModified.py and imported by every
# figure that has to match it, so the CMS block cannot drift between the limit
# curves and the paper panels.
from plotPaperLRModified import (CHANNEL_POS, CHANNEL_SIZE,
                                 CMS_LABEL_POS, CMS_LABEL_SIZE)

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
parser.add_argument("--method", type=str, required=True, help="Baseline, ParticleNet")
parser.add_argument("--limit_type", type=str, default="Asymptotic",
                    choices=["Asymptotic"], help="Limit type (Asymptotic only in V4)")
parser.add_argument("--blind", action="store_true", help="Hide observed limit (for blinded results)")
parser.add_argument("--ymax", type=float, default=None,
                    help="Override the automatic y-axis maximum (2x max exp+2sigma) — "
                         "for synchronizing scales across method/mHc comparisons")
parser.add_argument("--signal-source", type=str, default="mc-signal",
                    choices=["mc-signal", "interp-signal"],
                    help="Which collected-limits JSON to read (interp-signal: "
                         "the scan-grid collection; Baseline only)")
parser.add_argument("--stack_baseline", action="store_true", help="Show baseline expected limit on top (only for ParticleNet method)")
parser.add_argument("--mhc", type=int, default=None,
                    help="Plot only this MHc value. Baseline defaults to 160 when omitted; ParticleNet uses all trained points when omitted.")
parser.add_argument("--compare-mhc", dest="compare_mhc", action="store_true",
                    help="Overlay median expected limits for several MHc values (Baseline only)")
parser.add_argument("--mhc-list", dest="mhc_list", type=str, default="70,85,100,115,130,145,160",
                    help="Comma-separated MHc values for --compare-mhc (default: 70,85,100,115,130,145,160)")
parser.add_argument("--ratio-ref", dest="ratio_ref", type=int, default=160,
                    help="MHc used as the denominator of the --compare-mhc ratio "
                         "panel (default: 160). Must be one of --mhc-list.")
parser.add_argument("--ratio-range", dest="ratio_range", type=float, nargs=2,
                    default=None, metavar=("RMIN", "RMAX"),
                    help="Override the automatic y-range of the ratio panel.")
parser.add_argument("--panels-per-row", dest="panels_per_row", type=int,
                    default=3, choices=[2, 3],
                    help="how many of these panels share a row in the paper. "
                         "Text prints at t*W/N/A, so keeping one physical text "
                         "size across rows needs the canvas aspect A to scale "
                         "with 1/N: 3 -> square 600x600, 2 -> 900x600. Both "
                         "then print at the same height, W/3 (default: 3)")
parser.add_argument("--mode", type=str, default="BR", choices=["BR", "xsec"],
                    help="Limit unit: BR (relative branching ratio, default) or xsec (sigma_sig = 2 sigma(ttbar) B_sig in fb)")
args = parser.parse_args()

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


def _ma_of_point(mp):
    """m_A of a mass-point key, p-notation-safe (MA87p5)."""
    return float(srspaths.parse_ma_token(mp.split("_")[1][2:]))


def create_ratio_graph(limits_dict, ref_by_ma):
    """exp0(this MHc) / exp0(reference MHc), point by point in m_A.

    Nothing is interpolated. Every mHc column of a scan is drawn on the same
    m_A lattice (the bands of configs/grid.json do not depend on mHc), so each
    column's lattice is a subset of the widest one's and the ratio exists
    exactly at the points the column already carries. A point the reference
    column does NOT carry -- it only happens where the reference stops short in
    m_A -- is skipped and reported, the collectLimits.py convention: there is no
    ratio to draw there, and filling it in would silently interpolate the
    reference in m_A, which is a model statement this figure is not making.

    Returns (TGraph, [skipped m_A]).
    """
    shared = sorted((mp for mp in limits_dict if _ma_of_point(mp) in ref_by_ma),
                    key=_ma_of_point)
    skipped = sorted(_ma_of_point(mp) for mp in limits_dict
                     if _ma_of_point(mp) not in ref_by_ma)

    x = array('d', [_ma_of_point(mp) for mp in shared])
    y = array('d', [limits_dict[mp]["exp0"] / ref_by_ma[_ma_of_point(mp)]
                    for mp in shared])
    g = ROOT.TGraph(len(x), x, y)
    g.SetLineWidth(2)
    g.SetMarkerStyle(20)
    g.SetMarkerSize(0.8)
    return g, skipped


# The CMS 2016 (HEPData ins1735729) and ATLAS Run 2 (HEPData ins2654723) reference curves
# are no longer overlaid on the mH+ = 160 GeV BR plots; the paper figures show this result
# alone. The HEPData YAML files stay under results/yaml/ for reference.

# Setup CMS style
CMS.SetExtraText("Preliminary")
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
}

# Information text, stacked under the CMS block and anchored on the paper
# caption position. The final state keeps the caption size; the mass line below
# it is an annotation, not a caption, so it stays smaller. There is deliberately
# NO bold "SR" tag -- this is a limit, not a region plot.
_INFO_X = CHANNEL_POS[0]
_INFO_Y_CHANNEL = CHANNEL_POS[1]
# Both lines at one size, tighter than a paper caption: these are annotations
# on a limit curve, not a region tag, and the pair reads as a single block.
# The row pitch is just over the text height so the two sit close together.
_INFO_SIZE_CHANNEL = 0.045
_INFO_SIZE_MHC = _INFO_SIZE_CHANNEL
_INFO_ROW = 0.050
_INFO_Y_MHC = _INFO_Y_CHANNEL - _INFO_ROW

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
# Where the info block sits on a two-per-row panel: a fraction of the run from
# the left frame edge to the legend, and a drop below the CMS block's top line
# so it reads as a second, lower group rather than a continuation of it.
# See _shape_panel().
_INFO_X_FRACTION = 0.42
_INFO_TOP_DROP = 0.035

_LEGEND_TEXT_SIZE = 0.045
_LEGEND_ROW = 0.062
_LEGEND_X1 = 0.56
_LEGEND_X2 = 0.95
_LEGEND_Y2 = 0.90

# compareMHc carries seven entries instead of the Brazilian panel's four, in an
# upper pad the ratio panel has cut to ~69% of the canvas. Two columns of a
# slightly smaller entry keep the block clear of the Z peak; the row pitch
# tracks the text size the same way _LEGEND_ROW does.
_COMPARE_LEGEND_COLUMNS = 2
_COMPARE_LEGEND_TEXT_SIZE = 0.038
_COMPARE_LEGEND_ROW = 0.052
# Two columns need more than the single-column box, but the box grows to the
# LEFT (it stays flush against the right frame edge) and "Preliminary" already
# reaches x ~ 0.40. 1.28x is the widest that still clears it.
_COMPARE_LEGEND_WIDTH_SCALE = 1.28

# cmsDiCanvas sizes the y-axis TITLE for short labels ("Events / bin", "Obs /
# Exp") -- it scales cmsstyle's 0.04 up by H_ref/Hpad so the glyphs keep their
# physical height in a pad that is only 69% / 31% of the canvas. These two
# titles are long ("95% CL median expected sigma_sig [fb]" is 39 characters),
# and a title is laid out along the pad HEIGHT, which the split has cut in the
# same proportion -- so at that size both run off their pad, and the upper one
# lost its leading "9". Both are scaled by one factor, which keeps the two
# titles at a common physical size, rather than shortening the wording.
_COMPARE_YTITLE_SCALE = 0.70


def _shape_panel(canv, cms_state):
    """Widen the canvas for a two-per-row layout, undoing every side effect.

    Margins and y-axis label/title offsets are fractions of pad WIDTH, so a
    1.5x wider canvas puts 1.5x the physical gap into each of them; dividing by
    the stretch keeps the frame, the axis title and the caption exactly where
    the square panels have them. Returns the x for the info block (a fixed
    inset from the LEFT frame edge) and the legend box (a fixed width flush
    against the RIGHT frame edge) -- the two anchor to opposite sides, so one
    mapping cannot serve both.
    """
    if _WIDTH_STRETCH == 1.0:
        return ((_INFO_X, _INFO_Y_CHANNEL), (_INFO_X, _INFO_Y_MHC),
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

    if cms_state is not None:
        cms_state["posX"] = left + (cms_state["posX"] - left0) / _WIDTH_STRETCH
    legend_x2 = 1.0 - right
    legend_x1 = legend_x2 - (_LEGEND_X2 - _LEGEND_X1) / _WIDTH_STRETCH

    # A two-per-row panel is wide enough to carry the info block at the TOP,
    # centred in the frame and level with the CMS block, the way the postfit
    # summary does. The square three-per-row panels are NOT: "Preliminary"
    # already reaches x ~ 0.42 there and the legend starts at 0.56, so the block
    # stays under the CMS text on those.
    # Middle-LEFT of the space the legend leaves. Measured on the 900x600
    # panel: the CMS block ends at x = 0.233 and the legend starts at 0.716, so
    # this fraction of that span clears "Preliminary" comfortably while the
    # longest line still ends well short of the legend. Centring it (0.5)
    # straddles the frame midpoint and reads as "middle", not "middle-left".
    info_x = left + _INFO_X_FRACTION * (legend_x1 - left)
    canv.Modified()
    canv.Update()
    info_top = CMS_LABEL_POS[1] - _INFO_TOP_DROP
    return ((info_x, info_top),
            (info_x, info_top - _INFO_ROW),
            (legend_x1, legend_x2))


def _shape_dicanvas(canv, cms_state):
    """_shape_panel() for the two-pad compareMHc canvas.

    Same arithmetic, applied per PAD: on a cmsDiCanvas the margins and the
    y-axis offsets live on the sub-pads, not on the canvas, so the single-pad
    helper would read cmsstyle's untouched canvas margins and rescale nothing.
    Mirrors plotPaperPostfitSummary.resize_canvas(), which solves the identical
    problem for the stitched postfit panel.
    """
    if _WIDTH_STRETCH == 1.0:
        return ((_INFO_X, _INFO_Y_CHANNEL), (_INFO_X, _INFO_Y_MHC),
                (_LEGEND_X1, _LEGEND_X2))

    upper = canv.cd(1)
    left0, right0 = upper.GetLeftMargin(), upper.GetRightMargin()
    canv.SetCanvasSize(_PANEL_WIDTH, _PANEL_HEIGHT)
    canv.SetWindowSize(_PANEL_WIDTH, _PANEL_HEIGHT)
    for pad_index in (1, 2):
        pad = canv.cd(pad_index)
        pad.SetLeftMargin(pad.GetLeftMargin() / _WIDTH_STRETCH)
        pad.SetRightMargin(pad.GetRightMargin() / _WIDTH_STRETCH)
        axis = CMS.GetCmsCanvasHist(pad).GetYaxis()
        axis.SetLabelOffset(axis.GetLabelOffset() / _WIDTH_STRETCH)
        axis.SetTitleOffset(axis.GetTitleOffset() / _WIDTH_STRETCH)
    reanchor_lumi_header(canv.cd(1), _PANEL_WIDTH, _PANEL_HEIGHT)

    left = left0 / _WIDTH_STRETCH
    if cms_state is not None:
        cms_state["posX"] = left + (cms_state["posX"] - left0) / _WIDTH_STRETCH
    legend_x2 = 1.0 - right0 / _WIDTH_STRETCH
    legend_x1 = legend_x2 - (_LEGEND_X2 - _LEGEND_X1) / _WIDTH_STRETCH
    info_x = left + _INFO_X_FRACTION * (legend_x1 - left)
    canv.Modified()
    canv.Update()
    info_top = CMS_LABEL_POS[1] - _INFO_TOP_DROP
    return ((info_x, info_top),
            (info_x, info_top - _INFO_ROW),
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

# compareMHc draws MEDIANS ONLY -- no +-2sigma band -- so scaling its ceiling
# off exp+2sigma (what _ymax_from does for a Brazilian panel) leaves the curves
# in the bottom quarter of a frame whose top three quarters are empty, and the
# ratio panel makes the upper pad shorter still. The ceiling here is the drawn
# quantity instead, with just enough headroom to clear the legend block: at 1.55
# the Z peak -- the tallest median on the figure -- tops out at two thirds of the
# frame, comfortably below the legend's bottom row.
_COMPARE_HEADROOM = 1.55
# Ratio panel: pad the observed spread by a tenth of it and round outward to a
# 0.05 tick, so the extremes sit inside the frame rather than on the axis.
_RATIO_PAD_FRACTION = 0.10
_RATIO_TICK = 0.05


def _compare_ymax(limits_dicts):
    """Upper-pad ceiling for compareMHc: headroom over the tallest MEDIAN."""
    if args.ymax:
        return args.ymax
    return _COMPARE_HEADROOM * max(v["exp0"] for d in limits_dicts
                                   for v in d.values())


def _resolve_ratio_range(graphs):
    """Ratio-panel y-range from the drawn ratios, or the explicit override."""
    if args.ratio_range:
        return tuple(args.ratio_range)
    values = [g.GetPointY(i) for g in graphs for i in range(g.GetN())]
    lo, hi = min(values), max(values)
    pad = _RATIO_PAD_FRACTION * max(hi - lo, _RATIO_TICK)
    return (math.floor((lo - pad) / _RATIO_TICK) * _RATIO_TICK,
            math.ceil((hi + pad) / _RATIO_TICK) * _RATIO_TICK)


_ch_suffix = "" if args.channel == "Combined" else f".{args.channel}"

def _filter_by_mhc(limits_dict, mhc_value):
    """Return only mass points whose MHc matches mhc_value."""
    prefix = f"MHc{mhc_value}_"
    return {mp: v for mp, v in limits_dict.items() if mp.startswith(prefix)}


_json_dir = f"results/json/{args.mode}/{args.era}"

_source_infix = "" if args.signal_source == "mc-signal" \
    else f".{args.signal_source}"

if args.method == "Baseline":
    with open(f"{_json_dir}/limits.{args.era}{_ch_suffix}.{args.limit_type}"
              f".Baseline{_source_infix}.json") as f:
        limits = json.load(f)

    if args.compare_mhc:
        mhc_values = [int(v) for v in args.mhc_list.split(",") if v.strip()]
        per_mhc_limits = {}
        for mhc in mhc_values:
            sub = _filter_by_mhc(limits, mhc)
            if sub:
                per_mhc_limits[mhc] = sub
        if not per_mhc_limits:
            raise RuntimeError(f"No mass points found for any MHc in {mhc_values}")

        # The ratio panel needs a denominator that is actually drawn, and the
        # module's convention is to fail on bad input rather than quietly pick
        # a substitute -- a silently reassigned reference would relabel every
        # ratio on the panel.
        if args.ratio_ref not in per_mhc_limits:
            raise ValueError(
                f"--ratio-ref MHc{args.ratio_ref} is not among the drawn MHc "
                f"values {sorted(per_mhc_limits)}. Add it to --mhc-list or "
                f"pick another reference.")
        ref_by_ma = {_ma_of_point(mp): v["exp0"]
                     for mp, v in per_mhc_limits[args.ratio_ref].items()}

        # compareMHc spans every mHc at once, so it is not one of the
        # paper panels and keeps the dynamic scale.
        y_max = _compare_ymax(list(per_mhc_limits.values()))

        graphs_compare = []
        skipped_total = {}
        for idx, mhc in enumerate(sorted(per_mhc_limits.keys())):
            g = create_graphs(per_mhc_limits[mhc])
            color = PALETTE_LONG[idx % len(PALETTE_LONG)]
            g['exp'].SetLineColor(color)
            g['exp'].SetLineStyle(ROOT.kSolid)
            g['exp'].SetLineWidth(2)
            g['exp'].SetMarkerStyle(20)
            g['exp'].SetMarkerSize(0.8)
            g['exp'].SetMarkerColor(color)
            g_ratio, skipped = create_ratio_graph(per_mhc_limits[mhc], ref_by_ma)
            g_ratio.SetLineColor(color)
            g_ratio.SetMarkerColor(color)
            if skipped:
                skipped_total[mhc] = skipped
            graphs_compare.append((mhc, g, g_ratio))

        r_min, r_max = _resolve_ratio_range([r for _, _, r in graphs_compare])
        _cms_state = configure_cms_label(_CMS_LABEL_CONFIG)
        canv = CMS.cmsDiCanvas("limit", 15., 155., 0., y_max, r_min, r_max,
                               "m_{A} [GeV]", y_label_median,
                               f"Ratio to {args.ratio_ref} GeV",
                               square=True, iPos=_iPos, extraSpace=0.01)
        restore_cms_label(_cms_state)
        _ch_pos, _mhc_pos, (_leg_x1, _leg_x2) = _shape_dicanvas(canv, _cms_state)
        for pad_index in (1, 2):
            _axis = CMS.GetCmsCanvasHist(canv.cd(pad_index)).GetYaxis()
            _axis.SetTitleSize(_axis.GetTitleSize() * _COMPARE_YTITLE_SCALE)
            # The offset is measured in units of the title size, so shrinking
            # the text alone would also pull it in towards the axis.
            _axis.SetTitleOffset(_axis.GetTitleOffset() / _COMPARE_YTITLE_SCALE)
        _leg_x1 = _leg_x2 - (_leg_x2 - _leg_x1) * _COMPARE_LEGEND_WIDTH_SCALE
        canv.cd(1)
        draw_cms_label(_cms_state)

        for _, g, _r in graphs_compare:
            CMS.cmsObjectDraw(g['exp'], "LP same")
        canv.cd(1).RedrawAxis()

        ch_label = ROOT.TLatex()
        ch_label.SetNDC(True)
        ch_label.SetTextFont(42)
        ch_label.SetTextSize(_INFO_SIZE_CHANNEL)
        ch_label.DrawLatex(*_ch_pos, _channel_label_txt)

        # Two columns: seven entries in one column reach halfway down the
        # upper pad, which the ratio panel has already shortened, and the
        # legend then sits on the Z peak -- the tallest thing on the figure.
        n_rows = -(-len(graphs_compare) // _COMPARE_LEGEND_COLUMNS)
        leg = CMS.cmsLeg(_leg_x1, _LEGEND_Y2 - _COMPARE_LEGEND_ROW*n_rows,
                         _leg_x2, _LEGEND_Y2,
                         textSize=_COMPARE_LEGEND_TEXT_SIZE,
                         columns=_COMPARE_LEGEND_COLUMNS)
        for mhc, g, _r in graphs_compare:
            leg.AddEntry(g['exp'], f"m_{{H^{{+}}}} = {mhc} GeV", "lp")

        canv.cd(2)
        unity = ROOT.TLine()
        unity.SetLineStyle(ROOT.kDotted)
        unity.SetLineColor(ROOT.kBlack)
        unity.SetLineWidth(2)
        unity.DrawLine(15., 1.0, 155., 1.0)
        for mhc, _g, g_ratio in graphs_compare:
            # The reference divides itself: a flat 1 carries no information and
            # would only overdraw the guide line.
            if mhc == args.ratio_ref:
                continue
            CMS.cmsObjectDraw(g_ratio, "L same")
        canv.cd(2).RedrawAxis()

        print(f"Created MHc-comparison plot ({len(graphs_compare)} MHc values, Baseline)"
              f"; ratio reference MHc{args.ratio_ref}, "
              f"panel range [{r_min:.2f}, {r_max:.2f}]")
        for mhc, skipped in sorted(skipped_total.items()):
            print(f"  MHc{mhc}: {len(skipped)} point(s) have no MHc{args.ratio_ref} "
                  f"counterpart and carry no ratio: "
                  f"{', '.join(f'{m:g}' for m in skipped[:8])}"
                  f"{' ...' if len(skipped) > 8 else ''}")
    else:
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
        _ch_pos, _mhc_pos, (_leg_x1, _leg_x2) = _shape_panel(canv, _cms_state)
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

        n_entries = 4 if not args.blind else 3
        leg = CMS.cmsLeg(_leg_x1, _LEGEND_Y2 - _LEGEND_ROW*n_entries,
                         _leg_x2, _LEGEND_Y2, textSize=_LEGEND_TEXT_SIZE)
        if not args.blind:
            leg.AddEntry(graphs['obs'], "Observed", "lp")
        leg.AddEntry(graphs['exp'], "Expected", "l")
        leg.AddEntry(graphs['exp1sigma'], "Expected #pm1#sigma", "f")
        leg.AddEntry(graphs['exp2sigma'], "Expected #pm2#sigma", "f")

        print(f"Created Brazilian plot with {len(limits)} mass points (Baseline, MHc{mhc_value})")

elif args.method == "ParticleNet":
    # Load limits. For interp-signal both files carry the source infix: the
    # ParticleNet reach [82.5, 97.5] comes from the pnet scan, everything
    # outside from the Baseline interp scan (the two arms coexist).
    with open(f"{_json_dir}/limits.{args.era}{_ch_suffix}.{args.limit_type}.Baseline{_source_infix}.json") as f:
        limits_baseline = json.load(f)
    with open(f"{_json_dir}/limits.{args.era}{_ch_suffix}.{args.limit_type}.ParticleNet{_source_infix}.json") as f:
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
    _ch_pos, _mhc_pos, (_leg_x1, _leg_x2) = _shape_panel(canv, _cms_state)
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
    n_entries = (4 if not args.blind else 3) + (1 if args.stack_baseline else 0)
    leg = CMS.cmsLeg(_leg_x1, _LEGEND_Y2 - _LEGEND_ROW*n_entries,
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
if args.method != "Baseline":
    _mode_suffix = f".MHc{args.mhc}" if args.mhc is not None else ""
elif args.compare_mhc:
    _mode_suffix = ".compareMHc"
else:
    _mode_suffix = f".MHc{args.mhc if args.mhc is not None else 160}"

output_base = (f"results/plots/{args.mode}/{args.era}/limit.{args.era}"
               f"{_ch_suffix}.{args.limit_type}.{args.method}"
               f"{_source_infix}{_mode_suffix}")
os.makedirs(os.path.dirname(output_base), exist_ok=True)

canv.SaveAs(f"{output_base}.png")
canv.SaveAs(f"{output_base}.pdf")
