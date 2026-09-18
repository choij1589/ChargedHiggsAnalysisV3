#!/usr/bin/env python3
"""Supplementary: the V4 Combined limit panel with the frozen V3 limits on top.

Redraws the production Brazilian panel of plotLimits.py (interp-signal scan,
All / Combined) and overlays SignalRegionStudyV3's direct-MC limits at its MC
mass points as markers -- observed filled, median expected open -- so the
reproduction reads point by point off the published curve.

Validation-only, like compareToV3.py: it reads V3's frozen *output* JSONs
through --v3-dir, which is required and has no default, and never imports or
executes V3 code.

The panel itself is a copy of plotLimits.py's Combined path (layout, fixed
ceiling, ParticleNet window stitching), since that script runs at module level
and cannot be imported. Keep the two in step when the production panel changes.
"""
import argparse
import json
import os
import statistics
from array import array

import ROOT
import cmsstyle as CMS

import srspaths
from plotter import (LumiInfo, EnergyInfo, PALETTE_LONG,
                     configure_cms_label, draw_cms_label, restore_cms_label,
                     reanchor_lumi_header)
from plotPaperLRModified import CHANNEL_POS, CMS_LABEL_POS, CMS_LABEL_SIZE

ROOT.gROOT.SetBatch(ROOT.kTRUE)

ERA = "All"
PNET_MHC = (100, 115, 130, 145, 160)

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--v3-dir", dest="v3_dir", required=True,
                    help="SignalRegionStudyV3 directory; only its "
                         "results/json/{mode}/All/*.unblind.json are read")
parser.add_argument("--method", required=True, choices=["Baseline", "ParticleNet"])
parser.add_argument("--mhc", type=int, required=True)
parser.add_argument("--mode", default="BR", choices=["BR", "xsec"],
                    help="Limit unit: BR (default) or xsec (sigma_sig in fb)")
parser.add_argument("--panels-per-row", dest="panels_per_row", type=int,
                    default=None, choices=[2, 3],
                    help="Panel shape as in plotLimits.py (3 -> 600x600, "
                         "2 -> 900x600). Default follows the production "
                         "panels: 3 for mHc <= 100, 2 above.")
args = parser.parse_args()

if args.method == "ParticleNet" and args.mhc not in PNET_MHC:
    raise ValueError(f"ParticleNet has no arm at MHc{args.mhc}; trained mHc are {PNET_MHC}")
if args.panels_per_row is None:
    args.panels_per_row = 3 if args.mhc <= 100 else 2


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------
def _load(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


def v4_json(arm):
    return _load(f"results/json/{args.mode}/{ERA}/"
                 f"limits.{ERA}.Asymptotic.{arm}.interp-signal.json")


def v3_json(arm):
    return _load(os.path.join(args.v3_dir, "results", "json", args.mode, ERA,
                              f"limits.{ERA}.Asymptotic.{arm}.unblind.json"))


def ma_of(mp):
    """m_A of a mass-point key, p-notation-safe (MA87p5)."""
    return float(srspaths.parse_ma_token(mp.split("_MA")[1]))


def filter_mhc(limits, mhc):
    prefix = f"MHc{mhc}_"
    return {mp: v for mp, v in limits.items() if mp.startswith(prefix)}


# ---------------------------------------------------------------------------
# Style -- identical to plotLimits.py for everything the production panel draws
# ---------------------------------------------------------------------------
CMS.SetExtraText("Preliminary")
CMS.ResetAdditionalInfo()
CMS.SetLumi(None, run=(f"{LumiInfo['Run2']:g} fb^{{#minus1}} ({EnergyInfo['Run2']:g} TeV) + "
                       f"{LumiInfo['Run3']:g} fb^{{#minus1}}"))
CMS.SetEnergy(0, unit=f"{EnergyInfo['Run3']:g} TeV")

Y_LABEL = {"BR": "95% CL upper limit on #it{B}_{sig}",
           "xsec": "95% CL upper limit on #sigma_{sig} [fb]"}[args.mode]
CHANNEL_LABEL = "e#mu#mu + #mu#mu#mu"

_iPos = 11
_CMS_LABEL_CONFIG = {"cmsPosX": CMS_LABEL_POS[0], "cmsPosY": CMS_LABEL_POS[1],
                     "cmsLabelSize": CMS_LABEL_SIZE}

_INFO_X = CHANNEL_POS[0]
_INFO_Y_CHANNEL = CHANNEL_POS[1]
_INFO_SIZE = 0.045
_INFO_ROW = 0.050
_INFO_Y_MHC = _INFO_Y_CHANNEL - _INFO_ROW

_PANEL_HEIGHT = 600
_WIDTH_STRETCH = 3.0 / args.panels_per_row
_PANEL_WIDTH = int(round(_PANEL_HEIGHT * _WIDTH_STRETCH))
_INFO_X_FRACTION = 0.42
_INFO_TOP_DROP = 0.035

# Departure from plotLimits.py (0.045 / 0.062): six rows instead of four. At the
# production pitch the V3 rows reach the Z-peak +-2sigma band on the fixed-ceiling
# panels (ParticleNet MHc115: band top at y~0.55 NDC, bottom row at 0.56).
_LEGEND_TEXT_SIZE = 0.040
_LEGEND_ROW = 0.052
_LEGEND_ROWS = 6
_LEGEND_X1 = 0.56
_LEGEND_X2 = 0.95
_LEGEND_Y2 = 0.90
_SQUARE_LEGEND_SHIFT = 0.07   # see shape_panel()

_FIXED_YMAX = {"BR": 15e-6, "xsec": 25.0}

# The V3 overlay. Red and purple from the long palette: both read against the
# yellow and the light-blue band, where a blue marker would vanish in +-2sigma.
V3_OBS_COLOR = PALETTE_LONG[2]   # #bd1f01
V3_EXP_COLOR = PALETTE_LONG[4]   # #832db6
V3_MARKER_SIZE = 1.0
V3_OBS_MARKER = 20   # filled circle
V3_EXP_MARKER = 21   # filled square
# In-plot label only. The frozen V3 limits are the ones quoted in paper draft
# V5, which is how the reader knows them; code and docs keep calling them V3.
V3_LEGEND_TAG = "V5"


def create_graphs(limits):
    points = sorted(limits, key=ma_of)
    x = array('d', [ma_of(mp) for mp in points])
    n = len(x)
    vals = {key: array('d', [limits[mp][key] for mp in points])
            for key in ["obs", "exp0", "exp-1", "exp-2", "exp+1", "exp+2"]}

    g_obs = ROOT.TGraph(n, x, vals["obs"])
    g_obs.SetLineWidth(2)
    g_obs.SetMarkerStyle(20)
    g_obs.SetMarkerSize(0.8)

    g_exp = ROOT.TGraph(n, x, vals["exp0"])
    g_exp.SetLineWidth(2)
    g_exp.SetLineStyle(ROOT.kDashed)
    g_exp.SetLineColor(ROOT.kBlack)

    g_1s = ROOT.TGraphAsymmErrors(n)
    g_2s = ROOT.TGraphAsymmErrors(n)
    for i in range(n):
        for g in (g_1s, g_2s):
            g.SetPoint(i, x[i], vals["exp0"][i])
        g_1s.SetPointError(i, 0, 0, vals["exp0"][i] - vals["exp-1"][i],
                           vals["exp+1"][i] - vals["exp0"][i])
        g_2s.SetPointError(i, 0, 0, vals["exp0"][i] - vals["exp-2"][i],
                           vals["exp+2"][i] - vals["exp0"][i])
    return {"obs": g_obs, "exp": g_exp, "exp1sigma": g_1s, "exp2sigma": g_2s}


def create_v3_points(limits, key, color, style):
    points = sorted(limits, key=ma_of)
    g = ROOT.TGraph(len(points), array('d', [ma_of(mp) for mp in points]),
                    array('d', [limits[mp][key] for mp in points]))
    g.SetMarkerStyle(style)
    g.SetMarkerSize(V3_MARKER_SIZE)
    g.SetMarkerColor(color)
    g.SetLineColor(color)
    return g


def shape_panel(canv, cms_state):
    """plotLimits._shape_panel(): widen for a two-per-row layout.

    One departure: on the square panels the six-row legend is shifted left by
    _SQUARE_LEGEND_SHIFT. Its two V3 rows reach down to the Z-peak +-2sigma
    band, which on MHc100 sits at the right frame edge (text ended at x~0.92,
    band starts at ~0.88); shifted, it still starts right of "Preliminary".
    """
    if _WIDTH_STRETCH == 1.0:
        return ((_INFO_X, _INFO_Y_CHANNEL), (_INFO_X, _INFO_Y_MHC),
                (_LEGEND_X1 - _SQUARE_LEGEND_SHIFT, _LEGEND_X2 - _SQUARE_LEGEND_SHIFT))

    left0, right0 = canv.GetLeftMargin(), canv.GetRightMargin()
    canv.SetCanvasSize(_PANEL_WIDTH, _PANEL_HEIGHT)
    canv.SetWindowSize(_PANEL_WIDTH, _PANEL_HEIGHT)
    left, right = left0 / _WIDTH_STRETCH, right0 / _WIDTH_STRETCH
    canv.SetLeftMargin(left)
    canv.SetRightMargin(right)

    axis = CMS.GetCmsCanvasHist(canv).GetYaxis()
    axis.SetLabelOffset(axis.GetLabelOffset() / _WIDTH_STRETCH)
    axis.SetTitleOffset(axis.GetTitleOffset() / _WIDTH_STRETCH)
    reanchor_lumi_header(canv, _PANEL_WIDTH, _PANEL_HEIGHT)

    if cms_state is not None:
        cms_state["posX"] = left + (cms_state["posX"] - left0) / _WIDTH_STRETCH
    legend_x2 = 1.0 - right
    legend_x1 = legend_x2 - (_LEGEND_X2 - _LEGEND_X1) / _WIDTH_STRETCH

    info_x = left + _INFO_X_FRACTION * (legend_x1 - left)
    canv.Modified()
    canv.Update()
    info_top = CMS_LABEL_POS[1] - _INFO_TOP_DROP
    return ((info_x, info_top), (info_x, info_top - _INFO_ROW),
            (legend_x1, legend_x2))


def resolve_ymax(v4_dicts, v3_dicts):
    """plotLimits._resolve_ymax(), with the V3 markers held to the same rule.

    Fixed ceiling on the paper panels, 2x the V4 exp+2sigma peak elsewhere --
    so this panel shares its axis with the production one. A V3 marker above
    the ceiling raises instead of being clipped off the figure.
    """
    v4_dicts = [d for d in v4_dicts if d]
    is_paper = (args.method == "ParticleNet") == (args.mhc >= 100)
    if is_paper:
        ceiling = _FIXED_YMAX[args.mode]
    else:
        ceiling = 2.0 * max(v["exp+2"] for d in v4_dicts for v in d.values())
    tallest = max([max(v["exp+2"], v["obs"]) for d in v4_dicts for v in d.values()]
                  + [max(v["exp0"], v["obs"]) for d in v3_dicts for v in d.values()])
    if tallest > ceiling:
        raise ValueError(f"y-max {ceiling:g} would clip {args.method}/MHc{args.mhc} "
                         f"({args.mode}): tallest drawn point is {tallest:g}")
    return ceiling


def draw_bands(graphs):
    CMS.cmsObjectDraw(graphs["exp2sigma"], "E3 same", FillColor=ROOT.TColor.GetColor("#85D1FBff"))
    CMS.cmsObjectDraw(graphs["exp1sigma"], "E3 same", FillColor=ROOT.TColor.GetColor("#FFDF7Fff"))


# ---------------------------------------------------------------------------
# Limits: V4 as the production panel stitches them, V3 onto the same regions
# ---------------------------------------------------------------------------
v4_base = filter_mhc(v4_json("Baseline"), args.mhc)
v3_base = filter_mhc(v3_json("Baseline"), args.mhc)
if not v4_base:
    raise RuntimeError(f"No V4 Baseline points at MHc{args.mhc}")
if not v3_base:
    raise RuntimeError(f"No V3 Baseline points at MHc{args.mhc}")

pnet_window = None
if args.method == "Baseline":
    v4_regions = [v4_base]            # drawn with bands
    v4_obs_regions = [v4_base]
    v4_lookup = v4_base
    v3_points = dict(v3_base)
else:
    v4_pnet = filter_mhc(v4_json("ParticleNet"), args.mhc)
    v3_pnet = filter_mhc(v3_json("ParticleNet"), args.mhc)
    if not v4_pnet or not v3_pnet:
        raise RuntimeError(f"ParticleNet JSON missing MHc{args.mhc} "
                           f"(V4 {len(v4_pnet)}, V3 {len(v3_pnet)} points)")
    pnet_min = min(ma_of(mp) for mp in v4_pnet)
    pnet_max = max(ma_of(mp) for mp in v4_pnet)
    pnet_window = (pnet_min, pnet_max)

    # Same stitching as plotLimits.py: Baseline below/above the window, each
    # continued onto the window edge by the Baseline point sitting there.
    anchor_below = {mp: v for mp, v in v4_base.items() if abs(ma_of(mp) - pnet_min) < 1e-9}
    anchor_above = {mp: v for mp, v in v4_base.items() if abs(ma_of(mp) - pnet_max) < 1e-9}
    below = {**{mp: v for mp, v in v4_base.items() if ma_of(mp) < pnet_min}, **anchor_below}
    above = {**{mp: v for mp, v in v4_base.items() if ma_of(mp) > pnet_max}, **anchor_above}
    below_exp = below if len(below) >= 2 else {}
    above_exp = above if len(above) >= 2 else {}
    v4_regions = [v4_pnet, below_exp, above_exp]
    v4_obs_regions = [v4_pnet, below, above]

    # V3 mirrors it: its ParticleNet points inside the window, its Baseline
    # points outside, so no m_A carries two V3 markers.
    outside = [mp for mp in v3_pnet if not pnet_min <= ma_of(mp) <= pnet_max]
    if outside:
        raise RuntimeError(f"V3 ParticleNet points outside the V4 window "
                           f"[{pnet_min:g}, {pnet_max:g}]: {outside}")
    v3_points = {**{mp: v for mp, v in v3_base.items()
                    if not pnet_min <= ma_of(mp) <= pnet_max}, **v3_pnet}
    v4_lookup = {**v4_base, **v4_pnet}

missing = sorted(set(v3_points) - set(v4_lookup), key=ma_of)
if missing:
    raise RuntimeError(f"V3 points with no V4 counterpart: {missing}")

# ---------------------------------------------------------------------------
# Canvas
# ---------------------------------------------------------------------------
y_max = resolve_ymax(v4_regions, [v3_points])
x_min, x_max = 15.0, float(args.mhc - 5)

_cms_state = configure_cms_label(_CMS_LABEL_CONFIG)
canv = CMS.cmsCanvas("limit", x_min, x_max, 0., y_max, "m_{A} [GeV]", Y_LABEL,
                     square=True, iPos=_iPos, extraSpace=0.01)
restore_cms_label(_cms_state)
_ch_pos, _mhc_pos, (_leg_x1, _leg_x2) = shape_panel(canv, _cms_state)
canv.cd()
draw_cms_label(_cms_state)

graphs = [create_graphs(d) if d else None for d in v4_regions]
graphs_obs = [create_graphs(d) if d else None for d in v4_obs_regions]
for g in graphs:
    if g:
        draw_bands(g)
for g in reversed(graphs):         # Baseline regions first, ParticleNet on top
    if g:
        CMS.cmsObjectDraw(g["exp"], "L same")
for g in graphs_obs:
    if g:
        CMS.cmsObjectDraw(g["obs"], "LP same")

if pnet_window:
    # plotLimits.py stops the window guides at 0.63 y_max, just under its
    # four-row legend. The six-row legend reaches lower, so each guide stops
    # a small gap under the legend box instead (whichever is lower).
    bottom, top = canv.GetBottomMargin(), canv.GetTopMargin()
    legend_bottom = _LEGEND_Y2 - _LEGEND_ROW * _LEGEND_ROWS - 0.02
    guide_top = min(0.63, (legend_bottom - bottom) / (1.0 - bottom - top)) * y_max
    line = ROOT.TLine()
    line.SetLineColor(ROOT.kBlack)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    for edge in pnet_window:
        line.DrawLine(edge, 0, edge, guide_top)
    for g in graphs_obs:
        if g:
            CMS.cmsObjectDraw(g["obs"], "LP same")

# V3 on top of everything V4 draws.
# Observed last: where obs ~ exp the circle stays visible on the square.
g_v3_exp = create_v3_points(v3_points, "exp0", V3_EXP_COLOR, V3_EXP_MARKER)
g_v3_obs = create_v3_points(v3_points, "obs", V3_OBS_COLOR, V3_OBS_MARKER)
CMS.cmsObjectDraw(g_v3_exp, "P same")
CMS.cmsObjectDraw(g_v3_obs, "P same")
canv.RedrawAxis()

info = ROOT.TLatex()
info.SetNDC(True)
info.SetTextFont(42)
info.SetTextSize(_INFO_SIZE)
info.DrawLatex(*_ch_pos, CHANNEL_LABEL)
info.DrawLatex(*_mhc_pos, f"m_{{H^{{+}}}} = {args.mhc} GeV")

leg = CMS.cmsLeg(_leg_x1, _LEGEND_Y2 - _LEGEND_ROW * _LEGEND_ROWS, _leg_x2, _LEGEND_Y2,
                 textSize=_LEGEND_TEXT_SIZE)
ref = graphs[0]
leg.AddEntry(graphs_obs[0]["obs"], "Observed", "lp")
leg.AddEntry(ref["exp"], "Expected", "l")
leg.AddEntry(ref["exp1sigma"], "Expected #pm1#sigma", "f")
leg.AddEntry(ref["exp2sigma"], "Expected #pm2#sigma", "f")
leg.AddEntry(g_v3_obs, f"Observed ({V3_LEGEND_TAG})", "p")
leg.AddEntry(g_v3_exp, f"Expected ({V3_LEGEND_TAG})", "p")

# ---------------------------------------------------------------------------
# Numbers and output
# ---------------------------------------------------------------------------
ratio = {key: [v4_lookup[mp][key] / v3_points[mp][key] for mp in v3_points]
         for key in ("exp0", "obs")}
print(f"{args.method} MHc{args.mhc}: {len(v3_points)} V3 points"
      f"{f' (window [{pnet_window[0]:g}, {pnet_window[1]:g}])' if pnet_window else ''}; "
      f"V4/V3 exp0 median {statistics.median(ratio['exp0']):.3f} "
      f"[{min(ratio['exp0']):.3f}, {max(ratio['exp0']):.3f}], "
      f"obs median {statistics.median(ratio['obs']):.3f} "
      f"[{min(ratio['obs']):.3f}, {max(ratio['obs']):.3f}]")

output_base = (f"results/plots/{args.mode}/{ERA}/limit.{ERA}.Asymptotic."
               f"{args.method}.interp-signal.MHc{args.mhc}.compareV3")
os.makedirs(os.path.dirname(output_base), exist_ok=True)
canv.SaveAs(f"{output_base}.png")
canv.SaveAs(f"{output_base}.pdf")
