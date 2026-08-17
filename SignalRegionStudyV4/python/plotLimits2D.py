#!/usr/bin/env python3
"""2D limit map over the (m_H+, m_A) plane: m_H+ on x, m_A on y, colour = the
95% CL upper limit.

The scan is dense in m_A (0.1-1 GeV lattice over 15 GeV .. m_H+ - 5) and has
exactly seven m_H+ values, because the interpolation model interpolates in
m_A ALONE (docs/interpolation/WORKFLOW.md, "No mHc interpolation"). The map
is drawn the same way: one vertical COLUMN per measured m_H+, each filled by
linear interpolation along its own m_A curve. Nothing is ever interpolated
between columns, so the picture cannot claim more than the model delivers.

Cells outside a column's m_A reach stay empty (ROOT leaves zero-content bins
unpainted), which is exactly the kinematic boundary m_A <= m_H+ - 5 and
leaves the upper-left of the frame white for the information text.

  python3 python/plotLimits2D.py --era All --method Baseline \
      --signal-source interp-signal
  python3 python/plotLimits2D.py --era All --method ParticleNet \
      --signal-source interp-signal --quantity obs --mode xsec
"""
import argparse
import json
import os

import numpy as np
import ROOT
import cmsstyle as CMS

import srspaths
from plotter import LumiInfo, LumiInfoExact, EnergyInfo, get_CoM_energy

ROOT.gROOT.SetBatch(ROOT.kTRUE)

_LUMI_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
                               "Common", "Data", "Luminosity.json")
with open(_LUMI_JSON_PATH) as f:
    _LUMI_CONFIG = json.load(f)

# y binning of the map. The scan's own lattice is 0.1-1 GeV, so 0.5 GeV cells
# resolve every feature the limits actually have (the Z peak above all)
# without inventing structure between scan points.
MA_BIN_WIDTH = 0.5
MA_MIN, MA_MAX = 15.0, 155.0

# Fixed colour range, so every map of the campaign is read on one scale.
# The two modes are the same limit in different units (xsec = BR x
# sigma_ttbar(13 TeV) = 833.9 pb, see collectLimits.py), so the ranges are
# each other's image and the two renderings are pixel-for-pixel comparable.
DEFAULT_ZRANGE = {"BR": (5e-7, 1e-5), "xsec": (0.41695, 8.339)}

# Delaunay rendering grid for --interpolate-mhc, ~0.5 GeV in both directions.
SMOOTH_NPX, SMOOTH_NPY = 180, 280

# The ParticleNet arm's m_A window (configs/pnet_grid.json). It is the same
# for every trained m_H+; MHc100 simply has no scan points above its m_A = 95
# MC endpoint, and Baseline covers that corner. The two methods are NEVER
# interpolated into each other: each is rendered from its own points alone
# and the boundary is a hard edge at these two values.
PNET_WINDOW = (82.5, 97.5)

_CHANNEL_LABELS = {
    "Combined": "e#mu#mu + #mu#mu#mu",
    "SR1E2Mu": "e#mu#mu",
    "SR3Mu": "#mu#mu#mu",
}

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--era", type=str, required=True,
                    help="2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, "
                         "2023, 2023BPix, Run2, Run3, All")
parser.add_argument("--channel", type=str, default="Combined",
                    choices=["Combined", "SR1E2Mu", "SR3Mu"],
                    help="Analysis channel (default: Combined)")
parser.add_argument("--method", type=str, default="Baseline",
                    choices=["Baseline", "ParticleNet"],
                    help="Baseline draws one method over the whole plane; "
                         "ParticleNet stitches the ParticleNet scan into its "
                         "m_A window on the columns that have it, and marks "
                         "the window edges (as plotLimits.py does in 1D)")
parser.add_argument("--limit_type", type=str, default="Asymptotic",
                    choices=["Asymptotic"], help="Limit type (Asymptotic only in V4)")
parser.add_argument("--mode", type=str, default="BR", choices=["BR", "xsec"],
                    help="Limit unit: BR (relative branching ratio, default) "
                         "or xsec (sigma(pp->ttbar) x B_sig in fb)")
parser.add_argument("--quantity", type=str, default="exp0",
                    choices=["exp0", "obs"],
                    help="Median expected (default) or observed limit")
parser.add_argument("--signal-source", type=str, default="interp-signal",
                    choices=["mc-signal", "interp-signal"],
                    help="Which collected-limits JSON to read (default: the "
                         "scan grid, the only one dense enough for a map)")
parser.add_argument("--blind", action="store_true",
                    help="Read the {method}_blind collection")
parser.add_argument("--zrange", type=float, nargs=2, default=None,
                    metavar=("ZMIN", "ZMAX"),
                    help="Override the fixed colour range of --mode")
parser.add_argument("--interpolate-mhc", dest="interpolate_mhc",
                    action="store_true",
                    help="Interpolate BETWEEN the measured m_H+ as well: the "
                         "scan points are handed to a TGraph2D and ROOT's "
                         "Delaunay triangulation fills the plane, so the "
                         "kinematic edge comes out as the straight line "
                         "m_A = m_H+ - 5 instead of a staircase. This is a "
                         "rendering choice, not a prediction — the model "
                         "itself has no m_H+ interpolation.")
args = parser.parse_args()

VALID_ERAS = [
    "2016preVFP", "2016postVFP", "2017", "2018",
    "2022", "2022EE", "2023", "2023BPix",
    "Run2", "Run3", "All"
]
if args.era not in VALID_ERAS:
    raise ValueError(f"Invalid era: {args.era}. Must be one of {VALID_ERAS}")
if args.quantity == "obs" and args.blind:
    raise ValueError("--quantity obs is meaningless with --blind")


def load_limits(method):
    path = srspaths.limits_json(args.era, args.channel, method,
                                mode=args.mode, blind=args.blind,
                                source=args.signal_source)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Collected-limits JSON not found: {path}")
    with open(path) as f:
        return json.load(f)


def by_mhc(limits):
    """{mHc: {mA: value}} for the requested quantity."""
    cols = {}
    for mp, v in limits.items():
        mhc, ma = srspaths.masspoint_mhc_ma(mp)
        cols.setdefault(int(mhc), {})[float(ma)] = v[args.quantity]
    return cols


cols = by_mhc(load_limits("Baseline"))
pnet_cols = {}
if args.method == "ParticleNet":
    # Kept SEPARATE from the Baseline points on purpose. The arm covers a
    # window in m_A only, on the m_H+ it was trained for; Baseline holds
    # everywhere else, and the two are rendered from disjoint point sets so
    # that no cell and no triangle ever mixes the methods.
    for mhc, pnet_col in by_mhc(load_limits("ParticleNet")).items():
        if mhc not in cols:
            raise RuntimeError(f"ParticleNet column MHc{mhc} has no Baseline "
                               f"column beside it")
        outside = [ma for ma in pnet_col
                   if not PNET_WINDOW[0] <= ma <= PNET_WINDOW[1]]
        if outside:
            raise RuntimeError(f"ParticleNet MHc{mhc} has points outside "
                               f"{PNET_WINDOW}: {sorted(outside)}")
        pnet_cols[mhc] = pnet_col

if not cols:
    raise RuntimeError("No mass points in the collected-limits JSON")

mhcs = sorted(cols)
if len(mhcs) < 2:
    raise RuntimeError(f"A map needs at least two m_H+ columns, found {mhcs}")

# --- The surface. Column mode bins x at the measured m_H+ with edges at the
# midpoints; smooth mode hands the raw scan points to a TGraph2D, whose
# Delaunay triangulation covers their convex hull — and every column's top
# point sits on m_A = m_H+ - 5, so the hull's upper edge IS that line.
half_lo = (mhcs[1] - mhcs[0]) / 2.0
half_hi = (mhcs[-1] - mhcs[-2]) / 2.0
x_edges = ([mhcs[0] - half_lo]
           + [(a + b) / 2.0 for a, b in zip(mhcs, mhcs[1:])]
           + [mhcs[-1] + half_hi])
n_y = int(round((MA_MAX - MA_MIN) / MA_BIN_WIDTH))
y_edges = [MA_MIN + i * MA_BIN_WIDTH for i in range(n_y + 1)]
y_centers = np.array([(y_edges[i] + y_edges[i + 1]) / 2.0 for i in range(n_y)])

z_min, z_max = args.zrange if args.zrange else DEFAULT_ZRANGE[args.mode]

def make_graph2d(name, columns):
    """Delaunay surface over one method's points alone."""
    pts = [(mhc, ma, val)
           for mhc, col in sorted(columns.items())
           for ma, val in sorted(col.items())]
    g = ROOT.TGraph2D(len(pts))
    g.SetName(name)
    for i, (px, py, pz) in enumerate(pts):
        g.SetPoint(i, px, py, pz)
    g.SetNpx(SMOOTH_NPX)
    g.SetNpy(SMOOTH_NPY)
    g.SetMinimum(z_min)
    g.SetMaximum(z_max)
    return g


def column_values(col):
    """Linear in m_A within one column, np.nan outside its reach — those are
    the cells ROOT must leave unpainted."""
    ma = sorted(col)
    return np.interp(y_centers, ma, [col[m] for m in ma],
                     left=np.nan, right=np.nan)


if args.interpolate_mhc:
    surface = make_graph2d("limit2D", cols)
    # A second, independent triangulation for the arm. Drawn on top, so its
    # window replaces the Baseline surface there without a single triangle
    # spanning the two methods.
    pnet_surface = make_graph2d("limit2D_pnet", pnet_cols) if pnet_cols else None
    x_lo, x_hi = float(mhcs[0]), float(mhcs[-1])
else:
    pnet_surface = None
    surface = ROOT.TH2D("limit2D", "", len(mhcs), np.array(x_edges), n_y,
                        np.array(y_edges))
    surface.SetDirectory(0)
    for ix, mhc in enumerate(mhcs, start=1):
        filled = column_values(cols[mhc])
        if mhc in pnet_cols:
            # Each cell takes ONE method: ParticleNet wherever the arm has
            # points, Baseline elsewhere. Because the two grids share the
            # 0.5 GeV lattice, the switch lands exactly on the window edge.
            pnet_filled = column_values(pnet_cols[mhc])
            filled = np.where(np.isfinite(pnet_filled), pnet_filled, filled)
        for iy, val in enumerate(filled, start=1):
            if np.isfinite(val):
                surface.SetBinContent(ix, iy, val)
    if not any(surface.GetBinContent(ix, iy) > 0
               for ix in range(1, len(mhcs) + 1) for iy in range(1, n_y + 1)):
        raise RuntimeError("Every map cell is empty")
    surface.SetMinimum(z_min)
    surface.SetMaximum(z_max)
    x_lo, x_hi = x_edges[0], x_edges[-1]

# --- CMS style. The luminosity string is drawn by hand (below) so it can be
# right-aligned to the edge of the whole panel, palette and z title included,
# rather than to the frame; cmsstyle would stop it at the frame. Each run
# period is quoted with its own energy, as in plotLimits.py. It is scaled
# slightly down to clear the y-axis title on the widened panel.
CMS.SetExtraText("Preliminary")
CMS.ResetAdditionalInfo()
CMS.SetLumi(None, run="")
CMS.SetEnergy(0, unit="")
if args.era == "All":
    lumi_text = (f"{LumiInfoExact['Run2']:g} fb^{{#minus1}} "
                 f"({EnergyInfo['Run2']:g} TeV) + "
                 f"{LumiInfoExact['Run3']:g} fb^{{#minus1}} "
                 f"({EnergyInfo['Run3']:g} TeV)")
elif args.era in ("Run2", "Run3"):
    lumi_text = (f"{LumiInfoExact[args.era]:g} fb^{{#minus1}} "
                 f"({EnergyInfo[args.era]:g} TeV)")
else:
    lumi_text = (f"{LumiInfo[args.era]:g} fb^{{#minus1}} "
                 f"({get_CoM_energy(args.era):g} TeV)")

_qualifier = "Expected" if args.quantity == "exp0" else "Observed"
if args.mode == "xsec":
    z_title = "95% CL upper limit on #sigma_{sig} [fb]"
else:
    z_title = "95% CL upper limit on #it{B}_{sig}"

ROOT.gStyle.SetNumberContours(99)
canv = CMS.cmsCanvas("limit2D", x_lo, x_hi, MA_MIN, MA_MAX,
                     "m_{H^{+}} [GeV]", "m_{A} [GeV]",
                     square=True, iPos=11, extraSpace=0.02, with_z_axis=True)
# cmsCanvas sizes the right margin for a bare palette; the z title needs more.
canv.SetRightMargin(0.19)
canv.SetLogz(1)
canv.cd()

# 70..160 in ROOT's default steps of 10 gives x labels that touch. The frame
# has to be grabbed here: once TGraph2D has drawn, it is no longer the
# canvas primitive that FindObject returns.
if args.interpolate_mhc:
    CMS.GetCmsCanvasHist(canv).GetXaxis().SetNdivisions(505)

if args.interpolate_mhc:
    # A first Draw is what triggers the triangulation; the histogram it
    # produces is the object that actually gets painted and carries the axes,
    # and it comes back with its own ranges and no titles.
    surface.SetTitle(";m_{H^{+}} [GeV];m_{A} [GeV];")
    surface.Draw("COLZ")
    canv.Update()
    drawn = surface.GetHistogram()
    drawn.GetXaxis().SetTitle("m_{H^{+}} [GeV]")
    drawn.GetYaxis().SetTitle("m_{A} [GeV]")
    drawn.GetXaxis().SetNdivisions(505)
    drawn.GetYaxis().SetNdivisions(510)
    drawn.SetMinimum(z_min)
    drawn.SetMaximum(z_max)
    drawn.GetXaxis().SetRangeUser(x_lo, x_hi)
    drawn.GetYaxis().SetRangeUser(MA_MIN, MA_MAX)
    drawn.Draw("COLZ same")
    if pnet_surface is not None:
        # COL, not COLZ: the palette is already there, and a second one would
        # be drawn on top of the first.
        pnet_surface.Draw("COL same")
        canv.Update()
        pnet_drawn = pnet_surface.GetHistogram()
        pnet_drawn.SetMinimum(z_min)
        pnet_drawn.SetMaximum(z_max)
        pnet_drawn.Draw("COL same")
else:
    drawn = surface
    drawn.Draw("COLZ same")
canv.Update()
CMS.SetCMSPalette()

zax = drawn.GetZaxis()
zax.SetTitle(z_title)
zax.SetTitleOffset(1.30)
zax.SetTitleSize(0.032)
# The topmost palette label sits ON the frame top, half of it above the line,
# right where the luminosity text runs: small labels keep the two apart.
zax.SetLabelSize(0.028)
zax.SetLabelOffset(0.005)

# In column mode the column edges are a statement about the model (no
# interpolation in m_H+), so they are drawn rather than left implicit. In
# smooth mode there are no columns to separate.
if not args.interpolate_mhc:
    sep = ROOT.TLine()
    sep.SetLineColor(ROOT.kGray + 2)
    sep.SetLineStyle(3)
    sep.SetLineWidth(1)
    for edge in x_edges[1:-1]:
        sep.DrawLine(edge, MA_MIN, edge, MA_MAX)

# On-Z / off-Z boundary: the ParticleNet window, at its two fixed m_A values.
# Drawn only over the m_H+ that actually have the arm, so the mark can never
# suggest ParticleNet coverage where there is none.
if pnet_cols:
    trained = sorted(pnet_cols)
    if args.interpolate_mhc:
        wp_x = (float(trained[0]), float(trained[-1]))
    else:
        wp_x = (x_edges[mhcs.index(trained[0])],
                x_edges[mhcs.index(trained[-1]) + 1])
    wp = ROOT.TLine()
    wp.SetLineColor(ROOT.kBlack)
    wp.SetLineStyle(2)
    wp.SetLineWidth(2)
    for edge in PNET_WINDOW:
        wp.DrawLine(wp_x[0], edge, wp_x[1], edge)

# Information text, stacked under the in-frame CMS block. Both sit in the
# kinematically forbidden m_A > m_H+ - 5 corner, which the map leaves white —
# that is what makes plain black text readable on a colour map at all.
label = ROOT.TLatex()
label.SetNDC(True)
label.SetTextFont(42)
label.SetTextSize(0.04)
label.SetTextColor(ROOT.kBlack)
label.DrawLatex(0.21, 0.76, _CHANNEL_LABELS[args.channel])
label.DrawLatex(0.21, 0.71, _qualifier)

t = canv.GetTopMargin()
CMS.drawText(lumi_text, posX=1 - 0.01, posY=1 - t + CMS.lumiTextOffset * t,
             font=42, align=31, size=CMS.lumiTextSize * t * 0.85)
# TGraph2D's COLZ pass paints over the frame decorations.
CMS.CMS_lumi(canv, 11)
canv.RedrawAxis()

_ch_suffix = "" if args.channel == "Combined" else f".{args.channel}"
_source_infix = ("" if args.signal_source == "mc-signal"
                 else f".{args.signal_source}")
_style_infix = ".smooth" if args.interpolate_mhc else ""
output_base = (f"results/plots/{args.mode}/{args.era}/limit2D.{args.era}"
               f"{_ch_suffix}.{args.limit_type}."
               f"{srspaths.method_segment(args.method, args.blind)}"
               f"{_source_infix}{_style_infix}.{args.quantity}")
os.makedirs(os.path.dirname(output_base), exist_ok=True)
canv.SaveAs(f"{output_base}.png")
canv.SaveAs(f"{output_base}.pdf")

print(f"Saved: {output_base}.{{png,pdf}}")
print("  style: " + ("Delaunay surface over all scan points — INTERPOLATED "
                     "in m_H+, which the model does not do"
                     if args.interpolate_mhc else
                     "one column per measured m_H+, no m_H+ interpolation"))
print(f"  m_H+ columns: {mhcs}")
print(f"  Baseline points per column: { {m: len(cols[m]) for m in mhcs} }")
if pnet_cols:
    print(f"  ParticleNet window m_A in {PNET_WINDOW}, rendered from its own "
          f"points alone (no Baseline/ParticleNet interpolation):")
    for mhc, col in sorted(pnet_cols.items()):
        print(f"    MHc{mhc}: m_A {min(col):g}-{max(col):g} GeV "
              f"({len(col)} points)")
    missing = [m for m in mhcs if m not in pnet_cols]
    if missing:
        print(f"  Baseline-only columns (no ParticleNet arm): {missing}")
print(f"  colour range: [{z_min:.3g}, {z_max:.3g}]"
      f"{' (--zrange)' if args.zrange else ''}")
