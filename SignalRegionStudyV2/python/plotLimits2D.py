#!/usr/bin/env python3
"""Plot limits as multiple TGraphs (one per MHc) vs mA on the same canvas.

Uses Baseline limits as the base, replacing the near-mZ mass points with
ParticleNet limits (MHc100_MA95, MHc130_MA90, MHc160_MA85), mirroring the
overlay logic in plotLimits.py.
"""
import os
from array import array
import argparse
import ROOT
import json
import cmsstyle as CMS
from plotter import LumiInfo, get_CoM_energy, PALETTE_LONG

ROOT.gROOT.SetBatch(ROOT.kTRUE)

# Load luminosity configuration from JSON
_LUMI_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Common", "Data", "Luminosity.json")
with open(_LUMI_JSON_PATH, "r") as f:
    _LUMI_CONFIG = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--era", type=str, required=True,
                    help="2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix, Run2, Run3, All")
parser.add_argument("--limit_type", type=str, required=True, help="Asymptotic / HybridNew")
parser.add_argument("--unblind", action="store_true", help="Load limits from unblind JSON")
parser.add_argument("--blind", action="store_true", help="Hide observed limit (for blinded results)")
parser.add_argument("--cnc", action="store_true", help="Load CnC limits (uses .CnC suffix in JSON/plot filenames)")
parser.add_argument("--nsigma", type=float, default=3.0, help="CnC mass window half-width in sigma_voigt (default: 3.0)")
parser.add_argument("--style", type=str, default="lines", choices=["lines", "colz"],
                    help="Plot style: 'lines' (one TGraph per MHc, default) or 'colz' (2D color map via TGraph2D)")
args = parser.parse_args()

# Validate
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


def get_CoM_energy_extended(era):
    if era in ["2016preVFP", "2016postVFP", "2017", "2018", "Run2"]:
        return "13"
    elif era in ["2022", "2022EE", "2023", "2023BPix", "Run3"]:
        return "13.6"
    elif era == "All":
        return "13+13.6"
    else:
        raise ValueError(f"Unknown era: {era}")


# On-Z mass points not covered by ParticleNet — excluded from all plots
_ON_Z_EXCL = {"MHc115_MA87", "MHc145_MA92", "MHc160_MA98"}

_nsigma_tag = f"{args.nsigma:g}sigma"
_cnc_suffix = f".CnC_{_nsigma_tag}" if args.cnc else ""
_unblind_suffix = ".unblind" if args.unblind else ""

# Load Baseline limits (35 mass points, multiple mA per MHc)
with open(f"results/json/limits.{args.era}.{args.limit_type}.Baseline{_cnc_suffix}{_unblind_suffix}.json") as f:
    limits_baseline = json.load(f)

# Load ParticleNet limits (3 near-mZ mass points)
with open(f"results/json/limits.{args.era}.{args.limit_type}.ParticleNet{_cnc_suffix}{_unblind_suffix}.json") as f:
    limits_pnet = json.load(f)

# Merge: start from Baseline, replace near-mZ entries with ParticleNet values.
# Also remove on-Z exclusions (mass points with no ParticleNet coverage).
limits_merged = {mp: v for mp, v in limits_baseline.items() if mp not in _ON_Z_EXCL}
limits_merged.update(limits_pnet)

# Track which mass points use ParticleNet (for marker distinction)
pnet_mps = set(limits_pnet.keys())

# Group by MHc: {mhc_val: [(mA_val, limits_dict, is_pnet), ...]}
by_mhc = {}
for mp, v in limits_merged.items():
    parts = mp.split("_")
    mhc = int(parts[0][3:])   # "MHc130" → 130
    ma = int(parts[1][2:])    # "MA90"  → 90
    by_mhc.setdefault(mhc, []).append((ma, v, mp in pnet_mps))

for mhc in by_mhc:
    by_mhc[mhc].sort(key=lambda t: t[0])

mhc_values = sorted(by_mhc.keys())
if len(mhc_values) > len(PALETTE_LONG):
    raise RuntimeError(f"Too many MHc values ({len(mhc_values)}) for PALETTE_LONG ({len(PALETTE_LONG)} colors)")

# Setup CMS style
CMS.SetExtraText("Preliminary")
CMS.ResetAdditionalInfo()

if args.era == "All":
    run2_lumi = _LUMI_CONFIG["Run2"]["combined"]
    run3_lumi = _LUMI_CONFIG["Run3"]["combined"]
    CMS.SetLumi(None, run=f"Run 2+3, {run2_lumi}+{run3_lumi} fb^{{#minus1}}")
    CMS.SetEnergy(0, unit="13/13.6 TeV")
    y_max = 20e-6
elif args.era == "Run2":
    CMS.SetLumi(LumiInfo_extended["Run2"], run="Run2")
    CMS.SetEnergy(13)
    y_max = 20e-6
elif args.era == "Run3":
    CMS.SetLumi(LumiInfo_extended["Run3"], run="Run3")
    CMS.SetEnergy(13.6)
    y_max = 20e-6
else:
    CMS.SetLumi(LumiInfo.get(args.era, LumiInfo_extended.get(args.era)), run=args.era)
    CMS.SetEnergy(get_CoM_energy(args.era))
    y_max = 45e-6

output_base = f"results/plots/limit2D.{args.era}.{args.limit_type}.ParticleNet{_cnc_suffix}{_unblind_suffix}"
os.makedirs(os.path.dirname(output_base), exist_ok=True)

if args.style == "lines":
    # Drop sparse MHc rows with too few mA points for a meaningful line
    _MHC_SKIP = {85, 115, 145}
    mhc_values = [m for m in mhc_values if m not in _MHC_SKIP]

    # Scale each MHc curve by ×10^i to separate overlapping lines on log scale
    # Largest MHc at bottom (×1), smallest MHc at top (×10^(n-1))
    scale_factors = [10**i for i in range(len(mhc_values) - 1, -1, -1)]

    y_min = 1e-7
    y_max = 1e2

    canv = CMS.cmsCanvas("limit2D", 10., 160., y_min, y_max,
                         "m_{A} [GeV]", "95% CL limit on B_{sig}",
                         square=True, iPos=11, extraSpace=0.01)
    canv.SetLogy(1)
    canv.cd()

    # CMS color palette for lines
    line_colors = CMS.getPettroffColorSet(len(mhc_values))
    _color_2s = ROOT.TColor.GetColor("#85D1FBff")  # same as plotLimits.py ±2σ
    _color_1s = ROOT.TColor.GetColor("#FFDF7Fff")  # same as plotLimits.py ±1σ

    # Build one TGraph per MHc (all points merged)
    graphs_exp = []
    graphs_obs = []
    bands_1s = []
    bands_2s = []

    for i, mhc in enumerate(mhc_values):
        pts = by_mhc[mhc]
        n = len(pts)
        sf = scale_factors[i]
        color = line_colors[i]

        x_arr = array('d', [p[0] for p in pts])
        g_exp = ROOT.TGraph(n, x_arr, array('d', [p[1]["exp0"] * sf for p in pts]))
        g_exp.SetLineWidth(2); g_exp.SetLineStyle(1); g_exp.SetLineColor(color)
        g_exp.SetMarkerStyle(20); g_exp.SetMarkerSize(0.8); g_exp.SetMarkerColor(color)
        graphs_exp.append(g_exp)

        if not args.blind:
            g_obs = ROOT.TGraph(n, x_arr, array('d', [p[1]["obs"] * sf for p in pts]))
            g_obs.SetLineWidth(2); g_obs.SetLineStyle(2); g_obs.SetLineColor(color)
            g_obs.SetMarkerStyle(24); g_obs.SetMarkerSize(0.8); g_obs.SetMarkerColor(color)
            graphs_obs.append(g_obs)

        b2 = ROOT.TGraphAsymmErrors(n)
        b1 = ROOT.TGraphAsymmErrors(n)
        for j, p in enumerate(pts):
            v = p[1]
            b2.SetPoint(j, p[0], v["exp0"] * sf)
            b2.SetPointError(j, 0, 0, (v["exp0"] - v["exp-2"]) * sf, (v["exp+2"] - v["exp0"]) * sf)
            b1.SetPoint(j, p[0], v["exp0"] * sf)
            b1.SetPointError(j, 0, 0, (v["exp0"] - v["exp-1"]) * sf, (v["exp+1"] - v["exp0"]) * sf)
        bands_2s.append(b2)
        bands_1s.append(b1)

    # Draw order: bands → lines → observed → markers
    for g in bands_2s:
        CMS.cmsObjectDraw(g, "E3 same", FillColor=_color_2s)
    for g in bands_1s:
        CMS.cmsObjectDraw(g, "E3 same", FillColor=_color_1s)
    for g in graphs_exp:
        CMS.cmsObjectDraw(g, "LP same")
    if not args.blind:
        for g in graphs_obs:
            CMS.cmsObjectDraw(g, "LP same")

    canv.RedrawAxis()

    # Legend: one entry per MHc + band guide entries
    n_entries = len(mhc_values) + 2 + (2 if not args.blind else 0)
    leg = CMS.cmsLeg(0.50, 0.87 - 0.03 * n_entries, 0.90, 0.87, textSize=0.030)
    n = len(mhc_values)
    for i, mhc in enumerate(mhc_values):
        exp = n - 1 - i
        sf_str = f" (#times10^{{{exp}}})" if exp > 0 else ""
        leg.AddEntry(graphs_exp[i], f"m_{{H^{{+}}}}={mhc} GeV{sf_str}", "lp")

    leg.AddEntry(bands_1s[0], "Expected #pm1#sigma", "f")
    leg.AddEntry(bands_2s[0], "Expected #pm2#sigma", "f")

    # Style guide for line types when unblinded
    if not args.blind:
        dummy_exp = ROOT.TGraph(1, array('d', [0.]), array('d', [0.]))
        dummy_exp.SetLineStyle(1); dummy_exp.SetLineColor(ROOT.kBlack); dummy_exp.SetLineWidth(2)
        dummy_obs = ROOT.TGraph(1, array('d', [0.]), array('d', [0.]))
        dummy_obs.SetLineStyle(2); dummy_obs.SetLineColor(ROOT.kBlack); dummy_obs.SetLineWidth(2)
        leg.AddEntry(dummy_exp, "Expected (solid)", "l")
        leg.AddEntry(dummy_obs, "Observed (dashed)", "l")

    output_path = f"{output_base}.png"
    canv.SaveAs(output_path)
    print(f"Saved: {output_path}")

else:  # colz
    # True 2D color map via TGraph2D + Delaunay interpolation
    ROOT.gStyle.SetNumberContours(99)

    # Collect all points into a single TGraph2D
    all_pts = []
    for mhc, pts in by_mhc.items():
        for ma, v, _ in pts:
            all_pts.append((ma, mhc, v["exp0"]))

    exp0_vals = [p[2] for p in all_pts]
    z_min = min(exp0_vals) * 0.7
    z_max = max(exp0_vals) * 1.3

    g2d_exp = ROOT.TGraph2D(len(all_pts))
    g2d_exp.SetName("g2d_exp")
    g2d_exp.SetTitle(";m_{H^{+}} [GeV];m_{A} [GeV];")
    for idx, (ma, mhc, val) in enumerate(all_pts):
        g2d_exp.SetPoint(idx, mhc, ma, val)
    g2d_exp.SetNpx(200)
    g2d_exp.SetNpy(200)
    g2d_exp.SetMinimum(z_min)
    g2d_exp.SetMaximum(z_max)

    canv_colz = CMS.cmsCanvas("limit2D_colz", 65., 165., 15., 160.,
                               "m_{H^{+}} [GeV]", "m_{A} [GeV]",
                               square=True, iPos=11, extraSpace=0.01, with_z_axis=True)
    canv_colz.SetLogz(1)
    canv_colz.cd()

    # Draw once to trigger Delaunay computation, then smooth the internal TH2
    g2d_exp.Draw("COLZ")
    canv_colz.Update()
    h2d_smooth = g2d_exp.GetHistogram().Clone("h2d_smooth")
    h2d_smooth.Smooth(2)  # 2 passes of the default 5-neighbour kernel
    h2d_smooth.SetMinimum(z_min)
    h2d_smooth.SetMaximum(z_max)
    h2d_smooth.Draw("COLZ same")  # overwrites unsmoothed rendering

    # Z-axis title and style
    h2d_smooth.GetZaxis().SetTitle("Expected 95% CL limit on B_{sig}")
    h2d_smooth.GetZaxis().SetTitleOffset(1.35)
    h2d_smooth.GetZaxis().SetTitleSize(0.035)
    h2d_smooth.GetZaxis().SetLabelSize(0.035)
    h2d_smooth.GetZaxis().SetLabelOffset(0.005)

    # Fix axis ranges (COLZ resets them)
    h2d_smooth.GetXaxis().SetRangeUser(65., 165.)
    h2d_smooth.GetYaxis().SetRangeUser(15., 160.)

    # Apply official CMS palette after drawing (matches example pattern)
    CMS.SetCMSPalette()

    if not args.blind:
        obs_pts = [(ma, mhc, v["obs"]) for mhc, pts in by_mhc.items() for ma, v, _ in pts]
        g2d_obs = ROOT.TGraph2D(len(obs_pts))
        g2d_obs.SetName("g2d_obs")
        for idx, (ma, mhc, val) in enumerate(obs_pts):
            g2d_obs.SetPoint(idx, mhc, ma, val)
        g2d_obs.SetNpx(200)
        g2d_obs.SetNpy(200)
        g2d_obs.SetLineColor(ROOT.kBlack)
        g2d_obs.SetLineWidth(2)
        g2d_obs.Draw("CONT3 same")

    # Overlay markers at the actual interpolation grid points
    pts_x = array('d', [mhc for _, mhc, _ in all_pts])
    pts_y = array('d', [ma  for ma, _, _ in all_pts])
    g_pts = ROOT.TGraph(len(all_pts), pts_x, pts_y)
    g_pts.SetMarkerStyle(20)
    g_pts.SetMarkerSize(0.8)
    g_pts.SetMarkerColor(ROOT.kBlack)
    g_pts.Draw("P same")

    # Re-apply CMS label after COLZ draw (which overrides the cmsCanvas frame)
    CMS.CMS_lumi(canv_colz, 11)
    canv_colz.RedrawAxis()

    output_path = f"{output_base}.colz.png"
    canv_colz.SaveAs(output_path)
    print(f"Saved: {output_path}")

print(f"MHc values: {mhc_values}")
print(f"ParticleNet mass points: {sorted(pnet_mps)}")
print(f"Total mass points per MHc: { {mhc: len(by_mhc[mhc]) for mhc in mhc_values} }")
