#!/usr/bin/env python3
import os
from array import array
import argparse
import ROOT
import json
import yaml
import cmsstyle as CMS
from plotter import LumiInfo, LumiInfoExact, EnergyInfo, get_CoM_energy, PALETTE_LONG

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
parser.add_argument("--limit_type", type=str, required=True, help="Asymptotic / HybridNew")
parser.add_argument("--unblind", action="store_true", help="Load limits from unblind JSON")
parser.add_argument("--blind", action="store_true", help="Hide observed limit (for blinded results)")
parser.add_argument("--stack_baseline", action="store_true", help="Show baseline expected limit on top (only for ParticleNet method)")
parser.add_argument("--cnc", action="store_true", help="Load CnC limits (uses .CnC suffix in JSON/plot filenames)")
parser.add_argument("--nsigma", type=float, default=3.0, help="CnC mass window half-width in sigma_voigt (default: 3.0)")
parser.add_argument("--mhc", type=int, default=None,
                    help="Plot only this MHc value. Baseline defaults to 160 when omitted; ParticleNet uses all trained points when omitted.")
parser.add_argument("--compare-mhc", dest="compare_mhc", action="store_true",
                    help="Overlay median expected limits for several MHc values (Baseline only)")
parser.add_argument("--mhc-list", dest="mhc_list", type=str, default="70,85,100,115,130,145,160",
                    help="Comma-separated MHc values for --compare-mhc (default: 70,85,100,115,130,145,160)")
parser.add_argument("--mode", type=str, default="BR", choices=["BR", "xsec"],
                    help="Limit unit: BR (relative branching ratio, default) or xsec (sigma(pp->ttbar) x B_sig in fb)")
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
    mass_points = sorted(limits_dict.keys(), key=lambda mp: int(mp.split("_")[1][2:]))
    x = array('d', [int(mp.split("_")[1][2:]) for mp in mass_points])
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


# Load CMS reference limits (HEPData ins1735729, Figure 2b).
# This curve is for mH+ = 160 GeV only; xsec mode skips the overlay.
cms_ref_path = "results/yaml/HEPData-ins1735729-v2-Figure_2b.yaml"
if args.mode == "BR" and os.path.exists(cms_ref_path):
    with open(cms_ref_path) as f:
        cms_yaml = yaml.safe_load(f)
    x_cms_ref = array('d', [v["value"] for v in cms_yaml["independent_variables"][0]["values"]])
    # dependent_variables[1] is the expected upper limit in units of ×10⁻⁶
    exp_cms_ref = array('d', [v["value"] * 1e-6 for v in cms_yaml["dependent_variables"][1]["values"]])
    g_cms_ref = ROOT.TGraph(len(x_cms_ref), x_cms_ref, exp_cms_ref)
    g_cms_ref.SetLineWidth(2)
    g_cms_ref.SetLineStyle(2)
    g_cms_ref.SetLineColor(ROOT.kGreen+2)
    has_cms_ref = True
else:
    has_cms_ref = False
    if args.mode == "BR":
        print(f"Warning: CMS reference limits file not found at {cms_ref_path}")

# Load ATLAS reference limits (HEPData ins2654723, Table 9 / Figure 5d).
# This curve is for mH+ = 160 GeV only; xsec mode skips the overlay.
atlas_ref_path = "results/yaml/HEPData-ins2654723-v1-Table_9.yaml"
if args.mode == "BR" and os.path.exists(atlas_ref_path):
    with open(atlas_ref_path) as f:
        atlas_yaml = yaml.safe_load(f)
    x_atlas_ref = array('d', [v["value"] for v in atlas_yaml["independent_variables"][0]["values"]])
    # dependent_variables[1] is the expected limit, already in absolute B_sig units
    exp_atlas_ref = array('d', [v["value"] for v in atlas_yaml["dependent_variables"][1]["values"]])
    g_atlas_ref = ROOT.TGraph(len(x_atlas_ref), x_atlas_ref, exp_atlas_ref)
    g_atlas_ref.SetLineWidth(2)
    g_atlas_ref.SetLineStyle(2)
    g_atlas_ref.SetLineColor(ROOT.kBlue+1)
    has_atlas_ref = True
else:
    has_atlas_ref = False
    if args.mode == "BR":
        print(f"Warning: ATLAS reference limits file not found at {atlas_ref_path}")

# Setup CMS style
CMS.SetExtraText("Preliminary")
CMS.ResetAdditionalInfo()

# Luminosity header follows the paper convention shared with plotPaperPostfitSummary.py,
# plotPaperLRModified.py and plotPaperTemplates.py: un-rounded per-period luminosities from
# LumiInfoExact, each run period quoted with its own energy, and no "Run2,"-style prefix.
if args.era == "All":
    # cmsstyle renders "<cms_lumi> (<cms_energy>)", so only the Run3 energy can live in
    # SetEnergy; the whole Run2 term is baked into the run label.
    CMS.SetLumi(None, run=(f"{LumiInfoExact['Run2']:g} fb^{{#minus1}} ({EnergyInfo['Run2']:g} TeV) + "
                           f"{LumiInfoExact['Run3']:g} fb^{{#minus1}}"))
    CMS.SetEnergy(0, unit=f"{EnergyInfo['Run3']:g} TeV")
elif args.era in ("Run2", "Run3"):
    CMS.SetLumi(None, run=f"{LumiInfoExact[args.era]:g} fb^{{#minus1}}")
    CMS.SetEnergy(EnergyInfo[args.era])
else:
    # Individual era
    CMS.SetLumi(LumiInfo.get(args.era, LumiInfo_extended.get(args.era)), run=args.era)
    CMS.SetEnergy(get_CoM_energy(args.era))

if args.mode == "xsec":
    y_label_full = "95% CL limit on #sigma_{sig} [fb]"
    y_label_median = "95% CL median expected #sigma_{sig} [fb]"
else:
    y_label_full = "95% CL limit on #it{B}_{sig}"
    y_label_median = "95% CL median expected #it{B}_{sig}"

# Channel label drawn near MHc text on every plot.
_CHANNEL_LABELS = {
    "Combined": "e#mu#mu + #mu#mu#mu",
    "SR1E2Mu":  "e#mu#mu",
    "SR3Mu":    "#mu#mu#mu",
}
_channel_label_txt = _CHANNEL_LABELS[args.channel]

# CMS label position: iPos=11 puts "CMS"/"Preliminary" inside the frame at top-left, matching
# the paper figures. Kept for both modes -- with iPos=0 the out-of-frame "Preliminary" runs
# into the luminosity string on the xsec plots, whose y-axis title is wider.
_iPos = 11


def _ymax_from(limits_dict):
    """Dynamic y-max: 2x the maximum exp+2sigma across all mass points."""
    return 2.0 * max(v["exp+2"] for v in limits_dict.values())

_nsigma_tag = f"{args.nsigma:g}sigma"
_cnc_suffix = f".CnC_{_nsigma_tag}" if args.cnc else ""
_unblind_suffix = ".unblind" if args.unblind else ""
_ch_suffix = "" if args.channel == "Combined" else f".{args.channel}"

def _filter_by_mhc(limits_dict, mhc_value):
    """Return only mass points whose MHc matches mhc_value."""
    prefix = f"MHc{mhc_value}_"
    return {mp: v for mp, v in limits_dict.items() if mp.startswith(prefix)}


_json_dir = f"results/json/{args.mode}/{args.era}"

if args.method == "Baseline":
    with open(f"{_json_dir}/limits.{args.era}{_ch_suffix}.{args.limit_type}.Baseline{_cnc_suffix}{_unblind_suffix}.json") as f:
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

        y_max = max(_ymax_from(d) for d in per_mhc_limits.values())
        canv = CMS.cmsCanvas("limit", 15., 155., 0., y_max,
                             "m_{A} [GeV]", y_label_median,
                             square=True, iPos=_iPos, extraSpace=0.01)
        canv.cd()

        graphs_compare = []
        for idx, mhc in enumerate(sorted(per_mhc_limits.keys())):
            g = create_graphs(per_mhc_limits[mhc])
            color = PALETTE_LONG[idx % len(PALETTE_LONG)]
            g['exp'].SetLineColor(color)
            g['exp'].SetLineStyle(ROOT.kSolid)
            g['exp'].SetLineWidth(2)
            g['exp'].SetMarkerStyle(20)
            g['exp'].SetMarkerSize(0.8)
            g['exp'].SetMarkerColor(color)
            graphs_compare.append((mhc, g))

        for _, g in graphs_compare:
            CMS.cmsObjectDraw(g['exp'], "LP same")
        canv.RedrawAxis()

        ch_label = ROOT.TLatex()
        ch_label.SetNDC(True)
        ch_label.SetTextFont(42)
        ch_label.SetTextSize(0.04)
        ch_label.DrawLatex(0.20, 0.76, _channel_label_txt)

        n_entries = len(graphs_compare)
        leg = CMS.cmsLeg(0.65, 0.90 - 0.05*n_entries, 0.90, 0.90, textSize=0.035)
        for mhc, g in graphs_compare:
            leg.AddEntry(g['exp'], f"m_{{H^{{+}}}} = {mhc} GeV", "lp")

        print(f"Created MHc-comparison plot ({len(graphs_compare)} MHc values, Baseline)")
    else:
        mhc_value = args.mhc if args.mhc is not None else 160
        limits = _filter_by_mhc(limits, mhc_value)
        if not limits:
            raise RuntimeError(f"No mass points found for MHc{mhc_value} in JSON")
        graphs = create_graphs(limits)
        draw_cms_ref = has_cms_ref and mhc_value == 160
        draw_atlas_ref = has_atlas_ref and mhc_value == 160

        y_max = _ymax_from(limits)
        x_min = 15.0
        x_max = float(mhc_value - 5)
        canv = CMS.cmsCanvas("limit", x_min, x_max, 0., y_max,
                             "m_{A} [GeV]", y_label_full,
                             square=True, iPos=_iPos, extraSpace=0.01)
        canv.cd()

        CMS.cmsObjectDraw(graphs['exp2sigma'], "E3", FillColor=ROOT.TColor.GetColor("#85D1FBff"))
        CMS.cmsObjectDraw(graphs['exp1sigma'], "E3 same", FillColor=ROOT.TColor.GetColor("#FFDF7Fff"))
        CMS.cmsObjectDraw(graphs['exp'], "L same")
        if not args.blind:
            CMS.cmsObjectDraw(graphs['obs'], "LP same")
        if draw_cms_ref:
            CMS.cmsObjectDraw(g_cms_ref, "L same")
        if draw_atlas_ref:
            CMS.cmsObjectDraw(g_atlas_ref, "L same")
        if not args.blind:
            CMS.cmsObjectDraw(graphs['obs'], "LP same")
        canv.RedrawAxis()

        mhc_label_txt = ROOT.TLatex()
        mhc_label_txt.SetNDC(True)
        mhc_label_txt.SetTextFont(42)
        mhc_label_txt.SetTextSize(0.04)
        mhc_label_txt.DrawLatex(0.20, 0.71, f"m_{{H^{{+}}}} = {mhc_value} GeV")
        mhc_label_txt.DrawLatex(0.20, 0.76, _channel_label_txt)

        n_entries = (4 if not args.blind else 3) + (1 if draw_cms_ref else 0) + (1 if draw_atlas_ref else 0)
        leg = CMS.cmsLeg(0.65, 0.90 - 0.05*n_entries, 0.90, 0.90, textSize=0.035)
        if not args.blind:
            leg.AddEntry(graphs['obs'], "Observed", "lp")
        leg.AddEntry(graphs['exp'], "Expected", "l")
        leg.AddEntry(graphs['exp1sigma'], "Expected #pm1#sigma", "f")
        leg.AddEntry(graphs['exp2sigma'], "Expected #pm2#sigma", "f")
        if draw_cms_ref:
            leg.AddEntry(g_cms_ref, "CMS 2016", "l")
        if draw_atlas_ref:
            leg.AddEntry(g_atlas_ref, "ATLAS Run 2", "l")

        print(f"Created Brazilian plot with {len(limits)} mass points (Baseline, MHc{mhc_value})")

elif args.method == "ParticleNet":
    # Load limits
    with open(f"{_json_dir}/limits.{args.era}{_ch_suffix}.{args.limit_type}.Baseline{_cnc_suffix}{_unblind_suffix}.json") as f:
        limits_baseline = json.load(f)
    with open(f"{_json_dir}/limits.{args.era}{_ch_suffix}.{args.limit_type}.ParticleNet{_cnc_suffix}{_unblind_suffix}.json") as f:
        limits_pnet = json.load(f)
    draw_cms_ref = has_cms_ref and args.mhc == 160
    draw_atlas_ref = has_atlas_ref and args.mhc == 160

    if args.mhc is not None:
        limits_pnet = _filter_by_mhc(limits_pnet, args.mhc)
        if not limits_pnet:
            raise RuntimeError(f"No ParticleNet mass points found for MHc{args.mhc} in JSON")
        limits_baseline = _filter_by_mhc(limits_baseline, args.mhc)
    else:
        limits_baseline = {mp: v for mp, v in limits_baseline.items() if mp in _BASELINE_CURATED}

    # Split regions
    pnet_mass = [int(mp.split("_")[1][2:]) for mp in limits_pnet.keys()]
    pnet_min, pnet_max = min(pnet_mass), max(pnet_mass)

    limits_below = {mp: limits_baseline[mp] for mp in limits_baseline if int(mp.split("_")[1][2:]) < pnet_min}
    limits_above = {mp: limits_baseline[mp] for mp in limits_baseline if int(mp.split("_")[1][2:]) > pnet_max}

    # Expected line and bands are continued onto the ParticleNet window edge using the
    # Baseline point sitting exactly at m_A = pnet_min / pnet_max, so the two regions meet
    # instead of leaving a hole. Observed keeps the strict split: only the ParticleNet
    # observed is drawn at the boundary mass. Skipped when --mhc is omitted, where the
    # curated list can hold several entries at one m_A (e.g. two at m_A = 95).
    def _boundary_anchor(target_ma):
        if args.mhc is None:
            return {}
        return {mp: v for mp, v in limits_baseline.items()
                if int(mp.split("_")[1][2:]) == target_ma}

    limits_below_exp = {**limits_below, **_boundary_anchor(pnet_min)}
    limits_above_exp = {**limits_above, **_boundary_anchor(pnet_max)}
    # A single point draws neither a line nor a band; treat it as absent so it also
    # stays out of the y_max scan (matters for MHc100, which has no "above" region).
    if len(limits_below_exp) < 2:
        limits_below_exp = {}
    if len(limits_above_exp) < 2:
        limits_above_exp = {}

    # Create graphs. graphs_below/graphs_above feed the observed markers only; the
    # *_exp graphs carry the boundary-anchored expected line and bands.
    graphs_pnet = create_graphs(limits_pnet)
    graphs_below = create_graphs(limits_below) if limits_below else None
    graphs_above = create_graphs(limits_above) if limits_above else None
    graphs_below_exp = create_graphs(limits_below_exp) if limits_below_exp else None
    graphs_above_exp = create_graphs(limits_above_exp) if limits_above_exp else None

    y_max = max(_ymax_from(d) for d in (limits_pnet, limits_below_exp, limits_above_exp) if d)
    x_max = float(args.mhc - 5) if args.mhc is not None else 155.0
    canv = CMS.cmsCanvas("limit", 15., x_max, 0., y_max,
                         "m_{A} [GeV]", y_label_full,
                         square=True, iPos=_iPos, extraSpace=0.01)
    canv.cd()

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
    if draw_cms_ref:
        CMS.cmsObjectDraw(g_cms_ref, "L same")
    if draw_atlas_ref:
        CMS.cmsObjectDraw(g_atlas_ref, "L same")

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
    ch_label_pn.SetTextSize(0.04)
    if args.mhc is not None:
        ch_label_pn.DrawLatex(0.20, 0.71, f"m_{{H^{{+}}}} = {args.mhc} GeV")
    ch_label_pn.DrawLatex(0.20, 0.76, _channel_label_txt)

    # Legend
    n_entries = (
        (4 if not args.blind else 3)
        + (1 if args.stack_baseline else 0)
        + (1 if draw_cms_ref else 0)
        + (1 if draw_atlas_ref else 0)
    )
    leg = CMS.cmsLeg(0.65, 0.90 - 0.05*n_entries, 0.90, 0.90, textSize=0.035)
    if not args.blind:
        leg.AddEntry(graphs_pnet['obs'], "Observed", "lp")
    leg.AddEntry(graphs_pnet['exp'], "Expected", "l")
    leg.AddEntry(graphs_pnet['exp1sigma'], "Expected #pm1#sigma", "f")
    leg.AddEntry(graphs_pnet['exp2sigma'], "Expected #pm2#sigma", "f")
    if args.stack_baseline:
        leg.AddEntry(graphs_baseline_at_pnet['exp'], "w/o ParticleNet", "l")
    if draw_cms_ref:
        leg.AddEntry(g_cms_ref, "CMS 2016", "l")
    if draw_atlas_ref:
        leg.AddEntry(g_atlas_ref, "ATLAS Run 2", "l")

    mhc_msg = f", MHc{args.mhc}" if args.mhc is not None else ""
    print(f"Created Brazilian plot with ParticleNet ({pnet_min}-{pnet_max} GeV{mhc_msg})")
    print(f"  ParticleNet: {len(limits_pnet)} mass points")
    if graphs_below:
        n_anchor = len(limits_below_exp) - len(limits_below)
        print(f"  Baseline (below): {len(limits_below)} mass points"
              f"{f' (+{n_anchor} boundary anchor at MA{pnet_min})' if n_anchor > 0 else ''}")
    if graphs_above:
        n_anchor = len(limits_above_exp) - len(limits_above)
        print(f"  Baseline (above): {len(limits_above)} mass points"
              f"{f' (+{n_anchor} boundary anchor at MA{pnet_max})' if n_anchor > 0 else ''}")
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

output_base = f"results/plots/{args.mode}/{args.era}/limit.{args.era}{_ch_suffix}.{args.limit_type}.{args.method}{_mode_suffix}{_cnc_suffix}{_unblind_suffix}"
os.makedirs(os.path.dirname(output_base), exist_ok=True)

canv.SaveAs(f"{output_base}.png")
canv.SaveAs(f"{output_base}.pdf")
