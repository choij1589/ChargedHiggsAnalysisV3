#!/usr/bin/env python3
import os
from array import array
import argparse
import ROOT
import json
import cmsstyle as CMS
from plotter import LumiInfo, get_CoM_energy

ROOT.gROOT.SetBatch(ROOT.kTRUE)

parser = argparse.ArgumentParser()
parser.add_argument("--era", type=str, required=True,
                    help="2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix, Run2, Run3, All")
parser.add_argument("--method", type=str, required=True, help="Baseline, ParticleNet")
parser.add_argument("--limit_type", type=str, required=True, help="Asymptotic / HybridNew")
parser.add_argument("--stack_baseline", action="store_true", help="Show baseline expected limit on top (only for ParticleNet method)")
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
LumiInfo_extended["All"] = 200  # Run2 (138) + Run3 (62)


def get_CoM_energy_extended(era):
    """Get center-of-mass energy string for era."""
    if era in ["2016preVFP", "2016postVFP", "2017", "2018", "Run2"]:
        return "13"
    elif era in ["2022", "2022EE", "2023", "2023BPix", "Run3"]:
        return "13.6"
    elif era == "All":
        return "13+13.6"
    else:
        raise ValueError(f"Unknown era: {era}")


def create_graphs(limits_dict):
    """Create TGraph objects from limits dictionary."""
    mass_points = sorted(limits_dict.keys(), key=lambda mp: int(mp.split("_")[1][2:]))
    x = array('d', [int(mp.split("_")[1][2:]) for mp in mass_points])
    n = len(x)

    # Extract limit values (divide by BR factor to get branching ratio)
    BR_TTBAR_TO_LEPTON = 2 * 0.5456
    limits = {key: array('d', [limits_dict[mp][key] / BR_TTBAR_TO_LEPTON for mp in mass_points])
              for key in ["obs", "exp0", "exp-1", "exp-2", "exp+1", "exp+2"]}

    # Create graphs
    g_obs = ROOT.TGraph(n, x, limits["obs"])
    g_obs.SetLineWidth(2)
    g_obs.SetMarkerStyle(20)
    g_obs.SetMarkerSize(0.8)

    g_exp = ROOT.TGraph(n, x, limits["exp0"])
    g_exp.SetLineWidth(2)
    g_exp.SetLineStyle(2)
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


# Cross section constant used for reference limit conversion
TTBAR_XEC_13TEV = 832.0e3  # fb

# Load CMS reference limits (PhysRevLett.123.131801)
cms_ref_path = "results/json/limits.PhysRevLett.123.131801.json"
if os.path.exists(cms_ref_path):
    with open(cms_ref_path) as f:
        limits_cms_ref = json.load(f)
    x_cms_ref = array('d', sorted([float(mp) for mp in limits_cms_ref.keys()]))
    # Divide by 2 because H+ can appear in either top quark
    # Use expected limit for reference
    exp_cms_ref = array('d', [limits_cms_ref[str(mp)]["exp0"] / TTBAR_XEC_13TEV / 2 for mp in x_cms_ref])
    g_cms_ref = ROOT.TGraph(len(x_cms_ref), x_cms_ref, exp_cms_ref)
    g_cms_ref.SetLineWidth(2)
    g_cms_ref.SetLineStyle(2)
    g_cms_ref.SetLineColor(ROOT.kGreen+2)
    has_cms_ref = True
else:
    has_cms_ref = False
    print(f"Warning: CMS reference limits file not found at {cms_ref_path}")

# Load ATLAS reference limits (PhysRevD.108.092007)
atlas_ref_path = "results/json/limits.PhysRevLett.108.092007.json"
if os.path.exists(atlas_ref_path):
    with open(atlas_ref_path) as f:
        atlas_data = json.load(f)
    # Use hplus_160GeV table (closest to our H+ mass assumption)
    atlas_table = atlas_data["tables"]["hplus_160GeV"]
    x_atlas_ref = array('d', atlas_table["ma_GeV"])
    # Values are already BR(t->H+b) x BR(H+->Wa), use directly
    exp_atlas_ref = array('d', atlas_table["expected_pb"])
    g_atlas_ref = ROOT.TGraph(len(x_atlas_ref), x_atlas_ref, exp_atlas_ref)
    g_atlas_ref.SetLineWidth(2)
    g_atlas_ref.SetLineStyle(2)
    g_atlas_ref.SetLineColor(ROOT.kBlue+1)
    has_atlas_ref = True
else:
    has_atlas_ref = False
    print(f"Warning: ATLAS reference limits file not found at {atlas_ref_path}")

# Setup CMS style
CMS.SetExtraText("Preliminary")
CMS.ResetAdditionalInfo()

if args.era == "All":
    # Combined Run2+Run3: 138+62 fb⁻¹ at 13/13.6 TeV
    CMS.SetLumi(None, run="Run 2+3, 138+62 fb^{#minus1}")
    CMS.SetEnergy(0, unit="13/13.6 TeV")  # energy=0 uses unit string directly
    y_max = 14e-6
elif args.era == "Run2":
    CMS.SetLumi(LumiInfo_extended["Run2"], run="Run2")
    CMS.SetEnergy(13)
    y_max = 14e-6
elif args.era == "Run3":
    CMS.SetLumi(LumiInfo_extended["Run3"], run="Run3")
    CMS.SetEnergy(13.6)
    y_max = 14e-6
else:
    # Individual era
    CMS.SetLumi(LumiInfo.get(args.era, LumiInfo_extended.get(args.era)), run=args.era)
    CMS.SetEnergy(get_CoM_energy(args.era))
    y_max = 45e-6

if args.method == "Baseline":
    with open(f"results/json/limits.{args.era}.{args.limit_type}.Baseline.json") as f:
        limits = json.load(f)

    graphs = create_graphs(limits)
    
    canv = CMS.cmsCanvas("limit", 15., 155., 0., y_max,
                         "m_{A} [GeV]", "95% CL limit on B_{sig}",
                         square=True, iPos=11, extraSpace=0.01)
    canv.cd()

    # Draw
    CMS.cmsObjectDraw(graphs['exp2sigma'], "E3", FillColor=ROOT.TColor.GetColor("#85D1FBff"))
    CMS.cmsObjectDraw(graphs['exp1sigma'], "E3 same", FillColor=ROOT.TColor.GetColor("#FFDF7Fff"))
    CMS.cmsObjectDraw(graphs['exp'], "L same")
    CMS.cmsObjectDraw(graphs['obs'], "LP same")
    if has_cms_ref:
        CMS.cmsObjectDraw(g_cms_ref, "L same")
    if has_atlas_ref:
        CMS.cmsObjectDraw(g_atlas_ref, "L same")
    canv.RedrawAxis()

    # Legend
    n_entries = 4 + (1 if has_cms_ref else 0) + (1 if has_atlas_ref else 0)
    leg = CMS.cmsLeg(0.65, 0.90 - 0.05*n_entries, 0.90, 0.90, textSize=0.035)
    leg.AddEntry(graphs['obs'], "Observed", "lp")
    leg.AddEntry(graphs['exp'], "Expected", "l")
    leg.AddEntry(graphs['exp1sigma'], "Expected #pm1#sigma", "f")
    leg.AddEntry(graphs['exp2sigma'], "Expected #pm2#sigma", "f")
    if has_cms_ref:
        leg.AddEntry(g_cms_ref, "CMS 2016", "l")
    if has_atlas_ref:
        leg.AddEntry(g_atlas_ref, "ATLAS Run 2", "l")

    print(f"Created Brazilian plot with {len(limits)} mass points (Baseline)")

elif args.method == "ParticleNet":
    # Load limits
    with open(f"results/json/limits.{args.era}.{args.limit_type}.Baseline.json") as f:
        limits_baseline = json.load(f)
    with open(f"results/json/limits.{args.era}.{args.limit_type}.ParticleNet.json") as f:
        limits_pnet = json.load(f)

    # Split regions
    pnet_mass = [int(mp.split("_")[1][2:]) for mp in limits_pnet.keys()]
    pnet_min, pnet_max = min(pnet_mass), max(pnet_mass)

    limits_below = {mp: limits_baseline[mp] for mp in limits_baseline if int(mp.split("_")[1][2:]) < pnet_min}
    limits_above = {mp: limits_baseline[mp] for mp in limits_baseline if int(mp.split("_")[1][2:]) > pnet_max}

    # Create graphs
    graphs_pnet = create_graphs(limits_pnet)
    graphs_below = create_graphs(limits_below) if limits_below else None
    graphs_above = create_graphs(limits_above) if limits_above else None

    canv = CMS.cmsCanvas("limit", 15., 155., 0., y_max,
                         "m_{A} [GeV]", "95% CL limit on B_{sig}",
                         square=True, iPos=11, extraSpace=0.01)
    canv.cd()

    # Draw all regions (bands and observed)
    for g in [graphs_pnet, graphs_below, graphs_above]:
        if g:
            CMS.cmsObjectDraw(g['exp2sigma'], "E3 same", FillColor=ROOT.TColor.GetColor("#85D1FBff"))
            CMS.cmsObjectDraw(g['exp1sigma'], "E3 same", FillColor=ROOT.TColor.GetColor("#FFDF7Fff"))

    # Optionally draw baseline for comparison
    if args.stack_baseline:
        graphs_baseline_full = create_graphs(limits_baseline)
        graphs_baseline_full['exp'].SetLineColor(ROOT.kRed+1)
        graphs_baseline_full['exp'].SetLineWidth(2)
        CMS.cmsObjectDraw(graphs_baseline_full['exp'], "L same")

    # Draw expected lines (baseline regions first, then ParticleNet on top)
    if graphs_below:
        CMS.cmsObjectDraw(graphs_below['exp'], "L same")
    if graphs_above:
        CMS.cmsObjectDraw(graphs_above['exp'], "L same")
    CMS.cmsObjectDraw(graphs_pnet['exp'], "L same")

    # Draw observed points
    for g in [graphs_pnet, graphs_below, graphs_above]:
        if g:
            CMS.cmsObjectDraw(g['obs'], "LP same")

    # Draw vertical lines marking ParticleNet region
    line = ROOT.TLine()
    line.SetLineColor(ROOT.kBlack)
    line.SetLineStyle(2)
    line.SetLineWidth(2)
    line.DrawLine(pnet_min, 0, pnet_min, y_max)
    line.DrawLine(pnet_max, 0, pnet_max, y_max)

    # Draw references
    if has_cms_ref:
        CMS.cmsObjectDraw(g_cms_ref, "L same")
    if has_atlas_ref:
        CMS.cmsObjectDraw(g_atlas_ref, "L same")
    canv.RedrawAxis()

    # Legend
    n_entries = 4 + (1 if args.stack_baseline else 0) + (1 if has_cms_ref else 0) + (1 if has_atlas_ref else 0)
    leg = CMS.cmsLeg(0.65, 0.90 - 0.05*n_entries, 0.90, 0.90, textSize=0.035)
    leg.AddEntry(graphs_pnet['obs'], "Observed", "lp")
    leg.AddEntry(graphs_pnet['exp'], "Expected", "l")
    leg.AddEntry(graphs_pnet['exp1sigma'], "Expected #pm1#sigma", "f")
    leg.AddEntry(graphs_pnet['exp2sigma'], "Expected #pm2#sigma", "f")
    if args.stack_baseline:
        leg.AddEntry(graphs_baseline_full['exp'], "w/o ParticleNet", "l")
    if has_cms_ref:
        leg.AddEntry(g_cms_ref, "CMS 2016", "l")
    if has_atlas_ref:
        leg.AddEntry(g_atlas_ref, "ATLAS Run 2", "l")

    print(f"Created Brazilian plot with ParticleNet ({pnet_min}-{pnet_max} GeV)")
    print(f"  ParticleNet: {len(limits_pnet)} mass points")
    if graphs_below:
        print(f"  Baseline (below): {len(limits_below)} mass points")
    if graphs_above:
        print(f"  Baseline (above): {len(limits_above)} mass points")
    if args.stack_baseline:
        print(f"  Baseline (full range): {len(limits_baseline)} mass points overlay")

else:
    raise ValueError(f"Method {args.method} is not supported")

# Save outputs
output_base = f"results/plots/limit.{args.era}.{args.limit_type}.{args.method}"
os.makedirs(os.path.dirname(output_base), exist_ok=True)

canv.SaveAs(f"{output_base}.png")