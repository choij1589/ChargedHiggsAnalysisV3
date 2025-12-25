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
parser.add_argument("--era", type=str, required=True, help="2016preVFP, 2016postVFP, 2017, 2018, FullRun2")
parser.add_argument("--channel", type=str, required=True, help="SR1E2Mu, SR3Mu, Combined")
parser.add_argument("--method", type=str, required=True, help="Baseline, ParticleNet")
parser.add_argument("--limit_type", type=str, required=True, help="Asymptotic / HybridNew")
parser.add_argument("--stack_baseline", action="store_true", help="Show baseline expected limit on top (only for ParticleNet method)")
args = parser.parse_args()

def create_graphs(limits_dict):
    """Create TGraph objects from limits dictionary"""
    mass_points = sorted(limits_dict.keys(), key=lambda mp: int(mp.split("_")[1][2:]))
    x = array('d', [int(mp.split("_")[1][2:]) for mp in mass_points])
    n = len(x)

    # Extract limit values
    limits = {key: array('d', [limits_dict[mp][key]/(2*0.5456) for mp in mass_points])
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

# Load reference limits (HIG-18-020)
with open("results/json/limits.PhysRevLett.123.131801.json") as f:
    limits_ref = json.load(f)
x_ref = array('d', sorted([float(mp) for mp in limits_ref.keys()]))
TTBAR_XEC_13TEV = 832.0e3  # fb
obs_ref = array('d', [limits_ref[str(mp)]["obs"] / TTBAR_XEC_13TEV for mp in x_ref])
g_ref = ROOT.TGraph(len(x_ref), x_ref, obs_ref)
g_ref.SetLineWidth(2)
g_ref.SetLineStyle(2)
g_ref.SetLineColor(ROOT.kGreen+2)

# Setup CMS style
era = "Run2" if args.era == "FullRun2" else args.era
CMS.SetExtraText("Preliminary")
CMS.SetLumi(LumiInfo[era], run=era)
CMS.SetEnergy(get_CoM_energy(era))
CMS.ResetAdditionalInfo()

if args.method == "Baseline":
    with open(f"results/json/limits.{args.era}.{args.channel}.{args.limit_type}.Baseline.json") as f:
        limits = json.load(f)

    graphs = create_graphs(limits)
    y_max = max(graphs['values'] + list(obs_ref)) * 1.5

    canv = CMS.cmsCanvas("limit", 15., 155., 0., y_max,
                         "m_{A} [GeV]", "95% CL limit on B_{sig} [fb]",
                         square=True, iPos=11, extraSpace=0.01)
    canv.cd()

    # Draw
    CMS.cmsObjectDraw(graphs['exp2sigma'], "E3", FillColor=ROOT.TColor.GetColor("#85D1FBff"))
    CMS.cmsObjectDraw(graphs['exp1sigma'], "E3 same", FillColor=ROOT.TColor.GetColor("#FFDF7Fff"))
    CMS.cmsObjectDraw(graphs['exp'], "L same")
    CMS.cmsObjectDraw(graphs['obs'], "LP same")
    CMS.cmsObjectDraw(g_ref, "L same")

    # Legend
    leg = CMS.cmsLeg(0.63, 0.90 - 0.05*5, 0.95, 0.90, textSize=0.04)
    leg.AddEntry(graphs['obs'], "Observed", "lp")
    leg.AddEntry(graphs['exp'], "Expected", "l")
    leg.AddEntry(graphs['exp1sigma'], "Expected #pm1#sigma", "f")
    leg.AddEntry(graphs['exp2sigma'], "Expected #pm2#sigma", "f")
    leg.AddEntry(g_ref, "HIG-18-020", "l")

    print(f"Created Brazilian plot with {len(limits)} mass points (Baseline)")

elif args.method == "ParticleNet":
    # Load limits
    with open(f"results/json/limits.{args.era}.{args.channel}.{args.limit_type}.Baseline.json") as f:
        limits_baseline = json.load(f)
    with open(f"results/json/limits.{args.era}.{args.channel}.{args.limit_type}.ParticleNet.json") as f:
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

    # Calculate y-axis range
    all_values = graphs_pnet['values'] + list(obs_ref)
    if graphs_below: all_values += graphs_below['values']
    if graphs_above: all_values += graphs_above['values']
    y_max = max(all_values) * 1.5

    canv = CMS.cmsCanvas("limit", 15., 155., 0., y_max,
                         "m_{A} [GeV]", "95% CL limit on B_{sig} [fb]",
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

    # Draw reference
    CMS.cmsObjectDraw(g_ref, "L same")
    canv.RedrawAxis()

    # Legend
    n_entries = 6 if args.stack_baseline else 5
    leg = CMS.cmsLeg(0.63, 0.90 - 0.05*n_entries, 0.95, 0.90, textSize=0.04)
    leg.AddEntry(graphs_pnet['obs'], "Observed", "lp")
    leg.AddEntry(graphs_pnet['exp'], "Expected" if args.stack_baseline else "Expected", "l")
    leg.AddEntry(graphs_pnet['exp1sigma'], "Expected #pm1#sigma", "f")
    leg.AddEntry(graphs_pnet['exp2sigma'], "Expected #pm2#sigma", "f")
    if args.stack_baseline:
        leg.AddEntry(graphs_baseline_full['exp'], "w/o ParticleNet", "l")
    leg.AddEntry(g_ref, "HIG-18-020", "l")

    print(f"Created Brazilian plot with ParticleNet ({pnet_min}-{pnet_max} GeV)")
    print(f"  ParticleNet: {len(limits_pnet)} mass points")
    if graphs_below: print(f"  Baseline (below Z): {len(limits_below)} mass points")
    if graphs_above: print(f"  Baseline (above Z): {len(limits_above)} mass points")
    if args.stack_baseline: print(f"  Baseline (full range): {len(limits_baseline)} mass points overlay")

else:
    raise ValueError(f"Method {args.method} is not supported")

output_path = f"results/plots/limit.{args.era}.{args.channel}.{args.limit_type}.{args.method}.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
canv.SaveAs(output_path)
print(f"Saved plot to {output_path}")
