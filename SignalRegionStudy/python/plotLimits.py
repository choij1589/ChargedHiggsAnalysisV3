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
args = parser.parse_args()

# load json
with open(f"results/json/limits.{args.era}.{args.channel}.{args.limit_type}.{args.method}.json") as f:
    limits = json.load(f)

# Sort by mass point for proper ordering
mass_points = sorted(limits.keys(), key=lambda mp: int(mp.split("_")[1][2:]))
x = [int(mp.split("_")[1][2:]) for mp in mass_points]
n_points = len(x)

# Load reference limits from HIG-18-020
with open("results/json/limits.PhysRevLett.123.131801.json") as f:
    limits_ref = json.load(f)

# Convert reference limits to arrays
x_ref = [float(mp) for mp in sorted(limits_ref.keys(), key=float)]
obs_ref = [limits_ref[str(mp)]["obs"] * 832. * 10**-3 for mp in x_ref]
n_points_ref = len(x_ref)

# Extract all limit values
obs = [limits[mp]["obs"] for mp in mass_points]
exp0 = [limits[mp]["exp0"] for mp in mass_points]
exp_m1 = [limits[mp]["exp-1"] for mp in mass_points]
exp_m2 = [limits[mp]["exp-2"] for mp in mass_points]
exp_p1 = [limits[mp]["exp+1"] for mp in mass_points]
exp_p2 = [limits[mp]["exp+2"] for mp in mass_points]

# Create TGraphs for Brazilian plot
# Observed limit
g_obs = ROOT.TGraph(n_points, array('d', x), array('d', obs))
g_obs.SetLineWidth(2)
g_obs.SetLineStyle(1)
g_obs.SetMarkerStyle(20)
g_obs.SetMarkerSize(0.8)

# Expected limit
g_exp = ROOT.TGraph(n_points, array('d', x), array('d', exp0))
g_exp.SetLineWidth(2)
g_exp.SetLineStyle(2)
g_exp.SetLineColor(ROOT.kBlack)

# 1-sigma band (green)
g_exp1sigma = ROOT.TGraphAsymmErrors(n_points)
for i in range(n_points):
    g_exp1sigma.SetPoint(i, x[i], exp0[i])
    g_exp1sigma.SetPointError(i, 0, 0, exp0[i] - exp_m1[i], exp_p1[i] - exp0[i])
g_exp1sigma.SetFillColor(ROOT.kGreen)
g_exp1sigma.SetLineColor(ROOT.kGreen)

# 2-sigma band (yellow)
g_exp2sigma = ROOT.TGraphAsymmErrors(n_points)
for i in range(n_points):
    g_exp2sigma.SetPoint(i, x[i], exp0[i])
    g_exp2sigma.SetPointError(i, 0, 0, exp0[i] - exp_m2[i], exp_p2[i] - exp0[i])
g_exp2sigma.SetFillColor(ROOT.kYellow)
g_exp2sigma.SetLineColor(ROOT.kYellow)

# Reference limit graph (HIG-18-020)
g_ref = ROOT.TGraph(n_points_ref, array('d', x_ref), array('d', obs_ref))
g_ref.SetLineWidth(3)
g_ref.SetLineStyle(2)
g_ref.SetLineColor(ROOT.kRed)

# Determine y-axis range
all_values = obs + exp0 + exp_m1 + exp_m2 + exp_p1 + exp_p2 + obs_ref
y_max = max(all_values) * 1.5

print(f"Created Brazilian plot with {n_points} mass points")

CMS.SetExtraText("Preliminary")
CMS.SetLumi(LumiInfo[args.era], run=args.era)
CMS.SetEnergy(get_CoM_energy(args.era))
CMS.ResetAdditionalInfo()
canv = CMS.cmsCanvas("limit", 15., 155., 0., y_max,
                        "m_{A} [GeV]", "95% CL limit on #sigma_{sig} [fb]", square=True, iPos=0, extraSpace=0.01)
canv.cd()

# Draw in order: 2sigma, 1sigma, expected, observed, reference
CMS.cmsObjectDraw(g_exp2sigma, "E3", FillColor=ROOT.TColor.GetColor("#85D1FBff"))
CMS.cmsObjectDraw(g_exp1sigma, "E3 same", FillColor=ROOT.TColor.GetColor("#FFDF7Fff"))
CMS.cmsObjectDraw(g_exp, "L same")
CMS.cmsObjectDraw(g_obs, "LP same")
CMS.cmsObjectDraw(g_ref, "L same")

# Legend
leg = CMS.cmsLeg(0.63, 0.90 - 0.05*5, 0.95, 0.90, textSize=0.04)
leg.AddEntry(g_obs, "Observed", "lp")
leg.AddEntry(g_exp, "Expected", "l")
leg.AddEntry(g_exp1sigma, "Expected #pm1#sigma", "f")
leg.AddEntry(g_exp2sigma, "Expected #pm2#sigma", "f")
leg.AddEntry(g_ref, "HIG-18-020", "l")

output_path = f"results/plots/limit.{args.era}.{args.channel}.{args.limit_type}.{args.method}.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
canv.SaveAs(output_path)