#!/usr/bin/env python3
"""
Overlay chi2/ndf vs systematic uncertainty profiles across eras for a given
channel / histkey / syst combination.

Usage:
    python python/plotChi2Profiles.py --eras 2022 2022EE 2023 2023BPix Run3 \
        --channel Run1E2Mu --histkey pair/mass --syst Central
"""
import os
import argparse
import json
import ROOT
import numpy as np

WORKDIR = os.environ["WORKDIR"]

# Colour cycle (ROOT colour indices)
COLORS = [
    ROOT.kBlue + 1,
    ROOT.kRed + 1,
    ROOT.kGreen + 2,
    ROOT.kMagenta + 1,
    ROOT.kOrange + 1,
    ROOT.kCyan + 2,
    ROOT.kViolet + 2,
    ROOT.kPink + 1,
]
MARKERS = [20, 21, 22, 23, 24, 25, 26, 27]


def load_profile(era, channel, histkey, syst):
    variable = histkey.replace("/", "_").lower()
    path = (f"{WORKDIR}/MeasFakeRateV4/plots/{era}/{channel}/{syst}"
            f"/closure_{variable}_yield.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        data = json.load(f)
    if "chi2_profile" not in data:
        raise KeyError(f"No chi2_profile in {path} — re-run plotClosure.py first")
    return data


parser = argparse.ArgumentParser()
parser.add_argument("--eras",    nargs="+", required=True)
parser.add_argument("--channel", required=True)
parser.add_argument("--histkey", required=True)
parser.add_argument("--syst",    default="Central")
args = parser.parse_args()

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)

graphs = []
legend_entries = []

x_max = 100.0  # max syst_pct on x-axis

for i, era in enumerate(args.eras):
    try:
        data = load_profile(era, args.channel, args.histkey, args.syst)
    except (FileNotFoundError, KeyError) as e:
        print(f"WARNING: skipping {era}: {e}")
        continue

    profile = data["chi2_profile"]
    xs = np.array([e["syst_pct"] for e in profile], dtype=float)
    ys = np.array([min(e["chi2_per_ndf"], 10.0) for e in profile], dtype=float)

    rec = data.get("recommended_systematic_pct", None)

    g = ROOT.TGraph(len(xs), xs, ys)
    color  = COLORS[i % len(COLORS)]
    marker = MARKERS[i % len(MARKERS)]
    g.SetLineColor(color)
    g.SetLineWidth(2)
    g.SetMarkerColor(color)
    g.SetMarkerStyle(marker)
    g.SetMarkerSize(0.9)
    graphs.append(g)

    label = era if rec is None else f"{era} (rec. {rec}%)"
    legend_entries.append((g, label))

if not graphs:
    raise RuntimeError("No valid profiles found — check eras and paths.")

# Canvas
canv = ROOT.TCanvas("canv", "", 700, 550)
canv.SetLeftMargin(0.13)
canv.SetBottomMargin(0.13)

# Draw first graph to set axes
graphs[0].SetTitle(f";Systematic uncertainty [%];#chi^{{2}}/ndf")
graphs[0].GetXaxis().SetLimits(0.0, x_max)
graphs[0].GetYaxis().SetRangeUser(0.0, min(10.0, max(g.GetY()[0] for g in graphs) * 1.15))
graphs[0].Draw("ALP")
for g in graphs[1:]:
    g.Draw("LP same")

# chi2/ndf = 1 reference line
line = ROOT.TLine(0.0, 1.0, x_max, 1.0)
line.SetLineColor(ROOT.kBlack)
line.SetLineWidth(2)
line.SetLineStyle(2)
line.Draw("same")

# Legend
leg = ROOT.TLegend(0.45, 0.55, 0.88, 0.88)
leg.SetBorderSize(0)
leg.SetFillStyle(0)
leg.SetTextSize(0.033)
for g, label in legend_entries:
    leg.AddEntry(g, label, "lp")
leg.AddEntry(line, "#chi^{2}/ndf = 1", "l")
leg.Draw()

# Save
variable = args.histkey.replace("/", "_").lower()
era_tag = "_".join(args.eras)
outdir = f"{WORKDIR}/MeasFakeRateV4/plots/chi2profiles"
os.makedirs(outdir, exist_ok=True)
outpath = f"{outdir}/{args.channel}_{variable}_{args.syst}_{era_tag}.png"
canv.SaveAs(outpath)
print(f"Saved: {outpath}")
