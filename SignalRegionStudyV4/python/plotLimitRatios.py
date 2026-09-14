#!/usr/bin/env python3
"""Ratio of median expected limits: dimuon-mass-only / ParticleNet-selected.

One graph per mHc over the ParticleNet scan (82.5 <= mA <= 97.5), from the
same two JSONs used by plotLimits.py: the Baseline arm evaluated at the
150 ParticleNet mass points against the ParticleNet arm. A ratio above
one means the ParticleNet-selected templates give the more stringent
expected limit. Replaces the per-mHc median/min/max table in the AN with
the mA trend (the improvement peaks on the Z resonance).

  python3 python/plotLimitRatios.py [--era All] [--channel Combined]
"""

import argparse
import json
import re

import ROOT
import cmsstyle as CMS
from plotter import LumiInfo, EnergyInfo, PALETTE_LONG

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--era", default="All", choices=["All", "Run2", "Run3"])
parser.add_argument("--channel", default="Combined",
                    choices=["Combined", "SR1E2Mu", "SR3Mu"])
parser.add_argument("--limit-type", dest="limit_type", default="Asymptotic")
parser.add_argument("--signal-source", dest="signal_source",
                    default="interp-signal",
                    choices=["mc-signal", "interp-signal"])
args = parser.parse_args()

MASSPOINT_RE = re.compile(r"^MHc(?P<mhc>\d+)_MA(?P<ma>\d+(?:p\d+)?)$")


def parse_masspoint(mp):
    m = MASSPOINT_RE.match(mp)
    return int(m.group("mhc")), float(m.group("ma").replace("p", "."))


_ch_suffix = "" if args.channel == "Combined" else f".{args.channel}"
_source_infix = "" if args.signal_source == "mc-signal" \
    else f".{args.signal_source}"
_json_dir = f"results/json/BR/{args.era}"

with open(f"{_json_dir}/limits.{args.era}{_ch_suffix}.{args.limit_type}"
          f".Baseline{_source_infix}.json") as f:
    baseline = json.load(f)
with open(f"{_json_dir}/limits.{args.era}{_ch_suffix}.{args.limit_type}"
          f".ParticleNet{_source_infix}.json") as f:
    particlenet = json.load(f)

ratios = {}
for mp, pn in particlenet.items():
    if mp not in baseline:
        raise RuntimeError(f"{mp} present in ParticleNet JSON but not in Baseline")
    mhc, ma = parse_masspoint(mp)
    ratios.setdefault(mhc, []).append((ma, baseline[mp]["exp0"] / pn["exp0"]))

# Setup CMS style; header conventions follow plotLimits.py.
CMS.SetExtraText("Preliminary")
CMS.ResetAdditionalInfo()
if args.era == "All":
    CMS.SetLumi(None, run=(f"{LumiInfo['Run2']:g} fb^{{#minus1}} ({EnergyInfo['Run2']:g} TeV) + "
                           f"{LumiInfo['Run3']:g} fb^{{#minus1}}"))
    CMS.SetEnergy(0, unit=f"{EnergyInfo['Run3']:g} TeV")
else:
    CMS.SetLumi(None, run=f"{LumiInfo[args.era]:g} fb^{{#minus1}}")
    CMS.SetEnergy(EnergyInfo[args.era])

_CHANNEL_LABELS = {
    "Combined": "e#mu#mu + #mu#mu#mu",
    "SR1E2Mu":  "e#mu#mu",
    "SR3Mu":    "#mu#mu#mu",
}
_iPos = 11
_INFO_X = 0.20
_INFO_Y_CHANNEL = 0.76

all_values = [r for pts in ratios.values() for _, r in pts]
y_max = 1.15 * max(all_values)
y_min = 0.9

canv = CMS.cmsCanvas("limit_ratio", 82., 98., y_min, y_max,
                     "m_{A} [GeV]",
                     "Median expected limit ratio",
                     square=True, iPos=_iPos, extraSpace=0.01)
canv.cd()

unity = ROOT.TLine(82., 1., 98., 1.)
unity.SetLineStyle(ROOT.kDashed)
unity.SetLineColor(ROOT.kGray + 2)
unity.Draw()

graphs = []
for idx, mhc in enumerate(sorted(ratios)):
    pts = sorted(ratios[mhc])
    g = ROOT.TGraph(len(pts))
    for i, (ma, r) in enumerate(pts):
        g.SetPoint(i, ma, r)
    color = PALETTE_LONG[idx % len(PALETTE_LONG)]
    g.SetLineColor(color)
    g.SetLineStyle(ROOT.kSolid)
    g.SetLineWidth(2)
    g.SetMarkerStyle(20)
    g.SetMarkerSize(0.8)
    g.SetMarkerColor(color)
    graphs.append((mhc, g))

for _, g in graphs:
    CMS.cmsObjectDraw(g, "LP same")
canv.RedrawAxis()

ch_label = ROOT.TLatex()
ch_label.SetNDC(True)
ch_label.SetTextFont(42)
ch_label.SetTextSize(0.04)
ch_label.DrawLatex(_INFO_X, _INFO_Y_CHANNEL, _CHANNEL_LABELS[args.channel])

leg = CMS.cmsLeg(0.65, 0.90 - 0.05 * len(graphs), 0.90, 0.90, textSize=0.035)
for mhc, g in graphs:
    leg.AddEntry(g, f"m_{{H^{{+}}}} = {mhc} GeV", "lp")

output_base = (f"results/plots/BR/{args.era}/limit_ratio.{args.era}{_ch_suffix}"
               f".{args.limit_type}{_source_infix}")
canv.SaveAs(f"{output_base}.png")
canv.SaveAs(f"{output_base}.pdf")

# Console summary for the AN text.
from statistics import median
print(f"Ratio summary ({args.era}, {args.channel}, {args.signal_source}):")
print(f"  overall: min {min(all_values):.2f}, median {median(all_values):.2f}, "
      f"max {max(all_values):.2f}")
for mhc, pts in sorted(ratios.items()):
    pts = sorted(pts)
    peak_ma, peak_r = max(pts, key=lambda p: p[1])
    print(f"  MHc{mhc}: edge ratios {pts[0][1]:.2f} (mA={pts[0][0]:g}) / "
          f"{pts[-1][1]:.2f} (mA={pts[-1][0]:g}), "
          f"peak {peak_r:.2f} at mA={peak_ma:g}")
