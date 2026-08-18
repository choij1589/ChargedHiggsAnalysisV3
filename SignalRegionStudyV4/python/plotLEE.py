#!/usr/bin/env python3
"""Look-elsewhere diagnostics: the scan curve and its upcrossing ladder.

Two panels per (arm, mHc), the two things a reader has to be able to
check about a Gross-Vitells trials estimate:

  scan.*        observed Z(m_{A}) for the three channels, with the
                threshold ladder drawn -- this is the field whose
                excursions are being counted, so its smoothness and the
                width of its features are visible directly.
  upcrossings.* N_u(u) against u with the fitted N_0 exp(-u/2) overlaid.
                A straight line on this semilog panel is the asymptotic
                law holding; curvature is the warning that the
                extrapolation up to Z_max is not supported.

Reads the same inputs as estimateLEE.py and re-derives the curves through
its loaders, so the plots cannot show a different statistic than the
numbers.

    python3 python/plotLEE.py --all
    python3 python/plotLEE.py --mhc 145 --statistic bandpull
"""
import argparse
import math
import os
from array import array

import ROOT
import cmsstyle as CMS

import estimateLEE
import srspaths
from plotter import LumiInfo, LumiInfoExact, EnergyInfo, get_CoM_energy

ROOT.gROOT.SetBatch(True)

CHANNEL_STYLE = {
    "Combined": (ROOT.kBlack, 20),
    "SR1E2Mu": (ROOT.kAzure + 2, 21),
    "SR3Mu": (ROOT.kRed + 1, 22),
}
CHANNEL_LABEL = {
    "Combined": "e#mu#mu + #mu#mu#mu",
    "SR1E2Mu": "e#mu#mu",
    "SR3Mu": "#mu#mu#mu",
}
# Fixed so every panel of a campaign is read on one scale; the scan
# maxima are ~+3 and the deepest deficits ~-2.6.
Z_RANGE = (-4.0, 6.0)
NU_RANGE = (0.5, 200.0)


def set_lumi_header(era):
    """Same header as plotGoFPValues.py / plotLimits.py."""
    CMS.ResetAdditionalInfo()
    if era == "All":
        CMS.SetLumi(None, run=(
            f"{LumiInfoExact['Run2']:g} fb^{{#minus1}} "
            f"({EnergyInfo['Run2']:g} TeV) + "
            f"{LumiInfoExact['Run3']:g} fb^{{#minus1}}"))
        CMS.SetEnergy(0, unit=f"{EnergyInfo['Run3']:g} TeV")
    elif era in ("Run2", "Run3"):
        CMS.SetLumi(None, run=f"{LumiInfoExact[era]:g} fb^{{#minus1}}")
        CMS.SetEnergy(EnergyInfo[era])
    else:
        CMS.SetLumi(LumiInfo[era], run=era)
        CMS.SetEnergy(get_CoM_energy(era))


def load_curves(args):
    if args.statistic == "significance":
        curves, _ = estimateLEE.load_significance_curves(args.era,
                                                         args.signal_source)
        return curves, None
    calib = estimateLEE.fit_bandpull_calibration(args.era, args.signal_source)
    curves = estimateLEE.load_bandpull_curves(args.era, args.signal_source,
                                              calib["slope"])
    return curves, calib


def base_name(kind, args, method, mhc):
    outdir = os.path.join(srspaths.module_dir(), "results", "plots", "lee")
    os.makedirs(outdir, exist_ok=True)
    stat = "" if args.statistic == "significance" else "bandpull."
    return os.path.join(outdir, f"{kind}.{args.era}.{method}.{stat}"
                                f"{args.signal_source}.MHc{mhc}")


def plot_scan(args, method, mhc, columns):
    """Z(mA) for every channel of one study, with the threshold ladder."""
    xs_all = [ma for pts in columns.values() for ma, _ in pts]
    x_lo, x_hi = min(xs_all) - 2, max(xs_all) + 2

    CMS.SetExtraText("Preliminary")
    set_lumi_header(args.era)
    canv = CMS.cmsCanvas(f"lee_scan_{method}_{mhc}", x_lo, x_hi,
                         Z_RANGE[0], Z_RANGE[1], "m_{A} [GeV]",
                         "observed local significance Z", square=True,
                         iPos=11, extraSpace=0.01)

    lines = []
    for level in estimateLEE.THRESHOLDS:
        ln = ROOT.TLine(x_lo, level, x_hi, level)
        ln.SetLineStyle(ROOT.kDashed)
        ln.SetLineColor(ROOT.kGray + 1)
        ln.Draw("same")
        lines.append(ln)

    graphs, leg = {}, CMS.cmsLeg(0.58, 0.70, 0.90, 0.90, textSize=0.030)
    leg.SetHeader(f"m_{{H^{{#pm}}}} = {mhc} GeV, {method}")
    for channel in ("SR3Mu", "SR1E2Mu", "Combined"):
        pts = sorted(columns.get(channel, []))
        if not pts:
            continue
        g = ROOT.TGraph(len(pts), array("d", [p[0] for p in pts]),
                        array("d", [p[1] for p in pts]))
        color, _ = CHANNEL_STYLE[channel]
        g.SetLineColor(color)
        g.SetLineWidth(2)
        CMS.cmsObjectDraw(g, "L same")
        leg.AddEntry(g, CHANNEL_LABEL[channel], "l")
        graphs[channel] = g
    leg.Draw("same")

    base = base_name("scan", args, method, mhc)
    canv.SaveAs(f"{base}.png")
    canv.SaveAs(f"{base}.pdf")
    return base


def plot_upcrossings(args, method, mhc, columns):
    """N_u(u) with the fitted N_0 exp(-u/2); u = Z^2 on the x axis."""
    CMS.SetExtraText("Preliminary")
    set_lumi_header(args.era)
    u_hi = max(estimateLEE.THRESHOLDS) ** 2 + 1.0
    canv = CMS.cmsCanvas(f"lee_nu_{method}_{mhc}", 0.0, u_hi,
                         NU_RANGE[0], NU_RANGE[1], "u = Z^{2}",
                         "upcrossings N_{u}", square=True, iPos=11,
                         extraSpace=0.01)
    canv.SetLogy(True)

    keep, leg = [], CMS.cmsLeg(0.55, 0.68, 0.90, 0.90, textSize=0.030)
    leg.SetHeader(f"m_{{H^{{#pm}}}} = {mhc} GeV, {method}")
    for channel in ("SR3Mu", "SR1E2Mu", "Combined"):
        pts = sorted(columns.get(channel, []))
        if not pts:
            continue
        row = estimateLEE.analyse_column(method, channel, mhc, pts)
        us, ns = [], []
        for z in estimateLEE.THRESHOLDS:
            n_u = row["upcrossings"][f"{z:g}"]
            if n_u > 0:
                us.append(z * z)
                ns.append(float(n_u))
        if not us:
            continue
        color, marker = CHANNEL_STYLE[channel]
        g = ROOT.TGraph(len(us), array("d", us), array("d", ns))
        g.SetMarkerColor(color)
        g.SetMarkerStyle(marker)
        g.SetMarkerSize(1.2)
        CMS.cmsObjectDraw(g, "P same")
        keep.append(g)
        if row["n0"]:
            fn = ROOT.TF1(f"n0_{channel}_{mhc}",
                          f"{row['n0']}*exp(-0.5*x)", 0.0, u_hi)
            fn.SetLineColor(color)
            fn.SetLineStyle(ROOT.kDashed)
            fn.SetLineWidth(2)
            fn.Draw("same")
            keep.append(fn)
            leg.AddEntry(g, f"{CHANNEL_LABEL[channel]}  "
                            f"N_{{0}} = {row['n0']:.0f}", "p")
        else:
            leg.AddEntry(g, CHANNEL_LABEL[channel], "p")
    leg.Draw("same")

    base = base_name("upcrossings", args, method, mhc)
    canv.SaveAs(f"{base}.png")
    canv.SaveAs(f"{base}.pdf")
    return base


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mhc", type=int, action="append", default=None)
    parser.add_argument("--all", action="store_true", help="every study")
    parser.add_argument("--method", nargs="+", default=estimateLEE.METHODS,
                        choices=estimateLEE.METHODS)
    parser.add_argument("--statistic", default="significance",
                        choices=["significance", "bandpull"])
    parser.add_argument("--era", default="All")
    parser.add_argument("--signal-source", default="interp-signal",
                        choices=["interp-signal", "mc-signal"])
    parser.add_argument("--min-points", type=int, default=10)
    args = parser.parse_args()
    if not args.all and not args.mhc:
        raise ValueError("use --all or --mhc N")

    curves, calib = load_curves(args)
    if calib:
        print(f"Band-pull calibration: Z = {calib['slope']:.3f} x pull "
              f"(rms {calib['rms']:.3f}, n = {calib['n_points']})")

    n_panels = 0
    for method in args.method:
        by_mhc = {}
        for channel, per_mhc in curves.get(method, {}).items():
            for mhc, pts in per_mhc.items():
                if args.mhc and mhc not in args.mhc:
                    continue
                if len(pts) < args.min_points:
                    continue
                by_mhc.setdefault(mhc, {})[channel] = pts
        if not by_mhc:
            print(f"{method}: nothing to plot")
            continue
        for mhc in sorted(by_mhc):
            base = plot_scan(args, method, mhc, by_mhc[mhc])
            plot_upcrossings(args, method, mhc, by_mhc[mhc])
            n_panels += 2
            print(f"{method} MHc{mhc}: {sum(len(v) for v in by_mhc[mhc].values())}"
                  f" scan points -> {base}.png")
    print(f"Wrote {n_panels} panels")


if __name__ == "__main__":
    main()
