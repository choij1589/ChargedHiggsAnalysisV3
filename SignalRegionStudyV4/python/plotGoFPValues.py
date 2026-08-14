#!/usr/bin/env python3
"""Per-mHc summary of the saturated-GoF background-only p-values.

Reads each group seed's combine_output/gof/gof.json (written by
runGoF.sh --step collect) for the requested targets and plots p vs the
seed mA on a log scale with the p = 0.05 reference line. p = 0 is
floored at 0.5/ntoys (finite-toy resolution), mirroring V3's
plotGoFPValues.

  python3 python/plotGoFPValues.py --mhc 160
  python3 python/plotGoFPValues.py --all
"""
import argparse
import json
import os
from array import array

import ROOT
import cmsstyle as CMS

import srspaths
import interpolation_config
from interpolation_config import masspoint_name

ROOT.gROOT.SetBatch(True)

CHANNEL_STYLE = {
    "Combined": (ROOT.kBlack, 20),
    "SR1E2Mu": (ROOT.kAzure + 2, 21),
    "SR3Mu": (ROOT.kRed + 1, 22),
}


def gof_json_path(seed_mp, era, channel, method_segment, source):
    method = ("ParticleNet" if method_segment.startswith("ParticleNet")
              else "Baseline")
    tdir = srspaths.template_dir(seed_mp, method, era, channel,
                                 source=source)
    # method_segment differs from the method only for blind runs, which
    # the GoF chain does not produce by default; keep the hook anyway.
    if method_segment != method:
        tdir = tdir.replace(f"/{method}/", f"/{method_segment}/")
    return os.path.join(tdir, "combine_output", "gof", "gof.json")


def read_pvalue(path, floor):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    entry = payload.get("120.0") or next(iter(payload.values()))
    p = float(entry["p"])
    return max(p, floor)


def plot_mhc(mhc, args):
    cfg = (srspaths.grid_config()
           if args.method.startswith("Baseline")
           else srspaths.pnet_grid_config())
    grids = cfg["grids"][f"MHc{mhc}"]
    seeds = [g["seed"] for g in grids["groups"]]
    floor = 0.5 / args.ntoys

    graphs, n_missing = {}, 0
    for channel in args.channels:
        xs, ys = [], []
        for seed in seeds:
            mp = masspoint_name(seed, mhc)
            p = read_pvalue(gof_json_path(mp, args.era, channel,
                                          args.method, args.signal_source),
                            floor)
            if p is None:
                n_missing += 1
                continue
            xs.append(float(seed))
            ys.append(p)
        if xs:
            g = ROOT.TGraph(len(xs), array("d", xs), array("d", ys))
            color, marker = CHANNEL_STYLE.get(channel,
                                              (ROOT.kGray + 2, 24))
            g.SetMarkerColor(color)
            g.SetLineColor(color)
            g.SetMarkerStyle(marker)
            g.SetMarkerSize(0.7)
            graphs[channel] = g
    if n_missing:
        print(f"MHc{mhc}: {n_missing} missing gof.json entries "
              "(chain incomplete there)")
    if not graphs:
        print(f"MHc{mhc}: nothing to plot")
        return

    CMS.SetExtraText("Preliminary")
    x_lo, x_hi = min(seeds) - 2, max(seeds) + 2
    canv = CMS.cmsCanvas(f"gof_{mhc}", x_lo, x_hi, floor / 2, 2.0,
                         "m_{A} [GeV]", "saturated GoF p-value (B-only)",
                         square=True, iPos=11, extraSpace=0.01)
    canv.SetLogy(True)
    line = ROOT.TLine(x_lo, 0.05, x_hi, 0.05)
    line.SetLineStyle(ROOT.kDashed)
    line.SetLineColor(ROOT.kGray + 2)
    line.Draw("same")
    legend = CMS.cmsLeg(0.6, 0.75, 0.9, 0.9, textSize=0.03)
    for channel, g in graphs.items():
        CMS.cmsObjectDraw(g, "P same")
        legend.AddEntry(g, f"All/{channel}", "p")
    legend.Draw("same")

    outdir = "results/plots/gof"
    os.makedirs(outdir, exist_ok=True)
    base = (f"{outdir}/pvalue.{args.era}.{args.method}."
            f"{args.signal_source}.MHc{mhc}")
    canv.SaveAs(f"{base}.png")
    canv.SaveAs(f"{base}.pdf")
    worst = min((min(g.GetY()[i] for i in range(g.GetN())), ch)
                for ch, g in graphs.items())
    print(f"MHc{mhc}: {sum(g.GetN() for g in graphs.values())} points "
          f"plotted; worst p = {worst[0]:.3g} ({worst[1]}) -> {base}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", type=int, action="append", default=None)
    parser.add_argument("--all", action="store_true",
                        help="all seven mHc")
    parser.add_argument("--era", default="All")
    parser.add_argument("--channels", nargs="+",
                        default=["Combined", "SR1E2Mu", "SR3Mu"])
    parser.add_argument("--method", default="Baseline")
    parser.add_argument("--signal-source", default="interp-signal",
                        choices=["mc-signal", "interp-signal"])
    parser.add_argument("--ntoys", type=int, default=500,
                        help="p-value floor = 0.5/ntoys")
    args = parser.parse_args()

    if args.all:
        if args.method.startswith("ParticleNet"):
            import pnet_interp_config
            mhcs = [pnet_interp_config.mhc_int(m)
                    for m in pnet_interp_config.pn_mhc_list()]
        else:
            mhcs = interpolation_config.mhc_grid()
    else:
        mhcs = args.mhc or []
    if not mhcs:
        parser.error("--mhc N or --all required")
    for mhc in mhcs:
        plot_mhc(mhc, args)


if __name__ == "__main__":
    main()
