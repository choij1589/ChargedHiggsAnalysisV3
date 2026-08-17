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
from plotter import LumiInfo, LumiInfoExact, EnergyInfo, get_CoM_energy

ROOT.gROOT.SetBatch(True)

# Fixed y range so every mHc panel of a campaign is read on one scale.
# The lower edge sits below the finite-toy floor of both toy counts in use
# (0.5/500 = 2e-3, 0.5/2000 = 2.5e-4), so floored points stay on the plot;
# the headroom above 1 keeps the legend clear of the p ~ 1 band.
Y_RANGE = (1e-4, 100.0)

# Marker size. The Baseline panels carry ~95 seeds per channel, so this
# trades legibility against crowding in the dense mA > 60 region.
MARKER_SIZE = 1.0

CHANNEL_STYLE = {
    "Combined": (ROOT.kBlack, 20),
    "SR1E2Mu": (ROOT.kAzure + 2, 21),
    "SR3Mu": (ROOT.kRed + 1, 22),
}

# The ParticleNet overlay reuses the channel colour and marker family and
# only opens the marker, so shape reads as channel and fill reads as method.
OPEN_MARKER = {20: 24, 21: 25, 22: 26, 24: 24, 25: 25, 26: 26}

# Final-state labels, as in plotLimits.py / plotLimits2D.py. The era is
# already in the luminosity header, so it is not repeated here.
CHANNEL_LABEL = {
    "Combined": "e#mu#mu + #mu#mu#mu",
    "SR1E2Mu": "e#mu#mu",
    "SR3Mu": "#mu#mu#mu",
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
    """p, floored at the finite-toy resolution of THIS point.

    The toy count varies across the grid (the low-mA cells were rerun with
    2000), so the floor is taken from the entry's own toy list; `floor`
    (from --ntoys) is only the fallback for records without one.
    """
    if not os.path.exists(path):
        return None
    with open(path) as f:
        payload = json.load(f)
    entry = payload.get("120.0") or next(iter(payload.values()))
    p = float(entry["p"])
    ntoys = len(entry.get("toy") or ())
    return max(p, 0.5 / ntoys if ntoys else floor)


def set_lumi_header(era):
    """Luminosity header shared with plotLimits.py / plotLimits2D.py.

    Un-rounded per-period luminosities from LumiInfoExact, each run period
    quoted with its own energy, no "Run2,"-style prefix. cmsstyle renders
    "<cms_lumi> (<cms_energy>)", so for All only the Run3 energy can live in
    SetEnergy and the whole Run2 term is baked into the run label.
    """
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


def seeds_for(mhc, method):
    """Group seeds of this mHc study, or None if the method has no grid here.

    ParticleNet covers only five of the seven mHc, so the overlay is simply
    absent at mHc = 70, 85 rather than an error.
    """
    cfg = (srspaths.grid_config() if method.startswith("Baseline")
           else srspaths.pnet_grid_config())
    grids = cfg["grids"].get(f"MHc{mhc}")
    if grids is None:
        return None
    return [g["seed"] for g in grids["groups"]]


def collect_graphs(mhc, seeds, method, args, open_markers=False):
    """{channel: TGraph} of p vs seed mA, plus the missing-file count."""
    floor = 0.5 / args.ntoys
    graphs, n_missing = {}, 0
    for channel in args.channels:
        xs, ys = [], []
        for seed in seeds:
            mp = masspoint_name(seed, mhc)
            p = read_pvalue(gof_json_path(mp, args.era, channel,
                                          method, args.signal_source),
                            floor)
            if p is None:
                n_missing += 1
                continue
            xs.append(float(seed))
            ys.append(p)
        if not xs:
            continue
        g = ROOT.TGraph(len(xs), array("d", xs), array("d", ys))
        color, marker = CHANNEL_STYLE.get(channel, (ROOT.kGray + 2, 24))
        if open_markers:
            marker = OPEN_MARKER[marker]
        g.SetMarkerColor(color)
        g.SetLineColor(color)
        g.SetMarkerStyle(marker)
        g.SetMarkerSize(MARKER_SIZE)
        graphs[channel] = g
    return graphs, n_missing


def plot_mhc(mhc, args):
    seeds = seeds_for(mhc, args.method)
    if seeds is None:
        print(f"MHc{mhc}: no {args.method} grid")
        return
    graphs, n_missing = collect_graphs(mhc, seeds, args.method, args)

    # ParticleNet overlaid on the Baseline panel: same colours, open markers.
    # Its mA reach is a subset of the Baseline one, so the axis is unchanged.
    overlay = {}
    both_arms = bool(args.overlay) and args.method.startswith("Baseline")
    if both_arms:
        ov_seeds = seeds_for(mhc, args.overlay)
        if ov_seeds:
            overlay, ov_missing = collect_graphs(
                mhc, ov_seeds, args.overlay, args, open_markers=True)
            n_missing += ov_missing

    if n_missing:
        print(f"MHc{mhc}: {n_missing} missing gof.json entries "
              "(chain incomplete there)")
    if not graphs and not overlay:
        print(f"MHc{mhc}: nothing to plot")
        return

    CMS.SetExtraText("Preliminary")
    set_lumi_header(args.era)
    x_lo, x_hi = min(seeds) - 2, max(seeds) + 2
    canv = CMS.cmsCanvas(f"gof_{mhc}", x_lo, x_hi, Y_RANGE[0], Y_RANGE[1],
                         "m_{A} [GeV]", "saturated GoF p-value (B-only)",
                         square=True, iPos=11, extraSpace=0.01)
    canv.SetLogy(True)
    line = ROOT.TLine(x_lo, 0.05, x_hi, 0.05)
    line.SetLineStyle(ROOT.kDashed)
    line.SetLineColor(ROOT.kGray + 2)
    line.Draw("same")

    # One legend per method when both are shown, so the channel label is not
    # repeated with a fill qualifier glued onto it.
    legends = []
    if overlay:
        columns = [(0.40, args.method, graphs), (0.66, args.overlay, overlay)]
    else:
        columns = [(0.60, None, graphs)]
    for x_left, header, entries in columns:
        leg = CMS.cmsLeg(x_left, 0.68, x_left + 0.26, 0.90, textSize=0.028)
        if header:
            leg.SetHeader(header)
        for channel, g in entries.items():
            CMS.cmsObjectDraw(g, "P same")
            leg.AddEntry(g, CHANNEL_LABEL.get(channel, channel), "p")
        leg.Draw("same")
        legends.append(leg)

    # The method token is dropped for a both-arms run, since the panel is no
    # longer specific to either; single-method runs keep it. Keyed on the
    # REQUEST, not on `overlay`, so mHc 70 and 85 -- which have no
    # ParticleNet grid -- are named like the rest of the campaign.
    outdir = "results/plots/gof"
    os.makedirs(outdir, exist_ok=True)
    method_token = "" if both_arms else f"{args.method}."
    base = (f"{outdir}/pvalue.{args.era}.{method_token}"
            f"{args.signal_source}.MHc{mhc}")
    canv.SaveAs(f"{base}.png")
    canv.SaveAs(f"{base}.pdf")
    all_graphs = list(graphs.items()) + [(f"{ch} ({args.overlay})", g)
                                         for ch, g in overlay.items()]
    worst = min((min(g.GetY()[i] for i in range(g.GetN())), ch)
                for ch, g in all_graphs)
    print(f"MHc{mhc}: {sum(g.GetN() for _, g in all_graphs)} points "
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
    parser.add_argument("--overlay", default="ParticleNet",
                        help="second method drawn on the Baseline panels as "
                             "open markers; empty string disables it")
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
