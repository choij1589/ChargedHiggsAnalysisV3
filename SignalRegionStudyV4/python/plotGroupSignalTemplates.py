#!/usr/bin/env python3
"""Supplementary: every signal template of one interp-signal group on the
group's shared background.

A template group's SEED builds the backgrounds; its members inject only
their parametric signal (docs/interpolation/WORKFLOW.md "Template
production"). This draws, per Run x Channel category of the seed's
{era}/{channel} target, the seed's background stack + data + Data/Pred.
ratio -- the validation stack_background panel -- with the signal template
of every group member overlaid.

That the backgrounds are shared is what the figure asserts, so it is
checked, not assumed: every member's background components and data_obs
must equal the seed's bin by bin, or the script fails.

Outputs go to {seed template dir}/validation/group_signals/ -- a subdir of
its own, because validateRunPeriodTemplates.py rmtree's the per-category
validation dirs on every rerun.
"""

import argparse
import json
import os
from collections import OrderedDict

import ROOT

import validateRunPeriodTemplates as vrt  # also sets batch mode + Common/Tools path
import cmsstyle as CMS
import interpolation_config
import srspaths


Y_HEADROOM = 2.0
SEED_LINE_WIDTH = 3
MEMBER_LINE_WIDTH = 2


class GroupSignalCanvas(vrt.ValidationComparisonCanvas):
    """Validation stack with N signal lines instead of one."""

    def drawPadUp(self):
        self._cd_main()
        self.hs = CMS.buildTHStack(list(self.hists.values()), self.palette, LineColor=-1, FillColor=-1)
        CMS.cmsObjectDraw(self.hs, "hist")
        CMS.cmsObjectDraw(self.systematics, "FE2", FillStyle=3004, LineWidth=0, FillColor=12, MarkerSize=0)

        for signal in self.config["groupSignals"]:
            hist = signal["hist"]
            hist.SetLineColor(signal["color"])
            hist.SetLineWidth(signal["width"])
            hist.SetLineStyle(ROOT.kSolid)
            hist.SetFillStyle(0)
            hist.SetMarkerSize(0)
            hist.SetStats(0)
            hist.Draw("HIST SAME")

        CMS.cmsObjectDraw(self.incl, "PE", MarkerStyle=ROOT.kFullCircle, MarkerSize=1.0, MarkerColor=1)

        entries = [(self.incl, self.incl.GetTitle(), "PE")]
        entries.extend((self.hists[name], name, "F") for name in reversed(list(self.hists.keys())))
        entries.append((self.systematics, self.config.get("systSrc", "Stat+Syst"), " FE2"))
        CMS.addToLegend(self.leg, *entries)

        self._draw_channel_text(self.config)
        size = self.config.get("channelSize", 0.05)
        CMS.drawText(self.config["mhcLabel"], posX=self.config.get("channelPosX", 0.2),
                     posY=self.config.get("channelPosY", 0.7) - 2 * size,
                     font=42, align=0, size=size)
        self._cd_main().RedrawAxis()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--masspoint", required=True, help="group SEED, e.g. MHc130_MA90")
    parser.add_argument("--method", required=True, choices=["Baseline", "ParticleNet"])
    parser.add_argument("--era", default="All")
    parser.add_argument("--channel", default="Combined")
    parser.add_argument("--show-ma", type=float, nargs="+", default=None,
                        help="mA values of the members to draw (default: every member); "
                             "the shared-background check always covers the whole group")
    return parser.parse_args()


def group_members(seed, method):
    """Every member of the seed's group (seed included), ordered by mA."""
    if interpolation_config.group_seed(seed, method) != seed:
        raise ValueError(f"{seed} is not a {method} group seed "
                         f"(its seed is {interpolation_config.group_seed(seed, method)})")
    mhc, seed_ma = srspaths.masspoint_mhc_ma(seed)
    grid = srspaths.pnet_grid_config() if method == "ParticleNet" else srspaths.grid_config()
    groups = [g for g in grid["grids"][f"MHc{mhc}"]["groups"] if float(g["seed"]) == float(seed_ma)]
    if len(groups) != 1:
        raise RuntimeError(f"expected exactly one group seeded at {seed}, found {len(groups)}")
    return [(float(ma), interpolation_config.masspoint_name(ma, mhc))
            for ma in sorted(groups[0]["members"])]


def member_template_dir(seed, member, method, era, channel):
    if member == seed:
        return srspaths.template_dir(seed, method, era, channel, source="interp-signal")
    return srspaths.interp_member_dir(seed, member, era, channel, method=method)


def open_shapes(tdir):
    path = os.path.join(tdir, "shapes.root")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    f = ROOT.TFile.Open(path, "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Failed to open {path}")
    return f


def same_hist(a, b):
    if a.GetNbinsX() != b.GetNbinsX():
        return False
    return all(a.GetBinContent(i) == b.GetBinContent(i) and a.GetBinError(i) == b.GetBinError(i)
               for i in range(a.GetNbinsX() + 2))


def check_shared_background(seed_dir, member_dir, category, payload, member):
    names = ["data_obs"] + [meta["name"] for meta in payload["processes"] if not meta.get("is_signal", False)]
    for name in names:
        ref = seed_dir.Get(name)
        hist = member_dir.Get(name)
        if not ref or not hist:
            raise RuntimeError(f"{member}/{category}: missing {name}")
        if not same_hist(ref, hist):
            raise RuntimeError(f"{member}/{category}/{name} differs from the seed's -- "
                               "the group does not share its background")


def assign_colors(signals, seed):
    """Seed black; the other drawn members on a blue -> red ramp in mA."""
    ROOT.gStyle.SetPalette(ROOT.kRainBow)
    palette = ROOT.TColor.GetPalette()
    # Clip both ends: the dark violet start reads as the black seed line,
    # the far red end is indistinguishable from its neighbour.
    lo, hi = int(0.05 * (palette.GetSize() - 1)), int(0.95 * (palette.GetSize() - 1))
    others = [s for s in signals if s["masspoint"] != seed]
    for idx, signal in enumerate(others):
        signal["color"] = palette[lo + int(round(idx * (hi - lo) / max(len(others) - 1, 1)))]
    for signal in signals:
        if signal["masspoint"] == seed:
            signal["color"] = ROOT.kBlack
    return signals


def make_category_plot(seed, args, category, payload, physics_groups, seed_file, member_files,
                       members, shown_ma, datacard_nuisances, output_dir):
    directory = seed_file.Get(category)
    if not directory:
        raise RuntimeError(f"{seed}: missing category directory {category}")
    data = directory.Get("data_obs")
    if not data:
        raise RuntimeError(f"{seed}/{category}: missing data_obs")

    group_hists = OrderedDict()
    colors = []
    for group, names in vrt.ordered_physics_groups(physics_groups).items():
        hist = vrt.sum_hists(directory, names, f"{category}_{group}")
        if hist and hist.Integral() > 0:
            hist.SetTitle(f"{group} ({hist.Integral():.1f})")
            group_hists[group] = hist
            colors.append(vrt.GROUP_COLORS.get(group, ROOT.kGray + 1))
    if not group_hists:
        raise RuntimeError(f"{seed}/{category}: no background with positive yield")
    total = vrt.total_from_hists(group_hists.values(), f"{category}_total_bkg")

    signals = []
    for ma, member in members:
        member_dir = member_files[member].Get(category)
        if not member_dir:
            raise RuntimeError(f"{member}: missing category directory {category}")
        check_shared_background(directory, member_dir, category, payload, member)
        hist = vrt.category_signal_hist(member_dir, category, payload)
        if hist is None or hist.Integral() <= 0:
            raise RuntimeError(f"{member}/{category}: no signal template")
        hist.SetName(f"{category}_signal_{member}")
        is_seed = member == seed
        signals.append({
            "masspoint": member,
            "mA": ma,
            "hist": hist,
            "width": SEED_LINE_WIDTH if is_seed else MEMBER_LINE_WIDTH,
        })

    data_draw = vrt.clone_hist(data, f"{category}_data_obs_draw")
    data_draw.SetTitle("Data")
    plot_era = vrt.plot_era_for_category(args, payload)
    plot_channel = payload.get("channel", args.channel)
    ymax = max([total.GetMaximum(), data_draw.GetMaximum()] + [s["hist"].GetMaximum() for s in signals])
    mhc = srspaths.masspoint_mhc_ma(seed)[0]

    config = {
        "era": plot_era,
        "CoM": vrt.plot_com_energy(plot_era),
        "iPos": vrt.plot_ipos(plot_era),
        "channel": vrt.plot_region_label(plot_channel),
        "region": vrt.plot_channel_label(plot_channel),
        "mhcLabel": f"m_{{H^{{#pm}}}} = {mhc} GeV",
        "xTitle": vrt.plot_x_title(plot_channel),
        "yTitle": "Events",
        "yRange": [0.0, ymax * Y_HEADROOM],
        "rTitle": "Data / Pred.",
        "rRange": [0.0, 2.0],
        "maxDigits": 3,
        "systSrc": "Stat+Syst",
        "colors": colors,
        "legend": vrt.POSTFIT_SUMMARY_LEGEND,
        "legendColumns": 2,
        "legendTextSize": vrt.POSTFIT_SUMMARY_LEGEND_TEXT_SIZE,
        "groupSignals": assign_colors([s for s in signals if s["mA"] in shown_ma], seed),
    }
    plotter = GroupSignalCanvas(data_draw, group_hists, config)
    vrt.apply_prefit_uncertainty_band(plotter.systematics, directory, category, payload, datacard_nuisances)
    vrt.apply_ratio_uncertainty_band(plotter)
    plotter.drawPadUp()
    plotter.drawPadDown()
    vrt.overdraw_lumi_header(plotter.canv, plot_era)

    os.makedirs(output_dir, exist_ok=True)
    output_base = os.path.join(output_dir, "group_signals")
    for ext in ("png", "pdf"):
        plotter.canv.SaveAs(f"{output_base}.{ext}")
    plotter.canv.Close()

    return {
        "plot": f"{output_base}.png",
        "data_yield": data_draw.Integral(),
        "background_yield": total.Integral(),
        "groups": {group: hist.Integral() for group, hist in group_hists.items()},
        "signal_yields": {s["masspoint"]: s["hist"].Integral() for s in signals},
    }


def main():
    args = parse_args()
    seed = args.masspoint
    members = group_members(seed, args.method)
    member_ma = {ma for ma, _ in members}
    shown_ma = member_ma if args.show_ma is None else set(args.show_ma)
    if not shown_ma <= member_ma:
        raise ValueError(f"--show-ma {sorted(shown_ma - member_ma)} not in the {seed} group "
                         f"(members: {sorted(member_ma)})")
    seed_tdir = member_template_dir(seed, seed, args.method, args.era, args.channel)

    categories = vrt.load_json(os.path.join(seed_tdir, "categories.json"))["categories"]
    physics_groups = vrt.load_json(os.path.join(seed_tdir, "process_list.json"))["physics_groups"]
    datacard_path = os.path.join(seed_tdir, "datacard.txt")
    if not os.path.exists(datacard_path):
        raise FileNotFoundError(datacard_path)
    datacard_nuisances = vrt.parse_datacard_nuisance_rows(datacard_path)

    seed_file = open_shapes(seed_tdir)
    member_files = {}
    try:
        for _, member in members:
            member_files[member] = (seed_file if member == seed else
                                    open_shapes(member_template_dir(seed, member, args.method,
                                                                    args.era, args.channel)))
        output_root = os.path.join(seed_tdir, "validation", "group_signals")
        per_category = OrderedDict()
        for category, payload in categories.items():
            per_category[category] = make_category_plot(
                seed, args, category, payload, physics_groups, seed_file, member_files,
                members, shown_ma, datacard_nuisances, os.path.join(output_root, category))
            print(f"[plotGroupSignalTemplates] {category}: {per_category[category]['plot']}")
    finally:
        for f in {id(f): f for f in list(member_files.values()) + [seed_file]}.values():
            f.Close()

    summary_path = os.path.join(output_root, "summary.json")
    with open(summary_path, "w") as fout:
        json.dump({
            "seed": seed,
            "method": args.method,
            "era": args.era,
            "channel": args.channel,
            "members": [member for _, member in members],
            "drawn": [member for ma, member in members if ma in shown_ma],
            "categories": per_category,
        }, fout, indent=2)
    print(f"[plotGroupSignalTemplates] wrote {summary_path}")


if __name__ == "__main__":
    main()
