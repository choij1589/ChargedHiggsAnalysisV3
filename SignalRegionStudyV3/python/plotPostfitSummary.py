#!/usr/bin/env python3
"""Full-mA prefit/postfit summary plots for fixed mHc values."""

import argparse
import json
import logging
import math
import os
import re
from array import array
from pathlib import Path
from statistics import median
from types import SimpleNamespace

import ROOT

import plotPostfitMass as pm


MASSPOINT_RE = re.compile(r"^MHc(?P<mhc>\d+)_MA(?P<ma>\d+)$")
VALID_ERAS = ("Run2", "Run3", "All")
VALID_CHANNELS = ("SR1E2Mu", "SR3Mu", "Combined")
VALID_METHODS = ("Baseline", "ParticleNet")
SOURCE_METHOD_PRIORITY = {
    "ParticleNet": 0,
    "Baseline": 1,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw full-mA prefit/postfit summary plots for fixed mHc values."
    )
    parser.add_argument("--mhc", nargs="+", type=int, default=[70, 160],
                        help="mHc values to plot")
    parser.add_argument("--methods", nargs="+", choices=VALID_METHODS,
                        default=list(VALID_METHODS))
    parser.add_argument("--eras", nargs="+", choices=VALID_ERAS,
                        default=list(VALID_ERAS),
                        help="Fit sources and era scopes to plot")
    parser.add_argument("--channels", nargs="+", choices=VALID_CHANNELS,
                        default=list(VALID_CHANNELS),
                        help="Channel scopes to plot")
    parser.add_argument("--fit-channel", default="Combined",
                        help="Channel segment used in the fitDiagnostics path")
    parser.add_argument("--binning", default="extended")
    parser.add_argument("--nuisance", default="fallback_lnn",
                        choices=["fallback_lnn", "preserve_shape"])
    parser.add_argument("--fit-type", default="b", choices=["b", "s", "both"],
                        help="Postfit variant to draw")
    parser.add_argument("--bin-width", type=float, default=1.0,
                        help="Summary bin width in GeV")
    parser.add_argument("--output-dir", default="results/plots/postfit_summary")
    parser.add_argument("--signal-line", choices=["none", "median"], default="none",
                        help="Overlay a signal template line; 'median' uses the available mass point nearest the median mA")
    parser.add_argument("--wide-mhc", nargs="+", type=int, default=[160],
                        help="mHc values drawn with enlarged canvas width")
    parser.add_argument("--wide-factor", type=float, default=2.0,
                        help="Canvas width scale for --wide-mhc values")
    parser.add_argument("--plot-only", action="store_true",
                        help="Require cached fine-mass hists from plotPostfitMass.py")
    parser.add_argument("--debug", action="store_true")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--unblind", action="store_true")
    mode.add_argument("--partial-unblind", action="store_true", dest="partial_unblind")
    mode.add_argument("--blind", action="store_true")
    return parser.parse_args()


def extract_mhc_ma(masspoint):
    match = MASSPOINT_RE.match(masspoint)
    if not match:
        raise ValueError(f"Invalid masspoint name: {masspoint}")
    return int(match.group("mhc")), int(match.group("ma"))


def blinding_tag(args):
    if args.unblind:
        return "unblind"
    if args.partial_unblind:
        return "partial_unblind"
    return "blind"


def template_suffix(args, method):
    return pm._compute_binning_suffix(SimpleNamespace(
        method=method,
        binning=args.binning,
        unblind=args.unblind,
        partial_unblind=args.partial_unblind,
        blind=args.blind,
        nuisance=args.nuisance,
    ))


def fitdiagnostics_path(args, era, method, masspoint):
    suffix = template_suffix(args, method)
    return (
        Path(pm.WORKDIR)
        / "SignalRegionStudyV3"
        / "templates"
        / era
        / args.fit_channel
        / masspoint
        / method
        / suffix
        / "combine_output"
        / "fitdiag"
        / f"fitDiagnostics.{masspoint}.{method}.{suffix}.root"
    )


def discover_method_masspoints(args, era, method, mhc):
    base = (Path(pm.WORKDIR) / "SignalRegionStudyV3" / "templates" / era / args.fit_channel)
    out = []
    for mp_dir in sorted(base.glob(f"MHc{mhc}_MA*")):
        if not MASSPOINT_RE.match(mp_dir.name):
            continue
        fitdiag = fitdiagnostics_path(args, era, method, mp_dir.name)
        if fitdiag.exists():
            out.append(mp_dir.name)
    return sorted(set(out), key=lambda mp: extract_mhc_ma(mp)[1])


def discover_masspoint_sources(args, era, method, mhc):
    if method != "ParticleNet":
        return [
            {"masspoint": masspoint, "source_method": method}
            for masspoint in discover_method_masspoints(args, era, method, mhc)
        ]

    selected = {}
    for source_method in ("Baseline", "ParticleNet"):
        for masspoint in discover_method_masspoints(args, era, source_method, mhc):
            selected[masspoint] = source_method

    return [
        {"masspoint": masspoint, "source_method": selected[masspoint]}
        for masspoint in sorted(selected, key=lambda mp: extract_mhc_ma(mp)[1])
    ]


def fit_types_from_arg(fit_type):
    return ["b", "s"] if fit_type == "both" else [fit_type]


def make_pm_args(args, era, method, channel, masspoint):
    return SimpleNamespace(
        era=era,
        masspoint=masspoint,
        method=method,
        binning=args.binning,
        nuisance=args.nuisance,
        era_scope=era,
        channel_scope=channel,
        fit_channel=args.fit_channel,
        fit_type=args.fit_type,
        unblind=args.unblind,
        partial_unblind=args.partial_unblind,
        blind=args.blind,
        bin_width=args.bin_width,
        plot_only=args.plot_only,
        debug=args.debug,
    )


def ordered_backgrounds(sub_cfgs):
    bkg_union = []
    for cfg in sub_cfgs.values():
        if cfg.get("components"):
            for component in cfg["components"]:
                group = component.get("physics_group")
                if (
                    group
                    and group != "signal"
                    and group in pm.BKG_ORDER
                    and group not in bkg_union
                ):
                    bkg_union.append(group)
            continue
        for bkg in cfg["separate_processes"]:
            if bkg in pm.BKG_ORDER and bkg not in bkg_union:
                bkg_union.append(bkg)
    if "others" not in bkg_union:
        bkg_union.append("others")
    return [b for b in pm.BKG_ORDER if b in bkg_union]


def build_grouped_process_aggregates(fit_file, kept, cfgs, ordered_bkgs, fit_type,
                                     hist_edges):
    """Aggregate V3 run-period component processes into physics groups.

    FitDiagnostics uses component process names like ``WZ_2018`` while the
    publication-style stack should stay grouped as ``WZ``.  The fine mass shape
    is refilled from the component's base sample, then scaled with the matching
    component histogram from FitDiagnostics.
    """
    edges_arr = array("d", hist_edges)
    n_uniform = len(hist_edges) - 1
    prefit = {
        bkg: ROOT.TH1D(f"{bkg}_pre_grouped_{id(kept)}", "", n_uniform, edges_arr)
        for bkg in ordered_bkgs
    }
    postfit = {
        bkg: ROOT.TH1D(f"{bkg}_post_grouped_{id(kept)}", "", n_uniform, edges_arr)
        for bkg in ordered_bkgs
    }
    for hist in list(prefit.values()) + list(postfit.values()):
        hist.SetDirectory(0)
    pre_signal = ROOT.TH1D(f"signal_pre_grouped_{id(kept)}", "", n_uniform, edges_arr)
    post_signal = ROOT.TH1D(f"signal_post_grouped_{id(kept)}", "", n_uniform, edges_arr)
    pre_signal.SetDirectory(0)
    post_signal.SetDirectory(0)
    signal_available = False
    missing_signal_warned = set()

    for subch in kept:
        cfg = cfgs[subch]
        source_edges = tuple(float(x) for x in cfg["bin_edges"])
        for component in cfg.get("components", []):
            group = component.get("physics_group")
            if group == "signal":
                process = component["base_process"]
                if process == "signal" or component.get("is_signal"):
                    process = pm.args.masspoint
                channel = component.get("channel", cfg.get("channel"))
                try:
                    fine = pm.cached_run_period_fine(
                        subch,
                        component["subera"],
                        channel,
                        process,
                        cfg,
                        hist_edges,
                        f"{component['name']}_{subch}_signal_summary",
                        is_data=False,
                    )
                except KeyError:
                    if getattr(pm.args, "plot_only", False):
                        if process not in missing_signal_warned:
                            logging.warning(
                                "Signal line skipped for %s: cache lacks %s. "
                                "Re-run without --plot-only to rebuild.",
                                pm.args.masspoint,
                                process,
                            )
                            missing_signal_warned.add(process)
                        continue
                    raise
                if fine.Integral() <= 0:
                    continue
                signal_available = True
                pre_signal.Add(fine)
                post_clone = fine.Clone(f"{component['name']}_{subch}_{fit_type}_signal_summary")
                post_clone.SetDirectory(0)
                scales = pm.get_coarse_scale(
                    fit_file,
                    subch,
                    component["name"],
                    fit_type,
                    len(source_edges) - 1,
                )
                pm.apply_coarse_scale(post_clone, scales, source_edges)
                post_signal.Add(post_clone)
                continue
            if group not in prefit:
                continue
            process = component["base_process"]
            if process == "signal" or component.get("is_signal"):
                process = pm.args.masspoint
            channel = component.get("channel", cfg.get("channel"))
            fine = pm.cached_run_period_fine(
                subch,
                component["subera"],
                channel,
                process,
                cfg,
                hist_edges,
                f"{component['name']}_{subch}_{group}_summary",
                is_data=False,
            )
            if fine.Integral() <= 0:
                continue
            prefit[group].Add(fine)

            post_clone = fine.Clone(f"{component['name']}_{subch}_{fit_type}_summary")
            post_clone.SetDirectory(0)
            scales = pm.get_coarse_scale(
                fit_file,
                subch,
                component["name"],
                fit_type,
                len(source_edges) - 1,
            )
            pm.apply_coarse_scale(post_clone, scales, source_edges)
            postfit[group].Add(post_clone)

    prefit = {name: hist for name, hist in prefit.items() if hist.Integral() > 0}
    postfit = {name: hist for name, hist in postfit.items() if hist.Integral() > 0}

    data = ROOT.TH1D(f"data_grouped_{id(kept)}", "data", n_uniform, edges_arr)
    data.SetDirectory(0)
    data.SetTitle("data")
    if pm.args.blind:
        for hist in prefit.values():
            data.Add(hist)
    else:
        seen_data = set()
        for subch in kept:
            cfg = cfgs[subch]
            for component in cfg.get("components", []):
                channel = component.get("channel", cfg.get("channel"))
                seen_data.add((subch, component["subera"], channel))
        for subch, subera, channel in sorted(seen_data):
            cfg = cfgs[subch]
            data.Add(pm.cached_run_period_fine(
                subch,
                subera,
                channel,
                "data",
                cfg,
                hist_edges,
                f"data_{subch}_{subera}_{channel}_summary",
                is_data=True,
            ))

    return prefit, postfit, pre_signal, post_signal, data, signal_available


def load_one_masspoint(args, era, source_method, channel, masspoint, fit_types):
    mp_args = make_pm_args(args, era, source_method, channel, masspoint)
    pm.entry_setup(mp_args, require_fitdiag=True, make_output_dir=False)
    pm._FINE_CACHE.clear()
    pm.CACHE_PATH = (
        f"{pm.CACHE_DIR}/mass_hists_v2_bw{args.bin_width:g}"
        f".summary_{era}_{channel}.root"
    )

    if os.path.exists(pm.CACHE_PATH):
        pm.load_cache_from_file(pm.CACHE_PATH)
    elif args.plot_only:
        raise FileNotFoundError(
            f"--plot-only requires cached fine-mass hists at {pm.CACHE_PATH}"
        )

    fit_file = ROOT.TFile.Open(pm.FITDIAG_PATH, "READ")
    if not fit_file or fit_file.IsZombie():
        raise RuntimeError(f"Failed to open {pm.FITDIAG_PATH}")

    all_subchannels = pm.discover_channels(fit_file)
    all_cfgs = {}
    for subch in all_subchannels:
        era_i, ch_i = pm.parse_subchannel(subch, era)
        all_cfgs[subch] = pm.load_subchannel_config(era_i, ch_i)

    global_lo = min(cfg["mass_min"] for cfg in all_cfgs.values())
    global_hi = max(cfg["mass_max"] for cfg in all_cfgs.values())
    pm.set_global_edges(pm.build_uniform_edges(global_lo, global_hi, args.bin_width))

    kept = [
        subch for subch in all_subchannels
        if pm.keep_by_era(subch, era, era) and pm.keep_by_channel(subch, channel)
    ]
    if not kept:
        fit_file.Close()
        raise RuntimeError(f"No sub-channels match {era}/{channel} for {masspoint}")

    sub_cfgs = {subch: all_cfgs[subch] for subch in kept}
    ordered_bkgs = ordered_backgrounds(sub_cfgs)

    result = {
        "masspoint": masspoint,
        "source_method": source_method,
        "ma": extract_mhc_ma(masspoint)[1],
        # Use the median sub-channel window as the summary ownership window.
        # A single bad/atypical sub-channel range would otherwise let one
        # masspoint claim unrelated gaps across the full mA axis.
        "mass_min": median(cfg["mass_min"] for cfg in sub_cfgs.values()),
        "mass_max": median(cfg["mass_max"] for cfg in sub_cfgs.values()),
        "ordered_bkgs": ordered_bkgs,
        "per_fit": {},
    }

    for fit_type in fit_types:
        if any(cfg.get("components") for cfg in sub_cfgs.values()):
            pre, post, pre_signal, post_signal, data, signal_available = build_grouped_process_aggregates(
                fit_file, kept, sub_cfgs, ordered_bkgs, fit_type,
                tuple(pm._GLOBAL_EDGES),
            )
        else:
            pre, post, pre_signal, post_signal, data, *_uncertainties = pm.build_process_aggregates_cached(
                fit_file, kept, sub_cfgs, ordered_bkgs, fit_type, tuple(pm._GLOBAL_EDGES)
            )
            signal_available = pre_signal.Integral() > 0
        result["per_fit"][fit_type] = {
            "pre_bkgs": pre,
            "post_bkgs": post,
            "pre_signal": pre_signal,
            "post_signal": post_signal,
            "signal_available": signal_available,
            "data": data,
        }

    fit_file.Close()
    if not args.plot_only:
        pm.save_cache_to_file(pm.CACHE_PATH)
    return result


def build_edges(results, bin_width):
    lo = math.floor(min(item["mass_min"] for item in results))
    hi = math.ceil(max(item["mass_max"] for item in results))
    if hi <= lo:
        hi = lo + bin_width
    n_bins = max(1, int(round((hi - lo) / bin_width)))
    return [lo + i * bin_width for i in range(n_bins + 1)]


def owner_index(x, results):
    candidates = []
    for idx, item in enumerate(results):
        if item["mass_min"] <= x <= item["mass_max"]:
            source_priority = SOURCE_METHOD_PRIORITY.get(item.get("source_method"), 99)
            candidates.append((source_priority, abs(x - item["ma"]), item["ma"], idx))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][3]


def stitch_histograms(hist_list, results, edges, name):
    out = ROOT.TH1D(name, "", len(edges) - 1, array("d", edges))
    out.SetDirectory(0)
    for ibin in range(1, out.GetNbinsX() + 1):
        x = out.GetBinCenter(ibin)
        idx = owner_index(x, results)
        if idx is None:
            continue
        src = hist_list[idx]
        if src is None:
            continue
        src_bin = src.FindBin(x)
        if src_bin < 1 or src_bin > src.GetNbinsX():
            continue
        out.SetBinContent(ibin, src.GetBinContent(src_bin))
        out.SetBinError(ibin, src.GetBinError(src_bin))
    return out


def collect_ownership(results, edges):
    owners = []
    for ibin in range(len(edges) - 1):
        center = 0.5 * (edges[ibin] + edges[ibin + 1])
        idx = owner_index(center, results)
        owners.append(None if idx is None else results[idx]["masspoint"])

    intervals = []
    if not owners:
        return intervals
    start = edges[0]
    current = owners[0]
    for idx, owner in enumerate(owners[1:], start=1):
        if owner == current:
            continue
        intervals.append({
            "start": start,
            "end": edges[idx],
            "masspoint": current,
        })
        start = edges[idx]
        current = owner
    intervals.append({
        "start": start,
        "end": edges[-1],
        "masspoint": current,
    })
    return intervals


def all_backgrounds(results):
    bkg_union = []
    for item in results:
        for bkg in item["ordered_bkgs"]:
            if bkg not in bkg_union:
                bkg_union.append(bkg)
    ordered = [b for b in pm.BKG_ORDER if b in bkg_union]
    ordered += [b for b in bkg_union if b not in ordered]
    return ordered


def build_stitched_content(results, fit_type, edges):
    ordered_bkgs = all_backgrounds(results)
    pre_bkgs = {}
    post_bkgs = {}
    for bkg in ordered_bkgs:
        pre_list = [item["per_fit"][fit_type]["pre_bkgs"].get(bkg) for item in results]
        post_list = [item["per_fit"][fit_type]["post_bkgs"].get(bkg) for item in results]
        if any(hist is not None for hist in pre_list):
            pre_bkgs[bkg] = stitch_histograms(pre_list, results, edges, f"{bkg}_pre")
        if any(hist is not None for hist in post_list):
            post_bkgs[bkg] = stitch_histograms(post_list, results, edges, f"{bkg}_post_{fit_type}")

    data = stitch_histograms(
        [item["per_fit"][fit_type]["data"] for item in results],
        results,
        edges,
        "data",
    )
    data.SetTitle("data")
    return pre_bkgs, post_bkgs, data


def median_signal_result(results):
    if not results:
        return None
    median_ma = median(item["ma"] for item in results)
    return min(results, key=lambda item: (abs(item["ma"] - median_ma), item["ma"]))


def signal_hist_for(results, fit_type, mode):
    if mode == "none":
        return None, None
    if mode != "median":
        raise ValueError(f"Unsupported signal line mode: {mode}")
    item = median_signal_result(results)
    if item is None:
        return None, None
    if not item["per_fit"][fit_type].get("signal_available", False):
        return None, None
    signal = item["per_fit"][fit_type].get("pre_signal")
    label = f"mA={item['ma']}GeV (r=1)" if signal else None
    return signal, label


def ownership_boundaries(intervals, x_range):
    boundaries = set()
    x_min, x_max = x_range
    for interval in intervals:
        if interval["masspoint"] is None:
            continue
        boundaries.add(float(interval["start"]))
        boundaries.add(float(interval["end"]))
    return sorted(x for x in boundaries if x_min < x < x_max)


def draw_ownership_guides(canvas, intervals, x_range):
    boundaries = ownership_boundaries(intervals, x_range)
    if not boundaries:
        return

    guides = []
    for pad_idx in (1, 2):
        pad = canvas.cd(pad_idx)
        y_min = pad.GetUymin()
        y_max = pad.GetUymax()
        for boundary in boundaries:
            line = ROOT.TLine(boundary, y_min, boundary, y_max)
            line.SetLineColor(ROOT.kGray + 2)
            line.SetLineStyle(ROOT.kDashed)
            line.SetLineWidth(1)
            line.Draw("same")
            guides.append(line)
    canvas._ownership_guides = guides


def summary_header_text(era_scope):
    if era_scope == "All":
        return "Run 2+3, 200 fb^{#minus1} (13/13.6 TeV)"
    return pm._build_header_text(era_scope)


def overdraw_summary_lumi_header(canvas, era_scope, channel_scope):
    if era_scope != "All" and channel_scope != "Combined":
        return

    upper = canvas.cd(1)
    pad_y_lo = upper.GetYlowNDC()
    pad_y_hi = pad_y_lo + upper.GetHNDC()
    pad_x_lo = upper.GetXlowNDC()
    pad_x_hi = pad_x_lo + upper.GetWNDC()
    top_margin = upper.GetTopMargin()
    right_margin = upper.GetRightMargin()

    def sub_to_canvas(x_sub, y_sub):
        return (pad_x_lo + x_sub * (pad_x_hi - pad_x_lo),
                pad_y_lo + y_sub * (pad_y_hi - pad_y_lo))

    text_x, text_y = sub_to_canvas(
        1.0 - right_margin,
        1.0 - top_margin + 0.2 * top_margin,
    )
    text_x += 0.002
    erase_x1, erase_y1 = sub_to_canvas(0.0, 1.0 - top_margin + 0.001)
    erase_x2, erase_y2 = sub_to_canvas(1.0, 1.0)

    canvas.cd()
    erase = ROOT.TPave(erase_x1, erase_y1, erase_x2, erase_y2, 0, "brNDC")
    erase.SetFillColor(ROOT.kWhite)
    erase.SetBorderSize(0)
    erase.Draw()
    canvas._summary_lumi_erase = erase

    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextAlign(31)
    latex.SetTextSize(0.6 * top_margin * (pad_y_hi - pad_y_lo))
    latex.DrawLatex(text_x, text_y, summary_header_text(era_scope))
    canvas._summary_lumi_latex = latex


def apply_summary_axis_offsets(canvas, is_wide):
    if not is_wide:
        return

    upper_frame = pm.CMS.GetCmsCanvasHist(canvas.cd(1))
    upper_frame.GetYaxis().SetTitleOffset(0.54)
    upper_frame.GetYaxis().CenterTitle(False)
    canvas.cd(1).RedrawAxis()

    lower_frame = pm.CMS.GetCmsCanvasHist(canvas.cd(2))
    lower_frame.GetYaxis().SetTitleOffset(0.26)
    lower_frame.GetYaxis().CenterTitle(False)
    canvas.cd(2).RedrawAxis()


def draw_stack(data, backgrounds, signal, signal_label, label_top, out_base,
               args, era, channel, method, mhc, edges, intervals):
    if not backgrounds:
        logging.warning(f"No backgrounds; skipping {out_base}")
        return

    is_wide = mhc in args.wide_mhc and args.wide_factor > 1.0
    x_range = [12.0, float(mhc)]
    colors = [pm.BKG_COLORS.get(bkg, ROOT.kGray) for bkg in backgrounds.keys()]
    stack_total = next(iter(backgrounds.values())).Clone("_summary_stack_total")
    stack_total.SetDirectory(0)
    for hist in list(backgrounds.values())[1:]:
        stack_total.Add(hist)
    y_max = max(
        stack_total.GetMaximum(),
        data.GetMaximum() if data.Integral() > 0 else 0.0,
        signal.GetMaximum() if signal is not None and signal.Integral() > 0 else 0.0,
    ) * 2.0
    config = pm.make_canvas_config(era, {
        "xTitle": "M(#mu^{+}#mu^{-}) [GeV]",
        "yTitle": f"Events / {args.bin_width:g} GeV",
        "xRange": x_range,
        "yRange": [0, y_max],
        "rTitle": "Data / Pred",
        "rRange": [0, 5],
        "maxDigits": 3,
        "systSrc": "Stat+Syst",
        "iPos": 11,
        "legend": [0.5, 0.62, 0.99, 0.89],
        "legendColumns": 2,
        "legendTextSize": 0.035,
        "colors": colors,
    })
    plotter = pm.select_comparison_cls(era)(data, backgrounds, config)
    if is_wide:
        width = int(plotter.canv.GetWindowWidth() * args.wide_factor)
        height = plotter.canv.GetWindowHeight()
        plotter.canv.SetCanvasSize(width, height)
        plotter.canv.SetWindowSize(width, height)
        plotter.canv.Modified()
        plotter.canv.Update()
        plotter.leg.SetX1NDC(0.66)
        plotter.leg.SetX2NDC(0.99)
        plotter.leg.SetY1NDC(0.62)
        plotter.leg.SetY2NDC(0.89)
        if hasattr(plotter.leg, "SetColumnSeparation"):
            plotter.leg.SetColumnSeparation(0.02)
    plotter.drawPadUp()
    if signal is not None:
        plotter.canv.cd(1)
        signal.SetLineColor(ROOT.kBlack)
        signal.SetLineWidth(2)
        signal.SetLineStyle(1)
        if signal.Integral() > 0:
            signal.Draw("HIST SAME")
        for obj in ROOT.gPad.GetListOfPrimitives():
            if obj.InheritsFrom("TLegend"):
                obj.AddEntry(signal, signal_label or "Signal", "l")
                break
    plotter.drawPadDown()
    draw_ownership_guides(plotter.canv, intervals, x_range)

    plotter.canv.cd()
    pm.CMS.drawText(pm.scope_label(channel), posX=0.2, posY=0.80,
                    font=42, align=0, size=0.04)
    mass_label = f"m_{{H^{{+}}}} = {mhc} GeV"
    if is_wide:
        mass_label += f", {method}"
    pm.CMS.drawText(mass_label,
                    posX=0.2, posY=0.76, font=42, align=0, size=0.035)
    pm.CMS.drawText(label_top, posX=0.2, posY=0.72,
                    font=62, align=0, size=0.032)
    apply_summary_axis_offsets(plotter.canv, is_wide)
    overdraw_summary_lumi_header(plotter.canv, era, channel)

    for ext in ("png", "pdf"):
        output = f"{out_base}.{ext}"
        plotter.canv.SaveAs(output)
        logging.info(f"Saved: {output}")


def write_sidecar(path, args, era, channel, method, mhc, results, edges):
    intervals = collect_ownership(results, edges)
    x_range = [12.0, float(mhc)]
    signal_result = median_signal_result(results) if args.signal_line == "median" else None
    signal_available = (
        any(per_fit.get("signal_available", False)
            for per_fit in signal_result["per_fit"].values())
        if signal_result else False
    )
    payload = {
        "mhc": mhc,
        "method": method,
        "era": era,
        "channel": channel,
        "bin_width": args.bin_width,
        "binning": args.binning,
        "suffix": template_suffix(args, method),
        "fit_type": args.fit_type,
        "signal_line": args.signal_line,
        "signal_line_masspoint": signal_result["masspoint"] if signal_result else None,
        "signal_line_available": signal_available,
        "masspoints": [
            {
                "name": item["masspoint"],
                "source_method": item["source_method"],
                "ma": item["ma"],
                "mass_min": item["mass_min"],
                "mass_max": item["mass_max"],
            }
            for item in results
        ],
        "x_range": x_range,
        "hist_range": [edges[0], edges[-1]],
        "n_bins": len(edges) - 1,
        "ownership": intervals,
        "ownership_boundaries": ownership_boundaries(intervals, x_range),
    }
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2)
    logging.info(f"Wrote: {path}")


def process_target(args, era, channel, method, mhc, fit_types):
    masspoint_sources = discover_masspoint_sources(args, era, method, mhc)
    if not masspoint_sources:
        logging.warning(f"No fitDiagnostics found for mHc={mhc}, {era}, {channel}, {method}")
        return

    logging.info(
        "%s/%s/%s/mHc%s: %s",
        era,
        channel,
        method,
        mhc,
        [
            f"{item['masspoint']}({item['source_method']})"
            for item in masspoint_sources
        ],
    )
    results = []
    for item in masspoint_sources:
        results.append(
            load_one_masspoint(
                args,
                era,
                item["source_method"],
                channel,
                item["masspoint"],
                fit_types,
            )
        )

    edges = build_edges(results, args.bin_width)
    intervals = collect_ownership(results, edges)
    output_dir = Path(args.output_dir) / f"mHc{mhc}" / method
    os.makedirs(output_dir, exist_ok=True)
    tag = blinding_tag(args)
    prefix = output_dir / f"postfit_summary.mHc{mhc}.{era}.{channel}.{method}"

    prefit_done = False
    for fit_type in fit_types:
        pre_bkgs, post_bkgs, data = build_stitched_content(results, fit_type, edges)
        if not prefit_done:
            signal, signal_label = signal_hist_for(results, fit_type, args.signal_line)
            draw_stack(data, pre_bkgs, signal, signal_label, "Pre-fit",
                       f"{prefix}.prefit.{tag}", args, era, channel, method, mhc,
                       edges, intervals)
            prefit_done = True

        fit_label = "B-only" if fit_type == "b" else "S+B"
        signal, signal_label = signal_hist_for(results, fit_type, args.signal_line)
        draw_stack(data, post_bkgs, signal, signal_label, f"Post-fit {fit_label}",
                   f"{prefix}.postfit_{fit_type}.{tag}", args, era, channel, method, mhc,
                   edges, intervals)

    write_sidecar(f"{prefix}.{tag}.json", args, era, channel, method, mhc, results, edges)


def main():
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s - %(message)s")
    ROOT.gROOT.SetBatch(True)
    ROOT.gStyle.SetOptStat(0)

    fit_types = fit_types_from_arg(args.fit_type)
    for mhc in args.mhc:
        for method in args.methods:
            for era in args.eras:
                for channel in args.channels:
                    process_target(args, era, channel, method, mhc, fit_types)


if __name__ == "__main__":
    main()
