#!/usr/bin/env python3
"""
Stitch multiple masspoints' 1-GeV post-fit mass plots into a single plot
whose x-axis is partitioned into per-masspoint slices.

Example — partial-unblind test case with 3 ParticleNet masspoints:

    MHc160_MA85    : bins [75, 87]
    MHc130_MA90    : bins [87, 93]
    MHc100_MA95    : bins [93, 105]

Each 1-GeV fine bin in the combined plot pulls its content (per process)
from the owning masspoint's cached fine-mass hists produced by
``plotPostfitMass.py``. Because every cache is on integer-aligned 1-GeV
bins, no re-binning is required — only a bin-wise source selector.

Usage:
    python3 plotCombinedMass.py --era All --method ParticleNet \\
        --binning extended --masspoint-set partial_unblind \\
        --partial-unblind --fit-type both --plot-only
"""
import os
import sys
import json
import math
import logging
import argparse
from array import array
from types import SimpleNamespace

import ROOT

# Make plotPostfitMass importable and reuse all its helpers.
import plotPostfitMass as pm


def parse_args():
    p = argparse.ArgumentParser(description="Combined-masspoints real-mass plot")
    p.add_argument("--era", required=True, type=str,
                   help="Fit source (All/Run2/Run3/per-era). Each masspoint "
                        "uses its own fitDiagnostics file at this era level.")
    p.add_argument("--method", required=True, choices=["Baseline", "ParticleNet"])
    p.add_argument("--binning", default="extended",
                   choices=["extended", "uniform"])
    src = p.add_mutually_exclusive_group(required=False)
    src.add_argument("--masspoint-set", dest="masspoint_set",
                     help="Key from configs/masspoints.json (e.g. partial_unblind, "
                          "particlenet, baseline, limits). Defaults to 'limits' "
                          "for --blind and must be specified otherwise.")
    src.add_argument("--masspoints", dest="masspoints_csv",
                     help="Comma-separated list of masspoints.")
    p.add_argument("--fit-type", default="both", choices=["b", "s", "both"])
    p.add_argument("--era-scope", dest="era_scope", default="All")
    p.add_argument("--channel-scope", dest="channel_scope", default="Combined",
                   choices=["SR1E2Mu", "SR3Mu", "Combined"])
    p.add_argument("--unblind", action="store_true")
    p.add_argument("--partial-unblind", action="store_true", dest="partial_unblind")
    p.add_argument("--blind", action="store_true",
                   help="Asimov mode: data = sum of pre-fit backgrounds. "
                        "Default masspoint set is 'limits' (28 distinct mA values).")
    p.add_argument("--bin-width", default=1.0, type=float)
    p.add_argument("--slice-boundaries", dest="slice_boundaries_csv", default=None,
                   help="Explicit slice boundaries in GeV (comma-separated). "
                        "Default: int(round((mA_i + mA_{i+1}) / 2)).")
    p.add_argument("--x-range", dest="x_range_csv", default=None,
                   help="Global x-axis range in GeV, 'xmin,xmax'. "
                        "Default: union of masspoint mass windows.")
    p.add_argument("--plot-only", action="store_true", dest="plot_only",
                   help="Require per-masspoint caches to exist; no tree reads.")
    p.add_argument("--signal-masspoints", dest="signal_masspoints_csv", default=None,
                   help="Comma-separated masspoint subset whose signal curves "
                        "should be drawn. Default for --blind: ~6 masspoints "
                        "auto-spread across the sorted mA range.")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


# Colors for per-masspoint signal curves (black reserved for the default
# plotPostfitMass signal line). Use a distinct palette so 6 overlaid signal
# curves remain visually distinguishable.
SIGNAL_PALETTE = [
    ROOT.kBlack,
    ROOT.kAzure + 2,
    ROOT.kGreen + 2,
    ROOT.kOrange + 7,
    ROOT.kViolet + 1,
    ROOT.kTeal + 3,
    ROOT.kRed + 1,
    ROOT.kGray + 2,
]


# =============================================================================
# Masspoint list resolution & slice geometry
# =============================================================================

def _dedupe_by_mA(masspoints):
    """Keep the first masspoint encountered for each mA value (lowest MHc
    when the list is sorted by MHc, which is the natural order in
    configs/masspoints.json). Logs a warning for each drop."""
    seen = {}
    kept = []
    dropped = []
    for m in masspoints:
        ma = extract_mA(m)
        if ma in seen:
            dropped.append((m, seen[ma]))
            continue
        seen[ma] = m
        kept.append(m)
    if dropped:
        for d, k in dropped:
            logging.warning(f"Dropping {d} (duplicate mA={extract_mA(d)}, kept {k})")
    return kept


def resolve_masspoints(args):
    if args.masspoints_csv is not None:
        return [m.strip() for m in args.masspoints_csv.split(",") if m.strip()]

    key = args.masspoint_set
    if key is None:
        if args.blind:
            key = "limits"
            logging.info("--blind given without --masspoint-set; defaulting to 'limits'")
        else:
            raise ValueError("--masspoint-set or --masspoints is required "
                             "(or use --blind for the 'limits' default)")

    path = f"{pm.WORKDIR}/SignalRegionStudyV3/configs/masspoints.json"
    with open(path, "r") as f:
        d = json.load(f)
    if key not in d:
        raise KeyError(f"--masspoint-set '{key}' not found in {path}. "
                       f"Available: {sorted(d.keys())}")
    val = d[key]
    if isinstance(val, dict):
        # nested (e.g. 'impact' -> {'baseline': [...], 'particlenet': [...]})
        sub = args.method.lower()
        if sub not in val:
            raise KeyError(f"nested set '{key}' has no '{sub}' entry")
        val = val[sub]
    return _dedupe_by_mA(list(val))


def extract_mA(masspoint):
    """'MHc130_MA90' -> 90."""
    return int(masspoint.split("_")[1][2:])


# Default signal overlays for the blind combined plot. Must all exist in
# the `limits` masspoint set (the one loaded by default under --blind).
DEFAULT_BLIND_SIGNAL_MPS = [
    "MHc85_MA15",
    "MHc130_MA55",
    "MHc100_MA95",
    "MHc145_MA140",
]


def resolve_blind_signal_subset(args, sorted_mps):
    """Return the list of masspoints whose signal curve should overlay the
    blind combined plot. Falls back to DEFAULT_BLIND_SIGNAL_MPS when
    `--signal-masspoints` isn't given. Drops entries that aren't in the
    loaded set (with a warning) so the caller never trips on a KeyError.
    """
    if args.signal_masspoints_csv is not None:
        requested = [m.strip() for m in args.signal_masspoints_csv.split(",") if m.strip()]
    else:
        requested = list(DEFAULT_BLIND_SIGNAL_MPS)
    available = set(sorted_mps)
    out = []
    for m in requested:
        if m in available:
            out.append(m)
        else:
            logging.warning(f"Requested signal masspoint {m} not in loaded set; skipping")
    return out


def default_slice_boundaries(sorted_masspoints):
    """Midpoint between consecutive mA values, rounded away from the centre
    of the masspoint list.

    For the partial-unblind case (mA = [85, 90, 95]) this produces [87, 93]:
    the left boundary (between 85 and 90, midpoint 87.5) rounds DOWN to 87,
    and the right boundary (between 90 and 95, midpoint 92.5) rounds UP to 93.
    Effect: edge masspoints get a bit more of the overlap region than the
    middle masspoint, which matches the test-plot expectation.
    """
    mAs = [extract_mA(m) for m in sorted_masspoints]
    mid_idx = (len(mAs) - 1) / 2.0
    bounds = []
    for i in range(len(mAs) - 1):
        mp = (mAs[i] + mAs[i + 1]) / 2.0
        if i < mid_idx:
            bounds.append(int(math.floor(mp)))
        else:
            bounds.append(int(math.ceil(mp)))
    return bounds


def owning_masspoint_index(bin_center, slice_boundaries, n_masspoints):
    """Return the index of the masspoint owning the slice containing bin_center.

    Slice i covers [b_{i-1}, b_i) with b_{-1} = -inf and b_{n-1} = +inf.
    """
    for i, b in enumerate(slice_boundaries):
        if bin_center < b:
            return i
    return n_masspoints - 1


# =============================================================================
# Per-masspoint loader (reuses plotPostfitMass helpers)
# =============================================================================

def load_one_masspoint(masspoint, base_args, fit_types):
    """Populate plotPostfitMass state for `masspoint`, open fitdiag, build
    per-process aggregates for each fit type. Returns a dict.
    """
    mp_args = SimpleNamespace(**vars(base_args))
    mp_args.masspoint = masspoint
    mp_args.fit_type = "both"
    pm.entry_setup(mp_args, require_fitdiag=True, make_output_dir=False)
    pm._FINE_CACHE.clear()
    pm.CACHE_PATH = f"{pm.CACHE_DIR}/mass_hists_v2_bw{base_args.bin_width:g}.root"

    if base_args.plot_only:
        if not pm.load_cache_from_file(pm.CACHE_PATH):
            raise FileNotFoundError(
                f"--plot-only requires a cache at {pm.CACHE_PATH}. "
                f"Run ./automize/plotPostfitMass.sh first.")

    f = ROOT.TFile.Open(pm.FITDIAG_PATH, "READ")
    if not f or f.IsZombie():
        raise RuntimeError(f"Failed to open {pm.FITDIAG_PATH}")

    all_subchannels = pm.discover_channels(f)
    all_cfgs = {}
    for sc in all_subchannels:
        era_i, ch_i = pm.parse_subchannel(sc, mp_args.era)
        all_cfgs[sc] = pm.load_subchannel_config(era_i, ch_i)

    global_lo = min(cfg["mass_min"] for cfg in all_cfgs.values())
    global_hi = max(cfg["mass_max"] for cfg in all_cfgs.values())
    pm.set_global_edges(pm.build_uniform_edges(global_lo, global_hi, base_args.bin_width))

    kept = [sc for sc in all_subchannels
            if pm.keep_by_era(sc, base_args.era_scope, mp_args.era)
            and pm.keep_by_channel(sc, base_args.channel_scope)]
    if not kept:
        raise RuntimeError(
            f"No sub-channels match era={base_args.era_scope} "
            f"channel={base_args.channel_scope} for {masspoint}")

    sub_cfgs = {sc: all_cfgs[sc] for sc in kept}
    bkg_union = []
    for sc in kept:
        for bkg in sub_cfgs[sc]["separate_processes"]:
            if bkg in pm.BKG_ORDER and bkg not in bkg_union:
                bkg_union.append(bkg)
    if "others" not in bkg_union:
        bkg_union.append("others")
    ordered_bkgs = [b for b in pm.BKG_ORDER if b in bkg_union]

    mp_result = {
        "global_lo": global_lo,
        "global_hi": global_hi,
        "ordered_bkgs": ordered_bkgs,
        "per_ft": {},
    }

    for ft in fit_types:
        pre, post, pre_sig, post_sig, data = pm.build_process_aggregates_cached(
            f, kept, sub_cfgs, ordered_bkgs, ft, tuple(pm._GLOBAL_EDGES))
        mp_result["per_ft"][ft] = {
            "pre_bkgs": pre,
            "post_bkgs": post,
            "pre_signal": pre_sig,
            "post_signal": post_sig,
            "data": data,
        }
    f.Close()
    return mp_result


# =============================================================================
# Stitching
# =============================================================================

def stitch_histogram(mp_hists_by_mp, slice_boundaries, combined_edges, name):
    """For each bin in `combined_edges`, copy from the owning masspoint's hist.

    mp_hists_by_mp: list of TH1D (one per masspoint, same order as sorted_mps).
    """
    edges_arr = array('d', combined_edges)
    result = ROOT.TH1D(name, "", len(combined_edges) - 1, edges_arr)
    result.SetDirectory(0)
    n_mps = len(mp_hists_by_mp)
    for j in range(1, result.GetNbinsX() + 1):
        c = result.GetBinCenter(j)
        idx = owning_masspoint_index(c, slice_boundaries, n_mps)
        src = mp_hists_by_mp[idx]
        if src is None:
            continue
        src_bin = src.FindBin(c)
        if src_bin < 1 or src_bin > src.GetNbinsX():
            continue
        result.SetBinContent(j, src.GetBinContent(src_bin))
        result.SetBinError(j, src.GetBinError(src_bin))
    return result


def stitch_histogram_max(mp_hists_by_mp, mp_windows_by_mp, combined_edges, name):
    """For each bin, pick the masspoint with the highest content among those
    whose mass window covers the bin center. Used for blind-mode backgrounds
    where neighboring masspoints may disagree in overlapping bins (different
    preselection statistics / fine-bin edges) — taking the max hides dips
    caused by sparse low-stat bins in a single masspoint's cache.

    mp_hists_by_mp:   list of TH1D (same order as sorted_mps; may contain None).
    mp_windows_by_mp: list of (lo, hi) mass-window tuples per masspoint.
    """
    edges_arr = array('d', combined_edges)
    result = ROOT.TH1D(name, "", len(combined_edges) - 1, edges_arr)
    result.SetDirectory(0)
    for j in range(1, result.GetNbinsX() + 1):
        c = result.GetBinCenter(j)
        best_val = None
        best_err = 0.0
        for h, (lo, hi) in zip(mp_hists_by_mp, mp_windows_by_mp):
            if h is None or c < lo or c > hi:
                continue
            src_bin = h.FindBin(c)
            if src_bin < 1 or src_bin > h.GetNbinsX():
                continue
            val = h.GetBinContent(src_bin)
            if best_val is None or val > best_val:
                best_val = val
                best_err = h.GetBinError(src_bin)
        if best_val is not None:
            result.SetBinContent(j, best_val)
            result.SetBinError(j, best_err)
    return result


def per_mp_signal_in_slice(mp, mp_hist, mp_index, slice_boundaries,
                           combined_edges, name):
    """Return a TH1D on `combined_edges` holding `mp_hist` content ONLY inside
    the slice owned by masspoint at `mp_index`. Bins outside that slice are 0.

    Used to draw per-masspoint signal curves that only appear within each
    masspoint's assigned slice (not stitched across the full range).
    """
    edges_arr = array('d', combined_edges)
    result = ROOT.TH1D(name, "", len(combined_edges) - 1, edges_arr)
    result.SetDirectory(0)
    if mp_hist is None:
        return result
    n_total = len(slice_boundaries) + 1
    for j in range(1, result.GetNbinsX() + 1):
        c = result.GetBinCenter(j)
        if owning_masspoint_index(c, slice_boundaries, n_total) != mp_index:
            continue
        src_bin = mp_hist.FindBin(c)
        if src_bin < 1 or src_bin > mp_hist.GetNbinsX():
            continue
        result.SetBinContent(j, mp_hist.GetBinContent(src_bin))
        result.SetBinError(j, mp_hist.GetBinError(src_bin))
    return result


def build_combined_edges(sorted_mps, mp_results, args):
    if args.x_range_csv is not None:
        lo_s, hi_s = args.x_range_csv.split(",")
        lo, hi = float(lo_s), float(hi_s)
    else:
        lo = min(mp_results[m]["global_lo"] for m in sorted_mps)
        hi = max(mp_results[m]["global_hi"] for m in sorted_mps)
    return pm.build_uniform_edges(lo, hi, args.bin_width)


# =============================================================================
# Drawing
# =============================================================================

def _output_dir(base_args):
    suffix = pm._compute_binning_suffix(SimpleNamespace(
        binning=base_args.binning,
        unblind=base_args.unblind,
        partial_unblind=base_args.partial_unblind,
        blind=base_args.blind,
    ))
    return (f"{pm.WORKDIR}/SignalRegionStudyV3/results/plots/templates/"
            f"{base_args.era}/Combined/_combined_{base_args.method}_{suffix}/plots_mass")


def draw_combined_stack(agg_data, agg_bkgs, agg_signal, label_top, systSrc,
                        out_path, xrange, slice_boundaries, base_args,
                        show_signal=True):
    """Combined-masspoints stack plot. Same text formatting as
    `plotPostfitMass._make_stack` with the masspoint line dropped.

    `show_signal=False` skips both the signal curve and its legend entry —
    used on the post-fit stacks in blind mode, where the stitched 28-mA
    signal curve would be visually misleading.
    """
    if not agg_bkgs:
        logging.warning(f"No backgrounds; skipping {out_path}")
        return
    agg_data.SetTitle("data")
    colors = [pm.BKG_COLORS.get(b, ROOT.kGray) for b in agg_bkgs.keys()]
    config = pm.make_canvas_config(base_args.era_scope, {
        # Omitting "channel" suppresses the masspoint label (meaningless
        # across stitched masspoints).
        "xTitle": "M(#mu^{+}#mu^{-}) [GeV]",
        "yTitle": f"Events / {base_args.bin_width:g} GeV",
        "xRange": list(xrange),
        "rTitle": "Data / Pred",
        "rRange": [0, 2.5],
        "maxDigits": 3,
        "systSrc": systSrc,
        "colors": colors,
    })
    plotter = pm.select_comparison_cls(base_args.era_scope)(agg_data, agg_bkgs, config)
    plotter.drawPadUp()

    if show_signal and agg_signal is not None:
        plotter.canv.cd(1)
        agg_signal.SetLineColor(ROOT.kBlack)
        agg_signal.SetLineWidth(2)
        if agg_signal.Integral() > 0:
            agg_signal.Draw("HIST SAME")
        for obj in ROOT.gPad.GetListOfPrimitives():
            if obj.InheritsFrom("TLegend"):
                obj.AddEntry(agg_signal, "Signal", "l")
                break

    # Dashed vertical lines at slice boundaries
    plotter.canv.cd(1)
    ymin = 0
    ymax = ROOT.gPad.GetUymax()
    ln = ROOT.TLine()
    ln.SetLineStyle(ROOT.kDashed)
    ln.SetLineColor(ROOT.kGray + 2)
    ln.SetLineWidth(1)
    for b in slice_boundaries:
        ln.DrawLine(b, ymin, b, ymax)

    plotter.drawPadDown()
    plotter.canv.cd()
    pm.CMS.drawText(pm.scope_label(base_args.channel_scope),
                    posX=0.2, posY=0.8, font=42, align=0, size=0.04)
    blind = pm._blinding_label()
    line = label_top + (f" ({blind})" if blind else "")
    pm.CMS.drawText(line, posX=0.2, posY=0.76, font=62, align=0, size=0.032)

    # Emulate the All-era lumi-header overdraw regardless of scope
    pm._overdraw_lumi_header(plotter.canv, pm.PlotTarget(
        era_scope=base_args.era_scope,
        channel_scope=base_args.channel_scope,
        xrange=tuple(xrange),
    ))

    plotter.canv.SaveAs(out_path)
    logging.info(f"Saved: {out_path}")
    sig_int = agg_signal.Integral() if agg_signal else 0.0
    logging.info(f"  total_bkg: {sum(h.Integral() for h in agg_bkgs.values()):.2f}  "
                 f"data: {agg_data.Integral():.0f}  signal: {sig_int:.2f}")


# =============================================================================
# Blind-mode drawing (single wide pad, no data, no ratio, per-mp signal overlays)
# =============================================================================

def draw_combined_blind_stack(agg_bkgs, signal_per_mp, label_top, systSrc,
                              out_path, xrange, base_args):
    """Blind-mode combined stack: single wide pad, no data, no ratio.

    Parameters
    ----------
    agg_bkgs : dict[str, TH1D]
        Stitched per-process pre- or post-fit background hists.
    signal_per_mp : list[tuple[str, TH1D]]
        List of (masspoint_label, signal_hist) pairs to overlay as
        differently-coloured lines. Usually ~6 curated entries.
    label_top : str
        Bold fit-label text (e.g. "Pre-fit", "Post-fit B-only").
    systSrc : str
        Legend label for the background systematic-error entry.
    out_path : str
        Output PNG path.
    xrange : tuple[float, float]
        (x_min, x_max) in GeV.
    base_args : argparse.Namespace
    """
    if not agg_bkgs:
        logging.warning(f"No backgrounds; skipping {out_path}")
        return

    # ComparisonCanvas hardwires a data+ratio pad and cmsCanvas's internal
    # SetCanvasSize stretches the lumi TLatex (fb^{-1} spacing quirk). Build
    # a plain TCanvas and style it by hand.
    pm.CMS.setCMSStyle()
    run2_lumi = pm._LUMI_CONFIG["Run2"]["combined"]
    run3_lumi = pm._LUMI_CONFIG["Run3"]["combined"]
    run2_energy = pm._LUMI_CONFIG["Run2"]["energy_TeV"]
    run3_energy = pm._LUMI_CONFIG["Run3"]["energy_TeV"]
    pm.CMS.SetLumi(None, run=f"Run 2+3, {run2_lumi}+{run3_lumi} fb^{{#minus1}}")
    pm.CMS.SetEnergy(0, unit=f"{run2_energy:g}/{run3_energy:g} TeV")
    pm.CMS.SetExtraText("Preliminary")

    # Build the summed background hist for the shaded syst band + y-max probe.
    first = agg_bkgs[list(agg_bkgs.keys())[0]]
    n_bins = first.GetNbinsX()
    edges = array('d', [first.GetBinLowEdge(i) for i in range(1, n_bins + 2)])
    total = ROOT.TH1D(f"_blind_total_{id(agg_bkgs)}", "", n_bins, edges)
    total.SetDirectory(0)
    for h in agg_bkgs.values():
        total.Add(h)
    ymax = total.GetMaximum() * 1.6

    # Plain TCanvas at 1200 × 600, CMS-style margins.
    canv = ROOT.TCanvas(f"combined_blind_{id(agg_bkgs)}", "", 1200, 600)
    canv.SetFillColor(0)
    canv.SetBorderMode(0)
    canv.SetFrameFillStyle(0)
    canv.SetFrameBorderMode(0)
    canv.SetFrameLineColor(0)
    canv.SetFrameLineWidth(0)
    canv.SetLeftMargin(0.07)
    canv.SetRightMargin(0.05)
    canv.SetTopMargin(0.09)
    canv.SetBottomMargin(0.13)

    frame = canv.DrawFrame(xrange[0], 0.0, xrange[1], ymax)
    frame.GetXaxis().SetTitle("M(#mu^{+}#mu^{-}) [GeV]")
    frame.GetYaxis().SetTitle(f"Events / {base_args.bin_width:g} GeV")
    frame.GetYaxis().SetTitleOffset(0.6)
    frame.GetXaxis().SetTitleOffset(1.0)
    frame.GetYaxis().SetMaxDigits(3)
    frame.Draw("AXIS")

    canv.cd()
    # Build ordered list of background TH1Ds in BKG_ORDER for consistent colours
    ordered = [b for b in pm.BKG_ORDER if b in agg_bkgs]
    ordered += [b for b in agg_bkgs if b not in ordered]
    stack_hists = [agg_bkgs[b] for b in ordered]
    palette = [pm.BKG_COLORS.get(b, ROOT.kGray) for b in ordered]

    hs = pm.CMS.buildTHStack(stack_hists, palette, LineColor=-1, FillColor=-1)
    pm.CMS.cmsObjectDraw(hs, "hist")

    # Syst band (hatched) on total bkg
    pm.CMS.cmsObjectDraw(total, "FE2", FillStyle=3004, LineWidth=0,
                         FillColor=12, MarkerSize=0)

    # Per-masspoint signal overlays (curated subset)
    signal_entries = []
    for idx, (mp, h) in enumerate(signal_per_mp):
        if h is None or h.Integral() <= 0:
            continue
        col = SIGNAL_PALETTE[idx % len(SIGNAL_PALETTE)]
        h.SetLineColor(col)
        h.SetLineWidth(2)
        h.SetLineStyle(1)
        h.Draw("HIST SAME")
        signal_entries.append((h, mp, col))

    # Two-column legend: backgrounds + syst on the left, signals on the right.
    n_bkg = len(ordered) + 1  # +1 for syst band
    bkg_leg = pm.CMS.cmsLeg(0.69, max(0.45, 0.88 - 0.035 * n_bkg),
                            0.82, 0.88, textSize=0.028)
    for b in reversed(ordered):
        pm.CMS.addToLegend(bkg_leg, (agg_bkgs[b], b, "F"))
    pm.CMS.addToLegend(bkg_leg, (total, systSrc, " FE2"))

    if signal_entries:
        n_sig = len(signal_entries)
        sig_leg = pm.CMS.cmsLeg(0.83, max(0.65, 0.88 - 0.035 * n_sig),
                                0.98, 0.88, textSize=0.028)
        for h, mp, _ in signal_entries:
            pm.CMS.addToLegend(sig_leg, (h, pm.masspoint_label(mp), "l"))

    canv.RedrawAxis()

    # CMS logo + "Preliminary" + lumi header — call cmsstyle's helper directly
    # on the pre-sized canvas (no SetCanvasSize afterwards, so no fb^{-1} gap).
    pm.CMS.CMS_lumi(canv, 11)

    # Label block (top-left, below "Preliminary"): channel scope + fit-state
    # line. Same y positions as plotPostfitMass._make_stack.
    pm.CMS.drawText(pm.scope_label(base_args.channel_scope),
                    posX=0.2, posY=0.78, font=42, align=0, size=0.04)
    pm.CMS.drawText(label_top, posX=0.2, posY=0.72, font=62, align=0, size=0.035)

    canv.SaveAs(out_path)
    logging.info(f"Saved: {out_path}")
    logging.info(f"  total_bkg: {total.Integral():.2f}  "
                 f"signal curves: {len(signal_entries)}")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()
    chosen = sum(bool(x) for x in (args.unblind, args.partial_unblind, args.blind))
    if chosen != 1:
        raise ValueError("Exactly one of --blind / --partial-unblind / --unblind is required")

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(levelname)s - %(message)s')

    masspoints = resolve_masspoints(args)
    if len(masspoints) < 2:
        raise ValueError(f"Need at least 2 masspoints to combine; got {masspoints}")
    sorted_mps = sorted(masspoints, key=extract_mA)
    logging.info(f"Masspoints (sorted by mA): {sorted_mps}")

    fit_types = ["b", "s"] if args.fit_type == "both" else [args.fit_type]

    # Per-masspoint load
    mp_results = {}
    for mp in sorted_mps:
        logging.info(f"Loading {mp} ...")
        mp_results[mp] = load_one_masspoint(mp, args, fit_types)
        logging.info(f"  mass window [{mp_results[mp]['global_lo']:.2f}, "
                     f"{mp_results[mp]['global_hi']:.2f}] GeV, "
                     f"bkgs: {mp_results[mp]['ordered_bkgs']}")

    # Slice boundaries
    if args.slice_boundaries_csv is not None:
        slice_boundaries = [float(b) for b in args.slice_boundaries_csv.split(",")]
    else:
        slice_boundaries = [float(b) for b in default_slice_boundaries(sorted_mps)]
    logging.info(f"Slice boundaries: {slice_boundaries}")

    # Signal subset for the blind-mode wide plot (curated overlay).
    if args.blind:
        signal_subset = resolve_blind_signal_subset(args, sorted_mps)
        logging.info(f"Signal overlay masspoints: {signal_subset}")
    else:
        signal_subset = []

    # Combined grid
    combined_edges = build_combined_edges(sorted_mps, mp_results, args)
    xrange = (combined_edges[0], combined_edges[-1])
    logging.info(f"Combined grid: [{xrange[0]}, {xrange[1]}] "
                 f"({len(combined_edges) - 1} bins)")

    # Background process union across all masspoints
    bkg_union = []
    for mp in sorted_mps:
        for bkg in mp_results[mp]["ordered_bkgs"]:
            if bkg not in bkg_union:
                bkg_union.append(bkg)
    ordered_bkgs = [b for b in pm.BKG_ORDER if b in bkg_union]
    if "others" in bkg_union and "others" not in ordered_bkgs:
        ordered_bkgs.append("others")

    out_dir = _output_dir(args)
    os.makedirs(out_dir, exist_ok=True)
    logging.info(f"Output directory: {out_dir}")

    mp_windows = [(mp_results[mp]["global_lo"], mp_results[mp]["global_hi"])
                  for mp in sorted_mps]

    def stitch_bkg(hist_list, name):
        if args.blind:
            return stitch_histogram_max(hist_list, mp_windows, combined_edges, name)
        return stitch_histogram(hist_list, slice_boundaries, combined_edges, name)

    # Prefit content is fit-type-independent; build combined_pre_bkgs once.
    ft0 = fit_types[0]
    combined_pre_bkgs = {}
    for bkg in ordered_bkgs:
        pre_list = [mp_results[mp]["per_ft"][ft0]["pre_bkgs"].get(bkg) for mp in sorted_mps]
        if any(h is not None for h in pre_list):
            combined_pre_bkgs[bkg] = stitch_bkg(pre_list, f"{bkg}_pre_combined")

    # Signals and data are used only by the non-blind render path.
    if not args.blind:
        combined_pre_signal = stitch_histogram(
            [mp_results[mp]["per_ft"][ft0]["pre_signal"] for mp in sorted_mps],
            slice_boundaries, combined_edges, "signal_pre_combined")
        combined_data = stitch_histogram(
            [mp_results[mp]["per_ft"][ft0]["data"] for mp in sorted_mps],
            slice_boundaries, combined_edges, "data_combined")
        combined_data.SetTitle("data")

    # Stitch postfit & render per fit type
    for ft in fit_types:
        combined_post_bkgs = {}
        for bkg in ordered_bkgs:
            post_list = [mp_results[mp]["per_ft"][ft]["post_bkgs"].get(bkg) for mp in sorted_mps]
            if any(h is not None for h in post_list):
                combined_post_bkgs[bkg] = stitch_bkg(post_list, f"{bkg}_post_combined_{ft}")

        if not args.blind:
            combined_post_signal = stitch_histogram(
                [mp_results[mp]["per_ft"][ft]["post_signal"] for mp in sorted_mps],
                slice_boundaries, combined_edges, f"signal_post_combined_{ft}")

        fit_label = "B-only" if ft == "b" else "S+B"

        if args.blind:
            # Per-masspoint per-slice signal hists for the curated subset
            # (each signal lives only inside its owning masspoint's slice).
            def _build_signal_overlay(kind):
                """kind in {'pre_signal', 'post_signal'}."""
                out = []
                for mp in signal_subset:
                    if mp not in mp_results:
                        logging.warning(f"signal-masspoint {mp} not in loaded set; skipping")
                        continue
                    mp_idx = sorted_mps.index(mp)
                    src = mp_results[mp]["per_ft"][ft][kind]
                    h = per_mp_signal_in_slice(
                        mp, src, mp_idx, slice_boundaries, combined_edges,
                        f"{kind}_{mp}_{ft}_blind")
                    out.append((mp, h))
                return out

            if ft == fit_types[0]:
                draw_combined_blind_stack(
                    combined_pre_bkgs,
                    _build_signal_overlay("pre_signal"),
                    label_top="Pre-fit", systSrc="Pre-fit",
                    out_path=f"{out_dir}/combined_prefit_mass_{args.era_scope}_{args.channel_scope}.png",
                    xrange=xrange,
                    base_args=args,
                )

            draw_combined_blind_stack(
                combined_post_bkgs,
                _build_signal_overlay("post_signal"),
                label_top=f"Post-fit {fit_label}",
                systSrc=f"Post-fit ({fit_label})",
                out_path=f"{out_dir}/combined_postfit_{ft}_mass_{args.era_scope}_{args.channel_scope}.png",
                xrange=xrange,
                base_args=args,
            )
            continue

        # Non-blind path (partial-unblind / unblind): existing stacked plot
        # with data points, ratio pad, dashed slice lines.
        if ft == fit_types[0]:
            draw_combined_stack(
                combined_data, combined_pre_bkgs, combined_pre_signal,
                label_top="Pre-fit", systSrc="Pre-fit",
                out_path=f"{out_dir}/combined_prefit_mass_{args.era_scope}_{args.channel_scope}.png",
                xrange=xrange,
                slice_boundaries=slice_boundaries,
                base_args=args,
                show_signal=True,
            )

        draw_combined_stack(
            combined_data, combined_post_bkgs, combined_post_signal,
            label_top=f"Post-fit {fit_label}", systSrc=f"Post-fit ({fit_label})",
            out_path=f"{out_dir}/combined_postfit_{ft}_mass_{args.era_scope}_{args.channel_scope}.png",
            xrange=xrange,
            slice_boundaries=slice_boundaries,
            base_args=args,
            show_signal=True,
        )

    # Sidecar JSON
    sidecar = {
        "masspoints": sorted_mps,
        "mA": [extract_mA(m) for m in sorted_mps],
        "slice_boundaries": slice_boundaries,
        "x_range": list(xrange),
        "n_bins": len(combined_edges) - 1,
        "bin_width": args.bin_width,
        "era": args.era,
        "era_scope": args.era_scope,
        "channel_scope": args.channel_scope,
        "method": args.method,
        "binning_suffix": pm._compute_binning_suffix(SimpleNamespace(
            binning=args.binning, unblind=args.unblind,
            partial_unblind=args.partial_unblind)),
    }
    with open(f"{out_dir}/combined_config.json", "w") as f:
        json.dump(sidecar, f, indent=2)
    logging.info(f"Wrote {out_dir}/combined_config.json")
    logging.info("Done.")


if __name__ == "__main__":
    main()
