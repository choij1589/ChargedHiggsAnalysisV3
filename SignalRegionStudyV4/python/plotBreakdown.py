#!/usr/bin/env python3
"""Cross-point summary of the grouped uncertainty breakdown.

One row per mass point, one stacked bar per row, each segment a component
of sigma(r) drawn as its FRACTIONAL contribution sigma_i / sigma_total.
The points differ in sensitivity by a factor ~5, so the fraction is what
makes them comparable on one axis; the absolute sigma_total is printed at
the right of each row. The per-point plot1DScan.py panel remains the
detailed view.

Reads results/json/breakdown.{era}[.{source}].json (collectBreakdown.py).

  python3 python/plotBreakdown.py
  python3 python/plotBreakdown.py --channel SR3Mu
"""
import argparse
import json
import os

import ROOT
import cmsstyle as CMS

import srspaths
import nuisanceGroups
from plotter import LumiInfo, LumiInfoExact, EnergyInfo, get_CoM_energy

ROOT.gROOT.SetBatch(True)

# One colour per component, in cumulative freeze order then the residual.
# Statistical is deliberately the neutral grey: it is the irreducible
# part, not a systematic anyone can act on.
COMPONENT_COLOR = {
    "signal_theory": ROOT.kAzure + 2,
    "prompt_norm": ROOT.kOrange + 1,
    "nonprompt_norm": ROOT.kRed + 1,
    "experimental": ROOT.kGreen + 2,
    "stat": ROOT.kGray + 1,
}
FALLBACK_COLORS = [ROOT.kMagenta + 1, ROOT.kCyan + 2, ROOT.kYellow + 2]

# Blank rows reserved at the top of the frame for the CMS block and the
# legend.  Bars occupy rows 0..n-1 below it.
HEADROOM_ROWS = 2.4

# Legend box and final-state label, both inside the frame headroom (NDC).
LEGEND_BOX = (0.46, 0.735, 0.93, 0.875)
CHANNEL_LABEL_NDC = (0.29, 0.745)

CHANNEL_LABEL = {
    "Combined": "e#mu#mu + #mu#mu#mu",
    "SR1E2Mu": "e#mu#mu",
    "SR3Mu": "#mu#mu#mu",
}


def set_lumi_header(era):
    """Identical to plotGoFPValues / plotLimits: for All the Run2 term is
    baked into the run label because cmsstyle renders one energy."""
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


def symmetrized(entry, component):
    """Half the up+down width, or None where the component is unmeasured.

    collectBreakdown records a negative quadrature subtraction as null
    rather than zero; such a component cannot be drawn and is reported
    instead of silently vanishing into the bar.
    """
    payload = entry.get(component)
    if not payload:
        return None
    up, dn = payload.get("up"), payload.get("dn")
    if up is None or dn is None:
        return None
    return 0.5 * (abs(up) + abs(dn))


def rows_from(record, channel):
    """(label, entry) per point, Baseline then ParticleNet, mHc then mA."""
    rows = []
    for method in ("Baseline", "ParticleNet"):
        points = record.get(method, {})
        def sort_key(mp):
            mhc, ma = srspaths.masspoint_mhc_ma(mp)
            return (mhc, ma)
        for mp in sorted(points, key=sort_key):
            entry = points[mp].get(channel)
            if not entry:
                continue
            mhc, ma = srspaths.masspoint_mhc_ma(mp)
            label = (f"#splitline{{{method}}}"
                     f"{{m_{{H^{{#pm}}}}={mhc}, m_{{A}}={ma:g}}}")
            rows.append((label, entry))
    return rows


def draw(record, era, channel, components, labels, outdir):
    rows = rows_from(record, channel)
    if not rows:
        raise SystemExit(f"ERROR: no entries for channel {channel}")

    CMS.SetExtraText("Preliminary")
    set_lumi_header(era)
    n = len(rows)
    # Headroom above the bars for the CMS block, the channel label and the
    # legend, so none of them is ever drawn over a bar.
    y_max = n + HEADROOM_ROWS
    canv = CMS.cmsCanvas("breakdown", 0.0, 1.0, 0.0, y_max,
                         "fraction of #sigma^{2}(r)", "",
                         square=False, iPos=11, extraSpace=0.02)
    canv.SetLeftMargin(0.26)
    canv.SetRightMargin(0.13)
    canv.SetTopMargin(0.08)

    frame = canv.GetPrimitive("hframe")
    if frame:
        frame.GetYaxis().SetLabelSize(0.0)
        frame.GetYaxis().SetTickLength(0.0)
        frame.GetXaxis().SetNdivisions(505)

    keep = []
    # The legend lives in the headroom, to the right of the CMS block, so
    # it never overlaps a bar.
    leg = CMS.cmsLeg(LEGEND_BOX[0], LEGEND_BOX[1], LEGEND_BOX[2],
                     LEGEND_BOX[3], textSize=0.028)
    leg.SetNColumns(2)
    legend_done = set()
    unmeasured = []

    latex = ROOT.TLatex()
    latex.SetNDC(False)
    latex.SetTextFont(42)
    latex.SetTextSize(0.026)

    for irow, (label, entry) in enumerate(rows):
        y_lo = n - irow - 0.80
        y_hi = n - irow - 0.20
        total = symmetrized(entry, "total")
        widths = {c: symmetrized(entry, c) for c in components}
        missing = [c for c, w in widths.items() if w is None]
        if missing:
            unmeasured.append(f"{label}: {', '.join(missing)}")
        # Normalize on the components actually drawn, so the bar always
        # spans the axis and the reader compares shares, not lengths.
        drawn = {c: w for c, w in widths.items() if w is not None}
        norm = sum(w * w for w in drawn.values()) ** 0.5
        if norm <= 0:
            continue
        x = 0.0
        for idx, comp in enumerate(components):
            w = drawn.get(comp)
            if w is None:
                continue
            frac = (w * w) / (norm * norm)
            box = ROOT.TBox(x, y_lo, x + frac, y_hi)
            color = COMPONENT_COLOR.get(
                comp, FALLBACK_COLORS[idx % len(FALLBACK_COLORS)])
            box.SetFillColor(color)
            box.SetLineColor(ROOT.kBlack)
            box.SetLineWidth(1)
            box.Draw("l same")
            keep.append(box)
            if comp not in legend_done:
                marker = ROOT.TH1F(f"leg_{comp}", "", 1, 0, 1)
                marker.SetFillColor(color)
                marker.SetLineColor(ROOT.kBlack)
                leg.AddEntry(marker, labels.get(comp, comp), "f")
                keep.append(marker)
                legend_done.add(comp)
            x += frac

        latex.SetTextAlign(32)
        latex.DrawLatex(-0.02, 0.5 * (y_lo + y_hi), label)
        if total is not None:
            latex.SetTextAlign(12)
            latex.DrawLatex(1.02, 0.5 * (y_lo + y_hi),
                            f"#sigma={total:.2f}")

    # Final state under the CMS block, inside the frame headroom -- the
    # luminosity header above the frame is already crowded.
    latex.SetNDC(True)
    latex.SetTextAlign(11)
    latex.SetTextSize(0.030)
    latex.DrawLatex(CHANNEL_LABEL_NDC[0], CHANNEL_LABEL_NDC[1],
                    CHANNEL_LABEL.get(channel, channel))
    leg.Draw()
    canv.RedrawAxis()

    os.makedirs(outdir, exist_ok=True)
    ch_suffix = "" if channel == "Combined" else f".{channel}"
    base = os.path.join(outdir, f"breakdown.{era}{ch_suffix}")
    canv.SaveAs(f"{base}.png")
    canv.SaveAs(f"{base}.pdf")
    print(f"Wrote {base}.png / .pdf ({len(rows)} points)")
    if unmeasured:
        print("WARNING: components with no measured width (negative "
              "quadrature subtraction), omitted from the bar:")
        for line in unmeasured:
            print(f"  {line}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era", default="All")
    parser.add_argument("--channel", default="Combined",
                        choices=["Combined", "SR1E2Mu", "SR3Mu"])
    parser.add_argument("--signal-source", default="interp-signal",
                        choices=list(srspaths.SIGNAL_SOURCES))
    parser.add_argument("--input")
    parser.add_argument("--outdir")
    args = parser.parse_args()

    source_infix = ("" if args.signal_source == "mc-signal"
                    else f".{args.signal_source}")
    path = args.input or os.path.join(
        srspaths.module_dir(), "results", "json",
        f"breakdown.{args.era}{source_infix}.json")
    if not os.path.exists(path):
        raise SystemExit(f"ERROR: no breakdown JSON at {path}; run "
                         "python3 python/collectBreakdown.py first")
    with open(path) as f:
        record = json.load(f)

    config = nuisanceGroups.load_config()
    components = nuisanceGroups.component_names(config)
    labels = nuisanceGroups.component_labels(config)
    outdir = args.outdir or os.path.join(
        srspaths.module_dir(), "results", "plots", "breakdown")
    draw(record, args.era, args.channel, components, labels, outdir)


if __name__ == "__main__":
    main()
