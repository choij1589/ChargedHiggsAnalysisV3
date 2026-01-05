#!/usr/bin/env python3
"""
plotInjectionDist.py - Plot distributions of recovered r values for signal injection test

Creates histograms showing the distribution of fitted signal strengths
for each injected r value, using CMS style formatting.

Usage:
    # With automatic path inference:
    python plotInjectionDist.py --masspoint MHc130_MA90 --era Run2 --channel Combined \
                                --method Baseline --binning extended

    # With explicit paths:
    python plotInjectionDist.py --input /path/to/injection/dir --output plot.pdf \
                                --masspoint MHc130_MA90 --era Run2 --channel Combined
"""

import argparse
import os
import glob
import ROOT
import cmsstyle as CMS

CMS.setCMSStyle()
ROOT.gROOT.SetBatch(True)

# Get script directory for path inference
PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
WORKDIR = os.path.dirname(PYTHON_DIR)  # SignalRegionStudyV1 directory

# Color palette for different r values
PALETTE = [
    ROOT.TColor.GetColor("#5790fc"),  # blue
    ROOT.TColor.GetColor("#f89c20"),  # orange
    ROOT.TColor.GetColor("#e42536"),  # red
    ROOT.TColor.GetColor("#964a8b"),  # purple
    ROOT.TColor.GetColor("#9c9ca1"),  # gray
    ROOT.TColor.GetColor("#7a21dd"),  # violet
]


def parse_args():
    parser = argparse.ArgumentParser(description="Plot distributions of recovered r values")
    parser.add_argument("--input", help="Input injection directory (auto-inferred if not provided)")
    parser.add_argument("--output", help="Output plot file (auto-inferred if not provided)")
    parser.add_argument("--masspoint", required=True, help="Signal mass point (e.g., MHc130_MA90)")
    parser.add_argument("--era", required=True, help="Data-taking era (e.g., Run2)")
    parser.add_argument("--channel", required=True, help="Analysis channel (e.g., Combined)")
    parser.add_argument("--method", default="Baseline", help="Template method (e.g., Baseline, ParticleNet)")
    parser.add_argument("--binning", default="uniform", help="Binning scheme (e.g., uniform, extended)")
    return parser.parse_args()


def get_injection_dir(era, channel, masspoint, method, binning):
    """Construct the injection output directory path."""
    return os.path.join(
        WORKDIR, "templates", era, channel, masspoint, method, binning,
        "combine_output", "injection"
    )


def read_fitted_r_values(fit_file):
    """Read fitted r values from MultiDimFit output ROOT file."""
    r_values = []

    f = ROOT.TFile.Open(fit_file)
    if not f or f.IsZombie():
        print(f"WARNING: Could not open {fit_file}")
        return r_values

    tree = f.Get("limit")
    if not tree:
        print(f"WARNING: No 'limit' tree in {fit_file}")
        f.Close()
        return r_values

    for i in range(tree.GetEntries()):
        tree.GetEntry(i)
        # Filter for best-fit entries only (quantileExpected == -1)
        if tree.quantileExpected < -0.5:
            r_values.append(float(tree.r))

    f.Close()
    return r_values


def find_r_directories(injection_dir):
    """Find all r{value} subdirectories and extract r values."""
    r_dirs = glob.glob(os.path.join(injection_dir, "r*"))
    r_values = {}

    for r_dir in r_dirs:
        dirname = os.path.basename(r_dir)
        if dirname.startswith("r"):
            try:
                r_val = float(dirname[1:])  # Remove 'r' prefix
                r_values[r_val] = r_dir
            except ValueError:
                continue

    return r_values


def create_distribution_plot(r_distributions, masspoint, era, channel, method, binning, output_file):
    """Create the distribution plot with CMS style."""

    if not r_distributions:
        print("ERROR: No distributions to plot")
        return False

    # Get all values for determining histogram range
    all_values = []
    injected_values = []
    for r_inj, r_vals in r_distributions.items():
        all_values.extend(r_vals)
        injected_values.append(r_inj)

    if not all_values:
        print("ERROR: No fitted r values found")
        return False

    # Determine histogram range
    x_min = min(min(all_values), min(injected_values)) - 0.5
    x_max = max(max(all_values), max(injected_values)) + 0.5

    # Create histograms for each injected r value
    histograms = {}
    lines = []  # Keep references to prevent garbage collection

    for i, (r_inj, r_vals) in enumerate(sorted(r_distributions.items())):
        h = ROOT.TH1F(f"h_r{r_inj}", "", 50, x_min, x_max)
        h.SetDirectory(0)  # Prevent ROOT ownership

        for v in r_vals:
            h.Fill(v)

        # Normalize to probability density
        if h.Integral() > 0:
            h.Scale(1.0 / h.Integral())

        h.SetStats(0)
        histograms[r_inj] = (h, PALETTE[i % len(PALETTE)])

    # Configure CMS style
    CMS.SetExtraText("Simulation Preliminary")
    CMS.SetLumi(138)
    CMS.SetEnergy(13)

    # Calculate y-axis range
    y_max = max(h.GetMaximum() for h, _ in histograms.values()) * 1.5

    # Create CMS canvas
    canv = CMS.cmsCanvas("dist", x_min, x_max, 0, y_max,
                         "Recovered r", "PDF",
                         square=True, iPos=11, extraSpace=0.01)

    # Draw histograms using cmsObjectDraw
    for i, (r_inj, (h, color)) in enumerate(sorted(histograms.items())):
        CMS.cmsObjectDraw(h, "hist", LineColor=color, LineWidth=2, LineStyle=ROOT.kSolid)

        # Draw vertical dashed line at injected r value
        line = ROOT.TLine(r_inj, 0, r_inj, y_max * 0.7)
        line.SetLineStyle(2)
        line.SetLineWidth(2)
        line.SetLineColor(color)
        line.Draw()
        lines.append(line)  # Keep reference

    # Create legend
    n_entries = len(histograms)
    leg_height = 0.05 * n_entries
    leg = CMS.cmsLeg(0.55, 0.88 - leg_height, 0.92, 0.88, textSize=0.030)

    for r_inj, (h, color) in sorted(histograms.items()):
        mean = h.GetMean()
        std = h.GetStdDev()
        leg.AddEntry(h, f"r={r_inj} (#mu={mean:.2f} #pm {std:.2f})", "l")
    leg.Draw()

    # Draw channel info
    CMS.drawText(f"masspoint: {masspoint}", posX=0.62, posY=0.6, font=42, align=11, size=0.030)
    CMS.drawText(f"channel: {channel}", posX=0.62, posY=0.55, font=42, align=11, size=0.030)
    CMS.drawText(f"method: {method}", posX=0.62, posY=0.5, font=42, align=11, size=0.030)
    CMS.drawText(f"binning: {binning}", posX=0.62, posY=0.45, font=42, align=11, size=0.030)

    canv.RedrawAxis()

    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    canv.SaveAs(output_file)
    print(f"Plot saved to {output_file}")

    return True


def main():
    args = parse_args()

    # Infer input/output paths if not provided
    if args.input is None:
        args.input = get_injection_dir(
            args.era, args.channel, args.masspoint, args.method, args.binning
        )
    if args.output is None:
        args.output = os.path.join(args.input, "injection_dist.png")

    print(f"Input directory: {args.input}")
    print(f"Output file:     {args.output}")

    # Check input directory exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input directory not found: {args.input}")
        return 1

    # Find all r value directories
    r_dirs = find_r_directories(args.input)
    if not r_dirs:
        print(f"ERROR: No r* directories found in {args.input}")
        return 1

    print(f"Found {len(r_dirs)} injection points: {sorted(r_dirs.keys())}")

    # Read fitted r values for each injected r
    r_distributions = {}
    for r_inj, r_dir in sorted(r_dirs.items()):
        # Find MultiDimFit output file
        fit_files = glob.glob(os.path.join(r_dir, "higgsCombine.recovery_r*.MultiDimFit.mH120.*.root"))
        if not fit_files:
            print(f"  WARNING: No MultiDimFit file for r={r_inj}")
            continue

        fit_file = fit_files[0]  # Take first match
        r_vals = read_fitted_r_values(fit_file)
        if r_vals:
            r_distributions[r_inj] = r_vals
            print(f"  r={r_inj}: {len(r_vals)} fitted values (mean={sum(r_vals)/len(r_vals):.3f})")
        else:
            print(f"  WARNING: No fitted r values for r={r_inj}")

    if not r_distributions:
        print("ERROR: No valid distributions found")
        return 1

    # Create plot
    success = create_distribution_plot(
        r_distributions,
        args.masspoint, args.era, args.channel, args.method, args.binning,
        args.output
    )

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
