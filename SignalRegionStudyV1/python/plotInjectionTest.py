#!/usr/bin/env python3
"""
plotInjectionTest.py - Plot signal injection test results

Creates a plot comparing injected vs recovered signal strengths
using CMS style formatting.

Usage:
    # With explicit paths:
    python plotInjectionTest.py --input injection_data.csv --output injection_test.pdf \
                                --masspoint MHc130_MA90 --era Run2 --channel Combined

    # With automatic path inference:
    python plotInjectionTest.py --masspoint MHc130_MA90 --era Run2 --channel Combined \
                                --method Baseline --binning extended
"""

import argparse
import os
import ROOT
import cmsstyle as CMS

CMS.setCMSStyle()
ROOT.gROOT.SetBatch(True)

# Get script directory for path inference
PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
WORKDIR = os.path.dirname(PYTHON_DIR)  # SignalRegionStudyV1 directory


def parse_args():
    parser = argparse.ArgumentParser(description="Plot signal injection test results")
    parser.add_argument("--input", help="Input CSV file (auto-inferred if not provided)")
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


def read_data(input_file):
    """Read injection test data from CSV file."""
    x_inj, y_rec, y_err = [], [], []

    with open(input_file) as f:
        next(f)  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            x_inj.append(float(parts[0]))
            y_rec.append(float(parts[1]))
            y_err.append(float(parts[2]))

    return x_inj, y_rec, y_err


def create_plot(x_inj, y_rec, y_err, masspoint, era, channel, method, binning, output_file):
    """Create the injection test plot with CMS style."""

    n = len(x_inj)
    if n == 0:
        print("No data to plot")
        return False

    # Create TGraphErrors
    gr = ROOT.TGraphErrors(n)
    for i in range(n):
        gr.SetPoint(i, x_inj[i], y_rec[i])
        gr.SetPointError(i, 0, y_err[i])

    # Style for data points
    gr.SetMarkerStyle(20)
    gr.SetMarkerSize(1.2)
    gr.SetMarkerColor(ROOT.kBlue)
    gr.SetLineColor(ROOT.kBlue)
    gr.SetLineWidth(2)

    # Configure CMS style
    CMS.SetExtraText("Simulation Preliminary")
    CMS.SetLumi(138)
    CMS.SetEnergy(13)

    # Calculate axis range
    xmax = max(x_inj)
    rng = xmax * 1.3

    # Create CMS canvas
    canv = CMS.cmsCanvas("injection_test",
                         -0.2, rng,        # x range
                         -0.5, rng + 0.5,  # y range
                         "Injected r",
                         "Recovered r",
                         square=True,
                         iPos=11,
                         extraSpace=0.01)

    # Reference line y=x
    line = ROOT.TLine(-0.2, -0.2, rng, rng)
    line.SetLineStyle(2)
    line.SetLineColor(ROOT.kRed)
    line.SetLineWidth(2)
    line.Draw()

    # Draw points
    gr.Draw("P same")

    # Create CMS legend (positioned to avoid overlap with CMS label)
    leg = CMS.cmsLeg(0.55, 0.18, 0.92, 0.35, textSize=0.035)
    leg.AddEntry(gr, "Recovered (mean #pm std)", "ep")
    leg.AddEntry(line, "Ideal (y = x)", "l")
    leg.Draw()

    # Draw channel info
    CMS.drawText(f"masspoint: {masspoint}", posX=0.20, posY=0.75, font=42, align=11, size=0.030)
    CMS.drawText(f"channel: {channel}", posX=0.20, posY=0.70, font=42, align=11, size=0.030)
    CMS.drawText(f"method: {method}", posX=0.20, posY=0.65, font=42, align=11, size=0.030)
    CMS.drawText(f"binning: {binning}", posX=0.20, posY=0.60, font=42, align=11, size=0.030)

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
    if args.input is None or args.output is None:
        injection_dir = get_injection_dir(
            args.era, args.channel, args.masspoint, args.method, args.binning
        )
        if args.input is None:
            args.input = os.path.join(injection_dir, "injection_data.csv")
        if args.output is None:
            args.output = os.path.join(injection_dir, "injection_test.png")

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")

    # Check input file exists
    if not os.path.exists(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        return 1

    # Read data
    x_inj, y_rec, y_err = read_data(args.input)

    if len(x_inj) == 0:
        print("ERROR: No data found in input file")
        return 1

    # Create plot
    success = create_plot(x_inj, y_rec, y_err,
                          args.masspoint, args.era, args.channel,
                          args.method, args.binning,
                          args.output)

    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
