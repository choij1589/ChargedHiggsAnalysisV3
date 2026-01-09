#!/usr/bin/env python3
"""
plotBiasTest.py - Plot bias distribution (r - r_inj) for signal injection test

B2G requirement: Distribution of (r - r_inj) with Gaussian fit
- Mean should be ~0 for unbiased estimator

Usage:
    python plotBiasTest.py --era Run2 --channel Combined --masspoint MHc130_MA90 \
                           --method Baseline --binning extended
"""
import argparse
import os
import json
import ROOT
import cmsstyle as CMS

CMS.setCMSStyle()
ROOT.gROOT.SetBatch(True)

# Color palette
PALETTE = [
    ROOT.TColor.GetColor("#5790fc"),  # blue
    ROOT.TColor.GetColor("#f89c20"),  # orange
    ROOT.TColor.GetColor("#e42536"),  # red
    ROOT.TColor.GetColor("#964a8b"),  # purple
]

PYTHON_DIR = os.path.dirname(os.path.abspath(__file__))
WORKDIR = os.path.dirname(PYTHON_DIR)


def get_injection_dir(era, channel, masspoint, method, binning):
    """Construct the injection output directory path."""
    return os.path.join(
        WORKDIR, "templates", era, channel, masspoint, method, binning,
        "combine_output", "injection"
    )


def create_bias_plot(results_data, masspoint, era, channel, method, binning, output_file):
    """Create bias distribution plot with Gaussian fits."""

    results = results_data['results']

    if not results:
        raise ValueError("No results to plot")

    # Determine histogram range from all bias values
    all_bias = []
    for r_inj_str, fits in results.items():
        r_inj = float(r_inj_str)
        for fit in fits:
            all_bias.append(fit['r'] - r_inj)

    if not all_bias:
        raise ValueError("No bias values found")

    x_min = min(all_bias) - 0.2
    x_max = max(all_bias) + 0.2

    # Create histograms and fit for each r_inj
    histograms = {}
    fit_results = {}
    fit_funcs = []  # Keep references

    for i, (r_inj_str, fits) in enumerate(sorted(results.items(), key=lambda x: float(x[0]))):
        r_inj = float(r_inj_str)

        h = ROOT.TH1F(f"h_bias_{r_inj}", "", 50, x_min, x_max)
        h.SetDirectory(0)

        for fit in fits:
            bias = fit['r'] - r_inj
            h.Fill(bias)

        # Normalize to PDF
        if h.Integral() > 0:
            h.Scale(1.0 / h.Integral())

        # Gaussian fit
        gaus = ROOT.TF1(f"gaus_{r_inj}", "gaus", x_min, x_max)
        gaus.SetParameters(h.GetMaximum(), h.GetMean(), h.GetStdDev())
        h.Fit(gaus, "Q0")

        histograms[r_inj] = (h, PALETTE[i % len(PALETTE)])
        fit_results[r_inj] = {
            'mean': gaus.GetParameter(1),
            'mean_err': gaus.GetParError(1),
            'sigma': gaus.GetParameter(2),
            'sigma_err': gaus.GetParError(2)
        }
        fit_funcs.append(gaus)

    # Configure CMS style
    CMS.SetExtraText("Simulation Preliminary")
    CMS.SetLumi(138)
    CMS.SetEnergy(13)

    # Calculate y-axis range
    y_max = max(h.GetMaximum() for h, _ in histograms.values()) * 1.6

    # Create canvas
    canv = CMS.cmsCanvas("bias", x_min, x_max, 0, y_max,
                         "r - r_{inj}", "PDF",
                         square=True, iPos=11, extraSpace=0.01)

    # Draw histograms and fits
    draw_funcs = []
    for r_inj, (h, color) in sorted(histograms.items()):
        CMS.cmsObjectDraw(h, "hist", LineColor=color, LineWidth=2)

        # Draw fit function
        fr = fit_results[r_inj]
        gaus = ROOT.TF1(f"gaus_draw_{r_inj}", "gaus", x_min, x_max)
        # Scale amplitude for normalized histogram
        bin_width = h.GetBinWidth(1)
        amplitude = bin_width / (ROOT.TMath.Sqrt(2 * ROOT.TMath.Pi()) * fr['sigma'])
        gaus.SetParameters(amplitude, fr['mean'], fr['sigma'])
        gaus.SetLineColor(color)
        gaus.SetLineStyle(2)
        gaus.SetLineWidth(2)
        gaus.Draw("same")
        draw_funcs.append(gaus)

    # Draw vertical line at 0 (ideal)
    line = ROOT.TLine(0, 0, 0, y_max * 0.8)
    line.SetLineStyle(3)
    line.SetLineWidth(2)
    line.SetLineColor(ROOT.kGray + 2)
    line.Draw()

    # Legend with fit parameters
    n_entries = len(histograms)
    leg_height = 0.06 * n_entries
    leg = CMS.cmsLeg(0.55, 0.88 - leg_height, 0.95, 0.88, textSize=0.025)

    for r_inj, (h, color) in sorted(histograms.items()):
        fr = fit_results[r_inj]
        leg.AddEntry(h, f"r_{{inj}}={r_inj:.2f}: #mu={fr['mean']:.3f}#pm{fr['mean_err']:.3f}", "l")
    leg.Draw()

    # Draw info text
    CMS.drawText(f"{masspoint}", posX=0.20, posY=0.75, font=42, align=11, size=0.030)
    CMS.drawText(f"{era} / {channel}", posX=0.20, posY=0.7, font=42, align=11, size=0.030)
    CMS.drawText(f"{method}", posX=0.20, posY=0.65, font=42, align=11, size=0.030)

    canv.RedrawAxis()

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    canv.SaveAs(output_file)
    print(f"Bias plot saved to {output_file}")

    # Print summary
    print("\nBias Test Summary:")
    print("-" * 60)
    print(f"{'r_inj':<10} {'mean':<20} {'sigma':<20}")
    print("-" * 60)
    for r_inj in sorted(fit_results.keys()):
        fr = fit_results[r_inj]
        print(f"{r_inj:<10.2f} {fr['mean']:.4f}+/-{fr['mean_err']:.4f}     {fr['sigma']:.4f}+/-{fr['sigma_err']:.4f}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Plot bias distribution for signal injection")
    parser.add_argument("--era", required=True, help="Data-taking era")
    parser.add_argument("--channel", required=True, help="Analysis channel")
    parser.add_argument("--masspoint", required=True, help="Signal mass point")
    parser.add_argument("--method", default="Baseline", help="Template method")
    parser.add_argument("--binning", default="uniform", help="Binning scheme")
    parser.add_argument("--input", help="Input JSON (auto-inferred if not provided)")
    parser.add_argument("--output", help="Output plot (auto-inferred if not provided)")
    args = parser.parse_args()

    injection_dir = get_injection_dir(
        args.era, args.channel, args.masspoint, args.method, args.binning
    )

    input_file = args.input or os.path.join(injection_dir, "injection_results.json")
    output_file = args.output or os.path.join(injection_dir, "bias_test.png")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file) as f:
        results_data = json.load(f)

    create_bias_plot(results_data, args.masspoint, args.era, args.channel,
                     args.method, args.binning, output_file)


if __name__ == "__main__":
    main()
