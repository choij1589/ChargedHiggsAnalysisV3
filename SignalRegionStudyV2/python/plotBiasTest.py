#!/usr/bin/env python3
"""
plotBiasTest.py - Plot bias and pull distributions for signal injection test

B2G requirements:
- Bias: Distribution of (r - r_inj) with Gaussian fit. Mean should be ~0.
- Pull: Distribution of (r - r_inj) / sigma_r with Gaussian fit.
        Mean ~0 and sigma ~1 indicates good coverage.

Usage:
    # Bias plot
    python plotBiasTest.py --input injection_results.json --output bias_test.pdf \
                           --plot-type bias --masspoint MHc130_MA90 --era All --method Baseline

    # Pull plot
    python plotBiasTest.py --input injection_results.json --output pull_dist.pdf \
                           --plot-type pull --masspoint MHc130_MA90 --era All --method Baseline
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

# Luminosity and energy configuration
LUMI_INFO = {
    "2016preVFP": (19.5, 13),
    "2016postVFP": (16.8, 13),
    "2017": (41.5, 13),
    "2018": (59.8, 13),
    "2022": (7.9, 13.6),
    "2022EE": (26.7, 13.6),
    "2023": (17.8, 13.6),
    "2023BPix": (9.5, 13.6),
    "Run2": (138, 13),
    "Run3": (62, 13.6),
    "All": (200, 0),  # Special handling for combined
}


def configure_cms_style(era):
    """Configure CMS style labels for the given era."""
    CMS.SetExtraText("Simulation Preliminary")

    if era == "All":
        CMS.SetLumi(None, run="Run 2+3, 138+62 fb^{#minus1}")
        CMS.SetEnergy(0, unit="13/13.6 TeV")
    elif era in LUMI_INFO:
        lumi, energy = LUMI_INFO[era]
        CMS.SetLumi(lumi, run=era)
        CMS.SetEnergy(energy)
    else:
        CMS.SetLumi(-1)


def calculate_pull(r, r_inj, rLoErr, rHiErr):
    """Calculate pull with asymmetric error selection."""
    bias = r - r_inj
    # Use asymmetric error: rHiErr if bias > 0, else rLoErr
    r_err = rHiErr if bias > 0 else rLoErr

    if r_err <= 0:
        return None

    return bias / r_err


def create_bias_plot(results_data, masspoint, era, method, output_file):
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
    fit_funcs = []

    for i, (r_inj_str, fits) in enumerate(sorted(results.items(), key=lambda x: float(x[0]))):
        r_inj = float(r_inj_str)

        h = ROOT.TH1F(f"h_bias_{r_inj}", "", 50, x_min, x_max)
        h.SetDirectory(0)

        for fit in fits:
            bias = fit['r'] - r_inj
            h.Fill(bias)

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

    configure_cms_style(era)

    y_max = max(h.GetMaximum() for h, _ in histograms.values()) * 1.6

    canv = CMS.cmsCanvas("bias", x_min, x_max, 0, y_max,
                         "r - r_{inj}", "PDF",
                         square=True, iPos=11, extraSpace=0.01)

    # Draw histograms and fits
    draw_funcs = []
    for r_inj, (h, color) in sorted(histograms.items()):
        CMS.cmsObjectDraw(h, "hist", LineColor=color, LineWidth=2)

        fr = fit_results[r_inj]
        gaus = ROOT.TF1(f"gaus_draw_{r_inj}", "gaus", x_min, x_max)
        bin_width = h.GetBinWidth(1)
        amplitude = bin_width / (ROOT.TMath.Sqrt(2 * ROOT.TMath.Pi()) * fr['sigma'])
        gaus.SetParameters(amplitude, fr['mean'], fr['sigma'])
        gaus.SetLineColor(color)
        gaus.SetLineStyle(2)
        gaus.SetLineWidth(2)
        gaus.Draw("same")
        draw_funcs.append(gaus)

    # Vertical line at 0 (ideal)
    line = ROOT.TLine(0, 0, 0, y_max * 0.8)
    line.SetLineStyle(3)
    line.SetLineWidth(2)
    line.SetLineColor(ROOT.kGray + 2)
    line.Draw()

    # Legend
    n_entries = len(histograms)
    leg_height = 0.06 * n_entries
    leg = CMS.cmsLeg(0.55, 0.88 - leg_height, 0.95, 0.88, textSize=0.025)

    for r_inj, (h, color) in sorted(histograms.items()):
        fr = fit_results[r_inj]
        leg.AddEntry(h, f"r_{{inj}}={r_inj:.2f}: #mu={fr['mean']:.3f}#pm{fr['mean_err']:.3f}", "l")
    leg.Draw()

    # Info text
    CMS.drawText(f"{masspoint}", posX=0.20, posY=0.75, font=42, align=11, size=0.030)
    CMS.drawText(f"{era} / Combined", posX=0.20, posY=0.7, font=42, align=11, size=0.030)
    CMS.drawText(f"{method}", posX=0.20, posY=0.65, font=42, align=11, size=0.030)

    canv.RedrawAxis()

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    canv.SaveAs(output_file)
    print(f"Bias plot saved to {output_file}")

    # Print summary
    print("\nBias Test Summary:")
    print("-" * 60)
    print(f"{'r_inj':<10} {'mean':<20} {'sigma':<20}")
    print("-" * 60)
    for r_inj in sorted(fit_results.keys()):
        fr = fit_results[r_inj]
        print(f"{r_inj:<10.4f} {fr['mean']:.4f}+/-{fr['mean_err']:.4f}     "
              f"{fr['sigma']:.4f}+/-{fr['sigma_err']:.4f}")


def create_pull_plot(results_data, masspoint, era, method, output_file):
    """Create pull distribution plot with Gaussian fits."""
    results = results_data['results']

    if not results:
        raise ValueError("No results to plot")

    x_min, x_max = -5, 5

    histograms = {}
    fit_results = {}
    fit_funcs = []

    for i, (r_inj_str, fits) in enumerate(sorted(results.items(), key=lambda x: float(x[0]))):
        r_inj = float(r_inj_str)

        h = ROOT.TH1F(f"h_pull_{r_inj}", "", 40, x_min, x_max)
        h.SetDirectory(0)

        n_skipped = 0
        for fit in fits:
            pull = calculate_pull(fit['r'], r_inj, fit['rLoErr'], fit['rHiErr'])
            if pull is not None:
                h.Fill(pull)
            else:
                n_skipped += 1

        if n_skipped > 0:
            print(f"  r={r_inj}: Skipped {n_skipped} entries with zero error")

        if h.Integral() > 0:
            h.Scale(1.0 / h.Integral())

        # Gaussian fit with initial guess mean=0, sigma=1
        gaus = ROOT.TF1(f"gaus_{r_inj}", "gaus", x_min, x_max)
        gaus.SetParameters(h.GetMaximum(), 0, 1)
        h.Fit(gaus, "Q0")

        histograms[r_inj] = (h, PALETTE[i % len(PALETTE)])
        fit_results[r_inj] = {
            'mean': gaus.GetParameter(1),
            'mean_err': gaus.GetParError(1),
            'sigma': gaus.GetParameter(2),
            'sigma_err': gaus.GetParError(2)
        }
        fit_funcs.append(gaus)

    configure_cms_style(era)

    y_max = max(h.GetMaximum() for h, _ in histograms.values()) * 1.6

    canv = CMS.cmsCanvas("pull", x_min, x_max, 0, y_max,
                         "(r - r_{inj}) / #sigma_{r}", "PDF",
                         square=True, iPos=11, extraSpace=0.01)

    bin_width = (x_max - x_min) / 40.0

    # Draw histograms and fits
    draw_funcs = []
    for r_inj, (h, color) in sorted(histograms.items()):
        CMS.cmsObjectDraw(h, "hist", LineColor=color, LineWidth=2)

        fr = fit_results[r_inj]
        gaus = ROOT.TF1(f"gaus_draw_{r_inj}", "gaus", x_min, x_max)
        amplitude = bin_width / (ROOT.TMath.Sqrt(2 * ROOT.TMath.Pi()) * fr['sigma'])
        gaus.SetParameters(amplitude, fr['mean'], fr['sigma'])
        gaus.SetLineColor(color)
        gaus.SetLineStyle(2)
        gaus.SetLineWidth(2)
        gaus.Draw("same")
        draw_funcs.append(gaus)

    # Legend
    n_entries = len(histograms) + 1
    leg_height = 0.05 * n_entries
    leg = CMS.cmsLeg(0.55, 0.88 - leg_height, 0.95, 0.88, textSize=0.025)

    for r_inj, (h, color) in sorted(histograms.items()):
        fr = fit_results[r_inj]
        leg.AddEntry(h, f"r_{{inj}}={r_inj:.2f}: #mu={fr['mean']:.2f}, #sigma={fr['sigma']:.2f}", "l")
    leg.Draw()

    # Info text
    CMS.drawText(f"{masspoint}", posX=0.20, posY=0.75, font=42, align=11, size=0.030)
    CMS.drawText(f"{era} / Combined", posX=0.20, posY=0.7, font=42, align=11, size=0.030)
    CMS.drawText(f"{method}", posX=0.20, posY=0.65, font=42, align=11, size=0.030)

    canv.RedrawAxis()

    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    canv.SaveAs(output_file)
    print(f"Pull plot saved to {output_file}")

    # Print summary
    print("\nPull Distribution Summary:")
    print("-" * 70)
    print(f"{'r_inj':<10} {'mean':<20} {'sigma':<20} {'Status':<15}")
    print("-" * 70)
    for r_inj in sorted(fit_results.keys()):
        fr = fit_results[r_inj]
        mean_ok = abs(fr['mean']) < 0.3
        sigma_ok = abs(fr['sigma'] - 1) < 0.3
        status = "OK" if (mean_ok and sigma_ok) else "CHECK"
        print(f"{r_inj:<10.4f} {fr['mean']:.3f}+/-{fr['mean_err']:.3f}       "
              f"{fr['sigma']:.3f}+/-{fr['sigma_err']:.3f}       {status}")


def main():
    parser = argparse.ArgumentParser(description="Plot bias/pull distributions for signal injection")
    parser.add_argument("--input", required=True, help="Input JSON file (injection_results.json)")
    parser.add_argument("--output", required=True, help="Output plot file (e.g., bias_test.pdf)")
    parser.add_argument("--plot-type", required=True, choices=["bias", "pull"],
                        help="Plot type: bias or pull")
    parser.add_argument("--masspoint", required=True, help="Signal mass point")
    parser.add_argument("--era", required=True, help="Data-taking era")
    parser.add_argument("--method", default="Baseline", help="Template method")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    with open(args.input) as f:
        results_data = json.load(f)

    if args.plot_type == "bias":
        create_bias_plot(results_data, args.masspoint, args.era, args.method, args.output)
    elif args.plot_type == "pull":
        create_pull_plot(results_data, args.masspoint, args.era, args.method, args.output)


if __name__ == "__main__":
    main()
