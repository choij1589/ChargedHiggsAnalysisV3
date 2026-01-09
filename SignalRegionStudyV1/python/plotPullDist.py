#!/usr/bin/env python3
"""
plotPullDist.py - Plot pull distribution for signal injection test

B2G requirement: Pull = (r - r_inj) / r_Err with Gaussian fit
- Uses asymmetric errors: r_Err = rHiErr if (r - r_inj) > 0 else rLoErr
- Mean should be ~0, width should be ~1 for good coverage

Usage:
    python plotPullDist.py --era Run2 --channel Combined --masspoint MHc130_MA90 \
                           --method Baseline --binning extended
"""
import argparse
import os
import json
import ROOT
import cmsstyle as CMS

CMS.setCMSStyle()
ROOT.gROOT.SetBatch(True)

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


def calculate_pull(r, r_inj, rLoErr, rHiErr):
    """Calculate pull with asymmetric error selection."""
    bias = r - r_inj
    # Use asymmetric error: rHiErr if bias > 0, else rLoErr
    r_err = rHiErr if bias > 0 else rLoErr

    # Protect against zero error
    if r_err <= 0:
        return None

    return bias / r_err


def create_pull_plot(results_data, masspoint, era, channel, method, binning, output_file):
    """Create pull distribution plot with Gaussian fits."""

    results = results_data['results']

    if not results:
        raise ValueError("No results to plot")

    # Standard pull range
    x_min, x_max = -5, 5

    # Create histograms and fit for each r_inj
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

        # Normalize to PDF
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

    # Configure CMS style
    CMS.SetExtraText("Simulation Preliminary")
    CMS.SetLumi(138)
    CMS.SetEnergy(13)

    # Calculate y-axis range
    y_max = max(h.GetMaximum() for h, _ in histograms.values()) * 1.6

    # Create canvas
    canv = CMS.cmsCanvas("pull", x_min, x_max, 0, y_max,
                         "(r - r_{inj}) / #sigma_{r}", "PDF",
                         square=True, iPos=11, extraSpace=0.01)

    bin_width = (x_max - x_min) / 40.0

    # Draw histograms and fits
    draw_funcs = []
    for r_inj, (h, color) in sorted(histograms.items()):
        CMS.cmsObjectDraw(h, "hist", LineColor=color, LineWidth=2)

        # Draw fit
        fr = fit_results[r_inj]
        gaus = ROOT.TF1(f"gaus_draw_{r_inj}", "gaus", x_min, x_max)
        amplitude = bin_width / (ROOT.TMath.Sqrt(2 * ROOT.TMath.Pi()) * fr['sigma'])
        gaus.SetParameters(amplitude, fr['mean'], fr['sigma'])
        gaus.SetLineColor(color)
        gaus.SetLineStyle(2)
        gaus.SetLineWidth(2)
        gaus.Draw("same")
        draw_funcs.append(gaus)

    # Legend with fit parameters
    n_entries = len(histograms) + 1  # +1 for reference
    leg_height = 0.05 * n_entries
    leg = CMS.cmsLeg(0.55, 0.88 - leg_height, 0.95, 0.88, textSize=0.025)

    for r_inj, (h, color) in sorted(histograms.items()):
        fr = fit_results[r_inj]
        leg.AddEntry(h, f"r_{{inj}}={r_inj:.2f}: #mu={fr['mean']:.2f}, #sigma={fr['sigma']:.2f}", "l")
    leg.Draw()

    # Draw info text
    CMS.drawText(f"{masspoint}", posX=0.20, posY=0.75, font=42, align=11, size=0.030)
    CMS.drawText(f"{era} / {channel}", posX=0.20, posY=0.7, font=42, align=11, size=0.030)
    CMS.drawText(f"{method}", posX=0.20, posY=0.65, font=42, align=11, size=0.030)

    canv.RedrawAxis()

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    canv.SaveAs(output_file)
    canv.SaveAs(output_file.replace('.png', '.pdf'))
    print(f"Pull plot saved to {output_file}")

    # Print summary
    print("\nPull Distribution Summary:")
    print("-" * 70)
    print(f"{'r_inj':<10} {'mean':<20} {'sigma':<20} {'Status':<15}")
    print("-" * 70)
    for r_inj in sorted(fit_results.keys()):
        fr = fit_results[r_inj]
        # Check if mean ~ 0 and sigma ~ 1
        mean_ok = abs(fr['mean']) < 0.3
        sigma_ok = abs(fr['sigma'] - 1) < 0.3
        status = "OK" if (mean_ok and sigma_ok) else "CHECK"
        print(f"{r_inj:<10.2f} {fr['mean']:.3f}+/-{fr['mean_err']:.3f}       {fr['sigma']:.3f}+/-{fr['sigma_err']:.3f}       {status}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Plot pull distribution for signal injection")
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
    output_file = args.output or os.path.join(injection_dir, "pull_dist.png")

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with open(input_file) as f:
        results_data = json.load(f)

    create_pull_plot(results_data, args.masspoint, args.era, args.channel,
                     args.method, args.binning, output_file)


if __name__ == "__main__":
    main()
