#!/usr/bin/env python3
"""
Plot test statistic distributions from HybridNew toys.

Usage:
    python plotTestStatDist.py higgsCombine.r0.2000.seed1000.HybridNew.mH120.1000.root -o test_stat.pdf
    python plotTestStatDist.py hybridnew_grid.root --r 0.2 -o test_stat.pdf
"""
import os
import argparse
import glob
import ROOT
from array import array

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)


def get_hypotestresult(filename):
    """Extract HypoTestResult from a HybridNew output file."""
    f = ROOT.TFile.Open(filename)
    if not f or f.IsZombie():
        return None, None

    toys = f.Get("toys")
    if not toys:
        f.Close()
        return None, None

    htr = None
    for key in toys.GetListOfKeys():
        if "HypoTestResult" in key.GetName():
            htr = key.ReadObj()
            break

    return htr, f


def merge_hypotestresults(filenames):
    """Merge HypoTestResults from multiple files."""
    merged_htr = None
    files = []

    for fname in filenames:
        htr, f = get_hypotestresult(fname)
        if htr:
            files.append(f)
            if merged_htr is None:
                merged_htr = htr
            else:
                merged_htr.Append(htr)

    return merged_htr, files


def plot_test_stat_distribution(htr, output, r_value=None, title=""):
    """Plot the test statistic distributions for S+B and B-only hypotheses."""
    # Get distributions
    sb_dist = htr.GetAltDistribution()
    b_dist = htr.GetNullDistribution()
    q_obs = htr.GetTestStatisticData()

    # Get values
    sb_vals = list(sb_dist.GetSamplingDistribution())
    b_vals = list(b_dist.GetSamplingDistribution())

    # Find range
    all_vals = sb_vals + b_vals + [q_obs]
    xmin = min(all_vals) - 0.5
    xmax = max(all_vals) + 0.5

    # Create histograms
    nbins = 50
    h_sb = ROOT.TH1F("h_sb", "", nbins, xmin, xmax)
    h_b = ROOT.TH1F("h_b", "", nbins, xmin, xmax)

    for v in sb_vals:
        h_sb.Fill(v)
    for v in b_vals:
        h_b.Fill(v)

    # Normalize
    if h_sb.Integral() > 0:
        h_sb.Scale(1.0 / h_sb.Integral())
    if h_b.Integral() > 0:
        h_b.Scale(1.0 / h_b.Integral())

    # Styling
    h_sb.SetLineColor(ROOT.kRed)
    h_sb.SetFillColor(ROOT.kRed)
    h_sb.SetFillStyle(3004)
    h_sb.SetLineWidth(2)

    h_b.SetLineColor(ROOT.kBlue)
    h_b.SetFillColor(ROOT.kBlue)
    h_b.SetFillStyle(3005)
    h_b.SetLineWidth(2)

    # Create canvas
    canvas = ROOT.TCanvas("c", "Test Statistic Distribution", 800, 600)
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.05)
    canvas.SetBottomMargin(0.12)

    # Draw
    ymax = max(h_sb.GetMaximum(), h_b.GetMaximum()) * 1.3
    h_b.SetMaximum(ymax)
    h_b.SetMinimum(0)

    if title:
        h_b.SetTitle(f"{title};Test statistic q;Probability density")
    elif r_value is not None:
        h_b.SetTitle(f"Test Statistic Distribution (r = {r_value:.2f});Test statistic q;Probability density")
    else:
        h_b.SetTitle("Test Statistic Distribution;Test statistic q;Probability density")

    h_b.Draw("HIST")
    h_sb.Draw("HIST SAME")

    # Observed line
    line_obs = ROOT.TLine(q_obs, 0, q_obs, ymax * 0.8)
    line_obs.SetLineColor(ROOT.kBlack)
    line_obs.SetLineStyle(2)
    line_obs.SetLineWidth(2)
    line_obs.Draw()

    # Arrow showing observed
    arrow = ROOT.TArrow(q_obs, ymax * 0.75, q_obs, ymax * 0.1, 0.02, "|>")
    arrow.SetLineColor(ROOT.kBlack)
    arrow.SetLineWidth(2)
    arrow.Draw()

    # Legend
    legend = ROOT.TLegend(0.55, 0.65, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.035)
    legend.AddEntry(h_b, f"B-only ({len(b_vals)} toys)", "f")
    legend.AddEntry(h_sb, f"S+B ({len(sb_vals)} toys)", "f")
    legend.AddEntry(line_obs, f"Observed: q = {q_obs:.3f}", "l")
    legend.Draw()

    # Add CLs values
    latex = ROOT.TLatex()
    latex.SetNDC()
    latex.SetTextSize(0.035)
    latex.SetTextFont(42)
    latex.DrawLatex(0.15, 0.85, f"CL_{{s+b}} = {htr.CLsplusb():.4f}")
    latex.DrawLatex(0.15, 0.80, f"CL_{{b}} = {htr.CLb():.4f}")
    latex.DrawLatex(0.15, 0.75, f"CL_{{s}} = {htr.CLs():.4f}")

    # Save
    canvas.SaveAs(output)
    print(f"Saved: {output}")


def main():
    parser = argparse.ArgumentParser(description="Plot test statistic distributions from HybridNew")
    parser.add_argument("input", help="Input file (single toy file or hybridnew_grid.root)")
    parser.add_argument("-o", "--output", default="test_stat_dist.pdf", help="Output plot file")
    parser.add_argument("--r", type=float, default=None,
                        help="r value to plot (if using grid file or directory)")
    parser.add_argument("--title", default="", help="Plot title")
    args = parser.parse_args()

    # Check if input is a single file or needs merging
    if "seed" in args.input:
        # Single toy file
        htr, f = get_hypotestresult(args.input)
        if htr is None:
            print(f"ERROR: Could not read HypoTestResult from {args.input}")
            return 1

        # Try to extract r from filename
        r_val = args.r
        if r_val is None:
            try:
                basename = os.path.basename(args.input)
                parts = basename.split('.')
                r_str = parts[1] + '.' + parts[2]
                r_val = float(r_str[1:])
            except:
                pass

        plot_test_stat_distribution(htr, args.output, r_val, args.title)
        f.Close()

    else:
        # Grid file or directory - need to find individual files
        if args.r is None:
            print("ERROR: --r required when using grid file")
            return 1

        dirname = os.path.dirname(os.path.abspath(args.input))
        r_str = f"r{args.r:.4f}".replace(".", ".")  # e.g., r0.2000

        # Find matching files
        pattern = os.path.join(dirname, f"higgsCombine.r{args.r:.4f}*.seed*.HybridNew.*.root")
        toy_files = glob.glob(pattern)

        # Also try without leading zero
        if not toy_files and args.r < 1:
            pattern2 = os.path.join(dirname, f"higgsCombine.r{args.r}*.seed*.HybridNew.*.root")
            toy_files = glob.glob(pattern2)

        if not toy_files:
            print(f"ERROR: No toy files found for r={args.r}")
            print(f"Searched: {pattern}")
            return 1

        print(f"Found {len(toy_files)} toy files for r={args.r}")

        # Merge HypoTestResults
        htr, files = merge_hypotestresults(toy_files)
        if htr is None:
            print("ERROR: Could not merge HypoTestResults")
            return 1

        plot_test_stat_distribution(htr, args.output, args.r, args.title)

        for f in files:
            f.Close()

    return 0


if __name__ == "__main__":
    exit(main())
