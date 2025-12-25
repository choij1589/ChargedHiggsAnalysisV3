#!/usr/bin/env python3
"""
Plot HybridNew grid results: CLs vs r curve using ROOT.

Usage:
    python plotHybridNewGrid.py hybridnew_grid.root -o cls_vs_r.pdf
    python plotHybridNewGrid.py hybridnew_grid.root --rvalues 0.0,0.2,0.4,0.6,0.8,1.0 --nseeds 10
"""
import os
import argparse
import ROOT
import numpy as np
from collections import defaultdict
from array import array

ROOT.gROOT.SetBatch(True)
ROOT.gStyle.SetOptStat(0)


def read_hybridnew_grid(filename, r_values=None, n_seeds=10):
    """
    Read HybridNew grid and extract CLs vs r.

    If r_values is provided, uses those to assign r to entries.
    Otherwise tries to read 'r' branch or infer from structure.
    """
    f = ROOT.TFile.Open(filename)
    if not f or f.IsZombie():
        raise FileNotFoundError(f"Cannot open {filename}")

    tree = f.Get("limit")
    if not tree:
        f.Close()
        raise ValueError("No 'limit' tree found in file")

    n_entries = tree.GetEntries()

    # Check if 'r' branch exists (PyROOT null pointer handling)
    branch_names = [b.GetName() for b in tree.GetListOfBranches()]
    has_r_branch = "r" in branch_names

    # Collect data
    data = defaultdict(list)

    if has_r_branch:
        # Direct read if r branch exists
        for i in range(n_entries):
            tree.GetEntry(i)
            r = tree.r
            data[r].append({
                'limit': tree.limit,
                'limitErr': tree.limitErr,
                'quantile': tree.quantileExpected,
            })
    elif r_values is not None:
        # Assign r values based on known structure
        n_r = len(r_values)
        expected_entries = n_r * n_seeds

        if n_entries != expected_entries:
            print(f"Warning: Expected {expected_entries} entries ({n_r} r-values × {n_seeds} seeds), got {n_entries}")

        for i in range(n_entries):
            tree.GetEntry(i)
            r_idx = i // n_seeds
            if r_idx < n_r:
                r = r_values[r_idx]
                data[r].append({
                    'limit': tree.limit,
                    'limitErr': tree.limitErr,
                    'quantile': tree.quantileExpected,
                })
    else:
        # Try to infer from file structure
        print("No 'r' branch found. Attempting to infer from individual files...")

        # Try reading individual files from the directory
        dirname = os.path.dirname(os.path.abspath(filename))
        import glob
        toy_files = glob.glob(os.path.join(dirname, "higgsCombine.r*.seed*.HybridNew.*.root"))

        if toy_files:
            print(f"Found {len(toy_files)} individual toy files, extracting r values...")
            for toy_file in toy_files:
                basename = os.path.basename(toy_file)
                # Parse r value from filename: higgsCombine.r0.0000.seed1000.HybridNew.mH120.1000.root
                # Split gives: ['higgsCombine', 'r0', '0000', 'seed1000', ...]
                try:
                    parts = basename.split('.')
                    r_str = parts[1] + '.' + parts[2]  # "r0.0000"
                    r = float(r_str[1:])  # Remove 'r' prefix -> 0.0000

                    tf = ROOT.TFile.Open(toy_file)
                    tt = tf.Get("limit")
                    for j in range(tt.GetEntries()):
                        tt.GetEntry(j)
                        data[r].append({
                            'limit': tt.limit,
                            'limitErr': tt.limitErr,
                            'quantile': tt.quantileExpected,
                        })
                    tf.Close()
                except (IndexError, ValueError) as e:
                    print(f"  Could not parse {basename}: {e}")
        else:
            print("No individual toy files found. Please provide --rvalues and --nseeds.")
            f.Close()
            return None

    f.Close()
    return data


def compute_cls_statistics(data):
    """Compute mean and std of CLs for each r point."""
    r_values = []
    cls_mean = []
    cls_std = []

    for r in sorted(data.keys()):
        entries = data[r]
        cls_vals = [e['limit'] for e in entries]

        r_values.append(r)
        cls_mean.append(np.mean(cls_vals))
        cls_std.append(np.std(cls_vals))

    return np.array(r_values), np.array(cls_mean), np.array(cls_std)


def find_limit_crossing(r_values, cls_values, threshold=0.05):
    """Find r value where CLs crosses threshold using linear interpolation."""
    crossings = []

    for i in range(len(r_values) - 1):
        if (cls_values[i] > threshold and cls_values[i+1] <= threshold):
            # Linear interpolation
            r_cross = r_values[i] + (threshold - cls_values[i]) * (r_values[i+1] - r_values[i]) / (cls_values[i+1] - cls_values[i])
            crossings.append(r_cross)

    return crossings[0] if crossings else None


def plot_cls_vs_r(r_values, cls_mean, cls_std, output, title="", threshold=0.05):
    """Plot CLs vs r with 95% CL line using ROOT."""
    n = len(r_values)

    # Create arrays for TGraphErrors
    r_arr = array('d', r_values)
    cls_arr = array('d', cls_mean)
    r_err = array('d', [0.0] * n)
    cls_err = array('d', cls_std)

    # Create canvas
    canvas = ROOT.TCanvas("c", "CLs vs r", 800, 600)
    canvas.SetGrid()
    canvas.SetLeftMargin(0.12)
    canvas.SetRightMargin(0.05)
    canvas.SetBottomMargin(0.12)

    # Create graph with errors
    graph = ROOT.TGraphErrors(n, r_arr, cls_arr, r_err, cls_err)
    graph.SetTitle(title if title else "HybridNew: CL_{s} vs r;Signal strength r;CL_{s}")
    graph.SetMarkerStyle(20)
    graph.SetMarkerSize(1.0)
    graph.SetMarkerColor(ROOT.kBlue)
    graph.SetLineColor(ROOT.kBlue)
    graph.SetLineWidth(2)
    graph.SetFillColor(ROOT.kBlue - 9)
    graph.SetFillStyle(3001)

    # Set axis ranges
    graph.GetXaxis().SetLimits(min(r_values) - 0.05, max(r_values) * 1.05)
    graph.SetMinimum(0.0)
    graph.SetMaximum(1.1)

    # Draw with error band
    graph.Draw("A3")  # Fill area for error band
    graph.Draw("LP SAME")  # Line and points

    # 95% CL line
    line_95 = ROOT.TLine(min(r_values) - 0.05, threshold, max(r_values) * 1.05, threshold)
    line_95.SetLineColor(ROOT.kRed)
    line_95.SetLineStyle(2)
    line_95.SetLineWidth(2)
    line_95.Draw()

    # Find and mark crossing point
    limit = find_limit_crossing(r_values, cls_mean, threshold)

    if limit is not None:
        # Vertical line at limit
        line_limit = ROOT.TLine(limit, 0.0, limit, threshold)
        line_limit.SetLineColor(ROOT.kGreen + 2)
        line_limit.SetLineStyle(3)
        line_limit.SetLineWidth(2)
        line_limit.Draw()

        # Star marker at crossing
        marker = ROOT.TMarker(limit, threshold, 29)
        marker.SetMarkerColor(ROOT.kGreen + 2)
        marker.SetMarkerSize(2.0)
        marker.Draw()

        # Text box with limit value
        latex = ROOT.TLatex()
        latex.SetNDC()
        latex.SetTextSize(0.04)
        latex.SetTextFont(42)
        latex.DrawLatex(0.15, 0.85, f"95% CL Limit: r < {limit:.4f}")

    # Legend
    legend = ROOT.TLegend(0.55, 0.7, 0.9, 0.9)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.SetTextSize(0.035)
    legend.AddEntry(graph, "Observed CL_{s} #pm 1#sigma", "lpf")
    legend.AddEntry(line_95, f"95% CL ({threshold})", "l")
    if limit is not None:
        legend.AddEntry(line_limit, f"Limit: r < {limit:.4f}", "l")
    legend.Draw()

    # Save
    canvas.SaveAs(output)
    print(f"Saved: {output}")

    return limit


def main():
    parser = argparse.ArgumentParser(description="Plot HybridNew grid results")
    parser.add_argument("input", help="Input hybridnew_grid.root file")
    parser.add_argument("-o", "--output", default="cls_vs_r.pdf", help="Output plot file")
    parser.add_argument("--title", default="", help="Plot title")
    parser.add_argument("--rvalues", default=None,
                        help="Comma-separated r values if not stored in tree (e.g., 0.0,0.2,0.4)")
    parser.add_argument("--nseeds", type=int, default=10,
                        help="Number of seeds per r-value (default: 10)")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="CLs threshold for limit (default: 0.05 for 95% CL)")
    args = parser.parse_args()

    # Parse r values if provided
    r_values = None
    if args.rvalues:
        r_values = [float(x) for x in args.rvalues.split(',')]
        print(f"Using provided r values: {r_values}")

    # Read grid
    data = read_hybridnew_grid(args.input, r_values, args.nseeds)

    if data is None or len(data) == 0:
        print("ERROR: Could not read data from grid file.")
        print("\nTry providing r values explicitly:")
        print(f"  python {__file__} {args.input} --rvalues 0.0,0.2,0.4,0.6,0.8,1.0 --nseeds 10")
        return 1

    print(f"\nFound {len(data)} r-value points:")
    for r in sorted(data.keys()):
        print(f"  r={r:.4f}: {len(data[r])} entries, mean CLs={np.mean([e['limit'] for e in data[r]]):.4f}")

    # Compute statistics and plot
    r_arr, cls_mean, cls_std = compute_cls_statistics(data)
    limit = plot_cls_vs_r(r_arr, cls_mean, cls_std, args.output, args.title, args.threshold)

    if limit is not None:
        print(f"\n95% CL Limit: r < {limit:.4f}")
    else:
        print("\nCould not find limit crossing. CLs may not cross threshold in scanned range.")

    return 0


if __name__ == "__main__":
    exit(main())
