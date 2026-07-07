#!/usr/bin/env python3
"""Collect LEE toy fits and compute the global p-value."""

import argparse
import glob
import json
import math
import os
from datetime import datetime, timezone

import ROOT

ROOT.gROOT.SetBatch(ROOT.kTRUE)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masspoint", default="MHc70_MA18",
                        help="Observed maximum-excess mass point and LEE namespace")
    parser.add_argument("--start-toy", type=int, default=1,
                        help="First toy index to collect")
    parser.add_argument("--ntoys", type=int, default=1000,
                        help="Number of toy fit outputs to collect")
    parser.add_argument("--output-dir", default="results/lee",
                        help="Output directory for summary JSON/text and plots")
    parser.add_argument("--debug", action="store_true",
                        help="Print extra diagnostics")
    return parser.parse_args()


def load_lee_masspoints(config_path="configs/masspoints.json"):
    with open(config_path) as handle:
        payload = json.load(handle)
    masspoints = payload.get("LEE", [])
    if not masspoints:
        raise ValueError(f"No 'LEE' masspoint list found in {config_path}")
    return list(masspoints)


def read_observed_significance(masspoint):
    pattern = os.path.join(
        "templates",
        "All",
        "Combined",
        masspoint,
        "Baseline",
        "extended_unblind",
        "combine_output",
        "significance",
        "higgsCombine*.root",
    )
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No observed significance ROOT output found: {pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple observed significance ROOT outputs found: {matches}")

    root_path = matches[0]
    root_file = ROOT.TFile.Open(root_path)
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open observed significance ROOT file: {root_path}")
    try:
        tree = root_file.Get("limit")
        if not tree:
            raise KeyError(f"{root_path} does not contain a 'limit' tree")
        if tree.GetEntries() < 1:
            raise RuntimeError(f"{root_path}: 'limit' tree is empty")
        tree.GetEntry(0)
        z_obs = float(tree.limit)
    finally:
        root_file.Close()

    if not math.isfinite(z_obs):
        raise RuntimeError(f"Observed significance is not finite in {root_path}: {z_obs}")
    return z_obs, root_path


def toy_path(masspoint, toy):
    return os.path.join("LEE", masspoint, "fits", f"toy_{toy:04d}.json")


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def read_toy_fit(path, expected_masspoints):
    with open(path) as handle:
        payload = json.load(handle)

    problems = []
    z_values = payload.get("Z")
    if not isinstance(z_values, dict):
        problems.append("missing Z map")
        z_values = {}

    expected = set(expected_masspoints)
    observed = set(z_values)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        problems.append(f"missing Z entries: {', '.join(missing)}")
    if extra:
        problems.append(f"unexpected Z entries: {', '.join(extra)}")

    nonfinite = sorted(mp for mp in expected if mp in z_values and not finite_number(z_values[mp]))
    if nonfinite:
        problems.append(f"non-finite Z entries: {', '.join(nonfinite)}")

    failure_count = payload.get("failure_count", 0)
    if failure_count != 0:
        problems.append(f"failure_count={failure_count}")

    failed_masspoints = payload.get("failed_masspoints", [])
    if failed_masspoints:
        problems.append(f"failed_masspoints={failed_masspoints}")

    if problems:
        return None, problems

    values = [float(z_values[mp]) for mp in expected_masspoints]
    z_max = max(values)
    return {
        "toy": int(payload.get("toy", os.path.basename(path)[4:8])),
        "seed": payload.get("seed"),
        "Z_max": z_max,
        "Z_max_from_file": payload.get("Z_max"),
    }, []


def pvalue_uncertainty(p_value, n_toys):
    if n_toys <= 0:
        return None
    return math.sqrt(p_value * (1.0 - p_value) / n_toys)


def z_global_from_pvalue(p_value):
    if p_value >= 0.5:
        return None
    return float(ROOT.Math.normal_quantile_c(p_value, 1.0))


def write_text_summary(path, summary):
    lines = [
        "LEE global p-value summary",
        "==========================",
        f"masspoint: {summary['masspoint']}",
        f"observed Z: {summary['observed']['Z_obs']:.6f}",
        f"observed source: {summary['observed']['source']}",
        f"valid toys: {summary['toys']['valid']}",
        f"exceeding toys: {summary['global_pvalue']['n_exceed']}",
        (
            "p_global: "
            f"{summary['global_pvalue']['p']:.6f} +/- "
            f"{summary['global_pvalue']['uncertainty']:.6f}"
        ),
    ]
    if summary["global_pvalue"]["Z_global"] is not None:
        lines.append(f"Z_global: {summary['global_pvalue']['Z_global']:.6f}")
    else:
        lines.append("Z_global: not quoted because p_global >= 0.5")

    with open(path, "w") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def draw_zmax_distribution(zmax_values, z_obs, output_base):
    if not zmax_values:
        raise ValueError("No Z_max values to plot")

    z_min = min(zmax_values + [z_obs])
    z_max = max(zmax_values + [z_obs])
    if z_min == z_max:
        z_min -= 1.0
        z_max += 1.0
    padding = max(0.2, 0.10 * (z_max - z_min))
    xmin = z_min - padding
    xmax = z_max + padding

    nbins = min(60, max(20, int(math.sqrt(len(zmax_values)) * 2)))
    hist = ROOT.TH1D("h_lee_zmax", "", nbins, xmin, xmax)
    for value in zmax_values:
        hist.Fill(value)

    canvas = ROOT.TCanvas("c_lee_zmax", "LEE Zmax", 800, 700)
    hist.SetStats(False)
    hist.SetLineColor(ROOT.kAzure + 1)
    hist.SetFillColorAlpha(ROOT.kAzure + 1, 0.35)
    hist.SetLineWidth(2)
    hist.GetXaxis().SetTitle("max Z per toy")
    hist.GetYaxis().SetTitle("Toys")
    hist.Draw("HIST")

    ymax = hist.GetMaximum() * 1.25
    hist.SetMaximum(ymax)
    line = ROOT.TLine(z_obs, 0.0, z_obs, ymax * 0.92)
    line.SetLineColor(ROOT.kRed + 1)
    line.SetLineWidth(3)
    line.SetLineStyle(2)
    line.Draw("SAME")

    legend = ROOT.TLegend(0.56, 0.74, 0.88, 0.88)
    legend.SetBorderSize(0)
    legend.SetFillStyle(0)
    legend.AddEntry(hist, f"B-only toys ({len(zmax_values)})", "f")
    legend.AddEntry(line, f"Observed Z = {z_obs:.3f}", "l")
    legend.Draw()

    canvas.SaveAs(f"{output_base}.png")
    canvas.SaveAs(f"{output_base}.pdf")


def main():
    args = parse_args()
    if args.start_toy < 1:
        raise ValueError("--start-toy must be >= 1")
    if args.ntoys < 1:
        raise ValueError("--ntoys must be >= 1")

    masspoints = load_lee_masspoints()
    if args.masspoint not in masspoints:
        raise ValueError(f"{args.masspoint} is not in configs/masspoints.json:LEE")

    z_obs, observed_path = read_observed_significance(args.masspoint)
    last_toy = args.start_toy + args.ntoys - 1

    valid_toys = []
    invalid_toys = []
    missing_toys = []
    for toy in range(args.start_toy, last_toy + 1):
        path = toy_path(args.masspoint, toy)
        if not os.path.exists(path):
            missing_toys.append({"toy": toy, "path": path})
            continue
        record, problems = read_toy_fit(path, masspoints)
        if problems:
            invalid_toys.append({"toy": toy, "path": path, "problems": problems})
            continue
        valid_toys.append(record)

    if missing_toys or invalid_toys:
        print("ERROR: LEE toy collection is incomplete.")
        print(f"  Missing toys: {len(missing_toys)}")
        print(f"  Invalid toys: {len(invalid_toys)}")
        for item in missing_toys[:10]:
            print(f"  missing toy {item['toy']}: {item['path']}")
        for item in invalid_toys[:10]:
            print(f"  invalid toy {item['toy']}: {'; '.join(item['problems'])}")
        raise SystemExit(1)

    zmax_values = [record["Z_max"] for record in valid_toys]
    n_valid = len(valid_toys)
    n_exceed = sum(1 for value in zmax_values if value >= z_obs)
    p_global = (1.0 + n_exceed) / (n_valid + 1.0)
    p_unc = pvalue_uncertainty(p_global, n_valid)
    z_global = z_global_from_pvalue(p_global)

    output_dir = args.output_dir
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "masspoint": args.masspoint,
        "method": "Baseline",
        "era": "All",
        "channel": "Combined",
        "binning": "extended_unblind",
        "trials_set": {
            "source": "configs/masspoints.json:LEE",
            "n_masspoints": len(masspoints),
            "masspoints": masspoints,
        },
        "observed": {
            "Z_obs": z_obs,
            "source": observed_path,
        },
        "toys": {
            "start_toy": args.start_toy,
            "last_toy": last_toy,
            "expected": args.ntoys,
            "valid": n_valid,
            "missing": 0,
            "invalid": 0,
        },
        "global_pvalue": {
            "side": "excess",
            "estimator": "(1 + N_exceed) / (N_valid + 1)",
            "n_exceed": n_exceed,
            "p": p_global,
            "uncertainty": p_unc,
            "Z_global": z_global,
        },
        "zmax_distribution": {
            "min": min(zmax_values),
            "median": sorted(zmax_values)[n_valid // 2],
            "max": max(zmax_values),
        },
    }

    json_path = os.path.join(output_dir, "global_pvalue.json")
    text_path = os.path.join(output_dir, "global_pvalue.txt")
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_text_summary(text_path, summary)
    draw_zmax_distribution(zmax_values, z_obs, os.path.join(plot_dir, "zmax_distribution"))

    print(f"Wrote {json_path}")
    print(f"Wrote {text_path}")
    print(f"Wrote {plot_dir}/zmax_distribution.png")
    print(
        "LEE p_global(excess) = "
        f"{p_global:.6f} +/- {p_unc:.6f} "
        f"({n_exceed}/{n_valid} toys with Z_max >= {z_obs:.6f})"
    )
    if z_global is not None:
        print(f"Equivalent Z_global = {z_global:.6f}")


if __name__ == "__main__":
    main()
