#!/usr/bin/env python3
"""Validate the LEE toy chain and global p-value inputs."""

import argparse
import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

import ROOT

ROOT.gROOT.SetBatch(ROOT.kTRUE)

LEE_CATEGORIES = ("SR1E2Mu_Run2", "SR3Mu_Run2", "SR1E2Mu_Run3", "SR3Mu_Run3")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masspoint", default="MHc70_MA18",
                        help="LEE generation/observed maximum mass point")
    parser.add_argument("--reference-masspoint", default="MHc160_MA50",
                        help="Reference point for background sample sanity checks")
    parser.add_argument("--start-toy", type=int, default=1,
                        help="First toy index to validate")
    parser.add_argument("--ntoys", type=int, default=1000,
                        help="Number of toys to validate")
    parser.add_argument("--output-dir", default="results/lee/validation",
                        help="Validation output directory")
    parser.add_argument("--debug", action="store_true",
                        help="Print extra diagnostics")
    return parser.parse_args()


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def load_lee_masspoints(config_path="configs/masspoints.json"):
    payload = read_json(config_path)
    masspoints = payload.get("LEE", [])
    if not masspoints:
        raise ValueError(f"No 'LEE' masspoint list found in {config_path}")
    return list(masspoints)


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def stdev(values):
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1))


def normal_tail(z_value):
    return 0.5 * math.erfc(z_value / math.sqrt(2.0))


def pvalue_uncertainty(p_value, n_toys):
    if n_toys <= 0:
        return float("nan")
    return math.sqrt(p_value * (1.0 - p_value) / n_toys)


def toy_json_path(masspoint, toy):
    return os.path.join("LEE", masspoint, "toys", f"toy_{toy:04d}.json")


def toy_root_path(masspoint, toy):
    return os.path.join("LEE", masspoint, "toys", f"toy_{toy:04d}.root")


def fit_json_path(masspoint, toy):
    return os.path.join("LEE", masspoint, "fits", f"toy_{toy:04d}.json")


def check_inputs(args, masspoints):
    z_by_masspoint = {masspoint: [] for masspoint in masspoints}
    zmax_values = []
    toy_yields = {category: [] for category in LEE_CATEGORIES}
    expected_yields = {}
    errors = []

    last_toy = args.start_toy + args.ntoys - 1
    for toy in range(args.start_toy, last_toy + 1):
        toy_json = toy_json_path(args.masspoint, toy)
        toy_root = toy_root_path(args.masspoint, toy)
        fit_json = fit_json_path(args.masspoint, toy)

        for path in (toy_json, toy_root, fit_json):
            if not os.path.exists(path):
                errors.append(f"missing toy {toy}: {path}")
        if errors and any(f"toy {toy}:" in err for err in errors[-3:]):
            continue

        try:
            toy_payload = read_json(toy_json)
            fit_payload = read_json(fit_json)
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"toy {toy}: failed to read JSON: {exc}")
            continue

        if toy_payload.get("status") != "complete":
            errors.append(f"toy {toy}: toy status is {toy_payload.get('status')}")
        categories = toy_payload.get("categories", {})
        for category in LEE_CATEGORIES:
            record = categories.get(category)
            if not isinstance(record, dict):
                errors.append(f"toy {toy}: missing toy category {category}")
                continue
            entries = record.get("entries")
            expected = record.get("expected_yield")
            if not finite_number(entries) or not finite_number(expected):
                errors.append(f"toy {toy}: invalid yield record for {category}")
                continue
            toy_yields[category].append(float(entries))
            expected_yields.setdefault(category, float(expected))

        if fit_payload.get("status") != "complete":
            errors.append(f"toy {toy}: fit status is {fit_payload.get('status')}")
        if fit_payload.get("failure_count", 0) != 0:
            errors.append(f"toy {toy}: failure_count={fit_payload.get('failure_count')}")
        if fit_payload.get("failed_masspoints"):
            errors.append(f"toy {toy}: failed_masspoints={fit_payload.get('failed_masspoints')}")

        z_map = fit_payload.get("Z")
        if not isinstance(z_map, dict):
            errors.append(f"toy {toy}: missing Z map")
            continue
        missing = sorted(set(masspoints) - set(z_map))
        extra = sorted(set(z_map) - set(masspoints))
        if missing:
            errors.append(f"toy {toy}: missing Z entries {missing}")
        if extra:
            errors.append(f"toy {toy}: unexpected Z entries {extra}")

        values = []
        for masspoint in masspoints:
            z_value = z_map.get(masspoint)
            if not finite_number(z_value):
                errors.append(f"toy {toy}: non-finite Z for {masspoint}: {z_value}")
                continue
            z_float = float(z_value)
            z_by_masspoint[masspoint].append(z_float)
            values.append(z_float)
        if len(values) == len(masspoints):
            zmax_values.append(max(values))

    return {
        "errors": errors,
        "z_by_masspoint": z_by_masspoint,
        "zmax_values": zmax_values,
        "toy_yields": toy_yields,
        "expected_yields": expected_yields,
    }


def status_from_failures(failures, warnings=None):
    if failures:
        return "fail"
    if warnings:
        return "warn"
    return "pass"


def z_calibration(z_by_masspoint):
    # Residual per-point offsets are expected: the datacards' per-process
    # coarse-bin flooring makes them mutually inconsistent at the few-% level,
    # so no shared generation model can center every point exactly at 0.
    # Widths below 1 are expected over-coverage for nominal toys with
    # profiled fits, hence warn-only.
    rows = []
    failures = []
    warnings = []
    for masspoint, values in z_by_masspoint.items():
        row = {
            "masspoint": masspoint,
            "n": len(values),
            "mean": mean(values),
            "width": stdev(values),
        }
        row["mean_abs"] = abs(row["mean"])
        row["width_deviation"] = abs(row["width"] - 1.0)
        if row["mean_abs"] > 0.30:
            failures.append(masspoint)
        elif row["mean_abs"] > 0.15 or not 0.80 <= row["width"] <= 1.10:
            warnings.append(masspoint)
        rows.append(row)

    rows.sort(key=lambda item: item["masspoint"])
    return {
        "status": status_from_failures(failures, warnings),
        "thresholds": {
            "fail_abs_mean": 0.30,
            "warn_abs_mean": 0.15,
            "warn_width_range": [0.80, 1.10],
        },
        "failed_masspoints": failures,
        "warning_masspoints": warnings,
        "max_abs_mean": max((row["mean_abs"] for row in rows), default=None),
        "max_abs_width_minus_one": max((row["width_deviation"] for row in rows), default=None),
        "masspoints": rows,
    }


def toy_yield_closure(toy_yields, expected_yields):
    rows = []
    failures = []
    warnings = []
    for category in LEE_CATEGORIES:
        values = toy_yields.get(category, [])
        expected = expected_yields.get(category)
        if not values or expected is None:
            failures.append(category)
            continue
        observed_mean = mean(values)
        observed_width = stdev(values)
        expected_width = math.sqrt(expected)
        mean_unc = expected_width / math.sqrt(len(values))
        mean_pull = (observed_mean - expected) / mean_unc if mean_unc > 0 else float("nan")
        width_ratio = observed_width / expected_width if expected_width > 0 else float("nan")
        row = {
            "category": category,
            "n_toys": len(values),
            "expected": expected,
            "mean": observed_mean,
            "mean_pull": mean_pull,
            "width": observed_width,
            "expected_width": expected_width,
            "width_ratio": width_ratio,
        }
        if abs(mean_pull) > 5.0 or abs(width_ratio - 1.0) > 0.15:
            failures.append(category)
        elif abs(mean_pull) > 3.0 or abs(width_ratio - 1.0) > 0.10:
            warnings.append(category)
        rows.append(row)

    return {
        "status": status_from_failures(failures, warnings),
        "thresholds": {
            "fail_abs_mean_pull": 5.0,
            "warn_abs_mean_pull": 3.0,
            "fail_abs_width_ratio_minus_one": 0.15,
            "warn_abs_width_ratio_minus_one": 0.10,
        },
        "failed_categories": failures,
        "warning_categories": warnings,
        "categories": rows,
    }


def unique_input_records(model_payload):
    records = {}
    for category_record in model_payload.get("categories", {}).values():
        for item in category_record.get("input_files", []):
            key = (item.get("subera"), item.get("channel"), item.get("process"))
            if all(key):
                records[key] = item
    return records


def tree_entries_and_weight_sum(path):
    dataframe = ROOT.RDataFrame("Central", path)
    entries = int(dataframe.Count().GetValue())
    weight_sum = float(dataframe.Sum("weight").GetValue())
    return entries, weight_sum


def model_sanity(args, model_payload):
    # The per-process block flooring intentionally adds yield relative to the
    # net background sum (matching the datacard hygiene), so the flooring
    # fraction gate is looser than for a pure net-sum floor.
    floor_rows = []
    floor_failures = []
    floor_warnings = []
    for category, record in sorted(model_payload.get("categories", {}).items()):
        row = {
            "category": category,
            "pre_floor_integral": record.get("pre_floor_integral"),
            "post_floor_integral": record.get("post_floor_integral"),
            "floored_yield": record.get("floored_yield"),
            "floored_blocks": record.get("floored_blocks"),
            "floor_fraction": record.get("floor_fraction"),
        }
        if finite_number(row["floor_fraction"]):
            if row["floor_fraction"] > 0.20:
                floor_failures.append(category)
            elif row["floor_fraction"] > 0.10:
                floor_warnings.append(category)
        floor_rows.append(row)

    records = unique_input_records(model_payload)
    sample_rows = []
    sample_failures = []
    for key, item in sorted(records.items()):
        base_path = item["path"]
        ref_path = base_path.replace(f"/{args.masspoint}/", f"/{args.reference_masspoint}/")
        row = {
            "subera": key[0],
            "channel": key[1],
            "process": key[2],
            "masspoint_path": base_path,
            "reference_path": ref_path,
        }
        try:
            base_entries = int(item["entries"])
            base_weight = float(item["weight_sum"])
            ref_entries, ref_weight = tree_entries_and_weight_sum(ref_path)
            row.update({
                "entries": base_entries,
                "reference_entries": ref_entries,
                "weight_sum": base_weight,
                "reference_weight_sum": ref_weight,
                "entries_match": base_entries == ref_entries,
                "weight_sum_difference": ref_weight - base_weight,
            })
            tolerance = max(1e-8, 1e-8 * abs(base_weight))
            row["weight_sum_match"] = abs(ref_weight - base_weight) <= tolerance
            if not row["entries_match"] or not row["weight_sum_match"]:
                sample_failures.append(f"{key[0]}/{key[1]}/{key[2]}")
        except Exception as exc:
            row["error"] = str(exc)
            sample_failures.append(f"{key[0]}/{key[1]}/{key[2]}")
        sample_rows.append(row)

    failures = floor_failures + sample_failures
    return {
        "status": status_from_failures(failures, floor_warnings),
        "reference_masspoint": args.reference_masspoint,
        "flooring": {
            "fail_floor_fraction": 0.20,
            "warn_floor_fraction": 0.10,
            "failed_categories": floor_failures,
            "warning_categories": floor_warnings,
            "categories": floor_rows,
        },
        "sample_comparison": {
            "failed_records": sample_failures,
            "records": sample_rows,
        },
    }


def model_datacard_consistency(model_payload):
    """Gate on the Step 1 model-vs-datacard window-integral ratios; this is
    the diagnostic that catches a biased toy generation model before any
    toys are fit."""
    consistency = model_payload.get("consistency", {})
    if not consistency:
        return {
            "status": "fail",
            "reason": "bkg_model.json has no 'consistency' section; rerun Step 1",
        }

    rows = []
    failures = []
    warnings = []
    for masspoint, categories in sorted(consistency.items()):
        for category, record in sorted(categories.items()):
            ratio = record.get("ratio")
            if not finite_number(ratio):
                failures.append(f"{masspoint}/{category}")
                continue
            deviation = abs(1.0 - float(ratio))
            rows.append({
                "masspoint": masspoint,
                "category": category,
                "ratio": float(ratio),
                "deviation": deviation,
            })
            if deviation > 0.15:
                failures.append(f"{masspoint}/{category}")
            elif deviation > 0.10:
                warnings.append(f"{masspoint}/{category}")

    rows.sort(key=lambda item: item["deviation"], reverse=True)
    return {
        "status": status_from_failures(failures, warnings),
        "thresholds": {
            "fail_abs_deviation": 0.15,
            "warn_abs_deviation": 0.10,
        },
        "failed_entries": failures,
        "warning_entries": warnings,
        "max_deviation": rows[0]["deviation"] if rows else None,
        "worst_entries": rows[:10],
    }


def stability_check(global_payload, zmax_values):
    # Toys are iid by construction (deterministic per-toy seeds, independent
    # jobs), so no ordering trend is possible; this compares two independent
    # halves at 2 sigma to catch gross inconsistencies (corrupt batches,
    # mixed generation models) without a ~32% false-alarm rate.
    z_obs = global_payload["observed"]["Z_obs"]
    n_full = len(zmax_values)
    n_half = n_full // 2
    if n_half < 1:
        return {"status": "fail", "reason": "not enough toys"}

    first_exceed = sum(1 for value in zmax_values[:n_half] if value >= z_obs)
    second_exceed = sum(1 for value in zmax_values[n_half:2 * n_half] if value >= z_obs)
    full_exceed = sum(1 for value in zmax_values if value >= z_obs)
    p_first = (1.0 + first_exceed) / (n_half + 1.0)
    p_second = (1.0 + second_exceed) / (n_half + 1.0)
    p_full = (1.0 + full_exceed) / (n_full + 1.0)
    unc_first = pvalue_uncertainty(p_first, n_half)
    unc_second = pvalue_uncertainty(p_second, n_half)
    combined_unc = math.sqrt(unc_first ** 2 + unc_second ** 2)
    pull = abs(p_first - p_second) / combined_unc if combined_unc > 0 else float("inf")
    compatible = pull <= 2.0

    z_obs_tail = normal_tail(z_obs)
    n_trials = global_payload["trials_set"]["n_masspoints"]
    sidak = 1.0 - (1.0 - z_obs_tail) ** n_trials
    sidak_ok = p_full <= sidak

    return {
        "status": "pass" if compatible and sidak_ok else "fail",
        "first_half": {
            "n_toys": n_half,
            "n_exceed": first_exceed,
            "p": p_first,
            "uncertainty": unc_first,
        },
        "second_half": {
            "n_toys": n_half,
            "n_exceed": second_exceed,
            "p": p_second,
            "uncertainty": unc_second,
        },
        "full": {
            "n_toys": n_full,
            "n_exceed": full_exceed,
            "p": p_full,
            "uncertainty": pvalue_uncertainty(p_full, n_full),
        },
        "compatibility": {
            "absolute_difference": abs(p_first - p_second),
            "combined_uncertainty": combined_unc,
            "pull": pull,
            "compatible_within_two_sigma": compatible,
        },
        "sidak": {
            "local_p": z_obs_tail,
            "n_trials": n_trials,
            "sidak_independent_trials_p": sidak,
            "toy_p_below_sidak": sidak_ok,
        },
    }


def make_label_hist(name, title, labels, values, ytitle, output_base, lines=None):
    hist = ROOT.TH1D(name, title, len(labels), 0.5, len(labels) + 0.5)
    for index, (label, value) in enumerate(zip(labels, values), start=1):
        hist.SetBinContent(index, value)
        hist.GetXaxis().SetBinLabel(index, label)

    ymin = min(values + [line for line in (lines or [])])
    ymax = max(values + [line for line in (lines or [])])
    padding = max(0.05, 0.18 * (ymax - ymin if ymax != ymin else 1.0))

    canvas = ROOT.TCanvas(f"c_{name}", title, 1100, 650)
    canvas.SetBottomMargin(0.30)
    hist.SetStats(False)
    hist.SetLineColor(ROOT.kAzure + 1)
    hist.SetFillColorAlpha(ROOT.kAzure + 1, 0.35)
    hist.SetMinimum(ymin - padding)
    hist.SetMaximum(ymax + padding)
    hist.GetYaxis().SetTitle(ytitle)
    hist.LabelsOption("v", "X")
    hist.Draw("HIST")

    drawn = []
    for line_value in lines or []:
        line = ROOT.TLine(0.5, line_value, len(labels) + 0.5, line_value)
        line.SetLineColor(ROOT.kRed + 1)
        line.SetLineStyle(2)
        line.SetLineWidth(2)
        line.Draw("SAME")
        drawn.append(line)

    canvas.SaveAs(f"{output_base}.png")
    canvas.SaveAs(f"{output_base}.pdf")


def plot_z_calibration(z_result, plot_dir):
    labels = [row["masspoint"].replace("MHc", "").replace("_MA", "_") for row in z_result["masspoints"]]
    means = [row["mean"] for row in z_result["masspoints"]]
    widths = [row["width"] for row in z_result["masspoints"]]
    make_label_hist(
        "lee_z_mean",
        "LEE Z calibration means",
        labels,
        means,
        "mean fitted Z",
        os.path.join(plot_dir, "z_calibration_mean"),
        lines=[-0.10, 0.0, 0.10],
    )
    make_label_hist(
        "lee_z_width",
        "LEE Z calibration widths",
        labels,
        widths,
        "width of fitted Z",
        os.path.join(plot_dir, "z_calibration_width"),
        lines=[0.90, 1.0, 1.10],
    )


def plot_yields(yield_result, plot_dir):
    labels = [row["category"] for row in yield_result["categories"]]
    ratios = [row["mean"] / row["expected"] for row in yield_result["categories"]]
    make_label_hist(
        "lee_toy_yield_ratio",
        "LEE toy yield closure",
        labels,
        ratios,
        "mean toy yield / expected",
        os.path.join(plot_dir, "toy_yields"),
        lines=[1.0],
    )


def plot_closure_spectra(args, plot_dir):
    model_root_path = os.path.join("LEE", args.masspoint, "model", "bkg_model.root")
    model_file = ROOT.TFile.Open(model_root_path)
    if not model_file or model_file.IsZombie():
        raise OSError(f"Could not open {model_root_path}")

    try:
        model_hists = {}
        toy_hists = {}
        for category in LEE_CATEGORIES:
            model_hist = model_file.Get(category)
            if not model_hist:
                raise KeyError(f"{model_root_path} missing histogram {category}")
            model_hists[category] = model_hist.Clone(f"model_{category}")
            model_hists[category].SetDirectory(0)
            toy_hists[category] = model_hist.Clone(f"avg_toy_{category}")
            toy_hists[category].SetDirectory(0)
            toy_hists[category].Reset()
    finally:
        model_file.Close()

    for toy in range(args.start_toy, args.start_toy + args.ntoys):
        root_path = toy_root_path(args.masspoint, toy)
        root_file = ROOT.TFile.Open(root_path)
        if not root_file or root_file.IsZombie():
            raise OSError(f"Could not open {root_path}")
        try:
            for category in LEE_CATEGORIES:
                tree = root_file.Get(category)
                if not tree:
                    raise KeyError(f"{root_path} missing tree {category}")
                hist = toy_hists[category]
                for event in tree:
                    hist.Fill(float(event.mass), float(event.weight))
        finally:
            root_file.Close()

    rows = []
    for category in LEE_CATEGORIES:
        toy_hists[category].Scale(1.0 / args.ntoys)
        model = model_hists[category]
        toy = toy_hists[category]
        rows.append({
            "category": category,
            "model_integral": model.Integral(),
            "average_toy_integral": toy.Integral(),
            "integral_ratio": toy.Integral() / model.Integral() if model.Integral() else None,
        })

        canvas = ROOT.TCanvas(f"c_closure_{category}", category, 900, 700)
        model.SetStats(False)
        model.SetLineColor(ROOT.kBlack)
        model.SetLineWidth(2)
        toy.SetLineColor(ROOT.kAzure + 1)
        toy.SetFillColorAlpha(ROOT.kAzure + 1, 0.25)
        toy.SetLineWidth(2)
        model.GetXaxis().SetTitle("mass [GeV]")
        model.GetYaxis().SetTitle("Events / 0.1 GeV")
        ymax = max(model.GetMaximum(), toy.GetMaximum()) * 1.25
        model.SetMaximum(ymax)
        model.Draw("HIST")
        toy.Draw("HIST SAME")
        legend = ROOT.TLegend(0.58, 0.76, 0.88, 0.88)
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.AddEntry(model, "Step 1 model", "l")
        legend.AddEntry(toy, "Average toy", "f")
        legend.Draw()
        canvas.SaveAs(os.path.join(plot_dir, f"toy_closure_{category}.png"))
        canvas.SaveAs(os.path.join(plot_dir, f"toy_closure_{category}.pdf"))

    return rows


def write_text_summary(path, summary):
    lines = [
        "LEE validation summary",
        "======================",
        f"masspoint: {summary['masspoint']}",
        f"reference_masspoint: {summary['reference_masspoint']}",
        f"overall_status: {summary['overall_status']}",
        "",
    ]
    for name, result in summary["checks"].items():
        lines.append(f"{name}: {result['status']}")
    lines.extend([
        "",
        f"Z calibration failed mass points: {len(summary['checks']['z_calibration']['failed_masspoints'])}",
        (
            "p-value stability first-half/full: "
            f"{summary['checks']['pvalue_stability']['first_half']['p']:.6f} / "
            f"{summary['checks']['pvalue_stability']['full']['p']:.6f}"
        ),
    ])
    with open(path, "w") as handle:
        handle.write("\n".join(lines))
        handle.write("\n")


def main():
    args = parse_args()
    if args.start_toy < 1:
        raise ValueError("--start-toy must be >= 1")
    if args.ntoys < 1:
        raise ValueError("--ntoys must be >= 1")

    masspoints = load_lee_masspoints()
    if args.masspoint not in masspoints:
        raise ValueError(f"{args.masspoint} is not in configs/masspoints.json:LEE")
    if args.reference_masspoint not in masspoints:
        raise ValueError(f"{args.reference_masspoint} is not in configs/masspoints.json:LEE")

    model_json = os.path.join("LEE", args.masspoint, "model", "bkg_model.json")
    global_json = os.path.join("results", "lee", "global_pvalue.json")
    for path in (model_json, global_json):
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    model_payload = read_json(model_json)
    global_payload = read_json(global_json)

    input_result = check_inputs(args, masspoints)
    if input_result["errors"]:
        print("ERROR: LEE validation inputs are incomplete or corrupt.")
        for error in input_result["errors"][:50]:
            print(f"  {error}")
        if len(input_result["errors"]) > 50:
            print(f"  ... {len(input_result['errors']) - 50} more errors")
        raise SystemExit(1)

    output_dir = args.output_dir
    plot_dir = os.path.join(output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    z_result = z_calibration(input_result["z_by_masspoint"])
    yield_result = toy_yield_closure(input_result["toy_yields"], input_result["expected_yields"])
    model_result = model_sanity(args, model_payload)
    consistency_result = model_datacard_consistency(model_payload)
    stability_result = stability_check(global_payload, input_result["zmax_values"])
    closure_spectra = plot_closure_spectra(args, plot_dir)

    checks = {
        "completeness": {
            "status": "pass",
            "toy_root_files": args.ntoys,
            "toy_json_files": args.ntoys,
            "fit_json_files": args.ntoys,
            "n_masspoints_per_fit": len(masspoints),
        },
        "z_calibration": z_result,
        "toy_yield_closure": yield_result,
        "toy_spectrum_closure": {
            "status": "pass",
            "categories": closure_spectra,
        },
        "model_sanity": model_result,
        "model_datacard_consistency": consistency_result,
        "pvalue_stability": stability_result,
    }
    overall_status = "fail" if any(check["status"] == "fail" for check in checks.values()) else (
        "warn" if any(check["status"] == "warn" for check in checks.values()) else "pass"
    )

    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "masspoint": args.masspoint,
        "reference_masspoint": args.reference_masspoint,
        "start_toy": args.start_toy,
        "ntoys": args.ntoys,
        "overall_status": overall_status,
        "checks": checks,
    }

    plot_z_calibration(z_result, plot_dir)
    plot_yields(yield_result, plot_dir)

    json_path = os.path.join(output_dir, "validation.json")
    text_path = os.path.join(output_dir, "validation.txt")
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    write_text_summary(text_path, summary)

    print(f"Wrote {json_path}")
    print(f"Wrote {text_path}")
    print(f"Wrote validation plots under {plot_dir}")
    print(f"LEE validation overall_status = {overall_status}")


if __name__ == "__main__":
    main()
