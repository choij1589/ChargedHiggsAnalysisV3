#!/usr/bin/env python3
"""Selected-dimuon pT cut yield scan for 2018 signal-region samples.

The scan reads baseline-selected SKNano output trees directly. For each
masspoint and signal-region channel it selects the same dimuon candidate used
by SignalRegionStudyV3, derives a V3-style signal mass window, then scans
pt_selected >= threshold in 5 GeV steps.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import ROOT


ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

ERA = "2018"
CHANNEL_INPUT = OrderedDict(
    [
        ("SR1E2Mu", "Run1E2Mu"),
        ("SR3Mu", "Run3Mu"),
    ]
)
SIGNAL_POINTS = [
    "MHc70_MA15",
    "MHc100_MA60",
    "MHc130_MA90",
    "MHc160_MA155",
]
BACKGROUND_CATEGORIES = [
    "nonprompt",
    "WZ",
    "ZZ",
    "ttW",
    "ttZ",
    "ttH",
    "tZq",
    "conversion",
    "others",
]
REQUIRED_BRANCHES = ["mass1", "mass2", "pT1", "pT2", "weight"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_repo = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--repo-root",
        default=str(default_repo),
        help="ChargedHiggsAnalysisV3 repository root.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/2018",
        help="Output directory relative to SignalOpimize, unless absolute.",
    )
    parser.add_argument(
        "--plot-dir",
        default="plots/2018/pt_cut_scan",
        help="Plot output directory relative to SignalOpimize, unless absolute.",
    )
    parser.add_argument("--pt-min", default=0.0, type=float, help="Minimum threshold in GeV.")
    parser.add_argument("--pt-max", default=500.0, type=float, help="Maximum threshold in GeV.")
    parser.add_argument("--pt-step", default=5.0, type=float, help="Threshold step in GeV.")
    parser.add_argument(
        "--masspoints",
        nargs="+",
        default=SIGNAL_POINTS,
        help="Masspoints to scan, without the TTToHcToWAToMuMu- prefix.",
    )
    parser.add_argument("--debug", action="store_true", help="Print detailed fit information.")
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    with path.open() as handle:
        return json.load(handle)


def load_samplegroups(repo_root: Path) -> Dict:
    return load_json(repo_root / "SignalRegionStudyV3" / "configs" / "samplegroups.json")


def load_conv_sf(repo_root: Path, channel: str) -> float:
    payload = load_json(repo_root / "Common" / "Data" / "ConvSF.json")
    channel_key = channel.replace("SR", "")
    return float(payload[channel_key][ERA]["central"])


def load_kfactors(repo_root: Path) -> Dict[str, float]:
    payload = load_json(repo_root / "Common" / "Data" / "KFactors.json")
    return {
        sample: float(info["kFactor"])
        for sample, info in payload.get("Run2", {}).items()
        if "kFactor" in info
    }


def thresholds(pt_min: float, pt_max: float, pt_step: float) -> List[float]:
    if pt_step <= 0:
        raise ValueError("--pt-step must be positive")
    n_steps = int(round((pt_max - pt_min) / pt_step))
    values = [pt_min + i * pt_step for i in range(n_steps + 1)]
    return [round(value, 6) for value in values if value <= pt_max + 1e-9]


def masspoint_masses(masspoint: str) -> Tuple[int, int]:
    mhc = int(masspoint.split("_")[0].replace("MHc", ""))
    ma = int(masspoint.split("_")[1].replace("MA", ""))
    return mhc, ma


def selected_exprs(channel: str, masspoint: str) -> Tuple[str, str]:
    """Return selected mass and selected pT expressions for RDataFrame."""
    if channel == "SR1E2Mu":
        return "mass1", "pT1"

    mhc, ma = masspoint_masses(masspoint)
    if mhc >= 100 and ma >= 60:
        return (
            "(mass1 >= mass2 ? mass1 : mass2)",
            "(mass1 >= mass2 ? pT1 : pT2)",
        )
    return (
        "(mass1 < mass2 ? mass1 : mass2)",
        "(mass1 < mass2 ? pT1 : pT2)",
    )


def tree_branches(path: Path, tree_name: str) -> Tuple[bool, List[str], str, int]:
    if not path.exists():
        return False, [], "missing_file", 0

    root_file = ROOT.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie():
        return False, [], "open_failed", 0

    tree = root_file.Get(tree_name)
    if not tree:
        root_file.Close()
        return False, [], "missing_tree", 0

    branches = [branch.GetName() for branch in tree.GetListOfBranches()]
    entries = int(tree.GetEntries())
    root_file.Close()
    return True, branches, "ok", entries


def audit_file(path: Path, tree_name: str, required: Iterable[str]) -> Dict:
    ok, branches, status, entries = tree_branches(path, tree_name)
    missing = [] if status != "ok" else [name for name in required if name not in branches]
    return {
        "path": str(path),
        "tree": tree_name,
        "status": "ok" if ok and not missing else ("missing_branch" if missing else status),
        "missing_branches": missing,
        "entries": entries,
        "n_branches": len(branches),
    }


def signal_path(repo_root: Path, channel: str, masspoint: str) -> Path:
    input_channel = CHANNEL_INPUT[channel]
    return (
        repo_root
        / "SKNanoOutput"
        / "PromptAnalyzer"
        / f"{input_channel}_RunSyst_RunTheoryUnc"
        / ERA
        / f"TTToHcToWAToMuMu-{masspoint}.root"
    )


def background_path(repo_root: Path, channel: str, category: str, sample: str) -> Tuple[Path, str]:
    input_channel = CHANNEL_INPUT[channel]
    if category == "nonprompt":
        return (
            repo_root
            / "SKNanoOutput"
            / "MatrixAnalyzer"
            / input_channel
            / ERA
            / f"Skim_TriLep_{sample}.root",
            "Events",
        )
    return (
        repo_root
        / "SKNanoOutput"
        / "PromptAnalyzer"
        / f"{input_channel}_RunSyst"
        / ERA
        / f"Skim_TriLep_{sample}.root",
        "Events_Central",
    )


def data_path(repo_root: Path, channel: str, sample: str) -> Tuple[Path, str]:
    input_channel = CHANNEL_INPUT[channel]
    return (
        repo_root
        / "SKNanoOutput"
        / "PromptAnalyzer"
        / input_channel
        / ERA
        / f"Skim_TriLep_{sample}.root",
        "Events_Central",
    )


def make_selected_rdf(path: Path, tree_name: str, channel: str, masspoint: str):
    mass_expr, pt_expr = selected_exprs(channel, masspoint)
    return (
        ROOT.RDataFrame(tree_name, str(path))
        .Define("mass_selected", mass_expr)
        .Define("pt_selected", pt_expr)
    )


def weighted_mean_rms(path: Path, channel: str, masspoint: str, lo: float, hi: float) -> Tuple[float, float]:
    rdf = make_selected_rdf(path, "Events_Central", channel, masspoint)
    rdf = rdf.Filter(f"mass_selected >= {lo} && mass_selected <= {hi}")
    sum_w = float(rdf.Sum("weight").GetValue())
    if sum_w <= 0:
        return (lo + hi) / 2.0, max((hi - lo) / 10.0, 1.0)

    rdf = rdf.Define("wm", "weight * mass_selected").Define(
        "wm2", "weight * mass_selected * mass_selected"
    )
    mean = float(rdf.Sum("wm").GetValue()) / sum_w
    mean2 = float(rdf.Sum("wm2").GetValue()) / sum_w
    rms = math.sqrt(max(mean2 - mean * mean, 1e-6))
    return mean, rms


def fit_mass_window(path: Path, channel: str, masspoint: str, debug: bool = False) -> Dict:
    """Fit selected signal mass and return the V3-style mass window."""
    _mhc, ma = masspoint_masses(masspoint)
    wide_lo = max(ma - ma / 3.0, 12.0)
    wide_hi = ma + ma / 3.0
    fallback_mean, fallback_rms = weighted_mean_rms(path, channel, masspoint, wide_lo, wide_hi)

    try:
        rdf = make_selected_rdf(path, "Events_Central", channel, masspoint)
        hist = rdf.Histo1D(
            (f"h_wide_{channel}_{masspoint}", "", 100, wide_lo, wide_hi),
            "mass_selected",
            "weight",
        ).GetValue()
        hist.SetDirectory(0)
        if hist.Integral() <= 0:
            raise RuntimeError("empty signal mass histogram")

        mass_w = ROOT.RooRealVar("mass_w", "mass", wide_lo, wide_hi)
        data_w = ROOT.RooDataHist("data_w", "", ROOT.RooArgList(mass_w), hist)
        pre_x0 = ROOT.RooRealVar("pre_x0", "x0", fallback_mean, wide_lo, wide_hi)
        pre_s_l = ROOT.RooRealVar("pre_s_l", "sL", max(0.8 * fallback_rms, 0.05), 0.01, 10.0)
        pre_s_r = ROOT.RooRealVar("pre_s_r", "sR", max(0.8 * fallback_rms, 0.05), 0.01, 10.0)
        pre_a_l = ROOT.RooRealVar("pre_a_l", "aL", 1.5, 0.5, 10.0)
        pre_n_l = ROOT.RooRealVar("pre_n_l", "nL", 2.0, 0.1, 50.0)
        pre_a_r = ROOT.RooRealVar("pre_a_r", "aR", 1.5, 0.5, 10.0)
        pre_n_r = ROOT.RooRealVar("pre_n_r", "nR", 2.0, 0.1, 50.0)
        pre_dcb = ROOT.RooCrystalBall(
            "pre_dcb", "", mass_w, pre_x0,
            pre_s_l, pre_s_r, pre_a_l, pre_n_l, pre_a_r, pre_n_r
        )
        pre_dcb.fitTo(
            data_w,
            ROOT.RooFit.SumW2Error(True),
            ROOT.RooFit.Save(),
            ROOT.RooFit.PrintLevel(-1),
        )

        fitted_ma = float(pre_x0.getVal())
        pre_sigma = math.sqrt(0.5 * (pre_s_l.getVal() ** 2 + pre_s_r.getVal() ** 2))
        fit_lo = max(fitted_ma - 10.0 * pre_sigma, 12.0)
        fit_hi = fitted_ma + 10.0 * pre_sigma

        hist_n = rdf.Histo1D(
            (f"h_narrow_{channel}_{masspoint}", "", 100, fit_lo, fit_hi),
            "mass_selected",
            "weight",
        ).GetValue()
        hist_n.SetDirectory(0)
        mass_n = ROOT.RooRealVar("mass_n", "mass", fit_lo, fit_hi)
        data_n = ROOT.RooDataHist("data_n", "", ROOT.RooArgList(mass_n), hist_n)
        dcb_x0 = ROOT.RooRealVar("dcb_x0", "x0", fitted_ma, fit_lo, fit_hi)
        dcb_s_l = ROOT.RooRealVar("dcb_s_l", "sigmaL", max(0.8 * pre_sigma, 0.05), 0.01, 20.0)
        dcb_s_r = ROOT.RooRealVar("dcb_s_r", "sigmaR", max(0.8 * pre_sigma, 0.05), 0.01, 20.0)
        dcb_a_l = ROOT.RooRealVar("dcb_a_l", "alphaL", 1.5, 0.5, 10.0)
        dcb_n_l = ROOT.RooRealVar("dcb_n_l", "nL", 2.0, 0.1, 50.0)
        dcb_a_r = ROOT.RooRealVar("dcb_a_r", "alphaR", 1.5, 0.5, 10.0)
        dcb_n_r = ROOT.RooRealVar("dcb_n_r", "nR", 2.0, 0.1, 50.0)
        dcb = ROOT.RooCrystalBall(
            "dcb", "", mass_n, dcb_x0,
            dcb_s_l, dcb_s_r, dcb_a_l, dcb_n_l, dcb_a_r, dcb_n_r
        )
        dcb.fitTo(
            data_n,
            ROOT.RooFit.SumW2Error(True),
            ROOT.RooFit.Save(),
            ROOT.RooFit.PrintLevel(-1),
        )

        sigma_eff = math.sqrt(0.5 * (dcb_s_l.getVal() ** 2 + dcb_s_r.getVal() ** 2))
        x0 = float(dcb_x0.getVal())
        mass_min = max(x0 - 10.0 * sigma_eff, 12.0)
        mass_max = x0 + 10.0 * sigma_eff
        status = "fit"
    except Exception as exc:
        x0 = fallback_mean
        sigma_eff = max(fallback_rms, 0.1)
        mass_min = max(x0 - 10.0 * sigma_eff, 12.0)
        mass_max = x0 + 10.0 * sigma_eff
        status = f"fallback: {exc}"

    if debug:
        print(
            f"{channel} {masspoint}: x0={x0:.3f}, sigma_eff={sigma_eff:.3f}, "
            f"window=[{mass_min:.3f}, {mass_max:.3f}], status={status}"
        )

    return {
        "x0": float(x0),
        "sigma_eff": float(sigma_eff),
        "mass_min": float(mass_min),
        "mass_max": float(mass_max),
        "fit_status": status,
        "wide_window": [float(wide_lo), float(wide_hi)],
    }


def empty_threshold_map(threshold_values: Iterable[float]) -> OrderedDict:
    return OrderedDict((format_threshold(t), 0.0) for t in threshold_values)


def format_threshold(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def histogram_yields(
    path: Path,
    tree_name: str,
    channel: str,
    masspoint: str,
    mass_min: float,
    mass_max: float,
    threshold_values: List[float],
    pt_min: float,
    pt_max: float,
    pt_step: float,
    weight_scale: float,
) -> Tuple[OrderedDict, int]:
    rdf = make_selected_rdf(path, tree_name, channel, masspoint)
    rdf = rdf.Filter(f"mass_selected >= {mass_min} && mass_selected <= {mass_max}")
    if weight_scale == 1.0:
        rdf = rdf.Define("scan_weight", "weight")
    else:
        rdf = rdf.Define("scan_weight", f"weight * {weight_scale:.17g}")

    nbins = int(round((pt_max - pt_min) / pt_step))
    hist = rdf.Histo1D(("h_pt_selected", "", nbins, pt_min, pt_max), "pt_selected", "scan_weight")
    count = int(rdf.Count().GetValue())
    h = hist.GetValue()
    h.SetDirectory(0)

    yields = OrderedDict()
    for threshold in threshold_values:
        key = format_threshold(threshold)
        if threshold >= pt_max:
            first_bin = nbins + 1
        else:
            first_bin = int(math.floor((threshold - pt_min) / pt_step)) + 1
            first_bin = max(1, min(first_bin, nbins + 1))
        yields[key] = float(h.Integral(first_bin, nbins + 1))
    return yields, count


def add_yields(target: OrderedDict, source: OrderedDict) -> None:
    for key, value in source.items():
        target[key] += value


def collect_backgrounds_for_point(
    repo_root: Path,
    samplegroups: Dict,
    kfactors: Dict[str, float],
    conv_sf: float,
    channel: str,
    masspoint: str,
    mass_window: Dict,
    threshold_values: List[float],
    args: argparse.Namespace,
) -> Tuple[Dict, List[Dict]]:
    by_process = OrderedDict()
    audits: List[Dict] = []
    samples_for_channel = samplegroups[ERA][channel]

    for category in BACKGROUND_CATEGORIES:
        process_yields = empty_threshold_map(threshold_values)
        process_count = 0
        process_complete = True

        for sample in samples_for_channel.get(category, []):
            path, tree_name = background_path(repo_root, channel, category, sample)
            audit = audit_file(path, tree_name, REQUIRED_BRANCHES)
            audit.update(
                {
                    "era": ERA,
                    "channel": channel,
                    "masspoint": masspoint,
                    "kind": "background",
                    "process": category,
                    "sample": sample,
                }
            )
            audits.append(audit)
            if audit["status"] != "ok":
                process_complete = False
                continue

            weight_scale = kfactors.get(sample, 1.0)
            if category == "conversion":
                weight_scale *= conv_sf

            sample_yields, sample_count = histogram_yields(
                path,
                tree_name,
                channel,
                masspoint,
                mass_window["mass_min"],
                mass_window["mass_max"],
                threshold_values,
                args.pt_min,
                args.pt_max,
                args.pt_step,
                weight_scale,
            )
            add_yields(process_yields, sample_yields)
            process_count += sample_count

        by_process[category] = {
            "yield": process_yields,
            "count_after_mass_window": process_count,
            "complete": process_complete,
        }

    total = empty_threshold_map(threshold_values)
    for payload in by_process.values():
        add_yields(total, payload["yield"])

    return {
        "by_process": by_process,
        "total": total,
        "complete": all(payload["complete"] for payload in by_process.values()),
    }, audits


def audit_data_inputs(repo_root: Path, samplegroups: Dict) -> List[Dict]:
    audits: List[Dict] = []
    for channel in CHANNEL_INPUT:
        for sample in samplegroups[ERA][channel].get("data", []):
            path, tree_name = data_path(repo_root, channel, sample)
            audit = audit_file(path, tree_name, REQUIRED_BRANCHES)
            audit.update(
                {
                    "era": ERA,
                    "channel": channel,
                    "kind": "data",
                    "process": "data",
                    "sample": sample,
                }
            )
            audits.append(audit)
    return audits


def significance(signal: Optional[float], background: Optional[float]) -> Optional[float]:
    if signal is None or background is None or background <= 0:
        return None
    return signal / math.sqrt(background)


def ratio(value: Optional[float], baseline: Optional[float]) -> Optional[float]:
    if value is None or baseline in (None, 0):
        return None
    return value / baseline


def build_scan(args: argparse.Namespace) -> Dict:
    repo_root = Path(args.repo_root).resolve()
    samplegroups = load_samplegroups(repo_root)
    kfactors = load_kfactors(repo_root)
    threshold_values = thresholds(args.pt_min, args.pt_max, args.pt_step)

    rows: List[Dict] = []
    mass_windows = OrderedDict()
    signal_yields = OrderedDict()
    background_yields = OrderedDict()
    audits: List[Dict] = []
    audits.extend(audit_data_inputs(repo_root, samplegroups))

    for channel in CHANNEL_INPUT:
        conv_sf = load_conv_sf(repo_root, channel)
        mass_windows[channel] = OrderedDict()
        signal_yields[channel] = OrderedDict()
        background_yields[channel] = OrderedDict()

        for masspoint in args.masspoints:
            sig_path = signal_path(repo_root, channel, masspoint)
            signal_audit = audit_file(sig_path, "Events_Central", REQUIRED_BRANCHES)
            signal_audit.update(
                {
                    "era": ERA,
                    "channel": channel,
                    "masspoint": masspoint,
                    "kind": "signal",
                    "process": "signal",
                    "sample": masspoint,
                }
            )
            audits.append(signal_audit)

            if signal_audit["status"] != "ok":
                mass_windows[channel][masspoint] = None
                signal_yields[channel][masspoint] = {
                    "yield": empty_threshold_map(threshold_values),
                    "count_after_mass_window": 0,
                    "complete": False,
                }
                background_yields[channel][masspoint] = None
                for threshold in threshold_values:
                    rows.append(
                        make_row(
                            channel,
                            masspoint,
                            threshold,
                            None,
                            None,
                            None,
                            None,
                            False,
                            None,
                        )
                    )
                continue

            mass_window = fit_mass_window(sig_path, channel, masspoint, args.debug)
            mass_windows[channel][masspoint] = mass_window

            sig_scan, sig_count = histogram_yields(
                sig_path,
                "Events_Central",
                channel,
                masspoint,
                mass_window["mass_min"],
                mass_window["mass_max"],
                threshold_values,
                args.pt_min,
                args.pt_max,
                args.pt_step,
                1.0 / 3.0,
            )
            signal_yields[channel][masspoint] = {
                "yield": sig_scan,
                "count_after_mass_window": sig_count,
                "complete": True,
            }

            bkg_payload, bkg_audits = collect_backgrounds_for_point(
                repo_root,
                samplegroups,
                kfactors,
                conv_sf,
                channel,
                masspoint,
                mass_window,
                threshold_values,
                args,
            )
            background_yields[channel][masspoint] = bkg_payload
            audits.extend(bkg_audits)

            baseline_key = format_threshold(threshold_values[0])
            baseline_s = sig_scan[baseline_key]
            baseline_b = bkg_payload["total"][baseline_key]
            baseline_z = significance(baseline_s, baseline_b)
            for threshold in threshold_values:
                key = format_threshold(threshold)
                signal = sig_scan[key]
                background = bkg_payload["total"][key]
                z_value = significance(signal, background)
                rows.append(
                    make_row(
                        channel,
                        masspoint,
                        threshold,
                        signal,
                        background,
                        {
                            proc: payload["yield"][key]
                            for proc, payload in bkg_payload["by_process"].items()
                        },
                        z_value,
                        signal_audit["status"] == "ok" and bkg_payload["complete"],
                        {
                            "S_eff_vs_pt0": ratio(signal, baseline_s),
                            "B_eff_vs_pt0": ratio(background, baseline_b),
                            "S_over_sqrtB_vs_pt0": ratio(z_value, baseline_z),
                        },
                    )
                )

    return {
        "era": ERA,
        "channels": list(CHANNEL_INPUT.keys()),
        "signal_points": list(args.masspoints),
        "thresholds": threshold_values,
        "required_branches": REQUIRED_BRANCHES,
        "mass_windows": mass_windows,
        "signal_yields": signal_yields,
        "background_yields": background_yields,
        "rows": rows,
        "audit": {
            "era": ERA,
            "required_branches": REQUIRED_BRANCHES,
            "entries": audits,
            "missing_or_skipped": [item for item in audits if item["status"] != "ok"],
        },
    }


def make_row(
    channel: str,
    masspoint: str,
    threshold: float,
    signal: Optional[float],
    background: Optional[float],
    background_by_process: Optional[Dict[str, float]],
    z_value: Optional[float],
    complete: bool,
    ratios: Optional[Dict[str, Optional[float]]],
) -> Dict:
    ratios = ratios or {}
    return {
        "era": ERA,
        "channel": channel,
        "masspoint": masspoint,
        "pt_threshold": threshold,
        "selection_expr": f"pt_selected >= {threshold:g}",
        "S": signal,
        "B_total": background,
        "B_by_process": background_by_process,
        "S_over_sqrtB": z_value,
        "S_eff_vs_pt0": ratios.get("S_eff_vs_pt0"),
        "B_eff_vs_pt0": ratios.get("B_eff_vs_pt0"),
        "S_over_sqrtB_vs_pt0": ratios.get("S_over_sqrtB_vs_pt0"),
        "complete": complete,
    }


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "era",
        "channel",
        "masspoint",
        "pt_threshold",
        "selection_expr",
        "S",
        "B_total",
        "B_by_process",
        "S_over_sqrtB",
        "S_eff_vs_pt0",
        "B_eff_vs_pt0",
        "S_over_sqrtB_vs_pt0",
        "complete",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["B_by_process"] = (
                json.dumps(row["B_by_process"], sort_keys=True)
                if row["B_by_process"] is not None
                else ""
            )
            writer.writerow(csv_row)


def graph_from_rows(rows: List[Dict], channel: str, masspoint: str, metric: str):
    points = [
        (row["pt_threshold"], row[metric])
        for row in rows
        if row["channel"] == channel
        and row["masspoint"] == masspoint
        and row["complete"]
        and row[metric] is not None
    ]
    if not points:
        return None, points

    graph = ROOT.TGraph(len(points))
    for idx, (x_value, y_value) in enumerate(points):
        graph.SetPoint(idx, float(x_value), float(y_value))
    return graph, points


def draw_metric_plots(rows: List[Dict], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    masspoints = sorted({row["masspoint"] for row in rows})
    metrics = OrderedDict(
        [
            ("S", "Signal yield"),
            ("B_total", "Background yield"),
            ("S_over_sqrtB", "S / #sqrt{B}"),
            ("S_over_sqrtB_vs_pt0", "(S / #sqrt{B}) / pt0"),
        ]
    )
    colors = {
        "MHc70_MA15": ROOT.kRed + 1,
        "MHc100_MA60": ROOT.kOrange + 7,
        "MHc130_MA90": ROOT.kBlue + 1,
        "MHc160_MA155": ROOT.kGreen + 2,
    }

    for channel in CHANNEL_INPUT:
        for metric, y_title in metrics.items():
            canvas = ROOT.TCanvas(f"c_{channel}_{metric}", "", 900, 700)
            canvas.SetGrid()
            legend = ROOT.TLegend(0.55, 0.70, 0.88, 0.88)
            legend.SetBorderSize(0)
            legend.SetFillStyle(0)
            drawn = False
            max_y = 0.0
            graphs = []
            for masspoint in masspoints:
                graph, points = graph_from_rows(rows, channel, masspoint, metric)
                if not points:
                    continue
                graph.SetLineColor(colors.get(masspoint, ROOT.kBlack))
                graph.SetMarkerColor(colors.get(masspoint, ROOT.kBlack))
                graph.SetMarkerStyle(20)
                graph.SetLineWidth(2)
                graph.SetTitle(f"{channel};p_{{T}} threshold [GeV];{y_title}")
                max_y = max(max_y, max(y for _x, y in points))
                graphs.append((masspoint, graph))

            for masspoint, graph in graphs:
                if not drawn:
                    graph.Draw("ALP")
                    graph.GetYaxis().SetRangeUser(0.0, max_y * 1.25 if max_y > 0 else 1.0)
                    drawn = True
                else:
                    graph.Draw("LP SAME")
                legend.AddEntry(graph, masspoint, "lp")
            if drawn:
                legend.Draw()
                canvas.SaveAs(str(plot_dir / f"{channel}_{metric}.png"))
                canvas.SaveAs(str(plot_dir / f"{channel}_{metric}.pdf"))


def best_rows(rows: List[Dict]) -> List[Dict]:
    best = []
    masspoints = sorted({row["masspoint"] for row in rows})
    for channel in CHANNEL_INPUT:
        for masspoint in masspoints:
            candidates = [
                row for row in rows
                if row["channel"] == channel
                and row["masspoint"] == masspoint
                and row["complete"]
                and row["S_over_sqrtB"] is not None
            ]
            if candidates:
                best.append(max(candidates, key=lambda row: row["S_over_sqrtB"]))
    return best


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    plot_dir = Path(args.plot_dir)
    if not plot_dir.is_absolute():
        plot_dir = Path.cwd() / plot_dir

    payload = build_scan(args)

    write_json(output_dir / "pt_cut_branch_audit.json", payload["audit"])
    write_json(output_dir / "pt_cut_yield_scan.json", payload)
    write_csv(output_dir / "pt_cut_yield_scan.csv", payload["rows"])
    draw_metric_plots(payload["rows"], plot_dir)

    skipped = len(payload["audit"]["missing_or_skipped"])
    print(f"Wrote {output_dir / 'pt_cut_yield_scan.csv'}")
    print(f"Wrote {output_dir / 'pt_cut_yield_scan.json'}")
    print(f"Wrote {output_dir / 'pt_cut_branch_audit.json'}")
    print(f"Wrote plots under {plot_dir}")
    print(f"Skipped files with missing inputs: {skipped}")
    print("Best per-channel thresholds:")
    for row in best_rows(payload["rows"]):
        print(
            f"  {row['channel']} {row['masspoint']}: "
            f"pT >= {row['pt_threshold']:.0f} GeV, "
            f"S={row['S']:.4g}, B={row['B_total']:.4g}, "
            f"S/sqrt(B)={row['S_over_sqrtB']:.4g}, "
            f"ratio={row['S_over_sqrtB_vs_pt0']:.4g}"
        )


if __name__ == "__main__":
    main()
