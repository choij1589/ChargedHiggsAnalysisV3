#!/usr/bin/env python3
"""Loose-btag yield study for 2018 signal-region samples.

This script reads the baseline-selected SKNano output trees directly and
compares weighted yields for baseline, nB_loose >= 2, and nB_loose >= 3.
No mass window is applied.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import ROOT


ROOT.gROOT.SetBatch(True)

ERA = "2018"
CHANNEL_INPUT = {
    "SR1E2Mu": "Run1E2Mu",
    "SR3Mu": "Run3Mu",
}
SIGNAL_POINTS = [
    "MHc70_MA15",
    "MHc100_MA60",
    "MHc130_MA90",
    "MHc160_MA155",
]
SELECTIONS = OrderedDict(
    [
        ("baseline", "1"),
        ("nB_loose_ge1", "nB_loose >= 1"),
        ("nB_loose_ge2", "nB_loose >= 2"),
        ("nB_loose_ge3", "nB_loose >= 3"),
    ]
)
REQUIRED_BRANCHES = ["weight", "nB_loose"]
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
    return parser.parse_args()


def load_samplegroups(repo_root: Path) -> Dict:
    path = repo_root / "SignalRegionStudyV3" / "configs" / "samplegroups.json"
    with path.open() as handle:
        return json.load(handle)


def tree_branches(path: Path, tree_name: str) -> Tuple[bool, List[str], str]:
    if not path.exists():
        return False, [], "missing_file"

    root_file = ROOT.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie():
        return False, [], "open_failed"

    tree = root_file.Get(tree_name)
    if not tree:
        root_file.Close()
        return False, [], "missing_tree"

    branches = [branch.GetName() for branch in tree.GetListOfBranches()]
    root_file.Close()
    return True, branches, "ok"


def audit_file(path: Path, tree_name: str, required: Iterable[str]) -> Dict:
    ok, branches, status = tree_branches(path, tree_name)
    missing = [] if status != "ok" else [name for name in required if name not in branches]
    return {
        "path": str(path),
        "tree": tree_name,
        "status": "ok" if ok and not missing else ("missing_branch" if missing else status),
        "missing_branches": missing,
        "n_branches": len(branches),
    }


def sum_weight(path: Path, tree_name: str, selection: str) -> float:
    rdf = ROOT.RDataFrame(tree_name, str(path))
    if selection != "1":
        rdf = rdf.Filter(selection)
    result = rdf.Sum("weight")
    return float(result.GetValue())


def sum_weight_and_error(path: Path, tree_name: str, selection: str, scale: float = 1.0) -> Tuple[float, float]:
    rdf = ROOT.RDataFrame(tree_name, str(path))
    if selection != "1":
        rdf = rdf.Filter(selection)
    rdf = rdf.Define("__weight_scaled", f"weight * {scale:.17g}")
    rdf = rdf.Define("__weight_scaled_sq", "__weight_scaled * __weight_scaled")
    total = float(rdf.Sum("__weight_scaled").GetValue())
    variance = float(rdf.Sum("__weight_scaled_sq").GetValue())
    return total, math.sqrt(max(variance, 0.0))


def sum_count(path: Path, tree_name: str, selection: str) -> int:
    rdf = ROOT.RDataFrame(tree_name, str(path))
    if selection != "1":
        rdf = rdf.Filter(selection)
    return int(rdf.Count().GetValue())


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


def empty_selection_map() -> Dict[str, float]:
    return {name: 0.0 for name in SELECTIONS}


def empty_error2_map() -> Dict[str, float]:
    return {name: 0.0 for name in SELECTIONS}


def empty_count_map() -> Dict[str, int]:
    return {name: 0 for name in SELECTIONS}


def collect_backgrounds(repo_root: Path, samplegroups: Dict) -> Tuple[Dict, List[Dict]]:
    yields: Dict[str, Dict] = {}
    audits: List[Dict] = []

    for channel in CHANNEL_INPUT:
        yields[channel] = {
            "by_process": OrderedDict(),
            "total": empty_selection_map(),
            "error": empty_selection_map(),
            "counts": empty_count_map(),
            "complete": True,
        }
        samples_for_channel = samplegroups[ERA][channel]

        for category in BACKGROUND_CATEGORIES:
            process_yield = empty_selection_map()
            process_error2 = empty_error2_map()
            process_counts = empty_count_map()
            process_complete = True
            for sample in samples_for_channel.get(category, []):
                path, tree_name = background_path(repo_root, channel, category, sample)
                audit = audit_file(path, tree_name, REQUIRED_BRANCHES)
                audit.update(
                    {
                        "era": ERA,
                        "channel": channel,
                        "kind": "background",
                        "process": category,
                        "sample": sample,
                    }
                )
                audits.append(audit)
                if audit["status"] != "ok":
                    process_complete = False
                    yields[channel]["complete"] = False
                    continue

                for selection_name, expression in SELECTIONS.items():
                    value, error = sum_weight_and_error(path, tree_name, expression)
                    process_yield[selection_name] += value
                    process_error2[selection_name] += error * error
                    process_counts[selection_name] += sum_count(path, tree_name, expression)

            yields[channel]["by_process"][category] = {
                "yield": process_yield,
                "error": {
                    selection_name: math.sqrt(process_error2[selection_name])
                    for selection_name in SELECTIONS
                },
                "count": process_counts,
                "complete": process_complete,
            }
            for selection_name in SELECTIONS:
                yields[channel]["total"][selection_name] += process_yield[selection_name]
                yields[channel]["error"][selection_name] = math.sqrt(
                    yields[channel]["error"][selection_name] ** 2
                    + process_error2[selection_name]
                )
                yields[channel]["counts"][selection_name] += process_counts[selection_name]

    return yields, audits


def collect_signals(repo_root: Path) -> Tuple[Dict, List[Dict]]:
    yields: Dict[str, Dict] = {}
    audits: List[Dict] = []

    for channel in CHANNEL_INPUT:
        yields[channel] = {}
        for masspoint in SIGNAL_POINTS:
            path = signal_path(repo_root, channel, masspoint)
            audit = audit_file(path, "Events_Central", REQUIRED_BRANCHES)
            audit.update(
                {
                    "era": ERA,
                    "channel": channel,
                    "kind": "signal",
                    "process": "signal",
                    "sample": masspoint,
                }
            )
            audits.append(audit)

            point_yields = empty_selection_map()
            point_errors = empty_selection_map()
            point_counts = empty_count_map()
            complete = audit["status"] == "ok"
            if complete:
                for selection_name, expression in SELECTIONS.items():
                    value, error = sum_weight_and_error(
                        path, "Events_Central", expression, scale=1.0 / 3.0
                    )
                    point_yields[selection_name] = value
                    point_errors[selection_name] = error
                    point_counts[selection_name] = sum_count(
                        path, "Events_Central", expression
                    )

            yields[channel][masspoint] = {
                "yield": point_yields,
                "error": point_errors,
                "count": point_counts,
                "complete": complete,
            }

    return yields, audits


def ratio_to_baseline(values: Dict[str, Optional[float]], selection: str) -> Optional[float]:
    baseline = values["baseline"]
    value = values[selection]
    if baseline is None or value is None or baseline == 0:
        return None
    return value / baseline


def significance(signal: Optional[float], background: Optional[float]) -> Optional[float]:
    if signal is None or background is None or background <= 0:
        return None
    return signal / math.sqrt(background)


def significance_error(
    signal: Optional[float],
    signal_error: Optional[float],
    background: Optional[float],
    background_error: Optional[float],
) -> Optional[float]:
    if (
        signal is None
        or signal_error is None
        or background is None
        or background_error is None
        or background <= 0
    ):
        return None
    dz_ds = 1.0 / math.sqrt(background)
    dz_db = -0.5 * signal / (background ** 1.5)
    variance = (dz_ds * signal_error) ** 2 + (dz_db * background_error) ** 2
    return math.sqrt(max(variance, 0.0))


def build_rows(signal_yields: Dict, background_yields: Dict) -> List[Dict]:
    rows: List[Dict] = []

    for channel in list(CHANNEL_INPUT) + ["Combined"]:
        for masspoint in SIGNAL_POINTS:
            if channel == "Combined":
                signal_complete = all(
                    signal_yields[ch][masspoint]["complete"] for ch in CHANNEL_INPUT
                )
                sig_values = {
                    selection: (
                        sum(
                            signal_yields[ch][masspoint]["yield"][selection]
                            for ch in CHANNEL_INPUT
                        )
                        if signal_complete
                        else None
                    )
                    for selection in SELECTIONS
                }
                sig_errors = {
                    selection: (
                        math.sqrt(
                            sum(
                                signal_yields[ch][masspoint]["error"][selection] ** 2
                                for ch in CHANNEL_INPUT
                            )
                        )
                        if signal_complete
                        else None
                    )
                    for selection in SELECTIONS
                }
                bkg_values = {
                    selection: sum(
                        background_yields[ch]["total"][selection] for ch in CHANNEL_INPUT
                    )
                    for selection in SELECTIONS
                }
                bkg_errors = {
                    selection: math.sqrt(
                        sum(
                            background_yields[ch]["error"][selection] ** 2
                            for ch in CHANNEL_INPUT
                        )
                    )
                    for selection in SELECTIONS
                }
                bkg_by_process = OrderedDict()
                bkg_by_process_err = OrderedDict()
                for process in BACKGROUND_CATEGORIES:
                    bkg_by_process[process] = {
                        selection: sum(
                            background_yields[ch]["by_process"][process]["yield"][selection]
                            for ch in CHANNEL_INPUT
                        )
                        for selection in SELECTIONS
                    }
                    bkg_by_process_err[process] = {
                        selection: math.sqrt(
                            sum(
                                background_yields[ch]["by_process"][process]["error"][selection] ** 2
                                for ch in CHANNEL_INPUT
                            )
                        )
                        for selection in SELECTIONS
                    }
                background_complete = all(background_yields[ch]["complete"] for ch in CHANNEL_INPUT)
                complete = signal_complete and background_complete
            else:
                signal_complete = signal_yields[channel][masspoint]["complete"]
                sig_values = (
                    signal_yields[channel][masspoint]["yield"]
                    if signal_complete
                    else {selection: None for selection in SELECTIONS}
                )
                sig_errors = (
                    signal_yields[channel][masspoint]["error"]
                    if signal_complete
                    else {selection: None for selection in SELECTIONS}
                )
                bkg_values = background_yields[channel]["total"]
                bkg_errors = background_yields[channel]["error"]
                bkg_by_process = OrderedDict(
                    (process, payload["yield"])
                    for process, payload in background_yields[channel]["by_process"].items()
                )
                bkg_by_process_err = OrderedDict(
                    (process, payload["error"])
                    for process, payload in background_yields[channel]["by_process"].items()
                )
                background_complete = background_yields[channel]["complete"]
                complete = signal_complete and background_complete

            for selection_name in SELECTIONS:
                signal_value = sig_values[selection_name]
                signal_error = sig_errors[selection_name]
                background_value = bkg_values[selection_name]
                background_error = bkg_errors[selection_name]
                rows.append(
                    {
                        "era": ERA,
                        "channel": channel,
                        "masspoint": masspoint,
                        "selection": selection_name,
                        "selection_expr": SELECTIONS[selection_name],
                        "S": signal_value,
                        "S_err": signal_error,
                        "B_total": background_value,
                        "B_total_err": background_error,
                        "B_by_process": {
                            process: values[selection_name]
                            for process, values in bkg_by_process.items()
                        },
                        "B_by_process_err": {
                            process: values[selection_name]
                            for process, values in bkg_by_process_err.items()
                        },
                        "S_over_sqrtB": significance(signal_value, background_value),
                        "S_over_sqrtB_err": significance_error(
                            signal_value, signal_error, background_value, background_error
                        ),
                        "S_eff_vs_baseline": ratio_to_baseline(sig_values, selection_name),
                        "B_eff_vs_baseline": ratio_to_baseline(bkg_values, selection_name),
                        "signal_complete": signal_complete,
                        "background_complete": background_complete,
                        "complete": complete,
                    }
                )

    return rows


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "era",
        "channel",
        "masspoint",
        "selection",
        "selection_expr",
        "S",
        "S_err",
        "B_total",
        "B_total_err",
        "B_by_process",
        "B_by_process_err",
        "S_over_sqrtB",
        "S_over_sqrtB_err",
        "S_eff_vs_baseline",
        "B_eff_vs_baseline",
        "signal_complete",
        "background_complete",
        "complete",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            csv_row = dict(row)
            csv_row["B_by_process"] = json.dumps(row["B_by_process"], sort_keys=True)
            csv_row["B_by_process_err"] = json.dumps(row["B_by_process_err"], sort_keys=True)
            writer.writerow(csv_row)


def fmt_pm(value: Optional[float], error: Optional[float], precision: int = 3) -> str:
    if value is None or error is None:
        return "-"
    return f"{value:.{precision}f} +/- {error:.{precision}f}"


def write_markdown(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("# Loose b-tag yield study\n\n")
        handle.write("No mass window is applied. Errors are statistical sumw2 errors.\n\n")
        handle.write(
            "| channel | masspoint | selection | S | B | S/sqrt(B) | "
            "S eff. | B eff. | complete |\n"
        )
        handle.write("|---|---|---|---:|---:|---:|---:|---:|---|\n")
        for row in rows:
            handle.write(
                "| {channel} | {masspoint} | {selection} | {s} | {b} | {z} | "
                "{seff} | {beff} | {complete} |\n".format(
                    channel=row["channel"],
                    masspoint=row["masspoint"],
                    selection=row["selection"],
                    s=fmt_pm(row["S"], row["S_err"]),
                    b=fmt_pm(row["B_total"], row["B_total_err"]),
                    z=fmt_pm(row["S_over_sqrtB"], row["S_over_sqrtB_err"]),
                    seff="-" if row["S_eff_vs_baseline"] is None else f"{row['S_eff_vs_baseline']:.3f}",
                    beff="-" if row["B_eff_vs_baseline"] is None else f"{row['B_eff_vs_baseline']:.3f}",
                    complete=row["complete"],
                )
            )


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    samplegroups = load_samplegroups(repo_root)
    background_yields, background_audits = collect_backgrounds(repo_root, samplegroups)
    signal_yields, signal_audits = collect_signals(repo_root)
    rows = build_rows(signal_yields, background_yields)

    audit = {
        "era": ERA,
        "required_branches": REQUIRED_BRANCHES,
        "entries": signal_audits + background_audits,
        "missing_or_skipped": [
            item for item in signal_audits + background_audits if item["status"] != "ok"
        ],
    }
    payload = {
        "era": ERA,
        "signal_points": SIGNAL_POINTS,
        "selections": SELECTIONS,
        "required_branches": REQUIRED_BRANCHES,
        "signal_yields": signal_yields,
        "background_yields": background_yields,
        "rows": rows,
    }

    write_json(output_dir / "branch_audit.json", audit)
    write_json(output_dir / "loose_btag_yields.json", payload)
    write_csv(output_dir / "loose_btag_yields.csv", rows)
    write_markdown(output_dir / "loose_btag_yields.md", rows)

    skipped = len(audit["missing_or_skipped"])
    print(f"Wrote {output_dir / 'loose_btag_yields.csv'}")
    print(f"Wrote {output_dir / 'loose_btag_yields.json'}")
    print(f"Wrote {output_dir / 'loose_btag_yields.md'}")
    print(f"Wrote {output_dir / 'branch_audit.json'}")
    print(f"Skipped files with missing inputs: {skipped}")


if __name__ == "__main__":
    main()
