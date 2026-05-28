#!/usr/bin/env python3
"""
Compare two preprocessed sample directories.

The comparison is content-oriented rather than byte-oriented: ROOT files can
legitimately differ at the byte level after regeneration, so this checks file,
tree, branch, entry, weight, and ParticleNet score summaries.
"""
import argparse
import math
from pathlib import Path

import ROOT


def parse_args():
    parser = argparse.ArgumentParser(description="Compare preprocessed ROOT output directories")
    parser.add_argument("--reference", required=True, help="Backup/reference directory")
    parser.add_argument("--candidate", required=True, help="New/candidate directory")
    parser.add_argument("--masspoint", required=True, help="Mass point, e.g. MHc130_MA90")
    parser.add_argument(
        "--stats-tree",
        action="append",
        default=None,
        help="Tree name to scan for weight/score summaries. Repeatable. Defaults to all trees.",
    )
    parser.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance")
    parser.add_argument("--atol", type=float, default=1e-8, help="Absolute tolerance")
    return parser.parse_args()


def root_files(base):
    return sorted(path.relative_to(base) for path in base.rglob("*.root"))


def tree_names(tfile):
    names = []
    for key in tfile.GetListOfKeys():
        obj = key.ReadObj()
        if obj.InheritsFrom("TTree"):
            names.append(key.GetName())
    return sorted(names)


def branch_names(tree):
    return sorted(branch.GetName() for branch in tree.GetListOfBranches())


def tree_summary(tree, masspoint, collect_stats=True):
    branches = branch_names(tree)
    selected = ["weight"]
    selected.extend(
        name for name in branches
        if name.startswith(f"score_{masspoint}_")
    )

    stats = {
        "entries": int(tree.GetEntries()),
        "branches": branches,
        "selected": {},
    }
    if not collect_stats:
        return stats

    for name in selected:
        if name not in branches:
            continue
        stats["selected"][name] = {
            "sum": 0.0,
            "sum2": 0.0,
            "min": None,
            "max": None,
        }

    for ientry in range(stats["entries"]):
        tree.GetEntry(ientry)
        for name, values in stats["selected"].items():
            value = float(getattr(tree, name))
            values["sum"] += value
            values["sum2"] += value * value
            values["min"] = value if values["min"] is None else min(values["min"], value)
            values["max"] = value if values["max"] is None else max(values["max"], value)

    for values in stats["selected"].values():
        entries = stats["entries"]
        values["mean"] = values["sum"] / entries if entries else 0.0
        if values["min"] is None:
            values["min"] = 0.0
            values["max"] = 0.0

    return stats


def file_summary(path, masspoint, stats_trees):
    tfile = ROOT.TFile.Open(str(path))
    if not tfile or tfile.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {path}")

    try:
        summary = {}
        for name in tree_names(tfile):
            tree = tfile.Get(name)
            collect_stats = stats_trees is None or name in stats_trees
            summary[name] = tree_summary(tree, masspoint, collect_stats)
        return summary
    finally:
        tfile.Close()


def close_enough(left, right, rtol, atol):
    return math.isclose(left, right, rel_tol=rtol, abs_tol=atol)


def compare_values(label, left, right, rtol, atol, errors):
    if not close_enough(left, right, rtol, atol):
        errors.append(f"{label}: {left} != {right}")


def compare_summaries(relpath, reference, candidate, rtol, atol):
    errors = []
    ref_trees = set(reference)
    cand_trees = set(candidate)
    if ref_trees != cand_trees:
        errors.append(f"{relpath}: tree set differs: {sorted(ref_trees ^ cand_trees)}")
        return errors

    for tree_name in sorted(ref_trees):
        ref_tree = reference[tree_name]
        cand_tree = candidate[tree_name]
        label = f"{relpath}:{tree_name}"

        if ref_tree["entries"] != cand_tree["entries"]:
            errors.append(f"{label}: entries {ref_tree['entries']} != {cand_tree['entries']}")

        if ref_tree["branches"] != cand_tree["branches"]:
            ref_branches = set(ref_tree["branches"])
            cand_branches = set(cand_tree["branches"])
            errors.append(f"{label}: branch set differs: {sorted(ref_branches ^ cand_branches)}")

        ref_selected = set(ref_tree["selected"])
        cand_selected = set(cand_tree["selected"])
        if ref_selected != cand_selected:
            errors.append(f"{label}: selected branch set differs: {sorted(ref_selected ^ cand_selected)}")
            continue

        for branch in sorted(ref_selected):
            for key in ["sum", "sum2", "min", "max", "mean"]:
                compare_values(
                    f"{label}:{branch}:{key}",
                    ref_tree["selected"][branch][key],
                    cand_tree["selected"][branch][key],
                    rtol,
                    atol,
                    errors,
                )
    return errors


def main():
    args = parse_args()
    reference = Path(args.reference)
    candidate = Path(args.candidate)

    if not reference.is_dir():
        raise NotADirectoryError(reference)
    if not candidate.is_dir():
        raise NotADirectoryError(candidate)

    ref_files = root_files(reference)
    cand_files = root_files(candidate)
    if ref_files != cand_files:
        missing = sorted(set(ref_files) ^ set(cand_files))
        raise RuntimeError(f"ROOT file inventory differs: {missing}")

    stats_trees = set(args.stats_tree) if args.stats_tree else None
    all_errors = []
    for relpath in ref_files:
        ref_summary = file_summary(reference / relpath, args.masspoint, stats_trees)
        cand_summary = file_summary(candidate / relpath, args.masspoint, stats_trees)
        all_errors.extend(
            compare_summaries(relpath, ref_summary, cand_summary, args.rtol, args.atol)
        )

    if all_errors:
        print("FAILED")
        for error in all_errors[:200]:
            print(error)
        if len(all_errors) > 200:
            print(f"... {len(all_errors) - 200} more differences")
        raise SystemExit(1)

    print(f"OK: {len(ref_files)} ROOT files match within tolerance")


if __name__ == "__main__":
    main()
