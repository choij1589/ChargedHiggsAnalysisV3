#!/usr/bin/env python3
"""Merge split V3 run-period template outputs into one output directory.

The expensive construction is done independently for SR1E2Mu and SR3Mu.  This
script performs the lightweight file-level merge expected by printDatacard.py:
all category directories are copied into one shapes.root, and the metadata JSON
files are unioned without changing category names, process names, or binning.
"""

import argparse
import json
import logging
import os
import shutil
import sys
from collections import OrderedDict

import ROOT

from run_period_utils import PHYSICS_PROCESS_ORDER, SR_CHANNELS


ROOT.gROOT.SetBatch(True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era", required=True, choices=["Run2", "Run3", "All"])
    parser.add_argument("--channel", default="Combined", choices=["Combined", "SR1E2Mu", "SR3Mu"])
    parser.add_argument("--masspoint", required=True)
    parser.add_argument("--method", required=True,
                        choices=["Baseline", "ParticleNet", "PTOptimized"])
    parser.add_argument("--binning", default="extended", choices=["extended", "uniform"])
    parser.add_argument(
        "--sources",
        help=("Comma-separated input template directories as ERA:CHANNEL pairs. "
              "Default is <era>:SR1E2Mu,<era>:SR3Mu."),
    )
    parser.add_argument("--unblind", action="store_true")
    parser.add_argument("--partial-unblind", action="store_true")
    parser.add_argument("--nuisance", default="fallback_lnn",
                        choices=["fallback_lnn", "preserve_shape"])
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.unblind and args.partial_unblind:
        parser.error("--unblind and --partial-unblind are mutually exclusive")
    return args


def binning_suffix(args):
    suffix = args.binning
    if args.partial_unblind:
        suffix = f"{suffix}_partial_unblind"
    elif args.unblind:
        suffix = f"{suffix}_unblind"
    if args.nuisance == "preserve_shape":
        suffix = f"{suffix}_preserve_shape"
    return suffix


def load_json(path):
    with open(path) as handle:
        return json.load(handle, object_pairs_hook=OrderedDict)


def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def source_dir(workdir, args, era, channel):
    return os.path.join(
        workdir, "SignalRegionStudyV4", "templates", era, channel,
        args.masspoint, args.method, binning_suffix(args)
    )


def output_dir(workdir, args):
    return os.path.join(
        workdir, "SignalRegionStudyV4", "templates", args.era, args.channel,
        args.masspoint, args.method, binning_suffix(args)
    )


def parse_sources(args):
    if not args.sources:
        if args.channel == "Combined":
            return [(args.era, channel) for channel in SR_CHANNELS]
        if args.era == "All":
            return [("Run2", args.channel), ("Run3", args.channel)]
        raise ValueError(
            "Single-channel merge without --sources is only defined for --era All"
        )
    sources = []
    for item in args.sources.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid --sources item '{item}'. Use ERA:CHANNEL.")
        era, channel = item.split(":", 1)
        if not era or not channel:
            raise ValueError(f"Invalid --sources item '{item}'. Use ERA:CHANNEL.")
        sources.append((era, channel))
    if not sources:
        raise ValueError("--sources did not contain any source entries")
    return sources


def source_shapes_path(src_dir):
    """Return the pre-prune shapes file of a source component directory.

    printDatacard.py prunes low-stat shape systematics out of shapes.root in
    place and keeps the pre-prune content in shapes_original.root.  Component
    datacard jobs and combined merge jobs are independent DAG nodes, so reading
    shapes.root here makes the merged category inherit whichever pruning state
    the component happened to be in when the merge ran.  A merged category must
    start from the unpruned shapes so it can re-derive its own low-stat
    fallbacks; otherwise those nuisances are dropped outright instead of
    degrading to lnN.

    makeBinnedTemplates.py rmtree's the output directory before rebuilding, so
    shapes_original.root exists iff it is the pre-prune snapshot of the current
    shapes.root -- it can never be stale relative to it.
    """
    original = os.path.join(src_dir, "shapes_original.root")
    if os.path.exists(original):
        return original
    return os.path.join(src_dir, "shapes.root")


def copy_root_object(obj, outdir):
    """Recursively copy ROOT directories and objects into outdir."""
    name = obj.GetName()
    if obj.InheritsFrom("TDirectory"):
        target = outdir.mkdir(name)
        target.cd()
        for key in obj.GetListOfKeys():
            child = key.ReadObj()
            copy_root_object(child, target)
        return

    outdir.cd()
    clone = obj.Clone(name) if hasattr(obj, "Clone") else obj
    clone.Write(name)


def merge_shapes(source_dirs, out_path):
    output = ROOT.TFile.Open(out_path, "RECREATE")
    if not output or output.IsZombie():
        raise OSError(f"Could not create {out_path}")

    seen_categories = set()
    try:
        for src_dir in source_dirs:
            src_path = source_shapes_path(src_dir)
            if not os.path.exists(src_path):
                raise FileNotFoundError(f"Missing source shapes.root: {src_path}")
            logging.debug("Merging shapes from %s", src_path)
            infile = ROOT.TFile.Open(src_path, "READ")
            if not infile or infile.IsZombie():
                raise OSError(f"Could not open {src_path}")
            try:
                for key in infile.GetListOfKeys():
                    obj = key.ReadObj()
                    name = obj.GetName()
                    if name in seen_categories:
                        raise ValueError(f"Duplicate ROOT top-level object/category: {name}")
                    seen_categories.add(name)
                    copy_root_object(obj, output)
            finally:
                infile.Close()
    finally:
        output.Close()


def merge_categories(source_dirs):
    merged_categories = OrderedDict()
    for src_dir in source_dirs:
        payload = load_json(os.path.join(src_dir, "categories.json"))
        if payload.get("construction") != "run_period_components":
            raise ValueError(f"Unexpected construction in {src_dir}/categories.json")
        for cat, cat_payload in payload["categories"].items():
            if cat in merged_categories:
                raise ValueError(f"Duplicate category in metadata: {cat}")
            merged_categories[cat] = cat_payload
    return OrderedDict([
        ("construction", "run_period_components"),
        ("categories", merged_categories),
    ])


def merge_binning(source_dirs):
    merged = None
    for src_dir in source_dirs:
        payload = load_json(os.path.join(src_dir, "binning.json"))
        if merged is None:
            merged = OrderedDict((k, v) for k, v in payload.items() if k != "categories")
            merged["categories"] = OrderedDict()
        for cat, cat_payload in payload.get("categories", {}).items():
            if cat in merged["categories"]:
                raise ValueError(f"Duplicate category in binning metadata: {cat}")
            merged["categories"][cat] = cat_payload
    if merged is None:
        raise ValueError("No binning metadata found")
    return merged


def merge_category_keyed_json(source_dirs, filename):
    merged = None
    saw_file = False
    for src_dir in source_dirs:
        path = os.path.join(src_dir, filename)
        if not os.path.exists(path):
            continue
        saw_file = True
        payload = load_json(path)
        if isinstance(payload, dict) and "categories" in payload:
            if merged is None:
                merged = OrderedDict((k, v) for k, v in payload.items() if k != "categories")
                merged["categories"] = OrderedDict()
            category_payload = payload["categories"]
        else:
            if merged is None:
                merged = OrderedDict()
            category_payload = payload
        for cat, cat_payload in category_payload.items():
            if cat in merged.get("categories", merged):
                raise ValueError(f"Duplicate category in {filename}: {cat}")
            if "categories" in merged:
                merged["categories"][cat] = cat_payload
            else:
                merged[cat] = cat_payload
    return merged if saw_file else None


def build_process_list(categories):
    components = []
    for cat, payload in categories["categories"].items():
        for proc in payload["processes"]:
            entry = OrderedDict()
            entry["category"] = cat
            for key, value in proc.items():
                entry[key] = value
            components.append(entry)

    def unique_names(entries):
        seen = set()
        names = []
        for entry in entries:
            name = entry["name"]
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
        return names

    return OrderedDict([
        ("construction", "run_period_components"),
        ("separate_processes", unique_names(p for p in components if not p["is_signal"])),
        ("signal_processes", unique_names(p for p in components if p["is_signal"])),
        ("components", components),
        ("physics_groups", OrderedDict(
            (group, unique_names(p for p in components if p["physics_group"] == group))
            for group in PHYSICS_PROCESS_ORDER
        )),
        ("merged_to_others", []),
        ("description", "Run-period categories with subera component processes"),
    ])


def copy_auxiliary_files(source_dirs, out_dir):
    skip = {
        "shapes.root",
        "categories.json",
        "process_list.json",
        "binning.json",
        "background_validation.json",
        "threshold.json",
        "background_weights.json",
        "datacard.txt",
        "lowstat.json",
        "shapes_original.root",
    }
    skip_dirs = {"validation", "combine_output", "logs", "scores"}
    for src_dir in source_dirs:
        for name in os.listdir(src_dir):
            src = os.path.join(src_dir, name)
            dst = os.path.join(out_dir, name)
            if name in skip or name in skip_dirs:
                continue
            if os.path.isdir(src):
                if os.path.exists(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    workdir = os.environ.get("WORKDIR")
    if not workdir:
        raise RuntimeError("WORKDIR is not set")

    src_dirs = [source_dir(workdir, args, era, channel)
                for era, channel in parse_sources(args)]
    for src_dir in src_dirs:
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(f"Missing split template directory: {src_dir}")

    out_dir = output_dir(workdir, args)
    if os.path.abspath(out_dir) in {os.path.abspath(src_dir) for src_dir in src_dirs}:
        raise ValueError(f"Refusing to merge a template directory into itself: {out_dir}")

    logging.info("Merging split templates into %s", out_dir)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    merge_shapes(src_dirs, os.path.join(out_dir, "shapes.root"))
    categories = merge_categories(src_dirs)
    save_json(categories, os.path.join(out_dir, "categories.json"))
    save_json(merge_binning(src_dirs), os.path.join(out_dir, "binning.json"))
    save_json(build_process_list(categories), os.path.join(out_dir, "process_list.json"))

    for filename in ["background_validation.json", "threshold.json", "background_weights.json"]:
        payload = merge_category_keyed_json(src_dirs, filename)
        if payload:
            save_json(payload, os.path.join(out_dir, filename))

    copy_auxiliary_files(src_dirs, out_dir)
    logging.info("Merged %d categories", len(categories["categories"]))
    logging.info("Done: %s/shapes.root", out_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logging.error("%s", exc)
        sys.exit(1)
