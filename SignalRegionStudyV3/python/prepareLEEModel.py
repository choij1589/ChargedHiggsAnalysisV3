#!/usr/bin/env python3
"""Build the fine-binned background model for LEE pseudo-experiments."""

import argparse
import json
import logging
import os
from datetime import datetime, timezone

import ROOT


ROOT.gROOT.SetBatch(True)

DEFAULT_MASSPOINT = "MHc70_MA18"
MODEL_XMIN = 10.0
MODEL_XMAX = 100.0
MODEL_BIN_WIDTH = 0.1
MODEL_NBINS = int(round((MODEL_XMAX - MODEL_XMIN) / MODEL_BIN_WIDTH))
LEE_CATEGORIES = ("SR1E2Mu_Run2", "SR3Mu_Run2", "SR1E2Mu_Run3", "SR3Mu_Run3")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masspoint", default=DEFAULT_MASSPOINT)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def module_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def output_dir(base_dir, masspoint):
    return os.path.join(base_dir, "LEE", masspoint, "model")


def template_dir(base_dir, masspoint):
    return os.path.join(
        base_dir,
        "templates",
        "All",
        "Combined",
        masspoint,
        "Baseline",
        "extended_unblind",
    )


def validate_masspoint(base_dir, masspoint):
    config_path = os.path.join(base_dir, "configs", "masspoints.json")
    payload = load_json(config_path)
    lee_masspoints = payload.get("LEE", [])
    if not lee_masspoints:
        raise ValueError(f"No 'LEE' masspoint list found in {config_path}")
    if masspoint not in lee_masspoints:
        raise ValueError(
            f"Mass point '{masspoint}' is not in configs/masspoints.json:LEE"
        )
    return lee_masspoints


def validate_trial_binning(base_dir, lee_masspoints):
    windows = {}
    for masspoint in lee_masspoints:
        path = os.path.join(template_dir(base_dir, masspoint), "binning.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"LEE trial binning metadata not found: {path}")
        payload = load_json(path)
        categories = payload.get("categories", {})
        missing = [cat for cat in LEE_CATEGORIES if cat not in categories]
        if missing:
            raise KeyError(f"{path} missing LEE categories: {', '.join(missing)}")
        windows[masspoint] = {}
        for cat in LEE_CATEGORIES:
            mass_min = float(categories[cat]["mass_min"])
            mass_max = float(categories[cat]["mass_max"])
            if mass_min < MODEL_XMIN or mass_max > MODEL_XMAX:
                raise ValueError(
                    f"{masspoint}/{cat} mass window [{mass_min}, {mass_max}] "
                    f"falls outside model range [{MODEL_XMIN}, {MODEL_XMAX}]"
                )
            windows[masspoint][cat] = {"mass_min": mass_min, "mass_max": mass_max}
    return windows


def required_category_inputs(base_dir, masspoint):
    categories_path = os.path.join(template_dir(base_dir, masspoint), "categories.json")
    if not os.path.exists(categories_path):
        raise FileNotFoundError(f"Category metadata not found: {categories_path}")
    payload = load_json(categories_path)
    categories = payload.get("categories", {})

    required = {}
    for cat in LEE_CATEGORIES:
        if cat not in categories:
            raise KeyError(f"{categories_path} missing category '{cat}'")
        cat_payload = categories[cat]
        channel = cat_payload["channel"]
        entries = []
        seen = set()
        for proc in cat_payload.get("processes", []):
            if proc.get("is_signal"):
                continue
            subera = proc["subera"]
            base_process = proc["base_process"]
            key = (subera, channel, base_process)
            if key in seen:
                continue
            seen.add(key)
            path = os.path.join(
                base_dir,
                "samples",
                subera,
                channel,
                masspoint,
                f"{base_process}.root",
            )
            entries.append(
                {
                    "subera": subera,
                    "channel": channel,
                    "process": base_process,
                    "path": path,
                }
            )
        required[cat] = entries
    return required


def observable_expression(channel, mass1, mass2):
    if channel == "SR1E2Mu":
        return mass1
    if channel == "SR3Mu":
        return min(mass1, mass2)
    raise ValueError(f"Unsupported LEE channel '{channel}'")


def validate_tree(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required LEE input file not found: {path}")
    root_file = ROOT.TFile.Open(path, "READ")
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open ROOT file: {path}")
    tree = root_file.Get("Central")
    if not tree:
        root_file.Close()
        raise RuntimeError(f"Tree 'Central' not found in {path}")
    branches = {branch.GetName() for branch in tree.GetListOfBranches()}
    missing = {"mass1", "mass2", "weight"} - branches
    if missing:
        root_file.Close()
        raise RuntimeError(f"{path}/Central missing branches: {sorted(missing)}")
    return root_file, tree


def fill_model_hist(category, inputs):
    hist = ROOT.TH1D(category, category, MODEL_NBINS, MODEL_XMIN, MODEL_XMAX)
    hist.Sumw2()
    hist.SetDirectory(0)

    input_records = []
    for item in inputs:
        logging.info("Reading %s", item["path"])
        root_file, tree = validate_tree(item["path"])
        try:
            n_entries = int(tree.GetEntries())
            weight_sum = 0.0
            in_range_weight_sum = 0.0
            for event in tree:
                obs = observable_expression(item["channel"], event.mass1, event.mass2)
                weight = float(event.weight)
                weight_sum += weight
                if MODEL_XMIN <= obs < MODEL_XMAX:
                    in_range_weight_sum += weight
                    hist.Fill(obs, weight)
            input_records.append(
                {
                    "subera": item["subera"],
                    "channel": item["channel"],
                    "process": item["process"],
                    "path": item["path"],
                    "entries": n_entries,
                    "weight_sum": weight_sum,
                    "in_model_range_weight_sum": in_range_weight_sum,
                }
            )
        finally:
            root_file.Close()

    pre_floor_integral = hist.Integral()
    floored_yield = 0.0
    floored_bins = 0
    for idx in range(1, hist.GetNbinsX() + 1):
        content = hist.GetBinContent(idx)
        if content < 0.0:
            floored_yield += -content
            floored_bins += 1
            hist.SetBinContent(idx, 0.0)
            hist.SetBinError(idx, 0.0)
    post_floor_integral = hist.Integral()
    floor_fraction = (
        floored_yield / post_floor_integral if post_floor_integral > 0.0 else 0.0
    )

    return hist, {
        "input_files": input_records,
        "pre_floor_integral": pre_floor_integral,
        "post_floor_integral": post_floor_integral,
        "floored_yield": floored_yield,
        "floored_bins": floored_bins,
        "floor_fraction": floor_fraction,
    }


def write_outputs(outdir, hists, metadata):
    os.makedirs(outdir, exist_ok=True)
    root_path = os.path.join(outdir, "bkg_model.root")
    json_path = os.path.join(outdir, "bkg_model.json")

    root_file = ROOT.TFile.Open(root_path, "RECREATE")
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not create ROOT output: {root_path}")
    try:
        for hist in hists.values():
            root_file.cd()
            hist.Write(hist.GetName())
    finally:
        root_file.Close()

    with open(json_path, "w") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)

    return root_path, json_path


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    base_dir = module_dir()
    lee_masspoints = validate_masspoint(base_dir, args.masspoint)
    trial_windows = validate_trial_binning(base_dir, lee_masspoints)
    inputs = required_category_inputs(base_dir, args.masspoint)

    hists = {}
    category_metadata = {}
    for category in LEE_CATEGORIES:
        logging.info("Building LEE background model category %s", category)
        hist, meta = fill_model_hist(category, inputs[category])
        hists[category] = hist
        category_metadata[category] = meta
        logging.info(
            "%s: B=%.6f, floored %.6f in %d bins (fraction %.6g)",
            category,
            meta["post_floor_integral"],
            meta["floored_yield"],
            meta["floored_bins"],
            meta["floor_fraction"],
        )

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "masspoint": args.masspoint,
        "method": "Baseline",
        "era": "All",
        "channel": "Combined",
        "binning": "extended_unblind",
        "tree": "Central",
        "observable": {
            "SR1E2Mu": "mass1",
            "SR3Mu": "min(mass1, mass2)",
        },
        "model_binning": {
            "xmin": MODEL_XMIN,
            "xmax": MODEL_XMAX,
            "bin_width": MODEL_BIN_WIDTH,
            "nbins": MODEL_NBINS,
        },
        "trials_masspoints": lee_masspoints,
        "trial_windows": trial_windows,
        "categories": category_metadata,
    }

    root_path, json_path = write_outputs(output_dir(base_dir, args.masspoint), hists, metadata)
    logging.info("Wrote %s", root_path)
    logging.info("Wrote %s", json_path)


if __name__ == "__main__":
    main()
