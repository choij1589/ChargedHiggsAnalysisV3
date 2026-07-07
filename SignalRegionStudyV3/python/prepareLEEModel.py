#!/usr/bin/env python3
"""Build the fine-binned background model for LEE pseudo-experiments.

The model is built per background process (per subera input file) and each
process histogram is floored in coarse blocks before summing: within each
block the total is clipped at zero and redistributed proportionally to the
positive fine-bin content. This mimics the per-process negative-bin flooring
applied by makeBinnedTemplates.py to the datacard templates, so the toy
expectation tracks the frozen datacard backgrounds. A net-sum flooring would
undershoot them by up to ~10% (see docs/LEE.md).
"""

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
FLOOR_BLOCK_WIDTH = 0.5
LEE_CATEGORIES = ("SR1E2Mu_Run2", "SR3Mu_Run2", "SR1E2Mu_Run3", "SR3Mu_Run3")
CONSISTENCY_WARN_THRESHOLD = 0.10


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masspoint", default=DEFAULT_MASSPOINT)
    parser.add_argument("--floor-block-width", type=float, default=FLOOR_BLOCK_WIDTH,
                        help="Block width in GeV for per-process negative-content "
                             f"flooring [default: {FLOOR_BLOCK_WIDTH}]")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    ratio = args.floor_block_width / MODEL_BIN_WIDTH
    if args.floor_block_width <= 0 or abs(ratio - round(ratio)) > 1e-9:
        parser.error(
            f"--floor-block-width must be a positive multiple of {MODEL_BIN_WIDTH}"
        )
    return args


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


def observable_expression(channel):
    if channel == "SR1E2Mu":
        return "(double)mass1"
    if channel == "SR3Mu":
        return "std::min((double)mass1, (double)mass2)"
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


def block_floor_contents(contents, block_nbins):
    """Floor negative content per block, redistributing each block total
    proportionally to the positive fine-bin content within the block."""
    floored = [0.0] * len(contents)
    floored_blocks = 0
    for start in range(0, len(contents), block_nbins):
        block = contents[start:start + block_nbins]
        total = sum(block)
        if any(value < 0.0 for value in block):
            floored_blocks += 1
        if total <= 0.0:
            continue
        positive = [max(0.0, value) for value in block]
        positive_sum = sum(positive)
        if positive_sum > 0.0:
            for offset, value in enumerate(positive):
                floored[start + offset] = total * value / positive_sum
        else:
            share = total / len(block)
            for offset in range(len(block)):
                floored[start + offset] = share
    return floored, floored_blocks


def fill_process_contents(item):
    logging.info("Reading %s", item["path"])
    root_file, tree = validate_tree(item["path"])
    n_entries = int(tree.GetEntries())
    root_file.Close()

    frame = ROOT.RDataFrame("Central", item["path"]).Define(
        "lee_obs", observable_expression(item["channel"])
    )
    weight_sum_result = frame.Sum("weight")
    hist_result = frame.Histo1D(
        (f"lee_fine_{item['subera']}_{item['process']}", "",
         MODEL_NBINS, MODEL_XMIN, MODEL_XMAX),
        "lee_obs",
        "weight",
    )
    weight_sum = float(weight_sum_result.GetValue())
    hist = hist_result.GetValue()
    contents = [hist.GetBinContent(index) for index in range(1, MODEL_NBINS + 1)]
    return contents, {
        "entries": n_entries,
        "weight_sum": weight_sum,
        "in_model_range_weight_sum": sum(contents),
    }


def fill_model_hist(category, inputs, block_nbins):
    hist = ROOT.TH1D(category, category, MODEL_NBINS, MODEL_XMIN, MODEL_XMAX)
    hist.SetDirectory(0)

    total_contents = [0.0] * MODEL_NBINS
    pre_floor_integral = 0.0
    floored_blocks_total = 0
    input_records = []
    for item in inputs:
        contents, stats = fill_process_contents(item)
        raw_in_range = stats["in_model_range_weight_sum"]
        floored, floored_blocks = block_floor_contents(contents, block_nbins)
        floored_sum = sum(floored)
        pre_floor_integral += raw_in_range
        floored_blocks_total += floored_blocks
        for index, value in enumerate(floored):
            total_contents[index] += value
        input_records.append(
            {
                "subera": item["subera"],
                "channel": item["channel"],
                "process": item["process"],
                "path": item["path"],
                "entries": stats["entries"],
                "weight_sum": stats["weight_sum"],
                "in_model_range_weight_sum": raw_in_range,
                "post_floor_weight_sum": floored_sum,
                "floored_blocks": floored_blocks,
            }
        )

    for index, value in enumerate(total_contents):
        hist.SetBinContent(index + 1, value)
        hist.SetBinError(index + 1, 0.0)

    post_floor_integral = hist.Integral()
    floored_yield = post_floor_integral - pre_floor_integral
    floor_fraction = (
        floored_yield / post_floor_integral if post_floor_integral > 0.0 else 0.0
    )

    return hist, {
        "input_files": input_records,
        "pre_floor_integral": pre_floor_integral,
        "post_floor_integral": post_floor_integral,
        "floored_yield": floored_yield,
        "floored_blocks": floored_blocks_total,
        "floor_fraction": floor_fraction,
    }


def datacard_background_total(base_dir, masspoint, category):
    shapes_path = os.path.join(template_dir(base_dir, masspoint), "shapes.root")
    if not os.path.exists(shapes_path):
        raise FileNotFoundError(f"shapes.root not found: {shapes_path}")
    root_file = ROOT.TFile.Open(shapes_path, "READ")
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open shapes file: {shapes_path}")
    try:
        directory = root_file.Get(category)
        if not directory:
            raise KeyError(f"Category '{category}' not found in {shapes_path}")
        # Deduplicate ROOT key cycles by name.
        names = sorted({key.GetName() for key in directory.GetListOfKeys()})
        total = 0.0
        for name in names:
            if name == "data_obs" or name.startswith(("signal", "MHc")):
                continue
            if name.endswith(("Up", "Down")):
                continue
            hist = directory.Get(name)
            if not hist.InheritsFrom("TH1"):
                continue
            total += hist.Integral()
    finally:
        root_file.Close()
    return total


def model_window_integral(hist, mass_min, mass_max):
    total = 0.0
    for index in range(1, hist.GetNbinsX() + 1):
        low = hist.GetBinLowEdge(index)
        high = hist.GetBinLowEdge(index + 1)
        if high <= mass_min or low >= mass_max:
            continue
        overlap = (min(high, mass_max) - max(low, mass_min)) / (high - low)
        total += hist.GetBinContent(index) * overlap
    return total


def check_consistency(base_dir, lee_masspoints, trial_windows, hists):
    """Compare the model projection over each trial window against the
    datacard total background. Records ratios; ratios far from 1 mean the
    toy expectation is biased against the frozen fit model."""
    consistency = {}
    worst = None
    for masspoint in lee_masspoints:
        consistency[masspoint] = {}
        for cat in LEE_CATEGORIES:
            window = trial_windows[masspoint][cat]
            model_integral = model_window_integral(
                hists[cat], window["mass_min"], window["mass_max"]
            )
            datacard_total = datacard_background_total(base_dir, masspoint, cat)
            if datacard_total <= 0.0:
                raise ValueError(
                    f"Non-positive datacard background for {masspoint}/{cat}: "
                    f"{datacard_total}"
                )
            ratio = model_integral / datacard_total
            consistency[masspoint][cat] = {
                "model_integral": model_integral,
                "datacard_background": datacard_total,
                "ratio": ratio,
            }
            deviation = abs(1.0 - ratio)
            if worst is None or deviation > worst[0]:
                worst = (deviation, masspoint, cat, ratio)
            if deviation > CONSISTENCY_WARN_THRESHOLD:
                logging.warning(
                    "Model/datacard mismatch %s/%s: ratio=%.4f",
                    masspoint,
                    cat,
                    ratio,
                )
    logging.info(
        "Consistency check worst deviation: %s/%s ratio=%.4f",
        worst[1],
        worst[2],
        worst[3],
    )
    return consistency


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
    block_nbins = int(round(args.floor_block_width / MODEL_BIN_WIDTH))

    hists = {}
    category_metadata = {}
    for category in LEE_CATEGORIES:
        logging.info("Building LEE background model category %s", category)
        hist, meta = fill_model_hist(category, inputs[category], block_nbins)
        hists[category] = hist
        category_metadata[category] = meta
        logging.info(
            "%s: B=%.6f, flooring added %.6f over %d blocks (fraction %.6g)",
            category,
            meta["post_floor_integral"],
            meta["floored_yield"],
            meta["floored_blocks"],
            meta["floor_fraction"],
        )

    consistency = check_consistency(base_dir, lee_masspoints, trial_windows, hists)

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
        "flooring_scheme": {
            "name": "per_process_block_floor",
            "block_width": args.floor_block_width,
            "block_nbins": block_nbins,
            "description": (
                "Per background process (per subera input file), block totals "
                "are clipped at zero and redistributed proportionally to the "
                "positive fine-bin content, mimicking the per-process "
                "negative-bin flooring of the datacard templates."
            ),
        },
        "trials_masspoints": lee_masspoints,
        "trial_windows": trial_windows,
        "categories": category_metadata,
        "consistency": consistency,
    }

    root_path, json_path = write_outputs(output_dir(base_dir, args.masspoint), hists, metadata)
    logging.info("Wrote %s", root_path)
    logging.info("Wrote %s", json_path)


if __name__ == "__main__":
    main()
