#!/usr/bin/env python3
"""Generate LEE pseudo-event toys from the nominal background model."""

import argparse
import json
import logging
import os
from array import array
from datetime import datetime, timezone

import ROOT


ROOT.gROOT.SetBatch(True)

DEFAULT_MASSPOINT = "MHc70_MA18"
DEFAULT_SEED_OFFSET = 12345
LEE_CATEGORIES = ("SR1E2Mu_Run2", "SR3Mu_Run2", "SR1E2Mu_Run3", "SR3Mu_Run3")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masspoint", default=DEFAULT_MASSPOINT)
    parser.add_argument("--toy", type=int, help="Generate one toy index")
    parser.add_argument("--ntoys", type=int, default=1, help="Number of toys to generate")
    parser.add_argument("--start-toy", type=int, default=1, help="First toy index for --ntoys")
    parser.add_argument("--force", action="store_true", help="Overwrite complete existing toys")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.toy is not None and args.toy < 1:
        parser.error("--toy must be >= 1")
    if args.start_toy < 1:
        parser.error("--start-toy must be >= 1")
    if args.ntoys < 1:
        parser.error("--ntoys must be >= 1")
    return args


def module_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def validate_masspoint(base_dir, masspoint):
    config_path = os.path.join(base_dir, "configs", "masspoints.json")
    payload = load_json(config_path)
    lee_masspoints = payload.get("LEE", [])
    if masspoint not in lee_masspoints:
        raise ValueError(
            f"Mass point '{masspoint}' is not in configs/masspoints.json:LEE"
        )


def model_paths(base_dir, masspoint):
    model_dir = os.path.join(base_dir, "LEE", masspoint, "model")
    return (
        os.path.join(model_dir, "bkg_model.root"),
        os.path.join(model_dir, "bkg_model.json"),
    )


def toy_paths(base_dir, masspoint, toy):
    toy_dir = os.path.join(base_dir, "LEE", masspoint, "toys")
    label = f"toy_{toy:04d}"
    return (
        toy_dir,
        os.path.join(toy_dir, f"{label}.root"),
        os.path.join(toy_dir, f"{label}.json"),
    )


def toy_indices(args):
    if args.toy is not None:
        return [args.toy]
    return list(range(args.start_toy, args.start_toy + args.ntoys))


def is_complete(root_path, json_path):
    if not (os.path.exists(root_path) and os.path.exists(json_path)):
        return False
    try:
        payload = load_json(json_path)
    except (json.JSONDecodeError, OSError):
        return False
    return payload.get("status") == "complete"


def load_model(base_dir, masspoint):
    root_path, json_path = model_paths(base_dir, masspoint)
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"LEE background model ROOT file not found: {root_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"LEE background model JSON file not found: {json_path}")

    metadata = load_json(json_path)
    root_file = ROOT.TFile.Open(root_path, "READ")
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open LEE background model ROOT file: {root_path}")

    hists = {}
    for category in LEE_CATEGORIES:
        hist = root_file.Get(category)
        if not hist:
            root_file.Close()
            raise KeyError(f"Histogram '{category}' not found in {root_path}")
        clone = hist.Clone(category)
        clone.SetDirectory(0)
        hists[category] = clone
    root_file.Close()
    return root_path, json_path, metadata, hists


def generate_one_toy(base_dir, masspoint, toy, model_root_path, model_json_path,
                     model_metadata, hists, force=False):
    toy_dir, root_path, json_path = toy_paths(base_dir, masspoint, toy)
    if not force and is_complete(root_path, json_path):
        logging.info("Skipping complete toy %04d: %s", toy, root_path)
        return {"toy": toy, "status": "skipped", "root": root_path, "json": json_path}

    os.makedirs(toy_dir, exist_ok=True)
    tmp_root_path = f"{root_path}.tmp"
    tmp_json_path = f"{json_path}.tmp"
    for tmp_path in (tmp_root_path, tmp_json_path):
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    seed = DEFAULT_SEED_OFFSET + toy
    rng = ROOT.TRandom3(seed)
    category_counts = {}

    root_file = ROOT.TFile.Open(tmp_root_path, "RECREATE")
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not create toy ROOT file: {tmp_root_path}")

    try:
        for category in LEE_CATEGORIES:
            hist = hists[category]
            tree = ROOT.TTree(category, category)
            mass = array("d", [0.0])
            weight = array("d", [1.0])
            tree.Branch("mass", mass, "mass/D")
            tree.Branch("weight", weight, "weight/D")

            count = 0
            for idx in range(1, hist.GetNbinsX() + 1):
                mu = max(0.0, float(hist.GetBinContent(idx)))
                n_events = int(rng.Poisson(mu))
                if n_events <= 0:
                    continue
                low = float(hist.GetBinLowEdge(idx))
                high = float(hist.GetBinLowEdge(idx + 1))
                for _ in range(n_events):
                    mass[0] = float(rng.Uniform(low, high))
                    weight[0] = 1.0
                    tree.Fill()
                count += n_events

            root_file.cd()
            tree.Write(category)
            category_counts[category] = count
    finally:
        root_file.Close()

    payload = {
        "status": "complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "masspoint": masspoint,
        "toy": toy,
        "seed": seed,
        "seed_offset": DEFAULT_SEED_OFFSET,
        "source_model_root": model_root_path,
        "source_model_json": model_json_path,
        "model_binning": model_metadata.get("model_binning", {}),
        "categories": {
            category: {
                "entries": category_counts[category],
                "expected_yield": float(hists[category].Integral()),
            }
            for category in LEE_CATEGORIES
        },
    }
    with open(tmp_json_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)

    os.replace(tmp_root_path, root_path)
    os.replace(tmp_json_path, json_path)
    logging.info(
        "Wrote toy %04d: %s (%d total entries)",
        toy,
        root_path,
        sum(category_counts.values()),
    )
    return {"toy": toy, "status": "written", "root": root_path, "json": json_path}


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    base_dir = module_dir()
    validate_masspoint(base_dir, args.masspoint)
    model_root_path, model_json_path, model_metadata, hists = load_model(
        base_dir, args.masspoint
    )

    results = []
    for toy in toy_indices(args):
        results.append(
            generate_one_toy(
                base_dir,
                args.masspoint,
                toy,
                model_root_path,
                model_json_path,
                model_metadata,
                hists,
                force=args.force,
            )
        )

    written = sum(1 for item in results if item["status"] == "written")
    skipped = sum(1 for item in results if item["status"] == "skipped")
    logging.info("Toy generation complete: written=%d skipped=%d", written, skipped)


if __name__ == "__main__":
    main()
