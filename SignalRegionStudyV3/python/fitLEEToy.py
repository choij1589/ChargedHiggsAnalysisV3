#!/usr/bin/env python3
"""Project one LEE toy into frozen templates and fit all LEE trial points."""

import argparse
import glob
import json
import logging
import os
import shutil
import subprocess
import tempfile
from array import array
from datetime import datetime, timezone

import ROOT


ROOT.gROOT.SetBatch(True)

DEFAULT_MASSPOINT = "MHc70_MA18"
LEE_CATEGORIES = ("SR1E2Mu_Run2", "SR3Mu_Run2", "SR1E2Mu_Run3", "SR3Mu_Run3")
BINNING_SUFFIX = "extended_unblind"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--masspoint", default=DEFAULT_MASSPOINT)
    parser.add_argument("--toy", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-workdir", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.toy < 1:
        parser.error("--toy must be >= 1")
    return args


def module_dir():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_json(path):
    with open(path) as handle:
        return json.load(handle)


def save_json_atomic(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    os.replace(tmp_path, path)


def configured_lee_masspoints(base_dir, masspoint):
    config_path = os.path.join(base_dir, "configs", "masspoints.json")
    payload = load_json(config_path)
    masspoints = payload.get("LEE", [])
    if masspoint not in masspoints:
        raise ValueError(
            f"Mass point '{masspoint}' is not in configs/masspoints.json:LEE"
        )
    return masspoints


def template_dir(base_dir, trial_masspoint):
    return os.path.join(
        base_dir,
        "templates",
        "All",
        "Combined",
        trial_masspoint,
        "Baseline",
        BINNING_SUFFIX,
    )


def toy_paths(base_dir, masspoint, toy):
    toy_dir = os.path.join(base_dir, "LEE", masspoint, "toys")
    label = f"toy_{toy:04d}"
    return (
        os.path.join(toy_dir, f"{label}.root"),
        os.path.join(toy_dir, f"{label}.json"),
    )


def fit_json_path(base_dir, masspoint, toy):
    return os.path.join(base_dir, "LEE", masspoint, "fits", f"toy_{toy:04d}.json")


def output_is_complete(path):
    if not os.path.exists(path):
        return False
    try:
        payload = load_json(path)
    except (json.JSONDecodeError, OSError):
        return False
    return payload.get("status") == "complete"


def load_toy(base_dir, masspoint, toy):
    root_path, json_path = toy_paths(base_dir, masspoint, toy)
    if not os.path.exists(root_path):
        raise FileNotFoundError(f"LEE toy ROOT file not found: {root_path}")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"LEE toy JSON file not found: {json_path}")
    metadata = load_json(json_path)
    if metadata.get("status") != "complete":
        raise RuntimeError(f"LEE toy JSON is not complete: {json_path}")

    root_file = ROOT.TFile.Open(root_path, "READ")
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open LEE toy ROOT file: {root_path}")
    return root_path, json_path, metadata, root_file


def load_binning(base_dir, trial_masspoint):
    path = os.path.join(template_dir(base_dir, trial_masspoint), "binning.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Binning metadata not found: {path}")
    payload = load_json(path)
    categories = payload.get("categories", {})
    missing = [cat for cat in LEE_CATEGORIES if cat not in categories]
    if missing:
        raise KeyError(f"{path} missing categories: {', '.join(missing)}")
    return {
        cat: [float(edge) for edge in categories[cat]["bin_edges"]]
        for cat in LEE_CATEGORIES
    }


def project_toy_histograms(toy_file, binning):
    hists = {}
    counts = {}
    for category, edges in binning.items():
        tree = toy_file.Get(category)
        if not tree:
            raise KeyError(f"Toy tree '{category}' not found")
        hist = ROOT.TH1D("data_obs", "data_obs", len(edges) - 1, array("d", edges))
        hist.Sumw2()
        hist.SetDirectory(0)
        in_window = 0
        for event in tree:
            mass = float(event.mass)
            if edges[0] <= mass <= edges[-1]:
                weight = float(getattr(event, "weight", 1.0))
                hist.Fill(mass, weight)
                in_window += 1
        hists[category] = hist
        counts[category] = {
            "toy_entries": int(tree.GetEntries()),
            "in_window_entries": int(in_window),
            "data_obs_integral": float(hist.Integral()),
        }
    return hists, counts


def write_toy_datacard(src_path, dst_path):
    with open(src_path) as handle:
        lines = handle.readlines()
    replaced = False
    with open(dst_path, "w") as handle:
        for line in lines:
            if line.lstrip().startswith("observation"):
                handle.write("observation  -1  -1  -1  -1\n")
                replaced = True
            else:
                handle.write(line)
    if not replaced:
        raise RuntimeError(f"No observation line found in datacard: {src_path}")


def overwrite_data_obs(shapes_path, hists):
    root_file = ROOT.TFile.Open(shapes_path, "UPDATE")
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open staged shapes.root for update: {shapes_path}")
    try:
        for category, hist in hists.items():
            directory = root_file.Get(category)
            if not directory:
                raise KeyError(f"Category directory '{category}' not found in {shapes_path}")
            directory.cd()
            directory.Delete("data_obs;*")
            out_hist = hist.Clone("data_obs")
            out_hist.SetDirectory(directory)
            out_hist.Write("data_obs", ROOT.TObject.kOverwrite)
        root_file.Write("", ROOT.TObject.kOverwrite)
    finally:
        root_file.Close()


def parse_significance(root_path):
    root_file = ROOT.TFile.Open(root_path, "READ")
    if not root_file or root_file.IsZombie():
        raise OSError(f"Could not open combine output: {root_path}")
    try:
        tree = root_file.Get("limit")
        if not tree:
            raise RuntimeError(f"No 'limit' tree in combine output: {root_path}")
        if tree.GetEntries() < 1:
            raise RuntimeError(f"Empty 'limit' tree in combine output: {root_path}")
        tree.GetEntry(0)
        return float(tree.limit)
    finally:
        root_file.Close()


def run_combine(stage_dir, rmin, rmax):
    cmd = [
        "combine",
        "-M",
        "Significance",
        "datacard.txt",
        "--uncapped=1",
        f"--rMin={rmin}",
        f"--rMax={rmax}",
        "-m",
        "120",
        "-n",
        ".lee_toy",
    ]
    proc = subprocess.run(
        cmd,
        cwd=stage_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    log_name = f"combine_r{str(rmin).replace('-', 'm')}_{str(rmax).replace('-', 'm')}.log"
    with open(os.path.join(stage_dir, log_name), "w") as handle:
        handle.write(proc.stdout)
    if proc.returncode != 0:
        raise RuntimeError(f"combine failed with exit code {proc.returncode}")
    outputs = sorted(
        glob.glob(os.path.join(stage_dir, "higgsCombine*.Significance.mH120*.root")),
        key=os.path.getmtime,
    )
    if not outputs:
        raise RuntimeError("combine did not produce a Significance ROOT output")
    return parse_significance(outputs[-1])


def fit_trial(base_dir, toy_file, toy, trial_masspoint, keep_workdir=False):
    source_dir = template_dir(base_dir, trial_masspoint)
    datacard_src = os.path.join(source_dir, "datacard.txt")
    shapes_src = os.path.join(source_dir, "shapes.root")
    if not os.path.exists(datacard_src):
        raise FileNotFoundError(f"Datacard not found: {datacard_src}")
    if not os.path.exists(shapes_src):
        raise FileNotFoundError(f"shapes.root not found: {shapes_src}")

    binning = load_binning(base_dir, trial_masspoint)
    hists, projection = project_toy_histograms(toy_file, binning)

    scratch_base = os.environ.get("TMPDIR") or tempfile.gettempdir()
    stage_dir = tempfile.mkdtemp(
        prefix=f"lee_toy{toy:04d}_{trial_masspoint}_",
        dir=scratch_base,
    )
    try:
        write_toy_datacard(datacard_src, os.path.join(stage_dir, "datacard.txt"))
        shutil.copy2(shapes_src, os.path.join(stage_dir, "shapes.root"))
        overwrite_data_obs(os.path.join(stage_dir, "shapes.root"), hists)

        attempts = []
        for rmin, rmax in [(-20, 20), (-40, 40)]:
            try:
                z_value = run_combine(stage_dir, rmin, rmax)
                attempts.append({"rMin": rmin, "rMax": rmax, "status": "success"})
                return {
                    "status": "success",
                    "Z": z_value,
                    "attempts": attempts,
                    "projection": projection,
                    "stage_dir": stage_dir if keep_workdir else None,
                }
            except Exception as exc:
                attempts.append({
                    "rMin": rmin,
                    "rMax": rmax,
                    "status": "failed",
                    "error": str(exc),
                })
        return {
            "status": "failed",
            "Z": None,
            "attempts": attempts,
            "projection": projection,
            "stage_dir": stage_dir if keep_workdir else None,
        }
    finally:
        if not keep_workdir:
            shutil.rmtree(stage_dir, ignore_errors=True)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    base_dir = module_dir()
    fit_path = fit_json_path(base_dir, args.masspoint, args.toy)
    if not args.force and output_is_complete(fit_path):
        logging.info("Skipping complete Step 3 fit output: %s", fit_path)
        return

    trial_masspoints = configured_lee_masspoints(base_dir, args.masspoint)
    toy_root_path, toy_json_path, toy_metadata, toy_file = load_toy(
        base_dir, args.masspoint, args.toy
    )
    try:
        results = {}
        failures = []
        z_values = []
        for idx, trial_masspoint in enumerate(trial_masspoints, start=1):
            logging.info(
                "Toy %04d: fitting %s (%d/%d)",
                args.toy,
                trial_masspoint,
                idx,
                len(trial_masspoints),
            )
            result = fit_trial(
                base_dir,
                toy_file,
                args.toy,
                trial_masspoint,
                keep_workdir=args.keep_workdir,
            )
            results[trial_masspoint] = result
            if result["Z"] is None:
                failures.append(trial_masspoint)
            else:
                z_values.append(result["Z"])

        payload = {
            "status": "complete",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "masspoint": args.masspoint,
            "toy": args.toy,
            "seed": toy_metadata.get("seed"),
            "source_toy_root": toy_root_path,
            "source_toy_json": toy_json_path,
            "trials_masspoints": trial_masspoints,
            "Z": {
                masspoint: result["Z"]
                for masspoint, result in results.items()
            },
            "Z_max": max(z_values) if z_values else None,
            "Z_min": min(z_values) if z_values else None,
            "failure_count": len(failures),
            "failed_masspoints": failures,
            "results": results,
        }
        save_json_atomic(payload, fit_path)
        logging.info(
            "Wrote %s (Z_max=%s, failures=%d)",
            fit_path,
            payload["Z_max"],
            payload["failure_count"],
        )
    finally:
        toy_file.Close()


if __name__ == "__main__":
    main()
