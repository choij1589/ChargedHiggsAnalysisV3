#!/usr/bin/env python3
"""Generate Run-period component templates for the TTZ2E1Mu control region."""

import argparse
import json
import logging
import os
import shutil
from collections import OrderedDict

import numpy as np
import ROOT

from run_period_utils import (
    PHYSICS_PROCESS_ORDER,
    category_name,
    component_name,
    resolve_run_periods,
)
from template_utils import (
    BIN_FLOOR_VALUE,
    save_json,
    get_output_tree_name,
    combine_suffix_from_tree,
    ensure_positive_integral,
    cap_stat_errors,
    categorize_systematics,
    create_scaled_hist,
    check_binning_quality,
    apply_syst_driven_merging,
)


CHANNEL = "TTZ2E1Mu"
SYSTEMATICS_CHANNEL = "SR1E2Mu"
METHOD = "CR"
BINNING_TAG = "ZWin_adaptive"
DEFAULT_MASSPOINT = "MHc130_MA90"
SYST_MERGE_THRESHOLD = 2.0
MIN_NBINS = 5

BACKGROUND_PROCESSES = [
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--era", required=True, help="Run-period target: Run2, Run3, or All")
    parser.add_argument("--masspoint", default=DEFAULT_MASSPOINT,
                        help="Reference masspoint directory under samples/<subera>/TTZ2E1Mu/")
    parser.add_argument("--window-min", default=81.2, type=float,
                        help="Lower edge of Z mass window in GeV")
    parser.add_argument("--window-max", default=101.2, type=float,
                        help="Upper edge of Z mass window in GeV")
    parser.add_argument("--nbins-init", default=15, type=int,
                        help="Initial number of uniform bins before adaptive scan")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def sample_basedir(workdir, subera, masspoint):
    return f"{workdir}/SignalRegionStudyV3/samples/{subera}/{CHANNEL}/{masspoint}"


def output_dir(workdir, era, masspoint):
    return f"{workdir}/SignalRegionStudyV3/templates/{era}/{CHANNEL}/{masspoint}/{METHOD}/{BINNING_TAG}"


def load_systematics(workdir, subera):
    config_path = f"{workdir}/SignalRegionStudyV3/configs/systematics.{subera}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Systematics config not found: {config_path}")
    with open(config_path) as f:
        config = json.load(f)
    if SYSTEMATICS_CHANNEL not in config:
        raise ValueError(f"Channel '{SYSTEMATICS_CHANNEL}' not found in {config_path}")
    return config[SYSTEMATICS_CHANNEL]


def shape_tree_names(syst_name, variations):
    if isinstance(variations, dict):
        return [f"{syst_name}_{direction}" for direction in ("Up", "Down") if direction in variations]
    return [get_output_tree_name(syst_name, variation) for variation in variations]


def make_hist(file_path, tree_name, bin_edges, mass_min, mass_max, hist_name, use_weight=True):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample file not found: {file_path}")

    rfile = ROOT.TFile.Open(file_path, "READ")
    if not rfile or rfile.IsZombie():
        raise OSError(f"Failed to open: {file_path}")
    tree = rfile.Get(tree_name)
    if not tree:
        rfile.Close()
        raise RuntimeError(f"Tree '{tree_name}' not found in {file_path}")
    rfile.Close()

    edges = ROOT.std.vector["double"]([float(edge) for edge in bin_edges])
    rdf = ROOT.RDataFrame(tree_name, file_path)
    rdf = rdf.Filter(f"mass >= {mass_min} && mass <= {mass_max}")
    if use_weight:
        hist = rdf.Histo1D((hist_name, "", len(bin_edges) - 1, edges.data()), "mass", "weight")
    else:
        hist = rdf.Histo1D((hist_name, "", len(bin_edges) - 1, edges.data()), "mass")
    out = hist.GetValue()
    out.SetDirectory(0)
    return out


def get_central(basedir, process, bin_edges, mass_min, mass_max, hist_name):
    return make_hist(
        f"{basedir}/{process}.root", "Central", bin_edges,
        mass_min, mass_max, hist_name, use_weight=True
    )


def get_data(basedir, bin_edges, mass_min, mass_max, hist_name):
    return make_hist(
        f"{basedir}/data.root", "Central", bin_edges,
        mass_min, mass_max, hist_name, use_weight=False
    )


def get_syst(basedir, process, bin_edges, mass_min, mass_max, tree_name, hist_name):
    try:
        return make_hist(
            f"{basedir}/{process}.root", tree_name, bin_edges,
            mass_min, mass_max, hist_name, use_weight=True
        )
    except (FileNotFoundError, RuntimeError):
        return None


def empty_hist(name, bin_edges):
    edges = ROOT.std.vector["double"]([float(edge) for edge in bin_edges])
    hist = ROOT.TH1D(name, name, len(bin_edges) - 1, edges.data())
    hist.SetDirectory(0)
    return hist


def dummy_signal_hist(name, bin_edges):
    hist = empty_hist(name, bin_edges)
    for ibin in range(1, hist.GetNbinsX() + 1):
        hist.SetBinContent(ibin, BIN_FLOOR_VALUE)
        hist.SetBinError(ibin, BIN_FLOOR_VALUE)
    return hist


def uniform_edges(args, nbins):
    return np.linspace(args.window_min, args.window_max, nbins + 1)


def process_file_exists(basedir, process):
    return os.path.exists(f"{basedir}/{process}.root")


def build_component_templates(basedir, base_process, component, bin_edges, args, syst_categories):
    floor_mode = "floor" if base_process == "others" else "zero"
    central = get_central(
        basedir, base_process, bin_edges,
        args.window_min, args.window_max, component
    )
    central.SetName(component)
    ensure_positive_integral(central, floor_mode=floor_mode)
    cap_stat_errors(central)
    proc_map = {"nominal": central}

    for syst_name, variations, group in syst_categories["preprocessed_shape"]:
        if base_process not in group:
            continue
        for tree_name in shape_tree_names(syst_name, variations):
            suffix = combine_suffix_from_tree(tree_name)
            hist = get_syst(
                basedir, base_process, bin_edges,
                args.window_min, args.window_max, tree_name,
                f"{component}_{suffix}"
            )
            if hist is None:
                logging.debug("Missing CR variation %s/%s", component, tree_name)
                continue
            hist.SetName(f"{component}_{suffix}")
            ensure_positive_integral(hist, floor_mode=floor_mode)
            cap_stat_errors(hist)
            proc_map[suffix] = hist

    for syst_name, value, group in syst_categories["valued_shape"]:
        if base_process not in group:
            continue
        for direction in ("up", "down"):
            hist = create_scaled_hist(central, component, syst_name, value, direction)
            ensure_positive_integral(hist, floor_mode=floor_mode)
            cap_stat_errors(hist)
            suffix = f"{syst_name}Up" if direction == "up" else f"{syst_name}Down"
            proc_map[suffix] = hist

    return proc_map


def choose_category_binning(workdir, suberas, args):
    # TTZ CR has no sideband/core split: the full Z window is the core region.
    # Match the V3 adaptive policy by scanning every integer bin count from the
    # requested starting point down to the minimum allowed core-bin count.
    candidates = list(range(args.nbins_init, MIN_NBINS - 1, -1))
    candidates = [n for n in candidates if n >= MIN_NBINS]
    if not candidates:
        raise ValueError(f"--nbins-init must be >= {MIN_NBINS}, got {args.nbins_init}")

    final_edges = uniform_edges(args, candidates[-1])
    final_nbins = candidates[-1]
    passed = False
    diagnostics_by_n = {}

    for nbins in candidates:
        edges = uniform_edges(args, nbins)
        test_hists = {}
        for subera in suberas:
            basedir = sample_basedir(workdir, subera, args.masspoint)
            for process in BACKGROUND_PROCESSES:
                if not process_file_exists(basedir, process):
                    continue
                component = component_name(process, subera)
                try:
                    test_hists[component] = get_central(
                        basedir, process, edges, args.window_min, args.window_max, component
                    )
                except (FileNotFoundError, RuntimeError) as exc:
                    logging.warning("Skipping %s/%s in bin scan: %s", subera, process, exc)
        ok, diagnostics = check_binning_quality(test_hists)
        diagnostics_by_n[nbins] = diagnostics
        logging.info("  Trying %d CR bins: %s", nbins, "PASS" if ok else f"FAIL ({len(diagnostics)} issues)")
        if ok:
            final_edges = edges
            final_nbins = nbins
            passed = True
            break

    if not passed:
        logging.warning("All CR bin-count candidates failed n_eff; keeping %d bins", final_nbins)

    return final_edges, final_nbins, passed, diagnostics_by_n


def build_category(workdir, outdir, period, suberas, args):
    cat = category_name(CHANNEL, period)
    logging.info("=" * 60)
    logging.info("Building CR category %s", cat)
    logging.info("=" * 60)

    for subera in suberas:
        basedir = sample_basedir(workdir, subera, args.masspoint)
        if not os.path.isdir(basedir):
            raise FileNotFoundError(f"Sample directory not found: {basedir}")

    bin_edges, n_init, passed, diagnostics_by_n = choose_category_binning(workdir, suberas, args)
    logging.info("Final CR bin edges for %s: %s", cat, [round(float(edge), 3) for edge in bin_edges])

    templates = OrderedDict()
    process_metadata = []
    dropped_missing = []
    data_obs = empty_hist("data_obs", bin_edges)

    for subera in suberas:
        basedir = sample_basedir(workdir, subera, args.masspoint)
        data_hist = get_data(basedir, bin_edges, args.window_min, args.window_max, f"data_obs_{subera}")
        data_obs.Add(data_hist)

        signal_component = component_name("signal", subera, is_signal=True)
        templates[signal_component] = {"nominal": dummy_signal_hist(signal_component, bin_edges)}
        process_metadata.append({
            "name": signal_component,
            "base_process": "signal",
            "physics_group": "signal",
            "subera": subera,
            "is_signal": True,
            "dummy_signal": True,
        })

        syst_categories = categorize_systematics(load_systematics(workdir, subera))
        for base_process in BACKGROUND_PROCESSES:
            component = component_name(base_process, subera)
            if not process_file_exists(basedir, base_process):
                dropped_missing.append({
                    "category": cat,
                    "subera": subera,
                    "base_process": base_process,
                    "component": component,
                })
                continue
            try:
                templates[component] = build_component_templates(
                    basedir, base_process, component, bin_edges, args, syst_categories
                )
            except (FileNotFoundError, RuntimeError) as exc:
                logging.warning("Dropping %s: %s", component, exc)
                dropped_missing.append({
                    "category": cat,
                    "subera": subera,
                    "base_process": base_process,
                    "component": component,
                    "reason": str(exc),
                })
                continue
            process_metadata.append({
                "name": component,
                "base_process": base_process,
                "physics_group": base_process,
                "subera": subera,
                "is_signal": False,
            })

    templates["data_obs"] = data_obs
    bkg_components = [meta["name"] for meta in process_metadata if not meta["is_signal"]]
    pre_merge_nbins = len(bin_edges) - 1
    bin_edges, templates, n_merges = apply_syst_driven_merging(
        bin_edges, templates, bkg_components,
        max_rel_syst=SYST_MERGE_THRESHOLD, logger=logging
    )
    if n_merges > 0:
        logging.info("CR syst-merge for %s: %d -> %d bins", cat, pre_merge_nbins, len(bin_edges) - 1)

    return cat, {
        "period": period,
        "channel": CHANNEL,
        "systematics_channel": SYSTEMATICS_CHANNEL,
        "suberas": list(suberas),
        "processes": process_metadata,
        "templates": templates,
        "dropped_missing": dropped_missing,
        "binning": {
            "nbins": len(bin_edges) - 1,
            "bin_edges": [float(edge) for edge in bin_edges],
            "method": "CR_Z_window_run_period_components",
            "window_min": float(args.window_min),
            "window_max": float(args.window_max),
            "nbins_init": int(n_init),
            "min_nbins": MIN_NBINS,
            "adaptive_scan_nbins": [int(nbins) for nbins in sorted(diagnostics_by_n, reverse=True)],
            "adaptive_region": "full_Z_window",
            "automc_binning_passed": bool(passed),
            "diagnostics_by_nbins": {
                str(nbins): diagnostics for nbins, diagnostics in diagnostics_by_n.items()
            },
            "syst_merge_applied": n_merges > 0,
            "n_bins_merged": int(n_merges),
            "syst_merge_threshold": SYST_MERGE_THRESHOLD,
        },
    }


def write_shapes(outdir, categories):
    out = ROOT.TFile.Open(f"{outdir}/shapes.root", "RECREATE")
    for cat, payload in categories.items():
        directory = out.mkdir(cat)
        directory.cd()
        payload["templates"]["data_obs"].Write("data_obs")
        for meta in payload["processes"]:
            process = meta["name"]
            proc_map = payload["templates"].get(process, {})
            nominal = proc_map.get("nominal")
            if not nominal:
                continue
            nominal.Write(process)
            for key, hist in proc_map.items():
                if key == "nominal":
                    continue
                hist.Write(f"{process}_{key}")
        out.cd()
    out.Close()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(levelname)s - %(message)s",
    )

    workdir = os.getenv("WORKDIR")
    if not workdir:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh' first.")

    periods = resolve_run_periods(args.era)
    outdir = output_dir(workdir, args.era, args.masspoint)
    for _, suberas in periods:
        for subera in suberas:
            basedir = sample_basedir(workdir, subera, args.masspoint)
            if not os.path.isdir(basedir):
                raise FileNotFoundError(f"Sample directory not found: {basedir}")

    logging.info("TTZ2E1Mu CR Run-period component template generation")
    logging.info("  Era request: %s", args.era)
    logging.info("  Masspoint reference: %s", args.masspoint)
    logging.info("  Z window: [%.1f, %.1f] GeV", args.window_min, args.window_max)
    logging.info("  Output directory: %s", outdir)

    if os.path.exists(outdir):
        logging.info("Removing existing output directory: %s", outdir)
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    categories = OrderedDict()
    dropped_missing = []
    for period, suberas in periods:
        cat, payload = build_category(workdir, outdir, period, suberas, args)
        categories[cat] = payload
        dropped_missing.extend(payload["dropped_missing"])

    write_shapes(outdir, categories)

    categories_json = {
        "construction": "run_period_components",
        "analysis": "TTZ2E1Mu_CR",
        "data_obs": "real_data",
        "dummy_signal": True,
        "categories": OrderedDict(),
    }
    binning_json = {
        "construction": "run_period_components",
        "analysis": "TTZ2E1Mu_CR",
        "binning_type": BINNING_TAG,
        "categories": OrderedDict(),
    }
    process_components = []
    physics_groups = OrderedDict((group, []) for group in PHYSICS_PROCESS_ORDER)

    for cat, payload in categories.items():
        categories_json["categories"][cat] = {
            "period": payload["period"],
            "channel": payload["channel"],
            "systematics_channel": payload["systematics_channel"],
            "suberas": payload["suberas"],
            "processes": payload["processes"],
        }
        binning_json["categories"][cat] = payload["binning"]
        for meta in payload["processes"]:
            entry = dict(meta)
            entry["category"] = cat
            process_components.append(entry)
            physics_groups.setdefault(meta["physics_group"], []).append(meta["name"])

    save_json(categories_json, f"{outdir}/categories.json")
    save_json(binning_json, f"{outdir}/binning.json")
    save_json({
        "construction": "run_period_components",
        "analysis": "TTZ2E1Mu_CR",
        "signal_is_dummy": True,
        "process_components": process_components,
        "physics_groups": physics_groups,
        "dropped_missing": dropped_missing,
        "description": "TTZ2E1Mu CR Run-period categories with subera component processes.",
    }, f"{outdir}/process_list.json")

    logging.info("=" * 60)
    logging.info("CR Run-period template generation complete")
    logging.info("Output: %s/shapes.root", outdir)
    for cat, payload in categories.items():
        data_obs = payload["templates"]["data_obs"]
        total_bkg = sum(
            payload["templates"][meta["name"]]["nominal"].Integral()
            for meta in payload["processes"]
            if not meta["is_signal"]
        )
        logging.info("  %s: data_obs=%.2f total_bkg=%.2f components=%d",
                     cat, data_obs.Integral(), total_bkg, len(payload["processes"]))


if __name__ == "__main__":
    main()
