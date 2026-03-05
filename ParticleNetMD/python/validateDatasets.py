#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Validate saved PyTorch Geometric datasets by reconstructing physics observables
and plotting distributions.

Two phases:
  Fill phase  (default, slow, parallelized):
      Load .pt → extract observables → fill histograms → save ROOT files
  Plot phase  (--plotting, fast):
      Read ROOT files → merge folds → draw with KinematicCanvas → save PNGs

class_overlay : All processes (3 signals + 3 backgrounds) shape comparison
fold_overlay  : Per-process fold balance check (fold 0-4 overlaid)

Output structure:
    DataAugment/validation/histograms/{process}/{channel}_fold{fold}.root   (fill phase)
    DataAugment/validation/{RUN}/{CHANNEL}/class_overlay/{obs}.png          (plot phase)
    DataAugment/validation/{RUN}/{CHANNEL}/fold_overlay/{process}/{obs}.png
    DataAugment/validation/{RUN}/{CHANNEL}/summary.json

Usage:
    # Phase 1: Fill histograms (parallel) and save to ROOT
    python python/validateDatasets.py [--channel CHANNEL] [--folds 0,1,2,3,4] [--workers 8]

    # Phase 2: Read saved ROOT files and plot
    python python/validateDatasets.py --plotting [--channel CHANNEL] [--folds 0,1,2,3,4]
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import json
import argparse
import logging
from collections import OrderedDict
from multiprocessing import Pool

# Environment
WORKDIR = os.environ.get("WORKDIR")
if not WORKDIR:
    raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'")

sys.path.insert(0, os.path.join(WORKDIR, "Common", "Tools"))

# ─── Constants ─────────────────────────────────────────────────────────────────

CHANNELS = ["Run1E2Mu", "Run3Mu"]
CHANNEL_DISPLAY = {"Run1E2Mu": "SR1E2Mu", "Run3Mu": "SR3Mu"}

SIGNALS = [
    "TTToHcToWAToMuMu-MHc130_MA90",
    "TTToHcToWAToMuMu-MHc100_MA95",
    "TTToHcToWAToMuMu-MHc160_MA85",
]
SIGNAL_DISPLAY = {
    "TTToHcToWAToMuMu-MHc130_MA90": "MHc130_MA90",
    "TTToHcToWAToMuMu-MHc100_MA95": "MHc100_MA95",
    "TTToHcToWAToMuMu-MHc160_MA85": "MHc160_MA85",
}

BG_GROUPS = ["nonprompt", "diboson", "ttX"]

RUN2_ERAS = {"2016preVFP", "2016postVFP", "2017", "2018"}
RUN3_ERAS = {"2022", "2022EE", "2023", "2023BPix"}

# ─── CLI ───────────────────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(description="Validate ParticleNetMD datasets")
    parser.add_argument("--channel", default="all",
                        choices=["Run1E2Mu", "Run3Mu", "all"])
    parser.add_argument("--folds", default="0,1,2,3,4",
                        help="Comma-separated fold indices to include")
    parser.add_argument("--config", default=None,
                        help="Path to SglConfig.json")
    parser.add_argument("--obs-config", default=None,
                        help="Path to histkeys_validate.json")
    parser.add_argument("--plotting", "--plot-only", action="store_true",
                        help="Plot phase: read ROOT files and produce plots")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers for fill phase")
    return parser.parse_args()


# ─── Configuration Loading ─────────────────────────────────────────────────────


def load_json_config(path):
    with open(path) as f:
        return json.load(f)


def resolve_background_samples(config, backgrounds_dir):
    """Resolve background group config names to actual directories on disk."""
    prefix = config["dataset_config"]["background_prefix"]
    era_mapping = config["dataset_config"].get("era_sample_mapping", {})
    groups = config["background_config"]["background_groups"]

    resolved = {}
    for group_name, sample_list in groups.items():
        dirs = []
        for sample in sample_list:
            primary = prefix + sample
            if os.path.isdir(os.path.join(backgrounds_dir, primary)):
                dirs.append(primary)
                continue
            if sample in era_mapping:
                for variant in era_mapping[sample].values():
                    variant_dir = prefix + variant
                    if os.path.isdir(os.path.join(backgrounds_dir, variant_dir)):
                        if variant_dir not in dirs:
                            dirs.append(variant_dir)
                continue
            logging.warning(f"Directory not found for sample '{sample}' "
                            f"(tried {primary})")
        resolved[group_name] = dirs
    return resolved


# ─── Observable Extraction ─────────────────────────────────────────────────────


def extract_observables(data):
    """Extract physics observables from a single PyG Data object.

    Node features: [E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]
    B-jets have isJet=1 AND isBjet=1; regular jets have isJet=1, isBjet=0.
    """
    import torch

    x = data.x
    Px, Py, Pz = x[:, 1], x[:, 2], x[:, 3]
    is_muon, is_electron = x[:, 5], x[:, 6]
    is_jet, is_bjet = x[:, 7], x[:, 8]

    pT = torch.sqrt(Px ** 2 + Py ** 2)
    p = torch.sqrt(Px ** 2 + Py ** 2 + Pz ** 2)
    eta = torch.arctanh(torch.clamp(Pz / (p + 1e-10), -1 + 1e-7, 1 - 1e-7))
    phi = torch.atan2(Py, Px)

    obs = {}

    # --- Leptons (all) sorted by pT ---
    lep_mask = (is_muon > 0.5) | (is_electron > 0.5)
    lep_pt = pT[lep_mask]
    lep_eta = eta[lep_mask]
    lep_phi = phi[lep_mask]
    idx = torch.argsort(lep_pt, descending=True)
    lep_pt, lep_eta, lep_phi = lep_pt[idx], lep_eta[idx], lep_phi[idx]
    n = lep_pt.size(0)
    obs["lep1_pt"]  = lep_pt[0].item()  if n >= 1 else None
    obs["lep2_pt"]  = lep_pt[1].item()  if n >= 2 else None
    obs["lep3_pt"]  = lep_pt[2].item()  if n >= 3 else None
    obs["lep1_eta"] = lep_eta[0].item() if n >= 1 else None
    obs["lep2_eta"] = lep_eta[1].item() if n >= 2 else None
    obs["lep3_eta"] = lep_eta[2].item() if n >= 3 else None
    obs["lep1_phi"] = lep_phi[0].item() if n >= 1 else None
    obs["lep2_phi"] = lep_phi[1].item() if n >= 2 else None
    obs["lep3_phi"] = lep_phi[2].item() if n >= 3 else None

    # --- Muons sorted by pT ---
    mu_mask = is_muon > 0.5
    mu_pt = pT[mu_mask]
    mu_eta = eta[mu_mask]
    mu_phi = phi[mu_mask]
    idx = torch.argsort(mu_pt, descending=True)
    mu_pt, mu_eta, mu_phi = mu_pt[idx], mu_eta[idx], mu_phi[idx]
    nmu = mu_pt.size(0)
    obs["mu1_pt"]  = mu_pt[0].item()  if nmu >= 1 else None
    obs["mu2_pt"]  = mu_pt[1].item()  if nmu >= 2 else None
    obs["mu1_eta"] = mu_eta[0].item() if nmu >= 1 else None
    obs["mu2_eta"] = mu_eta[1].item() if nmu >= 2 else None
    obs["mu1_phi"] = mu_phi[0].item() if nmu >= 1 else None
    obs["mu2_phi"] = mu_phi[1].item() if nmu >= 2 else None

    # --- Electrons sorted by pT ---
    ele_mask = is_electron > 0.5
    ele_pt = pT[ele_mask]
    ele_eta = eta[ele_mask]
    ele_phi = phi[ele_mask]
    idx = torch.argsort(ele_pt, descending=True)
    ele_pt, ele_eta, ele_phi = ele_pt[idx], ele_eta[idx], ele_phi[idx]
    nele = ele_pt.size(0)
    obs["ele1_pt"]  = ele_pt[0].item()  if nele >= 1 else None
    obs["ele1_eta"] = ele_eta[0].item() if nele >= 1 else None
    obs["ele1_phi"] = ele_phi[0].item() if nele >= 1 else None

    # --- Regular jets (isJet=1, isBjet=0) sorted by pT ---
    jet_mask = (is_jet > 0.5) & (is_bjet < 0.5)
    jet_pt = pT[jet_mask]
    jet_eta = eta[jet_mask]
    jet_phi = phi[jet_mask]
    idx = torch.argsort(jet_pt, descending=True)
    jet_pt, jet_eta, jet_phi = jet_pt[idx], jet_eta[idx], jet_phi[idx]
    nj = jet_pt.size(0)
    obs["jet1_pt"]  = jet_pt[0].item()  if nj >= 1 else None
    obs["jet1_eta"] = jet_eta[0].item() if nj >= 1 else None
    obs["jet1_phi"] = jet_phi[0].item() if nj >= 1 else None
    obs["jet2_pt"]  = jet_pt[1].item()  if nj >= 2 else None
    obs["jet2_eta"] = jet_eta[1].item() if nj >= 2 else None
    obs["jet2_phi"] = jet_phi[1].item() if nj >= 2 else None

    # --- B-jets (isBjet=1) sorted by pT ---
    bjet_mask = is_bjet > 0.5
    bjet_pt = pT[bjet_mask]
    bjet_eta = eta[bjet_mask]
    bjet_phi = phi[bjet_mask]
    idx = torch.argsort(bjet_pt, descending=True)
    bjet_pt, bjet_eta, bjet_phi = bjet_pt[idx], bjet_eta[idx], bjet_phi[idx]
    nb = bjet_pt.size(0)
    obs["bjet1_pt"]  = bjet_pt[0].item()  if nb >= 1 else None
    obs["bjet1_eta"] = bjet_eta[0].item() if nb >= 1 else None
    obs["bjet1_phi"] = bjet_phi[0].item() if nb >= 1 else None
    obs["bjet2_pt"]  = bjet_pt[1].item()  if nb >= 2 else None
    obs["bjet2_eta"] = bjet_eta[1].item() if nb >= 2 else None
    obs["bjet2_phi"] = bjet_phi[1].item() if nb >= 2 else None

    # Multiplicities (nJets counts all jets including bjets)
    obs["nJets"] = int((is_jet > 0.5).sum().item())
    obs["nBjets"] = int(nb)
    obs["nParticles"] = x.size(0)

    # HT = sum pT of all jets (isJet=1 includes bjets)
    all_jet_mask = is_jet > 0.5
    obs["HT"] = pT[all_jet_mask].sum().item() if all_jet_mask.any() else 0.0

    # MET = negative vector sum of all visible particle pT
    met_px = -Px.sum().item()
    met_py = -Py.sum().item()
    import math
    obs["MET"] = math.sqrt(met_px ** 2 + met_py ** 2)
    obs["MET_phi"] = math.atan2(met_py, met_px)

    # OS muon pair masses (mass2 = -1 when only one pair exists)
    m1 = data.mass1.item() if hasattr(data, "mass1") else -1.0
    m2 = data.mass2.item() if hasattr(data, "mass2") else -1.0
    obs["mass1"] = m1 if m1 >= 0 else None
    obs["mass2"] = m2 if m2 >= 0 else None

    obs["weight"] = data.weight.item()
    return obs


# ─── Data Loading ──────────────────────────────────────────────────────────────


def load_fold_data(channel, samples_dir, sample_dirs, fold):
    """Load a single fold for given sample directories. Returns list of Data."""
    import torch

    data = []
    for sample_dir in sample_dirs:
        filepath = os.path.join(samples_dir, sample_dir,
                                f"{channel}_fold-{fold}.pt")
        if not os.path.exists(filepath):
            continue
        try:
            dataset = torch.load(filepath, weights_only=False)
        except (EOFError, RuntimeError) as e:
            logging.warning(f"Skipping corrupted file {filepath}: {e}")
            continue
        data.extend(dataset.data_list if hasattr(dataset, "data_list") else [])
    return data


def detect_runs(data_list):
    """Detect which run periods are present in a data list."""
    runs = set()
    for d in data_list:
        era = d.era if hasattr(d, "era") else ""
        if era in RUN2_ERAS:
            runs.add("Run2")
        elif era in RUN3_ERAS:
            runs.add("Run3")
    return sorted(runs)


def filter_by_run(data_list, run):
    """Filter events belonging to a run period."""
    eras = RUN2_ERAS if run == "Run2" else RUN3_ERAS
    return [d for d in data_list if (d.era if hasattr(d, "era") else "") in eras]


# ─── Histogram Utilities ──────────────────────────────────────────────────────


def create_histogram(name, obs_cfg):
    """Create a ROOT TH1D with appropriate binning."""
    import ROOT
    xmin, xmax = obs_cfg["xRange"]
    if obs_cfg.get("integer", False):
        nbins = int(xmax - xmin)
    else:
        nbins = max(200, int(xmax - xmin))
    h = ROOT.TH1D(name, "", nbins, xmin, xmax)
    h.SetDirectory(0)
    h.Sumw2()
    return h


def fill_histograms(data_list, obs_config, prefix):
    """Fill histograms for all observables from a list of Data objects."""
    hists = {}
    for obs_name, obs_cfg in obs_config.items():
        hists[obs_name] = create_histogram(f"{prefix}_{obs_name}", obs_cfg)

    for data in data_list:
        obs = extract_observables(data)
        w = obs["weight"]
        for obs_name in obs_config:
            val = obs.get(obs_name)
            if val is not None:
                hists[obs_name].Fill(val, w)

    return hists


# ─── ROOT I/O ─────────────────────────────────────────────────────────────────


def save_histograms_to_root(hists_by_run, metadata_by_run, output_path):
    """Save {run: {obs: TH1D}} + metadata to ROOT file."""
    import ROOT
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    f = ROOT.TFile.Open(output_path, "RECREATE")
    for run in sorted(hists_by_run.keys()):
        d = f.mkdir(run)
        d.cd()
        for obs_name, h in hists_by_run[run].items():
            h.Write(obs_name)
        metadata_by_run[run].Write("metadata")
    f.Close()


def load_histograms_from_root(filepath, obs_names):
    """Load {run: {obs: TH1D}} from ROOT file. Metadata stored under '_metadata' key."""
    import ROOT
    f = ROOT.TFile.Open(filepath, "READ")
    if not f or f.IsZombie():
        return {}
    result = {}
    for run in ["Run2", "Run3"]:
        d = f.Get(run)
        if not d:
            continue
        result[run] = {}
        for obs in obs_names:
            h = d.Get(obs)
            if h:
                h.SetDirectory(0)
                result[run][obs] = h
        meta = d.Get("metadata")
        if meta:
            meta.SetDirectory(0)
            result[run]["_metadata"] = meta
    f.Close()
    return result


# ─── Fill Phase (parallel) ────────────────────────────────────────────────────


def _init_worker():
    """Initializer for each worker process."""
    import torch
    import ROOT
    torch.set_num_threads(1)
    ROOT.gROOT.SetBatch(True)


def fill_worker(args):
    """Worker function: load one (process, channel, fold), fill histograms, save ROOT.

    Returns (output_path, log_message).
    """
    import ROOT
    (process_name, sample_dirs, samples_base_dir, channel, fold,
     obs_config, output_path) = args

    data_list = load_fold_data(channel, samples_base_dir, sample_dirs, fold)
    if not data_list:
        return (output_path, f"  SKIP {process_name}/{channel}/fold{fold}: no data")

    runs = detect_runs(data_list)
    hists_by_run = {}
    metadata_by_run = {}

    for run in runs:
        run_data = filter_by_run(data_list, run)
        if not run_data:
            continue
        prefix = f"{process_name}_{channel}_f{fold}_{run}"
        hists_by_run[run] = fill_histograms(run_data, obs_config, prefix)

        # Metadata: 1-bin histogram storing event count (content) and weight sum (error)
        meta = ROOT.TH1D(f"{prefix}_meta", "", 1, 0, 1)
        meta.SetDirectory(0)
        n_events = len(run_data)
        weight_sum = sum(d.weight.item() for d in run_data)
        meta.SetBinContent(1, n_events)
        meta.SetBinError(1, weight_sum)
        metadata_by_run[run] = meta

    if hists_by_run:
        save_histograms_to_root(hists_by_run, metadata_by_run, output_path)

    n_total = len(data_list)
    return (output_path, f"  OK {process_name}/{channel}/fold{fold}: {n_total} events → {output_path}")


def run_fill_phase(channels, folds, config, obs_config, n_workers):
    """Phase 1: Fill histograms in parallel and save ROOT files."""
    dataset_dir = os.path.join(WORKDIR, "ParticleNetMD", "dataset", "samples")
    signals_dir = os.path.join(dataset_dir, "signals")
    backgrounds_dir = os.path.join(dataset_dir, "backgrounds")
    results_base = os.path.join(WORKDIR, "ParticleNetMD", "DataAugment", "validation", "histograms")

    bg_samples = resolve_background_samples(config, backgrounds_dir)
    for group, dirs in bg_samples.items():
        logging.info(f"Background group '{group}': {dirs}")

    # Build job list: (process_name, sample_dirs, base_dir, channel, fold, obs_config, output_path)
    jobs = []
    for channel in channels:
        for fold in folds:
            # Signals
            for sig in SIGNALS:
                label = SIGNAL_DISPLAY[sig]
                output_path = os.path.join(results_base, label,
                                           f"{channel}_fold{fold}.root")
                jobs.append((label, [sig], signals_dir, channel, fold,
                             obs_config, output_path))
            # Backgrounds
            for group_name in BG_GROUPS:
                dirs = bg_samples.get(group_name, [])
                output_path = os.path.join(results_base, group_name,
                                           f"{channel}_fold{fold}.root")
                jobs.append((group_name, dirs, backgrounds_dir, channel, fold,
                             obs_config, output_path))

    logging.info(f"Fill phase: {len(jobs)} jobs with {n_workers} workers")

    with Pool(n_workers, initializer=_init_worker) as pool:
        results = pool.map(fill_worker, jobs)

    for _, msg in results:
        logging.info(msg)

    logging.info(f"Fill phase complete. ROOT files in {results_base}")


# ─── Plotting ──────────────────────────────────────────────────────────────────


def _add_overflow_underflow(hist, obs_cfg):
    """Add ROOT under/overflow bin content to first/last visible bins."""
    import math
    nbins = hist.GetNbinsX()
    if obs_cfg.get("overflow", False):
        last = nbins
        of_c = hist.GetBinContent(nbins + 1)
        of_e2 = hist.GetBinError(nbins + 1) ** 2
        hist.SetBinContent(last, hist.GetBinContent(last) + of_c)
        hist.SetBinError(last, math.sqrt(hist.GetBinError(last) ** 2 + of_e2))
        hist.SetBinContent(nbins + 1, 0)
        hist.SetBinError(nbins + 1, 0)
    if obs_cfg.get("underflow", False):
        uf_c = hist.GetBinContent(0)
        uf_e2 = hist.GetBinError(0) ** 2
        hist.SetBinContent(1, hist.GetBinContent(1) + uf_c)
        hist.SetBinError(1, math.sqrt(hist.GetBinError(1) ** 2 + uf_e2))
        hist.SetBinContent(0, 0)
        hist.SetBinError(0, 0)


def plot_overlay(hists_dict, obs_config, output_dir, canvas_config_base,
                 process_text=None):
    """Draw shape-comparison overlay plots (no ratio pad) for all observables.

    Args:
        hists_dict: OrderedDict of {label: {obs_name: TH1D}}.
        obs_config: Observable config dict.
        output_dir: Output directory for PNGs.
        canvas_config_base: Base config dict for KinematicCanvas.
        process_text: Optional process label drawn below channel text.
    """
    import ROOT
    import cmsstyle as CMS
    from plotter import KinematicCanvas

    os.makedirs(output_dir, exist_ok=True)

    for obs_name, obs_cfg in obs_config.items():
        ordered = OrderedDict()
        for label in hists_dict:
            h = hists_dict[label].get(obs_name)
            if h is not None and h.GetEntries() > 0:
                _add_overflow_underflow(h, obs_cfg)
                ordered[label] = h

        if len(ordered) < 2:
            continue

        config = dict(canvas_config_base)
        config["xTitle"] = obs_cfg["xTitle"]
        config["xRange"] = obs_cfg["xRange"]
        if "rebin" in obs_cfg:
            config["rebin"] = obs_cfg["rebin"]
        if obs_cfg.get("logy", False):
            config["logy"] = True
        n = len(ordered)
        config["legend"] = [0.60, 0.89 - 0.04 * n, 0.85, 0.89, 0.04]

        try:
            canvas = KinematicCanvas(ordered, config)
            canvas.drawPad()

            # Draw process text below channel text
            if process_text:
                CMS.drawText(process_text, posX=0.2, posY=0.65,
                             font=42, align=0, size=0.04)

            canvas.canv.RedrawAxis()
            canvas.canv.SaveAs(os.path.join(output_dir, f"{obs_name}.png"))
        except Exception as e:
            logging.warning(f"Failed to plot {obs_name}: {e}")


def run_plot_phase(channels, folds, obs_config):
    """Phase 2: Read ROOT files, merge folds, produce plots."""
    import ROOT
    ROOT.gROOT.SetBatch(True)

    results_base = os.path.join(WORKDIR, "ParticleNetMD", "DataAugment", "validation", "histograms")
    output_base = os.path.join(WORKDIR, "ParticleNetMD", "DataAugment", "validation")
    obs_names = list(obs_config.keys())

    # All process labels
    all_processes = [SIGNAL_DISPLAY[s] for s in SIGNALS] + list(BG_GROUPS)

    RUN_COM = {"Run2": 13, "Run3": 13.6}

    base_canvas_config = {
        "era": "Run2",
        "iPos": 0,
        "channelPosX": 0.18,
        "channelPosY": 0.82,
        "normalize": True,
        "yTitle": "Normalized",
    }

    for channel in channels:
        logging.info(f"\n{'=' * 60}")
        logging.info(f"Channel: {channel}")
        logging.info(f"{'=' * 60}")

        # Load all ROOT files for this channel
        # process -> fold -> {run: {obs: TH1D, '_metadata': TH1D}}
        loaded = {}
        for proc in all_processes:
            loaded[proc] = {}
            for fold in folds:
                filepath = os.path.join(results_base, proc,
                                        f"{channel}_fold{fold}.root")
                if os.path.exists(filepath):
                    loaded[proc][fold] = load_histograms_from_root(filepath, obs_names)
                else:
                    loaded[proc][fold] = {}

        # Detect available runs
        available_runs = set()
        for proc in all_processes:
            for fold in folds:
                for run in loaded[proc][fold]:
                    if run in ("Run2", "Run3"):
                        available_runs.add(run)
        available_runs = sorted(available_runs)
        if not available_runs:
            logging.warning(f"No ROOT files found for {channel}, skipping")
            continue
        logging.info(f"Available runs: {available_runs}")

        for run in available_runs:
            logging.info(f"\n--- {run} / {channel} ---")
            run_dir = os.path.join(output_base, run, channel)

            # ── Class overlay: merge folds per process, overlay processes ──
            logging.info("  class_overlay")
            class_hists = OrderedDict()
            for proc in all_processes:
                merged = {}
                for fold in folds:
                    fold_data = loaded[proc][fold].get(run, {})
                    for obs_name in obs_names:
                        h = fold_data.get(obs_name)
                        if h is None:
                            continue
                        if obs_name not in merged:
                            merged[obs_name] = h.Clone(f"cls_{run}_{channel}_{proc}_{obs_name}")
                            merged[obs_name].SetDirectory(0)
                        else:
                            merged[obs_name].Add(h)
                if merged:
                    class_hists[proc] = merged

            cls_config = dict(base_canvas_config)
            cls_config["era"] = run
            cls_config["run_label"] = run
            cls_config["CoM"] = RUN_COM[run]
            cls_config["channel"] = CHANNEL_DISPLAY.get(channel, channel)
            plot_overlay(
                class_hists, obs_config,
                os.path.join(run_dir, "class_overlay"),
                cls_config,
            )

            # ── Fold overlay: per process, fold 0-4 on one plot ──
            logging.info("  fold_overlay")
            for proc in all_processes:
                fold_hists = OrderedDict()
                for fold in folds:
                    fold_data = loaded[proc][fold].get(run, {})
                    obs_hists = {}
                    for obs_name in obs_names:
                        h = fold_data.get(obs_name)
                        if h is not None:
                            obs_hists[obs_name] = h
                    if obs_hists:
                        fold_hists[f"fold {fold}"] = obs_hists

                if len(fold_hists) < 2:
                    continue

                fold_config = dict(base_canvas_config)
                fold_config["era"] = run
                fold_config["run_label"] = run
                fold_config["CoM"] = RUN_COM[run]
                fold_config["channel"] = CHANNEL_DISPLAY.get(channel, channel)
                plot_overlay(
                    fold_hists, obs_config,
                    os.path.join(run_dir, "fold_overlay", proc),
                    fold_config,
                    process_text=proc,
                )

            # ── Summary JSON from metadata ──
            summary = {
                "run": run, "channel": channel,
                "processes": {},
            }
            for proc in all_processes:
                total_events = 0
                total_weight = 0.0
                fold_counts = {}
                for fold in folds:
                    fold_data = loaded[proc][fold].get(run, {})
                    meta = fold_data.get("_metadata")
                    if meta:
                        n_ev = int(meta.GetBinContent(1))
                        w_sum = meta.GetBinError(1)
                        total_events += n_ev
                        total_weight += w_sum
                        fold_counts[str(fold)] = n_ev
                    else:
                        fold_counts[str(fold)] = 0
                summary["processes"][proc] = {
                    "events": total_events,
                    "weight_sum": round(total_weight, 4),
                    "folds": fold_counts,
                }

            summary_path = os.path.join(run_dir, "summary.json")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)
            logging.info(f"  Summary: {summary_path}")

    logging.info(f"\nDone! Plots saved to {output_base}")


# ─── Main ──────────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
    args = parse_args()

    # Load configs
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    config_path = args.config or os.path.join(base_dir, "configs", "SglConfig.json")
    obs_config_path = args.obs_config or os.path.join(
        base_dir, "configs", "histkeys_validate.json"
    )
    config = load_json_config(config_path)
    obs_config = load_json_config(obs_config_path)

    folds = [int(f) for f in args.folds.split(",")]
    channels = CHANNELS if args.channel == "all" else [args.channel]

    if args.plotting:
        run_plot_phase(channels, folds, obs_config)
    else:
        run_fill_phase(channels, folds, config, obs_config, args.workers)


if __name__ == "__main__":
    main()
