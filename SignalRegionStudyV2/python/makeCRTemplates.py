#!/usr/bin/env python3
"""
Generate background-only binned templates for the TTZ2E1Mu control region.

The CR fits M(ee) in the Z window |M(ee) - 91.2| < 10 GeV. Initial binning is
15 uniform bins in [81.2, 101.2] GeV (1.33 GeV/bin), then the same adaptive
n_eff>=5 merge loop used by the SR pipeline shrinks the binning until every
bin satisfies Combine's autoMCStats criterion.

Output layout (mirrors SR five-segment paths so existing tools work):
    templates/{era}/TTZ2E1Mu/MHc130_MA90/CR/ZWin_adaptive/

The output shapes.root contains:
  - data_obs (real data, since the CR has no blinding)
  - 8 background processes (nonprompt, WZ, ZZ, ttW, ttZ, ttH, tZq, conversion) + others
  - Per-process Up/Down systematic variations driven by the SR1E2Mu config
    (preprocess.py:CHANNEL_CONFIG_MAP[TTZ2E1Mu] = "SR1E2Mu", so the same shape
    NPs, trigger SF NPs, nonprompt norm NPs, etc. are shared with SR1E2Mu.)
  - A dummy 'signal' placeholder histogram (1e-6/bin) so the datacard retains
    an `r` parameter and the existing partial-unblind GoF combine command
    (`--freezeParameters r --setParameters r=0`) works unchanged. With r=0
    the placeholder contributes zero to the likelihood.

Usage:
    python3 python/makeCRTemplates.py --era 2018
"""
import os
import shutil
import logging
import argparse
import json
import ROOT
import numpy as np

from template_utils import (
    BIN_FLOOR_VALUE,
    SHAPE_REL_ERR_THRESHOLD,
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


# Fixed parameters for the CR fit
CHANNEL = "TTZ2E1Mu"
METHOD = "CR"
BINNING_TAG = "ZWin_adaptive"
DEFAULT_MASSPOINT = "MHc130_MA90"   # reference masspoint dir (CR is masspoint-agnostic)
SR_CONFIG_CHANNEL = "SR1E2Mu"        # systematics block to read (same as preprocess.py)
SYST_MERGE_THRESHOLD = 2.0           # same threshold as SR pipeline


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--era", required=True, type=str,
                        help="Data-taking era (e.g., 2018, 2022EE, ...)")
    parser.add_argument("--masspoint", default=DEFAULT_MASSPOINT, type=str,
                        help="Reference masspoint dir under samples/{era}/TTZ2E1Mu/")
    parser.add_argument("--window-min", default=81.2, type=float,
                        help="Lower edge of Z mass window in GeV [default: 81.2]")
    parser.add_argument("--window-max", default=101.2, type=float,
                        help="Upper edge of Z mass window in GeV [default: 101.2]")
    parser.add_argument("--nbins-init", default=15, type=int,
                        help="Initial number of uniform bins before adaptive merging [default: 15]")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()


# =============================================================================
# Histogram helpers (CR-specific, no PN scoring, single mass branch)
# =============================================================================

def _make_hist(file_path, tree_name, bin_edges, mass_min, mass_max, hist_name,
               use_weight=True):
    """Fill a TH1 from a TTree filtered to the mass window. Returns detached TH1."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample file not found: {file_path}")

    rfile = ROOT.TFile.Open(file_path, "READ")
    if not rfile or rfile.IsZombie():
        raise IOError(f"Failed to open: {file_path}")
    if not rfile.GetListOfKeys().Contains(tree_name):
        rfile.Close()
        raise RuntimeError(f"Tree '{tree_name}' not found in {file_path}")
    rfile.Close()

    nbins = len(bin_edges) - 1
    bin_edges_vec = ROOT.std.vector["double"](bin_edges)

    rdf = ROOT.RDataFrame(tree_name, file_path)
    rdf = rdf.Filter(f"mass >= {mass_min} && mass <= {mass_max}")

    if use_weight:
        hist = rdf.Histo1D((hist_name, "", nbins, bin_edges_vec.data()), "mass", "weight")
    else:
        hist = rdf.Histo1D((hist_name, "", nbins, bin_edges_vec.data()), "mass")

    out = hist.GetValue()
    out.SetDirectory(0)
    return out


def _get_central(basedir, process, bin_edges, mass_min, mass_max):
    return _make_hist(f"{basedir}/{process}.root", "Central",
                      bin_edges, mass_min, mass_max, process, use_weight=True)


def _get_data(basedir, bin_edges, mass_min, mass_max):
    return _make_hist(f"{basedir}/data.root", "Central",
                      bin_edges, mass_min, mass_max, "data_obs", use_weight=False)


def _get_syst(basedir, process, bin_edges, mass_min, mass_max, tree_name):
    """Read a systematic-variation tree; returns None if the tree is absent."""
    file_path = f"{basedir}/{process}.root"
    if not os.path.exists(file_path):
        return None
    suffix = combine_suffix_from_tree(tree_name)
    try:
        return _make_hist(file_path, tree_name, bin_edges, mass_min, mass_max,
                          f"{process}_{suffix}", use_weight=True)
    except (FileNotFoundError, RuntimeError):
        return None


def _merged_central(basedir, processes, bin_edges, mass_min, mass_max, name):
    """Sum Central trees from multiple sample files into one TH1."""
    total = None
    for proc in processes:
        try:
            h = _make_hist(f"{basedir}/{proc}.root", "Central",
                           bin_edges, mass_min, mass_max, f"{name}_{proc}", use_weight=True)
        except (FileNotFoundError, RuntimeError) as e:
            logging.warning(f"  Skipping {proc} from merged '{name}': {e}")
            continue
        if total is None:
            total = h.Clone(name)
            total.SetDirectory(0)
        else:
            total.Add(h)
    if total is None:
        # Empty merged hist using the bin layout
        nbins = len(bin_edges) - 1
        bin_edges_vec = ROOT.std.vector["double"](bin_edges)
        total = ROOT.TH1D(name, name, nbins, bin_edges_vec.data())
        total.SetDirectory(0)
    return total


# =============================================================================
# Configuration loading
# =============================================================================

def load_systematics(workdir, era):
    """Load SR1E2Mu systematics block — shared with TTZ2E1Mu by design."""
    config_path = f"{workdir}/SignalRegionStudyV2/configs/systematics.{era}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Systematics config not found: {config_path}")
    with open(config_path) as f:
        cfg = json.load(f)
    if SR_CONFIG_CHANNEL not in cfg:
        raise ValueError(f"Channel '{SR_CONFIG_CHANNEL}' not found in {config_path}")
    return cfg[SR_CONFIG_CHANNEL]


# Background categories present in samples/{era}/TTZ2E1Mu/{masspoint}/
# Order matters for the datacard: nonprompt first, then prompt MC, then 'others'.
SEPARATE_BACKGROUNDS = ["nonprompt", "WZ", "ZZ", "ttW", "ttZ", "ttH", "tZq", "conversion"]
OTHERS_PROCESSES = ["others"]   # sample file is samples/.../others.root


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format="%(levelname)s - %(message)s")

    workdir = os.getenv("WORKDIR")
    if not workdir:
        raise EnvironmentError("WORKDIR not set. Run 'source setup.sh' first.")

    basedir = (f"{workdir}/SignalRegionStudyV2/samples/{args.era}/{CHANNEL}/{args.masspoint}")
    outdir = (f"{workdir}/SignalRegionStudyV2/templates/{args.era}/{CHANNEL}/"
              f"{args.masspoint}/{METHOD}/{BINNING_TAG}")

    if not os.path.isdir(basedir):
        raise FileNotFoundError(f"Sample directory not found: {basedir}")

    logging.info("=" * 60)
    logging.info("TTZ2E1Mu CR Template Generation")
    logging.info("=" * 60)
    logging.info(f"  Era: {args.era}")
    logging.info(f"  Masspoint (reference): {args.masspoint}")
    logging.info(f"  Z window: [{args.window_min:.1f}, {args.window_max:.1f}] GeV")
    logging.info(f"  Initial bins: {args.nbins_init} uniform")
    logging.info(f"  Sample dir: {basedir}")
    logging.info(f"  Output dir: {outdir}")

    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    syst_config = load_systematics(workdir, args.era)
    syst_categories = categorize_systematics(syst_config)
    logging.info(f"Loaded {len(syst_config)} systematics from SR1E2Mu block")
    logging.info(f"  preprocessed_shape: {len(syst_categories['preprocessed_shape'])}")
    logging.info(f"  valued_shape:       {len(syst_categories['valued_shape'])}")
    logging.info(f"  multi_variation:    {len(syst_categories['multi_variation'])}")
    logging.info(f"  valued_lnN:         {len(syst_categories['valued_lnN'])}")

    # =========================================================================
    # Stage 0: pick initial bin count by adaptive n_eff scan (mirrors SR's
    # standard extended loop over [15, 13, 11, 9, 7, 5]).
    # Largest n_init that yields n_eff >= AUTOMC_THRESHOLD in every bin wins;
    # if none pass, keep the smallest tried and rely on autoMCStats.
    # =========================================================================
    candidate_nbins = [n for n in [args.nbins_init, 13, 11, 9, 7, 5] if n <= args.nbins_init]
    bin_edges = np.linspace(args.window_min, args.window_max, candidate_nbins[0] + 1)
    nbins_final = candidate_nbins[0]
    apply_floor = False
    for n_init in candidate_nbins:
        candidate_edges = np.linspace(args.window_min, args.window_max, n_init + 1)
        test_hists = {}
        for proc in SEPARATE_BACKGROUNDS:
            try:
                test_hists[proc] = _get_central(basedir, proc, candidate_edges,
                                                args.window_min, args.window_max)
            except (FileNotFoundError, RuntimeError):
                pass
        try:
            test_hists["others"] = _merged_central(basedir, OTHERS_PROCESSES, candidate_edges,
                                                   args.window_min, args.window_max, "others")
        except (FileNotFoundError, RuntimeError):
            pass
        ok, diagnostics = check_binning_quality(test_hists)
        logging.info(f"  Trying {n_init} bins: {'PASS' if ok else 'FAIL ('+str(len(diagnostics))+' bins below threshold)'}")
        if ok:
            bin_edges = candidate_edges
            nbins_final = n_init
            break
    else:
        bin_edges = candidate_edges
        nbins_final = candidate_nbins[-1]
        apply_floor = True
        logging.warning(f"All candidate bin counts failed n_eff check; keeping {nbins_final} bins with floor")
    logging.info(f"Final bin count: {nbins_final}")
    logging.info(f"Final bin edges: {[round(float(e), 3) for e in bin_edges]}")

    # =========================================================================
    # Stage 1: build templates dict (process -> {'nominal'|'<syst>Up'|'<syst>Down': TH1})
    # =========================================================================
    templates = {}

    for process in SEPARATE_BACKGROUNDS:
        logging.info(f"Building {process}")
        try:
            central = _get_central(basedir, process, bin_edges, args.window_min, args.window_max)
        except (FileNotFoundError, RuntimeError) as e:
            logging.warning(f"  Skipping {process}: {e}")
            continue
        ensure_positive_integral(central, floor_mode="zero")
        cap_stat_errors(central)
        proc_map = {"nominal": central}
        templates[process] = proc_map
        logging.info(f"  Central integral: {central.Integral():.4f}")

        # Preprocessed shape systematics — read per-syst trees written by preprocess.py
        for syst_name, variations, group in syst_categories["preprocessed_shape"]:
            if process not in group:
                continue
            # Tree names in samples/.../{process}.root follow preprocess.py:
            # legacy form "PileupReweight_Up" → tree "<syst_name>_Up"
            # dict form {"Up": "Scale_4"} → tree "<syst_name>_Up" (preprocess renames)
            if isinstance(variations, dict):
                read_trees = [f"{syst_name}_{d}" for d in ("Up", "Down") if d in variations]
            else:
                read_trees = [get_output_tree_name(syst_name, v) for v in variations]
            for read_tree in read_trees:
                hist = _get_syst(basedir, process, bin_edges, args.window_min, args.window_max, read_tree)
                if hist is None:
                    logging.debug(f"  Missing variation {process}/{read_tree}, skipping")
                    continue
                ensure_positive_integral(hist, floor_mode="zero")
                cap_stat_errors(hist)
                proc_map[combine_suffix_from_tree(read_tree)] = hist

        # Valued shape systematics — scale Central by (1 +/- value)
        for syst_name, value, group in syst_categories["valued_shape"]:
            if process not in group:
                continue
            for direction in ("up", "down"):
                hist = create_scaled_hist(central, process, syst_name, value, direction)
                ensure_positive_integral(hist, floor_mode="zero")
                cap_stat_errors(hist)
                suffix = f"{syst_name}Up" if direction == "up" else f"{syst_name}Down"
                proc_map[suffix] = hist

    # 'others' merged background
    logging.info("Building others (merged)")
    h_others = _merged_central(basedir, OTHERS_PROCESSES, bin_edges,
                                args.window_min, args.window_max, "others")
    ensure_positive_integral(h_others, floor_mode="floor")
    cap_stat_errors(h_others)
    others_map = {"nominal": h_others}
    templates["others"] = others_map
    logging.info(f"  Central integral: {h_others.Integral():.4f}")

    # Others systematics — build merged variation by summing per-process variation hists
    for syst_name, variations, group in syst_categories["preprocessed_shape"]:
        if "others" not in group:
            continue
        if isinstance(variations, dict):
            read_trees = [f"{syst_name}_{d}" for d in ("Up", "Down") if d in variations]
        else:
            read_trees = [get_output_tree_name(syst_name, v) for v in variations]
        for read_tree in read_trees:
            merged = None
            for proc in OTHERS_PROCESSES:
                h = _get_syst(basedir, proc, bin_edges, args.window_min, args.window_max, read_tree)
                if h is None:
                    continue
                if merged is None:
                    merged = h.Clone(f"others_{combine_suffix_from_tree(read_tree)}")
                    merged.SetDirectory(0)
                else:
                    merged.Add(h)
            if merged is None:
                continue
            ensure_positive_integral(merged, floor_mode="floor")
            cap_stat_errors(merged)
            others_map[combine_suffix_from_tree(read_tree)] = merged

    # Others valued-shape (scale h_others)
    for syst_name, value, group in syst_categories["valued_shape"]:
        if "others" not in group:
            continue
        for direction in ("up", "down"):
            hist = create_scaled_hist(h_others, "others", syst_name, value, direction)
            ensure_positive_integral(hist, floor_mode="floor")
            cap_stat_errors(hist)
            suffix = f"{syst_name}Up" if direction == "up" else f"{syst_name}Down"
            others_map[suffix] = hist

    # Dummy signal placeholder — flat 1e-6 per bin so r is defined, contributes nothing at r=0
    nbins = len(bin_edges) - 1
    bin_edges_vec = ROOT.std.vector["double"](bin_edges)
    sig_hist = ROOT.TH1D("signal", "signal placeholder", nbins, bin_edges_vec.data())
    sig_hist.SetDirectory(0)
    for ib in range(1, nbins + 1):
        sig_hist.SetBinContent(ib, BIN_FLOOR_VALUE)
        sig_hist.SetBinError(ib, BIN_FLOOR_VALUE)
    templates["signal"] = {"nominal": sig_hist}

    # data_obs — real data (CR is unblinded by definition)
    data_obs = _get_data(basedir, bin_edges, args.window_min, args.window_max)
    templates["data_obs"] = data_obs
    logging.info(f"data_obs integral: {data_obs.Integral():.4f}")

    # =========================================================================
    # Stage 2: adaptive merge loop (n_eff>=5 + syst-driven)
    # =========================================================================
    bkg_processes_for_merge = [p for p in SEPARATE_BACKGROUNDS if p in templates] + ["others"]

    # First check: does the initial binning already pass autoMCStats?
    init_bkg_hists = {p: templates[p]["nominal"] for p in bkg_processes_for_merge if p in templates}
    ok, diagnostics = check_binning_quality(init_bkg_hists)
    if ok:
        logging.info("Initial binning already satisfies n_eff >= 5 in all bins")
    else:
        logging.info(f"Initial binning has {len(diagnostics)} bins below n_eff threshold:")
        for d in diagnostics[:5]:
            logging.info(f"  {d}")

    # Apply the SR's syst-driven merge loop (operates on snapshot, rebins TH1s once)
    pre_merge_nbins = len(bin_edges) - 1
    bin_edges, templates, n_merges = apply_syst_driven_merging(
        bin_edges, templates, bkg_processes_for_merge,
        max_rel_syst=SYST_MERGE_THRESHOLD, logger=logging)
    post_merge_nbins = len(bin_edges) - 1
    if n_merges > 0:
        logging.info(f"Syst-merge: {pre_merge_nbins} -> {post_merge_nbins} bins ({n_merges} merges)")

    # Refresh references after potential rebin
    data_obs = templates["data_obs"]
    sig_hist = templates["signal"]["nominal"]

    # =========================================================================
    # Stage 3: write outputs
    # =========================================================================
    save_json({
        "nbins": len(bin_edges) - 1,
        "bin_edges": [float(e) for e in bin_edges],
        "method": "CR_Z_window_adaptive",
        "mass_min": float(args.window_min),       # naming aligned with SR binning.json
        "mass_max": float(args.window_max),       # so plotPostfitMass.py can read both
        "window_min": float(args.window_min),
        "window_max": float(args.window_max),
        "nbins_init": int(args.nbins_init),
        "syst_merge_applied": n_merges > 0,
        "n_bins_merged": int(n_merges),
        "syst_merge_threshold": SYST_MERGE_THRESHOLD,
    }, f"{outdir}/binning.json")

    # process_list.json: same schema as SR — "signal" first as the dummy placeholder
    separate_processes = [p for p in SEPARATE_BACKGROUNDS if p in templates]
    save_json({
        "separate_processes": separate_processes,
        "merged_to_others": [],
        "dropped_missing": [p for p in SEPARATE_BACKGROUNDS if p not in templates],
        "description": "TTZ2E1Mu CR; signal is a dummy placeholder (1e-6/bin) for r-parameter compatibility.",
        "signal_is_dummy": True,
    }, f"{outdir}/process_list.json")

    # Compute relative stat errors for lowstat.json (consumed by printCRDatacard.py)
    SHAPE_REL_ERR_THRESHOLD = 0.30
    lowstat_processes = []
    for proc in separate_processes + ["others"]:
        h = templates[proc]["nominal"]
        integral = h.Integral()
        if integral <= 0:
            continue
        sum_err2 = sum(h.GetBinError(i) ** 2 for i in range(1, h.GetNbinsX() + 1))
        rel_err = (sum_err2 ** 0.5) / integral
        logging.info(f"  {proc:>12s}: integral={integral:.4f}, rel_err={rel_err*100:.1f}%")
        if rel_err > SHAPE_REL_ERR_THRESHOLD:
            lowstat_processes.append(proc)

    save_json({
        "threshold": SHAPE_REL_ERR_THRESHOLD,
        "processes": lowstat_processes,
        "fallbacks": {},   # printCRDatacard.py fills these
    }, f"{outdir}/lowstat.json")

    # shapes.root
    out_path = f"{outdir}/shapes.root"
    out_file = ROOT.TFile.Open(out_path, "RECREATE")
    out_file.cd()
    data_obs.Write("data_obs")
    sig_hist.Write("signal")
    # Signal has no per-syst variations (placeholder); skip writing them
    process_order = separate_processes + ["others"]
    for proc in process_order:
        proc_map = templates.get(proc, {})
        nominal = proc_map.get("nominal")
        if nominal is None:
            continue
        nominal.Write(proc)
        for key, hist in proc_map.items():
            if key == "nominal":
                continue
            hist.Write(f"{proc}_{key}")
    out_file.Close()

    # Summary
    logging.info("=" * 60)
    logging.info("CR template generation complete")
    logging.info(f"Output: {out_path}")
    logging.info(f"Final bins: {len(bin_edges)-1}")
    logging.info(f"data_obs:        {data_obs.Integral():>10.2f}")
    for proc in process_order:
        if proc in templates:
            logging.info(f"  {proc:>13s}: {templates[proc]['nominal'].Integral():>10.4f}")
    logging.info(f"  {'signal (dummy)':>13s}: {sig_hist.Integral():>10.4e}")
    if lowstat_processes:
        logging.info(f"Low-stat processes (rel_err > {SHAPE_REL_ERR_THRESHOLD*100:.0f}%): {lowstat_processes}")


if __name__ == "__main__":
    main()
