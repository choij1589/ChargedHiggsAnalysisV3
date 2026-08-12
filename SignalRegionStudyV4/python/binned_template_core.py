"""Signal-independent binned-template machinery.

Extracted verbatim from makeBinnedTemplates.py (2026-08-13) so the
direct-MC and parametric-signal (interp) template producers share one
implementation of everything that does not depend on how the signal is
made: tree readers, background validation, the adaptive binning scan,
background/data template construction and the sidecar JSON writers.
makeBinnedTemplates re-exports every public name, so existing importers
(measInterpShapeDeltas, the condor wrappers) are unaffected.

Behavior contract: the direct-MC template chain through this module is
BIT-IDENTICAL to the pre-refactor makeBinnedTemplates.py.
"""
import logging
import os
from collections import OrderedDict

import ROOT
import numpy as np

from template_utils import (
    save_json,
    ensure_positive_integral, cap_stat_errors, build_particlenet_score,
    create_scaled_hist,
    calculate_adaptive_bins, check_binning_quality,
    iter_shape_directions,
)
from run_period_utils import PHYSICS_PROCESS_ORDER, component_name
import srspaths

# Bins with total-bkg syst envelope / nominal above this threshold are merged
# into a neighbour. Addresses a stat-review concern about sudden-dip bins with
# very large relative systematic uncertainties underestimating backgrounds.
SYST_MERGE_THRESHOLD = 2.0

MIN_CORE_BINS = 5


def getHist(basedir, process, bin_edges, mass_min, mass_max, syst="Central",
            threshold=-999., upper_threshold=None, bg_weights=None, masspoint=None):
    """Create histogram from preprocessed tree using RDataFrame."""
    file_path = f"{basedir}/{process}.root"
    tree_name = syst

    # Combine expects no underscore before Up/Down
    syst_formatted = syst.replace("_Up", "Up").replace("_Down", "Down")
    hist_name = process if syst == "Central" else f"{process}_{syst_formatted}"

    logging.debug(f"Creating histogram: {hist_name} from tree {tree_name}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample file not found: {file_path}")

    nbins = len(bin_edges) - 1

    # Create ROOT vector for histogram binning
    bin_edges_vector = ROOT.std.vector['double'](bin_edges)

    # Check if tree exists and get branch list
    test_file = ROOT.TFile.Open(file_path)
    tree = test_file.Get(tree_name)
    if not tree:
        test_file.Close()
        raise RuntimeError(f"Tree '{tree_name}' not found in {file_path}")

    branches = [b.GetName() for b in tree.GetListOfBranches()]
    test_file.Close()

    # Create RDataFrame
    rdf = ROOT.RDataFrame(tree_name, file_path)

    # Apply mass window cut
    rdf = rdf.Filter(f"mass >= {mass_min} && mass <= {mass_max}")
    logging.debug(f"  Applied mass window cut: [{mass_min:.2f}, {mass_max:.2f}] GeV")

    # Apply ParticleNet score cut if threshold is provided
    if (threshold > -999. or upper_threshold is not None) and masspoint:
        score_sig = f"score_{masspoint}_signal"
        if score_sig in branches:
            score_formula = build_particlenet_score(masspoint, bg_weights)
            rdf = rdf.Define("score_PN", score_formula)
            if upper_threshold is not None:
                rdf = rdf.Filter(f"score_PN < {upper_threshold}")
                logging.debug(f"  Applied ParticleNet cut: score_PN < {upper_threshold:.3f}")
            elif threshold > -999.:
                rdf = rdf.Filter(f"score_PN >= {threshold}")
                logging.debug(f"  Applied ParticleNet cut: score_PN >= {threshold:.3f}")
        else:
            raise RuntimeError(
                f"ParticleNet score branches not found in {file_path}/{tree_name}\n"
                f"  Expected branch: {score_sig}"
            )

    # Fill histogram
    hist = rdf.Histo1D((hist_name, "", nbins, bin_edges_vector.data()), "mass", "weight")

    hist_result = hist.GetValue()
    hist_result.SetDirectory(0)

    logging.debug(f"Histogram {hist_name}: {hist_result.GetEntries()} entries, integral = {hist_result.Integral():.4f}")

    return hist_result


def createEnvelopeHists(basedir, process, bin_edges, mass_min, mass_max,
                        variations, syst_name, threshold=-999., upper_threshold=None,
                        bg_weights=None, masspoint=None):
    """Create up/down envelope histograms from multiple variations."""
    logging.debug(f"Creating envelope for {process}_{syst_name} from {len(variations)} variations")

    # Collect all variation histograms
    variation_hists = []
    for var in variations:
        try:
            hist = getHist(basedir, process, bin_edges, mass_min, mass_max,
                          var, threshold, upper_threshold, bg_weights, masspoint)
            variation_hists.append(hist)
        except (FileNotFoundError, RuntimeError) as e:
            logging.warning(f"  Skipping variation {var}: {e}")

    if not variation_hists:
        raise RuntimeError(f"No variation histograms found for {process}_{syst_name}")

    # Get central histogram for reference shape
    central_hist = getHist(basedir, process, bin_edges, mass_min, mass_max,
                           "Central", threshold, upper_threshold, bg_weights, masspoint)

    nbins = central_hist.GetNbinsX()

    # Create envelope histograms (same naming convention: {process}_{syst_name}Up/Down)
    hist_up = central_hist.Clone(f"{process}_{syst_name}Up")
    hist_down = central_hist.Clone(f"{process}_{syst_name}Down")
    hist_up.SetDirectory(0)
    hist_down.SetDirectory(0)

    # Calculate bin-by-bin envelope
    for i in range(1, nbins + 1):
        bin_values = [h.GetBinContent(i) for h in variation_hists]

        if bin_values:
            bin_max = max(bin_values)
            bin_min = min(bin_values)
            # Propagate errors as RMS of variations
            bin_err = np.std(bin_values) if len(bin_values) > 1 else 0.0

            hist_up.SetBinContent(i, bin_max)
            hist_up.SetBinError(i, bin_err)
            hist_down.SetBinContent(i, bin_min)
            hist_down.SetBinError(i, bin_err)

    logging.debug(f"  Envelope {syst_name}: up_integral={hist_up.Integral():.4f}, down_integral={hist_down.Integral():.4f}")

    return hist_up, hist_down


def getDataHist(basedir, bin_edges, mass_min, mass_max,
                threshold=-999., upper_threshold=None, bg_weights=None, masspoint=None):
    """Create data histogram from data.root file."""
    file_path = f"{basedir}/data.root"
    hist_name = "data_obs"

    logging.debug(f"Creating data histogram from {file_path}")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")

    nbins = len(bin_edges) - 1
    bin_edges_vector = ROOT.std.vector['double'](bin_edges)

    # Check if tree exists and get branch list
    test_file = ROOT.TFile.Open(file_path)
    tree = test_file.Get("Central")
    if not tree:
        test_file.Close()
        raise RuntimeError(f"Tree 'Central' not found in {file_path}")

    branches = [b.GetName() for b in tree.GetListOfBranches()]
    test_file.Close()

    # Create RDataFrame
    rdf = ROOT.RDataFrame("Central", file_path)

    # Apply mass window cut
    rdf = rdf.Filter(f"mass >= {mass_min} && mass <= {mass_max}")
    logging.debug(f"  Applied mass window cut: [{mass_min:.2f}, {mass_max:.2f}] GeV")

    # Apply ParticleNet score cut if threshold or upper_threshold is provided
    if (threshold > -999. or upper_threshold is not None) and masspoint:
        score_sig = f"score_{masspoint}_signal"
        if score_sig in branches:
            score_formula = build_particlenet_score(masspoint, bg_weights)
            rdf = rdf.Define("score_PN", score_formula)
            if upper_threshold is not None:
                rdf = rdf.Filter(f"score_PN < {upper_threshold}")
                logging.debug(f"  Applied ParticleNet cut: score_PN < {upper_threshold:.3f}")
            elif threshold > -999.:
                rdf = rdf.Filter(f"score_PN >= {threshold}")
                logging.debug(f"  Applied ParticleNet cut: score_PN >= {threshold:.3f}")
        else:
            raise RuntimeError(
                f"ParticleNet score branches not found in {file_path}/Central\n"
                f"  Expected branch: {score_sig}"
            )

    # Fill histogram (data uses weight=1)
    hist = rdf.Histo1D((hist_name, "", nbins, bin_edges_vector.data()), "mass")

    hist_result = hist.GetValue()
    hist_result.SetDirectory(0)

    logging.debug(f"Data histogram: {hist_result.GetEntries()} entries, integral = {hist_result.Integral():.4f}")

    return hist_result




def validateBackgroundStatistics(basedir, bin_edges, mass_min, mass_max,
                                  background_categories, masspoint, threshold=-999.,
                                  upper_threshold=None, bg_weights=None, min_total_events=1):
    """Validate statistical quality of each background process.

    Returns a dict of {process: {total_events, status, reason}} where status is
    one of: 'present', 'low_stat', 'empty', 'missing_file'.

    All processes except missing-file cases are kept as separate columns
    regardless of yield.
    """
    logging.info("Validating background statistics...")
    nbins = len(bin_edges) - 1
    results = {}

    for category in background_categories:
        # Map category name to output file name
        process = "conversion" if category == "conv" else category
        logging.info(f"  Validating {process}...")

        file_path = f"{basedir}/{process}.root"
        if not os.path.exists(file_path):
            logging.error(f"    {process}: sample file not found — will be dropped from this era/channel")
            results[process] = {
                "total_events": 0,
                "status": "missing_file",
                "reason": "file not found"
            }
            continue

        try:
            rdf = ROOT.RDataFrame("Central", file_path)
            rdf = rdf.Filter(f"mass >= {mass_min} && mass <= {mass_max}")

            if threshold > -999. or upper_threshold is not None:
                test_file = ROOT.TFile.Open(file_path, "READ")
                tree = test_file.Get("Central")
                if tree:
                    branches = [b.GetName() for b in tree.GetListOfBranches()]
                    test_file.Close()
                    score_sig = f"score_{masspoint}_signal"
                    if score_sig in branches:
                        score_formula = build_particlenet_score(masspoint, bg_weights)
                        rdf = rdf.Define("score_PN", score_formula)
                        if upper_threshold is not None:
                            rdf = rdf.Filter(f"score_PN < {upper_threshold}")
                        elif threshold > -999.:
                            rdf = rdf.Filter(f"score_PN >= {threshold}")
                else:
                    test_file.Close()

            bin_edges_vector = ROOT.std.vector['double'](bin_edges)
            hist = rdf.Histo1D(("temp", "", nbins, bin_edges_vector.data()), "mass", "weight")
            total_events = hist.GetValue().Integral()

        except Exception as e:
            logging.warning(f"    Error processing {process}: {e}")
            total_events = 0

        if total_events <= 0:
            status = "empty"
            reason = f"total events ({total_events:.2f}) <= 0"
        elif total_events < min_total_events:
            status = "low_stat"
            reason = f"total events ({total_events:.2f}) < {min_total_events} (low statistics)"
            logging.warning(f"    {process}: {reason} — kept as separate column; "
                            f"low-stat handling (floor + shape→lnN fallback) will apply")
        else:
            status = "present"
            reason = "passes statistical requirements"

        results[process] = {
            "total_events": total_events,
            "status": status,
            "reason": reason
        }

        logging.info(f"    Total events: {total_events:.2f}, status: {status}")

    return results




def load_systematics_block(era, channel):
    config = srspaths.systematics_config(era)
    if channel not in config:
        raise ValueError(f"Channel '{channel}' not found in systematics.{era}.json")
    return config[channel]


def load_sample_block(era, channel):
    samplegroups = srspaths.samplegroups_config()
    if era not in samplegroups:
        raise ValueError(f"Era '{era}' not found in samplegroups.json")
    if channel not in samplegroups[era]:
        raise ValueError(f"Channel '{channel}' not found for era '{era}'")
    return samplegroups[era][channel]




def category_background_processes(suberas, channel):
    """Return stable background process names and per-subera validation metadata."""
    processes = []
    by_subera = {}
    for subera in suberas:
        samples = load_sample_block(subera, channel)
        reserved = {"data"}
        subera_processes = []
        for key in PHYSICS_PROCESS_ORDER:
            if key == "signal" or key in reserved:
                continue
            if key in samples:
                subera_processes.append(key)
                if key not in processes:
                    processes.append(key)
        by_subera[subera] = subera_processes
    return processes, by_subera


def hist_exists(path, tree="Central"):
    if not os.path.exists(path):
        return False
    f = ROOT.TFile.Open(path, "READ")
    ok = bool(f and not f.IsZombie() and f.Get(tree))
    if f:
        f.Close()
    return ok


def build_component_shape_templates(basedir, base_process, component, bin_edges,
                                    mass_min, mass_max, syst_categories,
                                    threshold, upper_threshold, bg_weights,
                                    masspoint, floor_mode):
    proc_map = {}
    central = getHist(
        basedir, base_process, bin_edges, mass_min, mass_max,
        "Central", threshold, upper_threshold, bg_weights, masspoint
    )
    central.SetName(component)
    ensure_positive_integral(central, floor_mode=floor_mode)
    cap_stat_errors(central)
    proc_map["nominal"] = central

    if base_process == "nonprompt":
        for syst_name, value, group in syst_categories["valued_shape"]:
            if "nonprompt" not in group:
                continue
            for direction in ["up", "down"]:
                hist = create_scaled_hist(central, component, syst_name, value, direction)
                ensure_positive_integral(hist, floor_mode=floor_mode)
                cap_stat_errors(hist)
                suffix = f"{syst_name}Up" if direction == "up" else f"{syst_name}Down"
                proc_map[suffix] = hist
        return proc_map

    for syst_name, variations, group in syst_categories["preprocessed_shape"]:
        if base_process not in group:
            continue
        for direction in iter_shape_directions(variations):
            read_tree = f"{syst_name}_{direction}"
            combine_suffix = f"{syst_name}{direction}"
            try:
                hist = getHist(
                    basedir, base_process, bin_edges, mass_min, mass_max,
                    read_tree, threshold, upper_threshold, bg_weights, masspoint
                )
                hist.SetName(f"{component}_{combine_suffix}")
                ensure_positive_integral(hist, floor_mode=floor_mode)
                cap_stat_errors(hist)
                proc_map[combine_suffix] = hist
            except (FileNotFoundError, RuntimeError) as exc:
                logging.warning("    Skipping %s/%s/%s: %s", component, syst_name, direction, exc)

    for syst_name, value, group in syst_categories["valued_shape"]:
        if base_process not in group:
            continue
        for direction in ["up", "down"]:
            hist = create_scaled_hist(central, component, syst_name, value, direction)
            ensure_positive_integral(hist, floor_mode=floor_mode)
            cap_stat_errors(hist)
            suffix = f"{syst_name}Up" if direction == "up" else f"{syst_name}Down"
            proc_map[suffix] = hist

    return proc_map




def write_run_period_shapes(outdir, categories):
    output_file = ROOT.TFile(f"{outdir}/shapes.root", "RECREATE")
    for cat_name, cat_payload in categories.items():
        cat_dir = output_file.mkdir(cat_name)
        cat_dir.cd()
        cat_payload["templates"]["data_obs"].Write("data_obs")
        for process in cat_payload["process_order"]:
            proc_map = cat_payload["templates"].get(process, {})
            nominal = proc_map.get("nominal")
            if nominal is None:
                continue
            nominal.Write(process)
            for key, hist in proc_map.items():
                if key == "nominal":
                    continue
                hist.Write(f"{process}_{key}")
        output_file.cd()
    output_file.Close()


# =============================================================================
# Driver-level helpers extracted from build_run_period_templates
# =============================================================================

def validate_category_backgrounds(bg_by_subera, basedir_of, probe_edges,
                                  mass_min, mass_max, masspoint,
                                  threshold=-999., upper_threshold=None,
                                  bg_weights=None):
    """Per-subera background availability for one category.

    Returns (validation payload, active_by_subera). nonprompt/others are
    checked by file presence; validateBackgroundStatistics only sees the
    MC categories."""
    validation_by_subera = OrderedDict()
    active_by_subera = OrderedDict()
    for subera, processes in bg_by_subera.items():
        basedir = basedir_of(subera)
        validation = validateBackgroundStatistics(
            basedir, probe_edges, mass_min, mass_max,
            [p if p != "conversion" else "conv" for p in processes
             if p not in {"nonprompt", "others"}],
            masspoint, threshold, upper_threshold, bg_weights,
            min_total_events=1
        )
        for proc in ["nonprompt", "others"]:
            if proc not in processes:
                continue
            file_path = f"{basedir}/{proc}.root"
            status = "present" if hist_exists(file_path) else "missing_file"
            validation[proc] = {
                "total_events": 0.0,
                "status": status,
                "reason": "checked by file presence for component construction",
            }
        validation_by_subera[subera] = validation
        active_by_subera[subera] = [
            proc for proc in processes
            if validation.get(proc, {}).get("status") != "missing_file"
        ]
    return validation_by_subera, active_by_subera


def scan_binning(cat, active_by_subera, basedir_of, x0, sigma_eff,
                 mass_min, mass_max, masspoint,
                 threshold=-999., upper_threshold=None, bg_weights=None):
    """15 -> MIN_CORE_BINS adaptive scan on background-only test hists.

    Returns (bin_edges, n_core_final, apply_floor)."""
    core_bin_candidates = list(range(15, MIN_CORE_BINS - 1, -1))

    apply_floor = False
    n_core_final = core_bin_candidates[-1]
    bin_edges = calculate_adaptive_bins(x0, sigma_eff, n_core_final)
    for n_core in core_bin_candidates:
        candidate_edges = calculate_adaptive_bins(x0, sigma_eff, n_core)
        logging.info("Testing %s with %d core bins", cat, n_core)
        test_hists = {}
        for subera, processes in active_by_subera.items():
            basedir = basedir_of(subera)
            for proc in processes:
                try:
                    comp = component_name(proc, subera)
                    h = getHist(
                        basedir, proc, candidate_edges, mass_min, mass_max,
                        "Central", threshold, upper_threshold, bg_weights,
                        masspoint
                    )
                    test_hists[comp] = h
                except (FileNotFoundError, RuntimeError):
                    continue
        ok, diagnostics = check_binning_quality(test_hists)
        if ok:
            bin_edges = candidate_edges
            n_core_final = n_core
            logging.info("  PASS: %d core bins accepted for %s", n_core, cat)
            break
        logging.info("  FAIL: %d issues", len(diagnostics))
        for diag in diagnostics[:5]:
            logging.info("    %s", diag)
    else:
        apply_floor = True
        logging.warning("All candidates failed for %s. Keeping %d core bins "
                        "with floor handling.", cat, n_core_final)
    return bin_edges, n_core_final, apply_floor


def sum_data_obs(suberas, basedir_of, bin_edges, mass_min, mass_max,
                 threshold=-999., upper_threshold=None, bg_weights=None,
                 masspoint=None):
    """Unblind data_obs: per-subera data hists summed."""
    data_obs = None
    for subera in suberas:
        h_data = getDataHist(
            basedir_of(subera), bin_edges, mass_min, mass_max,
            threshold, upper_threshold, bg_weights, masspoint
        )
        if data_obs is None:
            data_obs = h_data.Clone("data_obs")
            data_obs.SetDirectory(0)
        else:
            data_obs.Add(h_data)
    return data_obs


def apply_floor_fallback(templates):
    """Post-scan-failure floor/zero handling for every template."""
    for proc_name, proc_map in templates.items():
        if proc_name == "data_obs":
            continue
        base = proc_name.rsplit("_", 1)[0]
        mode = "floor" if base in {"signal", "others"} else "zero"
        for hist in proc_map.values():
            ensure_positive_integral(hist, floor_mode=mode)
            cap_stat_errors(hist)


def write_template_sidecars(outdir, categories_json, binning_json,
                            background_validation, process_components):
    """categories/process_list/binning/background_validation JSONs."""
    save_json({"construction": "run_period_components",
               "categories": categories_json},
              f"{outdir}/categories.json")
    save_json({
        "construction": "run_period_components",
        "separate_processes": [p["name"] for p in process_components
                               if not p["is_signal"]],
        "signal_processes": [p["name"] for p in process_components
                             if p["is_signal"]],
        "components": process_components,
        "physics_groups": {
            group: [
                p["name"] for p in process_components
                if p["physics_group"] == group
            ]
            for group in PHYSICS_PROCESS_ORDER
        },
        "merged_to_others": [],
        "description": "Run-period categories with subera component processes",
    }, f"{outdir}/process_list.json")
    save_json(binning_json, f"{outdir}/binning.json")
    save_json(background_validation, f"{outdir}/background_validation.json")
