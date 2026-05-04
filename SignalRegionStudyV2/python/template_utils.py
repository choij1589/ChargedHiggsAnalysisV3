"""Utility functions for template generation."""
import os
import re
import json
import shutil
import logging
from array import array
import ROOT
import numpy as np

BIN_FLOOR_VALUE = 1e-6
AUTOMC_THRESHOLD = 5  # Combine autoMCStats threshold for BB-lite vs per-process
SHAPE_REL_ERR_THRESHOLD = 0.30  # Backgrounds above this rel-err drop shape systs (S1 lnN fallback)


def save_json(data, path):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def ensure_directory(path, clean=False):
    """Ensure directory exists, optionally cleaning it first."""
    if clean and os.path.exists(path):
        logging.info(f"Removing existing directory {path}")
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def parse_variations(variation_spec):
    """Parse variation specification.

    Supports:
      - list form, e.g. ["PileupReweight_Up", "PileupReweight_Down"]
      - range-pattern list, e.g. ["Scale_{0..8}//5,7"]
      - dict form with explicit direction → tree mapping, e.g.
        {"Up": "Scale_4", "Down": "Scale_3"}
    """
    if isinstance(variation_spec, dict):
        return variation_spec
    if not isinstance(variation_spec, list):
        return []
    if len(variation_spec) == 1 and '{' in variation_spec[0]:
        return _expand_range_pattern(variation_spec[0])
    return variation_spec


def _expand_range_pattern(pattern):
    """Expand range pattern like 'Scale_{0..8}//5,7' or 'pdf_{00..99}'."""
    exclusions = set()
    if '//' in pattern:
        pattern, excl_str = pattern.split('//')
        exclusions = set(int(x) for x in excl_str.split(','))

    match = re.search(r'\{(\d+)\.\.(\d+)\}', pattern)
    if not match:
        return [pattern]

    start, end = int(match.group(1)), int(match.group(2))
    start_str = match.group(1)
    pad_width = len(start_str) if start_str.startswith('0') and len(start_str) > 1 else 0

    prefix, suffix = pattern[:match.start()], pattern[match.end():]
    return [
        f"{prefix}{str(i).zfill(pad_width) if pad_width else str(i)}{suffix}"
        for i in range(start, end + 1) if i not in exclusions
    ]


def get_output_tree_name(syst_name, variation):
    """Get output tree name from systematic name and variation."""
    if variation.endswith("_Up") or variation.endswith("Up"):
        return f"{syst_name}_Up"
    elif variation.endswith("_Down") or variation.endswith("Down"):
        return f"{syst_name}_Down"
    return variation


def combine_suffix_from_tree(tree_name):
    """Convert tree name like 'CMS_pileup_13p6TeV_Up' to Combine suffix 'CMS_pileup_13p6TeVUp'."""
    if tree_name.endswith("_Up"):
        return tree_name[:-3] + "Up"
    if tree_name.endswith("_Down"):
        return tree_name[:-5] + "Down"
    return tree_name


def iter_shape_variations(syst_name, variations):
    """Yield (input_tree, output_tree) pairs for preprocessed shape systematics.

    Handles both the legacy list form (where each variation name encodes its
    direction via an 'Up'/'Down' suffix) and the dict form
    {"Up": <tree>, "Down": <tree>} used for QCD scale variations whose
    underlying tree names (e.g. "Scale_4") carry no direction marker.
    """
    if isinstance(variations, dict):
        for direction in ("Up", "Down"):
            if direction in variations:
                yield f"Events_{variations[direction]}", f"{syst_name}_{direction}"
    else:
        for var in variations:
            yield f"Events_{var}", get_output_tree_name(syst_name, var)


def iter_shape_directions(variations):
    """Yield 'Up' / 'Down' for each available direction in a variations spec."""
    if isinstance(variations, dict):
        for direction in ("Up", "Down"):
            if direction in variations:
                yield direction
    else:
        for var in variations:
            if var.endswith("Up"):
                yield "Up"
            elif var.endswith("Down"):
                yield "Down"


def calculate_weight_scale(value, direction):
    """Calculate weight scale for valued+shape systematics."""
    if value >= 1.0:
        return value if direction == 'up' else 2.0 - value
    return 1.0 + value if direction == 'up' else 1.0 - value


def ensure_positive_integral(hist, floor_mode="floor"):
    """Handle non-positive bins in histograms.

    Args:
        hist: ROOT TH1 histogram.
        floor_mode: "floor" sets empty/negative bins to BIN_FLOOR_VALUE (1e-6).
                    "zero" sets them to exactly 0 (content and error).

    floor_mode="floor" (default, for signal and 'others'):
      Guarantees positive bin content so Combine's vertical morphing has no
      divide-by-zero, and the total background is never empty in any bin.

    floor_mode="zero" (for individual background processes):
      Sets empty/negative bins to content=0, error=0. Combine sees sigma_p=0
      for this process in this bin and skips it in autoMCStats — no phantom NP.
    """
    modified = False
    for i in range(1, hist.GetNbinsX() + 1):
        if hist.GetBinContent(i) <= 0:
            if hist.GetBinContent(i) < 0:
                logging.warning(
                    f"  {hist.GetName()}, bin {i}: negative content "
                    f"{hist.GetBinContent(i):.3e}, setting to "
                    f"{'floor' if floor_mode == 'floor' else 'zero'}"
                )
            if floor_mode == "zero":
                hist.SetBinContent(i, 0.0)
                hist.SetBinError(i, 0.0)
            else:
                hist.SetBinContent(i, BIN_FLOOR_VALUE)
                hist.SetBinError(i, BIN_FLOOR_VALUE)
            modified = True
    return modified


def cap_stat_errors(hist):
    """Cap per-bin stat error at 100% of the bin content.

    For any bin where err > content > 0, sets err = content (exactly 100%
    relative error). This prevents low-stat processes from contributing
    err > content spikes into the total-background stat-error sum used by
    check_binning_quality, and ensures shapes.root always has well-defined
    per-bin errors for autoMCStats / BB-lite nuisances.

    No-op on empty bins (content <= 0) — those are handled by
    ensure_positive_integral / apply_floor.
    """
    modified = False
    for i in range(1, hist.GetNbinsX() + 1):
        bc = hist.GetBinContent(i)
        be = hist.GetBinError(i)
        if bc > 0 and be > bc:
            hist.SetBinError(i, bc)
            modified = True
    return modified


def build_particlenet_score(masspoint, bg_weights=None):
    """Build ParticleNet score formula string."""
    score_sig = f"score_{masspoint}_signal"
    score_nonprompt = f"score_{masspoint}_nonprompt"
    score_diboson = f"score_{masspoint}_diboson"
    score_ttZ = f"score_{masspoint}_ttZ"

    if bg_weights:
        w1 = bg_weights.get("nonprompt", 1.0)
        w2 = bg_weights.get("diboson", 1.0)
        w3 = bg_weights.get("ttX", 1.0)
        return f"({score_sig}) / ({score_sig} + {w1}*{score_nonprompt} + {w2}*{score_diboson} + {w3}*{score_ttZ})"
    return f"({score_sig}) / ({score_sig} + {score_nonprompt} + {score_diboson} + {score_ttZ})"


def create_filtered_rdf(file_path, tree_name, mass_min, mass_max, threshold, bg_weights, masspoint):
    """Create RDataFrame with mass window and optional ParticleNet filtering."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Sample file not found: {file_path}")

    test_file = ROOT.TFile.Open(file_path)
    tree = test_file.Get(tree_name)
    if not tree:
        test_file.Close()
        raise RuntimeError(f"Tree '{tree_name}' not found in {file_path}")

    branches = [b.GetName() for b in tree.GetListOfBranches()]
    test_file.Close()

    rdf = ROOT.RDataFrame(tree_name, file_path)
    rdf = rdf.Filter(f"mass >= {mass_min} && mass <= {mass_max}")

    if threshold > -999. and masspoint:
        score_sig = f"score_{masspoint}_signal"
        if score_sig in branches:
            score_formula = build_particlenet_score(masspoint, bg_weights)
            rdf = rdf.Define("score_PN", score_formula)
            rdf = rdf.Filter(f"score_PN >= {threshold}")
        else:
            raise RuntimeError(
                f"ParticleNet score branches not found in {file_path}/{tree_name}\n"
                f"  Expected branch: {score_sig}"
            )

    return rdf, branches


def create_scaled_hist(central_hist, process, syst_name, value, direction):
    """Create a scaled histogram for valued+shape systematics."""
    scale = calculate_weight_scale(value, direction)
    suffix = "Up" if direction == "up" else "Down"
    hist_name = f"{process}_{syst_name}{suffix}"

    hist = central_hist.Clone(hist_name)
    hist.SetDirectory(0)
    hist.Scale(scale)

    logging.debug(f"  Created {hist_name}: scale={scale:.4f}")
    return hist


# =============================================================================
# Run3 Signal Systematic Name Remapping
# =============================================================================
# When Run3 signal is scaled from Run2 (2018), the preprocessed trees have
# Run2-style systematic names. These functions help map between Run3 config
# names and Run2 tree names.

RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]


def is_run3_era(era):
    """Check if the era is a Run3 era."""
    return era in RUN3_ERAS


# Scaled-from-Run2 Run3 signals carry Run2 (NanoAODv9) LHEScaleWeight indexing.
# The Run3 configs, however, reference Run3 (NanoAODv13) indices. This table maps
# the Run3 QCD-scale nuisance+direction back to the literal Run2 Scale_N tree
# present in the scaled file.
_QCDSCALE_RUN3_TO_RUN2_TREE = {
    ("QCDScale_muF_BSMsignal_13p6TeV", "Up"): "Scale_4",
    ("QCDScale_muF_BSMsignal_13p6TeV", "Down"): "Scale_3",
    ("QCDScale_muR_BSMsignal_13p6TeV", "Up"): "Scale_6",
    ("QCDScale_muR_BSMsignal_13p6TeV", "Down"): "Scale_1",
}


def get_run2_tree_name_for_run3_syst(syst_name, direction, era):
    """
    Get the Run2 tree name that corresponds to a Run3 systematic.

    For Run3 scaled signal, the preprocessed trees have Run2 names.
    This maps Run3 systematic names back to their Run2 equivalents.

    Args:
        syst_name: Run3 systematic name (e.g., 'CMS_res_j_2023BPix')
        direction: 'Up' or 'Down'
        era: Target era (e.g., '2023BPix')

    Returns:
        Run2 tree name (e.g., 'CMS_res_j_2018_Up'), or the raw Scale_N name
        for QCD-scale nuisances (Run2 indexing has no direction suffix).
    """
    qcd_tree = _QCDSCALE_RUN3_TO_RUN2_TREE.get((syst_name, direction))
    if qcd_tree is not None:
        return qcd_tree

    # Era-specific systematics: {name}_{era} → {name}_2018
    if syst_name.endswith(f'_{era}'):
        base = syst_name[:-len(f'_{era}')]
        return f"{base}_2018_{direction}"

    # Energy-specific: 13p6TeV → 13TeV
    if '13p6TeV' in syst_name:
        run2_name = syst_name.replace('13p6TeV', '13TeV')
        return f"{run2_name}_{direction}"

    # No remapping needed (correlated systematics)
    return f"{syst_name}_{direction}"


def is_signal_scaled_from_run2(signal_file_path, era):
    """
    Check if signal file contains Run2 systematic names (scaled signal).

    For Run3 eras, if the signal was scaled from Run2, it will have
    Run2-style tree names like 'CMS_pileup_13TeV_Up' instead of
    'CMS_pileup_13p6TeV_Up'.

    Args:
        signal_file_path: Path to signal ROOT file
        era: Target era

    Returns:
        True if signal appears to be scaled from Run2, False otherwise
    """
    if not is_run3_era(era):
        return False

    rfile = ROOT.TFile.Open(signal_file_path, "READ")
    if not rfile or rfile.IsZombie():
        return False

    keys = [k.GetName() for k in rfile.GetListOfKeys()]
    rfile.Close()

    # Check for Run2-style tree names
    run2_indicators = [
        'CMS_pileup_13TeV_Up',
        'CMS_pileup_13TeV_Down',
        'CMS_res_j_2018_Up',
        'CMS_res_j_2018_Down'
    ]

    return any(indicator in keys for indicator in run2_indicators)


def categorize_systematics(config):
    """
    Categorize systematics from config into processing groups.

    Returns dict with keys:
    - preprocessed_shape: list of (syst_name, [variations], group)
    - valued_shape: list of (syst_name, value, group)
    - multi_variation: list of (syst_name, [variations], group)
    - valued_lnN: list of (syst_name, value, group)
    """
    result = {'preprocessed_shape': [], 'valued_shape': [], 'multi_variation': [], 'valued_lnN': []}

    for syst_name, syst_config in config.items():
        source = syst_config.get('source')
        syst_type = syst_config.get('type')
        group = syst_config.get('group', [])

        if source == 'preprocessed' and syst_type == 'shape':
            variations = parse_variations(syst_config.get('variations', []))
            if isinstance(variations, dict):
                result['preprocessed_shape'].append((syst_name, variations, group))
            elif len(variations) > 2:
                result['multi_variation'].append((syst_name, variations, group))
            elif len(variations) == 2:
                result['preprocessed_shape'].append((syst_name, variations, group))
            else:
                logging.warning(f"Unexpected variation count for {syst_name}: {variations}")

        elif source == 'valued' and syst_type == 'shape':
            result['valued_shape'].append((syst_name, syst_config.get('value'), group))

        elif source == 'valued' and syst_type == 'lnN':
            result['valued_lnN'].append((syst_name, syst_config.get('value'), group))

    return result


# =============================================================================
# Adaptive Binning
# =============================================================================

def calculate_adaptive_bins(x0, sigma_eff, n_core_bins):
    """
    Generate bin edges: 2 merged sideband bins + n_core uniform core bins.

    Layout: [-10σ, -5σ] + n_core uniform bins in [-5σ, +5σ] + [+5σ, +10σ]
    Total bins = n_core + 2

    Args:
        x0: Peak position (fitted A mass)
        sigma_eff: Effective width for bin scaling
        n_core_bins: Number of uniform core bins

    Returns:
        numpy array of bin edges (length n_core_bins + 3)
    """
    core_fracs = np.linspace(-5, 5, n_core_bins + 1)
    sigma_fractions = np.concatenate([[-10], core_fracs, [10]])
    return x0 + sigma_fractions * sigma_eff


def check_binning_quality(background_hists):
    """
    Check if binning produces acceptable background statistics.

    Single criterion matching Combine's autoMCStats algorithm:
      n_eff = round(y^2 / sigma^2) >= AUTOMC_THRESHOLD (=5)
    where y = total background content, sigma^2 = sum of squared errors.

    This guarantees every bin gets Barlow-Beeston-lite treatment (1 NP per
    bin) instead of per-process treatment (N_proc NPs per bin). Subsumes the
    old criteria (content > 0, stat_err < 100%).

    h_total is built via TH1::Add(), which propagates errors in quadrature.
    No per-process pre-processing is assumed — honest stats drive the
    binning decision. Numerical hygiene (ensure_positive_integral,
    cap_stat_errors) is applied post-selection only, not before this check.

    Args:
        background_hists: dict of {process_name: TH1} for central backgrounds

    Returns:
        (ok, diagnostics): ok is True if all criteria pass,
            diagnostics is list of problem descriptions
    """
    h_total = None
    for name, h in background_hists.items():
        if h_total is None:
            h_total = h.Clone("total_bkg_check")
            h_total.SetDirectory(0)
        else:
            h_total.Add(h)

    if h_total is None:
        return False, ["No background histograms"]

    nbins = h_total.GetNbinsX()
    diagnostics = []

    for i in range(1, nbins + 1):
        bc = h_total.GetBinContent(i)
        be = h_total.GetBinError(i)
        if bc <= 0:
            diagnostics.append(f"bin {i}: total bkg = {bc:.4f} (non-positive, n_eff undefined)")
            continue
        if be <= 0:
            continue  # perfect stats, no issue
        neff = round(bc * bc / (be * be))
        if neff < AUTOMC_THRESHOLD:
            diagnostics.append(
                f"bin {i}: n_eff = {neff} < {AUTOMC_THRESHOLD} "
                f"(content={bc:.4f}, error={be:.4f})"
            )

    h_total.Delete()
    ok = len(diagnostics) == 0
    return ok, diagnostics


# =============================================================================
# Syst-driven post-binning merge
# =============================================================================

def rebin_hist_with_edges(h, new_edges, name=None):
    """Rebin TH1 onto new_edges (must be a subset of current edges). Returns a new detached TH1."""
    edges_arr = array('d', [float(e) for e in new_edges])
    out_name = name if name is not None else h.GetName()
    rebinned = h.Rebin(len(edges_arr) - 1, out_name + "__tmp_rebin", edges_arr)
    rebinned.SetDirectory(0)
    rebinned.SetName(out_name)
    rebinned.SetTitle(h.GetTitle())
    return rebinned


def select_merge_neighbor(nominal, stat_err, i):
    """Pick neighbour of bin `i` with higher relative stat error. Returns neighbour index or None."""
    nbins = len(nominal)
    if nbins <= 1:
        return None
    left = i - 1 if i - 1 >= 0 else None
    right = i + 1 if i + 1 < nbins else None
    if left is None:
        return right
    if right is None:
        return left

    def rel_err(idx):
        c = nominal[idx]
        return float("inf") if c <= 0 else stat_err[idx] / c

    return right if rel_err(right) >= rel_err(left) else left


def _hist_to_content_var(h):
    n = h.GetNbinsX()
    contents = np.fromiter((h.GetBinContent(i + 1) for i in range(n)), dtype=float, count=n)
    errors = np.fromiter((h.GetBinError(i + 1) for i in range(n)), dtype=float, count=n)
    return contents, errors * errors


def _snapshot_templates(templates):
    """Capture every TH1 in `templates` as (content, variance) numpy arrays."""
    snap = {}
    for key, val in templates.items():
        if isinstance(val, dict):
            snap[key] = {sub: _hist_to_content_var(h) for sub, h in val.items()}
        else:
            snap[key] = _hist_to_content_var(val)
    return snap


def _merge_adjacent(cv, lo):
    c, v = cv
    new_c = np.concatenate([c[:lo], [c[lo] + c[lo + 1]], c[lo + 2:]])
    new_v = np.concatenate([v[:lo], [v[lo] + v[lo + 1]], v[lo + 2:]])
    return new_c, new_v


def _merge_snapshot(snap, lo):
    out = {}
    for key, val in snap.items():
        if isinstance(val, dict) and val and isinstance(next(iter(val.values())), tuple):
            out[key] = {sub: _merge_adjacent(cv, lo) for sub, cv in val.items()}
        else:
            out[key] = _merge_adjacent(val, lo)
    return out


def _total_bkg_syst_from_snapshot(snap, bkg_processes, syst_names):
    """Per-bin (nominal, sigma, stat_err) on the current snapshot binning."""
    # Nominal + stat from summed bkg nominals
    nbins = None
    nominal = None
    stat_var = None
    for p in bkg_processes:
        proc = snap.get(p)
        if not proc or "nominal" not in proc:
            continue
        c, v = proc["nominal"]
        if nominal is None:
            nbins = len(c)
            nominal = np.zeros(nbins)
            stat_var = np.zeros(nbins)
        nominal += c
        stat_var += v
    if nominal is None:
        raise RuntimeError("_total_bkg_syst_from_snapshot: no background nominals found")

    sigma_sq = np.zeros(nbins)
    for syst in syst_names:
        up_total = np.zeros(nbins)
        dn_total = np.zeros(nbins)
        for p in bkg_processes:
            proc = snap.get(p)
            if not proc or "nominal" not in proc:
                continue
            nom_c = proc["nominal"][0]
            up_total += proc.get(syst + "Up", (nom_c,))[0]
            dn_total += proc.get(syst + "Down", (nom_c,))[0]
        d_up = np.abs(up_total - nominal)
        d_dn = np.abs(dn_total - nominal)
        sigma_sq += np.maximum(d_up, d_dn) ** 2

    return nominal, np.sqrt(sigma_sq), np.sqrt(stat_var)


def _collect_syst_names(snap, bkg_processes):
    names = set()
    for p in bkg_processes:
        proc = snap.get(p, {})
        for key in proc:
            if key == "nominal":
                continue
            if key.endswith("Up"):
                names.add(key[:-2])
            elif key.endswith("Down"):
                names.add(key[:-4])
    return sorted(names)


def apply_syst_driven_merging(bin_edges, templates, bkg_processes,
                              max_rel_syst=2.0, min_nbins=3, logger=None):
    """
    Merge bins where total-bkg syst envelope / nominal > max_rel_syst.

    The merge loop operates on numpy (content, variance) snapshots; real TH1s
    are rebinned only once at the end. The merge target for a flagged bin is
    the neighbour with the higher relative stat error (see select_merge_neighbor).

    Returns (new_bin_edges, templates, n_merges). `templates` is a new dict;
    original TH1s are untouched if no merges occurred.
    """
    log = logger if logger is not None else logging

    snap = _snapshot_templates(templates)
    syst_names = _collect_syst_names(snap, bkg_processes)
    current_edges = np.asarray(bin_edges, dtype=float).copy()
    n_merges = 0

    while True:
        nbins = len(current_edges) - 1
        if nbins <= min_nbins:
            log.info(f"  Syst-merge: reached minimum nbins={min_nbins}, stopping")
            break

        nominal, sigma, stat_err = _total_bkg_syst_from_snapshot(snap, bkg_processes, syst_names)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = np.where(nominal > 0, sigma / nominal, 0.0)
        worst = int(np.argmax(rel))
        worst_rel = float(rel[worst])

        if worst_rel <= max_rel_syst:
            if n_merges == 0:
                log.info(f"  Syst-merge: max rel syst = {worst_rel:.3f} <= "
                         f"{max_rel_syst:.2f}, no merging needed")
            else:
                log.info(f"  Syst-merge: converged after {n_merges} merges "
                         f"(max rel syst = {worst_rel:.3f})")
            break

        neighbour = select_merge_neighbor(nominal, stat_err, worst)
        if neighbour is None:
            log.warning("  Syst-merge: cannot merge further (single bin)")
            break

        lo = min(worst, neighbour)
        drop_idx = lo + 1
        log.info(
            f"  Syst-merge #{n_merges + 1}: bin {worst} rel syst {worst_rel:.3f} > "
            f"{max_rel_syst:.2f}; merging with bin {neighbour}; "
            f"dropping edge at {current_edges[drop_idx]:.4f}"
        )

        snap = _merge_snapshot(snap, lo)
        current_edges = np.delete(current_edges, drop_idx)
        n_merges += 1

    if n_merges > 0:
        templates = {
            key: (
                {sub: rebin_hist_with_edges(h, current_edges) for sub, h in val.items()}
                if isinstance(val, dict) else rebin_hist_with_edges(val, current_edges)
            )
            for key, val in templates.items()
        }

    return current_edges, templates, n_merges
