"""Utility functions for template generation."""
import os
import re
import json
import logging
import ROOT


def save_json(data, path):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def parse_variations(variation_spec):
    """Parse variation specification strings (e.g., ["Scale_{0..8}//5,7"])."""
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


def calculate_weight_scale(value, direction):
    """Calculate weight scale for valued+shape systematics."""
    if value >= 1.0:
        return value if direction == 'up' else 2.0 - value
    return 1.0 + value if direction == 'up' else 1.0 - value


def ensure_positive_integral(hist, min_integral=1e-10):
    """Ensure histogram has positive integral for Combine normalization."""
    modified = False

    for i in range(1, hist.GetNbinsX() + 1):
        if hist.GetBinContent(i) < 0:
            logging.warning(f"  {hist.GetName()}, bin {i}: negative content, setting to 0")
            hist.SetBinContent(i, 0.0)
            hist.SetBinError(i, 0.0)
            modified = True

    if hist.Integral() <= 0:
        logging.warning(f"  {hist.GetName()} has non-positive integral")
        central_bin = hist.GetNbinsX() // 2 + 1
        hist.SetBinContent(central_bin, min_integral)
        hist.SetBinError(central_bin, min_integral)
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


def process_systematics(output_file, basedir, process, central_hist, syst_categories,
                        applicable_groups, hist_args, is_merged=False, merge_list=None):
    """
    Process all systematics for a given process.

    Args:
        output_file: ROOT TFile for output
        basedir: Sample directory
        process: Process name
        central_hist: Central histogram for scaling
        syst_categories: Dict from categorize_systematics()
        applicable_groups: List of group names to match
        hist_args: Dict with keys: bin_edges, mA, width, sigma, binning, threshold, bg_weights, masspoint
        is_merged: If True, use getHistMerged
        merge_list: List of processes to merge (required if is_merged=True)
    """
    from makeBinnedTemplates import getHist, getHistMerged, createEnvelopeHists

    def get_hist_func(syst):
        if is_merged:
            return getHistMerged(basedir, merge_list, hist_args['bin_edges'],
                                 hist_args['mA'], hist_args['width'], hist_args['sigma'],
                                 hist_args['binning'], syst, hist_args['threshold'],
                                 hist_args['bg_weights'], hist_args['masspoint'])
        return getHist(basedir, process, hist_args['bin_edges'],
                       hist_args['mA'], hist_args['width'], hist_args['sigma'],
                       hist_args['binning'], syst, hist_args['threshold'],
                       hist_args['bg_weights'], hist_args['masspoint'])

    # Preprocessed shape systematics
    for syst_name, variations, group in syst_categories['preprocessed_shape']:
        if not any(g in group for g in applicable_groups):
            continue
        for var in variations:
            output_tree = get_output_tree_name(syst_name, var)
            try:
                hist = get_hist_func(output_tree)
                ensure_positive_integral(hist)
                output_file.cd()
                hist.Write()
            except (FileNotFoundError, RuntimeError) as e:
                logging.warning(f"    Skipping {process}/{syst_name}/{var}: {e}")

    # Valued shape systematics
    for syst_name, value, group in syst_categories['valued_shape']:
        if not any(g in group for g in applicable_groups):
            continue
        for direction in ["up", "down"]:
            hist = create_scaled_hist(central_hist, process, syst_name, value, direction)
            ensure_positive_integral(hist)
            output_file.cd()
            hist.Write()

    # Multi-variation systematics (PDF/Scale envelopes) - signal only
    if "signal" in applicable_groups:
        for syst_name, variations, group in syst_categories['multi_variation']:
            if "signal" not in group:
                continue
            logging.info(f"  Creating envelope: {syst_name}")

            tree_variations = []
            for var in variations:
                if var.startswith("pdf_"):
                    tree_variations.append(f"PDF_{int(var.replace('pdf_', ''))}")
                elif var.startswith("Scale_"):
                    tree_variations.append(var)
                else:
                    tree_variations.append(var)

            try:
                hist_up, hist_down = createEnvelopeHists(
                    basedir, process, hist_args['bin_edges'],
                    hist_args['mA'], hist_args['width'], hist_args['sigma'],
                    hist_args['binning'], tree_variations, syst_name,
                    hist_args['threshold'], hist_args['bg_weights'], hist_args['masspoint']
                )
                ensure_positive_integral(hist_up)
                ensure_positive_integral(hist_down)
                output_file.cd()
                hist_up.Write()
                hist_down.Write()
            except RuntimeError as e:
                logging.warning(f"    Skipping envelope {syst_name}: {e}")
