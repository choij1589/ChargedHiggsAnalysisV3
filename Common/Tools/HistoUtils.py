#!/usr/bin/env python
"""
HistoUtils.py - Shared histogram utilities for DiLepton and TriLepton analyses

This module contains common functionality for:
- Missing histogram logging
- Histogram loading with error handling
- Systematic uncertainty calculation
- Era configuration loading
- Sample merging utilities
"""

import os
import logging
import json
import ROOT
from math import sqrt, pow
from array import array


def setup_missing_histogram_logging(args):
    """Setup logging for missing histograms

    Args:
        args: Argument parser object with era, channel, histkey, debug attributes

    Returns:
        logging.Logger: Configured logger for missing histograms
    """
    # Setup missing histogram logging
    log_file = f"logs/{args.era}/{args.channel}/{args.histkey.replace('/', '_')}.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create file logger for missing histograms
    missing_logger = logging.getLogger('missing_histograms')
    missing_logger.setLevel(logging.INFO)
    missing_logger.propagate = False  # Don't propagate to root logger (console)
    missing_handler = logging.FileHandler(log_file)
    missing_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    missing_logger.addHandler(missing_handler)

    # Add console handler only if debug mode is enabled
    if args.debug:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
        missing_logger.addHandler(console_handler)

    return missing_logger


def clip_negative_bins(h):
    """Zero out negative bin contents (and their errors) in place.

    Iterates all cells (including under/overflow) to match the loop style
    used in calculate_systematics. Silences THStack::BuildStack warnings
    that arise from negatively-weighted MC events and the matrix-method
    nonprompt estimate.
    """
    if h is None:
        return h
    for i in range(h.GetNcells()):
        if h.GetBinContent(i) < 0.0:
            h.SetBinContent(i, 0.0)
            h.SetBinError(i, 0.0)
    return h


def load_histogram(file_path, hist_path, era=None, missing_logger=None):
    """Load a single histogram from a ROOT file

    Args:
        file_path (str): Path to ROOT file
        hist_path (str): Path to histogram within ROOT file
        era (str, optional): Era information for logging
        missing_logger (logging.Logger, optional): Logger for missing histograms

    Returns:
        ROOT.TH1 or None: Loaded histogram or None if not found
    """
    era_info = f"[{era}] " if era else ""

    if not os.path.exists(file_path):
        if missing_logger:
            missing_logger.info(f"{era_info}MISSING_FILE: {file_path}")
        return None

    try:
        f = ROOT.TFile.Open(file_path)
        if not f or f.IsZombie():
            if f: f.Close()
            if missing_logger:
                missing_logger.info(f"{era_info}CANNOT_OPEN_FILE: {file_path}")
            return None

        h = f.Get(hist_path)
        if h and h.GetEntries() >= 0:
            h.SetDirectory(0)
            f.Close()
            return h
        else:
            f.Close()
            if missing_logger:
                missing_logger.info(f"{era_info}MISSING_CENTRAL: {hist_path} in {os.path.basename(file_path)}")
            return None

    except Exception as e:
        if missing_logger:
            missing_logger.info(f"{era_info}ERROR_LOADING: {hist_path} in {os.path.basename(file_path)} - {e}")
        return None


def calculate_systematics(h, systematics, file_path, args, era=None, missing_logger=None):
    """Calculate systematic uncertainties for a histogram

    Args:
        h (ROOT.TH1): Central histogram
        systematics (dict): Dictionary of systematic variations
        file_path (str): Path to ROOT file
        args: Argument parser object with channel, histkey, exclude attributes
        era (str, optional): Era information for logging
        missing_logger (logging.Logger, optional): Logger for missing histograms

    Returns:
        ROOT.TH1: Histogram with systematic uncertainties applied
    """
    clip_negative_bins(h)

    if args.exclude:
        return h

    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        return h

    era_info = f"[{era}] " if era else ""

    try:
        hSysts = []
        for syst, sources in systematics.items():
            syst_up, syst_down = tuple(sources)
            h_up = f.Get(f"{args.channel}/{syst_up}/{args.histkey}")
            h_down = f.Get(f"{args.channel}/{syst_down}/{args.histkey}")
            if h_up and h_down:
                h_up.SetDirectory(0)
                h_down.SetDirectory(0)
                clip_negative_bins(h_up)
                clip_negative_bins(h_down)
                hSysts.append((h_up, h_down))
            else:
                if missing_logger:
                    if not h_up:
                        missing_logger.info(f"{era_info}MISSING_SYSTEMATIC: {args.channel}/{syst_up}/{args.histkey} in {os.path.basename(file_path)}")
                    if not h_down:
                        missing_logger.info(f"{era_info}MISSING_SYSTEMATIC: {args.channel}/{syst_down}/{args.histkey} in {os.path.basename(file_path)}")

        # Apply systematic uncertainties bin by bin
        for bin in range(h.GetNcells()):
            stat_unc = h.GetBinError(bin)
            envelops = []
            for h_up, h_down in hSysts:
                systUp = abs(h_up.GetBinContent(bin) - h.GetBinContent(bin))
                systDown = abs(h_down.GetBinContent(bin) - h.GetBinContent(bin))
                envelops.append(max(systUp, systDown))
            total_unc = sqrt(pow(stat_unc, 2) + sum([pow(x, 2) for x in envelops]))
            h.SetBinError(bin, total_unc)

    finally:
        f.Close()

    return h


def sum_histograms(hist_list, name):
    """Sum a list of histograms

    Args:
        hist_list (list): List of ROOT histograms
        name (str): Name for the summed histogram

    Returns:
        ROOT.TH1 or None: Summed histogram or None if input list is empty
    """
    if not hist_list:
        return None

    total = hist_list[0].Clone(name)
    total.SetDirectory(0)
    for h in hist_list[1:]:
        total.Add(h)
    return total


def load_era_configs(args, era_list):
    """Load sample groups and systematics for all relevant eras

    Args:
        args: Argument parser object with channel attribute
        era_list (list): List of eras to process

    Returns:
        tuple: (era_samples, era_systematics) dictionaries
    """
    samplegroup_config = json.load(open("configs/samplegroup.json"))
    systematics_config = json.load(open("configs/systematics.json"))

    era_samples = {}
    era_systematics = {}

    for era in era_list:
        era_samples[era] = samplegroup_config[era][args.channel]
        # For systematics, use the Run2/Run3 key based on the era
        if era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
            run_key = "Run2"
        else:
            run_key = "Run3"
        era_systematics[era] = systematics_config[run_key][args.channel]

    return era_samples, era_systematics


def merge_systematics(era_systematics):
    """Merge systematics from all eras

    Args:
        era_systematics (dict): Dictionary of systematic variations per era

    Returns:
        list: List of all unique systematic variations
    """
    all_systs = []
    for era_systs in era_systematics.values():
        for syst in era_systs:
            if syst not in all_systs:
                all_systs.append(syst)
    return all_systs


def get_sample_lists(era_samples, categories):
    """Extract and organize sample lists

    Args:
        era_samples (dict): Dictionary of samples per era
        categories (list): List of sample categories to process

    Returns:
        tuple: (data_samples, mc_categories, all_mc_samples)
    """
    # Get era list from era_samples keys
    era_list = list(era_samples.keys())

    # All data samples (different names per era)
    data_samples = []
    for era in era_list:
        data_samples.extend(era_samples[era]["data"])

    # Unique MC samples by category
    mc_categories = {cat: set() for cat in categories}
    for era in era_list:
        for category in categories:
            if category in era_samples[era]:
                mc_categories[category].update(era_samples[era][category])

    # Convert to lists and create full MC list
    mc_lists = {cat: list(samples) for cat, samples in mc_categories.items()}
    all_mc_samples = sum(mc_lists.values(), [])

    return data_samples, mc_lists, all_mc_samples


def _merge_bins_for_chi2(h_obs, h_exp):
    """Merge bins failing ROOT's Chi2Test "UW" criteria into adjacent lower-yield neighbors.

    Criteria per bin:
    - Unweighted (data): content >= 1
    - Weighted (MC):     effective entries = content^2 / sum_w2 >= 10

    A bin failing either criterion is merged into its adjacent neighbor with
    lower MC yield. The same binning is applied to both histograms. Iterates
    until all bins pass or only one bin remains.

    Temporary histograms only — originals are not modified.

    Args:
        h_obs: Unweighted (data) histogram — no Sumw2
        h_exp: Weighted (MC) histogram — Sumw2 must be set

    Returns:
        tuple: (h_obs_merged, h_exp_merged) with merged bins
    """
    nbins = h_obs.GetNbinsX()

    def group_stats(group):
        obs_n = sum(h_obs.GetBinContent(i) for i in group)
        exp_c = sum(h_exp.GetBinContent(i) for i in group)
        exp_err2 = sum(h_exp.GetBinError(i)**2 for i in group)
        exp_neff = exp_c**2 / exp_err2 if exp_err2 > 0 else 0.0
        return obs_n, exp_c, exp_neff

    def fails(group):
        obs_n, _, exp_neff = group_stats(group)
        return obs_n < 1 or exp_neff < 10

    # Include overflow (nbins+1) as the last group so it can be merged if sparse.
    # Chi2Test "OF" adds overflow to the last visible bin; by folding it here we
    # ensure it meets the threshold. The output overflow slot is left at zero.
    groups = [[i] for i in range(1, nbins + 2)]

    while len(groups) > 1:
        made_change = False
        for idx in range(len(groups)):
            if not fails(groups[idx]):
                continue

            left = idx - 1 if idx > 0 else None
            right = idx + 1 if idx < len(groups) - 1 else None

            if left is None and right is None:
                break
            elif left is None:
                target = right
            elif right is None:
                target = left
            else:
                _, left_exp, _ = group_stats(groups[left])
                _, right_exp, _ = group_stats(groups[right])
                target = left if left_exp <= right_exp else right

            lo, hi = min(idx, target), max(idx, target)
            merged = sorted(groups[lo] + groups[hi])
            groups = groups[:lo] + [merged] + groups[lo + 2:]
            made_change = True
            break  # restart after structural change

        if not made_change:
            break

    if len(groups) == nbins + 1:
        return h_obs.Clone("h_obs_merged"), h_exp.Clone("h_exp_merged")

    logging.debug("Chi2 bin merging: %d -> %d bins (incl. overflow)", nbins, len(groups))

    new_nbins = len(groups)
    h_obs_out = ROOT.TH1D("h_obs_merged", "", new_nbins, 0, new_nbins)
    h_obs_out.SetDirectory(0)
    h_exp_out = ROOT.TH1D("h_exp_merged", "", new_nbins, 0, new_nbins)
    h_exp_out.SetDirectory(0)
    h_exp_out.Sumw2()

    # All groups (including any overflow-containing group) become visible bins.
    # The output overflow slot stays at zero, so Chi2Test "OF" adds nothing extra.
    for new_bin, grp in enumerate(groups, 1):
        obs_content = sum(h_obs.GetBinContent(i) for i in grp)
        exp_content = sum(h_exp.GetBinContent(i) for i in grp)
        exp_err2 = sum(h_exp.GetBinError(i)**2 for i in grp)
        h_obs_out.SetBinContent(new_bin, obs_content)
        h_exp_out.SetBinContent(new_bin, exp_content)
        h_exp_out.SetBinError(new_bin, sqrt(exp_err2))

    return h_obs_out, h_exp_out


def calculate_chi2(h_obs, h_exp, normalize=False):
    """
    Calculate chi^2 test between observed and expected histograms using ROOT's Chi2Test.

    Uses TH1::Chi2Test with "UW" option (unweighted data vs weighted MC),
    which properly handles Poisson statistics for low-count bins.

    Note: The `res` array parameter in Chi2Test returns incorrect values for "UW" option
    (ROOT bug). We use separate calls with "CHI2" and "CHI2/NDF" options as a workaround.

    Args:
        h_obs: Observed histogram (data, unweighted)
        h_exp: Expected histogram (sum of backgrounds, weighted)
        normalize: If True, normalize both to unit area (shape-only test)

    Returns:
        tuple: (chi2, ndf, p_value)
    """
    # Create a fresh histogram for data without Sumw2 to ensure ROOT treats it as unweighted.
    # Simply cloning preserves Sumw2 status which causes "UW" option warnings.
    h_obs_test = ROOT.TH1D("h_obs_chi2", "", h_obs.GetNbinsX(),
                           h_obs.GetXaxis().GetXmin(), h_obs.GetXaxis().GetXmax())
    h_obs_test.SetDirectory(0)
    for i in range(0, h_obs.GetNbinsX() + 2):  # Include under/overflow
        h_obs_test.SetBinContent(i, h_obs.GetBinContent(i))
        # Don't set errors - let ROOT compute sqrt(N) for unweighted histogram

    h_exp_test = h_exp.Clone()
    h_exp_test.SetDirectory(0)

    # Merge bins failing Chi2Test "UW" criteria before computing test statistic
    h_obs_test, h_exp_test = _merge_bins_for_chi2(h_obs_test, h_exp_test)

    # Build options string
    # "UW" = unweighted (data) vs weighted (MC) - proper Poisson handling
    # "OF" = include overflow bins - reduces dependence on xRange
    options = "UW OF"
    if normalize:
        options += " NORM"

    # Get values using separate calls - the `res` array parameter returns garbage
    # values for "UW" option in ROOT (tested in ROOT 6.32.08)
    p_value = h_obs_test.Chi2Test(h_exp_test, options)
    chi2 = h_obs_test.Chi2Test(h_exp_test, options + " CHI2")
    chi2_ndf = h_obs_test.Chi2Test(h_exp_test, options + " CHI2/NDF")

    # Calculate ndf from chi2 / (chi2/ndf)
    ndf = int(round(chi2 / chi2_ndf)) if chi2_ndf > 0 else 0

    return chi2, ndf, p_value