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
import numpy as np
from math import sqrt, pow


def build_sknanoutput_path(workdir, channel, flag, era, sample,
                           is_nonprompt=False, run_syst=False, no_wzsf=False):
    """Construct SKNanoOutput file path based on channel type and run mode.

    Control region channels (ZG*, WZ*) use new PromptAnalyzer/MatrixAnalyzer naming
    with _RunCR_NoTreeMode suffix. Signal region channels (SR*, ZFake*, TTZ*) use
    the traditional PromptSelector/MatrixSelector naming.

    Args:
        workdir: Base WORKDIR path
        channel: Analysis channel (SR1E2Mu, ZG1E2Mu, WZ3Mu, etc.)
        flag: Run flag (Run1E2Mu, Run3Mu)
        era: Data era (2017, 2022, etc.)
        sample: Sample name
        is_nonprompt: True for nonprompt (Matrix) samples
        run_syst: True to include _RunSyst suffix
        no_wzsf: True to include _RunNoWZSF suffix

    Returns:
        str: Full path to ROOT file
    """
    is_cr_channel = channel.startswith("ZG") or channel.startswith("WZ")

    if is_cr_channel:
        # New Analyzer naming for CR channels
        analyzer = "MatrixAnalyzer" if is_nonprompt else "PromptAnalyzer"
        flag_parts = [flag]
        if no_wzsf:
            flag_parts.append("RunNoWZSF")
        if run_syst:
            flag_parts.append("RunSyst")
        flag_parts.append("RunCR_NoTreeMode")
        full_flag = "_".join(flag_parts)
    else:
        # Old Selector naming for SR channels
        analyzer = "MatrixSelector" if is_nonprompt else "PromptSelector"
        flag_parts = [flag]
        if no_wzsf:
            flag_parts.append("RunNoWZSF")
        if run_syst:
            flag_parts.append("RunSyst")
        full_flag = "_".join(flag_parts)

    return f"{workdir}/SKNanoOutput/{analyzer}/{full_flag}/{era}/Skim_TriLep_{sample}.root"


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


def rebin_for_chi2_validity(h_obs, h_exp, min_expected=10.0):
    """
    Rebin histograms to ensure chi-squared test validity.
    ROOT's Chi2Test requires:
    - At least 1 event per bin in observed (unweighted)
    - At least 10 effective entries per bin in expected (weighted)

    This function merges adjacent bins until each bin has:
    - expected content >= min_expected
    - expected effective entries >= min_expected

    Args:
        h_obs: Observed histogram (data)
        h_exp: Expected histogram (sum of backgrounds)
        min_expected: Minimum expected count/effective entries per bin (default 10.0)

    Returns:
        Tuple of (h_obs_rebinned, h_exp_rebinned)
    """
    def get_effective_entries(content, err2):
        """Calculate effective entries: Neff = (sum w)^2 / (sum w^2)"""
        if err2 > 0:
            return content**2 / err2
        return 0.0

    # First pass: determine bin edges based on expected content AND effective entries
    bin_edges = [h_exp.GetBinLowEdge(1)]
    accumulated_exp = 0.0
    accumulated_err2 = 0.0

    for bin_idx in range(1, h_exp.GetNbinsX() + 1):
        accumulated_exp += h_exp.GetBinContent(bin_idx)
        accumulated_err2 += h_exp.GetBinError(bin_idx) ** 2

        eff_entries = get_effective_entries(accumulated_exp, accumulated_err2)
        should_split = (accumulated_exp >= min_expected and eff_entries >= min_expected)
        is_last_bin = (bin_idx == h_exp.GetNbinsX())

        if should_split or is_last_bin:
            bin_edges.append(h_exp.GetBinLowEdge(bin_idx + 1))
            accumulated_exp = 0.0
            accumulated_err2 = 0.0

    # If last bin has too few events, merge with previous bin
    if len(bin_edges) > 2:
        # Check last bin's effective entries
        last_bin_start = h_exp.FindBin(bin_edges[-2])
        last_bin_end = h_exp.GetNbinsX()
        last_exp = sum(h_exp.GetBinContent(i) for i in range(last_bin_start, last_bin_end + 1))
        last_err2 = sum(h_exp.GetBinError(i)**2 for i in range(last_bin_start, last_bin_end + 1))
        last_eff = get_effective_entries(last_exp, last_err2)

        if last_exp < min_expected or last_eff < min_expected:
            # Merge last bin with previous
            bin_edges.pop(-2)

    n_bins = len(bin_edges) - 1
    if n_bins < 1:
        return h_obs.Clone(), h_exp.Clone()

    bin_edges_arr = np.array(bin_edges, dtype=float)
    h_obs_rebinned = ROOT.TH1D(h_obs.GetName() + "_chi2", "", n_bins, bin_edges_arr)
    h_exp_rebinned = ROOT.TH1D(h_exp.GetName() + "_chi2", "", n_bins, bin_edges_arr)
    h_obs_rebinned.SetDirectory(0)
    h_exp_rebinned.SetDirectory(0)

    for new_bin in range(1, n_bins + 1):
        bin_low = h_obs_rebinned.GetBinLowEdge(new_bin)
        bin_up = h_obs_rebinned.GetBinLowEdge(new_bin + 1)
        orig_start = h_obs.FindBin(bin_low)
        orig_end = h_obs.FindBin(bin_up - 0.001)

        sum_obs, sum_exp = 0.0, 0.0
        sum_obs_err2, sum_exp_err2 = 0.0, 0.0

        for orig_bin in range(orig_start, orig_end + 1):
            sum_obs += h_obs.GetBinContent(orig_bin)
            sum_exp += h_exp.GetBinContent(orig_bin)
            sum_obs_err2 += h_obs.GetBinError(orig_bin) ** 2
            sum_exp_err2 += h_exp.GetBinError(orig_bin) ** 2

        h_obs_rebinned.SetBinContent(new_bin, sum_obs)
        h_exp_rebinned.SetBinContent(new_bin, sum_exp)
        h_obs_rebinned.SetBinError(new_bin, sqrt(sum_obs_err2))
        h_exp_rebinned.SetBinError(new_bin, sqrt(sum_exp_err2))

    return h_obs_rebinned, h_exp_rebinned


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