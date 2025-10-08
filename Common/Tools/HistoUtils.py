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