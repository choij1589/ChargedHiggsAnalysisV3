#!/usr/bin/env python
"""TriLepton-specific utilities for path construction."""
from math import sqrt


def apply_rate_uncertainty(hist, rel_unc):
    """Apply relative rate uncertainty to histogram errors (in quadrature).

    Args:
        hist: ROOT TH1 histogram to modify (in-place)
        rel_unc: Relative uncertainty as a fraction (e.g., 0.20 for 20%)
    """
    for bin_idx in range(hist.GetNcells()):
        content = hist.GetBinContent(bin_idx)
        stat_error = hist.GetBinError(bin_idx)
        rate_error = content * rel_unc
        hist.SetBinError(bin_idx, sqrt(stat_error**2 + rate_error**2))


def scale_with_variations(h_central, variations, factor):
    """Scale a central histogram and its systematic variations by the same factor.

    Args:
        h_central: ROOT TH1 to scale in place
        variations: {source: (h_up, h_down)} scaled alongside it
        factor: Multiplicative scale (K-factor, ConvSF, ...)
    """
    if factor == 1.0:
        return
    h_central.Scale(factor)
    for h_up, h_down in variations.values():
        h_up.Scale(factor)
        h_down.Scale(factor)


def build_sknanoutput_path(workdir, channel, flag, era, sample,
                           is_nonprompt=False, run_syst=False, no_wzsf=False,
                           no_hem_veto=False):
    """Construct SKNanoOutput file path based on channel type and run mode.

    All channels use PromptAnalyzer/MatrixAnalyzer naming.
    Control region channels (ZG*, WZ*) add _RunCR_NoTreeMode suffix.
    For HEM veto studies (2018, electron channels), no_hem_veto appends
    _RunNoHEMVeto_NoTreeMode instead of _RunCR_NoTreeMode.

    Args:
        workdir: Base WORKDIR path
        channel: Analysis channel (SR1E2Mu, ZG1E2Mu, WZ3Mu, etc.)
        flag: Run flag (Run1E2Mu, Run3Mu)
        era: Data era (2017, 2022, etc.)
        sample: Sample name
        is_nonprompt: True for nonprompt (Matrix) samples
        run_syst: True to include _RunSyst suffix
        no_wzsf: True to include _RunNoWZSF suffix
        no_hem_veto: True to use NoHEMVeto samples (2018 electron channels only)

    Returns:
        str: Full path to ROOT file
    """
    is_cr_channel = channel.startswith("ZG") or channel.startswith("WZ")

    # Use Analyzer naming for all channels
    analyzer = "MatrixAnalyzer" if is_nonprompt else "PromptAnalyzer"

    flag_parts = [flag]
    if no_wzsf:
        flag_parts.append("RunNoWZSF")
    if run_syst:
        flag_parts.append("RunSyst")
    if no_hem_veto:
        flag_parts.append("RunNoHEMVeto_NoTreeMode")
    elif is_cr_channel:
        flag_parts.append("RunCR_NoTreeMode")
    full_flag = "_".join(flag_parts)

    return f"{workdir}/SKNanoOutput/{analyzer}/{full_flag}/{era}/Skim_TriLep_{sample}.root"
