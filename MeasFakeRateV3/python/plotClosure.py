#!/usr/bin/env python
import os
import logging
import argparse
import ROOT
import json
import numpy as np
from plotter import ComparisonCanvas, get_era_list, get_CoM_energy

def get_adaptive_bin_edges(hist, max_frac_error=0.3, min_content=5.0, mass_range=(10.0, 160.0)):
    """
    Calculate variable bin edges based on fractional error threshold.
    """
    # Find bin range corresponding to mass range
    bin_start = hist.FindBin(mass_range[0])
    bin_end = hist.FindBin(mass_range[1])

    # Collect variable bin edges
    bin_edges = []
    accumulated_sumW = 0.0
    accumulated_sumW2 = 0.0  # sum of squared weights
    
    # Always start with the lower edge of first bin
    bin_edges.append(hist.GetBinLowEdge(bin_start))

    for bin in range(bin_start, bin_end + 1):
        bin_content = hist.GetBinContent(bin)
        bin_error = hist.GetBinError(bin)
        sumW2_this_bin = bin_error * bin_error  # sumW² for this bin

        # Accumulate
        accumulated_sumW += bin_content
        accumulated_sumW2 += sumW2_this_bin

        # Calculate fractional error
        accumulated_error = np.sqrt(accumulated_sumW2) if accumulated_sumW2 > 0 else 0.0
        frac_error = accumulated_error / accumulated_sumW if accumulated_sumW > 0 else float('inf')

        # Check if we should create a bin boundary
        # Criteria:
        # 1. Fractional error is below threshold AND
        # 2. Accumulated content is above minimum
        should_split = (frac_error <= max_frac_error and accumulated_sumW >= min_content)

        # Also split at the last bin in range
        is_last_bin = (bin == bin_end)

        if should_split or is_last_bin:
            # Create bin edge at the upper edge of current bin
            bin_edges.append(hist.GetBinLowEdge(bin + 1))

            # Reset accumulators for next bin group
            accumulated_sumW = 0.0
            accumulated_sumW2 = 0.0

    return np.array(bin_edges, dtype=float)

def apply_variable_binning(hist, bin_edges, suffix="_rebinned"):
    """
    Apply variable binning to a histogram.
    """
    n_bins = len(bin_edges) - 1
    if n_bins < 1:
        logging.warning("Rebinning produced less than 1 bin, returning original histogram")
        return hist.Clone()

    hist_name = hist.GetName() + suffix
    h_rebinned = ROOT.TH1D(hist_name, hist.GetTitle(), n_bins, bin_edges)
    h_rebinned.SetDirectory(0)

    # Fill the rebinned histogram
    # For each bin in the new histogram, sum up the corresponding bins from the original
    for new_bin in range(1, h_rebinned.GetNbinsX() + 1):
        bin_low_edge = h_rebinned.GetBinLowEdge(new_bin)
        bin_up_edge = h_rebinned.GetBinLowEdge(new_bin + 1)

        # Find corresponding bins in original histogram
        orig_bin_start = hist.FindBin(bin_low_edge)
        orig_bin_end = hist.FindBin(bin_up_edge - 0.001)  # Subtract small value to avoid boundary issues

        sum_content = 0.0
        sum_error2 = 0.0

        for orig_bin in range(orig_bin_start, orig_bin_end + 1):
            sum_content += hist.GetBinContent(orig_bin)
            sum_error2 += hist.GetBinError(orig_bin) ** 2

        h_rebinned.SetBinContent(new_bin, sum_content)
        h_rebinned.SetBinError(new_bin, np.sqrt(sum_error2))

    logging.debug(f"Rebinned {hist.GetName()} from {hist.GetNbinsX()} bins to {h_rebinned.GetNbinsX()} bins")
    
    return h_rebinned

def rebin_for_chi2_validity(h_obs, h_exp, min_expected=5.0):
    """
    Rebin histograms to ensure chi-squared test validity.

    Chi-squared test requires expected count >= 5 in each bin (classical rule).
    Merge consecutive bins until this criterion is met.

    Args:
        h_obs: Observed histogram
        h_exp: Expected histogram
        min_expected: Minimum expected count per bin (default 5.0)

    Returns:
        Tuple of (h_obs_rebinned, h_exp_rebinned)
    """
    # Collect variable bin edges
    bin_edges = []
    accumulated_exp = 0.0
    accumulated_obs = 0.0
    accumulated_exp_err2 = 0.0
    accumulated_obs_err2 = 0.0

    # Start with the lower edge of first bin
    bin_edges.append(h_exp.GetBinLowEdge(1))

    for bin in range(1, h_exp.GetNbinsX() + 1):
        exp_content = h_exp.GetBinContent(bin)
        obs_content = h_obs.GetBinContent(bin)
        exp_error = h_exp.GetBinError(bin)
        obs_error = h_obs.GetBinError(bin)

        # Accumulate
        accumulated_exp += exp_content
        accumulated_obs += obs_content
        accumulated_exp_err2 += exp_error * exp_error
        accumulated_obs_err2 += obs_error * obs_error

        # Check if we've accumulated enough expected events
        should_split = (accumulated_exp >= min_expected)
        is_last_bin = (bin == h_exp.GetNbinsX())

        if should_split or is_last_bin:
            bin_edges.append(h_exp.GetBinLowEdge(bin + 1))
            # Reset accumulators
            accumulated_exp = 0.0
            accumulated_obs = 0.0
            accumulated_exp_err2 = 0.0
            accumulated_obs_err2 = 0.0

    # Create new histograms with variable binning
    n_bins = len(bin_edges) - 1
    if n_bins < 1:
        logging.warning("Chi2 rebinning produced less than 1 bin, returning original histograms")
        return h_obs.Clone(), h_exp.Clone()

    h_obs_rebinned = ROOT.TH1D(h_obs.GetName() + "_chi2", h_obs.GetTitle(), n_bins, np.array(bin_edges, dtype=float))
    h_exp_rebinned = ROOT.TH1D(h_exp.GetName() + "_chi2", h_exp.GetTitle(), n_bins, np.array(bin_edges, dtype=float))
    h_obs_rebinned.SetDirectory(0)
    h_exp_rebinned.SetDirectory(0)

    # Fill the rebinned histograms
    for new_bin in range(1, h_obs_rebinned.GetNbinsX() + 1):
        bin_low_edge = h_obs_rebinned.GetBinLowEdge(new_bin)
        bin_up_edge = h_obs_rebinned.GetBinLowEdge(new_bin + 1)

        orig_bin_start = h_obs.FindBin(bin_low_edge)
        orig_bin_end = h_obs.FindBin(bin_up_edge - 0.001)

        sum_obs = 0.0
        sum_exp = 0.0
        sum_obs_err2 = 0.0
        sum_exp_err2 = 0.0

        for orig_bin in range(orig_bin_start, orig_bin_end + 1):
            sum_obs += h_obs.GetBinContent(orig_bin)
            sum_exp += h_exp.GetBinContent(orig_bin)
            sum_obs_err2 += h_obs.GetBinError(orig_bin) ** 2
            sum_exp_err2 += h_exp.GetBinError(orig_bin) ** 2

        h_obs_rebinned.SetBinContent(new_bin, sum_obs)
        h_exp_rebinned.SetBinContent(new_bin, sum_exp)
        h_obs_rebinned.SetBinError(new_bin, np.sqrt(sum_obs_err2))
        h_exp_rebinned.SetBinError(new_bin, np.sqrt(sum_exp_err2))

    logging.debug(f"Chi2 rebinning: {h_exp.GetNbinsX()} → {h_exp_rebinned.GetNbinsX()} bins (min_exp={min_expected})")

    return h_obs_rebinned, h_exp_rebinned

def calculate_chi2_with_scale(h_obs, h_exp, scale_factor, normalize=False):
    """
    Calculate chi-squared test with scaled uncertainties on h_exp.

    Args:
        h_obs: Observed histogram
        h_exp: Expected histogram
        scale_factor: Scale factor to apply to h_exp uncertainties
        normalize: If True, normalize histograms before chi2 calculation (shape-only test)

    Returns:
        tuple: (chi2, ndf)
    """
    # Clone histograms to avoid modifying originals
    h_obs_test = h_obs.Clone()
    h_exp_test = h_exp.Clone()

    if normalize:
        # Normalize both histograms to unit area
        obs_integral = h_obs_test.Integral()
        exp_integral = h_exp_test.Integral()

        if obs_integral > 0 and exp_integral > 0:
            h_obs_test.Scale(1.0 / obs_integral)
            h_exp_test.Scale(1.0 / exp_integral)

    chi2 = 0.0
    ndf = 0
    for bin in range(1, h_obs_test.GetNbinsX() + 1):
        obs_bin = h_obs_test.GetBinContent(bin)
        exp_bin = h_exp_test.GetBinContent(bin)
        obs_err = h_obs_test.GetBinError(bin)
        exp_err = h_exp_test.GetBinError(bin) * scale_factor

        if exp_bin > 0:
            sigma2 = obs_err**2 + exp_err**2
            if sigma2 > 0:
                chi2 += (obs_bin - exp_bin)**2 / sigma2
                ndf += 1

    return chi2, ndf

def calculate_chi2_root(h_obs, h_exp, normalize=True):
    """
    Calculate chi^2 test using ROOT's Chi2Test for weighted vs weighted histograms.

    In closure test, both histograms are MC (SR vs SB from TTLL_powheg),
    so we use "WW" option (weighted vs weighted).

    Note: ROOT's NORM option only works with UU (unweighted). For WW (weighted),
    we must manually normalize the histograms before the test.

    Args:
        h_obs: Observed histogram (MC from signal region)
        h_exp: Expected histogram (MC from sideband region)
        normalize: If True, perform shape-only test (normalize before comparison)

    Returns:
        tuple: (chi2, ndf, p_value)
    """
    h_obs_test = h_obs.Clone("h_obs_chi2_test")
    h_exp_test = h_exp.Clone("h_exp_chi2_test")
    h_obs_test.SetDirectory(0)
    h_exp_test.SetDirectory(0)

    # For shape-only test with weighted histograms, manually normalize
    # ROOT's NORM option only works with UU (unweighted), not WW (weighted)
    if normalize:
        obs_integral = h_obs_test.Integral()
        exp_integral = h_exp_test.Integral()
        if obs_integral > 0 and exp_integral > 0:
            h_obs_test.Scale(1.0 / obs_integral)
            h_exp_test.Scale(1.0 / exp_integral)

    options = "WW"  # weighted vs weighted (MC vs MC)

    p_value = h_obs_test.Chi2Test(h_exp_test, options)
    chi2 = h_obs_test.Chi2Test(h_exp_test, options + " CHI2")
    chi2_ndf = h_obs_test.Chi2Test(h_exp_test, options + " CHI2/NDF")
    ndf = int(round(chi2 / chi2_ndf)) if chi2_ndf > 0 else 0

    return chi2, ndf, p_value

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="Run1E2Mu / Run3Mu")
parser.add_argument("--histkey", required=True, type=str, help="histkey, e.g. Central/ZCand/mass")
parser.add_argument("--syst", default="Central", type=str, help="SB variation: Central, TT, bjet, cjet, ljet")
parser.add_argument("--rebin", default=5, type=int, help="rebin factor")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

# Validate syst argument
VALID_SYSTS = ["Central", "TT", "bjet", "cjet", "ljet"]
if args.syst not in VALID_SYSTS:
    raise ValueError(f"Invalid --syst value '{args.syst}'. Must be one of {VALID_SYSTS}")

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

WORKDIR = os.environ['WORKDIR']

with open("configs/histkeys.json") as f:
    config = json.load(f)[args.histkey]

# Handle merged eras
era_list = get_era_list(args.era)
logging.info(f"Processing {args.era} with eras: {era_list}")
logging.info(f"Using SB variation: {args.syst}")

config["era"] = args.era
config["CoM"] = get_CoM_energy(args.era)
config["rTitle"] = "Obs / Exp"
config["rRange"] = [0.0, 2.0]
config["systSrc"] = "error, stat"
if args.histkey == "nonprompt/eta" and args.channel == "Run3Mu":
    config["rebin"] = 4

# Get histograms from all eras and sum them
obs_hists = []
exp_hists = []

for era in era_list:
    file_path = f"{WORKDIR}/SKNanoOutput/ClosFakeRate/{args.channel}/{era}/Skim_TriLep_TTLL_powheg.root"
    
    if not os.path.exists(file_path):
        logging.warning(f"File not found: {file_path}")
        continue
    
    f = ROOT.TFile.Open(file_path)
    if not f or f.IsZombie():
        logging.warning(f"Cannot open file: {file_path}")
        if f: f.Close()
        continue
    
    if args.channel == "Run1E2Mu":
        h_obs_era = f.Get(f"SR1E2Mu/Central/{args.histkey}")
        h_exp_era = f.Get(f"SB1E2Mu/{args.syst}/{args.histkey}")
    elif args.channel == "Run3Mu":
        h_obs_era = f.Get(f"SR3Mu/Central/{args.histkey}")
        h_exp_era = f.Get(f"SB3Mu/{args.syst}/{args.histkey}")
    else:
        f.Close()
        raise KeyError(f"Wrong channel {args.channel}")
    
    if h_obs_era and h_exp_era:
        h_obs_era.SetDirectory(0)
        h_exp_era.SetDirectory(0)
        obs_hists.append(h_obs_era)
        exp_hists.append(h_exp_era)
        logging.debug(f"Loaded histograms from {era}")
    else:
        logging.warning(f"Cannot find histograms for {args.histkey} in {era}")
    
    f.Close()

# Sum histograms across eras
if not obs_hists or not exp_hists:
    raise RuntimeError(f"No valid histograms found for {args.histkey} in {args.channel}")

h_obs = obs_hists[0].Clone("observed_total")
h_exp = exp_hists[0].Clone("expected_total")
h_obs.SetDirectory(0)
h_exp.SetDirectory(0)

for h in obs_hists[1:]:
    h_obs.Add(h)
for h in exp_hists[1:]:
    h_exp.Add(h)

logging.info(f"Successfully merged histograms from {len(obs_hists)} eras")

# Create rebinned histograms for bin-by-bin uncertainty calculation (mass histograms only)
# For mass histograms, use adaptive rebinning based on fractional error threshold
# Keep original 1 GeV bins for chi-squared tests and plotting
h_obs_for_uncertainty = h_obs
h_exp_for_uncertainty = h_exp
mass_histkeys = ["pair/mass", "pair_lowM/mass", "pair_highM/mass"]
if any(mass_key in args.histkey for mass_key in mass_histkeys):
    logging.info(f"Mass histogram detected - applying adaptive rebinning for uncertainty calculation")
    logging.info(f"Rebinning criteria: max fractional error = 30%, min bin content = 5.0, mass range = [10, 160] GeV")

    # Apply adaptive rebinning based on fractional error threshold
    # This properly handles sumW² for MC histograms
    # First, derive bin edges from Expected histogram (reference statistics)
    bin_edges = get_adaptive_bin_edges(
        h_exp,
        max_frac_error=0.3,
        min_content=5.0,
        mass_range=(10.0, 160.0)
    )

    # Apply the SAME bin edges to both histograms
    h_obs_for_uncertainty = apply_variable_binning(h_obs, bin_edges, suffix="_obs_unc")
    h_exp_for_uncertainty = apply_variable_binning(h_exp, bin_edges, suffix="_exp_unc")

    logging.info(f"Original bins: {h_obs.GetNbinsX()}, Rebinned bins for uncertainty: {h_obs_for_uncertainty.GetNbinsX()}")

# Set systematic uncertainty to 30%
#for bin in range(1, h_exp.GetNbinsX()+1):
#    h_exp.SetBinError(bin, h_exp.GetBinContent(bin) * 0.30)

# Prepare histograms for plotting
h_obs.SetTitle("Observed")
exp_title = "Expected" if args.syst == "Central" else f"Expected ({args.syst})"
h_exp.SetTitle(exp_title)

obs = h_obs.Integral(0, h_obs.GetNbinsX()+1)
exp = h_exp.Integral(0, h_exp.GetNbinsX()+1)

# Calculate difference and store in JSON format
difference = (obs - exp) / exp if exp != 0 else float('inf')

# Calculate bin-by-bin fractional differences
max_bin_difference = 0.0
max_bin_difference_bin = -1
bin_differences = []  # Store all deviations for percentile calculation
bin_differences_filtered = []  # Store only bins passing stat threshold
snr_threshold = 3.0  # Signal-to-noise ratio threshold (content/error > 3)
mass_range_min = 10.0  # GeV
mass_range_max = 160.0  # GeV

# Statistics-filtered maximum (Option A)
max_bin_difference_filtered = 0.0
max_bin_difference_filtered_bin = -1

# Use rebinned histograms for bin-by-bin uncertainty calculation
for bin in range(1, h_obs_for_uncertainty.GetNbinsX() + 1):
    obs_bin = h_obs_for_uncertainty.GetBinContent(bin)
    exp_bin = h_exp_for_uncertainty.GetBinContent(bin)
    exp_err = h_exp_for_uncertainty.GetBinError(bin)
    bin_center = h_obs_for_uncertainty.GetBinCenter(bin)

    # Only consider bins within mass range
    if bin_center < mass_range_min or bin_center > mass_range_max:
        continue

    if exp_bin > 0:
        bin_diff = (obs_bin - exp_bin) / exp_bin
        abs_diff = abs(bin_diff)

        # Store all differences for percentile calculation
        bin_differences.append(abs_diff)

        # Track overall maximum (store absolute value for consistency with percentiles)
        if abs_diff > max_bin_difference:
            max_bin_difference = abs_diff
            max_bin_difference_bin = bin

        # Track filtered maximum and differences (only bins with good signal-to-noise ratio)
        # Check SNR for both Observed and Expected
        snr_exp = exp_bin / exp_err if exp_err > 0 else 0.0
        
        obs_err = h_obs_for_uncertainty.GetBinError(bin)
        snr_obs = obs_bin / obs_err if obs_err > 0 else 0.0
        
        if snr_exp > snr_threshold and snr_obs > snr_threshold:
            bin_differences_filtered.append(abs_diff)
            if abs_diff > max_bin_difference_filtered:
                max_bin_difference_filtered = abs_diff
                max_bin_difference_filtered_bin = bin

# Calculate percentile-based deviations (Option C)
if bin_differences_filtered:
    percentile_68_difference = np.percentile(bin_differences_filtered, 68)
    percentile_95_difference = np.percentile(bin_differences_filtered, 95)
    rms_bin_difference = np.sqrt(np.mean(np.array(bin_differences_filtered)**2))
else:
    percentile_68_difference = 0.0
    percentile_95_difference = 0.0
    rms_bin_difference = 0.0

# Calculate shape-only chi-squared test (normalized histograms)
# First, rebin to ensure chi-squared validity (expected >= 5 per bin)
logging.info("Applying chi-squared validity rebinning (min expected = 5.0)")
h_obs_chi2, h_exp_chi2 = rebin_for_chi2_validity(h_obs, h_exp, min_expected=5.0)
logging.info(f"Chi-squared test bins: {h_obs.GetNbinsX()} → {h_obs_chi2.GetNbinsX()}")

# Calculate shape-only chi2 using ROOT's Chi2Test with "WW NORM" options
# WW = weighted vs weighted (both histograms are MC)
# NORM = normalize before comparison (shape-only test)
chi2_shape, ndf_shape, p_value_shape = calculate_chi2_root(h_obs_chi2, h_exp_chi2, normalize=True)

# Find optimal uncertainty scale factor for shape-only closure
# Use chi2-rebinned histograms for this optimization as well
best_scale_shape = 1.0
best_diff_shape = abs(chi2_shape/ndf_shape - 1.0) if ndf_shape > 0 else float('inf')

for scale in [i * 0.05 for i in range(2, 101)]:  # 0.1 to 5.0 in steps of 0.05
    chi2_scaled, ndf_scaled = calculate_chi2_with_scale(h_obs_chi2, h_exp_chi2, scale, normalize=True)
    if ndf_scaled > 0:
        chi2_per_ndf = chi2_scaled / ndf_scaled
        diff = abs(chi2_per_ndf - 1.0)
        if diff < best_diff_shape:
            best_diff_shape = diff
            best_scale_shape = scale

recommended_systematic_shape = abs(best_scale_shape - 1.0) * 100  # in percent

results = {
    "syst": args.syst,
    "observed": obs,
    "expected": exp,
    "difference": difference,
    "max_bin_difference": max_bin_difference,
    "max_bin_difference_bin": max_bin_difference_bin,
    "max_bin_difference_filtered": max_bin_difference_filtered,
    "max_bin_difference_filtered_bin": max_bin_difference_filtered_bin,
    "snr_threshold": snr_threshold,
    "mass_range_min": mass_range_min,
    "mass_range_max": mass_range_max,
    "percentile_68_difference": percentile_68_difference,
    "percentile_95_difference": percentile_95_difference,
    "rms_bin_difference": rms_bin_difference,
    "chi2": chi2_shape,
    "ndf": ndf_shape,
    "chi2_per_ndf": chi2_shape/ndf_shape if ndf_shape > 0 else 0.0,
    "p_value": p_value_shape,
    "closure_uncertainty_scale": best_scale_shape,
    "recommended_systematic_pct": recommended_systematic_shape,
}

# Save results to JSON file
variable_name = args.histkey.replace('/', '_').lower()
json_output_path = f"{WORKDIR}/MeasFakeRateV3/plots/{args.era}/{args.channel}/{args.syst}/closure_{variable_name}_yield.json"
os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
with open(json_output_path, 'w') as json_file:
    json.dump(results, json_file, indent=2)

# Create background dictionary for ComparisonCanvas
# Expected (fake rate prediction) is the background
BKGs = {"Expected": h_exp}

# Plot configuration (already set above)

# Create output directory and filename
output_path = f"{WORKDIR}/MeasFakeRateV3/plots/{args.era}/{args.channel}/{args.syst}/closure_{variable_name}.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Create and draw the comparison plot
plotter = ComparisonCanvas(h_obs, BKGs, config)
plotter.drawPadUp()

# Add chi-squared test results to the plot
plotter.canv.cd(1)
chi2_text = f"#chi^{{2}}/ndf = {chi2_shape/ndf_shape:.2f} (p = {p_value_shape:.2f})"
# Need to import CMS for drawing text
import cmsstyle as CMS
CMS.drawText(chi2_text, posX=0.20, posY=0.62, font=42, align=0, size=0.04)

plotter.drawPadDown()
plotter.canv.SaveAs(output_path)

logging.info(f"Closure plot saved to: {output_path}")
logging.info(f"Chi2/ndf: {chi2_shape:.2f}/{ndf_shape} = {chi2_shape/ndf_shape:.2f}, p-value = {p_value_shape:.3f}")
logging.info(f"Recommended systematic: {recommended_systematic_shape:.1f}%")
logging.info(f"")
logging.info(f"Bin-by-bin deviation metrics (mass range: {mass_range_min}-{mass_range_max} GeV):")
logging.info(f"  Max bin difference (all bins): {max_bin_difference:.3f} (bin {max_bin_difference_bin})")
logging.info(f"  Max bin difference (SNR>{snr_threshold}): {max_bin_difference_filtered:.3f} (bin {max_bin_difference_filtered_bin})")
logging.info(f"  68th percentile: {percentile_68_difference:.3f}")
logging.info(f"  95th percentile: {percentile_95_difference:.3f}")
logging.info(f"  RMS deviation: {rms_bin_difference:.3f}")