#!/usr/bin/env python
import os
import logging
import argparse
import ROOT
import json
import numpy as np
from plotter import ComparisonCanvas, get_era_list, get_CoM_energy

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

def calculate_chi2_with_syst(h_obs, h_exp, syst_frac):
    """
    Calculate chi-squared on absolute yields with flat normalization systematic
    added in quadrature to the statistical uncertainty on h_exp.

    Args:
        h_obs: Observed histogram
        h_exp: Expected histogram
        syst_frac: Fractional systematic uncertainty (e.g. 0.20 for 20%)

    Returns:
        tuple: (chi2, ndf)
    """
    chi2 = 0.0
    ndf = 0
    for bin in range(1, h_obs.GetNbinsX() + 1):
        obs_bin = h_obs.GetBinContent(bin)
        exp_bin = h_exp.GetBinContent(bin)
        obs_err = h_obs.GetBinError(bin)
        exp_err = h_exp.GetBinError(bin)

        if exp_bin > 0:
            sigma2 = obs_err**2 + exp_err**2 + (syst_frac * exp_bin)**2
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

# Prepare histograms for plotting
h_obs.SetTitle("Observed")
exp_title = "Expected" if args.syst == "Central" else f"Expected ({args.syst})"
h_exp.SetTitle(exp_title)

obs = h_obs.Integral(0, h_obs.GetNbinsX()+1)
exp = h_exp.Integral(0, h_exp.GetNbinsX()+1)

# Calculate overall rate difference
difference = (obs - exp) / exp if exp != 0 else float('inf')

# Chi2 calculation with systematic uncertainty.
# For individual eras: scan a flat syst in 5% steps, pick the level closest to
# chi2/ndf = 1.
# For Run2/Run3 combined: use per-era systematics from FakeNorm.json applied
# era-by-era, then sum chi2 and ndf across eras.
COMBINED_ERAS = ("Run2", "Run3")

if args.era in COMBINED_ERAS:
    fake_norm_path = f"{WORKDIR}/Common/Data/FakeNorm.json"
    with open(fake_norm_path) as f:
        fake_norm = json.load(f)

    chi2_total, ndf_total = 0.0, 0
    era_chi2_breakdown = {}
    for era, h_obs_era, h_exp_era in zip(era_list, obs_hists, exp_hists):
        syst_frac = fake_norm[args.channel][era]
        chi2_era, ndf_era = calculate_chi2_with_syst(h_obs_era, h_exp_era, syst_frac)
        chi2_total += chi2_era
        ndf_total  += ndf_era
        era_chi2_breakdown[era] = {
            "syst_pct": round(syst_frac * 100),
            "chi2": chi2_era,
            "ndf": ndf_era,
            "chi2_per_ndf": chi2_era / ndf_era if ndf_era > 0 else float('inf'),
        }
        logging.info(f"  {era}: syst={syst_frac*100:.0f}%  chi2/ndf={chi2_era/ndf_era:.3f}" if ndf_era > 0 else f"  {era}: ndf=0")

    chi2_profile = []  # not applicable for combined eras
    recommended_systematic_pct = None  # per-era, stored in era_chi2_breakdown
    rec_chi2_per_ndf = chi2_total / ndf_total if ndf_total > 0 else float('inf')
    rec_p_value      = ROOT.TMath.Prob(chi2_total, ndf_total)

else:
    era_chi2_breakdown = {}
    # Scan flat normalization systematic in 5% steps using the original 1 GeV bins.
    # chi2 is computed on absolute yields (no normalization) because we assign a
    # rate (lnN) uncertainty.  The systematic term (syst_frac × exp_bin)² is added
    # in quadrature to the statistical uncertainties of both histograms, which keeps
    # even sparse bins well-behaved without requiring a minimum-expected rebinning.
    # Recommended uncertainty = syst_frac closest to chi2/ndf = 1.
    syst_levels = [i * 0.05 for i in range(0, 21)]  # 0%, 5%, ..., 100%
    chi2_profile = []
    for syst_frac in syst_levels:
        chi2_s, ndf_s = calculate_chi2_with_syst(h_obs, h_exp, syst_frac)
        chi2_profile.append({
            "syst_pct": round(syst_frac * 100),
            "chi2": chi2_s,
            "ndf": ndf_s,
            "chi2_per_ndf": chi2_s / ndf_s if ndf_s > 0 else float('inf'),
        })

    recommended_systematic_pct = min(
        chi2_profile, key=lambda e: abs(e["chi2_per_ndf"] - 1.0)
    )["syst_pct"]
    rec_entry        = next(e for e in chi2_profile if e["syst_pct"] == recommended_systematic_pct)
    rec_chi2_per_ndf = rec_entry["chi2_per_ndf"]
    rec_p_value      = ROOT.TMath.Prob(rec_entry["chi2"], rec_entry["ndf"])

# Reference chi2 via ROOT Chi2Test (needs expected >= 5 per bin for valid chi2 dist.)
h_obs_chi2, h_exp_chi2 = rebin_for_chi2_validity(h_obs, h_exp, min_expected=5.0)
logging.info(f"Chi-squared reference bins: {h_obs.GetNbinsX()} → {h_obs_chi2.GetNbinsX()}")
chi2_rate, ndf_rate, p_value_rate = calculate_chi2_root(h_obs_chi2, h_exp_chi2, normalize=False)
chi2_shape, ndf_shape, p_value_shape = calculate_chi2_root(h_obs_chi2, h_exp_chi2, normalize=True)

results = {
    "syst": args.syst,
    "observed": obs,
    "expected": exp,
    "difference": difference,
    "chi2_rate": chi2_rate,
    "ndf_rate": ndf_rate,
    "chi2_per_ndf_rate": chi2_rate / ndf_rate if ndf_rate > 0 else 0.0,
    "p_value_rate": p_value_rate,
    "chi2_shape": chi2_shape,
    "ndf_shape": ndf_shape,
    "chi2_per_ndf_shape": chi2_shape / ndf_shape if ndf_shape > 0 else 0.0,
    "p_value_shape": p_value_shape,
    "recommended_systematic_pct": recommended_systematic_pct,
    "chi2_profile": chi2_profile,
    "era_chi2_breakdown": era_chi2_breakdown,
}

# Save results to JSON file
variable_name = args.histkey.replace('/', '_').lower()
json_output_path = f"{WORKDIR}/MeasFakeRateV4/plots/{args.era}/{args.channel}/{args.syst}/closure_{variable_name}_yield.json"
os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
with open(json_output_path, 'w') as json_file:
    json.dump(results, json_file, indent=2)

# Create background dictionary for ComparisonCanvas
BKGs = {"Expected": h_exp}

# Create output directory and filename
output_path = f"{WORKDIR}/MeasFakeRateV4/plots/{args.era}/{args.channel}/{args.syst}/closure_{variable_name}.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Create and draw the comparison plot
plotter = ComparisonCanvas(h_obs, BKGs, config)
plotter.drawPadUp()

# Add chi-squared test result to the plot.
plotter.canv.cd(1)
import cmsstyle as CMS
if args.era in COMBINED_ERAS:
    rate_text = f"Rate: #chi^{{2}}/ndf = {rec_chi2_per_ndf:.2f} (p = {rec_p_value:.2f})"
else:
    rate_text = f"Rate: #chi^{{2}}/ndf = {rec_chi2_per_ndf:.2f} (p = {rec_p_value:.2f}), syst = {recommended_systematic_pct}%"
shape_text = f"Shape: #chi^{{2}}/ndf = {chi2_shape/ndf_shape:.2f} (p = {p_value_shape:.2f})"
CMS.drawText(rate_text,  posX=0.20, posY=0.62, font=42, align=0, size=0.04)
CMS.drawText(shape_text, posX=0.20, posY=0.57, font=42, align=0, size=0.04)

plotter.drawPadDown()
plotter.canv.SaveAs(output_path)

logging.info(f"Closure plot saved to: {output_path}")
logging.info(f"Chi2/ndf rate  (stat only): {chi2_rate:.2f}/{ndf_rate} = {chi2_rate/ndf_rate:.2f}, p-value = {p_value_rate:.3f}")
logging.info(f"Chi2/ndf shape (stat only): {chi2_shape:.2f}/{ndf_shape} = {chi2_shape/ndf_shape:.2f}, p-value = {p_value_shape:.3f}")
if args.era in COMBINED_ERAS:
    logging.info(f"Rate chi2/ndf (per-era syst): {rec_chi2_per_ndf:.3f}, p-value = {rec_p_value:.3f}")
    for era, breakdown in era_chi2_breakdown.items():
        logging.info(f"  {era}: syst={breakdown['syst_pct']}%  chi2/ndf={breakdown['chi2_per_ndf']:.3f}")
else:
    logging.info(f"Recommended systematic: {recommended_systematic_pct}%")
    logging.info(f"Chi2/ndf profile:")
    for entry in chi2_profile:
        logging.info(f"  syst = {entry['syst_pct']:3d}%  chi2/ndf = {entry['chi2_per_ndf']:.3f}")

    # Plot chi2/ndf vs systematic uncertainty profile (individual eras only)
    profile_syst   = np.array([e["syst_pct"] for e in chi2_profile], dtype=float)
    profile_chi2ndf = np.array([min(e["chi2_per_ndf"], 10.0) for e in chi2_profile], dtype=float)

    g_profile = ROOT.TGraph(len(profile_syst), profile_syst, profile_chi2ndf)
    g_profile.SetTitle(";Systematic uncertainty [%];#chi^{2}/ndf")
    g_profile.SetLineColor(ROOT.kBlue + 1)
    g_profile.SetLineWidth(2)
    g_profile.SetMarkerColor(ROOT.kBlue + 1)
    g_profile.SetMarkerStyle(20)
    g_profile.SetMarkerSize(0.8)

    line_one = ROOT.TLine(profile_syst[0], 1.0, profile_syst[-1], 1.0)
    line_one.SetLineColor(ROOT.kRed)
    line_one.SetLineWidth(2)
    line_one.SetLineStyle(2)

    line_rec = ROOT.TLine(recommended_systematic_pct, 0.0, recommended_systematic_pct, 1.0)
    line_rec.SetLineColor(ROOT.kGreen + 2)
    line_rec.SetLineWidth(2)
    line_rec.SetLineStyle(3)

    canv_profile = ROOT.TCanvas("canv_profile", "", 600, 500)
    canv_profile.SetLeftMargin(0.15)
    canv_profile.SetBottomMargin(0.15)

    g_profile.GetYaxis().SetRangeUser(0.0, min(profile_chi2ndf[0] * 1.2, 10.0))
    g_profile.Draw("ALP")
    line_one.Draw("same")
    line_rec.Draw("same")

    latex = ROOT.TLatex()
    latex.SetNDC(False)
    latex.SetTextSize(0.035)
    latex.SetTextColor(ROOT.kGreen + 2)
    latex.DrawLatex(recommended_systematic_pct + 1.0, 0.15, f"Rec. = {recommended_systematic_pct}%")

    profile_path = output_path.replace(".png", "_chi2profile.png")
    canv_profile.SaveAs(profile_path)
    logging.info(f"Chi2 profile plot saved to: {profile_path}")
