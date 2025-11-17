#!/usr/bin/env python
import os
import logging
import argparse
import ROOT
import json
from plotter import ComparisonCanvas, get_era_list, get_CoM_energy

def calculate_chi2_with_scale(h_obs, h_exp, scale_factor):
    """
    Calculate chi-squared test with scaled uncertainties on h_exp.

    Args:
        h_obs: Observed histogram
        h_exp: Expected histogram
        scale_factor: Scale factor to apply to h_exp uncertainties

    Returns:
        tuple: (chi2, ndf)
    """
    chi2 = 0.0
    ndf = 0
    for bin in range(1, h_obs.GetNbinsX() + 1):
        obs_bin = h_obs.GetBinContent(bin)
        exp_bin = h_exp.GetBinContent(bin)
        obs_err = h_obs.GetBinError(bin)
        exp_err = h_exp.GetBinError(bin) * scale_factor

        if exp_bin > 0:
            sigma2 = obs_err**2 + exp_err**2
            if sigma2 > 0:
                chi2 += (obs_bin - exp_bin)**2 / sigma2
                ndf += 1

    return chi2, ndf

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="Run1E2Mu / Run3Mu")
parser.add_argument("--histkey", required=True, type=str, help="histkey, e.g. Central/ZCand/mass")
parser.add_argument("--rebin", default=5, type=int, help="rebin factor")
parser.add_argument("--debug", default=False, action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

WORKDIR = os.environ['WORKDIR']

with open("configs/histkeys.json") as f:
    config = json.load(f)[args.histkey]

# Handle merged eras
era_list = get_era_list(args.era)
logging.info(f"Processing {args.era} with eras: {era_list}")

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
        h_exp_era = f.Get(f"SB1E2Mu/Central/{args.histkey}")
    elif args.channel == "Run3Mu":
        h_obs_era = f.Get(f"SR3Mu/Central/{args.histkey}")
        h_exp_era = f.Get(f"SB3Mu/Central/{args.histkey}")
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

# Set systematic uncertainty to 30%
#for bin in range(1, h_exp.GetNbinsX()+1):
#    h_exp.SetBinError(bin, h_exp.GetBinContent(bin) * 0.30)

# Prepare histograms for plotting
h_obs.SetTitle("Observed")
h_exp.SetTitle("Expected")

obs = h_obs.Integral(0, h_obs.GetNbinsX()+1)
exp = h_exp.Integral(0, h_exp.GetNbinsX()+1)

# Calculate difference and store in JSON format
difference = (obs - exp) / exp if exp != 0 else float('inf')

# Calculate chi-squared test
chi2 = 0.0
ndf = 0
for bin in range(1, h_obs.GetNbinsX() + 1):
    obs_bin = h_obs.GetBinContent(bin)
    exp_bin = h_exp.GetBinContent(bin)
    obs_err = h_obs.GetBinError(bin)
    exp_err = h_exp.GetBinError(bin)

    # Only include bins with non-zero expected value
    if exp_bin > 0:
        # Combined uncertainty: σ² = σ_obs² + σ_exp²
        sigma2 = obs_err**2 + exp_err**2
        if sigma2 > 0:
            chi2 += (obs_bin - exp_bin)**2 / sigma2
            ndf += 1

# Subtract 1 for degrees of freedom (no fit parameters, but general convention)
# Actually, for a simple comparison without fitting, ndf = number of bins used
# Keep ndf as is (number of bins)

# Calculate p-value using ROOT's TMath
p_value = ROOT.TMath.Prob(chi2, ndf) if ndf > 0 else 0.0

# Find optimal uncertainty scale factor for closure
# Search for scale that gives chi2/ndf closest to 1.0
best_scale = 1.0
best_diff = abs(chi2/ndf - 1.0) if ndf > 0 else float('inf')

for scale in [i * 0.05 for i in range(2, 101)]:  # 0.1 to 5.0 in steps of 0.05
    chi2_scaled, ndf_scaled = calculate_chi2_with_scale(h_obs, h_exp, scale)
    if ndf_scaled > 0:
        chi2_per_ndf = chi2_scaled / ndf_scaled
        diff = abs(chi2_per_ndf - 1.0)
        if diff < best_diff:
            best_diff = diff
            best_scale = scale

# Calculate recommended systematic uncertainty
# This is the additional uncertainty needed for closure
recommended_systematic = abs(best_scale - 1.0) * 100  # in percent

results = {
    "observed": obs,
    "expected": exp,
    "difference": difference,
    "chi2": chi2,
    "ndf": ndf,
    "chi2_per_ndf": chi2/ndf if ndf > 0 else 0.0,
    "p_value": p_value,
    "closure_uncertainty_scale": best_scale,
    "recommended_systematic_pct": recommended_systematic,
}

# Save results to JSON file
json_output_path = f"{WORKDIR}/MeasFakeRateV3/plots/{args.era}/{args.channel}/closure_yield.json"
os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
with open(json_output_path, 'w') as json_file:
    json.dump(results, json_file, indent=2)

# Create background dictionary for ComparisonCanvas
# Expected (fake rate prediction) is the background
BKGs = {"Expected": h_exp}

# Plot configuration (already set above)

# Create output directory and filename
output_path = f"{WORKDIR}/MeasFakeRateV3/plots/{args.era}/{args.channel}/closure_{args.histkey.replace('/', '_').lower()}.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Create and draw the comparison plot
plotter = ComparisonCanvas(h_obs, BKGs, config)
plotter.drawPadUp()

# Add chi-squared test results to the plot
plotter.canv.cd(1)
chi2_text = f"#chi^{{2}}/ndf = {chi2:.2f}/{ndf}"
syst_text = f"Closure syst: #pm{recommended_systematic:.1f}%"
# Need to import CMS for drawing text
import cmsstyle as CMS
CMS.drawText(chi2_text, posX=0.20, posY=0.62, font=42, align=0, size=0.04)
#CMS.drawText(syst_text, posX=0.20, posY=0.57, font=42, align=0, size=0.04)

plotter.drawPadDown()
plotter.canv.SaveAs(output_path)

logging.info(f"Closure plot saved to: {output_path}")