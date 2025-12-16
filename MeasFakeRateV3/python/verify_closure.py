#!/usr/bin/env python3
import os
import ROOT
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Hardcoded configuration for debugging
WORKDIR = "/home/choij/Sync/workspace/ChargedHiggsAnalysisV3"
ERA = "Run2"
CHANNEL = "Run1E2Mu"
HISTKEY = "pair/mass"
ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]

# Import the modified functions from plotClosure
import sys
sys.path.append(f"{WORKDIR}/MeasFakeRateV3/python")
from plotClosure import get_adaptive_bin_edges, apply_variable_binning

# Load and Sum
obs_hists = []
exp_hists = []
for era in ERAS:
    file_path = f"{WORKDIR}/SKNanoOutput/ClosFakeRate/{CHANNEL}/{era}/Skim_TriLep_TTLL_powheg.root"
    f = ROOT.TFile.Open(file_path)
    if not f: continue
    h_obs_era = f.Get(f"SR1E2Mu/Central/{HISTKEY}")
    h_exp_era = f.Get(f"SB1E2Mu/Central/{HISTKEY}")
    if h_obs_era and h_exp_era:
        h_obs_era.SetDirectory(0)
        h_exp_era.SetDirectory(0)
        obs_hists.append(h_obs_era)
        exp_hists.append(h_exp_era)
    f.Close()

h_obs = obs_hists[0].Clone("observed_total")
h_exp = exp_hists[0].Clone("expected_total")
for h in obs_hists[1:]: h_obs.Add(h)
for h in exp_hists[1:]: h_exp.Add(h)

# Test New Logic
logging.info("Testing new binning logic...")

# 1. Derive edges from Exp
bin_edges = get_adaptive_bin_edges(h_exp, max_frac_error=0.3, min_content=5.0)
logging.info(f"Derived {len(bin_edges)-1} bins from Expected")

# 2. Apply to both
h_obs_unc = apply_variable_binning(h_obs, bin_edges, "_obs")
h_exp_unc = apply_variable_binning(h_exp, bin_edges, "_exp")

# 3. Verify Consistency
if h_obs_unc.GetNbinsX() != h_exp_unc.GetNbinsX():
    logging.error("Bin count mismatch!")
else:
    logging.info("Bin count matches.")

# 4. Check SNR Logic
logging.info("\nChecking SNR filtering (Threshold > 3.0 for BOTH)...")
print(f"{'Bin':<5} {'Center':<10} {'Exp':<10} {'ExpErr':<10} {'SNR_Exp':<10} {'Obs':<10} {'ObsErr':<10} {'SNR_Obs':<10} {'Accepted?':<10}")

count_accepted = 0
for bin in range(1, h_obs_unc.GetNbinsX() + 1):
    center = h_obs_unc.GetBinCenter(bin)
    if center < 10.0 or center > 160.0: continue
    
    exp = h_exp_unc.GetBinContent(bin)
    exp_err = h_exp_unc.GetBinError(bin)
    obs = h_obs_unc.GetBinContent(bin)
    obs_err = h_obs_unc.GetBinError(bin)
    
    snr_exp = exp / exp_err if exp_err > 0 else 0
    snr_obs = obs / obs_err if obs_err > 0 else 0
    
    accepted = (snr_exp > 3.0 and snr_obs > 3.0)
    if accepted: count_accepted += 1
    
    if accepted or (snr_exp > 3.0 and snr_obs <= 3.0): # Show edge cases
        print(f"{bin:<5} {center:<10.1f} {exp:<10.1f} {exp_err:<10.1f} {snr_exp:<10.1f} {obs:<10.1f} {obs_err:<10.1f} {snr_obs:<10.1f} {str(accepted):<10}")

logging.info(f"\nTotal accepted bins: {count_accepted}")
