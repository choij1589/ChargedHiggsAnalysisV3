#!/bin/bash
ERAs=("2016preVFP" "2016postVFP" "2017" "2018")
VARs=("mass" "pt" "eta" "deltaR" "deltaEta" "deltaPhi" "MT_pair_MET" "MT_muSS_MET"
      "ptAsymmetry" "scalarPtSum" "gammaFactor" "gammaAcop" "deltaR_pair_mu3rd"
      "deltaPhi_pair_mu3rd" "deltaEta_pair_mu3rd" "ptRatio_mu3rd"
      "deltaPhi_pair_MET" "MT_asymmetry" "mu1_iso" "mu2_iso")

plot_kinematic() {
    local era=$1
    local variable=$2
    plotOSMuonSelection.py --era ${era} --channel SR3Mu --variable ${variable}
}
export PATH=$PWD/python:$PATH
export -f plot_kinematic
parallel -j 12 plot_kinematic ::: "${ERAs[@]}" ::: "${VARs[@]}"