#!/bin/bash
MASSPOINTs=(MHc130_MA90 MHc160_MA85 MHc100_MA95)
CHANNELs=(Run1E2Mu Run3Mu)

for MASSPOINT in ${MASSPOINTs[@]}; do
    for CHANNEL in ${CHANNELs[@]}; do
        echo $MASSPOINT $CHANNEL
        mkdir -p ../SKNanoAnalyzer/data/Run3_v13_Run2_v9/Run2/${CHANNEL}/Classifiers/ParticleNet
        cp results_bjets/${CHANNEL}/multiclass/TTToHcToWAToMuMu-${MASSPOINT}/fold-3/models/ParticleNet-*_scripted.pt ../SKNanoAnalyzer/data/Run3_v13_Run2_v9/Run2/${CHANNEL}/Classifiers/ParticleNet/${MASSPOINT}_scripted.pt
    done
done


