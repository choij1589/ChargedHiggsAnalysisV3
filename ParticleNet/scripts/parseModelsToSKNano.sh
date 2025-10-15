#!/bin/bash
MASSPOINTs=(MHc130_MA90 MHc160_MA85 MHc100_MA95 MHc115_MA87 MHc145_MA92 MHc160_MA98)
CHANNELs=(Run1E2Mu Run3Mu)


for MASSPOINT in ${MASSPOINTs[@]}; do
    for CHANNEL in ${CHANNELs[@]}; do
        echo $MASSPOINT $CHANNEL
        TO="../SKNanoAnalyzer/data/Run3_v13_Run2_v9/Run2/${CHANNEL}/Classifiers/ParticleNet/${MASSPOINT}/fold-3/models"
        mkdir -p $TO
        cp results_bjets/${CHANNEL}/multiclass/TTToHcToWAToMuMu-${MASSPOINT}/fold-3/models/ParticleNet-*3bg.pt $TO/ParticleNet.pt
    done
done


