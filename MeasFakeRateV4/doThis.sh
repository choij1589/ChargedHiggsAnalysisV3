#!/bin/bash
ERAs=("2016preVFP" "2016postVFP" "2017" "2018" "2022" "2022EE" "2023" "2023BPix")
MEASUREs=("electron" "muon")
#for era in "${ERAs[@]}"; do
#    for measure in "${MEASUREs[@]}"; do
#       ./scripts/measFakeRate.sh $era $measure
#    done
#done

ERAs=("2016preVFP" "2016postVFP" "2017" "2018" "2022" "2022EE" "2023" "2023BPix" "Run2" "Run3")
for era in "${ERAs[@]}"; do
    ./scripts/plotClosure.sh $era
done
