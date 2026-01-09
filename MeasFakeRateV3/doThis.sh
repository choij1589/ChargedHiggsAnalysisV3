#!/bin/bash
ERAs=("2016preVFP" "2016postVFP" "2017" "2018" "2022" "2022EE" "2023" "2023BPix" "Run2" "Run3")
MEASUREs=("electron" "muon")

for era in "${ERAs[@]}"; do
    for measure in "${MEASUREs[@]}"; do
        ./scripts/measFakeRate.sh $era $measure
        ./scripts/plotClosure.sh $era
    done
done
