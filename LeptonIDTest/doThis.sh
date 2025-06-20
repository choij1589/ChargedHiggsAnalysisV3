#!/bin/bash
ERAs=("2016preVFP" "2016postVFP" "2017" "2018" "2022" "2022EE" "2023" "2023BPix")

# Fill histograms
./scripts/fillHists.sh

# Plot histograms
for ERA in ${ERAs[@]}; do
    ./scripts/plot.sh $ERA
done
