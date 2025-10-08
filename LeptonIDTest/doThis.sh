#!/bin/bash
ERAs=("2022" "2022EE" "2023" "2023BPix")

# Fill histograms
./scripts/fillHists.sh
./scripts/optimize.sh

# Plot histograms
for ERA in ${ERAs[@]}; do
    ./scripts/plot.sh $ERA
done
