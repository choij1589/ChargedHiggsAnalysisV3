#!/bin/bash
#ERAs=("2016preVFP" "2016postVFP" "2017" "2018" "2022" "2022EE" "2023" "2023BPix" "Run2" "Run3")
ERAs=("Run2" "Run3")
CHANNELs=("DIMU", "EMU")

for ERA in "${ERAs[@]}"; do
  for CHANNEL in "${CHANNELs[@]}"; do
    echo "Processing ERA: $ERA, CHANNEL: $CHANNEL"
    ./scripts/drawPlots.sh $ERA $CHANNEL
  done
done
