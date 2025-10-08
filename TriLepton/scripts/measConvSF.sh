#!/bin/bash
ERAs=("2016preVFP" "2016postVFP" "2017" "2018" "2022" "2022EE" "2023" "2023BPix")
CHANNELs=("ZG1E2Mu" "ZG3Mu")

export PATH="$PWD/python:$PATH"
for ERA in "${ERAs[@]}"; do
    for CHANNEL in "${CHANNELs[@]}"; do
        echo "Processing ERA: $ERA, CHANNEL: $CHANNEL"
        measConvSF.py --era $ERA --channel $CHANNEL
    done
done