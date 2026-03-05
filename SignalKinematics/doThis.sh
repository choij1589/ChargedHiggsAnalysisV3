#!/bin/bash
set -euo pipefail
export PATH="${PWD}/python:${PATH}"

ERAs=("2016preVFP" "2016postVFP" "2017" "2018" "Run2" "2022" "2022EE" "2023" "2023BPix" "Run3")
CHANNELs=("SR3Mu" "SR1E2Mu")

for ERA in "${ERAs[@]}"; do
  for CHANNEL in "${CHANNELs[@]}"; do
    bash scripts/drawPlots.sh "$ERA" "$CHANNEL"
  done
done
