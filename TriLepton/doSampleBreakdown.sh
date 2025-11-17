#!/bin/bash
SRs=("SR1E2Mu" "SR3Mu")
CRs=("ZFake1E2Mu" "ZFake3Mu" "ZG1E2Mu" "ZG3Mu" "WZ1E2Mu" "WZ3Mu")
ERAs=("2016preVFP" "2016postVFP" "2017" "2018" "2022" "2022EE" "2023" "2023BPix" "Run2" "Run3")

export PATH=$PWD/python:$PATH
for ERA in "${ERAs[@]}"; do
  for REGION in "${SRs[@]}"; do
    echo "Processing ERA: $ERA, REGION: $REGION"
    sampleBreakdown.py --era $ERA --channel $REGION --blind
    sampleBreakdown.py --era $ERA --channel $REGION --blind --onZ
  done
  for REGION in "${CRs[@]}"; do
    echo "Processing ERA: $ERA, REGION: $REGION"
    sampleBreakdown.py --era $ERA --channel $REGION
  done
done

