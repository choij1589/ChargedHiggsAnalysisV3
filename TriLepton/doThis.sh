#!/bin/bash
ERAsRun2=("2016preVFP" "2016postVFP" "2017" "2018" "Run2")
ERAsRun3=("2022" "2022EE" "2023" "2023BPix" "Run3")
CHANNELsRun2=("ZFake1E2Mu" "ZFake3Mu" "ZG1E2Mu" "ZG3Mu" "SR1E2Mu" "SR3Mu" "TTZ2E1Mu")
CHANNELsRun3=("ZFake1E2Mu" "ZFake3Mu" "ZG1E2Mu" "ZG3Mu" "WZ1E2Mu" "WZ3Mu" "SR1E2Mu" "SR3Mu" "TTZ2E1Mu")

for ERA in "${ERAsRun2[@]}"; do
  for CHANNEL in "${CHANNELsRun2[@]}"; do
    echo "Processing ERA: $ERA, CHANNEL: $CHANNEL"
    ./scripts/drawPlots.sh $ERA $CHANNEL
  done
done

for ERA in "${ERAsRun3[@]}"; do
  for CHANNEL in "${CHANNELsRun3[@]}"; do
    echo "Processing ERA: $ERA, CHANNEL: $CHANNEL"
    ./scripts/drawPlots.sh $ERA $CHANNEL
  done
done
