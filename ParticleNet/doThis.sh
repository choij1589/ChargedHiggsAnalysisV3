#!/bin/bash
MASSPOINTs=("MHc100_MA95" "MHc130_MA90" "MHc160_MA85" "MHc115_MA87" "MHc145_MA92" "MHc160_MA98")
CHANNELs=("Run1E2Mu" "Run3Mu")

for MASSPOINT in "${MASSPOINTs[@]}"; do
  for CHANNEL in "${CHANNELs[@]}"; do
    echo "Launching GA optimization for masspoint: ${MASSPOINT}, channel: ${CHANNEL}"
    ./scripts/launchGAOptim.sh ${MASSPOINT} ${CHANNEL} cuda:0
  done
done
