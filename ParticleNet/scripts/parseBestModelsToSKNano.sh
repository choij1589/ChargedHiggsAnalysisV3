#!/bin/bash

# Check if input is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <INPUT_DIRECTORY>"
    echo "Example: $0 /path/to/experiment/results"
    exit 1
fi

INPUT=$1

# Check if input directory exists
if [ ! -d "$INPUT" ]; then
    echo "Error: Directory '$INPUT' does not exist"
    echo "Usage: $0 <INPUT_DIRECTORY>"
    exit 1
fi

MASSPOINTs=(MHc130_MA90 MHc130_MA100 MHc160_MA85 MHc100_MA95 MHc115_MA87 MHc145_MA92 MHc160_MA98)

for MASSPOINT in ${MASSPOINTs[@]}; do
    echo $MASSPOINT
    TO="../SKNanoAnalyzer/data/Run3_v13_Run2_v9/Run2/Combined/Classifiers/ParticleNet/${MASSPOINT}/"
    mkdir -p $TO
    cp -r  "$INPUT/Combined/multiclass/TTToHcToWAToMuMu-${MASSPOINT}/best_model" $TO/
done