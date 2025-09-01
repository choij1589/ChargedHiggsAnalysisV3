#!/bin/bash
export PATH=$PWD/python:$PATH

# Run3 eras for optimization
RUN2_ERAs=("2016preVFP" "2016postVFP" "2017" "2018")
RUN3_ERAs=("2022" "2022EE" "2023" "2023BPix")
OBJECTS=("muon" "electron")

optimize_era_object() {
    local era=$1
    local object=$2
    echo "Starting $object optimization for era: $era"
    optimize_tightID.py --era $era --object $object --format csv
    echo "Completed $object optimization for era: $era"
}

export -f optimize_era_object

# Run optimization in parallel for all Run3 eras and both objects
#parallel -j 8 optimize_era_object ::: ${RUN2_ERAs[@]} ::: ${OBJECTS[@]}
parallel -j 8 optimize_era_object ::: ${RUN3_ERAs[@]} ::: ${OBJECTS[@]}
