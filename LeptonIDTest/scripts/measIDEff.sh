#!/bin/bash
export PATH="$PWD/python:$PATH"

ERAs=("2016preVFP" "2016postVFP" "2017" "2018" "2022" "2022EE" "2023" "2023BPix")
OBJECTS=("muon" "electron")

meas_ideff() {
    local era=$1
    local object=$2
    echo "Measuring $object ID efficiency for era: $era"
    measIDEff.py --era "$era" --object "$object"
}
export -f meas_ideff

parallel -j 16 meas_ideff ::: "${ERAs[@]}" ::: "${OBJECTS[@]}"
