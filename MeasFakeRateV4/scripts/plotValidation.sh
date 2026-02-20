#!/bin/bash
set -euo pipefail

ERA=$1

export PATH=$PWD/python:$PATH

# Define HLTs for each measure
HLTs_El=("El8" "El12" "El23")
HLTs_Mu=("Mu8" "Mu17")

WPs=("loose" "tight")
SELECTIONs=("Central" "MotherJetPt_Up" "MotherJetPt_Down" "RequireHeavyTag")

plot_validation_zenriched() {
    local era=$1
    local hlt=$2
    local wp=$3
    local selection=$4
    plotValidation.py --era "$era" --hlt "$hlt" --wp "$wp" --region ZEnriched --selection "$selection" --histkey ZCand/mass
}

plot_validation_inclusive() {
    local era=$1
    local hlt=$2
    local wp=$3
    local selection=$4
    plotValidation.py --era "$era" --hlt "$hlt" --wp "$wp" --region Inclusive --selection "$selection" --histkey MT
    plotValidation.py --era "$era" --hlt "$hlt" --wp "$wp" --region Inclusive --selection "$selection" --histkey pt
    plotValidation.py --era "$era" --hlt "$hlt" --wp "$wp" --region Inclusive --selection "$selection" --histkey eta
}

export -f plot_validation_zenriched
export -f plot_validation_inclusive

# Electron HLTs
parallel plot_validation_zenriched ::: "$ERA" ::: "${HLTs_El[@]}" ::: "${WPs[@]}" ::: "${SELECTIONs[@]}"
parallel plot_validation_inclusive ::: "$ERA" ::: "${HLTs_El[@]}" ::: "${WPs[@]}" ::: "${SELECTIONs[@]}"

# Muon HLTs
parallel plot_validation_zenriched ::: "$ERA" ::: "${HLTs_Mu[@]}" ::: "${WPs[@]}" ::: "${SELECTIONs[@]}"
parallel plot_validation_inclusive ::: "$ERA" ::: "${HLTs_Mu[@]}" ::: "${WPs[@]}" ::: "${SELECTIONs[@]}"
