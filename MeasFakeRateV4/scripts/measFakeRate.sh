#!/bin/bash
set -euo pipefail

ERA=$1
MEASURE=$2

export PATH=$PWD/python:$PATH

parseIntegral.py --era "$ERA" --measure "$MEASURE"
measFakeRate.py --era "$ERA" --measure "$MEASURE"
measFakeRate.py --era "$ERA" --measure "$MEASURE" --isMC
plotFakeRate.py --era "$ERA" --measure "$MEASURE"
plotFakeRate.py --era "$ERA" --measure "$MEASURE" --isMC

HLTs=()
if [[ $MEASURE == "muon" ]]; then
    HLTs=("Mu8" "Mu17")
elif [[ $MEASURE == "electron" ]]; then
    HLTs=("El8" "El12" "El23")
else
    echo "Invalid measure: $MEASURE"
    exit 1
fi

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

parallel plot_validation_zenriched ::: "$ERA" ::: "${HLTs[@]}" ::: "${WPs[@]}" ::: "${SELECTIONs[@]}"
parallel plot_validation_inclusive ::: "$ERA" ::: "${HLTs[@]}" ::: "${WPs[@]}" ::: "${SELECTIONs[@]}"

plotSystematics.py --era "$ERA" --measure "$MEASURE" --etabin EB1
plotSystematics.py --era "$ERA" --measure "$MEASURE" --etabin EB2
plotSystematics.py --era "$ERA" --measure "$MEASURE" --etabin EE
