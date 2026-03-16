#!/bin/bash
set -euo pipefail

ERA=$1
MEASURE=$2
NO_HEM_VETO=false
if [[ "${3:-}" == "--noHEMVeto" ]]; then
    NO_HEM_VETO=true
fi

export PATH=$PWD/python:$PATH

if $NO_HEM_VETO; then
    parseIntegral.py --era "$ERA" --measure "$MEASURE" --noHEMVeto
    measFakeRate.py --era "$ERA" --measure "$MEASURE" --noHEMVeto
else
    parseIntegral.py --era "$ERA" --measure "$MEASURE"
    measFakeRate.py --era "$ERA" --measure "$MEASURE"
    measFakeRate.py --era "$ERA" --measure "$MEASURE" --isMC
fi

if $NO_HEM_VETO; then
    plotFakeRate.py --era "$ERA" --measure "$MEASURE" --noHEMVeto
else
    plotFakeRate.py --era "$ERA" --measure "$MEASURE"
    plotFakeRate.py --era "$ERA" --measure "$MEASURE" --isMC
fi

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
    local base_args="--era $era --hlt $hlt --wp $wp --region ZEnriched --selection $selection --histkey ZCand/mass"
    if $NO_HEM_VETO; then
        plotValidation.py $base_args --noHEMVeto
    else
        plotValidation.py $base_args
    fi
}

plot_validation_inclusive() {
    local era=$1
    local hlt=$2
    local wp=$3
    local selection=$4
    local base_args="--era $era --hlt $hlt --wp $wp --region Inclusive --selection $selection"
    if $NO_HEM_VETO; then
        plotValidation.py $base_args --histkey MT --noHEMVeto
        plotValidation.py $base_args --histkey pt --noHEMVeto
        plotValidation.py $base_args --histkey eta --noHEMVeto
    else
        plotValidation.py $base_args --histkey MT
        plotValidation.py $base_args --histkey pt
        plotValidation.py $base_args --histkey eta
    fi
}

export -f plot_validation_zenriched
export -f plot_validation_inclusive
export NO_HEM_VETO

if $NO_HEM_VETO; then
    parallel plot_validation_zenriched ::: "$ERA" ::: "${HLTs[@]}" ::: "${WPs[@]}" ::: "Central"
    parallel plot_validation_inclusive ::: "$ERA" ::: "${HLTs[@]}" ::: "${WPs[@]}" ::: "Central"
else
    parallel plot_validation_zenriched ::: "$ERA" ::: "${HLTs[@]}" ::: "${WPs[@]}" ::: "${SELECTIONs[@]}"
    parallel plot_validation_inclusive ::: "$ERA" ::: "${HLTs[@]}" ::: "${WPs[@]}" ::: "${SELECTIONs[@]}"
fi

if $NO_HEM_VETO; then
    plotCompareHEMVeto.py
else
    plotSystematics.py --era "$ERA" --measure "$MEASURE" --etabin EB1
    plotSystematics.py --era "$ERA" --measure "$MEASURE" --etabin EB2
    plotSystematics.py --era "$ERA" --measure "$MEASURE" --etabin EE
fi
