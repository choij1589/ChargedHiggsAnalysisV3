#!/bin/bash
ERA=$1
MEASURE=$2

export PATH=$PWD/python:$PATH

parseIntegral.py --era $ERA --measure $MEASURE
measFakeRate.py --era $ERA --measure $MEASURE
measFakeRate.py --era $ERA --measure $MEASURE --isMC
plotFakeRate.py --era $ERA --measure $MEASURE
plotFakeRate.py --era $ERA --measure $MEASURE --isMC

HLTs=()
if [[ $MEASURE == "muon" ]]; then
    HLTs=("MeasFakeMu8" "MeasFakeMu17")
elif [[ $MEASURE == "electron" ]]; then
    HLTs=("MeasFakeEl8" "MeasFakeEl12" "MeasFakeEl23")
else
    echo "Invalid measure: $MEASURE"
    exit 1
fi

WPs=("loose" "tight")
SELECTIONs=("Central" "MotherJetPt_Up" "MotherJetPt_Down" "RequireHeavyTag")

plot_normalization() {
    local era=$1
    local hlt=$2
    local wp=$3
    local selection=$4
    if [[ -n $selection ]]; then
        plotNormalization.py --era $era --hlt $hlt --wp $wp --selection $selection
    else
        plotNormalization.py --era $era --hlt $hlt --wp $wp
    fi
}

plot_validation_inclusive() {
    local era=$1
    local hlt=$2
    local wp=$3
    local selection=$4
    plotValidation.py --era $era --hlt $hlt --wp $wp --region Inclusive --selection $selection --histkey MT
    plotValidation.py --era $era --hlt $hlt --wp $wp --region Inclusive --selection $selection --histkey pt
    plotValidation.py --era $era --hlt $hlt --wp $wp --region Inclusive --selection $selection --histkey eta
}

plot_validation_zenriched() {
    local era=$1
    local hlt=$2
    local wp=$3
    local selection=$4
    plotValidation.py --era $era --hlt $hlt --wp $wp --region ZEnriched --selection $selection --histkey ZCand/mass
}
    

export -f plot_normalization
export -f plot_validation_inclusive
export -f plot_validation_zenriched

parallel plot_normalization ::: $ERA ::: ${HLTs[@]} ::: ${WPs[@]} ::: ${SELECTIONs[@]}
parallel plot_validation_inclusive ::: $ERA ::: ${HLTs[@]} ::: ${WPs[@]} ::: ${SELECTIONs[@]}
parallel plot_validation_zenriched ::: $ERA ::: ${HLTs[@]} ::: ${WPs[@]} ::: ${SELECTIONs[@]}
plotSystematics.py --era $ERA --measure $MEASURE --etabin EB1
plotSystematics.py --era $ERA --measure $MEASURE --etabin EB2
plotSystematics.py --era $ERA --measure $MEASURE --etabin EE
