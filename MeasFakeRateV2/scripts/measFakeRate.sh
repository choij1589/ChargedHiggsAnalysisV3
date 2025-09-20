#!/bin/bash
ERA=$1
MEASURE=$2

export PATH=$PWD/python:$PATH

parseIntegral.py --era $ERA --measure $MEASURE
measFakeRate.py --era $ERA --measure $MEASURE
measFakeRate.py --era $ERA --measure $MEASURE --isQCD
plotFakeRate.py --era $ERA --measure $MEASURE
plotFakeRate.py --era $ERA --measure $MEASURE --isQCD

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

plot_validation() {
    local era=$1
    local hlt=$2
    local wp=$3
    local region=$4
    local selection=$5
    plotValidation.py --era $era --hlt $hlt --wp $wp --region $region --selection $selection
}

export -f plot_normalization
export -f plot_validation

parallel plot_normalization ::: $ERA ::: ${HLTs[@]} ::: ${WPs[@]} ::: ${SELECTIONs[@]}
parallel plot_validation ::: $ERA ::: ${HLTs[@]} ::: ${WPs[@]} ::: Inclusive ::: ${SELECTIONs[@]}

# Determine eta bins based on era
if [[ $ERA == "2022" || $ERA == "2022EE" || $ERA == "2023" || $ERA == "2023BPix" ]]; then
    # Run3 eta bins
    plotSystematics.py --era $ERA --measure $MEASURE --etabin EEm
    plotSystematics.py --era $ERA --measure $MEASURE --etabin EB2m
    plotSystematics.py --era $ERA --measure $MEASURE --etabin EB1m
    plotSystematics.py --era $ERA --measure $MEASURE --etabin EB1p
    plotSystematics.py --era $ERA --measure $MEASURE --etabin EB2p
    plotSystematics.py --era $ERA --measure $MEASURE --etabin EEp
else
    # Run2 abseta bins
    plotSystematics.py --era $ERA --measure $MEASURE --etabin EB1
    plotSystematics.py --era $ERA --measure $MEASURE --etabin EB2
    plotSystematics.py --era $ERA --measure $MEASURE --etabin EE
fi