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

for HLT in ${HLTs[@]}; do
    plotNormalization.py --era $ERA --hlt $HLT --wp loose 
    plotNormalization.py --era $ERA --hlt $HLT --wp loose --selection MotherJetPt_Up
    plotNormalization.py --era $ERA --hlt $HLT --wp loose --selection MotherJetPt_Down
    plotNormalization.py --era $ERA --hlt $HLT --wp loose --selection RequireHeavyTag
    plotNormalization.py --era $ERA --hlt $HLT --wp tight 
    plotNormalization.py --era $ERA --hlt $HLT --wp tight --selection MotherJetPt_Up
    plotNormalization.py --era $ERA --hlt $HLT --wp tight --selection MotherJetPt_Down
    plotNormalization.py --era $ERA --hlt $HLT --wp tight --selection RequireHeavyTag
done