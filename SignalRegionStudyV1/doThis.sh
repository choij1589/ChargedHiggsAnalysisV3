#!/bin/bash
MASSPOINTs=("MHc70_MA15" "MHc100_MA60" "MHc130_MA90" "MHc160_MA155")
MASSPOINTsParticleNet=("MHc130_MA90")
ERAs=("2016preVFP" "2016postVFP" "2017" "2018")
CHANNELs=("SR1E2Mu" "SR3Mu")
BINNINGs=("uniform" "sigma")

## Use GNU parallel
#rm -rf samples
#rm -rf templates

function preprocess_baseline() {
    local era=$1
    local channel=$2
    local masspoint=$3
    preprocess.py --era $era --channel $channel --masspoint $masspoint
    makeBinnedTemplates.py --era $era --channel $channel --masspoint $masspoint --method Baseline --binning uniform
    makeBinnedTemplates.py --era $era --channel $channel --masspoint $masspoint --method Baseline --binning sigma
    checkTemplates.py --era $era --channel $channel --masspoint $masspoint --method Baseline --binning uniform
    checkTemplates.py --era $era --channel $channel --masspoint $masspoint --method Baseline --binning sigma
    printDatacard.py --era $era --channel $channel --masspoint $masspoint --method Baseline --binning uniform
    printDatacard.py --era $era --channel $channel --masspoint $masspoint --method Baseline --binning sigma
}

function preprocess_particleNet() {
    local channel=$1
    local era=$2
    local masspoint=$3
    makeBinnedTemplates.py --era $era --channel $channel --masspoint $masspoint --method ParticleNet --binning uniform
    plotParticleNetScore.py --era $era --channel $channel --masspoint $masspoint --binning uniform
    checkTemplates.py --era $era --channel $channel --masspoint $masspoint --method ParticleNet --binning uniform
    printDatacard.py --era $era --channel $channel --masspoint $masspoint --method ParticleNet --binning uniform

    makeBinnedTemplates.py --era $era --channel $channel --masspoint $masspoint --method ParticleNet --binning sigma
    plotParticleNetScore.py --era $era --channel $channel --masspoint $masspoint --binning sigma
    checkTemplates.py --era $era --channel $channel --masspoint $masspoint --method ParticleNet --binning sigma
    printDatacard.py --era $era --channel $channel --masspoint $masspoint --method ParticleNet --binning sigma
}

export -f preprocess_baseline
export -f preprocess_particleNet

parallel -j 16 preprocess_baseline {} {} {} ::: "${ERAs[@]}" ::: "${CHANNELs[@]}" ::: "${MASSPOINTs[@]}"

# ParticleNet: SR3Mu depends on SR1E2Mu fit results, so process SR1E2Mu first
# Function signature: preprocess_particleNet(channel, era, masspoint)
echo "Processing SR1E2Mu (ParticleNet)..."
parallel -j 4 preprocess_particleNet SR1E2Mu {} {} ::: "${ERAs[@]}" ::: "${MASSPOINTsParticleNet[@]}"

echo "Processing SR3Mu (ParticleNet)..."
parallel -j 4 preprocess_particleNet SR3Mu {} {} ::: "${ERAs[@]}" ::: "${MASSPOINTsParticleNet[@]}"
