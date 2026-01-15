#!/bin/bash
MASSPOINTs=("MHc70_MA15" "MHc100_MA60" "MHc130_MA90" "MHc160_MA155")
MASSPOINTsParticleNet=("MHc100_MA95" "MHc130_MA90" "MHc160_MA85")
ERAs=("2016preVFP" "2016postVFP" "2017" "2018")
CHANNELs=("SR1E2Mu" "SR3Mu")
BINNINGs=("uniform" "extended")

## Use GNU parallel
#rm -rf samples
#rm -rf templates

function preprocess_baseline() {
    local channel=$1
    local era=$2
    local masspoint=$3
    #preprocess.py --era $era --channel $channel --masspoint $masspoint
    #makeBinnedTemplates.py --era $era --channel $channel --masspoint $masspoint --method Baseline --binning uniform
    #makeBinnedTemplates.py --era $era --channel $channel --masspoint $masspoint --method Baseline --binning extended
    checkTemplates.py --era $era --channel $channel --masspoint $masspoint --method Baseline --binning uniform
    checkTemplates.py --era $era --channel $channel --masspoint $masspoint --method Baseline --binning extended
    #printDatacard.py --era $era --channel $channel --masspoint $masspoint --method Baseline --binning uniform
    #printDatacard.py --era $era --channel $channel --masspoint $masspoint --method Baseline --binning extended
}

function preprocess_particleNet() {
    local channel=$1
    local era=$2
    local masspoint=$3
    # Preprocess if samples don't exist yet (for mass points not in Baseline)
    if [ ! -d "samples/$era/$channel/$masspoint" ]; then
        preprocess.py --era $era --channel $channel --masspoint $masspoint
    fi
<<<<<<< HEAD
    makeBinnedTemplates.py --era $era --channel $channel --masspoint $masspoint --method ParticleNet --binning uniform
    plotParticleNetScore.py --era $era --channel $channel --masspoint $masspoint --binning uniform
    checkTemplates.py --era $era --channel $channel --masspoint $masspoint --method ParticleNet --binning uniform
    printDatacard.py --era $era --channel $channel --masspoint $masspoint --method ParticleNet --binning uniform
=======
    #makeBinnedTemplates.py --era $era --channel $channel --masspoint $masspoint --method ParticleNet --binning extended
    plotParticleNetScore.py --era $era --channel $channel --masspoint $masspoint --binning extended
    checkTemplates.py --era $era --channel $channel --masspoint $masspoint --method ParticleNet --binning extended
    #printDatacard.py --era $era --channel $channel --masspoint $masspoint --method ParticleNet --binning extended
>>>>>>> ae0b3f54c1d23c6ccdaaa7c30ba8fd7866fd26f4
}

function run_combined_asymptotic() {
    local masspoint=$1
    local method=$2
    local binning=$3
    ./scripts/runCombinedAsymptotic.sh --masspoint $masspoint --method $method --binning $binning
}

export -f preprocess_baseline
export -f preprocess_particleNet
export -f run_combined_asymptotic

parallel -j 16 preprocess_baseline SR1E2Mu {} {} ::: "${ERAs[@]}" ::: "${MASSPOINTs[@]}"
parallel -j 16 preprocess_baseline SR3Mu {} {} ::: "${ERAs[@]}" ::: "${MASSPOINTs[@]}"

echo "Processing SR1E2Mu (ParticleNet)..."
parallel -j 12 preprocess_particleNet SR1E2Mu {} {} ::: "${ERAs[@]}" ::: "${MASSPOINTsParticleNet[@]}"

echo "Processing SR3Mu (ParticleNet)..."
parallel -j 12 preprocess_particleNet SR3Mu {} {} ::: "${ERAs[@]}" ::: "${MASSPOINTsParticleNet[@]}"

# Run combined asymptotic limits for all mass points, methods, and binning options
echo ""
echo "Running Combined Asymptotic Limits (Baseline method)..."
#parallel -j 8 run_combined_asymptotic {1} Baseline {2} ::: "${MASSPOINTs[@]}" ::: "${BINNINGs[@]}"

echo ""
echo "Running Combined Asymptotic Limits (ParticleNet method)..."
#parallel -j 3 run_combined_asymptotic {1} ParticleNet {2} ::: "${MASSPOINTsParticleNet[@]}" ::: "${BINNINGs[@]}"

echo ""
echo "All Combined Asymptotic Limits completed."
