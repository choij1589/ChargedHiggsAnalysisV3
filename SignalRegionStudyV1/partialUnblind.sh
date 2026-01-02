#!/bin/bash
MASSPOINTs=("MHc160_MA85" "MHc130_MA90" "MHc100_MA95")
ERAs=("2016preVFP" "2016postVFP" "2017" "2018")

function partial_unblind() {
    local channel=$1
    local era=$2
    local masspoint=$3
    preprocess.py --era $era --channel $channel --masspoint $masspoint
    makeBinnedTemplates.py --era $era --channel $channel --masspoint $masspoint --method ParticleNet --binning extended --partial-unblind
    checkTemplates.py --era $era --channel $channel --masspoint $masspoint --method ParticleNet --binning extended --partial-unblind
    plotParticleNetScore.py --era $era --channel $channel --masspoint $masspoint --binning extended --partial-unblind
    printDatacard.py --era $era --channel $channel --masspoint $masspoint --method ParticleNet --binning extended --partial-unblind
}

export -f partial_unblind

# SR3Mu depends on SR1E2Mu fit results, so process SR1E2Mu first
#echo "Processing SR1E2Mu (partial-unblind)..."
#parallel -j 4 partial_unblind SR1E2Mu {} {} ::: "${ERAs[@]}" ::: "${MASSPOINTs[@]}"

#echo "Processing SR3Mu (partial-unblind)..."
#parallel -j 4 partial_unblind SR3Mu {} {} ::: "${ERAs[@]}" ::: "${MASSPOINTs[@]}"

#echo ""
#echo "Partial unblind template generation completed."
#echo ""

# Run combined asymptotic limits for each mass point
#echo "Running combined asymptotic limits..."
#for MP in "${MASSPOINTs[@]}"; do
#    echo "  Processing ${MP}..."
#    ./scripts/runCombinedAsymptotic.sh --masspoint "$MP" --method ParticleNet --binning extended --partial-unblind
#done

echo ""
echo "Running impact studies..."
for MP in "${MASSPOINTs[@]}"; do
    echo "  Processing ${MP}..."
    #./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint "$MP" --method ParticleNet --binning extended --partial-unblind --condor
    ./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint "$MP" --method ParticleNet --binning extended --partial-unblind --skip-initial --skip-fits
done

echo ""
echo "Partial unblind processing completed."
