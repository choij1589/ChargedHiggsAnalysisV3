#!/bin/bash
export PATH=$PATH:$PWD/python

signals=("TTToHcToWAToMuMu-MHc100_MA95" "TTToHcToWAToMuMu-MHc115_MA87" "TTToHcToWAToMuMu-MHc130_MA90"
         "TTToHcToWAToMuMu-MHc130_MA100" "TTToHcToWAToMuMu-MHc145_MA92" "TTToHcToWAToMuMu-MHc160_MA85" "TTToHcToWAToMuMu-MHc160_MA98")

backgrounds=("Skim_TriLep_TTLL_powheg" "Skim_TriLep_DYJets" "Skim_TriLep_DYJets10to50" "Skim_TriLep_WZTo3LNu_amcatnlo"
             "Skim_TriLep_ZZTo4L_powheg" "Skim_TriLep_TTZToLLNuNu" "Skim_TriLep_TTWToLNu" "Skim_TriLep_tZq")

channels=("Run1E2Mu" "Run3Mu")

# Parse arguments
FLAGS=""
[[ "$*" == *"--pilot"* ]] && FLAGS+=" --pilot"
[[ "$*" == *"--separate_bjets"* ]] && FLAGS+=" --separate_bjets"
export FLAGS

run_saveDataset_sample() {
    echo "Processing: $2/$1 for channel $3"
    saveDataset.py --sample $1 --sample-type $2 --channel $3 $FLAGS
}

export -f run_saveDataset_sample

echo "Processing signal samples..."
parallel run_saveDataset_sample {1} signal {2} ::: "${signals[@]}" ::: "${channels[@]}"

echo "Processing background samples..."
parallel run_saveDataset_sample {1} background {2} ::: "${backgrounds[@]}" ::: "${channels[@]}"

# Summary
DATASET_DIR="${WORKDIR}/ParticleNet/dataset$(grep -q separate_bjets <<< "$FLAGS" && echo _bjets || echo)/samples"
echo "Finished. Total files: $(find $DATASET_DIR -name "*.pt" 2>/dev/null | wc -l)"
