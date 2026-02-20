#!/bin/bash
# saveDatasets.sh - Create datasets for Mass-Decorrelated ParticleNet
#
# Usage: ./scripts/saveDatasets.sh
#
# Note: Saves all events (no subsampling). Subsampling is done at training time.
#       "Combined" channel is handled by DynamicDatasetLoader (loads Run1E2Mu + Run3Mu).

export PATH=$PATH:$PWD/python

signals=("TTToHcToWAToMuMu-MHc100_MA95"
         "TTToHcToWAToMuMu-MHc115_MA87"
         "TTToHcToWAToMuMu-MHc130_MA90"
         "TTToHcToWAToMuMu-MHc130_MA100"
         "TTToHcToWAToMuMu-MHc145_MA92"
         "TTToHcToWAToMuMu-MHc160_MA85"
         "TTToHcToWAToMuMu-MHc160_MA98"
     )

backgrounds=("Skim_TriLep_TTLL_powheg"
             "Skim_TriLep_TTLL_hdamp_down_powheg"
             "Skim_TriLep_TTLL_hdamp_up_powheg"
             "Skim_TriLep_TTLL_mtop169p5_powheg"
             "Skim_TriLep_TTLL_mtop171p5_powheg"
             "Skim_TriLep_TTLL_mtop173p5_powheg"
             "Skim_TriLep_TTLL_mtop175p5_powheg"
             "Skim_TriLep_TTLL_TuneCP5CR1_powheg"
             "Skim_TriLep_TTLL_TuneCP5CR2_powheg"
             "Skim_TriLep_TTLL_TuneCP5down_powheg"
             "Skim_TriLep_TTLL_TuneCP5_erdOn_powheg"
             "Skim_TriLep_TTLL_TuneCP5_RTT_powheg"
             "Skim_TriLep_TTLL_TuneCP5up_powheg"
             "Skim_TriLep_TTLL_widthx0p7_powheg"
             "Skim_TriLep_TTLL_widthx0p85_powheg"
             "Skim_TriLep_TTLL_widthx1p15_powheg"
             "Skim_TriLep_TTLL_widthx1p3_powheg"
             "Skim_TriLep_WZTo3LNu_amcatnlo"
             "Skim_TriLep_ZZTo4L_powheg"
             "Skim_TriLep_TTZToLLNuNu"
             "Skim_TriLep_TTWToLNu"
             "Skim_TriLep_tZq"
         )

# Save each channel separately; Combined is loaded on-the-fly by DynamicDatasetLoader
channels=("Run1E2Mu" "Run3Mu")

run_saveDataset_sample() {
    echo "Processing: $2/$1 for channel $3"
    saveDataset.py --sample $1 --sample-type $2 --channel $3
}

export -f run_saveDataset_sample

echo "Creating datasets for Mass-Decorrelated ParticleNet..."
echo "Node features: [E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]"
echo "Mass info: mass1, mass2 (OS muon pairs)"
echo ""

echo "Processing signal samples..."
parallel run_saveDataset_sample {1} signal {2} ::: "${signals[@]}" ::: "${channels[@]}"

echo "Processing background samples..."
parallel run_saveDataset_sample {1} background {2} ::: "${backgrounds[@]}" ::: "${channels[@]}"

# Summary
DATASET_DIR="${WORKDIR}/ParticleNetMD/dataset/samples"
echo ""
echo "Finished. Total files: $(find $DATASET_DIR -name "*.pt" 2>/dev/null | wc -l)"
echo "Dataset location: $DATASET_DIR"
