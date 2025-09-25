#!/bin/bash
export PATH=$PATH:$PWD/python

# Signal MC samples (use actual ROOT filename without .root)
signals=("TTToHcToWAToMuMu-MHc100_MA95" 
         "TTToHcToWAToMuMu-MHc115_MA87" 
         "TTToHcToWAToMuMu-MHc130_MA90" 
         "TTToHcToWAToMuMu-MHc130_MA100" 
         "TTToHcToWAToMuMu-MHc145_MA92" 
         "TTToHcToWAToMuMu-MHc160_MA85" 
         "TTToHcToWAToMuMu-MHc160_MA98")

# Background MC samples (use actual ROOT filename without .root)
backgrounds=("Skim_TriLep_TTLL_powheg" 
             "Skim_TriLep_DYJets"
             "Skim_TriLep_DYJets10to50"
             "Skim_TriLep_WZTo3LNu_amcatnlo" 
             "Skim_TriLep_TTZToLLNuNu"
             "Skim_TriLep_TTWToLNu"
             "Skim_TriLep_tZq")

channels=("Run1E2Mu" "Run3Mu")

# Check for pilot mode
PILOT_FLAG=""
if [[ "$1" == "--pilot" ]]; then
    echo "Running in pilot mode"
    PILOT_FLAG="--pilot"
    shift
fi

echo "Creating per-process ParticleNet datasets"
echo "=========================================="

run_saveDataset_sample() {
    local sample="$1"
    local sample_type="$2"
    local channel="$3"

    echo "Processing: $sample_type/$sample for channel $channel"
    saveDataset.py --sample $sample --sample-type $sample_type --channel $channel $PILOT_FLAG
}

export -f run_saveDataset_sample

# Process all signals
echo "Processing signal samples..."
parallel run_saveDataset_sample {1} signal {2} ::: "${signals[@]}" ::: "${channels[@]}"

# Process all backgrounds
echo "Processing background samples..."
parallel run_saveDataset_sample {1} background {2} ::: "${backgrounds[@]}" ::: "${channels[@]}"

echo "=========================================="
echo "Finished creating per-process datasets"

# Generate a summary report
echo ""
echo "Dataset Summary:"
echo "=================="
find ${WORKDIR}/ParticleNet/dataset/samples -name "*.pt" | wc -l | awk '{print "Total .pt files created: " $1}'

# Show directory structure
echo ""
echo "Directory structure:"
if command -v tree >/dev/null 2>&1; then
    tree ${WORKDIR}/ParticleNet/dataset/samples -I "*pilot*"
else
    find ${WORKDIR}/ParticleNet/dataset/samples -type d | head -20
fi
