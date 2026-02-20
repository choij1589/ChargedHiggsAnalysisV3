#!/bin/bash
set -euo pipefail

# Mass points
MASSPOINTs_BASELINE=(
    "MHc70_MA15" "MHc70_MA18" "MHc70_MA40" "MHc70_MA55" "MHc70_MA65"
    "MHc85_MA15" "MHc85_MA70" "MHc85_MA80"      # 85_21 is missng in 16a
    "MHc100_MA15" "MHc100_MA24" "MHc100_MA60" "MHc100_MA75" "MHc100_MA95"
    "MHc115_MA15" "MHc115_MA27" "MHc115_MA87" "MHc115_MA110"
    "MHc130_MA15" "MHc130_MA30" "MHc130_MA55" "MHc130_MA83" "MHc130_MA90" "MHc130_MA100" "MHc130_MA125"
    "MHc145_MA15" "MHc145_MA35" "MHc145_MA92" "MHc145_MA140"
    "MHc160_MA15" "MHc160_MA50" "MHc160_MA85" "MHc160_MA98" "MHc160_MA120" "MHc160_MA135" "MHc160_MA155"
)
MASSPOINTs_PARTICLENET=(
    "MHc100_MA95" "MHc130_MA90" "MHc160_MA85" "MHc115_MA87" "MHc145_MA92" "MHc160_MA98"
)

# Eras
ERAs_RUN2=("2016preVFP" "2016postVFP" "2017" "2018")
ERAs_RUN3=("2022" "2022EE" "2023" "2023BPix")

# Channels
CHANNELs_SR=("SR1E2Mu" "SR3Mu")
CHANNELs_TTZ=("TTZ2E1Mu")  # Only for ParticleNet masspoints

# Number of parallel jobs
NJOBS_RUN2=16
NJOBS_RUN3=12

# HTCondor settings
USE_CONDOR=false
DRY_RUN=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDOR_DIR="${SCRIPT_DIR}/../condor"

# Export PATH to include python scripts
export PATH="${PWD}/python:${PATH}"

# Function to preprocess a single sample
function preprocess_sample() {
    local era=$1
    local channel=$2
    local masspoint=$3
    local extra_args=${4:-}

    echo "Preprocessing: era=$era, channel=$channel, masspoint=$masspoint $extra_args"
    preprocess.py --era "$era" --channel "$channel" --masspoint "$masspoint" $extra_args
}

export -f preprocess_sample

# Function to submit HTCondor jobs
# Args: eras_array masspoints_array channels_array extra_args
function submit_condor_jobs() {
    local -n eras_ref=$1
    local -n masspoints_ref=$2
    local -n channels_ref=$3
    local extra_args=${4:-}

    # Create timestamped job directory
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local job_dir="$CONDOR_DIR/jobs_${timestamp}"
    mkdir -p "$job_dir/logs"

    # Generate job_params.txt with all era/channel/masspoint/extra_args combinations
    local params_file="$job_dir/job_params.txt"
    > "$params_file"

    for era in "${eras_ref[@]}"; do
        for channel in "${channels_ref[@]}"; do
            for masspoint in "${masspoints_ref[@]}"; do
                echo "$era,$channel,$masspoint,$extra_args" >> "$params_file"
            done
        done
    done

    local num_jobs=$(wc -l < "$params_file")
    echo "Generated $num_jobs jobs in $params_file"

    # Generate HTCondor submission file
    local sub_file="$job_dir/preprocess.sub"
    cat > "$sub_file" << 'CONDOR_SUB'
JobBatchName = preprocess
universe = vanilla
executable = WRAPPER_PATH
arguments = $(era) $(channel) $(masspoint) $(extra_args)
output = logs/preprocess_$(era)_$(channel)_$(masspoint).out
error = logs/preprocess_$(era)_$(channel)_$(masspoint).err
log = preprocess.log

request_cpus = 1
request_memory = 2GB
request_disk = 1GB

should_transfer_files = NO

queue era,channel,masspoint,extra_args from job_params.txt
CONDOR_SUB

    # Replace WRAPPER_PATH with actual path
    sed -i "s|WRAPPER_PATH|$CONDOR_DIR/preprocess_wrapper.sh|g" "$sub_file"

    echo "Created submission file: $sub_file"

    # Submit or dry-run
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] Would submit: condor_submit $sub_file"
        echo "[DRY-RUN] Job directory: $job_dir"
        echo "[DRY-RUN] First 10 jobs:"
        head -10 "$params_file" | while IFS=, read -r era channel masspoint extra; do
            echo "  era=$era channel=$channel masspoint=$masspoint extra_args=$extra"
        done
    else
        echo "Submitting to HTCondor..."
        cd "$job_dir"
        condor_submit preprocess.sub
        cd - > /dev/null
    fi
}

# Parse command line arguments
MODE=""  # Options: run2, run3, run3-scaled

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --condor)
            USE_CONDOR=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 --mode <run2|run3|run3-scaled> [--condor] [--dry-run]"
            echo ""
            echo "Modes:"
            echo "  run2         - Process Run2 only"
            echo "  run3         - Process Run3 with real signal MC (if available)"
            echo "  run3-scaled  - Process Run3 with signal scaled from 2018"
            echo "                 Requires: Run2 (2018) must be preprocessed first"
            echo ""
            echo "Options:"
            echo "  --condor     - Submit jobs to HTCondor instead of local parallel"
            echo "  --dry-run    - Generate submission files without submitting (requires --condor)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate mode is specified
if [[ -z "$MODE" ]]; then
    echo "Error: --mode is required. Use --help for usage."
    exit 1
fi

if [[ "$MODE" != "run2" && "$MODE" != "run3" && "$MODE" != "run3-scaled" ]]; then
    echo "Error: Invalid mode '$MODE'. Must be one of: run2, run3, run3-scaled"
    exit 1
fi

# Validate options
if [[ "$DRY_RUN" == "true" && "$USE_CONDOR" == "false" ]]; then
    echo "Error: --dry-run requires --condor"
    exit 1
fi

echo "============================================================"
echo "SignalRegionStudyV2 Preprocessing"
echo "Mode: $MODE"
if [[ "$USE_CONDOR" == "true" ]]; then
    echo "Execution: HTCondor (dry-run: $DRY_RUN)"
else
    echo "Execution: Local (GNU parallel)"
fi
echo "============================================================"

# Compute mass point categories for processing
# - MASSPOINTS_BASELINE_ONLY: baseline mass points that are NOT ParticleNet
# - MASSPOINTs_PARTICLENET: all ParticleNet mass points (need TTZ2E1Mu channel)
MASSPOINTS_BASELINE_ONLY=()

for mp in "${MASSPOINTs_BASELINE[@]}"; do
    if [[ ! " ${MASSPOINTs_PARTICLENET[*]} " =~ " ${mp} " ]]; then
        MASSPOINTS_BASELINE_ONLY+=("$mp")
    fi
done

# Run2 processing
if [[ "$MODE" == "run2" ]]; then
    echo ""
    echo "============================================================"
    echo "Processing Run2 eras..."
    echo "============================================================"

    if [[ "$USE_CONDOR" == "true" ]]; then
        echo "Submitting Run2 jobs to HTCondor..."
        # Baseline-only mass points (SR channels only)
        if [[ ${#MASSPOINTS_BASELINE_ONLY[@]} -gt 0 ]]; then
            echo "  Submitting baseline-only mass points (SR channels)..."
            submit_condor_jobs ERAs_RUN2 MASSPOINTS_BASELINE_ONLY CHANNELs_SR ""
        fi
        # ParticleNet mass points (SR channels)
        echo "  Submitting ParticleNet mass points (SR channels)..."
        submit_condor_jobs ERAs_RUN2 MASSPOINTs_PARTICLENET CHANNELs_SR ""
        # ParticleNet mass points (TTZ channel)
        echo "  Submitting ParticleNet mass points (TTZ2E1Mu channel)..."
        submit_condor_jobs ERAs_RUN2 MASSPOINTs_PARTICLENET CHANNELs_TTZ ""
    else
        # Baseline-only mass points (SR channels only)
        if [[ ${#MASSPOINTS_BASELINE_ONLY[@]} -gt 0 ]]; then
            echo "Processing Baseline-only mass points for Run2 (SR channels)..."
            for channel in "${CHANNELs_SR[@]}"; do
                parallel -j $NJOBS_RUN2 preprocess_sample {1} "$channel" {2} ::: "${ERAs_RUN2[@]}" ::: "${MASSPOINTS_BASELINE_ONLY[@]}"
            done
        fi

        # ParticleNet mass points (SR channels)
        echo "Processing ParticleNet mass points for Run2 (SR channels)..."
        for channel in "${CHANNELs_SR[@]}"; do
            parallel -j $NJOBS_RUN2 preprocess_sample {1} "$channel" {2} ::: "${ERAs_RUN2[@]}" ::: "${MASSPOINTs_PARTICLENET[@]}"
        done

        # ParticleNet mass points (TTZ channel)
        echo "Processing ParticleNet mass points for Run2 (TTZ2E1Mu channel)..."
        parallel -j $NJOBS_RUN2 preprocess_sample {1} "TTZ2E1Mu" {2} ::: "${ERAs_RUN2[@]}" ::: "${MASSPOINTs_PARTICLENET[@]}"
    fi

    echo "Run2 preprocessing complete!"
fi

# Run3 processing with scaled signal
if [[ "$MODE" == "run3-scaled" ]]; then
    echo ""
    echo "============================================================"
    echo "Processing Run3 eras (signal scaled from 2018)..."
    echo "============================================================"

    if [[ "$USE_CONDOR" == "true" ]]; then
        echo "Submitting Run3 (scaled) jobs to HTCondor..."
        # Baseline-only mass points (SR channels only)
        if [[ ${#MASSPOINTS_BASELINE_ONLY[@]} -gt 0 ]]; then
            echo "  Submitting baseline-only mass points (SR channels)..."
            submit_condor_jobs ERAs_RUN3 MASSPOINTS_BASELINE_ONLY CHANNELs_SR "--scale-from-run2"
        fi
        # ParticleNet mass points (SR channels)
        echo "  Submitting ParticleNet mass points (SR channels)..."
        submit_condor_jobs ERAs_RUN3 MASSPOINTs_PARTICLENET CHANNELs_SR "--scale-from-run2"
        # ParticleNet mass points (TTZ channel) - no --scale-from-run2 needed (no signal)
        echo "  Submitting ParticleNet mass points (TTZ2E1Mu channel)..."
        submit_condor_jobs ERAs_RUN3 MASSPOINTs_PARTICLENET CHANNELs_TTZ ""
    else
        # Baseline-only mass points (SR channels only)
        if [[ ${#MASSPOINTS_BASELINE_ONLY[@]} -gt 0 ]]; then
            echo "Processing Baseline-only mass points for Run3 (SR channels, scaled signal)..."
            for channel in "${CHANNELs_SR[@]}"; do
                parallel -j $NJOBS_RUN3 preprocess_sample {1} "$channel" {2} "--scale-from-run2" ::: "${ERAs_RUN3[@]}" ::: "${MASSPOINTS_BASELINE_ONLY[@]}"
            done
        fi

        # ParticleNet mass points (SR channels)
        echo "Processing ParticleNet mass points for Run3 (SR channels, scaled signal)..."
        for channel in "${CHANNELs_SR[@]}"; do
            parallel -j $NJOBS_RUN3 preprocess_sample {1} "$channel" {2} "--scale-from-run2" ::: "${ERAs_RUN3[@]}" ::: "${MASSPOINTs_PARTICLENET[@]}"
        done

        # ParticleNet mass points (TTZ channel) - no --scale-from-run2 needed (no signal)
        echo "Processing ParticleNet mass points for Run3 (TTZ2E1Mu channel)..."
        parallel -j $NJOBS_RUN3 preprocess_sample {1} "TTZ2E1Mu" {2} ::: "${ERAs_RUN3[@]}" ::: "${MASSPOINTs_PARTICLENET[@]}"
    fi

    echo "Run3 (scaled signal) preprocessing complete!"
fi

# Run3 processing with real signal (if available)
if [[ "$MODE" == "run3" ]]; then
    echo ""
    echo "============================================================"
    echo "Processing Run3 eras (real signal MC)..."
    echo "============================================================"

    if [[ "$USE_CONDOR" == "true" ]]; then
        echo "Submitting Run3 (real signal) jobs to HTCondor..."
        # Baseline-only mass points (SR channels only)
        if [[ ${#MASSPOINTS_BASELINE_ONLY[@]} -gt 0 ]]; then
            echo "  Submitting baseline-only mass points (SR channels)..."
            submit_condor_jobs ERAs_RUN3 MASSPOINTS_BASELINE_ONLY CHANNELs_SR ""
        fi
        # ParticleNet mass points (SR channels)
        echo "  Submitting ParticleNet mass points (SR channels)..."
        submit_condor_jobs ERAs_RUN3 MASSPOINTs_PARTICLENET CHANNELs_SR ""
        # ParticleNet mass points (TTZ channel)
        echo "  Submitting ParticleNet mass points (TTZ2E1Mu channel)..."
        submit_condor_jobs ERAs_RUN3 MASSPOINTs_PARTICLENET CHANNELs_TTZ ""
    else
        # Baseline-only mass points (SR channels only)
        if [[ ${#MASSPOINTS_BASELINE_ONLY[@]} -gt 0 ]]; then
            echo "Processing Baseline-only mass points for Run3 (SR channels)..."
            for channel in "${CHANNELs_SR[@]}"; do
                parallel -j $NJOBS_RUN3 preprocess_sample {1} "$channel" {2} ::: "${ERAs_RUN3[@]}" ::: "${MASSPOINTS_BASELINE_ONLY[@]}"
            done
        fi

        # ParticleNet mass points (SR channels)
        echo "Processing ParticleNet mass points for Run3 (SR channels)..."
        for channel in "${CHANNELs_SR[@]}"; do
            parallel -j $NJOBS_RUN3 preprocess_sample {1} "$channel" {2} ::: "${ERAs_RUN3[@]}" ::: "${MASSPOINTs_PARTICLENET[@]}"
        done

        # ParticleNet mass points (TTZ channel)
        echo "Processing ParticleNet mass points for Run3 (TTZ2E1Mu channel)..."
        parallel -j $NJOBS_RUN3 preprocess_sample {1} "TTZ2E1Mu" {2} ::: "${ERAs_RUN3[@]}" ::: "${MASSPOINTs_PARTICLENET[@]}"
    fi

    echo "Run3 (real signal) preprocessing complete!"
fi

echo ""
echo "============================================================"
echo "All preprocessing complete!"
echo "============================================================"
