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
CHANNELs=("SR1E2Mu" "SR3Mu")

# Number of parallel jobs
NJOBS_RUN2=16
NJOBS_RUN3=12

# HTCondor settings
USE_CONDOR=false
DRY_RUN=false
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDOR_DIR="$SCRIPT_DIR/condor"

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
function submit_condor_jobs() {
    local -n eras_ref=$1
    local -n masspoints_ref=$2
    local extra_args=${3:-}

    # Create timestamped job directory
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local job_dir="$CONDOR_DIR/jobs_${timestamp}"
    mkdir -p "$job_dir/logs"

    # Generate job_params.txt with all era/channel/masspoint/extra_args combinations
    local params_file="$job_dir/job_params.txt"
    > "$params_file"

    for era in "${eras_ref[@]}"; do
        for channel in "${CHANNELs[@]}"; do
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
MODE="all"  # Options: all, run2, run3, run3-scaled

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
            echo "Usage: $0 [--mode <all|run2|run3|run3-scaled>] [--condor] [--dry-run]"
            echo ""
            echo "Modes:"
            echo "  all          - Process Run2 and Run3 (Run3 with scaled signal)"
            echo "                 Note: With --condor, waits for Run2 completion before Run3"
            echo "  run2         - Process Run2 only"
            echo "  run3         - Process Run3 with real signal MC (if available)"
            echo "  run3-scaled  - Process Run3 with signal scaled from 2018"
            echo "                 Requires: Run2 (2018) must be preprocessed first"
            echo ""
            echo "Options:"
            echo "  --condor     - Submit jobs to HTCondor instead of local parallel"
            echo "  --dry-run    - Generate submission files without submitting (requires --condor)"
            echo ""
            echo "Dependencies:"
            echo "  Run3 --scale-from-run2 requires preprocessed 2018 samples as source."
            echo "  With --mode all, Run2 jobs complete before Run3 jobs start."
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

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

# Compute ParticleNet-only mass points (needed for both local and condor)
MASSPOINTS_PN_ONLY=()
for mp in "${MASSPOINTs_PARTICLENET[@]}"; do
    if [[ ! " ${MASSPOINTs_BASELINE[*]} " =~ " ${mp} " ]]; then
        MASSPOINTS_PN_ONLY+=("$mp")
    fi
done

# Combine baseline and ParticleNet-only mass points for condor submission
ALL_MASSPOINTS=("${MASSPOINTs_BASELINE[@]}")
for mp in "${MASSPOINTS_PN_ONLY[@]}"; do
    ALL_MASSPOINTS+=("$mp")
done

# Track condor cluster ID for dependency management
RUN2_CLUSTER_ID=""

# Run2 processing
if [[ "$MODE" == "all" || "$MODE" == "run2" ]]; then
    echo ""
    echo "============================================================"
    echo "Processing Run2 eras..."
    echo "============================================================"

    if [[ "$USE_CONDOR" == "true" ]]; then
        echo "Submitting Run2 jobs to HTCondor..."
        submit_condor_jobs ERAs_RUN2 ALL_MASSPOINTS ""
        # Capture cluster ID from the most recent submission
        if [[ "$DRY_RUN" == "false" ]]; then
            RUN2_CLUSTER_ID=$(condor_q -submitter "$USER" -af ClusterId 2>/dev/null | tail -1)
            echo "Run2 cluster ID: $RUN2_CLUSTER_ID"
        fi
    else
        # Baseline mass points
        echo "Processing Baseline mass points for Run2..."
        for channel in "${CHANNELs[@]}"; do
            parallel -j $NJOBS_RUN2 preprocess_sample {1} "$channel" {2} ::: "${ERAs_RUN2[@]}" ::: "${MASSPOINTs_BASELINE[@]}"
        done

        # ParticleNet mass points (that are not in Baseline)
        if [[ ${#MASSPOINTS_PN_ONLY[@]} -gt 0 ]]; then
            echo "Processing ParticleNet-only mass points for Run2..."
            for channel in "${CHANNELs[@]}"; do
                parallel -j $NJOBS_RUN2 preprocess_sample {1} "$channel" {2} ::: "${ERAs_RUN2[@]}" ::: "${MASSPOINTS_PN_ONLY[@]}"
            done
        fi
    fi

    echo "Run2 preprocessing complete!"
fi

# Run3 processing with scaled signal
if [[ "$MODE" == "all" || "$MODE" == "run3-scaled" ]]; then
    echo ""
    echo "============================================================"
    echo "Processing Run3 eras (signal scaled from 2018)..."
    echo "============================================================"

    if [[ "$USE_CONDOR" == "true" ]]; then
        # Wait for Run2 jobs to complete when using --mode all
        if [[ "$MODE" == "all" && -n "$RUN2_CLUSTER_ID" ]]; then
            echo ""
            echo "Waiting for Run2 jobs (cluster $RUN2_CLUSTER_ID) to complete..."
            echo "Run3 signal scaling depends on preprocessed 2018 samples."
            echo ""
            while condor_q "$RUN2_CLUSTER_ID" 2>/dev/null | grep -qE "[0-9]+ jobs"; do
                remaining=$(condor_q "$RUN2_CLUSTER_ID" 2>/dev/null | grep -oE "[0-9]+ jobs" | head -1)
                echo "  $(date '+%H:%M:%S') - $remaining remaining..."
                sleep 60
            done
            echo "Run2 jobs completed. Proceeding with Run3..."
        fi
        echo "Submitting Run3 (scaled) jobs to HTCondor..."
        submit_condor_jobs ERAs_RUN3 ALL_MASSPOINTS "--scale-from-run2"
    else
        # Baseline mass points
        echo "Processing Baseline mass points for Run3 (scaled signal)..."
        for channel in "${CHANNELs[@]}"; do
            parallel -j $NJOBS_RUN3 preprocess_sample {1} "$channel" {2} "--scale-from-run2" ::: "${ERAs_RUN3[@]}" ::: "${MASSPOINTs_BASELINE[@]}"
        done

        # ParticleNet mass points (that are not in Baseline)
        if [[ ${#MASSPOINTS_PN_ONLY[@]} -gt 0 ]]; then
            echo "Processing ParticleNet-only mass points for Run3 (scaled signal)..."
            for channel in "${CHANNELs[@]}"; do
                parallel -j $NJOBS_RUN3 preprocess_sample {1} "$channel" {2} "--scale-from-run2" ::: "${ERAs_RUN3[@]}" ::: "${MASSPOINTS_PN_ONLY[@]}"
            done
        fi
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
        submit_condor_jobs ERAs_RUN3 ALL_MASSPOINTS ""
    else
        # Baseline mass points
        echo "Processing Baseline mass points for Run3..."
        for channel in "${CHANNELs[@]}"; do
            parallel -j $NJOBS_RUN3 preprocess_sample {1} "$channel" {2} ::: "${ERAs_RUN3[@]}" ::: "${MASSPOINTs_BASELINE[@]}"
        done

        # ParticleNet mass points (that are not in Baseline)
        if [[ ${#MASSPOINTS_PN_ONLY[@]} -gt 0 ]]; then
            echo "Processing ParticleNet-only mass points for Run3..."
            for channel in "${CHANNELs[@]}"; do
                parallel -j $NJOBS_RUN3 preprocess_sample {1} "$channel" {2} ::: "${ERAs_RUN3[@]}" ::: "${MASSPOINTS_PN_ONLY[@]}"
            done
        fi
    fi

    echo "Run3 (real signal) preprocessing complete!"
fi

echo ""
echo "============================================================"
echo "All preprocessing complete!"
echo "============================================================"
