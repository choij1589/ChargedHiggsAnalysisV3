#!/bin/bash
################################################################################
# visualizeGAIteration.sh - Wrapper script for GA iteration visualization
#
# Generates comprehensive visualization and overfitting analysis for all models
# from a GA iteration in ParticleNetMD.
#
# Features:
#   - Multi-signal support with comma-separated signals
#   - Multi-device support with round-robin assignment
#   - Parallel processing of individual models
#   - Skip-eval mode for regenerating plots from existing results
#
# Usage:
#   ./scripts/visualizeGAIteration.sh <signal> <channel> <iteration> <device> [OPTIONS]
#
# Arguments:
#   signal      Signal name or comma-separated list (e.g., "MHc130_MA90" or "MHc130_MA90,MHc160_MA85")
#   channel     Channel name (e.g., "Combined", "Run1E2Mu", "Run3Mu")
#   iteration   GA iteration number (e.g., 0, 1, 2)
#   device      CUDA device or comma-separated list (e.g., "cuda:0" or "cuda:0,cuda:1")
#
# Options:
#   --input DIR     Input directory (default: GAOptim)
#   --pilot         Use pilot datasets
#   --skip-eval     Skip evaluation, reuse existing histograms
#   --parallel      Process models in parallel within each signal
#   --jobs N        Number of parallel jobs (default: 4)
#
# Examples:
#   # Single signal, single device
#   ./scripts/visualizeGAIteration.sh MHc130_MA90 Combined 0 cuda:0
#
#   # Multiple signals, single device (sequential)
#   ./scripts/visualizeGAIteration.sh 'MHc130_MA90,MHc160_MA85' Combined 0 cuda:0
#
#   # Multiple signals, multiple devices (parallel, round-robin assignment)
#   ./scripts/visualizeGAIteration.sh 'MHc130_MA90,MHc160_MA85' Combined 0 'cuda:0,cuda:1'
#
#   # With pilot datasets
#   ./scripts/visualizeGAIteration.sh MHc130_MA90 Combined 0 cuda:0 --pilot
#
#   # Skip evaluation (regenerate plots from existing results)
#   ./scripts/visualizeGAIteration.sh MHc130_MA90 Combined 0 cuda:0 --skip-eval
#
#   # Parallel model processing
#   ./scripts/visualizeGAIteration.sh MHc130_MA90 Combined 0 cuda:0 --parallel --jobs 4
#
################################################################################

set -e

# Add python directory to PATH
export PATH="${PWD}/python:${PATH}"

# Parse arguments
if [ $# -lt 4 ]; then
    echo "Usage: $0 <signal> <channel> <iteration> <device> [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --input DIR     Input directory (default: GAOptim)"
    echo "  --pilot         Use pilot datasets"
    echo "  --skip-eval     Skip evaluation, reuse existing histograms"
    echo "  --parallel      Process models in parallel within each signal"
    echo "  --jobs N        Number of parallel jobs (default: 4)"
    exit 1
fi

SIGNAL="$1"
CHANNEL="$2"
ITERATION="$3"
DEVICE="$4"
shift 4

# Default options
INPUT_DIR="GAOptim"
PILOT_FLAG=""
SKIP_EVAL_FLAG=""
PARALLEL_MODE=false
N_JOBS=4

# Parse optional arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_DIR="$2"
            shift 2
            ;;
        --pilot)
            PILOT_FLAG="--pilot"
            shift
            ;;
        --skip-eval)
            SKIP_EVAL_FLAG="--skip-eval"
            shift
            ;;
        --parallel)
            PARALLEL_MODE=true
            shift
            ;;
        --jobs)
            N_JOBS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Convert comma-separated signals and devices to arrays
IFS=',' read -ra SIGNALS <<< "$SIGNAL"
IFS=',' read -ra DEVICES <<< "$DEVICE"

NUM_SIGNALS=${#SIGNALS[@]}
NUM_DEVICES=${#DEVICES[@]}

echo "================================================================================"
echo "GA ITERATION VISUALIZATION (ParticleNetMD)"
echo "================================================================================"
echo "Signals: ${SIGNALS[*]} (${NUM_SIGNALS} total)"
echo "Channel: ${CHANNEL}"
echo "Iteration: ${ITERATION}"
echo "Devices: ${DEVICES[*]} (${NUM_DEVICES} total)"
echo "Input directory: ${INPUT_DIR}"
echo "Pilot mode: ${PILOT_FLAG:-disabled}"
echo "Skip evaluation: ${SKIP_EVAL_FLAG:-disabled}"
echo "Parallel mode: ${PARALLEL_MODE}"
echo "Jobs: ${N_JOBS}"
echo "================================================================================"

# Function to process a single signal
process_signal() {
    local signal=$1
    local device=$2
    local channel=$3
    local iteration=$4
    local input_dir=$5
    local pilot_flag=$6
    local skip_eval_flag=$7
    local parallel_mode=$8
    local n_jobs=$9

    local signal_full="TTToHcToWAToMuMu-${signal}"
    local base_dir="${input_dir}/${channel}/multiclass/${signal_full}/GA-iter${iteration}"
    local log_file="${base_dir}/visualization.log"

    # Create output directory
    mkdir -p "${base_dir}"

    echo ""
    echo "Processing signal: ${signal}"
    echo "  Device: ${device}"
    echo "  Base directory: ${base_dir}"
    echo "  Log file: ${log_file}"

    if [ "$parallel_mode" = true ]; then
        # Parallel mode: process each model independently using GNU parallel
        local json_dir="${base_dir}/json"
        local model_info="${json_dir}/model_info.csv"

        if [ ! -f "$model_info" ]; then
            echo "  Error: model_info.csv not found at ${model_info}"
            return 1
        fi

        # Extract model indices from model_info.csv
        local model_indices=$(tail -n +2 "$model_info" | cut -d',' -f1 | sed 's/model//')
        local model_count=$(echo "$model_indices" | wc -w)

        echo "  Found ${model_count} models to process"
        echo "  Running in parallel with ${n_jobs} jobs"

        # Run models in parallel
        echo "$model_indices" | tr ' ' '\n' | \
            parallel --jobs "$n_jobs" --bar \
            "python3 python/visualizeGAIteration.py \
                --signal ${signal} \
                --channel ${channel} \
                --iteration ${iteration} \
                --device ${device} \
                --input ${input_dir} \
                ${pilot_flag} \
                ${skip_eval_flag} \
                --model-indices {}" 2>&1 | tee -a "${log_file}"

        # Generate summary plots (requires all models to be done)
        echo "  Generating summary plots..."
        python3 python/visualizeGAIteration.py \
            --signal "${signal}" \
            --channel "${channel}" \
            --iteration "${iteration}" \
            --device "${device}" \
            --input "${input_dir}" \
            ${pilot_flag} \
            --skip-eval 2>&1 | tee -a "${log_file}"

    else
        # Sequential mode: process all models in a single invocation
        python3 python/visualizeGAIteration.py \
            --signal "${signal}" \
            --channel "${channel}" \
            --iteration "${iteration}" \
            --device "${device}" \
            --input "${input_dir}" \
            ${pilot_flag} \
            ${skip_eval_flag} 2>&1 | tee "${log_file}"
    fi

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo "  Signal ${signal} completed successfully"
    else
        echo "  Signal ${signal} failed with exit code ${exit_code}"
    fi

    return $exit_code
}

# Export function and variables for parallel execution
export -f process_signal
export CHANNEL ITERATION INPUT_DIR PILOT_FLAG SKIP_EVAL_FLAG PARALLEL_MODE N_JOBS

# Process signals
if [ $NUM_SIGNALS -eq 1 ]; then
    # Single signal: run directly
    process_signal "${SIGNALS[0]}" "${DEVICES[0]}" "$CHANNEL" "$ITERATION" \
        "$INPUT_DIR" "$PILOT_FLAG" "$SKIP_EVAL_FLAG" "$PARALLEL_MODE" "$N_JOBS"
    exit_code=$?
else
    # Multiple signals: run in parallel with round-robin device assignment
    echo ""
    echo "Processing ${NUM_SIGNALS} signals with ${NUM_DEVICES} devices..."

    # Array to store background process PIDs
    declare -a pids=()
    declare -a signal_names=()

    for i in "${!SIGNALS[@]}"; do
        signal="${SIGNALS[$i]}"
        # Round-robin device assignment
        device_idx=$((i % NUM_DEVICES))
        device="${DEVICES[$device_idx]}"

        echo "Launching signal ${signal} on device ${device}..."

        # Run in background
        process_signal "$signal" "$device" "$CHANNEL" "$ITERATION" \
            "$INPUT_DIR" "$PILOT_FLAG" "$SKIP_EVAL_FLAG" "$PARALLEL_MODE" "$N_JOBS" &

        pids+=($!)
        signal_names+=("$signal")
    done

    # Wait for all background processes
    echo ""
    echo "Waiting for ${#pids[@]} background processes..."
    echo "PIDs: ${pids[*]}"

    exit_code=0
    for i in "${!pids[@]}"; do
        pid="${pids[$i]}"
        signal="${signal_names[$i]}"

        if wait "$pid"; then
            echo "Signal ${signal} (PID ${pid}) completed successfully"
        else
            status=$?
            echo "Signal ${signal} (PID ${pid}) failed with exit code ${status}"
            exit_code=1
        fi
    done
fi

echo ""
echo "================================================================================"
if [ $exit_code -eq 0 ]; then
    echo "ALL SIGNALS COMPLETED SUCCESSFULLY"
else
    echo "SOME SIGNALS FAILED - CHECK LOGS FOR DETAILS"
fi
echo "================================================================================"

exit $exit_code
