#!/bin/bash

# Wrapper script for visualizing GA iteration results
# Usage: ./scripts/visualizeGAIteration.sh <signal> <channel> <iteration> <device> [OPTIONS]
#
# Example:
#   ./scripts/visualizeGAIteration.sh MHc130_MA90 Run1E2Mu 3 cuda:0 --pilot
#   ./scripts/visualizeGAIteration.sh MHc130_MA90 Run1E2Mu 3 cuda:0 --parallel --jobs 8
#   ./scripts/visualizeGAIteration.sh MHc130_MA90 Run1E2Mu 3 cuda:0 --skip-eval
#   ./scripts/visualizeGAIteration.sh MHc130_MA90 Run1E2Mu 3 cuda:0 --input GAOptim_bjets_maxepoch50
#
# Multiple signals with different devices:
#   ./scripts/visualizeGAIteration.sh "MHc130_MA90,MHc160_MA85" Run1E2Mu 3 "cuda:0,cuda:1" --pilot
#   ./scripts/visualizeGAIteration.sh "MHc130_MA90,MHc160_MA85,MHc145_MA92" Run1E2Mu 3 "cuda:0,cuda:1" --parallel --jobs 8

if [ $# -lt 4 ]; then
    echo "Usage: $0 <signal> <channel> <iteration> <device> [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  signal     : Signal name(s) - single or comma-separated"
    echo "               e.g., MHc130_MA90  OR  'MHc130_MA90,MHc160_MA85'"
    echo "  channel    : Channel (e.g., Run1E2Mu, Run3Mu)"
    echo "  iteration  : GA iteration number (e.g., 0, 1, 2, 3)"
    echo "  device     : Device(s) - single or comma-separated"
    echo "               e.g., cuda:0  OR  'cuda:0,cuda:1,cuda:2'"
    echo "               If fewer devices than signals, will use round-robin"
    echo ""
    echo "Options:"
    echo "  --input DIR  : Input directory path (default: GAOptim_bjets)"
    echo "  --pilot      : Use pilot datasets"
    echo "  --skip-eval  : Skip evaluation, reuse existing histograms"
    echo "  --parallel   : Process models in parallel"
    echo "  --jobs N     : Number of parallel jobs (default: 4)"
    echo ""
    echo "Examples:"
    echo "  # Single signal"
    echo "  $0 MHc130_MA90 Run1E2Mu 3 cuda:0 --pilot"
    echo "  $0 MHc130_MA90 Run1E2Mu 3 cuda:0 --parallel --jobs 8"
    echo ""
    echo "  # Multiple signals on different devices"
    echo "  $0 'MHc130_MA90,MHc160_MA85' Run1E2Mu 3 'cuda:0,cuda:1' --pilot"
    echo "  $0 'MHc130_MA90,MHc160_MA85,MHc145_MA92' Run1E2Mu 3 'cuda:0,cuda:1' --parallel --jobs 8"
    exit 1
fi

SIGNAL_INPUT=$1
CHANNEL=$2
ITERATION=$3
DEVICE_INPUT=$4
shift 4

# Parse optional flags
PILOT_FLAG=""
SKIP_EVAL_FLAG=""
PARALLEL=false
JOBS=4
INPUT_DIR="GAOptim_bjets"

while [[ $# -gt 0 ]]; do
    case $1 in
        --input)
            INPUT_DIR=$2
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
            PARALLEL=true
            shift
            ;;
        --jobs)
            JOBS=$2
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Setup paths
export PATH="${PWD}/python:${PATH}"

# Parse comma-separated signals and devices
IFS=',' read -ra SIGNALS <<< "$SIGNAL_INPUT"
IFS=',' read -ra DEVICES <<< "$DEVICE_INPUT"

# Get number of signals and devices
NUM_SIGNALS=${#SIGNALS[@]}
NUM_DEVICES=${#DEVICES[@]}

# Print configuration
echo "======================================================================"
echo "Visualizing GA Iteration Results"
echo "======================================================================"
echo "Signal(s)    : ${SIGNALS[*]}"
echo "Channel      : $CHANNEL"
echo "Iteration    : $ITERATION"
echo "Device(s)    : ${DEVICES[*]}"
echo "Input dir    : $INPUT_DIR"
echo "Pilot mode   : ${PILOT_FLAG:-false}"
echo "Skip eval    : ${SKIP_EVAL_FLAG:-false}"
echo "Parallel     : $PARALLEL"
if [ "$PARALLEL" = true ]; then
    echo "Parallel jobs: $JOBS"
fi
echo "======================================================================"
echo ""

# Check if we have multiple signals
if [ $NUM_SIGNALS -gt 1 ]; then
    echo "Processing $NUM_SIGNALS signals across $NUM_DEVICES device(s)..."
    echo ""
fi

# Function to process a single signal
process_signal() {
    local SIGNAL=$1
    local DEVICE=$2
    local LOG_FILE=$3

    echo "[$SIGNAL on $DEVICE] Starting..." | tee -a "$LOG_FILE"

    if [ "$PARALLEL" = true ]; then
        # Parallel execution: process each model independently
        echo "[$SIGNAL on $DEVICE] Processing models in parallel with $JOBS jobs..." | tee -a "$LOG_FILE"

        # Get list of models from model_info.csv
        JSON_DIR="${INPUT_DIR}/${CHANNEL}/multiclass/TTToHcToWAToMuMu-${SIGNAL}/GA-iter${ITERATION}/json"
        MODEL_INFO="${JSON_DIR}/model_info.csv"

        if [ ! -f "$MODEL_INFO" ]; then
            echo "[$SIGNAL on $DEVICE] ERROR: Model info file not found: $MODEL_INFO" | tee -a "$LOG_FILE"
            return 1
        fi

        # Extract model indices (skip header, get model column)
        MODEL_INDICES=$(tail -n +2 "$MODEL_INFO" | cut -d',' -f1 | sed 's/model//g')

        # Process each model in parallel
        echo "$MODEL_INDICES" | parallel -j $JOBS \
            "python3 python/visualizeGAIteration.py \
                --signal '$SIGNAL' \
                --channel '$CHANNEL' \
                --iteration $ITERATION \
                --device '$DEVICE' \
                --input '$INPUT_DIR' \
                --model-indices {} \
                $PILOT_FLAG $SKIP_EVAL_FLAG" >> "$LOG_FILE" 2>&1

        if [ $? -ne 0 ]; then
            echo "[$SIGNAL on $DEVICE] ERROR: Parallel processing failed!" | tee -a "$LOG_FILE"
            return 1
        fi

        # Run summary generation (processes all models but skips individual evaluation)
        echo "[$SIGNAL on $DEVICE] Generating summary plots..." | tee -a "$LOG_FILE"
        python3 python/visualizeGAIteration.py \
            --signal "$SIGNAL" \
            --channel "$CHANNEL" \
            --iteration $ITERATION \
            --device "$DEVICE" \
            --input "$INPUT_DIR" \
            --skip-eval \
            $PILOT_FLAG >> "$LOG_FILE" 2>&1
    else
        # Sequential execution (original behavior)
        python3 python/visualizeGAIteration.py \
            --signal "$SIGNAL" \
            --channel "$CHANNEL" \
            --iteration $ITERATION \
            --device "$DEVICE" \
            --input "$INPUT_DIR" \
            $PILOT_FLAG $SKIP_EVAL_FLAG >> "$LOG_FILE" 2>&1
    fi

    local EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "[$SIGNAL on $DEVICE] Completed successfully!" | tee -a "$LOG_FILE"
        return 0
    else
        echo "[$SIGNAL on $DEVICE] ERROR: Visualization failed!" | tee -a "$LOG_FILE"
        return 1
    fi
}

# Export function and variables for parallel execution
export -f process_signal
export CHANNEL ITERATION INPUT_DIR PILOT_FLAG SKIP_EVAL_FLAG PARALLEL JOBS

# Process signals (either single or multiple)
if [ $NUM_SIGNALS -eq 1 ]; then
    # Single signal - original behavior
    SIGNAL=${SIGNALS[0]}
    DEVICE=${DEVICES[0]}
    LOG_FILE="${INPUT_DIR}/${CHANNEL}/multiclass/TTToHcToWAToMuMu-${SIGNAL}/GA-iter${ITERATION}/visualization.log"

    # Create log directory if needed
    mkdir -p "$(dirname "$LOG_FILE")"

    # Process the signal
    process_signal "$SIGNAL" "$DEVICE" "$LOG_FILE"
    FINAL_EXIT_CODE=$?

else
    # Multiple signals - distribute across devices
    echo "Launching visualization for $NUM_SIGNALS signals..."
    echo ""

    # Array to track PIDs and signals
    declare -a PIDS
    declare -a SIGNAL_NAMES
    declare -a LOG_FILES

    # Launch each signal in background
    for i in "${!SIGNALS[@]}"; do
        SIGNAL="${SIGNALS[$i]}"
        # Use round-robin for device assignment
        DEVICE_IDX=$((i % NUM_DEVICES))
        DEVICE="${DEVICES[$DEVICE_IDX]}"

        LOG_FILE="${INPUT_DIR}/${CHANNEL}/multiclass/TTToHcToWAToMuMu-${SIGNAL}/GA-iter${ITERATION}/visualization.log"

        # Create log directory if needed
        mkdir -p "$(dirname "$LOG_FILE")"

        echo "Launching: $SIGNAL on $DEVICE (log: $LOG_FILE)"

        # Launch in background
        process_signal "$SIGNAL" "$DEVICE" "$LOG_FILE" &
        PIDS+=($!)
        SIGNAL_NAMES+=("$SIGNAL")
        LOG_FILES+=("$LOG_FILE")

        # Small delay to avoid race conditions
        sleep 2
    done

    echo ""
    echo "All signals launched. Waiting for completion..."
    echo "======================================================================"
    echo ""

    # Wait for all processes and collect exit codes
    FINAL_EXIT_CODE=0
    for i in "${!PIDS[@]}"; do
        PID=${PIDS[$i]}
        SIGNAL="${SIGNAL_NAMES[$i]}"

        wait $PID
        EXIT_CODE=$?

        if [ $EXIT_CODE -eq 0 ]; then
            echo "✓ $SIGNAL completed successfully"
        else
            echo "✗ $SIGNAL failed (exit code: $EXIT_CODE)"
            FINAL_EXIT_CODE=1
        fi
    done
fi

# Final status report
echo ""
echo "======================================================================"
if [ $FINAL_EXIT_CODE -eq 0 ]; then
    echo "All visualizations completed successfully!"
    echo "======================================================================"
    echo ""
    echo "Check the following directories for results:"
    for SIGNAL in "${SIGNALS[@]}"; do
        echo "  - ${INPUT_DIR}/${CHANNEL}/multiclass/TTToHcToWAToMuMu-${SIGNAL}/GA-iter${ITERATION}/overfitting_diagnostics/"
        echo "  - ${INPUT_DIR}/${CHANNEL}/multiclass/TTToHcToWAToMuMu-${SIGNAL}/GA-iter${ITERATION}/plots/"
    done
    echo ""
else
    echo "ERROR: One or more visualizations failed!"
    echo "======================================================================"
    echo ""
    echo "Check log files for details:"
    for SIGNAL in "${SIGNALS[@]}"; do
        echo "  - ${INPUT_DIR}/${CHANNEL}/multiclass/TTToHcToWAToMuMu-${SIGNAL}/GA-iter${ITERATION}/visualization.log"
    done
    echo ""
    exit 1
fi
