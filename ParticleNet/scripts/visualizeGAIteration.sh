#!/bin/bash

# Wrapper script for visualizing GA iteration results
# Usage: ./scripts/visualizeGAIteration.sh <signal> <channel> <iteration> <device> [OPTIONS]
#
# Example:
#   ./scripts/visualizeGAIteration.sh MHc130_MA90 Run1E2Mu 3 cuda:0 --pilot
#   ./scripts/visualizeGAIteration.sh MHc130_MA90 Run1E2Mu 3 cuda:0 --parallel --jobs 8
#   ./scripts/visualizeGAIteration.sh MHc130_MA90 Run1E2Mu 3 cuda:0 --skip-eval

if [ $# -lt 4 ]; then
    echo "Usage: $0 <signal> <channel> <iteration> <device> [OPTIONS]"
    echo ""
    echo "Arguments:"
    echo "  signal     : Signal name (e.g., MHc130_MA90)"
    echo "  channel    : Channel (e.g., Run1E2Mu, Run3Mu)"
    echo "  iteration  : GA iteration number (e.g., 0, 1, 2, 3)"
    echo "  device     : Device to use (e.g., cuda:0, cpu)"
    echo ""
    echo "Options:"
    echo "  --pilot      : Use pilot datasets"
    echo "  --skip-eval  : Skip evaluation, reuse existing histograms"
    echo "  --parallel   : Process models in parallel"
    echo "  --jobs N     : Number of parallel jobs (default: 4)"
    echo ""
    echo "Examples:"
    echo "  $0 MHc130_MA90 Run1E2Mu 3 cuda:0 --pilot"
    echo "  $0 MHc130_MA90 Run1E2Mu 3 cuda:0 --parallel --jobs 8"
    echo "  $0 MHc130_MA90 Run1E2Mu 3 cuda:0 --skip-eval"
    exit 1
fi

SIGNAL=$1
CHANNEL=$2
ITERATION=$3
DEVICE=$4
shift 4

# Parse optional flags
PILOT_FLAG=""
SKIP_EVAL_FLAG=""
PARALLEL=false
JOBS=4

while [[ $# -gt 0 ]]; do
    case $1 in
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

# Print configuration
echo "======================================================================"
echo "Visualizing GA Iteration Results"
echo "======================================================================"
echo "Signal       : $SIGNAL"
echo "Channel      : $CHANNEL"
echo "Iteration    : $ITERATION"
echo "Device       : $DEVICE"
echo "Pilot mode   : ${PILOT_FLAG:-false}"
echo "Skip eval    : ${SKIP_EVAL_FLAG:-false}"
echo "Parallel     : $PARALLEL"
if [ "$PARALLEL" = true ]; then
    echo "Parallel jobs: $JOBS"
fi
echo "======================================================================"
echo ""

if [ "$PARALLEL" = true ]; then
    # Parallel execution: process each model independently
    echo "Processing models in parallel with $JOBS jobs..."
    echo ""

    # Get list of models from model_info.csv
    JSON_DIR="GAOptim_bjets/${CHANNEL}/multiclass/TTToHcToWAToMuMu-${SIGNAL}/GA-iter${ITERATION}/json"
    MODEL_INFO="${JSON_DIR}/model_info.csv"

    if [ ! -f "$MODEL_INFO" ]; then
        echo "ERROR: Model info file not found: $MODEL_INFO"
        exit 1
    fi

    # Extract model indices (skip header, get model column)
    MODEL_INDICES=$(tail -n +2 "$MODEL_INFO" | cut -d',' -f1 | sed 's/model//g')

    # Process each model in parallel
    echo "$MODEL_INDICES" | parallel -j $JOBS --progress \
        "python3 python/visualizeGAIteration.py \
            --signal '$SIGNAL' \
            --channel '$CHANNEL' \
            --iteration $ITERATION \
            --device '$DEVICE' \
            --model-indices {} \
            $PILOT_FLAG $SKIP_EVAL_FLAG"

    if [ $? -ne 0 ]; then
        echo ""
        echo "======================================================================"
        echo "ERROR: Parallel processing failed!"
        echo "======================================================================"
        exit 1
    fi

    # Run summary generation (processes all models but skips individual evaluation)
    echo ""
    echo "Generating summary plots..."
    python3 python/visualizeGAIteration.py \
        --signal "$SIGNAL" \
        --channel "$CHANNEL" \
        --iteration $ITERATION \
        --device "$DEVICE" \
        --skip-eval \
        $PILOT_FLAG
else
    # Sequential execution (original behavior)
    python3 python/visualizeGAIteration.py \
        --signal "$SIGNAL" \
        --channel "$CHANNEL" \
        --iteration $ITERATION \
        --device "$DEVICE" \
        $PILOT_FLAG $SKIP_EVAL_FLAG
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================"
    echo "Visualization completed successfully!"
    echo "======================================================================"
    echo ""
    echo "Check the following directories for results:"
    echo "  - GAOptim_bjets/${CHANNEL}/multiclass/TTToHcToWAToMuMu-${SIGNAL}/GA-iter${ITERATION}/overfitting_diagnostics/"
    echo "  - GAOptim_bjets/${CHANNEL}/multiclass/TTToHcToWAToMuMu-${SIGNAL}/GA-iter${ITERATION}/plots/"
    echo ""
else
    echo ""
    echo "======================================================================"
    echo "ERROR: Visualization failed!"
    echo "======================================================================"
    exit 1
fi
