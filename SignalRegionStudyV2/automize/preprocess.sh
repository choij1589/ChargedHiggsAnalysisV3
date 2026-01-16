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

# Parse command line arguments
MODE="all"  # Options: all, run2, run3, run3-scaled

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--mode <all|run2|run3|run3-scaled>]"
            echo ""
            echo "Modes:"
            echo "  all          - Process Run2 and Run3 (Run3 with scaled signal)"
            echo "  run2         - Process Run2 only"
            echo "  run3         - Process Run3 with real signal MC (if available)"
            echo "  run3-scaled  - Process Run3 with signal scaled from 2018"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "SignalRegionStudyV2 Preprocessing"
echo "Mode: $MODE"
echo "============================================================"

# Run2 processing
if [[ "$MODE" == "all" || "$MODE" == "run2" ]]; then
    echo ""
    echo "============================================================"
    echo "Processing Run2 eras..."
    echo "============================================================"

    # Baseline mass points
    echo "Processing Baseline mass points for Run2..."
    for channel in "${CHANNELs[@]}"; do
        parallel -j $NJOBS_RUN2 preprocess_sample {1} "$channel" {2} ::: "${ERAs_RUN2[@]}" ::: "${MASSPOINTs_BASELINE[@]}"
    done

    # ParticleNet mass points (that are not in Baseline)
    echo "Processing ParticleNet-only mass points for Run2..."
    MASSPOINTS_PN_ONLY=()
    for mp in "${MASSPOINTs_PARTICLENET[@]}"; do
        if [[ ! " ${MASSPOINTs_BASELINE[*]} " =~ " ${mp} " ]]; then
            MASSPOINTS_PN_ONLY+=("$mp")
        fi
    done

    if [[ ${#MASSPOINTS_PN_ONLY[@]} -gt 0 ]]; then
        for channel in "${CHANNELs[@]}"; do
            parallel -j $NJOBS_RUN2 preprocess_sample {1} "$channel" {2} ::: "${ERAs_RUN2[@]}" ::: "${MASSPOINTS_PN_ONLY[@]}"
        done
    fi

    echo "Run2 preprocessing complete!"
fi

# Run3 processing with scaled signal
if [[ "$MODE" == "all" || "$MODE" == "run3-scaled" ]]; then
    echo ""
    echo "============================================================"
    echo "Processing Run3 eras (signal scaled from 2018)..."
    echo "============================================================"

    # Baseline mass points
    echo "Processing Baseline mass points for Run3 (scaled signal)..."
    for channel in "${CHANNELs[@]}"; do
        parallel -j $NJOBS_RUN3 preprocess_sample {1} "$channel" {2} "--scale-from-run2" ::: "${ERAs_RUN3[@]}" ::: "${MASSPOINTs_BASELINE[@]}"
    done

    # ParticleNet mass points (that are not in Baseline)
    echo "Processing ParticleNet-only mass points for Run3 (scaled signal)..."
    if [[ ${#MASSPOINTS_PN_ONLY[@]} -gt 0 ]]; then
        for channel in "${CHANNELs[@]}"; do
            parallel -j $NJOBS_RUN3 preprocess_sample {1} "$channel" {2} "--scale-from-run2" ::: "${ERAs_RUN3[@]}" ::: "${MASSPOINTS_PN_ONLY[@]}"
        done
    fi

    echo "Run3 (scaled signal) preprocessing complete!"
fi

# Run3 processing with real signal (if available)
if [[ "$MODE" == "run3" ]]; then
    echo ""
    echo "============================================================"
    echo "Processing Run3 eras (real signal MC)..."
    echo "============================================================"

    # Baseline mass points
    echo "Processing Baseline mass points for Run3..."
    for channel in "${CHANNELs[@]}"; do
        parallel -j $NJOBS_RUN3 preprocess_sample {1} "$channel" {2} ::: "${ERAs_RUN3[@]}" ::: "${MASSPOINTs_BASELINE[@]}"
    done

    # ParticleNet mass points (that are not in Baseline)
    echo "Processing ParticleNet-only mass points for Run3..."
    if [[ ${#MASSPOINTS_PN_ONLY[@]} -gt 0 ]]; then
        for channel in "${CHANNELs[@]}"; do
            parallel -j $NJOBS_RUN3 preprocess_sample {1} "$channel" {2} ::: "${ERAs_RUN3[@]}" ::: "${MASSPOINTS_PN_ONLY[@]}"
        done
    fi

    echo "Run3 (real signal) preprocessing complete!"
fi

echo ""
echo "============================================================"
echo "All preprocessing complete!"
echo "============================================================"
