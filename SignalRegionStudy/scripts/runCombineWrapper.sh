#!/bin/bash
# Wrapper script for parallel execution of combine workflow
# Arguments: $1 = masspoint (e.g., MHc70_MA15), $2 = method (e.g., Baseline)

set -e  # Exit immediately if any command fails

masspoint=$1
method=$2

# Debug output
echo "=========================================="
echo "runCombineWrapper.sh called with:"
echo "  masspoint: $masspoint"
echo "  method: $method"
echo "  pwd: $(pwd)"
echo "=========================================="

# Validate arguments
if [ -z "$masspoint" ] || [ -z "$method" ]; then
    echo "ERROR: Missing arguments"
    echo "Usage: $0 <masspoint> <method>"
    exit 1
fi

# Set up environment
SCRIPT_DIR="/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/SignalRegionStudy"
cd "$SCRIPT_DIR"
source "$SCRIPT_DIR/setup.sh"

# Define eras
ERAs=("2016preVFP" "2016postVFP" "2017" "2018")

echo "Running combine workflow for $masspoint with method $method"

# 1E2Mu channel first (needed for fitting)
echo "Processing SR1E2Mu channel..."
for era in ${ERAs[@]}; do
    echo "  Era: $era"
    "$SCRIPT_DIR/scripts/prepareCombine.sh" $era SR1E2Mu $masspoint $method
    "$SCRIPT_DIR/scripts/runCombine.sh" $era SR1E2Mu $masspoint $method
done

# 3Mu channel
echo "Processing SR3Mu channel..."
for era in ${ERAs[@]}; do
    echo "  Era: $era"
    "$SCRIPT_DIR/scripts/prepareCombine.sh" $era SR3Mu $masspoint $method
    "$SCRIPT_DIR/scripts/runCombine.sh" $era SR3Mu $masspoint $method
done

# Combined channels
echo "Processing Combined channels..."
for era in ${ERAs[@]}; do
    echo "  Era: $era"
    "$SCRIPT_DIR/scripts/runCombine.sh" $era Combined $masspoint $method
done

# Full Run2
echo "Processing Full Run2..."
"$SCRIPT_DIR/scripts/runCombine.sh" FullRun2 SR1E2Mu $masspoint $method
"$SCRIPT_DIR/scripts/runCombine.sh" FullRun2 SR3Mu $masspoint $method
"$SCRIPT_DIR/scripts/runCombine.sh" FullRun2 Combined $masspoint $method

echo "=========================================="
echo "Completed combine workflow for $masspoint"
echo "=========================================="
