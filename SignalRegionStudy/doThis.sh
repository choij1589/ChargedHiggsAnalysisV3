#!/bin/bash
# Running combine for all mass points, all channels, all eras

# Set working directory
SCRIPT_DIR="/home/choij/Sync/workspace/ChargedHiggsAnalysisV3/SignalRegionStudy"
cd "$SCRIPT_DIR"

MASSPOINTs=(
    "MHc70_MA15" "MHc70_MA18" "MHc70_MA40" "MHc70_MA55" "MHc70_MA65"
    "MHc85_MA15" "MHc85_MA70" "MHc85_MA80"      # 85_21 is missng in 16a
    "MHc100_MA15" "MHc100_MA24" "MHc100_MA60" "MHc100_MA75" "MHc100_MA95"
    "MHc115_MA15" "MHc115_MA27" "MHc115_MA87" "MHc115_MA110"
    "MHc130_MA15" "MHc130_MA30" "MHc130_MA55" "MHc130_MA83" "MHc130_MA90" "MHc130_MA100" "MHc130_MA125"
    "MHc145_MA15" "MHc145_MA35" "MHc145_MA92" "MHc145_MA140"
    "MHc160_MA15" "MHc160_MA50" "MHc160_MA85" "MHc160_MA98" "MHc160_MA120" "MHc160_MA135" "MHc160_MA155"
)
#MPForOptimized=("MHc100_MA95" "MHc130_MA90" "MHc160_MA85" "MHc115_MA87" "MHc145_MA92", "MHc160_MA98")
MPForOptimized=("MHc145_MA92")
# No longer need combine function - using wrapper script instead

echo "Cleaning up"
#rm -rf samples
#rm -rf templates

# Use the wrapper script with parallel
# The wrapper script handles all environment setup internally
#parallel -j 18 "$SCRIPT_DIR/scripts/runCombineWrapper.sh" {1} {2} ::: "${MASSPOINTs[@]}" ::: "Baseline"
parallel -j 18 "$SCRIPT_DIR/scripts/runCombineWrapper.sh" {1} {2} ::: "${MPForOptimized[@]}" ::: "ParticleNet"
