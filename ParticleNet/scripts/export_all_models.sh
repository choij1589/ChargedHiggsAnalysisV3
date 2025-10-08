#!/bin/bash
# Export all ParticleNet models to TorchScript format
#
# This script exports all 6 trained multiclass ParticleNet models
# (3 signal mass points × 2 channels) from fold-3 checkpoints.
#
# Usage:
#   bash scripts/export_all_models.sh
#
# Author: Claude Code + Human
# Date: 2025-10-08

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "======================================================================"
echo "  TorchScript Model Export for ParticleNet"
echo "======================================================================"
echo ""

# Configuration
CHANNELS=("Run1E2Mu" "Run3Mu")
SIGNALS=("MHc160_MA85" "MHc130_MA90" "MHc100_MA95")
FOLD=3
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

echo "Project directory: $PROJECT_DIR"
echo "Python script: python/export_to_torchscript.py"
echo "Fold: $FOLD"
echo ""

# Check if export script exists
if [ ! -f "$PROJECT_DIR/python/export_to_torchscript.py" ]; then
    echo -e "${RED}Error: export_to_torchscript.py not found${NC}"
    exit 1
fi

# Change to project directory
cd "$PROJECT_DIR"

# Export counters
TOTAL_MODELS=$((${#CHANNELS[@]} * ${#SIGNALS[@]}))
SUCCESS_COUNT=0
FAILED_MODELS=()

echo "Models to export: $TOTAL_MODELS"
echo ""

# Export each model
for channel in "${CHANNELS[@]}"; do
    for signal in "${SIGNALS[@]}"; do
        MODEL_ID="${channel}/${signal}/fold-${FOLD}"
        echo "----------------------------------------------------------------------"
        echo -e "${YELLOW}Exporting: $MODEL_ID${NC}"
        echo "----------------------------------------------------------------------"

        if python python/export_to_torchscript.py \
            --channel "$channel" \
            --signal "$signal" \
            --fold "$FOLD"; then
            echo -e "${GREEN}✓ Successfully exported $MODEL_ID${NC}"
            ((SUCCESS_COUNT++))
        else
            echo -e "${RED}✗ Failed to export $MODEL_ID${NC}"
            FAILED_MODELS+=("$MODEL_ID")
        fi

        echo ""
    done
done

# Summary
echo "======================================================================"
echo "  EXPORT SUMMARY"
echo "======================================================================"
echo "Total models:  $TOTAL_MODELS"
echo -e "Successful:    ${GREEN}$SUCCESS_COUNT${NC}"
echo -e "Failed:        ${RED}${#FAILED_MODELS[@]}${NC}"
echo ""

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "Failed models:"
    for model in "${FAILED_MODELS[@]}"; do
        echo -e "  ${RED}✗${NC} $model"
    done
    echo ""
    exit 1
else
    echo -e "${GREEN}✓ All models exported successfully!${NC}"
    echo ""

    # List exported models
    echo "Exported TorchScript models:"
    find results_bjets -name "*_scripted.pt" -type f | sort | while read -r file; do
        size=$(du -h "$file" | cut -f1)
        echo "  ✓ $file ($size)"
    done
    echo ""

    echo "Next steps:"
    echo "  1. Validate exports: python python/validate_torchscript_export.py --all --fold $FOLD"
    echo "  2. Review C++ integration: cat docs/torchscript_cpp_integration.md"
    echo ""
fi

exit 0
