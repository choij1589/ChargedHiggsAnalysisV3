#!/bin/bash

# Generate discrimination variable heatmaps and split JSON files
# This script performs two operations:
# 1. Splits combined JSON into individual mass point files
# 2. Generates 27 heatmaps (one per discrimination variable)

ERA="2018"
CHANNEL="SR3Mu"

# Setup environment
export PATH="${PWD}/python:${PATH}"

summarizeDiscrimination.py --era ${ERA} --channel ${CHANNEL}

echo "======================================================================"
echo "Discrimination Variable Analysis - Heatmap Generation"
echo "======================================================================"
echo "Era: ${ERA}"
echo "Channel: ${CHANNEL}"
echo "======================================================================"

# Check if input file exists
INPUT_JSON="results/${ERA}/${CHANNEL}/discrimination_summary.json"
if [ ! -f "${INPUT_JSON}" ]; then
    echo "ERROR: Input file not found: ${INPUT_JSON}"
    echo "Please run summarizeDiscrimination.py first to generate the input file."
    exit 1
fi

echo ""
echo "Step 1: Splitting JSON into individual mass point files..."
echo "----------------------------------------------------------------------"
python python/splitDiscriminationJSON.py \
    --input ${INPUT_JSON} \
    --output results/${ERA}/${CHANNEL}/masspoints/

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to split JSON files"
    exit 1
fi

echo ""
echo "Step 2: Generating 27 discrimination variable heatmaps..."
echo "----------------------------------------------------------------------"
python python/plotDiscriminationHeatmap.py \
    --input ${INPUT_JSON} \
    --era ${ERA} \
    --channel ${CHANNEL}

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to generate heatmaps"
    exit 1
fi

echo ""
echo "======================================================================"
echo "All tasks completed successfully!"
echo "======================================================================"
echo ""
echo "Output locations:"
echo "  - Individual JSON files: results/${ERA}/${CHANNEL}/masspoints/"
echo "  - Heatmap images: plots/DiscriminationHeatmaps/${ERA}/${CHANNEL}/"
echo ""
echo "Summary:"
echo "  - 36 JSON files created (one per mass point)"
echo "  - 27 heatmap PNG files created (one per discrimination variable)"
echo "======================================================================"
