#!/bin/bash
# Script to visualize multi-class ParticleNet training results
#
# Supports both individual and grouped background models
# Organizes outputs in ParticleNet/plots/analysis/ directory

# Usage: ./visualizeResults.sh <CHANNEL> <SIGNAL> <FOLD> [--pilot] [--model-pattern]

if [ $# -lt 3 ]; then
    echo "Usage: $0 <CHANNEL> <SIGNAL> <FOLD> [--pilot] [--model-pattern PATTERN]"
    echo ""
    echo "Examples:"
    echo "  Individual backgrounds: $0 Run1E2Mu TTToHcToWAToMuMu-MHc130_MA100 0"
    echo "  Grouped backgrounds:    $0 Run1E2Mu TTToHcToWAToMuMu-MHc130_MA90 0 --pilot"
    echo "  Specific model pattern: $0 Run1E2Mu TTToHcToWAToMuMu-MHc130_MA90 0 --pilot --model-pattern '*3grp*'"
    echo ""
    echo "Arguments:"
    echo "  CHANNEL       Channel name (e.g., Run1E2Mu, Run3Mu)"
    echo "  SIGNAL        Signal sample name (e.g., TTToHcToWAToMuMu-MHc130_MA90)"
    echo "  FOLD          Cross-validation fold (0-4)"
    echo "  --pilot       Use pilot dataset results"
    echo "  --model-pattern  Pattern to match specific models (default: all models)"
    exit 1
fi

CHANNEL=$1
SIGNAL=$2
FOLD=$3
PILOT_MODE=false
MODEL_PATTERN=""

# Parse additional arguments
shift 3
while [[ $# -gt 0 ]]; do
    case $1 in
        --pilot)
            PILOT_MODE=true
            shift
            ;;
        --model-pattern)
            MODEL_PATTERN="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set paths based on pilot mode
if [ "$PILOT_MODE" = true ]; then
    BASE_DIR="${WORKDIR}/ParticleNet/results/multiclass/${CHANNEL}/${SIGNAL}/pilot"
else
    BASE_DIR="${WORKDIR}/ParticleNet/results/multiclass/${CHANNEL}/${SIGNAL}/fold-${FOLD}"
fi

# Find the model files with optional pattern matching
if [ -n "$MODEL_PATTERN" ]; then
    MODEL_SEARCH_PATTERN="${BASE_DIR}/models/${MODEL_PATTERN}.pt"
else
    MODEL_SEARCH_PATTERN="${BASE_DIR}/models/*.pt"
fi

MODEL_FILES=($(ls $MODEL_SEARCH_PATTERN 2>/dev/null))

if [ ${#MODEL_FILES[@]} -eq 0 ]; then
    echo "Error: No model files found matching pattern: $MODEL_SEARCH_PATTERN"
    echo "Available models:"
    ls "${BASE_DIR}/models/" 2>/dev/null || echo "  No models directory found"
    exit 1
fi

echo "Found ${#MODEL_FILES[@]} model(s) matching pattern"

# Process each model file
for MODEL_FILE in "${MODEL_FILES[@]}"; do
    MODEL_NAME=$(basename $MODEL_FILE .pt)

    # Set file paths
    ROOT_FILE="${BASE_DIR}/trees/${MODEL_NAME}.root"
    CSV_FILE="${BASE_DIR}/CSV/${MODEL_NAME}.csv"

    # Determine analysis type and create organized output directory
    if [[ $MODEL_NAME == *"grp"* ]]; then
        ANALYSIS_TYPE="grouped_backgrounds"
        # Extract group info from model name
        if [[ $MODEL_NAME == *"3grp"* ]]; then
            ANALYSIS_SUBTYPE="3groups"
        elif [[ $MODEL_NAME == *"4grp"* ]]; then
            ANALYSIS_SUBTYPE="4groups"
        else
            ANALYSIS_SUBTYPE="groups"
        fi
    else
        ANALYSIS_TYPE="individual_backgrounds"
        # Extract number of backgrounds from model name
        if [[ $MODEL_NAME == *"bg"* ]]; then
            ANALYSIS_SUBTYPE=$(echo $MODEL_NAME | grep -o '[0-9]\+bg' | head -1)
        else
            ANALYSIS_SUBTYPE="standard"
        fi
    fi

    OUTPUT_DIR="${WORKDIR}/ParticleNet/plots/analysis/${ANALYSIS_TYPE}/${ANALYSIS_SUBTYPE}/${CHANNEL}/${SIGNAL}"
    if [ "$PILOT_MODE" = true ]; then
        OUTPUT_DIR="${OUTPUT_DIR}/pilot"
    else
        OUTPUT_DIR="${OUTPUT_DIR}/fold-${FOLD}"
    fi

    # Check if files exist
    if [ ! -f "$ROOT_FILE" ]; then
        echo "Warning: ROOT file not found for $MODEL_NAME: $ROOT_FILE"
        echo "Skipping this model..."
        continue
    fi

    if [ ! -f "$CSV_FILE" ]; then
        echo "Warning: CSV file not found for $MODEL_NAME: $CSV_FILE"
        echo "Skipping this model..."
        continue
    fi

    echo ""
    echo "=========================================="
    echo "Visualizing results for:"
    echo "  Channel: $CHANNEL"
    echo "  Signal: $SIGNAL"
    echo "  Fold: $FOLD"
    echo "  Model: $MODEL_NAME"
    echo "  Analysis Type: $ANALYSIS_TYPE ($ANALYSIS_SUBTYPE)"
    echo "  Output: $OUTPUT_DIR"
    echo "=========================================="

    # Create output directory
    mkdir -p "$OUTPUT_DIR"

    # Run visualization from ParticleNet directory for proper path handling
    cd "${WORKDIR}/ParticleNet"

    python python/visualizeMultiClass.py \
        --root "$ROOT_FILE" \
        --csv "$CSV_FILE" \
        --output "$OUTPUT_DIR" \
        --channel "$CHANNEL" \
        --signal "$SIGNAL"

    if [ $? -eq 0 ]; then
        echo "Visualization completed successfully!"
        echo "Plots saved to: $OUTPUT_DIR"

        # List generated files
        echo ""
        echo "Generated files:"
        ls -la "$OUTPUT_DIR/"

        # Create a comprehensive summary file with model information and class imbalance analysis
        echo "Model: $MODEL_NAME" > "$OUTPUT_DIR/model_info.txt"
        echo "Analysis Type: $ANALYSIS_TYPE" >> "$OUTPUT_DIR/model_info.txt"
        echo "Analysis Subtype: $ANALYSIS_SUBTYPE" >> "$OUTPUT_DIR/model_info.txt"
        echo "Channel: $CHANNEL" >> "$OUTPUT_DIR/model_info.txt"
        echo "Signal: $SIGNAL" >> "$OUTPUT_DIR/model_info.txt"
        echo "Fold: $FOLD" >> "$OUTPUT_DIR/model_info.txt"
        echo "Pilot Mode: $PILOT_MODE" >> "$OUTPUT_DIR/model_info.txt"
        echo "Generated: $(date)" >> "$OUTPUT_DIR/model_info.txt"
        echo "" >> "$OUTPUT_DIR/model_info.txt"

        # Add class imbalance analysis information
        if [[ $ANALYSIS_TYPE == "grouped_backgrounds" ]]; then
            echo "=== CLASS IMBALANCE HANDLING ===" >> "$OUTPUT_DIR/model_info.txt"
            echo "This model uses grouped background classification with advanced" >> "$OUTPUT_DIR/model_info.txt"
            echo "class imbalance handling. The following metrics are provided:" >> "$OUTPUT_DIR/model_info.txt"
            echo "" >> "$OUTPUT_DIR/model_info.txt"
            echo "• Standard Accuracy: Sample-count based (may favor majority classes)" >> "$OUTPUT_DIR/model_info.txt"
            echo "• Weighted Accuracy: Physics-weight normalized" >> "$OUTPUT_DIR/model_info.txt"
            echo "• Group-Balanced Accuracy: Equal class contribution (training-consistent)" >> "$OUTPUT_DIR/model_info.txt"
            echo "" >> "$OUTPUT_DIR/model_info.txt"
            echo "Additional visualizations:" >> "$OUTPUT_DIR/model_info.txt"
            echo "• class_imbalance_analysis_*.png: Sample vs weight distributions" >> "$OUTPUT_DIR/model_info.txt"
            echo "• class_imbalance_metrics_*.csv: Detailed imbalance metrics" >> "$OUTPUT_DIR/model_info.txt"
            echo "• confusion_matrix_*.png: 3-panel (count, normalized, weighted)" >> "$OUTPUT_DIR/model_info.txt"
            echo "• summary_plot.png: Enhanced with weight-aware metrics" >> "$OUTPUT_DIR/model_info.txt"
            echo "" >> "$OUTPUT_DIR/model_info.txt"
        else
            echo "=== STANDARD CLASSIFICATION ===" >> "$OUTPUT_DIR/model_info.txt"
            echo "This model uses individual background classification." >> "$OUTPUT_DIR/model_info.txt"
            echo "Standard accuracy metrics are used." >> "$OUTPUT_DIR/model_info.txt"
            echo "" >> "$OUTPUT_DIR/model_info.txt"
        fi

        # Check if class imbalance analysis files were generated and report key metrics
        if [[ -f "$OUTPUT_DIR/class_imbalance_metrics_test.csv" ]]; then
            echo "=== KEY PERFORMANCE METRICS (Test Set) ===" >> "$OUTPUT_DIR/model_info.txt"

            # Extract key metrics from the CSV file (if it exists)
            if command -v python3 > /dev/null 2>&1; then
                python3 -c "
import pandas as pd
import sys
try:
    df = pd.read_csv('$OUTPUT_DIR/class_imbalance_metrics_test.csv')
    overall_row = df[df['Class'] == 'OVERALL']
    if not overall_row.empty:
        print('Standard Accuracy: {:.3f}'.format(overall_row.iloc[0]['Standard_Accuracy']))
        print('Weighted Accuracy: {:.3f}'.format(overall_row.iloc[0]['Weighted_Accuracy']))
        print('Group-Balanced Accuracy: {:.3f}'.format(overall_row.iloc[0]['Group_Balanced_Accuracy']))
        print('Total Events: {:.0f}'.format(overall_row.iloc[0]['Sample_Count']))
        print('Total Weight: {:.1f}'.format(overall_row.iloc[0]['Total_Weight']))
except Exception as e:
    print('Could not extract metrics from CSV file')
" >> "$OUTPUT_DIR/model_info.txt" 2>/dev/null
            fi
            echo "" >> "$OUTPUT_DIR/model_info.txt"
        fi

        # Add file listing
        echo "=== GENERATED FILES ===" >> "$OUTPUT_DIR/model_info.txt"
        ls -la "$OUTPUT_DIR/" | grep -E '\.(png|pdf|csv|txt)$' | awk '{print $9, "("$5" bytes)"}' >> "$OUTPUT_DIR/model_info.txt"

    else
        echo "Error: Visualization failed for $MODEL_NAME!"
        echo "Continuing with other models..."
        continue
    fi

done

echo ""
echo "=========================================="
echo "All visualizations completed!"
echo "Results organized in: ${WORKDIR}/ParticleNet/plots/analysis/"
echo ""

# Check if any grouped background models were processed
GROUPED_PROCESSED=false
for MODEL_FILE in "${MODEL_FILES[@]}"; do
    MODEL_NAME=$(basename $MODEL_FILE .pt)
    if [[ $MODEL_NAME == *"grp"* ]]; then
        GROUPED_PROCESSED=true
        break
    fi
done

if [ "$GROUPED_PROCESSED" = true ]; then
    echo "🎯 CLASS IMBALANCE ANALYSIS COMPLETED!"
    echo "──────────────────────────────────────"
    echo "Enhanced visualizations generated for grouped background models:"
    echo "• Group-balanced accuracy metrics (training-consistent evaluation)"
    echo "• Class imbalance analysis plots showing sample vs weight distributions"
    echo "• Weight-aware confusion matrices with physics-normalized metrics"
    echo "• Detailed CSV files with per-class imbalance statistics"
    echo ""
    echo "Key files to review:"
    echo "• class_imbalance_analysis_test.png - Visual analysis of class balance"
    echo "• class_imbalance_metrics_test.csv - Detailed numeric analysis"
    echo "• summary_plot.png - Enhanced summary with weight-aware metrics"
    echo "• model_info.txt - Comprehensive analysis summary"
    echo ""
    echo "The group-balanced accuracy provides the most reliable evaluation"
    echo "metric for grouped background classification models."
else
    echo "Standard multi-class visualization completed for individual background models."
fi
