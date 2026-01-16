#!/bin/bash
set -euo pipefail

# Eras to plot
ERAs=("2016preVFP" "2016postVFP" "2017" "2018" "2022" "2022EE" "2023" "2023BPix" "Run2" "Run3" "All")

# Methods
METHODs=("Baseline" "ParticleNet")

# Limit type
LIMIT_TYPE="Asymptotic"

# Use local python scripts (not the ones from Combine)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
COLLECT_LIMITS="${PROJECT_DIR}/python/collectLimits.py"
PLOT_LIMITS="${PROJECT_DIR}/python/plotLimits.py"

# Change to project directory (script uses relative paths)
cd "$PROJECT_DIR"

# Parse command line arguments
STACK_BASELINE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --stack-baseline)
            STACK_BASELINE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--stack-baseline]"
            echo ""
            echo "Options:"
            echo "  --stack-baseline  Show baseline expected limit overlay on ParticleNet plots"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "============================================================"
echo "SignalRegionStudyV2 Limit Collection and Plotting"
echo "============================================================"

# Collect Baseline limits
echo ""
echo "Collecting Baseline limits..."
for era in "${ERAs[@]}"; do
    echo "  Collecting: era=$era, method=Baseline"
    $COLLECT_LIMITS --era "$era" --method Baseline
done

# Collect ParticleNet limits
echo ""
echo "Collecting ParticleNet limits..."
for era in "${ERAs[@]}"; do
    echo "  Collecting: era=$era, method=ParticleNet"
    $COLLECT_LIMITS --era "$era" --method ParticleNet
done

# Plot Baseline limits
echo ""
echo "Plotting Baseline limits..."
for era in "${ERAs[@]}"; do
    echo "  Plotting: era=$era, method=Baseline"
    $PLOT_LIMITS --era "$era" --method Baseline --limit_type "$LIMIT_TYPE"
done

# Plot ParticleNet limits
echo ""
echo "Plotting ParticleNet limits..."
for era in "${ERAs[@]}"; do
    if [[ "$STACK_BASELINE" == true ]]; then
        echo "  Plotting: era=$era, method=ParticleNet (with baseline overlay)"
        $PLOT_LIMITS --era "$era" --method ParticleNet --limit_type "$LIMIT_TYPE" --stack_baseline
    else
        echo "  Plotting: era=$era, method=ParticleNet"
        $PLOT_LIMITS --era "$era" --method ParticleNet --limit_type "$LIMIT_TYPE"
    fi
done

echo ""
echo "============================================================"
echo "All limit plots complete!"
echo "Output saved to: results/plots/"
echo "============================================================"
