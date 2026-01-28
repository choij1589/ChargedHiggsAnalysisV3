#!/bin/bash
#
# impact.sh - Batch impact plot calculation for all mass points
#
# This script orchestrates impact plot calculation across all mass points
# for Run2, Run3, or combined (All) datasets.
#
# Usage:
#   # Run2 only
#   ./impact.sh --mode run2 --method Baseline --binning extended
#
#   # Run3 only
#   ./impact.sh --mode run3 --method Baseline --binning extended
#
#   # All (Run2, Run3, and combined)
#   ./impact.sh --mode all --method Baseline --binning extended
#
#   # ParticleNet method
#   ./impact.sh --mode all --method ParticleNet --binning extended --partial-unblind
#
#   # Single era mode
#   ./impact.sh --era 2018 --method Baseline --binning extended
#
#   # Submit to HTCondor
#   ./impact.sh --mode all --method Baseline --binning extended --condor
#

set -euo pipefail

# Get the script directory (SignalRegionStudyV2/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Mass points (same as makeBinnedTemplates.sh)
MASSPOINTs_BASELINE=(
    "MHc70_MA15" "MHc100_MA60" "MHc130_MA90" "MHc160_MA155"
)
MASSPOINTs_PARTICLENET=(
    "MHc100_MA95" "MHc130_MA90" "MHc160_MA85"
)

# Default values
MODE="all"  # Options: all, run2, run3
SINGLE_ERA=""  # Single era mode (overrides MODE)
METHOD="Baseline"  # Options: Baseline, ParticleNet
BINNING="extended"
PARTIAL_UNBLIND=false
EXPECT_SIGNAL=1  # Default: inject signal (use 0 for background-only)
CONDOR=false
PLOT_ONLY=false
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="${2,,}"  # Convert to lowercase
            shift 2
            ;;
        --era)
            SINGLE_ERA="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --binning)
            BINNING="$2"
            shift 2
            ;;
        --partial-unblind)
            PARTIAL_UNBLIND=true
            shift
            ;;
        --expect-signal)
            EXPECT_SIGNAL="$2"
            shift 2
            ;;
        --condor)
            CONDOR=true
            shift
            ;;
        --plot-only)
            PLOT_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Modes:"
            echo "  --mode all    - Process Run2, Run3, and All (default)"
            echo "  --mode run2   - Process Run2 only"
            echo "  --mode run3   - Process Run3 only"
            echo ""
            echo "Single Era:"
            echo "  --era ERA     - Process single era only (e.g., --era 2018, --era Run2)"
            echo ""
            echo "Template Options:"
            echo "  --method METHOD        - Baseline or ParticleNet (default: Baseline)"
            echo "  --binning BINNING      - extended or uniform (default: extended)"
            echo "  --partial-unblind      - Use partial-unblind templates"
            echo "  --expect-signal N      - Expected signal for Asimov (0 or 1) [default: 1]"
            echo ""
            echo "Execution Options:"
            echo "  --condor               - Submit to HTCondor via DAG"
            echo "  --plot-only            - Only generate impact plots from existing json"
            echo ""
            echo "Other:"
            echo "  --dry-run              - Print commands without executing"
            echo ""
            echo "Examples:"
            echo "  # Run impacts for all combined mass points"
            echo "  $0 --mode all --method Baseline --binning extended"
            echo ""
            echo "  # Submit to HTCondor"
            echo "  $0 --mode all --method Baseline --binning extended --condor"
            echo ""
            echo "  # ParticleNet method with partial unblinding"
            echo "  $0 --mode all --method ParticleNet --binning extended --partial-unblind"
            echo ""
            echo "  # Only regenerate plots from existing results"
            echo "  $0 --mode all --method Baseline --plot-only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Select mass points based on method
if [[ "$METHOD" == "ParticleNet" ]]; then
    MASSPOINTs=("${MASSPOINTs_PARTICLENET[@]}")
else
    MASSPOINTs=("${MASSPOINTs_BASELINE[@]}")
fi

# Build extra args for runImpacts.sh
EXTRA_ARGS=""
if [[ "$PARTIAL_UNBLIND" == true ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --partial-unblind"
fi
EXTRA_ARGS="$EXTRA_ARGS --expect-signal $EXPECT_SIGNAL"
if [[ "$CONDOR" == true ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --condor"
fi
if [[ "$PLOT_ONLY" == true ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --plot-only"
fi
if [[ "$DRY_RUN" == true ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --dry-run"
fi

echo "============================================================"
echo "SignalRegionStudyV2 Impact Plot Batch Submission"
if [[ -n "$SINGLE_ERA" ]]; then
    echo "Era: $SINGLE_ERA (single era mode)"
else
    echo "Mode: $MODE"
fi
echo "Method: $METHOD"
echo "Binning: $BINNING"
echo "Mass points: ${#MASSPOINTs[@]} total"
echo "Partial unblind: $PARTIAL_UNBLIND"
echo "Expect signal: $EXPECT_SIGNAL"
echo "Condor: $CONDOR"
echo "Plot only: $PLOT_ONLY"
echo "Dry run: $DRY_RUN"
echo "============================================================"
echo ""

# Function to run impacts for a single era and mass point
run_impacts_single() {
    local era=$1
    local masspoint=$2

    local cmd="bash ${SCRIPT_DIR}/scripts/runImpacts.sh"
    cmd="$cmd --era $era"
    cmd="$cmd --channel Combined"
    cmd="$cmd --masspoint $masspoint"
    cmd="$cmd --method $METHOD"
    cmd="$cmd --binning $BINNING"
    cmd="$cmd $EXTRA_ARGS"

    echo ">>> Running: $cmd"
    eval "$cmd"
    echo ""
}

# Function to process all mass points for a given era
process_era() {
    local era=$1

    echo ""
    echo "============================================================"
    echo "Processing era: $era"
    echo "============================================================"

    for masspoint in "${MASSPOINTs[@]}"; do
        run_impacts_single "$era" "$masspoint"
    done
}

# ============================================================
# Main Execution
# ============================================================

# Single era mode
if [[ -n "$SINGLE_ERA" ]]; then
    process_era "$SINGLE_ERA"
    echo ""
    echo "============================================================"
    echo "Single era processing complete: $SINGLE_ERA"
    echo "============================================================"
    exit 0
fi

# Multi-era mode
case "$MODE" in
    run2)
        process_era "Run2"
        ;;
    run3)
        process_era "Run3"
        ;;
    all)
        # All three: Run2, Run3, and combined
        process_era "Run2"
        process_era "Run3"
        process_era "All"
        ;;
    *)
        echo "ERROR: Unknown mode '$MODE'"
        echo "Valid modes: all, run2, run3"
        exit 1
        ;;
esac

echo ""
echo "============================================================"
echo "Batch submission complete!"
echo ""
if [[ "$CONDOR" == true ]]; then
    echo "To monitor jobs:"
    echo "  condor_q -dag"
fi
echo ""
echo "After completion, generate plots with:"
echo "  $0 --mode $MODE --method $METHOD --plot-only"
echo "============================================================"
