#!/bin/bash
#
# hybridnew.sh - Batch HybridNew limit calculation for all mass points
#
# This script orchestrates HybridNew limit calculation across all mass points
# for Run2, Run3, or combined (All) datasets. Each mass point is submitted
# as a separate HTCondor DAG workflow.
#
# Usage:
#   # Run2 only
#   ./hybridnew.sh --mode run2 --method Baseline --binning extended --auto-grid
#
#   # Run3 only
#   ./hybridnew.sh --mode run3 --method Baseline --binning extended --auto-grid
#
#   # All (Run2 + Run3 + combined)
#   ./hybridnew.sh --mode all --method Baseline --binning extended --auto-grid
#
#   # ParticleNet method
#   ./hybridnew.sh --mode all --method ParticleNet --binning extended --partial-unblind --auto-grid
#
#   # Single era mode
#   ./hybridnew.sh --era 2018 --method Baseline --binning extended --auto-grid
#

set -euo pipefail

# Get the script directory (SignalRegionStudyV2/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Mass points (same as makeBinnedTemplates.sh)
MASSPOINTs_BASELINE=(
    "MHc70_MA15" "MHc70_MA18" "MHc70_MA40" "MHc70_MA55" "MHc70_MA65"
    "MHc85_MA15" "MHc85_MA70" "MHc85_MA80"
    "MHc100_MA15" "MHc100_MA24" "MHc100_MA60" "MHc100_MA75" "MHc100_MA95"
    "MHc115_MA15" "MHc115_MA27" "MHc115_MA87" "MHc115_MA110"
    "MHc130_MA15" "MHc130_MA30" "MHc130_MA55" "MHc130_MA83" "MHc130_MA90" "MHc130_MA100" "MHc130_MA125"
    "MHc145_MA15" "MHc145_MA35" "MHc145_MA92" "MHc145_MA140"
    "MHc160_MA15" "MHc160_MA50" "MHc160_MA85" "MHc160_MA98" "MHc160_MA120" "MHc160_MA135" "MHc160_MA155"
)
MASSPOINTs_PARTICLENET=(
    "MHc100_MA95" "MHc130_MA90" "MHc160_MA85" "MHc115_MA87" "MHc145_MA92" "MHc160_MA98"
)

# Default values
MODE="all"  # Options: all, run2, run3
SINGLE_ERA=""  # Single era mode (overrides MODE)
METHOD="Baseline"  # Options: Baseline, ParticleNet
BINNING="extended"
PARTIAL_UNBLIND=false
AUTO_GRID=true  # Default to auto-grid
NTOYS=50
NJOBS=10
DRY_RUN=false

# Post-processing flags
MERGE_ONLY=false
EXTRACT_ONLY=false
PARTIAL_EXTRACT=false
PLOT_ONLY=false

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
        --auto-grid)
            AUTO_GRID=true
            shift
            ;;
        --no-auto-grid)
            AUTO_GRID=false
            shift
            ;;
        --ntoys)
            NTOYS="$2"
            shift 2
            ;;
        --njobs)
            NJOBS="$2"
            shift 2
            ;;
        --merge-only)
            MERGE_ONLY=true
            shift
            ;;
        --extract-only)
            EXTRACT_ONLY=true
            shift
            ;;
        --partial-extract)
            PARTIAL_EXTRACT=true
            shift
            ;;
        --plot)
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
            echo "  --mode all    - Process combined Run2+Run3 (default)"
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
            echo ""
            echo "HybridNew Options:"
            echo "  --auto-grid            - Auto-tune r-range from Asymptotic results (default: true)"
            echo "  --no-auto-grid         - Disable auto-grid, use manual rmin/rmax/rstep"
            echo "  --ntoys N              - Toys per job (default: 100)"
            echo "  --njobs N              - Jobs per r-point (default: 10)"
            echo ""
            echo "Post-processing:"
            echo "  --merge-only           - Only merge existing toys"
            echo "  --extract-only         - Only extract limits from merged grid"
            echo "  --partial-extract      - Extract limits from incomplete condor jobs"
            echo "  --plot                 - Generate plots from existing output"
            echo ""
            echo "Other:"
            echo "  --dry-run              - Print commands without executing"
            echo ""
            echo "Examples:"
            echo "  # Run HybridNew for all Run2 mass points"
            echo "  $0 --mode run2 --method Baseline --binning extended --auto-grid"
            echo ""
            echo "  # ParticleNet method with partial unblinding"
            echo "  $0 --mode all --method ParticleNet --binning extended --partial-unblind"
            echo ""
            echo "  # Extract limits from partial results"
            echo "  $0 --mode run2 --method Baseline --partial-extract"
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

# Build extra args for runHybridNew.sh
EXTRA_ARGS=""
if [[ "$PARTIAL_UNBLIND" == true ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --partial-unblind"
fi
if [[ "$AUTO_GRID" == true ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --auto-grid"
fi
if [[ "$MERGE_ONLY" == true ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --merge-only"
fi
if [[ "$EXTRACT_ONLY" == true ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --extract-only"
fi
if [[ "$PARTIAL_EXTRACT" == true ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --partial-extract"
fi
if [[ "$PLOT_ONLY" == true ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --plot"
fi
if [[ "$DRY_RUN" == true ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --dry-run"
fi

echo "============================================================"
echo "SignalRegionStudyV2 HybridNew Batch Submission"
if [[ -n "$SINGLE_ERA" ]]; then
    echo "Era: $SINGLE_ERA (single era mode)"
else
    echo "Mode: $MODE"
fi
echo "Method: $METHOD"
echo "Binning: $BINNING"
echo "Mass points: ${#MASSPOINTs[@]} total"
echo "Partial unblind: $PARTIAL_UNBLIND"
echo "Auto-grid: $AUTO_GRID"
echo "Toys per job: $NTOYS"
echo "Jobs per r-point: $NJOBS"
echo "Dry run: $DRY_RUN"
echo "============================================================"
echo ""

# Function to run HybridNew for a single era and mass point
run_hybridnew_single() {
    local era=$1
    local masspoint=$2

    local cmd="bash ${SCRIPT_DIR}/scripts/runHybridNew.sh"
    cmd="$cmd --era $era"
    cmd="$cmd --channel Combined"
    cmd="$cmd --masspoint $masspoint"
    cmd="$cmd --method $METHOD"
    cmd="$cmd --binning $BINNING"
    cmd="$cmd --ntoys $NTOYS"
    cmd="$cmd --njobs $NJOBS"
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
        run_hybridnew_single "$era" "$masspoint"
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
        # Combined Run2+Run3 only
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
echo "To monitor jobs:"
echo "  condor_q -dag"
echo ""
echo "After completion, extract results with:"
echo "  $0 --mode $MODE --method $METHOD --partial-extract"
echo ""
echo "Generate plots with:"
echo "  $0 --mode $MODE --method $METHOD --plot"
echo "============================================================"
