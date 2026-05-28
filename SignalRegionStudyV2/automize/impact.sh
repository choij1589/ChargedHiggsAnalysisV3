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

# Mass points (loaded from configs/masspoints.json)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_masspoints.sh"

# Default values
MODE="all"  # Options: all, run2, run3
SINGLE_ERA=""  # Single era mode (overrides MODE)
METHOD="Baseline"  # Options: Baseline, ParticleNet
BINNING="extended"
NUISANCE="fallback_lnn"
PARTIAL_UNBLIND=false
UNBLIND=false
BLIND_RESULT=false
EXPECT_SIGNAL=1  # Default: inject signal (use 0 for background-only)
AUTO_EXPECT_SIGNAL=false
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
        --nuisance)
            NUISANCE="$2"
            shift 2
            ;;
        --partial-unblind)
            PARTIAL_UNBLIND=true
            shift
            ;;
        --unblind)
            UNBLIND=true
            shift
            ;;
        --blind-result)
            BLIND_RESULT=true
            shift
            ;;
        --expect-signal)
            EXPECT_SIGNAL="$2"
            shift 2
            ;;
        --auto-expect-signal)
            AUTO_EXPECT_SIGNAL=true
            shift
            ;;
        --condor)
            echo "NOTE: --condor is now the default (and only) execution mode. Flag ignored."
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
            echo "  --binning BINNING      - extended, uniform, or extended_coarser_binning (default: extended)"
            echo "  --nuisance MODE        - fallback_lnn (default) or preserve_shape"
            echo "  --partial-unblind      - Use partial-unblind templates"
            echo "  --unblind              - Use fully unblinded templates; auto-selects unblind mass-point subset"
            echo "  --blind-result         - Hide observed r in impact plot (stage-1 unblinding; use with --unblind)"
            echo "  --expect-signal N      - Expected signal for Asimov (0 or 1) [default: 1]
  --auto-expect-signal   - Use median expected limit as expectSignal (output: impacts_med)"
            echo ""
            echo "Execution Options:"
            echo "  --condor               - (No-op, condor is now the only execution mode)"
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

# Select mass points based on method and blinding mode
# --unblind auto-selects the curated unblind subset
if [[ "$METHOD" == "ParticleNet" ]]; then
    if [[ "$UNBLIND" == true ]]; then
        MASSPOINTs=("${MASSPOINTs_UNBLIND_PN[@]}")
    else
        MASSPOINTs=("${MASSPOINTs_IMPACT_PN[@]}")
    fi
else
    if [[ "$UNBLIND" == true ]]; then
        MASSPOINTs=("${MASSPOINTs_UNBLIND_BASELINE[@]}")
    else
        MASSPOINTs=("${MASSPOINTs_IMPACT_BASELINE[@]}")
    fi
fi

# Validate mutual exclusion
if [[ "$UNBLIND" == true && "$PARTIAL_UNBLIND" == true ]]; then
    echo "ERROR: --unblind and --partial-unblind are mutually exclusive"
    exit 1
fi
case "$NUISANCE" in
    fallback_lnn|preserve_shape) ;;
    *)
        echo "ERROR: Invalid --nuisance value '$NUISANCE'"
        echo "Valid values: fallback_lnn, preserve_shape"
        exit 1
        ;;
esac

# Build extra args for runImpacts.sh
EXTRA_ARGS=""
if [[ "$UNBLIND" == true ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --unblind"
elif [[ "$PARTIAL_UNBLIND" == true ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --partial-unblind"
fi
[[ "$NUISANCE" == "preserve_shape" ]] && EXTRA_ARGS="$EXTRA_ARGS --nuisance preserve_shape"
[[ "$BLIND_RESULT" == true ]] && EXTRA_ARGS="$EXTRA_ARGS --blind-result"
if [[ "$AUTO_EXPECT_SIGNAL" == true ]]; then
    EXTRA_ARGS="$EXTRA_ARGS --auto-expect-signal"
else
    EXTRA_ARGS="$EXTRA_ARGS --expect-signal $EXPECT_SIGNAL"
fi
EXTRA_ARGS="$EXTRA_ARGS --condor"
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
echo "Nuisance mode: $NUISANCE"
echo "Mass points: ${#MASSPOINTs[@]} total"
echo "Unblind: $UNBLIND"
echo "Partial unblind: $PARTIAL_UNBLIND"
echo "Blind result: $BLIND_RESULT"
echo "Expect signal: $([ "$AUTO_EXPECT_SIGNAL" == true ] && echo "auto (median expected)" || echo "$EXPECT_SIGNAL")"
echo "Execution: HTCondor (condor-only)"
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
echo "To monitor jobs:"
echo "  condor_q -dag"
echo ""
echo "After completion, generate plots with:"
echo "  $0 --mode $MODE --method $METHOD --plot-only"
echo "============================================================"
