#!/bin/bash
#
# gof.sh - Batch Goodness-of-Fit test for all mass points
#
# Runs GoF tests across mass points for Run2, Run3, or combined (All) datasets.
#
# Usage:
#   # ParticleNet, partial-unblind (preferred: real sideband data)
#   ./gof.sh --mode all --method ParticleNet --partial-unblind
#
#   # Baseline, blinded (Asimov data)
#   ./gof.sh --mode all --method Baseline
#
#   # Single era
#   ./gof.sh --era All --method ParticleNet --partial-unblind
#
#   # Plot only from existing results
#   ./gof.sh --mode all --method ParticleNet --partial-unblind --plot-only
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_masspoints.sh"

# Default values
MODE="all"
SINGLE_ERA=""
METHOD="Baseline"
BINNING="extended"
NTOYS=500
NBATCHES=5
PARTIAL_UNBLIND=false
UNBLIND=false
PLOT_ONLY=false
DRY_RUN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="${2,,}"
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
        --ntoys)
            NTOYS="$2"
            shift 2
            ;;
        --nbatches)
            NBATCHES="$2"
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
            echo "  --era ERA     - Process single era only (e.g., --era All, --era Run2)"
            echo ""
            echo "Template Options:"
            echo "  --method METHOD     - Baseline or ParticleNet [default: Baseline]"
            echo "  --binning BINNING   - extended or uniform [default: extended]"
            echo "  --partial-unblind   - Use partial-unblind templates (real data, score < 0.3)"
            echo "  --unblind           - Use fully unblinded templates"
            echo "  --ntoys N           - Total toys for p-value [default: 500]"
            echo "  --nbatches N        - HTCondor batches for toys [default: 5]"
            echo ""
            echo "Execution Options:"
            echo "  --condor            - (No-op, condor is now the only execution mode)"
            echo "  --plot-only         - Only collect+plot from existing outputs"
            echo ""
            echo "Other:"
            echo "  --dry-run           - Print commands without executing"
            echo ""
            echo "Examples:"
            echo "  # ParticleNet with partial-unblind (real sideband data)"
            echo "  $0 --mode all --method ParticleNet --partial-unblind"
            echo ""
            echo "  # Baseline with Asimov (blinded)"
            echo "  $0 --mode all --method Baseline"
            echo ""
            echo "  # Re-plot from existing results"
            echo "  $0 --mode all --method ParticleNet --partial-unblind --plot-only"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ "$UNBLIND" == true && "$PARTIAL_UNBLIND" == true ]]; then
    echo "ERROR: --unblind and --partial-unblind are mutually exclusive"
    exit 1
fi

# Select mass points based on method
if [[ "$METHOD" == "ParticleNet" ]]; then
    MASSPOINTs=("${MASSPOINTs_GOF_PN[@]}")
else
    MASSPOINTs=("${MASSPOINTs_GOF_BASELINE[@]}")
fi

# Build extra args
EXTRA_ARGS="--ntoys ${NTOYS} --nbatches ${NBATCHES} --condor"
[[ "$PARTIAL_UNBLIND" == true ]] && EXTRA_ARGS="$EXTRA_ARGS --partial-unblind"
[[ "$UNBLIND"         == true ]] && EXTRA_ARGS="$EXTRA_ARGS --unblind"
[[ "$PLOT_ONLY"       == true ]] && EXTRA_ARGS="$EXTRA_ARGS --plot-only"
[[ "$DRY_RUN"         == true ]] && EXTRA_ARGS="$EXTRA_ARGS --dry-run"

echo "============================================================"
echo "SignalRegionStudyV2 GoF Batch Submission"
if [[ -n "$SINGLE_ERA" ]]; then
    echo "Era: $SINGLE_ERA (single era mode)"
else
    echo "Mode: $MODE"
fi
echo "Method:          $METHOD"
echo "Binning:         $BINNING"
echo "Mass points:     ${#MASSPOINTs[@]} total"
echo "Partial-unblind: $PARTIAL_UNBLIND"
echo "Unblind:         $UNBLIND"
echo "Toys:            $NTOYS (${NBATCHES} batches)"
echo "Plot only:       $PLOT_ONLY"
echo "Dry run:         $DRY_RUN"
echo "============================================================"
echo ""

run_gof_single() {
    local era=$1
    local masspoint=$2

    local cmd="bash ${SCRIPT_DIR}/scripts/runGoF.sh"
    cmd="$cmd --era ${era}"
    cmd="$cmd --channel Combined"
    cmd="$cmd --masspoint ${masspoint}"
    cmd="$cmd --method ${METHOD}"
    cmd="$cmd --binning ${BINNING}"
    cmd="$cmd ${EXTRA_ARGS}"

    echo ">>> Running: $cmd"
    eval "$cmd"
    echo ""
}

process_era() {
    local era=$1
    echo ""
    echo "============================================================"
    echo "Processing era: $era"
    echo "============================================================"
    for masspoint in "${MASSPOINTs[@]}"; do
        run_gof_single "$era" "$masspoint"
    done
}

# Main execution
if [[ -n "$SINGLE_ERA" ]]; then
    process_era "$SINGLE_ERA"
    echo ""
    echo "============================================================"
    echo "Single era processing complete: $SINGLE_ERA"
    echo "============================================================"
    exit 0
fi

case "$MODE" in
    run2)
        process_era "Run2"
        ;;
    run3)
        process_era "Run3"
        ;;
    all)
        process_era "Run2"
        process_era "Run3"
        process_era "All"
        ;;
    *)
        echo "ERROR: Unknown mode '$MODE'. Valid: all, run2, run3"
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
echo "  $0 --mode $MODE --method $METHOD $([[ "$PARTIAL_UNBLIND" == true ]] && echo --partial-unblind) --plot-only"
echo "============================================================"
