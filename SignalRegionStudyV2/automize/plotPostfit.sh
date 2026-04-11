#!/bin/bash
#
# plotPostfit.sh - Batch pre-fit/post-fit plots from FitDiagnostics output
#
# Wraps python/plotPostfit.py over all mass points for Run2, Run3, and All.
# Requires --partial-unblind or --unblind (blinded fits don't need postfit plots).
#
# Usage:
#   # ParticleNet partial-unblind
#   ./plotPostfit.sh --mode all --method ParticleNet --partial-unblind
#
#   # Fully unblinded
#   ./plotPostfit.sh --mode all --method Baseline --unblind
#   ./plotPostfit.sh --mode all --method ParticleNet --unblind
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_masspoints.sh"

MODE="all"
SINGLE_ERA=""
METHOD="Baseline"
BINNING="extended"
FIT_TYPE="b"
PARTIAL_UNBLIND=false
UNBLIND=false
DRY_RUN=false

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
        --fit-type)
            FIT_TYPE="$2"
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
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Required (one of):"
            echo "  --partial-unblind   Use partial-unblind templates (real data, sideband)"
            echo "  --unblind           Use fully unblinded templates"
            echo ""
            echo "Modes:"
            echo "  --mode all    Run Run2, Run3, and All (default)"
            echo "  --mode run2   Run2 only"
            echo "  --mode run3   Run3 only"
            echo "  --era ERA     Single era only (e.g. All, Run2, 2018)"
            echo ""
            echo "Template options:"
            echo "  --method METHOD     Baseline or ParticleNet [default: Baseline]"
            echo "  --binning BINNING   extended or uniform [default: extended]"
            echo "  --fit-type T        b (B-only) or s (S+B) [default: b]"
            echo ""
            echo "Other:"
            echo "  --dry-run           Print commands without executing"
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

if [[ "$UNBLIND" != true && "$PARTIAL_UNBLIND" != true ]]; then
    echo "ERROR: must specify --partial-unblind or --unblind (blinded fits do not need postfit plots)"
    exit 1
fi

if [[ "$METHOD" == "ParticleNet" ]]; then
    MASSPOINTs=("${MASSPOINTs_PARTICLENET[@]}")
else
    MASSPOINTs=("${MASSPOINTs_BASELINE[@]}")
fi

EXTRA_ARGS=""
[[ "$PARTIAL_UNBLIND" == true ]] && EXTRA_ARGS="$EXTRA_ARGS --partial-unblind"
[[ "$UNBLIND"         == true ]] && EXTRA_ARGS="$EXTRA_ARGS --unblind"

echo "============================================================"
echo "SignalRegionStudyV2 Post-fit Plot Batch"
if [[ -n "$SINGLE_ERA" ]]; then
    echo "Era:             $SINGLE_ERA (single era mode)"
else
    echo "Mode:            $MODE"
fi
echo "Method:          $METHOD"
echo "Binning:         $BINNING"
echo "Fit type:        $FIT_TYPE"
echo "Mass points:     ${#MASSPOINTs[@]} total"
echo "Partial-unblind: $PARTIAL_UNBLIND"
echo "Unblind:         $UNBLIND"
echo "Dry run:         $DRY_RUN"
echo "============================================================"
echo ""

run_postfit_single() {
    local era=$1
    local masspoint=$2

    local cmd="python3 ${SCRIPT_DIR}/python/plotPostfit.py"
    cmd="$cmd --era ${era}"
    cmd="$cmd --channel Combined"
    cmd="$cmd --masspoint ${masspoint}"
    cmd="$cmd --method ${METHOD}"
    cmd="$cmd --binning ${BINNING}"
    cmd="$cmd --fit-type ${FIT_TYPE}"
    cmd="$cmd ${EXTRA_ARGS}"

    echo ">>> Running: $cmd"
    if [[ "$DRY_RUN" != true ]]; then
        eval "$cmd"
    fi
    echo ""
}

process_era() {
    local era=$1
    echo ""
    echo "============================================================"
    echo "Processing era: $era"
    echo "============================================================"
    for masspoint in "${MASSPOINTs[@]}"; do
        run_postfit_single "$era" "$masspoint"
    done
}

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
echo "Post-fit plot batch complete!"
echo "Outputs under: templates/{era}/Combined/{masspoint}/${METHOD}/${BINNING}*/combine_output/fitdiag/plots/"
echo "============================================================"
