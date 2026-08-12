#!/bin/bash
#
# HTCondor wrapper script for the mA-interpolation chain DAGMan workflow
# (automize/interpolation.sh).
#
# Unlike makeBinnedTemplates_wrapper.sh's template step, every step here is
# light I/O: samples are read straight off the shared samples/ tree over
# NFS (should_transfer_files = NO), no pnfs staging is needed.
#
# Usage:
#   ./interpolation_wrapper.sh <STEP> <MHC> <MASSPOINT> [EXTRA_ARGS]
#
# MASSPOINT is "-" for steps that are not per-masspoint (polynomials,
# yield_model, delta_model, merge_*, export_uncertainties).
#
set -eo pipefail

STEP=$1
MHC=$2
MASSPOINT=$3
EXTRA_ARGS="${*:4}"

if [[ -z "$STEP" || -z "$MHC" || -z "$MASSPOINT" ]]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 STEP MHC MASSPOINT [EXTRA_ARGS]"
    exit 1
fi

echo "============================================================"
echo "HTCondor Job: interpolation chain"
echo "Step: $STEP"
echo "MHc: $MHC"
echo "Masspoint: $MASSPOINT"
echo "Extra args: $EXTRA_ARGS"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "_CONDOR_SCRATCH_DIR: ${_CONDOR_SCRATCH_DIR:-not set}"
echo "============================================================"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
srs_setup_cmssw

# Include user site-packages for cmsstyle, same as the template wrapper.
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$SRS_REPO_DIR/Common/Tools:$PYTHONPATH"
export WORKDIR="$SRS_REPO_DIR"
cd "$SRS_MODULE_DIR"
export PATH="${PWD}/python:${PATH}"

INTERP_DIR="$(srs_interp_dir "$MHC")"

part_output() {
    # $1 = subdir under $INTERP_DIR, $2 = file basename prefix
    local subdir=$1
    local prefix=$2
    mkdir -p "$INTERP_DIR/$subdir/parts"
    echo "$INTERP_DIR/$subdir/parts/${prefix}.${MASSPOINT}.json"
}

case "$STEP" in
    fit_float)
        python3 python/fitInterpShapes.py --mhc "$MHC" --pass floating \
            --masspoints "$MASSPOINT" \
            --output "$(part_output fits dcb_fits_floating)" $EXTRA_ARGS
        ;;
    merge_float)
        python3 python/mergeInterpResults.py --mhc "$MHC" --stage fits-floating $EXTRA_ARGS
        ;;
    fit_frozen)
        python3 python/fitInterpShapes.py --mhc "$MHC" --pass frozen \
            --masspoints "$MASSPOINT" \
            --output "$(part_output fits dcb_fits)" $EXTRA_ARGS
        ;;
    merge_frozen)
        python3 python/mergeInterpResults.py --mhc "$MHC" --stage fits $EXTRA_ARGS
        ;;
    polynomials)
        python3 python/fitInterpPolynomials.py --mhc "$MHC" $EXTRA_ARGS
        ;;
    closure)
        python3 python/closInterpShapes.py --mhc "$MHC" \
            --masspoints "$MASSPOINT" \
            --output "$(part_output closure closure)" $EXTRA_ARGS
        ;;
    merge_closure)
        python3 python/mergeInterpResults.py --mhc "$MHC" --stage closure $EXTRA_ARGS
        ;;
    yields)
        python3 python/measInterpYields.py --mhc "$MHC" \
            --masspoints "$MASSPOINT" \
            --output "$(part_output yields yields)" $EXTRA_ARGS
        ;;
    merge_yields)
        python3 python/mergeInterpResults.py --mhc "$MHC" --stage yields $EXTRA_ARGS
        ;;
    yield_model)
        python3 python/fitInterpYieldModel.py --mhc "$MHC" $EXTRA_ARGS
        ;;
    yield_closure)
        python3 python/closInterpYields.py --mhc "$MHC" \
            --masspoints "$MASSPOINT" \
            --output "$(part_output yields yield_closure)" $EXTRA_ARGS
        ;;
    merge_yield_closure)
        python3 python/mergeInterpResults.py --mhc "$MHC" --stage yield-closure $EXTRA_ARGS
        ;;
    deltas)
        python3 python/measInterpShapeDeltas.py --mhc "$MHC" \
            --masspoints "$MASSPOINT" \
            --output "$(part_output shape_deltas shape_deltas)" $EXTRA_ARGS
        ;;
    merge_deltas)
        python3 python/mergeInterpResults.py --mhc "$MHC" --stage shape-deltas $EXTRA_ARGS
        ;;
    delta_model)
        python3 python/fitInterpShapeDeltas.py --mhc "$MHC" $EXTRA_ARGS
        ;;
    export_uncertainties)
        python3 python/exportInterpUncertainties.py --mhc "$MHC" $EXTRA_ARGS
        ;;
    loo)
        # One full leave-one-out iteration for MASSPOINT = MHc{X}_MA{Y}:
        # refit the shape polynomials and the yield model on the full grid
        # minus this point, then close both at this point only. Outputs go
        # to tests/interpolation/MHc{X}_MA{Y}/.
        LOO_MA="${MASSPOINT##*_MA}"
        if [[ -z "$LOO_MA" || "$LOO_MA" == "$MASSPOINT" ]]; then
            echo "ERROR: loo step needs MASSPOINT of the form MHc{X}_MA{Y}, got '$MASSPOINT'"
            exit 1
        fi
        python3 python/fitInterpPolynomials.py --mhc "$MHC" --loo-ma "$LOO_MA" $EXTRA_ARGS
        python3 python/fitInterpYieldModel.py  --mhc "$MHC" --loo-ma "$LOO_MA" $EXTRA_ARGS
        python3 python/closInterpShapes.py     --mhc "$MHC" --loo-ma "$LOO_MA" $EXTRA_ARGS
        python3 python/closInterpYields.py     --mhc "$MHC" --loo-ma "$LOO_MA" $EXTRA_ARGS
        ;;
    export_loo)
        python3 python/exportInterpUncertainties.py --loo --mhc "$MHC" $EXTRA_ARGS
        ;;
    *)
        echo "ERROR: Unknown step '$STEP'"
        echo "Valid steps: fit_float, merge_float, fit_frozen, merge_frozen, polynomials,"
        echo "  closure, merge_closure, yields, merge_yields, yield_model, yield_closure,"
        echo "  merge_yield_closure, deltas, merge_deltas, delta_model, export_uncertainties,"
        echo "  loo, export_loo"
        exit 1
        ;;
esac

EXIT_CODE=$?
echo "============================================================"
echo "Job completed with exit code: $EXIT_CODE"
echo "Time: $(date)"
echo "============================================================"

exit $EXIT_CODE
