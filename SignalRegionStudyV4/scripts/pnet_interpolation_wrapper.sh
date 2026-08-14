#!/bin/bash
#
# HTCondor wrapper script for the ParticleNet-interpolation study chain
# (automize/pnetInterpolation.sh).
#
# Every step is light I/O like the Baseline interpolation wrapper: samples
# are read straight off the shared samples/ tree over NFS
# (should_transfer_files = NO), outputs land in the git-tracked
# fits/pnet/ and closure/pnet/ trees.
#
# Usage:
#   ./pnet_interpolation_wrapper.sh <STEP> <MHC> [EXTRA_ARGS]
#
# MHC is "-" for the cross-study steps (export, summarize).
#
set -eo pipefail

STEP=$1
MHC=$2
EXTRA_ARGS="${*:3}"

if [[ -z "$STEP" || -z "$MHC" ]]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 STEP MHC [EXTRA_ARGS]"
    exit 1
fi

echo "============================================================"
echo "HTCondor Job: ParticleNet interpolation chain"
echo "Step: $STEP"
echo "MHc: $MHC"
echo "Extra args: $EXTRA_ARGS"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "_CONDOR_SCRATCH_DIR: ${_CONDOR_SCRATCH_DIR:-not set}"
echo "============================================================"

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# getenv = False leaves HOME unset on the worker; cmssw + user
# site-packages need it (same guard as interp_templates_wrapper.sh).
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
# Include user site-packages for cmsstyle, same as the template wrapper.
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$SRS_REPO_DIR/Common/Tools:${PYTHONPATH:-}"
export WORKDIR="$SRS_REPO_DIR"
cd "$SRS_MODULE_DIR"
export PATH="${PWD}/python:${PATH}"

srs_setup_cmssw

case "$STEP" in
    thresholds)
        python3 python/measPnetThresholds.py --mhc "$MHC" $EXTRA_ARGS
        ;;
    shapes)
        python3 python/closPnetShapes.py --mhc "$MHC" $EXTRA_ARGS
        ;;
    yields)
        python3 python/closPnetYields.py --mhc "$MHC" $EXTRA_ARGS
        ;;
    eps_model)
        python3 python/fitPnetEpsModel.py --mhc "$MHC" $EXTRA_ARGS
        ;;
    export)
        python3 python/exportPnetUncertainties.py $EXTRA_ARGS
        ;;
    closure)
        python3 python/closPnetTemplates.py --mhc "$MHC" $EXTRA_ARGS
        ;;
    summarize)
        python3 python/summarizePnetStudies.py $EXTRA_ARGS
        ;;
    *)
        echo "ERROR: Unknown step '$STEP'"
        echo "Valid steps: thresholds, shapes, yields, eps_model, export,"
        echo "  closure, summarize"
        exit 1
        ;;
esac

EXIT_CODE=$?
echo "============================================================"
echo "Job completed with exit code: $EXIT_CODE"
echo "=== EXITING WITH STATUS $EXIT_CODE ==="
echo "Time: $(date)"
echo "============================================================"

exit $EXIT_CODE
