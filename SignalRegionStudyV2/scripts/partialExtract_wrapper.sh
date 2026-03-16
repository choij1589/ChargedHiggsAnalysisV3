#!/bin/bash
#
# HTCondor wrapper script for partial-extract step of HybridNew limits
#
# Usage:
#   ./partialExtract_wrapper.sh <ERA> <MASSPOINT> <METHOD> <BINNING>
#
set -eo pipefail

ERA=$1
MASSPOINT=$2
METHOD=$3
BINNING=$4

if [[ -z "$ERA" || -z "$MASSPOINT" || -z "$METHOD" || -z "$BINNING" ]]; then
    echo "ERROR: Missing required arguments"
    echo "Usage: $0 ERA MASSPOINT METHOD BINNING"
    exit 1
fi

echo "============================================================"
echo "HTCondor Job: partialExtract"
echo "Era: $ERA"
echo "Masspoint: $MASSPOINT"
echo "Method: $METHOD"
echo "Binning: $BINNING"
echo "Host: $(hostname)"
echo "Time: $(date)"
echo "============================================================"

SCRATCH_WORKDIR="/u/user/choij/scratch/ChargedHiggsAnalysisV3"

# Setup CMSSW environment
source /cvmfs/cms.cern.ch/cmsset_default.sh
local_cmssw_dir="$SCRATCH_WORKDIR/Common/CMSSW_14_1_0_pre4/src"
if [[ -d "$local_cmssw_dir" ]]; then
    cd "$local_cmssw_dir"
    eval $(scramv1 runtime -sh)
    cd - > /dev/null
else
    echo "ERROR: CMSSW directory not found: $local_cmssw_dir"
    exit 1
fi

export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$SCRATCH_WORKDIR/Common/Tools:$PYTHONPATH"
export LD_LIBRARY_PATH="$SCRATCH_WORKDIR/Common/Tools/cpp/lib:$LD_LIBRARY_PATH"
export WORKDIR="$SCRATCH_WORKDIR"

cd "$SCRATCH_WORKDIR/SignalRegionStudyV2"

echo "Running runHybridNew.sh --partial-extract for $MASSPOINT ($ERA/$METHOD/$BINNING)..."
bash scripts/runHybridNew.sh \
    --era "$ERA" \
    --channel Combined \
    --masspoint "$MASSPOINT" \
    --method "$METHOD" \
    --binning "$BINNING" \
    --partial-extract

EXIT_CODE=$?
echo "============================================================"
echo "Job completed with exit code: $EXIT_CODE"
echo "Time: $(date)"
echo "============================================================"
exit $EXIT_CODE
