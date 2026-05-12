#!/bin/bash
#
# Condor wrapper for plotPostfitSummary.py.
#
set -eo pipefail

REPO_DIR="/u/user/choij/scratch/ChargedHiggsAnalysisV3"

cd "$REPO_DIR/Common/CMSSW_14_1_0_pre4/src"
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval $(scramv1 runtime -sh)

export PATH="$REPO_DIR/SignalRegionStudyV2/python:$PATH"
export LD_LIBRARY_PATH="$REPO_DIR/SignalRegionStudyV2/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$REPO_DIR/Common/Tools:${PYTHONPATH:-}"
export WORKDIR="$REPO_DIR"

cd "$REPO_DIR/SignalRegionStudyV2"
python3 python/plotPostfitSummary.py "$@"
