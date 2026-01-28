#!/bin/bash
set -eo pipefail

# Re-exec in clean environment if not already clean
if [[ -z "$_CLEAN_ENV" ]]; then
    exec env -i \
        HOME="$HOME" \
        USER="$USER" \
        TERM="$TERM" \
        TMPDIR="${TMPDIR:-/tmp}" \
        PATH="/usr/bin:/bin:/usr/sbin:/sbin" \
        _CLEAN_ENV=1 \
        bash "$0" "$@"
fi

ERA=$1
CHANNEL=$2
MASSPOINT=$3
EXTRA_ARGS=${4:-}

# Paths
PNFS_BASE="/pnfs/knu.ac.kr/data/cms/store/user/choij"
REPO_DIR="/u/user/choij/scratch/ChargedHiggsAnalysisV3"
LOCAL_WORKDIR="${TMPDIR:-/tmp}/workdir_$$"

# Setup CMSSW environment from scratch
cd $REPO_DIR/Common/CMSSW_14_1_0_pre4/src
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval $(scramv1 runtime -sh)

export PATH=$REPO_DIR/SignalRegionStudyV2/python:$PATH
export LD_LIBRARY_PATH=$REPO_DIR/SignalRegionStudyV2/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$REPO_DIR/Common/Tools:$PYTHONPATH

# Create local WORKDIR structure
mkdir -p "$LOCAL_WORKDIR"

# Copy config files (small, needed for processing)
mkdir -p "$LOCAL_WORKDIR/SignalRegionStudyV2"
cp -r "$REPO_DIR/SignalRegionStudyV2/configs" "$LOCAL_WORKDIR/SignalRegionStudyV2/"

mkdir -p "$LOCAL_WORKDIR/Common/Data"
cp -r "$REPO_DIR/Common/Data"/* "$LOCAL_WORKDIR/Common/Data/"

mkdir -p "$LOCAL_WORKDIR/TriLepton/results"
cp -r "$REPO_DIR/TriLepton/results"/* "$LOCAL_WORKDIR/TriLepton/results/" 2>/dev/null || true

# Symlink pnfs SKNanoOutput for input (read via NFS)
ln -sf "$PNFS_BASE/SKNanoOutput" "$LOCAL_WORKDIR/SKNanoOutput"

# For scaled signal: link to existing preprocessed samples on pnfs
mkdir -p "$LOCAL_WORKDIR/SignalRegionStudyV1"
ln -sf "$PNFS_BASE/SignalRegionStudyV1/samples" "$LOCAL_WORKDIR/SignalRegionStudyV1/samples" 2>/dev/null || true
ln -sf "$PNFS_BASE/SignalRegionStudyV2/samples" "$LOCAL_WORKDIR/SignalRegionStudyV2/samples_source" 2>/dev/null || true

# Run preprocessing with local WORKDIR
export WORKDIR="$LOCAL_WORKDIR"
python3 "$REPO_DIR/SignalRegionStudyV2/python/preprocess.py" \
    --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" $EXTRA_ARGS

# Copy output to pnfs
LOCAL_OUTPUT="$LOCAL_WORKDIR/SignalRegionStudyV2/samples/$ERA/$CHANNEL/$MASSPOINT"
PNFS_OUTPUT="$PNFS_BASE/SignalRegionStudyV2/samples/$ERA/$CHANNEL/$MASSPOINT"

mkdir -p "$PNFS_OUTPUT"
cp -r "$LOCAL_OUTPUT"/* "$PNFS_OUTPUT/"

# Cleanup
rm -rf "$LOCAL_WORKDIR"

echo "Successfully preprocessed $ERA/$CHANNEL/$MASSPOINT -> $PNFS_OUTPUT"
