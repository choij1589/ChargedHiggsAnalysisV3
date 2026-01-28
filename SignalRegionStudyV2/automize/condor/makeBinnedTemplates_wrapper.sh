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
METHOD=$4
BINNING=$5
EXTRA_ARGS=${6:-}

# Paths
PNFS_BASE="/pnfs/knu.ac.kr/data/cms/store/user/choij"
REPO_DIR="/u/user/choij/scratch/ChargedHiggsAnalysisV3"
LOCAL_WORKDIR="${TMPDIR:-/tmp}/workdir_$$"

# Setup CMSSW environment from scratch
cd "$REPO_DIR/Common/CMSSW_14_1_0_pre4/src"
source /cvmfs/cms.cern.ch/cmsset_default.sh
eval $(scramv1 runtime -sh)

export PATH="$REPO_DIR/SignalRegionStudyV2/python:$PATH"
export LD_LIBRARY_PATH="$REPO_DIR/SignalRegionStudyV2/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$REPO_DIR/Common/Tools:$PYTHONPATH"

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

# Symlink preprocessed samples from pnfs
mkdir -p "$LOCAL_WORKDIR/SignalRegionStudyV2"
ln -sf "$PNFS_BASE/SignalRegionStudyV2/samples" "$LOCAL_WORKDIR/SignalRegionStudyV2/samples"

# Run template generation with local WORKDIR
export WORKDIR="$LOCAL_WORKDIR"
cd "$REPO_DIR/SignalRegionStudyV2"

python3 python/makeBinnedTemplates.py \
    --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
    --method "$METHOD" --binning "$BINNING" $EXTRA_ARGS

# Determine output directory based on extra args
BINNING_SUFFIX="$BINNING"
if [[ "$EXTRA_ARGS" == *"--partial-unblind"* ]]; then
    BINNING_SUFFIX="${BINNING}_partial_unblind"
elif [[ "$EXTRA_ARGS" == *"--unblind"* ]]; then
    BINNING_SUFFIX="${BINNING}_unblind"
fi

# Copy output templates to pnfs
LOCAL_OUTPUT="$LOCAL_WORKDIR/SignalRegionStudyV2/templates/$ERA/$CHANNEL/$MASSPOINT/$METHOD/$BINNING_SUFFIX"
PNFS_OUTPUT="$PNFS_BASE/SignalRegionStudyV2/templates/$ERA/$CHANNEL/$MASSPOINT/$METHOD/$BINNING_SUFFIX"

if [[ -d "$LOCAL_OUTPUT" ]]; then
    mkdir -p "$PNFS_OUTPUT"
    cp -r "$LOCAL_OUTPUT"/* "$PNFS_OUTPUT/"
    echo "Copied templates to $PNFS_OUTPUT"
else
    echo "WARNING: No output found at $LOCAL_OUTPUT"
fi

# Cleanup
rm -rf "$LOCAL_WORKDIR"

echo "Successfully processed $ERA/$CHANNEL/$MASSPOINT ($METHOD/$BINNING_SUFFIX)"
