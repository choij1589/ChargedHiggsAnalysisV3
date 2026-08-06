#!/bin/bash
set -eo pipefail

ERA=$1
CHANNEL=$2
MASSPOINT=$3
EXTRA_ARGS=${4:-}

# Paths (site constants and module location come from env.sh)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
PNFS_BASE="$PNFS_USER_BASE"
SE_BASE="$XROOTD_USER_BASE"
REPO_DIR="$SRS_REPO_DIR"
LOCAL_WORKDIR="${TMPDIR:-/tmp}/workdir_$$"

# Setup CMSSW environment
srs_setup_cmssw

export PATH=$REPO_DIR/$SRS_MODULE_NAME/python:$PATH
export LD_LIBRARY_PATH=$REPO_DIR/$SRS_MODULE_NAME/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$REPO_DIR/Common/Tools:$PYTHONPATH

# Create local WORKDIR structure
mkdir -p "$LOCAL_WORKDIR"

# Copy config files (small, needed for processing)
mkdir -p "$LOCAL_WORKDIR/$SRS_MODULE_NAME"
cp -r "$REPO_DIR/$SRS_MODULE_NAME/configs" "$LOCAL_WORKDIR/$SRS_MODULE_NAME/"

mkdir -p "$LOCAL_WORKDIR/Common/Data"
cp -r "$REPO_DIR/Common/Data"/* "$LOCAL_WORKDIR/Common/Data/"

mkdir -p "$LOCAL_WORKDIR/TriLepton/results"
cp -r "$REPO_DIR/TriLepton/results"/* "$LOCAL_WORKDIR/TriLepton/results/" 2>/dev/null || true

# Symlink pnfs SKNanoOutput for input (read via NFS)
ln -sf "$PNFS_BASE/SKNanoOutput" "$LOCAL_WORKDIR/SKNanoOutput"

# For scaled signal: link to this module's preprocessed sample area on pnfs.
ln -sf "$PNFS_BASE/$SRS_MODULE_NAME/samples" "$LOCAL_WORKDIR/$SRS_MODULE_NAME/samples_source" 2>/dev/null || true

# Run preprocessing with local WORKDIR
export WORKDIR="$LOCAL_WORKDIR"
python3 "$REPO_DIR/$SRS_MODULE_NAME/python/preprocess.py" \
    --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" $EXTRA_ARGS

# Copy output to pnfs via xrootd (faster than NFS for writes)
LOCAL_OUTPUT="$LOCAL_WORKDIR/$SRS_MODULE_NAME/samples/$ERA/$CHANNEL/$MASSPOINT"
SE_OUTPUT="$SE_BASE/$SRS_MODULE_NAME/samples/$ERA/$CHANNEL/$MASSPOINT"
PNFS_OUTPUT="$PNFS_BASE/$SRS_MODULE_NAME/samples/$ERA/$CHANNEL/$MASSPOINT"

mkdir -p "$PNFS_OUTPUT"

for f in "$LOCAL_OUTPUT"/*.root; do
    xrdcp -s -f "$f" "$SE_OUTPUT/$(basename "$f")"
done

# Cleanup
rm -rf "$LOCAL_WORKDIR"

echo "Successfully preprocessed $ERA/$CHANNEL/$MASSPOINT -> $SE_OUTPUT"
