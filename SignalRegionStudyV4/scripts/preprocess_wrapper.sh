#!/bin/bash
set -eo pipefail

ERA=$1
CHANNEL=$2
MASSPOINT=$3
EXTRA_ARGS="${*:4}"

# Paths (site constants and module location come from env.sh)
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
PNFS_BASE="$PNFS_USER_BASE"
SE_BASE="$XROOTD_USER_BASE"
REPO_DIR="$SRS_REPO_DIR"
LOCAL_WORKDIR="${TMPDIR:-/tmp}/workdir_$$"

# Setup CMSSW environment
srs_setup_cmssw

export PATH=$REPO_DIR/$SRS_MODULE_NAME/python:$PATH
export PYTHONPATH=$REPO_DIR/Common/Tools:$PYTHONPATH

# Create local WORKDIR structure
mkdir -p "$LOCAL_WORKDIR"

# Copy config files (small, needed for processing)
mkdir -p "$LOCAL_WORKDIR/$SRS_MODULE_NAME"
cp -r "$REPO_DIR/$SRS_MODULE_NAME/configs" "$LOCAL_WORKDIR/$SRS_MODULE_NAME/"

mkdir -p "$LOCAL_WORKDIR/Common/Data"
cp -r "$REPO_DIR/Common/Data"/* "$LOCAL_WORKDIR/Common/Data/"

# Symlink pnfs SKNanoOutput for input (read via NFS)
ln -sf "$PNFS_BASE/SKNanoOutput" "$LOCAL_WORKDIR/SKNanoOutput"

# Run preprocessing with local WORKDIR. MASSPOINT may be the literal 'none'
# for shared-backgrounds nodes (which take no --masspoint).
export WORKDIR="$LOCAL_WORKDIR"
PREPROCESS_ARGS=(--era "$ERA" --channel "$CHANNEL")
if [[ -n "$MASSPOINT" && "$MASSPOINT" != "none" ]]; then
    PREPROCESS_ARGS+=(--masspoint "$MASSPOINT")
fi
python3 "$REPO_DIR/$SRS_MODULE_NAME/python/preprocess.py" \
    "${PREPROCESS_ARGS[@]}" $EXTRA_ARGS

# Copy everything preprocess.py wrote to pnfs via xrootd, mirroring the
# local samples subtree. Mode-agnostic: works for per-masspoint dirs,
# shared background dirs, and shared signal files alike.
LOCAL_SAMPLES="$LOCAL_WORKDIR/$SRS_MODULE_NAME/samples"
if [[ ! -d "$LOCAL_SAMPLES" ]]; then
    echo "ERROR: preprocess.py produced no samples/ output"
    exit 1
fi

cd "$LOCAL_SAMPLES"
find . -name "*.root" -type f | while read -r relpath; do
    relpath="${relpath#./}"
    destdir="$PNFS_BASE/$SRS_MODULE_NAME/samples/$(dirname "$relpath")"
    mkdir -p "$destdir"
    xrdcp -s -f "$LOCAL_SAMPLES/$relpath" \
        "$SE_BASE/$SRS_MODULE_NAME/samples/$relpath"
done
n_files=$(find . -name "*.root" -type f | wc -l)
cd - > /dev/null

# Cleanup
rm -rf "$LOCAL_WORKDIR"

echo "Successfully preprocessed $ERA/$CHANNEL/${MASSPOINT:-shared} ($n_files files) -> $SE_BASE/$SRS_MODULE_NAME/samples/"
