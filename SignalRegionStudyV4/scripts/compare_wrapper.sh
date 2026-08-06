#!/bin/bash
# HTCondor wrapper for python/compareToV3.py.
#
# The samples stage opens every tree of every preprocessed file (heavy I/O),
# so reproduction comparisons run on condor, not on the login node.
# All arguments are passed through to compareToV3.py.
set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
srs_setup_cmssw

export WORKDIR="$SRS_REPO_DIR"
cd "$SRS_REPO_DIR/$SRS_MODULE_NAME"
export PATH="${PWD}/python:${PATH}"

python3 python/compareToV3.py "$@"
