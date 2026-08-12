#!/bin/bash
# Shared environment definitions for SignalRegionStudyV4 shell scripts and
# HTCondor wrappers. This is the single place where site constants and the
# module location are defined — no other shell script may hard-code them.
#
# Self-locates via BASH_SOURCE, which works both interactively and when a
# wrapper runs as a condor executable (jobs use the absolute NFS repo path
# with should_transfer_files = NO).

SRS_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRS_MODULE_DIR="$(dirname "$SRS_ENV_SCRIPT_DIR")"
SRS_MODULE_NAME="$(basename "$SRS_MODULE_DIR")"
SRS_REPO_DIR="$(dirname "$SRS_MODULE_DIR")"

# Site constants (overridable via environment)
: "${PNFS_USER_BASE:=/pnfs/knu.ac.kr/data/cms/store/user/choij}"
: "${XROOTD_USER_BASE:=root://cluster142.knu.ac.kr//store/user/choij}"
: "${SRS_CMSSW_REL:=CMSSW_14_1_0_pre4}"

export SRS_MODULE_DIR SRS_MODULE_NAME SRS_REPO_DIR
export PNFS_USER_BASE XROOTD_USER_BASE SRS_CMSSW_REL

# Source the CMSSW + Combine environment. Callers keep their own PATH /
# PYTHONPATH / LD_LIBRARY_PATH exports on top of this.
srs_setup_cmssw() {
    source /cvmfs/cms.cern.ch/cmsset_default.sh
    local cmssw_dir="$SRS_REPO_DIR/Common/$SRS_CMSSW_REL/src"
    if [[ ! -d "$cmssw_dir" ]]; then
        echo "ERROR: CMSSW directory not found: $cmssw_dir" >&2
        exit 1
    fi
    cd "$cmssw_dir"
    eval $(scramv1 runtime -sh)
    cd - > /dev/null
}

# SR3Mu pairing variant for a mass point (mirrors srspaths.pairing_variant):
# highM iff mHc >= 100 && mA >= 60.
# Usage: srs_pairing_variant MHc130_MA90  ->  highM
srs_pairing_variant() {
    local mp=$1
    local mhc=${mp%%_*}; mhc=${mhc#MHc}
    local ma=${mp##*_MA}
    if (( mhc >= 100 && ma >= 60 )); then
        echo "highM"
    else
        echo "lowM"
    fi
}

# Method directory segment for template paths ({method} or {method}_blind).
# The only shell-side construction site; python-side equivalent lives in
# python/srspaths.py (method_segment).
# Usage: srs_method_segment METHOD "EXTRA_ARGS"
srs_method_segment() {
    local method=$1
    local extra_args=${2:-}
    if [[ "$extra_args" == *"--blind"* ]]; then
        echo "${method}_blind"
    else
        echo "$method"
    fi
}

# Interpolation-study output directory for one mHc (mirrors
# srspaths.interpolation_dir). Usage: srs_interp_dir 160 [variant]
# The optional variant selects the fit-model variant tree
# (tests/interpolation/variants/{variant}/MHc{X}).
srs_interp_dir() {
    if [[ -n "${2:-}" ]]; then
        echo "$SRS_MODULE_DIR/tests/interpolation/variants/$2/MHc$1"
    else
        echo "$SRS_MODULE_DIR/tests/interpolation/MHc$1"
    fi
}
