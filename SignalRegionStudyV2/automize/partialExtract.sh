#!/bin/bash
#
# partialExtract.sh - Batch partial-extract of HybridNew limits via HTCondor DAG
#
# Generates a condor DAG with one job per mass point. Each job runs
# scripts/runHybridNew.sh --partial-extract for a single mass point.
#
# Usage:
#   bash automize/partialExtract.sh --mode all --method Baseline
#   bash automize/partialExtract.sh --mode all --method ParticleNet
#   bash automize/partialExtract.sh --mode all --method Baseline --dry-run
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Load mass point arrays
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_masspoints.sh"

# Defaults
MODE="all"
METHOD="Baseline"
BINNING="extended"
ERA="All"
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="${2,,}"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --binning)
            BINNING="$2"
            shift 2
            ;;
        --era)
            ERA="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode MODE      Processing mode: all (default)"
            echo "  --method METHOD  Baseline or ParticleNet (default: Baseline)"
            echo "  --binning BINNG  extended or uniform (default: extended)"
            echo "  --era ERA        Era to process (default: All)"
            echo "  --dry-run        Print DAG without submitting"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Select mass points
if [[ "$METHOD" == "ParticleNet" ]]; then
    MASSPOINTs=("${MASSPOINTs_PARTICLENET[@]}")
elif [[ "$METHOD" == "Baseline" ]]; then
    MASSPOINTs=("${MASSPOINTs_BASELINE[@]}")
else
    echo "ERROR: Unknown method '$METHOD'. Must be Baseline or ParticleNet"
    exit 1
fi

CONDOR_DIR="${SCRIPT_DIR}/condor/jobs_partialExtract_${METHOD}"
mkdir -p "${CONDOR_DIR}/logs"

WRAPPER="${SCRIPT_DIR}/scripts/partialExtract_wrapper.sh"
if [[ ! -f "$WRAPPER" ]]; then
    echo "ERROR: Wrapper not found: $WRAPPER"
    exit 1
fi

echo "============================================================"
echo "SignalRegionStudyV2 Partial-Extract Batch Submission"
echo "Method: $METHOD"
echo "Binning: $BINNING"
echo "Era: $ERA"
echo "Mass points: ${#MASSPOINTs[@]}"
echo "Condor dir: $CONDOR_DIR"
echo "Dry run: $DRY_RUN"
echo "============================================================"
echo ""

# ========== condor submit file ==========
SUB_FILE="${CONDOR_DIR}/partialExtract.sub"
cat > "$SUB_FILE" << EOF
universe            = vanilla
executable          = ${WRAPPER}
arguments           = \$(era) \$(masspoint) \$(method) \$(binning)
output              = ${CONDOR_DIR}/logs/job_\$(masspoint).out
error               = ${CONDOR_DIR}/logs/job_\$(masspoint).err
log                 = ${CONDOR_DIR}/partialExtract.log

request_cpus        = 1
request_memory      = 4GB
request_disk        = 2GB

should_transfer_files = NO

queue
EOF

# ========== DAG file ==========
DAG_FILE="${CONDOR_DIR}/partialExtract.dag"

{
    echo "# partialExtract DAG — generated $(date)"
    echo "# Method: $METHOD | Era: $ERA | Binning: $BINNING"
    echo "# Mass points: ${#MASSPOINTs[@]}"
    echo ""
    echo "CONFIG ${SCRIPT_DIR}/configs/dagman.config"
    echo ""

    for masspoint in "${MASSPOINTs[@]}"; do
        job_name="extract_${masspoint}"
        echo "JOB ${job_name} ${SUB_FILE}"
        echo "VARS ${job_name} era=\"${ERA}\" masspoint=\"${masspoint}\" method=\"${METHOD}\" binning=\"${BINNING}\""
        echo ""
    done
} > "$DAG_FILE"

echo "Generated DAG: $DAG_FILE"
echo "  Jobs: ${#MASSPOINTs[@]}"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY-RUN] Would submit:"
    echo "  condor_submit_dag $DAG_FILE"
    echo ""
    echo "DAG contents:"
    cat "$DAG_FILE"
else
    echo "Submitting DAG..."
    condor_submit_dag "$DAG_FILE"
    echo ""
    echo "Monitor with: condor_q -dag"
    echo ""
    echo "After completion, collect limits with:"
    echo "  python3 python/collectLimits.py --era $ERA --method $METHOD --limit_type HybridNew"
fi
