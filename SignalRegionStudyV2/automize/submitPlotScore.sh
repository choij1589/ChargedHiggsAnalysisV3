#!/bin/bash
set -euo pipefail

# Default values
BINNING="extended"
UNBLIND_OPT=""
CHANNELS=("SR1E2Mu" "SR3Mu")
MODE=""  # run2 or run3
MASSPOINTS=()
DRY_RUN=false

# Parse arguments
usage() {
    echo "Usage: $0 --mode <run2|run3> [OPTIONS]"
    echo ""
    echo "Submit DAG workflow for plotParticleNetScore.py"
    echo "Single-era jobs run first, then combination job runs after all complete."
    echo ""
    echo "Required:"
    echo "  --mode MODE         Mode: 'run2' or 'run3'"
    echo ""
    echo "Options:"
    echo "  --channel CHANNEL   Channel: SR1E2Mu, SR3Mu (can be repeated, default: both)"
    echo "  --masspoint MP      Masspoint like MHc130_MA90 (can be repeated)"
    echo "  --all-masspoints    Use all available masspoints"
    echo "  --binning BINNING   Binning scheme (default: extended)"
    echo "  --partial-unblind   Enable partial unblinding"
    echo "  --unblind           Enable full unblinding"
    echo "  --dry-run           Print what would be submitted without submitting"
    echo "  -h, --help          Show this help"
    echo ""
    echo "Examples:"
    echo "  $0 --mode run2 --channel SR1E2Mu --masspoint MHc130_MA90"
    echo "  $0 --mode run2 --all-masspoints --partial-unblind"
    echo "  $0 --mode run3 --channel SR1E2Mu --all-masspoints"
}

CUSTOM_CHANNELS=()
ALL_MASSPOINTS=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --channel)
            CUSTOM_CHANNELS+=("$2")
            shift 2
            ;;
        --masspoint)
            MASSPOINTS+=("$2")
            shift 2
            ;;
        --all-masspoints)
            ALL_MASSPOINTS=true
            shift
            ;;
        --binning)
            BINNING="$2"
            shift 2
            ;;
        --partial-unblind)
            UNBLIND_OPT="--partial-unblind"
            shift
            ;;
        --unblind)
            UNBLIND_OPT="--unblind"
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate mode
if [ -z "$MODE" ]; then
    echo "Error: --mode is required (run2 or run3)"
    usage
    exit 1
fi

# Set eras based on mode
case "$MODE" in
    run2|Run2|RUN2)
        SINGLE_ERAS=("2016preVFP" "2016postVFP" "2017" "2018")
        COMBINED_ERA="Run2"
        ;;
    run3|Run3|RUN3)
        SINGLE_ERAS=("2022" "2022EE" "2023" "2023BPix")
        COMBINED_ERA="Run3"
        ;;
    *)
        echo "Error: Invalid mode '$MODE'. Use 'run2' or 'run3'"
        exit 1
        ;;
esac

# Use custom channels if provided
if [ ${#CUSTOM_CHANNELS[@]} -gt 0 ]; then
    CHANNELS=("${CUSTOM_CHANNELS[@]}")
fi

# Get all masspoints if requested
if [ "$ALL_MASSPOINTS" = true ]; then
    MASSPOINTS=(
        "MHc100_MA95" "MHc115_MA87" "MHc130_MA90" 
        "MHc145_MA92" "MHc160_MA85" "MHc160_MA98"
    )
fi

# Validate
if [ ${#MASSPOINTS[@]} -eq 0 ]; then
    echo "Error: No masspoints specified. Use --masspoint or --all-masspoints"
    exit 1
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
CONDOR_DIR="$REPO_DIR/condor"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_DIR="$CONDOR_DIR/jobs_dag_${MODE}_${TIMESTAMP}"

echo "=== Plot Score DAG Workflow ==="
echo "Mode: $MODE"
echo "Single eras: ${SINGLE_ERAS[*]}"
echo "Combined era: $COMBINED_ERA"
echo "Channels: ${CHANNELS[*]}"
echo "Masspoints: ${#MASSPOINTS[@]} total"
echo "Binning: $BINNING"
echo "Unblind option: ${UNBLIND_OPT:-none}"
echo "Job directory: $JOB_DIR"
echo ""

# Create job directory structure
mkdir -p "$JOB_DIR/logs"

# Create submit file template for single-era jobs
SINGLE_SUB="$JOB_DIR/single_era.sub"
cat > "$SINGLE_SUB" << EOF
universe = vanilla
executable = $CONDOR_DIR/plotScore_wrapper.sh
arguments = \$(era) \$(channel) \$(masspoint) \$(extra_args)
output = logs/single_\$(era)_\$(channel)_\$(masspoint).out
error = logs/single_\$(era)_\$(channel)_\$(masspoint).err
log = single_era.log

request_cpus = 1
request_memory = 4GB
request_disk = 2GB

should_transfer_files = NO

queue
EOF

# Create submit file template for combined-era jobs
COMBINED_SUB="$JOB_DIR/combined_era.sub"
cat > "$COMBINED_SUB" << EOF
universe = vanilla
executable = $CONDOR_DIR/plotScore_wrapper.sh
arguments = \$(era) \$(channel) \$(masspoint) \$(extra_args)
output = logs/combined_\$(era)_\$(channel)_\$(masspoint).out
error = logs/combined_\$(era)_\$(channel)_\$(masspoint).err
log = combined_era.log

request_cpus = 1
request_memory = 2GB
request_disk = 1GB

should_transfer_files = NO

queue
EOF

# Create DAG file
DAG_FILE="$JOB_DIR/workflow.dag"
> "$DAG_FILE"

# Track job counts
SINGLE_JOB_COUNT=0
COMBINED_JOB_COUNT=0

for channel in "${CHANNELS[@]}"; do
    for mp in "${MASSPOINTS[@]}"; do
        # Create single-era jobs for this channel/masspoint
        PARENT_JOBS=()

        for era in "${SINGLE_ERAS[@]}"; do
            JOB_NAME="single_${era}_${channel}_${mp}"

            echo "JOB $JOB_NAME $SINGLE_SUB" >> "$DAG_FILE"
            echo "VARS $JOB_NAME era=\"$era\" channel=\"$channel\" masspoint=\"$mp\" extra_args=\"$UNBLIND_OPT\"" >> "$DAG_FILE"

            PARENT_JOBS+=("$JOB_NAME")
            SINGLE_JOB_COUNT=$((SINGLE_JOB_COUNT + 1))
        done

        # Create combined-era job that depends on all single-era jobs
        COMBINED_JOB_NAME="combined_${COMBINED_ERA}_${channel}_${mp}"

        echo "JOB $COMBINED_JOB_NAME $COMBINED_SUB" >> "$DAG_FILE"
        echo "VARS $COMBINED_JOB_NAME era=\"$COMBINED_ERA\" channel=\"$channel\" masspoint=\"$mp\" extra_args=\"$UNBLIND_OPT\"" >> "$DAG_FILE"

        # Add parent-child relationship
        echo "PARENT ${PARENT_JOBS[*]} CHILD $COMBINED_JOB_NAME" >> "$DAG_FILE"
        echo "" >> "$DAG_FILE"

        COMBINED_JOB_COUNT=$((COMBINED_JOB_COUNT + 1))
    done
done

TOTAL_JOBS=$((SINGLE_JOB_COUNT + COMBINED_JOB_COUNT))
echo "Generated DAG with:"
echo "  - $SINGLE_JOB_COUNT single-era jobs (${#SINGLE_ERAS[@]} eras x ${#CHANNELS[@]} channels x ${#MASSPOINTS[@]} masspoints)"
echo "  - $COMBINED_JOB_COUNT combined-era jobs (${#CHANNELS[@]} channels x ${#MASSPOINTS[@]} masspoints)"
echo "  - $TOTAL_JOBS total jobs"
echo ""

# Submit or dry-run
if [ "$DRY_RUN" = true ]; then
    echo "=== DRY RUN - Would submit: ==="
    echo "condor_submit_dag $DAG_FILE"
    echo ""
    echo "DAG structure (first 30 lines):"
    head -30 "$DAG_FILE"
    echo "..."
else
    cd "$JOB_DIR"
    condor_submit_dag "$DAG_FILE"
    echo ""
    echo "Submitted DAG workflow"
    echo "Monitor with: condor_q -dag"
    echo "DAG status:   condor_q -nobatch"
    echo "Logs in:      $JOB_DIR/logs/"
    echo "DAG log:      $JOB_DIR/workflow.dag.dagman.out"
fi
