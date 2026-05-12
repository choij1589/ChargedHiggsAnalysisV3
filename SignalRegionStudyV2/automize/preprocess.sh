#!/bin/bash
set -euo pipefail

# voms-proxy-init --voms=cms --valid=168:00  # skipped: existing /tmp/x509up_u$(id -u) is valid

# Mass points (loaded from configs/masspoints.json)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/load_masspoints.sh"

# Eras
ERAs_RUN2=("2016preVFP" "2016postVFP" "2017" "2018")
ERAs_RUN3=("2022" "2022EE" "2023" "2023BPix")

# Channels
CHANNELs_SR=("SR1E2Mu" "SR3Mu")
CHANNELs_TTZ=("TTZ2E1Mu")  # Only for ParticleNet masspoints

# HTCondor settings
DRY_RUN=false
CONDOR_DIR="${SCRIPT_DIR}/../condor"

# Export PATH to include python scripts
export PATH="${SCRIPT_DIR}/../python:${PATH}"

# Function to submit HTCondor jobs
# Args: eras_array masspoints_array channels_array extra_args
function submit_condor_jobs() {
    local -n eras_ref=$1
    local -n masspoints_ref=$2
    local -n channels_ref=$3
    local extra_args=${4:-}

    # Create timestamped job directory
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local job_dir="$CONDOR_DIR/jobs_${timestamp}"
    mkdir -p "$job_dir/logs"

    # Generate job_params.txt with all era/channel/masspoint/extra_args combinations
    local params_file="$job_dir/job_params.txt"
    > "$params_file"

    for era in "${eras_ref[@]}"; do
        for channel in "${channels_ref[@]}"; do
            for masspoint in "${masspoints_ref[@]}"; do
                echo "$era,$channel,$masspoint,$extra_args" >> "$params_file"
            done
        done
    done

    local num_jobs=$(wc -l < "$params_file")
    echo "Generated $num_jobs jobs in $params_file"

    # Generate HTCondor submission file
    local sub_file="$job_dir/preprocess.sub"
    cat > "$sub_file" << CONDOR_SUB
JobBatchName = preprocess
universe = vanilla
executable = WRAPPER_PATH
arguments = \$(era) \$(channel) \$(masspoint) \$(extra_args)
output = logs/preprocess_\$(era)_\$(channel)_\$(masspoint).out
error = logs/preprocess_\$(era)_\$(channel)_\$(masspoint).err
log = preprocess.log

request_cpus = 1
request_memory = 2GB
request_disk = 1GB

should_transfer_files = NO
use_x509userproxy = True
x509userproxy = /tmp/x509up_u$(id -u)

queue era,channel,masspoint,extra_args from job_params.txt
CONDOR_SUB

    # Replace WRAPPER_PATH with actual path
    sed -i "s|WRAPPER_PATH|$SCRIPT_DIR/../scripts/preprocess_wrapper.sh|g" "$sub_file"

    echo "Created submission file: $sub_file"

    # Submit or dry-run
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] Would submit: condor_submit $sub_file"
        echo "[DRY-RUN] Job directory: $job_dir"
        echo "[DRY-RUN] First 10 jobs:"
        head -10 "$params_file" | while IFS=, read -r era channel masspoint extra; do
            echo "  era=$era channel=$channel masspoint=$masspoint extra_args=$extra"
        done
    else
        echo "Submitting to HTCondor..."
        cd "$job_dir"
        condor_submit preprocess.sub
        cd - > /dev/null
    fi
}

# Parse command line arguments
MODE=""    # Options: run2, run3, all
UNBLIND=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --unblind)
            UNBLIND=true
            shift
            ;;
        --condor)
            echo "NOTE: --condor is now the default (and only) execution mode. Flag ignored."
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 --mode <run2|run3|all> [--unblind] [--dry-run]"
            echo ""
            echo "Modes:"
            echo "  run2  - Process Run2 only"
            echo "  run3  - Process Run3 only (real signal MC)"
            echo "  all   - Process Run2 + Run3 as a DAG (per-masspoint)"
            echo ""
            echo "Options:"
            echo "  --unblind    - Use the curated unblind mass-point subset"
            echo "                 (configs/masspoints.json: unblind.{baseline,particlenet})."
            echo "                 Implies --mode all (or omit --mode)."
            echo "  --condor     - (No-op, condor is now the only execution mode)"
            echo "  --dry-run    - Generate submission files without submitting"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# --unblind: filter mass points and force --mode all
if [[ "$UNBLIND" == "true" ]]; then
    if [[ -z "$MODE" ]]; then
        MODE="all"
    elif [[ "$MODE" != "all" ]]; then
        echo "Error: --unblind requires --mode all (or omit --mode)."
        exit 1
    fi

    # Override the master baseline list with the deduplicated union of unblind subsets.
    # is_particlenet still resolves against the unchanged MASSPOINTs_PARTICLENET, so per-point
    # PN classification (which gates TTZ2E1Mu jobs) keeps working.
    mapfile -t MASSPOINTs_BASELINE < <(
        printf '%s\n' "${MASSPOINTs_UNBLIND_BASELINE[@]}" "${MASSPOINTs_UNBLIND_PN[@]}" \
        | awk 'NF && !seen[$0]++'
    )
    if [[ ${#MASSPOINTs_BASELINE[@]} -eq 0 ]]; then
        echo "Error: --unblind selected an empty mass-point set (check configs/masspoints.json)."
        exit 1
    fi
fi

# Validate mode is specified
if [[ -z "$MODE" ]]; then
    echo "Error: --mode is required. Use --help for usage."
    exit 1
fi

if [[ "$MODE" != "run2" && "$MODE" != "run3" && "$MODE" != "all" ]]; then
    echo "Error: Invalid mode '$MODE'. Must be one of: run2, run3, all"
    exit 1
fi

echo "============================================================"
echo "SignalRegionStudyV2 Preprocessing"
echo "Mode: $MODE"
echo "Unblind subset: $UNBLIND"
echo "Execution: HTCondor (dry-run: $DRY_RUN)"
echo "============================================================"

# =============================================================================
# DAGMan workflow for --mode all
# =============================================================================

# Helper: check if a masspoint is a ParticleNet masspoint
function is_particlenet() {
    local mp=$1
    [[ " ${MASSPOINTs_PARTICLENET[*]} " =~ " ${mp} " ]]
}

# Generate DAG file for a single masspoint.
# Emits independent Run2 + Run3 jobs (real MC only — no scaling). preprocess.py raises
# FileNotFoundError on Run3 eras for masspoints without real Run3 MC; that node fails
# while the other DAG nodes complete.
function generate_preprocess_dag_file() {
    local masspoint=$1
    local dag_file=$2
    local is_pn=$3      # true/false

    cat > "$dag_file" << EOF
# Preprocess DAG for $masspoint
CONFIG dagman.config

EOF

    # Determine channels for this masspoint
    local -a channels_all=("${CHANNELs_SR[@]}")
    if [[ "$is_pn" == "true" ]]; then
        channels_all+=("${CHANNELs_TTZ[@]}")
    fi

    local mp="$masspoint"  # shorthand for node names

    # Run2 jobs (all eras, all applicable channels)
    for era in "${ERAs_RUN2[@]}"; do
        for channel in "${channels_all[@]}"; do
            echo "JOB ${mp}_run2_${channel}_${era} jobs.sub" >> "$dag_file"
            echo "VARS ${mp}_run2_${channel}_${era} era=\"${era}\" channel=\"${channel}\" masspoint=\"${masspoint}\" extra_args=\"\"" >> "$dag_file"
        done
    done

    # Run3 jobs (all eras, all applicable channels) — independent of Run2
    for era in "${ERAs_RUN3[@]}"; do
        for channel in "${channels_all[@]}"; do
            echo "JOB ${mp}_run3_${channel}_${era} jobs.sub" >> "$dag_file"
            echo "VARS ${mp}_run3_${channel}_${era} era=\"${era}\" channel=\"${channel}\" masspoint=\"${masspoint}\" extra_args=\"\"" >> "$dag_file"
        done
    done
}

# Submit DAGs for all masspoints
function submit_preprocess_dags() {
    local timestamp=$(date +%Y%m%d_%H%M%S)
    local job_dir="$CONDOR_DIR/jobs_dag_preprocess_all_${timestamp}"
    mkdir -p "$job_dir"

    local wrapper_path
    wrapper_path="$(cd "$SCRIPT_DIR/.." && pwd)/scripts/preprocess_wrapper.sh"

    for masspoint in "${MASSPOINTs_BASELINE[@]}"; do
        local mp_dir="$job_dir/$masspoint"
        mkdir -p "$mp_dir/logs"

        local is_pn="false"
        is_particlenet "$masspoint" && is_pn="true"

        # Create jobs.sub
        cat > "$mp_dir/jobs.sub" << EOF
JobBatchName = preprocess_${masspoint}
universe = vanilla
executable = ${wrapper_path}
arguments = \$(era) \$(channel) \$(masspoint) \$(extra_args)
output = logs/\$(era)_\$(channel).out
error = logs/\$(era)_\$(channel).err
log = dag.log
request_cpus = 1
request_memory = 2GB
request_disk = 1GB
should_transfer_files = NO
use_x509userproxy = True
x509userproxy = /tmp/x509up_u$(id -u)
queue
EOF

        # Copy dagman.config
        cp "$SCRIPT_DIR/../configs/dagman.config" "$mp_dir/"

        # Generate DAG file
        generate_preprocess_dag_file "$masspoint" "$mp_dir/dag.dag" "$is_pn"

        echo "Generated DAG: $mp_dir/dag.dag (pn=$is_pn)"
    done

    # Create submit_all.sh
    cat > "$job_dir/submit_all.sh" << 'SUBMIT_EOF'
#!/bin/bash
set -e
for mp_dir in */; do
    if [[ -f "$mp_dir/dag.dag" ]]; then
        echo "Submitting DAG for ${mp_dir%/}..."
        (cd "$mp_dir" && condor_submit_dag dag.dag)
    fi
done
echo "All DAGs submitted!"
SUBMIT_EOF
    chmod +x "$job_dir/submit_all.sh"

    # Create status_all.sh
    cat > "$job_dir/status_all.sh" << 'STATUS_EOF'
#!/bin/bash
echo "Preprocess DAG Status:"
echo "======================"
for mp_dir in */; do
    if [[ -f "$mp_dir/dag.dag" ]]; then
        mp_name="${mp_dir%/}"
        if [[ -f "$mp_dir/dag.dag.dagman.out" ]]; then
            done=$(grep -c "ULOG_JOB_TERMINATED" "$mp_dir/dag.dag.dagman.out" 2>/dev/null || echo 0)
            total=$(grep -c "^JOB " "$mp_dir/dag.dag" 2>/dev/null || echo 0)
            echo "$mp_name: $done/$total jobs completed"
        else
            echo "$mp_name: not started"
        fi
    fi
done
STATUS_EOF
    chmod +x "$job_dir/status_all.sh"

    echo ""
    echo "========================================"
    echo "Generated DAGMan workflows in: $job_dir"
    echo "Total masspoints: ${#MASSPOINTs_BASELINE[@]}"
    echo ""

    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] To submit all DAGs:"
        echo "  cd $job_dir && ./submit_all.sh"
        echo ""
        echo "[DRY-RUN] To check status:"
        echo "  cd $job_dir && ./status_all.sh"
        echo "  condor_q -dag"
    else
        echo "Submitting all DAGs..."
        cd "$job_dir" && ./submit_all.sh
        cd - > /dev/null
        echo ""
        echo "Monitor with:"
        echo "  condor_q -dag"
        echo "  cd $job_dir && ./status_all.sh"
    fi
}

# =============================================================================
# Execution
# =============================================================================

# --mode all: submit everything as a DAG with dependencies
if [[ "$MODE" == "all" ]]; then
    echo ""
    echo "============================================================"
    echo "Submitting all preprocessing as DAGMan workflows..."
    echo "  Run2 + Run3 (real MC) per mass point: ${#MASSPOINTs_BASELINE[@]} mass points"
    echo "============================================================"
    submit_preprocess_dags
    echo ""
    echo "============================================================"
    echo "All preprocessing DAGs submitted!"
    echo "============================================================"
    exit 0
fi

# Run2 processing
if [[ "$MODE" == "run2" ]]; then
    # Compute baseline-only mass points (BASELINE minus PARTICLENET)
    MASSPOINTs_BASELINE_ONLY=()
    for mp in "${MASSPOINTs_BASELINE[@]}"; do
        is_particlenet "$mp" || MASSPOINTs_BASELINE_ONLY+=("$mp")
    done

    echo ""
    echo "============================================================"
    echo "Processing Run2 eras..."
    echo "============================================================"

    echo "Submitting Run2 jobs to HTCondor..."
    if [[ ${#MASSPOINTs_BASELINE_ONLY[@]} -gt 0 ]]; then
        echo "  Submitting baseline-only mass points (SR channels)..."
        submit_condor_jobs ERAs_RUN2 MASSPOINTs_BASELINE_ONLY CHANNELs_SR ""
    fi
    echo "  Submitting ParticleNet mass points (SR channels)..."
    submit_condor_jobs ERAs_RUN2 MASSPOINTs_PARTICLENET CHANNELs_SR ""
    echo "  Submitting ParticleNet mass points (TTZ2E1Mu channel)..."
    submit_condor_jobs ERAs_RUN2 MASSPOINTs_PARTICLENET CHANNELs_TTZ ""

    echo "Run2 preprocessing complete!"
fi

# Run3 processing (real signal MC). Mass points without Run3 MC will fail at preprocess.py
# with FileNotFoundError; that is expected and surfaces missing inputs explicitly.
if [[ "$MODE" == "run3" ]]; then
    MASSPOINTs_BASELINE_ONLY=()
    for mp in "${MASSPOINTs_BASELINE[@]}"; do
        is_particlenet "$mp" || MASSPOINTs_BASELINE_ONLY+=("$mp")
    done

    echo ""
    echo "============================================================"
    echo "Processing Run3 eras..."
    echo "============================================================"

    echo "Submitting Run3 jobs to HTCondor..."
    if [[ ${#MASSPOINTs_BASELINE_ONLY[@]} -gt 0 ]]; then
        echo "  Submitting baseline-only mass points (SR channels)..."
        submit_condor_jobs ERAs_RUN3 MASSPOINTs_BASELINE_ONLY CHANNELs_SR ""
    fi
    if [[ ${#MASSPOINTs_PARTICLENET[@]} -gt 0 ]]; then
        echo "  Submitting ParticleNet mass points (SR channels)..."
        submit_condor_jobs ERAs_RUN3 MASSPOINTs_PARTICLENET CHANNELs_SR ""
        echo "  Submitting ParticleNet mass points (TTZ2E1Mu channel)..."
        submit_condor_jobs ERAs_RUN3 MASSPOINTs_PARTICLENET CHANNELs_TTZ ""
    fi

    echo "Run3 preprocessing complete!"
fi

echo ""
echo "============================================================"
echo "All preprocessing complete!"
echo "============================================================"
