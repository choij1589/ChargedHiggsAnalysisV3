#!/bin/bash
set -euo pipefail

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
    cat > "$sub_file" << 'CONDOR_SUB'
JobBatchName = preprocess
universe = vanilla
executable = WRAPPER_PATH
arguments = $(era) $(channel) $(masspoint) $(extra_args)
output = logs/preprocess_$(era)_$(channel)_$(masspoint).out
error = logs/preprocess_$(era)_$(channel)_$(masspoint).err
log = preprocess.log

request_cpus = 1
request_memory = 2GB
request_disk = 1GB

should_transfer_files = NO

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

# Function to process Run3 mass points (shared by run3 and run3-scaled modes)
# Args: baseline_only_array pn_array signal_extra_args label
function process_run3() {
    local -n baseline_only_ref=$1
    local -n pn_ref=$2
    local signal_extra_args=$3
    local label=$4

    echo "Submitting Run3 ($label) jobs to HTCondor..."
    if [[ ${#baseline_only_ref[@]} -gt 0 ]]; then
        echo "  Submitting baseline-only mass points (SR channels)..."
        submit_condor_jobs ERAs_RUN3 "$1" CHANNELs_SR "$signal_extra_args"
    fi
    if [[ ${#pn_ref[@]} -gt 0 ]]; then
        echo "  Submitting ParticleNet mass points (SR channels)..."
        submit_condor_jobs ERAs_RUN3 "$2" CHANNELs_SR "$signal_extra_args"
        echo "  Submitting ParticleNet mass points (TTZ2E1Mu channel)..."
        submit_condor_jobs ERAs_RUN3 "$2" CHANNELs_TTZ ""
    fi
}

# Parse command line arguments
MODE=""  # Options: run2, run3, run3-scaled, all

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
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
            echo "Usage: $0 --mode <run2|run3|run3-scaled|all> [--dry-run]"
            echo ""
            echo "Modes:"
            echo "  run2         - Process Run2 only"
            echo "  run3         - Process Run3 with real signal MC (if available)"
            echo "  run3-scaled  - Process Run3 with signal scaled from 2018"
            echo "                 Requires: Run2 (2018) must be preprocessed first"
            echo "  all          - Process Run2 + Run3 (real MC + scaled) as a DAG"
            echo "                 (DAG enforces 2018 -> Run3-scaled deps)"
            echo ""
            echo "Options:"
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

# Validate mode is specified
if [[ -z "$MODE" ]]; then
    echo "Error: --mode is required. Use --help for usage."
    exit 1
fi

if [[ "$MODE" != "run2" && "$MODE" != "run3" && "$MODE" != "run3-scaled" && "$MODE" != "all" ]]; then
    echo "Error: Invalid mode '$MODE'. Must be one of: run2, run3, run3-scaled, all"
    exit 1
fi

# Validate options

echo "============================================================"
echo "SignalRegionStudyV2 Preprocessing"
echo "Mode: $MODE"
echo "Execution: HTCondor (dry-run: $DRY_RUN)"
echo "============================================================"

# =============================================================================
# DAGMan workflow for --mode all
# =============================================================================

# Helper: check if a masspoint has real Run3 MC
function has_real_run3_mc() {
    local mp=$1
    [[ " ${MASSPOINTs_Run3[*]} " =~ " ${mp} " ]]
}

# Helper: check if a masspoint is a ParticleNet masspoint
function is_particlenet() {
    local mp=$1
    [[ " ${MASSPOINTs_PARTICLENET[*]} " =~ " ${mp} " ]]
}

# Generate DAG file for a single masspoint
# Writes Run2 jobs, Run3 real/scaled jobs, and dependency lines
function generate_preprocess_dag_file() {
    local masspoint=$1
    local dag_file=$2
    local is_pn=$3      # true/false
    local is_run3=$4    # true if real Run3 MC available

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

    # --- Layer 0: Run2 jobs (all eras, all applicable channels) ---
    for era in "${ERAs_RUN2[@]}"; do
        for channel in "${channels_all[@]}"; do
            echo "JOB ${mp}_run2_${channel}_${era} jobs.sub" >> "$dag_file"
            echo "VARS ${mp}_run2_${channel}_${era} era=\"${era}\" channel=\"${channel}\" masspoint=\"${masspoint}\" extra_args=\"\"" >> "$dag_file"
        done
    done

    # --- Layer 1: Run3 jobs ---
    if [[ "$is_run3" == "true" ]]; then
        # Real Run3 MC: all independent (no deps on 2018)
        for era in "${ERAs_RUN3[@]}"; do
            for channel in "${channels_all[@]}"; do
                echo "JOB ${mp}_run3_${channel}_${era} jobs.sub" >> "$dag_file"
                echo "VARS ${mp}_run3_${channel}_${era} era=\"${era}\" channel=\"${channel}\" masspoint=\"${masspoint}\" extra_args=\"\"" >> "$dag_file"
            done
        done
    else
        # Scaled from Run2: SR channels need --scale-from-run2, depend on 2018
        for era in "${ERAs_RUN3[@]}"; do
            for channel in "${CHANNELs_SR[@]}"; do
                echo "JOB ${mp}_run3s_${channel}_${era} jobs.sub" >> "$dag_file"
                echo "VARS ${mp}_run3s_${channel}_${era} era=\"${era}\" channel=\"${channel}\" masspoint=\"${masspoint}\" extra_args=\"--scale-from-run2\"" >> "$dag_file"
            done
        done
        # TTZ2E1Mu for ParticleNet: no signal, no scaling dep
        if [[ "$is_pn" == "true" ]]; then
            for era in "${ERAs_RUN3[@]}"; do
                echo "JOB ${mp}_run3_TTZ2E1Mu_${era} jobs.sub" >> "$dag_file"
                echo "VARS ${mp}_run3_TTZ2E1Mu_${era} era=\"${era}\" channel=\"TTZ2E1Mu\" masspoint=\"${masspoint}\" extra_args=\"\"" >> "$dag_file"
            done
        fi
    fi

    # --- Dependencies ---
    echo "" >> "$dag_file"
    echo "# Dependencies" >> "$dag_file"

    if [[ "$is_run3" == "false" ]]; then
        # Scaled Run3 SR jobs depend on their 2018 counterpart
        for channel in "${CHANNELs_SR[@]}"; do
            local run3s_jobs=""
            for era in "${ERAs_RUN3[@]}"; do
                run3s_jobs+="${mp}_run3s_${channel}_${era} "
            done
            echo "PARENT ${mp}_run2_${channel}_2018 CHILD $run3s_jobs" >> "$dag_file"
        done
    fi
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
        local is_run3="false"
        has_real_run3_mc "$masspoint" && is_run3="true"

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
queue
EOF

        # Copy dagman.config
        cp "$SCRIPT_DIR/../configs/dagman.config" "$mp_dir/"

        # Generate DAG file
        generate_preprocess_dag_file "$masspoint" "$mp_dir/dag.dag" "$is_pn" "$is_run3"

        echo "Generated DAG: $mp_dir/dag.dag (pn=$is_pn, run3_mc=$is_run3)"
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
    echo "  Run2: all eras, all mass points"
    echo "  Run3 real MC: ${MASSPOINTs_Run3[*]}"
    echo "  Run3 scaled: remaining mass points (depend on 2018)"
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

# Run3 processing with scaled signal (mass points without real Run3 MC)
if [[ "$MODE" == "run3-scaled" ]]; then
    # Compute scaled mass points (BASELINE minus Run3 real MC), split by ParticleNet
    MASSPOINTs_Run3_SCALED_BASELINE_ONLY=()
    MASSPOINTs_Run3_SCALED_PN=()
    for mp in "${MASSPOINTs_BASELINE[@]}"; do
        has_real_run3_mc "$mp" && continue
        if is_particlenet "$mp"; then
            MASSPOINTs_Run3_SCALED_PN+=("$mp")
        else
            MASSPOINTs_Run3_SCALED_BASELINE_ONLY+=("$mp")
        fi
    done

    echo ""
    echo "============================================================"
    echo "Processing Run3 eras (signal scaled from 2018)..."
    echo "  Baseline-only: ${MASSPOINTs_Run3_SCALED_BASELINE_ONLY[*]}"
    echo "  ParticleNet:   ${MASSPOINTs_Run3_SCALED_PN[*]}"
    echo "============================================================"

    process_run3 MASSPOINTs_Run3_SCALED_BASELINE_ONLY MASSPOINTs_Run3_SCALED_PN \
        "--scale-from-run2" "scaled signal"

    echo "Run3 (scaled signal) preprocessing complete!"
fi

# Run3 processing with real signal MC
if [[ "$MODE" == "run3" ]]; then
    # Split Run3 real MC mass points by ParticleNet
    MASSPOINTs_Run3_BASELINE_ONLY=()
    MASSPOINTs_Run3_PN=()
    for mp in "${MASSPOINTs_Run3[@]}"; do
        if is_particlenet "$mp"; then
            MASSPOINTs_Run3_PN+=("$mp")
        else
            MASSPOINTs_Run3_BASELINE_ONLY+=("$mp")
        fi
    done

    echo ""
    echo "============================================================"
    echo "Processing Run3 eras (real signal MC)..."
    echo "  Baseline-only: ${MASSPOINTs_Run3_BASELINE_ONLY[*]}"
    echo "  ParticleNet:   ${MASSPOINTs_Run3_PN[*]}"
    echo "============================================================"

    process_run3 MASSPOINTs_Run3_BASELINE_ONLY MASSPOINTs_Run3_PN \
        "" "real signal"

    echo "Run3 (real signal) preprocessing complete!"
fi

echo ""
echo "============================================================"
echo "All preprocessing complete!"
echo "============================================================"
