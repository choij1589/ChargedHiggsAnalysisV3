#!/bin/bash
# Shared DAGMan job-directory helpers for automize/ drivers.
# Source this file; do not execute it.
#
# The per-node JOB/VARS emission and the jobs.sub heredoc stay in each driver
# (they are genuinely workflow-specific); everything around them — timestamped
# job directory, dagman.config copy, submit_all.sh / status_all.sh — is
# defined once here.

DAG_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DAG_MODULE_DIR="$(dirname "$DAG_LIB_DIR")"

# Create a timestamped job directory under condor/ and print its path.
# Usage: job_dir=$(dag_new_jobdir <prefix>)
dag_new_jobdir() {
    local prefix=$1
    local timestamp
    # PID suffix: two driver invocations within the same second must never
    # share a job dir (a reused dir makes submit_all.sh resubmit earlier
    # DAGs).
    timestamp="$(date +%Y%m%d_%H%M%S)_$$"
    local job_dir="$DAG_MODULE_DIR/condor/jobs_${prefix}_${timestamp}"
    mkdir -p "$job_dir"
    cp "$DAG_MODULE_DIR/configs/dagman.config" "$job_dir/"
    echo "$job_dir"
}

# Prepare a per-masspoint subdirectory (logs/ + dagman.config) inside a job
# directory and print its path.
# Usage: mp_dir=$(dag_new_masspoint_dir <job_dir> <masspoint>)
dag_new_masspoint_dir() {
    local job_dir=$1
    local masspoint=$2
    local mp_dir="$job_dir/$masspoint"
    mkdir -p "$mp_dir/logs"
    cp "$job_dir/dagman.config" "$mp_dir/"
    echo "$mp_dir"
}

# Write submit_all.sh into a job directory.
# Usage: dag_write_submit_all <job_dir>
dag_write_submit_all() {
    local job_dir=$1
    cat > "$job_dir/submit_all.sh" << 'EOF'
#!/bin/bash
set -e
for mp_dir in */; do
    if [[ -f "$mp_dir/dag.dag" ]]; then
        echo "Submitting DAG for ${mp_dir%/}..."
        (cd "$mp_dir" && condor_submit_dag -f dag.dag)
    fi
done
echo "All DAGs submitted!"
EOF
    chmod +x "$job_dir/submit_all.sh"
}

# Write status_all.sh into a job directory.
# Usage: dag_write_status_all <job_dir>
dag_write_status_all() {
    local job_dir=$1
    cat > "$job_dir/status_all.sh" << 'EOF'
#!/bin/bash
echo "DAG Status:"
echo "==========="
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
EOF
    chmod +x "$job_dir/status_all.sh"
}

# Submit (or dry-run) every per-masspoint DAG in a job directory.
# Usage: dag_submit_or_dryrun <job_dir> <dry_run:true|false>
dag_submit_or_dryrun() {
    local job_dir=$1
    local dry_run=$2
    echo "Generated DAGMan workflows in: $job_dir"
    if [[ "$dry_run" == "true" ]]; then
        echo "[DRY-RUN] To submit: cd $job_dir && ./submit_all.sh"
        echo "[DRY-RUN] To check status: cd $job_dir && ./status_all.sh"
    else
        cd "$job_dir" && ./submit_all.sh
        cd - > /dev/null
        echo ""
        echo "Monitor with:"
        echo "  condor_q -dag"
        echo "  cd $job_dir && ./status_all.sh"
    fi
}
