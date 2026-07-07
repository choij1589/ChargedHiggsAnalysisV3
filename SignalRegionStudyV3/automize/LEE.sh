#!/bin/bash
#
# LEE.sh - DAG-based automation for the look-elsewhere-effect workflow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_masspoints.sh"

MASSPOINT="MHc70_MA18"
REFERENCE_MASSPOINT="MHc160_MA50"
STEP="1"
RUN_LOCAL=false
DRY_RUN=false
DEBUG=false
NTOYS=1000
START_TOY=1
SINGLE_TOY=""
FORCE=false
KEEP_WORKDIR=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --step)
            STEP="$2"
            shift 2
            ;;
        --masspoint)
            MASSPOINT="$2"
            shift 2
            ;;
        --reference-masspoint)
            REFERENCE_MASSPOINT="$2"
            shift 2
            ;;
        --condor)
            RUN_LOCAL=false
            shift
            ;;
        --local)
            RUN_LOCAL=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --debug)
            DEBUG=true
            shift
            ;;
        --ntoys)
            NTOYS="$2"
            shift 2
            ;;
        --start-toy)
            START_TOY="$2"
            shift 2
            ;;
        --toy)
            SINGLE_TOY="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --keep-workdir)
            KEEP_WORKDIR=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--step 1|2|3|4|5] [--masspoint MASSPOINT] [--condor|--local] [--dry-run]"
            echo ""
            echo "LEE workflow automation. Condor is the default; --condor is accepted as an explicit no-op."
            echo ""
            echo "Options:"
            echo "  --step N               LEE step number [default: 1]"
            echo "  --masspoint MASSPOINT  LEE model mass point [default: MHc70_MA18]"
            echo "  --reference-masspoint MASSPOINT"
            echo "                         Step 5 background sample comparison point [default: MHc160_MA50]"
            echo "  --condor               Submit via Condor (default)"
            echo "  --local                Run the step inline"
            echo "  --dry-run              Generate/print commands without submitting or running"
            echo "  --debug                Pass debug logging to the step script"
            echo "  --toy N                Steps 2-3: process one toy"
            echo "  --start-toy N          Steps 2-5: first toy index for --ntoys [default: 1]"
            echo "  --ntoys N              Steps 2-5: number of toys [default: 1000]"
            echo "  --force                Steps 2-3: overwrite complete existing outputs"
            echo "  --keep-workdir         Step 3: keep staged per-trial fit directories"
            exit 0
            ;;
        *)
            echo "ERROR: Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

case "$STEP" in
    1|2|3|4|5) ;;
    *) echo "ERROR: invalid --step '$STEP'. Valid: 1, 2, 3, 4, 5" >&2; exit 1 ;;
esac
case "$NTOYS" in
    ''|*[!0-9]*) echo "ERROR: --ntoys must be a positive integer" >&2; exit 1 ;;
esac
case "$START_TOY" in
    ''|*[!0-9]*) echo "ERROR: --start-toy must be a positive integer" >&2; exit 1 ;;
esac
if [[ "$NTOYS" -lt 1 ]]; then
    echo "ERROR: --ntoys must be >= 1" >&2
    exit 1
fi
if [[ "$START_TOY" -lt 1 ]]; then
    echo "ERROR: --start-toy must be >= 1" >&2
    exit 1
fi
if [[ -n "$SINGLE_TOY" ]]; then
    case "$SINGLE_TOY" in
        ''|*[!0-9]*) echo "ERROR: --toy must be a positive integer" >&2; exit 1 ;;
    esac
    if [[ "$SINGLE_TOY" -lt 1 ]]; then
        echo "ERROR: --toy must be >= 1" >&2
        exit 1
    fi
fi

masspoint_is_configured=false
for mp in "${MASSPOINTs_LEE[@]}"; do
    if [[ "$mp" == "$MASSPOINT" ]]; then
        masspoint_is_configured=true
        break
    fi
done
if [[ "$masspoint_is_configured" != true ]]; then
    echo "ERROR: masspoint '$MASSPOINT' is not configured in configs/masspoints.json:LEE" >&2
    echo "Configured LEE mass points: ${MASSPOINTs_LEE[*]}" >&2
    exit 1
fi

reference_masspoint_is_configured=false
for mp in "${MASSPOINTs_LEE[@]}"; do
    if [[ "$mp" == "$REFERENCE_MASSPOINT" ]]; then
        reference_masspoint_is_configured=true
        break
    fi
done
if [[ "$STEP" == "5" && "$reference_masspoint_is_configured" != true ]]; then
    echo "ERROR: reference masspoint '$REFERENCE_MASSPOINT' is not configured in configs/masspoints.json:LEE" >&2
    echo "Configured LEE mass points: ${MASSPOINTs_LEE[*]}" >&2
    exit 1
fi

if [[ "$STEP" != "1" && "$STEP" != "2" && "$STEP" != "3" && "$STEP" != "4" && "$STEP" != "5" ]]; then
    echo "ERROR: LEE step $STEP is recognized but not implemented yet." >&2
    exit 1
fi

run_step1_local() {
    local step_args=(--masspoint "$MASSPOINT")
    if [[ "$DEBUG" == true ]]; then
        step_args+=(--debug)
    fi
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] bash scripts/prepareLEEModel.sh ${step_args[*]}"
        return
    fi
    cd "$SCRIPT_DIR"
    bash scripts/prepareLEEModel.sh "${step_args[@]}"
}

submit_step1_condor() {
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local job_dir="$SCRIPT_DIR/LEE/$MASSPOINT/condor/jobs_dag_step1_${timestamp}"
    local submit_debug_args=""
    if [[ "$DEBUG" == true ]]; then
        submit_debug_args=" --debug"
    fi

    mkdir -p "$job_dir/logs"
    cp "$SCRIPT_DIR/configs/dagman.config" "$job_dir/"

    cat > "$job_dir/jobs.sub" << EOFSUB
JobBatchName = lee_step1_${MASSPOINT}
universe = vanilla
executable = $SCRIPT_DIR/scripts/prepareLEEModel.sh
arguments = --masspoint \$(masspoint)${submit_debug_args}
output = logs/lee_model.out
error  = logs/lee_model.err
log    = dag.log

request_cpus = 1
request_memory = 4GB
request_disk = 4GB

getenv = True
should_transfer_files = NO

queue
EOFSUB

    cat > "$job_dir/dag.dag" << EOFDAG
# LEE Step 1 background model DAG
CONFIG dagman.config

JOB lee_model jobs.sub
VARS lee_model masspoint="${MASSPOINT}"
EOFDAG

    cat > "$job_dir/submit_all.sh" << 'EOF_SUBMIT'
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
condor_submit_dag -config dagman.config dag.dag
EOF_SUBMIT
    chmod +x "$job_dir/submit_all.sh"

    cat > "$job_dir/status_all.sh" << 'EOF_STATUS'
#!/bin/bash
cd "$(dirname "$0")"
echo "DAG status:"
if [[ -f dag.dag.dagman.out ]]; then
    done=$(grep -c "ULOG_JOB_TERMINATED" dag.dag.dagman.out 2>/dev/null || echo 0)
    total=$(grep -c "^JOB " dag.dag 2>/dev/null || echo 0)
    echo "  $done / $total jobs completed"
else
    echo "  not started"
fi
condor_q -dag 2>/dev/null || true
EOF_STATUS
    chmod +x "$job_dir/status_all.sh"

    echo "Generated LEE Step 1 DAG: $job_dir/dag.dag"
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] To submit:"
        echo "  cd $job_dir && ./submit_all.sh"
        echo "[DRY-RUN] To monitor:"
        echo "  cd $job_dir && ./status_all.sh"
        return
    fi

    echo "Submitting LEE Step 1 DAG..."
    (cd "$job_dir" && ./submit_all.sh)
    echo "Monitor with:"
    echo "  cd $job_dir && ./status_all.sh"
    echo "  condor_q -dag"
}

step2_args_for_local() {
    printf -- '--masspoint %q ' "$MASSPOINT"
    if [[ -n "$SINGLE_TOY" ]]; then
        printf -- '--toy %q ' "$SINGLE_TOY"
    else
        printf -- '--start-toy %q --ntoys %q ' "$START_TOY" "$NTOYS"
    fi
    [[ "$FORCE" == true ]] && printf -- '--force '
    [[ "$DEBUG" == true ]] && printf -- '--debug '
}

run_step2_local() {
    local step_args=(--masspoint "$MASSPOINT")
    if [[ -n "$SINGLE_TOY" ]]; then
        step_args+=(--toy "$SINGLE_TOY")
    else
        step_args+=(--start-toy "$START_TOY" --ntoys "$NTOYS")
    fi
    [[ "$FORCE" == true ]] && step_args+=(--force)
    [[ "$DEBUG" == true ]] && step_args+=(--debug)

    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] bash scripts/generateLEEToys.sh $(step2_args_for_local)"
        return
    fi
    cd "$SCRIPT_DIR"
    bash scripts/generateLEEToys.sh "${step_args[@]}"
}

submit_step2_condor() {
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local job_dir="$SCRIPT_DIR/LEE/$MASSPOINT/condor/jobs_dag_step2_${timestamp}"
    local submit_extra_args=""
    [[ "$FORCE" == true ]] && submit_extra_args="${submit_extra_args} --force"
    [[ "$DEBUG" == true ]] && submit_extra_args="${submit_extra_args} --debug"

    local first_toy last_toy
    if [[ -n "$SINGLE_TOY" ]]; then
        first_toy="$SINGLE_TOY"
        last_toy="$SINGLE_TOY"
    else
        first_toy="$START_TOY"
        last_toy=$((START_TOY + NTOYS - 1))
    fi

    mkdir -p "$job_dir/logs"
    cp "$SCRIPT_DIR/configs/dagman.config" "$job_dir/"

    cat > "$job_dir/jobs.sub" << EOFSUB
JobBatchName = lee_step2_${MASSPOINT}
universe = vanilla
executable = $SCRIPT_DIR/scripts/generateLEEToys.sh
arguments = --masspoint \$(masspoint) --toy \$(toy)${submit_extra_args}
output = logs/toy_\$(toy).out
error  = logs/toy_\$(toy).err
log    = dag.log

request_cpus = 1
request_memory = 2GB
request_disk = 2GB

getenv = True
should_transfer_files = NO

queue
EOFSUB

    {
        echo "# LEE Step 2 toy generation DAG"
        echo "CONFIG dagman.config"
        echo ""
        local toy toy_label
        for ((toy = first_toy; toy <= last_toy; toy++)); do
            printf -v toy_label "%04d" "$toy"
            echo "JOB toy_${toy_label} jobs.sub"
            echo "VARS toy_${toy_label} masspoint=\"${MASSPOINT}\" toy=\"${toy}\""
        done
    } > "$job_dir/dag.dag"

    cat > "$job_dir/submit_all.sh" << 'EOF_SUBMIT'
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
condor_submit_dag -config dagman.config dag.dag
EOF_SUBMIT
    chmod +x "$job_dir/submit_all.sh"

    cat > "$job_dir/status_all.sh" << 'EOF_STATUS'
#!/bin/bash
cd "$(dirname "$0")"
echo "DAG status:"
if [[ -f dag.dag.dagman.out ]]; then
    done=$(grep -c "ULOG_JOB_TERMINATED" dag.dag.dagman.out 2>/dev/null || echo 0)
    total=$(grep -c "^JOB " dag.dag 2>/dev/null || echo 0)
    echo "  $done / $total jobs completed"
else
    echo "  not started"
fi
condor_q -dag 2>/dev/null || true
EOF_STATUS
    chmod +x "$job_dir/status_all.sh"

    echo "Generated LEE Step 2 DAG: $job_dir/dag.dag"
    echo "Toy range: $first_toy-$last_toy"
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] To submit:"
        echo "  cd $job_dir && ./submit_all.sh"
        echo "[DRY-RUN] To monitor:"
        echo "  cd $job_dir && ./status_all.sh"
        return
    fi

    echo "Submitting LEE Step 2 DAG..."
    (cd "$job_dir" && ./submit_all.sh)
    echo "Monitor with:"
    echo "  cd $job_dir && ./status_all.sh"
    echo "  condor_q -dag"
}

step3_args_for_one() {
    local toy=$1
    printf -- '--masspoint %q --toy %q ' "$MASSPOINT" "$toy"
    [[ "$FORCE" == true ]] && printf -- '--force '
    [[ "$KEEP_WORKDIR" == true ]] && printf -- '--keep-workdir '
    [[ "$DEBUG" == true ]] && printf -- '--debug '
}

step3_toy_range() {
    if [[ -n "$SINGLE_TOY" ]]; then
        echo "$SINGLE_TOY $SINGLE_TOY"
    else
        echo "$START_TOY $((START_TOY + NTOYS - 1))"
    fi
}

run_step3_local() {
    local first_toy last_toy toy
    read -r first_toy last_toy <<< "$(step3_toy_range)"
    if [[ "$DRY_RUN" == true ]]; then
        for ((toy = first_toy; toy <= last_toy; toy++)); do
            echo "[DRY-RUN] bash scripts/fitLEEToy.sh $(step3_args_for_one "$toy")"
        done
        return
    fi
    cd "$SCRIPT_DIR"
    for ((toy = first_toy; toy <= last_toy; toy++)); do
        # shellcheck disable=SC2206
        local step_args=( $(step3_args_for_one "$toy") )
        bash scripts/fitLEEToy.sh "${step_args[@]}"
    done
}

submit_step3_condor() {
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local job_dir="$SCRIPT_DIR/LEE/$MASSPOINT/condor/jobs_dag_step3_${timestamp}"
    local submit_extra_args=""
    [[ "$FORCE" == true ]] && submit_extra_args="${submit_extra_args} --force"
    [[ "$KEEP_WORKDIR" == true ]] && submit_extra_args="${submit_extra_args} --keep-workdir"
    [[ "$DEBUG" == true ]] && submit_extra_args="${submit_extra_args} --debug"

    local first_toy last_toy
    read -r first_toy last_toy <<< "$(step3_toy_range)"

    mkdir -p "$job_dir/logs"
    cp "$SCRIPT_DIR/configs/dagman.config" "$job_dir/"

    cat > "$job_dir/jobs.sub" << EOFSUB
JobBatchName = lee_step3_${MASSPOINT}
universe = vanilla
executable = $SCRIPT_DIR/scripts/fitLEEToy.sh
arguments = --masspoint \$(masspoint) --toy \$(toy)${submit_extra_args}
output = logs/toy_\$(toy).out
error  = logs/toy_\$(toy).err
log    = dag.log

request_cpus = 1
request_memory = 4GB
request_disk = 6GB

getenv = True
should_transfer_files = NO

queue
EOFSUB

    {
        echo "# LEE Step 3 toy projection and fit DAG"
        echo "CONFIG dagman.config"
        echo ""
        local toy toy_label
        for ((toy = first_toy; toy <= last_toy; toy++)); do
            printf -v toy_label "%04d" "$toy"
            echo "JOB toy_${toy_label} jobs.sub"
            echo "VARS toy_${toy_label} masspoint=\"${MASSPOINT}\" toy=\"${toy}\""
        done
    } > "$job_dir/dag.dag"

    cat > "$job_dir/submit_all.sh" << 'EOF_SUBMIT'
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
condor_submit_dag -config dagman.config dag.dag
EOF_SUBMIT
    chmod +x "$job_dir/submit_all.sh"

    cat > "$job_dir/status_all.sh" << 'EOF_STATUS'
#!/bin/bash
cd "$(dirname "$0")"
echo "DAG status:"
if [[ -f dag.dag.dagman.out ]]; then
    done=$(grep -c "ULOG_JOB_TERMINATED" dag.dag.dagman.out 2>/dev/null || echo 0)
    total=$(grep -c "^JOB " dag.dag 2>/dev/null || echo 0)
    echo "  $done / $total jobs completed"
else
    echo "  not started"
fi
condor_q -dag 2>/dev/null || true
EOF_STATUS
    chmod +x "$job_dir/status_all.sh"

    echo "Generated LEE Step 3 DAG: $job_dir/dag.dag"
    echo "Toy range: $first_toy-$last_toy"
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] To submit:"
        echo "  cd $job_dir && ./submit_all.sh"
        echo "[DRY-RUN] To monitor:"
        echo "  cd $job_dir && ./status_all.sh"
        return
    fi

    echo "Submitting LEE Step 3 DAG..."
    (cd "$job_dir" && ./submit_all.sh)
    echo "Monitor with:"
    echo "  cd $job_dir && ./status_all.sh"
    echo "  condor_q -dag"
}

step4_args() {
    printf -- '--masspoint %q --start-toy %q --ntoys %q ' "$MASSPOINT" "$START_TOY" "$NTOYS"
    [[ "$DEBUG" == true ]] && printf -- '--debug '
}

run_step4_local() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] bash scripts/collectLEE.sh $(step4_args)"
        return
    fi
    cd "$SCRIPT_DIR"
    # shellcheck disable=SC2206
    local step_args=( $(step4_args) )
    bash scripts/collectLEE.sh "${step_args[@]}"
}

submit_step4_condor() {
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local job_dir="$SCRIPT_DIR/LEE/$MASSPOINT/condor/jobs_dag_step4_${timestamp}"
    local submit_extra_args=""
    [[ "$DEBUG" == true ]] && submit_extra_args="${submit_extra_args} --debug"

    mkdir -p "$job_dir/logs"
    cp "$SCRIPT_DIR/configs/dagman.config" "$job_dir/"

    cat > "$job_dir/jobs.sub" << EOFSUB
JobBatchName = lee_step4_${MASSPOINT}
universe = vanilla
executable = $SCRIPT_DIR/scripts/collectLEE.sh
arguments = --masspoint \$(masspoint) --start-toy \$(start_toy) --ntoys \$(ntoys)${submit_extra_args}
output = logs/collect.out
error  = logs/collect.err
log    = dag.log

request_cpus = 1
request_memory = 2GB
request_disk = 2GB

getenv = True
should_transfer_files = NO

queue
EOFSUB

    cat > "$job_dir/dag.dag" << EOFDAG
# LEE Step 4 global p-value collection DAG
CONFIG dagman.config

JOB collect jobs.sub
VARS collect masspoint="${MASSPOINT}" start_toy="${START_TOY}" ntoys="${NTOYS}"
EOFDAG

    cat > "$job_dir/submit_all.sh" << 'EOF_SUBMIT'
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
condor_submit_dag -config dagman.config dag.dag
EOF_SUBMIT
    chmod +x "$job_dir/submit_all.sh"

    cat > "$job_dir/status_all.sh" << 'EOF_STATUS'
#!/bin/bash
cd "$(dirname "$0")"
echo "DAG status:"
if [[ -f dag.dag.dagman.out ]]; then
    done=$(grep -c "ULOG_JOB_TERMINATED" dag.dag.dagman.out 2>/dev/null || echo 0)
    total=$(grep -c "^JOB " dag.dag 2>/dev/null || echo 0)
    echo "  $done / $total jobs completed"
else
    echo "  not started"
fi
condor_q -dag 2>/dev/null || true
EOF_STATUS
    chmod +x "$job_dir/status_all.sh"

    echo "Generated LEE Step 4 DAG: $job_dir/dag.dag"
    echo "Toy range: $START_TOY-$((START_TOY + NTOYS - 1))"
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] To submit:"
        echo "  cd $job_dir && ./submit_all.sh"
        echo "[DRY-RUN] To monitor:"
        echo "  cd $job_dir && ./status_all.sh"
        return
    fi

    echo "Submitting LEE Step 4 DAG..."
    (cd "$job_dir" && ./submit_all.sh)
    echo "Monitor with:"
    echo "  cd $job_dir && ./status_all.sh"
    echo "  condor_q -dag"
}

step5_args() {
    printf -- '--masspoint %q --reference-masspoint %q --start-toy %q --ntoys %q ' \
        "$MASSPOINT" "$REFERENCE_MASSPOINT" "$START_TOY" "$NTOYS"
    [[ "$DEBUG" == true ]] && printf -- '--debug '
}

run_step5_local() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] bash scripts/validateLEE.sh $(step5_args)"
        return
    fi
    cd "$SCRIPT_DIR"
    # shellcheck disable=SC2206
    local step_args=( $(step5_args) )
    bash scripts/validateLEE.sh "${step_args[@]}"
}

submit_step5_condor() {
    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local job_dir="$SCRIPT_DIR/LEE/$MASSPOINT/condor/jobs_dag_step5_${timestamp}"
    local submit_extra_args=""
    [[ "$DEBUG" == true ]] && submit_extra_args="${submit_extra_args} --debug"

    mkdir -p "$job_dir/logs"
    cp "$SCRIPT_DIR/configs/dagman.config" "$job_dir/"

    cat > "$job_dir/jobs.sub" << EOFSUB
JobBatchName = lee_step5_${MASSPOINT}
universe = vanilla
executable = $SCRIPT_DIR/scripts/validateLEE.sh
arguments = --masspoint \$(masspoint) --reference-masspoint \$(reference_masspoint) --start-toy \$(start_toy) --ntoys \$(ntoys)${submit_extra_args}
output = logs/validate.out
error  = logs/validate.err
log    = dag.log

request_cpus = 1
request_memory = 4GB
request_disk = 4GB

getenv = True
should_transfer_files = NO

queue
EOFSUB

    cat > "$job_dir/dag.dag" << EOFDAG
# LEE Step 5 validation DAG
CONFIG dagman.config

JOB validate jobs.sub
VARS validate masspoint="${MASSPOINT}" reference_masspoint="${REFERENCE_MASSPOINT}" start_toy="${START_TOY}" ntoys="${NTOYS}"
EOFDAG

    cat > "$job_dir/submit_all.sh" << 'EOF_SUBMIT'
#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
condor_submit_dag -config dagman.config dag.dag
EOF_SUBMIT
    chmod +x "$job_dir/submit_all.sh"

    cat > "$job_dir/status_all.sh" << 'EOF_STATUS'
#!/bin/bash
cd "$(dirname "$0")"
echo "DAG status:"
if [[ -f dag.dag.dagman.out ]]; then
    done=$(grep -c "ULOG_JOB_TERMINATED" dag.dag.dagman.out 2>/dev/null || echo 0)
    total=$(grep -c "^JOB " dag.dag 2>/dev/null || echo 0)
    echo "  $done / $total jobs completed"
else
    echo "  not started"
fi
condor_q -dag 2>/dev/null || true
EOF_STATUS
    chmod +x "$job_dir/status_all.sh"

    echo "Generated LEE Step 5 DAG: $job_dir/dag.dag"
    echo "Toy range: $START_TOY-$((START_TOY + NTOYS - 1))"
    echo "Reference mass point: $REFERENCE_MASSPOINT"
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] To submit:"
        echo "  cd $job_dir && ./submit_all.sh"
        echo "[DRY-RUN] To monitor:"
        echo "  cd $job_dir && ./status_all.sh"
        return
    fi

    echo "Submitting LEE Step 5 DAG..."
    (cd "$job_dir" && ./submit_all.sh)
    echo "Monitor with:"
    echo "  cd $job_dir && ./status_all.sh"
    echo "  condor_q -dag"
}

echo "============================================================"
echo "LEE workflow"
echo "Step: $STEP"
echo "Mass point: $MASSPOINT"
echo "Mode: $([[ "$RUN_LOCAL" == true ]] && echo local || echo condor)"
echo "Dry run: $DRY_RUN"
if [[ "$STEP" == "1" ]]; then
    echo "Output: LEE/$MASSPOINT/model"
elif [[ "$STEP" == "2" ]]; then
    if [[ -n "$SINGLE_TOY" ]]; then
        echo "Toy: $SINGLE_TOY"
    else
        echo "Toy range: $START_TOY-$((START_TOY + NTOYS - 1))"
    fi
    echo "Output: LEE/$MASSPOINT/toys"
elif [[ "$STEP" == "3" ]]; then
    if [[ -n "$SINGLE_TOY" ]]; then
        echo "Toy: $SINGLE_TOY"
    else
        echo "Toy range: $START_TOY-$((START_TOY + NTOYS - 1))"
    fi
    echo "Output: LEE/$MASSPOINT/fits"
else
    echo "Toy range: $START_TOY-$((START_TOY + NTOYS - 1))"
    if [[ "$STEP" == "4" ]]; then
        echo "Output: results/lee"
    else
        echo "Reference mass point: $REFERENCE_MASSPOINT"
        echo "Output: results/lee/validation"
    fi
fi
echo "============================================================"

if [[ "$STEP" == "1" && "$RUN_LOCAL" == true ]]; then
    run_step1_local
elif [[ "$STEP" == "1" ]]; then
    submit_step1_condor
elif [[ "$STEP" == "2" && "$RUN_LOCAL" == true ]]; then
    run_step2_local
elif [[ "$STEP" == "2" ]]; then
    submit_step2_condor
elif [[ "$STEP" == "3" && "$RUN_LOCAL" == true ]]; then
    run_step3_local
elif [[ "$STEP" == "3" ]]; then
    submit_step3_condor
elif [[ "$STEP" == "4" && "$RUN_LOCAL" == true ]]; then
    run_step4_local
elif [[ "$STEP" == "4" ]]; then
    submit_step4_condor
elif [[ "$RUN_LOCAL" == true ]]; then
    run_step5_local
else
    submit_step5_condor
fi
