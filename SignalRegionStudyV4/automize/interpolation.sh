#!/bin/bash
# One-liner driver for the mA-interpolation chain: per-mHc DAG covering
# per-point DCB fits (floating -> frozen-n passes) -> shape parametrizations
# -> shape closure / window yields (parallel) -> yield model -> yield
# closure -> shape-systematic deltas -> delta model -> uncertainty export.
#
# Usage:
#   ./interpolation.sh --mhc 160 [--dry-run]
#   ./interpolation.sh --all [--start-from STEP] [--local]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dag_lib.sh"

CONDOR_DIR="$SCRIPT_DIR/condor"
export PATH="${PWD}/python:${PATH}"

MHCS=()
ALL_MHC=false
START_FROM="fit-floating"
DRY_RUN=false
LOCAL_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mhc)
            MHCS+=("$2")
            shift 2
            ;;
        --all)
            ALL_MHC=true
            shift
            ;;
        --start-from)
            START_FROM="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --local)
            LOCAL_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 --mhc N [--mhc N ...] | --all [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mhc N              Run the chain for mHc=N (repeatable)"
            echo "  --all                Run for every mHc in configs/interpolation.json"
            echo "  --start-from STEP    fit-floating, fit-frozen, polynomials, closure,"
            echo "                       yields, yield-model, yield-closure, deltas, export"
            echo "                       [default: fit-floating]"
            echo "  --local              Serial execution without condor (insurance for"
            echo "                       local-server condor availability)"
            echo "  --dry-run            Generate DAGs without submitting (condor mode only)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

if [[ "$ALL_MHC" == "false" && ${#MHCS[@]} -eq 0 ]]; then
    echo "ERROR: pass --mhc N (one or more) or --all"
    exit 1
fi

if [[ "$ALL_MHC" == "true" ]]; then
    _interp_json="$SCRIPT_DIR/configs/interpolation.json"
    _mhc_list=$(python3 -c "
import json
d = json.load(open('$_interp_json'))
print(' '.join(sorted(d['fit_points'], key=int)))
")
    read -ra MHCS <<< "$_mhc_list"
fi

case "$START_FROM" in
    fit-floating|fit-frozen|polynomials|closure|yields|yield-model|yield-closure|deltas|export) ;;
    *) echo "ERROR: Invalid --start-from '$START_FROM'"; exit 1 ;;
esac

# Full baseline mA grid of one mHc study (interpolation_config.study()["all"]).
# Fails loudly: a silent empty list would generate a DAG with no per-mass-point
# nodes, which looks like a successful (but useless) submission.
study_masspoints_for() {
    local mhc=$1
    local out err_file status
    # stderr goes to a file, never into $out: a stray python warning would
    # otherwise be parsed as an extra "mass point".
    err_file=$(mktemp)
    out=$(WORKDIR="$(dirname "$SCRIPT_DIR")" python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/python')
import interpolation_config as ic
grid = ic.study($mhc)['all']
print(' '.join(ic.masspoint_name(m, $mhc) for m in grid))
" 2>"$err_file")
    status=$?
    if [[ $status -ne 0 ]]; then
        echo "ERROR: cannot resolve the mass-point grid for mHc=$mhc." >&2
        echo "       Did you 'source setup.sh' first? (numpy/ROOT come from CMSSW)" >&2
        sed 's/^/       /' "$err_file" >&2
        rm -f "$err_file"
        exit 1
    fi
    rm -f "$err_file"
    if [[ -z "${out// }" ]]; then
        echo "ERROR: empty mass-point grid for mHc=$mhc — check configs/interpolation.json" >&2
        exit 1
    fi
    echo "$out"
}

step_to_level() {
    case "$1" in
        fit_float|merge_float) echo 0 ;;
        fit_frozen|merge_frozen) echo 1 ;;
        polynomials) echo 2 ;;
        closure|merge_closure|yields|merge_yields) echo 3 ;;
        yield_model) echo 4 ;;
        yield_closure|merge_yield_closure) echo 5 ;;
        deltas|merge_deltas|delta_model) echo 6 ;;
        export_uncertainties) echo 7 ;;
        *) echo 0 ;;
    esac
}

start_from_level() {
    case "$1" in
        fit-floating) echo 0 ;;
        fit-frozen) echo 1 ;;
        polynomials) echo 2 ;;
        closure|yields) echo 3 ;;
        yield-model) echo 4 ;;
        yield-closure) echo 5 ;;
        deltas) echo 6 ;;
        export) echo 7 ;;
    esac
}

generate_dag_file() {
    local mhc=$1
    local dag_file=$2
    local start_level
    start_level=$(start_from_level "$START_FROM")
    local masspoints mp_list
    # Separate assignment so $? is the command substitution's status: the
    # helper exits non-zero in its subshell and we must abort here.
    mp_list=$(study_masspoints_for "$mhc") || exit 1
    read -r -a masspoints <<< "$mp_list"

    job_done_suffix() {
        local level
        level=$(step_to_level "$1")
        [[ $level -lt $start_level ]] && echo " DONE" || echo ""
    }

    cat > "$dag_file" << EOF
# DAG for MHc${mhc} interpolation chain (start-from: $START_FROM)
CONFIG dagman.config

EOF

    local mp
    local done_sfx

    done_sfx=$(job_done_suffix fit_float)
    for mp in "${masspoints[@]}"; do
        echo "JOB fit_float_${mp} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS fit_float_${mp} step=\"fit_float\" mhc=\"${mhc}\" masspoint=\"${mp}\" extra_args=\"\" job_request_memory=\"2048\"" >> "$dag_file"
    done
    done_sfx=$(job_done_suffix merge_float)
    echo "JOB merge_float jobs.sub${done_sfx}" >> "$dag_file"
    echo "VARS merge_float step=\"merge_float\" mhc=\"${mhc}\" masspoint=\"-\" extra_args=\"\" job_request_memory=\"2048\"" >> "$dag_file"

    done_sfx=$(job_done_suffix fit_frozen)
    for mp in "${masspoints[@]}"; do
        echo "JOB fit_frozen_${mp} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS fit_frozen_${mp} step=\"fit_frozen\" mhc=\"${mhc}\" masspoint=\"${mp}\" extra_args=\"\" job_request_memory=\"2048\"" >> "$dag_file"
    done
    done_sfx=$(job_done_suffix merge_frozen)
    echo "JOB merge_frozen jobs.sub${done_sfx}" >> "$dag_file"
    echo "VARS merge_frozen step=\"merge_frozen\" mhc=\"${mhc}\" masspoint=\"-\" extra_args=\"\" job_request_memory=\"2048\"" >> "$dag_file"

    done_sfx=$(job_done_suffix polynomials)
    echo "JOB polynomials jobs.sub${done_sfx}" >> "$dag_file"
    echo "VARS polynomials step=\"polynomials\" mhc=\"${mhc}\" masspoint=\"-\" extra_args=\"\" job_request_memory=\"2048\"" >> "$dag_file"

    done_sfx=$(job_done_suffix closure)
    for mp in "${masspoints[@]}"; do
        echo "JOB closure_${mp} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS closure_${mp} step=\"closure\" mhc=\"${mhc}\" masspoint=\"${mp}\" extra_args=\"\" job_request_memory=\"2048\"" >> "$dag_file"
    done
    done_sfx=$(job_done_suffix merge_closure)
    echo "JOB merge_closure jobs.sub${done_sfx}" >> "$dag_file"
    echo "VARS merge_closure step=\"merge_closure\" mhc=\"${mhc}\" masspoint=\"-\" extra_args=\"\" job_request_memory=\"2048\"" >> "$dag_file"

    done_sfx=$(job_done_suffix yields)
    for mp in "${masspoints[@]}"; do
        echo "JOB yields_${mp} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS yields_${mp} step=\"yields\" mhc=\"${mhc}\" masspoint=\"${mp}\" extra_args=\"\" job_request_memory=\"2048\"" >> "$dag_file"
    done
    done_sfx=$(job_done_suffix merge_yields)
    echo "JOB merge_yields jobs.sub${done_sfx}" >> "$dag_file"
    echo "VARS merge_yields step=\"merge_yields\" mhc=\"${mhc}\" masspoint=\"-\" extra_args=\"\" job_request_memory=\"2048\"" >> "$dag_file"

    done_sfx=$(job_done_suffix yield_model)
    echo "JOB yield_model jobs.sub${done_sfx}" >> "$dag_file"
    echo "VARS yield_model step=\"yield_model\" mhc=\"${mhc}\" masspoint=\"-\" extra_args=\"\" job_request_memory=\"2048\"" >> "$dag_file"

    done_sfx=$(job_done_suffix yield_closure)
    for mp in "${masspoints[@]}"; do
        echo "JOB yield_closure_${mp} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS yield_closure_${mp} step=\"yield_closure\" mhc=\"${mhc}\" masspoint=\"${mp}\" extra_args=\"\" job_request_memory=\"2048\"" >> "$dag_file"
    done
    done_sfx=$(job_done_suffix merge_yield_closure)
    echo "JOB merge_yield_closure jobs.sub${done_sfx}" >> "$dag_file"
    echo "VARS merge_yield_closure step=\"merge_yield_closure\" mhc=\"${mhc}\" masspoint=\"-\" extra_args=\"\" job_request_memory=\"2048\"" >> "$dag_file"

    done_sfx=$(job_done_suffix deltas)
    for mp in "${masspoints[@]}"; do
        echo "JOB deltas_${mp} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS deltas_${mp} step=\"deltas\" mhc=\"${mhc}\" masspoint=\"${mp}\" extra_args=\"\" job_request_memory=\"3072\"" >> "$dag_file"
    done
    done_sfx=$(job_done_suffix merge_deltas)
    echo "JOB merge_deltas jobs.sub${done_sfx}" >> "$dag_file"
    echo "VARS merge_deltas step=\"merge_deltas\" mhc=\"${mhc}\" masspoint=\"-\" extra_args=\"\" job_request_memory=\"2048\"" >> "$dag_file"

    done_sfx=$(job_done_suffix delta_model)
    echo "JOB delta_model jobs.sub${done_sfx}" >> "$dag_file"
    echo "VARS delta_model step=\"delta_model\" mhc=\"${mhc}\" masspoint=\"-\" extra_args=\"\" job_request_memory=\"2048\"" >> "$dag_file"

    done_sfx=$(job_done_suffix export_uncertainties)
    echo "JOB export_uncertainties jobs.sub${done_sfx}" >> "$dag_file"
    echo "VARS export_uncertainties step=\"export_uncertainties\" mhc=\"${mhc}\" masspoint=\"-\" extra_args=\"\" job_request_memory=\"2048\"" >> "$dag_file"

    echo "" >> "$dag_file"
    echo "# Dependencies" >> "$dag_file"
    for mp in "${masspoints[@]}"; do
        echo "PARENT fit_float_${mp} CHILD merge_float" >> "$dag_file"
    done
    echo "PARENT merge_float CHILD ${masspoints[*]/#/fit_frozen_}" >> "$dag_file"
    for mp in "${masspoints[@]}"; do
        echo "PARENT fit_frozen_${mp} CHILD merge_frozen" >> "$dag_file"
    done
    echo "PARENT merge_frozen CHILD polynomials" >> "$dag_file"
    echo "PARENT polynomials CHILD ${masspoints[*]/#/closure_}" >> "$dag_file"
    for mp in "${masspoints[@]}"; do
        echo "PARENT closure_${mp} CHILD merge_closure" >> "$dag_file"
    done
    echo "PARENT polynomials CHILD ${masspoints[*]/#/yields_}" >> "$dag_file"
    for mp in "${masspoints[@]}"; do
        echo "PARENT yields_${mp} CHILD merge_yields" >> "$dag_file"
    done
    echo "PARENT merge_yields CHILD yield_model" >> "$dag_file"
    echo "PARENT yield_model CHILD ${masspoints[*]/#/yield_closure_}" >> "$dag_file"
    for mp in "${masspoints[@]}"; do
        echo "PARENT yield_closure_${mp} CHILD merge_yield_closure" >> "$dag_file"
    done
    echo "PARENT polynomials CHILD ${masspoints[*]/#/deltas_}" >> "$dag_file"
    for mp in "${masspoints[@]}"; do
        echo "PARENT deltas_${mp} CHILD merge_deltas" >> "$dag_file"
    done
    echo "PARENT merge_deltas CHILD delta_model" >> "$dag_file"
    echo "PARENT merge_closure CHILD export_uncertainties" >> "$dag_file"
    echo "PARENT merge_yield_closure CHILD export_uncertainties" >> "$dag_file"
}

submit_condor_dags() {
    local job_dir
    job_dir=$(dag_new_jobdir "interp")

    local mhc
    for mhc in "${MHCS[@]}"; do
        local mp_dir
        mp_dir=$(dag_new_masspoint_dir "$job_dir" "MHc${mhc}")
        cat > "$mp_dir/jobs.sub" << EOF
JobBatchName = MHc${mhc}
universe = vanilla
executable = $SCRIPT_DIR/scripts/interpolation_wrapper.sh
arguments = "'\$(step)' '\$(mhc)' '\$(masspoint)' '\$(extra_args)'"
output = logs/\$(step).out
error = logs/\$(step).err
log = dag.log
request_cpus = 1
RequestMemory = \$(job_request_memory)
request_disk = 2GB
getenv = True
should_transfer_files = NO
queue
EOF
        generate_dag_file "$mhc" "$mp_dir/dag.dag"
        echo "Generated DAG: $mp_dir/dag.dag"
    done

    dag_write_submit_all "$job_dir"
    dag_write_status_all "$job_dir"
    dag_submit_or_dryrun "$job_dir" "$DRY_RUN"
}

run_local() {
    local mhc=$1
    local masspoints mp_list
    mp_list=$(study_masspoints_for "$mhc") || exit 1
    read -r -a masspoints <<< "$mp_list"
    local wrapper="$SCRIPT_DIR/scripts/interpolation_wrapper.sh"
    local mp

    echo "=== MHc${mhc}: local serial run (${#masspoints[@]} mass points) ==="
    for mp in "${masspoints[@]}"; do "$wrapper" fit_float "$mhc" "$mp"; done
    "$wrapper" merge_float "$mhc" -
    for mp in "${masspoints[@]}"; do "$wrapper" fit_frozen "$mhc" "$mp"; done
    "$wrapper" merge_frozen "$mhc" -
    "$wrapper" polynomials "$mhc" -
    for mp in "${masspoints[@]}"; do "$wrapper" closure "$mhc" "$mp"; done
    "$wrapper" merge_closure "$mhc" -
    for mp in "${masspoints[@]}"; do "$wrapper" yields "$mhc" "$mp"; done
    "$wrapper" merge_yields "$mhc" -
    "$wrapper" yield_model "$mhc" -
    for mp in "${masspoints[@]}"; do "$wrapper" yield_closure "$mhc" "$mp"; done
    "$wrapper" merge_yield_closure "$mhc" -
    for mp in "${masspoints[@]}"; do "$wrapper" deltas "$mhc" "$mp"; done
    "$wrapper" merge_deltas "$mhc" -
    "$wrapper" delta_model "$mhc" -
    "$wrapper" export_uncertainties "$mhc" -
}

echo "============================================================"
echo "SignalRegionStudyV4 mA-interpolation chain"
echo "mHc values: ${MHCS[*]}"
echo "Start from: $START_FROM"
echo "Local: $LOCAL_RUN"
echo "Dry run: $DRY_RUN"
echo "============================================================"

if [[ "$LOCAL_RUN" == "true" ]]; then
    for mhc in "${MHCS[@]}"; do
        run_local "$mhc"
    done
else
    submit_condor_dags
fi
