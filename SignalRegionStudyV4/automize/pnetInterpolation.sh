#!/bin/bash
# One-liner driver for the ParticleNet mA-interpolation study chain:
# working points -> shape/yield closures -> eps model -> uncertainty
# export -> template closure -> summary, over the five trained mHc.
#
# Unlike the Baseline chain (one DAG per mHc, three barrier passes), this
# is ONE DAG: the only cross-study barrier -- exportPnetUncertainties
# reads every study's shards -- is expressed as a node with all ten
# shapes/yields parents, so a full rebuild is a single submission.
#
#   thresholds_{mhc}  ->  shapes_{mhc} + yields_{mhc}
#   yields_{mhc}      ->  eps_model_{mhc}
#   ALL shapes+yields ->  export  (configs/pnet_interpolation_uncertainties.json)
#   export + eps_model_{mhc} -> closure_{mhc} -> summarize
#
# Usage:
#   ./pnetInterpolation.sh --all [--dry-run]
#   ./pnetInterpolation.sh --mhc 115 --mhc 145
#   ./pnetInterpolation.sh --all --start-from closure
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/dag_lib.sh"

MHCS=()
ALL_MHC=false
START_FROM="thresholds"
DRY_RUN=false
LOCAL_RUN=false
STOP_AFTER=""
STOP_AFTER_SET=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mhc)
            MHCS+=("${2#MHc}")
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
        --stop-after)
            STOP_AFTER="$2"
            STOP_AFTER_SET=true
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
            echo "  --mhc N              Run for mHc=N (repeatable; 115 or MHc115)"
            echo "  --all                Run for every trained mHc (configs/masspoints.json)"
            echo "  --start-from STEP    thresholds, shapes, yields, eps-model, export,"
            echo "                       closure, summarize  [default: thresholds]"
            echo "  --stop-after STEP    same vocabulary; nodes above that level are"
            echo "                       emitted DONE (dependency graph kept)"
            echo "  --local              Serial execution without condor"
            echo "  --dry-run            Generate the DAG without submitting"
            echo ""
            echo "NOTE: export and closure need ALL five studies' shards; a partial"
            echo "      --mhc run marks them DONE unless every study is included."
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

# Trained mHc list from configs/masspoints.json, via the config module --
# fails loudly if the environment is not set up.
_mhc_all=$(WORKDIR="$(dirname "$SCRIPT_DIR")" python3 -c "
import sys
sys.path.insert(0, '$SCRIPT_DIR/python')
import pnet_interp_config as pic
print(' '.join(m.replace('MHc', '') for m in pic.pn_mhc_list()))
") || { echo "ERROR: cannot resolve the trained mHc list (source setup.sh first?)"; exit 1; }
if [[ "$ALL_MHC" == "true" ]]; then
    read -ra MHCS <<< "$_mhc_all"
fi

_valid_step() {
    case "$1" in
        thresholds|shapes|yields|eps-model|export|closure|summarize) return 0 ;;
        *) return 1 ;;
    esac
}
_valid_step "$START_FROM" || { echo "ERROR: Invalid --start-from '$START_FROM'"; exit 1; }
if [[ "$STOP_AFTER_SET" == "true" ]]; then
    _valid_step "$STOP_AFTER" || { echo "ERROR: Invalid --stop-after '$STOP_AFTER'"; exit 1; }
fi

step_to_level() {
    case "$1" in
        thresholds) echo 0 ;;
        shapes|yields) echo 1 ;;
        eps_model) echo 2 ;;
        export) echo 3 ;;
        closure) echo 4 ;;
        summarize) echo 5 ;;
        *) echo 0 ;;
    esac
}

start_from_level() {
    case "$1" in
        thresholds) echo 0 ;;
        shapes|yields) echo 1 ;;
        eps-model) echo 2 ;;
        export) echo 3 ;;
        closure) echo 4 ;;
        summarize) echo 5 ;;
    esac
}

# The export and summarize barriers read EVERY study; a partial --mhc run
# would build them from stale/missing shards. Mark them DONE unless the
# run covers the full trained list.
FULL_COVERAGE=true
for m in $_mhc_all; do
    _found=false
    for sel in "${MHCS[@]}"; do
        [[ "$sel" == "$m" ]] && _found=true
    done
    [[ "$_found" == "false" ]] && FULL_COVERAGE=false
done
if [[ "$FULL_COVERAGE" == "false" ]]; then
    echo "NOTE: partial mHc selection -- export/closure/summarize nodes will be"
    echo "      marked DONE (they need all five studies; rerun with --all)"
fi

MEMORY_thresholds=6144
MEMORY_shapes=6144
MEMORY_yields=4096
MEMORY_eps_model=2048
MEMORY_export=2048
MEMORY_closure=4096
MEMORY_summarize=2048

generate_dag_file() {
    local dag_file=$1
    local start_level stop_level
    start_level=$(start_from_level "$START_FROM")
    stop_level=99
    [[ "$STOP_AFTER_SET" == "true" ]] && stop_level=$(start_from_level "$STOP_AFTER")

    # A node outside [start_level, stop_level] is emitted but marked DONE,
    # so DAGMan keeps the full dependency graph and simply does not run it
    # (same convention as automize/interpolation.sh).
    job_done_suffix() {
        local level
        level=$(step_to_level "$1")
        if [[ $level -lt $start_level || $level -gt $stop_level ]]; then
            echo " DONE"
            return
        fi
        # Cross-study barrier nodes need every study in this run.
        case "$1" in
            export|closure|summarize)
                [[ "$FULL_COVERAGE" == "false" ]] && echo " DONE" && return
                ;;
        esac
        echo ""
    }

    cat > "$dag_file" << EOF
# DAG for the ParticleNet interpolation study chain
# (start-from: $START_FROM, stop-after: ${STOP_AFTER:-none}, mHc: ${MHCS[*]})
CONFIG dagman.config

EOF

    local mhc done_sfx
    for mhc in "${MHCS[@]}"; do
        done_sfx=$(job_done_suffix thresholds)
        echo "JOB thresholds_MHc${mhc} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS thresholds_MHc${mhc} step=\"thresholds\" mhc=\"MHc${mhc}\" extra_args=\"\" job_request_memory=\"$MEMORY_thresholds\"" >> "$dag_file"
        done_sfx=$(job_done_suffix shapes)
        echo "JOB shapes_MHc${mhc} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS shapes_MHc${mhc} step=\"shapes\" mhc=\"MHc${mhc}\" extra_args=\"\" job_request_memory=\"$MEMORY_shapes\"" >> "$dag_file"
        done_sfx=$(job_done_suffix yields)
        echo "JOB yields_MHc${mhc} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS yields_MHc${mhc} step=\"yields\" mhc=\"MHc${mhc}\" extra_args=\"\" job_request_memory=\"$MEMORY_yields\"" >> "$dag_file"
        done_sfx=$(job_done_suffix eps_model)
        echo "JOB eps_model_MHc${mhc} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS eps_model_MHc${mhc} step=\"eps_model\" mhc=\"MHc${mhc}\" extra_args=\"\" job_request_memory=\"$MEMORY_eps_model\"" >> "$dag_file"
        done_sfx=$(job_done_suffix closure)
        echo "JOB closure_MHc${mhc} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS closure_MHc${mhc} step=\"closure\" mhc=\"MHc${mhc}\" extra_args=\"\" job_request_memory=\"$MEMORY_closure\"" >> "$dag_file"
    done

    done_sfx=$(job_done_suffix export)
    echo "JOB export jobs.sub${done_sfx}" >> "$dag_file"
    echo "VARS export step=\"export\" mhc=\"-\" extra_args=\"\" job_request_memory=\"$MEMORY_export\"" >> "$dag_file"
    done_sfx=$(job_done_suffix summarize)
    echo "JOB summarize jobs.sub${done_sfx}" >> "$dag_file"
    echo "VARS summarize step=\"summarize\" mhc=\"-\" extra_args=\"\" job_request_memory=\"$MEMORY_summarize\"" >> "$dag_file"

    echo "" >> "$dag_file"
    for mhc in "${MHCS[@]}"; do
        echo "PARENT thresholds_MHc${mhc} CHILD shapes_MHc${mhc} yields_MHc${mhc}" >> "$dag_file"
        echo "PARENT yields_MHc${mhc} CHILD eps_model_MHc${mhc}" >> "$dag_file"
        echo "PARENT shapes_MHc${mhc} yields_MHc${mhc} CHILD export" >> "$dag_file"
        echo "PARENT export eps_model_MHc${mhc} CHILD closure_MHc${mhc}" >> "$dag_file"
        echo "PARENT closure_MHc${mhc} CHILD summarize" >> "$dag_file"
    done

    echo "" >> "$dag_file"
    for mhc in "${MHCS[@]}"; do
        for node in thresholds shapes yields eps_model closure; do
            echo "RETRY ${node}_MHc${mhc} 1" >> "$dag_file"
        done
    done
    echo "RETRY export 1" >> "$dag_file"
    echo "RETRY summarize 1" >> "$dag_file"
}

run_local() {
    local start_level stop_level level
    start_level=$(start_from_level "$START_FROM")
    stop_level=99
    [[ "$STOP_AFTER_SET" == "true" ]] && stop_level=$(start_from_level "$STOP_AFTER")
    _should_run() {
        level=$(step_to_level "$1")
        [[ $level -ge $start_level && $level -le $stop_level ]]
    }
    local wrapper="$SCRIPT_DIR/scripts/pnet_interpolation_wrapper.sh"
    local mhc
    for mhc in "${MHCS[@]}"; do
        _should_run thresholds && "$wrapper" thresholds "MHc${mhc}"
    done
    for mhc in "${MHCS[@]}"; do
        _should_run shapes && "$wrapper" shapes "MHc${mhc}"
        _should_run yields && "$wrapper" yields "MHc${mhc}"
    done
    for mhc in "${MHCS[@]}"; do
        _should_run eps_model && "$wrapper" eps_model "MHc${mhc}"
    done
    if [[ "$FULL_COVERAGE" == "true" ]]; then
        _should_run export && "$wrapper" export -
        for mhc in "${MHCS[@]}"; do
            _should_run closure && "$wrapper" closure "MHc${mhc}"
        done
        _should_run summarize && "$wrapper" summarize -
    fi
}

echo "ParticleNet interpolation chain"
echo "mHc: ${MHCS[*]}"
echo "start-from: $START_FROM   stop-after: ${STOP_AFTER:-none}"
echo "Local: $LOCAL_RUN   Dry-run: $DRY_RUN"

if [[ "$LOCAL_RUN" == "true" ]]; then
    run_local
    exit 0
fi

JOB_DIR=$(dag_new_jobdir "pnet_interp")
STUDY_DIR=$(dag_new_masspoint_dir "$JOB_DIR" "study")

cat > "$STUDY_DIR/jobs.sub" << EOF
JobBatchName = pnet_interp
universe = vanilla
executable = $SCRIPT_DIR/scripts/pnet_interpolation_wrapper.sh
arguments = "'\$(step)' '\$(mhc)' '\$(extra_args)'"
output = logs/\$(step).\$(mhc).out
error = logs/\$(step).\$(mhc).err
log = dag.log
request_cpus = 1
RequestMemory = \$(job_request_memory)
request_disk = 2GB
getenv = False
should_transfer_files = NO
queue
EOF

generate_dag_file "$STUDY_DIR/dag.dag"
echo "Generated DAG: $STUDY_DIR/dag.dag"

dag_write_submit_all "$JOB_DIR"
dag_write_status_all "$JOB_DIR"
dag_submit_or_dryrun "$JOB_DIR" "$DRY_RUN"
