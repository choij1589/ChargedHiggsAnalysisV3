#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_masspoints.sh"

CONDOR_DIR="$SCRIPT_DIR/condor"
export PATH="${PWD}/python:${PATH}"

MODE="all"
SINGLE_ERA=""
SINGLE_MASSPOINT=""
MASSPOINT_SET=""
METHOD="Baseline"
BINNING="extended"
NUISANCE="fallback_lnn"
EXTRA_ARGS=""
PULL_FIT="b"
DO_PRINT_DATACARD=true
DO_RUN_ASYMPTOTIC=true
DO_FITDIAG=false
DO_PLOT_SCORE_SET=false
DO_PLOT_SCORE=false
DRY_RUN=false
START_FROM="template"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="${2,,}"
            shift 2
            ;;
        --era)
            SINGLE_ERA="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --masspoint)
            SINGLE_MASSPOINT="$2"
            shift 2
            ;;
        --masspoint-set)
            MASSPOINT_SET="$2"
            shift 2
            ;;
        --binning)
            BINNING="$2"
            shift 2
            ;;
        --nuisance)
            NUISANCE="$2"
            EXTRA_ARGS="$EXTRA_ARGS --nuisance $2"
            shift 2
            ;;
        --pull-fit)
            PULL_FIT="$2"
            shift 2
            ;;
        --unblind)
            EXTRA_ARGS="$EXTRA_ARGS --unblind"
            shift
            ;;
        --partial-unblind)
            EXTRA_ARGS="$EXTRA_ARGS --partial-unblind"
            shift
            ;;
        --debug)
            EXTRA_ARGS="$EXTRA_ARGS --debug"
            shift
            ;;
        --start-from)
            START_FROM="$2"
            shift 2
            ;;
        --plot-score)
            DO_PLOT_SCORE_SET=true
            DO_PLOT_SCORE=true
            shift
            ;;
        --no-plot-score)
            DO_PLOT_SCORE_SET=true
            DO_PLOT_SCORE=false
            shift
            ;;
        --no-printDatacard)
            DO_PRINT_DATACARD=false
            shift
            ;;
        --no-runAsymptotic)
            DO_RUN_ASYMPTOTIC=false
            shift
            ;;
        --fitdiag)
            DO_FITDIAG=true
            shift
            ;;
        --no-fitdiag)
            DO_FITDIAG=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--mode all|run2|run3] [--era Run2|Run3|All] [--method Baseline|ParticleNet|PTOptimized] [OPTIONS]"
            echo ""
            echo "V3 default builds merged Run-period component templates directly."
            echo "No per-subera combineCards.py stage is used."
            echo ""
            echo "Options:"
            echo "  --mode all|run2|run3       Run-period targets [default: all]"
            echo "  --era Run2|Run3|All        Build a single Run-period target"
            echo "  --method METHOD            Baseline or ParticleNet [default: Baseline]"
            echo "  --masspoint MP             Build a single mass point only"
            echo "  --masspoint-set SET        Override the mass-point list:"
            echo "                             baseline | particlenet | ptoptimized"
            echo "                             (default: chosen from --method)"
            echo "  --binning BINNING          extended or uniform [default: extended]"
            echo "  --nuisance MODE            fallback_lnn or preserve_shape"
            echo "  --unblind                  Use full real data"
            echo "  --partial-unblind          Use ParticleNet sideband data"
            echo "  --start-from STEP          template, merge_template, datacard, validate, asymptotic, fitdiag, plotpostfit, plotpulls"
            echo "  --fitdiag                  Run FitDiagnostics and post-fit plots"
            echo "  --dry-run                  Generate DAGs without submitting"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

case "$MODE" in
    all|run2|run3) ;;
    *) echo "ERROR: Invalid --mode '$MODE'"; exit 1 ;;
esac
if [[ -n "$SINGLE_ERA" ]]; then
    case "$SINGLE_ERA" in
        Run2|Run3|All) ;;
        *) echo "ERROR: Invalid --era '$SINGLE_ERA'. V3 template targets are Run2, Run3, or All."; exit 1 ;;
    esac
fi
case "$METHOD" in
    Baseline|ParticleNet|PTOptimized) ;;
    *) echo "ERROR: Invalid --method '$METHOD'"; exit 1 ;;
esac
case "$BINNING" in
    extended|uniform) ;;
    *) echo "ERROR: Invalid --binning '$BINNING'"; exit 1 ;;
esac
case "$NUISANCE" in
    fallback_lnn|preserve_shape) ;;
    *) echo "ERROR: Invalid --nuisance '$NUISANCE'"; exit 1 ;;
esac
case "$PULL_FIT" in
    b|both) ;;
    *) echo "ERROR: Invalid --pull-fit '$PULL_FIT'"; exit 1 ;;
esac
case "$START_FROM" in
    template|merge_template|datacard|validate|asymptotic|fitdiag|plotpostfit|plotpulls|plot_score) ;;
    *) echo "ERROR: Invalid --start-from '$START_FROM'"; exit 1 ;;
esac
if [[ "$DO_PLOT_SCORE_SET" == "false" && "$METHOD" == "ParticleNet" ]]; then
    DO_PLOT_SCORE=true
fi

case "$MASSPOINT_SET" in
    "")
        if [[ "$METHOD" == "ParticleNet" ]]; then
            MASSPOINTs=("${MASSPOINTs_PARTICLENET[@]}")
        elif [[ "$METHOD" == "PTOptimized" ]]; then
            MASSPOINTs=("${MASSPOINTs_PTOPTIMIZED[@]}")
        else
            MASSPOINTs=("${MASSPOINTs_BASELINE[@]}")
        fi
        ;;
    baseline)    MASSPOINTs=("${MASSPOINTs_BASELINE[@]}") ;;
    particlenet) MASSPOINTs=("${MASSPOINTs_PARTICLENET[@]}") ;;
    ptoptimized) MASSPOINTs=("${MASSPOINTs_PTOPTIMIZED[@]}") ;;
    *)
        # Any other value is looked up as a top-level array key in
        # configs/masspoints.json, so ad-hoc subsets (e.g. "baseline_todo")
        # can be driven without editing this script.
        _mp_json="$SCRIPT_DIR/configs/masspoints.json"
        _mp_list=$(python3 -c "
import json,sys
d=json.load(open('$_mp_json'))
v=d.get('$MASSPOINT_SET')
if not isinstance(v,list):
    sys.stderr.write(\"no such array key: $MASSPOINT_SET\n\"); sys.exit(1)
print(' '.join(v))
") || { echo "ERROR: --masspoint-set '$MASSPOINT_SET' is not an array key in configs/masspoints.json"; exit 1; }
        read -ra MASSPOINTs <<< "$_mp_list"
        ;;
esac
if [[ ${#MASSPOINTs[@]} -eq 0 ]]; then
    echo "ERROR: empty mass-point list for method '$METHOD'${MASSPOINT_SET:+ / set '$MASSPOINT_SET'}"
    exit 1
fi
if [[ -n "$SINGLE_MASSPOINT" ]]; then
    found=false
    for mp in "${MASSPOINTs[@]}"; do
        if [[ "$mp" == "$SINGLE_MASSPOINT" ]]; then
            found=true
            break
        fi
    done
    if [[ "$found" != "true" ]]; then
        echo "ERROR: masspoint '$SINGLE_MASSPOINT' is not configured for method '$METHOD'" >&2
        echo "Configured mass points: ${MASSPOINTs[*]}" >&2
        exit 1
    fi
    MASSPOINTs=("$SINGLE_MASSPOINT")
fi

targets_for_request() {
    if [[ -n "$SINGLE_ERA" ]]; then
        echo "$SINGLE_ERA"
        return
    fi
    case "$MODE" in
        run2) echo "Run2" ;;
        run3) echo "Run3" ;;
        all) echo "Run2 Run3 All" ;;
    esac
}

downstream_channels_for_request() {
    echo "SR1E2Mu SR3Mu Combined"
}

print_downstream_grid() {
    local targets
    local channels
    read -r -a targets <<< "$(targets_for_request)"
    read -r -a channels <<< "$(downstream_channels_for_request)"

    local entries=()
    local target
    local channel
    for target in "${targets[@]}"; do
        for channel in "${channels[@]}"; do
            entries+=("${target}/${channel}")
        done
    done
    printf '%s ' "${entries[@]}"
    echo
}

template_periods_for_request() {
    if [[ -n "$SINGLE_ERA" ]]; then
        case "$SINGLE_ERA" in
            Run2) echo "Run2" ;;
            Run3) echo "Run3" ;;
            All) echo "Run2 Run3" ;;
        esac
        return
    fi
    case "$MODE" in
        run2) echo "Run2" ;;
        run3) echo "Run3" ;;
        all) echo "Run2 Run3" ;;
    esac
}

merge_sources_for_target() {
    case "$1" in
        Run2) echo "Run2:SR1E2Mu,Run2:SR3Mu" ;;
        Run3) echo "Run3:SR1E2Mu,Run3:SR3Mu" ;;
        All) echo "Run2:SR1E2Mu,Run2:SR3Mu,Run3:SR1E2Mu,Run3:SR3Mu" ;;
    esac
}

merge_sources_for_channel_target() {
    local target=$1
    local channel=$2
    case "$channel" in
        Combined)
            merge_sources_for_target "$target"
            ;;
        SR1E2Mu|SR3Mu)
            case "$target" in
                All) echo "Run2:${channel},Run3:${channel}" ;;
                *) echo "${target}:${channel}" ;;
            esac
            ;;
    esac
}

fitdiag_target_enabled() {
    local target=$1
    if [[ "$DO_FITDIAG" != "true" ]]; then
        return 1
    fi
    if [[ -n "$SINGLE_ERA" ]]; then
        return 0
    fi
    case "$MODE" in
        run2|run3) return 0 ;;
        all) [[ "$target" == "All" ]] && return 0 || return 1 ;;
    esac
    return 1
}

step_to_level() {
    case "$1" in
        template) echo 0 ;;
        merge_template) echo 1 ;;
        datacard) echo 2 ;;
        validate) echo 3 ;;
        asymptotic) echo 4 ;;
        fitdiag) echo 5 ;;
        plotpostfit) echo 6 ;;
        plotpulls) echo 6 ;;
        plot_score) echo 7 ;;
        *) echo 0 ;;
    esac
}

generate_dag_file() {
    local masspoint=$1
    local method=$2
    local binning=$3
    local extra_args=${4:-}
    local dag_file=$5
    local start_from=${6:-template}
    read -r -a targets <<< "$(targets_for_request)"
    read -r -a template_periods <<< "$(template_periods_for_request)"

    local start_level
    start_level=$(step_to_level "$start_from")

    job_done_suffix() {
        local step=$1
        local level
        level=$(step_to_level "$step")
        if [[ $level -lt $start_level ]]; then
            echo " DONE"
            return
        fi
        case "$step" in
            datacard)
                [[ "$DO_PRINT_DATACARD" == "false" ]] && echo " DONE" && return
                ;;
            asymptotic)
                [[ "$DO_RUN_ASYMPTOTIC" == "false" ]] && echo " DONE" && return
                ;;
            fitdiag|plotpostfit|plotpulls)
                [[ "$DO_FITDIAG" == "false" ]] && echo " DONE" && return
                ;;
            plot_score)
                [[ "$DO_PLOT_SCORE" == "false" ]] && echo " DONE" && return
                ;;
        esac
        echo ""
    }

    local asymptotic_extra_args=""
    if [[ "$extra_args" == *"--partial-unblind"* ]]; then
        asymptotic_extra_args="--partial-unblind"
    elif [[ "$extra_args" == *"--unblind"* ]]; then
        asymptotic_extra_args="--unblind"
    fi
    if [[ "$extra_args" == *"--nuisance preserve_shape"* ]]; then
        asymptotic_extra_args="$asymptotic_extra_args --nuisance preserve_shape"
    fi
    local pull_extra_args="$asymptotic_extra_args --pull-fit $PULL_FIT"

    cat > "$dag_file" << EOF
# DAG for $masspoint (Run-period component V3 workflow, start-from: $start_from)
CONFIG dagman.config

EOF

    local target
    local period
    local done_sfx
    done_sfx=$(job_done_suffix template)
    for period in "${template_periods[@]}"; do
        echo "JOB template_SR1E2Mu_${period} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS template_SR1E2Mu_${period} step=\"template\" era=\"${period}\" channel=\"SR1E2Mu\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" output_era=\"\" extra_args=\"${extra_args}\" request_cpus=\"1\" job_request_memory=\"6144\"" >> "$dag_file"
        echo "JOB template_SR3Mu_${period} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS template_SR3Mu_${period} step=\"template\" era=\"${period}\" channel=\"SR3Mu\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" output_era=\"\" extra_args=\"${extra_args}\" request_cpus=\"1\" job_request_memory=\"6144\"" >> "$dag_file"
    done

    for target in "${targets[@]}"; do
        done_sfx=$(job_done_suffix merge_template)
        local merge_sources
        merge_sources=$(merge_sources_for_target "$target")
        echo "JOB merge_template_Combined_${target} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS merge_template_Combined_${target} step=\"merge_template\" era=\"${target}\" channel=\"Combined\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" output_era=\"\" extra_args=\"${extra_args} --sources ${merge_sources}\" request_cpus=\"1\" job_request_memory=\"2048\"" >> "$dag_file"

        done_sfx=$(job_done_suffix datacard)
        echo "JOB datacard_Combined_${target} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS datacard_Combined_${target} step=\"datacard\" era=\"${target}\" channel=\"Combined\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" output_era=\"\" extra_args=\"${extra_args}\" request_cpus=\"1\" job_request_memory=\"2048\"" >> "$dag_file"

        done_sfx=$(job_done_suffix validate)
        echo "JOB validate_Combined_${target} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS validate_Combined_${target} step=\"validate\" era=\"${target}\" channel=\"Combined\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" output_era=\"\" extra_args=\"${extra_args}\" request_cpus=\"1\" job_request_memory=\"2048\"" >> "$dag_file"

        done_sfx=$(job_done_suffix asymptotic)
        echo "JOB asymptotic_Combined_${target} jobs.sub${done_sfx}" >> "$dag_file"
        echo "VARS asymptotic_Combined_${target} step=\"asymptotic\" era=\"${target}\" channel=\"Combined\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" output_era=\"\" extra_args=\"${asymptotic_extra_args}\" request_cpus=\"1\" job_request_memory=\"2048\"" >> "$dag_file"

        if fitdiag_target_enabled "$target"; then
            done_sfx=$(job_done_suffix fitdiag)
            echo "JOB fitdiag_Combined_${target} jobs.sub${done_sfx}" >> "$dag_file"
            echo "VARS fitdiag_Combined_${target} step=\"fitdiag\" era=\"${target}\" channel=\"Combined\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" output_era=\"\" extra_args=\"${asymptotic_extra_args}\" request_cpus=\"1\" job_request_memory=\"2048\"" >> "$dag_file"

            done_sfx=$(job_done_suffix plotpostfit)
            echo "JOB plotpostfit_Combined_${target} jobs.sub${done_sfx}" >> "$dag_file"
            echo "VARS plotpostfit_Combined_${target} step=\"plotpostfit\" era=\"${target}\" channel=\"Combined\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" output_era=\"\" extra_args=\"${asymptotic_extra_args}\" request_cpus=\"1\" job_request_memory=\"2048\"" >> "$dag_file"

            done_sfx=$(job_done_suffix plotpulls)
            echo "JOB plotpulls_Combined_${target} jobs.sub${done_sfx}" >> "$dag_file"
            echo "VARS plotpulls_Combined_${target} step=\"plotpulls\" era=\"${target}\" channel=\"Combined\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" output_era=\"\" extra_args=\"${pull_extra_args}\" request_cpus=\"1\" job_request_memory=\"2048\"" >> "$dag_file"
        fi

        if [[ "$method" == "ParticleNet" && "$DO_PLOT_SCORE" == "true" ]]; then
            done_sfx=$(job_done_suffix plot_score)
            echo "JOB plot_score_Combined_${target} jobs.sub${done_sfx}" >> "$dag_file"
            echo "VARS plot_score_Combined_${target} step=\"plot_score\" era=\"${target}\" channel=\"Combined\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" output_era=\"\" extra_args=\"${extra_args}\" request_cpus=\"1\" job_request_memory=\"2048\"" >> "$dag_file"
        fi

        for channel in SR1E2Mu SR3Mu; do
            if [[ "$target" == "All" ]]; then
                done_sfx=$(job_done_suffix merge_template)
                merge_sources=$(merge_sources_for_channel_target "$target" "$channel")
                echo "JOB merge_template_${channel}_${target} jobs.sub${done_sfx}" >> "$dag_file"
                echo "VARS merge_template_${channel}_${target} step=\"merge_template\" era=\"${target}\" channel=\"${channel}\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" output_era=\"\" extra_args=\"${extra_args} --sources ${merge_sources}\" request_cpus=\"1\" job_request_memory=\"2048\"" >> "$dag_file"
            fi

            done_sfx=$(job_done_suffix datacard)
            echo "JOB datacard_${channel}_${target} jobs.sub${done_sfx}" >> "$dag_file"
            echo "VARS datacard_${channel}_${target} step=\"datacard\" era=\"${target}\" channel=\"${channel}\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" output_era=\"\" extra_args=\"${extra_args}\" request_cpus=\"1\" job_request_memory=\"2048\"" >> "$dag_file"

            done_sfx=$(job_done_suffix validate)
            echo "JOB validate_${channel}_${target} jobs.sub${done_sfx}" >> "$dag_file"
            echo "VARS validate_${channel}_${target} step=\"validate\" era=\"${target}\" channel=\"${channel}\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" output_era=\"\" extra_args=\"${extra_args}\" request_cpus=\"1\" job_request_memory=\"2048\"" >> "$dag_file"

            done_sfx=$(job_done_suffix asymptotic)
            echo "JOB asymptotic_${channel}_${target} jobs.sub${done_sfx}" >> "$dag_file"
            echo "VARS asymptotic_${channel}_${target} step=\"asymptotic\" era=\"${target}\" channel=\"${channel}\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" output_era=\"\" extra_args=\"${asymptotic_extra_args}\" request_cpus=\"1\" job_request_memory=\"2048\"" >> "$dag_file"

            if [[ "$method" == "ParticleNet" && "$DO_PLOT_SCORE" == "true" ]]; then
                done_sfx=$(job_done_suffix plot_score)
                echo "JOB plot_score_${channel}_${target} jobs.sub${done_sfx}" >> "$dag_file"
                echo "VARS plot_score_${channel}_${target} step=\"plot_score\" era=\"${target}\" channel=\"${channel}\" masspoint=\"${masspoint}\" method=\"${method}\" binning=\"${binning}\" output_era=\"\" extra_args=\"${extra_args}\" request_cpus=\"1\" job_request_memory=\"2048\"" >> "$dag_file"
            fi
        done
    done

    echo "" >> "$dag_file"
    echo "# Dependencies" >> "$dag_file"
    for target in "${targets[@]}"; do
        case "$target" in
            Run2|Run3)
                echo "PARENT template_SR1E2Mu_${target} template_SR3Mu_${target} CHILD merge_template_Combined_${target}" >> "$dag_file"
                ;;
            All)
                echo "PARENT template_SR1E2Mu_Run2 template_SR3Mu_Run2 template_SR1E2Mu_Run3 template_SR3Mu_Run3 CHILD merge_template_Combined_All" >> "$dag_file"
                ;;
        esac
        echo "PARENT merge_template_Combined_${target} CHILD datacard_Combined_${target}" >> "$dag_file"
        echo "PARENT datacard_Combined_${target} CHILD validate_Combined_${target}" >> "$dag_file"
        echo "PARENT validate_Combined_${target} CHILD asymptotic_Combined_${target}" >> "$dag_file"
        if fitdiag_target_enabled "$target"; then
            echo "PARENT validate_Combined_${target} CHILD fitdiag_Combined_${target}" >> "$dag_file"
            echo "PARENT fitdiag_Combined_${target} CHILD plotpostfit_Combined_${target} plotpulls_Combined_${target}" >> "$dag_file"
        fi
        if [[ "$method" == "ParticleNet" && "$DO_PLOT_SCORE" == "true" ]]; then
            echo "PARENT merge_template_Combined_${target} CHILD plot_score_Combined_${target}" >> "$dag_file"
        fi

        for channel in SR1E2Mu SR3Mu; do
            if [[ "$target" == "All" ]]; then
                echo "PARENT template_${channel}_Run2 template_${channel}_Run3 CHILD merge_template_${channel}_All" >> "$dag_file"
                echo "PARENT merge_template_${channel}_All CHILD datacard_${channel}_All" >> "$dag_file"
            else
                echo "PARENT template_${channel}_${target} CHILD datacard_${channel}_${target}" >> "$dag_file"
            fi
            echo "PARENT datacard_${channel}_${target} CHILD validate_${channel}_${target}" >> "$dag_file"
            echo "PARENT validate_${channel}_${target} CHILD asymptotic_${channel}_${target}" >> "$dag_file"
            if [[ "$method" == "ParticleNet" && "$DO_PLOT_SCORE" == "true" ]]; then
                if [[ "$target" == "All" ]]; then
                    echo "PARENT merge_template_${channel}_All CHILD plot_score_${channel}_All" >> "$dag_file"
                else
                    echo "PARENT template_${channel}_${target} CHILD plot_score_${channel}_${target}" >> "$dag_file"
                fi
            fi
        done
    done
}

submit_condor_dags() {
    local -n masspoints_ref=$1
    local method=$2
    local binning=$3
    local extra_args=${4:-}

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local dir_suffix="$MODE"
    [[ -n "$SINGLE_ERA" ]] && dir_suffix="$SINGLE_ERA"
    local job_dir="$CONDOR_DIR/jobs_dag_${dir_suffix}_${method}_${timestamp}"
    mkdir -p "$job_dir"
    cp "$SCRIPT_DIR/configs/dagman.config" "$job_dir/"

    for masspoint in "${masspoints_ref[@]}"; do
        local mp_dir="$job_dir/$masspoint"
        mkdir -p "$mp_dir/logs"
        cat > "$mp_dir/jobs.sub" << EOF
JobBatchName = ${masspoint}
universe = vanilla
executable = $SCRIPT_DIR/scripts/makeBinnedTemplates_wrapper.sh
arguments = "'\$(step)' '\$(era)' '\$(channel)' '\$(masspoint)' '\$(method)' '\$(binning)' '\$(output_era)' '\$(extra_args)'"
output = logs/\$(step)_\$(channel)_\$(era).out
error = logs/\$(step)_\$(channel)_\$(era).err
log = dag.log
request_cpus = \$(request_cpus)
RequestMemory = \$(job_request_memory)
request_disk = 4GB
getenv = True
should_transfer_files = NO
queue
EOF
        cp "$job_dir/dagman.config" "$mp_dir/"
        generate_dag_file "$masspoint" "$method" "$binning" "$extra_args" "$mp_dir/dag.dag" "$START_FROM"
        echo "Generated DAG: $mp_dir/dag.dag"
    done

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

    echo "Generated DAGMan workflows in: $job_dir"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY-RUN] To submit: cd $job_dir && ./submit_all.sh"
    else
        cd "$job_dir" && ./submit_all.sh
        cd - > /dev/null
    fi
}

echo "============================================================"
echo "SignalRegionStudyV3 Run-period Component Workflow"
echo "Mode: $MODE"
[[ -n "$SINGLE_ERA" ]] && echo "Single target: $SINGLE_ERA"
echo "Method: $METHOD"
[[ -n "$SINGLE_MASSPOINT" ]] && echo "Single mass point: $SINGLE_MASSPOINT"
echo "Binning: $BINNING"
echo "Nuisance: $NUISANCE"
echo "Targets: $(targets_for_request)"
echo "Downstream datacard/validation/asymptotic grid: $(print_downstream_grid)"
echo "Mass points: ${MASSPOINTs[*]}"
echo "FitDiagnostics: $DO_FITDIAG"
echo "Dry run: $DRY_RUN"
echo "============================================================"

submit_condor_dags MASSPOINTs "$METHOD" "$BINNING" "$EXTRA_ARGS"
