#!/bin/bash
#
# ttz_cr_gof.sh - Top-level orchestrator for the TTZ2E1Mu Run-period CR GoF DAG
#
# Builds and submits one HTCondor DAG using the same V3 Run-period component
# template/datacard construction as the signal-region workflow.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER="$SCRIPT_DIR/scripts/ttz_cr_wrapper.sh"
DAGMAN_CONFIG="$SCRIPT_DIR/configs/dagman.config"
CONDOR_DIR="$SCRIPT_DIR/condor"

CHANNEL="TTZ2E1Mu"
MASSPOINT="MHc130_MA90"
METHOD="CR"
BINNING_TAG="ZWin_adaptive"
TARGETS=("Run2" "Run3" "All")

NTOYS=500
NBATCHES=5
START_FROM="template"
PLOT_ONLY=false
DO_IMPACTS=true
DO_PULLS=true
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --ntoys)      NTOYS="$2"; shift 2 ;;
        --nbatches)   NBATCHES="$2"; shift 2 ;;
        --start-from) START_FROM="$2"; shift 2 ;;
        --plot-only)  PLOT_ONLY=true; shift ;;
        --skip-impacts) DO_IMPACTS=false; shift ;;
        --skip-pulls) DO_PULLS=false; shift ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "  --ntoys N           Total GoF toys per Run-period target [default: 500]"
            echo "  --nbatches N        Toy batches per target [default: 5]"
            echo "  --start-from STEP   template, datacard, validate, workspace, gof, fitdiag, impact, plotpulls"
            echo "  --plot-only         Run only gof_collect + plotpostfit + plotpulls"
            echo "  --skip-impacts      Do not add observed impact jobs"
            echo "  --skip-pulls        Do not add postfit pull-plot jobs"
            echo "  --dry-run           Generate DAG without submitting"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "$START_FROM" in
    template|datacard|validate|workspace|gof|fitdiag|impact|plotpulls) ;;
    *) echo "ERROR: invalid --start-from '$START_FROM'"; exit 1 ;;
esac

TOYS_PER_BATCH=$((NTOYS / NBATCHES))
[[ "$TOYS_PER_BATCH" -lt 1 ]] && TOYS_PER_BATCH=1

step_level() {
    case "$1" in
        template)  echo 0 ;;
        datacard)  echo 1 ;;
        validate)  echo 2 ;;
        workspace) echo 3 ;;
        gof)       echo 4 ;;
        fitdiag)   echo 4 ;;
        impact)    echo 5 ;;
        collect)   echo 5 ;;
        plot)      echo 5 ;;
        plotpulls) echo 5 ;;
        *) echo 0 ;;
    esac
}
START_LEVEL=$(step_level "$START_FROM")

job_done() {
    local step=$1
    if [[ "$START_FROM" == "impact" ]]; then
        [[ "$step" == "impact" ]] && echo "" || echo " DONE"
        return
    fi
    if [[ "$START_FROM" == "plotpulls" ]]; then
        [[ "$step" == "plotpulls" ]] && echo "" || echo " DONE"
        return
    fi
    if [[ "$PLOT_ONLY" == true ]]; then
        case "$step" in
            collect|plot|plotpulls) echo "" ;;
            impact)       echo " DONE" ;;
            *)            echo " DONE" ;;
        esac
        return
    fi
    local lvl
    lvl=$(step_level "$step")
    if [[ "$lvl" -lt "$START_LEVEL" ]]; then
        echo " DONE"
    else
        echo ""
    fi
}

echo "============================================================"
echo "TTZ2E1Mu CR Run-period GoF DAG"
echo "  Channel: $CHANNEL  Masspoint: $MASSPOINT  Method: $METHOD  Binning: $BINNING_TAG"
echo "  Targets: ${TARGETS[*]}"
echo "  Toys per target: $NTOYS ($NBATCHES batches x $TOYS_PER_BATCH)"
echo "  Start from: $START_FROM (plot-only: $PLOT_ONLY)"
echo "  Impacts: $DO_IMPACTS  Pull plots: $DO_PULLS"
echo "  Dry-run: $DRY_RUN"
echo "============================================================"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
mkdir -p "$CONDOR_DIR"
idx=0
while true; do
    if [[ "$idx" -eq 0 ]]; then
        JOB_DIR="$CONDOR_DIR/jobs_dag_ttz_cr_${TIMESTAMP}"
    else
        JOB_DIR="$CONDOR_DIR/jobs_dag_ttz_cr_${TIMESTAMP}_${idx}"
    fi
    if mkdir "$JOB_DIR" 2>/dev/null; then
        break
    fi
    idx=$((idx + 1))
done
mkdir -p "$JOB_DIR/logs"
cp "$DAGMAN_CONFIG" "$JOB_DIR/"

cat > "$JOB_DIR/jobs.sub" << EOFSUB
JobBatchName = ttz_cr_${TIMESTAMP}
universe = vanilla
executable = $WRAPPER
arguments = \$(step) \$(era) \$(seed) \$(tpb)
output = logs/\$(step)_\$(era)_\$(seed).out
error  = logs/\$(step)_\$(era)_\$(seed).err
log    = dag.log

request_cpus   = 1
request_memory = 4GB
request_disk   = 2GB

getenv = True
should_transfer_files = NO

queue
EOFSUB

DAG_FILE="$JOB_DIR/dag.dag"
{
    echo "# TTZ2E1Mu CR Run-period GoF DAG (timestamp: $TIMESTAMP)"
    echo "CONFIG dagman.config"
    echo ""

    for target in "${TARGETS[@]}"; do
        sfx=$(job_done template)
        echo "JOB template_${target} jobs.sub${sfx}"
        echo "VARS template_${target} step=\"template\" era=\"${target}\" seed=\"0\" tpb=\"0\""

        sfx=$(job_done datacard)
        echo "JOB datacard_${target} jobs.sub${sfx}"
        echo "VARS datacard_${target} step=\"datacard\" era=\"${target}\" seed=\"0\" tpb=\"0\""

        sfx=$(job_done validate)
        echo "JOB validate_${target} jobs.sub${sfx}"
        echo "VARS validate_${target} step=\"validate\" era=\"${target}\" seed=\"0\" tpb=\"0\""

        sfx=$(job_done workspace)
        echo "JOB workspace_${target} jobs.sub${sfx}"
        echo "VARS workspace_${target} step=\"workspace\" era=\"${target}\" seed=\"0\" tpb=\"0\""

        sfx_g=$(job_done gof)
        echo "JOB gof_data_${target} jobs.sub${sfx_g}"
        echo "VARS gof_data_${target} step=\"gof_data\" era=\"${target}\" seed=\"0\" tpb=\"0\""
        for seed in $(seq 1 "$NBATCHES"); do
            echo "JOB gof_toys_${target}_s${seed} jobs.sub${sfx_g}"
            echo "VARS gof_toys_${target}_s${seed} step=\"gof_toys\" era=\"${target}\" seed=\"${seed}\" tpb=\"${TOYS_PER_BATCH}\""
        done

        sfx_f=$(job_done fitdiag)
        echo "JOB fitdiag_${target} jobs.sub${sfx_f}"
        echo "VARS fitdiag_${target} step=\"fitdiag\" era=\"${target}\" seed=\"0\" tpb=\"0\""

        sfx_c=$(job_done collect)
        echo "JOB gof_collect_${target} jobs.sub${sfx_c}"
        echo "VARS gof_collect_${target} step=\"gof_collect\" era=\"${target}\" seed=\"0\" tpb=\"0\""

        sfx_p=$(job_done plot)
        echo "JOB plotpostfit_${target} jobs.sub${sfx_p}"
        echo "VARS plotpostfit_${target} step=\"plotpostfit\" era=\"${target}\" seed=\"0\" tpb=\"0\""

        if [[ "$DO_PULLS" == true ]]; then
            sfx_pull=$(job_done plotpulls)
            echo "JOB plotpulls_${target} jobs.sub${sfx_pull}"
            echo "VARS plotpulls_${target} step=\"plotpulls\" era=\"${target}\" seed=\"0\" tpb=\"0\""
        fi

        if [[ "$DO_IMPACTS" == true ]]; then
            sfx_i=$(job_done impact)
            echo "JOB impact_${target} jobs.sub${sfx_i}"
            echo "VARS impact_${target} step=\"impact\" era=\"${target}\" seed=\"0\" tpb=\"0\""
        fi
        echo ""
    done

    echo "# Dependencies"
    for target in "${TARGETS[@]}"; do
        toy_jobs=""
        for seed in $(seq 1 "$NBATCHES"); do
            toy_jobs+="gof_toys_${target}_s${seed} "
        done
        echo "PARENT template_${target} CHILD datacard_${target}"
        echo "PARENT datacard_${target} CHILD validate_${target}"
        echo "PARENT validate_${target} CHILD workspace_${target}"
        echo "PARENT workspace_${target} CHILD gof_data_${target} ${toy_jobs}fitdiag_${target}"
        echo "PARENT gof_data_${target} ${toy_jobs}CHILD gof_collect_${target}"
        children="plotpostfit_${target}"
        if [[ "$DO_PULLS" == true ]]; then
            children+=" plotpulls_${target}"
        fi
        echo "PARENT fitdiag_${target} CHILD ${children}"
        if [[ "$DO_IMPACTS" == true ]]; then
            echo "PARENT workspace_${target} CHILD impact_${target}"
        fi
    done
} > "$DAG_FILE"

echo ""
echo "Generated DAG: $DAG_FILE"
echo "Total JOB lines: $(grep -c '^JOB ' "$DAG_FILE")"
echo "Done-skipped:   $(grep -c ' DONE$' "$DAG_FILE")"
echo ""

cat > "$JOB_DIR/submit_all.sh" << 'SUBMIT_EOF'
#!/bin/bash
set -e
cd "$(dirname "$0")"
condor_submit_dag -config dagman.config dag.dag
SUBMIT_EOF
chmod +x "$JOB_DIR/submit_all.sh"

cat > "$JOB_DIR/status_all.sh" << 'STATUS_EOF'
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
STATUS_EOF
chmod +x "$JOB_DIR/status_all.sh"

if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY-RUN] To submit:"
    echo "  cd $JOB_DIR && ./submit_all.sh"
    echo "[DRY-RUN] To monitor:"
    echo "  cd $JOB_DIR && ./status_all.sh"
else
    echo "Submitting DAG..."
    cd "$JOB_DIR"
    ./submit_all.sh
    cd - > /dev/null
    echo ""
    echo "Monitor with:"
    echo "  cd $JOB_DIR && ./status_all.sh"
    echo "  condor_q -dag"
fi
