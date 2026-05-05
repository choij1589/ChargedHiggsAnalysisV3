#!/bin/bash
#
# ttz_cr_gof.sh - Top-level orchestrator for the TTZ2E1Mu control-region GoF DAG
#
# Builds and submits a single HTCondor DAG covering:
#   Layer 0: per-era template generation (8 eras)
#   Layer 1: per-era validate + datacard
#   Layer 2: era combination (Run2, Run3, All)
#   Layer 3: per-combined-era GoF (data + N toy batches), FitDiagnostics
#   Layer 4: per-combined-era GoF collect+plot, post-fit plotting
#
# All work uses the CR-fixed configuration: TTZ2E1Mu / MHc130_MA90 / CR / ZWin_adaptive.
#
# Usage:
#   ./automize/ttz_cr_gof.sh                               # full pipeline, 500 toys / 5 batches
#   ./automize/ttz_cr_gof.sh --ntoys 1000 --nbatches 10
#   ./automize/ttz_cr_gof.sh --start-from datacard         # skip templates+validate
#   ./automize/ttz_cr_gof.sh --plot-only                   # only re-collect + re-plot
#   ./automize/ttz_cr_gof.sh --dry-run                     # build DAG but don't submit

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WRAPPER="$SCRIPT_DIR/scripts/ttz_cr_wrapper.sh"
DAGMAN_CONFIG="$SCRIPT_DIR/configs/dagman.config"
CONDOR_DIR="$SCRIPT_DIR/condor"

# CR fixed parameters
CHANNEL="TTZ2E1Mu"
MASSPOINT="MHc130_MA90"
METHOD="CR"
BINNING_TAG="ZWin_adaptive"

ERAS_RUN2=("2016preVFP" "2016postVFP" "2017" "2018")
ERAS_RUN3=("2022" "2022EE" "2023" "2023BPix")
ERAS_ALL=("${ERAS_RUN2[@]}" "${ERAS_RUN3[@]}")
COMBINED_ERAS=("Run2" "Run3" "All")

# Defaults
NTOYS=500
NBATCHES=5
START_FROM="template"
PLOT_ONLY=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --ntoys)      NTOYS="$2"; shift 2 ;;
        --nbatches)   NBATCHES="$2"; shift 2 ;;
        --start-from) START_FROM="$2"; shift 2 ;;
        --plot-only)  PLOT_ONLY=true; shift ;;
        --dry-run)    DRY_RUN=true; shift ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo "  --ntoys N           Total GoF toys per combined era [default: 500]"
            echo "  --nbatches N        Toy batches per combined era [default: 5]"
            echo "  --start-from STEP   Skip steps up to this point as DONE in the DAG."
            echo "                      Values: template, validate, datacard, combine_era, gof, fitdiag"
            echo "  --plot-only         Run only gof_collect + plotpostfit (everything else DONE)"
            echo "  --dry-run           Generate DAG without submitting"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

case "$START_FROM" in
    template|validate|datacard|combine_era|gof|fitdiag) ;;
    *) echo "ERROR: invalid --start-from '$START_FROM'"; exit 1 ;;
esac

TOYS_PER_BATCH=$((NTOYS / NBATCHES))
[[ "$TOYS_PER_BATCH" -lt 1 ]] && TOYS_PER_BATCH=1

# Step level mapping (for --start-from). Steps with level < start_level are DONE.
# gof and fitdiag run in parallel but are independent — split levels so users
# can skip one without the other.
step_level() {
    case "$1" in
        template)    echo 0 ;;
        validate)    echo 1 ;;
        datacard)    echo 1 ;;
        combine_era) echo 2 ;;
        gof)         echo 3 ;;
        fitdiag)     echo 3 ;;
        collect)     echo 4 ;;
        plot)        echo 4 ;;
        *) echo 0 ;;
    esac
}
START_LEVEL=$(step_level "$START_FROM")

# job_done: prints " DONE" if a step is to be skipped, empty otherwise.
job_done() {
    local step=$1
    # --plot-only: only gof_collect + plotpostfit run; everything else is DONE
    if [[ "$PLOT_ONLY" == true ]]; then
        case "$step" in
            collect|plot) echo "" ;;
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
echo "TTZ2E1Mu CR GoF DAG"
echo "  Channel: $CHANNEL  Masspoint: $MASSPOINT  Method: $METHOD  Binning: $BINNING_TAG"
echo "  Toys per combined era: $NTOYS ($NBATCHES batches × $TOYS_PER_BATCH)"
echo "  Start from: $START_FROM (plot-only: $PLOT_ONLY)"
echo "  Dry-run: $DRY_RUN"
echo "============================================================"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_DIR="$CONDOR_DIR/jobs_dag_ttz_cr_${TIMESTAMP}"
mkdir -p "$JOB_DIR/logs"
cp "$DAGMAN_CONFIG" "$JOB_DIR/"

# One jobs.sub for every node; VARS supplies step/era/seed/tpb per JOB line
cat > "$JOB_DIR/jobs.sub" << EOFSUB
JobBatchName = ttz_cr_${TIMESTAMP}
universe = vanilla
executable = $WRAPPER
arguments = \$(step) \$(era) \$(seed) \$(tpb)
output = logs/\$(step)_\$(era)_\$(seed).out
error  = logs/\$(step)_\$(era)_\$(seed).err
log    = dag.log

request_cpus   = 1
request_memory = 2GB
request_disk   = 2GB

getenv = True
should_transfer_files = NO

queue
EOFSUB

DAG_FILE="$JOB_DIR/dag.dag"
{
    echo "# TTZ2E1Mu CR GoF DAG (timestamp: $TIMESTAMP)"
    echo "CONFIG dagman.config"
    echo ""

    # Layer 0: per-era templates
    sfx=$(job_done template)
    for era in "${ERAS_ALL[@]}"; do
        echo "JOB template_${era} jobs.sub${sfx}"
        echo "VARS template_${era} step=\"template\" era=\"${era}\" seed=\"0\" tpb=\"0\""
    done
    echo ""

    # Layer 1: per-era validate + datacard
    sfx_v=$(job_done validate)
    sfx_d=$(job_done datacard)
    for era in "${ERAS_ALL[@]}"; do
        echo "JOB validate_${era} jobs.sub${sfx_v}"
        echo "VARS validate_${era} step=\"validate\" era=\"${era}\" seed=\"0\" tpb=\"0\""
        echo "JOB datacard_${era} jobs.sub${sfx_d}"
        echo "VARS datacard_${era} step=\"datacard\" era=\"${era}\" seed=\"0\" tpb=\"0\""
    done
    echo ""

    # Layer 2: era combination
    sfx=$(job_done combine_era)
    for cera in "${COMBINED_ERAS[@]}"; do
        echo "JOB combine_era_${cera} jobs.sub${sfx}"
        echo "VARS combine_era_${cera} step=\"combine_era\" era=\"${cera}\" seed=\"0\" tpb=\"0\""
    done
    echo ""

    # Layer 3: per-combined-era GoF (data + toys) + FitDiagnostics (parallel with GoF)
    sfx_g=$(job_done gof)
    sfx_f=$(job_done fitdiag)
    for cera in "${COMBINED_ERAS[@]}"; do
        echo "JOB gof_data_${cera} jobs.sub${sfx_g}"
        echo "VARS gof_data_${cera} step=\"gof_data\" era=\"${cera}\" seed=\"0\" tpb=\"0\""
        for s in $(seq 1 "$NBATCHES"); do
            echo "JOB gof_toys_${cera}_s${s} jobs.sub${sfx_g}"
            echo "VARS gof_toys_${cera}_s${s} step=\"gof_toys\" era=\"${cera}\" seed=\"${s}\" tpb=\"${TOYS_PER_BATCH}\""
        done
        echo "JOB fitdiag_${cera} jobs.sub${sfx_f}"
        echo "VARS fitdiag_${cera} step=\"fitdiag\" era=\"${cera}\" seed=\"0\" tpb=\"0\""
    done
    echo ""

    # Layer 4: per-combined-era collect + post-fit plot
    sfx_c=$(job_done collect)
    sfx_p=$(job_done plot)
    for cera in "${COMBINED_ERAS[@]}"; do
        echo "JOB gof_collect_${cera} jobs.sub${sfx_c}"
        echo "VARS gof_collect_${cera} step=\"gof_collect\" era=\"${cera}\" seed=\"0\" tpb=\"0\""
        echo "JOB plotpostfit_${cera} jobs.sub${sfx_p}"
        echo "VARS plotpostfit_${cera} step=\"plotpostfit\" era=\"${cera}\" seed=\"0\" tpb=\"0\""
    done
    echo ""

    # Dependencies
    echo "# Dependencies"
    for era in "${ERAS_ALL[@]}"; do
        echo "PARENT template_${era} CHILD validate_${era} datacard_${era}"
    done

    # Per-era datacards feed the appropriate combine_era
    run2_dc_list=""
    for era in "${ERAS_RUN2[@]}"; do run2_dc_list+="datacard_${era} "; done
    run3_dc_list=""
    for era in "${ERAS_RUN3[@]}"; do run3_dc_list+="datacard_${era} "; done
    echo "PARENT $run2_dc_list CHILD combine_era_Run2"
    echo "PARENT $run3_dc_list CHILD combine_era_Run3"
    echo "PARENT combine_era_Run2 combine_era_Run3 CHILD combine_era_All"

    # combine_era → gof_data, gof_toys, fitdiag
    for cera in "${COMBINED_ERAS[@]}"; do
        toy_jobs=""
        for s in $(seq 1 "$NBATCHES"); do
            toy_jobs+="gof_toys_${cera}_s${s} "
        done
        echo "PARENT combine_era_${cera} CHILD gof_data_${cera} ${toy_jobs}fitdiag_${cera}"

        # gof_data + all toy batches → gof_collect
        echo "PARENT gof_data_${cera} ${toy_jobs}CHILD gof_collect_${cera}"

        # fitdiag → plotpostfit
        echo "PARENT fitdiag_${cera} CHILD plotpostfit_${cera}"
    done
} > "$DAG_FILE"

echo ""
echo "Generated DAG: $DAG_FILE"
echo "Total JOB lines: $(grep -c '^JOB ' "$DAG_FILE")"
echo "Done-skipped:   $(grep -c ' DONE$' "$DAG_FILE")"
echo ""

# submit_all.sh / status_all.sh helpers — same convention as automize/preprocess.sh
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
