#!/bin/bash
#
# test_dropRun3_partial.sh
#
# Sensitivity test: rebuild the "All" combined datacard dropping
# 2016postVFP, 2022, 2023BPix (keep 2016preVFP, 2017, 2018, 2022EE, 2023)
# and run asymptotic limits for 4 representative mass points.
#
# Submits one HTCondor DAG per mass point (combine_era -> asymptotic),
# reusing scripts/makeBinnedTemplates_wrapper.sh.
#
# Usage:
#   bash scripts/test_dropRun3_partial.sh [--dry-run] [--local]
#     --dry-run : generate DAGs but do not submit
#     --local   : run sequentially in the current shell (no condor)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDOR_DIR="$SCRIPT_DIR/condor"

ERAS_CSV="2016preVFP,2017,2018,2022EE,2023"
OUTPUT_ERA="All_drop_16post_22_23BPix"
CHANNEL="Combined"
METHOD="Baseline"
BINNING="extended"

MASSPOINTS=(
    "MHc70_MA15"
    "MHc130_MA55"
    "MHc100_MA95"
    "MHc160_MA155"
)

DRY_RUN=false
LOCAL_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run) DRY_RUN=true; shift ;;
        --local)   LOCAL_MODE=true; shift ;;
        -h|--help)
            sed -n '2,18p' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

echo "=============================================================="
echo " Test: drop 2016postVFP, 2022, 2023BPix from 'All' combination"
echo "   kept eras   : ${ERAS_CSV}"
echo "   output era  : ${OUTPUT_ERA}"
echo "   mass points : ${MASSPOINTS[*]}"
echo "   execution   : $([[ $LOCAL_MODE == true ]] && echo local || echo 'HTCondor DAGMan') (dry-run: ${DRY_RUN})"
echo "=============================================================="

# -----------------------------------------------------------------------------
# Local (sequential) mode — kept for quick single-machine testing
# -----------------------------------------------------------------------------
if [[ "$LOCAL_MODE" == true ]]; then
    if [[ -z "${WORKDIR:-}" ]]; then
        echo "ERROR: WORKDIR not set. Run 'source setup.sh'." >&2
        exit 1
    fi
    cd "$SCRIPT_DIR"

    run() {
        if [[ "$DRY_RUN" == true ]]; then
            echo "[DRY-RUN] $*"
        else
            echo "+ $*"
            eval "$*"
        fi
    }

    for MP in "${MASSPOINTS[@]}"; do
        echo ""
        echo "----- ${MP} -----"
        run "python3 python/combineDatacards.py \
                --mode era --eras ${ERAS_CSV} --output-era ${OUTPUT_ERA} \
                --channel ${CHANNEL} --masspoint ${MP} \
                --method ${METHOD} --binning ${BINNING}"
        run "bash scripts/runAsymptotic.sh \
                --era ${OUTPUT_ERA} --channel ${CHANNEL} --masspoint ${MP} \
                --method ${METHOD} --binning ${BINNING}"
    done

    echo ""
    echo "Done. Compare results with: bash scripts/test_dropRun3_compare.sh"
    exit 0
fi

# -----------------------------------------------------------------------------
# HTCondor DAGMan mode — one DAG per mass point
# -----------------------------------------------------------------------------
cd "$SCRIPT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_DIR="$CONDOR_DIR/jobs_dag_test_dropRun3_${TIMESTAMP}"
mkdir -p "$JOB_DIR"

# Share dagman.config
cp "$SCRIPT_DIR/configs/dagman.config" "$JOB_DIR/"

for MP in "${MASSPOINTS[@]}"; do
    MP_DIR="$JOB_DIR/$MP"
    mkdir -p "$MP_DIR/logs"

    # Per-mass-point jobs.sub (reuses the existing wrapper)
    cat > "$MP_DIR/jobs.sub" << EOF
JobBatchName = test_dropRun3_${MP}
universe = vanilla
executable = $SCRIPT_DIR/scripts/makeBinnedTemplates_wrapper.sh
arguments = \$(step) \$(era) \$(channel) \$(masspoint) \$(method) \$(binning) \$(extra_args)
output = logs/\$(step)_\$(channel)_\$(era).out
error  = logs/\$(step)_\$(channel)_\$(era).err
log    = dag.log
request_cpus = 1
request_memory = 2GB
request_disk = 2GB
getenv = True
should_transfer_files = NO
queue
EOF

    cp "$JOB_DIR/dagman.config" "$MP_DIR/"

    # Per-mass-point DAG: combine_era -> asymptotic
    # For combine_era: ERA=<csv input eras>, CHANNEL=<output era name>
    # For asymptotic : ERA=<output era name>, CHANNEL=Combined
    DAG="$MP_DIR/dag.dag"
    {
        echo "# DAG for ${MP}: drop 16postVFP/2022/2023BPix test"
        echo "CONFIG dagman.config"
        echo ""
        echo "JOB combine_era jobs.sub"
        echo "VARS combine_era step=\"combine_era\" era=\"${ERAS_CSV}\" channel=\"${OUTPUT_ERA}\" masspoint=\"${MP}\" method=\"${METHOD}\" binning=\"${BINNING}\" extra_args=\"\""
        echo ""
        echo "JOB asymptotic jobs.sub"
        echo "VARS asymptotic step=\"asymptotic\" era=\"${OUTPUT_ERA}\" channel=\"${CHANNEL}\" masspoint=\"${MP}\" method=\"${METHOD}\" binning=\"${BINNING}\" extra_args=\"\""
        echo ""
        echo "PARENT combine_era CHILD asymptotic"
    } > "$DAG"

    echo "Generated DAG: $DAG"
done

# submit_all.sh helper
cat > "$JOB_DIR/submit_all.sh" << 'EOF'
#!/bin/bash
set -e
for mp_dir in */; do
    if [[ -f "$mp_dir/dag.dag" ]]; then
        echo "Submitting DAG for ${mp_dir%/}..."
        (cd "$mp_dir" && condor_submit_dag dag.dag)
    fi
done
echo "All DAGs submitted."
EOF
chmod +x "$JOB_DIR/submit_all.sh"

# status_all.sh helper
cat > "$JOB_DIR/status_all.sh" << 'EOF'
#!/bin/bash
echo "DAG Status Summary"
echo "==================="
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
chmod +x "$JOB_DIR/status_all.sh"

echo ""
echo "========================================"
echo "Generated DAGMan workflows in: $JOB_DIR"
echo "Mass points: ${#MASSPOINTS[@]}"

if [[ "$DRY_RUN" == true ]]; then
    echo ""
    echo "[DRY-RUN] To submit: cd $JOB_DIR && ./submit_all.sh"
    echo "[DRY-RUN] To check : cd $JOB_DIR && ./status_all.sh"
else
    echo ""
    echo "Submitting all DAGs..."
    (cd "$JOB_DIR" && ./submit_all.sh)
    echo ""
    echo "Monitor with:"
    echo "  condor_q -dag"
    echo "  cd $JOB_DIR && ./status_all.sh"
fi

echo ""
echo "After completion, compare with: bash scripts/test_dropRun3_compare.sh"
