#!/bin/bash
#
# runGoF.sh - Run Goodness-of-Fit test via HTCondor DAG
#
# Measures GoF using the saturated model algorithm on data (or Asimov when blinded),
# generates toys for p-value calculation, then collects and plots results.
#
# B2G Requirements:
# - Uses saturated model (--algo=saturated)
# - Signal frozen to 0 (--freezeParameters r --setParameters r=0)
# - Uses --toysFrequentist to randomise NPs in toys
# - Uses --bypassFrequentistFit when blinded (no real data in SR)
# - Produces gof.json + gof_plot.{pdf,png} via combineTool + plotGof.py
#
# Usage:
#   ./runGoF.sh --era All --channel Combined --masspoint MHc130_MA90 --method ParticleNet \
#               --binning extended --partial-unblind
#   ./runGoF.sh --era All --channel Combined --masspoint MHc130_MA55 --method Baseline \
#               --binning extended
#

set -euo pipefail

# Default values
ERA=""
CHANNEL="Combined"
MASSPOINT=""
METHOD="Baseline"
BINNING="extended"
NTOYS=500
NBATCHES=5
PARTIAL_UNBLIND=false
UNBLIND=false
CONDOR=false
PLOT_ONLY=false
DRY_RUN=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --era)
            ERA="$2"
            shift 2
            ;;
        --channel)
            CHANNEL="$2"
            shift 2
            ;;
        --masspoint)
            MASSPOINT="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --binning)
            BINNING="$2"
            shift 2
            ;;
        --ntoys)
            NTOYS="$2"
            shift 2
            ;;
        --nbatches)
            NBATCHES="$2"
            shift 2
            ;;
        --partial-unblind)
            PARTIAL_UNBLIND=true
            shift
            ;;
        --unblind)
            UNBLIND=true
            shift
            ;;
        --condor)
            CONDOR=true
            shift
            ;;
        --plot-only)
            PLOT_ONLY=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --era ERA --masspoint MASSPOINT [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --era ERA            Data-taking period (Run2, Run3, All, 2018, ...)"
            echo "  --masspoint MP       Signal mass point (e.g., MHc130_MA90)"
            echo ""
            echo "Options:"
            echo "  --channel CHANNEL    Analysis channel [default: Combined]"
            echo "  --method METHOD      Baseline or ParticleNet [default: Baseline]"
            echo "  --binning BINNING    extended or uniform [default: extended]"
            echo "  --ntoys N            Total toys for p-value [default: 500]"
            echo "  --nbatches N         HTCondor batches for toys [default: 5]"
            echo "  --partial-unblind    Use partial-unblind templates (real data, score < 0.3)"
            echo "  --unblind            Use fully unblinded templates"
            echo "  --condor             Submit via HTCondor DAG"
            echo "  --plot-only          Only collect+plot from existing outputs"
            echo "  --dry-run            Print commands without executing"
            echo "  --verbose            Enable verbose logging"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate
if [[ -z "$ERA" || -z "$MASSPOINT" ]]; then
    echo "ERROR: --era and --masspoint are required"
    exit 1
fi
if [[ "$UNBLIND" == true && "$PARTIAL_UNBLIND" == true ]]; then
    echo "ERROR: --unblind and --partial-unblind are mutually exclusive"
    exit 1
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Resolve template directory (same logic as runFitDiagnostics.sh)
BINNING_SUFFIX="${BINNING}"
if [[ "$UNBLIND" == true ]]; then
    BINNING_SUFFIX="${BINNING}_unblind"
elif [[ "$PARTIAL_UNBLIND" == true ]]; then
    BINNING_SUFFIX="${BINNING}_partial_unblind"
fi
TEMPLATE_DIR="${WORKDIR}/SignalRegionStudyV2/templates/${ERA}/${CHANNEL}/${MASSPOINT}/${METHOD}/${BINNING_SUFFIX}"

if [[ ! -d "$TEMPLATE_DIR" ]]; then
    echo "ERROR: Template directory not found: $TEMPLATE_DIR"
    exit 1
fi
if [[ ! -f "$TEMPLATE_DIR/datacard.txt" ]]; then
    echo "ERROR: Datacard not found: $TEMPLATE_DIR/datacard.txt"
    exit 1
fi

OUTPUT_DIR="${TEMPLATE_DIR}/combine_output/gof"
CMSSW_BASE="${WORKDIR}/Common/CMSSW_14_1_0_pre4/src"
TOYS_PER_BATCH=$((NTOYS / NBATCHES))

# Toy generation options:
# - partial-unblind/unblind: real data present → --toysFrequentist only (fit data first)
# - blinded (Asimov): no real data → --toysFrequentist --bypassFrequentistFit
if [[ "$PARTIAL_UNBLIND" == true || "$UNBLIND" == true ]]; then
    TOY_OPTS="--toysFrequentist"
    DATA_OPTS=""
else
    TOY_OPTS="--toysFrequentist --bypassFrequentistFit"
    DATA_OPTS="-t -1"  # Asimov data
fi
FREEZE_OPTS="--freezeParameters r --setParameters r=0"

log() { [[ "$VERBOSE" == true ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" || true; }
run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then echo "[DRY-RUN] $1"; else log "Running: $1"; eval "$1"; fi
}

echo "============================================================"
echo "Goodness-of-Fit Test (B2G-compliant, saturated model)"
echo "============================================================"
echo "  Era:       ${ERA}"
echo "  Channel:   ${CHANNEL}"
echo "  Masspoint: ${MASSPOINT}"
echo "  Method:    ${METHOD}"
echo "  Binning:   ${BINNING_SUFFIX}"
echo "  Toys:      ${NTOYS} (${NBATCHES} batches x ${TOYS_PER_BATCH})"
echo "  Mode:      $([ "$PARTIAL_UNBLIND" == true ] && echo partial-unblind || ([ "$UNBLIND" == true ] && echo unblind || echo blinded/Asimov))"
echo "  Condor:    ${CONDOR}"
echo ""

# ============================================================
# Plot-only mode
# ============================================================
if [[ "$PLOT_ONLY" == true ]]; then
    echo "Running in plot-only mode..."
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        echo "ERROR: No GoF output directory: $OUTPUT_DIR"
        exit 1
    fi
    cd "$OUTPUT_DIR"

    DATA_FILE=$(ls higgsCombine.gof_data.GoodnessOfFit.mH120.root 2>/dev/null | head -1)
    TOY_FILES=$(ls higgsCombine.gof_toys_s*.GoodnessOfFit.mH120.*.root 2>/dev/null | tr '\n' ' ')

    if [[ -z "$DATA_FILE" || -z "$TOY_FILES" ]]; then
        echo "ERROR: Missing data or toy GoF files in ${OUTPUT_DIR}"
        exit 1
    fi

    run_cmd "combineTool.py -M CollectGoodnessOfFit \
        --input ${DATA_FILE} ${TOY_FILES} \
        -o gof.json"
    run_cmd "plotGof.py gof.json \
        --statistic saturated --mass 120.0 \
        -o gof_plot \
        --title-right=\"${ERA} ${MASSPOINT} ${METHOD}\""
    echo "Done. Results in ${OUTPUT_DIR}/"
    exit 0
fi

# ============================================================
# Pre-DAG steps: create workspace
# ============================================================
cd "$TEMPLATE_DIR"
log "Working directory: $(pwd)"

echo "===== Creating workspace ====="
if [[ ! -f "workspace.root" ]] || [[ "datacard.txt" -nt "workspace.root" ]]; then
    run_cmd "text2workspace.py datacard.txt -o workspace.root"
else
    echo "  workspace.root is up-to-date"
fi

# ============================================================
# Local execution
# ============================================================
if [[ "$CONDOR" == false ]]; then
    mkdir -p "$OUTPUT_DIR"

    echo ""
    echo "===== Step 1: GoF on data/Asimov ====="
    run_cmd "combine -M GoodnessOfFit workspace.root \
        --algo=saturated \
        ${FREEZE_OPTS} \
        ${DATA_OPTS} \
        -n .gof_data \
        -m 120 2>&1 | tee ${OUTPUT_DIR}/gof_data.log"

    if [[ "$DRY_RUN" == false ]]; then
        mv -f higgsCombine.gof_data.GoodnessOfFit.mH120.root "$OUTPUT_DIR/" 2>/dev/null || true
    fi

    echo ""
    echo "===== Step 2: GoF on toys ====="
    for SEED in $(seq 1 "$NBATCHES"); do
        run_cmd "combine -M GoodnessOfFit workspace.root \
            --algo=saturated \
            ${FREEZE_OPTS} \
            ${TOY_OPTS} \
            -t ${TOYS_PER_BATCH} \
            -n .gof_toys_s${SEED} \
            -m 120 -s ${SEED} 2>&1 | tee ${OUTPUT_DIR}/gof_toys_s${SEED}.log"
        if [[ "$DRY_RUN" == false ]]; then
            mv -f "higgsCombine.gof_toys_s${SEED}.GoodnessOfFit.mH120.${SEED}.root" \
                "$OUTPUT_DIR/" 2>/dev/null || true
        fi
    done

    echo ""
    echo "===== Step 3: Collect + plot ====="
    if [[ "$DRY_RUN" == false ]]; then
        cd "$OUTPUT_DIR"
        DATA_FILE="higgsCombine.gof_data.GoodnessOfFit.mH120.root"
        TOY_FILES=$(ls higgsCombine.gof_toys_s*.GoodnessOfFit.mH120.*.root 2>/dev/null | tr '\n' ' ')

        combineTool.py -M CollectGoodnessOfFit \
            --input "${DATA_FILE}" ${TOY_FILES} \
            -o gof.json

        plotGof.py gof.json \
            --statistic saturated --mass 120.0 \
            -o gof_plot \
            --title-right="${ERA} ${MASSPOINT} ${METHOD}"
    else
        run_cmd "combineTool.py -M CollectGoodnessOfFit --input [data+toy files] -o gof.json"
        run_cmd "plotGof.py gof.json --statistic saturated --mass 120.0 -o gof_plot"
    fi

    echo ""
    echo "============================================================"
    echo "Results saved to ${OUTPUT_DIR}/"
    echo "============================================================"
    exit 0
fi

# ============================================================
# HTCondor DAG workflow
# ============================================================
echo ""
echo "===== Preparing HTCondor DAG workflow ====="

rm -rf "$OUTPUT_DIR" || true
mkdir -p "${OUTPUT_DIR}/logs"
CONDOR_DIR="$OUTPUT_DIR"

if [[ "$DRY_RUN" == false ]]; then
    cp workspace.root "${CONDOR_DIR}/"
else
    echo "[DRY-RUN] Would copy workspace.root to ${CONDOR_DIR}/"
fi

# ===== Data GoF job =====
cat > "${CONDOR_DIR}/data_gof.sh" << EOFDATA
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}

echo "GoF on data/Asimov: ${MASSPOINT} ${ERA}"
combine -M GoodnessOfFit workspace.root \\
    --algo=saturated \\
    ${FREEZE_OPTS} \\
    ${DATA_OPTS} \\
    -n .gof_data \\
    -m 120
echo "Done."
ls -la higgsCombine.gof_data.GoodnessOfFit.mH120.root
EOFDATA
chmod +x "${CONDOR_DIR}/data_gof.sh"

cat > "${CONDOR_DIR}/data_gof.sub" << EOFSUB
universe = vanilla
executable = data_gof.sh
output = logs/data_gof.out
error  = logs/data_gof.err
log    = gof.log

request_cpus   = 1
request_memory = 2GB
request_disk   = 1GB

should_transfer_files   = YES
transfer_input_files    = workspace.root
transfer_output_files   = higgsCombine.gof_data.GoodnessOfFit.mH120.root
when_to_transfer_output = ON_EXIT

queue
EOFSUB

# ===== Toy GoF jobs =====
TOY_OUTPUT_FILES=""
for SEED in $(seq 1 "$NBATCHES"); do
    TOY_FILE="higgsCombine.gof_toys_s${SEED}.GoodnessOfFit.mH120.${SEED}.root"
    [[ -n "$TOY_OUTPUT_FILES" ]] && TOY_OUTPUT_FILES="${TOY_OUTPUT_FILES},"
    TOY_OUTPUT_FILES="${TOY_OUTPUT_FILES}${TOY_FILE}"

    cat > "${CONDOR_DIR}/toys_gof_s${SEED}.sh" << EOFTOY
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}

echo "GoF toys: seed=${SEED}, n=${TOYS_PER_BATCH}"
combine -M GoodnessOfFit workspace.root \\
    --algo=saturated \\
    ${FREEZE_OPTS} \\
    ${TOY_OPTS} \\
    -t ${TOYS_PER_BATCH} \\
    -n .gof_toys_s${SEED} \\
    -m 120 -s ${SEED}
echo "Done."
ls -la ${TOY_FILE}
EOFTOY
    chmod +x "${CONDOR_DIR}/toys_gof_s${SEED}.sh"

    cat > "${CONDOR_DIR}/toys_gof_s${SEED}.sub" << EOFSUB
universe = vanilla
executable = toys_gof_s${SEED}.sh
output = logs/toys_gof_s${SEED}.out
error  = logs/toys_gof_s${SEED}.err
log    = gof.log

request_cpus   = 1
request_memory = 2GB
request_disk   = 1GB

should_transfer_files   = YES
transfer_input_files    = workspace.root
transfer_output_files   = ${TOY_FILE}
when_to_transfer_output = ON_EXIT

queue
EOFSUB
done

# ===== Collect + plot job =====
ALL_INPUT_FILES="higgsCombine.gof_data.GoodnessOfFit.mH120.root,${TOY_OUTPUT_FILES}"

cat > "${CONDOR_DIR}/collect_plot.sh" << EOFCOLLECT
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}

echo "Collecting GoF results..."
ls -la higgsCombine.*.GoodnessOfFit.mH120*.root

DATA_FILE="higgsCombine.gof_data.GoodnessOfFit.mH120.root"
TOY_FILES=\$(ls higgsCombine.gof_toys_s*.GoodnessOfFit.mH120.*.root 2>/dev/null | tr '\\n' ' ')

combineTool.py -M CollectGoodnessOfFit \\
    --input \${DATA_FILE} \${TOY_FILES} \\
    -o gof.json

plotGof.py gof.json \\
    --statistic saturated --mass 120.0 \\
    -o gof_plot \\
    --title-right="${ERA} ${MASSPOINT} ${METHOD}"

echo "Done."
ls -la gof.json gof_plot.*
EOFCOLLECT
chmod +x "${CONDOR_DIR}/collect_plot.sh"

cat > "${CONDOR_DIR}/collect_plot.sub" << EOFSUB
universe = vanilla
executable = collect_plot.sh
output = logs/collect_plot.out
error  = logs/collect_plot.err
log    = gof.log

request_cpus   = 1
request_memory = 2GB
request_disk   = 1GB

should_transfer_files   = YES
transfer_input_files    = ${ALL_INPUT_FILES}
transfer_output_files   = gof.json,gof_plot.pdf,gof_plot.png
when_to_transfer_output = ON_EXIT

queue
EOFSUB

# ===== DAG file =====
DAG_FILE="${CONDOR_DIR}/gof.dag"
{
    echo "# Goodness-of-Fit DAG: ${ERA} ${CHANNEL} ${MASSPOINT} ${METHOD} ${BINNING_SUFFIX}"
    echo ""
    echo "JOB data_gof   data_gof.sub"
    for SEED in $(seq 1 "$NBATCHES"); do
        echo "JOB toys_gof_s${SEED}   toys_gof_s${SEED}.sub"
    done
    echo "JOB collect_plot   collect_plot.sub"
    echo ""
    # Dependencies: data_gof + all toy jobs → collect_plot
    TOY_JOB_NAMES=$(seq 1 "$NBATCHES" | xargs -I{} echo -n "toys_gof_s{} ")
    echo "PARENT data_gof ${TOY_JOB_NAMES}CHILD collect_plot"
} > "$DAG_FILE"

CONFIG_FILE="${SCRIPT_DIR}/../configs/dagman.config"
echo ""
echo "Submitting DAG: ${DAG_FILE}"
if [[ "$DRY_RUN" == false ]]; then
    cd "$CONDOR_DIR"
    condor_submit_dag -config "${CONFIG_FILE}" gof.dag
else
    echo "[DRY-RUN] condor_submit_dag -config ${CONFIG_FILE} ${DAG_FILE}"
fi

echo ""
echo "============================================================"
echo "GoF DAG submitted. Monitor with: condor_q -dag"
echo "Results will be in: ${OUTPUT_DIR}/"
echo "============================================================"
