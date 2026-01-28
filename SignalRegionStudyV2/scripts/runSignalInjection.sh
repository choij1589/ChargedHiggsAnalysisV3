#!/bin/bash
#
# runSignalInjection.sh - Run B2G-compliant signal injection tests via HTCondor DAG
#
# Injects signal at expected limit r values (0, exp-1sigma, exp0, exp+1sigma)
# and verifies recovery using FitDiagnostics.
#
# B2G Requirements:
# - Uses --bypassFrequentistFit for data-like toy generation
# - Uses FitDiagnostics (not MultiDimFit) for fitting
# - Filters on fit_status == 0
# - Plots bias (r - r_inj) and pull distributions with Gaussian fits
#
# Usage:
#   ./runSignalInjection.sh --era All --channel Combined --masspoint MHc130_MA90 --method Baseline --binning extended --condor
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
            echo "  --era        Data-taking period (Run2, Run3, All, or single era)"
            echo "  --masspoint  Signal mass point (e.g., MHc130_MA90)"
            echo ""
            echo "Options:"
            echo "  --channel    Analysis channel [default: Combined]"
            echo "  --method     Template method (Baseline, ParticleNet) [default: Baseline]"
            echo "  --binning    Binning scheme (uniform, extended) [default: extended]"
            echo "  --ntoys      Total number of toys [default: 500]"
            echo "  --nbatches   Number of condor batches per r-value [default: 5]"
            echo "  --condor     Submit via HTCondor DAG"
            echo "  --plot-only  Only run collect+plot from existing results"
            echo "  --dry-run    Print commands without executing"
            echo "  --verbose    Enable verbose output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$ERA" || -z "$MASSPOINT" ]]; then
    echo "ERROR: --era and --masspoint are required"
    exit 1
fi

# Get WORKDIR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PYTHON_DIR="$(dirname "$SCRIPT_DIR")/python"

# Template directory
TEMPLATE_DIR="${WORKDIR}/SignalRegionStudyV2/templates/${ERA}/${CHANNEL}/${MASSPOINT}/${METHOD}/${BINNING}"

# Check if template directory exists
if [[ ! -d "$TEMPLATE_DIR" ]]; then
    echo "ERROR: Template directory not found: $TEMPLATE_DIR"
    exit 1
fi

# Output directory
OUTPUT_DIR="${TEMPLATE_DIR}/combine_output/injection"

# CMSSW path for Combine
CMSSW_BASE="${WORKDIR}/Common/CMSSW_14_1_0_pre4/src"

# Log function
log() {
    if [[ "$VERBOSE" == true ]]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    fi
}

# Run function (respects dry-run)
run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] $1"
    else
        log "Running: $1"
        eval "$1"
    fi
}

echo "============================================================"
echo "Signal Injection Test (B2G-compliant)"
echo "============================================================"
echo "  Era:       ${ERA}"
echo "  Channel:   ${CHANNEL}"
echo "  Masspoint: ${MASSPOINT}"
echo "  Method:    ${METHOD}"
echo "  Binning:   ${BINNING}"
echo "  Toys:      ${NTOYS} (${NBATCHES} batches)"
echo "  Condor:    ${CONDOR}"
echo "  Plot only: ${PLOT_ONLY}"
echo ""

# Calculate toys per batch
TOYS_PER_BATCH=$((NTOYS / NBATCHES))
if [[ $((TOYS_PER_BATCH * NBATCHES)) -ne $NTOYS ]]; then
    echo "WARNING: NTOYS=${NTOYS} not evenly divisible by NBATCHES=${NBATCHES}"
    echo "         Using ${TOYS_PER_BATCH} toys/batch (total: $((TOYS_PER_BATCH * NBATCHES)))"
fi

# ============================================================
# Plot-only mode: just run extraction and plotting
# ============================================================
if [[ "$PLOT_ONLY" == true ]]; then
    echo "Running in plot-only mode..."
    if [[ ! -d "$OUTPUT_DIR" ]]; then
        echo "ERROR: No injection output directory found: $OUTPUT_DIR"
        exit 1
    fi

    cd "$TEMPLATE_DIR"

    echo "Extracting results..."
    python3 "${PYTHON_DIR}/extractInjectionResults.py" \
        --input-dir "${OUTPUT_DIR}" \
        --output "${OUTPUT_DIR}/injection_results.json"

    echo ""
    echo "Creating bias test plot..."
    python3 "${PYTHON_DIR}/plotBiasTest.py" \
        --input "${OUTPUT_DIR}/injection_results.json" \
        --output "${OUTPUT_DIR}/bias_test.pdf" \
        --plot-type bias \
        --masspoint "${MASSPOINT}" --era "${ERA}" --method "${METHOD}"

    echo ""
    echo "Creating pull distribution plot..."
    python3 "${PYTHON_DIR}/plotBiasTest.py" \
        --input "${OUTPUT_DIR}/injection_results.json" \
        --output "${OUTPUT_DIR}/pull_dist.pdf" \
        --plot-type pull \
        --masspoint "${MASSPOINT}" --era "${ERA}" --method "${METHOD}"

    echo ""
    echo "Done. Results in ${OUTPUT_DIR}/"
    exit 0
fi

# ============================================================
# Pre-DAG steps: run locally
# ============================================================

cd "$TEMPLATE_DIR"
log "Working directory: $(pwd)"

# Step 1: Locate asymptotic output and extract expected limits
echo "===== Step 1: Extracting expected limits from asymptotic output ====="

ASYM_FILE=$(find "${TEMPLATE_DIR}/combine_output/asymptotic/" \
    -name "higgsCombine*.AsymptoticLimits.mH120.root" 2>/dev/null | head -1)

if [[ -z "$ASYM_FILE" || ! -f "$ASYM_FILE" ]]; then
    echo "ERROR: Asymptotic limits file not found in:"
    echo "  ${TEMPLATE_DIR}/combine_output/asymptotic/"
    echo "Run asymptotic limits first before signal injection."
    exit 1
fi

echo "  Using: $ASYM_FILE"

if [[ "$DRY_RUN" == false ]]; then
    EXPECTED_LIMITS=$(root -l -b -q -e "
        TFile *f = TFile::Open(\"${ASYM_FILE}\");
        TTree *t = (TTree*)f->Get(\"limit\");
        double r; t->SetBranchAddress(\"limit\", &r);
        t->GetEntry(1); double exp_m1 = r;
        t->GetEntry(2); double exp_0 = r;
        t->GetEntry(3); double exp_p1 = r;
        printf(\"%.6f,%.6f,%.6f\", exp_m1, exp_0, exp_p1);
        f->Close();
    " 2>/dev/null | tail -1)

    EXP_M1=$(echo "$EXPECTED_LIMITS" | cut -d',' -f1)
    EXP_0=$(echo "$EXPECTED_LIMITS" | cut -d',' -f2)
    EXP_P1=$(echo "$EXPECTED_LIMITS" | cut -d',' -f3)

    # Validate extracted values are non-empty numbers
    for val in "$EXP_M1" "$EXP_0" "$EXP_P1"; do
        if [[ -z "$val" ]] || ! [[ "$val" =~ ^[0-9]+\.[0-9]+$ ]]; then
            echo "ERROR: Failed to extract expected limits from asymptotic file"
            echo "  Raw output: '${EXPECTED_LIMITS}'"
            exit 1
        fi
    done

    echo "  Expected limits:"
    echo "    exp-1sigma: r < ${EXP_M1}"
    echo "    exp median: r < ${EXP_0}"
    echo "    exp+1sigma: r < ${EXP_P1}"

    R_VALUES=(0 "${EXP_M1}" "${EXP_0}" "${EXP_P1}")
    R_LABELS=("r0" "rM1" "rMed" "rP1")
else
    R_VALUES=(0 0.5 1.0 1.5)
    R_LABELS=("r0" "rM1" "rMed" "rP1")
    echo "  [DRY-RUN] Using dummy r values: ${R_VALUES[*]}"
fi

echo ""
echo "Injection r values: ${R_VALUES[*]}"
echo ""

# Step 2: Create workspace
echo "===== Step 2: Creating workspace ====="
if [[ ! -f "workspace.root" ]] || [[ "datacard.txt" -nt "workspace.root" ]]; then
    run_cmd "text2workspace.py datacard.txt -o workspace.root"
else
    echo "  workspace.root is up-to-date"
fi

# ============================================================
# Local execution (no --condor)
# ============================================================
if [[ "$CONDOR" == false ]]; then
    echo ""
    echo "===== Running locally ====="

    # Cleanup previous results
    rm -rf "$OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"

    for idx in "${!R_VALUES[@]}"; do
        R="${R_VALUES[$idx]}"
        LABEL="${R_LABELS[$idx]}"

        echo ""
        echo "--- Injection r=${R} (${LABEL}) ---"

        for SEED in $(seq 1 "$NBATCHES"); do
            echo "  Batch ${SEED}/${NBATCHES}: Generating ${TOYS_PER_BATCH} toys..."

            run_cmd "combine -M GenerateOnly workspace.root \
                -t ${TOYS_PER_BATCH} \
                --expectSignal ${R} \
                --saveToys \
                --bypassFrequentistFit \
                -n .inject_${LABEL}_s${SEED} \
                -m 120 -s ${SEED}"

            TOY_FILE="higgsCombine.inject_${LABEL}_s${SEED}.GenerateOnly.mH120.${SEED}.root"
            if [[ "$DRY_RUN" == false && ! -f "$TOY_FILE" ]]; then
                echo "  ERROR: Toy file not generated: $TOY_FILE"
                continue
            fi

            echo "  Batch ${SEED}/${NBATCHES}: Running FitDiagnostics..."

            run_cmd "combine -M FitDiagnostics workspace.root \
                -t ${TOYS_PER_BATCH} \
                --toysFile ${TOY_FILE} \
                --rMin -5 --rMax 5 \
                -n .recovery_${LABEL}_s${SEED} \
                -m 120"

            # Move output to injection directory
            if [[ "$DRY_RUN" == false ]]; then
                mv -f "fitDiagnostics.recovery_${LABEL}_s${SEED}.root" "${OUTPUT_DIR}/" 2>/dev/null || true
                rm -f "$TOY_FILE"
                rm -f "higgsCombine.recovery_${LABEL}_s${SEED}.FitDiagnostics.mH120."*.root 2>/dev/null || true
            fi
        done
    done

    # Extract results and plot
    echo ""
    echo "===== Extracting results ====="
    if [[ "$DRY_RUN" == false ]]; then
        python3 "${PYTHON_DIR}/extractInjectionResults.py" \
            --input-dir "${OUTPUT_DIR}" \
            --output "${OUTPUT_DIR}/injection_results.json"

        echo ""
        echo "Creating bias test plot..."
        python3 "${PYTHON_DIR}/plotBiasTest.py" \
            --input "${OUTPUT_DIR}/injection_results.json" \
            --output "${OUTPUT_DIR}/bias_test.pdf" \
            --plot-type bias \
            --masspoint "${MASSPOINT}" --era "${ERA}" --method "${METHOD}"

        echo ""
        echo "Creating pull distribution plot..."
        python3 "${PYTHON_DIR}/plotBiasTest.py" \
            --input "${OUTPUT_DIR}/injection_results.json" \
            --output "${OUTPUT_DIR}/pull_dist.pdf" \
            --plot-type pull \
            --masspoint "${MASSPOINT}" --era "${ERA}" --method "${METHOD}"
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

# Cleanup and create condor directory
rm -rf "$OUTPUT_DIR"
mkdir -p "${OUTPUT_DIR}/condor/logs"

CONDOR_DIR="${OUTPUT_DIR}/condor"

# Save r values to file for collect_plot job
printf "%s\n" "${R_VALUES[@]}" > "${CONDOR_DIR}/r_values.txt"
printf "%s\n" "${R_LABELS[@]}" > "${CONDOR_DIR}/r_labels.txt"

# Copy workspace and python scripts to condor directory
if [[ "$DRY_RUN" == false ]]; then
    cp workspace.root "${CONDOR_DIR}/"
    cp "${PYTHON_DIR}/extractInjectionResults.py" "${CONDOR_DIR}/"
    cp "${PYTHON_DIR}/plotBiasTest.py" "${CONDOR_DIR}/"
else
    echo "[DRY-RUN] Would copy workspace.root and python scripts to ${CONDOR_DIR}/"
fi

# ========== Generate inject job scripts and .sub files ==========
echo "Generating inject job scripts..."

INJECT_JOB_NAMES=()

for idx in "${!R_VALUES[@]}"; do
    R="${R_VALUES[$idx]}"
    LABEL="${R_LABELS[$idx]}"

    for SEED in $(seq 1 "$NBATCHES"); do
        JOB_NAME="inject_${LABEL}_s${SEED}"
        INJECT_JOB_NAMES+=("$JOB_NAME")

        # Job script
        cat > "${CONDOR_DIR}/${JOB_NAME}.sh" << EOFINJECT
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}

echo "Signal injection: r=${R}, seed=${SEED}, toys=${TOYS_PER_BATCH}"

# Phase 1: Generate toys
combine -M GenerateOnly workspace.root \\
    -t ${TOYS_PER_BATCH} \\
    --expectSignal ${R} \\
    --saveToys \\
    --bypassFrequentistFit \\
    -n .inject_${LABEL}_s${SEED} \\
    -m 120 -s ${SEED}

# Phase 2: FitDiagnostics
combine -M FitDiagnostics workspace.root \\
    -t ${TOYS_PER_BATCH} \\
    --toysFile higgsCombine.inject_${LABEL}_s${SEED}.GenerateOnly.mH120.${SEED}.root \\
    --rMin -5 --rMax 5 \\
    -n .recovery_${LABEL}_s${SEED} \\
    -m 120

echo "Done. Output:"
ls -la fitDiagnostics.*.root 2>/dev/null || echo "No fitDiagnostics output!"
EOFINJECT
        chmod +x "${CONDOR_DIR}/${JOB_NAME}.sh"

        # Submit file
        cat > "${CONDOR_DIR}/${JOB_NAME}.sub" << EOFSUB
universe = vanilla
executable = ${JOB_NAME}.sh
output = logs/${JOB_NAME}.out
error = logs/${JOB_NAME}.err
log = injection.log

request_cpus = 1
request_memory = 2GB
request_disk = 1GB

should_transfer_files = YES
transfer_input_files = workspace.root
transfer_output_files = fitDiagnostics.recovery_${LABEL}_s${SEED}.root
when_to_transfer_output = ON_EXIT

queue
EOFSUB
    done
done

NUM_INJECT=${#INJECT_JOB_NAMES[@]}
echo "  Created ${NUM_INJECT} inject jobs (${#R_VALUES[@]} r-values x ${NBATCHES} seeds)"

# ========== Generate collect_plot job ==========
echo "Generating collect_plot job..."

# Build list of fitDiagnostics files that will be transferred in
FITDIAG_FILES=""
for idx in "${!R_VALUES[@]}"; do
    LABEL="${R_LABELS[$idx]}"
    for SEED in $(seq 1 "$NBATCHES"); do
        if [[ -n "$FITDIAG_FILES" ]]; then
            FITDIAG_FILES="${FITDIAG_FILES},"
        fi
        FITDIAG_FILES="${FITDIAG_FILES}fitDiagnostics.recovery_${LABEL}_s${SEED}.root"
    done
done

cat > "${CONDOR_DIR}/collect_plot.sh" << EOFCOLLECT
#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd ${CMSSW_BASE}
eval \$(scramv1 runtime -sh)
cd \${_CONDOR_SCRATCH_DIR}

echo "Collecting injection results..."
ls -la fitDiagnostics.*.root

# Extract results
python3 extractInjectionResults.py \\
    --input-dir . \\
    --output injection_results.json

# Plot bias distribution
python3 plotBiasTest.py \\
    --input injection_results.json \\
    --output bias_test.pdf \\
    --plot-type bias \\
    --masspoint ${MASSPOINT} --era ${ERA} --method ${METHOD}

# Plot pull distribution
python3 plotBiasTest.py \\
    --input injection_results.json \\
    --output pull_dist.pdf \\
    --plot-type pull \\
    --masspoint ${MASSPOINT} --era ${ERA} --method ${METHOD}

echo "Done. Output files:"
ls -la injection_results.json bias_test.pdf pull_dist.pdf
EOFCOLLECT
chmod +x "${CONDOR_DIR}/collect_plot.sh"

cat > "${CONDOR_DIR}/collect_plot.sub" << EOFSUB
universe = vanilla
executable = collect_plot.sh
output = logs/collect_plot.out
error = logs/collect_plot.err
log = injection.log

request_cpus = 1
request_memory = 2GB
request_disk = 1GB

should_transfer_files = YES
transfer_input_files = workspace.root,extractInjectionResults.py,plotBiasTest.py,r_values.txt,r_labels.txt,${FITDIAG_FILES}
transfer_output_files = injection_results.json,bias_test.pdf,pull_dist.pdf
when_to_transfer_output = ON_EXIT

queue
EOFSUB

# ========== Generate DAG file ==========
echo "Generating DAG file..."

DAG_FILE="${CONDOR_DIR}/injection.dag"
{
    echo "# Signal Injection DAG workflow"
    echo "# ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING})"
    echo "# Total jobs: ${NUM_INJECT} inject + 1 collect_plot = $((NUM_INJECT + 1))"
    echo "# Toys: ${NTOYS} total (${NBATCHES} batches x ${TOYS_PER_BATCH} toys)"
    echo ""

    # Inject jobs (all independent)
    echo "# Inject jobs (${NUM_INJECT} total)"
    for JOB_NAME in "${INJECT_JOB_NAMES[@]}"; do
        echo "JOB ${JOB_NAME} ${JOB_NAME}.sub"
    done
    echo ""

    # Collect + plot job
    echo "# Collect and plot job"
    echo "JOB collect_plot collect_plot.sub"
    echo ""

    # Dependencies: all inject -> collect_plot
    echo "# Dependencies"
    echo "PARENT ${INJECT_JOB_NAMES[*]} CHILD collect_plot"

} > "$DAG_FILE"

# DAGMan configuration
cat > "${CONDOR_DIR}/dagman.config" << EOF
DAGMAN_MAX_JOBS_SUBMITTED = 50
DAGMAN_MAX_JOBS_IDLE = 25
EOF

echo ""
echo "=========================================="
echo "DAG workflow prepared in:"
echo "  ${CONDOR_DIR}"
echo ""
echo "Jobs created:"
echo "  ${NUM_INJECT} inject jobs (${#R_VALUES[@]} r-values x ${NBATCHES} seeds)"
echo "  1 collect_plot job"
echo "  Total: $((NUM_INJECT + 1)) jobs"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo "[DRY-RUN] Would submit: condor_submit_dag injection.dag"
    echo "=========================================="
    exit 0
fi

# Submit DAG
echo "Submitting DAG workflow..."
cd "${CONDOR_DIR}"
condor_submit_dag -config dagman.config injection.dag
echo ""
echo "Monitor with: condor_q -dag"
echo "DAG log: ${CONDOR_DIR}/injection.dag.dagman.out"
echo ""
echo "After completion, results will be in:"
echo "  ${CONDOR_DIR}/injection_results.json"
echo "  ${CONDOR_DIR}/bias_test.pdf"
echo "  ${CONDOR_DIR}/pull_dist.pdf"
echo "=========================================="
