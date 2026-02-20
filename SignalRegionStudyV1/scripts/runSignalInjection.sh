#!/bin/bash
#
# runSignalInjection.sh - Run B2G-compliant signal injection tests
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
#   ./runSignalInjection.sh --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --method Baseline --binning uniform
#

set -e

# Default values
ERA=""
CHANNEL=""
MASSPOINT=""
METHOD="Baseline"
BINNING="uniform"
NTOYS=500
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
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --era ERA --channel CHANNEL --masspoint MASSPOINT [OPTIONS]"
            echo ""
            echo "Required:"
            echo "  --era        Data-taking period"
            echo "  --channel    Analysis channel (SR1E2Mu, SR3Mu, Combined)"
            echo "  --masspoint  Signal mass point (e.g., MHc130_MA90)"
            echo ""
            echo "Options:"
            echo "  --method     Template method (Baseline, ParticleNet) [default: Baseline]"
            echo "  --binning    Binning scheme (uniform, extended) [default: uniform]"
            echo "  --ntoys      Number of toys per injection [default: 500]"
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
if [[ -z "$ERA" || -z "$CHANNEL" || -z "$MASSPOINT" ]]; then
    echo "ERROR: --era, --channel, and --masspoint are required"
    exit 1
fi

# Get WORKDIR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
PYTHON_DIR="$(dirname "$SCRIPT_DIR")/python"

# Template directory
TEMPLATE_DIR="${WORKDIR}/SignalRegionStudyV1/templates/${ERA}/${CHANNEL}/${MASSPOINT}/${METHOD}/${BINNING}"

# Check if template directory exists
if [[ ! -d "$TEMPLATE_DIR" ]]; then
    echo "ERROR: Template directory not found: $TEMPLATE_DIR"
    exit 1
fi

# Create output directory
OUTPUT_DIR="${TEMPLATE_DIR}/combine_output/injection"
mkdir -p "$OUTPUT_DIR"

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

# Change to template directory
cd "$TEMPLATE_DIR"
log "Working directory: $(pwd)"

echo "============================================================"
echo "Signal Injection Test (B2G-compliant)"
echo "============================================================"
echo "  Era:       ${ERA}"
echo "  Channel:   ${CHANNEL}"
echo "  Masspoint: ${MASSPOINT}"
echo "  Method:    ${METHOD}"
echo "  Binning:   ${BINNING}"
echo "  Toys:      ${NTOYS}"
echo ""

# Create workspace if needed
if [[ ! -f "workspace.root" ]] || [[ "datacard.txt" -nt "workspace.root" ]]; then
    echo "Creating workspace..."
    run_cmd "text2workspace.py datacard.txt -o workspace.root"
fi

# Cleanup previous injection results
if [[ -d "$OUTPUT_DIR" ]]; then
    echo "Cleaning up previous injection results..."
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

# ===== Phase 0: Get expected limits =====
echo ""
echo "===== Phase 0: Getting expected limits ====="

LIMIT_FILE="${OUTPUT_DIR}/higgsCombine.forInjection.AsymptoticLimits.mH120.root"
run_cmd "combine -M AsymptoticLimits workspace.root -m 120 -n .forInjection 2>&1 | tee ${OUTPUT_DIR}/asymptotic.out"

if [[ "$DRY_RUN" == false ]]; then
    mv -f higgsCombine.forInjection.AsymptoticLimits.mH120.root "${OUTPUT_DIR}/" 2>/dev/null || true

    # Extract exp-1sigma, exp0 (median), exp+1sigma r values
    EXPECTED_LIMITS=$(root -l -b -q -e "
        TFile *f = TFile::Open(\"${LIMIT_FILE}\");
        TTree *t = (TTree*)f->Get(\"limit\");
        double r; t->SetBranchAddress(\"limit\", &r);
        t->GetEntry(1); double exp_m1 = r;  // exp-1sigma (index 1)
        t->GetEntry(2); double exp_0 = r;   // exp median (index 2)
        t->GetEntry(3); double exp_p1 = r;  // exp+1sigma (index 3)
        printf(\"%.4f,%.4f,%.4f\", exp_m1, exp_0, exp_p1);
        f->Close();
    " 2>/dev/null | tail -1)

    EXP_M1=$(echo $EXPECTED_LIMITS | cut -d',' -f1)
    EXP_0=$(echo $EXPECTED_LIMITS | cut -d',' -f2)
    EXP_P1=$(echo $EXPECTED_LIMITS | cut -d',' -f3)

    echo "  Expected limits extracted:"
    echo "    exp-1sigma: r < ${EXP_M1}"
    echo "    exp median: r < ${EXP_0}"
    echo "    exp+1sigma: r < ${EXP_P1}"

    # Build injection r values: 0, exp-1, exp0, exp+1
    R_VALUES=(0 ${EXP_M1} ${EXP_0} ${EXP_P1})
else
    # Dummy values for dry-run
    R_VALUES=(0 0.5 1.0 1.5)
fi

echo ""
echo "Injection r values: ${R_VALUES[*]}"
echo ""

# ===== Phase 1: Generate toys for all r values =====
echo "===== Phase 1: Generating toys (with --bypassFrequentistFit) ====="
for R in "${R_VALUES[@]}"; do
    echo "  Generating toys with r=${R} signal..."

    R_OUTPUT_DIR="${OUTPUT_DIR}/r${R}"
    mkdir -p "$R_OUTPUT_DIR"

    run_cmd "combine -M GenerateOnly workspace.root \
        -t ${NTOYS} \
        --expectSignal ${R} \
        --saveToys \
        --bypassFrequentistFit \
        -n .inject_r${R} \
        -m 120 \
        -s 12345 \
        2>&1 | tee ${R_OUTPUT_DIR}/generate.out"

    # Move toy file to output directory
    TOY_FILE="higgsCombine.inject_r${R}.GenerateOnly.mH120.12345.root"
    if [[ "$DRY_RUN" == false ]]; then
        if [[ ! -f "$TOY_FILE" ]]; then
            echo "  ERROR: Toy file not generated for r=${R}"
            continue
        fi
        mv -f "$TOY_FILE" "${R_OUTPUT_DIR}/"
    fi
done

# ===== Phase 2: Run FitDiagnostics in parallel =====
echo ""
echo "===== Phase 2: Running FitDiagnostics (4 parallel jobs) ====="

# Function to run FitDiagnostics for a single r value
run_fitdiagnostics() {
    local R=$1
    local OUTPUT_DIR=$2
    local NTOYS=$3
    local R_OUTPUT_DIR="${OUTPUT_DIR}/r${R}"
    local TOY_FILE="${R_OUTPUT_DIR}/higgsCombine.inject_r${R}.GenerateOnly.mH120.12345.root"

    echo "  [r=${R}] Starting FitDiagnostics..."
    combine -M FitDiagnostics workspace.root \
        -t ${NTOYS} \
        --toysFile ${TOY_FILE} \
        --rMin -5 --rMax 5 \
        -n .recovery_r${R} \
        -m 120 \
        2>&1 | tee ${R_OUTPUT_DIR}/fitdiag.out

    # Move output files
    mv -f "fitDiagnostics.recovery_r${R}.root" "${R_OUTPUT_DIR}/" 2>/dev/null || true
    mv -f "higgsCombine.recovery_r${R}.FitDiagnostics.mH120."*.root "${R_OUTPUT_DIR}/" 2>/dev/null || true
    echo "  [r=${R}] FitDiagnostics complete"
}
export -f run_fitdiagnostics

if [[ "$DRY_RUN" == false ]]; then
    # Run FitDiagnostics in parallel with 4 jobs
    parallel -j 4 run_fitdiagnostics {} "${OUTPUT_DIR}" "${NTOYS}" ::: "${R_VALUES[@]}"
else
    for R in "${R_VALUES[@]}"; do
        echo "[DRY-RUN] combine -M FitDiagnostics workspace.root -t ${NTOYS} --toysFile ... --rMin -5 --rMax 5 -n .recovery_r${R} -m 120"
    done
fi

# ===== Phase 3: Extract results =====
echo ""
echo "===== Phase 3: Extracting results ====="

if [[ "$DRY_RUN" == false ]]; then
    python3 "${PYTHON_DIR}/extractInjectionResults.py" \
        --masspoint "${MASSPOINT}" \
        --era "${ERA}" \
        --channel "${CHANNEL}" \
        --method "${METHOD}" \
        --binning "${BINNING}" \
        --output "${OUTPUT_DIR}/injection_results.json"
else
    echo "[DRY-RUN] python3 extractInjectionResults.py --masspoint ${MASSPOINT} --era ${ERA} --channel ${CHANNEL} --method ${METHOD} --binning ${BINNING}"
fi

# ===== Phase 4: Generate B2G-compliant plots =====
echo ""
echo "===== Phase 4: Generating plots ====="

if [[ "$DRY_RUN" == false ]]; then
    # Plot 1: Bias distribution (r - r_inj)
    echo "Creating bias test plot..."
    python3 "${PYTHON_DIR}/plotBiasTest.py" \
        --masspoint "${MASSPOINT}" \
        --era "${ERA}" \
        --channel "${CHANNEL}" \
        --method "${METHOD}" \
        --binning "${BINNING}"

    # Plot 2: Pull distribution (r - r_inj) / r_Err
    echo ""
    echo "Creating pull distribution plot..."
    python3 "${PYTHON_DIR}/plotPullDist.py" \
        --masspoint "${MASSPOINT}" \
        --era "${ERA}" \
        --channel "${CHANNEL}" \
        --method "${METHOD}" \
        --binning "${BINNING}"
else
    echo "[DRY-RUN] python3 plotBiasTest.py ..."
    echo "[DRY-RUN] python3 plotPullDist.py ..."
fi

echo ""
echo "============================================================"
echo "Results saved to ${OUTPUT_DIR}/"
echo "  - injection_results.json  : Fit results with fit_status"
echo "  - bias_test.png/pdf       : Bias distribution (r - r_inj)"
echo "  - pull_dist.png/pdf       : Pull distribution"
echo "============================================================"
echo "Done."
