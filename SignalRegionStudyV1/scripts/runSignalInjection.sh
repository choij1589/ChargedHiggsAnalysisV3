#!/bin/bash
#
# runSignalInjection.sh - Run signal injection tests
#
# Injects signal at specified strengths and verifies recovery.
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
INJECT_R="0,1,2"
NTOYS=100
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
        --inject-r)
            INJECT_R="$2"
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
            echo "  --binning    Binning scheme (uniform, sigma) [default: uniform]"
            echo "  --inject-r   Comma-separated r values to inject [default: 0,1,2]"
            echo "  --ntoys      Number of toys per injection [default: 100]"
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

echo "Signal injection test for ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING})"
echo "  Injection r values: ${INJECT_R}"
echo "  Toys per injection: ${NTOYS}"
echo ""

# Create workspace if needed
if [[ ! -f "workspace.root" ]] || [[ "datacard.txt" -nt "workspace.root" ]]; then
    echo "Creating workspace..."
    run_cmd "text2workspace.py datacard.txt -o workspace.root"
fi

# Parse injection r values
IFS=',' read -ra R_VALUES <<< "$INJECT_R"

# Summary results
declare -A RESULTS

for R in "${R_VALUES[@]}"; do
    echo ""
    echo "===== Injecting signal at r = $R ====="

    R_OUTPUT_DIR="${OUTPUT_DIR}/r${R}"
    mkdir -p "$R_OUTPUT_DIR"

    # Step 1: Generate toys with injected signal
    echo "  Generating toys with r=${R} signal..."
    run_cmd "combine -M GenerateOnly workspace.root \
        -t ${NTOYS} \
        --expectSignal ${R} \
        --saveToys \
        -n .inject_r${R} \
        -m 120 \
        -s 12345 \
        2>&1 | tee ${R_OUTPUT_DIR}/generate.out"

    # Find the generated toy file
    TOY_FILE="higgsCombine.inject_r${R}.GenerateOnly.mH120.12345.root"

    if [[ "$DRY_RUN" == false ]]; then
        if [[ ! -f "$TOY_FILE" ]]; then
            echo "  ERROR: Toy file not generated"
            continue
        fi
        mv -f "$TOY_FILE" "${R_OUTPUT_DIR}/"
        TOY_FILE="${R_OUTPUT_DIR}/$(basename $TOY_FILE)"
    fi

    # Step 2: Run limit extraction on injected toys
    echo "  Running limit extraction on injected toys..."
    run_cmd "combine -M AsymptoticLimits workspace.root \
        -t ${NTOYS} \
        --toysFile ${TOY_FILE} \
        -n .recovery_r${R} \
        -m 120 \
        2>&1 | tee ${R_OUTPUT_DIR}/recovery.out"

    if [[ "$DRY_RUN" == false ]]; then
        mv -f higgsCombine.*.AsymptoticLimits.*.root "${R_OUTPUT_DIR}/" 2>/dev/null || true

        # Extract and display results
        LIMIT_FILE="${R_OUTPUT_DIR}/higgsCombine.recovery_r${R}.AsymptoticLimits.mH120.root"
        if [[ -f "$LIMIT_FILE" ]]; then
            # Get median expected limit
            MEDIAN=$(root -l -b -q -e "
                TFile *f = TFile::Open(\"${LIMIT_FILE}\");
                TTree *limit = (TTree*)f->Get(\"limit\");
                double r; limit->SetBranchAddress(\"limit\", &r);
                limit->GetEntry(2);  // Median expected
                printf(\"%.4f\", r);
                f->Close();
            " 2>/dev/null | tail -1)

            echo "  Injected: r = ${R}"
            echo "  Recovered (median expected): r = ${MEDIAN}"
            RESULTS[$R]=$MEDIAN
        fi
    fi
done

# Print summary
echo ""
echo "===== Signal Injection Summary ====="
echo "Injected r | Recovered r (median exp)"
echo "-----------|------------------------"
for R in "${R_VALUES[@]}"; do
    if [[ -n "${RESULTS[$R]}" ]]; then
        printf "    %-7s | %s\n" "$R" "${RESULTS[$R]}"
    else
        printf "    %-7s | N/A\n" "$R"
    fi
done
echo ""

echo "Results saved to ${OUTPUT_DIR}/"
echo "Done."
