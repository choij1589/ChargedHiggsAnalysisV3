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
INJECT_R="0,0.5,1,1.5,2"
NTOYS=1000
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
            echo "  --binning    Binning scheme (uniform, extended) [default: uniform]"
            echo "  --inject-r   Comma-separated r values to inject [default: 0,0.5,1,1.5,2]"
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

# Cleanup previous injection results
if [[ -d "$OUTPUT_DIR" ]]; then
    echo "Cleaning up previous injection results..."
    rm -rf "$OUTPUT_DIR"
fi
mkdir -p "$OUTPUT_DIR"

# Parse injection r values
IFS=',' read -ra R_VALUES <<< "$INJECT_R"

# Summary results
declare -A RESULTS

# ===== Phase 1: Generate toys for all r values =====
echo ""
echo "===== Phase 1: Generating toys ====="
for R in "${R_VALUES[@]}"; do
    echo "  Generating toys with r=${R} signal..."

    R_OUTPUT_DIR="${OUTPUT_DIR}/r${R}"
    mkdir -p "$R_OUTPUT_DIR"

    run_cmd "combine -M GenerateOnly workspace.root \
        -t ${NTOYS} \
        --expectSignal ${R} \
        --saveToys \
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

# ===== Phase 2: Run MultiDimFit in parallel =====
echo ""
echo "===== Phase 2: Running MultiDimFit (5 parallel jobs) ====="

# Function to run MultiDimFit for a single r value
run_multidimfit() {
    local R=$1
    local OUTPUT_DIR=$2
    local NTOYS=$3
    local R_OUTPUT_DIR="${OUTPUT_DIR}/r${R}"
    local TOY_FILE="${R_OUTPUT_DIR}/higgsCombine.inject_r${R}.GenerateOnly.mH120.12345.root"

    echo "  [r=${R}] Starting MultiDimFit..."
    combine -M MultiDimFit workspace.root \
        -t ${NTOYS} \
        --toysFile ${TOY_FILE} \
        --algo singles \
        --rMin -5 --rMax 5 \
        -n .recovery_r${R} \
        -m 120 \
        2>&1 | tee ${R_OUTPUT_DIR}/recovery.out

    # Move only the specific file for this r value (avoid race condition in parallel)
    mv -f "higgsCombine.recovery_r${R}.MultiDimFit.mH120."*.root "${R_OUTPUT_DIR}/" 2>/dev/null || true
    echo "  [r=${R}] MultiDimFit complete"
}
export -f run_multidimfit

if [[ "$DRY_RUN" == false ]]; then
    # Run MultiDimFit in parallel with 4 jobs
    parallel -j 5 run_multidimfit {} "${OUTPUT_DIR}" "${NTOYS}" ::: "${R_VALUES[@]}"
else
    for R in "${R_VALUES[@]}"; do
        echo "[DRY-RUN] combine -M MultiDimFit workspace.root -t ${NTOYS} --toysFile ... --algo singles --rMin -5 --rMax 5 -n .recovery_r${R} -m 120"
    done
fi

# ===== Phase 3: Extract results =====
echo ""
echo "===== Phase 3: Extracting results ====="
for R in "${R_VALUES[@]}"; do
    R_OUTPUT_DIR="${OUTPUT_DIR}/r${R}"

    if [[ "$DRY_RUN" == false ]]; then
        # Extract and display results (filename includes seed when using toys)
        FIT_FILE=$(ls ${R_OUTPUT_DIR}/higgsCombine.recovery_r${R}.MultiDimFit.mH120.*.root 2>/dev/null | head -1)
        if [[ -f "$FIT_FILE" ]]; then
            # Get mean and std of fitted r values (best-fit only, quantileExpected == -1)
            RESULT=$(root -l -b -q -e "
                TFile *f = TFile::Open(\"${FIT_FILE}\");
                TTree *limit = (TTree*)f->Get(\"limit\");
                float r, quantile;
                limit->SetBranchAddress(\"r\", &r);
                limit->SetBranchAddress(\"quantileExpected\", &quantile);
                double sum = 0, sum2 = 0;
                int n = 0;
                for (int i = 0; i < limit->GetEntries(); i++) {
                    limit->GetEntry(i);
                    if (quantile < -0.5) {  // best-fit has quantileExpected = -1
                        sum += r;
                        sum2 += r*r;
                        n++;
                    }
                }
                double mean = sum / n;
                double std = sqrt(sum2/n - mean*mean);
                printf(\"%.4f +/- %.4f\", mean, std);
                f->Close();
            " 2>/dev/null | tail -1)

            echo "  r=${R}: recovered ${RESULT}"
            RESULTS[$R]=$RESULT
        else
            echo "  r=${R}: FIT FILE NOT FOUND"
        fi
    fi
done

# Print summary
echo ""
echo "===== Signal Injection Summary ====="
echo "Injected r | Recovered r (mean +/- std)"
echo "-----------|---------------------------"
for R in "${R_VALUES[@]}"; do
    if [[ -n "${RESULTS[$R]}" ]]; then
        printf "    %-7s | %s\n" "$R" "${RESULTS[$R]}"
    else
        printf "    %-7s | N/A\n" "$R"
    fi
done
echo ""

# Create plot of injected vs recovered r values
if [[ "$DRY_RUN" == false ]]; then
    echo "Creating injection test plot..."

    # Build data file for plotting
    DATA_FILE="${OUTPUT_DIR}/injection_data.csv"
    echo "injected,recovered,error" > "$DATA_FILE"
    for R in "${R_VALUES[@]}"; do
        if [[ -n "${RESULTS[$R]}" ]]; then
            MEAN=$(echo "${RESULTS[$R]}" | awk '{print $1}')
            STD=$(echo "${RESULTS[$R]}" | awk '{print $3}')
            echo "${R},${MEAN},${STD}" >> "$DATA_FILE"
        fi
    done

    PLOT_FILE="${OUTPUT_DIR}/injection_test.pdf"

    # Call plotting scripts
    PYTHON_DIR="$(dirname "$SCRIPT_DIR")/python"

    # Plot 1: Injected vs Recovered summary
    python3 "${PYTHON_DIR}/plotInjectionTest.py" \
        --masspoint "${MASSPOINT}" \
        --era "${ERA}" \
        --channel "${CHANNEL}" \
        --method "${METHOD}" \
        --binning "${BINNING}"

    # Plot 2: Distribution of recovered r for each injection point
    python3 "${PYTHON_DIR}/plotInjectionDist.py" \
        --masspoint "${MASSPOINT}" \
        --era "${ERA}" \
        --channel "${CHANNEL}" \
        --method "${METHOD}" \
        --binning "${BINNING}"
fi

echo "Results saved to ${OUTPUT_DIR}/"
echo "Done."
