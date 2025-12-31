#!/bin/bash
#
# runAsymptotic.sh - Run asymptotic limit calculation using HiggsCombine
#
# Usage:
#   ./runAsymptotic.sh --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --method Baseline --binning uniform
#

set -e

# Default values
ERA=""
CHANNEL=""
MASSPOINT=""
METHOD="Baseline"
BINNING="uniform"
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
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --era ERA --channel CHANNEL --masspoint MASSPOINT [--method METHOD] [--binning BINNING] [--dry-run] [--verbose]"
            echo ""
            echo "Options:"
            echo "  --era        Data-taking period (2016preVFP, 2017, 2018, etc.)"
            echo "  --channel    Analysis channel (SR1E2Mu, SR3Mu, Combined)"
            echo "  --masspoint  Signal mass point (e.g., MHc130_MA90)"
            echo "  --method     Template method (Baseline, ParticleNet) [default: Baseline]"
            echo "  --binning    Binning scheme (uniform, extended) [default: uniform]"
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

# Check if datacard exists
if [[ ! -f "$TEMPLATE_DIR/datacard.txt" ]]; then
    echo "ERROR: Datacard not found: $TEMPLATE_DIR/datacard.txt"
    exit 1
fi

# Create output directory
OUTPUT_DIR="${TEMPLATE_DIR}/combine_output/asymptotic"
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

# Run AsymptoticLimits
echo "Running AsymptoticLimits for ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING})..."

COMBINE_CMD="combine -M AsymptoticLimits datacard.txt \
    -n .${MASSPOINT}.${METHOD}.${BINNING} \
    -m 120 \
    --rAbsAcc 0.0001 \
    --rRelAcc 0.01 \
    2>&1 | tee ${OUTPUT_DIR}/combine_logger.out"

run_cmd "$COMBINE_CMD"

# Move output files to output directory
if [[ "$DRY_RUN" == false ]]; then
    mv -f higgsCombine.*.AsymptoticLimits.*.root "$OUTPUT_DIR/" 2>/dev/null || true
    mv -f roostats-*.root "$OUTPUT_DIR/" 2>/dev/null || true

    # Check if output was created
    if ls "${OUTPUT_DIR}"/higgsCombine.*.AsymptoticLimits.*.root 1>/dev/null 2>&1; then
        echo "SUCCESS: Output saved to ${OUTPUT_DIR}/"

        # Print limit summary
        echo ""
        echo "Limit summary:"
        root -l -b -q -e "
            TFile *f = TFile::Open(\"${OUTPUT_DIR}/higgsCombine.${MASSPOINT}.${METHOD}.${BINNING}.AsymptoticLimits.mH120.root\");
            TTree *limit = (TTree*)f->Get(\"limit\");
            double r;
            limit->SetBranchAddress(\"limit\", &r);
            const char* labels[] = {\"Exp -2sigma\", \"Exp -1sigma\", \"Exp median\", \"Exp +1sigma\", \"Exp +2sigma\", \"Observed\"};
            for (int i = 0; i < limit->GetEntries(); i++) {
                limit->GetEntry(i);
                printf(\"  %s: %.4f\\n\", labels[i], r);
            }
            f->Close();
        " 2>/dev/null || echo "  (Could not print summary)"
    else
        echo "WARNING: No output file created"
    fi
fi

echo "Done."
