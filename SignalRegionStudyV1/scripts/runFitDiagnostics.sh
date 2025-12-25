#!/bin/bash
#
# runFitDiagnostics.sh - Run FitDiagnostics for pre/post-fit analysis
#
# Usage:
#   ./runFitDiagnostics.sh --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --method Baseline --binning uniform
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
OUTPUT_DIR="${TEMPLATE_DIR}/combine_output/fitDiagnostics"
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

echo "Running FitDiagnostics for ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING})..."

# Run FitDiagnostics
run_cmd "combine -M FitDiagnostics datacard.txt \
    -n .${MASSPOINT}.${METHOD}.${BINNING} \
    -m 120 \
    --saveShapes \
    --saveNormalizations \
    --saveWithUncertainties \
    --robustFit 1 \
    --robustHesse 1 \
    -t -1 \
    --expectSignal 0 \
    2>&1 | tee ${OUTPUT_DIR}/combine_logger.out"

# Move outputs
if [[ "$DRY_RUN" == false ]]; then
    mv -f fitDiagnostics*.root "${OUTPUT_DIR}/" 2>/dev/null || true
    mv -f higgsCombine.*.FitDiagnostics.*.root "${OUTPUT_DIR}/" 2>/dev/null || true

    FITDIAG_FILE="${OUTPUT_DIR}/fitDiagnostics.${MASSPOINT}.${METHOD}.${BINNING}.root"
    if [[ -f "$FITDIAG_FILE" ]]; then
        echo ""
        echo "Generating nuisance parameter plots..."

        # Try to run diffNuisances if available
        if command -v python3 &> /dev/null; then
            DIFFNUISANCES="${CMSSW_BASE}/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py"
            if [[ -f "$DIFFNUISANCES" ]]; then
                run_cmd "python3 ${DIFFNUISANCES} ${FITDIAG_FILE} \
                    --all \
                    -g ${OUTPUT_DIR}/nuisance_pulls.root \
                    2>&1 | tee ${OUTPUT_DIR}/nuisance_pulls.txt"
            else
                echo "  diffNuisances.py not found, skipping nuisance plots"
            fi
        fi

        # Print fit result summary
        echo ""
        echo "Fit result summary:"
        root -l -b -q -e "
            TFile *f = TFile::Open(\"${FITDIAG_FILE}\");
            RooFitResult *fit_s = (RooFitResult*)f->Get(\"fit_s\");
            RooFitResult *fit_b = (RooFitResult*)f->Get(\"fit_b\");

            if (fit_b) {
                printf(\"Background-only fit:\\n\");
                printf(\"  Status: %d\\n\", fit_b->status());
                printf(\"  EDM: %.6f\\n\", fit_b->edm());
                printf(\"  NLL: %.4f\\n\", fit_b->minNll());
            }

            if (fit_s) {
                printf(\"Signal+background fit:\\n\");
                printf(\"  Status: %d\\n\", fit_s->status());
                printf(\"  EDM: %.6f\\n\", fit_s->edm());
                printf(\"  NLL: %.4f\\n\", fit_s->minNll());

                RooRealVar *r = (RooRealVar*)fit_s->floatParsFinal().find(\"r\");
                if (r) {
                    printf(\"  r = %.4f +/- %.4f\\n\", r->getVal(), r->getError());
                }
            }
            f->Close();
        " 2>/dev/null || echo "  (Could not print fit summary)"

        echo ""
        echo "SUCCESS: FitDiagnostics saved to ${OUTPUT_DIR}/"
        echo ""
        echo "Output files:"
        ls -la "${OUTPUT_DIR}/"
    else
        echo "WARNING: fitDiagnostics output not found"
    fi
fi

echo "Done."
