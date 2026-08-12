#!/bin/bash
#
# runFitDiagnostics.sh - Run FitDiagnostics to produce post-fit shapes
#
# Usage:
#   ./runFitDiagnostics.sh --era 2018 --channel Combined --masspoint MHc130_MA90 --method Baseline --binning extended
#

set -eo pipefail

# Default values
ERA=""
CHANNEL=""
MASSPOINT=""
METHOD="Baseline"
SIGNAL_SOURCE="mc-signal"
SEED=""
BLIND=false
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
        --blind)
            BLIND=true
            shift
            ;;
        --signal-source)
            SIGNAL_SOURCE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
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
            echo "Usage: $0 --era ERA --channel CHANNEL --masspoint MASSPOINT [--method METHOD] [--blind] [--dry-run] [--verbose]"
            echo ""
            echo "Options:"
            echo "  --era        Run-period target (Run2, Run3, All)"
            echo "  --channel    Analysis channel (SR1E2Mu, SR3Mu, Combined)"
            echo "  --masspoint  Signal mass point (e.g., MHc130_MA90)"
            echo "  --method     Template method (Baseline, ParticleNet) [default: Baseline]"
            echo "  --blind      Use the {method}_blind template segment (Asimov fit: -t -1)"
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


source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

# Template directory (V4 layout: templates/{masspoint}/{method}/{era}/{channel})
METHOD_SEGMENT="$METHOD"
[[ "$BLIND" == true ]] && METHOD_SEGMENT="${METHOD}_blind"
TEMPLATE_DIR="$(srs_template_dir "$MASSPOINT" "$METHOD_SEGMENT" "$SIGNAL_SOURCE" "$ERA" "$CHANNEL")"
# interp-signal group members nest under their seed
if [[ -n "$SEED" && "$SEED" != "$MASSPOINT" ]]; then
    TEMPLATE_DIR="$(srs_template_dir "$SEED" "$METHOD_SEGMENT" "$SIGNAL_SOURCE" "$ERA" "$CHANNEL")/points/$MASSPOINT"
fi

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
OUTPUT_DIR="${TEMPLATE_DIR}/combine_output/fitdiag"
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

# Determine Asimov options (blind mode only)
ASIMOV_OPTIONS=""
if [[ "$BLIND" == true ]]; then
    ASIMOV_OPTIONS="-t -1 --expectSignal 1"
fi

# Run FitDiagnostics
echo "Running FitDiagnostics for ${MASSPOINT} (${METHOD_SEGMENT}/${ERA}/${CHANNEL})..."

COMBINE_CMD="combine -M FitDiagnostics datacard.txt \
    --saveShapes --saveWithUncertainties \
    -n .${MASSPOINT}.${METHOD_SEGMENT} \
    -m 120 --robustFit 1 \
    --cminDefaultMinimizerStrategy 0 \
    --cminDefaultMinimizerTolerance 0.1 \
    --setParameterRanges r=-5,5 \
    ${ASIMOV_OPTIONS} \
    2>&1 | tee ${OUTPUT_DIR}/combine_logger.out"

run_cmd "$COMBINE_CMD"

# Move output files to output directory
if [[ "$DRY_RUN" == false ]]; then
    mv -f fitDiagnostics.${MASSPOINT}.${METHOD_SEGMENT}.root "$OUTPUT_DIR/" 2>/dev/null || true
    mv -f higgsCombine.*.FitDiagnostics.*.root "$OUTPUT_DIR/" 2>/dev/null || true
    mv -f roostats-*.root "$OUTPUT_DIR/" 2>/dev/null || true

    # Check if output was created
    if ls "${OUTPUT_DIR}"/fitDiagnostics.*.root 1>/dev/null 2>&1; then
        echo "SUCCESS: Output saved to ${OUTPUT_DIR}/"

        # Print fit status summary
        echo ""
        echo "Fit summary:"
        root -l -b -q -e "
            TFile *f = TFile::Open(\"${OUTPUT_DIR}/fitDiagnostics.${MASSPOINT}.${METHOD_SEGMENT}.root\");
            if (!f || f->IsZombie()) { printf(\"  Could not open fitDiagnostics file\\n\"); return; }

            // Print best-fit r
            RooFitResult *fit_b = (RooFitResult*)f->Get(\"fit_b\");
            RooFitResult *fit_s = (RooFitResult*)f->Get(\"fit_s\");

            if (fit_b) {
                printf(\"  B-only fit status: %d (0=converged)\\n\", fit_b->status());
                printf(\"  B-only fit quality: %d (3=full accurate cov matrix)\\n\", fit_b->covQual());
            }
            if (fit_s) {
                printf(\"  S+B fit status: %d (0=converged)\\n\", fit_s->status());
                printf(\"  S+B fit quality: %d (3=full accurate cov matrix)\\n\", fit_s->covQual());
                RooRealVar *r = (RooRealVar*)fit_s->floatParsFinal().find(\"r\");
                if (r) printf(\"  Best-fit r = %.4f +/- %.4f\\n\", r->getVal(), r->getError());
            }

            // Print post-fit yields from shapes_fit_b
            TDirectory *dir = (TDirectory*)f->Get(\"shapes_fit_b\");
            if (dir) {
                printf(\"\\n  Post-fit B-only yields:\\n\");
                TH1 *total = (TH1*)dir->Get(\"total_background\");
                if (!total) {
                    // Try subdirectories for combined channels
                    TList *keys = dir->GetListOfKeys();
                    TIter next(keys);
                    TKey *key;
                    while ((key = (TKey*)next())) {
                        TDirectory *subdir = (TDirectory*)dir->Get(key->GetName());
                        if (subdir && subdir->InheritsFrom(\"TDirectory\")) {
                            TH1 *sub_total = (TH1*)subdir->Get(\"total_background\");
                            TH1 *sub_data = (TH1*)subdir->Get(\"total_data\");
                            if (sub_total) printf(\"    %s total_background: %.2f\\n\", key->GetName(), sub_total->Integral());
                            if (sub_data) printf(\"    %s total_data: %.2f\\n\", key->GetName(), sub_data->Integral());
                        }
                    }
                } else {
                    printf(\"    total_background: %.2f\\n\", total->Integral());
                    TH1 *data_h = (TH1*)dir->Get(\"total_data\");
                    if (data_h) printf(\"    total_data: %.2f\\n\", data_h->Integral());
                }
            }
            f->Close();
        " 2>/dev/null || echo "  (Could not print summary)"
    else
        echo "ERROR: No output file created"
        exit 1
    fi
fi

echo "Done."
