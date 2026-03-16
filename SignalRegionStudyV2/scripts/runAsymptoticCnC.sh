#!/bin/bash
#
# runAsymptoticCnC.sh - Run asymptotic limit calculation using Cut-and-Count datacard
#
# Runs combine from combine_output/asymptotic_cnc/ to avoid interfering with
# the standard binned-template asymptotic outputs in combine_output/asymptotic/.
#
# Usage:
#   ./runAsymptoticCnC.sh --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --method Baseline --binning extended
#

set -euo pipefail

ERA=""
CHANNEL=""
MASSPOINT=""
METHOD="Baseline"
BINNING="uniform"
NSIGMA="3.0"
PARTIAL_UNBLIND=false
UNBLIND=false
DRY_RUN=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --era)            ERA="$2";        shift 2 ;;
        --channel)        CHANNEL="$2";    shift 2 ;;
        --masspoint)      MASSPOINT="$2";  shift 2 ;;
        --method)         METHOD="$2";     shift 2 ;;
        --binning)        BINNING="$2";    shift 2 ;;
        --nsigma)         NSIGMA="$2";     shift 2 ;;
        --partial-unblind) PARTIAL_UNBLIND=true; shift ;;
        --unblind)        UNBLIND=true;    shift ;;
        --dry-run)        DRY_RUN=true;    shift ;;
        --verbose)        VERBOSE=true;    shift ;;
        -h|--help)
            echo "Usage: $0 --era ERA --channel CHANNEL --masspoint MASSPOINT [options]"
            echo ""
            echo "Options:"
            echo "  --era            Data-taking period (2016preVFP, 2017, 2018, Run2, etc.)"
            echo "  --channel        Analysis channel (SR1E2Mu, SR3Mu, Combined)"
            echo "  --masspoint      Signal mass point (e.g., MHc130_MA90)"
            echo "  --method         Template method (Baseline, ParticleNet) [default: Baseline]"
            echo "  --binning        Binning scheme (uniform, extended) [default: uniform]"
            echo "  --partial-unblind  Use partial-unblind templates"
            echo "  --unblind        Use full unblind templates"
            echo "  --dry-run        Print commands without executing"
            echo "  --verbose        Enable verbose output"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$ERA" || -z "$CHANNEL" || -z "$MASSPOINT" ]]; then
    echo "ERROR: --era, --channel, and --masspoint are required"
    exit 1
fi

if [[ "$UNBLIND" == true && "$PARTIAL_UNBLIND" == true ]]; then
    echo "ERROR: --unblind and --partial-unblind are mutually exclusive"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

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

NSIGMA_TAG="${NSIGMA%.*}sigma"
# Handle non-integer nsigma (e.g. 2.5 -> 2.5sigma)
[[ "$NSIGMA" == *.* && "${NSIGMA##*.}" != "0" ]] && NSIGMA_TAG="${NSIGMA}sigma"

DATACARD="${TEMPLATE_DIR}/datacard_cnc_${NSIGMA_TAG}.txt"
if [[ ! -f "$DATACARD" ]]; then
    echo "ERROR: CnC datacard not found: $DATACARD"
    echo "       Run printDatacardCnC.py --nsigma $NSIGMA first."
    exit 1
fi

OUTPUT_DIR="${TEMPLATE_DIR}/combine_output/asymptotic_cnc_${NSIGMA_TAG}"
mkdir -p "$OUTPUT_DIR"

log() { [[ "$VERBOSE" == true ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" || true; }
run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] $1"
    else
        log "Running: $1"
        eval "$1"
    fi
}

# Run combine from the isolated output directory.
# datacard_cnc.txt has no shapes directive so no relative path to shapes.root.
cd "$OUTPUT_DIR"
log "Working directory: $(pwd)"

NAME=".${MASSPOINT}.${METHOD}.${BINNING_SUFFIX}.CnC_${NSIGMA_TAG}"
echo "Running CnC AsymptoticLimits for ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING_SUFFIX}, ${NSIGMA_TAG})..."

COMBINE_CMD="combine -M AsymptoticLimits \"${DATACARD}\" \
    -n \"${NAME}\" \
    -m 120 \
    --rAbsAcc 0.0001 \
    --rRelAcc 0.01 \
    2>&1 | tee combine_logger.out"

run_cmd "$COMBINE_CMD"

if [[ "$DRY_RUN" == false ]]; then
    ROOT_FILE="${OUTPUT_DIR}/higgsCombine${NAME}.AsymptoticLimits.mH120.root"
    if [[ -f "$ROOT_FILE" ]]; then
        echo "SUCCESS: Output saved to ${OUTPUT_DIR}/"
        echo ""
        if [[ "$PARTIAL_UNBLIND" == true ]]; then
            echo "Limit calculation completed (r values hidden for partial-unblind mode)"
        else
            echo "Limit summary (CnC):"
            root -l -b -q -e "
                TFile *f = TFile::Open(\"${ROOT_FILE}\");
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
        fi
    else
        echo "WARNING: No output file created"
    fi
fi

echo "Done."
