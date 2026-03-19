#!/bin/bash
#
# runPullPlots.sh - Generate NP pull plots from FitDiagnostics output
#
# Runs diffNuisances.py on fitDiagnostics.root to produce:
#   - nuisance_pulls.txt   (text table of pre/post-fit NP values and uncertainties)
#   - nuisance_pulls.pdf   (pull plot canvas)
#
# Must be run after runFitDiagnostics.sh has produced combine_output/fitdiag/.
#
# Usage:
#   ./runPullPlots.sh --era All --channel Combined --masspoint MHc130_MA90 \
#                     --method ParticleNet --binning extended --partial-unblind
#

set -e

# Default values
ERA=""
CHANNEL=""
MASSPOINT=""
METHOD="Baseline"
BINNING="extended"
PARTIAL_UNBLIND=false
UNBLIND=false
DRY_RUN=false
VERBOSE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --era)           ERA="$2";           shift 2 ;;
        --channel)       CHANNEL="$2";       shift 2 ;;
        --masspoint)     MASSPOINT="$2";     shift 2 ;;
        --method)        METHOD="$2";        shift 2 ;;
        --binning)       BINNING="$2";       shift 2 ;;
        --partial-unblind) PARTIAL_UNBLIND=true; shift ;;
        --unblind)       UNBLIND=true;       shift ;;
        --dry-run)       DRY_RUN=true;       shift ;;
        --verbose)       VERBOSE=true;       shift ;;
        -h|--help)
            echo "Usage: $0 --era ERA --channel CHANNEL --masspoint MASSPOINT [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --method METHOD         Baseline or ParticleNet [default: Baseline]"
            echo "  --binning BINNING       extended or uniform [default: extended]"
            echo "  --partial-unblind       Use partial-unblind templates"
            echo "  --unblind               Use fully unblinded templates"
            echo "  --dry-run               Print commands without executing"
            echo "  --verbose               Enable verbose logging"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Validate
if [[ -z "$ERA" || -z "$CHANNEL" || -z "$MASSPOINT" ]]; then
    echo "ERROR: --era, --channel, and --masspoint are required"
    exit 1
fi
if [[ "$UNBLIND" == true && "$PARTIAL_UNBLIND" == true ]]; then
    echo "ERROR: --unblind and --partial-unblind are mutually exclusive"
    exit 1
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Resolve binning suffix
BINNING_SUFFIX="${BINNING}"
if   [[ "$UNBLIND"         == true ]]; then BINNING_SUFFIX="${BINNING}_unblind"
elif [[ "$PARTIAL_UNBLIND" == true ]]; then BINNING_SUFFIX="${BINNING}_partial_unblind"
fi

TEMPLATE_DIR="${WORKDIR}/SignalRegionStudyV2/templates/${ERA}/${CHANNEL}/${MASSPOINT}/${METHOD}/${BINNING_SUFFIX}"
FITDIAG_DIR="${TEMPLATE_DIR}/combine_output/fitdiag"

if [[ ! -d "$FITDIAG_DIR" ]]; then
    echo "ERROR: FitDiagnostics output directory not found: $FITDIAG_DIR"
    echo "  Run runFitDiagnostics.sh first."
    exit 1
fi

FITDIAG_FILE=$(ls "${FITDIAG_DIR}"/fitDiagnostics.*.root 2>/dev/null | head -1)
if [[ -z "$FITDIAG_FILE" ]]; then
    echo "ERROR: No fitDiagnostics.root file found in ${FITDIAG_DIR}"
    exit 1
fi

DIFFNUIS="${CMSSW_BASE}/src/HiggsAnalysis/CombinedLimit/test/diffNuisances.py"
if [[ ! -f "$DIFFNUIS" ]]; then
    echo "ERROR: diffNuisances.py not found at ${DIFFNUIS}"
    echo "  Make sure CMSSW environment is set up."
    exit 1
fi

log() { [[ "$VERBOSE" == true ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" || true; }
run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then echo "[DRY-RUN] $1"; else log "Running: $1"; eval "$1"; fi
}

echo "============================================================"
echo "NP Pull Plots (diffNuisances.py)"
echo "============================================================"
echo "  Era:       ${ERA}"
echo "  Channel:   ${CHANNEL}"
echo "  Masspoint: ${MASSPOINT}"
echo "  Method:    ${METHOD}"
echo "  Binning:   ${BINNING_SUFFIX}"
echo "  Input:     ${FITDIAG_FILE}"
echo ""

# Text table
echo "===== Generating text pull table ====="
run_cmd "python3 '${DIFFNUIS}' '${FITDIAG_FILE}' \
    --all \
    -f text \
    2>/dev/null > '${FITDIAG_DIR}/nuisance_pulls.txt'"

# ROOT canvas
echo "===== Generating pull plot canvas ====="
run_cmd "python3 '${DIFFNUIS}' '${FITDIAG_FILE}' \
    --all \
    -g '${FITDIAG_DIR}/nuisance_pulls.root' \
    2>/dev/null"

# Convert ROOT canvas to PDF
if [[ "$DRY_RUN" == false && -f "${FITDIAG_DIR}/nuisance_pulls.root" ]]; then
    echo "===== Converting canvas to PDF ====="
    root -l -b -q -e "
        TFile *f = TFile::Open(\"${FITDIAG_DIR}/nuisance_pulls.root\");
        if (!f || f->IsZombie()) { printf(\"ERROR: cannot open nuisance_pulls.root\\n\"); return; }
        TCanvas *c = (TCanvas*)f->Get(\"nuisances\");
        if (c) {
            c->SaveAs(\"${FITDIAG_DIR}/nuisance_pulls.pdf\");
            printf(\"Saved: nuisance_pulls.pdf\\n\");
        } else {
            printf(\"WARNING: 'nuisances' canvas not found in ROOT file\\n\");
        }
        f->Close();
    " 2>/dev/null || echo "WARNING: PDF conversion failed — check nuisance_pulls.root manually"
fi

echo ""
echo "============================================================"
if [[ "$DRY_RUN" == false ]]; then
    echo "Output files:"
    ls -lh "${FITDIAG_DIR}/nuisance_pulls".* 2>/dev/null || echo "  (none produced)"
fi
echo "Done."
echo "============================================================"
