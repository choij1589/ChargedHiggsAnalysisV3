#!/bin/bash
#
# runSignificance.sh - observed local significance of the signal hypothesis
# at one mass point, from the same datacard the limit is computed on.
#
# Run UNCAPPED (--uncapped 1 --rMin -5): the capped default floors a
# downward fluctuation at Z = 0, which would make every deficit point
# report the same "0" and hide exactly the thing a deficit scan is for.
# Uncapped, an excess gives the usual positive Z and a deficit its
# negative counterpart; the two agree wherever Z > 0.
#
# Usage:
#   ./runSignificance.sh --era All --channel Combined --masspoint MHc130_MA90 \
#       --method Baseline --signal-source interp-signal --seed MHc130_MA90
#

set -e

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ERA=""
CHANNEL=""
MASSPOINT=""
METHOD="Baseline"
SIGNAL_SOURCE="mc-signal"
SEED=""
RMIN="-5"
BLIND=false
DRY_RUN=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --era)           ERA="$2";           shift 2 ;;
        --channel)       CHANNEL="$2";       shift 2 ;;
        --masspoint)     MASSPOINT="$2";     shift 2 ;;
        --method)        METHOD="$2";        shift 2 ;;
        --signal-source) SIGNAL_SOURCE="$2"; shift 2 ;;
        --seed)          SEED="$2";          shift 2 ;;
        --rmin)          RMIN="$2";          shift 2 ;;
        --blind)         BLIND=true;         shift ;;
        --dry-run)       DRY_RUN=true;       shift ;;
        --verbose)       VERBOSE=true;       shift ;;
        -h|--help)
            echo "Usage: $0 --era ERA --channel CHANNEL --masspoint MASSPOINT"
            echo "          [--method M] [--signal-source S] [--seed SEED_MP]"
            echo "          [--rmin R] [--blind] [--dry-run] [--verbose]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$ERA" || -z "$CHANNEL" || -z "$MASSPOINT" ]]; then
    echo "ERROR: --era, --channel, and --masspoint are required"
    exit 1
fi

METHOD_SEGMENT="$METHOD"
[[ "$BLIND" == true ]] && METHOD_SEGMENT="${METHOD}_blind"
TEMPLATE_DIR="$(srs_template_dir "$MASSPOINT" "$METHOD_SEGMENT" "$SIGNAL_SOURCE" "$ERA" "$CHANNEL")"
# interp-signal group members nest under their seed
if [[ -n "$SEED" && "$SEED" != "$MASSPOINT" ]]; then
    TEMPLATE_DIR="$(srs_template_dir "$SEED" "$METHOD_SEGMENT" "$SIGNAL_SOURCE" "$ERA" "$CHANNEL")/points/$MASSPOINT"
fi

if [[ ! -f "$TEMPLATE_DIR/datacard.txt" ]]; then
    echo "ERROR: Datacard not found: $TEMPLATE_DIR/datacard.txt"
    exit 1
fi

OUTPUT_DIR="${TEMPLATE_DIR}/combine_output/significance"
mkdir -p "$OUTPUT_DIR"

log() { [[ "$VERBOSE" == true ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" || true; }
run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then echo "[DRY-RUN] $1"; else log "Running: $1"; eval "$1"; fi
}

cd "$TEMPLATE_DIR"
echo "Running Significance for ${MASSPOINT} (${METHOD_SEGMENT}/${ERA}/${CHANNEL})..."

COMBINE_CMD="combine -M Significance datacard.txt \
    -n .${MASSPOINT}.${METHOD_SEGMENT} \
    -m 120 \
    --uncapped 1 --rMin ${RMIN} \
    2>&1 | tee ${OUTPUT_DIR}/combine_logger.out"

run_cmd "$COMBINE_CMD"

if [[ "$DRY_RUN" == false ]]; then
    mv -f higgsCombine.*.Significance.*.root "$OUTPUT_DIR/" 2>/dev/null || true
    mv -f roostats-*.root "$OUTPUT_DIR/" 2>/dev/null || true

    OUT_ROOT="${OUTPUT_DIR}/higgsCombine.${MASSPOINT}.${METHOD_SEGMENT}.Significance.mH120.root"
    if [[ -f "$OUT_ROOT" ]]; then
        echo "SUCCESS: Output saved to ${OUTPUT_DIR}/"
        root -l -b -q -e "
            TFile *f = TFile::Open(\"${OUT_ROOT}\");
            TTree *t = (TTree*)f->Get(\"limit\");
            double z;
            t->SetBranchAddress(\"limit\", &z);
            t->GetEntry(0);
            printf(\"  Observed significance (uncapped): %+.4f sigma\\n\", z);
            f->Close();
        " 2>/dev/null || echo "  (Could not print summary)"
    else
        echo "ERROR: No output file created"
        exit 1
    fi
fi

echo "Done."
