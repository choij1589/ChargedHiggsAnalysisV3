#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
source setup.sh

METHOD="Baseline"
BINNING="extended"
NTOYS=1000
NBATCHES=10
FORCE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ntoys)
            NTOYS="$2"
            shift 2
            ;;
        --nbatches)
            NBATCHES="$2"
            shift 2
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--ntoys N] [--nbatches N] [--force] [--dry-run]"
            echo ""
            echo "Submit GoF DAGs only for missing GoF JSONs in the five review targets:"
            echo "  All/Combined, All/SR1E2Mu, All/SR3Mu, Run2/Combined, Run3/Combined"
            echo "Mass points come from configs/masspoints.json: baseline, falling back to gof.baseline."
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

mapfile -t MASSPOINTS < <(
    python3 - <<'PY'
import json
d = json.load(open("configs/masspoints.json"))
masspoints = d.get("baseline") or d.get("gof", {}).get("baseline", [])
for mp in masspoints:
    print(mp)
PY
)

if [[ ${#MASSPOINTS[@]} -eq 0 ]]; then
    echo "ERROR: no Baseline mass points configured in configs/masspoints.json" >&2
    exit 1
fi

targets=(
    "All Combined"
    "All SR1E2Mu"
    "All SR3Mu"
    "Run2 Combined"
    "Run3 Combined"
)

suffix="${BINNING}_unblind"
submitted=0
skipped_done=0
missing_datacard=0

for masspoint in "${MASSPOINTS[@]}"; do
    for target in "${targets[@]}"; do
        read -r era channel <<< "$target"
        template_dir="${WORKDIR}/SignalRegionStudyV3/templates/${era}/${channel}/${masspoint}/${METHOD}/${suffix}"
        datacard="${template_dir}/datacard.txt"
        gof_json="${template_dir}/combine_output/gof/gof.json"

        if [[ -f "$gof_json" && "$FORCE" != true ]]; then
            ((skipped_done += 1))
            continue
        fi
        if [[ ! -f "$datacard" ]]; then
            echo "MISSING_DATACARD ${era}/${channel}/${masspoint}: $datacard"
            ((missing_datacard += 1))
            continue
        fi

        cmd=(
            bash scripts/runGoF.sh
            --era "$era"
            --channel "$channel"
            --masspoint "$masspoint"
            --method "$METHOD"
            --binning "$BINNING"
            --unblind
            --ntoys "$NTOYS"
            --nbatches "$NBATCHES"
            --condor
        )

        echo "SUBMIT_GOF ${era}/${channel}/${masspoint}"
        if [[ "$DRY_RUN" == true ]]; then
            printf '[DRY-RUN]'; printf ' %q' "${cmd[@]}"; echo
        else
            "${cmd[@]}"
        fi
        ((submitted += 1))
    done
done

echo "============================================================"
echo "GoF rescue summary"
echo "Configured mass points: ${#MASSPOINTS[@]}"
echo "Submitted: $submitted"
echo "Skipped existing gof.json: $skipped_done"
echo "Missing datacard: $missing_datacard"
echo "Dry run: $DRY_RUN"
echo "============================================================"

if [[ "$missing_datacard" -gt 0 ]]; then
    echo "Run automize/rescueGoFDatacards.sh first, then re-run this script." >&2
    exit 2
fi

