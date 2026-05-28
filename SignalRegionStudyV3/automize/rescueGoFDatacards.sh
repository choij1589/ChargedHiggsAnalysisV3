#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"
source setup.sh

METHOD="Baseline"
BINNING="extended"
EXTRA_ARGS=(--unblind)
FORCE=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--force] [--dry-run]"
            echo ""
            echo "Regenerate the known missing datacards needed before GoF backfill:"
            echo "  All/SR1E2Mu/MHc100_MA24"
            echo "  Run2/Combined/MHc130_MA100"
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

rows=(
    "All SR1E2Mu MHc100_MA24 Run2:SR1E2Mu,Run3:SR1E2Mu"
    "Run2 Combined MHc130_MA100 Run2:SR1E2Mu,Run2:SR3Mu"
)

suffix="${BINNING}_unblind"

for row in "${rows[@]}"; do
    read -r era channel masspoint sources <<< "$row"
    template_dir="${WORKDIR}/SignalRegionStudyV3/templates/${era}/${channel}/${masspoint}/${METHOD}/${suffix}"
    datacard="${template_dir}/datacard.txt"

    if [[ -f "$datacard" && "$FORCE" != true ]]; then
        echo "SKIP ${era}/${channel}/${masspoint}: datacard exists"
        continue
    fi

    echo "RESCUE ${era}/${channel}/${masspoint}: merge templates and print datacard"
    merge_cmd=(
        python3 python/mergeRunPeriodTemplates.py
        --era "$era"
        --channel "$channel"
        --masspoint "$masspoint"
        --method "$METHOD"
        --binning "$BINNING"
        "${EXTRA_ARGS[@]}"
        --sources "$sources"
    )
    card_cmd=(
        python3 python/printDatacard.py
        --era "$era"
        --channel "$channel"
        --masspoint "$masspoint"
        --method "$METHOD"
        --binning "$BINNING"
        "${EXTRA_ARGS[@]}"
    )

    if [[ "$DRY_RUN" == true ]]; then
        printf '[DRY-RUN]'; printf ' %q' "${merge_cmd[@]}"; echo
        printf '[DRY-RUN]'; printf ' %q' "${card_cmd[@]}"; echo
        continue
    fi

    "${merge_cmd[@]}"
    "${card_cmd[@]}"

    if [[ ! -f "$datacard" ]]; then
        echo "ERROR: datacard was not created: $datacard" >&2
        exit 1
    fi
    echo "OK ${datacard}"
done

