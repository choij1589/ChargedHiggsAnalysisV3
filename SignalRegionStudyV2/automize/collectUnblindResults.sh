#!/bin/bash
#
# collectUnblindResults.sh - Gather unblind diagnostic plots by mass point.
#
# Reads configs/masspoints.json unblind.{baseline,particlenet} and copies
# existing artifacts into results/unblind/<masspoint>/<method>/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${SCRIPT_DIR}/configs/masspoints.json"
METHOD="all"
BINNING="extended"
NUISANCE="fallback_lnn"
CHANNEL="Combined"
ERAS="Run2,Run3,All"
DRY_RUN=false
STRICT=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --method)
            METHOD="$2"
            shift 2
            ;;
        --binning)
            BINNING="$2"
            shift 2
            ;;
        --nuisance)
            NUISANCE="$2"
            shift 2
            ;;
        --channel)
            CHANNEL="$2"
            shift 2
            ;;
        --eras)
            ERAS="$2"
            shift 2
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --strict)
            STRICT=true
            shift
            ;;
        --help|-h)
            cat <<EOF
Usage: $0 [OPTIONS]

Options:
  --method METHOD     Baseline, ParticleNet, or all [default: all]
  --binning BINNING   Template binning name [default: extended]
  --nuisance MODE     fallback_lnn or preserve_shape [default: fallback_lnn]
  --channel CHANNEL   Combined, SR1E2Mu, or SR3Mu [default: Combined]
  --eras LIST         Comma-separated source eras [default: Run2,Run3,All]
  --config PATH       Masspoint JSON [default: configs/masspoints.json]
  --dry-run           Print collection summary without copying
  --strict            Exit nonzero if any artifact is missing

Output:
  results/unblind/MASSPOINT/METHOD/
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

case "$METHOD" in
    Baseline|ParticleNet|all) ;;
    *)
        echo "ERROR: --method must be Baseline, ParticleNet, or all" >&2
        exit 1
        ;;
esac
case "$NUISANCE" in
    fallback_lnn|preserve_shape) ;;
    *)
        echo "ERROR: --nuisance must be fallback_lnn or preserve_shape" >&2
        exit 1
        ;;
esac
case "$CHANNEL" in
    Combined|SR1E2Mu|SR3Mu) ;;
    *)
        echo "ERROR: --channel must be Combined, SR1E2Mu, or SR3Mu" >&2
        exit 1
        ;;
esac

methods=()
if [[ "$METHOD" == "all" ]]; then
    methods=(Baseline ParticleNet)
else
    methods=("$METHOD")
fi

load_masspoints() {
    local method="$1"
    python3 - "$CONFIG" "$method" <<'PY'
import json
import sys

config, method = sys.argv[1], sys.argv[2]
key = "baseline" if method == "Baseline" else "particlenet"
with open(config) as handle:
    data = json.load(handle)
for masspoint in data.get("unblind", {}).get(key, []):
    print(masspoint)
PY
}

extra_args=(
    --binning "$BINNING"
    --nuisance "$NUISANCE"
    --eras "$ERAS"
)
[[ "$DRY_RUN" == true ]] && extra_args+=(--dry-run)
[[ "$STRICT"  == true ]] && extra_args+=(--strict)

echo "============================================================"
echo "SignalRegionStudyV2 Unblind Result Collection"
echo "Method:        $METHOD"
echo "Channel:       $CHANNEL"
echo "Binning:       $BINNING"
echo "Nuisance mode: $NUISANCE"
echo "Eras:          $ERAS"
echo "Config:        $CONFIG"
echo "Dry run:       $DRY_RUN"
echo "Strict:        $STRICT"
echo "============================================================"

status=0
total=0
for method in "${methods[@]}"; do
    mapfile -t masspoints < <(load_masspoints "$method")
    echo ""
    echo ">>> $method: ${#masspoints[@]} mass points"
    for masspoint in "${masspoints[@]}"; do
        total=$((total + 1))
        if ! python3 "${SCRIPT_DIR}/python/collectUnblindMasspoint.py" \
            --masspoint "$masspoint" \
            --method "$method" \
            --channel "$CHANNEL" \
            "${extra_args[@]}"; then
            status=1
        fi
    done
done

echo ""
echo "Collection complete: $total masspoint-method entries processed."
echo "Output root: ${SCRIPT_DIR}/results/unblind"
exit "$status"
