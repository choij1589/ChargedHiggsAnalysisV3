#!/bin/bash
#
# collectUnblindResults.sh - Gather unblind diagnostic plots by mass point.
#
# Reads configs/masspoints.json baseline_done/particlenet_done and copies existing
# artifacts into results/unblind/<masspoint>/<method>/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${SCRIPT_DIR}/configs/masspoints.json"
METHOD="all"
BINNING="extended"
NUISANCE="fallback_lnn"
ERAS="Run2,Run3,All"
POSTFIT_TARGETS="Run2:SR1E2Mu,Run2:SR3Mu,Run2:Combined,Run3:SR1E2Mu,Run3:SR3Mu,Run3:Combined,All:SR1E2Mu,All:SR3Mu,All:Combined"
GOF_TARGETS="All:Combined,All:SR1E2Mu,All:SR3Mu,Run2:Combined,Run3:Combined"
SCORE_TARGETS="Run2:SR1E2Mu,Run2:SR3Mu,Run2:Combined,Run3:SR1E2Mu,Run3:SR3Mu,Run3:Combined,All:SR1E2Mu,All:SR3Mu,All:Combined"
CHANNEL_OVERRIDE=""
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
            CHANNEL_OVERRIDE="$2"
            shift 2
            ;;
        --eras)
            ERAS="$2"
            shift 2
            ;;
        --postfit-targets)
            POSTFIT_TARGETS="$2"
            shift 2
            ;;
        --gof-targets)
            GOF_TARGETS="$2"
            shift 2
            ;;
        --score-targets)
            SCORE_TARGETS="$2"
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
  --channel CHANNEL   Legacy shortcut: collect only one channel across --eras
  --eras LIST         Comma-separated source eras for legacy --channel and fitdiag extras [default: Run2,Run3,All]
  --postfit-targets LIST
                     Comma-separated ERA:CHANNEL targets for prefit, B-only,
                     and S+B postfit mass distributions
                     [default: {Run2,Run3,All} x {SR1E2Mu,SR3Mu,Combined}]
  --gof-targets LIST  Comma-separated ERA:CHANNEL targets for GoF results
                     [default: All x {Combined,SR1E2Mu,SR3Mu}, Run2 x Combined, Run3 x Combined]
  --score-targets LIST
                     Comma-separated ERA:CHANNEL targets for ParticleNet score
                     distributions
                     [default: {Run2,Run3,All} x {SR1E2Mu,SR3Mu,Combined}]
  --config PATH       Masspoint JSON [default: configs/masspoints.json]
                     Uses baseline_done for Baseline and particlenet_done for
                     ParticleNet.
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
validate_targets() {
    local label="$1"
    local raw="$2"
    local item era channel
    IFS=',' read -ra items <<< "$raw"
    for item in "${items[@]}"; do
        [[ -z "$item" ]] && continue
        if [[ "$item" != *:* ]]; then
            echo "ERROR: $label target '$item' must use ERA:CHANNEL format" >&2
            exit 1
        fi
        era="${item%%:*}"
        channel="${item#*:}"
        case "$era" in
            Run2|Run3|All) ;;
            *) echo "ERROR: $label target has invalid era '$era'" >&2; exit 1 ;;
        esac
        case "$channel" in
            Combined|SR1E2Mu|SR3Mu) ;;
            *) echo "ERROR: $label target has invalid channel '$channel'" >&2; exit 1 ;;
        esac
    done
}

if [[ -n "$CHANNEL_OVERRIDE" ]]; then
    case "$CHANNEL_OVERRIDE" in
        Combined|SR1E2Mu|SR3Mu) ;;
        *)
            echo "ERROR: --channel must be Combined, SR1E2Mu, or SR3Mu" >&2
            exit 1
            ;;
    esac
    IFS=',' read -ra legacy_eras <<< "$ERAS"
    POSTFIT_TARGETS=""
    GOF_TARGETS=""
    SCORE_TARGETS=""
    for era in "${legacy_eras[@]}"; do
        [[ -z "$era" ]] && continue
        POSTFIT_TARGETS+="${POSTFIT_TARGETS:+,}${era}:${CHANNEL_OVERRIDE}"
        GOF_TARGETS+="${GOF_TARGETS:+,}${era}:${CHANNEL_OVERRIDE}"
        SCORE_TARGETS+="${SCORE_TARGETS:+,}${era}:${CHANNEL_OVERRIDE}"
    done
fi

validate_targets "--postfit-targets" "$POSTFIT_TARGETS"
validate_targets "--gof-targets" "$GOF_TARGETS"
validate_targets "--score-targets" "$SCORE_TARGETS"

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
key = "baseline_done" if method == "Baseline" else "particlenet_done"
with open(config) as handle:
    data = json.load(handle)
for masspoint in data.get(key, []):
    print(masspoint)
PY
}

extra_args=(
    --binning "$BINNING"
    --nuisance "$NUISANCE"
    --eras "$ERAS"
    --postfit-targets "$POSTFIT_TARGETS"
    --gof-targets "$GOF_TARGETS"
    --score-targets "$SCORE_TARGETS"
)
[[ "$DRY_RUN" == true ]] && extra_args+=(--dry-run)
[[ "$STRICT"  == true ]] && extra_args+=(--strict)

echo "============================================================"
echo "SignalRegionStudyV3 Unblind Result Collection"
echo "Method:        $METHOD"
echo "Binning:       $BINNING"
echo "Nuisance mode: $NUISANCE"
echo "Eras:          $ERAS"
echo "Postfit grid:  $POSTFIT_TARGETS"
echo "GoF grid:      $GOF_TARGETS"
echo "Score grid:    $SCORE_TARGETS"
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
            "${extra_args[@]}"; then
            status=1
        fi
    done
done

echo ""
echo "Collection complete: $total masspoint-method entries processed."
echo "Output root: ${SCRIPT_DIR}/results/unblind"
exit "$status"
