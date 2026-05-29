#!/bin/bash
#
# runSignificance.sh - Run observed Combine Significance from workspace.root.
#
# Usage:
#   ./scripts/runSignificance.sh --era All --channel Combined \
#     --masspoint MHc160_MA85 --method ParticleNet --binning extended --unblind
#

set -euo pipefail

ERA=""
CHANNEL="Combined"
MASSPOINT=""
METHOD="Baseline"
BINNING="extended"
NUISANCE="fallback_lnn"
WORKSPACE="workspace.root"
MASS="120"
RMIN="-20"
RMAX="20"
PARTIAL_UNBLIND=false
UNBLIND=false
DRY_RUN=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --era) ERA="$2"; shift 2 ;;
        --channel) CHANNEL="$2"; shift 2 ;;
        --masspoint) MASSPOINT="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --binning) BINNING="$2"; shift 2 ;;
        --nuisance) NUISANCE="$2"; shift 2 ;;
        --workspace) WORKSPACE="$2"; shift 2 ;;
        --mass) MASS="$2"; shift 2 ;;
        --rMin) RMIN="$2"; shift 2 ;;
        --rMax) RMAX="$2"; shift 2 ;;
        --partial-unblind) PARTIAL_UNBLIND=true; shift ;;
        --unblind) UNBLIND=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        --verbose) VERBOSE=true; shift ;;
        -h|--help)
            echo "Usage: $0 --era ERA --masspoint MASSPOINT [OPTIONS]"
            echo "  --channel CHANNEL    Analysis channel [default: Combined]"
            echo "  --method METHOD      Baseline or ParticleNet [default: Baseline]"
            echo "  --binning BINNING    Binning scheme [default: extended]"
            echo "  --nuisance MODE      fallback_lnn (default) or preserve_shape"
            echo "  --workspace FILE     Workspace file in template dir [default: workspace.root]"
            echo "  --mass MASS          Combine mass label [default: 120]"
            echo "  --rMin VALUE         POI lower bound [default: -20]"
            echo "  --rMax VALUE         POI upper bound [default: 20]"
            echo "  --partial-unblind    Use partial-unblind template suffix"
            echo "  --unblind            Use full-unblind template suffix"
            echo "  --dry-run            Print commands without executing"
            echo "  --verbose            Enable verbose logging"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$ERA" || -z "$MASSPOINT" ]]; then
    echo "ERROR: --era and --masspoint are required"
    exit 1
fi
if [[ "$UNBLIND" == true && "$PARTIAL_UNBLIND" == true ]]; then
    echo "ERROR: --unblind and --partial-unblind are mutually exclusive"
    exit 1
fi
case "$NUISANCE" in
    fallback_lnn|preserve_shape) ;;
    *) echo "ERROR: Invalid --nuisance value '$NUISANCE'"; exit 1 ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

BINNING_SUFFIX="${BINNING}"
if [[ "$UNBLIND" == true ]]; then
    BINNING_SUFFIX="${BINNING}_unblind"
elif [[ "$PARTIAL_UNBLIND" == true ]]; then
    BINNING_SUFFIX="${BINNING}_partial_unblind"
fi
if [[ "$NUISANCE" == "preserve_shape" ]]; then
    BINNING_SUFFIX="${BINNING_SUFFIX}_preserve_shape"
fi

TEMPLATE_DIR="${WORKDIR}/SignalRegionStudyV3/templates/${ERA}/${CHANNEL}/${MASSPOINT}/${METHOD}/${BINNING_SUFFIX}"
if [[ "$DRY_RUN" != true && ! -d "$TEMPLATE_DIR" ]]; then
    echo "ERROR: Template directory not found: $TEMPLATE_DIR"
    exit 1
fi
if [[ "$DRY_RUN" != true && ! -f "${TEMPLATE_DIR}/${WORKSPACE}" ]]; then
    echo "ERROR: Workspace not found: ${TEMPLATE_DIR}/${WORKSPACE}"
    exit 1
fi

log() { [[ "$VERBOSE" == true ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" || true; }
run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then echo "[DRY-RUN] $1"; else log "Running: $1"; eval "$1"; fi
}

OUTPUT_DIR="combine_output/significance"
NAME=".${MASSPOINT}.${METHOD}.${BINNING_SUFFIX}.Significance"

echo "Running Significance for ${MASSPOINT} (${ERA}/${CHANNEL}/${METHOD}/${BINNING_SUFFIX})"
echo "  Workspace: ${WORKSPACE}"
echo "  r range:   [${RMIN}, ${RMAX}]"

cd "$TEMPLATE_DIR"
mkdir -p "$OUTPUT_DIR"

COMBINE_CMD="combine -M Significance ${WORKSPACE} \
    --uncapped=1 \
    --rMin=${RMIN} --rMax=${RMAX} \
    -n ${NAME} \
    -m ${MASS} \
    2>&1 | tee ${OUTPUT_DIR}/combine_logger.out"

run_cmd "$COMBINE_CMD"

if [[ "$DRY_RUN" == false ]]; then
    mv -f higgsCombine.*.Significance.*.root "$OUTPUT_DIR/" 2>/dev/null || true
    mv -f roostats-*.root "$OUTPUT_DIR/" 2>/dev/null || true
    ls -lh "$OUTPUT_DIR"
fi

