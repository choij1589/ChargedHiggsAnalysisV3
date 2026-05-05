#!/bin/bash
#
# runCRGoF.sh - Goodness-of-Fit for the TTZ2E1Mu control region
#
# This script is a CR-specific copy of scripts/runGoF.sh whose combine commands
# match the partial-unblind GoF block VERBATIM:
#   combine -M GoodnessOfFit --algo=saturated \
#     --freezeParameters r --setParameters r=0 \
#     --toysFrequentist [-t N -s SEED] \
#     workspace.root -m 120
#
# The CR datacard carries a dummy 1e-6/bin signal placeholder so the `r`
# parameter exists; freezing r=0 makes the fit physically background-only.
#
# Differences vs runGoF.sh:
#   - Templates path is the CR five-segment layout with method=CR,
#     masspoint=MHc130_MA90, binning=ZWin_adaptive (no _unblind/_partial_unblind suffix).
#   - --step {data,toys,collect} dispatcher for HTCondor wrapper use.
#   - No blind/Asimov path: the CR is always real-data.
#
# Usage:
#   ./runCRGoF.sh --era Run2 --ntoys 500 --nbatches 5
#   ./runCRGoF.sh --era Run2 --step data
#   ./runCRGoF.sh --era Run2 --step toys --seed 3 --toys-per-batch 100
#   ./runCRGoF.sh --era Run2 --step collect

set -euo pipefail

ERA=""
NTOYS=500
NBATCHES=5
STEP="all"
SEED=0
TOYS_PER_BATCH=0
DRY_RUN=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --era)            ERA="$2"; shift 2 ;;
        --ntoys)          NTOYS="$2"; shift 2 ;;
        --nbatches)       NBATCHES="$2"; shift 2 ;;
        --step)           STEP="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --toys-per-batch) TOYS_PER_BATCH="$2"; shift 2 ;;
        --dry-run)        DRY_RUN=true; shift ;;
        --verbose)        VERBOSE=true; shift ;;
        -h|--help)
            echo "Usage: $0 --era ERA [--ntoys N] [--nbatches N] [--step STEP] [--seed N] [--toys-per-batch N]"
            echo "  STEP: all (default) | data | toys | collect"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$ERA" ]]; then
    echo "ERROR: --era is required"
    exit 1
fi

# Fixed CR layout (mirrors makeCRTemplates.py / printCRDatacard.py constants)
CHANNEL="TTZ2E1Mu"
MASSPOINT="MHc130_MA90"
METHOD="CR"
BINNING_SUFFIX="ZWin_adaptive"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR_DEFAULT="$(dirname "$(dirname "$SCRIPT_DIR")")"
: "${WORKDIR:=$WORKDIR_DEFAULT}"

TEMPLATE_DIR="${WORKDIR}/SignalRegionStudyV2/templates/${ERA}/${CHANNEL}/${MASSPOINT}/${METHOD}/${BINNING_SUFFIX}"
OUTPUT_DIR="${TEMPLATE_DIR}/combine_output/gof"

if [[ ! -d "$TEMPLATE_DIR" ]]; then
    echo "ERROR: Template directory not found: $TEMPLATE_DIR"
    exit 1
fi
if [[ ! -f "$TEMPLATE_DIR/datacard.txt" ]]; then
    echo "ERROR: Datacard not found: $TEMPLATE_DIR/datacard.txt"
    exit 1
fi

if [[ "$TOYS_PER_BATCH" -le 0 ]]; then
    TOYS_PER_BATCH=$((NTOYS / NBATCHES))
fi

# COMBINE command flags — copied verbatim from the partial-unblind block in
# scripts/runGoF.sh. Do not modify these.
TOY_OPTS="--toysFrequentist"
DATA_OPTS=""
FREEZE_OPTS="--freezeParameters r --setParameters r=0"

log() { [[ "$VERBOSE" == true ]] && echo "[$(date '+%H:%M:%S')] $1" || true; }
run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then echo "[DRY-RUN] $1"; else log "Running: $1"; eval "$1"; fi
}

mkdir -p "$OUTPUT_DIR"
cd "$TEMPLATE_DIR"

# workspace.root is built once in the combine_era step (DAG parent); only
# regenerate here if absent (e.g. standalone invocation outside the DAG).
# A simultaneous regeneration across parallel toy jobs corrupts the file.
if [[ ! -f "workspace.root" ]]; then
    run_cmd "text2workspace.py datacard.txt -o workspace.root"
fi

run_data() {
    echo "===== TTZ2E1Mu CR GoF: data ====="
    run_cmd "combine -M GoodnessOfFit workspace.root \
        --algo=saturated \
        ${FREEZE_OPTS} \
        ${DATA_OPTS} \
        -n .gof_data \
        -m 120 2>&1 | tee ${OUTPUT_DIR}/gof_data.log"
    if [[ "$DRY_RUN" == false ]]; then
        mv -f higgsCombine.gof_data.GoodnessOfFit.mH120.root "$OUTPUT_DIR/" 2>/dev/null || true
    fi
}

run_toys_one() {
    local seed=$1
    echo "===== TTZ2E1Mu CR GoF: toys (seed=${seed}, n=${TOYS_PER_BATCH}) ====="
    run_cmd "combine -M GoodnessOfFit workspace.root \
        --algo=saturated \
        ${FREEZE_OPTS} \
        ${TOY_OPTS} \
        -t ${TOYS_PER_BATCH} \
        -n .gof_toys_s${seed} \
        -m 120 -s ${seed} 2>&1 | tee ${OUTPUT_DIR}/gof_toys_s${seed}.log"
    if [[ "$DRY_RUN" == false ]]; then
        mv -f "higgsCombine.gof_toys_s${seed}.GoodnessOfFit.mH120.${seed}.root" \
            "$OUTPUT_DIR/" 2>/dev/null || true
    fi
}

run_toys_all() {
    for s in $(seq 1 "$NBATCHES"); do
        run_toys_one "$s"
    done
}

run_collect() {
    echo "===== TTZ2E1Mu CR GoF: collect + plot ====="
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] combineTool.py -M CollectGoodnessOfFit ... && plotGof.py ..."
        return
    fi
    cd "$OUTPUT_DIR"
    DATA_FILE="higgsCombine.gof_data.GoodnessOfFit.mH120.root"
    TOY_FILES=$(ls higgsCombine.gof_toys_s*.GoodnessOfFit.mH120.*.root 2>/dev/null | tr '\n' ' ')
    if [[ -z "$TOY_FILES" ]]; then
        echo "ERROR: No toy GoF files found in $OUTPUT_DIR"
        exit 1
    fi
    combineTool.py -M CollectGoodnessOfFit \
        --input "${DATA_FILE}" ${TOY_FILES} \
        -o gof.json
    plotGof.py gof.json \
        --statistic saturated --mass 120.0 \
        -o gof_plot \
        --title-right="${ERA} TTZ2E1Mu CR"
    cd "$TEMPLATE_DIR"
}

case "$STEP" in
    all)
        run_data
        run_toys_all
        run_collect
        ;;
    data)
        run_data
        ;;
    toys)
        if [[ "$SEED" -le 0 ]]; then
            echo "ERROR: --step toys requires --seed N (>0)"
            exit 1
        fi
        run_toys_one "$SEED"
        ;;
    collect)
        run_collect
        ;;
    *)
        echo "ERROR: Unknown --step '$STEP'. Valid: all, data, toys, collect"
        exit 1
        ;;
esac

echo "Results in: $OUTPUT_DIR"
