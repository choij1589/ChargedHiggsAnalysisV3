#!/bin/bash
#
# runGoF.sh - saturated goodness-of-fit test on one template dir
# (mirrors the V3 recipe: background-only, frequentist toys, real data).
#
# Steps (run per step so the condor DAG can parallelize toy batches):
#   workspace  text2workspace.py datacard.txt -o workspace.root
#   data       observed test statistic
#   toys       one toy batch (--toy-seed N gives the batch its seed)
#   collect    CollectGoodnessOfFit -> gof.json + plotGof.py canvas
#
# Usage:
#   ./runGoF.sh --era All --channel Combined --masspoint MHc160_MA90 \
#       --signal-source interp-signal --seed MHc160_MA90 --step data
#

set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ERA=""
CHANNEL=""
MASSPOINT=""
METHOD="Baseline"
SIGNAL_SOURCE="mc-signal"
SEED=""
STEP=""
NTOYS=500
NBATCHES=5
TOY_SEED=1
BLIND=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --era)           ERA="$2";           shift 2 ;;
        --channel)       CHANNEL="$2";       shift 2 ;;
        --masspoint)     MASSPOINT="$2";     shift 2 ;;
        --method)        METHOD="$2";        shift 2 ;;
        --signal-source) SIGNAL_SOURCE="$2"; shift 2 ;;
        --seed)          SEED="$2";          shift 2 ;;
        --step)          STEP="$2";          shift 2 ;;
        --ntoys)         NTOYS="$2";         shift 2 ;;
        --nbatches)      NBATCHES="$2";      shift 2 ;;
        --toy-seed)      TOY_SEED="$2";      shift 2 ;;
        --blind)         BLIND=true;         shift ;;
        --dry-run)       DRY_RUN=true;       shift ;;
        -h|--help)
            echo "Usage: $0 --era ERA --channel CH --masspoint MP --step {workspace,data,toys,collect}"
            echo "          [--method M] [--signal-source S] [--seed SEED_MP]"
            echo "          [--ntoys N] [--nbatches N] [--toy-seed N] [--blind] [--dry-run]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$ERA" || -z "$CHANNEL" || -z "$MASSPOINT" || -z "$STEP" ]]; then
    echo "ERROR: --era, --channel, --masspoint and --step are required"
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

OUTPUT_DIR="$TEMPLATE_DIR/combine_output/gof"
TOYS_PER_BATCH=$((NTOYS / NBATCHES))

# Background-only, saturated statistic (V3 recipe; B2G stat recs).
FREEZE_OPTS="--freezeParameters r --setParameters r=0"
# Unblind (V4 default): real data_obs, frequentist toys.
TOY_OPTS="--toysFrequentist"
DATA_OPTS=""
if [[ "$BLIND" == true ]]; then
    TOY_OPTS="--toysFrequentist --bypassFrequentistFit"
    DATA_OPTS="-t -1"
fi

run() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] $*"
    else
        echo "+ $*"
        "$@"
    fi
}

cd "$TEMPLATE_DIR"
mkdir -p "$OUTPUT_DIR"

case "$STEP" in
    workspace)
        # Skip when up to date (several DAG nodes share the workspace; the
        # gof data node owns its creation).
        if [[ -f workspace.root && workspace.root -nt datacard.txt ]]; then
            echo "workspace.root up to date"
        else
            run text2workspace.py datacard.txt -o workspace.root
        fi
        ;;
    data)
        if [[ ! -f workspace.root || datacard.txt -nt workspace.root ]]; then
            run text2workspace.py datacard.txt -o workspace.root
        fi
        run combine -M GoodnessOfFit workspace.root \
            --algo=saturated \
            $FREEZE_OPTS $DATA_OPTS \
            -n .gof_data -m 120
        run mv -f higgsCombine.gof_data.GoodnessOfFit.mH120.root "$OUTPUT_DIR/"
        ;;
    toys)
        run combine -M GoodnessOfFit workspace.root \
            --algo=saturated \
            $FREEZE_OPTS $TOY_OPTS \
            -t "$TOYS_PER_BATCH" \
            -n ".gof_toys_s${TOY_SEED}" -m 120 -s "$TOY_SEED"
        run mv -f "higgsCombine.gof_toys_s${TOY_SEED}.GoodnessOfFit.mH120.${TOY_SEED}.root" "$OUTPUT_DIR/"
        ;;
    collect)
        cd "$OUTPUT_DIR"
        TOY_FILES=(higgsCombine.gof_toys_s*.GoodnessOfFit.mH120.*.root)
        if [[ ! -f higgsCombine.gof_data.GoodnessOfFit.mH120.root || ! -e "${TOY_FILES[0]}" ]]; then
            echo "ERROR: data and/or toy outputs missing in $OUTPUT_DIR"
            exit 1
        fi
        run combineTool.py -M CollectGoodnessOfFit \
            --input higgsCombine.gof_data.GoodnessOfFit.mH120.root "${TOY_FILES[@]}" \
            -o gof.json
        run plotGof.py gof.json --statistic saturated --mass 120.0 \
            -o gof_plot --title-right="${ERA} ${CHANNEL} ${MASSPOINT} ${METHOD_SEGMENT}"
        ;;
    *)
        echo "ERROR: unknown --step '$STEP'"
        exit 1
        ;;
esac
