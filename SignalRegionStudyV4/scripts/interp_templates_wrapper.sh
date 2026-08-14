#!/bin/bash
# Condor leaf wrapper for the interp-signal template DAGs
# (automize/interpTemplates.sh). Runs IN PLACE on the NFS module dir —
# unlike the MC template wrapper there is no scratch stage-out: seed jobs
# read pnfs samples via the same shared dirs, member/merge/datacard/
# validate/asymptotic jobs are JSON/histogram-level.
#
# Arguments: STEP MASSPOINT SEED ERA CHANNEL [EXTRA...]
#   STEP: template | merge | datacard | validate | asymptotic
#         | gof-data | gof-toys (EXTRA = toy seed) | gof-collect | impacts
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

STEP=$1
MASSPOINT=$2
SEED=$3
ERA=$4
CHANNEL=$5
shift 5
# --method may appear anywhere in EXTRA (ParticleNet interp arm, added by
# the DAG generator); default Baseline. Extracted here and re-injected as
# exactly one --method per tool call, so old DAGs stay valid unchanged.
METHOD="Baseline"
_rest=()
while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--method" ]]; then
        METHOD="$2"; shift 2
    else
        _rest+=("$1"); shift
    fi
done
set -- ${_rest[@]+"${_rest[@]}"}
EXTRA_ARGS="$*"

export WORKDIR="$SRS_REPO_DIR"
cd "$SRS_MODULE_DIR"
export PATH="${PWD}/python:${PATH}"
export HOME="${HOME:-$(getent passwd "$(id -un)" | cut -d: -f6)}"
export PYTHONPATH="$HOME/.local/lib/python3.9/site-packages:$SRS_REPO_DIR/Common/Tools:${PYTHONPATH:-}"

srs_setup_cmssw

SRC_ARGS=(--signal-source interp-signal)

case "$STEP" in
    template)
        if [[ "$MASSPOINT" == "$SEED" ]]; then
            # seed: one heavy job per category (era/channel = the category)
            python3 python/makeBinnedTemplates.py \
                --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
                --method "$METHOD" "${SRC_ARGS[@]}" $EXTRA_ARGS
        else
            # member: signal injection is cheap — build all four component
            # dirs (the merge step consumes them) in this single job.
            for cat in "Run2 SR1E2Mu" "Run2 SR3Mu" "Run3 SR1E2Mu" "Run3 SR3Mu"; do
                read -r cat_era cat_channel <<< "$cat"
                python3 python/makeBinnedTemplates.py \
                    --era "$cat_era" --channel "$cat_channel" \
                    --masspoint "$MASSPOINT" \
                    --method "$METHOD" "${SRC_ARGS[@]}" $EXTRA_ARGS
            done
        fi
        ;;
    merge)
        # CHANNEL is the merge target; sources depend on it.
        case "$CHANNEL" in
            Combined) SOURCES="Run2:SR1E2Mu,Run2:SR3Mu,Run3:SR1E2Mu,Run3:SR3Mu" ;;
            SR1E2Mu|SR3Mu) SOURCES="Run2:${CHANNEL},Run3:${CHANNEL}" ;;
            *) echo "ERROR: unknown merge target $CHANNEL"; exit 1 ;;
        esac
        python3 python/mergeRunPeriodTemplates.py \
            --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
            --method "$METHOD" --sources "$SOURCES" "${SRC_ARGS[@]}" $EXTRA_ARGS
        ;;
    datacard)
        python3 python/printDatacard.py \
            --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
            --method "$METHOD" "${SRC_ARGS[@]}" $EXTRA_ARGS
        ;;
    validate)
        python3 python/validateRunPeriodTemplates.py \
            --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
            --method "$METHOD" "${SRC_ARGS[@]}" $EXTRA_ARGS
        ;;
    asymptotic)
        ./scripts/runAsymptotic.sh \
            --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
            --method "$METHOD" --signal-source interp-signal \
            --seed "$SEED" $EXTRA_ARGS
        ;;
    gof-data|gof-collect)
        ./scripts/runGoF.sh \
            --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
            --method "$METHOD" --signal-source interp-signal \
            --seed "$SEED" --step "${STEP#gof-}" $EXTRA_ARGS
        ;;
    gof-toys)
        # first EXTRA arg = toy batch seed
        TOY_SEED=$1; shift
        ./scripts/runGoF.sh \
            --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
            --method "$METHOD" --signal-source interp-signal \
            --seed "$SEED" --step toys --toy-seed "$TOY_SEED" "$@"
        ;;
    impacts)
        ./scripts/runImpacts.sh \
            --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
            --method "$METHOD" --signal-source interp-signal \
            --seed "$SEED" $EXTRA_ARGS
        ;;
    plot-score)
        # ParticleNet interp arm only: score distributions at the group
        # seed (masspoint == seed). plotParticleNetScore.py has no
        # --method; interp-signal implies ParticleNet.
        python3 python/plotParticleNetScore.py \
            --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
            --signal-source interp-signal $EXTRA_ARGS
        ;;
    fitdiag)
        ./scripts/runFitDiagnostics.sh \
            --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
            --method "$METHOD" --signal-source interp-signal \
            --seed "$SEED" $EXTRA_ARGS
        ;;
    plotpostfit)
        python3 python/plotPostfitMass.py \
            --era "$ERA" --masspoint "$MASSPOINT" --method "$METHOD" \
            --signal-source interp-signal --fit-type both $EXTRA_ARGS
        ;;
    plotpulls)
        ./scripts/runPullPlots.sh \
            --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
            --method "$METHOD" --signal-source interp-signal \
            --seed "$SEED" --pull-fit both $EXTRA_ARGS
        ;;
    *)
        echo "ERROR: unknown step '$STEP'"
        exit 1
        ;;
esac
