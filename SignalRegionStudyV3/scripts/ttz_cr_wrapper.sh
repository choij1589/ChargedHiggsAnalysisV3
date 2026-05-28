#!/bin/bash
#
# ttz_cr_wrapper.sh - HTCondor wrapper for TTZ2E1Mu CR GoF DAG steps
#
# Step set:
#   template          $ERA  (Run2 / Run3 / All)
#   datacard          $ERA
#   validate          $ERA
#   workspace         $ERA
#   gof_data          $ERA
#   gof_toys          $ERA $SEED $TOYS_PER_BATCH
#   gof_collect       $ERA
#   fitdiag           $ERA
#   plotpostfit       $ERA
#   plotpulls         $ERA
#   impact            $ERA
#
# All steps assume the CR Run-period component templates layout:
#   templates/{ERA}/TTZ2E1Mu/MHc130_MA90/CR/ZWin_adaptive/
#
# Usage:
#   ./ttz_cr_wrapper.sh <STEP> <ERA> [SEED] [TOYS_PER_BATCH]

set -euo pipefail

STEP="${1:-}"
ERA="${2:-}"

if [[ -z "$STEP" || -z "$ERA" ]]; then
    echo "Usage: $0 <STEP> <ERA> [SEED] [TOYS_PER_BATCH]"
    exit 1
fi

# CR layout constants
CHANNEL="TTZ2E1Mu"
MASSPOINT="MHc130_MA90"
METHOD="CR"
BINNING_TAG="ZWin_adaptive"

# Resolve repo root from this script's location, then source setup.sh from
# inside SignalRegionStudyV3 so cvmfs+CMSSW (incl. pyROOT) is on the worker's
# environment. setup.sh derives WORKDIR from $PWD/.. so we must cd first.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
cd "$REPO_DIR"
source ./setup.sh

echo "============================================================"
echo "TTZ2E1Mu CR wrapper: step=$STEP, era=$ERA, host=$(hostname)"
echo "============================================================"

case "$STEP" in
    template)
        python3 python/makeCRTemplates.py --era "$ERA"
        ;;
    validate)
        python3 python/checkTemplates.py \
            --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
            --method "$METHOD" --binning "$BINNING_TAG"
        ;;
    datacard)
        python3 python/printCRDatacard.py --era "$ERA"
        ;;
    workspace)
        TPL_DIR="${WORKDIR}/SignalRegionStudyV3/templates/${ERA}/${CHANNEL}/${MASSPOINT}/${METHOD}/${BINNING_TAG}"
        (cd "$TPL_DIR" && text2workspace.py datacard.txt -o workspace.root)
        ;;
    gof_data)
        bash scripts/runCRGoF.sh --era "$ERA" --step data
        ;;
    gof_toys)
        SEED="${3:-}"
        TPB="${4:-}"
        if [[ -z "$SEED" || -z "$TPB" ]]; then
            echo "ERROR: gof_toys requires SEED and TOYS_PER_BATCH"
            exit 1
        fi
        bash scripts/runCRGoF.sh --era "$ERA" --step toys --seed "$SEED" --toys-per-batch "$TPB"
        ;;
    gof_collect)
        bash scripts/runCRGoF.sh --era "$ERA" --step collect
        ;;
    fitdiag)
        bash scripts/runFitDiagnostics.sh \
            --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
            --method "$METHOD" --binning "$BINNING_TAG"
        ;;
    plotpostfit)
        # plotPostfitMass.py: real-mass unbinned post-fit + pre-fit plots.
        # For CR, --method=CR triggers the no-blinding branch (real data, no
        # _unblind/_partial_unblind suffix), and --fit-channel=TTZ2E1Mu picks
        # the CR datacard path (vs SR's "Combined" rollup).
        python3 python/plotPostfitMass.py \
            --era "$ERA" --masspoint "$MASSPOINT" --method "$METHOD" --binning "$BINNING_TAG" \
            --channel-scope "$CHANNEL" --fit-channel "$CHANNEL" \
            --fit-type b
        ;;
    plotpulls)
        bash scripts/runPullPlots.sh \
            --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
            --method "$METHOD" --binning "$BINNING_TAG"
        ;;
    impact)
        # Run inside this DAG node rather than submitting a nested per-nuisance
        # Condor DAG. CR mode uses observed data with the plain ZWin_adaptive
        # template path.
        bash scripts/runImpacts.sh \
            --era "$ERA" --channel "$CHANNEL" --masspoint "$MASSPOINT" \
            --method "$METHOD" --binning "$BINNING_TAG" \
            --parallel 1
        ;;
    *)
        echo "ERROR: unknown step '$STEP'"
        echo "Valid: template datacard validate workspace gof_data gof_toys gof_collect fitdiag plotpostfit plotpulls impact"
        exit 1
        ;;
esac

echo "============================================================"
echo "Step '$STEP' for era '$ERA' completed."
echo "============================================================"
