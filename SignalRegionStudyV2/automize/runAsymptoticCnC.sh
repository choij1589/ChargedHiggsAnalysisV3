#!/bin/bash
#
# runAsymptoticCnC.sh - Full CnC asymptotic limit pipeline (local, no condor)
#
# Runs the complete CnC pipeline for all mass points in a given mode:
#   1. printDatacardCnC.py  -- generate CnC datacards per (era, channel)
#   2. combineDatacardsCnC.py --mode channel  -- SR1E2Mu+SR3Mu -> Combined per era
#   3. scripts/runAsymptoticCnC.sh  -- asymptotic on Combined per individual era
#   4. combineDatacardsCnC.py --mode era  -- combine eras -> Run2/Run3/All
#   5. scripts/runAsymptoticCnC.sh  -- asymptotic on combined-era Combined
# Then after all mass points:
#   6. collectLimits.py --cnc
#   7. plotLimits.py --cnc
#
# Usage:
#   automize/runAsymptoticCnC.sh --mode run2|run3|all --method Baseline|ParticleNet \
#       [--binning extended] [--unblind] [--partial-unblind] [--dry-run]
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Load mass point arrays
# shellcheck source=automize/load_masspoints.sh
source "${SCRIPT_DIR}/load_masspoints.sh"

# Era definitions
ERAs_RUN2=("2016preVFP" "2016postVFP" "2017" "2018")
ERAs_RUN3=("2022" "2022EE" "2023" "2023BPix")
CHANNELs=("SR1E2Mu" "SR3Mu")

# Default options
MODE=""
METHOD=""
BINNING="extended"
NSIGMA="3.0"
UNBLIND=false
PARTIAL_UNBLIND=false
DRY_RUN=false

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            MODE="$2"
            shift 2
            ;;
        --method)
            METHOD="$2"
            shift 2
            ;;
        --binning)
            BINNING="$2"
            shift 2
            ;;
        --nsigma)
            NSIGMA="$2"
            shift 2
            ;;
        --unblind)
            UNBLIND=true
            shift
            ;;
        --partial-unblind)
            PARTIAL_UNBLIND=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --mode run2|run3|all --method Baseline|ParticleNet [options]"
            echo ""
            echo "Required:"
            echo "  --mode    Era group to process: run2, run3, or all"
            echo "  --method  Template method: Baseline or ParticleNet"
            echo ""
            echo "Options:"
            echo "  --binning         Binning scheme (default: extended)"
            echo "  --nsigma          Mass window half-width in sigma_voigt (default: 3.0)"
            echo "  --unblind         Use full unblind templates"
            echo "  --partial-unblind Use partial-unblind templates"
            echo "  --dry-run         Print commands without executing"
            echo ""
            echo "Pipeline (per mass point):"
            echo "  1. printDatacardCnC.py per (era, channel)"
            echo "  2. combineDatacardsCnC.py --mode channel per era"
            echo "  3. scripts/runAsymptoticCnC.sh per individual era"
            echo "  4. combineDatacardsCnC.py --mode era -> Run2/Run3/All"
            echo "  5. scripts/runAsymptoticCnC.sh on combined era"
            echo "Then: collectLimits.py --cnc + plotLimits.py --cnc for each era group"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
if [[ -z "$MODE" ]]; then
    echo "ERROR: --mode is required (run2, run3, or all)"
    exit 1
fi
if [[ "$MODE" != "run2" && "$MODE" != "run3" && "$MODE" != "all" ]]; then
    echo "ERROR: --mode must be run2, run3, or all (got: $MODE)"
    exit 1
fi
if [[ -z "$METHOD" ]]; then
    echo "ERROR: --method is required (Baseline or ParticleNet)"
    exit 1
fi
if [[ "$METHOD" != "Baseline" && "$METHOD" != "ParticleNet" ]]; then
    echo "ERROR: --method must be Baseline or ParticleNet (got: $METHOD)"
    exit 1
fi
if [[ "$UNBLIND" == true && "$PARTIAL_UNBLIND" == true ]]; then
    echo "ERROR: --unblind and --partial-unblind are mutually exclusive"
    exit 1
fi

# ---------------------------------------------------------------------------
# Resolve era lists and combined era names
# ---------------------------------------------------------------------------
declare -a INDIVIDUAL_ERAs=()
declare -a COMBINED_ERAs=()

case "$MODE" in
    run2)
        INDIVIDUAL_ERAs=("${ERAs_RUN2[@]}")
        COMBINED_ERAs=("Run2")
        ;;
    run3)
        INDIVIDUAL_ERAs=("${ERAs_RUN3[@]}")
        COMBINED_ERAs=("Run3")
        ;;
    all)
        INDIVIDUAL_ERAs=("${ERAs_RUN2[@]}" "${ERAs_RUN3[@]}")
        COMBINED_ERAs=("Run2" "Run3" "All")
        ;;
esac

# ---------------------------------------------------------------------------
# Resolve mass points
# ---------------------------------------------------------------------------
if [[ "$METHOD" == "Baseline" ]]; then
    MASSPOINTs=("${MASSPOINTs_BASELINE[@]}")
else
    MASSPOINTs=("${MASSPOINTs_PARTICLENET[@]}")
fi

# ---------------------------------------------------------------------------
# Build blinding flags to forward to sub-commands
# ---------------------------------------------------------------------------
BLIND_FLAGS=""
if [[ "$UNBLIND" == true ]]; then
    BLIND_FLAGS="--unblind"
elif [[ "$PARTIAL_UNBLIND" == true ]]; then
    BLIND_FLAGS="--partial-unblind"
fi

# ---------------------------------------------------------------------------
# Helper: run or dry-run a command
# ---------------------------------------------------------------------------
run_cmd() {
    local cmd="$1"
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] $cmd"
    else
        eval "$cmd"
    fi
}

# ---------------------------------------------------------------------------
# Change to project directory (scripts use relative paths)
# ---------------------------------------------------------------------------
cd "$PROJECT_DIR"

echo "============================================================"
echo "SignalRegionStudyV2 CnC Asymptotic Limit Pipeline"
echo "  Mode:    $MODE"
echo "  Method:  $METHOD"
echo "  Binning: $BINNING"
echo "  Flags:   ${BLIND_FLAGS:-(none)}"
echo "  Dry-run: $DRY_RUN"
echo "============================================================"

# ===========================================================================
# Per-mass-point pipeline
# ===========================================================================
for masspoint in "${MASSPOINTs[@]}"; do
    echo ""
    echo "------------------------------------------------------------"
    echo "Processing: $masspoint"
    echo "------------------------------------------------------------"

    # -----------------------------------------------------------------------
    # Step 1: printDatacardCnC.py for each (era, channel)
    # -----------------------------------------------------------------------
    echo ""
    echo "[Step 1] Generate CnC datacards..."
    for era in "${INDIVIDUAL_ERAs[@]}"; do
        for channel in "${CHANNELs[@]}"; do
            echo "  printDatacardCnC: era=$era, channel=$channel, masspoint=$masspoint"
            run_cmd "python3 python/printDatacardCnC.py \
                --era \"$era\" \
                --channel \"$channel\" \
                --masspoint \"$masspoint\" \
                --method \"$METHOD\" \
                --binning \"$BINNING\" \
                --nsigma \"$NSIGMA\" \
                $BLIND_FLAGS"
        done
    done

    # -----------------------------------------------------------------------
    # Step 2: combineDatacardsCnC.py --mode channel per era
    # -----------------------------------------------------------------------
    echo ""
    echo "[Step 2] Combine channels (SR1E2Mu+SR3Mu -> Combined) per era..."
    for era in "${INDIVIDUAL_ERAs[@]}"; do
        echo "  combineDatacardsCnC (channel): era=$era, masspoint=$masspoint"
        run_cmd "python3 python/combineDatacardsCnC.py \
            --mode channel \
            --era \"$era\" \
            --masspoint \"$masspoint\" \
            --method \"$METHOD\" \
            --binning \"$BINNING\" \
            --nsigma \"$NSIGMA\" \
            $BLIND_FLAGS"
    done

    # -----------------------------------------------------------------------
    # Step 3: runAsymptoticCnC.sh on Combined per individual era
    # -----------------------------------------------------------------------
    echo ""
    echo "[Step 3] Run asymptotic on individual eras..."
    for era in "${INDIVIDUAL_ERAs[@]}"; do
        echo "  runAsymptoticCnC: era=$era, channel=Combined, masspoint=$masspoint"
        DRY_RUN_FLAG=""
        [[ "$DRY_RUN" == true ]] && DRY_RUN_FLAG="--dry-run"
        run_cmd "bash scripts/runAsymptoticCnC.sh \
            --era \"$era\" \
            --channel Combined \
            --masspoint \"$masspoint\" \
            --method \"$METHOD\" \
            --binning \"$BINNING\" \
            --nsigma \"$NSIGMA\" \
            $BLIND_FLAGS \
            $DRY_RUN_FLAG"
    done

    # -----------------------------------------------------------------------
    # Step 4 & 5: combine eras and run asymptotic on combined eras
    # -----------------------------------------------------------------------
    if [[ "$MODE" == "run2" || "$MODE" == "all" ]]; then
        echo ""
        echo "[Step 4] Combine Run2 eras..."
        run2_eras_csv=$(IFS=','; echo "${ERAs_RUN2[*]}")
        run_cmd "python3 python/combineDatacardsCnC.py \
            --mode era \
            --channel Combined \
            --eras \"$run2_eras_csv\" \
            --output-era Run2 \
            --masspoint \"$masspoint\" \
            --method \"$METHOD\" \
            --binning \"$BINNING\" \
            --nsigma \"$NSIGMA\" \
            $BLIND_FLAGS"

        echo ""
        echo "[Step 5] Run asymptotic on Run2 Combined..."
        DRY_RUN_FLAG=""
        [[ "$DRY_RUN" == true ]] && DRY_RUN_FLAG="--dry-run"
        run_cmd "bash scripts/runAsymptoticCnC.sh \
            --era Run2 \
            --channel Combined \
            --masspoint \"$masspoint\" \
            --method \"$METHOD\" \
            --binning \"$BINNING\" \
            --nsigma \"$NSIGMA\" \
            $BLIND_FLAGS \
            $DRY_RUN_FLAG"
    fi

    if [[ "$MODE" == "run3" || "$MODE" == "all" ]]; then
        echo ""
        echo "[Step 4] Combine Run3 eras..."
        run3_eras_csv=$(IFS=','; echo "${ERAs_RUN3[*]}")
        run_cmd "python3 python/combineDatacardsCnC.py \
            --mode era \
            --channel Combined \
            --eras \"$run3_eras_csv\" \
            --output-era Run3 \
            --masspoint \"$masspoint\" \
            --method \"$METHOD\" \
            --binning \"$BINNING\" \
            --nsigma \"$NSIGMA\" \
            $BLIND_FLAGS"

        echo ""
        echo "[Step 5] Run asymptotic on Run3 Combined..."
        DRY_RUN_FLAG=""
        [[ "$DRY_RUN" == true ]] && DRY_RUN_FLAG="--dry-run"
        run_cmd "bash scripts/runAsymptoticCnC.sh \
            --era Run3 \
            --channel Combined \
            --masspoint \"$masspoint\" \
            --method \"$METHOD\" \
            --binning \"$BINNING\" \
            --nsigma \"$NSIGMA\" \
            $BLIND_FLAGS \
            $DRY_RUN_FLAG"
    fi

    if [[ "$MODE" == "all" ]]; then
        echo ""
        echo "[Step 4] Combine All eras..."
        all_eras_csv=$(IFS=','; echo "${ERAs_RUN2[*]},${ERAs_RUN3[*]}")
        run_cmd "python3 python/combineDatacardsCnC.py \
            --mode era \
            --channel Combined \
            --eras \"$all_eras_csv\" \
            --output-era All \
            --masspoint \"$masspoint\" \
            --method \"$METHOD\" \
            --binning \"$BINNING\" \
            --nsigma \"$NSIGMA\" \
            $BLIND_FLAGS"

        echo ""
        echo "[Step 5] Run asymptotic on All Combined..."
        DRY_RUN_FLAG=""
        [[ "$DRY_RUN" == true ]] && DRY_RUN_FLAG="--dry-run"
        run_cmd "bash scripts/runAsymptoticCnC.sh \
            --era All \
            --channel Combined \
            --masspoint \"$masspoint\" \
            --method \"$METHOD\" \
            --binning \"$BINNING\" \
            --nsigma \"$NSIGMA\" \
            $BLIND_FLAGS \
            $DRY_RUN_FLAG"
    fi

done  # end mass points loop

# ===========================================================================
# Steps 6 & 7: collect and plot limits for each combined era
# ===========================================================================
echo ""
echo "============================================================"
echo "Collecting and plotting CnC limits..."
echo "============================================================"

# Determine which combined eras + individual eras to collect/plot
COLLECT_ERAs=("${COMBINED_ERAs[@]}" "${INDIVIDUAL_ERAs[@]}")

COLLECT_FLAGS="--cnc --nsigma $NSIGMA"
PLOT_FLAGS="--cnc --nsigma $NSIGMA"
if [[ "$UNBLIND" == true ]]; then
    COLLECT_FLAGS="$COLLECT_FLAGS --unblind"
    PLOT_FLAGS="$PLOT_FLAGS --unblind"
else
    PLOT_FLAGS="$PLOT_FLAGS --blind"
fi

echo ""
echo "[Step 6] Collecting CnC limits..."
for era in "${COLLECT_ERAs[@]}"; do
    echo "  collectLimits: era=$era, method=$METHOD"
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] python3 python/collectLimits.py --era \"$era\" --method \"$METHOD\" $COLLECT_FLAGS"
    else
        python3 python/collectLimits.py --era "$era" --method "$METHOD" $COLLECT_FLAGS
    fi
done

echo ""
echo "[Step 7] Plotting CnC limits..."
for era in "${COLLECT_ERAs[@]}"; do
    echo "  plotLimits: era=$era, method=$METHOD"
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] python3 python/plotLimits.py --era \"$era\" --method \"$METHOD\" --limit_type Asymptotic $PLOT_FLAGS"
    else
        python3 python/plotLimits.py --era "$era" --method "$METHOD" --limit_type Asymptotic $PLOT_FLAGS
    fi
done

echo ""
echo "============================================================"
echo "CnC asymptotic pipeline complete!"
echo "Output saved to: results/plots/"
echo "============================================================"
