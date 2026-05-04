#!/bin/bash
set -euo pipefail

ERAs=("2016preVFP" "2016postVFP" "2017" "2018"
      "2022" "2022EE" "2023" "2023BPix"
      "Run2" "Run3" "All")

# Per-channel asymptotic results exist only at run-period level
# (combine_era_${ch}_${run} → asymptotic_${ch}_${run} in makeBinnedTemplates DAG).
PER_CHANNEL_ERAs=("Run2" "Run3" "All")
CHANNELs=("SR1E2Mu" "SR3Mu")

# MHc values for fixed-MHc per-mA plots (covers all rows in baseline mass-point list).
MHC_LIST="70 85 100 115 130 145 160"
# MHc values overlaid in the median-only comparison plot.
COMPARE_MHC_LIST="70,85,100,115,130,145,160"

# Output modes: BR = relative branching ratio (default), xsec = sigma(pp->ttbar) x B_sig in fb.
MODES=("BR" "xsec")

# Parallel slot count (override with COLLECTLIMITS_JOBS=N).
JOBS="${COLLECTLIMITS_JOBS:-4}"

UNBLIND=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --unblind)
            UNBLIND=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--unblind] [-j N | --jobs N]"
            echo ""
            echo "  --unblind     Read from templates/.../extended_unblind/ and emit"
            echo "                limits.{era}.Asymptotic.Baseline.unblind.json plus"
            echo "                limit.{era}.Asymptotic.Baseline.*.unblind.png. Observed"
            echo "                limit is drawn (--blind is dropped)."
            echo "  -j, --jobs N  Number of parallel pipeline jobs (default: 4, env: COLLECTLIMITS_JOBS)."
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if $UNBLIND; then
    COLLECT_FLAGS_STR="--unblind"
    PLOT_FLAGS_STR="--unblind"
else
    COLLECT_FLAGS_STR=""
    PLOT_FLAGS_STR="--blind"
fi

run_pipeline() {
    local mode="$1" era="$2" channel="$3"

    # Reconstruct flag arrays from exported strings (parallel runs in a child shell).
    local collect_flags=() plot_flags=() mhcs=() channel_arg=()
    [[ -n "$COLLECT_FLAGS_STR" ]] && read -r -a collect_flags <<< "$COLLECT_FLAGS_STR"
    [[ -n "$PLOT_FLAGS_STR" ]]    && read -r -a plot_flags    <<< "$PLOT_FLAGS_STR"
    read -r -a mhcs <<< "$MHC_LIST"
    [[ "$channel" != "Combined" ]] && channel_arg=(--channel "$channel")

    python3 python/collectLimits.py --era "$era" --method Baseline    --limit_type Asymptotic --mode "$mode" "${channel_arg[@]}" "${collect_flags[@]}"
    python3 python/collectLimits.py --era "$era" --method ParticleNet --limit_type Asymptotic --mode "$mode" "${channel_arg[@]}" "${collect_flags[@]}"

    # Per-MHc Brazilian band plots
    for mhc in "${mhcs[@]}"; do
        python3 python/plotLimits.py --era "$era" --method Baseline --limit_type Asymptotic \
            --mode "$mode" "${channel_arg[@]}" --mhc "$mhc" "${plot_flags[@]}"
    done

    # Median-expected-only overlay across MHc values
    python3 python/plotLimits.py --era "$era" --method Baseline --limit_type Asymptotic \
        --mode "$mode" "${channel_arg[@]}" --compare-mhc --mhc-list "$COMPARE_MHC_LIST" "${plot_flags[@]}"

    # ParticleNet (3 trained mass points) overlay on Baseline.
    python3 python/plotLimits.py --era "$era" --method ParticleNet --limit_type Asymptotic \
        --mode "$mode" "${channel_arg[@]}" --stack_baseline "${plot_flags[@]}"
}
export -f run_pipeline
export COLLECT_FLAGS_STR PLOT_FLAGS_STR MHC_LIST COMPARE_MHC_LIST

# Build (mode, era, channel) tuples and dispatch via GNU parallel.
{
    for mode in "${MODES[@]}"; do
        for era in "${ERAs[@]}"; do
            echo "$mode $era Combined"
        done
        for era in "${PER_CHANNEL_ERAs[@]}"; do
            for ch in "${CHANNELs[@]}"; do
                echo "$mode $era $ch"
            done
        done
    done
} | parallel --colsep ' ' -j "$JOBS" --halt soon,fail=1 --line-buffer --tagstring '[{1} {2} {3}]' run_pipeline {1} {2} {3}
