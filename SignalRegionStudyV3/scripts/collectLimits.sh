#!/bin/bash
set -euo pipefail

# V3 limit outputs are run-period component likelihoods, not per-subera
# combineCards products.
ERAs=("Run2" "Run3" "All")

# Per-channel asymptotic results, when produced, also live at run-period level.
PER_CHANNEL_ERAs=("Run2" "Run3" "All")
CHANNELs=("SR1E2Mu" "SR3Mu")

# Output modes: BR = relative branching ratio (default), xsec = sigma(pp->ttbar) x B_sig in fb.
MODES=("BR" "xsec")

# Per-MHc stitched ParticleNet plots: Baseline off-Z regions + ParticleNet on-Z region.
PARTICLENET_MHCS=(100 115 130 145 160)
PARTICLENET_MHCS_STR="${PARTICLENET_MHCS[*]}"

# Parallel slot count (override with COLLECTLIMITS_JOBS=N).
JOBS="${COLLECTLIMITS_JOBS:-4}"

UNBLIND=false
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --unblind)
            UNBLIND=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--unblind] [--dry-run] [-j N | --jobs N]"
            echo ""
            echo "  --unblind     Read from templates/.../extended_unblind/ and emit"
            echo "                limits.{era}.Asymptotic.Baseline.unblind.json plus"
            echo "                limit.{era}.Asymptotic.Baseline.*.unblind.png. Observed"
            echo "                limit is drawn (--blind is dropped)."
            echo "  --dry-run     Print discovered V3 tasks without collecting or plotting."
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

BINNING_SUFFIX="extended"
if $UNBLIND; then
    BINNING_SUFFIX="extended_unblind"
fi

has_outputs() {
    local era="$1" channel="$2" method="$3"
    compgen -G "templates/${era}/${channel}/*/${method}/${BINNING_SUFFIX}/combine_output/asymptotic/higgsCombine*.AsymptoticLimits.mH120.root" >/dev/null
}

json_path_for() {
    local mode="$1" era="$2" channel="$3" method="$4"
    local unblind_suffix="" ch_suffix=""
    [[ "$UNBLIND" == true ]] && unblind_suffix=".unblind"
    [[ "$channel" != "Combined" ]] && ch_suffix=".${channel}"
    printf 'results/json/%s/%s/limits.%s%s.Asymptotic.%s%s.json' \
        "$mode" "$era" "$era" "$ch_suffix" "$method" "$unblind_suffix"
}

available_mhcs_from_json() {
    local json_path="$1"
    python3 -c 'import json, re, sys
with open(sys.argv[1]) as f:
    data = json.load(f)
vals = sorted({int(m.group(1)) for mp in data for m in [re.match(r"MHc(\d+)_MA\d+$", mp)] if m})
print(" ".join(map(str, vals)))' "$json_path"
}

pnet_json_has_mhc() {
    local json_path="$1" mhc="$2"
    python3 -c 'import json, sys
path, mhc = sys.argv[1], sys.argv[2]
with open(path) as f:
    data = json.load(f)
sys.exit(0 if any(mp.startswith(f"MHc{mhc}_") for mp in data) else 1)' "$json_path" "$mhc"
}

run_pipeline() {
    local mode="$1" era="$2" channel="$3"

    # Reconstruct flag arrays from exported strings (parallel runs in a child shell).
    local collect_flags=() plot_flags=() mhcs=() pnet_mhcs=() channel_arg=()
    [[ -n "$COLLECT_FLAGS_STR" ]] && read -r -a collect_flags <<< "$COLLECT_FLAGS_STR"
    [[ -n "$PLOT_FLAGS_STR" ]]    && read -r -a plot_flags    <<< "$PLOT_FLAGS_STR"
    [[ -n "$PARTICLENET_MHCS_STR" ]] && read -r -a pnet_mhcs <<< "$PARTICLENET_MHCS_STR"
    [[ "$channel" != "Combined" ]] && channel_arg=(--channel "$channel")

    if has_outputs "$era" "$channel" "Baseline"; then
        python3 python/collectLimits.py --era "$era" --method Baseline --limit_type Asymptotic \
            --mode "$mode" "${channel_arg[@]}" --available-only "${collect_flags[@]}"

        local baseline_json
        baseline_json="$(json_path_for "$mode" "$era" "$channel" "Baseline")"
        read -r -a mhcs <<< "$(available_mhcs_from_json "$baseline_json")"

        # Per-MHc Brazilian band plots for only the MHc values present in the JSON.
        for mhc in "${mhcs[@]}"; do
            python3 python/plotLimits.py --era "$era" --method Baseline --limit_type Asymptotic \
                --mode "$mode" "${channel_arg[@]}" --mhc "$mhc" "${plot_flags[@]}"
        done

        # Median-expected-only overlay across available MHc values.
        if (( ${#mhcs[@]} > 1 )); then
            local mhc_csv
            mhc_csv="$(IFS=,; echo "${mhcs[*]}")"
            python3 python/plotLimits.py --era "$era" --method Baseline --limit_type Asymptotic \
                --mode "$mode" "${channel_arg[@]}" --compare-mhc --mhc-list "$mhc_csv" "${plot_flags[@]}"
        fi
    else
        echo "Skipping ${mode}/${era}/${channel}/Baseline: no ${BINNING_SUFFIX} AsymptoticLimits ROOT files"
    fi

    if has_outputs "$era" "$channel" "ParticleNet"; then
        python3 python/collectLimits.py --era "$era" --method ParticleNet --limit_type Asymptotic \
            --mode "$mode" "${channel_arg[@]}" --available-only "${collect_flags[@]}"

        local baseline_json pnet_json
        baseline_json="$(json_path_for "$mode" "$era" "$channel" "Baseline")"
        pnet_json="$(json_path_for "$mode" "$era" "$channel" "ParticleNet")"
        if [[ -s "$baseline_json" && -s "$pnet_json" ]]; then
            # ParticleNet trained points overlaid with Baseline.
            python3 python/plotLimits.py --era "$era" --method ParticleNet --limit_type Asymptotic \
                --mode "$mode" "${channel_arg[@]}" --stack_baseline "${plot_flags[@]}"

            for mhc in "${pnet_mhcs[@]}"; do
                if pnet_json_has_mhc "$pnet_json" "$mhc"; then
                    python3 python/plotLimits.py --era "$era" --method ParticleNet --limit_type Asymptotic \
                        --mode "$mode" "${channel_arg[@]}" --mhc "$mhc" --stack_baseline "${plot_flags[@]}"
                else
                    echo "Skipping ${mode}/${era}/${channel}/ParticleNet MHc${mhc}: no ParticleNet points in JSON"
                fi
            done
        else
            echo "Skipping ${mode}/${era}/${channel}/ParticleNet plot: missing Baseline or ParticleNet JSON"
        fi
    else
        echo "Skipping ${mode}/${era}/${channel}/ParticleNet: no ${BINNING_SUFFIX} AsymptoticLimits ROOT files"
    fi
}
export -f run_pipeline
export -f has_outputs json_path_for available_mhcs_from_json pnet_json_has_mhc
export COLLECT_FLAGS_STR PLOT_FLAGS_STR BINNING_SUFFIX UNBLIND PARTICLENET_MHCS_STR

build_tasks() {
    for mode in "${MODES[@]}"; do
        for era in "${ERAs[@]}"; do
            if has_outputs "$era" "Combined" "Baseline" || has_outputs "$era" "Combined" "ParticleNet"; then
                echo "$mode $era Combined"
            fi
        done
        for era in "${PER_CHANNEL_ERAs[@]}"; do
            for ch in "${CHANNELs[@]}"; do
                if has_outputs "$era" "$ch" "Baseline" || has_outputs "$era" "$ch" "ParticleNet"; then
                    echo "$mode $era $ch"
                fi
            done
        done
    done
}

mapfile -t TASKS < <(build_tasks)

if (( ${#TASKS[@]} == 0 )); then
    echo "No ${BINNING_SUFFIX} AsymptoticLimits ROOT files found under templates/."
    exit 1
fi

printf 'Discovered %d V3 limit task(s):\n' "${#TASKS[@]}"
printf '  %s\n' "${TASKS[@]}"

if $DRY_RUN; then
    exit 0
fi

if command -v parallel >/dev/null 2>&1; then
    printf '%s\n' "${TASKS[@]}" | parallel --colsep ' ' -j "$JOBS" --halt soon,fail=1 --line-buffer --tagstring '[{1} {2} {3}]' run_pipeline {1} {2} {3}
else
    echo "GNU parallel not found; running sequentially."
    for task in "${TASKS[@]}"; do
        read -r mode era channel <<< "$task"
        run_pipeline "$mode" "$era" "$channel"
    done
fi
