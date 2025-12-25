#!/bin/bash
#
# runCombineWrapper.sh - Main orchestration script for Combine analysis
#
# Runs combine analysis across multiple eras, channels, and mass points.
#
# Usage:
#   ./runCombineWrapper.sh --action asymptotic --method Baseline --binning uniform
#   ./runCombineWrapper.sh --action all --masspoints MHc130_MA90 --parallel 4
#

set -e

# Default values
ACTION=""
ERAS="all"
CHANNELS="all"
MASSPOINTS="all"
METHOD="Baseline"
BINNING="uniform"
PARALLEL=4
DRY_RUN=false
VERBOSE=false
CONDOR=false

# Mass point definitions
BASELINE_MASSPOINTS=(
    "MHc70_MA15" "MHc70_MA40" "MHc70_MA65"
    "MHc100_MA15" "MHc100_MA60" "MHc100_MA95"
    "MHc130_MA15" "MHc130_MA55" "MHc130_MA90" "MHc130_MA125"
    "MHc160_MA15" "MHc160_MA85" "MHc160_MA120" "MHc160_MA155"
)

PARTICLENET_MASSPOINTS=(
    "MHc100_MA95" "MHc130_MA90" "MHc160_MA85"
)

RUN2_ERAS=("2016preVFP" "2016postVFP" "2017" "2018")
RUN3_ERAS=("2022" "2022EE" "2023" "2023BPix")
ALL_ERAS=("${RUN2_ERAS[@]}" "${RUN3_ERAS[@]}")

ALL_CHANNELS=("SR1E2Mu" "SR3Mu")

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --action)
            ACTION="$2"
            shift 2
            ;;
        --eras)
            ERAS="$2"
            shift 2
            ;;
        --channels)
            CHANNELS="$2"
            shift 2
            ;;
        --masspoints)
            MASSPOINTS="$2"
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
        --parallel)
            PARALLEL="$2"
            shift 2
            ;;
        --condor)
            CONDOR=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 --action ACTION [OPTIONS]"
            echo ""
            echo "Actions:"
            echo "  asymptotic   Run AsymptoticLimits"
            echo "  hybridnew    Run HybridNew with toys"
            echo "  injection    Run signal injection tests"
            echo "  fitdiag      Run FitDiagnostics"
            echo "  impacts      Run impact calculation"
            echo "  combine      Combine datacards (channels/eras)"
            echo "  collect      Collect limits to JSON"
            echo "  plot         Generate limit plots"
            echo "  all          Run full analysis chain"
            echo ""
            echo "Options:"
            echo "  --eras       Comma-separated eras or 'all', 'run2', 'run3' [default: all]"
            echo "  --channels   Comma-separated channels or 'all' [default: all]"
            echo "  --masspoints Comma-separated masspoints or 'all', 'baseline', 'particlenet' [default: all]"
            echo "  --method     Template method (Baseline, ParticleNet) [default: Baseline]"
            echo "  --binning    Binning scheme (uniform, sigma) [default: uniform]"
            echo "  --parallel   Number of parallel jobs [default: 4]"
            echo "  --condor     Use HTCondor for heavy jobs"
            echo "  --dry-run    Print commands without executing"
            echo "  --verbose    Enable verbose output"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate action
if [[ -z "$ACTION" ]]; then
    echo "ERROR: --action is required"
    exit 1
fi

# Get WORKDIR
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(dirname "$(dirname "$SCRIPT_DIR")")"

# Resolve era list
resolve_eras() {
    case "$ERAS" in
        all)
            echo "${ALL_ERAS[@]}"
            ;;
        run2)
            echo "${RUN2_ERAS[@]}"
            ;;
        run3)
            echo "${RUN3_ERAS[@]}"
            ;;
        *)
            echo "$ERAS" | tr ',' ' '
            ;;
    esac
}

# Resolve channel list
resolve_channels() {
    case "$CHANNELS" in
        all)
            echo "${ALL_CHANNELS[@]}"
            ;;
        *)
            echo "$CHANNELS" | tr ',' ' '
            ;;
    esac
}

# Resolve mass point list
resolve_masspoints() {
    case "$MASSPOINTS" in
        all)
            if [[ "$METHOD" == "ParticleNet" ]]; then
                echo "${PARTICLENET_MASSPOINTS[@]}"
            else
                echo "${BASELINE_MASSPOINTS[@]}"
            fi
            ;;
        baseline)
            echo "${BASELINE_MASSPOINTS[@]}"
            ;;
        particlenet)
            echo "${PARTICLENET_MASSPOINTS[@]}"
            ;;
        *)
            echo "$MASSPOINTS" | tr ',' ' '
            ;;
    esac
}

# Get resolved lists
ERA_LIST=($(resolve_eras))
CHANNEL_LIST=($(resolve_channels))
MASSPOINT_LIST=($(resolve_masspoints))

echo "=========================================="
echo "Combine Analysis Wrapper"
echo "=========================================="
echo "Action: $ACTION"
echo "Method: $METHOD"
echo "Binning: $BINNING"
echo "Eras: ${ERA_LIST[*]}"
echo "Channels: ${CHANNEL_LIST[*]}"
echo "Masspoints: ${MASSPOINT_LIST[*]}"
echo "Parallel: $PARALLEL"
echo "Condor: $CONDOR"
echo "=========================================="
echo ""

# Build job list
build_job_list() {
    local script_name="$1"
    local extra_args="$2"

    for era in "${ERA_LIST[@]}"; do
        for channel in "${CHANNEL_LIST[@]}"; do
            for masspoint in "${MASSPOINT_LIST[@]}"; do
                template_dir="${WORKDIR}/SignalRegionStudyV1/templates/${era}/${channel}/${masspoint}/${METHOD}/${BINNING}"
                if [[ -d "$template_dir" ]]; then
                    echo "${SCRIPT_DIR}/${script_name} --era ${era} --channel ${channel} --masspoint ${masspoint} --method ${METHOD} --binning ${BINNING} ${extra_args}"
                else
                    echo "# SKIP: Template not found: ${template_dir}" >&2
                fi
            done
        done
    done
}

# Run action
run_action() {
    local action="$1"

    case "$action" in
        asymptotic)
            echo "Running AsymptoticLimits..."
            JOBS=$(build_job_list "runAsymptotic.sh" "")
            ;;
        hybridnew)
            echo "Running HybridNew..."
            EXTRA=""
            [[ "$CONDOR" == true ]] && EXTRA="--condor"
            JOBS=$(build_job_list "runHybridNew.sh" "$EXTRA")
            ;;
        injection)
            echo "Running Signal Injection..."
            JOBS=$(build_job_list "runSignalInjection.sh" "")
            ;;
        fitdiag)
            echo "Running FitDiagnostics..."
            JOBS=$(build_job_list "runFitDiagnostics.sh" "")
            ;;
        impacts)
            echo "Running Impacts..."
            EXTRA=""
            [[ "$CONDOR" == true ]] && EXTRA="--condor"
            JOBS=$(build_job_list "runImpacts.sh" "$EXTRA")
            ;;
        combine)
            echo "Combining datacards..."
            # First combine channels
            for era in "${ERA_LIST[@]}"; do
                for masspoint in "${MASSPOINT_LIST[@]}"; do
                    echo "  Combining channels for ${era}/${masspoint}..."
                    if [[ "$DRY_RUN" == false ]]; then
                        python3 "${WORKDIR}/SignalRegionStudyV1/python/combineDatacards.py" \
                            --mode channel \
                            --era "$era" \
                            --masspoint "$masspoint" \
                            --method "$METHOD" \
                            --binning "$BINNING" || true
                    fi
                done
            done

            # Then combine eras for FullRun2
            echo "  Combining Run2 eras..."
            for channel in "${CHANNEL_LIST[@]}" "Combined"; do
                for masspoint in "${MASSPOINT_LIST[@]}"; do
                    if [[ "$DRY_RUN" == false ]]; then
                        python3 "${WORKDIR}/SignalRegionStudyV1/python/combineDatacards.py" \
                            --mode era \
                            --eras "2016preVFP,2016postVFP,2017,2018" \
                            --channel "$channel" \
                            --masspoint "$masspoint" \
                            --method "$METHOD" \
                            --binning "$BINNING" \
                            --output-era "FullRun2" || true
                    fi
                done
            done
            return
            ;;
        collect)
            echo "Collecting limits..."
            for era in "${ERA_LIST[@]}" "FullRun2"; do
                for channel in "${CHANNEL_LIST[@]}" "Combined"; do
                    echo "  Collecting ${era}/${channel}..."
                    if [[ "$DRY_RUN" == false ]]; then
                        python3 "${WORKDIR}/SignalRegionStudyV1/python/collectLimits.py" \
                            --era "$era" \
                            --channel "$channel" \
                            --method "$METHOD" \
                            --binning "$BINNING" \
                            --limit-type Asymptotic || true
                    fi
                done
            done
            return
            ;;
        plot)
            echo "Generating limit plots..."
            for era in "${ERA_LIST[@]}" "FullRun2"; do
                for channel in "${CHANNEL_LIST[@]}" "Combined"; do
                    echo "  Plotting ${era}/${channel}..."
                    if [[ "$DRY_RUN" == false ]]; then
                        python3 "${WORKDIR}/SignalRegionStudyV1/python/plotLimits.py" \
                            --era "$era" \
                            --channel "$channel" \
                            --method "$METHOD" \
                            --binning "$BINNING" || true
                    fi
                done
            done
            return
            ;;
        all)
            echo "Running full analysis chain..."
            run_action "asymptotic"
            run_action "impacts"
            run_action "fitdiag"
            run_action "combine"
            run_action "collect"
            run_action "plot"
            return
            ;;
        *)
            echo "ERROR: Unknown action: $action"
            exit 1
            ;;
    esac

    # Execute jobs
    if [[ -n "$JOBS" ]]; then
        if [[ "$DRY_RUN" == true ]]; then
            echo "$JOBS"
        else
            if command -v parallel &> /dev/null && [[ "$PARALLEL" -gt 1 ]]; then
                echo "$JOBS" | parallel -j "$PARALLEL" --progress
            else
                echo "$JOBS" | while read -r cmd; do
                    [[ -n "$cmd" ]] && eval "$cmd"
                done
            fi
        fi
    fi
}

# Run the requested action
run_action "$ACTION"

echo ""
echo "Done."
