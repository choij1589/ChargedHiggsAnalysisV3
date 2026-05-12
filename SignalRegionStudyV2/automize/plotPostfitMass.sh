#!/bin/bash
#
# plotPostfitMass.sh - Batch real-mass post-fit plots (fine 1-GeV bins).
#
# Wraps python/plotPostfitMass.py over all mass points. By default it uses the
# All/Combined fit result and lets the Python script render every applicable
# era scope and channel scope.
#
# Usage:
#   # Default: All fit, all era scopes, all channel scopes, both fit types.
#   ./plotPostfitMass.sh --method ParticleNet --partial-unblind
#
#   # Run on a different fit source / era-scope / channel-scope:
#   ./plotPostfitMass.sh --method Baseline --unblind --era Run2 --era-scope 2018 --channel-scope SR1E2Mu
#
#   # Re-render plots from cached fine-mass hists (style edits only):
#   ./plotPostfitMass.sh --method ParticleNet --partial-unblind --plot-only
#
#   # Submit as HTCondor jobs (one per masspoint):
#   ./plotPostfitMass.sh --method ParticleNet --partial-unblind --condor
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/load_masspoints.sh"

# Defaults chosen so the shell produces the full plot grid from the All fit.
ERA="All"
ERA_SCOPE=""
CHANNEL_SCOPE=""
METHOD="Baseline"
BINNING="extended"
NUISANCE="fallback_lnn"
FIT_TYPE="both"
BIN_WIDTH="auto"
PARTIAL_UNBLIND=false
UNBLIND=false
BLIND=false
PLOT_ONLY=false
CONDOR=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --era)             ERA="$2"; shift 2 ;;
        --era-scope)       ERA_SCOPE="$2"; shift 2 ;;
        --channel|--channel-scope) CHANNEL_SCOPE="$2"; shift 2 ;;
        --method)          METHOD="$2"; shift 2 ;;
        --binning)         BINNING="$2"; shift 2 ;;
        --nuisance)        NUISANCE="$2"; shift 2 ;;
        --fit-type)        FIT_TYPE="$2"; shift 2 ;;
        --bin-width)       BIN_WIDTH="$2"; shift 2 ;;
        --partial-unblind) PARTIAL_UNBLIND=true; shift ;;
        --unblind)         UNBLIND=true; shift ;;
        --blind)           BLIND=true; shift ;;
        --plot-only)       PLOT_ONLY=true; shift ;;
        --condor)          CONDOR=true; shift ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --help|-h)
            cat <<EOF
Usage: $0 [OPTIONS]

Required (one of):
  --partial-unblind   Use partial-unblind templates (real sideband data)
  --unblind           Use fully unblinded templates
  --blind             Use blinded templates (Asimov data = sum of pre-fit backgrounds)

Defaults (produce all applicable era/channel scopes):
  --era All           Fit source
  --era-scope SCOPE   Optional era slice within the fit
  --channel CHANNEL   Optional channel slice (alias: --channel-scope)

Template options:
  --method METHOD     Baseline or ParticleNet [default: Baseline]
  --binning BINNING   extended or uniform     [default: extended]
  --nuisance MODE     fallback_lnn (default) or preserve_shape
  --fit-type T        b | s | both            [default: both]
  --bin-width W       Fine-grid bin width     [default: auto]

Execution:
  --condor            Submit one HTCondor job per masspoint
  --plot-only         Re-render plots from cached fine-mass hists (fast)
  --dry-run           Print commands without executing
EOF
            exit 0
            ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

CHOSEN=0
[[ "$PARTIAL_UNBLIND" == true ]] && CHOSEN=$((CHOSEN + 1))
[[ "$UNBLIND"         == true ]] && CHOSEN=$((CHOSEN + 1))
[[ "$BLIND"           == true ]] && CHOSEN=$((CHOSEN + 1))
if [[ "$CHOSEN" -ne 1 ]]; then
    echo "ERROR: must specify exactly one of --blind / --partial-unblind / --unblind" >&2
    exit 1
fi
case "$NUISANCE" in
    fallback_lnn|preserve_shape) ;;
    *)
        echo "ERROR: Invalid --nuisance value '$NUISANCE'" >&2
        echo "Valid values: fallback_lnn, preserve_shape" >&2
        exit 1
        ;;
esac

if [[ "$METHOD" == "ParticleNet" ]]; then
    if [[ "$UNBLIND" == true ]]; then
        MASSPOINTs=("${MASSPOINTs_UNBLIND_PN[@]}")
    else
        MASSPOINTs=("${MASSPOINTs_PARTICLENET[@]}")
    fi
else
    if [[ "$UNBLIND" == true ]]; then
        MASSPOINTs=("${MASSPOINTs_UNBLIND_BASELINE[@]}")
    else
        MASSPOINTs=("${MASSPOINTs_BASELINE[@]}")
    fi
fi

# Common extra args (same order for condor and local runs)
EXTRA_ARGS=(
    --era "${ERA}"
    --method "${METHOD}"
    --binning "${BINNING}"
    --nuisance "${NUISANCE}"
    --fit-type "${FIT_TYPE}"
    --bin-width "${BIN_WIDTH}"
)
[[ -n "$ERA_SCOPE" ]] && EXTRA_ARGS+=(--era-scope "${ERA_SCOPE}")
[[ -n "$CHANNEL_SCOPE" ]] && EXTRA_ARGS+=(--channel-scope "${CHANNEL_SCOPE}")
[[ "$PARTIAL_UNBLIND" == true ]] && EXTRA_ARGS+=(--partial-unblind)
[[ "$UNBLIND"         == true ]] && EXTRA_ARGS+=(--unblind)
[[ "$BLIND"           == true ]] && EXTRA_ARGS+=(--blind)
[[ "$PLOT_ONLY"       == true ]] && EXTRA_ARGS+=(--plot-only)

echo "============================================================"
echo "SignalRegionStudyV2 Real-Mass Post-fit Plot Batch"
echo "Era (fit):       $ERA"
echo "Era scope:       ${ERA_SCOPE:-all}"
echo "Channel scope:   ${CHANNEL_SCOPE:-all}"
echo "Method:          $METHOD"
echo "Binning:         $BINNING"
echo "Nuisance mode:   $NUISANCE"
echo "Fit type:        $FIT_TYPE"
echo "Bin width:       $BIN_WIDTH GeV"
echo "Mass points:     ${#MASSPOINTs[@]} total"
echo "Partial-unblind: $PARTIAL_UNBLIND"
echo "Unblind:         $UNBLIND"
echo "Blind:           $BLIND"
echo "Plot only:       $PLOT_ONLY"
echo "Execution:       $([[ "$CONDOR" == true ]] && echo HTCondor || echo Local)"
echo "Dry run:         $DRY_RUN"
echo "============================================================"

run_local() {
    for mp in "${MASSPOINTs[@]}"; do
        echo ">>> python3 plotPostfitMass.py --masspoint ${mp} ${EXTRA_ARGS[*]}"
        if [[ "$DRY_RUN" != true ]]; then
            python3 "${SCRIPT_DIR}/python/plotPostfitMass.py" \
                --masspoint "${mp}" "${EXTRA_ARGS[@]}"
        fi
    done
    echo ""
    echo "Local batch complete."
}

submit_condor() {
    local tag
    if [[ "$PARTIAL_UNBLIND" == true ]]; then tag="partial_unblind"
    elif [[ "$UNBLIND"       == true ]]; then tag="unblind"
    else tag="blinded"
    fi
    if [[ "$NUISANCE" == "preserve_shape" ]]; then
        tag="${tag}_preserve_shape"
    fi

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    local job_dir="${SCRIPT_DIR}/condor/jobs_plotPostfitMass_${METHOD}_${tag}_${timestamp}"
    mkdir -p "${job_dir}/logs"

    local submit_file="${job_dir}/plotPostfitMass.sub"
    local extra_str="${EXTRA_ARGS[*]}"
    cat > "${submit_file}" <<EOF
universe = vanilla
executable = ${SCRIPT_DIR}/scripts/plotPostfitMass_wrapper.sh
arguments = --masspoint \$(masspoint) ${extra_str}
output = logs/\$(masspoint).out
error  = logs/\$(masspoint).err
log    = plotPostfitMass.log

request_cpus   = 1
request_memory = 4GB
request_disk   = 2GB

should_transfer_files = NO

EOF

    {
        echo "queue masspoint from ("
        for mp in "${MASSPOINTs[@]}"; do
            echo "    ${mp}"
        done
        echo ")"
    } >> "${submit_file}"

    echo "Submit file: ${submit_file}"
    echo "Job dir:     ${job_dir}"

    if [[ "$DRY_RUN" == true ]]; then
        echo "=== DRY RUN - would submit: ==="
        echo "cd ${job_dir} && condor_submit ${submit_file}"
        echo ""
        echo "Submit file preview:"
        head -40 "${submit_file}"
        return 0
    fi

    cd "${job_dir}"
    condor_submit "${submit_file}"
    echo ""
    echo "Submitted ${#MASSPOINTs[@]} HTCondor jobs."
    echo "Monitor with: condor_q"
    echo "Logs:         ${job_dir}/logs/"
}

if [[ "$CONDOR" == true ]]; then
    submit_condor
else
    run_local
fi
