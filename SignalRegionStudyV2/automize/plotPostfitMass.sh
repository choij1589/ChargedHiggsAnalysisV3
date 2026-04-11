#!/bin/bash
#
# plotPostfitMass.sh - Batch real-mass post-fit plots (fine 1-GeV bins).
#
# Wraps python/plotPostfitMass.py over all mass points for a single
# (era-scope, channel-scope) pair. Defaults to the All/Combined fit result
# with era-scope=All and channel-scope=Combined, since those are the only
# plots we routinely care about.
#
# Usage:
#   # Default: All fit, era-scope All, channel-scope Combined, both fit types.
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

# Defaults chosen so the shell produces only the top-level All/Combined plot.
ERA="All"
ERA_SCOPE="All"
CHANNEL_SCOPE="Combined"
METHOD="Baseline"
BINNING="extended"
FIT_TYPE="both"
BIN_WIDTH="1.0"
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

Defaults (produce only top-level All/Combined plots):
  --era All           Fit source
  --era-scope All     Era slice within the fit (pass through to Python)
  --channel Combined  Channel slice (alias: --channel-scope)

Template options:
  --method METHOD     Baseline or ParticleNet [default: Baseline]
  --binning BINNING   extended or uniform     [default: extended]
  --fit-type T        b | s | both            [default: both]
  --bin-width W       Fine-grid bin width     [default: 1.0 GeV]

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

if [[ "$METHOD" == "ParticleNet" ]]; then
    MASSPOINTs=("${MASSPOINTs_PARTICLENET[@]}")
else
    MASSPOINTs=("${MASSPOINTs_BASELINE[@]}")
fi

# Common extra args (same order for condor and local runs)
EXTRA_ARGS=(
    --era "${ERA}"
    --era-scope "${ERA_SCOPE}"
    --channel-scope "${CHANNEL_SCOPE}"
    --method "${METHOD}"
    --binning "${BINNING}"
    --fit-type "${FIT_TYPE}"
    --bin-width "${BIN_WIDTH}"
)
[[ "$PARTIAL_UNBLIND" == true ]] && EXTRA_ARGS+=(--partial-unblind)
[[ "$UNBLIND"         == true ]] && EXTRA_ARGS+=(--unblind)
[[ "$BLIND"           == true ]] && EXTRA_ARGS+=(--blind)
[[ "$PLOT_ONLY"       == true ]] && EXTRA_ARGS+=(--plot-only)

echo "============================================================"
echo "SignalRegionStudyV2 Real-Mass Post-fit Plot Batch"
echo "Era (fit):       $ERA"
echo "Era scope:       $ERA_SCOPE"
echo "Channel scope:   $CHANNEL_SCOPE"
echo "Method:          $METHOD"
echo "Binning:         $BINNING"
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
