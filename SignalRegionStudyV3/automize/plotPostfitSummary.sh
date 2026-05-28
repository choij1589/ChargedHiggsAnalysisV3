#!/bin/bash
#
# plotPostfitSummary.sh - Batch full-mA postfit summary plots.
#
# Job granularity is (mHc, method, era, channel). Summary plotting uses
# channel-scoped refill caches so these jobs can run concurrently safely.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MHC_VALUES=(70 160)
METHODS=(Baseline ParticleNet)
ERAS=(Run2 Run3 All)
CHANNELS=(SR1E2Mu SR3Mu Combined)
BINNING="extended"
NUISANCE="fallback_lnn"
FIT_TYPE="b"
BIN_WIDTH="1"
OUTPUT_DIR="results/plots/postfit_summary"
SIGNAL_LINE="none"
PARTIAL_UNBLIND=false
UNBLIND=false
BLIND=false
PLOT_ONLY=false
CONDOR=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mhc)
            shift
            MHC_VALUES=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                MHC_VALUES+=("$1")
                shift
            done
            ;;
        --methods|--method)
            shift
            METHODS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                METHODS+=("$1")
                shift
            done
            ;;
        --eras|--era)
            shift
            ERAS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                ERAS+=("$1")
                shift
            done
            ;;
        --channels|--channel|--channel-scope)
            shift
            CHANNELS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                CHANNELS+=("$1")
                shift
            done
            ;;
        --binning)         BINNING="$2"; shift 2 ;;
        --nuisance)        NUISANCE="$2"; shift 2 ;;
        --fit-type)        FIT_TYPE="$2"; shift 2 ;;
        --bin-width)       BIN_WIDTH="$2"; shift 2 ;;
        --output-dir)      OUTPUT_DIR="$2"; shift 2 ;;
        --signal-line)     SIGNAL_LINE="$2"; shift 2 ;;
        --partial-unblind) PARTIAL_UNBLIND=true; shift ;;
        --unblind)         UNBLIND=true; shift ;;
        --blind)           BLIND=true; shift ;;
        --plot-only)       PLOT_ONLY=true; shift ;;
        --condor)          CONDOR=true; shift ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --help|-h)
            cat <<EOF
Usage: $0 [OPTIONS]

Required mode, exactly one:
  --unblind | --partial-unblind | --blind

Defaults:
  --mhc 70 160
  --methods Baseline ParticleNet
  --eras Run2 Run3 All
  --channels SR1E2Mu SR3Mu Combined
  --binning extended (also accepts extended)
  --fit-type b
  --bin-width 1
  --signal-line none (choices: none, median)

Execution:
  --condor       Submit one job per (mHc, method, era, channel)
  --plot-only    Re-render from existing caches
  --dry-run      Print commands / submit file only
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
    echo "ERROR: specify exactly one of --blind / --partial-unblind / --unblind" >&2
    exit 1
fi

case "$FIT_TYPE" in b|s|both) ;; *) echo "ERROR: invalid --fit-type '$FIT_TYPE'" >&2; exit 1 ;; esac
case "$NUISANCE" in fallback_lnn|preserve_shape) ;; *) echo "ERROR: invalid --nuisance '$NUISANCE'" >&2; exit 1 ;; esac
case "$SIGNAL_LINE" in none|median) ;; *) echo "ERROR: invalid --signal-line '$SIGNAL_LINE'" >&2; exit 1 ;; esac

MODE_ARGS=()
[[ "$PARTIAL_UNBLIND" == true ]] && MODE_ARGS+=(--partial-unblind)
[[ "$UNBLIND"         == true ]] && MODE_ARGS+=(--unblind)
[[ "$BLIND"           == true ]] && MODE_ARGS+=(--blind)
[[ "$PLOT_ONLY"       == true ]] && MODE_ARGS+=(--plot-only)

COMMON_ARGS=(
    --binning "$BINNING"
    --nuisance "$NUISANCE"
    --fit-type "$FIT_TYPE"
    --bin-width "$BIN_WIDTH"
    --output-dir "$OUTPUT_DIR"
    --signal-line "$SIGNAL_LINE"
    "${MODE_ARGS[@]}"
)

echo "============================================================"
echo "SignalRegionStudyV3 Postfit Summary Batch"
echo "mHc:        ${MHC_VALUES[*]}"
echo "Methods:    ${METHODS[*]}"
echo "Eras:       ${ERAS[*]}"
echo "Channels:   ${CHANNELS[*]}"
echo "Binning:    $BINNING"
echo "Fit type:   $FIT_TYPE"
echo "Bin width:  $BIN_WIDTH GeV"
echo "Signal:     $SIGNAL_LINE"
echo "Output dir: $OUTPUT_DIR"
echo "Execution:  $([[ "$CONDOR" == true ]] && echo HTCondor || echo Local)"
echo "Dry run:    $DRY_RUN"
echo "============================================================"

run_local() {
    for mhc in "${MHC_VALUES[@]}"; do
        for method in "${METHODS[@]}"; do
            for era in "${ERAS[@]}"; do
                cmd=(python3 "${SCRIPT_DIR}/python/plotPostfitSummary.py"
                     --mhc "$mhc"
                     --methods "$method"
                     --eras "$era"
                     --channels "${CHANNELS[@]}"
                     "${COMMON_ARGS[@]}")
                echo ">>> ${cmd[*]}"
                [[ "$DRY_RUN" == true ]] || "${cmd[@]}"
            done
        done
    done
}

submit_condor() {
    local tag
    if [[ "$PARTIAL_UNBLIND" == true ]]; then tag="partial_unblind"
    elif [[ "$UNBLIND"       == true ]]; then tag="unblind"
    else tag="blind"
    fi

    local timestamp job_dir submit_file common_str
    timestamp=$(date +%Y%m%d_%H%M%S)
    job_dir="${SCRIPT_DIR}/condor/jobs_plotPostfitSummary_${tag}_${timestamp}"
    mkdir -p "${job_dir}/logs"
    submit_file="${job_dir}/plotPostfitSummary.sub"
    common_str="${COMMON_ARGS[*]}"

    cat > "$submit_file" <<EOF
universe = vanilla
executable = ${SCRIPT_DIR}/scripts/plotPostfitSummary_wrapper.sh
arguments = --mhc \$(mhc) --methods \$(method) --eras \$(era) --channels \$(channel) ${common_str}
output = logs/mHc\$(mhc).\$(method).\$(era).\$(channel).out
error  = logs/mHc\$(mhc).\$(method).\$(era).\$(channel).err
log    = plotPostfitSummary.log

request_cpus   = 1
request_memory = 6GB
request_disk   = 4GB

should_transfer_files = NO

EOF

    {
        echo "queue mhc, method, era, channel from ("
        for mhc in "${MHC_VALUES[@]}"; do
            for method in "${METHODS[@]}"; do
                for era in "${ERAS[@]}"; do
                    for channel in "${CHANNELS[@]}"; do
                        echo "    $mhc $method $era $channel"
                    done
                done
            done
        done
        echo ")"
    } >> "$submit_file"

    echo "Submit file: $submit_file"
    echo "Job dir:     $job_dir"
    if [[ "$DRY_RUN" == true ]]; then
        echo "=== DRY RUN - would submit: ==="
        echo "cd $job_dir && condor_submit $submit_file"
        echo ""
        head -60 "$submit_file"
        return
    fi

    cd "$job_dir"
    condor_submit "$submit_file"
}

if [[ "$CONDOR" == true ]]; then
    submit_condor
else
    run_local
fi
