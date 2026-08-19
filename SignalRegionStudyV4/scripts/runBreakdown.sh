#!/bin/bash
#
# runBreakdown.sh - grouped-nuisance uncertainty breakdown of sigma(r)
# at one mass point (docs/BREAKDOWN.md).
#
# One total MultiDimFit likelihood scan of r, then a chain of scans that
# cumulatively freeze nuisance groups; each group's contribution is the
# quadrature difference between consecutive scans, and whatever still
# floats at the end (data statistics + the autoMCStats prop_bin
# parameters) is the residual 'stat' component.
#
# The scan range is derived PER POINT from the point's own asymptotic
# limit, not fixed.  V3 scanned r over [-5,5] with 100 points, a step of
# 0.1, while sigma(r) here is 0.08-0.45 -- 0.2 to 1.2 grid steps per
# sigma, so the 2*deltaNLL=1 crossing would be splined from one or two
# points and the quadrature subtraction of two such numbers is noise.
# See resolve_scan_range() below.
#
# Steps (one condor node each, see automize/breakdown.sh):
#   setup   -> grouped datacard + workspace
#   bestfit -> --algo none --saveWorkspace: the common best-fit snapshot
#   total   -> the unfrozen scan, off that snapshot
#   freeze  -> the Nth cumulative freeze scan (--group-index N, 1-based)
#   plot    -> plot1DScan.py --breakdown
#
# EVERY scan starts from the ONE bestfit snapshot. Quadrature subtraction
# compares intervals across scans, which is only meaningful if they share
# a minimum; letting the total grid scan be its own snapshot source (V3's
# recipe) lets the frozen scans re-minimize to a different r, and the
# subtraction then goes negative. Measured: at the three template points
# whose total scan pinned r-hat at 0.000, freezing RAISED sigma.
#
# Usage:
#   ./runBreakdown.sh --era All --channel Combined --masspoint MHc130_MA90 \
#       --method Baseline --signal-source interp-signal --seed MHc130_MA90 \
#       --step total
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
GROUP_INDEX=""
POINTS="200"
R_RANGE=""
SIGMA_WINDOW="5"
BLIND=false
DRY_RUN=false
VERBOSE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --era)           ERA="$2";           shift 2 ;;
        --channel)       CHANNEL="$2";       shift 2 ;;
        --masspoint)     MASSPOINT="$2";     shift 2 ;;
        --method)        METHOD="$2";        shift 2 ;;
        --signal-source) SIGNAL_SOURCE="$2"; shift 2 ;;
        --seed)          SEED="$2";          shift 2 ;;
        --step)          STEP="$2";          shift 2 ;;
        --group-index)   GROUP_INDEX="$2";   shift 2 ;;
        --points)        POINTS="$2";        shift 2 ;;
        --r-range)       R_RANGE="$2";       shift 2 ;;
        --sigma-window)  SIGMA_WINDOW="$2";  shift 2 ;;
        --blind)         BLIND=true;         shift ;;
        --dry-run)       DRY_RUN=true;       shift ;;
        --verbose)       VERBOSE=true;       shift ;;
        -h|--help)
            echo "Usage: $0 --era ERA --channel CHANNEL --masspoint MASSPOINT"
            echo "          --step {setup,total,freeze,plot} [--group-index N]"
            echo "          [--method M] [--signal-source S] [--seed SEED_MP]"
            echo "          [--points N] [--r-range MIN,MAX] [--sigma-window W]"
            echo "          [--blind] [--dry-run] [--verbose]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$ERA" || -z "$CHANNEL" || -z "$MASSPOINT" ]]; then
    echo "ERROR: --era, --channel, and --masspoint are required"
    exit 1
fi
case "$STEP" in
    setup|bestfit|total|freeze|plot) ;;
    *) echo "ERROR: --step must be setup, bestfit, total, freeze or plot"; exit 1 ;;
esac
if [[ "$STEP" == "freeze" && -z "$GROUP_INDEX" ]]; then
    echo "ERROR: --step freeze requires --group-index N (1-based)"
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

OUTPUT_DIR="${TEMPLATE_DIR}/combine_output/breakdown"
mkdir -p "$OUTPUT_DIR"

TAG="${MASSPOINT}.${METHOD_SEGMENT}"
TOTAL_FILE="higgsCombine.${TAG}.total.MultiDimFit.mH120.root"
BESTFIT_FILE="higgsCombine.${TAG}.bestfit.MultiDimFit.mH120.root"
GROUPED_DATACARD="${OUTPUT_DIR}/grouped_datacard.txt"
GROUPED_WORKSPACE="${OUTPUT_DIR}/grouped_workspace.root"
FREEZE_GROUPS_FILE="${OUTPUT_DIR}/freeze_groups.txt"

ASIMOV_OPTIONS=""
[[ "$BLIND" == true ]] && ASIMOV_OPTIONS="-t -1 --expectSignal 1"

log() { [[ "$VERBOSE" == true ]] && echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" || true; }
run_cmd() {
    if [[ "$DRY_RUN" == true ]]; then echo "[DRY-RUN] $1"; else log "Running: $1"; eval "$1"; fi
}

# The scan window, in units of the point's own sigma(r).
#
# The asymptotic limit tree already holds r (no unit conversion), and its
# median expected upper limit is ~1.96 sigma for r near zero, so
# sigma ~ exp0/1.96 is a free and reliable scale.  Scanning
# +-SIGMA_WINDOW sigma with --points N puts 2*N/(2*SIGMA_WINDOW) = 20
# grid points inside each sigma at the default 5 / 200, whatever the
# point's sensitivity.
#
# NOTE: scripts/runImpacts.sh has a resolve_r_range() that looks similar
# but only ever WIDENS past +-5; for every template point
# 2 x exp+2sigma < 5, so it returns -5,5 and cannot be reused here.
require_groups_file() {
    if [[ ! -s "$FREEZE_GROUPS_FILE" ]]; then
        echo "ERROR: $FREEZE_GROUPS_FILE missing or empty;" \
             "run --step setup first" >&2
        exit 1
    fi
}

resolve_scan_range() {
    local root_file="$TEMPLATE_DIR/combine_output/asymptotic/higgsCombine.${TAG}.AsymptoticLimits.mH120.root"
    if [[ -n "$R_RANGE" ]]; then
        echo "$R_RANGE"
        return
    fi
    if [[ -f "$root_file" ]]; then
        local rng
        rng=$(python3 -c "
import ROOT
ROOT.gROOT.SetBatch(True)
f = ROOT.TFile.Open('$root_file')
t = f.Get('limit')
vals = [e.limit for e in t]
f.Close()
exp0 = vals[2]
sigma = exp0 / 1.96
r = $SIGMA_WINDOW * sigma
print(f'-{r:.4f},{r:.4f}' if r > 0 else '')
" 2>/dev/null)
        if [[ -n "$rng" ]]; then
            echo "$rng"
            return
        fi
    fi
    # Falling back to V3's fixed window badly under-resolves the crossing
    # at this analysis's sensitivity -- loud, not silent.
    echo "WARNING: no usable asymptotic limit at ${root_file};" \
         "falling back to r=-5,5, which under-resolves sigma(r) here." >&2
    echo "-5,5"
}

cd "$TEMPLATE_DIR"

case "$STEP" in
    setup)
        echo "Breakdown setup for ${MASSPOINT} (${METHOD_SEGMENT}/${ERA}/${CHANNEL})..."
        run_cmd "python3 '$SRS_MODULE_DIR/python/nuisanceGroups.py' \
            --datacard '$TEMPLATE_DIR/datacard.txt' \
            --output '$GROUPED_DATACARD' \
            --group-json '${OUTPUT_DIR}/group_members.json' \
            --groups-file '$FREEZE_GROUPS_FILE'"
        # shapes.root is referenced by a relative path in the datacard
        run_cmd "ln -sf '$TEMPLATE_DIR/shapes.root' '${OUTPUT_DIR}/shapes.root'"
        # Its own workspace: only this one carries the group = lines, so
        # it must not be confused with the GoF/impacts workspace.root.
        run_cmd "cd '$OUTPUT_DIR' && text2workspace.py grouped_datacard.txt \
            -o grouped_workspace.root 2>&1 | tee text2workspace.out"
        if [[ "$DRY_RUN" == false && ! -f "$GROUPED_WORKSPACE" ]]; then
            echo "ERROR: workspace not created: $GROUPED_WORKSPACE"
            exit 1
        fi
        ;;

    bestfit)
        RANGE=$(resolve_scan_range)
        echo "Breakdown best fit for ${MASSPOINT} (${METHOD_SEGMENT}/${ERA}/${CHANNEL}), r range ${RANGE}"
        if [[ ! -f "$GROUPED_WORKSPACE" && "$DRY_RUN" == false ]]; then
            echo "ERROR: grouped workspace missing; run --step setup first"
            exit 1
        fi
        run_cmd "cd '$OUTPUT_DIR' && combine -M MultiDimFit grouped_workspace.root \
            --algo none \
            --setParameterRanges r=${RANGE} \
            --saveWorkspace \
            -n .${TAG}.bestfit \
            -m 120 \
            ${ASIMOV_OPTIONS} \
            2>&1 | tee bestfit.out"
        if [[ "$DRY_RUN" == false && ! -f "${OUTPUT_DIR}/${BESTFIT_FILE}" ]]; then
            echo "ERROR: best fit produced no output: ${OUTPUT_DIR}/${BESTFIT_FILE}"
            exit 1
        fi
        ;;

    total)
        RANGE=$(resolve_scan_range)
        echo "Breakdown total scan for ${MASSPOINT} (${METHOD_SEGMENT}/${ERA}/${CHANNEL}), r range ${RANGE}, ${POINTS} points"
        if [[ ! -f "${OUTPUT_DIR}/${BESTFIT_FILE}" && "$DRY_RUN" == false ]]; then
            echo "ERROR: best-fit snapshot missing; run --step bestfit first"
            exit 1
        fi
        run_cmd "cd '$OUTPUT_DIR' && combine -M MultiDimFit ${BESTFIT_FILE} \
            --snapshotName MultiDimFit \
            --algo grid \
            --points ${POINTS} \
            --setParameterRanges r=${RANGE} \
            -n .${TAG}.total \
            -m 120 \
            ${ASIMOV_OPTIONS} \
            2>&1 | tee total_scan.out"
        if [[ "$DRY_RUN" == false && ! -f "${OUTPUT_DIR}/${TOTAL_FILE}" ]]; then
            echo "ERROR: total scan produced no output: ${OUTPUT_DIR}/${TOTAL_FILE}"
            exit 1
        fi
        ;;

    freeze)
        RANGE=$(resolve_scan_range)
        require_groups_file
        mapfile -t FREEZE_GROUPS < "$FREEZE_GROUPS_FILE"
        if [[ "$GROUP_INDEX" -lt 1 || "$GROUP_INDEX" -gt ${#FREEZE_GROUPS[@]} ]]; then
            echo "ERROR: --group-index $GROUP_INDEX out of range (1..${#FREEZE_GROUPS[@]})"
            exit 1
        fi
        # Cumulative: freeze groups 1..GROUP_INDEX.  plot1DScan.py
        # --breakdown requires exactly this nesting.
        CUMULATIVE_CSV="$(IFS=','; echo "${FREEZE_GROUPS[*]:0:$GROUP_INDEX}")"
        CUMULATIVE_TAG="${CUMULATIVE_CSV//,/_}"
        # The last scan leaves only prop_bin* floating; the analytic
        # minimizer can shortcut there and write the best-fit point alone
        # instead of a full grid.
        EXTRA_RTD=""
        [[ "$GROUP_INDEX" -eq ${#FREEZE_GROUPS[@]} ]] && EXTRA_RTD="--X-rtd MINIMIZER_no_analytic"
        FREEZE_FILE="higgsCombine.${TAG}.freeze_${CUMULATIVE_TAG}.MultiDimFit.mH120.root"
        echo "Breakdown freeze scan ${GROUP_INDEX}/${#FREEZE_GROUPS[@]} (${CUMULATIVE_CSV}) for ${MASSPOINT}, r range ${RANGE}"
        if [[ ! -f "${OUTPUT_DIR}/${BESTFIT_FILE}" && "$DRY_RUN" == false ]]; then
            echo "ERROR: best-fit snapshot missing; run --step bestfit first"
            exit 1
        fi
        run_cmd "cd '$OUTPUT_DIR' && combine -M MultiDimFit ${BESTFIT_FILE} \
            --snapshotName MultiDimFit \
            --algo grid \
            --points ${POINTS} \
            --setParameterRanges r=${RANGE} \
            --freezeNuisanceGroups ${CUMULATIVE_CSV} \
            ${EXTRA_RTD} \
            -n .${TAG}.freeze_${CUMULATIVE_TAG} \
            -m 120 \
            ${ASIMOV_OPTIONS} \
            2>&1 | tee freeze_${CUMULATIVE_TAG}.out"
        if [[ "$DRY_RUN" == false && ! -f "${OUTPUT_DIR}/${FREEZE_FILE}" ]]; then
            echo "ERROR: freeze scan produced no output: ${OUTPUT_DIR}/${FREEZE_FILE}"
            exit 1
        fi
        ;;

    plot)
        require_groups_file
        mapfile -t FREEZE_GROUPS < "$FREEZE_GROUPS_FILE"
        COLORS=(2 4 6 8 9 28 46 38)
        OTHER_ARGS=()
        CUMULATIVE=()
        for IDX in "${!FREEZE_GROUPS[@]}"; do
            CUMULATIVE+=("${FREEZE_GROUPS[$IDX]}")
            CUM_TAG="$(IFS='_'; echo "${CUMULATIVE[*]}")"
            OTHER_ARGS+=("higgsCombine.${TAG}.freeze_${CUM_TAG}.MultiDimFit.mH120.root:${FREEZE_GROUPS[$IDX]}:${COLORS[$IDX]}")
        done
        BREAKDOWN_LABELS="$(IFS=','; echo "${FREEZE_GROUPS[*]}"),stat"
        echo "Breakdown plot for ${MASSPOINT}: ${BREAKDOWN_LABELS}"
        run_cmd "cd '$OUTPUT_DIR' && plot1DScan.py ${TOTAL_FILE} \
            --main-label Total \
            --main-color 1 \
            --POI r \
            --others ${OTHER_ARGS[*]} \
            --breakdown ${BREAKDOWN_LABELS} \
            --y-cut 100 \
            --y-max 10 \
            -o breakdown \
            2>&1 | tee plot.out"
        if [[ "$DRY_RUN" == false && ! -f "${OUTPUT_DIR}/breakdown.png" ]]; then
            echo "ERROR: plot1DScan produced no output"
            exit 1
        fi
        # plot1DScan.py substitutes 0 for a negative quadrature
        # subtraction and only says so on stdout; surface it here so a
        # pathological point cannot look like a clean zero.
        if [[ "$DRY_RUN" == false ]] && grep -q "SUBTRACTION IS NEGATIVE" "${OUTPUT_DIR}/plot.out"; then
            echo "WARNING: negative quadrature subtraction at this point:"
            grep "SUBTRACTION IS NEGATIVE" "${OUTPUT_DIR}/plot.out" | sed 's/^/  /'
        fi
        ;;
esac

echo "Done."
