#!/bin/bash
#
# runImpacts.sh - nuisance impacts on one template dir (mirrors the V3
# combineTool workflow: initial fit -> per-nuisance fits -> collect,
# --robustFit 1, r-range from the point's own asymptotic limit). Runs the
# whole workflow in ONE invocation (V3's local mode) so a condor slot with
# request_cpus = --parallel executes it as a single fat job.
#
# Usage:
#   ./runImpacts.sh --era All --channel Combined --masspoint MHc160_MA90 \
#       --signal-source interp-signal --seed MHc160_MA90 [--parallel 4]
#

set -eo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

ERA=""
CHANNEL=""
MASSPOINT=""
METHOD="Baseline"
SIGNAL_SOURCE="mc-signal"
SEED=""
PARALLEL=4
R_RANGE=""
BLIND=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --era)           ERA="$2";           shift 2 ;;
        --channel)       CHANNEL="$2";       shift 2 ;;
        --masspoint)     MASSPOINT="$2";     shift 2 ;;
        --method)        METHOD="$2";        shift 2 ;;
        --signal-source) SIGNAL_SOURCE="$2"; shift 2 ;;
        --seed)          SEED="$2";          shift 2 ;;
        --parallel)      PARALLEL="$2";      shift 2 ;;
        --r-range)       R_RANGE="$2";       shift 2 ;;
        --blind)         BLIND=true;         shift ;;
        --dry-run)       DRY_RUN=true;       shift ;;
        -h|--help)
            echo "Usage: $0 --era ERA --channel CH --masspoint MP"
            echo "          [--method M] [--signal-source S] [--seed SEED_MP]"
            echo "          [--parallel N] [--r-range MIN,MAX] [--blind] [--dry-run]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ -z "$ERA" || -z "$CHANNEL" || -z "$MASSPOINT" ]]; then
    echo "ERROR: --era, --channel and --masspoint are required"
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

# Unblind output dir name (V3 convention); a blinded Asimov run would be a
# different mode and is not wired here (V4 unblind default).
OUTPUT_DIR="$TEMPLATE_DIR/combine_output/impacts_obs"
SUFFIX="${MASSPOINT}.${METHOD_SEGMENT}"

# r range: 2 x (expected +2sigma) from the point's own asymptotic output,
# used when it exceeds the +-5 fallback (V3 rule).
resolve_r_range() {
    local root_file="$TEMPLATE_DIR/combine_output/asymptotic/higgsCombine.${MASSPOINT}.${METHOD_SEGMENT}.AsymptoticLimits.mH120.root"
    if [[ -n "$R_RANGE" ]]; then
        echo "$R_RANGE"
        return
    fi
    if [[ -f "$root_file" ]]; then
        local rmax
        rmax=$(python3 -c "
import ROOT
f = ROOT.TFile.Open('$root_file')
t = f.Get('limit')
vals = [e.limit for e in t]
f.Close()
r = 2.0 * vals[4]   # 2 x exp+2sigma
print(f'{r:.4f}' if abs(r) > 5 else '')
" 2>/dev/null)
        if [[ -n "$rmax" ]]; then
            echo "-${rmax},${rmax}"
            return
        fi
    fi
    echo "-5,5"
}

RANGE=$(resolve_r_range)

run() {
    if [[ "$DRY_RUN" == true ]]; then
        echo "[DRY-RUN] $*"
    else
        echo "+ $*"
        "$@"
    fi
}

cd "$TEMPLATE_DIR"
mkdir -p "$OUTPUT_DIR"
echo "Impacts for $MASSPOINT ($ERA/$CHANNEL, $SIGNAL_SOURCE), r range $RANGE, parallel $PARALLEL"

if [[ ! -f workspace.root || datacard.txt -nt workspace.root ]]; then
    run text2workspace.py datacard.txt -o workspace.root
fi

# Fit artifacts land in CWD; run inside the output dir so they stay there.
cd "$OUTPUT_DIR"
run combineTool.py -M Impacts -d ../../workspace.root -m 120 \
    --doInitialFit --robustFit 1 \
    --setParameterRanges "r=${RANGE}" \
    -n ".${SUFFIX}"
run combineTool.py -M Impacts -d ../../workspace.root -m 120 \
    --doFits --robustFit 1 --parallel "$PARALLEL" \
    --setParameterRanges "r=${RANGE}" \
    -n ".${SUFFIX}"
run combineTool.py -M Impacts -d ../../workspace.root -m 120 \
    -n ".${SUFFIX}" -o impacts.json
run python3 "$SRS_MODULE_DIR/python/filterImpacts.py" \
    -i impacts.json -o impacts_filtered.json
run plotImpacts.py -i impacts.json -o impacts
run plotImpacts.py -i impacts_filtered.json -o impacts_filtered --summary
echo "Impacts complete: $OUTPUT_DIR/impacts.pdf"
