#!/bin/bash
# ParticleNet score-distribution plots at every interpolation group seed
# (mirrors V3's plot_score coverage: {Run2, Run3, All} x {SR1E2Mu, SR3Mu,
# Combined} per point, with the TTZ2E1Mu control region auto-emitted by the
# per-channel jobs).
#
# Node ordering per seed: combined modes sum the per-era cached
# histograms.root files, so period x channel jobs run first, then the
# All-channel jobs, then the Combined-channel jobs:
#
#   {Run2,Run3} x {SR1E2Mu,SR3Mu}  (4)  ->  All x {SR1E2Mu,SR3Mu}  (2)
#                                        ->  {Run2,Run3,All} x Combined (3)
#
# Usage:
#   ./automize/pnetScorePlots.sh --all [--dry-run]
#   ./automize/pnetScorePlots.sh --mhc 115 [--group MHc115_MA90]
#
# --replot passes --skip-histogram to every node: the cached histograms.root
# files are re-drawn without touching the score trees. That is the cheap pass
# to run after a styling-only change (~30 s per node instead of the full
# reprocessing).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/automize/dag_lib.sh"
source "$SCRIPT_DIR/scripts/env.sh"

MHC_LIST=()
GROUP_SEED=""
DRY_RUN=false
REPLOT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mhc) MHC_LIST+=("${2#MHc}"); shift 2 ;;
        --all) MHC_LIST=(100 115 130 145 160); shift ;;
        --group) GROUP_SEED="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        --replot) REPLOT=true; shift ;;
        -h|--help) grep '^#' "$0" | head -16; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
[[ ${#MHC_LIST[@]} -gt 0 ]] || { echo "ERROR: --mhc N or --all required"; exit 1; }

EXTRA=""
if $REPLOT; then EXTRA="--skip-histogram"; fi

generate_dag() {
    local mhc=$1
    local dag_dir=$2
    local dag_file="$dag_dir/dag.dag"

    local seeds
    seeds=$(WORKDIR="$SRS_REPO_DIR" python3 - "$mhc" <<'PYEOF'
import sys, os
sys.path.insert(0, os.path.join(os.environ["WORKDIR"],
                                "SignalRegionStudyV4", "python"))
import srspaths
from interpolation_config import masspoint_name
mhc = int(sys.argv[1])
for grp in srspaths.pnet_grid_config()["grids"][f"MHc{mhc}"]["groups"]:
    print(masspoint_name(grp["seed"], mhc))
PYEOF
    )

    cat > "$dag_dir/jobs.sub" << EOF
universe                = vanilla
executable              = $SRS_MODULE_DIR/scripts/interp_templates_wrapper.sh
arguments               = "\$(step) \$(masspoint) \$(seed) \$(era) \$(channel) \$(extra)"
output                  = logs/\$(step).\$(masspoint).\$(era).\$(channel).out
error                   = logs/\$(step).\$(masspoint).\$(era).\$(channel).err
log                     = logs/dag.log
request_memory          = 4096
request_cpus            = 1
should_transfer_files   = NO
getenv                  = False
queue
EOF

    : > "$dag_file"
    local n_nodes=0

    while IFS= read -r seed; do
        [[ -z "$seed" ]] && continue
        [[ -n "$GROUP_SEED" && "$seed" != "$GROUP_SEED" ]] && continue

        local node era channel
        local period_nodes=() allch_nodes=()
        for era in Run2 Run3; do
            for channel in SR1E2Mu SR3Mu; do
                node="score_${seed}_${era}_${channel}"
                period_nodes+=("$node")
                {
                    echo "JOB $node jobs.sub"
                    echo "VARS $node step=\"plot-score\" masspoint=\"$seed\" seed=\"$seed\" era=\"$era\" channel=\"$channel\" extra=\"$EXTRA\""
                    echo "RETRY $node 1"
                } >> "$dag_file"
                n_nodes=$((n_nodes + 1))
            done
        done
        for channel in SR1E2Mu SR3Mu; do
            node="score_${seed}_All_${channel}"
            allch_nodes+=("$node")
            {
                echo "JOB $node jobs.sub"
                echo "VARS $node step=\"plot-score\" masspoint=\"$seed\" seed=\"$seed\" era=\"All\" channel=\"$channel\" extra=\"$EXTRA\""
                echo "PARENT ${period_nodes[*]} CHILD $node"
                echo "RETRY $node 1"
            } >> "$dag_file"
            n_nodes=$((n_nodes + 1))
        done
        for era in Run2 Run3 All; do
            node="score_${seed}_${era}_Combined"
            {
                echo "JOB $node jobs.sub"
                echo "VARS $node step=\"plot-score\" masspoint=\"$seed\" seed=\"$seed\" era=\"$era\" channel=\"Combined\" extra=\"$EXTRA\""
                echo "PARENT ${period_nodes[*]} ${allch_nodes[*]} CHILD $node"
                echo "RETRY $node 1"
            } >> "$dag_file"
            n_nodes=$((n_nodes + 1))
        done
    done <<< "$seeds"

    echo "CONFIG dagman.config" >> "$dag_file"
    echo "Generated $dag_file: $n_nodes nodes"
}

echo "============================================================"
echo "ParticleNet interp-signal score plots"
echo "mHc values: ${MHC_LIST[*]}"
echo "Dry run: $DRY_RUN"
echo "============================================================"

JOB_DIR=$(dag_new_jobdir "pnet_scores")
for mhc in "${MHC_LIST[@]}"; do
    MP_DIR=$(dag_new_masspoint_dir "$JOB_DIR" "MHc${mhc}")
    generate_dag "$mhc" "$MP_DIR"
done
dag_write_submit_all "$JOB_DIR"
dag_write_status_all "$JOB_DIR"
dag_submit_or_dryrun "$JOB_DIR" "$DRY_RUN"
