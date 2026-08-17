#!/bin/bash
# Stitched per-mHc postfit summary panels, parallelized at the SEED level.
#
# The summary stitches every group seed of a study (Baseline 64-95 per mHc,
# ParticleNet 3 on top), and the expensive part is building each seed's
# fine-mass histograms out of its fitDiagnostics shapes -- 160 hists per
# (seed, channel). That work is per-seed independent and cached under the
# seed's own combine_output/fitdiag/cached/, so the DAG is:
#
#   postfit-cache (one node per seed, all 3 channels)  ... in parallel
#                              |
#                              v
#   postfit-summary (one node per mHc, stitches from cache in seconds)
#
# Run serially inside one summary job this took hours per mHc; fanned out it
# is bounded by the slowest single seed.
#
# ORDER MATTERS: the ParticleNet panel stitches the Baseline seeds too and
# reads their caches, so run the Baseline campaign first. Each arm warms only
# its own seeds, so the two never write the same cache file.
#
# Usage:
#   ./automize/postfitSummary.sh --all                        # Baseline, 7 mHc, 579 nodes
#   ./automize/postfitSummary.sh --all --method ParticleNet    # then: 5 mHc, 20 nodes
#   ./automize/postfitSummary.sh --mhc 160 [--dry-run]
#   ./automize/postfitSummary.sh --all --summary-only          # caches already warm
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/automize/dag_lib.sh"
source "$SCRIPT_DIR/scripts/env.sh"

MHC_LIST=()
DRY_RUN=false
METHOD="Baseline"
ALL_MHC=false
SUMMARY_ONLY=false
CACHE_MEMORY=4096
SUMMARY_MEMORY=32768

while [[ $# -gt 0 ]]; do
    case $1 in
        --mhc) MHC_LIST+=("${2#MHc}"); shift 2 ;;
        --all) ALL_MHC=true; shift ;;
        --method) METHOD="$2"; shift 2 ;;
        --summary-only) SUMMARY_ONLY=true; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) grep '^#' "$0" | head -22; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
case "$METHOD" in
    Baseline|ParticleNet) ;;
    *) echo "ERROR: --method must be Baseline or ParticleNet"; exit 1 ;;
esac
if [[ "$ALL_MHC" == "true" ]]; then
    if [[ "$METHOD" == "ParticleNet" ]]; then
        MHC_LIST=(100 115 130 145 160)
    else
        MHC_LIST=(70 85 100 115 130 145 160)
    fi
fi
[[ ${#MHC_LIST[@]} -gt 0 ]] || { echo "ERROR: --mhc N or --all required"; exit 1; }

# Group seeds of one (mHc, arm), from grid.json / pnet_grid.json.
seed_list() {
    local mhc=$1 arm=$2
    SRS_SEED_ARM="$arm" python3 - "$mhc" << 'PYEOF'
import sys, os
sys.path.insert(0, os.path.join(os.environ["WORKDIR"],
                                "SignalRegionStudyV4", "python"))
import srspaths
from interpolation_config import masspoint_name
mhc = int(sys.argv[1])
arm = os.environ["SRS_SEED_ARM"]
cfg = (srspaths.grid_config() if arm == "Baseline"
       else srspaths.pnet_grid_config())
key = f"MHc{mhc}"
for grp in cfg["grids"].get(key, {}).get("groups", []):
    print(masspoint_name(grp["seed"], mhc))
PYEOF
}

generate_dag() {
    local mhc=$1
    local dag_dir=$2
    local dag_file="$dag_dir/dag.dag"

    cat > "$dag_dir/jobs.sub" << EOF
universe                = vanilla
executable              = $SRS_MODULE_DIR/scripts/interp_templates_wrapper.sh
arguments               = "\$(step) \$(masspoint) \$(seed) \$(era) \$(channel) \$(extra)"
output                  = logs/\$(step).\$(seed).\$(arm).out
error                   = logs/\$(step).\$(seed).\$(arm).err
log                     = logs/dag.log
request_memory          = \$(memory)
request_cpus            = 1
should_transfer_files   = NO
getenv                  = False
queue
EOF

    : > "$dag_file"
    local n_nodes=0
    local parents=""

    if [[ "$SUMMARY_ONLY" == "false" ]]; then
        # Each arm warms ONLY its own seeds. A ParticleNet panel also stitches
        # every Baseline seed, but those caches are the Baseline campaign's
        # files -- warming them here too would have two jobs writing one ROOT
        # file. So run the Baseline campaign first; this one adds its 3 seeds
        # per mHc on top.
        local arms=("$METHOD")
        for arm in "${arms[@]}"; do
            local arm_extra=""
            [[ "$arm" == "ParticleNet" ]] && arm_extra="--method ParticleNet"
            while IFS= read -r seed; do
                [[ -z "$seed" ]] && continue
                local node="cache_${arm}_${seed}"
                {
                    echo "JOB $node jobs.sub"
                    echo "VARS $node step=\"postfit-cache\" masspoint=\"MHc${mhc}\" seed=\"$seed\" era=\"All\" channel=\"-\" extra=\"$arm_extra\" memory=\"$CACHE_MEMORY\" arm=\"$arm\""
                    echo "RETRY $node 1"
                } >> "$dag_file"
                parents="$parents $node"
                n_nodes=$((n_nodes + 1))
            done <<< "$(seed_list "$mhc" "$arm")"
        done
        # An unset WORKDIR makes seed_list import-fail and return nothing,
        # which would otherwise emit a summary-only DAG that quietly stitches
        # from whatever caches happen to exist. Fail instead.
        if [[ $n_nodes -eq 0 ]]; then
            echo "ERROR: no seeds found for mHc${mhc}. Did you 'source" \
                 "setup.sh'? (WORKDIR=${WORKDIR:-UNSET})" >&2
            exit 1
        fi
    fi

    local sum_extra=""
    [[ "$METHOD" == "ParticleNet" ]] && sum_extra="--method ParticleNet"
    {
        echo "JOB summary jobs.sub"
        echo "VARS summary step=\"postfit-summary\" masspoint=\"MHc${mhc}\" seed=\"MHc${mhc}\" era=\"All\" channel=\"-\" extra=\"$sum_extra\" memory=\"$SUMMARY_MEMORY\" arm=\"$METHOD\""
        echo "RETRY summary 1"
    } >> "$dag_file"
    n_nodes=$((n_nodes + 1))
    [[ -n "$parents" ]] && echo "PARENT$parents CHILD summary" >> "$dag_file"

    echo "CONFIG dagman.config" >> "$dag_file"
    echo "Generated $dag_file: $n_nodes nodes"
}

echo "============================================================"
echo "Postfit summary campaign (seed-level cache fan-out)"
echo "Method: $METHOD"
echo "mHc values: ${MHC_LIST[*]}"
echo "Summary only: $SUMMARY_ONLY"
echo "Dry run: $DRY_RUN"
echo "============================================================"

JOB_PREFIX="postfit_summary"
[[ "$METHOD" == "ParticleNet" ]] && JOB_PREFIX="pnet_postfit_summary"
JOB_DIR=$(dag_new_jobdir "$JOB_PREFIX")
for mhc in "${MHC_LIST[@]}"; do
    MP_DIR=$(dag_new_masspoint_dir "$JOB_DIR" "MHc${mhc}")
    generate_dag "$mhc" "$MP_DIR"
done
dag_write_submit_all "$JOB_DIR"
dag_write_status_all "$JOB_DIR"
dag_submit_or_dryrun "$JOB_DIR" "$DRY_RUN"
