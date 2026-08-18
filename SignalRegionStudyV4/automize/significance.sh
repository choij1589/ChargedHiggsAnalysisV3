#!/bin/bash
# Observed local significance, All x 3 channels.
#
# Two modes, deliberately separate:
#
#   --point / --template-points  the QUOTED points.  The significance is a
#       follow-up on points the limit scan singles out, so this list stays
#       explicit; --template-points takes the curated bundle set (the same
#       list collectTemplatePlots.py promotes, read from there so the two
#       cannot drift).
#
#   --grid / --pnet-grid         the whole scan, one node per (point,
#       channel).  This is the look-elsewhere input: the Gross-Vitells
#       trials estimate counts upcrossings of the observed Z(mA) curve, so
#       it needs Z at every scan point, not at a chosen few.  See
#       docs/LEE.md.  One DAG per (arm, mHc).
#
# Usage:
#   ./automize/significance.sh --template-points
#   ./automize/significance.sh --point Baseline:MHc145_MA17p5
#   ./automize/significance.sh --grid                 # 2467 x 3 nodes
#   ./automize/significance.sh --grid --mhc 145       # one column
#   ./automize/significance.sh --pnet-grid            # 150 x 3 nodes
#   ./automize/significance.sh --grid --skip-existing # resume a killed run
#   ./automize/significance.sh --grid --dry-run
#
# --skip-existing emits only the (point, channel) nodes whose Significance
# output is not already on disk. A DAG killed without writing a rescue file
# (schedd restart) leaves no other way to resume, and the fits are
# deterministic, so re-running a completed one only wastes the slot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/automize/dag_lib.sh"
source "$SCRIPT_DIR/scripts/env.sh"

POINTS=()
GRID_ARMS=()
MHC_FILTER=""
TEMPLATE_POINTS=false
SKIP_EXISTING=false
DRY_RUN=false
CHANNELS=(Combined SR1E2Mu SR3Mu)
ERA="All"
# Measured 33 s / 250 MB per node; 1 GB is already generous.
MEMORY=1024

while [[ $# -gt 0 ]]; do
    case $1 in
        --point) POINTS+=("$2"); shift 2 ;;
        --template-points) TEMPLATE_POINTS=true; shift ;;
        --grid) GRID_ARMS+=("Baseline"); shift ;;
        --pnet-grid) GRID_ARMS+=("ParticleNet"); shift ;;
        --mhc) MHC_FILTER="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        --era) ERA="$2"; shift 2 ;;
        --channels) IFS=',' read -ra CHANNELS <<< "$2"; shift 2 ;;
        --memory) MEMORY="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) grep '^#' "$0" | head -26; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [[ "$TEMPLATE_POINTS" == "true" ]]; then
    while IFS= read -r spec; do
        [[ -n "$spec" ]] && POINTS+=("$spec")
    done < <(python3 - << 'PYEOF'
import sys, os
sys.path.insert(0, os.path.join(os.environ["WORKDIR"],
                                "SignalRegionStudyV4", "python"))
from collectTemplatePlots import DEFAULT_POINTS
for method in sorted(DEFAULT_POINTS):
    for mp in DEFAULT_POINTS[method]:
        print(f"{method}:{mp}")
PYEOF
)
fi

if [[ ${#POINTS[@]} -eq 0 && ${#GRID_ARMS[@]} -eq 0 ]]; then
    echo "ERROR: no points. Use --template-points, --point METHOD:MP," \
         "--grid or --pnet-grid." \
         "(WORKDIR=${WORKDIR:-UNSET} -- did you 'source setup.sh'?)" >&2
    exit 1
fi

JOB_DIR=$(dag_new_jobdir "significance")

# METHOD MASSPOINT SEED MHC CHANNELS for every point of an arm's scan
# grid, taken from the frozen grid config (a group carries its seed and
# its members, so no per-point group_seed lookup is needed).  With
# --skip-existing, CHANNELS holds only the channels whose Significance
# output is not already on disk, and points with none left are dropped.
enumerate_grid() {
    local method=$1
    SRS_METHOD="$method" SRS_MHC="$MHC_FILTER" \
    SRS_CHANNELS="$(IFS=,; echo "${CHANNELS[*]}")" \
    SRS_SKIP="$SKIP_EXISTING" python3 - << 'PYEOF'
import os, sys
sys.path.insert(0, os.path.join(os.environ["WORKDIR"],
                                "SignalRegionStudyV4", "python"))
import srspaths
from interpolation_config import masspoint_name

method = os.environ["SRS_METHOD"]
channels = os.environ["SRS_CHANNELS"].split(",")
skip = os.environ["SRS_SKIP"] == "true"
cfg = (srspaths.grid_config() if method == "Baseline"
       else srspaths.pnet_grid_config())
wanted = {int(x) for x in os.environ["SRS_MHC"].replace(",", " ").split()}


def todo(mp, seed, chans):
    if not skip:
        return chans
    left = []
    for ch in chans:
        base = srspaths.template_dir(mp, method, "All", ch,
                                     source="interp-signal")
        if seed != mp:
            base = srspaths.interp_member_dir(seed, mp, "All", ch,
                                              method=method)
        out = os.path.join(base, "combine_output", "significance",
                           f"higgsCombine.{mp}.{method}.Significance"
                           ".mH120.root")
        if not os.path.exists(out):
            left.append(ch)
    return left


for key in sorted(cfg["grids"], key=lambda k: int(k[3:])):
    mhc = int(key[3:])
    if wanted and mhc not in wanted:
        continue
    for grp in cfg["grids"][key]["groups"]:
        seed = masspoint_name(grp["seed"], mhc)
        for ma in grp["members"]:
            mp = masspoint_name(ma, mhc)
            left = todo(mp, seed, channels)
            if left:
                print(method, mp, seed, mhc, ",".join(left))
PYEOF
}

write_submit() {
    cat > "$1/jobs.sub" << EOF
universe                = vanilla
executable              = $SRS_MODULE_DIR/scripts/interp_templates_wrapper.sh
arguments               = "\$(step) \$(masspoint) \$(seed) \$(era) \$(channel) \$(extra)"
output                  = logs/\$(step).\$(masspoint).\$(channel).\$(arm).out
error                   = logs/\$(step).\$(masspoint).\$(channel).\$(arm).err
log                     = logs/dag.log
request_memory          = $MEMORY
request_cpus            = 1
should_transfer_files   = NO
getenv                  = False
queue
EOF
}

# stdin: "METHOD MASSPOINT SEED [MHC] CHANNELS" lines (CHANNELS comma-
# separated) -> one DAG, one node per (point, channel).
emit_dag() {
    local dag_dir=$1
    local dag_file="$dag_dir/dag.dag"
    local n=0
    write_submit "$dag_dir"
    : > "$dag_file"
    while read -r method masspoint seed chans; do
        [[ -z "$method" ]] && continue
        local extra=""
        [[ "$method" == "ParticleNet" ]] && extra="--method ParticleNet"
        local channel_list
        IFS=',' read -ra channel_list <<< "$chans"
        for channel in "${channel_list[@]}"; do
            local node="sig_${method}_${masspoint}_${channel}"
            {
                echo "JOB $node jobs.sub"
                echo "VARS $node step=\"significance\" masspoint=\"$masspoint\" seed=\"$seed\" era=\"$ERA\" channel=\"$channel\" extra=\"$extra\" arm=\"$method\""
                echo "RETRY $node 1"
            } >> "$dag_file"
            n=$((n + 1))
        done
    done
    echo "CONFIG dagman.config" >> "$dag_file"
    echo "$n"
}

TOTAL_NODES=0
DAG_COUNT=0

# The quoted points: one flat DAG, as before.
if [[ ${#POINTS[@]} -gt 0 ]]; then
    MP_DIR=$(dag_new_masspoint_dir "$JOB_DIR" "points")
    specs=""
    for spec in "${POINTS[@]}"; do
        method="${spec%%:*}"
        masspoint="${spec#*:}"
        if [[ "$method" != "Baseline" && "$method" != "ParticleNet" ]]; then
            echo "ERROR: --point METHOD must be Baseline or ParticleNet: $spec" >&2
            exit 1
        fi
        # The seed owns the group's shared backgrounds, so a member's
        # artifacts nest under it; the wrapper needs the seed to resolve
        # the path.
        seed=$(SRS_MP="$masspoint" SRS_METHOD="$method" python3 - << 'PYEOF'
import sys, os
sys.path.insert(0, os.path.join(os.environ["WORKDIR"],
                                "SignalRegionStudyV4", "python"))
import interpolation_config
print(interpolation_config.group_seed(os.environ["SRS_MP"],
                                      os.environ["SRS_METHOD"]))
PYEOF
)
        specs+="$method $masspoint $seed $(IFS=,; echo "${CHANNELS[*]}")"$'\n'
    done
    n=$(printf '%s' "$specs" | emit_dag "$MP_DIR")
    TOTAL_NODES=$((TOTAL_NODES + n))
    DAG_COUNT=$((DAG_COUNT + 1))
    echo "  points (${#POINTS[@]} points) -> $n nodes"
fi

# The scan grids: one DAG per (arm, mHc), so a column can be resubmitted
# on its own and DAGMan is never handed 7401 nodes at once.
for arm in ${GRID_ARMS[@]+"${GRID_ARMS[@]}"}; do
    grid_specs=$(enumerate_grid "$arm")
    if [[ -z "$grid_specs" ]]; then
        # With --skip-existing an empty list is the success case: the arm
        # is already complete. Without it, it means the grid config or the
        # --mhc filter selected nothing, which is an error.
        if [[ "$SKIP_EXISTING" == "true" ]]; then
            echo "  $arm: nothing left to run (--mhc ${MHC_FILTER:-all})"
            continue
        fi
        echo "ERROR: empty grid for $arm (--mhc $MHC_FILTER)" >&2
        exit 1
    fi
    while IFS= read -r mhc; do
        [[ -z "$mhc" ]] && continue
        MP_DIR=$(dag_new_masspoint_dir "$JOB_DIR" "${arm}_MHc${mhc}")
        n=$(awk -v m="$mhc" '$4 == m {print $1, $2, $3, $5}' <<< "$grid_specs" \
            | emit_dag "$MP_DIR")
        TOTAL_NODES=$((TOTAL_NODES + n))
        DAG_COUNT=$((DAG_COUNT + 1))
        echo "  ${arm}_MHc${mhc} -> $n nodes"
    done < <(awk '{print $4}' <<< "$grid_specs" | sort -n -u)
done

echo "============================================================"
echo "Significance campaign"
if [[ ${#POINTS[@]} -gt 0 ]]; then
    echo "Points:   ${POINTS[*]}"
fi
if [[ ${#GRID_ARMS[@]} -gt 0 ]]; then
    echo "Grids:    ${GRID_ARMS[*]} (mhc: ${MHC_FILTER:-all})"
fi
echo "Era:      $ERA"
echo "Channels: ${CHANNELS[*]}"
echo "DAGs:     $DAG_COUNT"
echo "Nodes:    $TOTAL_NODES"
echo "Memory:   $MEMORY"
echo "Dry run:  $DRY_RUN"
echo "============================================================"

dag_write_submit_all "$JOB_DIR"
dag_write_status_all "$JOB_DIR"
dag_submit_or_dryrun "$JOB_DIR" "$DRY_RUN"
