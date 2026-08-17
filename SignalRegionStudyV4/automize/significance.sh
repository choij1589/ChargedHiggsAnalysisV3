#!/bin/bash
# Observed local significance at selected mass points, All x 3 channels.
#
# Not part of the scan: the significance is a follow-up on points the
# limit scan singles out, so the point list is explicit. --template-points
# takes the curated bundle set (the same list collectTemplatePlots.py
# promotes, read from there so the two cannot drift); --point adds any
# other point, e.g. the extremes of an obs/exp sweep.
#
# One node per (point, channel); the fit is cheap, so they all run flat.
#
# Usage:
#   ./automize/significance.sh --template-points
#   ./automize/significance.sh --point Baseline:MHc145_MA17p5 \
#                              --point Baseline:MHc160_MA39
#   ./automize/significance.sh --template-points --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/automize/dag_lib.sh"
source "$SCRIPT_DIR/scripts/env.sh"

POINTS=()
TEMPLATE_POINTS=false
DRY_RUN=false
CHANNELS=(Combined SR1E2Mu SR3Mu)
ERA="All"
MEMORY=4096

while [[ $# -gt 0 ]]; do
    case $1 in
        --point) POINTS+=("$2"); shift 2 ;;
        --template-points) TEMPLATE_POINTS=true; shift ;;
        --era) ERA="$2"; shift 2 ;;
        --channels) IFS=',' read -ra CHANNELS <<< "$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) grep '^#' "$0" | head -16; exit 0 ;;
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

if [[ ${#POINTS[@]} -eq 0 ]]; then
    echo "ERROR: no points. Use --template-points and/or --point METHOD:MP." \
         "(WORKDIR=${WORKDIR:-UNSET} -- did you 'source setup.sh'?)" >&2
    exit 1
fi

JOB_DIR=$(dag_new_jobdir "significance")
MP_DIR=$(dag_new_masspoint_dir "$JOB_DIR" "points")
DAG_FILE="$MP_DIR/dag.dag"

cat > "$MP_DIR/jobs.sub" << EOF
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

: > "$DAG_FILE"
n_nodes=0
for spec in "${POINTS[@]}"; do
    method="${spec%%:*}"
    masspoint="${spec#*:}"
    if [[ "$method" != "Baseline" && "$method" != "ParticleNet" ]]; then
        echo "ERROR: --point METHOD must be Baseline or ParticleNet: $spec" >&2
        exit 1
    fi
    # The seed owns the group's shared backgrounds, so a member's artifacts
    # nest under it; the wrapper needs the seed to resolve the path.
    seed=$(SRS_MP="$masspoint" SRS_METHOD="$method" python3 - << 'PYEOF'
import sys, os
sys.path.insert(0, os.path.join(os.environ["WORKDIR"],
                                "SignalRegionStudyV4", "python"))
import interpolation_config
print(interpolation_config.group_seed(os.environ["SRS_MP"],
                                      os.environ["SRS_METHOD"]))
PYEOF
)
    extra=""
    [[ "$method" == "ParticleNet" ]] && extra="--method ParticleNet"
    for channel in "${CHANNELS[@]}"; do
        node="sig_${method}_${masspoint}_${channel}"
        {
            echo "JOB $node jobs.sub"
            echo "VARS $node step=\"significance\" masspoint=\"$masspoint\" seed=\"$seed\" era=\"$ERA\" channel=\"$channel\" extra=\"$extra\" arm=\"$method\""
            echo "RETRY $node 1"
        } >> "$DAG_FILE"
        n_nodes=$((n_nodes + 1))
    done
done
echo "CONFIG dagman.config" >> "$DAG_FILE"

echo "============================================================"
echo "Significance campaign"
echo "Points:   ${POINTS[*]}"
echo "Era:      $ERA"
echo "Channels: ${CHANNELS[*]}"
echo "Nodes:    $n_nodes"
echo "Dry run:  $DRY_RUN"
echo "============================================================"

dag_write_submit_all "$JOB_DIR"
dag_write_status_all "$JOB_DIR"
dag_submit_or_dryrun "$JOB_DIR" "$DRY_RUN"
