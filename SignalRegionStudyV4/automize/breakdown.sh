#!/bin/bash
# Grouped-nuisance uncertainty breakdown of sigma(r) (docs/BREAKDOWN.md).
#
# Like the significance step, the point list is EXPLICIT: the breakdown is
# a follow-up on the points the analysis quotes, not a scan.
# --template-points takes the curated bundle set (the same list
# collectTemplatePlots.py promotes, read from there so the two cannot
# drift); any other point is named with --point METHOD:MASSPOINT.
#
# Per point the DAG is a fan-out between two barriers:
#
#   setup -> bestfit -> {total, freeze_1 .. freeze_N} (parallel) -> plot
#
# EVERY scan hangs off the one bestfit snapshot, so they share a minimum
# -- quadrature subtraction across scans is only meaningful then. V4 runs
# in place on NFS (should_transfer_files = NO), so the scans simply open
# the snapshot in the point's combine_output/breakdown dir, no staging.
#
# Usage:
#   ./automize/breakdown.sh --template-points
#   ./automize/breakdown.sh --point Baseline:MHc145_MA90
#   ./automize/breakdown.sh --template-points --skip-existing
#   ./automize/breakdown.sh --template-points --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/automize/dag_lib.sh"
source "$SCRIPT_DIR/scripts/env.sh"

POINTS=()
TEMPLATE_POINTS=false
SKIP_EXISTING=false
DRY_RUN=false
CHANNELS=(Combined)
ERA="All"
SCAN_POINTS="200"
SIGMA_WINDOW="5"
# The scan nodes carry a full workspace and a 200-point grid; 4 GB
# matches V3's request and the impacts node.  setup/plot are light.
SCAN_MEMORY=4096
LIGHT_MEMORY=2048

while [[ $# -gt 0 ]]; do
    case $1 in
        --point) POINTS+=("$2"); shift 2 ;;
        --template-points) TEMPLATE_POINTS=true; shift ;;
        --era) ERA="$2"; shift 2 ;;
        --channels) IFS=',' read -ra CHANNELS <<< "$2"; shift 2 ;;
        --scan-points) SCAN_POINTS="$2"; shift 2 ;;
        --sigma-window) SIGMA_WINDOW="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        --memory) SCAN_MEMORY="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) grep '^#' "$0" | head -22; exit 0 ;;
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
    echo "ERROR: no points. Use --template-points or --point METHOD:MP." \
         "(WORKDIR=${WORKDIR:-UNSET} -- did you 'source setup.sh'?)" >&2
    exit 1
fi

# Group count fixes the number of freeze nodes, and the config is the one
# place that decides it.
NGROUPS=$(python3 - << 'PYEOF'
import sys, os
sys.path.insert(0, os.path.join(os.environ["WORKDIR"],
                                "SignalRegionStudyV4", "python"))
import nuisanceGroups
print(len(nuisanceGroups.group_names()))
PYEOF
)
[[ "$NGROUPS" -ge 1 ]] || { echo "ERROR: no nuisance groups configured"; exit 1; }

JOB_DIR=$(dag_new_jobdir "breakdown")
MP_DIR=$(dag_new_masspoint_dir "$JOB_DIR" "points")

cat > "$MP_DIR/jobs.sub" << EOF
universe                = vanilla
executable              = $SRS_MODULE_DIR/scripts/interp_templates_wrapper.sh
arguments               = "\$(step) \$(masspoint) \$(seed) \$(era) \$(channel) \$(extra)"
output                  = logs/\$(step).\$(masspoint).\$(channel).\$(arm).\$(tag).out
error                   = logs/\$(step).\$(masspoint).\$(channel).\$(arm).\$(tag).err
log                     = logs/dag.log
request_memory          = \$(memory)
request_cpus            = 1
should_transfer_files   = NO
getenv                  = False
queue
EOF

DAG_FILE="$MP_DIR/dag.dag"
: > "$DAG_FILE"
TOTAL_NODES=0
SKIPPED=0

for spec in "${POINTS[@]}"; do
    method="${spec%%:*}"
    masspoint="${spec#*:}"
    if [[ "$method" != "Baseline" && "$method" != "ParticleNet" ]]; then
        echo "ERROR: --point METHOD must be Baseline or ParticleNet: $spec" >&2
        exit 1
    fi
    # The seed owns the group's shared backgrounds, so a member's
    # artifacts nest under it; the wrapper needs the seed to resolve the
    # path.
    seed=$(SRS_MP="$masspoint" SRS_METHOD="$method" python3 - << 'PYEOF'
import sys, os
sys.path.insert(0, os.path.join(os.environ["WORKDIR"],
                                "SignalRegionStudyV4", "python"))
import interpolation_config
print(interpolation_config.group_seed(os.environ["SRS_MP"],
                                      os.environ["SRS_METHOD"]))
PYEOF
)
    method_extra=""
    [[ "$method" == "ParticleNet" ]] && method_extra=" --method ParticleNet"
    scan_extra="--points $SCAN_POINTS --sigma-window $SIGMA_WINDOW${method_extra}"

    for channel in "${CHANNELS[@]}"; do
        if [[ "$SKIP_EXISTING" == "true" ]]; then
            done_plot=$(SRS_MP="$masspoint" SRS_SEED="$seed" \
                        SRS_METHOD="$method" SRS_CH="$channel" \
                        SRS_ERA="$ERA" python3 - << 'PYEOF'
import sys, os
sys.path.insert(0, os.path.join(os.environ["WORKDIR"],
                                "SignalRegionStudyV4", "python"))
import srspaths
mp, seed = os.environ["SRS_MP"], os.environ["SRS_SEED"]
method, ch, era = os.environ["SRS_METHOD"], os.environ["SRS_CH"], os.environ["SRS_ERA"]
base = srspaths.template_dir(mp, method, era, ch, source="interp-signal")
if seed != mp:
    base = srspaths.interp_member_dir(seed, mp, era, ch, method=method)
print("yes" if os.path.exists(os.path.join(
    base, "combine_output", "breakdown", "breakdown.png")) else "no")
PYEOF
)
            if [[ "$done_plot" == "yes" ]]; then
                SKIPPED=$((SKIPPED + 1))
                continue
            fi
        fi

        base="${method}_${masspoint}_${channel}"
        freeze_nodes=()
        scan_nodes=("total_${base}")
        {
            echo "JOB setup_${base} jobs.sub"
            echo "VARS setup_${base} step=\"breakdown-setup\" masspoint=\"$masspoint\" seed=\"$seed\" era=\"$ERA\" channel=\"$channel\" extra=\"${method_extra# }\" memory=\"$LIGHT_MEMORY\" arm=\"$method\" tag=\"setup\""
            echo "RETRY setup_${base} 1"
            echo "JOB bestfit_${base} jobs.sub"
            echo "VARS bestfit_${base} step=\"breakdown-bestfit\" masspoint=\"$masspoint\" seed=\"$seed\" era=\"$ERA\" channel=\"$channel\" extra=\"$scan_extra\" memory=\"$SCAN_MEMORY\" arm=\"$method\" tag=\"bestfit\""
            echo "RETRY bestfit_${base} 1"
            echo "JOB total_${base} jobs.sub"
            echo "VARS total_${base} step=\"breakdown-total\" masspoint=\"$masspoint\" seed=\"$seed\" era=\"$ERA\" channel=\"$channel\" extra=\"$scan_extra\" memory=\"$SCAN_MEMORY\" arm=\"$method\" tag=\"total\""
            echo "RETRY total_${base} 1"
            echo "PARENT setup_${base} CHILD bestfit_${base}"
        } >> "$DAG_FILE"
        TOTAL_NODES=$((TOTAL_NODES + 3))

        for idx in $(seq 1 "$NGROUPS"); do
            freeze_nodes+=("freeze_${base}_${idx}")
            scan_nodes+=("freeze_${base}_${idx}")
            {
                echo "JOB freeze_${base}_${idx} jobs.sub"
                echo "VARS freeze_${base}_${idx} step=\"breakdown-freeze\" masspoint=\"$masspoint\" seed=\"$seed\" era=\"$ERA\" channel=\"$channel\" extra=\"$idx $scan_extra\" memory=\"$SCAN_MEMORY\" arm=\"$method\" tag=\"freeze${idx}\""
                echo "RETRY freeze_${base}_${idx} 1"
            } >> "$DAG_FILE"
            TOTAL_NODES=$((TOTAL_NODES + 1))
        done

        {
            echo "JOB plot_${base} jobs.sub"
            echo "VARS plot_${base} step=\"breakdown-plot\" masspoint=\"$masspoint\" seed=\"$seed\" era=\"$ERA\" channel=\"$channel\" extra=\"${method_extra# }\" memory=\"$LIGHT_MEMORY\" arm=\"$method\" tag=\"plot\""
            echo "RETRY plot_${base} 1"
            echo "PARENT bestfit_${base} CHILD ${scan_nodes[*]}"
            echo "PARENT ${scan_nodes[*]} CHILD plot_${base}"
        } >> "$DAG_FILE"
        TOTAL_NODES=$((TOTAL_NODES + 1))
    done
done

echo "CONFIG dagman.config" >> "$DAG_FILE"

echo "============================================================"
echo "Uncertainty breakdown campaign"
echo "Points:   ${POINTS[*]}"
echo "Era:      $ERA"
echo "Channels: ${CHANNELS[*]}"
echo "Groups:   $NGROUPS (+ residual stat)"
echo "Scan:     $SCAN_POINTS points over +-$SIGMA_WINDOW sigma"
[[ "$SKIPPED" -gt 0 ]] && echo "Skipped:  $SKIPPED already-complete (point, channel)"
echo "Nodes:    $TOTAL_NODES"
if [[ "$TOTAL_NODES" -eq 0 ]]; then
    echo "Nothing to run."
    exit 0
fi

dag_write_submit_all "$JOB_DIR"
dag_write_status_all "$JOB_DIR"
dag_submit_or_dryrun "$JOB_DIR" "$DRY_RUN"
