#!/bin/bash
# MC-vs-interpolation closure of the final binned signal template, one node
# per (mass point, run period, channel).
#
# Scope is the mass points that HAVE MC: configs/masspoints.json 'baseline'
# (78) and 'particlenet' (17) -- exactly the mc_points of the two scan
# grids -- times {Run2, Run3} x {SR1E2Mu, SR3Mu}. 312 + 68 = 380 nodes.
# Every node is independent (it only reads frozen templates and frozen
# samples), so the DAG carries no PARENT/CHILD edges.
#
# Usage:
#   ./automize/templateClosure.sh --method Baseline
#   ./automize/templateClosure.sh --method ParticleNet
#   ./automize/templateClosure.sh --masspoint MHc130_MA90
#   ./automize/templateClosure.sh --mhc 160 --skip-existing
#   ./automize/templateClosure.sh --method Baseline --dry-run
#
# --skip-existing emits only the nodes whose closure PNG is not already on
# disk: the job is deterministic, so re-running a finished one only wastes
# a slot.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/automize/dag_lib.sh"
source "$SCRIPT_DIR/automize/load_masspoints.sh"
source "$SCRIPT_DIR/scripts/env.sh"

METHOD="Baseline"
POINTS=()
MHC_FILTER=""
SKIP_EXISTING=false
DRY_RUN=false
ERAS=(Run2 Run3)
CHANNELS=(SR1E2Mu SR3Mu)
# Measured on Baseline MHc130_MA90/Run2/SR1E2Mu; the ParticleNet arm is the
# heavy case (RDataFrame over the shared-scores files).
MEMORY=4096

while [[ $# -gt 0 ]]; do
    case $1 in
        --method) METHOD="$2"; shift 2 ;;
        --masspoint) POINTS+=("$2"); shift 2 ;;
        --mhc) MHC_FILTER="$2"; shift 2 ;;
        --skip-existing) SKIP_EXISTING=true; shift ;;
        --eras) IFS=',' read -ra ERAS <<< "$2"; shift 2 ;;
        --channels) IFS=',' read -ra CHANNELS <<< "$2"; shift 2 ;;
        --memory) MEMORY="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) grep '^#' "$0" | head -20; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

if [[ "$METHOD" != "Baseline" && "$METHOD" != "ParticleNet" ]]; then
    echo "ERROR: --method must be Baseline or ParticleNet" >&2
    exit 1
fi

if [[ ${#POINTS[@]} -eq 0 ]]; then
    if [[ "$METHOD" == "ParticleNet" ]]; then
        POINTS=("${MASSPOINTs_PARTICLENET[@]}")
    else
        POINTS=("${MASSPOINTs_BASELINE[@]}")
    fi
fi

if [[ -n "$MHC_FILTER" ]]; then
    _filtered=()
    for mp in "${POINTS[@]}"; do
        [[ "$mp" == "MHc${MHC_FILTER}_"* ]] && _filtered+=("$mp")
    done
    POINTS=(${_filtered[@]+"${_filtered[@]}"})
    if [[ ${#POINTS[@]} -eq 0 ]]; then
        echo "ERROR: --mhc $MHC_FILTER selected no mass point of $METHOD" >&2
        exit 1
    fi
fi

# "MASSPOINT SEED ERA CHANNEL" per node. The seed is what the wrapper needs
# to resolve a member's nested template dir; --skip-existing drops the
# (point, category) pairs whose PNG already exists.
enumerate() {
    SRS_METHOD="$METHOD" SRS_POINTS="${POINTS[*]}" \
    SRS_ERAS="${ERAS[*]}" SRS_CHANNELS="${CHANNELS[*]}" \
    SRS_SKIP="$SKIP_EXISTING" python3 - << 'PYEOF'
import os
import sys
sys.path.insert(0, os.path.join(os.environ["WORKDIR"],
                                "SignalRegionStudyV4", "python"))
import interpolation_config
import run_period_utils
import srspaths

method = os.environ["SRS_METHOD"]
skip = os.environ["SRS_SKIP"] == "true"
for mp in os.environ["SRS_POINTS"].split():
    seed = interpolation_config.group_seed(mp, method)
    for era in os.environ["SRS_ERAS"].split():
        for channel in os.environ["SRS_CHANNELS"].split():
            if seed == mp:
                leaf = srspaths.template_dir(mp, method, era, channel,
                                             source="interp-signal")
            else:
                leaf = srspaths.interp_member_dir(seed, mp, era, channel,
                                                  method=method)
            if skip:
                cat = run_period_utils.category_name(channel, era)
                png = os.path.join(leaf, "closure", f"closure.{cat}.png")
                if os.path.exists(png):
                    continue
            print(mp, seed, era, channel)
PYEOF
}

write_submit() {
    cat > "$1/jobs.sub" << EOF
universe                = vanilla
executable              = $SRS_MODULE_DIR/scripts/interp_templates_wrapper.sh
arguments               = "\$(step) \$(masspoint) \$(seed) \$(era) \$(channel) \$(extra)"
output                  = logs/\$(step).\$(masspoint).\$(era).\$(channel).out
error                   = logs/\$(step).\$(masspoint).\$(era).\$(channel).err
log                     = logs/dag.log
request_memory          = $MEMORY
request_cpus            = 1
should_transfer_files   = NO
getenv                  = False
queue
EOF
}

# stdin: "MASSPOINT SEED ERA CHANNEL" lines -> one flat DAG.
emit_dag() {
    local dag_dir=$1
    local dag_file="$dag_dir/dag.dag"
    local n=0
    local extra=""
    [[ "$METHOD" == "ParticleNet" ]] && extra="--method ParticleNet"
    write_submit "$dag_dir"
    : > "$dag_file"
    while read -r masspoint seed era channel; do
        [[ -z "$masspoint" ]] && continue
        local node="clos_${masspoint}_${era}_${channel}"
        {
            echo "JOB $node jobs.sub"
            echo "VARS $node step=\"template-closure\" masspoint=\"$masspoint\" seed=\"$seed\" era=\"$era\" channel=\"$channel\" extra=\"$extra\""
            echo "RETRY $node 1"
        } >> "$dag_file"
        n=$((n + 1))
    done
    echo "CONFIG dagman.config" >> "$dag_file"
    echo "$n"
}

SPECS=$(enumerate)
if [[ -z "$SPECS" ]]; then
    if [[ "$SKIP_EXISTING" == "true" ]]; then
        echo "Nothing left to run for $METHOD (--mhc ${MHC_FILTER:-all})"
        exit 0
    fi
    echo "ERROR: no nodes enumerated for $METHOD" >&2
    exit 1
fi

JOB_DIR=$(dag_new_jobdir "template_closure")
MP_DIR=$(dag_new_masspoint_dir "$JOB_DIR" "$METHOD")
TOTAL_NODES=$(printf '%s\n' "$SPECS" | emit_dag "$MP_DIR")

echo "============================================================"
echo "Template closure (MC vs interpolation)"
echo "Method:   $METHOD"
echo "Points:   ${#POINTS[@]} (mhc: ${MHC_FILTER:-all})"
echo "Eras:     ${ERAS[*]}"
echo "Channels: ${CHANNELS[*]}"
echo "Nodes:    $TOTAL_NODES"
echo "Memory:   $MEMORY"
echo "Dry run:  $DRY_RUN"
echo "============================================================"

dag_write_submit_all "$JOB_DIR"
dag_write_status_all "$JOB_DIR"
dag_submit_or_dryrun "$JOB_DIR" "$DRY_RUN"
