#!/bin/bash
# interp-signal template scan driver: one DAG per mHc over configs/grid.json.
#
# Node topology per group (grid.json "groups"):
#   seed_{cat}      4 heavy nodes (Run2|Run3 x SR1E2Mu|SR3Mu), 6 GB —
#                   backgrounds built once with the seed's mean/sigma
#   member_{mp}     1 light node per member point (all 4 categories,
#                   signal injection only), parent = the 4 seed nodes
#   per point:      merge_{tgt} -> datacard_{tgt} -> asymptotic_{tgt}
#                   for tgt in All/{SR1E2Mu,SR3Mu,Combined}
#   validate        1 per group, at the seed's All/Combined datacard
#
# Usage:
#   ./automize/interpTemplates.sh --mhc 160 [--dry-run]
#   ./automize/interpTemplates.sh --all [--dry-run]
#   ./automize/interpTemplates.sh --all --method ParticleNet
#
# --method ParticleNet drives the ParticleNet arm over configs/pnet_grid.json
# (155 points / 15 groups / 5 mHc, reach mA in [82.5, 97.5]): same topology,
# templates at templates/{seed}/ParticleNet/interp-signal/..., score cut and
# eps(mA) from the frozen fits/pnet artifacts.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/automize/dag_lib.sh"
source "$SCRIPT_DIR/scripts/env.sh"

MHC_LIST=()
GROUP_SEED=""
DRY_RUN=false
METHOD="Baseline"
ALL_MHC=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mhc) MHC_LIST+=("$2"); shift 2 ;;
        --all) ALL_MHC=true; shift ;;
        --group) GROUP_SEED="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help)
            grep '^#' "$0" | head -20; exit 0 ;;
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
METHOD_EXTRA=""
[[ "$METHOD" == "ParticleNet" ]] && METHOD_EXTRA="--method ParticleNet"

TARGETS=(SR1E2Mu SR3Mu Combined)
CATEGORIES=("Run2 SR1E2Mu" "Run2 SR3Mu" "Run3 SR1E2Mu" "Run3 SR3Mu")

generate_dag() {
    local mhc=$1
    local dag_dir=$2
    local dag_file="$dag_dir/dag.dag"

    # groups from grid.json via python (masspoint names in p-notation)
    local groups_py
    groups_py=$(WORKDIR="$SRS_REPO_DIR" SRS_INTERP_METHOD="$METHOD" python3 - "$mhc" <<'PYEOF'
import sys, os
sys.path.insert(0, os.path.join(os.environ["WORKDIR"],
                                "SignalRegionStudyV4", "python"))
import srspaths
from interpolation_config import masspoint_name
mhc = int(sys.argv[1])
grid_name = os.environ.get("SRS_INTERP_METHOD", "Baseline")
cfg = (srspaths.grid_config() if grid_name == "Baseline"
       else srspaths.pnet_grid_config())["grids"][f"MHc{mhc}"]
for grp in cfg["groups"]:
    seed = masspoint_name(grp["seed"], mhc)
    members = [masspoint_name(v, mhc) for v in grp["members"]]
    print(seed + ":" + ",".join(m for m in members if m != seed))
PYEOF
    )

    cat > "$dag_dir/jobs.sub" << EOF
universe                = vanilla
executable              = $SRS_MODULE_DIR/scripts/interp_templates_wrapper.sh
arguments               = "\$(step) \$(masspoint) \$(seed) \$(era) \$(channel) \$(extra)"
output                  = logs/\$(step).\$(masspoint).\$(era).\$(channel).out
error                   = logs/\$(step).\$(masspoint).\$(era).\$(channel).err
log                     = logs/dag.log
request_memory          = \$(memory)
request_cpus            = 1
should_transfer_files   = NO
getenv                  = False
queue
EOF

    : > "$dag_file"
    local n_nodes=0

    while IFS= read -r line; do
        [[ -z "$line" ]] && continue
        local seed="${line%%:*}"
        local members_csv="${line#*:}"
        # --group SEED_MP: emit only that group's nodes (verification /
        # ad-hoc single-group runs)
        [[ -n "$GROUP_SEED" && "$seed" != "$GROUP_SEED" ]] && continue

        # 4 heavy seed-category nodes
        local cat_nodes=()
        local era channel
        for cat in "${CATEGORIES[@]}"; do
            read -r era channel <<< "$cat"
            local node="seedtpl_${seed}_${era}_${channel}"
            cat_nodes+=("$node")
            {
                echo "JOB $node jobs.sub"
                echo "VARS $node step=\"template\" masspoint=\"$seed\" seed=\"$seed\" era=\"$era\" channel=\"$channel\" memory=\"6144\" extra=\"$METHOD_EXTRA\""
                echo "RETRY $node 1"
            } >> "$dag_file"
            n_nodes=$((n_nodes + 1))
        done

        # per-point chain generator (seed + members share it)
        emit_point_chain() {
            local mp=$1 parents=$2
            for tgt in "${TARGETS[@]}"; do
                local m="merge_${mp}_${tgt}" d="card_${mp}_${tgt}" a="asym_${mp}_${tgt}"
                {
                    echo "JOB $m jobs.sub"
                    echo "VARS $m step=\"merge\" masspoint=\"$mp\" seed=\"$seed\" era=\"All\" channel=\"$tgt\" memory=\"2048\" extra=\"$METHOD_EXTRA\""
                    echo "JOB $d jobs.sub"
                    echo "VARS $d step=\"datacard\" masspoint=\"$mp\" seed=\"$seed\" era=\"All\" channel=\"$tgt\" memory=\"2048\" extra=\"$METHOD_EXTRA\""
                    echo "JOB $a jobs.sub"
                    echo "VARS $a step=\"asymptotic\" masspoint=\"$mp\" seed=\"$seed\" era=\"All\" channel=\"$tgt\" memory=\"2048\" extra=\"$METHOD_EXTRA\""
                    echo "PARENT $parents CHILD $m"
                    echo "PARENT $m CHILD $d"
                    echo "PARENT $d CHILD $a"
                    echo "RETRY $m 1"
                    echo "RETRY $d 1"
                    echo "RETRY $a 1"
                } >> "$dag_file"
            done
            n_nodes=$((n_nodes + 9))
        }

        emit_point_chain "$seed" "${cat_nodes[*]}"

        # validate once per group, at the seed's All/Combined datacard
        {
            echo "JOB validate_${seed} jobs.sub"
            echo "VARS validate_${seed} step=\"validate\" masspoint=\"$seed\" seed=\"$seed\" era=\"All\" channel=\"Combined\" memory=\"4096\" extra=\"$METHOD_EXTRA\""
            echo "PARENT card_${seed}_Combined CHILD validate_${seed}"
            echo "RETRY validate_${seed} 1"
        } >> "$dag_file"
        n_nodes=$((n_nodes + 1))

        # member nodes: one cheap template job (all 4 categories) per point
        if [[ -n "$members_csv" ]]; then
            IFS=',' read -r -a members <<< "$members_csv"
            for mp in "${members[@]}"; do
                [[ -z "$mp" ]] && continue
                local node="membertpl_${mp}"
                {
                    echo "JOB $node jobs.sub"
                    echo "VARS $node step=\"template\" masspoint=\"$mp\" seed=\"$seed\" era=\"All\" channel=\"Combined\" memory=\"2048\" extra=\"$METHOD_EXTRA\""
                    echo "PARENT ${cat_nodes[*]} CHILD $node"
                    echo "RETRY $node 1"
                } >> "$dag_file"
                n_nodes=$((n_nodes + 1))
                emit_point_chain "$mp" "$node"
            done
        fi
    done <<< "$groups_py"

    echo "CONFIG dagman.config" >> "$dag_file"
    echo "Generated $dag_file: $n_nodes nodes"
}

echo "============================================================"
echo "interp-signal template scan"
echo "Method: $METHOD"
echo "mHc values: ${MHC_LIST[*]}"
echo "Dry run: $DRY_RUN"
echo "============================================================"

JOB_PREFIX="interp_templates"
[[ "$METHOD" == "ParticleNet" ]] && JOB_PREFIX="pnet_interp_templates"
JOB_DIR=$(dag_new_jobdir "$JOB_PREFIX")
for mhc in "${MHC_LIST[@]}"; do
    MP_DIR=$(dag_new_masspoint_dir "$JOB_DIR" "MHc${mhc}")
    generate_dag "$mhc" "$MP_DIR"
done
dag_write_submit_all "$JOB_DIR"
dag_write_status_all "$JOB_DIR"
dag_submit_or_dryrun "$JOB_DIR" "$DRY_RUN"
