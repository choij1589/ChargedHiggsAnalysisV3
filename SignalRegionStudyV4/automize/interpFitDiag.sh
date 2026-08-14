#!/bin/bash
# FitDiagnostics + prefit/postfit mass plots + nuisance pulls over the
# interp-signal GROUP SEEDS (members share the seed's backgrounds bitwise,
# so per-member fits are near-duplicates -- user decision 2026-08-15).
#
# Per seed, at All/Combined (V3 convention):
#   fitdiag (combine -M FitDiagnostics --saveShapes --saveWithUncertainties)
#     -> plotpostfit (fine-grid refill, prefit + postfit_b + postfit_s)
#      + plotpulls   (diffNuisances, --pull-fit both)
#
# Usage:
#   ./automize/interpFitDiag.sh --all --method ParticleNet   # 15 seeds, 45 nodes
#   ./automize/interpFitDiag.sh --all                        # Baseline: 572 seeds, 1716 nodes
#   ./automize/interpFitDiag.sh --mhc 160 [--group MHc160_MA90] [--dry-run]
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
        --mhc) MHC_LIST+=("${2#MHc}"); shift 2 ;;
        --all) ALL_MHC=true; shift ;;
        --group) GROUP_SEED="$2"; shift 2 ;;
        --method) METHOD="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) grep '^#' "$0" | head -15; exit 0 ;;
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
[[ "$METHOD" == "ParticleNet" ]] && METHOD_EXTRA=" --method ParticleNet"

generate_dag() {
    local mhc=$1
    local dag_dir=$2
    local dag_file="$dag_dir/dag.dag"

    local seeds
    seeds=$(WORKDIR="$SRS_REPO_DIR" SRS_INTERP_METHOD="$METHOD" python3 - "$mhc" <<'PYEOF'
import sys, os
sys.path.insert(0, os.path.join(os.environ["WORKDIR"],
                                "SignalRegionStudyV4", "python"))
import srspaths
from interpolation_config import masspoint_name
mhc = int(sys.argv[1])
method = os.environ.get("SRS_INTERP_METHOD", "Baseline")
cfg = (srspaths.grid_config() if method == "Baseline"
       else srspaths.pnet_grid_config())
for grp in cfg["grids"][f"MHc{mhc}"]["groups"]:
    print(masspoint_name(grp["seed"], mhc))
PYEOF
    )

    cat > "$dag_dir/jobs.sub" << EOF
universe                = vanilla
executable              = $SRS_MODULE_DIR/scripts/interp_templates_wrapper.sh
arguments               = "\$(step) \$(masspoint) \$(seed) \$(era) \$(channel) \$(extra)"
output                  = logs/\$(step).\$(masspoint).out
error                   = logs/\$(step).\$(masspoint).err
log                     = logs/dag.log
request_memory          = \$(memory)
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
        {
            echo "JOB fitdiag_${seed} jobs.sub"
            echo "VARS fitdiag_${seed} step=\"fitdiag\" masspoint=\"$seed\" seed=\"$seed\" era=\"All\" channel=\"Combined\" extra=\"${METHOD_EXTRA# }\" memory=\"2048\""
            echo "JOB plotpostfit_${seed} jobs.sub"
            echo "VARS plotpostfit_${seed} step=\"plotpostfit\" masspoint=\"$seed\" seed=\"$seed\" era=\"All\" channel=\"Combined\" extra=\"${METHOD_EXTRA# }\" memory=\"4096\""
            echo "JOB plotpulls_${seed} jobs.sub"
            echo "VARS plotpulls_${seed} step=\"plotpulls\" masspoint=\"$seed\" seed=\"$seed\" era=\"All\" channel=\"Combined\" extra=\"${METHOD_EXTRA# }\" memory=\"2048\""
            echo "PARENT fitdiag_${seed} CHILD plotpostfit_${seed} plotpulls_${seed}"
            echo "RETRY fitdiag_${seed} 1"
            echo "RETRY plotpostfit_${seed} 1"
            echo "RETRY plotpulls_${seed} 1"
        } >> "$dag_file"
        n_nodes=$((n_nodes + 3))
    done <<< "$seeds"

    echo "CONFIG dagman.config" >> "$dag_file"
    echo "Generated $dag_file: $n_nodes nodes"
}

echo "============================================================"
echo "interp-signal FitDiagnostics + postfit + pulls campaign"
echo "Method: $METHOD"
echo "mHc values: ${MHC_LIST[*]}"
echo "Dry run: $DRY_RUN"
echo "============================================================"

JOB_PREFIX="interp_fitdiag"
[[ "$METHOD" == "ParticleNet" ]] && JOB_PREFIX="pnet_interp_fitdiag"
JOB_DIR=$(dag_new_jobdir "$JOB_PREFIX")
for mhc in "${MHC_LIST[@]}"; do
    MP_DIR=$(dag_new_masspoint_dir "$JOB_DIR" "MHc${mhc}")
    generate_dag "$mhc" "$MP_DIR"
done
dag_write_submit_all "$JOB_DIR"
dag_write_status_all "$JOB_DIR"
dag_submit_or_dryrun "$JOB_DIR" "$DRY_RUN"
