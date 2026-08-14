#!/bin/bash
# GoF + impacts campaign over the interp-signal group seeds (mirrors V3's
# gof.sh / impact.sh, adapted to the grouped-template layout: every group
# seed's dir is where these diagnostics live).
#
# Per seed and per GoF target (All x {Combined, SR1E2Mu, SR3Mu}):
#   gofdata (workspace + observed q) -> goftoys_s{1..NBATCHES} -> gofcollect
# Per seed (All/Combined only, V3 convention):
#   impacts (fat node: initial fit + per-nuisance fits --parallel + plots)
#
# Usage:
#   ./automize/interpGofImpacts.sh --all
#   ./automize/interpGofImpacts.sh --mhc 160 [--group MHc160_MA90]
#   Options: --gof-only | --impacts-only, --ntoys N (500), --nbatches N (5),
#            --dry-run
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$SCRIPT_DIR/automize/dag_lib.sh"
source "$SCRIPT_DIR/scripts/env.sh"

MHC_LIST=()
GROUP_SEED=""
DO_GOF=true
DO_IMPACTS=true
NTOYS=500
NBATCHES=5
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --mhc) MHC_LIST+=("$2"); shift 2 ;;
        --all) MHC_LIST=(70 85 100 115 130 145 160); shift ;;
        --group) GROUP_SEED="$2"; shift 2 ;;
        --gof-only) DO_IMPACTS=false; shift ;;
        --impacts-only) DO_GOF=false; shift ;;
        --ntoys) NTOYS="$2"; shift 2 ;;
        --nbatches) NBATCHES="$2"; shift 2 ;;
        --dry-run) DRY_RUN=true; shift ;;
        -h|--help) grep '^#' "$0" | head -14; exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done
[[ ${#MHC_LIST[@]} -gt 0 ]] || { echo "ERROR: --mhc N or --all required"; exit 1; }

GOF_TARGETS=(Combined SR1E2Mu SR3Mu)

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
for grp in srspaths.grid_config()["grids"][f"MHc{mhc}"]["groups"]:
    print(masspoint_name(grp["seed"], mhc))
PYEOF
    )

    cat > "$dag_dir/jobs.sub" << EOF
universe                = vanilla
executable              = $SRS_MODULE_DIR/scripts/interp_templates_wrapper.sh
arguments               = "\$(step) \$(masspoint) \$(seed) \$(era) \$(channel) \$(extra)"
output                  = logs/\$(step).\$(masspoint).\$(era).\$(channel).\$(tag).out
error                   = logs/\$(step).\$(masspoint).\$(era).\$(channel).\$(tag).err
log                     = logs/dag.log
request_memory          = \$(memory)
request_cpus            = \$(cpus)
should_transfer_files   = NO
getenv                  = False
queue
EOF

    : > "$dag_file"
    local n_nodes=0

    while IFS= read -r seed; do
        [[ -z "$seed" ]] && continue
        [[ -n "$GROUP_SEED" && "$seed" != "$GROUP_SEED" ]] && continue

        if [[ "$DO_GOF" == true ]]; then
            for tgt in "${GOF_TARGETS[@]}"; do
                local base="${seed}_${tgt}"
                local toy_nodes=()
                {
                    echo "JOB gofdata_${base} jobs.sub"
                    echo "VARS gofdata_${base} step=\"gof-data\" masspoint=\"$seed\" seed=\"$seed\" era=\"All\" channel=\"$tgt\" extra=\"\" memory=\"2048\" cpus=\"1\" tag=\"data\""
                    echo "RETRY gofdata_${base} 1"
                } >> "$dag_file"
                local s
                for s in $(seq 1 "$NBATCHES"); do
                    toy_nodes+=("goftoys_${base}_s${s}")
                    {
                        echo "JOB goftoys_${base}_s${s} jobs.sub"
                        echo "VARS goftoys_${base}_s${s} step=\"gof-toys\" masspoint=\"$seed\" seed=\"$seed\" era=\"All\" channel=\"$tgt\" extra=\"$s --ntoys $NTOYS --nbatches $NBATCHES\" memory=\"2048\" cpus=\"1\" tag=\"s${s}\""
                        echo "RETRY goftoys_${base}_s${s} 1"
                    } >> "$dag_file"
                done
                {
                    echo "JOB gofcollect_${base} jobs.sub"
                    echo "VARS gofcollect_${base} step=\"gof-collect\" masspoint=\"$seed\" seed=\"$seed\" era=\"All\" channel=\"$tgt\" extra=\"--ntoys $NTOYS --nbatches $NBATCHES\" memory=\"2048\" cpus=\"1\" tag=\"collect\""
                    echo "PARENT gofdata_${base} CHILD ${toy_nodes[*]}"
                    echo "PARENT gofdata_${base} ${toy_nodes[*]} CHILD gofcollect_${base}"
                    echo "RETRY gofcollect_${base} 1"
                } >> "$dag_file"
                n_nodes=$((n_nodes + 2 + NBATCHES))
            done
        fi

        if [[ "$DO_IMPACTS" == true ]]; then
            {
                echo "JOB impacts_${seed} jobs.sub"
                echo "VARS impacts_${seed} step=\"impacts\" masspoint=\"$seed\" seed=\"$seed\" era=\"All\" channel=\"Combined\" extra=\"--parallel 4\" memory=\"4096\" cpus=\"4\" tag=\"impacts\""
                echo "RETRY impacts_${seed} 1"
            } >> "$dag_file"
            # share the workspace with the Combined gof-data node instead of
            # racing text2workspace in the same dir
            if [[ "$DO_GOF" == true ]]; then
                echo "PARENT gofdata_${seed}_Combined CHILD impacts_${seed}" >> "$dag_file"
            fi
            n_nodes=$((n_nodes + 1))
        fi
    done <<< "$seeds"

    echo "CONFIG dagman.config" >> "$dag_file"
    echo "Generated $dag_file: $n_nodes nodes"
}

echo "============================================================"
echo "interp-signal GoF + impacts campaign"
echo "mHc values: ${MHC_LIST[*]}  (gof: $DO_GOF, impacts: $DO_IMPACTS)"
echo "GoF: $NTOYS toys in $NBATCHES batches; targets All x {${GOF_TARGETS[*]}}"
echo "Dry run: $DRY_RUN"
echo "============================================================"

JOB_DIR=$(dag_new_jobdir "interp_gof")
for mhc in "${MHC_LIST[@]}"; do
    MP_DIR=$(dag_new_masspoint_dir "$JOB_DIR" "MHc${mhc}")
    generate_dag "$mhc" "$MP_DIR"
done
dag_write_submit_all "$JOB_DIR"
dag_write_status_all "$JOB_DIR"
dag_submit_or_dryrun "$JOB_DIR" "$DRY_RUN"
