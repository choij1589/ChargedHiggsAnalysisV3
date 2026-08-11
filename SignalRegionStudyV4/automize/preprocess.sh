#!/bin/bash
set -euo pipefail

# Preprocessing driver for the shared-sample layout (V4 default):
#
#   samples/{era}/SR1E2Mu/                  shared bkg/nonprompt/data + all signals
#   samples/{era}/SR3Mu_{lowM,highM}/       shared bkg/nonprompt/data + ALL signals
#                                           (both pairing variants, for interpolation)
#   samples/{era}/{channel}/{masspoint}/    ParticleNet per-masspoint dirs
#                                           (scores + NoHistMode skims; incl. TTZ2E1Mu)
#
# Job structure:
#   shared_backgrounds DAG : 8 eras x {SR1E2Mu, SR3Mu:lowM, SR3Mu:highM} = 24 nodes,
#                            run ONCE for the whole analysis (mass-independent)
#   per-masspoint DAG      : 8 eras x {SR1E2Mu, SR3Mu} shared-signal nodes
#                            (+ 8 eras x {SR1E2Mu, SR3Mu, TTZ2E1Mu} full ParticleNet
#                             nodes for ParticleNet-trained points)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/load_masspoints.sh"
source "$SCRIPT_DIR/dag_lib.sh"

ERAs=("2016preVFP" "2016postVFP" "2017" "2018" "2022" "2022EE" "2023" "2023BPix")

SINGLE_MASSPOINT=""
DO_BACKGROUNDS=true
DO_SIGNALS=true
DO_PARTICLENET=true
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --masspoint)        SINGLE_MASSPOINT="$2"; shift 2 ;;
        --skip-backgrounds) DO_BACKGROUNDS=false;  shift ;;
        --backgrounds-only) DO_SIGNALS=false;      shift ;;
        --skip-particlenet) DO_PARTICLENET=false;  shift ;;
        --dry-run)          DRY_RUN=true;          shift ;;
        --help)
            echo "Usage: $0 [--masspoint MP] [--skip-backgrounds] [--backgrounds-only]"
            echo "          [--skip-particlenet] [--dry-run]"
            echo ""
            echo "  Default: shared-background DAG + one DAG per mass point"
            echo "  (baseline+particlenet union from configs/masspoints.json)."
            echo "  --masspoint MP       - signals/ParticleNet production for MP only"
            echo "  --skip-backgrounds   - omit the shared-background DAG (already produced)"
            echo "  --backgrounds-only   - only the shared-background DAG"
            echo "  --skip-particlenet   - shared backgrounds/signals only; omit the"
            echo "                         per-masspoint ParticleNet nodes (Baseline-only"
            echo "                         production, which needs no NoHistMode inputs)"
            echo "  --dry-run            - generate DAGs without submitting"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

mapfile -t MASSPOINTs_PREPROCESS < <(
    printf '%s\n' "${MASSPOINTs_BASELINE[@]}" "${MASSPOINTs_PARTICLENET[@]}" \
    | awk 'NF && !seen[$0]++'
)
if [[ -n "$SINGLE_MASSPOINT" ]]; then
    found=false
    for mp in "${MASSPOINTs_PREPROCESS[@]}"; do
        [[ "$mp" == "$SINGLE_MASSPOINT" ]] && found=true && break
    done
    if [[ "$found" != "true" ]]; then
        echo "Error: masspoint '$SINGLE_MASSPOINT' is not in the baseline+particlenet set." >&2
        exit 1
    fi
    MASSPOINTs_PREPROCESS=("$SINGLE_MASSPOINT")
fi

is_particlenet() {
    [[ " ${MASSPOINTs_PARTICLENET[*]} " =~ " $1 " ]]
}

WRAPPER_PATH="$(cd "$SCRIPT_DIR/.." && pwd)/scripts/preprocess_wrapper.sh"

write_jobs_sub() {
    local mp_dir=$1
    local batch=$2
    cat > "$mp_dir/jobs.sub" << EOF
JobBatchName = ${batch}
universe = vanilla
executable = ${WRAPPER_PATH}
arguments = \$(era) \$(channel) \$(masspoint) \$(extra_args)
output = logs/\$(era)_\$(channel)_\$(tag).out
error = logs/\$(era)_\$(channel)_\$(tag).err
log = dag.log
request_cpus = 1
request_memory = 2GB
request_disk = 1GB
should_transfer_files = NO
use_x509userproxy = True
x509userproxy = /tmp/x509up_u$(id -u)
queue
EOF
}

dag_node() {
    local dag_file=$1 name=$2 era=$3 channel=$4 masspoint=$5 extra=$6 tag=$7
    echo "JOB ${name} jobs.sub" >> "$dag_file"
    echo "VARS ${name} era=\"${era}\" channel=\"${channel}\" masspoint=\"${masspoint}\" extra_args=\"${extra}\" tag=\"${tag}\"" >> "$dag_file"
}

job_dir=$(dag_new_jobdir "preprocess")

# --- Shared-background DAG (mass-independent; run once) ---
if [[ "$DO_BACKGROUNDS" == "true" ]]; then
    bkg_dir=$(dag_new_masspoint_dir "$job_dir" "shared_backgrounds")
    write_jobs_sub "$bkg_dir" "preprocess_shared_bkg"
    {
        echo "# Shared-background preprocessing (mass-independent)"
        echo "CONFIG dagman.config"
        echo ""
    } > "$bkg_dir/dag.dag"
    for era in "${ERAs[@]}"; do
        dag_node "$bkg_dir/dag.dag" "bkg_SR1E2Mu_${era}" "$era" "SR1E2Mu" "none" \
            "--shared-backgrounds" "bkg"
        for pairing in lowM highM; do
            dag_node "$bkg_dir/dag.dag" "bkg_SR3Mu_${pairing}_${era}" "$era" "SR3Mu" "none" \
                "--shared-backgrounds --pairing ${pairing}" "bkg_${pairing}"
        done
    done
    echo "Generated DAG: $bkg_dir/dag.dag (24 shared-background nodes)"
fi

# --- Per-masspoint DAGs: shared signals (+ full ParticleNet production) ---
if [[ "$DO_SIGNALS" == "true" ]]; then
    for masspoint in "${MASSPOINTs_PREPROCESS[@]}"; do
        mp_dir=$(dag_new_masspoint_dir "$job_dir" "$masspoint")
        write_jobs_sub "$mp_dir" "preprocess_${masspoint}"
        {
            echo "# Preprocess DAG for $masspoint (shared signals + ParticleNet)"
            echo "CONFIG dagman.config"
            echo ""
        } > "$mp_dir/dag.dag"

        for era in "${ERAs[@]}"; do
            for channel in SR1E2Mu SR3Mu; do
                dag_node "$mp_dir/dag.dag" "sig_${channel}_${era}" "$era" "$channel" \
                    "$masspoint" "--shared-signal" "sig"
            done
        done

        if [[ "$DO_PARTICLENET" == "true" ]] && is_particlenet "$masspoint"; then
            for era in "${ERAs[@]}"; do
                for channel in SR1E2Mu SR3Mu TTZ2E1Mu; do
                    dag_node "$mp_dir/dag.dag" "pnet_${channel}_${era}" "$era" "$channel" \
                        "$masspoint" "" "pnet"
                done
            done
        fi
        echo "Generated DAG: $mp_dir/dag.dag"
    done
fi

dag_write_submit_all "$job_dir"
dag_write_status_all "$job_dir"

echo ""
echo "============================================================"
echo "Shared-background DAG: $DO_BACKGROUNDS | Signal DAGs: $DO_SIGNALS | ParticleNet nodes: $DO_PARTICLENET"
[[ "$DO_SIGNALS" == "true" ]] && echo "Mass points: ${#MASSPOINTs_PREPROCESS[@]}"
dag_submit_or_dryrun "$job_dir" "$DRY_RUN"
