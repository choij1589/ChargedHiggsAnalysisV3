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
PNET_SCORES=false
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --masspoint)        SINGLE_MASSPOINT="$2"; shift 2 ;;
        --skip-backgrounds) DO_BACKGROUNDS=false;  shift ;;
        --backgrounds-only) DO_SIGNALS=false;      shift ;;
        --skip-particlenet) DO_PARTICLENET=false;  shift ;;
        --pnet-scores)      PNET_SCORES=true;      shift ;;
        --dry-run)          DRY_RUN=true;          shift ;;
        --help)
            echo "Usage: $0 [--masspoint MP] [--skip-backgrounds] [--backgrounds-only]"
            echo "          [--skip-particlenet] [--pnet-scores] [--dry-run]"
            echo ""
            echo "  Default: shared-background DAG + one DAG per mass point"
            echo "  (baseline+particlenet union from configs/masspoints.json)."
            echo "  --masspoint MP       - signals/ParticleNet production for MP only"
            echo "  --skip-backgrounds   - omit the shared-background DAG (already produced)"
            echo "  --backgrounds-only   - only the shared-background DAG"
            echo "  --skip-particlenet   - shared backgrounds/signals only; omit the"
            echo "                         per-masspoint ParticleNet nodes (Baseline-only"
            echo "                         production, which needs no NoHistMode inputs)"
            echo "  --pnet-scores        - ONLY the per-mHc shared-scores DAG"
            echo "                         (preprocess.py --shared-scores, full"
            echo "                         systematics): 5 mHc x 8 eras x"
            echo "                         {SR1E2Mu, SR3Mu, TTZ2E1Mu} = 120 nodes."
            echo "                         Input layout of the ParticleNet"
            echo "                         mA-interpolation chain."
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
    local memory=${3:-2GB}
    local disk=${4:-1GB}
    cat > "$mp_dir/jobs.sub" << EOF
JobBatchName = ${batch}
universe = vanilla
executable = ${WRAPPER_PATH}
arguments = \$(era) \$(channel) \$(masspoint) \$(extra_args)
output = logs/\$(era)_\$(channel)_\$(tag).out
error = logs/\$(era)_\$(channel)_\$(tag).err
log = dag.log
request_cpus = 1
request_memory = ${memory}
request_disk = ${disk}
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

# --- Per-mHc shared-scores DAG (ParticleNet interpolation inputs) ---
# One dir per (era, channel, mHc): every trained mass point with EVERY
# net's score branches plus one shared background/nonprompt/data set
# (samples/{era}/{channel}/{mHc}/). Full systematics -- these dirs are
# template-production inputs, unlike the study's --central-only copies.
if [[ "$PNET_SCORES" == "true" ]]; then
    _pnet_mhcs=$(WORKDIR="$(cd "$SCRIPT_DIR/.." && pwd)/.." python3 -c "
import sys
sys.path.insert(0, '$(cd "$SCRIPT_DIR/.." && pwd)/python')
import pnet_interp_config as pic
print(' '.join(pic.pn_mhc_list()))
") || { echo "ERROR: cannot resolve the trained mHc list"; exit 1; }
    read -ra PNET_MHCS <<< "$_pnet_mhcs"
    scores_dir=$(dag_new_masspoint_dir "$job_dir" "pnet_shared_scores")
    # 4GB memory / 8GB disk: one job writes every trained mass point of its
    # mHc with ALL systematic trees (~370 MB per signal file, 3-4 points)
    # plus the shared backgrounds, all carrying every net's score branches.
    write_jobs_sub "$scores_dir" "preprocess_pnet_scores" "4GB" "8GB"
    {
        echo "# Per-mHc shared-scores preprocessing (ParticleNet interpolation)"
        echo "CONFIG dagman.config"
        echo ""
    } > "$scores_dir/dag.dag"
    for mhc in "${PNET_MHCS[@]}"; do
        for era in "${ERAs[@]}"; do
            for channel in SR1E2Mu SR3Mu TTZ2E1Mu; do
                dag_node "$scores_dir/dag.dag" "scores_${mhc}_${channel}_${era}" \
                    "$era" "$channel" "none" "--shared-scores --mhc ${mhc}" \
                    "scores_${mhc}"
            done
        done
    done
    n_nodes=$(( ${#PNET_MHCS[@]} * ${#ERAs[@]} * 3 ))
    echo "Generated DAG: $scores_dir/dag.dag (${n_nodes} shared-scores nodes)"
    dag_write_submit_all "$job_dir"
    dag_write_status_all "$job_dir"
    dag_submit_or_dryrun "$job_dir" "$DRY_RUN"
    exit 0
fi

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
