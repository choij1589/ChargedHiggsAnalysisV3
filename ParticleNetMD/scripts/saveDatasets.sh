#!/bin/bash
# saveDatasets.sh - Create datasets for Mass-Decorrelated ParticleNet
#
# 4 sample types:
#   signal    - Tight+Bjet (standard conversion)
#   nonprompt - LNT+Bjet + FR weights (9 TTLL variants)
#   diboson   - 0-tag promoted + calibration weights (WZ + ZZ)
#   ttX       - Tight+Bjet (TTZ + tZq)
#
# Era-dependent samples (WZ, TTZ) are processed in two waves:
#   Wave 1: all independent jobs (no append dependency) run in parallel
#   Wave 2: run3 append jobs (must follow wave 1 per output file)
#
# Parallelism: controlled by JOBS env var (default: nproc)
#   export JOBS=64; ./scripts/saveDatasets.sh
#
# "Combined" channel is handled by DynamicDatasetLoader at training time.

set -euo pipefail
export PATH="${PWD}/python:${PATH}"

JOBS="${JOBS:-$(nproc)}"

channels=("Run1E2Mu" "Run3Mu")

# ===========================================================================
# Signals (Tight+Bjet, all eras)
# ===========================================================================
signals=("TTToHcToWAToMuMu-MHc100_MA95"
         "TTToHcToWAToMuMu-MHc115_MA87"
         "TTToHcToWAToMuMu-MHc130_MA90"
         "TTToHcToWAToMuMu-MHc145_MA92"
         "TTToHcToWAToMuMu-MHc160_MA85"
         "TTToHcToWAToMuMu-MHc160_MA98")

# ===========================================================================
# Nonprompt (LNT+Bjet + FR weights, all eras, 9 TTLL variants)
# ===========================================================================
nonprompt=("Skim_TriLep_TTLL_powheg"
           "Skim_TriLep_TTLL_mtop171p5_powheg"
           "Skim_TriLep_TTLL_mtop173p5_powheg"
           "Skim_TriLep_TTLL_TuneCP5up_powheg"
           "Skim_TriLep_TTLL_TuneCP5down_powheg"
           "Skim_TriLep_TTLL_TuneCP5CR1_powheg"
           "Skim_TriLep_TTLL_TuneCP5CR2_powheg"
           "Skim_TriLep_TTLL_hdamp_up_powheg"
           "Skim_TriLep_TTLL_hdamp_down_powheg")

# ===========================================================================
# Diboson (0-tag promoted + calibration weights)
#   ZZ: same name across all eras
#   WZ: run2=amcatnlo, run3=powheg → merge into Skim_TriLep_WZTo3LNu
# ===========================================================================
diboson_zz="Skim_TriLep_ZZTo4L_powheg"
wz_run2="Skim_TriLep_WZTo3LNu_amcatnlo"
wz_run3="Skim_TriLep_WZTo3LNu_powheg"
wz_output="Skim_TriLep_WZTo3LNu"

# ===========================================================================
# ttX (Tight+Bjet)
#   tZq: same name across all eras
#   TTZ: run2=TTZToLLNuNu, run3=TTZ_M50 → merge into Skim_TriLep_TTZ
# ===========================================================================
ttx_tZq="Skim_TriLep_tZq"
ttz_run2="Skim_TriLep_TTZToLLNuNu"
ttz_run3="Skim_TriLep_TTZ_M50"
ttz_output="Skim_TriLep_TTZ"

# ===========================================================================
# Helper function
# run_save sample sample_type channel [extra_args...]
# ===========================================================================
run_save() {
    local sample="$1"
    local sample_type="$2"
    local channel="$3"
    local extra_args="${4:-}"
    echo "Processing: ${sample_type}/${sample} for channel ${channel} ${extra_args}"
    python3 python/saveDataset.py --sample "${sample}" --sample-type "${sample_type}" \
        --channel "${channel}" ${extra_args}
}
export -f run_save

echo "============================================================"
echo "Creating datasets for Mass-Decorrelated ParticleNet"
echo "  Node features: [E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]"
echo "  Mass info: mass1, mass2 (OS muon pairs)"
echo "  Parallel jobs: ${JOBS}"
echo "============================================================"
echo ""

# ===========================================================================
# Wave 1: all independent first-pass jobs
#   - signals × channels
#   - nonprompt × channels
#   - diboson ZZ × channels
#   - diboson WZ run2 pass × channels  (creates output files)
#   - ttX tZq × channels
#   - ttX TTZ run2 pass × channels     (creates output files)
# All jobs write to distinct output directories → fully parallel.
# ===========================================================================
echo "--- Wave 1: all independent jobs (${JOBS} parallel) ---"

{
    # Signals
    for sig in "${signals[@]}"; do
        for ch in "${channels[@]}"; do
            printf 'signal:::%s:::%s:::\n' "${sig}" "${ch}"
        done
    done

    # Nonprompt
    for smp in "${nonprompt[@]}"; do
        for ch in "${channels[@]}"; do
            printf 'nonprompt:::%s:::%s:::\n' "${smp}" "${ch}"
        done
    done

    # Diboson ZZ (all eras, single name)
    for ch in "${channels[@]}"; do
        printf 'diboson:::%s:::%s:::\n' "${diboson_zz}" "${ch}"
    done

    # Diboson WZ run2 pass (creates output files for wave-2 append)
    for ch in "${channels[@]}"; do
        printf 'diboson:::%s:::%s:::--eras run2 --output-name %s\n' \
            "${wz_run2}" "${ch}" "${wz_output}"
    done

    # ttX tZq (all eras, single name)
    for ch in "${channels[@]}"; do
        printf 'ttX:::%s:::%s:::\n' "${ttx_tZq}" "${ch}"
    done

    # ttX TTZ run2 pass (creates output files for wave-2 append)
    for ch in "${channels[@]}"; do
        printf 'ttX:::%s:::%s:::--eras run2 --output-name %s\n' \
            "${ttz_run2}" "${ch}" "${ttz_output}"
    done

} | parallel --jobs "${JOBS}" --colsep ':::' run_save {2} {1} {3} {4}

echo ""

# ===========================================================================
# Wave 2: append jobs (run3 passes that extend wave-1 output files)
#   Each pair (WZ-run3, TTZ-run3) per channel is independent → parallel.
# ===========================================================================
echo "--- Wave 2: append jobs (run3 passes) ---"

{
    for ch in "${channels[@]}"; do
        printf 'diboson:::%s:::%s:::--eras run3 --output-name %s --append\n' \
            "${wz_run3}" "${ch}" "${wz_output}"
        printf 'ttX:::%s:::%s:::--eras run3 --output-name %s --append\n' \
            "${ttz_run3}" "${ch}" "${ttz_output}"
    done
} | parallel --jobs "${JOBS}" --colsep ':::' run_save {2} {1} {3} {4}

echo ""

# ===========================================================================
# Summary
# ===========================================================================
DATASET_DIR="${WORKDIR}/ParticleNetMD/dataset/samples"
echo "============================================================"
echo "Finished. Total files: $(find "${DATASET_DIR}" -name "*.pt" 2>/dev/null | wc -l)"
echo "Dataset location: ${DATASET_DIR}"
echo "============================================================"
