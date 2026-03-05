#!/bin/bash
set -euo pipefail
export PATH="${PWD}/python:${PATH}"

CHANNEL="Combined"
SIGNALS=("MHc130_MA90" "MHc100_MA95" "MHc160_MA85")
LAMBDAS=(0.0 0.005 0.01 0.02 0.05 0.1 0.2 0.5)
PILOT=false

# Parse flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pilot) PILOT=true ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
    shift
done

mkdir -p logs
echo "Lambda sweep | channel=${CHANNEL} | signals=${SIGNALS[*]} | lambdas=${LAMBDAS[*]} | pilot=${PILOT}"

# Launch one launchLambdaSweep.py per signal, each pinned to its own GPU.
# CUDA_VISIBLE_DEVICES restricts all 8 lambda workers of that launcher to one GPU,
# so 3 GPUs run in parallel instead of 24 workers piling onto cuda:0.
pids=(); labels=()
gpu_id=0
for sig in "${SIGNALS[@]}"; do
    label="${sig}_sweep"
    log="logs/sweep_${CHANNEL}_${label}.log"
    extra_args=""
    if $PILOT; then extra_args="--pilot"; fi
    CUDA_VISIBLE_DEVICES=${gpu_id} python python/launchLambdaSweep.py \
        --signal "${sig}" \
        --channel "${CHANNEL}" \
        --lambdas "${LAMBDAS[*]}" \
        ${extra_args} > "${log}" 2>&1 &
    pids+=($!); labels+=("${label}")
    echo "  Launched ${label} on GPU ${gpu_id} → ${log}"
    (( gpu_id++ )) || true
done

# Wait and verify each launcher produced one "Training complete" per lambda
wait
fail=0
for label in "${labels[@]}"; do
    log="logs/sweep_${CHANNEL}_${label}.log"
    count=$(grep -c "Training complete  lambda=" "${log}" 2>/dev/null || echo 0)
    if [[ "${count}" -eq "${#LAMBDAS[@]}" ]]; then
        echo "OK:   ${label} (${count}/${#LAMBDAS[@]})"
    else
        echo "FAIL: ${label} (${count}/${#LAMBDAS[@]})"
        fail=1
    fi
done
[[ $fail -eq 0 ]] || exit 1

echo ""
echo "Training sweep complete. Running per-model visualization..."

viz_pids=(); viz_labels=()
for sig in "${SIGNALS[@]}"; do
    for lam in "${LAMBDAS[@]}"; do
        lam_str="${lam//./p}"           # 0.005 → 0p005
        label="${sig}_lambda${lam}"
        log="logs/viz_${CHANNEL}_${label}.log"
        extra_args=""
        if $PILOT; then extra_args="--pilot"; fi
        python python/visualizeMultiClass.py \
            --signal "${sig}" \
            --channel "${CHANNEL}" \
            --model-name "discoL${lam_str}" \
            ${extra_args} > "${log}" 2>&1 &
        viz_pids+=($!); viz_labels+=("${label}")
        echo "  Launched viz ${label} → ${log}"
    done
done

wait
viz_fail=0
for label in "${viz_labels[@]}"; do
    log="logs/viz_${CHANNEL}_${label}.log"
    if grep -qE "Traceback|SyntaxError|Error" "${log}" 2>/dev/null; then
        echo "VIZ FAIL: ${label}"
        viz_fail=1
    else
        echo "VIZ OK:   ${label}"
    fi
done
[[ $viz_fail -eq 0 ]] || exit 1

echo ""
echo "All done. Next: run comparison plots across lambdas:"
for sig in "${SIGNALS[@]}"; do
    echo "  python python/compareDecorrelation.py --signal ${sig} --channel ${CHANNEL}"
done
