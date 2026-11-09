#!/bin/bash
# Profile the dominant CUDA kernels of one GA worker's training with Nsight Compute (ncu).
#
# What it profiles: python/profileKernels.py, one process training one model the way a GA worker does,
# on a small cached slice of the data. ncu replays every kernel many times and serializes launches, so the
# GA itself (16 processes, ~6 M launches) is out of reach. Only kernels inside the driver's "ncu_measure"
# NVTX range are profiled (warm-up epochs are skipped), and each kernel gets its own ncu invocation:
# one combined filter with a launch count would never reach the later kernels.
#
# Kernel regexes match the full demangled name (--kernel-name-base demangled). Defaults are the top kernels
# by GPU time in the nsys run (docs/PROFILE.md, section 5.8).
#
# Needs a GPU node; refuses to run on a login node. On Perlmutter the DCGM host engine can hold the
# profiling counters ("a driver resource was unavailable"); DCGM_PAUSE=1 runs `dcgmi profile --pause`
# before and `--resume` after. Pausing affects the whole node: use it only on a node you have to yourself.
#
# Usage:  ./scripts/ncuKernels.sh <signal> <channel> [profileKernels.py options, e.g. --pilot --nNodes 256]
#   e.g.  ./scripts/ncuKernels.sh MHc130_MA90 Combined --pilot
# Env:    KERNELS='cutlass_80_simt_sgemm|knn_kernel|...'  kernel regexes; each alternative is its own ncu run
#         COUNT=8          launches profiled per kernel (--launch-count)
#         SET=full         ncu section set (basic | detailed | full)
#         CLOCK=base       --clock-control (base: ncu default, locked base clocks; none: run at boost clocks)
#         NCU_BIN=<ncu>    default: ncu in PATH, else the hpc_sdk Nsight Compute in /opt
#         DCGM_PAUSE=1     pause/resume DCGM profiling around the runs (node-wide)
#         OUT=<dir>        default profiling/ncu/<signal>_<channel>_n<nNodes>
#         FORCE=1          run even on a login node (not recommended; DCGM blocks the counters there)
# Output in $OUT, per kernel tag T: ncu_T.ncu-rep (git-ignored), details_T.txt, metrics_T.csv, run_T.log

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "usage: $0 <signal> <channel> [profileKernels.py options]   e.g. $0 MHc130_MA90 Combined --pilot"
    exit 1
fi
SIGNAL=$1
CHANNEL=$2
shift 2
DRIVER_ARGS=("$@")

HERE=$(cd "$(dirname "$0")/.." && pwd)     # ParticleNetMD/
cd "$HERE"
export PATH="${PWD}/python:${PATH}"

KERNELS=${KERNELS:-cutlass_80_simt_sgemm|knn_kernel|ampere_sgemm_32x128_tn|reduce_kernel|batch_norm_collect_statistics}
COUNT=${COUNT:-8}
SET=${SET:-full}
CLOCK=${CLOCK:-base}
# Default output folder is tagged with the layer width: cuBLAS picks GEMM kernels per matrix shape,
# so the kernel set differs between widths (e.g. cutlass_80_simt_sgemm only appears at --nNodes 256)
NNODES=128
for ((i = 0; i < ${#DRIVER_ARGS[@]}; i++)); do
    case "${DRIVER_ARGS[$i]}" in
        --nNodes) NNODES="${DRIVER_ARGS[$((i + 1))]:-128}";;
        --nNodes=*) NNODES="${DRIVER_ARGS[$i]#--nNodes=}";;
    esac
done
OUT=${OUT:-profiling/ncu/${SIGNAL}_${CHANNEL}_n${NNODES}}
if [ -z "${NCU_BIN:-}" ]; then
    NCU_BIN=$(command -v ncu || echo /opt/nvidia/hpc_sdk/Linux_x86_64/26.5/profilers/13.2/Nsight_Compute/ncu)
fi

# Metrics read from the report (requested explicitly: a section set does not necessarily contain them,
# and a report can only show what it collected)
STALLS=""
for s in long_scoreboard short_scoreboard no_instruction math_pipe_throttle mio_throttle lg_throttle wait barrier \
         branch_resolving imc_miss not_selected selected; do
    STALLS="${STALLS},smsp__average_warps_issue_stalled_${s}_per_issue_active.ratio"
done
M=gpu__time_duration.sum,launch__grid_size,launch__block_size,launch__registers_per_thread
M=${M},launch__occupancy_limit_registers,launch__occupancy_limit_shared_mem,launch__occupancy_limit_blocks
M=${M},launch__waves_per_multiprocessor,sm__warps_active.avg.pct_of_peak_sustained_active
M=${M},sm__throughput.avg.pct_of_peak_sustained_elapsed,gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed
M=${M},dram__throughput.avg.pct_of_peak_sustained_elapsed,smsp__issue_active.avg.pct_of_peak_sustained_active
M=${M},sm__inst_executed.avg.per_cycle_active,smsp__thread_inst_executed_per_inst_executed.ratio
M=${M},sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active
M=${M},l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,dram__bytes_read.sum,dram__bytes_write.sum
M="${M}${STALLS}"

# --- checks before doing anything --------------------------------------------------------------
case "$(hostname)" in
    login*)
        if [ "${FORCE:-0}" != 1 ]; then
            echo "[ncu] $(hostname) is a login node; get a GPU node first, e.g."
            echo "      salloc -A m2612 -C gpu -q interactive -N 1 --gpus=1 -c 64 -t 01:00:00"
            echo "      (FORCE=1 overrides this check)"
            exit 1
        fi;;
esac
nvidia-smi -L > /dev/null 2>&1 || { echo "[ncu] no GPU visible on $(hostname) (nvidia-smi -L failed)"; exit 1; }
[ -x "$NCU_BIN" ] || { echo "[ncu] ncu not found at '$NCU_BIN' (set NCU_BIN)"; exit 1; }
case "$CLOCK" in base|none) ;; *) echo "[ncu] CLOCK must be base or none, got '$CLOCK'"; exit 1;; esac
[[ "$COUNT" =~ ^[1-9][0-9]*$ ]] || { echo "[ncu] COUNT must be a positive integer, got '$COUNT'"; exit 1; }

mkdir -p "$OUT"

# --- DCGM: optional pause, always resumed on exit ----------------------------------------------
if [ "${DCGM_PAUSE:-0}" = 1 ]; then
    command -v dcgmi > /dev/null 2>&1 || { echo "[ncu] DCGM_PAUSE=1 but dcgmi is not in PATH"; exit 1; }
    dcgmi profile --pause
    trap 'dcgmi profile --resume || echo "[ncu] WARNING: dcgmi profile --resume failed"' EXIT
    echo "[ncu] DCGM profiling paused (node-wide) until this script exits"
fi

# --- data slice: build/cache it once, outside ncu ----------------------------------------------
echo "[ncu] preparing the data slice (cached after the first run)"
python "$(command -v profileKernels.py)" --signal "$SIGNAL" --channel "$CHANNEL" "${DRIVER_ARGS[@]}" --prepare-only

# --- profile one kernel: $1 = kernel regex ---------------------------------------------------------
profile_one() {
    local K=$1
    local TAG
    TAG=$(echo "$K" | tr -c 'A-Za-z0-9_\n' '_')
    local LOG="$OUT/run_${TAG}.log" REP="$OUT/ncu_${TAG}"
    echo "[ncu] ${K}: up to ${COUNT} launches inside ncu_measure, set=${SET}, clock-control=${CLOCK}"
    set +e
    "$NCU_BIN" --nvtx --nvtx-include "ncu_measure/" \
        --kernel-name-base demangled --kernel-name "regex:${K}" --launch-count "$COUNT" \
        --set "$SET" --metrics "$M" --clock-control "$CLOCK" --force-overwrite -o "$REP" \
        python "$(command -v profileKernels.py)" --signal "$SIGNAL" --channel "$CHANNEL" "${DRIVER_ARGS[@]}" \
        > "$LOG" 2>&1
    local rc=$?
    set -e
    echo "[ncu] ${K}: ncu exited with status ${rc}"
    if grep -q "ERR_NVGPUCTRPERM" "$LOG"; then
        echo "[ncu] GPU performance counters are restricted to admins on this node (ERR_NVGPUCTRPERM); see $LOG"
        return 1
    fi
    if grep -q "driver resource was unavailable" "$LOG"; then
        echo "[ncu] the profiling counters are in use (DCGM: pgrep -a nv-hostengine)."
        echo "      On a node you have to yourself, rerun with DCGM_PAUSE=1."
        return 1
    fi
    if grep -q "No kernels were profiled" "$LOG"; then
        echo "[ncu] ${K}: no kernel matched inside ncu_measure. The kernel set depends on the shapes: cuBLAS"
        echo "      picks GEMM kernels per width (e.g. cutlass_80_simt_sgemm only at --nNodes 256). Check the"
        echo "      regex against an nsys kernel list of the driver with the same options."
        return 1
    fi
    [ -e "${REP}.ncu-rep" ] || { echo "[ncu] ${K}: no report produced; see $LOG"; return 1; }

    # readable outputs (reading a report needs no GPU)
    "$NCU_BIN" -i "${REP}.ncu-rep" --page details > "$OUT/details_${TAG}.txt" 2>&1
    "$NCU_BIN" -i "${REP}.ncu-rep" --page raw --csv --metrics "$M" > "$OUT/metrics_${TAG}.csv" 2>&1
    # sanity: which launches (kernel name, grid) were measured
    python - "$OUT/metrics_${TAG}.csv" <<'PY'
import csv, sys
from collections import Counter
rows = list(csv.reader(open(sys.argv[1])))
try:
    h = next(i for i, r in enumerate(rows) if r and r[0] == "ID")
except StopIteration:
    print("[ncu] could not parse", sys.argv[1]); sys.exit(0)
hdr, data = rows[h], rows[h + 2:]          # second header row holds the units
kn, gs = hdr.index("Kernel Name"), hdr.index("Grid Size")
shapes = Counter((r[kn][:70], r[gs]) for r in data)
for (name, grid), n in shapes.most_common():
    print(f"[ncu]   {n} x grid {grid}  {name}")
PY
    return 0
}

IFS='|' read -ra KLIST <<< "$KERNELS"
status=0
for K in "${KLIST[@]}"; do
    profile_one "$K" || status=1
done
ls -lh "$OUT"
exit $status
