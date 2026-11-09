# Profiling the GA optimization (`launchGAOptim.sh`)

This note records how the GA hyperparameter optimization is profiled on Perlmutter GPU nodes, what the
profiles showed, and what was changed because of them. All measurements are from the pilot configuration
(`configs/GAConfig.pilot.json`: 1 generation, 16 models, 10 epochs, 20,000 events per fold and class) for
`MHc130_MA90`, channel `Combined`, on one A100 of a Perlmutter GPU node (`--gpus=1`, 64 CPUs), 2026-11-08.

Tools, in the order they were used: **perf** (CPU, all processes), **py-spy** (tried and dropped),
**nsys** (GPU timeline), **ncu** (single kernels, planned).

---

## 1. Setup

```bash
salloc -A m2612 -C gpu -q interactive -N 1 --gpus=1 -c 64 -t 02:00:00
cd /pscratch/sd/c/choij/workspace/ChargedHiggsAnalysisV3 && source setup.sh && cd ParticleNetMD
git clone https://github.com/brendangregg/FlameGraph     # once; or set FLAMEGRAPH_DIR
```

- Profile on a compute node, never a login node: other users' processes end up in the profile.
- `setup.sh` (NERSC branch) exports `NSYS_BIN` (the full `/opt/nvidia/hpc_sdk/.../Nsight_Systems/bin/nsys`),
  `NSYS_DELAY` and `NSYS_DURATION`.
- Do **not** `module load cudatoolkit` for these runs. It puts `/usr/local/cuda-13.2/compat` on
  `LD_LIBRARY_PATH`, and PyTorch then loads the forward-compatibility driver `libcuda.so.595.91.07` instead of
  the system `libcuda.so.580.178.04`, so the run is no longer comparable with the others.
- Do **not** edit `scripts/launchGAOptim.sh` while a run of it is in progress. bash reads a script from disk as it
  goes; inserted lines shift the byte offsets, and the post-processing after the `wait` loop can break. Python
  files are safe to edit (they are imported once at startup).
- `/pscratch` is purged after ~8 weeks without access. Raw profiles (`*.perf.data`, `*.nsys-rep`) are not in git;
  copy them to CFS if they must survive.

## 2. Launcher options

All options work with every argument format of `launchGAOptim.sh` and are mutually exclusive.
Each run writes its job log to `logs/GA_<signal>_<channel>.<tool>.log` (no suffix without a profiler).

| Option | Command run per job | Outputs |
|--------|--------------------|---------|
| `--perf` | `perf record -F 99 -g` with `PYTHONPERFSUPPORT=1` | `profiling/perf/GA_<sig>_<ch>.{perf.data,folded,svg}` and `.flamechart/` (per-process flame charts, `index.tsv`, `timeline.svg`) |
| `--py-spy` | `py-spy record --subprocesses --native --rate $PYSPY_RATE --format raw` | `profiling/py-spy/GA_<sig>_<ch>.{folded,svg}` |
| `--nsys` | `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --kill=none` | `profiling/nsys/GA_<sig>_<ch>.nsys-rep` and `GA_<sig>_<ch>_<report>.csv` |

Environment variables:

| Variable | Default | Meaning |
|----------|---------|---------|
| `FLAMEGRAPH_DIR` | `./FlameGraph` | Brendan Gregg's FlameGraph scripts |
| `PYSPY_RATE` | 100 | py-spy samples per second and process |
| `NSYS_BIN` | `nsys` in PATH, else `$WORKDIR/../tools/nsight_root/target-linux-x64/nsys` | nsys binary |
| `NSYS_IMPORTER` | next to `NSYS_BIN`, else from the nsys in PATH | `QdstrmImporter`, needed only by a collector-only nsys |
| `NSYS_DELAY`, `NSYS_DURATION` | 0, 0 (whole run) | recording window in seconds after launch; training continues after it |
| `NSYS_EXTRA_ARGS` | empty | appended to `nsys profile` (e.g. `--pytorch=autograd-nvtx`) |

nsys summary reports written after the run: `nvtx_sum`, `nvtx_gpu_proj_sum`, `cuda_gpu_kern_sum`,
`cuda_gpu_mem_time_sum`, `cuda_api_sum`. They are built from an SQLite export (`<stem>.sqlite`, git-ignored)
after `python/nsysFixRanges.py` has removed the NVTX ranges cut by the recording window: ranges still open
when recording stops get an end before their start, which gives negative durations in `nvtx_sum` and makes
`nvtx_gpu_proj_sum` fail (`rtree constraint failed: NVTX_EVENTS_RIDX.(startTS<=endTS)`). In the run of
section 5.8 this removed 9 of 80,392 ranges. To redo the summaries of an existing report:

```bash
nsys export --type sqlite --force-overwrite true -o <stem>.sqlite <stem>.nsys-rep
python python/nsysFixRanges.py <stem>.sqlite
nsys stats --report nvtx_sum,nvtx_gpu_proj_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_api_sum \
    --format csv --force-overwrite=true --output <stem> <stem>.sqlite
```

`--nsys` forces `--start-method spawn` (section 5.7). The copy of nsys in `tools/nsight_root` is the
collector-only package: it writes `.qdstrm` and needs a `QdstrmImporter` from a full install to produce a
`.nsys-rep`; the launcher converts automatically and stops before training if no importer is found.

Flame charts can be rebuilt from a `perf.data` file at any time:

```bash
python python/perfFlameChart.py --perf-data profiling/perf/GA_MHc130_MA90_Combined.perf.data
```

Run it on the node that recorded the data if Python function names matter (they are resolved from
`/tmp/perf-<pid>.map`, which only exists there).

## 3. Instrumentation in the code

### NVTX ranges (for nsys/ncu; near-zero cost without a profiler)

| Where | Ranges |
|-------|--------|
| `launchGAOptim.py` (parent) | `iter N load datasets`, `iter N spawn + train 16 workers` |
| `trainWorker.py` (per worker) | `worker N setup`, `worker N epoch E` → `train`, `validate`, `early_stop_checkpoint`; `worker N final_eval` |
| training batch | `train/collate`, `train/batch` → `train/to_device`, `train/phi_rotation`, `train/forward`, `train/loss`, `train/backward`, `train/optimizer_step`; then `train/epoch_end_sync`, `train/accuracy` |
| validation batch | `valid/to_device`, `valid/forward`, `valid/loss` |
| GPU→CPU syncs | `loss/item_sync` (4 `.item()` per call in `WeightedLoss.py`), `phi_rotation/item_sync` (`TrainingUtilities.py`) |

Each training batch makes **5 GPU→CPU syncs**: 4 in the DisCo loss bookkeeping and 1 in the phi rotation.

### `[PROFILE]` log lines

```bash
grep "\[PROFILE\]" logs/GA_MHc130_MA90_Combined.perf.log
```

They time the parallel dataset loading, balance+shuffle per split, the shared-memory copy, the whole
preparation, each worker's setup and training, the spawn+train phase, and the total GA run.

## 4. Reading the perf outputs

- **Flame graph** (`GA_<sig>_<ch>.svg`): box width = share of on-CPU samples (CPU cycles) summed over all
  processes; left-to-right order is alphabetical, not time. Off-CPU time (sleeping, blocked) is invisible.
- **Per-process flame charts** (`.flamechart/<pid>_<tid>_<comm>.svg`): time-ordered; gaps between samples
  longer than the sampling interval are filled with `[off-CPU]`, so the x-axis is wall-clock time.
  Processes with less than 10 s on-CPU are skipped (`--min-oncpu-sec`); only main threads are drawn.
- **Timeline** (`.flamechart/timeline.svg`): one row per process on a shared wall-clock axis; row height =
  on-CPU share of each bin, stacked by activity (Python interpreter, PyTorch C++ dispatch, waiting on the GPU,
  GPU→CPU copy, garbage collector, other); empty track = off-CPU. Rows are labelled **parent**, **loader N**
  (no GPU work) and **worker N** (kernel launches, stream syncs or GPU copies). Hover a bin for exact numbers.

Limitation: the conda-forge Python 3.12 and the PyTorch wheels are built without frame pointers, so `-g`
unwinding stops after 1-2 native frames and Python function names (`py::`) are almost never reached
(230 of 17,933 stacks in the first run). The activity categories come from the leaf frame only.

## 5. Results

### 5.1 Overview

| Run | Changes | Dataset loading | Worker starts | Total GA run |
|-----|---------|----------------|---------------|--------------|
| 1 | baseline: serial loading, `spawn` | 650 s | staggered over ~12 min | 2,084 s |
| 2 | parallel loading (separate train / valid pools) | 346 s (train 182 s + valid 102 s + shared memory) | staggered over 11 min 20 s | ~1,929 s |
| 3 | + merged train/valid pool, `forkserver` workers and loaders | **269 s** (16 tasks 211 s, shared memory 27 s) | **all 16 within 4 s** | **1,347 s** |

Raw numbers: run 3 `[PROFILE]` lines; runs 1-2 from log timestamps and `index.tsv`. The flame charts in
`profiling/perf/` are from run 3; runs 1 and 2 are in the git history (commits `9b67b62924`, `d60d0cfb01`).

### 5.2 Training workers

- On-CPU about 90-96% of their lifetime: the CPU side is busy with Python and PyTorch dispatch overhead and
  spin-waits on the GPU.
- Share of a worker's CPU time spent waiting on the GPU (stream syncs + GPU→CPU copies): 16-30% in run 1
  (workers starting one by one), 38% on average in run 3 (all 16 competing from the start). Python interpreter:
  30-40%.
- **The epoch `Time:` in the job log is cumulative** (seconds since the worker started training,
  `MLTools.py:167`), not per epoch. Per-epoch times are differences of consecutive values.
- Per-epoch time: ~68 s with the GPU nearly to itself, ~110-116 s with all 16 models running (run 2: worker 0
  went 68, 68, 69, 70, 72, 77, 86, 93, 107, 113 s as the others started; run 3: 115-118 s from the first
  epoch, dropping to ~84 s as early-stopped workers left). So 16 concurrent models give about 16/115 ≈ 0.14
  epochs/s against 1/68 ≈ 0.015 for one: **~9.5× the throughput of one model alone**. An earlier reading of
  the log as per-epoch time had concluded ~1.3×; that was wrong.

### 5.3 Dataset loading

- The `.pt` files are pickled lists of PyG `Data` objects; `torch.load` unpickles them one object at a time on
  one core: ~3,400 events/s, 5,180 events/s with the garbage collector off (one 75k-event file: 22.3 s → 14.5 s).
- A pilot load reads 112 files (7.8 GB, ~2.0 M events) and keeps 320 k (4 classes × 4 folds × 20,000).
- Run 1: the parent was ~99% on-CPU from 100 s to 650 s: CPU-bound, single-threaded, GC-heavy.
- Now: one task per (class, fold), 16 tasks in one process pool (`DynamicDatasetLoader.load_multiclass_splits`),
  results identical to the original serial loader (checked event by event: 80,000 + 80,000 events, all
  tensor fields, era and sample info).
- Run 3 timeline: each loader is ~95% on-CPU for 25-53 s, so the loaders' own work is done by ~130 s. The
  pool returns at ~290 s: the remaining ~150 s is the **parent unpickling the 320 k returned events, one result
  after another**. The loaders also start only at 78 s: they wait for the forkserver to finish its one-time
  import.

### 5.4 Worker start (the 11-minute diagonal)

- `torch.multiprocessing.spawn` starts workers in a loop, and with the `spawn` start method `start()` returns only
  when the child has read its arguments from a pipe. The child first imports the main module
  (`multiprocessing/spawn.py:_main`), and the arguments (~2.4 MB) exceed the pipe buffer, so the parent's write
  blocks until that import is done. Each worker's import time is paid back to back.
- Reproduced on the login node: with a GA-sized payload, 4 `spawn` workers started at +6.1, +12.2, +18.5,
  +24.7 s for a 6.0 s import; with a small payload they all started together.
- The import (torch, PyG, sklearn, ROOT via `evaluateGAModels` → `OverfittingDetector`) takes ~7 s on a warm
  login node and ~40-45 s on a GPU node, mostly off-CPU (run 1: 5-10 s on-CPU, 35-40 s off-CPU in each worker's
  first 45 s). The conda env lives in `$HOME`, which is slow at reading many small files on compute nodes
  (not measured directly on a GPU node).
- The shared dataset itself is cheap to send: the validation `Batch` pickles to 0.6 MB in ~0 s (tensors travel
  as shared-memory file descriptors).
- Fix: `--start-method forkserver` (default). `python/lib/forkserverPreload.py` imports everything once in the
  forkserver; workers and loaders are forked from it. Run 3: 16 workers within 4 s.

### 5.5 Pitfalls found on the way

- **`torch.__config__.show()` initializes the CUDA driver.** `torch_geometric/typing.py:24` calls it at import
  to look for `USE_MKL=OFF`. A process forked from one that did this cannot use CUDA (`CUDA error:
  initialization error`). `forkserverPreload.py` swaps `show()` for a CPU-only answer to the same question
  (`torch.backends.mkl.is_available()`) during the PyG import. `PYTORCH_NVML_BASED_CUDA_CHECK=1` does not help
  (it only changes `torch.cuda.is_available()`).
- **`set_forkserver_preload(["__main__"])` is silently ignored in Python 3.12**: the forkserver asks for
  `main_path`, but the preparation data only has `init_main_from_path`. Modules are preloaded by name instead.
- **Multiprocessing's default tensor transfer exhausts mmap**: returning ~10^5 small tensors per task gives each
  its own shared-memory segment (`unable to mmap ... Cannot allocate memory`). Loader results are returned as
  plain pickle bytes.
- **Threads do not parallelize unpickling** (it holds the GIL); `plotTrainTestLR.py` still uses a thread pool.

### 5.6 py-spy (dropped)

`py-spy record --subprocesses --native` on 17 processes fell more than 60 s behind at 50 samples/s, slowed the
single-threaded loading 2.5× (27 min 21 s vs 10 min 50 s), and the run was killed without output. perf gave the
per-process picture at much lower cost. For Python-level attribution of one process, a short
`py-spy record -p <pid> --duration 60` is still an option.

### 5.7 nsys and forked processes

nsys does not trace CUDA in processes created by `fork` without `exec`: a forkserver test under nsys recorded
no kernels (8 CUDA API calls with `--trace-fork-before-exec=true`, none without). `--nsys` therefore runs the
workers with `spawn`, and the ~42 s stagger is back in nsys runs only. With `NSYS_DELAY=1000`,
`NSYS_DURATION=120`, the window falls where all 16 workers train at the same time.

### 5.8 nsys results

Run: `--pilot --nsys`, `NSYS_DELAY=1000`, `NSYS_DURATION=120` (workers started with `spawn`, ~42 s apart).
Total GA run 1,684 s (dataset loading 262 s, spawn + train 1,422 s). In the 120 s window all 16 workers train.
Files: `profiling/nsys/GA_MHc130_MA90_Combined_<report>.csv`; the `.nsys-rep` (353 MB) and `.sqlite` (893 MB)
are git-ignored.

**The GPU is about half busy, with mostly tiny kernels.**

- Kernel time 64.4 s + memory copies/sets 1.3 s in 120 s: ~55% busy. Kernels of different processes time-slice
  (no MPS), so the sum is a fair busy-time estimate; "busy" does not mean the SMs are full.
- 5,780,771 kernel launches of 109 kernel types (~600 per batch, ~48,000 per second); 83% average under 10 µs,
  97% under 50 µs.

| Kernel | GPU time | Launches | Median |
|--------|---------|----------|--------|
| `cutlass_80_simt_sgemm_32x128_8x5_nt_align1` | 17.1% | 18,717 | 366 µs |
| `knn_kernel<float>` (torch_cluster, DynamicEdgeConv) | 10.9% | 19,372 | 375 µs |
| `ampere_sgemm_32x128_tn` | 8.4% | 81,398 | 58 µs |
| `reduce_kernel<128, 4, ...>` | 4.8% | 145,506 | 14.6 µs |
| `ampere_sgemm_32x32_sliced1x4_nt` | 4.7% | 63,839 | 24.3 µs |
| other `ampere_sgemm_*` (4 shapes) | 10.3% | | 25-100 µs |
| `reduce_kernel<512, 1, ...>` | 2.8% | 253,641 | 7.0 µs |
| `batch_norm_*_channels_last` (3 kernels) | 5.3% | ~350,000 | 6-11 µs |

**A training step, by NVTX range** (median CPU-side duration; includes waiting for the GPU under 16-way sharing):

| Range | ms |
|-------|----|
| `train/collate` (PyG batch assembly from the shared dataset, in the training process: `num_workers=0`) | **83.9** |
| `train/forward` | 37.7 |
| `train/loss` (DisCo) | 26.5 |
| `train/to_device` | 14.2 |
| `train/backward` | 9.4 |
| `train/phi_rotation` | 2.4 |
| `train/optimizer_step` | 1.4 |

About 177 ms per step, of which ~47% is batch assembly on the CPU. GPU operations per range
(`nvtx_gpu_proj_sum`): ~474 per training batch, of which ~283 in `forward` and ~150 in the DisCo loss.
`train/backward` shows no GPU work in its range: autograd launches the backward kernels from its own thread
(`pt_autograd_0`), outside the main thread's NVTX range. For `forward`, `loss` and `to_device` the projected
GPU span equals the CPU-side duration (syncs inside each range; under time-slicing the span also contains other
processes' turns), so it is not net GPU busy time.

**35% of worker time is blocked in syncs.** `cudaStreamSynchronize` (430,457 calls, 342 s) and `cudaMemcpyAsync`
(492,886 calls, 328 s; 333,374 of the copies are device-to-host) take 670 s of the 1,920 process-seconds in the
window, matching perf's ~38% "waiting on GPU". That is ~44 syncs per batch. Attributing each call to its
innermost NVTX range (from the SQLite export):

| Innermost range | Syncs per training batch | Share of blocked time |
|-----------------|-------------------------|-----------------------|
| `train/forward` | ~17 copies + ~17 syncs | 31% |
| `train/loss` (besides `loss/item_sync`) | ~12 + ~12 | 23% |
| `train/to_device` | ~10 | 12% |
| `loss/item_sync` + `phi_rotation/item_sync` (instrumented) | 5 | 5% |
| validation ranges | same pattern | ~26% |

### 5.9 The synchronization points, by source line

nsys `--python-backtrace=cuda` (with `--cudabacktrace=sync:0,memory:0` and CPU sampling) records native C++
stacks for every CUDA call, but in the SQLite export no table links a CUDA call to the Python frames: the Python
function names are in `StringIds` and referenced nowhere. The source lines therefore come from PyTorch itself:
`python/findSyncSites.py` runs the real `train_epoch_disco` / `evaluate_disco` on a small slice (one warm-up
epoch, then one epoch) with `torch.cuda.set_sync_debug_mode("warn")` and groups the warnings by stack.

```bash
python python/findSyncSites.py --signal MHc130_MA90 --channel Combined --pilot   # GPU, ~2-3 min
```

Result (`profiling/syncs/sync_sites_MHc130_MA90_Combined.csv`): **42.2 syncs per training batch, 36.2 per
validation batch**, i.e. ~96% of the ~44 that nsys counts (syncs made by C++/CUDA extensions themselves are
invisible to PyTorch's check). Per training batch:

| Syncs | Our code | Where it happens | Cause |
|-------|----------|------------------|-------|
| 10 | `trainWorker.py:98` `batch.to(device)` | `torch_geometric/data/data.py:364` | one blocking copy per batch tensor (`non_blocking=False`) |
| 8 | `MultiClassModels.py:70` `knn_graph` (2 dynamic EdgeConv layers × 4) | `torch_cluster/knn.py:66`, `:69` | `int(batch_x.max())`, `int(batch_y.max())` to find the batch size |
| | | `torch_cluster/knn.py:142` | `row[mask], col[mask]` (removing self-loops) |
| 3 | `MultiClassModels.py:71` `dropout_edge` (training only) | `torch_geometric/utils/dropout.py:212` | `edge_index[:, edge_mask]` |
| 1 | `MultiClassModels.py:143` `gn0` | `torch_geometric/nn/norm/graph_norm.py:70` | `int(batch.max())` |
| 1 | `MultiClassModels.py:152` `global_mean_pool` | `torch_geometric/utils/_scatter.py:48` | `int(index.max())` |
| 2 | `WeightedLoss.py:185`, `:196` | | `if validN.sum() > 1:` |
| 6 | `WeightedLoss.py:187-189`, `:198-200` | | `scores[valid]`, `mass[valid]`, `weight[valid]` |
| 4 | `WeightedLoss.py:121` (2 calls) | | `if dvar_X_sq > 1e-10 and dvar_Y_sq > 1e-10` |
| 4 | `WeightedLoss.py:216-219` | | `.item()` for loss logging |
| 2 | `trainWorker.py:121` `loss.backward()` | | |
| 1 | `TrainingUtilities.py:40` | | phi-rotation `.item()` |

The DisCo loss accounts for 16 (38%), only 4 of them the logging `.item()` calls; the batch copy for 10; PyG and
torch_cluster for 13, mostly batch sizes read back from the GPU although they are known on the CPU. Standard fixes
would be pinned `non_blocking` copies, passing `batch_size` / `dim_size` explicitly, and multiplying by masks
instead of indexing with them; none is applied (this study does not optimize further).

### 5.10 Reproducibility between runs

The GA population is not seeded (`GATools.randomGeneration`), so each run trains a different hyperparameter mix
(e.g. 5 vs 6 workers with `nNodes=256` in the perf and nsys runs). Kernel shares and per-range times therefore
move by a few percent between runs, and the window catches different epochs. The sync sites and counts depend on
the code path only. For before/after comparisons, seed the GA or pin the hyperparameters.

## 6. Nsight Compute (ncu)

ncu replays each kernel many times and serializes launches, so it profiles a small single-process driver, not the
GA (lessons from `cms-madgraph4gpu-integration/.../docs/REPORT.md`, section 9, and its `ncu_gridpack.sh`).

```bash
# on a GPU node you have to yourself (DCGM_PAUSE pauses DCGM profiling node-wide, resumed on exit)
DCGM_PAUSE=1 ./scripts/ncuKernels.sh MHc130_MA90 Combined --pilot
KERNELS='knn_kernel' COUNT=4 ./scripts/ncuKernels.sh MHc130_MA90 Combined --pilot --nNodes 256
```

- **Driver** `python/profileKernels.py`: trains one model the way a GA worker does (the real
  `train_epoch_disco` / `evaluate_disco`, GA batch size) on a cached slice (2,048 events per class and fold:
  16 train + 16 valid batches; `profiling/cache/`, git-ignored, built once outside ncu with `--prepare-only`).
  Phases are NVTX-marked `ncu_warmup` (1 epoch: allocator, cuBLAS heuristics, first kNN) and `ncu_measure`
  (`--measure-epochs`, `--phase train|valid|both`); the measured phase is also bracketed by
  `cudaProfilerStart/Stop` (`torch.cuda.profiler.start/stop`). The setup is shared with `findSyncSites.py`
  (`python/lib/ProfileSetup.py`); hyperparameters via `--nNodes --optimizer --scheduler --initLR --weight-decay`.
- **Launcher** `scripts/ncuKernels.sh <signal> <channel> [driver options]`: one ncu run per kernel regex
  (`KERNELS`, default `cutlass_80_simt_sgemm|knn_kernel|ampere_sgemm_32x128_tn|reduce_kernel|batch_norm_collect_statistics`,
  matched against the full demangled name), `--profile-from-start off`, `COUNT=8` launches,
  `SET=full`, an explicit `--metrics` list (launch shape, occupancy limits, waves per SM, SM / memory / DRAM
  throughput, FMA pipe, cache hit rates, warp-stall reasons), `CLOCK=base|none`. Output in
  `profiling/ncu/<signal>_<channel>/`: `ncu_<k>.ncu-rep` (git-ignored), `details_<k>.txt`, `metrics_<k>.csv`,
  `run_<k>.log`, plus a printed list of the (kernel, grid) shapes measured. Refuses login nodes (`FORCE=1`).
- **Do not select the measured phase with an NVTX filter.** NVTX ranges are per thread, and autograd launches
  every backward kernel from its own thread (`pt_autograd_0`): in the nsys GA run, 100% of
  `cutlass_80_simt_sgemm` and `batch_norm_backward_*` launches and 46% of `reduce_kernel` launches came from
  that thread. The first ncu runs used `--nvtx --nvtx-include "ncu_measure/"` and therefore never saw the
  cutlass GEMM and measured only forward `reduce_kernel` launches. `--profile-from-start off` with
  `cudaProfilerStart/Stop` covers every thread of the process. (Also: ncu reads `/` in an NVTX filter as range
  nesting, so names like `train/forward` cannot be used as filters at all.)
- **Counters on Perlmutter**: `RmProfilingAdminOnly` is 0, but the DCGM host engine (`nv-hostengine`) holds
  them ("a driver resource was unavailable"; seen on the login node). On a GPU node, `DCGM_PAUSE=1` runs
  `dcgmi profile --pause` / `--resume`; this affects every job on the node.
- ncu locks clocks to base by default, so absolute times differ from nsys (boost clocks, 16-way sharing): compare
  ratios, or use `CLOCK=none`. Do not run ncu on the GPU of a running nsys/perf job; a separate node is fine.
- cuBLAS picks GEMM kernels per matrix shape **and GPU model**: on the compute nodes, GA workers of every width
  (96-256) launch `cutlass_80_simt_sgemm`, while on the login node's A100 (80 GB PCIe) the driver uses it only
  at `nNodes=256`. Check kernel names with nsys on the same kind of node. The default output folder is tagged
  with the width (`profiling/ncu/<signal>_<channel>_n<nNodes>/`).

### 6.1 Results (`nNodes=128`, base clocks, 8 launches per kernel)

| Kernel | Duration | Grid x block | Waves/SM | Achieved occupancy | SM throughput | Top stall |
|--------|---------|--------------|----------|--------------------|---------------|-----------|
| `knn_kernel` | 306 µs | 14 x 256 | 0.03 | 11.8% | 0.8% | `long_scoreboard` 18.6 |
| `ampere_sgemm_32x128_tn` | 59 µs (17-74) | 112-264 x 256 | 0.26-0.61 | 25.5% | 26.8% (FMA pipe 34%) | `long_scoreboard` 8.7 |
| `batch_norm_collect_statistics_channels_last` | 14.6 µs | 56-132 x 512 | 0.26-0.61 | 30.6% | 20.1% (DRAM 14%) | `long_scoreboard` 3.0 |
| `reduce_kernel` (`max<long>`, `sum<long>`) | 6.9 µs | 1 x 512 | 0.00 | 23.9% | 0.1% | `imc_miss` 8.6 |
| `cutlass_80_simt_sgemm` (`128x32_nt`, `32x128_nt`, `128x32_nn`; backward) | 13.8 µs | 104-216 x 128 | 0.36 | 9.3% | 49% (FMA pipe 59%) | none significant |
| `batch_norm_backward_{reduce,elemt}_channels_last` | 10.2 µs | 4-56 x 512 | 0.03 | 24.1% | 1.5% | `long_scoreboard` 15.6 |

Medians; stalls in warps per issued instruction. Files: `profiling/ncu/MHc130_MA90_Combined_n128/`.
`knn_kernel`, `ampere_sgemm_32x128_tn` and `batch_norm_collect_statistics` were measured with the NVTX filter;
they run 100% on the main thread, so those results are complete. The last three rows use
`--profile-from-start off`.

- `COUNT=8` takes the **first** 8 matching launches of the measured epoch, i.e. the start of the first batch:
  all 8 `reduce_kernel` launches are forward-pass `max` / `sum` reductions, although 46% of `reduce_kernel`
  launches in the GA are backward. Use a larger `COUNT` to reach later launches.
- **The 17% "cutlass" share of GA GPU time is one GEMM shape at `nNodes=256`.** Per variant and width in the nsys
  run: `32x128_8x5_nt` at width 256: 17,766 launches, median 367 µs, p90 1.94 ms, grid (16, 1, 54); every
  variant at widths 96-192: 8.7-12 µs per launch. Measure it with
  `KERNELS='cutlass_80_simt_sgemm_32x128' COUNT=16 ./scripts/ncuKernels.sh MHc130_MA90 Combined --pilot --nNodes 256`.
- At width 128, cutlass is the best-used kernel measured (49% of SM capacity, FMA pipe 59%, no dominant stall),
  but each launch lasts ~14 µs; `batch_norm_backward` is as small as its forward counterpart.

### 6.2 The heavy cutlass GEMM at `nNodes=256`

`KERNELS='cutlass_80_simt_sgemm_32x128' COUNT=16 ./scripts/ncuKernels.sh MHc130_MA90 Combined --pilot --nNodes 256`
(files: `profiling/ncu/MHc130_MA90_Combined_n256/`). 14 of the 16 launches are the heavy shape, 2 are small
launches of the same variant (17 µs, grid (2, 1, 105)). Medians over all 16:

| Metric | `cutlass_80_simt_sgemm_32x128_8x5_nt_align1`, width 256 |
|--------|-------------------------------------------------------|
| Duration | 411 µs (nsys in the GA: 367 µs median at boost clocks; p90 1.94 ms from time-slicing) |
| Grid x block | 848-864 x 128 (grid z = 53-54): 2.6 waves per SM |
| Registers per thread | 152: at most 3 blocks per SM (register and shared-memory limits both 3) |
| Achieved occupancy | 17% (~11 of 64 warps per SM) |
| SM throughput / FMA pipe | 16% / 18% |
| DRAM throughput | 2% (17 MB read per launch, L2 hit rate 74%) |
| Top stall | `barrier` 7.2 warps per issue, then `long_scoreboard` 2.1 |

- The only measured kernel with more than one wave, and the most expensive kernel of the GA (17% of GPU time,
  all from the width-256 workers), yet it uses ~16-18% of the SMs' compute: 152 registers per thread allow
  only 12 warps per SM, and they mostly wait at block-wide barriers. DRAM bandwidth is not the limit.
- Grid z = 53-54 suggests split-K over the long reduction dimension of this backward weight-gradient GEMM
  (presumably the number of edges), consistent with the barrier stalls; the matrix shapes were not checked.
- It is an FP32 SIMT kernel: the Tensor Cores are idle. Allowing TF32
  (`torch.backends.cuda.matmul.allow_tf32 = True`) is the one setting that could plausibly change this kernel
  measurably; not tested, and it changes numerical precision.

- **No kernel fills the A100**: every launch is below one wave (0.00-0.61 waves per SM on 108 SMs).
- **`knn_kernel`** (torch_cluster, ~11% of GPU time in the GA) launches 14 blocks of 256 threads, one thread per
  node of the batch (~3,600 nodes for 512 graphs), each looping over its graph's nodes: 14 of 108 SMs busy,
  0.8% SM throughput, stalled on memory loads despite a 98% L1 hit rate. ncu measures 306 µs at base clocks
  (nsys: 375 µs median under 16-way sharing).
- **`reduce_kernel`** launches are 1-block `max` / `sum` reductions over `long`: the GPU side of the hidden
  syncs of section 5.9 (`int(batch.max())` in torch_cluster / PyG, `valid.sum() > 1` in the DisCo loss).
- The GEMMs are the only kernels doing substantial work (~25-35% of SM capacity), still under one wave.
- **Why 16 concurrent models give ~9.5x the throughput of one**: without MPS, kernels of different processes
  time-slice and never run at the same time, so the gain fills **temporal** gaps (each worker is mostly busy on
  the CPU: batch assembly, launches, syncs). The **spatial** gaps remain: a running kernel uses a small fraction
  of the SMs, which the ~55% "busy" of section 5.8 does not show. MPS (kernels of different processes running
  side by side) would target those; not tested.
