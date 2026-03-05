# Step 2: Mass Decorrelation — Lambda Sweep Experiment

How to train a series of mass-decorrelated models with varying DisCo strength, and produce comparison plots showing the decorrelation–performance tradeoff.

---

## 0. Prerequisites

Datasets from Step 1 must exist for both `Run1E2Mu` and `Run3Mu` (the `Combined` channel merges them on-the-fly):

```bash
ls dataset/samples/signals/TTToHcToWAToMuMu-MHc130_MA90/Run1E2Mu_fold-0.pt   # must exist
ls dataset/samples/signals/TTToHcToWAToMuMu-MHc100_MA95/Run1E2Mu_fold-0.pt   # must exist
ls dataset/samples/signals/TTToHcToWAToMuMu-MHc160_MA85/Run1E2Mu_fold-0.pt   # must exist
```

If missing, run `./scripts/saveDatasets.sh` first (see [STEP1_PREPROCESSING.md](STEP1_PREPROCESSING.md)).

---

## 1. Background: DisCo Loss

ParticleNetMD trains with a combined loss:

```
L_total = L_CE + λ · (DisCo(score_signal, mass1) + DisCo(score_signal, mass2))
```

- **L_CE**: weighted cross-entropy (standard classification loss)
- **DisCo(X, Y)**: event-weighted distance correlation — 0 when X and Y are independent, 1 when fully dependent
- **mass1, mass2**: OS muon pair invariant masses extracted during dataset creation
- **λ = 0**: plain cross-entropy, no decorrelation (equivalent to the ParticleNet baseline)
- **λ > 0**: penalises the classifier for correlating its output with the muon pair mass

The motivation is to prevent artificial mass bumps in background predictions when a score cut is applied. Without decorrelation, a high-scoring event region may preferentially select a particular mass window, biasing the background shape.

**Implementation:** [`WeightedLoss.py:DiScoWeightedLoss`](../python/WeightedLoss.py), [`WeightedLoss.py:distance_correlation()`](../python/WeightedLoss.py)

**Config default:** `disco_lambda = 0.1` in [`configs/GAConfig.json`](../configs/GAConfig.json) (used during GA optimisation). The per-lambda sweep training uses `--disco-lambda` as a CLI override; `SglConfig.json` is not consulted for the sweep.

---

## 2. Lambda Sweep

### Experiment parameters

| Parameter | Values |
|-----------|--------|
| Channel | `Combined` (Run1E2Mu + Run3Mu merged) |
| Signals | `MHc130_MA90`, `MHc100_MA95`, `MHc160_MA85` |
| λ values | 0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5 |
| Total runs | 24 (3 signals × 8 λ) |

λ = 0.0 is the no-decorrelation baseline. λ = 0.05 is the current production default.

### Model naming

The disco_lambda value is encoded in the model name as `discoL{value}`:

| λ | Loss label in model name |
|---|--------------------------|
| 0.0   | `discoL0p0`   |
| 0.005 | `discoL0p005` |
| 0.01  | `discoL0p01`  |
| 0.02  | `discoL0p02`  |
| 0.05  | `discoL0p05`  |
| 0.1   | `discoL0p1`   |
| 0.2   | `discoL0p2`   |
| 0.5   | `discoL0p5`   |

Example full model name:
```
ParticleNet-nNodes256-Adam-initLR0p0005-decay0p00004-ExponentialLR-discoL0p05-3grp-nonprompt-diboson-ttX
```

### Shared-dataset launcher

The sweep uses `launchLambdaSweep.py` rather than independent `trainMultiClass.py` calls.
For each signal it loads the dataset **once** and shares it across all 8 λ workers via PyTorch shared memory (`mp.spawn`).

Memory comparison (8 λ values):

| Approach | Peak RAM per signal |
|----------|---------------------|
| Old (8 independent processes) | 8 × 3 GB = **24 GB** |
| New (shared dataset + 8 workers) | 3 GB shared + 8 × 0.5 GB ≈ **7 GB** |

### Early stopping

Training uses early stopping on **validation loss** with `patience=10` (from `SglConfig.json`). If validation loss does not improve for 10 consecutive epochs, training stops and the best weights are restored. In pilot mode the small dataset often triggers early stopping within 10–15 epochs; full training typically runs 50–120 epochs.

---

## 3. Running the Sweep

### 3a. Pilot run (validate pipeline first)

```bash
bash scripts/runLambdaSweep.sh --pilot
```

Runs all 24 combinations with pilot mode (small dataset caps, fast training):
- Training log per signal: `logs/sweep_Combined_{signal}_sweep.log`
- Results: `LambdaSweep/Combined/{signal}/pilot/trees/` (8 ROOT files per signal)
- Expected runtime: ~20 minutes total on GPU

Check for failures:
```bash
# Expect 8 "Training complete" lines per signal (24 total)
grep -c "Training complete" logs/sweep_Combined_*_sweep.log
```

### 3b. Full training

```bash
bash scripts/runLambdaSweep.sh
```

- Same 24 combinations, full dataset (up to 50k events/fold/class), up to 120 epochs
- Training log per signal: `logs/sweep_Combined_{signal}_sweep.log`
- Results: `LambdaSweep/Combined/{signal}/fold-4/trees/` (8 ROOT files per signal)
- `compareDecorrelation.py` prefers `fold-4/` over `pilot/` when both exist

---

## 4. Output Structure

After the sweep completes:

```
LambdaSweep/Combined/
├── MHc130_MA90/
│   └── fold-4/
│       ├── models/
│       │   ├── ParticleNet-...-discoL0p0-3grp-....pt
│       │   ├── ...
│       │   └── ParticleNet-...-discoL0p5-3grp-....pt
│       ├── CSV/    (per-epoch loss/accuracy CSVs)
│       ├── trees/
│       │   ├── ParticleNet-...-discoL0p0-3grp-....root   ← input to compareDecorrelation.py
│       │   ├── ...
│       │   └── ParticleNet-...-discoL0p5-3grp-....root
│       ├── plots/discoL{lam}/    ← per-model visualizations
│       └── comparison/           ← cross-lambda comparison plots
├── MHc100_MA95/
│   └── fold-4/    (same structure)
└── MHc160_MA85/
    └── fold-4/    (same structure)
```

Each ROOT file contains a `TTree("Events")` with branches:
`score_signal`, `score_nonprompt`, `score_diboson`, `score_ttX`, `true_label`, `weight`, `mass1`, `mass2`, `train_mask`, `valid_mask`, `test_mask`

---

## 5. Per-model Visualization

To run `visualizeMultiClass.py` manually for a single model:

```bash
# Full results — MUST activate conda first (python = 2.7 in a raw shell)
source ~/miniconda3/bin/activate && conda activate Nano

python python/visualizeMultiClass.py --signal MHc130_MA90 --channel Combined \
    --model-name discoL0p1 --fold 4

# Pilot results
python python/visualizeMultiClass.py --signal MHc130_MA90 --channel Combined \
    --model-name discoL0p1 --pilot
```

The `--model-name` flag (substring filter) selects the correct model when multiple λ results share the same output directory. Always pass `--fold 4` for full results to avoid the auto-detect defaulting to pilot results.

To re-run all 24 combinations after a full sweep:

```bash
source ~/miniconda3/bin/activate && conda activate Nano
for sig in MHc130_MA90 MHc100_MA95 MHc160_MA85; do
  for lam in 0.0 0.005 0.01 0.02 0.05 0.1 0.2 0.5; do
    lam_str="${lam//./p}"
    python -u python/visualizeMultiClass.py \
      --signal "${sig}" --channel Combined \
      --model-name "discoL${lam_str}" --fold 4 \
      > "logs/viz_full_Combined_${sig}_lambda${lam}.log" 2>&1 &
  done
done
wait
```

Outputs go to `LambdaSweep/Combined/{signal}/fold-4/plots/discoL{lam}/` (11 PNGs each).

---

## 6. Comparison Plots

```bash
# Full results
for sig in MHc130_MA90 MHc100_MA95 MHc160_MA85; do
    python python/compareDecorrelation.py --signal ${sig} --channel Combined
done

# Pilot results
for sig in MHc130_MA90 MHc100_MA95 MHc160_MA85; do
    python python/compareDecorrelation.py --signal ${sig} --channel Combined --pilot
done
```

**Script:** [`python/compareDecorrelation.py`](../python/compareDecorrelation.py)

Discovers all `discoL*.root` files, parses λ from the model name, computes test-set metrics, and produces 4 plots per signal. With `--pilot`, only `pilot/trees/` is searched and outputs go to `LambdaSweep/Combined/{signal}/pilot/comparison/`. Without `--pilot`, `fold-4/` is preferred with `pilot/` as fallback.

### Output

```
LambdaSweep/Combined/{signal}/fold-4/comparison/
├── performance_vs_lambda.png    ← AUC vs λ
├── disco_vs_lambda.png          ← post-hoc DisCo(score, m1/m2) vs λ
├── tradeoff.png                 ← AUC vs DisCo scatter
├── mass_sculpting.png           ← background m1 shape at 3 score cuts, one subplot per λ
└── summary.json
```

### Plot descriptions

**`performance_vs_lambda.png`**
Signal-vs-rest AUC on the test set as a function of λ. Expect AUC to decrease as λ increases (stronger decorrelation trades off discriminating power).

**`disco_vs_lambda.png`**
Post-hoc DisCo(score_signal, mass1) and DisCo(score_signal, mass2) measured on the test set. Shows how effectively the network has been decorrelated. Expect both to decrease as λ increases.

**`tradeoff.png`**
AUC (y-axis) vs DisCo(score, mass1) (x-axis), one point per λ, annotated. The ideal operating point is in the upper-left (high AUC, low DisCo). Use this to select the production λ value.

**`mass_sculpting.png`**
For each λ (one subplot), background mass1 distribution at three signal-score cuts (>0.3, >0.5, >0.7) overlaid with the no-cut baseline, normalised to unit area. A well-decorrelated model produces flat ratios: the cut distributions should match the no-cut shape.

### `summary.json` schema

```json
[
  {"lambda": 0.0,   "auc": 0.912, "disco_mass1": 0.183, "disco_mass2": 0.147},
  {"lambda": 0.005, "auc": 0.910, "disco_mass1": 0.165, "disco_mass2": 0.132},
  {"lambda": 0.01,  "auc": 0.908, "disco_mass1": 0.142, "disco_mass2": 0.118},
  ...
]
```

---

## 7. Selecting the Production λ

Look at `tradeoff.png` for each signal. The recommended approach:

1. Identify the **knee** of the tradeoff curve — the point where DisCo drops significantly with only a small AUC loss.
2. Cross-check with `mass_sculpting.png` — verify that score cuts do not shift the mass1 peak.
3. A λ value that gives DisCo < 0.05 is generally considered well-decorrelated.

### Findings from full sweep (2026-02-26, Combined channel)

| Signal | AUC(λ=0) | AUC(λ=0.1) | AUC(λ=0.5) | DisCo(m2) at λ=0.1 |
|--------|----------|------------|------------|---------------------|
| MHc130_MA90 | ~flat | ~flat | drops | improves vs λ=0 |
| MHc100_MA95 | ~flat | ~flat | drops | improves vs λ=0 |
| MHc160_MA85 | ~flat | ~flat | drops | improves vs λ=0 |

AUC is flat from λ=0 to λ=0.2; decorrelation begins to degrade discrimination only at λ=0.5.
Mass sculpting plots show no visible shape difference between λ values in the di-muon mass distribution.

**Production choice: `disco_lambda = 0.1`** — already set in `configs/GAConfig.json`.

Update `GAConfig.json` if the production λ changes:
```json
"disco_parameters": {
    "disco_lambda": 0.1
}
```

---

## 8. Troubleshooting

**`No discoL*.root files found`**
The sweep has not finished or the wrong mode was used. For pilot results add `--pilot`; for full results check that `fold-4/trees/` exists.

**`CUDA out of memory` in sweep logs**
3 signals run simultaneously, each spawning 8 λ workers that share one GPU. Reduce `batch_size` in `SglConfig.json` (e.g., 128) before re-running.

**One λ worker failed, others succeeded**
Re-run only the failed combination directly:
```bash
python python/trainMultiClass.py --signal MHc130_MA90 --channel Combined --disco-lambda 0.5
```
`compareDecorrelation.py` will pick up however many ROOT files are present.

**Early stopping triggers immediately (epoch 10–15)**
Expected in pilot mode — tiny datasets (8k events) give a noisy loss landscape. In full training, jobs typically run 50–120 epochs before convergence.

**DisCo does not decrease with λ**
This can happen if training converged to a solution that ignores the DisCo term entirely (very low λ relative to CE loss). Try increasing λ or reducing `max_epochs` to prevent overfitting to the CE term before the DisCo term can take effect.

**`compareDecorrelation.py` killed (exit 137 / OOM) with empty log**
Root cause: `distance_correlation()` in `WeightedLoss.py` allocates an n×n pairwise matrix. With ~1M test events this is ~4 TB. Fixed in the script by subsampling to 5000 events before calling DisCo. If you see an empty log, always run with `python -u` for unbuffered output so partial output is visible.

**`compareDecorrelation.py` produces no new files despite exit 0**
When the command is piped (`... | tee ...`), `$?` captures the pipe's exit code (always 0), not Python's. Capture the exit code before piping, or check the log file for `Traceback`.

**`SyntaxError: invalid syntax` on f-strings when running viz interactively**
The system `python` on Perlmutter is Python 2.7. Scripts require Python 3.12 from the conda environment. Always activate conda before running interactively:
```bash
source ~/miniconda3/bin/activate && conda activate Nano
```
SLURM jobs are unaffected because `source setup.sh` activates conda automatically.

**Per-lambda viz plots went to `pilot/` instead of `fold-4/`**
This happens when `visualizeMultiClass.py` is called without `--fold 4` and the fold-4 results did not yet exist at viz time (viz ran before training finished). Re-run with `--fold 4` explicitly. The `runLambdaSweep.sh` viz step runs immediately after training `wait`, but if fold-4 was produced by a separate later run the viz must be triggered manually.

---

## Quick Reference

```bash
cd /path/to/ChargedHiggsAnalysisV3
source setup.sh
cd ParticleNetMD

# Validate pipeline (fast)
bash scripts/runLambdaSweep.sh --pilot
grep -c "Training complete" logs/sweep_Combined_*_sweep.log   # expect 8 per signal

# Full sweep
bash scripts/runLambdaSweep.sh

# Verify outputs
ls LambdaSweep/Combined/MHc130_MA90/fold-4/trees/*.root | wc -l  # expect 8

# Comparison plots (full)
for sig in MHc130_MA90 MHc100_MA95 MHc160_MA85; do
    python python/compareDecorrelation.py --signal ${sig} --channel Combined
done

# Comparison plots (pilot)
for sig in MHc130_MA90 MHc100_MA95 MHc160_MA85; do
    python python/compareDecorrelation.py --signal ${sig} --channel Combined --pilot
done

ls LambdaSweep/Combined/MHc130_MA90/fold-4/comparison/   # expect 4 PNGs + summary.json
```

---

## References

- [DisCo loss implementation](../python/WeightedLoss.py) — `distance_correlation()`, `DiScoWeightedLoss`
- [Lambda sweep launcher](../python/launchLambdaSweep.py) — shared-dataset launcher (one per signal)
- [Lambda sweep worker](../python/lambdaSweepWorker.py) — per-λ training worker
- [Lambda sweep script](../scripts/runLambdaSweep.sh) — outer shell driver (3 signals in parallel)
- [Shared worker utilities](../python/SharedWorkerUtils.py) — `setup_spawn_method()`, `make_dataloader_from_batch()`
- [Comparison script](../python/compareDecorrelation.py) — metrics computation and plots (`--pilot` supported)
- [Per-model viz script](../python/visualizeMultiClass.py) — `--model-name` / `--pilot` CLI flags
- [Training script](../python/trainMultiClass.py) — `--disco-lambda` / `--pilot` CLI flags (single-run fallback)
- [Training configuration](../configs/SglConfig.json) — `disco_parameters.disco_lambda`, `early_stopping_patience`
- [Dataset preparation](STEP1_PREPROCESSING.md) — prerequisite step
