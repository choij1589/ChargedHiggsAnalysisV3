# ParticleNetMD: Mass-Decorrelated ParticleNet Workflow

## Overview

ParticleNetMD trains a **4-class GNN classifier** (signal vs nonprompt/diboson/ttX) with
**DisCo loss** (Distance Correlation) regularization. The key innovation over `ParticleNet/` is
that the classifier is explicitly decorrelated from the OS di-muon pair mass distributions,
preventing artificial mass bumps in background estimates.

```
L_total = L_CE + λ · DisCo(score, mass1) + λ · DisCo(score, mass2)
```

- `mass1`, `mass2`: sorted OS muon pair invariant masses stored per event
- DisCo = 0 → score is statistically independent of mass (desired)
- `λ` (`disco_lambda`): decorrelation strength, tunable in config

For full dataset pipeline documentation, see [`docs/STEP1_PREPROCESSING.md`](STEP1_PREPROCESSING.md).

---

## Quick Start

```bash
# 0. Environment
cd /path/to/ChargedHiggsAnalysisV3
source setup.sh
cd ParticleNetMD

# 1. Diboson promotion tables (required before saveDatasets.sh)
python python/dibosonRankPromote.py

# 2. Nonprompt validation (recommended)
python python/nonpromptPromotion.py

# 3. Create all datasets
./scripts/saveDatasets.sh

# 4. Validate datasets
python python/validateDatasets.py --workers 30   # fill phase
python python/validateDatasets.py --plotting      # plot phase

# 5. Lambda sweep (select optimal DisCo lambda)
sbatch submit_sweep.slurm
# After completion: run compareDecorrelation.py (see Step 4 below)

# 6. GA hyperparameter optimization
./scripts/launchGAOptim.sh --signal MHc130_MA90 --channel Combined \
    --device cuda:0

# 7. Visualize GA iteration results
./scripts/visualizeGAIteration.sh MHc130_MA90 Combined 3 cuda:0 --parallel --jobs 8

# 8. Summarize GA loss and select best model
python python/summarizeGALoss.py --signal MHc130_MA90 --channel Combined
```

---

## Step-by-Step Workflow

### Step 1: Data Augmentation Prerequisites

Two scripts must run before dataset creation. See [`docs/STEP1_PREPROCESSING.md §2`](STEP1_PREPROCESSING.md#2-data-augmentation-prerequisites) for full details.

**Diboson conditional promotion tables (required):**
```bash
python python/dibosonRankPromote.py
```
Learns P(n_bjets, rank | nJets) from genuine Tight+Bjet diboson events. Outputs
`DataAugment/diboson/plots/rank_promote/conditional_tables.json` — required by
`saveDataset.py --sample-type diboson`.

Validation plots saved to `DataAugment/diboson/plots/rank_promote/{Run2|Run3}/{channel}/`.

**Nonprompt LNT promotion validation (recommended):**
```bash
python python/nonpromptPromotion.py
```
Validates that loose-not-tight events with fake rate weights reproduce the genuine tight
distribution. Validation plots saved to `DataAugment/nonprompt/plots/lnt_promote/{Run2|Run3}/{channel}/`.

---

### Step 2: Dataset Creation

```bash
./scripts/saveDatasets.sh
```

Processes 4 sample types in parallel. See [`docs/STEP1_PREPROCESSING.md §3`](STEP1_PREPROCESSING.md#3-dataset-creation) for per-class details.

| Class | Samples | Selection | Augmentation |
|-------|---------|-----------|--------------|
| Signal | 6 mass points | Tight+Bjet | None |
| Nonprompt | 9 TTLL variants | LNT+Bjet | Fake rate reweighting |
| Diboson | WZ + ZZ | Tight+0tag promoted | Rank-based b-jet promotion |
| ttX | TTZ + tZq | Tight+Bjet | None |

**Output:** `dataset/samples/{signals,backgrounds}/` — 190 `.pt` files total (6+13 samples × 2 channels × 5 folds).

---

### Step 3: Dataset Validation

Two-phase pipeline using [`python/validateDatasets.py`](../python/validateDatasets.py).

**Fill phase** (slow, parallel — run once):
```bash
python python/validateDatasets.py --workers 30
```
Loads `.pt` files, extracts 40 kinematic observables, saves ROOT histograms to
`results/Validation/{process}/{channel}_fold{fold}.root`.

**Plot phase** (fast — re-run freely):
```bash
python python/validateDatasets.py --plotting
```
Produces shape-comparison and fold-balance overlay plots:
```
plots/Validation/{Run2|Run3}/{channel}/
├── class_overlay/     (all 6 processes overlaid, shape-normalized)
├── fold_overlay/      (per-process, 5 folds overlaid — checks for fold bias)
└── summary.json       (event counts, weight sums per process/fold)
```

**Key checks:**
- `mass1`, `mass2` peak near Z mass for signal
- `bjet1_pt` of promoted diboson is physically reasonable
- No fold-dependent shape differences
- `weight` distribution shows no pathological fake-rate weights

---

### Step 4: Lambda Sweep (DisCo Decorrelation Scan)

Scans 8 λ values (`0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5`) across 3 pilot signals
(`MHc130_MA90`, `MHc100_MA95`, `MHc160_MA85`) on the `Combined` channel to identify the
optimal DisCo regularization strength before committing to GA optimization.

**Quick sanity check — single model:**
```bash
python python/trainMultiClass.py --signal MHc130_MA90 --channel Combined --pilot
```

**Submit to SLURM (recommended):**
```bash
sbatch submit_sweep.slurm
# Monitor: squeue -u $USER
# Logs: logs/sweep_Combined_{signal}_sweep.log    (training)
#        logs/viz_Combined_{signal}_lambda{lam}.log (per-model viz)
```

**Or run locally:**
```bash
bash scripts/runLambdaSweep.sh         # full run
bash scripts/runLambdaSweep.sh --pilot # quick sanity check
```

**What `runLambdaSweep.sh` does automatically:**
1. Trains 3 signals × 8 λ = 24 models in parallel (one signal per GPU via `CUDA_VISIBLE_DEVICES`)
2. Verifies each launcher logged exactly 8 `"Training complete"` lines
3. Runs all 24 per-model visualizations in parallel (`visualizeMultiClass.py`)

**Post-sweep: cross-lambda comparison (run after sbatch completes):**
```bash
for sig in MHc130_MA90 MHc100_MA95 MHc160_MA85; do
    python python/compareDecorrelation.py --signal ${sig} --channel Combined
done
```

Compares performance metrics and mass-sculpting across all λ values.
Output: `plots/Combined/multiclass/{signal}/lambda_comparison/`

**Model naming convention:** `discoL{lam_str}` where `.` → `p` (e.g. `0.005` → `discoL0p005`).

**Output locations:**
```
results/{signal}_Combined_discoL{lam_str}.json        # training history
results/trees/{signal}_Combined_discoL{lam_str}.root  # score trees
plots/Combined/multiclass/{signal}/discoL{lam_str}/   # per-model plots
plots/Combined/multiclass/{signal}/lambda_comparison/ # cross-λ comparison
```

**Config:** [`configs/SglConfig.json`](../configs/SglConfig.json) — set `disco_lambda` to
the chosen value before GA optimization.

After choosing λ: set `disco_parameters.disco_lambda` in [`configs/GAConfig.json`](../configs/GAConfig.json),
then proceed to Step 5.

---

### Step 5: GA Hyperparameter Optimization

Searches over `nNodes`, optimizer, scheduler, `initLR`, `weight_decay` with `disco_lambda`
fixed (see [`configs/GAConfig.json`](../configs/GAConfig.json)).
For full documentation see [`docs/STEP3_HYPERPARAM.md`](STEP3_HYPERPARAM.md).

```bash
# Single signal, two channels in parallel
./scripts/launchGAOptim.sh --signal MHc130_MA90 \
    --channel Run1E2Mu,Run3Mu \
    --device cuda:0,cuda:1

# Multiple signal:channel pairs
./scripts/launchGAOptim.sh \
    --config MHc130_MA90:Run1E2Mu,MHc130_MA90:Run3Mu \
    --device cuda:0,cuda:1

# Pilot run (single fold, 8000/2000 event caps, 10 epochs — completes in <5 min)
./scripts/launchGAOptim.sh --signal MHc130_MA90 --channel Combined \
    --device cuda:0 --pilot

# Resume after interruption (preserves iters 0..N-1, retrains N onward)
./scripts/launchGAOptim.sh \
    --config MHc130_MA90:Combined,MHc100_MA95:Combined \
    --device cuda:0,cuda:1 --resume-from 2
```

GA parameters (in `GAConfig.json`):
- `population_size: 16` — models per iteration
- `max_iterations: 4` — generations (iters 0–3)
- `fitness_metric: loss/valid` — minimize validation loss (lower is better)
- `overfitting_penalty_weight: 0.3` — penalize `valid_loss - train_loss`

Logs per job: `logs/GA_{signal}_{channel}.log`

Results: `GAOptim/{channel}/{signal}/fold-4/GA-iter{N}/`

---

### Step 6: GA Iteration Visualization

Per-model diagnostic plots for a given GA iteration. Supports parallel model evaluation
and multi-signal processing across GPUs.

```bash
# Single signal
./scripts/visualizeGAIteration.sh MHc130_MA90 Combined 3 cuda:0 --parallel --jobs 8

# Multiple signals on different GPUs
./scripts/visualizeGAIteration.sh "MHc100_MA95,MHc160_MA85,MHc130_MA90" Combined 3 \
    "cuda:0,cuda:1,cuda:2" --parallel --jobs 8

# Skip evaluation (reuse existing histograms, regenerate plots only)
./scripts/visualizeGAIteration.sh MHc130_MA90 Combined 3 cuda:0 --skip-eval
```

Produces per-model plots in `GAOptim/{channel}/{signal}/fold-4/GA-iter{N}/plots/`:
- Training curves (loss, accuracy, CE/DisCo decomposition)
- ROC curves and confusion matrix
- Score distributions per class
- Mass sculpting and score-vs-mass 2D profiles
- KS test heatmap (overfitting diagnostic)

Also generates summary plots comparing all models in the iteration.

Overfitting diagnostics saved to `GAOptim/{channel}/{signal}/fold-4/GA-iter{N}/overfitting_diagnostics/`
with 16 KS tests per model (4 true classes × 4 output scores, p < 0.05 = overfitted).

---

### Step 7: GA Loss Summary and Best Model Selection

Summarizes GA optimization across all iterations and copies the best model.

```bash
# Run for each signal
python python/summarizeGALoss.py --signal MHc100_MA95 --channel Combined
python python/summarizeGALoss.py --signal MHc160_MA85 --channel Combined
python python/summarizeGALoss.py --signal MHc130_MA90 --channel Combined
```

**What it does:**
1. Reads all `GA-iter*` directories and computes per-iteration statistics (mean/std/min/max
   of train/valid loss, CE/DisCo decomposition, accuracy)
2. Creates loss evolution plot (`ga_loss_evolution.png`) — mean ± std across iterations
3. Creates DisCo decomposition plot (`ga_disco_decomposition.png`) — CE vs DisCo term tracking
4. Copies best model (lowest validation loss from the last iteration) to `best_model/`

**Output:**
```
GAOptim/{channel}/{signal}/fold-4/
├── ga_loss_summary.json           # Per-iteration statistics
├── ga_loss_evolution.png          # Loss evolution plot
├── ga_disco_decomposition.png     # CE + DisCo term plot
└── best_model/
    ├── model.pt                   # Best model checkpoint
    ├── model_info.json            # Hyperparameters + training summary
    ├── confusion_matrix.png       # 7 validation plots copied from
    ├── ks_test_heatmap.png        #   the best model's GA-iter plots
    ├── mass_profile_vs_score.png
    ├── mass_sculpting.png
    ├── roc_curves.png
    ├── score_distributions_grid.png
    └── training_curves.png
```

---

## Directory Structure

```
ParticleNetMD/
├── configs/
│   ├── SglConfig.json             # Single-model training config
│   ├── GAConfig.json              # GA optimization config
│   └── histkeys_validate.json     # Observable config for validateDatasets.py
├── dataset/
│   └── samples/
│       ├── signals/               # 6 mass points × 2 channels × 5 folds
│       └── backgrounds/           # 13 backgrounds × 2 channels × 5 folds
├── DataAugment/
│   ├── diboson/
│   │   └── plots/rank_promote/    # Validation plots + conditional_tables.json
│   └── nonprompt/
│       └── plots/lnt_promote/     # Validation plots + summary.json
├── python/
│   ├── trainMultiClass.py         # Main training script
│   ├── saveDataset.py             # Per-sample dataset creation entry point
│   ├── launchGAOptim.py           # GA optimization driver
│   ├── evaluateGAModels.py        # Post-GA overfitting evaluation (KS tests)
│   ├── visualizeGAIteration.py    # GA iteration visualization
│   ├── summarizeGALoss.py         # GA loss summary + best model selection
│   ├── launchLambdaSweep.py       # DisCo lambda sweep
│   ├── visualizeMultiClass.py     # Single-model training result visualization
│   ├── compareDecorrelation.py    # Lambda sweep comparison
│   ├── validateDatasets.py        # Dataset validation (fill + plot phases)
│   ├── dibosonRankPromote.py      # Diboson promotion validation + table generation
│   ├── nonpromptPromotion.py      # Nonprompt LNT promotion validation
│   └── lib/                       # Library modules
│       ├── TrainingOrchestrator.py    # Training loop with DisCo integration
│       ├── MultiClassModels.py        # 4-class ParticleNet GNN
│       ├── WeightedLoss.py            # DisCo + weighted cross-entropy
│       ├── DataPipeline.py            # Dataset loading, DataLoader configuration
│       ├── DynamicDatasetLoader.py    # Combined channel on-the-fly merge
│       ├── Preprocess.py              # ROOT → PyG, mass1/mass2, fake rates, promotion
│       ├── SglConfig.py               # JSON config loader
│       ├── ResultPersistence.py       # Model checkpoints + GA-compatible JSON output
│       └── ROCCurveCalculator.py      # ROC curves and metrics
├── scripts/
│   ├── saveDatasets.sh            # Bulk dataset creation
│   ├── runLambdaSweep.sh          # Lambda sweep runner
│   ├── launchGAOptim.sh           # GA optimization launcher
│   └── visualizeGAIteration.sh    # GA iteration visualization wrapper
├── results/                       # Training outputs (models, JSON, ROOT trees)
├── logs/                          # Dataset stats JSON, GA training logs
├── plots/Validation/              # validateDatasets.py output
└── docs/
    ├── STEP1_PREPROCESSING.md     # Full data pipeline documentation
    └── WORKFLOW.md                # This file
```

---

## Key Concepts

### Node Features (9 dimensions)

```
[E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]
```

B-jets are a separate particle type — no `btagScore` feature. Edges connect k=4
nearest-neighbor particles by ΔR.

### Era Encoding (8 dimensions)

Graph-level input `graphInput` is an 8-dim one-hot vector:
```
[2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix]
```

### OS Muon Pair Masses

`mass1 ≤ mass2` (sorted). For Run1E2Mu (only 1 OS pair): `mass2 = -1`, ignored in DisCo.
For Run3Mu (2 OS pairs from 3 muons): both masses stored.

### DisCo Loss

Event-weighted distance correlation between classifier score and OS pair masses:

```
DisCo(X, Y) = 0  →  X and Y are statistically independent
DisCo(X, Y) = 1  →  X and Y are fully correlated
```

Reasonable `disco_lambda` range: 0.01–0.1. Start small and increase if mass
sculpting is observed. Over-decorrelation (`λ` too large) causes accuracy loss.

### Combined Channel

`Combined` channel is not pre-computed — [`DynamicDatasetLoader`](../python/DynamicDatasetLoader.py)
merges `Run1E2Mu + Run3Mu` folds on-the-fly at training time.

---

## Output Formats

### GA-Compatible JSON (`results/{model}.json`)

```json
{
  "hyperparameters": {
    "signal": "MHc130_MA90", "channel": "Run1E2Mu",
    "num_hidden": 256, "optimizer": "Adam",
    "initial_lr": 0.0005, "scheduler": "ExponentialLR",
    "loss_type": "disco", "disco_lambda": 0.05
  },
  "training_summary": {
    "best_epoch": 45, "best_valid_loss": 0.87,
    "best_valid_acc": 0.63, "total_epochs": 100
  },
  "epoch_history": {
    "train_loss": [...], "valid_loss": [...],
    "train_acc": [...],  "valid_acc": [...],
    "train_ce_loss": [...], "train_disco_term": [...],
    "valid_ce_loss": [...], "valid_disco_term": [...],
    "epoch": [...], "timestamp": [...]
  }
}
```

### ROOT Tree (`results/trees/{model}.root`)

| Branch | Description |
|--------|-------------|
| `score_signal/nonprompt/diboson/ttX` | Class probabilities |
| `true_label` | Ground truth (0=signal, 1=nonprompt, 2=diboson, 3=ttX) |
| `train_mask`, `valid_mask`, `test_mask` | Data split flags |
| `weight` | Physics event weight |
| `mass1`, `mass2` | OS muon pair masses (for decorrelation analysis) |

---

## References

- [Data pipeline details](STEP1_PREPROCESSING.md) — augmentation, fold structure, validation
- [Dataset event counts](../DataAugment/dataset.md) — per-sample statistics and training budget
- [Nonprompt augmentation](../DataAugment/nonprompt.md) — LNT promotion, sample selection
- [ttX sample selection](../DataAugment/ttX.md) — TTW/TTH exclusion rationale
- [Training config](../configs/SglConfig.json) — folds, DisCo lambda, background groups
- [GA config](../configs/GAConfig.json) — search space, fitness metric, population size
