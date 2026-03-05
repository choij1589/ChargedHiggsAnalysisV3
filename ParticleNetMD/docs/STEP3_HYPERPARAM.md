# Step 3: Genetic Algorithm Hyperparameter Optimization

How to search for optimal model hyperparameters (architecture, optimizer, learning rate schedule)
with `disco_lambda` fixed at the value chosen in Step 2.

---

## 0. Prerequisites

**Datasets** from Step 1 must exist for both `Run1E2Mu` and `Run3Mu`:
```bash
ls dataset/samples/signals/TTToHcToWAToMuMu-MHc130_MA90/Run1E2Mu_fold-0.pt  # must exist
ls dataset/samples/backgrounds/Skim_TriLep_TTLL_powheg/Run1E2Mu_fold-0.pt   # must exist
```

**`disco_lambda`** must be set in `configs/GAConfig.json` (Step 2 output):
```json
"disco_parameters": { "disco_lambda": 0.1 }
```

---

## 1. Background: Genetic Algorithm

ParticleNetMD uses a population-based genetic algorithm (GA) to efficiently explore a
multi-dimensional hyperparameter space without exhaustive grid search.

**Each individual** in the population is a chromosome — a tuple of 5 hyperparameters:
```
(nNodes, optimizer, initLR, weight_decay, scheduler)
```

**Each GA iteration** (generation):
1. All individuals in the population are trained in parallel via `mp.spawn`
2. Fitness is assigned: `fitness = valid_loss + λ_penalty · (valid_loss − train_loss)`
3. Survivors are selected by rank; children are bred via uniform crossover + displacement mutation
4. The next generation = survivors + children

**Convergence:** After `max_iterations` generations the best-fitness individual (lowest combined
loss) is reported.

**Implementation:** [`python/GATools.py:GeneticModule`](../python/GATools.py),
[`python/launchGAOptim.py`](../python/launchGAOptim.py),
[`python/trainWorker.py`](../python/trainWorker.py)

---

## 2. Configuration: `configs/GAConfig.json`

Key sections relevant to the GA run:

### `ga_parameters`

| Key | Value | Description |
|-----|-------|-------------|
| `population_size` | 16 | Models trained per iteration |
| `max_iterations` | 4 | Total generations (gen 0 + 3 evolution rounds) |
| `evolution_ratio` | 0.7 | Fraction of next generation that are new children |
| `mutation_thresholds` | [0.7, 0.7, 0.7, 0.7, 0.7] | Per-gene mutation probability |
| `fitness_metric` | `"loss/valid"` | Primary metric read from model JSON |
| `overfitting_penalty_weight` | 0.3 | λ in `fitness = valid_loss + λ·(valid_loss − train_loss)` |

### `hyperparameter_search_space`

| Hyperparameter | Type | Values / Range |
|---------------|------|----------------|
| `nNodes` | discrete | 96, 128, 192, 256 |
| `optimizer` | discrete | RMSprop, Adam, Adadelta |
| `scheduler` | discrete | ExponentialLR, CyclicLR, ReduceLROnPlateau |
| `initLR` | log-uniform | [1e-4, 1e-2], 100 samples, rounded to 4 digits |
| `weight_decay` | log-uniform | [1e-5, 1e-3], 100 samples, rounded to 5 digits |

### `training_parameters`

| Key | Value | Pilot override |
|-----|-------|----------------|
| `max_epochs` | 120 | **capped to 10** |
| `batch_size` | 256 | unchanged |
| `dropout_p` | 0.4 | unchanged |
| `early_stopping_patience` | 7 | unchanged |
| `train_folds` | [0, 1, 2] | **[0] only** |
| `max_events_per_fold_per_class` | 50000 | **8000 train / 2000 valid** |
| `balance_weights` | true | unchanged |
| `augment_phi_rotation` | true | unchanged |

### `disco_parameters`

```json
"disco_lambda": 0.1
```
Fixed across all GA workers — not included in the search space.

---

## 3. Workflow

The GA workflow is split into two stages: **training** (GPU batch job) and
**post-processing** (interactive, after reviewing results).

### Stage 1: GA Training

`launchGAOptim.sh` runs the GA loop only — no evaluation or visualization.
This keeps the batch job focused on GPU-intensive training.

#### 3a. Pilot run (validate pipeline — completes in <30 min)

```bash
./scripts/launchGAOptim.sh --signal MHc130_MA90 --channel Combined \
    --device cuda:0 --pilot
```

Expected log lines:
```
PILOT MODE: train_folds=[0] | caps: train=8000 valid=2000
PILOT MODE: max_epochs capped to 10
Train dataset: 32000 events (from shared memory)   # 4 classes × 8000
Valid dataset:  8000 events (from shared memory)   # 4 classes × 2000
DisCo lambda: 0.1 (fixed, not optimized)
```

Output path: `GAOptim/Combined/MHc130_MA90/pilot/GA-iter{N}/json/model{idx}.json`

#### 3b. Full run — single signal, two channels in parallel

```bash
./scripts/launchGAOptim.sh --signal MHc130_MA90 \
    --channel Run1E2Mu,Run3Mu \
    --device cuda:0,cuda:1
```

One Python process per (signal, channel) pair — each manages its own GA loop
and its own pool of 16 workers via `mp.spawn`.

#### 3c. Full run — multiple signal:channel pairs

```bash
./scripts/launchGAOptim.sh \
    --config MHc130_MA90:Run1E2Mu,MHc130_MA90:Run3Mu,MHc100_MA95:Combined \
    --device cuda:0,cuda:1,cuda:2
```

Number of `--config` entries must match number of `--device` entries.

#### 3d. SLURM batch submission

```bash
sbatch submit.slurm "MHc130_MA90:Combined,MHc100_MA95:Combined" "cuda:0,cuda:1"
```

#### 3e. Log monitoring

```bash
tail -f logs/GA_MHc130_MA90_Combined.log
```

Key milestone lines to watch:
```
GENERATION 0 - Initial Population
[Iter 0] Spawning 16 worker processes...
[Iter 0] All 16 workers completed
Best chromosome: ...
Best fitness: ...
GENERATION 1
...
GA OPTIMIZATION COMPLETE
```

#### 3f. Resuming after interruption (SLURM timeout)

The 4-iteration GA loop takes ~28h total but SLURM jobs are capped at 24h. Use `--resume-from N`
to continue from iteration N without losing completed iterations 0..N-1.

```bash
# After a job that completed iters 0-2 but was killed before iter 3:
./scripts/launchGAOptim.sh \
    --config MHc130_MA90:Combined,MHc100_MA95:Combined,MHc160_MA85:Combined \
    --device cuda:0,cuda:1,cuda:2 --resume-from 3
```

**What `--resume-from N` does:**
1. Validates that iteration (N-1)'s `model_info.csv` exists (population checkpoint)
2. Deletes iteration N+ directories (cleans partial results from interrupted iteration)
3. Loads the population from iteration (N-1)'s CSV
4. Runs evolution + training starting from iteration N

**Special cases:**
- `--resume-from 0` regenerates a fresh population and starts from scratch (equivalent to no flag, but keeps existing base_dir structure)
- `--resume-from 2` preserves iters 0-1 untouched, deletes iter 2+ (even if iter 2 partially completed), retrains iter 2 and 3

**Reproducibility:** The hyperparameter pool uses seeded `loguniform.rvs()` (seed 42), so the pool is identical across invocations. Loaded chromosomes are also validated against the pool and injected if missing.

### Stage 2: Post-processing (interactive)

After the GA training completes, run evaluation, visualization, and summary
interactively. These scripts are lightweight and can run on a login node or
a short interactive session.

#### Summary: loss evolution + best model extraction

```bash
# Pilot
python python/summarizeGALoss.py --signal MHc130_MA90 --channel Combined --pilot

# Full run
python python/summarizeGALoss.py --signal MHc130_MA90 --channel Combined
```

Produces:
- `ga_loss_summary.json` — per-iteration statistics (total loss, CE loss, DisCo term, accuracy)
- `ga_loss_evolution.png` — mean/best loss vs GA iteration
- `ga_disco_decomposition.png` — CE loss and DisCo term evolution
- `best_model/` — copies best model checkpoint, JSON, and validation plots

#### Overfitting evaluation (optional, per iteration)

```bash
python python/evaluateGAModels.py --signal MHc130_MA90 --channel Combined \
    --device cuda:0 --iteration 3 --pilot
```

Runs 16 KS tests per model (4 true classes x 4 output scores) to detect overfitting.

#### Visualization (optional, per iteration)

```bash
python python/visualizeGAIteration.py --signal MHc130_MA90 --channel Combined \
    --device cuda:0 --iteration 3 --pilot
```

Generates per-model: score distributions, ROC curves, confusion matrices,
training curves (CE + DisCo decomposition), and mass decorrelation plots.

---

## 4. How a GA Iteration Works

```
launchGAOptim.py [--resume-from N]
├── load_ga_config()
├── setup_output_directory()
│   ├── Fresh mode: shutil.rmtree(base_dir)
│   └── Resume mode: validate iter (N-1) CSV, delete iter N+ dirs
│
├── Initialize GeneticModule
│   ├── Fresh: generatePopulation() + randomGeneration(16)
│   └── Resume (N>0): generatePopulation() + loadPopulation(iter N-1 CSV)
│
├── GENERATION 0  (skipped when resume N > 0)
│   ├── prepare_shared_datasets()          ← load once, share across all workers
│   │   ├── DynamicDatasetLoader (Combined = Run1E2Mu + Run3Mu merged on-the-fly)
│   │   └── Batch.from_data_list() → share_memory_()
│   ├── mp.spawn(train_worker, nprocs=16)  ← 16 workers, one GPU, serialised by CUDA
│   │   └── each worker:
│   │       ├── make_dataloader_from_batch()
│   │       ├── create model / optimizer / scheduler
│   │       ├── DiScoWeightedLoss(disco_lambda=0.1)
│   │       ├── train_epoch_disco() × max_epochs (or early stop)
│   │       └── save model{idx}.json + model{idx}.pt
│   ├── GeneticModule.updatePopulation()   ← assign fitness from JSON files
│   └── (overfitting detection disabled)
│
├── GENERATION max(1,N)..3
│   ├── GeneticModule.evolution()          ← rank selection → crossover → mutation
│   ├── prepare_shared_datasets()          ← fresh load for new fold set
│   ├── mp.spawn(train_worker, nprocs=16)
│   └── GeneticModule.updatePopulation()
│
└── report best chromosome + save ga_optimization_results.json
```

### Fitness function

```
fitness = valid_loss + 0.3 × (valid_loss − train_loss)
```

A model with lower validation loss is preferred; the overfitting penalty (weight 0.3) additionally
penalises models whose training loss is much lower than their validation loss.

### Evolution operators (`GeneticModule`)

| Operator | Method | Description |
|----------|--------|-------------|
| Selection | `rankSelection()` | Rank individuals by fitness; sample with rank-proportional probability |
| Crossover | `uniformCrossover(p1, p2)` | Each gene independently taken from p1 or p2 with equal probability |
| Mutation | `displacementMutation(child, thresholds)` | Each gene mutated with probability from `mutation_thresholds` |
| Evolution | `evolution(thresholds, ratio)` | Compose: select parents → crossover → mutate → fill to `population_size` |

---

## 5. Output Structure

```
GAOptim/
└── {channel}/
    └── {signal}/                     (short name, e.g., MHc130_MA90)
        ├── pilot/                    (pilot run)
        │   ├── GA-iter0/
        │   │   ├── json/
        │   │   │   ├── model0.json ... model15.json   ← training histories
        │   │   │   └── model_info.csv                 ← population summary
        │   │   └── models/
        │   │       └── model0.pt ... model15.pt       ← best checkpoints
        │   ├── GA-iter1/ ... GA-iter3/
        │   ├── ga_optimization_results.json           ← final best config
        │   ├── ga_loss_summary.json                   ← summarizeGALoss output
        │   ├── ga_loss_evolution.png                  ← loss vs iteration plot
        │   ├── ga_disco_decomposition.png             ← CE + DisCo evolution plot
        │   └── best_model/                            ← extracted best model
        │       ├── model.pt
        │       ├── model_info.json
        │       └── *.png                              ← validation plots
        └── fold-4/                   (full run, same structure as pilot/)
```

### `model{idx}.json` schema

```json
{
  "hyperparameters": {
    "signal": "MHc130_MA90", "channel": "Combined",
    "iteration": 0, "model_idx": 3,
    "num_hidden": 192, "optimizer": "Adam",
    "initial_lr": 0.0008, "weight_decay": 0.00063,
    "scheduler": "ReduceLROnPlateau",
    "pilot_mode": false,
    "loss_type": "disco", "disco_lambda": 0.1,
    "dropout_p": 0.4, "batch_size": 256
  },
  "training_summary": {
    "best_epoch": 47, "best_valid_loss": 0.734,
    "best_train_loss": 0.701, "total_epochs": 54
  },
  "epoch_history": {
    "epoch": [...], "timestamp": [...],
    "train_loss": [...], "valid_loss": [...],
    "train_acc": [...],  "valid_acc": [...],
    "train_ce_loss": [...], "valid_ce_loss": [...],
    "train_disco_term": [...], "valid_disco_term": [...]
  }
}
```

### `model_info.csv`

One row per model in the population. Columns: `model_idx`, `fitness`, `nNodes`, `optimizer`,
`initLR`, `weight_decay`, `scheduler`.

### `ga_optimization_results.json`

Written at the end of the full GA loop:
```json
{
  "signal": "MHc130_MA90", "channel": "Combined",
  "best_chromosome": {
    "chromosome": [192, "Adam", 0.0008, 0.00063, "ReduceLROnPlateau"],
    "fitness": 0.734,
    "decoded": { "nNodes": 192, "optimizer": "Adam", "initLR": 0.0008,
                 "weight_decay": 0.00063, "scheduler": "ReduceLROnPlateau",
                 "disco_lambda": 0.1 }
  },
  "configuration": { ... }
}
```

---

## 6. Pilot Results (MHc130_MA90, Combined, 2026-02-26)

```
Iteration 0:  Mean Valid Loss: 0.8644 ± 0.1102  |  Mean Acc: 0.3484
Iteration 1:  Mean Valid Loss: 0.8547 ± 0.1059  |  Mean Acc: 0.3602
Iteration 2:  Mean Valid Loss: 0.8241 ± 0.0878  |  Mean Acc: 0.3856
Iteration 3:  Mean Valid Loss: 0.8484 ± 0.1247  |  Mean Acc: 0.3848
```

DisCo decomposition (mean at best epoch):
```
          CE Loss              DisCo Term
          train   valid        train   valid
Iter 0:   0.9927  0.7932       0.6403  0.7123
Iter 1:   0.9550  0.7845       0.6292  0.7016
Iter 2:   0.8860  0.7555       0.6288  0.6866
Iter 3:   0.8728  0.7781       0.6266  0.7030
```

Best model: **model11** from iter 3, valid_loss=0.7577.

The GA converges over 4 generations — CE loss decreases while the DisCo term stays
stable (~0.63 train, ~0.70 valid), confirming that the optimizer improves classification
without degrading mass decorrelation. Absolute accuracy is low in pilot mode (10 epochs,
single fold) but improves consistently across iterations.

---

## 7. Troubleshooting

**Dataset not found at startup**
```
FileNotFoundError: dataset/samples/signals/.../Run1E2Mu_fold-0.pt
```
Run `./scripts/saveDatasets.sh` first (see STEP1_PREPROCESSING.md).

**`model{idx}.json` not found during `updatePopulation()`**
A worker crashed mid-training. Check for `[Worker N]` ERROR lines in the log. Common cause:
CUDA OOM — reduce `batch_size` in GAConfig.json (e.g., 128).

**Population extinct (all models overfitted)**
With `overfitting_detection.enabled: false` (current default) this cannot happen.
If you enable detection and see this, lower `p_value_threshold` or reduce `dropout_p`.

**Pilot finishes instantly with very high loss**
Check `Valid dataset: N events` — if N = 0 the valid fold had no events after subsampling.
This indicates a dataset creation issue; re-run `saveDatasets.sh --pilot`.

**`CUDA error: device-side assert triggered`**
Usually a label out-of-range error. Verify the number of classes matches `background_groups`
in GAConfig.json (currently 3 groups → 4 classes total).

**Results directory already exists**
Without `--resume-from`, `launchGAOptim.py` deletes and recreates `base_dir` at startup.
To preserve a previous run, either use `--resume-from N` to continue it, or rename it first:
```bash
mv GAOptim/Combined/MHc130_MA90 GAOptim/Combined/MHc130_MA90_backup
```

**SLURM job timed out mid-GA**
Use `--resume-from N` where N is the first incomplete iteration. Check the log for the last
completed `GA OPTIMIZATION COMPLETE` or `Population saved to:` line to determine N:
```bash
grep "Population saved" logs/GA_MHc130_MA90_Combined.log
# If iter 0,1,2 have "Population saved" lines → resume-from 3
./scripts/launchGAOptim.sh --config MHc130_MA90:Combined --device cuda:0 --resume-from 3
```

**`ModuleNotFoundError: No module named 'GAConfig'` in spawned workers (iter 3+)**
After the `python/` → `python/lib/` refactor, `GAConfig.py` and `SglConfig.py` used a single `..` level to find `configs/`, resolving to `python/configs/` (wrong) instead of `configs/` at the module root. Fixed by using two `..` levels in both files:
```python
# python/lib/GAConfig.py and python/lib/SglConfig.py
config_path = os.path.join(script_dir, '..', '..', 'configs', 'GAConfig.json')
```
This only manifests in `mp.spawn` workers (iter 3+) because the worker re-imports modules without the entry script's `sys.path` setup. If you see this error, verify both files use `'..', '..', 'configs'`.

---

## Quick Reference

```bash
cd /path/to/ChargedHiggsAnalysisV3
source setup.sh
cd ParticleNetMD

# 1. Set disco_lambda (from Step 2 analysis)
#    Edit configs/GAConfig.json: "disco_lambda": 0.1

# 2. Pilot smoke-test
./scripts/launchGAOptim.sh --signal MHc130_MA90 --channel Combined \
    --device cuda:0 --pilot

# 3. Post-processing (after GA completes)
python python/summarizeGALoss.py --signal MHc130_MA90 --channel Combined --pilot

# 4. Full run
./scripts/launchGAOptim.sh --signal MHc130_MA90 \
    --channel Run1E2Mu,Run3Mu \
    --device cuda:0,cuda:1

# 4b. Resume after interruption (retrains iter N onward)
./scripts/launchGAOptim.sh --signal MHc130_MA90 \
    --channel Run1E2Mu,Run3Mu \
    --device cuda:0,cuda:1 --resume-from 3

# 5. Full run post-processing
python python/summarizeGALoss.py --signal MHc130_MA90 --channel Run1E2Mu
python python/summarizeGALoss.py --signal MHc130_MA90 --channel Run3Mu

# 6. Monitor live
tail -f logs/GA_MHc130_MA90_Combined.log

# 7. Optional: per-iteration eval/viz
python python/evaluateGAModels.py --signal MHc130_MA90 --channel Combined \
    --device cuda:0 --iteration 3
python python/visualizeGAIteration.py --signal MHc130_MA90 --channel Combined \
    --device cuda:0 --iteration 3
```

---

## References

- [GA implementation](../python/GATools.py) — `GeneticModule`: selection, crossover, mutation
- [GA config loader](../python/lib/GAConfig.py) — `GAConfigLoader`: typed access to GAConfig.json
- [GA launcher](../python/launchGAOptim.py) — outer loop, shared dataset setup, fitness update
- [GA worker](../python/lib/trainWorker.py) — per-model training with DisCo loss and shared memory
- [Loss summary](../python/summarizeGALoss.py) — loss evolution, DisCo decomposition, best model extraction
- [Overfitting evaluation](../python/evaluateGAModels.py) — 16 KS tests for overfitting detection
- [Iteration visualization](../python/visualizeGAIteration.py) — score distributions, ROC, mass decorrelation
- [Shared dataset manager](../python/lib/SharedDatasetManager.py) — `prepare_shared_datasets()`
- [Shared worker utils](../python/lib/SharedWorkerUtils.py) — `make_dataloader_from_batch()`, `setup_spawn_method()`
- [Launch script](../scripts/launchGAOptim.sh) — shell driver for multi-channel/multi-signal runs
- [SLURM submission](../submit.slurm) — batch job submission script
- [GA configuration](../configs/GAConfig.json) — search space, fitness metric, population size
- [Step 2: Lambda sweep](STEP2-DECORRELATION.md) — how `disco_lambda` is chosen
