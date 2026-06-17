# Parametric ParticleNetMD Study

## Overview

ParametricPN is a fixed-`mHc` ParticleNetMD study that trains one model for
multiple `mA` hypotheses:

```text
P(class | event, mA)
```

For the first study, `mHc` is fixed to `130` and the trained hypotheses are:

```text
mA = 85, 90, 95
```

The classifier output remains the same 4-class target used by the standard
ParticleNetMD training:

```text
signal, nonprompt, diboson, ttX
```

ParametricPN is intended to answer whether a single mass-conditioned
ParticleNetMD can match the performance of separately trained plain
ParticleNetMD models. The first comparison metrics are ROC curves for each mass
point, followed by score distributions and mass-sculpting checks.

## Model And Data Definition

The only model-input change is one extra graph-level feature:

```text
standard graphInput:   8 era one-hot features
ParametricPN graphInput: 8 era one-hot features + normalized mA
```

The default normalization is:

```text
mA_norm = (mA - 90.0) / 10.0
```

Therefore:

```text
mA=85 -> -0.5
mA=90 ->  0.0
mA=95 ->  0.5
```

No new dataset files are written. Existing `.pt` graph files are loaded, capped,
and exposed through a lazy hypothesis dataset.

Signal events are loaded from the matching signal sample:

```text
TTToHcToWAToMuMu-MHc130_MA85 -> label 0, mA=85
TTToHcToWAToMuMu-MHc130_MA90 -> label 0, mA=90
TTToHcToWAToMuMu-MHc130_MA95 -> label 0, mA=95
```

Background events are exposed once per `mA` hypothesis:

```text
same background event + mA=85 -> background label
same background event + mA=90 -> background label
same background event + mA=95 -> background label
```

This keeps the training target consistent with `P(class | event, mA)`: a
background-like event remains background for every tested signal hypothesis.
The implementation keeps selected base graphs once and materializes the
mA-conditioned copy only when a batch is requested, avoiding full in-memory
triplication of background graphs.

Signal mass points are event-count balanced by applying the per-fold event cap
independently to each `mA` signal sample. The usual class-level weight balancing
is then applied after signal loading and background expansion.

## Configuration

The default config is:

```text
configs/ParametricPNConfig.json
```

It uses the GA-best hyperparameters from the `MHc130_MA90` ParticleNetMD study:

```text
nNodes: 256
optimizer: Adam
initLR: 0.0001
weight_decay: 0.00003
scheduler: ReduceLROnPlateau
dropout_p: 0.4
batch_size: 512
max_epochs: 120
loss_type: disco
disco_lambda: 0.1
train_folds: [0, 1, 2]
valid_folds: [3]
test_folds: [4]
max_events_per_fold_per_class: 50000
augment_phi_rotation: true
```

The standard grouped backgrounds are reused:

```text
nonprompt: 9 TTLL variants
diboson: WZTo3LNu, ZZTo4L_powheg
ttX: TTZ, tZq
```

The output location is:

```text
ParametricPN/{channel}/MHc130_MA85_MA90_MA95/fold-4/
```

Pilot output uses:

```text
ParametricPN/{channel}/MHc130_MA85_MA90_MA95/pilot/
```

## Training

Set up the parent analysis environment first:

```bash
cd ..
source setup.sh
cd ParticleNetMD
```

Before launching training, confirm that all three signal datasets exist:

```bash
for ma in 85 90 95; do
  ls dataset/samples/signals/TTToHcToWAToMuMu-MHc130_MA${ma}/Run1E2Mu_fold-0.pt
  ls dataset/samples/signals/TTToHcToWAToMuMu-MHc130_MA${ma}/Run3Mu_fold-0.pt
done
```

Pilot run:

```bash
python python/trainParametricPN.py \
  --mhc 130 \
  --ma-values 85,90,95 \
  --channel Combined \
  --pilot
```

The default pilot caps are intentionally small:

```text
train: 1000 events per mA signal sample and per background group before expansion
valid/test: 300 events per mA signal sample and per background group before expansion
dataset loading workers: 4
```

With three mA hypotheses and three background groups, the default pilot split
sizes are:

```text
train: 12,000 events = 3,000 signal + 9,000 background
valid: 3,600 events = 900 signal + 2,700 background
test:  3,600 events = 900 signal + 2,700 background
```

Short smoke run with fewer epochs and tighter event caps:

```bash
python python/trainParametricPN.py \
  --mhc 130 \
  --ma-values 85,90,95 \
  --channel Combined \
  --pilot \
  --pilot-max-train-events-per-fold-per-class 200 \
  --pilot-max-eval-events-per-fold-per-class 100 \
  --data-loading-workers 4 \
  --max-epochs 2
```

Full run:

```bash
python python/trainParametricPN.py \
  --mhc 130 \
  --ma-values 85,90,95 \
  --channel Combined
```

Useful overrides:

```bash
python python/trainParametricPN.py \
  --mhc 130 \
  --ma-values 85,90,95 \
  --channel Combined \
  --device cuda:0 \
  --max-events-per-fold-per-class 20000
```

Pilot cap overrides:

```bash
python python/trainParametricPN.py \
  --mhc 130 \
  --ma-values 85,90,95 \
  --channel Combined \
  --pilot \
  --pilot-max-train-events-per-fold-per-class 500 \
  --pilot-max-eval-events-per-fold-per-class 200 \
  --data-loading-workers 8
```

`--data-loading-workers` parallelizes independent PyTorch dataset file reads
inside ParametricPN. The saved `.pt` files are still loaded as complete files
before subsampling, so increasing workers trades wall time for higher temporary
memory and disk I/O pressure. For full training, use a conservative worker
count such as `2` unless the machine has ample free memory.

The training script writes the same artifact types as standard ParticleNetMD:

```text
models/*.pt
CSV/*.csv
trees/*.root
*.json
*_model_info.json
*_performance.json
```

Parametric ROOT trees additionally contain:

```text
param_mA
param_mA_norm
```

Model metadata records:

```text
num_graph_features = 9
mhc = 130
ma_values = [85, 90, 95]
ma_center = 90.0
ma_scale = 10.0
```

## Comparison Plots

The comparison script evaluates the trained ParametricPN model against the
existing GA-best plain ParticleNetMD model for each mass point:

```text
GAOptim/Combined/MHc130_MA85/fold-4/best_model/model.pt
GAOptim/Combined/MHc130_MA90/fold-4/best_model/model.pt
GAOptim/Combined/MHc130_MA95/fold-4/best_model/model.pt
```

Run after ParametricPN training finishes:

```bash
python python/compareParametricPN.py \
  --mhc 130 \
  --ma-values 85,90,95 \
  --channel Combined
```

For a pilot comparison:

```bash
python python/compareParametricPN.py \
  --mhc 130 \
  --ma-values 85,90,95 \
  --channel Combined \
  --pilot \
  --max-events-per-class 2000
```

Default comparison output:

```text
ParametricPN/Combined/MHc130_MA85_MA90_MA95/fold-4/comparison/
```

Pilot comparison output:

```text
ParametricPN/Combined/MHc130_MA85_MA90_MA95/pilot/comparison/
```

The comparison produces one subdirectory per mass point:

```text
comparison/
├── MA85/
├── MA90/
├── MA95/
├── auc_summary.csv
├── dcor_summary.csv
└── summary.json
```

Per-mass plots include:

```text
roc_MA{ma}_nonprompt.png
roc_MA{ma}_diboson.png
roc_MA{ma}_ttX.png
score_signal_MA{ma}.png
mass_sculpting_MA{ma}_{background}_mass1.png
mass_sculpting_MA{ma}_{background}_mass2.png
```

ROC curves use the same likelihood-ratio score convention as the standard
ParticleNetMD tools:

```text
LR(signal vs bg) = P(signal) / (P(signal) + P(bg))
```

Mass-sculpting plots compare the no-cut mass shape with three LR regions:

```text
LR < 0.3
0.3 <= LR <= 0.7
LR > 0.7
```

The summary files contain AUC values and distance-correlation values between
the LR score and `mass1` or `mass2`. ROC, score, and mass-sculpting plots use
the full selected comparison sample. The exact pairwise distance-correlation
calculation is capped by `--dcor-max-events` per model/background/mass
combination, defaulting to `3000`, to avoid the O(N^2) memory cost of computing
dCor on the full 50k-event comparison slices.

## Interpretation Checklist

For each mass point, compare ParametricPN against the GA-best plain
ParticleNetMD baseline:

- ROC/AUC for signal vs nonprompt, diboson, and ttX.
- Signal-score distributions by true class.
- `mass1` and `mass2` sculpting in background classes.
- Distance-correlation summaries in `dcor_summary.csv`.

The ParametricPN model is promising if it keeps ROC performance close to the
single-mass baselines while showing comparable or better mass-sculpting control.

Watch especially for:

- Strong AUC degradation at only one endpoint mass (`MA85` or `MA95`).
- A visible background mass-shape change in the high-LR region.
- Larger LR-mass distance correlation than the plain ParticleNetMD baseline.
- A suspiciously strong dependence of the signal score on `mA` for backgrounds.

## Implementation Map

Main files:

```text
configs/ParametricPNConfig.json
python/trainParametricPN.py
python/compareParametricPN.py
python/lib/ParametricDataPipeline.py
```

Shared modules used without changing their public training behavior:

```text
python/lib/TrainingOrchestrator.py
python/lib/ResultPersistence.py
python/lib/WeightedLoss.py
python/lib/MultiClassModels.py
```

`TrainingOrchestrator` now reads `config.args.num_graph_features`, allowing
ParametricPN to build a 9-graph-feature model while standard ParticleNetMD
continues to use 8 graph features.

`ResultPersistence` writes `param_mA` and `param_mA_norm` branches only for
parametric runs. Standard training outputs are unchanged.
