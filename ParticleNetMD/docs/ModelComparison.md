# ModelComparison: Five-Classifier Benchmark

How to train and compare the five signal/background classifiers — BDT, DNN, DNN-MD, ParticleNet, ParticleNet-MD — on a shared dataset where the OS-dimuon transverse momenta are masked from the tabular features. This is the main model-comparison result of the analysis: it quantifies the discrimination–decorrelation trade-off across model classes and motivates the mass-decorrelated ParticleNet as the production classifier.

---

## 0. Prerequisites

Datasets from [Step 1](STEP1_PREPROCESSING.md) must exist for both `Run1E2Mu` and `Run3Mu` (the `Combined` channel merges them at load time):

```bash
ls dataset/samples/signals/TTToHcToWAToMuMu-MHc130_MA90/Run1E2Mu_fold-0.pt   # must exist
```

Two external inputs are consumed:

- **Masked table cache** — `ModelComparison/dataset/{signal}/fold-4/tables/{train,valid,test}_cap50000.npz`. Built automatically on first run (or with `--cache-only`); rebuilt with `--rebuild-dataset`. The cache drops `os_dimu1_pt` and `os_dimu2_pt` (84 → 82 tabular features) so the BDT/DNN cannot use the pair-pT proxies for the dimuon mass. Manifest: `ModelComparison/dataset/{signal}/fold-4/manifest.json`.
- **LR weights** — `$WORKDIR/SignalRegionStudyV2/configs/thresholds/{signal}.json`, used by `compute_lr_modified` for the per-era/channel background weights. If missing, unit weights are used (a warning is printed).

Environment as usual:

```bash
cd .. && source setup.sh && cd ParticleNetMD
```

---

## 1. Models

All five models are 4-class classifiers (`signal`, `nonprompt`, `diboson`, `ttX`) trained on identical events, folds (train `[0,1,2]`, valid `[3]`, test `[4]`), weights, and class caps (50 000 events/fold/class).

| Model | Trainer | Inputs | Loss | Output dir |
|-------|---------|--------|------|------------|
| BDT | [`trainBDT.py`](../python/trainBDT.py) (`sklearn.HistGradientBoostingClassifier`) | 82 masked tabular features | multiclass log-loss | `ModelComparison/BDT/` |
| DNN | [`trainDNN.py`](../python/trainDNN.py) (MLP, Linear+BatchNorm blocks) | 82 masked tabular features | weighted CE | `ModelComparison/DNN/` |
| DNN-MD | [`trainDNN.py`](../python/trainDNN.py) `--loss-type disco` | 82 masked tabular features | weighted CE + DisCo | `ModelComparison/DNN_MD/` |
| ParticleNet | [`trainMultiClass.py`](../python/trainMultiClass.py) + [canonical config](../ModelComparison/configs/ParticleNet/default_config.json) | particle graphs (no explicit pair-pT column) | weighted CE | `ModelComparison/ParticleNet/` |
| ParticleNet-MD | [`trainMultiClass.py`](../python/trainMultiClass.py) + [canonical MD config](../ModelComparison/configs/ParticleNet_MD/default_config.json) | particle graphs | weighted CE + DisCo (λ = 0.1) | `ModelComparison/ParticleNet_MD/` |

Each model's 4-class score vector is collapsed to a single discriminant:

```
LR_modified = p_sig / (p_sig + w_np·p_np + w_VV·p_VV + w_ttX·p_ttX)
```

with per-era/channel weights `w` from the SignalRegionStudyV2 threshold files. Prediction files (`predictions_{train,test}.npz`) store `y`, `weight`, `mass1`, `mass2`, `era`, `channel_id`, the score matrix (`bdt_scores`/`dnn_scores`/`pn_scores`), and the LR (`bdt_lr`/`dnn_lr`/`pn_lr`).

The dimuon masses are **never** model inputs — they are carried through only for decorrelation diagnostics.

---

## 2. Canonical ParticleNet Configuration

Both ParticleNet variants use the fixed configuration in `ModelComparison/configs/ParticleNet[_MD]/default_config.json` (signal-independent — one config serves all mass points):

| Parameter | Value |
|-----------|-------|
| Conv widths (`conv_channels`) | **512 → 256 → 256** (3 DynamicEdgeConv layers, k = 4) |
| Edge dropout (`edge_dropout_p`) | **0.0** (disabled) |
| Activation dropout (`dropout_p`) | 0.4 |
| Optimizer | Adam, `initLR = 5e-4`, `weight_decay = 3e-5` |
| Scheduler | **CyclicLR** (`base_lr = initLR/5`, `max_lr = 2·initLR`, hardcoded in [`TrainingUtilities.py`](../python/lib/TrainingUtilities.py)) |
| Epochs / batch | 120 max, early stopping patience 10, batch 512 |
| Augmentation | φ-rotation on |
| DisCo λ | 0.0 (ParticleNet) / **0.1** (ParticleNet-MD) |

**Provenance:** selected from a June-2026 LR/scheduler/capacity scan (uniform-256 GA-best baseline vs 512-256-256 at initLR {1e-4, 3e-4, 5e-4} × {ReduceLROnPlateau, CyclicLR}, and a no-edge-dropout test). The retired variants and the earlier GA-chromosome-based comparison outputs are archived under `BackUp/ModelComparison/`. Compared to the GA-best uniform-256 model, the canonical setting gains ≈ +0.012 average test AUC at equal or better mass decorrelation.

**Device caveat:** `trainMultiClass.py` has no `--device` flag — ParticleNet *training* runs on the device in the config's `system_config.device` (`cuda`). The driver's `--device` option affects only tabular training and ParticleNet *evaluation*.

---

## 3. Running the Comparison

One command per signal runs everything — cache build, all five trainings, prediction export, and all plots:

```bash
python python/runModelComparison.py --signal MHc130_MA90
python python/runModelComparison.py --signal MHc160_MA85
python python/runModelComparison.py --signal MHc100_MA95
# or all three sequentially:
python python/runModelComparison.py --all
```

Useful flags:

| Flag | Effect |
|------|--------|
| `--plots-only` | Rebuild all comparison plots from cached `predictions_*.npz` (no training/eval) |
| `--pn-only` | Skip training; re-export ParticleNet/ParticleNet-MD predictions from existing checkpoints, then rebuild plots |
| `--decorrelation-only` | Rebuild only the decorrelation plot suite from cached predictions |
| `--cache-only` | Build/validate the masked table cache and stop |
| `--rebuild-dataset` | Force-rebuild the masked table cache |
| `--pilot` | Smoke test on reduced data (PN reference export is skipped — pilot checkpoints must not overwrite full-statistics predictions) |
| `--device cuda:N` | Device for tabular training and PN evaluation (see device caveat above) |

Models that lack cached predictions are skipped with a warning, so partial runs (e.g. plots for a signal whose ParticleNets are not yet trained) do not crash.

---

## 4. Output Structure

```
ModelComparison/
├── configs/
│   ├── ParticleNet/default_config.json       # canonical CE config
│   └── ParticleNet_MD/default_config.json    # canonical DisCo config
├── dataset/{signal}/fold-4/
│   ├── manifest.json                          # masked-feature bookkeeping
│   └── tables/{split}_cap50000.npz            # shared masked tabular cache
├── {BDT,DNN,DNN_MD}/Combined/{signal}/fold-4/
│   ├── model.{joblib,pt}, feature_names.json, summary.json
│   └── predictions_{train,test}.npz
├── {ParticleNet,ParticleNet_MD}/Combined/{signal}/fold-4/
│   ├── models/*.pt                            # trainMultiClass checkpoints
│   ├── <model-name>.json                      # epoch history + hyperparameters
│   ├── summary.json                           # export metadata + AUC/dCor
│   └── predictions_{train,test}.npz
└── plots/{signal}/
    ├── summary.csv, summary.json              # headline metrics, all models
    ├── roc_*.png, lr_three_models_*.png, mass_sculpting_*.png
    └── decorrelation/
        ├── decorrelation_summary.{csv,json}
        └── mass_profile_*, score_vs_mass_*, roc_*, mass_sculpting_*
```

`BackUp/ModelComparison/` holds the retired exploratory variants (GA-based ParticleNet references, LR/scheduler scan, no-edge-dropout test).

---

## 5. Summaries and Plots

**`plots/{signal}/summary.csv`** — one row per model (test split): `test_average_auc`, per-background AUCs (`auc_nonprompt`, `auc_diboson`, `auc_ttX`), `dcor_psig_mass{1,2}` (distance correlation of the signal score with each dimuon mass), and Z-peak retention metrics (`dcor_lr_mass2_60_120`, `high_lr_fraction_60_120`, `peak_fraction_all_60_120`, `peak_fraction_high_lr_60_120`).

**Plot inventory** (per signal):

- `roc_{model}_train_test.png` — per-model train-vs-test ROC per background class (overfitting check)
- `roc_model_comparison_{background}.png` — all five models overlaid, test split
- `lr_three_models_{class}.png` — normalized LR distributions per true class
- `mass_sculpting_{model}[_{class}]_{mass}.png` — mass shapes in LR regions (< 0.3, 0.3–0.7, > 0.7) vs no-cut, with ratio panel and dCor values
- `decorrelation/mass_profile_*`, `decorrelation/score_vs_mass_*` — ⟨mass⟩ vs score profiles and 2D score–mass maps

---

## 6. Findings (2026-07-21, test split)

### MHc130_MA90

| Model | avg AUC | AUC NP | AUC VV | AUC ttX | dCor(p_sig, m1) | dCor(p_sig, m2) |
|-------|---------|--------|--------|---------|-----------------|-----------------|
| BDT | **0.9380** | 0.9139 | 0.9569 | 0.9433 | 0.195 | 0.320 |
| DNN | 0.9322 | 0.9012 | 0.9561 | 0.9394 | 0.184 | 0.272 |
| DNN-MD | 0.9292 | 0.8952 | 0.9557 | 0.9366 | 0.095 | 0.140 |
| ParticleNet | 0.9348 | 0.9120 | 0.9593 | 0.9330 | 0.078 | 0.130 |
| ParticleNet-MD | 0.9328 | 0.9089 | 0.9581 | 0.9315 | 0.065 | **0.099** |

### MHc160_MA85

| Model | avg AUC | AUC NP | AUC VV | AUC ttX | dCor(p_sig, m1) | dCor(p_sig, m2) |
|-------|---------|--------|--------|---------|-----------------|-----------------|
| BDT | **0.9379** | 0.9110 | 0.9552 | 0.9473 | 0.319 | 0.370 |
| DNN | 0.9289 | 0.8936 | 0.9529 | 0.9403 | 0.297 | 0.355 |
| DNN-MD | 0.9243 | 0.8865 | 0.9499 | 0.9364 | 0.200 | 0.145 |
| ParticleNet | 0.9262 | 0.9022 | 0.9502 | 0.9264 | 0.189 | 0.173 |
| ParticleNet-MD | 0.9251 | 0.9009 | 0.9483 | 0.9262 | 0.157 | **0.125** |

### MHc100_MA95

| Model | avg AUC | AUC NP | AUC VV | AUC ttX | dCor(p_sig, m1) | dCor(p_sig, m2) |
|-------|---------|--------|--------|---------|-----------------|-----------------|
| BDT | **0.9294** | 0.9157 | 0.9487 | 0.9239 | 0.194 | 0.108 |
| DNN | 0.9235 | 0.9031 | 0.9478 | 0.9195 | 0.203 | 0.108 |
| DNN-MD | 0.9210 | 0.8987 | 0.9463 | 0.9179 | 0.148 | 0.079 |
| ParticleNet | 0.9189 | 0.8900 | 0.9479 | 0.9187 | 0.173 | 0.098 |
| ParticleNet-MD | 0.9184 | 0.8896 | 0.9475 | 0.9180 | 0.140 | **0.070** |

### Interpretation

- The BDT has the best raw AUC at every mass point but sculpts the background mass hardest — at high LR the background Z-peak fraction rises to 0.62 (MHc130_MA90) and 0.63 (MHc160_MA85) versus ≈ 0.53 with no cut.
- The graph models are intrinsically less mass-correlated even without DisCo — their input has no explicit pair-pT column.
- **ParticleNet-MD has the lowest dCor(p_sig, mass2) and the least background-shape distortion at every mass point, at a ≤ 1.2% average-AUC cost versus the BDT** — the basis for using it as the production classifier.
- Train/test average AUCs agree to ≤ 0.0014 for the ParticleNet variants at all three mass points — no overfitting from the added capacity.
- MHc100_MA95 shows compressed separations and low dCor for all models: with MA = 95 GeV the signal sits essentially on the Z peak, so mass-shape discrimination is intrinsically limited.
- Provenance: the MHc130_MA90 checkpoints are the June-2026 scan models (reused, not retrained); MHc160_MA85 and MHc100_MA95 were trained 2026-07-20/21 with the one-command workflow in §3 using the canonical configs.

---

## 7. Troubleshooting

- **`No ParticleNet checkpoint found under .../models`** — the canonical training has not run for that signal; run `python python/runModelComparison.py --signal {signal}` (or without `--pn-only`).
- **`Warning: skipping <model> for <signal>; missing ...predictions_*.npz`** — expected for signals without trained models; plots are produced for the available subset.
- **Threshold file not found** — LR weights fall back to unity; check `$WORKDIR/SignalRegionStudyV2/configs/thresholds/`.
- **Pilot runs** — `--pilot` trains smoke-test models but intentionally skips PN prediction export; full-statistics predictions are never overwritten by pilot artifacts.
- **ROOT/cmsstyle errors** — plotting requires the analysis environment (`source setup.sh` from the repo root).

---

## Quick Reference

```bash
# Full pipeline, one signal
python python/runModelComparison.py --signal MHc130_MA90

# All three comparison signals
python python/runModelComparison.py --all

# Refresh PN predictions from existing checkpoints + all plots
python python/runModelComparison.py --signal MHc130_MA90 --pn-only

# Plots only (fast, from cached predictions)
python python/runModelComparison.py --signal MHc130_MA90 --plots-only
```

---

## References

- [Data pipeline](STEP1_PREPROCESSING.md) — dataset creation and validation
- [Lambda sweep](STEP2-DECORRELATION.md) — DisCo decorrelation scan
- [GA optimization](STEP3_HYPERPARAM.md) — hyperparameter search (historical baseline for this comparison)
- [Canonical CE config](../ModelComparison/configs/ParticleNet/default_config.json) / [MD config](../ModelComparison/configs/ParticleNet_MD/default_config.json)
- [`runModelComparison.py`](../python/runModelComparison.py) — driver
