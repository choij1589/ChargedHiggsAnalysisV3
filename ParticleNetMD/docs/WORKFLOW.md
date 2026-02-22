# ParticleNetMD: Mass-Decorrelated ParticleNet Workflow

## Overview

ParticleNetMD implements mass-decorrelated training for the charged Higgs analysis, preventing the neural network from learning the di-muon mass peak directly. This ensures the classifier doesn't sculpt the mass distribution and create artificial bumps in background estimates.

### Key Features

1. **DisCo Loss (Default)** - Distance Correlation regularization to decorrelate classifier output from mass
2. **Mass Storage** - Store both OS pair masses (mass1, mass2) in graph data
3. **Separate B-jets (Default)** - Use b-jet as separate particle type (no btagScore feature)

### Physics Context

For **3-muon events** (Run3Mu channel):
- 3 muons create 2 opposite-sign (OS) pairs
- Only one pair is from A -> mu mu (signal resonance)
- The other pair is combinatorial (one muon from A, one from W)
- We decorrelate from BOTH masses to prevent any mass-related learning

For **1e2mu events** (Run1E2Mu channel):
- 2 muons create 1 OS pair
- mass1 = di-muon mass, mass2 = -1 (ignored in DisCo loss)

---

## Quick Start

### 1. Create Datasets
```bash
cd /pscratch/sd/c/choij/workspace/ChargedHiggsAnalysisV3/ParticleNetMD

# Signal
python python/saveDataset.py --sample TTToHcToWAToMuMu-MHc130_MA90 --sample-type signal --channel Run3Mu

# Backgrounds
python python/saveDataset.py --sample Skim_TriLep_TTLL_powheg --sample-type background --channel Run3Mu
python python/saveDataset.py --sample Skim_TriLep_WZTo3LNu_amcatnlo --sample-type background --channel Run3Mu
python python/saveDataset.py --sample Skim_TriLep_ZZTo4L_powheg --sample-type background --channel Run3Mu
python python/saveDataset.py --sample Skim_TriLep_TTZToLLNuNu --sample-type background --channel Run3Mu
python python/saveDataset.py --sample Skim_TriLep_tZq --sample-type background --channel Run3Mu
```

### 2. Run Pilot Training
```bash
python python/trainMultiClass.py --signal MHc130_MA90 --channel Run3Mu
```

### 3. Full Dataset Creation (All Signals/Backgrounds)
```bash
./scripts/saveDatasets.sh
```

---

## Directory Structure

```
ParticleNetMD/
├── configs/
│   └── SglConfig.json              # Training config with DisCo params
├── dataset/
│   └── samples/
│       ├── signals/                # Signal samples with mass info
│       └── backgrounds/            # Background samples with mass info
├── DataAugment/
│   └── diboson/                    # Diboson GNN reweighting outputs
│       ├── dataset/                # Intermediate PyG datasets + kinematics JSON
│       ├── models/                 # Trained reweighting model checkpoint
│       └── plots/                  # Validation plots
├── python/
│   ├── DataFormat.py               # Particle classes (from ParticleNet)
│   ├── Preprocess.py               # [Modified] store mass1, mass2; 8-dim era encoding
│   ├── saveDataset.py              # [Modified] compute OS pair masses
│   ├── trainDibosonReweight.py     # Diboson augmentation via GNN reweighting
│   ├── DynamicDatasetLoader.py     # Dataset loading with mass attributes
│   ├── DataPipeline.py             # [Modified] ParticleNetMD paths
│   ├── WeightedLoss.py             # [Modified] DiScoWeightedLoss
│   ├── MultiClassModels.py         # ParticleNet GNN architecture
│   ├── MLTools.py                  # ML utilities
│   ├── TrainingUtilities.py        # [Modified] pass mass to loss
│   ├── TrainingOrchestrator.py     # [Modified] DisCo configuration
│   ├── trainMultiClass.py          # [Modified] DisCo as default
│   ├── SglConfig.py                # JSON config loader
│   ├── ResultPersistence.py        # Model saving utilities
│   └── ROCCurveCalculator.py       # ROC curve utilities
├── scripts/
│   └── saveDatasets.sh             # Dataset creation script
├── results/                        # Training outputs
└── docs/
    ├── DATASET.md                  # Dataset statistics and augmentation docs
    └── WORKFLOW.md                 # This file
```

---

## Node Features (9 dimensions)

```
[E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]
```

- No btagScore - b-jets are identified as separate particle type
- Particles: muons, electrons, jets, b-jets, MET

## Era Encoding (8 dimensions)

Graph-level input is an 8-dim one-hot vector covering all Run2 + Run3 eras:

```
[2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix]
```

Defined in `Preprocess.py` via `ERA_INDEX` dict. All downstream files use `num_graph_features=8`.

---

## DisCo Loss

The Distance Correlation (DisCo) loss decorrelates classifier output from mass:

```
L_total = L_CE + lambda * (DisCo(score, mass1) + DisCo(score, mass2))
```

Where:
- `L_CE`: Weighted cross-entropy loss
- `DisCo(X, Y)`: Distance correlation between X and Y (0 = independent, 1 = fully dependent)
- `lambda`: DisCo penalty weight (default: 0.1)
- `score`: Signal class probability (probs[:, 0])
- `mass1, mass2`: OS muon pair masses (mass2 = -1 for 1e2mu events, ignored)

### Event-Weighted DisCo

Both the CE loss and DisCo terms use physics event weights:
```
weight = genWeight * puWeight * prefireWeight
```

---

## Configuration (SglConfig.json)

```json
{
  "training_parameters": {
    "train_folds": [0, 1, 2],
    "valid_folds": [3],
    "test_folds": [4],
    "max_epochs": 100,
    "batch_size": 256,
    "loss_type": "disco",
    "balance_weights": true,
    "max_events_per_fold_per_class": 50000
  },
  "disco_parameters": {
    "disco_lambda": 0.1
  },
  "background_config": {
    "mode": "groups",
    "background_groups": {
      "nonprompt": ["TTLL_powheg"],
      "diboson": ["WZTo3LNu_amcatnlo", "ZZTo4L_powheg"],
      "ttX": ["TTZToLLNuNu", "tZq"]
    }
  }
}
```

---

## Diboson Augmentation

Diboson is the most statistics-limited class (~72k Tight+Bjet events, only ~8.4k from Run3). The b-jet requirement rejects ~93% of diboson events. To augment, we reweight non-b-tagged diboson events using a shallow GNN classifier (`trainDibosonReweight.py`).

See `docs/DATASET.md` for full details on:
- Per-class event statistics and imbalance
- The GNN reweighting strategy (b-jet promotion → density ratio learning → reweighting)
- Validation methodology

```bash
# Three-step workflow
python python/trainDibosonReweight.py --prepare-dataset   # ROOT → PyG graphs
python python/trainDibosonReweight.py --train              # Train ShallowGraphNet
python python/trainDibosonReweight.py --validation         # Validation plots
```

Outputs saved to `DataAugment/diboson/`.

---

## Implementation Status

### Phase 1: Data Preparation [COMPLETED]
- [x] Copy base files from ParticleNet
- [x] Modify `Preprocess.py` - add `compute_os_pair_masses()`, store mass1/mass2 in Data
- [x] Modify `saveDataset.py` - compute OS pair masses
- [x] Create `saveDatasets.sh` script

### Phase 2: DisCo Implementation [COMPLETED]
- [x] Implement `distance_correlation()` function in WeightedLoss.py
- [x] Implement `DiScoWeightedLoss` class
- [x] Update `create_loss_function()` factory

### Phase 3: Training Infrastructure [COMPLETED]
- [x] Modify `TrainingUtilities.py` - add `loss_requires_mass` parameter
- [x] Modify `TrainingOrchestrator.py` - DisCo configuration handling
- [x] Modify `trainMultiClass.py` - DisCo as default loss type

### Phase 4: Configuration [COMPLETED]
- [x] Create `SglConfig.json` with DisCo parameters
- [x] Update `DataPipeline.py` for ParticleNetMD paths

### Phase 5: Output Format & DisCo Tracking [COMPLETED]
- [x] Add `save_ga_compatible_json()` to ResultPersistence.py for visualizeGAIteration.py compatibility
- [x] Track timestamps in training history
- [x] Return decomposed loss (CE + DisCo) from WeightedLoss.py
- [x] Accumulate CE/DisCo losses separately in TrainingUtilities.py
- [x] Log decomposed losses in TrainingOrchestrator.py (train_ce_loss, train_disco_term, etc.)
- [x] Add mass1, mass2 branches to ROOT tree output

### Phase 6: Testing [IN PROGRESS]
- [x] Create pilot datasets
- [ ] Run pilot training to validate implementation
- [ ] Verify DisCo loss computation
- [ ] Verify GA-compatible JSON output

### Phase 7: GA Optimization [NOT STARTED]
- [ ] Create GAConfig.json with disco_lambda in search space
- [ ] Modify trainWorker.py for DisCo support
- [ ] Create launchGAOptim.sh

### Phase 8: Visualization [IN PROGRESS]
- [x] GA-compatible JSON output for visualizeGAIteration.py
- [ ] Create visualizeMultiClass.py with mass decorrelation plots
- [ ] Add score vs mass 2D plots
- [ ] Add mass profile vs score plots
- [ ] Add DisCo evolution plots

---

## Expected Results

### Good Decorrelation (disco_lambda ~ 0.1-0.5)
- DisCo(score, mass) < 0.1
- Flat score profile vs mass
- Slight decrease in raw classification accuracy (trade-off)

### Poor Decorrelation (disco_lambda too small)
- DisCo(score, mass) > 0.3
- Score peaks at signal mass region
- Higher raw accuracy but mass sculpting

### Over-decorrelation (disco_lambda too large)
- DisCo(score, mass) ~ 0
- Very flat score profile
- Significant accuracy loss, model learns nothing useful

---

## Files Modified from ParticleNet

| File | Modification |
|------|--------------|
| `Preprocess.py` | Added `compute_os_pair_masses()`, store mass1/mass2 in Data |
| `saveDataset.py` | Compute OS pair masses |
| `WeightedLoss.py` | Added `distance_correlation()`, `DiScoWeightedLoss` with decomposed loss tracking |
| `TrainingUtilities.py` | Added `loss_requires_mass`, returns decomposed CE/DisCo losses |
| `TrainingOrchestrator.py` | DisCo configuration, timestamps, decomposed loss logging |
| `trainMultiClass.py` | DisCo as default, calls `save_ga_compatible_json()` |
| `DataPipeline.py` | Changed dataset path to ParticleNetMD |
| `ResultPersistence.py` | Added `save_ga_compatible_json()`, mass1/mass2 ROOT branches |

---

## Output Format

### GA-Compatible JSON (for visualizeGAIteration.py)
Training produces `{model_name}.json` with:
```json
{
  "hyperparameters": {
    "signal": "MHc130_MA90",
    "channel": "Run3Mu",
    "num_hidden": 128,
    "optimizer": "Adam",
    "initial_lr": 0.001,
    "scheduler": "ReduceLROnPlateau",
    "loss_type": "disco",
    "disco_lambda": 0.1,
    ...
  },
  "training_summary": {
    "best_epoch": 45,
    "best_train_loss": 0.85,
    "best_valid_loss": 0.87,
    "best_train_acc": 0.65,
    "best_valid_acc": 0.63,
    "total_epochs": 100
  },
  "epoch_history": {
    "train_loss": [...],
    "valid_loss": [...],
    "train_acc": [...],
    "valid_acc": [...],
    "train_ce_loss": [...],      // DisCo only
    "train_disco_term": [...],   // DisCo only
    "valid_ce_loss": [...],      // DisCo only
    "valid_disco_term": [...],   // DisCo only
    "epoch": [...],
    "timestamp": [...]
  }
}
```

### ROOT Tree Branches
Output tree (`trees/{model_name}.root`) contains:
- `score_signal`, `score_nonprompt`, `score_diboson`, `score_ttX` - class probabilities
- `true_label` - ground truth (0=signal, 1=nonprompt, 2=diboson, 3=ttX)
- `train_mask`, `valid_mask`, `test_mask` - data split flags
- `has_bjet` - whether event contains b-jets (for analysis, not used in training)
- `weight` - physics event weight
- `mass1`, `mass2` - OS muon pair masses (for decorrelation analysis)

---

## Training Command Reference

### Single Model Training
```bash
# With default config (DisCo loss, lambda=0.1)
python python/trainMultiClass.py --signal MHc130_MA90 --channel Run3Mu

# With custom config
python python/trainMultiClass.py --signal MHc130_MA90 --channel Run3Mu --config configs/custom.json
```

### Available Loss Types
- `disco` (default): DisCo-regularized weighted cross-entropy
- `weighted_ce`: Standard weighted cross-entropy
- `focal`: Weighted focal loss
- `sample_normalized`: Per-class normalized weighted loss
