# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Module Overview

ParticleNetMD implements **mass-decorrelated** training for the ParticleNet GNN classifier used in the charged Higgs analysis. It is a specialized variant of `ParticleNet/` with a key innovation: DisCo (Distance Correlation) loss regularization that prevents the network from learning the OS di-muon pair mass distribution, avoiding artificial mass bumps in background predictions.

**Purpose:** Train a 4-class classifier (signal vs nonprompt/diboson/ttX backgrounds) that is explicitly decorrelated from the OS muon pair mass.

**Key difference from ParticleNet:** Uses DisCo loss (`L_total = L_CE + λ·DisCo(score, mass1) + λ·DisCo(score, mass2)`) instead of standard cross-entropy alone.

For comprehensive implementation documentation, see also `WORKFLOW.md` in this directory.

## Environment Setup

```bash
cd ..  # Navigate to ChargedHiggsAnalysisV3 root
source setup.sh
cd ParticleNetMD
```

## Key Commands

### Dataset Preparation
```bash
# Create datasets for all signals and backgrounds
./scripts/saveDatasets.sh

# Quick test with pilot datasets
./scripts/saveDatasets.sh --pilot

# Single sample
python3 python/saveDataset.py --sample TTToHcToWAToMuMu-MHc130MA100 --sample-type signal --channel Run1E2Mu
```

### Model Training
```bash
# Standard training with DisCo loss (default)
python python/trainMultiClass.py --signal MHc130_MA100 --channel Run1E2Mu

# Custom configuration
python python/trainMultiClass.py --signal MHc130_MA100 --channel Run1E2Mu --config configs/SglConfig-custom.json

# Combined channel (merges Run1E2Mu + Run3Mu datasets dynamically)
python python/trainMultiClass.py --signal MHc130_MA100 --channel Combined
```

## Directory Structure

```
ParticleNetMD/
├── configs/
│   └── SglConfig.json             # Training configuration (DisCo params, folds, model, etc.)
├── dataset/
│   └── samples/
│       ├── signals/               # PyTorch Geometric datasets per signal × channel × fold
│       └── backgrounds/           # PyTorch Geometric datasets per background × channel × fold
├── python/
│   ├── trainMultiClass.py         # Main training script
│   ├── TrainingOrchestrator.py    # Training loop with DisCo loss integration
│   ├── MultiClassModels.py        # 4-class ParticleNet GNN (3 EdgeConv layers)
│   ├── WeightedLoss.py            # DisCo + weighted cross-entropy loss functions
│   ├── DataPipeline.py            # Dataset loading, DataLoader configuration
│   ├── DynamicDatasetLoader.py    # On-the-fly loading for Combined channel
│   ├── Preprocess.py              # Extracts mass1, mass2 for decorrelation
│   ├── saveDataset.py             # ROOT → PyTorch Geometric conversion
│   ├── SglConfig.py               # JSON config loader
│   ├── ResultPersistence.py       # Model + GA-compatible JSON output
│   ├── ROCCurveCalculator.py      # ROC curves and performance metrics
│   ├── visualizeMultiClass.py     # Visualization utilities
│   └── (other shared utilities)
├── scripts/
│   └── saveDatasets.sh            # Bulk dataset creation
├── logs/                          # Dataset statistics JSON files from runs
├── models/                        # Trained model checkpoints
├── results/                       # Training outputs
├── plots/                         # Visualization outputs
├── BackUps/                       # Previous versions
└── WORKFLOW.md                    # Comprehensive implementation guide (305 lines)
```

## Configuration: configs/SglConfig.json

All training parameters are in `SglConfig.json`. Key sections:

```json
{
  "training_parameters": {
    "train_folds": [0,1,2], "valid_folds": [3], "test_folds": [4],
    "max_epochs": 50, "batch_size": 1024, "dropout_p": 0.1,
    "loss_type": "disco",
    "early_stopping_patience": 10
  },
  "disco_config": {
    "lambda_mass": 0.1,      // DisCo weight for OS mass decorrelation
    "lambda_bjet": 0.5       // DisCo weight for b-jet decorrelation
  },
  "model_config": {
    "default_model": "ParticleNet",
    "nNodes": 64
  },
  "optimization_config": {
    "optimizer": "Adam", "initLR": 0.001,
    "weight_decay": 0.0001, "scheduler": "StepLR"
  },
  "background_config": {
    "mode": "groups",
    "background_groups": {
      "nonprompt": [...], "diboson": [...], "ttX": [...]
    }
  }
}
```

To customize: `cp configs/SglConfig.json configs/SglConfig-custom.json` and edit.

## DisCo Loss

The core innovation of ParticleNetMD. Event-weighted distance correlation between classifier score and OS muon pair masses:

```
L_total = L_CE + λ_mass·DisCo(score, mass1) + λ_mass·DisCo(score, mass2) + λ_bjet·DisCo(score, bjet_info)
```

- **mass1, mass2:** OS muon pair masses extracted during dataset creation in `Preprocess.py`
- **DisCo = 0** when score is statistically independent of mass (desired)
- **λ** values control strength of decorrelation (tunable in SglConfig.json)

## Node Features

9 node features per particle: `[E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]`

## Model Architecture

4-class ParticleNet GNN (in `MultiClassModels.py`):
- 3 DynamicEdgeConv layers with k=4 nearest neighbors
- Graph normalization + residual connections + dropout
- Global mean pooling → dense classification head → 4-class softmax

## Dataset Creation

`saveDataset.py` converts ROOT files to PyTorch Geometric format:
- Extracts 9 node features + mass1, mass2 for decorrelation
- Assigns 5-fold train/validation/test split
- Applies event weights (genWeight × puWeight × prefireWeight)
- Special handling: diboson samples use relaxed b-jet requirement
- Per-era processing (all 8 eras: Run2 + Run3)

## Training Output

`ResultPersistence.py` produces:
- Model checkpoints (PyTorch `.pt` files)
- GA-compatible JSON with epoch-by-epoch loss/accuracy history and timestamps
- ROOT trees with class probabilities, true labels, and mass info for decorrelation analysis

## Files Modified from ParticleNet

| File | Modification |
|---|---|
| `Preprocess.py` | Added mass1, mass2 extraction |
| `saveDataset.py` | OS pair mass computation, diboson handling |
| `WeightedLoss.py` | DisCo loss implementation |
| `TrainingOrchestrator.py` | DisCo configuration, decomposed loss logging |
| `trainMultiClass.py` | DisCo as default loss type |
| `DataPipeline.py` | ParticleNetMD dataset paths |
| `ResultPersistence.py` | GA-compatible JSON output |

## Signal/Background Samples

**Signals:** TTToHcToWAToMuMu-MHc[100,115,130,145,160]_MA[85,87,90,92,95,98,100]

**Backgrounds:**
- **nonprompt:** Skim_TriLep_TTLL_powheg, DYJets
- **diboson:** WZTo3LNu_amcatnlo, ZZTo4L_powheg
- **ttX:** TTZToLLNuNu, TTWToLNu, tZq

**Channels:** Run1E2Mu (1e+2μ), Run3Mu (3μ), Combined (dynamic merge)

## Common Issues

**DisCo lambda too high:** Excessive decorrelation degrades signal/background discrimination. Start with λ_mass=0.1 and increase gradually.

**Combined channel dataset not found:** The `DynamicDatasetLoader.py` loads Run1E2Mu + Run3Mu on-the-fly for the Combined channel. Both must exist.

**Dataset not found:**
```bash
ls dataset/samples/signals/TTToHcToWAToMuMu-MHc130MA100/Run1E2Mu_fold-0.pt
./scripts/saveDatasets.sh --pilot  # Recreate
```

**GA-compatible JSON format:** The output JSON follows a specific schema for use with genetic algorithm hyperparameter optimization. See `ResultPersistence.py` for format details.

**CUDA out of memory:** Reduce `batch_size` in SglConfig.json.
