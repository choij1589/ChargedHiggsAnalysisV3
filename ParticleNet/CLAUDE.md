# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ParticleNet is a graph neural network-based machine learning pipeline for charged Higgs boson analysis within ChargedHiggsAnalysisV3. It provides signal/background discrimination for multi-lepton final states in CMS experiments, with advanced class imbalance handling and genetic algorithm hyperparameter optimization.

## Critical Setup

**ALWAYS** source the parent project's setup before any work:
```bash
cd ..  # Navigate to ChargedHiggsAnalysisV3 root
source setup.sh
cd ParticleNet
```

## Key Commands

### Dataset Preparation
```bash
# Full dataset creation (all signals and backgrounds)
./scripts/saveDatasets.sh

# Quick test with pilot datasets
./scripts/saveDatasets.sh --pilot

# Single sample
python3 python/saveDataset.py --sample TTToHcToWAToMuMu-MHc130MA100 --sample-type signal --channel Run1E2Mu
```

### Multi-Class Training

Training parameters are now configured via `configs/SglConfig.json`:

```bash
# Direct Python script usage
python trainMultiClass.py --signal MHc130_MA100 --channel Run1E2Mu
python trainMultiClass.py --signal MHc130_MA100 --channel Run1E2Mu --config configs/SglConfig-custom.json

# Batch training - Format 1: Paired signal:channel configs
./scripts/trainMultiClass.sh --config MHc130_MA100:Run1E2Mu,MHc130_MA90:Run3Mu
./scripts/trainMultiClass.sh --config MHc130_MA100:Run1E2Mu,MHc130_MA90:Run3Mu --param configs/SglConfig-custom.json

# Batch training - Format 2: Single signal, multiple channels
./scripts/trainMultiClass.sh --signal MHc130_MA100 --channels Run1E2Mu,Run3Mu
./scripts/trainMultiClass.sh --signal MHc130_MA100 --channels Run1E2Mu,Run3Mu --param configs/SglConfig-custom.json

# Batch training - Format 3: Single channel, multiple signals (backward compatible)
./scripts/trainMultiClass.sh --channel Run1E2Mu
./scripts/trainMultiClass.sh --channel Run1E2Mu --signals "MHc130_MA90,MHc160_MA85"
./scripts/trainMultiClass.sh --channel Run1E2Mu --signals "MHc130_MA90,MHc160_MA85" --param configs/SglConfig-custom.json
```

**Configuration**: All training parameters (model, optimizer, learning rate, backgrounds, folds, etc.) are specified in `configs/SglConfig.json`. To customize:
```bash
cp configs/SglConfig.json configs/SglConfig-custom.json
# Edit configs/SglConfig-custom.json to change:
#   - Model architecture (model_config.default_model, model_config.nNodes)
#   - Optimization (optimization_config.optimizer, initLR, scheduler)
#   - Background grouping (background_config.mode, background_groups)
#   - Training parameters (training_parameters.max_epochs, batch_size, loss_type)
#   - Fold configuration (training_parameters.train_folds, valid_folds, test_folds)
```

**Note**: The shell script uses `--param` for the parameter JSON file to avoid conflict with `--config` (used for signal:channel pairs in Format 1). The Python script still uses `--config` for the parameter file.

### Hyperparameter Optimization (GA)
```bash
# Standard GA optimization
./scripts/launchGAOptim.sh MHc130_MA100 Run1E2Mu cuda:0

# With pilot data for testing
./scripts/launchGAOptim.sh MHc130_MA100 Run1E2Mu cuda:0 --pilot

# Batch optimization for multiple mass points
./doThis.sh
```

### Visualization
```bash
# Visualize specific model results
./scripts/visualizeResults.sh Run1E2Mu TTToHcToWAToMuMu-MHc130_MA90 0

# Pattern matching for grouped models
./scripts/visualizeResults.sh Run1E2Mu TTToHcToWAToMuMu-MHc130_MA90 0 --model-pattern '*3grp*'
```

## Architecture and Data Flow

### Workflow Pipeline
1. **saveDatasets.sh**: ROOT → PyTorch Geometric datasets with physics event weights
2. **trainMultiClass.py**: N-class training with class balancing
3. **visualizeMultiClass.py**: Performance metrics and ROC curves
4. **launchGAOptim.sh**: Genetic algorithm hyperparameter search
5. **evalModels.sh**: Cross-validation with best hyperparameters
6. **toSKFlat.sh**: Deploy to analysis framework

### Directory Structure
```
ParticleNet/
├── configs/
│   ├── GAConfig.json          # GA optimization configuration
│   └── SglConfig.json         # Single-run training configuration
├── dataset/
│   └── samples/               # PyTorch datasets (auto-generated)
│       ├── signals/
│       └── backgrounds/
├── GAOptim_bjets/             # GA optimization results
│   └── {channel}/multiclass/{signal}/GA-iter{N}/
├── models/                    # Trained model checkpoints
├── plots/                     # Visualization outputs
├── python/                    # Core implementation
└── scripts/                   # Shell wrappers with GNU parallel
```

### Key Configuration Files

#### SglConfig.json - Single-Run Training
Controls trainMultiClass.py behavior:
- **training_parameters**: max_epochs, batch_size, dropout_p, loss_type, train_folds, valid_folds, test_folds
- **model_config**: default_model (ParticleNet variants), nNodes
- **optimization_config**: optimizer, initLR, weight_decay, scheduler
- **background_config**: mode (groups/individual), background_groups, backgrounds_list
- **dataset_config**: use_bjets, signal_prefix, background_prefix
- **system_config**: device, pilot mode, debug mode

**Fold Configuration**: Use train_folds=[0,1,2], valid_folds=[3], test_folds=[4] for standard setup. To perform 5-fold cross-validation, create 5 different config files with different fold splits.

#### GAConfig.json - Hyperparameter Optimization
Controls launchGAOptim.sh behavior:
- **GA parameters**: population_size=20, max_iterations=4, evolution_ratio=0.7
- **Hyperparameter space**: nNodes, optimizers, schedulers, initLR, weight_decay
- **Training config**: max_epochs=10, batch_size=1024, fold splits for GA
- **Background groups**: Physics-motivated sample grouping
- **Overfitting detection**: Kolmogorov-Smirnov tests with p-value threshold

## Class Imbalance Handling

Three-layer approach to handle 4-5 order magnitude differences in cross-sections:

1. **Dataset Normalization**: Equal total weight per class during training
2. **Physics Weights**: Per-event genWeight × puWeight × prefireWeight
3. **Balanced Metrics**: Equal contribution from each class in accuracy

### Loss Functions
- `weighted_ce`: Standard cross-entropy with event weights
- `sample_normalized`: Additional per-class normalization
- `focal`: Emphasizes hard examples with physics weights

## ParticleNet Score Integration

### Cross-Section Weighted Score
The ParticleNet score used in SignalRegionStudy is a Bayesian likelihood ratio:
```
score_PN = s₀ / (s₀ + w₁×s₁ + w₂×s₂ + w₃×s₃)
```
Where:
- s₀: signal score from network
- s₁, s₂, s₃: background scores (nonprompt, diboson, ttX)
- w₁, w₂, w₃: background weights from true cross-sections

This corrects for training with balanced classes vs real-world imbalanced rates.

## Development Patterns

### Python Scripts
- Always add `export PATH="${PWD}/python:${PATH}"` in shell scripts
- Use `ROOT.gROOT.SetBatch(True)` for non-interactive mode
- Handle ROOT memory: call `SetDirectory(0)` after reading histograms
- Check for pilot mode with `--pilot` flag for quick testing

### Parallel Processing
- GNU parallel used extensively for multi-core execution
- Fold-level parallelism in training scripts
- Process-level parallelism in GA optimization
- Add delays between parallel launches to avoid resource conflicts

### Error Handling
- No silent fallbacks - raise errors on unexpected behavior
- Validate file existence before processing
- Check GPU availability when device specified
- Verify dataset completeness before training

## Signal/Background Samples

**Signals** (7 mass points):
- TTToHcToWAToMuMu-MHc[100,115,130,145,160]_MA[85,87,90,92,95,98,100]

**Backgrounds**:
- **nonprompt**: Skim_TriLep_TTLL_powheg, DYJets
- **diboson**: WZTo3LNu_amcatnlo, ZZTo4L_powheg
- **ttX**: TTZToLLNuNu, TTWToLNu, tZq

**Channels**: Run1E2Mu (1e+2μ), Run3Mu (3μ), Combined

## Common Issues and Solutions

### Dataset Not Found
```bash
# Verify dataset exists
ls dataset/samples/signals/TTToHcToWAToMuMu-MHc130MA100/Run1E2Mu_fold-0.pt

# If missing, recreate
./scripts/saveDatasets.sh --pilot  # for testing
```

### CUDA Out of Memory
Reduce batch size in training command or GAConfig.json

### Overfitting Detection Failed
Check `GAOptim_bjets/{channel}/multiclass/{signal}/GA-iter{N}/overfitting_diagnostics/`
- Review per-model K-S test results
- Adjust p_value_threshold in GAConfig.json if needed

### Score Distribution Issues
```bash
# Visualize scores to debug
python python/visualizeMultiClass.py --model-dir models/{your_model} --plot-scores
```