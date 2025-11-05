# ParticleNet
---
## Introduction
ParticleNet-based machine learning pipeline for charged Higgs boson analysis. The workflow consists of 6 main steps to train graph neural networks for discriminating signal from background events, with advanced class imbalance handling for particle physics.

## Workflow Overview
1. **Dataset Preparation** (`saveDatasets.sh`) - Convert ROOT files to PyTorch Geometric datasets
2. **Multi-Class Training** (`trainMultiClass.py`) - N-class training with physics-aware weighting
3. **Visualization** (`visualizeMultiClass.py`) - Analysis plots and performance metrics
4. **Hyperparameter Optimization** (`launchGAOptim.sh`) - Genetic Algorithm optimization with shared memory for efficient parallel training
5. **Model Evaluation** (`evalModels.sh`) - Cross-validation training
6. **SKFlat Integration** (`toSKFlat.sh`) - Deploy models to analysis framework

---

## Step 1: Dataset Preparation (`saveDatasets.sh`)

Converts ROOT files into PyTorch Geometric datasets with per-process storage.

### Configuration
**Signal Samples** (7 mass points): TTToHcToWAToMuMu-MHc[100,115,130,145,160]MA[85,87,90,92,95,98,100]

**Background Samples**: Skim_TriLep_TTLL_powheg (nonprompt), DYJets (nonprompt), WZTo3LNu_amcatnlo + ZZTo4L_powheg (diboson), TTZToLLNuNu + TTWToLNu + tZq (ttX)

**Channels**: Run1E2Mu (1e+2μ), Run3Mu (3μ), Combined

**Features**: 9 per particle [E,Px,Py,Pz,Charge,IsMuon,IsElectron,IsJet,IsBjet] + 4 graph-level (era encoding) + event weights (genWeight×puWeight×prefireWeight)

### Usage
```bash
# Create all datasets
./scripts/saveDatasets.sh

# Create pilot datasets for testing
./scripts/saveDatasets.sh --pilot

# Individual sample creation
python3 python/saveDataset.py --sample TTToHcToWAToMuMu-MHc130MA100 --sample-type signal --channel Run1E2Mu
```

### Output Structure
```
dataset/samples/
├── signals/TTToHcToWAToMuMu-MHc*/Run1E2Mu_fold-*.pt
└── backgrounds/Skim_TriLep_*/Run1E2Mu_fold-*.pt
```

---

## Step 2: Multi-Class Training (`trainMultiClass.py`)

Modular training system with sophisticated class imbalance handling. See script docstrings for detailed architecture documentation.

### Class Imbalance Handling

**Three-Layer Strategy**:
1. **Dataset Normalization**: Between-group weight balancing (preserves within-group cross-sections)
2. **Physics-Aware Loss**: Event weights (genWeight×puWeight×prefireWeight) with weighted/focal loss
3. **Group-Balanced Metrics**: Equal class contribution regardless of sample count

**Background Training Modes**:
- **Individual**: Each sample as separate class (e.g., signal + TTLL + WZ + TTZ)
- **Grouped**: Physics-motivated categories (e.g., signal + nonprompt + diboson + ttX)

### Key Features
- 5-fold cross-validation (3 train / 1 valid / 1 test)
- Automatic weight normalization for class balance
- Early stopping with validation monitoring
- Real-time performance monitoring (memory, CPU, time)
- ROOT tree outputs with per-class score distributions

### Usage Examples

**Grouped Backgrounds (Recommended)**:
```bash
python trainMultiClass.py \
  --signal MHc130_MA100 --channel Run1E2Mu --fold 0 \
  --background_groups \
    "nonprompt:TTLL_powheg" \
    "diboson:WZTo3LNu_amcatnlo,ZZTo4L_powheg" \
    "ttX:TTZToLLNuNu,TTWToLNu,tZq" \
  --model OptimizedParticleNet --loss_type weighted_ce
```

**Individual Backgrounds**:
```bash
python trainMultiClass.py \
  --signal MHc130_MA100 --channel Run1E2Mu --fold 0 \
  --backgrounds TTLL_powheg WZTo3LNu_amcatnlo TTZToLLNuNu \
  --model ParticleNet --loss_type weighted_ce
```

**Batch Training**:
```bash
./scripts/trainMultiClass.sh --channel Run1E2Mu \
  --backgrounds "TTLL_powheg WZTo3LNu_amcatnlo TTZToLLNuNu" \
  --model ParticleNet
```

### Command Line Options
- `--signal`: Signal sample (without prefix, required)
- `--backgrounds`: Individual background samples (space-separated)
- `--background_groups`: Groups format 'groupname:sample1,sample2,...'
- `--model`: Architecture variant (ParticleNet, OptimizedParticleNet, etc.)
- `--loss_type`: weighted_ce (default), sample_normalized, focal
- `--balance`: Enable weight normalization (default: True)
- `--pilot`: Use pilot datasets for testing

---

## Step 3: Visualization (`visualizeMultiClass.py`)

Comprehensive visualization supporting both individual and grouped backgrounds.

### Features
- Multi-class ROC curves (one-vs-rest)
- Confusion matrices (normalized + raw)
- Score distributions per class
- Performance metrics (accuracy, precision, recall, F1)
- Automatic model type detection

### Usage
```bash
# Visualize specific model
./scripts/visualizeResults.sh Run1E2Mu TTToHcToWAToMuMu-MHc130_MA90 0 --pilot

# Pattern matching for grouped models
./scripts/visualizeResults.sh Run1E2Mu TTToHcToWAToMuMu-MHc130_MA90 0 --model-pattern '*3grp*'
```

### Output Structure
```
plots/analysis/
├── individual_backgrounds/standard/<CHANNEL>/<SIGNAL>/fold-<N>/
└── grouped_backgrounds/3groups/<CHANNEL>/<SIGNAL>/fold-<N>/
    ├── roc_curves.png
    ├── confusion_matrix.png
    ├── score_distributions.png
    └── model_info.txt
```

---

## Step 4: Hyperparameter Optimization (`launchGAOptim.sh`)

Genetic Algorithm (GA) optimization with statistical overfitting detection and filtering. All configuration managed via `configs/GAConfig.json`. See script docstrings for implementation details.

### Overview

**Key Features**:
- Configuration-driven approach (no hardcoded parameters)
- Log-uniform sampling for continuous hyperparameters
- Per-class overfitting detection using Kolmogorov-Smirnov tests
- Automatic filtering and population regeneration
- Parallel training and evaluation

**Hyperparameter Search**:
- **nNodes**: [64, 96, 128] hidden nodes
- **Optimizers**: [RMSprop, Adam, Adadelta]
- **Schedulers**: [ExponentialLR, CyclicLR, ReduceLROnPlateau]
- **initLR**: Log-uniform [1e-4, 1e-2] (100 samples)
- **weight_decay**: Log-uniform [1e-5, 1e-2] (1000 samples)

### Configuration Reference

Edit `configs/GAConfig.json` to adjust:
- **GA parameters**: population_size, max_iterations, evolution_ratio, mutation_thresholds
- **Training parameters**: max_epochs, batch_size, dropout_p, train/valid/test folds
- **Background groups**: Physics-motivated sample groupings
- **Overfitting detection**: Enable/disable, p-value threshold, test folds

### Overfitting Detection

**Strategy**: Independent test set (fold 3) with per-class K-S tests comparing train vs test score distributions. Models fail if ANY class has p < 0.05. Filtered models are immediately replaced to maintain population size.

**Diagnostics**: Detailed per-model histograms, p-values, and summary reports saved to `GAOptim_bjets/{channel}/multiclass/{signal}/GA-iter{N}/overfitting_diagnostics/`

### Usage

```bash
# Basic command
./scripts/launchGAOptim.sh MHc130_MA100 Run1E2Mu cuda:0

# With pilot data (recommended for testing)
./scripts/launchGAOptim.sh MHc130_MA100 Run1E2Mu cuda:0 --pilot

# Debug mode
./scripts/launchGAOptim.sh MHc130_MA100 Run1E2Mu cuda:0 --pilot --debug
```

### Output Structure

```
GAOptim_bjets/{channel}/multiclass/{signal}/
├── GA-iter0/                              # Generation 0
│   ├── models/model{N}.pt                 # Population checkpoints
│   ├── json/model{N}.json                 # Training metrics
│   ├── json/model_info.csv                # Population summary
│   └── overfitting_diagnostics/
│       ├── model{N}_iter0_histograms.root
│       ├── model{N}_iter0_ks_results.json
│       ├── overfitting_summary.json
│       └── overfitting_summary.txt
├── GA-iter1/                              # Generation 1 (evolved)
└── GA-iter2/                              # Generation 2
```

### Module List
- **`GAConfig.py`**: Configuration loading and validation
- **`GATools.py`**: Genetic algorithm operations (selection, crossover, mutation)
- **`launchGAOptim.py`**: Main optimization workflow
- **`trainWorker.py`**: Worker function for parallel training
- **`SharedDatasetManager.py`**: Shared memory dataset management
- **`evaluateGAModels.py`**: Batch overfitting evaluation
- **`OverfittingDetector.py`**: K-S test implementation

### Shared Memory Implementation

The GA optimization uses PyTorch's shared memory to enable efficient parallel training:

**How It Works**:
1. Dataset loaded once and shared via `.share_memory_()` to `/dev/shm`
2. Worker processes created with `mp.spawn()` for CUDA-safe operation
3. Workers access shared memory with zero-copy reads

**Performance Benefits**:
- **Memory**: ~10-20GB total (vs 200-400GB for independent loading)
- **Startup**: Significantly faster dataset initialization
- **Scalability**: Enables larger population sizes

**Technical Notes**:
- Shared tensors are read-only (workers must not modify)
- Requires Linux shared memory (`/dev/shm`)
- Uses `spawn` multiprocessing for clean CUDA contexts

---

## Step 5: Model Evaluation (`evalModels.sh`)

Cross-validation training using optimized hyperparameters.

**Features**: 5-fold parallel training, performance metrics (ROC, AUC), ROOT prediction outputs

```bash
./scripts/evalModels.sh MHc-130_MA-100 nonprompt Combined
```

---

## Step 6: SKFlat Integration (`toSKFlat.sh`)

Deploy trained models to SKFlat analysis framework.

```bash
./scripts/toSKFlat.sh Combined
```

**Deployment**: `SKFlatAnalyzer/data/Run2UltraLegacy_v3/Classifiers/ParticleNet/`

---

## Technical Details

### Model Architecture
- **Base**: ParticleNet graph neural network
- **Input**: 9 node + 4 graph features, k=4 nearest neighbors in ΔR
- **Output**: Configurable N-class (signal + N-1 backgrounds)
- **Cross-validation**: 5-fold deterministic splitting

### Class Imbalance Strategy

**Problem**: Signal/background cross-sections differ by 4-5 orders of magnitude

**Solution**: Multi-layered approach
1. **Weight Normalization**: `max_weight / sample_total_weight` per class
2. **Physics Weights**: Per-event `genWeight × puWeight × prefireWeight`
3. **Balanced Accuracy**: Within-class weighted, between-class equal contribution

**Loss Functions**:
- `weighted_ce`: Standard with event weights
- `sample_normalized`: Additional per-class normalization
- `focal`: Hard example focus with physics weights

### Requirements
- PyTorch + PyTorch Geometric
- ROOT + scikit-learn
- GNU Parallel
- CUDA recommended

### Current Status
✅ **Complete**: Dataset preparation, multi-class training, visualization, hyperparameter optimization with shared memory
🚧 **In Progress**: Visualizing hyperparameter optimization results, results parsing to SKFlat
