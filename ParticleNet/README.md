# ParticleNet
---
## Introduction
ParticleNet-based machine learning pipeline for charged Higgs boson analysis. The workflow consists of 6 main steps to train graph neural networks for discriminating signal from background events, with advanced class imbalance handling for particle physics.

## Workflow Overview
1. **Dataset Preparation** (`saveDatasets.sh`) - Convert ROOT files to PyTorch Geometric datasets
2. **Multi-Class Training** (`trainMultiClass.py`) - N-class training with physics-aware weighting and class imbalance handling
3. **Visualization** (`visualizeMultiClass.py`) - Analysis plots and performance metrics
4. **Hyperparameter Optimization** (`launchGAOptim.sh`) - Genetic Algorithm optimization
5. **Model Evaluation** (`evalModels.sh`) - Cross-validation training and evaluation
6. **SKFlat Integration** (`toSKFlat.sh`) - Deploy models to analysis framework

---

## Step 1: Dataset Preparation (`saveDatasets.sh`)

Converts ROOT files into PyTorch Geometric datasets with per-process storage.

### Configuration
**Signal Samples** (7 mass points): TTToHcToWAToMuMu-MHc[100,115,130,145,160]MA[85,87,90,92,95,98,100]

**Background Samples**: Skim_TriLep_TTLL_powheg (nonprompt), DYJets + DYJets10to50 (prompt), WZTo3LNu_amcatnlo + ZZTo4L_powheg (diboson), TTZToLLNuNu + TTWToLNu + tZq (ttX)

**Channels**: Run1E2Mu (1e+2μ), Run3Mu (3μ), Combined

**Features**: 9 per particle [E,Px,Py,Pz,Charge,BtagScore,IsMuon,IsElectron,IsJet] + 4 graph-level (era encoding) + event weights (genWeight×puWeight×prefireWeight)

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

**NEW**: Fully refactored modular training system with sophisticated class imbalance handling.

### Architecture Variants
| Model | Parameters | Use Case |
|-------|------------|----------|
| **ParticleNet** | ~270K | Standard baseline, individual backgrounds |
| **OptimizedParticleNet** | ~350K | **Recommended for grouped backgrounds** |
| **EfficientParticleNet** | ~500K | Complex classification, balanced performance |
| **EnhancedParticleNet** | ~1M+ | Maximum capacity (slower training) |

### Class Imbalance Handling

**Three-Layer Strategy**:
1. **Dataset Normalization**: Between-group weight balancing (preserves within-group cross-sections)
2. **Physics-Aware Loss**: Event weights (genWeight×puWeight×prefireWeight) with weighted/focal loss
3. **Group-Balanced Metrics**: Equal class contribution regardless of sample count

**Background Training Modes**:
- **Individual**: Each sample as separate class (e.g., signal + TTLL + WZ + TTZ)
- **Grouped**: Physics-motivated categories (e.g., signal + nonprompt + diboson + ttX)

### Modular Training System (NEW)

The training system has been refactored into focused modules for improved maintainability and extensibility:

```python
# Example usage with new modular system
from TrainingConfig import create_training_config
from DataPipeline import create_data_pipeline
from TrainingOrchestrator import create_training_orchestrator
from ResultPersistence import create_result_persistence

# 1. Configuration Management
config = create_training_config()  # Handles all argument parsing and validation

# 2. Data Pipeline Setup
data_pipeline = create_data_pipeline(config)
data_pipeline.create_datasets()
data_pipeline.create_data_loaders()

# 3. Training Orchestration
orchestrator = create_training_orchestrator(config, data_pipeline)
training_results = orchestrator.train()

# 4. Result Persistence
persistence = create_result_persistence(config)
persistence.save_predictions_to_root(model, data_pipeline, device, tree_path)
```

**Module Responsibilities**:
- **TrainingConfig**: Argument parsing, validation, path management, model naming
- **TrainingUtilities**: Core training functions, group-balanced metrics, device setup
- **DataPipeline**: Dataset creation, data loader management, integrity validation
- **TrainingOrchestrator**: Training loop execution, early stopping, progress monitoring
- **ResultPersistence**: ROOT tree creation, performance metrics, model information

### Key Training Features
- 5-fold cross-validation (3 train / 1 valid / 1 test)
- Automatic weight normalization for class balance
- Early stopping with validation monitoring
- Real-time performance monitoring (memory, CPU, time)
- ROOT tree outputs with per-class score distributions and event weights
- **NEW**: Comprehensive error handling and validation at each pipeline stage

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
- `--model`: Architecture variant
- `--loss_type`: weighted_ce (default), sample_normalized, focal
- `--balance`: Enable weight normalization (default: True)
- `--pilot`: Use pilot datasets

### Performance Impact
- **Individual backgrounds**: 17.7% accuracy (imbalanced)
- **Grouped backgrounds**: 36.5% accuracy (106% improvement)
- **Training stability**: 2x faster convergence with proper normalization

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

Genetic Algorithm optimization for systematic hyperparameter search.

**Search Space**: Nodes [64,96,128], Optimizers [RMSprop,Adam,Adadelta], Schedulers [ExponentialLR,CyclicLR,ReduceLROnPlateau], Learning rates [1e-4,1e-2], Weight decay [1e-5,1e-2]

```bash
./scripts/launchGAOptim.sh MHc-130_MA-100 nonprompt Combined cuda:0
```

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

### Class Imbalance Strategy Details

**Problem**: Signal/background cross-sections differ by 4-5 orders of magnitude

**Solution**: Multi-layered approach
1. **Weight Normalization**: `max_weight / sample_total_weight` per class
2. **Physics Weights**: Per-event `genWeight × puWeight × prefireWeight`
3. **Balanced Accuracy**: Within-class weighted, between-class equal contribution

**Loss Functions**:
- `weighted_ce`: Standard with event weights
- `sample_normalized`: Additional per-class normalization
- `focal`: Hard example focus with physics weights

### Refactored Modular Architecture (NEW)

**Training System Refactoring**: The original 691-line monolithic `trainMultiClass.py` has been refactored into a clean modular architecture with 80% code reduction in the main script (140 lines).

**Core Training Modules**:
- **`TrainingConfig.py`** - Centralized argument parsing, validation, and configuration management
- **`TrainingUtilities.py`** - Core training functions, group-balanced metrics, and performance monitoring
- **`DataPipeline.py`** - Dataset creation, data loader management, and integrity validation
- **`TrainingOrchestrator.py`** - Training loop execution, early stopping, and metrics collection
- **`ResultPersistence.py`** - ROOT tree creation, model saving, and output management
- **`trainMultiClass.py`** - Clean main entry point with high-level workflow orchestration

**Benefits**:
- **Maintainability**: Single responsibility per module
- **Testability**: Individual components can be unit tested
- **Reusability**: Training utilities shared across scripts
- **Extensibility**: Easy feature additions without affecting existing code
- **Backward Compatibility**: Same CLI interface and output structure

### Directory Structure
```
ParticleNet/
├── python/          # Core modules and refactored training system
│   ├── trainMultiClass.py          # Main entry point (140 lines, was 691)
│   ├── trainMultiClass_original.py # Original backup
│   ├── TrainingConfig.py           # Configuration management
│   ├── TrainingUtilities.py        # Core training functions
│   ├── DataPipeline.py             # Data loading and validation
│   ├── TrainingOrchestrator.py     # Training loop management
│   ├── ResultPersistence.py        # Output and persistence
│   ├── MultiClassModels.py         # Model architectures
│   ├── WeightedLoss.py             # Physics-aware loss functions
│   ├── DynamicDatasetLoader.py     # Dataset management
│   ├── visualizeMultiClass.py      # Visualization with class imbalance support
│   └── ...                         # Other utilities
├── scripts/         # Workflow automation
├── dataset/         # Processed datasets
├── results/         # Trained models and metrics
└── plots/          # Analysis visualizations
```

### Requirements
- PyTorch + PyTorch Geometric
- ROOT + scikit-learn
- GNU Parallel
- CUDA recommended

### Current Status
✅ **Complete**: Dataset preparation, multi-class training, visualization, class imbalance handling, performance monitoring, **modular refactoring**
🚧 **In Progress**: Hyperparameter optimization, results parsing, SKFlat deployment

### Key Features
- **NEW**: Modular training architecture with 80% code reduction and improved maintainability
- **NEW**: Comprehensive weight storage for class imbalance analysis in ROOT trees
- Per-process dataset storage (~60% size reduction)
- Physics-motivated background grouping with hierarchical weight normalization
- 6 architecture variants optimized for different complexity levels
- Advanced class imbalance handling preserving cross-section physics
- Real-time performance monitoring and computational optimization
- Comprehensive visualization pipeline with automatic adaptation
- 5-fold cross-validation with proven train/valid/test splitting scheme
- **NEW**: Full backward compatibility with enhanced error handling and validation