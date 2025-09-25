# ParticleNet
---
## Introduction
This module implements a ParticleNet-based machine learning pipeline for charged Higgs boson analysis. The workflow consists of 5 main steps to train graph neural networks for discriminating signal from background events.

## Workflow Overview
1. **Dataset Preparation** (`saveDatasets.sh`) - Convert ROOT files to PyTorch Geometric datasets
2. **Hyperparameter Optimization** (`launchGAOptim.sh`) - Use Genetic Algorithm to find optimal hyperparameters
3. **Model Training** (`evalModels.sh`) - Train models with best hyperparameters on all folds
4. **Results Parsing** (`parseBestResults.sh`) - Extract best models and organize results
5. **SKFlat Integration** (`toSKFlat.sh`) - Deploy trained models to analysis framework

## Detailed Workflow

### Step 1: Dataset Preparation (`saveDatasets.sh`)
Converts ROOT files into PyTorch Geometric datasets with per-process storage for maximum flexibility.

**Per-Process Architecture:**

#### Data Structure
- **Python modules**:
  - `python/DataFormat.py`: Particle class definitions (Muon, Electron, Jet, MET)
  - `python/Preprocess.py`: Graph conversion utilities and dataset creation
  - `python/saveDataset.py`: Per-process dataset preparation script
  - `python/DynamicDatasetLoader.py`: Dynamic dataset loading for training

- **C++ library** (optional, for performance):
  - `include/DataFormat.h`: Header with particle classes using ROOT RVec
  - `src/DataFormat.cc`: Implementation with RVec for improved performance
  - Build with: `make` (requires ROOT environment)

#### Configuration
- **Signal MC Samples** (7 mass points):
  - TTToHcToWAToMuMu-MHc100MA95
  - TTToHcToWAToMuMu-MHc115MA87
  - TTToHcToWAToMuMu-MHc130MA90
  - TTToHcToWAToMuMu-MHc130MA100
  - TTToHcToWAToMuMu-MHc145MA92
  - TTToHcToWAToMuMu-MHc160MA85
  - TTToHcToWAToMuMu-MHc160MA98

- **Background MC Samples**:
  - Skim_TriLep_TTLL_powheg (nonprompt leptons)
  - Skim_TriLep_DYJets (Drell-Yan production)
  - Skim_TriLep_DYJets10to50 (low-mass Drell-Yan)
  - Skim_TriLep_WZTo3LNu_amcatnlo (diboson production)
  - Skim_TriLep_TTZToLLNuNu (ttZ production)
  - Skim_TriLep_TTWToLNu (ttW production)
  - Skim_TriLep_tZq (single top + Z)

- **Channels**: 
  - Run1E2Mu (1 electron + 2 muons)
  - Run3Mu (3 muons)
  - Combined (both channels together)

- **Features**:
  - 9 features per particle: [E, Px, Py, Pz, Charge, BtagScore, IsMuon, IsElectron, IsJet]
  - Graph structure: k=4 nearest neighbors in ΔR
  - Event weights: genWeight × puWeight × prefireWeight
  - Era information: String format for [2016preVFP, 2016postVFP, 2017, 2018]
  - 5-fold cross-validation with deterministic splitting
  - Process metadata: Separate storage for signals and backgrounds

#### Usage

**Per-Process Dataset Creation**:
```bash
# Create all datasets (signals + backgrounds separately)
./scripts/saveDatasets.sh

# Create pilot datasets for testing
./scripts/saveDatasets.sh --pilot

# Create individual samples
python3 python/saveDataset.py --sample TTToHcToWAToMuMu-MHc100MA95 --sample-type signal --channel Run1E2Mu
python3 python/saveDataset.py --sample Skim_TriLep_TTLL_powheg --sample-type background --channel Run1E2Mu
```

**Dynamic Training Dataset Creation**:
```python
from DynamicDatasetLoader import DynamicDatasetLoader

loader = DynamicDatasetLoader(f"{WORKDIR}/ParticleNet/dataset")

# Binary classification: signal sample vs background category
dataset = loader.create_training_dataset({
    "mode": "binary",
    "signal": "TTToHcToWAToMuMu-MHc130MA100",  # MC sample name
    "background": "nonprompt",                 # Physics category
    "channel": "Run1E2Mu",
    "fold": 0
})

# Multi-class: signal vs all background categories
dataset = loader.create_training_dataset({
    "mode": "multiclass",
    "signal": "TTToHcToWAToMuMu-MHc130MA100",
    "channel": "Run1E2Mu",
    "fold": 0
})
```

**Background Category Mapping**:
The system maps MC samples to physics categories at loading time:
```python
background_categories = {
    "Skim_TriLep_TTLL_powheg": "nonprompt",      # Non-prompt leptons
    "Skim_TriLep_DYJets": "prompt",              # Prompt leptons (DY)
    "Skim_TriLep_DYJets10to50": "prompt",        # Low-mass DY
    "Skim_TriLep_WZTo3LNu_amcatnlo": "diboson",  # Diboson production
    "Skim_TriLep_TTZToLLNuNu": "ttZ",            # ttZ production
    "Skim_TriLep_TTWToLNu": "ttW",               # ttW production
    "Skim_TriLep_tZq": "rare_top",               # Rare top processes
}
```

**Options**:
- `--pilot`: Create smaller datasets (10k events) for testing
- `--debug`: Enable debug logging

#### Dataset Statistics Logging
The saveDataset.py script now automatically generates detailed statistics for each dataset:

1. **Statistics saved in two locations**:
   - In dataset directory: `<dataset_dir>/<signal>_vs_<background>_stats.json`
   - In logs directory: `logs/dataset_stats_<channel>_<signal>_vs_<background>_<timestamp>.json`

2. **Information tracked**:
   - Events loaded per era (2016preVFP, 2016postVFP, 2017, 2018)
   - Events per fold before and after balancing
   - Total event counts for signal and each background
   - Data usage efficiency

3. **Analyzing statistics**:
   ```bash
   # Analyze a specific stats file
   python3 python/analyze_dataset_stats.py dataset/Run1E2Mu__/MHc-100_MA-95_vs_ttZ_stats.json
   
   # Analyze the latest stats file
   python3 python/analyze_dataset_stats.py --latest
   ```

#### Output Structure
```
dataset/
└── samples/
    ├── signals/
    │   ├── TTToHcToWAToMuMu-MHc100MA95/
    │   │   ├── Run1E2Mu_fold-0.pt
    │   │   ├── Run1E2Mu_fold-1.pt
    │   │   └── ...
    │   ├── TTToHcToWAToMuMu-MHc130MA100/
    │   └── ... (7 signal samples)
    └── backgrounds/
        ├── Skim_TriLep_TTLL_powheg/
        │   ├── Run1E2Mu_fold-0.pt
        │   └── ...
        ├── Skim_TriLep_DYJets/
        ├── Skim_TriLep_WZTo3LNu_amcatnlo/
        ├── Skim_TriLep_TTZToLLNuNu/
        ├── Skim_TriLep_TTWToLNu/
        └── ... (7 background samples)
```

Each `.pt` file contains pure MC sample data with weights, no labels applied until training time.

### Step 2: Hyperparameter Optimization (`launchGAOptim.sh`)
Employs a Genetic Algorithm for systematic hyperparameter optimization.

**Search Space:**
- **Number of nodes**: [64, 96, 128]
- **Optimizers**: [RMSprop, Adam, Adadelta]
- **Learning rate schedulers**: [ExponentialLR, CyclicLR, ReduceLROnPlateau]
- **Initial learning rates**: log-uniform(1e-4, 1e-2)
- **Weight decay**: log-uniform(1e-5, 1e-2)

**Parameters:**
- Population size: 16
- Max iterations: 4
- Fitness threshold: 0.9

**Usage:**
```bash
./scripts/launchGAOptim.sh <SIGNAL> <BACKGROUND> <CHANNEL> <DEVICE>
# Example: ./scripts/launchGAOptim.sh MHc-100_MA-95 nonprompt Combined cuda:0
```

### Step 3: Model Training and Evaluation (`evalModels.sh`)
Trains ParticleNet models using the best hyperparameters from GA optimization.

**Configuration:**
- Trains separate models for each of 5 folds
- Max epochs: 81
- Parallel GPU training (cuda:0 and cuda:1)
- Generates performance metrics (ROC curves, AUC scores)
- Outputs ROOT files with model predictions

**Usage:**
```bash
./scripts/evalModels.sh <SIGNAL> <BACKGROUND> <CHANNEL>
# Example: ./scripts/evalModels.sh MHc-100_MA-95 nonprompt Combined
```

### Step 4: Results Parsing (`parseBestResults.sh`)
Extracts and organizes the best performing models from each fold.

**Output Structure:**
```
results/
└── <CHANNEL>__/
    └── <SIGNAL>_vs_<BACKGROUND>/
        └── fold-<N>/
            ├── summary.txt (hyperparameters and metrics)
            ├── model.pt (trained model)
            ├── training.csv (training history)
            └── predictions.root (model outputs)
```

**Usage:**
```bash
./scripts/parseBestResults.sh
```

### Step 5: SKFlat Integration (`toSKFlat.sh`)
Deploys trained models to the SKFlat analysis framework.

**Deployment Path:**
- `SKFlatAnalyzer/data/Run2UltraLegacy_v3/Classifiers/ParticleNet/`
- Automatically backs up existing models before overwriting

**Usage:**
```bash
./scripts/toSKFlat.sh <CHANNEL>
# Example: ./scripts/toSKFlat.sh Combined
```

## Model Architecture
- **Base**: ParticleNet graph neural network
- **Input features**: 9 node features + 4 graph-level features (era encoding)
- **Edge features**: 1 feature (ΔR distance between connected particles)
- **Output**: 
  - Binary classification: 2 classes (signal vs background)
  - Multi-class classification: 4 classes (signal, nonprompt, diboson, ttZ)
- **Architecture variants**: ParticleNet and ParticleNetV2 (to be implemented)

## Current Status
- **Dataset Preparation**: ✅ Per-process storage system fully implemented
- **Data Format**: ✅ Python and C++ (RVec) implementations complete
- **Dynamic Loading**: ✅ Flexible training dataset creation at runtime
- **Input Data**: Using SKNanoOutput ROOT files from EvtTreeProducer
- **Signal Points**: 7 mass points covering MHc=[100-160] GeV, MA=[85-100] GeV
- **Features**: 9 physics features per particle + event weights + process metadata
- **Graph Structure**: k=4 nearest neighbors in ΔR space
- **Cross-validation**: 5-fold with deterministic splitting based on MET
- **Statistical Weights**: Proper preservation of genWeight × puWeight × prefireWeight
- **Parallel Processing**: Enabled via GNU Parallel for dataset creation
- **Storage Efficiency**: ~60% reduction through per-process architecture

## Directory Structure
```
ParticleNet/
├── Makefile       # C++ build configuration
├── include/       # C++ header files
├── src/           # C++ source files
├── libs/          # Compiled C++ libraries
├── obj/           # Object files (build artifacts)
├── test/          # C++ test programs
├── python/        # Python modules
├── scripts/       # Shell scripts for workflow automation
├── dataset/       # Processed datasets (created by saveDatasets.sh)
├── condor/        # Condor job outputs (GA optimization and evaluation)
└── results/       # Final trained models and metrics
```

## Requirements
- PyTorch with CUDA support
- PyTorch Geometric
- ROOT
- scikit-learn
- GNU Parallel
- Multiple GPUs recommended for efficient training

## Implementation Status

### ✅ Completed Components

#### Per-Process Dataset System
- **Python Modules**:
  - `DataFormat.py` - Particle class definitions (Muon, Electron, Jet, MET)
  - `Preprocess.py` - Graph conversion with weights and process metadata
  - `saveDataset.py` - Per-process dataset creation with weight preservation
  - `DynamicDatasetLoader.py` - Runtime dataset assembly for training

- **C++ Library** (optional performance enhancement):
  - `include/DataFormat.h` & `src/DataFormat.cc` - ROOT RVec implementation
  - Makefile for compilation

- **Shell Scripts**:
  - `saveDatasets.sh` - Parallel per-process dataset creation

- **Documentation & Examples**:
  - `example_per_process_usage.py` - Comprehensive usage examples
  - `README_per_process_storage.md` - Detailed system documentation

### 🚧 To Be Implemented

#### Model Architecture
- **Models.py** - ParticleNet graph neural network implementation
- **MLTools.py** - Training utilities, metrics, and loss functions

#### Hyperparameter Optimization
- **GATools.py** - Genetic Algorithm framework
- **launchGAOptim.py** - GA optimization launcher
- **trainSglConfigForGA.py** - Single configuration training for GA
- **launchGAOptim.sh** - Shell wrapper for GA optimization

#### Model Training & Evaluation
- **evalModels.py** - Multi-fold model training and evaluation
- **trainModel.py** - Full model training script
- **evalModels.sh** - Parallel training launcher

#### Results Processing
- **parseBestResults.py** - Extract best models from each fold
- **parseBestResults.sh** - Automated results parsing
- **toSKFlat.sh** - Deploy models to SKFlat framework

### Key Features
- ✅ Per-process storage for maximum flexibility
- ✅ Dynamic training dataset creation (binary and multi-class)
- ✅ Statistical weight preservation (genWeight × puWeight × prefireWeight)
- ✅ 5-fold cross-validation with deterministic splitting
- ✅ Runtime class balancing with proper weight handling
- ✅ ROOT RVec integration for performance
- ✅ ~60% storage reduction through process separation
- 🚧 GPU-accelerated training
- 🚧 Genetic Algorithm hyperparameter optimization
- 🚧 Comprehensive performance metrics (ROC, AUC, confusion matrices)
