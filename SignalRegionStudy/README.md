# SignalRegionStudy

**Version**: 1.0
**Part of**: ChargedHiggsAnalysisV3
**Purpose**: Signal region optimization and statistical analysis framework for charged Higgs searches

---

## 📚 Documentation Index

### Quick Start
- **[5-Minute Demo](#quick-start)** - Get started immediately
- **[Installation Guide](#installation-steps-for-combine)** - Setup Combine and environment

### Core Documentation
- **[PROJECT_INDEX.md](docs/PROJECT_INDEX.md)** - Complete project overview
  - Architecture and design philosophy
  - Component descriptions (Preprocessor, AmassFitter)
  - Build system details
  - Usage workflows and examples

- **[API_REFERENCE.md](docs/API_REFERENCE.md)** - Detailed class documentation
  - Preprocessor API with usage examples
  - AmassFitter API with RooFit integration
  - Type definitions and best practices
  - Python/ROOT integration guide

- **[WORKFLOW_GUIDE.md](docs/WORKFLOW_GUIDE.md)** - Practical analysis workflows
  - Standard single/multi-masspoint processing
  - Score optimization and retraining
  - Systematic uncertainty profiling
  - Template creation pipeline
  - Troubleshooting and performance tips

- **[COMBINE_WORKFLOW.md](docs/COMBINE_WORKFLOW.md)** - Complete statistical analysis workflow
  - End-to-end combine execution guide
  - Template preparation (prepareCombine.sh)
  - Statistical analysis (runCombine.sh)
  - Batch processing and automation
  - Result extraction and interpretation

- **[BUILD_NOTES.md](docs/BUILD_NOTES.md)** - Build system and troubleshooting
  - Environment requirements
  - CMake configuration
  - Common build issues and solutions

### Parent Project
- **[ChargedHiggsAnalysisV3/CLAUDE.md](../CLAUDE.md)** - Framework-wide documentation

---

## Quick Start

### Prerequisites
```bash
# 1. Source environment (REQUIRED)
source setup.sh

# 2. Build library
./scripts/build.sh

# 3. Verify installation
ls lib/libSignalRegionStudy.so  # Should exist
```

### 5-Minute Example
```bash
# Process single masspoint
ERA="2022"
CHANNEL="SR1E2Mu"
MASSPOINT="MHc130_MA90"
METHOD="ParticleNet"

# Step 1: Preprocess data
preprocess.py --era $ERA --channel SR1E2Mu --signal $MASSPOINT

# Step 2: Create templates
makeBinnedTemplates.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method $METHOD

# Step 3: Validate templates
checkTemplates.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method $METHOD

# Step 4: Generate datacard
printDatacard.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method $METHOD
# Automatically saves to: templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt
```

→ **[Full Workflow Guide](docs/WORKFLOW_GUIDE.md)** for advanced usage

---

## Installation Steps for Combine

Combine should be installed locally in ChargedHiggsAnalysisV3/Common directory.
We use Combine v10.2.1 with CMSSW_14_1_0_pre4.

```bash
# In ChargedHiggsAnalysisV3/Common
export SCRAM_ARCH=el8_amd64_gcc12
cmsrel CMSSW_14_1_0_pre4
cd CMSSW_14_1_0_pre4/src
cmsenv

# Install HiggsAnalysis-CombinedLimit
git -c advice.detachedHead=false clone --depth 1 --branch v10.2.1 \
    https://github.com/cms-analysis/HiggsAnalysis-CombinedLimit.git \
    HiggsAnalysis/CombinedLimit

# Install CombineHarvester
cd $CMSSW_BASE/src
git clone https://github.com/cms-analysis/CombineHarvester.git CombineHarvester
cd CombineHarvester
git checkout v3.0.0-pre1

# Build
cd $CMSSW_BASE/src
scramv1 b clean; scramv1 b -j 12
```

→ **[Detailed Build Instructions](docs/BUILD_NOTES.md)**

---

## Project Structure

```
SignalRegionStudy/
├── include/              # C++ headers
│   ├── Preprocessor.h    # Data preprocessing
│   ├── AmassFitter.h     # Mass fitting with RooFit
│   └── LinkDef.h         # ROOT dictionary linkage
├── src/                  # C++ implementations
│   ├── Preprocessor.cc
│   └── AmassFitter.cc
├── scripts/              # Build and automation
│   └── build.sh          # CMake build script
├── docs/                 # Documentation
│   ├── PROJECT_INDEX.md      # Project overview
│   ├── API_REFERENCE.md      # Class documentation
│   ├── WORKFLOW_GUIDE.md     # Analysis workflows
│   └── BUILD_NOTES.md        # Build system details
├── lib/                  # Built libraries (generated)
├── setup.sh              # Environment setup
└── CMakeLists.txt        # Build configuration

Runtime outputs:
├── samples/              # Preprocessed ROOT files
└── templates/            # Statistical analysis templates
```

→ **[Architecture Details](docs/PROJECT_INDEX.md#architecture-overview)**

---

## Core Components

### Preprocessor Class
**Purpose**: Transform raw analysis trees into signal region templates

**Key Features**:
- Era/channel/masspoint parameterization
- Systematic variation handling
- Signal cross-section normalization
- Conversion electron scale factors
- Multi-fold training/validation support

**Example Usage**:
```cpp
Preprocessor prep("2022", "Skim1E2Mu", "SingleMuon");
prep.setConvSF(1.05, 0.10);
prep.setInputFile("input.root");
prep.setOutputFile("output.root");
prep.setInputTree("Central");
prep.fillOutTree("DYJets", "MHc130_MA90", "Central", true, true);
prep.saveTree();
```

→ **[Full Preprocessor API](docs/API_REFERENCE.md#preprocessor-class)**

### AmassFitter Class
**Purpose**: Voigtian fitting for pseudoscalar mass peak extraction

**Statistical Model**: RooVoigtian (Breit-Wigner ⊗ Gaussian)

**Example Usage**:
```cpp
AmassFitter fitter("signal.root", "fit_results.root");
fitter.fitMass(90.0, 80.0, 100.0);  // Fit mA=90 GeV
fitter.saveCanvas("plots/fit.pdf");

double mA = fitter.getRooMA()->getVal();
double mA_err = fitter.getRooMA()->getError();
```

→ **[Full AmassFitter API](docs/API_REFERENCE.md#amassfitter-class)**

---

## Standard Workflows

### Workflow 1: Standard Template Creation
For masspoints **without** score optimization:

```bash
preprocess.py --era $ERA --channel $CHANNEL --signal $MASSPOINT
makeBinnedTemplates.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method $METHOD
checkTemplates.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method $METHOD
printDatacard.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method $METHOD
```

### Workflow 2: Score Optimization
For masspoints **with** ML score updates:

```bash
preprocess.py --era $ERA --channel $CHANNEL --signal $MASSPOINT
makeBinnedTemplates.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method ParticleNet --update
checkTemplates.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method ParticleNet
printDatacard.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method ParticleNet \
    >> templates/$ERA/$CHANNEL/$MASSPOINT/Shape/ParticleNet/datacard.txt
```

**Note**: Updated scores are stored in `samples/` directory. To rerun, delete files and rerun preprocessing.

→ **[All Workflows](docs/WORKFLOW_GUIDE.md#standard-workflows)**

---

## Running Combine Analysis

### Complete Workflow Overview

The statistical analysis consists of **3 stages**:

1. **Template Preparation** - Create datacards and shapes
2. **Statistical Analysis** - Run HiggsCombine
3. **Batch Processing** - Automate across masspoints

### Quick Start: Single Masspoint

```bash
# Stage 1: Prepare templates (4 sub-steps internally)
./scripts/prepareCombine.sh $ERA $CHANNEL $MASSPOINT $METHOD
# → Creates: templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt

# Stage 2: Run HiggsCombine
./scripts/runCombine.sh $ERA $CHANNEL $MASSPOINT $METHOD
# → Creates: higgsCombineTest.AsymptoticLimits.mH120.root

# Example:
./scripts/prepareCombine.sh 2022 SR1E2Mu MHc130_MA90 ParticleNet
./scripts/runCombine.sh 2022 SR1E2Mu MHc130_MA90 ParticleNet
```

### Batch Processing: All Masspoints

```bash
# Stage 3: Process all masspoints in parallel (18 jobs)
# Edit doThis.sh to configure masspoints list
./doThis.sh

# Or process single masspoint, all eras/channels
./scripts/runCombineWrapper.sh MHc130_MA90 ParticleNet
```

### Script Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `ERA` | 2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix, **FullRun2** | Data-taking period |
| `CHANNEL` | SR1E2Mu, SR3Mu, **Combined** | Analysis channel |
| `MASSPOINT` | MHc100_MA95, MHc130_MA90, MHc160_MA85, etc. | Signal hypothesis |
| `METHOD` | Baseline, ParticleNet | Discrimination method |

**Special values**:
- `ERA=FullRun2` - Combines all Run2 eras (2016preVFP + 2016postVFP + 2017 + 2018)
- `CHANNEL=Combined` - Merges SR1E2Mu + SR3Mu channels

### Output Structure

```
templates/{era}/{channel}/{masspoint}/Shape/{method}/
├── shapes.root                                # All histogram templates (~159)
├── datacard.txt                               # HiggsCombine input
├── fit_result.root                            # Signal fit workspace
├── workspace.root                             # Full probability model
├── higgsCombineTest.FitDiagnostics.mH120.root    # Fit results
├── higgsCombineTest.AsymptoticLimits.mH120.root  # Limit values
└── validation/                                # Diagnostic plots
```

### Understanding Results

Extract limits from combine output:

```bash
# View limit tree
root -l templates/FullRun2/Combined/MHc130_MA90/Shape/ParticleNet/higgsCombineTest.AsymptoticLimits.mH120.root

# In ROOT:
limit->Scan("quantileExpected:limit")
# Shows: expected ±1σ, ±2σ bands and observed limit
```

### Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| "combineTool.py: command not found" | Install CombineHarvester in CMSSW |
| "Fit quality is not good" | Run `checkTemplates.py` to validate inputs |
| "Expected limit > 100" | No sensitivity to this mass point (not an error) |
| "Datacard not found for Combined" | Run prepareCombine.sh for SR1E2Mu and SR3Mu first |

→ **[Complete Combine Workflow Guide](docs/COMBINE_WORKFLOW.md)** for detailed instructions, advanced usage, and troubleshooting

---

## Advanced Topics

### Systematic Uncertainty Profiling
```bash
# Identify dominant uncertainties
combineTool.py -M Impacts -d workspace.root -m 90 --doFits
plotImpacts.py -i impacts.json -o impacts_plot
```

### Signal Region Optimization
```python
# Optimize mass window for maximum sensitivity
python3 optimize_sr_boundaries.py
```

### Multi-Era Combination
```bash
# Combine Run3 eras
combineCards.py \
    2022=templates/2022/.../datacard.txt \
    2023=templates/2023/.../datacard.txt \
    > datacard_combined.txt
```

→ **[Advanced Workflows](docs/WORKFLOW_GUIDE.md#advanced-workflows)**

---

## Channel Processing

### SR1E2Mu Channel
- Single Z candidate reconstruction
- Conversion scale factor applied
- Optimal for electron channel sensitivity

### SR3Mu Channel
- Dual Z candidate selection based on transverse mass
- No conversion background
- Pure muon final state

→ **[Detailed Processing Logic](docs/API_REFERENCE.md#fillOutTree)**

---

## Configuration

### Era Support
- **Run3**: 2022, 2022EE, 2023, 2023BPix
- **Run2**: 2016preVFP, 2016postVFP, 2017, 2018

### Methods
- **Mass**: Simple invariant mass binning
- **ParticleNet**: ML discriminator (recommended)
- **BDT**: Boosted Decision Tree scores
- **DNN**: Deep Neural Network scores

### Systematic Uncertainties
- **Experimental**: Lepton/jet reconstruction, energy scales, b-tagging
- **Data-driven**: Nonprompt background (shape), conversion (normalization)
- **Normalization**: Luminosity, cross-sections
- **Theory**: PDF, scale, parton shower (signal only)

All systematic definitions in `configs/systematics.json`

→ **[Systematic Details](docs/systematics.md)**

---

## Troubleshooting

### Common Issues

**"Events_Central tree not found"**
```bash
# Check file contents
root -l input.root
.ls
```

**"Systematic variation has zero entries"**
```python
# Verify systematics
tree = f.Get("Events_MuonIDSFUp")
tree.Print()
```

**"Combine fit fails to converge"**
```bash
# Simplify fit
combine -M FitDiagnostics datacard.txt --robustFit 1
```

→ **[Complete Troubleshooting Guide](docs/WORKFLOW_GUIDE.md#troubleshooting)**

---

## Best Practices

### Reproducibility
- Track configurations: `git add configs/*.json`
- Archive results: `tar -czf analysis_$(date +%Y%m%d).tar.gz`
- Document environment: `conda env export > environment.yml`

### Validation
- Template sanity checks: Non-zero integral, no negative bins
- Systematic symmetry: Up/Down variations reasonable
- Statistical tests: χ²/ndf, pull distributions

### Performance
- Parallel processing: `parallel -j 4 process_masspoint`
- Memory optimization: Process in chunks, reduce concurrency
- ROOT batch mode: `ROOT.gROOT.SetBatch(True)`

→ **[Best Practices Guide](docs/WORKFLOW_GUIDE.md#best-practices)**

---

## Documentation Maintenance

**Adding new features**:
1. Update relevant API documentation
2. Add workflow examples
3. Update troubleshooting section
4. Test all documented examples

**Reporting issues**:
- Submit to project repository
- Include error messages and context
- Attach minimal reproducible example

---

## Support and Contact

- **Project Repository**: ChargedHiggsAnalysisV3/SignalRegionStudy
- **Parent Framework**: ChargedHiggsAnalysisV3
- **Documentation Issues**: Submit to repository

---

**Last Updated**: 2025-10-09
**Version**: 1.0
**Maintainers**: ChargedHiggsAnalysisV3 Development Team
