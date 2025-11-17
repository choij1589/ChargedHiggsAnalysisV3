# SignalRegionStudy - Complete Combine Workflow

**Version**: 1.0
**Last Updated**: 2025-11-14
**Audience**: Analyzers running statistical analysis with HiggsCombine

---

## Table of Contents

- [Overview](#overview)
- [Workflow Diagram](#workflow-diagram)
- [Prerequisites](#prerequisites)
- [Stage 1: Template Preparation](#stage-1-template-preparation-preparecombinesh)
- [Stage 2: Statistical Analysis](#stage-2-statistical-analysis-runcom binesh)
- [Stage 3: Batch Processing](#stage-3-batch-processing-runcombinewrappersh)
- [Stage 4: Result Collection](#stage-4-result-collection)
- [Understanding Output Files](#understanding-output-files)
- [Advanced Usage](#advanced-usage)
- [Troubleshooting](#troubleshooting)

---

## Overview

This guide describes the **complete end-to-end workflow** for running statistical analysis with HiggsCombine. It bridges the gap between template creation (documented in [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)) and final limit extraction.

### The Complete Pipeline

```
Raw Data → Preprocessing → Templates → Combine → Limits
    │           │             │          │         │
    │           │             │          │         └─→ Result plots/tables
    │           │             │          └─→ Fit diagnostics, limits
    │           │             └─→ datacard.txt, shapes.root
    │           └─→ Sample ROOT files
    └─→ SKNanoOutput files
```

### Key Scripts

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `prepareCombine.sh` | Template preparation | SKNano files | datacards, shapes |
| `runCombine.sh` | Run HiggsCombine | datacards | fit results, limits |
| `runCombineWrapper.sh` | Batch orchestration | masspoint list | complete results |
| `collectLimits.py` | Extract limits | combine outputs | limit tables/plots |

---

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     doThis.sh (Master Script)                   │
│         Launches parallel processing for all masspoints         │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ├─→ GNU parallel (18 jobs)
                             │
                ┌────────────┴────────────┐
                │  runCombineWrapper.sh   │
                │  (per masspoint)        │
                └────────────┬────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        │                    │                    │
        ▼                    ▼                    ▼
   2016preVFP           2016postVFP            2017, 2018...
        │                    │                    │
        ├─→ SR1E2Mu          ├─→ SR1E2Mu          ├─→ SR1E2Mu
        │   ├─ prepareCombine.sh                  │
        │   │   ├─ preprocess.py                  │
        │   │   ├─ makeBinnedTemplates.py         │
        │   │   ├─ checkTemplates.py              │
        │   │   └─ printDatacard.py               │
        │   └─ runCombine.sh                      │
        │       ├─ text2workspace.py              │
        │       ├─ combine -M FitDiagnostics      │
        │       └─ combine -M AsymptoticLimits    │
        │                                          │
        └─→ SR3Mu                                  └─→ SR3Mu
            (same steps)                               (same steps)

        After individual channels:
        ├─→ Combined (SR1E2Mu + SR3Mu merger)
        │   └─ combineCards.py + combine
        │
        └─→ FullRun2 (all eras merger)
            └─ combineCards.py + combine
```

---

## Prerequisites

### 1. Environment Setup

```bash
# REQUIRED: Source environment
source setup.sh

# Verify CMSSW environment
echo $CMSSW_BASE  # Should show CMSSW_14_1_0_pre4 path

# Verify Combine installation
which text2workspace.py  # Should be in $CMSSW_BASE/bin
which combine            # Should be in $CMSSW_BASE/bin
```

### 2. Build Library

```bash
./scripts/build.sh
ls lib/libSignalRegionStudy.so  # Should exist
```

### 3. Input Data Requirements

```bash
# Verify SKNanoOutput files exist
ls $WORKDIR/SKNanoOutput/SignalRegion/2022/
# Should contain sample directories: TTLL_powheg, TTLJ_powheg, DYJets, etc.
```

### 4. Disk Space

Each masspoint requires approximately:
- **Template files**: ~500 MB per era/channel
- **Combine outputs**: ~100-200 MB per era/channel
- **Total per masspoint**: ~5-10 GB for FullRun2

---

## Stage 1: Template Preparation (`prepareCombine.sh`)

### Purpose

Converts raw SKNanoOutput files into HiggsCombine-ready templates (datacards and shape files).

### Syntax

```bash
./scripts/prepareCombine.sh <ERA> <CHANNEL> <MASSPOINT> <METHOD>
```

### Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `ERA` | 2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix | Data-taking period |
| `CHANNEL` | SR1E2Mu, SR3Mu, Combined | Analysis channel |
| `MASSPOINT` | MHc100_MA95, MHc130_MA90, etc. | Signal hypothesis (mH+, mA) |
| `METHOD` | Baseline, ParticleNet | Discrimination method |

### Example

```bash
./scripts/prepareCombine.sh 2022 SR1E2Mu MHc130_MA90 ParticleNet
```

### What It Does (4 Sub-Steps)

#### Sub-Step 1: Preprocessing (`preprocess.py`)

```bash
preprocess.py --era 2022 --channel SR1E2Mu --signal MHc130_MA90 --method Baseline
```

- Loads `Preprocessor` C++ class
- Reads trees from `$WORKDIR/SKNanoOutput/SignalRegion/2022/`
- Normalizes signal cross-sections (÷3 for 5 fb)
- Applies conversion electron scale factors (1E2Mu channel)
- Extracts mass variables (mass1, mass2 for 3Mu)

**Output**: `samples/2022/SR1E2Mu/MHc130_MA90/Baseline/*.root`
- nonprompt.root, diboson.root, ttX.root, conversion.root, others.root, MHc130_MA90.root

#### Sub-Step 2: Template Creation (`makeBinnedTemplates.py`)

```bash
makeBinnedTemplates.py --era 2022 --channel SR1E2Mu \
    --masspoint MHc130_MA90 --method ParticleNet
```

- Fits signal mass distribution with Double-Sided Crystal Ball
- Creates binned templates for all processes
- Stores ~159 histograms in shapes.root
- Applies smoothing and negative bin fixes

**Output**: `templates/2022/SR1E2Mu/MHc130_MA90/Shape/ParticleNet/shapes.root`

#### Sub-Step 3: Validation (`checkTemplates.py`)

```bash
checkTemplates.py --era 2022 --channel SR1E2Mu \
    --masspoint MHc130_MA90 --method ParticleNet
```

- Validates histogram integrity
- Checks for negative bins/integrals
- Generates validation plots
- Reports warnings/errors

**Output**: Validation plots in `validation/` subdirectory

#### Sub-Step 4: Datacard Generation (`printDatacard.py`)

```bash
printDatacard.py --era 2022 --channel SR1E2Mu \
    --masspoint MHc130_MA90 --method ParticleNet
```

- Generates HiggsCombine datacard
- Includes all systematic uncertainties
- Links to shapes.root
- Creates RooFit workspace for signal fit

**Output**:
- `datacard.txt` - Main HiggsCombine input file
- `fit_result.root` - RooFit workspace with signal parameterization
- `signal_fit.png` - Diagnostic plot of signal fit

### Output Directory Structure

```
templates/2022/SR1E2Mu/MHc130_MA90/Shape/ParticleNet/
├── shapes.root              # All histograms (~159)
├── datacard.txt             # HiggsCombine input
├── fit_result.root          # RooFit workspace
├── signal_fit.png           # Fit diagnostic
└── validation/              # QA plots
    ├── nonprompt_mass.png
    ├── diboson_mass.png
    └── ...
```

### Common Issues

**Issue**: "File not found: SKNanoOutput/..."
```bash
# Solution: Check input path
ls $WORKDIR/SKNanoOutput/SignalRegion/2022/SR1E2Mu/
```

**Issue**: "Negative bin detected"
```bash
# This is automatically fixed by the scripts
# Check validation plots to verify fix quality
```

**Issue**: "Signal fit failed"
```bash
# Check signal_fit.png for diagnostic
# May need to adjust fit range in makeBinnedTemplates.py
```

---

## Stage 2: Statistical Analysis (`runCombine.sh`)

### Purpose

Executes HiggsCombine statistical framework to extract limits and perform fits.

### Syntax

```bash
./scripts/runCombine.sh <ERA> <CHANNEL> <MASSPOINT> <METHOD>
```

### Parameters

Same as `prepareCombine.sh`, plus:
- `ERA` can be **FullRun2** for combined Run2 analysis
- `CHANNEL` can be **Combined** for SR1E2Mu+SR3Mu merger

### Example

```bash
# Single era, single channel
./scripts/runCombine.sh 2022 SR1E2Mu MHc130_MA90 ParticleNet

# Combined channel (must run SR1E2Mu and SR3Mu first)
./scripts/runCombine.sh 2022 Combined MHc130_MA90 ParticleNet

# Full Run2 (must run all eras first)
./scripts/runCombine.sh FullRun2 SR1E2Mu MHc130_MA90 ParticleNet
```

### What It Does

#### Step 1: Card Combination (if needed)

For **Combined** channel:
```bash
combineCards.py \
    ch1=templates/2022/SR1E2Mu/MHc130_MA90/Shape/ParticleNet/datacard.txt \
    ch2=templates/2022/SR3Mu/MHc130_MA90/Shape/ParticleNet/datacard.txt \
    > datacard.txt
```

For **FullRun2** era:
```bash
combineCards.py \
    era1=templates/2016preVFP/SR1E2Mu/MHc130_MA90/Shape/ParticleNet/datacard.txt \
    era2=templates/2016postVFP/SR1E2Mu/MHc130_MA90/Shape/ParticleNet/datacard.txt \
    era3=templates/2017/SR1E2Mu/MHc130_MA90/Shape/ParticleNet/datacard.txt \
    era4=templates/2018/SR1E2Mu/MHc130_MA90/Shape/ParticleNet/datacard.txt \
    > datacard.txt
```

#### Step 2: Create RooWorkspace

```bash
text2workspace.py datacard.txt -o workspace.root
```

- Converts datacard to RooFit workspace
- Builds probability model for all processes
- Includes systematic uncertainties as nuisance parameters

#### Step 3: Fit Diagnostics

```bash
combine -M FitDiagnostics workspace.root
```

- Performs maximum likelihood fit
- Generates pre-fit and post-fit plots
- Extracts best-fit parameters and uncertainties
- Creates covariance matrix

**Output**: `higgsCombineTest.FitDiagnostics.mH120.root`

#### Step 4: Asymptotic Limits

```bash
combine -M AsymptoticLimits workspace.root -t -1
```

- Calculates expected and observed limits
- Uses asymptotic formulas (CLs method)
- Flag `-t -1` generates Asimov dataset (expected limit)

**Output**: `higgsCombineTest.AsymptoticLimits.mH120.root`

### Understanding Combine Methods

| Method | Purpose | Output | Typical Runtime |
|--------|---------|--------|-----------------|
| `FitDiagnostics` | Best-fit parameters, pulls, impacts | Fit results, post-fit shapes | 1-5 min |
| `AsymptoticLimits` | Expected/observed limits (CLs) | Limit values, significance | 30 sec - 2 min |
| `HybridNew` | Toy-based limits (more accurate) | Limit bands (±1σ, ±2σ) | 30 min - 2 hr |
| `Impacts` | Nuisance parameter impacts | Impact plots | 10-30 min |

**Note**: `HybridNew` and `Impacts` are commented out in the script for speed. Uncomment for final results.

### Output Files

```
templates/2022/SR1E2Mu/MHc130_MA90/Shape/ParticleNet/
├── workspace.root                                    # RooFit workspace
├── higgsCombineTest.FitDiagnostics.mH120.root       # Fit results
├── higgsCombineTest.AsymptoticLimits.mH120.root     # Limits
└── fitDiagnostics.root                              # Detailed fit output
```

### Extracting Results

#### Asymptotic Limits

```bash
# View limit tree
root -l higgsCombineTest.AsymptoticLimits.mH120.root
root [1] limit->Scan("quantileExpected:limit")

# Output format:
# quantileExpected | limit
# -----------------+--------
#            -1.00 | <observed limit>
#             0.03 | <expected -2σ>
#             0.16 | <expected -1σ>
#             0.50 | <expected median>
#             0.84 | <expected +1σ>
#             0.97 | <expected +2σ>
```

#### Fit Diagnostics

```python
import ROOT
f = ROOT.TFile("higgsCombineTest.FitDiagnostics.mH120.root")

# Get fit results
fit_s = f.Get("fit_s")  # Signal+background fit
fit_b = f.Get("fit_b")  # Background-only fit

# Print parameters
fit_s.Print("v")

# Get nuisance parameter pulls
fit_s.floatParsFinal().Print("v")
```

### Common Issues

**Issue**: "Fit failed to converge"
```bash
# Solution: Check templates for issues
checkTemplates.py --era 2022 --channel SR1E2Mu \
    --masspoint MHc130_MA90 --method ParticleNet

# May need to increase iterations
combine -M FitDiagnostics workspace.root --cminDefaultMinimizerStrategy 0
```

**Issue**: "Negative signal strength"
```bash
# Expected for high mass points with low sensitivity
# Not necessarily an error, indicates limit > 100
```

**Issue**: "Cannot find datacard.txt"
```bash
# Solution: Run prepareCombine.sh first
./scripts/prepareCombine.sh 2022 SR1E2Mu MHc130_MA90 ParticleNet
```

---

## Stage 3: Batch Processing (`runCombineWrapper.sh`)

### Purpose

Automates the complete workflow for a single masspoint across all eras and channels.

### Syntax

```bash
./scripts/runCombineWrapper.sh <MASSPOINT> <METHOD>
```

### Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `MASSPOINT` | MHc100_MA95, MHc130_MA90, etc. | Signal hypothesis |
| `METHOD` | Baseline, ParticleNet | Discrimination method |

### Example

```bash
# Process single masspoint
./scripts/runCombineWrapper.sh MHc130_MA90 ParticleNet

# Or use doThis.sh for all masspoints
# (launches 18 parallel jobs)
```

### What It Does

The wrapper script runs **all combinations** for a single masspoint:

1. **Individual eras × channels** (8 combinations):
   ```
   2016preVFP/SR1E2Mu
   2016preVFP/SR3Mu
   2016postVFP/SR1E2Mu
   2016postVFP/SR3Mu
   2017/SR1E2Mu
   2017/SR3Mu
   2018/SR1E2Mu
   2018/SR3Mu
   ```

2. **Combined channels** (4 combinations):
   ```
   2016preVFP/Combined
   2016postVFP/Combined
   2017/Combined
   2018/Combined
   ```

3. **FullRun2** (3 combinations):
   ```
   FullRun2/SR1E2Mu
   FullRun2/SR3Mu
   FullRun2/Combined
   ```

**Total**: 15 combinations per masspoint

### Execution Flow

```bash
#!/bin/bash
# Simplified version of runCombineWrapper.sh

MASSPOINT=$1
METHOD=$2

ERAs=("2016preVFP" "2016postVFP" "2017" "2018")

# Process SR1E2Mu channel (all eras)
for era in ${ERAs[@]}; do
    ./scripts/prepareCombine.sh $era SR1E2Mu $MASSPOINT $METHOD
    ./scripts/runCombine.sh $era SR1E2Mu $MASSPOINT $METHOD
done

# Process SR3Mu channel (all eras)
for era in ${ERAs[@]}; do
    ./scripts/prepareCombine.sh $era SR3Mu $MASSPOINT $METHOD
    ./scripts/runCombine.sh $era SR3Mu $MASSPOINT $METHOD
done

# Process combined channels (all eras)
for era in ${ERAs[@]}; do
    ./scripts/runCombine.sh $era Combined $MASSPOINT $METHOD
done

# Process FullRun2 combinations
./scripts/runCombine.sh FullRun2 SR1E2Mu $MASSPOINT $METHOD
./scripts/runCombine.sh FullRun2 SR3Mu $MASSPOINT $METHOD
./scripts/runCombine.sh FullRun2 Combined $MASSPOINT $METHOD
```

### Parallel Processing with `doThis.sh`

```bash
# Define masspoints
MASSPOINTs=(
    "MHc100_MA95"
    "MHc115_MA87"
    "MHc130_MA90"
    "MHc145_MA92"
    "MHc160_MA85"
    "MHc160_MA98"
)

# Launch parallel jobs (18 concurrent)
parallel -j 18 "./scripts/runCombineWrapper.sh" {1} {2} \
    ::: "${MASSPOINTs[@]}" ::: "ParticleNet"
```

### Resource Requirements

| Resource | Per Job | 18 Jobs Total |
|----------|---------|---------------|
| **CPU cores** | 1 | 18 |
| **RAM** | 2-4 GB | 36-72 GB |
| **Disk I/O** | Moderate | High |
| **Runtime** | 30-60 min | 30-60 min (parallel) |

### Monitoring Progress

```bash
# Check running jobs
ps aux | grep prepareCombine

# Check completed templates
find templates/ -name "datacard.txt" | wc -l
# Should be: 15 × (number of masspoints completed)

# Check combine outputs
find templates/ -name "higgsCombineTest.AsymptoticLimits.mH120.root" | wc -l
```

### Known Issues

**Hardcoded Path Warning**:
The script contains a hardcoded path at line 26:
```bash
cd /data9/Users/choij/workspace/ChargedHiggsAnalysisV3/SignalRegionStudy
```

**Workaround**: Edit the script or ensure you run from the expected directory.

**Future Fix**: Use `$WORKDIR` environment variable instead.

---

## Stage 4: Result Collection

### Purpose

Collect and visualize limits from all masspoints.

### Using `collectLimits.py`

```bash
# Collect limits for specific configuration
python python/collectLimits.py \
    --era FullRun2 \
    --channel Combined \
    --method ParticleNet

# Output: limits table and plots
```

### Manual Limit Extraction

```python
#!/usr/bin/env python3
import ROOT
import json

def extract_limits(root_file):
    """Extract limit values from combine output"""
    f = ROOT.TFile(root_file)
    tree = f.Get("limit")

    limits = {}
    for entry in tree:
        quantile = entry.quantileExpected
        limit_val = entry.limit

        if quantile < 0:
            limits['observed'] = limit_val
        elif abs(quantile - 0.025) < 0.01:
            limits['exp_m2sigma'] = limit_val
        elif abs(quantile - 0.16) < 0.01:
            limits['exp_m1sigma'] = limit_val
        elif abs(quantile - 0.5) < 0.01:
            limits['exp_median'] = limit_val
        elif abs(quantile - 0.84) < 0.01:
            limits['exp_p1sigma'] = limit_val
        elif abs(quantile - 0.975) < 0.01:
            limits['exp_p2sigma'] = limit_val

    return limits

# Example usage
masspoints = [
    "MHc100_MA95", "MHc115_MA87", "MHc130_MA90",
    "MHc145_MA92", "MHc160_MA85", "MHc160_MA98"
]

results = {}
for mp in masspoints:
    limit_file = f"templates/FullRun2/Combined/{mp}/Shape/ParticleNet/higgsCombineTest.AsymptoticLimits.mH120.root"
    results[mp] = extract_limits(limit_file)

# Save to JSON
with open("limits_summary.json", "w") as f:
    json.dump(results, f, indent=2)

print("Limits extracted to limits_summary.json")
```

### Creating Limit Plots

```python
import matplotlib.pyplot as plt
import numpy as np

# Extract median expected limits
mHc = [100, 115, 130, 145, 160, 160]
mA = [95, 87, 90, 92, 85, 98]
limits = [2.5, 1.8, 1.2, 0.9, 0.7, 1.1]  # Example values

plt.figure(figsize=(10, 6))
plt.plot(mHc, limits, 'o-', label='Expected limit (median)')
plt.axhline(y=1.0, color='r', linestyle='--', label='σ(theory)')
plt.xlabel('$m_{H^+}$ [GeV]')
plt.ylabel('95% CL limit / σ(theory)')
plt.title('Expected Limits - FullRun2 Combined')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('limits_vs_mass.png', dpi=300)
print("Plot saved to limits_vs_mass.png")
```

---

## Understanding Output Files

### Complete File Tree

```
templates/FullRun2/Combined/MHc130_MA90/Shape/ParticleNet/
│
├── shapes.root                                      # All templates
│   ├── TH1: nonprompt_mass_bin0, bin1, ..., bin158
│   ├── TH1: diboson_mass_bin0, bin1, ..., bin158
│   ├── TH1: ttX_mass_bin0, bin1, ..., bin158
│   └── TH1: MHc130_MA90_mass_bin0, bin1, ..., bin158
│
├── datacard.txt                                     # HiggsCombine input
│   Format:
│   imax 159   # Number of bins
│   jmax 4     # Number of backgrounds
│   kmax *     # Number of systematics
│   shapes *  * shapes.root $CHANNEL/$PROCESS $CHANNEL/$PROCESS_$SYSTEMATIC
│   bin        bin0 bin1 ... bin158
│   observation  5    3   ...   2     # Data counts
│   bin          bin0 bin0 bin0 ...
│   process      0    1    2    ...    # 0=signal, 1+=background
│   rate         1.5  2.3  0.8  ...    # Expected yields
│   lnN CMS_...  1.02 1.03 1.05 ...    # Systematic uncertainties
│
├── fit_result.root                                  # Signal fit workspace
│   └── RooWorkspace: w
│       ├── RooRealVar: mass, mean, sigma, alphaL, alphaR, nL, nR
│       └── RooDoubleCB: signal_pdf
│
├── workspace.root                                   # Full probability model
│   └── RooWorkspace: w
│       ├── All processes as RooRealVar/RooHistFunc
│       ├── All systematics as RooRealVar
│       └── Combined likelihood: ModelConfig
│
├── higgsCombineTest.FitDiagnostics.mH120.root      # Fit results
│   ├── TTree: tree_fit_sb, tree_fit_b
│   ├── RooFitResult: fit_s, fit_b
│   ├── TH1: prefit/nonprompt, postfit/nonprompt, etc.
│   └── TH2: covariance_fit_s
│
└── higgsCombineTest.AsymptoticLimits.mH120.root    # Limit results
    └── TTree: limit
        ├── quantileExpected = -1.0, limit = <observed>
        ├── quantileExpected = 0.025, limit = <exp -2σ>
        ├── quantileExpected = 0.16, limit = <exp -1σ>
        ├── quantileExpected = 0.50, limit = <exp median>
        ├── quantileExpected = 0.84, limit = <exp +1σ>
        └── quantileExpected = 0.975, limit = <exp +2σ>
```

### File Sizes

| File | Typical Size | Purpose |
|------|--------------|---------|
| `shapes.root` | 5-20 MB | All histogram templates |
| `datacard.txt` | 50-200 KB | Human-readable input |
| `fit_result.root` | 100-500 KB | Signal parameterization |
| `workspace.root` | 5-50 MB | Full probability model |
| `FitDiagnostics` | 10-100 MB | Detailed fit results |
| `AsymptoticLimits` | 10-50 KB | Final limit values |

---

## Advanced Usage

### Running Toy-Based Limits (HybridNew)

For more accurate limits with ±1σ and ±2σ bands:

```bash
# Uncomment in runCombine.sh (lines 46-54)
combine -M HybridNew workspace.root \
    --LHCmode LHC-limits \
    -T 500 \                       # Number of toys
    --expectedFromGrid 0.5 \       # Median expected
    -m 120

# Then scan other quantiles
for q in 0.025 0.16 0.84 0.975; do
    combine -M HybridNew workspace.root \
        --LHCmode LHC-limits \
        -T 500 \
        --expectedFromGrid $q \
        -m 120
done
```

**Runtime**: ~30 minutes to 2 hours per quantile

### Impact Analysis

Determine which systematics have largest effect:

```bash
# Step 1: Initial fit
combineTool.py -M Impacts \
    -d workspace.root \
    -m 120 \
    --doInitialFit \
    --robustFit 1

# Step 2: Fit with each nuisance
combineTool.py -M Impacts \
    -d workspace.root \
    -m 120 \
    --doFits \
    --robustFit 1 \
    --parallel 8

# Step 3: Collect results
combineTool.py -M Impacts \
    -d workspace.root \
    -m 120 \
    -o impacts.json

# Step 4: Plot
plotImpacts.py -i impacts.json -o impacts
# Generates impacts.pdf
```

### Goodness-of-Fit Test

```bash
# Saturated model test
combine -M GoodnessOfFit workspace.root --algo=saturated

# Generate toys
combine -M GoodnessOfFit workspace.root --algo=saturated -t 500 -s 0:19:1

# Compare to data
python $CMSSW_BASE/src/HiggsAnalysis/CombinedLimit/test/gofTools/plotGof.py
```

### Significance Calculation

```bash
# Expected significance (Asimov)
combine -M Significance workspace.root -t -1 --expectSignal=1

# Observed significance
combine -M Significance workspace.root
```

---

## Troubleshooting

### Problem: "combineTool.py: command not found"

**Diagnosis**: CombineHarvester not installed

**Solution**:
```bash
cd $CMSSW_BASE/src
git clone https://github.com/cms-analysis/CombineHarvester.git
scram b -j 8
```

### Problem: "Fit quality is not good (status != 0)"

**Diagnosis**: Fit convergence issues

**Solutions**:
1. Check template quality:
   ```bash
   checkTemplates.py --era 2022 --channel SR1E2Mu \
       --masspoint MHc130_MA90 --method ParticleNet
   ```

2. Increase fit iterations:
   ```bash
   combine -M FitDiagnostics workspace.root \
       --cminDefaultMinimizerStrategy 0 \
       --cminDefaultMinimizerTolerance 0.01
   ```

3. Use robust fit:
   ```bash
   combine -M FitDiagnostics workspace.root --robustFit 1
   ```

### Problem: "Expected limit > 100"

**Diagnosis**: No sensitivity to this mass point

**Interpretation**:
- Not necessarily an error
- Indicates signal cross-section limit is > 100× theory
- May exclude this mass point from final results

**Check**:
```bash
# View fit results
root -l higgsCombineTest.FitDiagnostics.mH120.root
root [1] fit_s->Print("v")
# Look for r (signal strength) value
```

### Problem: "Negative bins after smoothing"

**Diagnosis**: Aggressive smoothing or low statistics

**Solutions**:
1. Automatic fix already applied in `makeBinnedTemplates.py`
2. Check validation plots:
   ```bash
   ls templates/*/validation/
   ```
3. Adjust smoothing parameters in config if needed

### Problem: "Workspace size > 1 GB"

**Diagnosis**: Too many bins or systematics

**Solutions**:
1. Reduce binning in `makeBinnedTemplates.py`
2. Remove negligible systematics from datacard
3. Use `--X-rtd REMOVE_CONSTANT_ZERO_POINT=1` in combine

### Problem: "Different results between AsymptoticLimits and HybridNew"

**Interpretation**:
- Normal, especially for low-mass points
- AsymptoticLimits uses asymptotic formulas (fast, approximate)
- HybridNew uses toys (slow, accurate)
- **Use HybridNew for final results, AsymptoticLimits for testing**

### Problem: "Datacard not found for Combined/FullRun2"

**Diagnosis**: Individual datacards not created yet

**Solution**: Run prepareCombine.sh for individual channels first:
```bash
# For Combined channel, need both:
./scripts/prepareCombine.sh 2022 SR1E2Mu MHc130_MA90 ParticleNet
./scripts/prepareCombine.sh 2022 SR3Mu MHc130_MA90 ParticleNet
# Then:
./scripts/runCombine.sh 2022 Combined MHc130_MA90 ParticleNet

# For FullRun2, need all eras:
for era in 2016preVFP 2016postVFP 2017 2018; do
    ./scripts/prepareCombine.sh $era SR1E2Mu MHc130_MA90 ParticleNet
done
# Then:
./scripts/runCombine.sh FullRun2 SR1E2Mu MHc130_MA90 ParticleNet
```

---

## Best Practices

### 1. Always Run Validation

```bash
# After every template creation
checkTemplates.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method $METHOD

# Check validation plots
ls -lh templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/validation/
```

### 2. Test Single Point First

```bash
# Before running all masspoints, test one:
./scripts/runCombineWrapper.sh MHc130_MA90 ParticleNet

# Verify outputs
ls -lh templates/FullRun2/Combined/MHc130_MA90/Shape/ParticleNet/
```

### 3. Monitor Disk Space

```bash
# Check space before running
df -h $PWD

# Monitor during execution
watch -n 60 'du -sh templates/'
```

### 4. Save Intermediate Results

```bash
# After template creation, backup
tar -czf templates_backup_$(date +%Y%m%d).tar.gz templates/

# After combine, save limits
python python/collectLimits.py --era FullRun2 --channel Combined --method ParticleNet
```

### 5. Use Screen/Tmux for Long Jobs

```bash
# Start screen session
screen -S combine_analysis

# Run workflow
./doThis.sh

# Detach: Ctrl+A, D
# Reattach: screen -r combine_analysis
```

---

## Quick Reference

### Typical Workflow Commands

```bash
# 1. Setup (ALWAYS FIRST)
source setup.sh

# 2. Single masspoint, all combinations
./scripts/runCombineWrapper.sh MHc130_MA90 ParticleNet

# 3. Or, all masspoints in parallel
# (Edit doThis.sh to configure masspoints)
./doThis.sh

# 4. Monitor progress
find templates/ -name "higgsCombineTest.AsymptoticLimits.mH120.root" | wc -l

# 5. Collect results
python python/collectLimits.py --era FullRun2 --channel Combined --method ParticleNet
```

### Important Paths

| Item | Path |
|------|------|
| Scripts | `./scripts/` |
| Python tools | `./python/` |
| Templates output | `./templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/` |
| Sample files | `./samples/$ERA/$CHANNEL/$MASSPOINT/$METHOD/` |
| SKNano input | `$WORKDIR/SKNanoOutput/SignalRegion/$ERA/` |

### Common Commands

```bash
# Check environment
echo $CMSSW_BASE
which combine

# Find all datacards
find templates/ -name "datacard.txt"

# Find all limit files
find templates/ -name "*AsymptoticLimits*.root"

# Extract limit from file
root -l -q 'templates/FullRun2/Combined/MHc130_MA90/Shape/ParticleNet/higgsCombineTest.AsymptoticLimits.mH120.root' \
    -e 'limit->Scan("quantileExpected:limit")'

# Check combine version
combine --version
```

---

## Related Documentation

- **[WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)** - Template creation workflows (Steps 1-4)
- **[API_REFERENCE.md](API_REFERENCE.md)** - Python script and C++ class API
- **[PARTICLENET_WORKFLOW.md](PARTICLENET_WORKFLOW.md)** - ParticleNet methodology
- **[NEGATIVE_BINS_HANDLING.md](NEGATIVE_BINS_HANDLING.md)** - Technical handling of negative bins
- **[PROJECT_INDEX.md](PROJECT_INDEX.md)** - Complete project overview
- **[CMS Combine Documentation](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/)** - Official Combine guide

---

**Document Version**: 1.0
**Last Updated**: 2025-11-14
**Maintained by**: ChargedHiggsAnalysisV3 Team
