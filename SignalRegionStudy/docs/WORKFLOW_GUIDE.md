# SignalRegionStudy - Workflow Guide

**Version**: 1.0
**Last Updated**: 2025-10-10
**Audience**: Analyzers performing charged Higgs signal region optimization

---

## Table of Contents

- [Quick Start](#quick-start)
- [Standard Workflows](#standard-workflows)
- [Advanced Workflows](#advanced-workflows)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

---

## Quick Start

### Prerequisites Checklist

```bash
# 1. Verify CMSSW environment
source setup.sh
echo $CMSSW_BASE  # Should show CMSSW_14_1_0_pre4 path

# 2. Verify ROOT with RooFit
root-config --version  # Should be 6.30.07 or higher
root -l -q -e 'gSystem->Load("libRooFit")'  # Should load without errors

# 3. Build SignalRegionStudy library
./scripts/build.sh
ls lib/libSignalRegionStudy.so  # Should exist

# 4. Verify input data structure
ls $WORKDIR/SKNanoOutput/SignalRegion/2022/  # Should show sample directories
```

### 5-Minute Demo

```bash
# Setup
source setup.sh

# Process single masspoint for 2022, 1E2Mu channel
ERA="2022"
CHANNEL="SR1E2Mu"
MASSPOINT="MHc130_MA90"
METHOD="ParticleNet"

# Full workflow (4 steps)
preprocess.py --era $ERA --channel Skim1E2Mu --signal $MASSPOINT

makeBinnedTemplates.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method $METHOD

checkTemplates.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method $METHOD

printDatacard.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method $METHOD

# Output location
ls templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/
# → datacard.txt, shapes.root
```

---

## Standard Workflows

### Workflow 1: Single Masspoint Analysis

**Use Case**: Analyze one signal hypothesis without score optimization

#### Step 1: Preprocessing

```bash
#!/bin/bash
# preprocess_single.sh

ERA="2022"
CHANNEL_SKIM="SR1E2Mu"  # Input channel from SKNanoOutput
SIGNAL="MHc130_MA90"

preprocess.py --era $ERA --channel $CHANNEL --signal $SIGNAL
```

**What This Does**:
- Loads `Preprocessor` C++ class via ROOT
- Reads trees from `$WORKDIR/SKNanoOutput/SignalRegion/$ERA/$CHANNEL_SKIM/`
- Applies signal cross-section normalization (÷3 for 5 fb)
- Applies conversion electron scale factors (1E2Mu channel)
- Extracts mass variables according to channel logic:
  - **1E2Mu**: Single entry per event (mass1)
  - **3Mu**: Two entries per event (mass1, mass2)
- Outputs to `samples/$ERA/SR{channel}/$SIGNAL/`

**Output Structure**:
```
samples/2022/SR1E2Mu/MHc130_MA90/
├── nonprompt.root      # Fake lepton background
├── diboson.root        # WZ, ZZ backgrounds
├── ttX.root            # ttW, ttZ, ttH, tZq background
├── MHc130_MA90.root    # Signal
└── ... (other backgrounds)
```

---

#### Step 2: Template Creation

```bash
#!/bin/bash
# make_templates.sh

ERA="2017"
CHANNEL="SR1E2Mu"
MASSPOINT="MHc130_MA90"
METHOD="Baseline"  # Currently: Baseline only (ParticleNet/GBDT: future)

makeBinnedTemplates.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method $METHOD
```

**What This Does**:

**Current Implementation: Baseline Method**
- Simple mass-based binning (no ML optimization)
- ParticleNet/GBDT methods deferred to future release

**Processing Steps**:

1. **A Mass Fitting** (using `AmassFitter` C++ class):
   - Fits Voigtian (Breit-Wigner ⊗ Gaussian) to signal mass distribution
   - Extracts three parameters:
     - **mA**: Fitted peak position (e.g., 89.84 GeV for nominal 90 GeV)
     - **Γ (width)**: Natural decay width (e.g., 0.893 GeV)
     - **σ (sigma)**: Detector resolution (e.g., 0.823 GeV)
   - Saves fit visualization: `fit_result.png`

2. **Binning Calculation**:
   - Formula: `mass_range = 5 × √(Γ² + σ²)`
   - Example: `5 × √(0.893² + 0.823²) = 6.07 GeV`
   - Result: `[mA - mass_range, mA + mass_range]` with 15 bins
   - Typical bin width: ~0.8 GeV

3. **Histogram Creation** (using RDataFrame):
   - Uses `mass` branch (already selected in preprocessing)
   - Works for both **SR1E2Mu** and **SR3Mu** channels
   - No channel-specific mass logic needed at this stage

4. **Systematic Processing**:
   - **Signal**: All prompt systematics (L1Prefire, PileupReweight, MuonIDSF, ElectronIDSF, EMuTrigSF, etc.)
   - **Nonprompt**: Weight-based variations (Nonprompt_Up/Down only)
   - **Conversion**: All prompt systematics (ConvSF as rate uncertainty in datacard)
   - **Diboson, ttX, Others**: All prompt systematics
   - **Theory systematics**: Not yet implemented (PDF, Scale, PS deferred)

5. **Background Categories**:
   - `nonprompt`: Data-driven fake lepton background
   - `conversion`: Conversion electron background (DYJets, TTG, WWG)
   - `diboson`: WZ, ZZ
   - `ttX`: ttW, ttZ, ttH, tZq
   - `others`: Rare processes (WWW, WWZ, WZZ, ZZZ, VH, tHq, TTTT)

6. **data_obs Creation**:
   - Sum of all background histograms
   - Used for unblinding and pseudo-experiments

**Output Structure**:
```
templates/2017/SR1E2Mu/MHc130_MA90/Shape/Baseline/
├── shapes.root        # All templates (~159 histograms)
├── fit_result.root    # RooFit workspace
└── fit_result.png     # Fit visualization
```

**Histogram Naming Convention**:
```
{process}                      # Central histogram
{process}_{systematic}Up       # Systematic up variation
{process}_{systematic}Down     # Systematic down variation

Examples:
- MHc130_MA90                      # Signal central
- MHc130_MA90_MuonIDSF_Up
- MHc130_MA90_BtagSF_HFcorr_Down
- nonprompt                        # Background central
- nonprompt_Nonprompt_Up
- diboson_L1Prefire_Down
- data_obs                         # Sum of all backgrounds
```

**Histogram Inventory** (typical output):
- **Signal**: 1 central + 30 systematic variations = 31 histograms
- **Nonprompt**: 1 central + 2 variations = 3 histograms
- **Conversion**: 1 central + 30 systematic variations = 31 histograms
- **Diboson**: 1 central + 30 systematic variations = 31 histograms
- **ttX**: 1 central + 30 systematic variations = 31 histograms
- **Others**: 1 central + 30 systematic variations = 31 histograms
- **data_obs**: 1 histogram
- **Total**: ~159 histograms

**Example Output** (2017, SR1E2Mu, MHc130_MA90, Baseline):
```
INFO:root:Fit results: mA = 89.84 GeV, Γ = 0.893 GeV, σ = 0.823 GeV
INFO:root:Mass window: [83.76, 95.91] GeV
INFO:root:Bin width: 0.810 GeV
INFO:root:Process yields:
INFO:root:  Signal (MHc130_MA90):      8.6779
INFO:root:  Nonprompt:                26.8971
INFO:root:  Conversion:                0.3643
INFO:root:  Diboson:                  25.3097
INFO:root:  ttX:                      50.8969
INFO:root:  Others:                    2.2112
INFO:root:  Total background:        105.6792
INFO:root:  S/B ratio:                 0.0821
```

**Expected Runtime**: ~2-5 minutes per masspoint

**Known Limitations**:
- Theory systematics (PDF, Scale, PS) not yet available in preprocessed files
- ConvSF treated as rate uncertainty (not shape systematic in templates)
- ParticleNet/GBDT methods not yet implemented

---

#### Step 3: Template Validation

```bash
#!/bin/bash
# check_templates.sh

ERA="2017"
CHANNEL="SR1E2Mu"
MASSPOINT="MHc130_MA90"
METHOD="Baseline"

python/checkTemplates.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method $METHOD
```

**What This Does**:

**Validation Checks**:

1. **Histogram Integrity**:
   - Existence: All expected histograms present
   - Non-zero integrals: No empty histograms
   - Positive yields: All process yields > 0
   - Bin contents: No negative bins (critical for HiggsCombine)

2. **Systematic Coverage**:
   - All prompt systematics present for signal and MC backgrounds
   - Nonprompt weight variations (Nonprompt_Up/Down)
   - Up/Down variations properly paired
   - Variation magnitudes within [0.5, 2.0] × nominal

3. **Diagnostic Plots** (CMS publication style):
   - **background_stack.png**: Stacked backgrounds with data_obs overlay using ComparisonCanvas
     - Includes ratio panel (Data / Prediction)
     - CMS headers with era and luminosity labels
   - **signal_vs_background.png**: Signal vs total background comparison using KinematicCanvas
     - Shows S/B ratio and event yields
     - Professional palette and formatting
   - **systematic_*.png**: Four separate plots for first 4 systematics using KinematicCanvas
     - Each plot shows Central, Up (+X%), and Down (−X%) variations
     - Automatic percentage calculation for systematic impacts

4. **Validation Report**:
   - Process yields summary
   - S/B ratio calculation
   - List of validation issues (critical failures)
   - List of warnings (non-critical anomalies)
   - Total histogram count

**Validation Criteria**:
```python
# Critical checks (must pass):
- Histogram integral > 0 for all processes
- No NaN or Inf bin contents
- Tree structures consistent across systematics

# Warning checks (flagged but non-blocking):
- Systematic variations > 2.0× or < 0.5× nominal
- Highly asymmetric uncertainties (>50% asymmetry)
- Low statistics (< min_entries threshold)
```

**Output Structure**:
```
templates/2017/SR1E2Mu/MHc130_MA90/Shape/Baseline/validation/
├── background_stack.png           # Stacked background plot with ratio panel (ComparisonCanvas)
├── signal_vs_background.png       # Signal overlaid on background (KinematicCanvas)
├── systematic_L1Prefire.png       # L1Prefire systematic variations (KinematicCanvas)
├── systematic_PileupReweight.png  # PileupReweight systematic variations (KinematicCanvas)
├── systematic_MuonIDSF.png        # MuonIDSF systematic variations (KinematicCanvas)
├── systematic_ElectronIDSF.png    # ElectronIDSF systematic variations (KinematicCanvas)
└── validation_report.txt          # Text summary report
```

**Example Output** (2017, SR1E2Mu, MHc130_MA90, Baseline):
```
INFO:root:Starting template validation for MHc130_MA90, 2017, SR1E2Mu, Baseline
INFO:root:Validating histogram integrity...
INFO:root:Validating systematic variations...
INFO:root:Generating diagnostic plots...
INFO:root:Creating background stack plot
INFO:root:Background stack saved: validation/background_stack.png
INFO:root:Total background yield: 105.68
INFO:root:Creating signal vs background plot
INFO:root:Signal vs background plot saved: validation/signal_vs_background.png
INFO:root:Creating systematic variation plots
INFO:root:Created 4 systematic variation plots
INFO:root:============================================================
INFO:root:Validation complete!
INFO:root:Report: validation/validation_report.txt
INFO:root:============================================================
INFO:root:Critical issues: 31
INFO:root:Warnings: 0
ERROR:root:✗ Validation FAILED
ERROR:root:  - conversion: Negative bin content at bin 4 (-0.0040)
ERROR:root:  - conversion_L1Prefire_Up: Negative bin content at bin 4 (-0.0046)
ERROR:root:  ... and 26 more issues
INFO:root:============================================================
```

**validation_report.txt Content**:
```
================================================================================
Template Validation Report
================================================================================
Era:        2017
Channel:    SR1E2Mu
Masspoint:  MHc130_MA90
Method:     Baseline

Process Yields:
  MHc130_MA90         :     8.6779 events
  nonprompt           :    26.8971 events
  conversion          :     0.3643 events
  diboson             :    25.3097 events
  ttX                 :    50.8969 events
  others              :     2.2112 events
  data_obs            :   105.6792 events

Signal/Background ratio: 0.0821

Validation Issues:
  ✗ conversion: Negative bin content at bin 4 (-0.0040)
  ✗ conversion_MuonEn_Up: Negative bin content at bin 6 (-0.0933)
  ... (31 total)

Warnings:
  ✓ No warnings

Total histograms: 159
================================================================================
```

**Common Issues**:

1. **Negative Bin Contents** (as shown above):
   - **Important**: Negative bins are **ACCEPTABLE** and physically meaningful (interference effects)
   - **What's NOT acceptable**: Negative histogram integrals (total normalizations)
   - **Automatic Handling**: Our framework automatically ensures positive integrals
   - **Validation reports**: May show "negative bin" warnings - these are informational only
   - **Action Required**: None - automatic fixes are applied during template generation
   - **For Details**: See [NEGATIVE_BINS_HANDLING.md](NEGATIVE_BINS_HANDLING.md) for comprehensive explanation

2. **Large Systematic Variations**:
   - **Cause**: Incorrect scale factor application or extreme weight fluctuations
   - **Investigation**: Check bin-by-bin ratios, verify SF ranges
   - **Action**: Review preprocessing step, check systematics.json

3. **Missing Systematics**:
   - **Cause**: Systematic not available in preprocessed files
   - **Action**: Rerun preprocessing with correct systematic list

**Expected Runtime**: ~30-60 seconds per masspoint

**Next Steps**:
- If validation **PASSED**: Proceed to datacard generation (Step 4)
- If validation **FAILED**: Address critical issues before continuing
  - Negative bins: Most common issue, especially in conversion background
  - Missing histograms: Check preprocessing step
  - Large variations: Investigate scale factors

---

#### Step 4: Datacard Generation

```bash
#!/bin/bash
# generate_datacard.sh

ERA="2022"
CHANNEL="SR1E2Mu"
MASSPOINT="MHc130_MA90"
METHOD="ParticleNet"

DATACARD_PATH="templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt"

# Generate datacard
printDatacard.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method $METHOD

echo "Datacard created: $DATACARD_PATH"
```

**What This Does**:
- Reads systematic configuration from `configs/systematics.json`
- Generates HiggsCombine-compatible datacard
- Includes all systematic uncertainties (experimental + data-driven + normalization + theory)
- Automatically applies era suffixes for uncorrelated uncertainties
- Saves directly to template directory

**Configuration-Driven**:
All systematics managed in single configuration file - no hardcoded systematic lists

**Expected Runtime**: <1 minute

---

### Workflow 2: Multi-Masspoint Batch Processing

**Use Case**: Process multiple signal hypotheses in parallel

```bash
#!/bin/bash
# batch_process_masspoints.sh

source setup.sh

ERA="2022"
CHANNEL_SKIM="Skim1E2Mu"
CHANNEL="SR1E2Mu"
METHOD="ParticleNet"

# Define masspoint grid
MASSPOINTS=(
    "MHc100_MA60"
    "MHc100_MA70"
    "MHc100_MA80"
    "MHc130_MA90"
    "MHc130_MA100"
    "MHc130_MA110"
    "MHc160_MA120"
    "MHc160_MA130"
)

# Process in parallel (4 jobs)
export ERA CHANNEL_SKIM CHANNEL METHOD

process_masspoint() {
    local MASSPOINT=$1
    echo "Processing $MASSPOINT..."

    # Step 1: Preprocess
    preprocess.py --era $ERA --channel $CHANNEL_SKIM --signal $MASSPOINT

    # Step 2: Templates
    makeBinnedTemplates.py --era $ERA --channel $CHANNEL \
        --masspoint $MASSPOINT --method $METHOD

    # Step 3: Validation
    checkTemplates.py --era $ERA --channel $CHANNEL \
        --masspoint $MASSPOINT --method $METHOD

    # Step 4: Datacard
    printDatacard.py --era $ERA --channel $CHANNEL \
        --masspoint $MASSPOINT --method $METHOD \
        >> templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt

    echo "Completed $MASSPOINT"
}

export -f process_masspoint

# GNU parallel execution (4 concurrent jobs)
parallel -j 4 process_masspoint ::: "${MASSPOINTS[@]}"

echo "Batch processing complete!"
```

**Performance Optimization**:
- Parallel processing: 4 masspoints simultaneously
- Total time: ~15-20 minutes for 8 masspoints (vs ~60 minutes sequential)
- Memory usage: ~4 GB per job (ensure 16 GB+ available)

---

### Workflow 3: Era Combination

**Use Case**: Combine multiple run periods for final result

```bash
#!/bin/bash
# combine_eras.sh

CHANNEL="SR1E2Mu"
MASSPOINT="MHc130_MA90"
METHOD="ParticleNet"

ERAS=("2022" "2022EE" "2023" "2023BPix")

# Process each era
for ERA in "${ERAS[@]}"; do
    echo "Processing era: $ERA"

    preprocess.py --era $ERA --channel Skim1E2Mu --signal $MASSPOINT

    makeBinnedTemplates.py --era $ERA --channel $CHANNEL \
        --masspoint $MASSPOINT --method $METHOD

    checkTemplates.py --era $ERA --channel $CHANNEL \
        --masspoint $MASSPOINT --method $METHOD
done

# Combine datacards
combineCards.py \
    2022=templates/2022/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt \
    2022EE=templates/2022EE/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt \
    2023=templates/2023/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt \
    2023BPix=templates/2023BPix/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt \
    > templates/Run3/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard_combined.txt

echo "Era combination complete!"
```

**Combined Datacard Structure**:
```
imax 4  number of channels (one per era)
jmax 4  number of backgrounds
kmax *  number of nuisance parameters

---
shapes * 2022    2022/shapes.root    $PROCESS_SR1E2Mu_$SYSTEMATIC
shapes * 2022EE  2022EE/shapes.root  $PROCESS_SR1E2Mu_$SYSTEMATIC
shapes * 2023    2023/shapes.root    $PROCESS_SR1E2Mu_$SYSTEMATIC
shapes * 2023BPix 2023BPix/shapes.root $PROCESS_SR1E2Mu_$SYSTEMATIC

---
bin          2022     2022EE   2023     2023BPix
observation  150.5    85.2     180.3    120.7
...
```

---

## Advanced Workflows

### Workflow 4: Score Optimization with Retraining

**Use Case**: Update ML discriminator scores after model retraining

#### Background

When ML models are retrained with new features or hyperparameters, scores must be updated in analysis trees. The `--update` flag enables this workflow.

#### Process

```bash
#!/bin/bash
# optimize_scores.sh

ERA="2022"
CHANNEL_SKIM="Skim1E2Mu"
CHANNEL="SR1E2Mu"
MASSPOINT="MHc130_MA90"
METHOD="ParticleNet"

# Step 0: Clear old scores (IMPORTANT!)
rm -rf samples/$ERA/$CHANNEL/$MASSPOINT/*.root
echo "Cleared old samples"

# Step 1: Preprocess with fresh data
preprocess.py --era $ERA --channel $CHANNEL_SKIM --signal $MASSPOINT
echo "Preprocessing complete"

# Step 2: Generate templates with score updates
makeBinnedTemplates.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method $METHOD --update
echo "Templates with updated scores created"

# Step 3: Validate updated templates
checkTemplates.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method $METHOD

# Step 4: Generate datacard
printDatacard.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method $METHOD

echo "Score optimization complete!"
```

**What `--update` Does**:
1. Loads updated ML model from `models/$METHOD/`
2. Re-evaluates scores for all events
3. Updates `scoreX`, `scoreY`, `scoreZ` branches in sample ROOT files
4. Regenerates templates with new score bins

**Critical Notes**:
- **Always clear old samples** before updating (prevents version mismatch)
- Updated scores persist in `samples/` directory
- To revert: delete `samples/` and rerun without `--update`

---

### Workflow 5: Systematic Uncertainty Profiling

**Use Case**: Identify dominant systematic uncertainties for optimization

```bash
#!/bin/bash
# profile_systematics.sh

ERA="2022"
CHANNEL="SR1E2Mu"
MASSPOINT="MHc130_MA90"
METHOD="ParticleNet"

DATACARD="templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt"

# Run Combine impact analysis
cd $CMSSW_BASE/src
cmsenv

# Initial fit
text2workspace.py $DATACARD -o workspace.root

# Compute impacts
combineTool.py -M Impacts -d workspace.root -m 90 \
    --doInitialFit --robustFit 1

combineTool.py -M Impacts -d workspace.root -m 90 \
    --doFits --robustFit 1 --parallel 8

combineTool.py -M Impacts -d workspace.root -m 90 \
    -o impacts.json

# Plot results
plotImpacts.py -i impacts.json -o impacts_plot

echo "Impact plot: impacts_plot.pdf"
```

**Output Interpretation**:
- **Impact**: Change in signal strength (μ) when nuisance varies by ±1σ
- **Pull**: Best-fit nuisance value relative to prior (should be ~0)
- **Constraint**: Post-fit uncertainty relative to prior (should be <1)

**Typical Results**:
```
Top 5 Impacts (example):
1. BtagSF       ±0.15  (largest uncertainty)
2. MuonIDSF     ±0.08
3. JetES        ±0.06
4. PileupRW     ±0.04
5. Luminosity   ±0.03
```

**Optimization Strategy**:
- Focus measurement efforts on high-impact systematics
- Consider additional control regions for top uncertainties
- Validate stability across different methods

---

### Workflow 6: Signal Region Boundary Optimization

**Use Case**: Optimize kinematic cuts for maximum sensitivity

```python
#!/usr/bin/env python3
# optimize_sr_boundaries.py

import ROOT
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Load preprocessed data
def load_data(era, channel, masspoint):
    signal_file = ROOT.TFile(f"samples/{era}/{channel}/{masspoint}/{masspoint}.root")
    bkg_file = ROOT.TFile(f"samples/{era}/{channel}/backgrounds_combined.root")

    signal_tree = signal_file.Get("Events_Central")
    bkg_tree = bkg_file.Get("Events_Central")

    return signal_tree, bkg_tree

# Compute significance
def compute_significance(signal, background):
    """Asimov significance: s/sqrt(b)"""
    if background <= 0:
        return 0
    return signal / np.sqrt(background)

# Optimize mass window
def optimize_mass_window(signal_tree, bkg_tree, mA_nominal):
    mass_windows = np.arange(5, 25, 1)  # 5-25 GeV window

    significances = []
    for window in mass_windows:
        mass_min = mA_nominal - window
        mass_max = mA_nominal + window

        cut = f"mass > {mass_min} && mass < {mass_max}"

        s = signal_tree.GetEntries(cut)
        b = bkg_tree.GetEntries(cut)

        sig = compute_significance(s, b)
        significances.append(sig)

    # Find optimal window
    optimal_idx = np.argmax(significances)
    optimal_window = mass_windows[optimal_idx]

    # Plot results
    plt.figure(figsize=(8, 6))
    plt.plot(mass_windows, significances, 'o-')
    plt.axvline(optimal_window, color='red', linestyle='--',
                label=f'Optimal: ±{optimal_window} GeV')
    plt.xlabel('Mass Window [GeV]')
    plt.ylabel('Significance (s/√b)')
    plt.title(f'Mass Window Optimization (mA={mA_nominal} GeV)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'optimization/mass_window_mA{int(mA_nominal)}.pdf')

    return optimal_window

# Main optimization
if __name__ == "__main__":
    era = "2022"
    channel = "SR1E2Mu"
    masspoint = "MHc130_MA90"
    mA_nominal = 90.0

    signal_tree, bkg_tree = load_data(era, channel, masspoint)

    optimal_window = optimize_mass_window(signal_tree, bkg_tree, mA_nominal)

    print(f"Optimal mass window: mA ± {optimal_window} GeV")
    print(f"Cut string: mass > {mA_nominal - optimal_window} && "
          f"mass < {mA_nominal + optimal_window}")
```

**Usage**:
```bash
python3 optimize_sr_boundaries.py
# Output: optimization/mass_window_mA90.pdf
```

---

### Workflow 7: Full Statistical Analysis

**Use Case**: Run complete statistical interpretation with Combine

```bash
#!/bin/bash
# full_statistical_analysis.sh

ERA="2022"
CHANNEL="SR1E2Mu"
MASSPOINT="MHc130_MA90"
METHOD="ParticleNet"

DATACARD="templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt"
WORKSPACE="workspace_${MASSPOINT}.root"

cd $CMSSW_BASE/src
cmsenv

# Step 1: Create workspace
text2workspace.py $DATACARD -o $WORKSPACE

# Step 2: Observed limit
combine -M AsymptoticLimits $WORKSPACE -m 90 -n Observed

# Step 3: Expected limit with bands
combine -M AsymptoticLimits $WORKSPACE -m 90 -n Expected --run expected

# Step 4: Significance
combine -M Significance $WORKSPACE -m 90 -n Significance --uncapped 1

# Step 5: Best-fit signal strength
combine -M FitDiagnostics $WORKSPACE -m 90 -n FitDiag --saveShapes --saveWithUncertainties

# Step 6: Likelihood scan
combine -M MultiDimFit $WORKSPACE -m 90 -n Scan --algo grid \
    --points 100 --setParameterRanges r=-2,5

# Collect results
echo "=== RESULTS ==="
echo "Observed limit:"
cat higgsCombineObserved.AsymptoticLimits.mH90.root | grep "Limit"

echo "Expected limit:"
cat higgsCombineExpected.AsymptoticLimits.mH90.root | grep "Limit"

echo "Significance:"
cat higgsCombineSignificance.Significance.mH90.root | grep "Significance"
```

**Typical Output**:
```
=== RESULTS ===
Observed limit: 0.85 (95% CL on signal strength)
Expected limit: 1.02 ± 0.30
  Expected -2σ: 0.45
  Expected -1σ: 0.65
  Expected median: 1.02
  Expected +1σ: 1.50
  Expected +2σ: 2.10
Significance: 1.85σ
```

**Interpretation**:
- **Observed < 1**: Data disfavors signal (exclusion)
- **Expected ~ 1**: Analysis has sensitivity to SM-like signal
- **Significance**: Evidence level (>3σ for evidence, >5σ for discovery)

---

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "Could not find Events_Central tree"

**Symptoms**:
```
Error in <TFile::Get>: Events_Central not found in file
```

**Causes**:
1. Input file not preprocessed yet
2. Wrong file path
3. Corrupted ROOT file

**Solutions**:
```bash
# Check file contents
root -l input.root
.ls  # List objects in file

# Verify tree exists
TTree *tree = (TTree*)gDirectory->Get("Events_Central");
tree->Print();

# If missing, rerun preprocessing
preprocess.py --era 2022 --channel Skim1E2Mu --signal MHc130_MA90
```

---

#### Issue 2: "Systematic variation has zero entries"

**Symptoms**:
```
Warning: Events_MuonIDSFUp has 0 entries
```

**Causes**:
1. Systematic not available in input
2. Selection cuts too tight for variation
3. Weight branch contains NaN/Inf

**Solutions**:
```python
# Check input file systematics
import ROOT
f = ROOT.TFile("input.root")
f.ls()  # List all trees

# Verify weight branch
tree = f.Get("Events_MuonIDSFUp")
tree->Scan("weight", "", "", 10)  # Print first 10 weights

# If weights invalid, investigate source
tree->Draw("weight >> h(100, -1, 5)")
# Look for unusual distribution
```

**Workaround**:
- Skip problematic systematic
- Use symmetric uncertainty (if only Up or Down available)
- Investigate upstream SKNano production

---

#### Issue 3: "Bogus norm" error from text2workspace.py

**Symptoms**:
```
RuntimeError: Bogus norm -0.004458621144294739 for channel signal_region,
              process conversion, systematic PileupReweight Down
```

**What This Means**:
- A histogram has **negative total integral** (sum of all bins ≤ 0)
- This is different from negative bins (which are acceptable)
- Makes the likelihood calculation undefined

**Automatic Fix**:
- Our framework automatically handles this in `makeBinnedTemplates.py`
- If you see this error, automatic handling may have been bypassed

**Solutions**:
```bash
# Solution 1: Regenerate templates (recommended)
rm -rf templates/2018/SR1E2Mu/MHc70_MA15/
./scripts/prepareCombine.sh 2018 SR1E2Mu MHc70_MA15 Baseline

# Solution 2: Check for manual edits
# Did you manually edit shapes.root or datacard.txt?
# If yes, regenerate from scratch

# Solution 3: Verify automatic handling is present
grep -n "ensurePositiveIntegral" python/makeBinnedTemplates.py
# Should show function definition at line ~135
```

**For More Details**: See [NEGATIVE_BINS_HANDLING.md](NEGATIVE_BINS_HANDLING.md)

---

#### Issue 4: "Combine fit fails to converge"

**Symptoms**:
```
Best fit failed with status != 0
Covariance matrix not positive definite
```

**Causes**:
1. Empty bins in templates (should not occur with automatic handling)
2. Inconsistent systematic variations
3. Overparameterized fit (too many nuisances)

**Solutions**:
```bash
# Diagnose with simplified fit
combine -M FitDiagnostics datacard.txt --robustFit 1 --stepSize 0.1

# Check template integrity
checkTemplates.py --era 2022 --channel SR1E2Mu \
    --masspoint MHc130_MA90 --method ParticleNet

# Reduce nuisance parameters
# Edit datacard: comment out low-impact systematics
# Start with only shape systematics, add normalization incrementally
```

**Debugging Combine**:
```bash
# Verbose output
combine -M FitDiagnostics datacard.txt -v 9

# Save fit diagnostics
combine -M FitDiagnostics datacard.txt --saveShapes \
    --saveNormalizations --plots

# Inspect fit results
root -l fitDiagnostics.root
fit_s->Print("V")  # Print all parameters
covariance->Print()  # Check correlations
```

---

#### Issue 5: "Score branches not found"

**Symptoms**:
```
Error: Branch score_MHc130_MA90_vs_nonprompt not found
```

**Causes**:
1. `isTrainedSample=false` in preprocessing
2. Score naming mismatch (signal name different from expected)
3. ML model not applied to input

**Solutions**:
```bash
# Verify branch existence
root -l samples/2022/SR1E2Mu/MHc130_MA90/MHc130_MA90.root
Events_Central->Print()  # List all branches

# If scores missing, rerun with training flag
preprocess.py --era 2022 --channel Skim1E2Mu --signal MHc130_MA90 --trained

# Check score naming convention
# Expected: score_{signal}_vs_{background}
# Example: score_MHc130_MA90_vs_nonprompt
```

---

#### Issue 6: "TTree::SetBranchAddress type mismatch"

**Symptoms**:
```
Error in <TTree::SetBranchAddress>: The pointer type given "Double_t" (8)
does not correspond to the type needed "Float_t" (5) by the branch: mass
```

**Cause**:
AmassFitter.cc used incorrect data type (Double_t) for branch reading when branches are stored as Float_t

**Solution** (Already fixed in current version):

The issue was in `src/AmassFitter.cc`. The correct implementation uses `float` instead of `double`:

```cpp
// CORRECT (current implementation):
float mass; tree->SetBranchAddress("mass", &mass);
float weight; tree->SetBranchAddress("weight", &weight);

// INCORRECT (old implementation):
double mass; tree->SetBranchAddress("mass", &mass);     // Type mismatch!
double weight; tree->SetBranchAddress("weight", &weight); // Type mismatch!
```

**If you encounter this error**:
1. Verify you have the latest version of AmassFitter.cc
2. Check lines 14-15 in src/AmassFitter.cc
3. Rebuild the library:
   ```bash
   ./scripts/build.sh
   ```

**Prevention**:
- Always check branch data types before setting branch addresses:
  ```cpp
  TTree *tree = file->Get("Central");
  tree->Print();  // Shows branch types
  ```
- Use ROOT's type-checking: `tree->SetBranchAddress()` will warn about mismatches

---

### Performance Optimization Tips

#### Memory Usage

**Problem**: Script crashes with "out of memory" error

**Solutions**:
```bash
# Process in smaller chunks
for SAMPLE in signal background1 background2; do
    preprocess_sample.py --sample $SAMPLE
done

# Use batch processing instead of all-at-once
parallel -j 2 process_masspoint ::: "${MASSPOINTS[@]}"
# Reduce from -j 4 to -j 2 (trades speed for memory)

# Monitor memory during execution
watch -n 1 'free -h; ps aux | grep python'
```

---

#### Slow RooFit Fitting

**Problem**: `AmassFitter::fitMass()` takes >10 minutes

**Solutions**:
```cpp
// In AmassFitter.cc, enable NumCPU
fit_result = roo_model->fitTo(*roo_data,
                               SumW2Error(kTRUE),
                               NumCPU(4),  // Parallel processing
                               Save());

// Tighten fit range (fewer data points)
// Before: fitMass(90.0, 60.0, 120.0)  // ±30 GeV
// After:  fitMass(90.0, 75.0, 105.0)  // ±15 GeV

// Reduce dataset size (testing only!)
RooDataSet *subset = (RooDataSet*)roo_data->reduce(EventRange(0, 10000));
```

---

## Best Practices

### Reproducibility

#### 1. Version Control Configuration
```bash
# Track configuration files
git add configs/*.json
git commit -m "Add histogram and systematic configurations"

# Document masspoint parameters
echo "MHc130_MA90: mH+=130 GeV, mA=90 GeV" >> masspoints.txt
```

#### 2. Environment Documentation
```bash
# Save environment snapshot
conda env export > environment.yml

# Document ROOT/CMSSW versions
root-config --version > versions.txt
scram version >> versions.txt
```

#### 3. Result Archiving
```bash
# Archive complete analysis
tar -czf analysis_2022_SR1E2Mu_$(date +%Y%m%d).tar.gz \
    samples/2022/SR1E2Mu/ \
    templates/2022/SR1E2Mu/ \
    configs/

# Store on CMS storage
xrdcp analysis_*.tar.gz root://eosuser.cern.ch//eos/user/u/username/
```

---

### Validation Strategy

#### 1. Template Sanity Checks
```python
# check_template_sanity.py
import ROOT

def validate_template(file_path, hist_name):
    f = ROOT.TFile(file_path)
    h = f.Get(hist_name)

    # Check 1: Non-zero integral
    assert h.Integral() > 0, f"{hist_name} has zero integral"

    # Check 2: No negative bins
    for i in range(1, h.GetNbinsX() + 1):
        assert h.GetBinContent(i) >= 0, f"{hist_name} bin {i} is negative"

    # Check 3: Reasonable statistics
    assert h.GetEntries() > 10, f"{hist_name} has too few entries"

    print(f"✓ {hist_name} passed validation")

# Run validation
validate_template("templates/2022/SR1E2Mu/MHc130_MA90/Shape/ParticleNet/shapes.root",
                  "MHc130_MA90_SR1E2Mu_Central")
```

#### 2. Systematic Variation Checks
```bash
# Check systematic symmetry
checkSystematics.py --era 2022 --channel SR1E2Mu --masspoint MHc130_MA90

# Expected output:
# ✓ MuonIDSF: Up/Down symmetric within 5%
# ✓ BtagSF: Variation within [0.8, 1.2]
# ⚠ PileupRW: Asymmetric (Up: +15%, Down: -8%)
```

---

### Documentation Standards

#### 1. Analysis Log
```markdown
# Analysis Log - MHc130_MA90

## 2025-10-09
- Preprocessed 2022 data for SR1E2Mu
- Applied ConvSF = 1.05 ± 0.10
- Generated ParticleNet templates
- Observed limit: 0.85 (95% CL)

## Issues Encountered
- BtagSF systematic showed unexpected asymmetry → investigated, found upstream issue
- Low statistics in high-score region → merged top two bins

## Next Steps
- Combine with 2023 data
- Optimize score binning for better sensitivity
```

#### 2. Configuration Documentation
```python
# configs/analysis_config.py

# MHc130_MA90 Analysis Configuration
MASSPOINT_CONFIG = {
    "signal_name": "MHc130_MA90",
    "mHc": 130.0,  # GeV
    "mA": 90.0,    # GeV
    "mass_window": 15.0,  # GeV (optimized in mass_window_optimization.py)
    "score_bins": [0.0, 0.3, 0.5, 0.7, 0.85, 1.0],  # ParticleNet score bins
    "systematics": [
        "MuonIDSF", "ElectronIDSF", "BtagSF", "JetES", "JetER",
        "PileupReweight", "L1Prefire", "TriggerSF"
    ]
}
```

---

### Collaboration Guidelines

#### 1. Naming Conventions
```bash
# Scripts
preprocess_{analysis}.py
make_templates_{method}.py
optimize_{parameter}.py

# Output directories
templates/{era}/{channel}/{masspoint}/Shape/{method}/
results/{analysis_tag}/{date}/

# Datacards
datacard_{era}_{channel}_{masspoint}_{method}.txt
```

#### 2. Code Review Checklist
- [ ] All hardcoded paths replaced with variables
- [ ] Era/channel/masspoint parameterized
- [ ] Error handling for missing files
- [ ] Validation checks included
- [ ] Documentation strings added
- [ ] Example usage provided

---

**Last Updated**: 2025-10-10
**Maintainer**: ChargedHiggsAnalysisV3 Development Team
**Feedback**: Submit issues to project repository
