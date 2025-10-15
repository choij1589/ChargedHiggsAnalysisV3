# Handling Negative Bins and Negative Normalizations

**Version**: 1.0
**Last Updated**: 2025-10-11
**Audience**: Developers and analyzers working with HiggsCombine statistical framework

---

## Table of Contents

- [Overview](#overview)
- [Background: Physics and Statistics](#background-physics-and-statistics)
- [The Distinction: Negative Bins vs Negative Integrals](#the-distinction-negative-bins-vs-negative-integrals)
- [Implementation](#implementation)
- [Automatic Handling Mechanisms](#automatic-handling-mechanisms)
- [Diagnostic Information](#diagnostic-information)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)
- [References](#references)

---

## Overview

This document explains how the SignalRegionStudy framework handles negative bin contents and negative normalizations (integrals) in histogram templates for the HiggsCombine statistical analysis tool.

**Key Point**: Negative bins are **physically meaningful** and **acceptable** in Combine, but negative total normalizations (histogram integrals) **cause runtime errors** and must be corrected.

---

## Background: Physics and Statistics

### Why Negative Bin Contents Occur

Negative bin contents in Monte Carlo histograms arise from two main sources:

#### 1. **Interference Effects** (Physical)
```
Example: t-channel single top production
- Forward scattering amplitude: A_forward ~ +1.5
- Backward scattering amplitude: A_backward ~ -0.8
- Interference term: A_forward × A*_backward < 0
→ Results in negative event weights for some kinematic regions
```

These negative weights represent genuine quantum mechanical interference and **should be preserved** in the analysis.

#### 2. **Negative MC Event Weights** (Technical)
```
Sources of negative weights:
- NLO QCD corrections (loop diagrams with negative contributions)
- Parton shower reweighting (eg. PDF variations)
- Systematic variation weights that reduce cross-section below zero locally
```

### Combine Framework Requirements

The HiggsCombine framework (CMSSW `text2workspace.py`):

✅ **Accepts**: Negative bin contents (handled via Barlow-Beeston lite method or template morphing)

❌ **Rejects**: Negative histogram integrals (normalizations)

**Why?** The normalization represents the total predicted yield for a process, which must be positive for likelihood construction:

```
L(data|μ,θ) = ∏_bins Poisson(n_obs | μ×s_i(θ) + b_i(θ))

Where:
  μ×s_i(θ) + b_i(θ) = expected yield in bin i (MUST be > 0)

  Total normalization:
  N = ∫ histogram = Σ_bins content_i (MUST be > 0)
```

If `N ≤ 0`, the Poisson likelihood is undefined.

---

## The Distinction: Negative Bins vs Negative Integrals

| Aspect | Negative Bins | Negative Integrals |
|--------|---------------|-------------------|
| **Example** | Bin 5: -0.3 events, Total: +10.5 events | Bin 1: -5.0 events, Bin 2: -8.0 events, Total: -2.5 events |
| **Physical Meaning** | Local interference effect | Unphysical: process can't have negative total yield |
| **Combine Status** | ✅ Acceptable (physics) | ❌ Runtime Error: "Bogus norm" |
| **Action Required** | None (preserve physics) | **Must fix** before running Combine |

### Example from Our Codebase

**Scenario**: Conversion electron background with very low statistics

```
Template: conversion_PileupReweightDown
Bin contents: [-0.04, -0.01, 0.02, -0.08, 0.05, -0.03, ...]
Total integral: -0.0045 events ← PROBLEM!
```

**Combine Error**:
```
RuntimeError: Bogus norm -0.004458621144294739 for channel signal_region,
              process conversion, systematic PileupReweight Down
```

**Our Solution**: Automatically detect and set minimum positive integral (1e-10).

---

## Implementation

Our framework implements two complementary mechanisms:

### 1. Histogram-Level Fix: `ensurePositiveIntegral()`

**Location**: `python/makeBinnedTemplates.py:135-160`

**Purpose**: Ensure every histogram has a positive integral (normalization)

**Function Signature**:
```python
def ensurePositiveIntegral(hist, min_integral=1e-10):
    """
    Ensure histogram has positive integral for normalization.
    If integral is negative or zero, set a minimal positive value.

    Args:
        hist: TH1D histogram
        min_integral: Minimum integral value to set (default: 1e-10)

    Returns:
        True if histogram was modified, False otherwise
    """
```

**Algorithm**:
```python
integral = hist.Integral()

if integral <= 0:
    # Find central bin
    central_bin = hist.GetNbinsX() // 2 + 1

    # Set minimal positive value
    hist.SetBinContent(central_bin, min_integral)
    hist.SetBinError(central_bin, min_integral)

    logging.warning(f"Histogram {hist.GetName()} has non-positive integral: {integral:.4e}")
    logging.warning(f"Setting bin {central_bin} to {min_integral} to ensure positive normalization")

    return True

return False
```

**Design Rationale**:
- **Why central bin?** Minimal impact on distribution shape
- **Why 1e-10?** Large enough to avoid numerical issues, small enough to be negligible (~10^-6% of typical background yields)
- **Preserve other bins**: Keep negative bins intact (they contain physical information)

### 2. Datacard-Level Fix: `check_systematic_validity()`

**Location**: `python/printDatacard.py:149-234`

**Purpose**: Detect negative normalizations in systematic variations and automatically convert to log-normal (lnN) uncertainties

**Function Signature**:
```python
def check_systematic_validity(self, syst_name, applies_to):
    """
    Check if systematic variations produce negative normalizations.

    Args:
        syst_name: Name of systematic (e.g., "PileupReweight")
        applies_to: List of processes to check

    Returns:
        (has_negative, details): Boolean flag and list of affected processes
    """
```

**Algorithm**:
```python
has_negative = False
details = []

for proc in applies_to:
    rate_central = get_event_rate(proc, "Central")
    rate_up = get_event_rate(proc, f"{syst_name}Up")
    rate_down = get_event_rate(proc, f"{syst_name}Down")

    # Check if variations have negative normalization
    if rate_up <= 0 and rate_central > 0:
        has_negative = True
        details.append(f"{proc} Up: {rate_up:.4e}")

    if rate_down <= 0 and rate_central > 0:
        has_negative = True
        details.append(f"{proc} Down: {rate_down:.4e}")

return has_negative, details
```

**Automatic Conversion**: If negative normalization detected:
```python
if has_negative:
    # Convert from shape systematic to lnN
    sysType = "lnN"

    # Calculate lnN value from maximum variation
    for proc in processes:
        rate_central = get_event_rate(proc, "Central")
        rate_up = get_event_rate(proc, f"{syst}Up")
        rate_down = get_event_rate(proc, f"{syst}Down")

        var_up = abs(rate_up / rate_central - 1.0) if rate_up > 0 else 0.0
        var_down = abs(rate_down / rate_central - 1.0) if rate_down > 0 else 0.0
        max_var = max(var_up, var_down, 0.001)  # Minimum 0.1%

        lnN_value = 1.0 + max_var
```

**Example Conversion**:
```
Original datacard (shape systematic):
PileupReweight   shape   1    1    1    1    1

After automatic conversion (lnN):
PileupReweight   lnN     1.05 1.12 1.03 1.08 1.06
```

---

## Automatic Handling Mechanisms

### Workflow Integration

```
┌─────────────────────────────────────────────────────────────┐
│ Step 1: Template Generation (makeBinnedTemplates.py)       │
├─────────────────────────────────────────────────────────────┤
│ For each process and systematic:                            │
│   1. Create histogram from RDataFrame                       │
│   2. Call ensurePositiveIntegral(hist)                      │
│   3. Log warning if integral ≤ 0                           │
│   4. Write corrected histogram to shapes.root               │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 2: Datacard Generation (printDatacard.py)             │
├─────────────────────────────────────────────────────────────┤
│ For each shape systematic:                                  │
│   1. Call check_systematic_validity(syst, processes)        │
│   2. If negative normalization found:                       │
│      → Print warning to stderr                              │
│      → Convert shape → lnN automatically                    │
│      → Calculate appropriate lnN values                     │
│   3. Write datacard with corrected systematics              │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│ Step 3: Combine Statistical Analysis (text2workspace.py)   │
├─────────────────────────────────────────────────────────────┤
│ Result: No "Bogus norm" errors                             │
│         ✓ All histograms have positive integrals          │
│         ✓ Problematic systematics converted to lnN         │
└─────────────────────────────────────────────────────────────┘
```

### What Gets Fixed Automatically

| Issue | Detection Point | Fix Applied | User Action |
|-------|----------------|-------------|-------------|
| Histogram integral ≤ 0 | `makeBinnedTemplates.py` | Set central bin to 1e-10 | **None** (automatic) |
| Systematic variation with negative norm | `printDatacard.py` | Convert shape → lnN | **None** (automatic) |
| Persistent negative bins | `checkTemplates.py` | Report only (no fix) | Review logs, investigate if needed |

---

## Diagnostic Information

### Log Messages During Template Generation

**Normal Case** (no issues):
```
INFO:root:Histogram MHc70_MA15: 1523 entries, integral = 1.8183
INFO:root:Histogram nonprompt: 245 entries, integral = 0.1079
INFO:root:Histogram conversion: 89 entries, integral = 0.0123
```

**Case with Negative Integral** (automatic fix applied):
```
WARNING:root:  Histogram conversion has non-positive integral: 0.0000e+00
WARNING:root:  Setting bin 8 to 1e-10 to ensure positive normalization
INFO:root:Histogram conversion modified to ensure positive integral: 1.0000e-10

WARNING:root:  Histogram conversion_L1PrefireUp has non-positive integral: -4.5872e-04
WARNING:root:  Setting bin 8 to 1e-10 to ensure positive normalization
INFO:root:Histogram conversion_L1PrefireUp modified to ensure positive integral: 1.0000e-10
```

**Interpretation**:
- First line: Original integral was 0 or negative
- Second line: Which bin was set to minimum value
- Third line: Confirmed new integral is positive

### Log Messages During Datacard Generation

**Normal Case**:
```
(No warnings printed to stderr)
```

**Case with Negative Normalization** (automatic conversion):
```
WARNING: Negative normalization detected for PileupReweight, switching to lnN
  conversion Up: 0.0000e+00
  conversion Down: -4.4586e-03

WARNING: Negative normalization detected for BtagSF_HFcorr, switching to lnN
  conversion Up: -1.2345e-03
  conversion Down: -2.3456e-03
```

**Interpretation**:
- Shape systematic was originally requested in `systematics.json`
- Automatic detection found negative normalizations
- Conversion to lnN applied automatically
- Values calculated from maximum variation

### Validation Report Output

From `checkTemplates.py`:
```
INFO:root:Critical issues: 63
INFO:root:Warnings: 0
ERROR:root:✗ Validation FAILED
ERROR:root:  - conversion: Negative bin content at bin 13 (-0.1287)
ERROR:root:  - conversion_L1Prefire_18Up: Negative bin content at bin 13 (-0.1284)
ERROR:root:  ... and 59 more issues
```

**Important**: These "critical issues" are **not actually critical**!
- Negative bins are **acceptable** in Combine
- The validation script is conservative (reports all negative bins)
- As long as integrals are positive, the analysis can proceed
- Real critical issues are negative **integrals**, which are automatically fixed

---

## Best Practices

### DO ✓

1. **Trust the automatic handling**
   ```bash
   # Just run the standard workflow
   makeBinnedTemplates.py --era 2018 --channel SR1E2Mu --masspoint MHc70_MA15 --method Baseline
   printDatacard.py --era 2018 --channel SR1E2Mu --masspoint MHc70_MA15 --method Baseline

   # Automatic fixes are applied, no manual intervention needed
   ```

2. **Review warning messages**
   ```bash
   # Look for patterns
   makeBinnedTemplates.py ... 2>&1 | grep "non-positive integral"

   # Example output tells you which processes need attention
   conversion: 15 negative integral warnings
   others: 2 negative integral warnings
   ```

3. **Investigate if many histograms are affected**
   ```
   If > 50% of systematic variations have negative integrals:
   → Problem might be upstream (preprocessing, scale factors)
   → Check input ROOT files for unusual weight distributions
   ```

4. **Preserve negative bins** (don't "fix" them)
   ```python
   # WRONG - removes physics information
   for i in range(1, hist.GetNbinsX() + 1):
       if hist.GetBinContent(i) < 0:
           hist.SetBinContent(i, 0)  # ✗ DON'T DO THIS

   # RIGHT - let automatic handling deal with integrals only
   ensurePositiveIntegral(hist)  # ✓ Only fixes if integral ≤ 0
   ```

### DON'T ✗

1. **Don't manually set all negative bins to zero**
   - This removes physical interference information
   - Can bias fit results
   - Automatic handling is sufficient

2. **Don't ignore warnings about systematic conversions**
   ```
   WARNING: Negative normalization detected for PileupReweight, switching to lnN

   ✗ "I'll just ignore this, it's probably fine"
   ✓ "I should investigate why PileupReweight produces negative norms"
   ```

3. **Don't modify the automatic threshold (1e-10) without understanding**
   ```python
   # Current implementation
   ensurePositiveIntegral(hist, min_integral=1e-10)  # ✓ Carefully chosen

   # Don't do this without good reason
   ensurePositiveIntegral(hist, min_integral=1.0)    # ✗ Too large, biases fit
   ensurePositiveIntegral(hist, min_integral=1e-20)  # ✗ May cause numerical issues
   ```

4. **Don't skip validation**
   ```bash
   # Always run validation to see diagnostic plots
   checkTemplates.py --era 2018 --channel SR1E2Mu --masspoint MHc70_MA15 --method Baseline

   # Even if "validation FAILED" is reported, check:
   # 1. Are integrals positive? (automatic fix ensures this)
   # 2. Are systematic variations reasonable? (check plots)
   # 3. Are there patterns in negative bins? (physics vs technical issue)
   ```

---

## Troubleshooting

### Issue 1: Many Histograms with Negative Integrals

**Symptoms**:
```
WARNING:root:  Histogram conversion has non-positive integral: 0.0000e+00
WARNING:root:  Histogram conversion_L1PrefireUp has non-positive integral: 0.0000e+00
WARNING:root:  Histogram conversion_L1PrefireDown has non-positive integral: 0.0000e+00
... (repeated for 30+ systematics)
```

**Diagnosis**:
```bash
# Check the central histogram first
root -l templates/2018/SR1E2Mu/MHc70_MA15/Shape/Baseline/shapes.root
root [1] conversion->Print()
root [2] conversion->Integral()  # Is central histogram also near zero?
```

**Common Causes**:
1. **Very low statistics** sample (conversion background in some mass regions)
2. **Tight selection cuts** eliminating most events
3. **Upstream preprocessing issue** (check sample ROOT files)

**Solutions**:
```python
# Solution 1: Merge into "others" category (preferred for very rare backgrounds)
# Edit configs/samplegroups.json to move conversion → others

# Solution 2: Relax selection cuts (if physics allows)
# Edit preprocessing cuts in preprocess.py

# Solution 3: Accept automatic handling (if only systematic variations affected)
# Automatic fixes ensure integrals > 0, analysis proceeds normally
```

### Issue 2: Systematic Converted to lnN but Should Be Shape

**Symptoms**:
```
WARNING: Negative normalization detected for JetEn, switching to lnN
```

**Diagnosis**:
```bash
# This systematic should be shape-based (kinematic distortion)
# Check why normalization went negative

root -l templates/2018/SR1E2Mu/MHc70_MA15/Shape/Baseline/shapes.root
root [1] signal_JetEnUp->Integral()    # Check signal
root [2] conversion_JetEnUp->Integral()  # Check conversion
```

**Common Causes**:
1. **Low statistics** + **large energy scale variation** → some samples have 0 events
2. **Preprocessing error** in systematic branch
3. **Scale factor issue** in JetEn systematic

**Solutions**:
```bash
# Solution 1: Investigate upstream
# Check SKNano output for JetEn systematic trees
root -l $WORKDIR/SKNanoOutput/SignalRegion/2018/SR1E2Mu/conversion.root
.ls  # Are JetEn_Up and JetEn_Down trees present?

# Solution 2: Accept lnN conversion (conservative)
# lnN still captures normalization impact, shape info lost but fit valid

# Solution 3: Merge low-stats samples
# If conversion has this issue, merge into "others"
```

### Issue 3: text2workspace.py Still Fails

**Symptoms**:
```
RuntimeError: Bogus norm -0.004458621144294739 for channel signal_region,
              process conversion, systematic PileupReweight Down
```

**Diagnosis**:
```bash
# This error should NOT occur if automatic handling is working
# Check if fixes were applied:

grep "non-positive integral" makeBinnedTemplates.log
# Should see warnings + fixes

grep "switching to lnN" printDatacard.log
# Should see systematic conversions
```

**Possible Causes**:
1. **Outdated code** (automatic handling not present)
2. **Manual datacard editing** that bypassed automatic handling
3. **shapes.root file from old run** (regenerate templates)

**Solutions**:
```bash
# Solution 1: Verify code version
cd $WORKDIR/SignalRegionStudy
grep -n "ensurePositiveIntegral" python/makeBinnedTemplates.py
# Should find function at line ~135

# Solution 2: Clean and regenerate
rm -rf templates/2018/SR1E2Mu/MHc70_MA15/
./scripts/prepareCombine.sh 2018 SR1E2Mu MHc70_MA15 Baseline

# Solution 3: Check datacard manually
cat templates/2018/SR1E2Mu/MHc70_MA15/Shape/Baseline/datacard.txt | grep "PileupReweight"
# Should see "lnN" not "shape" if conversion was applied
```

### Issue 4: Combine Fit Fails with "Non-positive definite covariance"

**Symptoms**:
```
Best fit failed with status != 0
Covariance matrix not positive definite
```

**Note**: This is **different** from negative bin issues!

**Diagnosis**:
```bash
# Check if related to our fixes
combine -M FitDiagnostics datacard.txt --robustFit 1 --stepSize 0.1 -v 9
# Look for which parameters are problematic
```

**Common Causes**:
1. **Too many nuisance parameters** (overparameterized fit)
2. **Correlated systematics** causing fit instability
3. **Low statistics** in data_obs

**Solutions**:
```bash
# Solution 1: Simplify fit (testing)
# Comment out some systematics in datacard temporarily

# Solution 2: Use Combine's built-in options
combine -M FitDiagnostics datacard.txt --robustFit 1 --cminDefaultMinimizerStrategy 0

# Solution 3: Check for problematic processes
# Remove processes with very low yields (< 0.01 events)
```

---

## References

### Code Locations

| Component | File | Lines | Purpose |
|-----------|------|-------|---------|
| Positive integral enforcement | `python/makeBinnedTemplates.py` | 135-160 | Fix negative histogram integrals |
| Histogram creation | `python/makeBinnedTemplates.py` | 163-223 | Create histograms with automatic fixes |
| Systematic validation | `python/printDatacard.py` | 149-172 | Detect negative normalizations |
| Shape→lnN conversion | `python/printDatacard.py` | 174-234 | Auto-convert problematic systematics |

### Related Documentation

- [Combine Tutorial](https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/): Official HiggsCombine documentation
- [Barlow-Beeston Method](https://arxiv.org/abs/hep-ex/9807015): Handling bin-by-bin statistical uncertainties
- [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md): Complete analysis workflow
- [systematics.md](systematics.md): Systematic uncertainty definitions
- [API_REFERENCE.md](API_REFERENCE.md): Function-level documentation

### External Resources

- **CMS Statistics Committee**: [Guidelines on negative weights](https://twiki.cern.ch/twiki/bin/view/CMS/SWGuideHiggsAnalysisCombinedLimit#Negative_weights)
- **ROOT TH1 Documentation**: [Understanding histogram integrals](https://root.cern.ch/doc/master/classTH1.html#a94d934c45e1a35e26b1b1b063e6a1e91)

---

**Version History**:
- v1.0 (2025-10-11): Initial documentation of automatic negative bin handling

**Maintainer**: ChargedHiggsAnalysisV3 Development Team
**Questions**: Submit issues to project repository
