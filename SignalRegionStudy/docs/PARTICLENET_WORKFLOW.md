# ParticleNet Workflow

**Version**: 2.0
**Last Updated**: 2025-10-16
**Changelog**: Version 2.0 adds cross-section weighted likelihood ratio formulation

---

## Overview

The ParticleNet method uses deep learning scores to improve signal/background separation beyond kinematic cuts alone. The workflow:

1. Fits signal mass distribution → determines mass window
2. Calculates background class weights from true cross-sections
3. **Optimizes ParticleNet score threshold** within mass window to maximize sensitivity
4. Applies mass window + score cuts → creates mass templates

**Key Concept**: ParticleNet score is used as a **CUT variable**, not the discrimination variable. After cuts, we still bin by MASS (not score).

**ParticleNet Score**: A cross-section weighted Bayesian likelihood ratio representing P(signal | event, true rates). See **Mathematical Formulation** section below for detailed derivation.

---

## Baseline vs ParticleNet

| Aspect | Baseline | ParticleNet |
|--------|----------|-------------|
| **Cuts** | Mass window only | Mass window + score ≥ threshold |
| **Signal** | 8.67 events | 5.07 events (58% efficiency) |
| **Background** | 105.62 events | 13.67 events (87% rejection) |
| **S/B Ratio** | 0.082 | 0.371 (**4.5× better**) |
| **Sensitivity** | Z = 0.833 | Z = 1.297 (56% improvement) |
| **Runtime** | ~2-5 min | ~5-10 min |
| **Requirements** | Mass only | ParticleNet scores (80 < mA < 100 GeV) |

**When to use ParticleNet**: When scores are available and background rejection is needed.

---

## Mathematical Formulation

### ParticleNet Network Output

The ParticleNet classifier produces four-class softmax outputs for each event:
- **s₀**: signal score
- **s₁**: nonprompt score
- **s₂**: diboson score
- **s₃**: ttZ score

Properties: sᵢ ∈ [0, 1] and ∑ᵢ sᵢ ≈ 1

### Bayesian Likelihood Ratio Construction

The goal is to compute the posterior probability:

```
P(signal | event, true rates)
```

Using Bayes' theorem:

```
P(class_i | event) = P(event | class_i) × P(class_i) / P(event)
```

For the signal class:

```
P(signal | event, true rates) = P(event | signal) × P(signal) / P(event)

where P(event) = ∑ᵢ P(event | class_i) × P(class_i)
```

Expanding the denominator:

```
P(signal | event, true rates) =
    P(event | signal) × P(signal) /
    [P(event | signal) × P(signal) +
     P(event | nonprompt) × P(nonprompt) +
     P(event | diboson) × P(diboson) +
     P(event | ttZ) × P(ttZ)]
```

### Network Training vs Inference

During **training**:
- All classes weighted to equal total weight → P(signal) = P(nonprompt) = P(diboson) = P(ttZ) = 0.25
- Network learns likelihoods P(event | class_i)
- Softmax outputs: sᵢ ∝ P(event | class_i) × 0.25

During **inference**:
- True prior probabilities differ: P(signal) ≠ P(nonprompt) ≠ P(diboson) ≠ P(ttZ)
- Must reweight to account for actual cross-sections

### Cross-Section Weighted Score

Define background weights proportional to true rates:

```
w₁ = yield_nonprompt / ∑(background yields)
w₂ = yield_diboson / ∑(background yields)
w₃ = yield_ttZ / ∑(background yields)

Constraint: w₁ + w₂ + w₃ = 1
```

The corrected likelihood ratio:

```
score_PN = s₀ / (s₀ + w₁ × s₁ + w₂ × s₂ + w₃ × s₃)
```

This represents P(signal | event, true rates) when properly normalized.

### Mathematical Properties

The weighted score satisfies:

**Boundedness**: score_PN ∈ [0, 1]
- Proof: Numerator s₀ ≥ 0, denominator ≥ s₀ (since wᵢ ≥ 0, sᵢ ≥ 0)

**Monotonicity**: Higher signal likelihood → higher score

**Normalization**:
- When s₀ = 1, others = 0 → score_PN = 1 (pure signal)
- When s₀ = 0 → score_PN = 0 (pure background)

**Comparison with unweighted score**:
```
Unweighted: score_PN = s₀ / (s₀ + s₁ + s₂ + s₃)
            Assumes equal background priors (training assumption)

Weighted:   score_PN = s₀ / (s₀ + w₁s₁ + w₂s₂ + w₃s₃)
            Corrects for true background composition
```

---

## Background Weight Calculation

Background weights are calculated from observed event yields in the mass window to correct for class imbalance during training.

### Process

1. **Measure yields** for each background process in mass window [mₐ - 5√(Γ² + σ²), mₐ + 5√(Γ² + σ²)]

2. **Calculate relative weights**:
   ```
   total = yield_nonprompt + yield_diboson + yield_ttX
   w₁ = yield_nonprompt / total
   w₂ = yield_diboson / total
   w₃ = yield_ttX / total
   ```

3. **Verify constraint**: w₁ + w₂ + w₃ = 1.0

### Output Format

Saved to `background_weights.json`:

```json
{
  "weights": {
    "nonprompt": 0.7234,
    "diboson": 0.1891,
    "ttX": 0.0875
  },
  "yields": {
    "nonprompt": 85.42,
    "diboson": 22.33,
    "ttX": 10.33
  },
  "total_yield": 118.08,
  "mass_window": [83.76, 95.91]
}
```

### Physical Interpretation

Weights represent the true relative background composition in the signal region, accounting for differences in production cross-sections and selection efficiencies. Dominant backgrounds (e.g., nonprompt) receive higher weights, while rare processes (e.g., ttZ) receive lower weights.

---

## Workflow

This section describes the **template creation** stage for the ParticleNet method. For the complete workflow including statistical analysis (combine), see the **[Integration with Statistical Analysis](#integration-with-statistical-analysis)** section below.

### Direct Command (Manual)

For manual execution or debugging, call `makeBinnedTemplates.py` directly:

```bash
ERA="2017"
CHANNEL="SR1E2Mu"
MASSPOINT="MHc130_MA90"

makeBinnedTemplates.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method ParticleNet
```

**Note**: This command is automatically called by `prepareCombine.sh` as part of the complete workflow.

### Automated Workflow (Recommended)

For production analysis, use the automated shell script:

```bash
./scripts/prepareCombine.sh $ERA $CHANNEL $MASSPOINT ParticleNet
# Calls makeBinnedTemplates.py internally (along with preprocess, check, datacard steps)
```

→ **[Complete Workflow Guide](COMBINE_WORKFLOW.md)** for automated processing

### What Happens (Automatic)

**Step 1: Mass Fitting**
- Fits Voigtian to signal → extracts mass, width, sigma
- Calculates mass window: `mass ± 5√(width² + sigma²)`
- Saves: `signal_fit.json`, `signal_fit.png`

**Step 2: Background Weight Calculation** (ParticleNet method only)
- Calculates event yields for nonprompt, diboson, ttX in mass window
- Normalizes weights to sum = 1.0
- Saves: `background_weights.json`

**Step 3: Threshold Optimization**
- Loads events within mass window
- Scans 101 thresholds (0.00 to 1.00) using cross-section weighted scores
- Maximizes Asimov significance: `Z = √(2((S+B)ln(1+S/B) - S))`
- Saves: `threshold.csv`, `score_optimization.png`

**Step 4: Template Creation**
- Applies mass window cut
- Applies ParticleNet score cut (threshold from Step 3) using weighted score definition
- Creates mass histograms with systematics
- Saves: `shapes.root`

### Output Files

```
templates/$ERA/$CHANNEL/$MASSPOINT/Shape/ParticleNet/
├── signal_fit.json           # {"mass": 89.84, "width": 0.893, "sigma": 0.823, ...}
├── signal_fit.png            # Mass fit diagnostic
├── background_weights.json   # Background class weights (w₁, w₂, w₃)
├── threshold.csv             # Optimized threshold (e.g., 0.52)
├── score_optimization.png    # Signal/background score distributions
└── shapes.root               # Mass templates for HiggsCombine
```

### Example Output

```
INFO:root:Fit results: mA = 89.84 GeV, Γ = 0.893 GeV, σ = 0.823 GeV
INFO:root:Mass window: [83.76, 95.91] GeV

INFO:root:Threshold optimization:
INFO:root:  Best threshold: 0.520
INFO:root:  Initial sensitivity: 0.833
INFO:root:  Max sensitivity: 1.297
INFO:root:  Improvement: 55.82%

INFO:root:Process yields:
INFO:root:  Signal:          5.07
INFO:root:  Background:     13.67
INFO:root:  S/B ratio:       0.371
```

---

## Integration with Statistical Analysis

The ParticleNet workflow described above creates optimized templates. This section shows how these templates integrate with HiggsCombine for statistical analysis.

### Template Creation → Statistical Analysis Pipeline

```
ParticleNet Template Creation      Statistical Analysis (HiggsCombine)
(makeBinnedTemplates.py)          (text2workspace, combine)
         │                                    │
         ├─→ signal_fit.json                 │
         ├─→ background_weights.json         │
         ├─→ threshold.csv                   │
         ├─→ shapes.root  ────────────────→  │
         └─→ score_optimization.png          │
                                              ├─→ workspace.root
                                              ├─→ FitDiagnostics
                                              └─→ Limits
```

### Automated Workflow with Shell Scripts

The `prepareCombine.sh` script automates the complete template creation workflow, including ParticleNet optimization:

```bash
ERA="2022"
CHANNEL="SR1E2Mu"
MASSPOINT="MHc130_MA90"
METHOD="ParticleNet"

# Stage 1: Automated template preparation (includes ParticleNet optimization)
./scripts/prepareCombine.sh $ERA $CHANNEL $MASSPOINT $METHOD
# Internally calls:
#   1. preprocess.py
#   2. makeBinnedTemplates.py --method ParticleNet (the workflow described above)
#   3. checkTemplates.py
#   4. printDatacard.py

# Stage 2: Run HiggsCombine
./scripts/runCombine.sh $ERA $CHANNEL $MASSPOINT $METHOD
# Executes:
#   1. text2workspace.py datacard.txt
#   2. combine -M FitDiagnostics
#   3. combine -M AsymptoticLimits
```

**Output**: Complete statistical analysis results including limits and fit diagnostics.

→ **[Complete Combine Workflow](COMBINE_WORKFLOW.md)** for detailed documentation

### Complete End-to-End Example: Single Masspoint

This example shows the full ParticleNet analysis from raw data to final limits:

```bash
#!/bin/bash
# complete_particlenet_analysis.sh

ERA="2022"
CHANNEL="SR1E2Mu"
MASSPOINT="MHc130_MA90"
METHOD="ParticleNet"

# Stage 1: Template Preparation (automated)
echo "Creating ParticleNet-optimized templates..."
./scripts/prepareCombine.sh $ERA $CHANNEL $MASSPOINT $METHOD

# What happens internally:
# ├─ Preprocessing: Creates sample ROOT files
# ├─ Signal fit: Determines mass window
# ├─ Background weights: Calculates w₁, w₂, w₃ from yields
# ├─ Threshold optimization: Finds optimal score cut (e.g., 0.52)
# ├─ Template creation: Applies cuts, creates mass histograms
# └─ Datacard generation: Prepares HiggsCombine input

# Check ParticleNet optimization results
echo "ParticleNet optimization results:"
cat templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/threshold.csv
# Example output: threshold,0.520

# View background weights
cat templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/background_weights.json
# Shows: {"weights": {"nonprompt": 0.723, "diboson": 0.189, "ttX": 0.088}}

# Stage 2: Statistical Analysis
echo "Running HiggsCombine..."
./scripts/runCombine.sh $ERA $CHANNEL $MASSPOINT $METHOD

# Stage 3: Extract Results
echo "Extracting limits..."
root -l -q templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/higgsCombineTest.AsymptoticLimits.mH120.root <<EOF
limit->Scan("quantileExpected:limit")
.q
EOF

echo "Analysis complete!"
echo "Templates: templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD/"
echo "ParticleNet plots: score_optimization.png, signal_fit.png"
echo "Limits: higgsCombineTest.AsymptoticLimits.mH120.root"
```

**Expected Output**:
```
ParticleNet optimization results:
threshold,0.520

Running HiggsCombine...
[Combine output...]

Extracting limits...
***********************************
*    Row   * quantileE * limit    *
***********************************
*        0 *   -1.0000 *  0.8234  *  (observed)
*        1 *    0.0250 *  0.4521  *  (expected -2σ)
*        2 *    0.1600 *  0.6123  *  (expected -1σ)
*        3 *    0.5000 *  0.8765  *  (expected median)
*        4 *    0.8400 *  1.2451  *  (expected +1σ)
*        5 *    0.9750 *  1.7234  *  (expected +2σ)
***********************************

Analysis complete!
```

### Batch Processing Multiple Masspoints

Process all ParticleNet-compatible masspoints in parallel:

```bash
#!/bin/bash
# batch_particlenet_analysis.sh

METHOD="ParticleNet"

# ParticleNet scores available for: 80 < mA < 100 GeV
MASSPOINTs=(
    "MHc115_MA87"
    "MHc130_MA90"
    "MHc145_MA92"
    "MHc160_MA85"
)

# Process all masspoints through complete workflow
parallel -j 18 "./scripts/runCombineWrapper.sh" {1} $METHOD \
    ::: "${MASSPOINTs[@]}"

# Collect results
echo "Collecting ParticleNet limits..."
python python/collectLimits.py \
    --era FullRun2 \
    --channel Combined \
    --method ParticleNet \
    --output limits_particlenet.json

echo "ParticleNet batch analysis complete!"
```

→ **[Batch Processing Guide](COMBINE_WORKFLOW.md#stage-3-batch-processing-runcombinewrappersh)** for details

### ParticleNet vs Baseline Comparison

Compare sensitivity between methods:

```bash
#!/bin/bash
# compare_methods.sh

ERA="2022"
CHANNEL="SR1E2Mu"
MASSPOINT="MHc130_MA90"

# Run both methods
for METHOD in Baseline ParticleNet; do
    echo "Processing $METHOD method..."
    ./scripts/prepareCombine.sh $ERA $CHANNEL $MASSPOINT $METHOD
    ./scripts/runCombine.sh $ERA $CHANNEL $MASSPOINT $METHOD
done

# Compare expected limits
echo "Baseline limit:"
root -l -q templates/$ERA/$CHANNEL/$MASSPOINT/Shape/Baseline/higgsCombineTest.AsymptoticLimits.mH120.root \
    -e 'limit->Scan("limit", "abs(quantileExpected-0.5)<0.01")'

echo "ParticleNet limit:"
root -l -q templates/$ERA/$CHANNEL/$MASSPOINT/Shape/ParticleNet/higgsCombineTest.AsymptoticLimits.mH120.root \
    -e 'limit->Scan("limit", "abs(quantileExpected-0.5)<0.01")'
```

**Typical Result**: ParticleNet expected limit ~40-50% better than Baseline (lower is better)

### Key Integration Points

| Stage | ParticleNet-Specific | Standard (Both Methods) |
|-------|---------------------|------------------------|
| **Preprocessing** | None - uses same samples | `preprocess.py` |
| **Template Creation** | Background weights, threshold optimization | Signal fitting, mass binning |
| **Validation** | Check score distributions | Check histogram integrity |
| **Datacard** | None - same format | `printDatacard.py` |
| **Combine** | None - method-agnostic | All combine commands |
| **Limits** | Typically 40-50% better | Same statistical framework |

**Key Point**: ParticleNet is a **template optimization method**. Once templates are created, the statistical analysis (combine) is identical to Baseline.

---

## Troubleshooting

### ParticleNet Scores Not Found

**Symptoms**:
```
WARNING:root:ParticleNet scores not found
WARNING:root:Proceeding without cuts (equivalent to Baseline)
```

**Cause**: Mass point outside training range (mA < 80 or > 100 GeV), or scores not in ROOT file.

**Solution**:
```bash
# Check if scores exist
root -l samples/2017/SR1E2Mu/MHc130_MA90/Baseline/MHc130_MA90.root
Central->Print()  # Look for score_MHc130_MA90_* branches

# If missing: use Baseline method or contact ParticleNet training team
```

### Threshold Optimization Returns 0.0

**Symptoms**:
```
INFO:root:  Best threshold: 0.000
INFO:root:  Improvement: 0.00%
```

**Cause**: No signal/background separation, or mass window has no events.

**Solution**:
```bash
# Check score distributions
plotParticleNetScore.py --era 2017 --channel SR1E2Mu --masspoint MHc130_MA90

# If no separation visible: use Baseline method
```

### S/B Worse Than Baseline

**Symptoms**: ParticleNet S/B < Baseline S/B

**Cause**: Threshold too loose or model overtrained.

**Solution**: Fall back to Baseline or manually adjust threshold in `threshold.csv`.

### Background Weight Calculation Fails

**Symptoms**:
```
WARNING:root: Cannot calculate weight for nonprompt: file not found
WARNING:root: Using equal weights
```

**Cause**: Missing background sample files or empty mass window.

**Solution**:
```bash
# Verify all background files exist
ls samples/$ERA/$CHANNEL/$MASSPOINT/Baseline/{nonprompt,diboson,ttX}.root

# Check mass window has events
root -l samples/$ERA/$CHANNEL/$MASSPOINT/Baseline/nonprompt.root
Central->Draw("mass")  # Should show events in mass window
```

If files are missing, run preprocessing. If mass window is empty, check signal fit parameters.

---

## Visualization (Optional)

Generate score distribution plots:

```bash
plotParticleNetScore.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT
```

Creates `score_distribution.png` showing signal vs background scores for events in mass window.

---

## Best Practices

1. **Always compare with Baseline**: Run both methods, verify ParticleNet improves S/B
2. **Check score distributions**: Use `plotParticleNetScore.py` to validate separation
3. **Monitor systematics**: ParticleNet cuts can amplify systematic uncertainties
4. **Document threshold choices**: If manually adjusting, document reasoning

---

## Related Documentation

### Complete Analysis Pipeline
- **[COMBINE_WORKFLOW.md](COMBINE_WORKFLOW.md)** - Complete statistical analysis workflow
  - End-to-end: template preparation → combine → limits
  - Shell script automation (prepareCombine.sh, runCombine.sh)
  - Batch processing and result collection
  - **Read this for**: Running complete analysis from templates to limits

### Template Creation and Methodology
- **[WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md)** - General workflow patterns
  - Single/multi-masspoint processing
  - Score optimization and retraining
  - Systematic uncertainty profiling
  - **Read this for**: Understanding workflow patterns and best practices

- **[NEGATIVE_BINS_HANDLING.md](NEGATIVE_BINS_HANDLING.md)** - Technical handling of negative bins
  - Automatic fix mechanisms
  - Statistical implications
  - **Read this for**: Understanding template validation

### Technical Reference
- **[API_REFERENCE.md](API_REFERENCE.md)** - Code documentation
  - C++ class API (Preprocessor, AmassFitter)
  - Python script parameters
  - Shell script API (prepareCombine.sh, runCombine.sh, runCombineWrapper.sh)
  - **Read this for**: Detailed API documentation and parameters

### Recommended Reading Order

**For new users**:
1. This document (PARTICLENET_WORKFLOW.md) - Understand the method
2. COMBINE_WORKFLOW.md - Learn complete pipeline
3. WORKFLOW_GUIDE.md - Explore workflow patterns

**For developers**:
1. API_REFERENCE.md - Understand code structure
2. This document - ParticleNet implementation details
3. NEGATIVE_BINS_HANDLING.md - Technical edge cases
