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

### Single Command

```bash
ERA="2017"
CHANNEL="SR1E2Mu"
MASSPOINT="MHc130_MA90"

makeBinnedTemplates.py --era $ERA --channel $CHANNEL \
    --masspoint $MASSPOINT --method ParticleNet
```

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

- [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md) - General workflow overview
- [NEGATIVE_BINS_HANDLING.md](NEGATIVE_BINS_HANDLING.md) - Handling negative bins
- [API_REFERENCE.md](API_REFERENCE.md) - Code documentation
