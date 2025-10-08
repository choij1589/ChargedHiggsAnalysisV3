# ParticleNet Classification Metrics Definition

## 1. Overview

This document defines how performance metrics are calculated for ParticleNet models in the ChargedHiggsAnalysisV3 framework. It covers both **binary classification** (signal vs. single background) and **multi-class classification** (signal vs. multiple backgrounds simultaneously), with emphasis on ensuring direct comparability between approaches.

**Purpose:**
- Document metric calculation methods for reproducibility
- Enable fair comparison between binary and multi-class models
- Provide reference for physics interpretation of results

**Scope:**
- ROC curves and AUC metrics
- Score distributions
- Physics event weighting
- Train/test evaluation

---

## 2. Data Structure

### Binary Classification

**Training Data:**
- Signal sample: TTToHcToWAToMuMu
- Single background: One of {Nonprompt (TTLL), Diboson (WZ), TTZ}

**Labels:**
- 0: Signal
- 1: Background

**Model Output:**
- Single probability: P(signal) ∈ [0, 1]
- Obtained from sigmoid activation (2-class softmax equivalent)

### Multi-Class Classification

**Training Data:**
- Signal sample: TTToHcToWAToMuMu
- Multiple backgrounds trained simultaneously: Nonprompt, Diboson, TTZ

**Labels:**
- 0: Signal
- 1: Nonprompt
- 2: Diboson
- 3: TTZ

**Model Output:**
- Probability vector: [P(signal), P(nonprompt), P(diboson), P(ttZ)]
- Constraint: ΣP = 1 (softmax output layer)
- Each element ∈ [0, 1]

---

## 3. Score Definitions

### Binary Classification Score

```
score = P(signal)
```

**Properties:**
- Direct output from binary classifier
- Range: [0, 1]
- Interpretation: Probability that event belongs to signal class

**Example:**
- score = 0.9 → Event is 90% likely to be signal
- score = 0.1 → Event is 10% likely to be signal (90% likely to be background)

### Multi-Class Classification Score (Likelihood Ratio)

```
For comparing Signal vs. Background_i:
  score = P(signal) / (P(signal) + P(background_i))
```

**Properties:**
- Combines relevant probabilities from multi-class softmax output
- Range: [0, 1]
- Interpretation: Relative likelihood of signal hypothesis vs. specific background hypothesis

**Mathematical Basis:**
- Derived from likelihood ratio test (Neyman-Pearson lemma)
- Optimal discriminant for binary hypothesis testing
- Monotonic transformation of log-likelihood ratio

**Implementation Detail:**
```python
score = P_signal / (P_signal + P_background_i + 1e-10)
```
Small epsilon (1e-10) added to denominator to prevent division by zero.

**Example:**
Given multi-class output [P(signal)=0.6, P(nonprompt)=0.3, P(diboson)=0.05, P(ttZ)=0.05]:
- Score for Signal vs. Nonprompt: 0.6 / (0.6 + 0.3) = 0.667
- Score for Signal vs. Diboson: 0.6 / (0.6 + 0.05) = 0.923
- Score for Signal vs. TTZ: 0.6 / (0.6 + 0.05) = 0.923

---

## 4. ROC Curve Calculation

### Binary Classification

**Step-by-step procedure:**

1. **Event Selection:**
   - Use all events: signal + background_i
   - No filtering applied

2. **Labels:**
   - y_true ∈ {0, 1}
   - 0 = signal, 1 = background

3. **Score:**
   - P(signal) from model output

4. **Weights:**
   - |event_weight| (absolute value of physics weights)
   - Handles negative NLO Monte Carlo weights

5. **Calculation:**
   ```python
   from sklearn.metrics import roc_curve, auc
   fpr, tpr, thresholds = roc_curve(y_true, score, sample_weight=weights)
   roc_auc = auc(fpr, tpr)
   ```

6. **Metrics:**
   - **FPR** (False Positive Rate): P(predict signal | true background)
   - **TPR** (True Positive Rate): P(predict signal | true signal)
   - **AUC**: Area Under ROC Curve, integrated performance measure

### Multi-Class Classification

**Step-by-step procedure:**

1. **Event Selection (FILTERING):**
   - Filter to signal + specific background_i only
   - Event mask: `(y_true == 0) | (y_true == background_idx)`
   - This creates a binary problem from multi-class dataset

2. **Labels:**
   - Binary conversion within filtered dataset
   ```python
   y_binary = (y_true[mask] == 0).astype(int)
   ```
   - Result: y_binary ∈ {0, 1} where 0 = background_i, 1 = signal

3. **Score:**
   - Likelihood ratio calculated for filtered events
   ```python
   P_signal = y_scores[mask, 0]
   P_background_i = y_scores[mask, background_idx]
   score = P_signal / (P_signal + P_background_i + 1e-10)
   ```

4. **Weights:**
   - |event_weight[mask]| (absolute value, filtered)

5. **Calculation:**
   ```python
   fpr, tpr, thresholds = roc_curve(y_binary, score, sample_weight=weights)
   roc_auc = auc(fpr, tpr)
   ```

6. **Metrics:**
   - Same FPR, TPR, AUC definitions as binary
   - Computed separately for each signal-background pair

**Key Difference:**
Multi-class filters events first (creating binary subsets), then computes likelihood ratio score. This makes metrics directly comparable to binary approach.

---

## 5. Score Distribution Plots

### Binary Classification

**Visualization:**
- **X-axis:** P(signal) ∈ [0, 1]
- **Y-axis:** Normalized event count (log scale)
- **Histograms:**
  - Signal events (blue color): Should peak near 1 for good separation
  - Background events (red color): Should peak near 0 for good separation

**Train/Test Overlay:**
- Training data: Solid lines
- Test data: Points with error bars
- Allows assessment of overfitting

**Interpretation:**
- Good classifier: Minimal overlap between signal and background distributions
- Overfitting: Large gap between train and test distributions

### Multi-Class Classification

**Two types of plots:**

#### Per-Class Score Distributions
Shows P(class_i) for all events, colored by true class. Demonstrates how well the multi-class model learned each class probability.

#### Likelihood Ratio Score Distributions
**Visualization:**
- **X-axis:** Likelihood ratio score ∈ [0, 1]
- **Y-axis:** Normalized event count (log scale)
- **Histograms:**
  - Signal events (blue color): Should peak near 1
  - Background_i events (red color): Should peak near 0

**Key Points:**
- Generated separately for each signal-background pair
- Uses only events from signal and that specific background (filtered)
- Directly comparable to binary score distributions

**Train/Test Overlay:**
- Same convention as binary (solid lines for train, points for test)

---

## 6. Physics Event Weighting

### Rationale

**Why weights are necessary:**
- CMS Monte Carlo samples include event weights representing:
  - Generator cross-section
  - Detector acceptance
  - Reconstruction efficiency
  - Pileup reweighting
- NLO generators (e.g., POWHEG) can produce negative event weights
- Metrics must be physics-aware for proper interpretation and comparison with data

### Implementation

**Weight Processing:**
```python
weights_abs = np.abs(event_weights)  # Take absolute value
```

**Applied to All Metrics:**

1. **ROC Curve Calculation:**
   ```python
   roc_curve(y_true, y_score, sample_weight=weights_abs)
   ```
   FPR and TPR are computed using weighted event counts

2. **AUC Calculation:**
   ```python
   roc_auc_score(y_true, y_score, sample_weight=weights_abs)
   ```
   Uses weighted integration under ROC curve

3. **Score Distribution Histograms:**
   ```python
   for score, weight in zip(scores, weights):
       histogram.Fill(score, weight)
   ```
   Each event contributes its physics weight to the bin

4. **Confusion Matrices:**
   ```python
   confusion_matrix(y_true, y_pred, sample_weight=weights_abs)
   ```

5. **Precision/Recall Curves:**
   ```python
   precision_recall_curve(y_true, y_score, sample_weight=weights_abs)
   ```

### Weight Statistics

Typical weight ranges in this analysis:
- Signal events: O(10⁻³) to O(10⁻¹)
- Background events: O(10⁻²) to O(1)
- Some events may have |weight| < 10⁻⁶ or > 10²

All histograms are normalized after weighting: `histogram.Scale(1.0 / histogram.Integral())`

---

## 7. Why Likelihood Ratio Enables Direct Comparison

### Problem with Using P(signal) Only

**Issue:**
- Multi-class model outputs 4 probabilities that sum to 1
- Using only P(signal) ignores background-specific information
- P(signal) is "diluted" by presence of multiple backgrounds
- Example: P(signal)=0.4 could mean:
  - Strong signal-like features, but 3 backgrounds share remaining 0.6
  - Weak signal-like features with one dominant background

**Consequence:**
- Unfair comparison: Binary model uses full output, multi-class uses partial output
- Multi-class model appears artificially worse

### Solution: Likelihood Ratio

**Formulation:**
```
score = P(signal) / (P(signal) + P(background_i))
```

**Advantages:**

1. **Utilizes Full Information:**
   - Combines both P(signal) and P(background_i)
   - Reflects how model distinguishes these two specific classes
   - Fair comparison: Both approaches use complete output

2. **Theoretical Optimality:**
   - Neyman-Pearson Lemma: For testing H₀ (background) vs. H₁ (signal), the likelihood ratio test is the most powerful test
   - Maximizes TPR for any given FPR
   - Standard approach in particle physics analyses

3. **Captures Multi-Class Advantage:**
   - Multi-class model learned to distinguish among all backgrounds
   - This knowledge helps separate signal from each individual background
   - Likelihood ratio exposes this learned structure

### Mathematical Justification

For binary hypothesis test:
- H₀: Event is background_i
- H₁: Event is signal

The optimal test statistic is the likelihood ratio:
```
Λ(x) = P(x|H₁) / P(x|H₀) = P(x|signal) / P(x|background_i)
```

By Bayes' theorem, this is proportional to:
```
Λ(x) ∝ P(signal|x) / P(background_i|x)
```

The quantity we use:
```
score = P(signal|x) / (P(signal|x) + P(background_i|x))
```

is a monotonic transformation of Λ, specifically:
```
score = Λ / (Λ + 1)
```

Since monotonic transformations preserve ROC curves, our score is equivalent to the optimal likelihood ratio test.

---

## 8. Implementation Details

### Numerical Stability

**Division by Zero Prevention:**
```python
score = P_signal / (P_signal + P_background + 1e-10)
```
Small epsilon (1e-10) added to prevent division errors when both probabilities are near zero.

**Floating-Point Precision:**
```python
fpr = np.clip(fpr, 0.0, 1.0)
tpr = np.clip(tpr, 0.0, 1.0)
```
Weighted ROC calculations can produce values slightly outside [0,1] due to floating-point arithmetic. Clipping ensures valid probability ranges.

**Negative Weights:**
```python
weights_abs = np.abs(weights)
```
NLO Monte Carlo generators (POWHEG, aMC@NLO) produce negative weights. Taking absolute value is standard practice for performance metrics.

### Train/Test Evaluation

**Purpose:**
- Assess overfitting
- Verify model generalization
- Standard machine learning practice

**Visualization Convention:**
- **Training data:**
  - ROC curves: Solid lines
  - Score distributions: Solid line histograms
  - Usually shows better performance (slight overfitting expected)

- **Test data:**
  - ROC curves: Dashed lines
  - Score distributions: Points with error bars
  - True model performance on unseen data

**Metrics Reported:**
- Separate AUC values for train and test
- Large gap indicates overfitting
- Test metrics used for model comparison

### sklearn Functions Used

```python
from sklearn.metrics import roc_curve, auc, roc_auc_score

# ROC Curve Computation
fpr, tpr, thresholds = roc_curve(
    y_true,           # True labels (0/1)
    y_score,          # Predicted scores
    sample_weight=w   # Physics event weights
)

# AUC by Integration
roc_auc = auc(fpr, tpr)

# AUC by Direct Calculation
roc_auc = roc_auc_score(
    y_true,
    y_score,
    sample_weight=w
)
```

**Notes:**
- `roc_curve`: Returns FPR/TPR arrays for plotting
- `auc`: Integrates using trapezoidal rule
- `roc_auc_score`: More robust for weighted metrics (preferred when available)

### ROOT Visualization

**CMS Style:**
```python
import cmsstyle as CMS
CMS.setCMSStyle()
CMS.SetEnergy(13)                    # 13 TeV
CMS.SetLumi(-1, run="Run2")          # Simulation (negative lumi)
CMS.SetExtraText("Simulation Preliminary")
```

**Histogram Setup:**
- 50 bins in [0, 1] for score distributions
- Log scale y-axis (better visualization of tails)
- Physics-weighted: `histogram.Fill(score, weight)`
- Normalized: `histogram.Scale(1.0 / histogram.Integral())`

**Color Palette:**
Consistent with `Common/Tools/plotter.py`:
- Signal: Blue (#5790fc)
- Background: Red (#e42536)
- Others: Orange, Purple, Gray as needed

---

## 9. Output Files Summary

### Binary Classification (`visualizeBinary.py`)

Generated for each signal-background pair (e.g., MHc160_MA85 vs. nonprompt):

**Primary Metrics:**
- `roc_curve.png`: ROC curve with train/test comparison
- `precision_recall_curve.png`: Alternative performance metric
- `score_distributions.png`: P(signal) distributions for signal and background

**Additional:**
- `confusion_matrix.png`: 2×2 classification matrix (raw counts and normalized)
- `training_curves.png`: Loss and accuracy vs. epoch
- `training_metrics.png`: Memory usage and epoch time
- `summary_report.json` / `.txt`: Numerical results

**Typical usage:**
```bash
python visualizeBinary.py --signal MHc160_MA85 --background nonprompt --channel Run1E2Mu --fold 3
```

### Multi-Class Classification (`visualizeMultiClass.py`)

Generated for signal point (e.g., MHc160_MA85) with all backgrounds:

**ROC Curves (3 files, one per background):**
- `signal_vs_nonprompt_roc_curve.png`
- `signal_vs_diboson_roc_curve.png`
- `signal_vs_ttz_roc_curve.png`

Each shows likelihood ratio discriminant ROC with train/test comparison.

**Likelihood Ratio Score Distributions (3 files):**
- `signal_vs_nonprompt_lr_score_distribution.png`
- `signal_vs_diboson_lr_score_distribution.png`
- `signal_vs_ttz_lr_score_distribution.png`

Each shows signal vs. specific background distributions.

**Per-Class Outputs:**
- `{classname}_score_distribution.png`: P(class) distributions (4 files)
- `{classname}_confusion_matrix.png`: Binary confusion for class vs. rest (4 files)
- `overall_confusion_matrix.png`: 4×4 classification matrix

**Additional:**
- `training_curves.png`: Combined loss and accuracy
- `{metric}_by_class.png`: Precision, recall, F1-score bar charts
- `summary_report.json` / `.txt`: Numerical results including all AUC values

**Typical usage:**
```bash
python visualizeMultiClass.py --signal MHc160_MA85 --channel Run1E2Mu --fold 3
```

---

## 10. References

### Theoretical Background

**Neyman-Pearson Lemma:**
- J. Neyman and E. S. Pearson, "On the Problem of the Most Efficient Tests of Statistical Hypotheses," Philosophical Transactions of the Royal Society A, 1933
- Establishes likelihood ratio test as uniformly most powerful test for simple hypotheses

**Likelihood Ratio in Particle Physics:**
- Standard discriminant in High Energy Physics multivariate analyses
- Used in Higgs discovery (2012), top quark measurements, etc.
- See: TMVA (ROOT) documentation, CMS/ATLAS analysis notes

### Implementation References

**scikit-learn Documentation:**
- ROC Curves: https://scikit-learn.org/stable/modules/model_evaluation.html#roc-metrics
- Weighted Metrics: https://scikit-learn.org/stable/modules/model_evaluation.html#from-binary-to-multiclass-and-multilabel
- Classification Metrics: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics

**ROOT and CMS Style:**
- ROOT TH1 documentation: https://root.cern/doc/master/classTH1.html
- cmsstyle package: CMS-standard plotting for Python
- CMS Publications: https://cms-results.web.cern.ch/

### Related Code in This Repository

**Visualization Scripts:**
- `ParticleNet/python/visualizeBinary.py`: Binary classification visualization
- `ParticleNet/python/visualizeMultiClass.py`: Multi-class classification visualization

**Shared Utilities:**
- `Common/Tools/plotter.py`: Common plotting functions and color schemes
- `Common/Tools/DataFormat.py`: Physics object definitions

**Training Scripts:**
- `ParticleNet/python/trainBinary.py`: Binary model training
- `ParticleNet/python/trainMultiClass.py`: Multi-class model training

**Model Architecture:**
- `ParticleNet/python/ParticleNet.py`: ParticleNet implementation in PyTorch
- Based on: "ParticleNet: Jet Tagging via Particle Clouds" (arXiv:1902.08570)

---

## Document Information

**Version:** 1.0
**Last Updated:** 2025-10-01
**Maintainer:** ChargedHiggsAnalysisV3 Team
**Related Documentation:** See `ParticleNet/README.md` for overview
