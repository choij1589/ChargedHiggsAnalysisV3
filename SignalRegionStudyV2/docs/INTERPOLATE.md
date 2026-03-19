# Signal Template Interpolation

## Overview

MHc = 85, 115, 145 GeV have only 3–4 simulated MA points each, leaving large gaps in the (MHc, MA) parameter space. Since the signal is a narrow resonance in the di-muon mass (mA), its shape and rate vary smoothly with MA and can be reliably interpolated from neighbouring mass points using an analytic parametric model.

This document describes the interpolation method, validation results, and the strategy for extending it to shape systematics.

---

## 1. Motivation

| MHc (GeV) | Existing MA points | Gap |
|-----------|-------------------|-----|
| 85  | 15, 70, 80 | 15 → 70 GeV (55 GeV gap) |
| 115 | 15, 27, 87, 110 | 27 → 87 GeV (60 GeV gap) |
| 145 | 15, 35, 92, 140 | 35 → 92 GeV (57 GeV gap) |

For comparison, MHc = 70, 100, 130, 160 GeV each have 5–7 MA points. The sparse coverage for the intermediate MHc values degrades limit interpolation when producing 2D exclusion contours.

---

## 2. Shape Model: Double Gaussian

### Why not Voigt?

The existing pipeline fits a Voigt function (RooFit) to each signal's di-muon mass distribution to determine the binning. However, the Voigt profile has heavy Lorentzian tails that systematically **overshoot the MC histogram in the off-peak wings** and **underpredict the central peak**. Quantitatively (self-closure test on MHc=130):

| Model | KS distance | Mean relative diff (core bins) |
|-------|-------------|-------------------------------|
| Voigt | ~0.09 | ~26% |
| Double Gaussian | ~0.04 | ~9% |

The Double Gaussian (DG) is 2–3× more accurate and has better-behaved tails.

### Double Gaussian definition

```
f(x; μ, σ₁, σ₂, f) = f · G(x; μ, σ₁) + (1−f) · G(x; μ, σ₂)
```

where σ₁ < σ₂ (narrow + wide component), G is the standard Gaussian PDF, and the overall normalization gives the signal rate. Parameters:

| Parameter | Physical meaning |
|-----------|-----------------|
| μ | Peak position (≈ fitted mA) |
| σ₁ | Narrow Gaussian width (core detector resolution) |
| σ₂ | Wide Gaussian width (non-Gaussian tails) |
| f | Fraction in narrow component |

### Fit procedure

**File:** `python/generate_interpolated_templates.py:fit_double_gaussian()`

- Fit using `scipy.optimize.curve_fit` (chi2 minimisation) against the weighted MC histogram
- Bin-integrated model (CDF differences, not point-sampled) for accuracy in wide tail bins
- Initial guess: σ₁ = 0.7 × σ_Voigt, σ₂ = 1.5 × σ_Voigt, f = 0.75
- Bounds enforce σ₁ < σ₂ and 0.05 < f < 0.95

Fitted parameters for existing mass points are smooth and physically interpretable:

| MHc | MA | σ₁ (GeV) | σ₂ (GeV) | f |
|-----|----|----------|----------|---|
| 130 | 15 | 0.136 | 0.324 | 0.76 |
| 130 | 55 | 0.571 | 1.355 | 0.78 |
| 130 | 90 | 1.042 | 2.304 | 0.78 |
| 130 | 125 | 1.635 | 3.715 | 0.79 |

Both σ₁ and σ₂ scale linearly with MA. The fraction f ≈ 0.76–0.79 is nearly constant — ideal for spline interpolation.

---

## 3. Interpolation Procedure

**File:** `python/generate_interpolated_templates.py`

### 3.1 Parameters interpolated

For each existing MA point the following quantities are stored, then interpolated as a function of MA:

| Parameter | Source | Used for |
|-----------|--------|---------|
| `mass` (μ) | DG fit | Histogram peak position and binning |
| `voigt_width` = √(w²+σ²) | Voigt fit | Bin edge calculation (consistent with pipeline) |
| `sigma1` | DG fit | Shape |
| `sigma2` | DG fit | Shape |
| `frac` | DG fit | Shape |
| `integral` (rate) | Histogram sum | Signal normalisation |

### 3.2 Spline interpolation

```python
from scipy.interpolate import make_interp_spline
k = min(3, n_points - 1)   # cubic for ≥4 pts, quadratic for 3
spline = make_interp_spline(ma_values, parameter_values, k=k)
```

Degree is automatically reduced for MHc=85 (3 existing points → quadratic).

### 3.3 Binning

Bin edges for interpolated templates use the same extended-binning formula as the main pipeline (`makeBinnedTemplates.py:calculateExtendedBins`):

```python
voigt_width = interpolated value of sqrt(width^2 + sigma^2)
bin_edges = mA_interp + sigma_fractions * voigt_width
# sigma_fractions: [-10, -7, -5..+5 (15 uniform), +7, +10] → 19 bins
```

### 3.4 Output

For each new mass point `MHc{X}_MA{Y}`:

```
templates/{era}/{channel}/MHc{X}_MA{Y}/Baseline/extended/
    shapes_dg.root      # TH1D signal histogram (central only, no systematics yet)
    signal_fit_dg.json  # Interpolated DG parameters + provenance
```

### 3.5 New mass points generated

| MHc | New MA points |
|-----|--------------|
| 85  | 25, 35, 45, 55 |
| 115 | 40, 55, 70 |
| 145 | 50, 65, 80 |

---

## 4. Validation

### 4.1 Self-closure test

For each existing mass point, generate the DG histogram from its own fitted parameters and compare to the real MC histogram. Measures the **model error** (DG vs full simulation), independent of interpolation.

| MHc | MA range | DG KS | Mean rel. diff (core) |
|-----|----------|-------|-----------------------|
| 85 | 15–80 | 0.03–0.04 | 8–9% |
| 115 | 15–110 | 0.03–0.04 | 7–11% |
| 145 | 15–140 | 0.03–0.05 | 7–11% |

### 4.2 Leave-one-out test

Hold out one existing mass point, interpolate from the remaining points, compare to the real MC histogram. Measures **interpolation error** on top of model error.

#### MHc=115 (4 existing points, cubic spline)

| Holdout MA | DG KS | Mean rel. diff | Rate error |
|------------|-------|----------------|-----------|
| 27 | 0.030 | 13% | +17% |
| 87 | 0.053 | 11% | −15% |

#### MHc=145 (4 existing points, cubic spline)

| Holdout MA | DG KS | Mean rel. diff | Rate error |
|------------|-------|----------------|-----------|
| 35 | 0.040 | 7% | +5% |
| 92 | 0.050 | 11% | −7% |

#### MHc=85 (3 existing points)
Leave-one-out not performed (only 2 training points after removal — insufficient for a spline).

### 4.3 Interpretation

**Shape accuracy:** KS ≈ 0.04–0.05 for interpolated templates, compared to 0.03–0.04 for self-closure. The interpolation degrades the KS by ~0.01, which is small relative to the total model uncertainty.

**Rate accuracy:**
- MHc=145: 5–7% rate error → assign ~10% lnN normalisation uncertainty on interpolated signal
- MHc=115: 15% rate error in the large 27→87 GeV gap → assign ~20% lnN normalisation uncertainty; the shape is well-reconstructed (KS still good) but the overall yield has more uncertainty
- MHc=85: Rate unvalidated in the 15→70 GeV gap (no LOO possible); assign ~20% lnN normalisation uncertainty

---

## 5. Shape Systematics Strategy

The interpolated `shapes_dg.root` currently contains **only the central signal histogram**. The 21 signal-group systematics in `configs/systematics.2018.json` must be handled separately.

### 5.1 Systematic types

| Type | Examples | Count | Strategy |
|------|----------|-------|----------|
| `lnN` | `lumi_13TeV_*` | 1 | Apply same value — no template needed |
| `valued shape` | `CMS_eff_e_trigger_2018` | 1 | Apply same ±% to DG central (shape-independent scale) |
| `preprocessed shape` | JES, JER, b-tag, muon/electron scale, PDF, ISR/FSR… | 20 | **Interpolate relative variations** (see below) |

### 5.2 Recommended approach: interpolate relative variations

For each `preprocessed shape` systematic `s` and each existing mass point `MA_i`:

1. Read the up/down histograms from `shapes.root`
2. Compute the bin-by-bin relative variation:
   ```
   δ_up[bin]   = (h_up[MA_i, bin]   − h_nom[MA_i, bin]) / h_nom[MA_i, bin]
   δ_down[bin] = (h_down[MA_i, bin] − h_nom[MA_i, bin]) / h_nom[MA_i, bin]
   ```
3. Fit a spline to `δ_up[bin](MA)` and `δ_down[bin](MA)` across existing MA points
4. Evaluate at the new MA to get `δ_up_interp[bin]` and `δ_down_interp[bin]`
5. Apply to the DG central histogram:
   ```
   h_syst_up[interp, bin]   = h_dg[bin] × (1 + δ_up_interp[bin])
   h_syst_down[interp, bin] = h_dg[bin] × (1 + δ_down_interp[bin])
   ```

**Physical justification:** Detector systematics (JES, lepton scale, b-tag) produce relative shape changes that are largely MA-independent in the signal window. The relative variation `δ` captures the fractional effect on each bin, which transfers well to a DG template at a nearby MA.

**For MHc=85** (3 existing points, quadratic spline): Use nearest-neighbour for theory systematics (PDF, QCD scale, ISR/FSR) where the MA-dependence is less smooth; use interpolation for detector systematics.

### 5.3 Implementation status

| Step | Status |
|------|--------|
| Central DG template generation | Done (`python/generate_interpolated_templates.py`) |
| Systematic relative-variation interpolation | **Planned** |
| Integration with `printDatacard.py` | Planned |
| Full pipeline test (Combine cards + limits) | Planned |

---

## 6. Relevant Files

| File | Role |
|------|------|
| `python/test_interpolation.py` | Voigt vs DG comparison study; leave-one-out on MHc=130 |
| `python/generate_interpolated_templates.py` | Main generation script for MHc=85,115,145 |
| `configs/masspoints.json` | Existing mass points (source of `existing` lists) |
| `templates/{era}/{channel}/MHc*/Baseline/extended/signal_fit_dg.json` | DG fit parameters + provenance for each interpolated point |
| `results/plots/interp_*.png` | Validation and overlay plots |

---

## 7. Usage

```bash
cd SignalRegionStudyV2
source setup.sh

# Run for all three MHc values (2018, SR1E2Mu)
python3 python/generate_interpolated_templates.py --mhc all

# Single MHc
python3 python/generate_interpolated_templates.py --mhc 115 --era 2018 --channel SR1E2Mu

# Reproduce the validation study (MHc=130 Voigt vs DG comparison)
python3 python/test_interpolation.py
```
