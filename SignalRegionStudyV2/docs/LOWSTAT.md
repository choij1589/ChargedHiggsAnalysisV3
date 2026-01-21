# Low-Statistics Background Treatment

## Overview

Low-statistics backgrounds require special handling to avoid statistical artifacts in Combine fits. This document summarizes all treatments implemented.

## 1. Negative Bin Handling

**File**: `python/template_utils.py:ensure_positive_integral()`

- Negative bin contents are set to 0
- If total histogram integral <= 0, a small value (1e-10) is added to the central bin

This ensures Combine can properly normalize all templates.

## 2. Process Merging

**File**: `python/makeBinnedTemplates.py:determineProcessList()`

**Threshold**: `min_total_events=1`

Backgrounds with fewer than 1 event are merged into "others", **except**:
- `nonprompt` and `conversion` are **always kept separate** (dedicated normalization systematics)

Processes with **negative yield** (after all selections) are kept separate but will be dropped from the datacard.

## 3. Dropping Negative/Zero Yield Processes

**File**: `python/printDatacard.py:__init__()`

Processes with non-positive yield are excluded from the datacard entirely:
- Warning logged when a process is dropped
- Prevents Combine issues with negative/zero rate templates

## 4. Shape Systematic Handling (Relative Error > 30%)

**File**: `python/printDatacard.py`

**Threshold**: `SHAPE_REL_ERR_THRESHOLD = 0.30` (30%)

Relative error is calculated as:
```
rel_err = sqrt(sum of bin_error^2) / integral
```

For backgrounds with rel_err > 30%:
- Shape systematics show "-" in datacard (not used)
- Rate effects from shape systematics are absorbed into normalization lnN

### Why relative error instead of absolute yield?
- Relative error directly measures statistical quality of the template
- A process with low yield but many MC entries (e.g., ttZ with 5 events from 7000 entries) is statistically well-determined
- A process with similar yield but few entries (e.g., conversion with 1 event from 7 entries) is not

## 5. Rate Effect Absorption

**File**: `python/printDatacard.py:collect_absorbed_rate_effects()`

For low-stat backgrounds (rel_err > 30%), rate effects from all applicable shape systematics are:
1. Collected (fractional change from Up/Down variations)
2. Combined in quadrature
3. Added to the normalization lnN value

```
combined_lnN = 1.0 + sqrt((base_lnN - 1)^2 + absorbed_frac^2)
```

Example: If base normalization is 1.13 (13% uncertainty) and absorbed effects total 9.2%:
```
combined = 1.0 + sqrt(0.13^2 + 0.092^2) = 1.159
```

## Config Changes

Normalization systematics (`CMS_B2G25013_Norm_*`) changed from `"type": "shape"` to `"type": "lnN"` in all `configs/systematics.*.json` files.

## Summary Table

| Treatment | Criterion | Location | Effect |
|-----------|-----------|----------|--------|
| Negative bins | content < 0 | `template_utils.py` | Set to 0 |
| Always separate | nonprompt, conversion | `makeBinnedTemplates.py` | Never merge |
| Process merging | integral < 1 | `makeBinnedTemplates.py` | Merge to "others" |
| Drop from datacard | yield <= 0 | `printDatacard.py` | Exclude process |
| Shape skip | rel_err > 30% | `printDatacard.py` | Show "-", absorb rate |
| Rate absorption | rel_err > 30% | `printDatacard.py` | Add to lnN in quadrature |

## Verification

```bash
python3 printDatacard.py --era 2017 --channel SR1E2Mu --masspoint MHc130_MA90 --method ParticleNet --binning extended --debug
```

Check:
- Relative errors printed for each background
- Only high rel_err backgrounds get "-" for shape systematics
- Normalization lnN values are enhanced for those backgrounds
