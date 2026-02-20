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

## 4. Shape-to-lnN Fallback via `shape?` (Relative Error > 30%)

**File**: `python/printDatacard.py`

**Threshold**: `SHAPE_REL_ERR_THRESHOLD = 0.30` (30%)

Relative error is calculated as:
```
rel_err = sqrt(sum of bin_error^2) / integral
```

For backgrounds with rel_err > 30%:
1. Per-systematic lnN fallback values are pre-computed from Up/Down integral ratios (`precompute_lnn_fallbacks()`)
2. Up/Down shape histograms are removed from `shapes.root` (`rewrite_shapes_root()`)
3. The datacard uses Combine's `shape?` type with per-column lnN values

Combine's `shape?` mechanism: if a process has shape histograms, it uses them; if not, it falls back to the lnN value in that column.

### Per-systematic lnN computation

```python
rate_effect = max(|Up_integral - Central_integral|, |Down_integral - Central_integral|) / Central_integral
lnN_value = 1.0 + rate_effect   # capped at MAX_LNN_VALUE = 2.0
```

- Effects < 0.1% are skipped (value = "-")
- Yields below `MIN_YIELD_THRESHOLD = 1e-6` are skipped

### Why relative error instead of absolute yield?
- Relative error directly measures statistical quality of the template
- A process with low yield but many MC entries (e.g., ttZ with 5 events from 7000 entries) is statistically well-determined
- A process with similar yield but few entries (e.g., conversion with 1 event from 7 entries) is not

## 5. Non-destructive `shapes.root` Rewrite

**File**: `python/printDatacard.py:rewrite_shapes_root()`

When low-stat shape histograms are removed:
1. Original `shapes.root` is renamed to `shapes_original.root`
2. A new `shapes.root` is written with only the kept histograms
3. Re-runs are handled: existing `shapes_original.root` is removed before rename

This preserves the full set of histograms for debugging/reprocessing.

## 6. `lowstat.json` Metadata

**File**: `python/printDatacard.py:write_lowstat_json()`

After pre-computing fallbacks, a JSON file is written to `{TEMPLATE_DIR}/lowstat.json`:

```json
{
  "threshold": 0.30,
  "processes": ["nonprompt", "conversion"],
  "fallbacks": {
    "nonprompt": {
      "CMS_pileup_13TeV": "1.050",
      "CMS_l1_prefiring_2016preVFP": "-"
    },
    "conversion": {
      "CMS_pileup_13TeV": "-"
    }
  }
}
```

This file is consumed by `checkTemplates.py` for correct error band calculation and validation.

## 7. `checkTemplates.py` Integration

**File**: `python/checkTemplates.py`

When `lowstat.json` is present:

1. **Error calculation** (`calculate_systematic_error()`): For low-stat processes with missing Up/Down histograms, the lnN fallback value is used:
   ```
   error_contribution = nominal_bin_content * (lnN_value - 1.0)
   ```
   This is added in quadrature with other systematic errors.

2. **Validation**: Missing histogram warnings are suppressed for (process, systematic) pairs listed in `lowstat.json["fallbacks"]`, since these histograms were intentionally removed.

3. **Systematic plots**: `make_systematic_plot` and `make_systematic_stack_plot` already handle missing histograms gracefully (skip or use central).

## Config Changes

Normalization systematics (`CMS_B2G25013_Norm_*`) changed from `"type": "shape"` to `"type": "lnN"` in all `configs/systematics.*.json` files.

## Summary Table

| Treatment | Criterion | Location | Effect |
|-----------|-----------|----------|--------|
| Negative bins | content < 0 | `template_utils.py` | Set to 0 |
| Always separate | nonprompt, conversion | `makeBinnedTemplates.py` | Never merge |
| Process merging | integral < 1 | `makeBinnedTemplates.py` | Merge to "others" |
| Drop from datacard | yield <= 0 | `printDatacard.py` | Exclude process |
| shape? fallback | rel_err > 30% | `printDatacard.py` | Per-syst lnN via `shape?` |
| Non-destructive rewrite | rel_err > 30% | `printDatacard.py` | `shapes_original.root` preserved |
| lowstat.json | rel_err > 30% | `printDatacard.py` | Metadata for downstream tools |
| Error band fallback | lowstat.json exists | `checkTemplates.py` | lnN-based error contribution |

## Verification

```bash
# Generate templates
python3 python/makeBinnedTemplates.py --era 2016postVFP --channel SR3Mu --masspoint MHc130_MA90 --method Baseline --binning extended

# Print datacard (creates shapes_original.root + filtered shapes.root + lowstat.json)
python3 python/printDatacard.py --era 2016postVFP --channel SR3Mu --masspoint MHc130_MA90 --method Baseline --binning extended --debug

# Verify artifacts
ls -la templates/2016postVFP/SR3Mu/MHc130_MA90/Baseline/extended/shapes*.root
cat templates/2016postVFP/SR3Mu/MHc130_MA90/Baseline/extended/lowstat.json

# Run checkTemplates (loads lowstat.json, correct error bands, no spurious "Missing" errors)
python3 python/checkTemplates.py --era 2016postVFP --channel SR3Mu --masspoint MHc130_MA90 --method Baseline --binning extended
```

Check:
- `shapes_original.root` exists alongside `shapes.root`
- `lowstat.json` lists low-stat processes and their per-systematic fallback values
- `checkTemplates.py` logs "Loaded lowstat.json" and produces no "Missing" issues for low-stat pairs
- Error bands in background stack plot include lnN fallback contributions
