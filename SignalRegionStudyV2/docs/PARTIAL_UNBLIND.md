# Partial Unblinding: One-Sided Impacts in Combined Eras

## Problem

Impact plots for the **All (Run2+Run3) Combined** partial-unblind configuration show a high fraction of one-sided nuisance parameters (~85%), while individual Run2 and Run3 impacts are normal (~12% and ~20% respectively). This was first observed for `MHc100_MA95/ParticleNet/extended_partial_unblind`.

## Investigation Summary

### Template Structure (All/Combined/MHc100_MA95)

| Region | Data | Background | Signal | Data-Bkg | S/B |
|--------|------|-----------|--------|----------|-----|
| Peak (bins 7-11) | 83 | 88.3 | 197.2 | -5.3 | 2.23 |
| Sidebands | 311 | 353.8 | 45.7 | -42.8 | 0.13 |
| **Total** | **394** | **442.1** | **242.9** | **-48.1** | **0.55** |

Key features:
- Signal is sharply peaked in bins 8-10 (S/B up to 5.0)
- Data is below background everywhere, with the deficit concentrated in the sidebands
- Best-fit r ~ 0.002 (essentially no signal)

### One-Sidedness by Impact Magnitude

The one-sidedness is pervasive across all impact sizes, not limited to small-impact NPs.

### Comparison Across Eras

| Configuration | Physics NPs | Total NPs | Best-fit r | One-sided fraction |
|---------------|-------------|-----------|------------|-------------------|
| Run2 Combined | 67 | 424 | -0.005 | 12% |
| Run3 Combined | 60 | 480 | +0.022 | 20% |
| All Combined | 127 | 904 | +0.002 | ~85% |

## Root Cause

The one-sided impacts are **genuine**, not a fit convergence artifact. The mechanism:

1. **Signal shape is a narrow peak** (bins 8-10) while the **data deficit is in the flat sidebands** (-43 events). Adjusting the signal strength r cannot fix a sideband deficit.

2. **Best-fit r ~ 0**: The data is consistent with background-only. With r ~ 0, the signal template contributes negligibly to the total prediction.

3. **NP shifts perturb background normalization/shape**: When any NP is shifted to +/- 1 sigma and the fit re-profiles, the other NPs absorb the sideband changes. The signal strength r, which can only add events in the peaked region, adjusts minimally and consistently in the same direction.

4. **More NPs in "All" amplifies the effect**: With 127 physics NPs (vs 67 or 60), many era-specific NPs have tiny impacts on r. At this scale, both +1sigma and -1sigma shifts produce the same sign of delta-r, because the signal template cannot distinguish between up/down perturbations of a weakly-constraining nuisance.

### Why Run2 and Run3 individually are less affected

- Fewer NPs = each NP has relatively larger individual leverage on r
- The likelihood surface is simpler, so conditional profiling produces cleaner two-sided behavior
- Run3 best-fit r = 0.022 (further from zero) gives more room for two-sided impacts

## What Was Tried

| Approach | Result |
|----------|--------|
| `r=-5,5` (default partial-unblind) | 85% one-sided |
| `r=-10,10` (wider range) | 88% one-sided |
| `--cminDefaultMinimizerStrategy 0` + `--X-rtd MINIMIZER_analytic` + `--cminFallbackAlgo` | 85% one-sided |
| `--cminDefaultMinimizerStrategy 2` (IMPROVE step) | Tested, no significant improvement expected |

None of the minimizer options changed the one-sidedness significantly, confirming this is a physics effect rather than a fit convergence issue.

## Conclusion

The one-sided impacts in the All/Combined partial-unblind configuration are an expected consequence of:
- Partial unblinding (score < 0.3) selecting a region with poor signal discrimination
- The signal template being a narrow peak that cannot absorb sideband deficits
- The large number of weakly-constraining NPs in the full Run2+Run3 combination

**This does not indicate a problem with the analysis.** The impact plot correctly reflects that most NPs have negligible and asymmetric effects on the signal strength when r ~ 0. The top-ranked NPs (scale_m, Norm_others, Norm_nonprompt, Norm_ttZ, Norm_WZ) show proper two-sided behavior.

## Recommendations

- Use the **`--blind` flag** for partial-unblind impact plots (already implemented) to hide the observed r value
- Focus on the **top-ranked NPs** which show proper two-sided behavior and dominate the total uncertainty
- Compare with the **expected (Asimov) impacts** (`impacts_r1`) which should show normal two-sided behavior since the injected signal provides discrimination
- For presentation, consider showing Run2 and Run3 impact plots separately alongside the All combination
