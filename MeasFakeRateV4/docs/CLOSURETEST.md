# Closure Test Uncertainty Design

## Overview

The closure test compares the signal region (SR) MC against the sideband prediction (SB) for
`pair/mass`, `pair_lowM/mass`, and `pair_highM/mass` distributions. The resulting per-era
normalization uncertainties are written to `Common/Data/FakeNorm.json` and consumed downstream
by `SignalRegionStudyV2` as `lnN` nuisance parameters.

## Uncertainty Derivation (Individual Eras)

### Chi2 formula

The chi2 is computed on **absolute yields** (no normalization), because the assigned uncertainty
is a rate (normalization) uncertainty. A flat fractional systematic `syst_frac` is added in
quadrature to the statistical uncertainties of both histograms in each bin:

```
σ²_bin = σ²_stat,obs + σ²_stat,exp + (syst_frac × exp_bin)²
chi2   = Σ_bins  (obs_bin - exp_bin)² / σ²_bin        [bins with exp_bin > 0]
```

All original 1 GeV bins are used — no rebinning is applied before the scan. Bins with
`exp_bin = 0` are skipped. The `(syst_frac × exp_bin)²` floor keeps sparse tail bins
well-behaved without requiring a minimum-expected merging step.

### Scan and recommendation

`syst_frac` is scanned in 5% steps from 0% to 100%. The **recommended systematic** is the
step where `chi2/ndf` is closest to 1.0:

```python
recommended_systematic_pct = min(chi2_profile, key=lambda e: abs(e["chi2_per_ndf"] - 1.0))["syst_pct"]
```

The full scan profile (`chi2_profile`) is stored in the output JSON for inspection and
overlaid across eras with `plotChi2Profiles.py`.

### Reference chi2 (for the plot)

A separate shape-only chi2 is computed via ROOT's `Chi2Test("WW")` on histograms rebinned
to `expected >= 5` per bin (classical validity condition). This is used only as a reference
p-value on the plot, not for deriving the systematic.

### Output per plot

Each closure plot shows two lines:

```
Rate:  χ²/ndf = X.XX (p = Y.YY), syst = Z%    ← at the recommended syst level
Shape: χ²/ndf = X.XX (p = Y.YY)               ← shape-only, stat errors only
```

A companion `_chi2profile.png` is also saved showing chi2/ndf vs assumed systematic (%) with
a red dashed line at chi2/ndf = 1 and a green marker at the recommended value.

## Uncertainty Derivation (Run2 / Run3 Combined)

For combined-era plots (era = `Run2` or `Run3`), a single flat syst scan is not appropriate
because individual eras can have genuine era-specific closure failures that cancel
accidentally in the combination (e.g. 2022EE showing the largest deviation despite having
the most statistics within Run3).

Instead, the chi2 is computed **era-by-era** using per-era systematics from
`Common/Data/FakeNorm.json`, then summed:

```
chi2_total = Σ_era  chi2(h_obs_era, h_exp_era, syst_era)
ndf_total  = Σ_era  ndf_era
```

This gives a combined chi2/ndf that reflects how well each era closes under its own assigned
uncertainty. The plot label shows:

```
Rate:  χ²/ndf = X.XX (p = Y.YY)
Shape: χ²/ndf = X.XX (p = Y.YY)
```

No syst scan profile is generated for combined eras.

## Source of Uncertainty per Channel

| Channel   | Histkey used                              |
|-----------|-------------------------------------------|
| Run1E2Mu  | `pair/mass`                               |
| Run3Mu    | `max(pair_lowM/mass, pair_highM/mass)`    |

Only the `Central` sideband variation is used for deriving the uncertainty.

## FakeNorm.json

`Common/Data/FakeNorm.json` stores per-era fractional uncertainties for individual eras
(Run2/Run3 combined values are **not** stored — they are computed on-the-fly from the
per-era entries):

```json
{
    "Run1E2Mu": {
        "2016preVFP":  0.10,
        "2016postVFP": 0.10,
        "2017":        0.10,
        "2018":        0.10,
        "2022":        0.25,
        "2022EE":      0.20,
        "2023":        0.15,
        "2023BPix":    0.20
    },
    "Run3Mu": { ... }
}
```

Generate / update with:

```bash
cd MeasFakeRateV4
bash scripts/parseFakeNorm.sh
```

## Diagnostic Tools

### Overlay chi2/ndf profiles across eras

```bash
python python/plotChi2Profiles.py \
    --eras 2022 2022EE 2023 2023BPix \
    --channel Run1E2Mu --histkey pair/mass --syst Central
```

Output saved to `plots/chi2profiles/`.

Use this to distinguish statistical fluctuations from genuine era-specific closure failures:
if a high-statistics era (e.g. 2022EE) still shows a large recommended systematic, the
closure failure is genuine, not a fluctuation.

## Relevant Files

| File | Role |
|------|------|
| `python/plotClosure.py` | Main closure test script; produces plots and yield JSON |
| `python/parseFakeNorm.py` | Reads per-era JSONs → writes `Common/Data/FakeNorm.json` |
| `python/plotChi2Profiles.py` | Overlays chi2/ndf profiles across eras |
| `scripts/plotClosure.sh` | Runs closure tests in parallel for all channels/histkeys/systs |
| `scripts/parseFakeNorm.sh` | Entry point for `parseFakeNorm.py` |
| `Common/Data/FakeNorm.json` | Per-era nonprompt normalization uncertainties (fractions) |
