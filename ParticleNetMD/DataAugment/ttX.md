# ttX Background for ParticleNetMD

## Sample Selection

**Included in ttX training class:**
- `TTZToLLNuNu` (Run2) / `TTZ_M50` (Run3)
- `tZq`
- `TTHToNonbb`

**Excluded:**
- `TTWToLNu` — see rationale below

## Rationale

### TTH inclusion

TTH is an irreducible background that contributes 6% of the ttX physics yield in the signal region (sumW = 37.7, smaller than tZq at 137.5). It shares the same tt+boson topology as TTZ and has ample training statistics (40k events after Tight+Bjet, all 8 eras covered). Excluding it would leave a background unmodeled.

> **Note:** The TTH Run2 sumW dropped from 121.3 to 27.4 in the updated TTHToNonbb files (event counts unchanged at 15,653). This is a real change in generator weights in the updated EvtTreeProducer output.

### TTW exclusion

TTW contributes only 8% of the ttX yield (sumW = 48.7) and has critically low Run3 statistics: **917 events** after Tight+Bjet across both channels (~180 events/fold). This is far too few for stable per-fold training. Including it would introduce noise from extreme per-event weight fluctuations in Run3 without meaningfully improving background modeling. Its small yield means the signal region impact is negligible.

## Physics Yield (sumW = genWeight x puWeight x prefireWeight)

| Sample | Run2 sumW | Run3 sumW | Total sumW | % of ttX |
|--------|----------:|----------:|-----------:|---------:|
| TTZ | 348.5 | 107.7 | 456.2 | 72% |
| tZq | 109.5 | 28.0 | 137.5 | 22% |
| TTH | 27.4 | 10.2 | 37.7 | 6% |
| **Included total** | **485.4** | **145.9** | **631.3** | **100%** |
| ~~TTW~~ (excluded) | ~~36.5~~ | ~~12.2~~ | ~~48.7~~ | — |

## Event Counts (Tight+Bjet, both channels)

### Included samples

| Sample | Run2 | Run3 | Total |
|--------|-----:|-----:|------:|
| TTZ (TTZToLLNuNu / TTZ_M50) | 479,096 | 59,087 | 538,183 |
| tZq | 341,656 | 39,453 | 381,109 |
| TTHToNonbb | 15,653 | 24,289 | 39,942 |
| **Included total** | **836,405** | **122,829** | **959,234** |

### Per-fold statistics (5 folds)

| Sample | Run2/fold | Run3/fold | Total/fold |
|--------|----------:|----------:|-----------:|
| TTZ | 95,819 | 11,817 | 107,637 |
| tZq | 68,331 | 7,891 | 76,222 |
| TTH | 3,131 | 4,858 | 7,988 |
| **Total** | **167,281** | **24,566** | **191,847** |

With `max_events_per_fold_per_class = 40,000`, the ttX class is comfortably capped. No augmentation needed.

### Excluded sample

| Sample | Run2 | Run3 | Total |
|--------|-----:|-----:|------:|
| TTWToLNu | 32,679 | **917** | 33,596 |

Run3 TTW: ~180 events/fold — insufficient for stable training.

## Full Event Counts Reference

Event counts under progressive selection cuts, summed over both channels (Run1E2Mu + Run3Mu).

### All Eras Combined

| Sample | Raw | Tight | Bjet | Tight+Bjet |
|--------|----:|------:|-----:|-----------:|
| TTZ | 846,487 | 631,242 | 716,788 | 540,183 |
| tZq | 645,903 | 479,323 | 509,316 | 383,109 |
| TTH | 76,981 | 48,111 | 63,027 | 39,942 |
| TTW (excluded) | 56,674 | 37,280 | 49,479 | 33,602 |

### Run2 (2016preVFP + 2016postVFP + 2017 + 2018)

| Sample | Raw | Tight | Bjet | Tight+Bjet |
|--------|----:|------:|-----:|-----------:|
| TTZToLLNuNu | 736,758 | 552,483 | 633,953 | 479,096 |
| tZq | 571,466 | 425,920 | 455,054 | 341,656 |
| TTHToNonbb | 28,067 | 17,664 | 24,418 | 15,653 |
| TTWToLNu (excluded) | 55,144 | 36,776 | 48,002 | 32,679 |

### Run3 (2022 + 2022EE + 2023 + 2023BPix)

| Sample | Raw | Tight | Bjet | Tight+Bjet |
|--------|----:|------:|-----:|-----------:|
| TTZ_M50 | 109,724 | 78,759 | 81,825 | 59,087 |
| tZq | 76,211 | 54,940 | 55,312 | 39,453 |
| TTHToNonbb | 52,214 | 31,299 | 39,614 | 24,289 |
| TTWToLNu (excluded) | 1,625 | 1,103 | 1,477 | 917 |
