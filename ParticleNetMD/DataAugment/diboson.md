# Diboson Event Counts for ParticleNetMD

Event counts for diboson background samples, summed over both channels (Run1E2Mu + Run3Mu).

- **Run2**: WZTo3LNu_amcatnlo + ZZTo4L_powheg
- **Run3**: WZTo3LNu_powheg + ZZTo4L_powheg

No WZ variants (powheg_mllmin4p0 excluded).

## All Eras Combined

| Sample | Raw | Tight | Bjet | Tight+Bjet |
|--------|----:|------:|-----:|-----------:|
| WZ (Run2+Run3) | 253,441 | 197,799 | 19,898 | 15,561 |
| ZZ (Run2+Run3) | 703,495 | 517,761 | 79,539 | 56,775 |
| **Diboson total** | **956,936** | **715,560** | **99,437** | **72,336** |

## Run2 (2016preVFP + 2016postVFP + 2017 + 2018)

| Sample | Raw | Tight | Bjet | Tight+Bjet |
|--------|----:|------:|-----:|-----------:|
| WZTo3LNu_amcatnlo | 224,219 | 176,467 | 17,806 | 14,041 |
| ZZTo4L_powheg | 578,768 | 431,692 | 69,523 | 49,882 |
| **Diboson total** | **802,987** | **608,159** | **87,329** | **63,923** |

## Run3 (2022 + 2022EE + 2023 + 2023BPix)

| Sample | Raw | Tight | Bjet | Tight+Bjet |
|--------|----:|------:|-----:|-----------:|
| WZTo3LNu_powheg | 29,222 | 21,332 | 2,092 | 1,520 |
| ZZTo4L_powheg | 124,727 | 86,069 | 10,016 | 6,893 |
| **Diboson total** | **153,949** | **107,401** | **12,108** | **8,413** |

Note: Run3 diboson Tight+Bjet is ~8.4k events (vs ~64k in Run2).
WZ in Run3 is especially thin (~1.5k after Tight+Bjet).

## Motivation

Diboson events that naturally contain at least one b-tagged jet ("genuine") have very low statistics, especially in Run 3 where we only have ~1.5k WZ events after Tight+Bjet selections (~8.4k total). These low statistics cause overtraining when training ML models like ParticleNet.

To increase training statistics, we use a "promotion" strategy: take diboson events with **zero** b-tagged jets (637k events), designate one or more jets as b-jets, and use the promoted events alongside the genuine ones.

### Problem with naive promotion (highest b-score)

The previous approach promoted the jet with the highest b-tag score in each 0-tag event. While `(pT, eta)` reweighting of the promoted b-jet can match single b-jet kinematics, it **distorts the correlated event kinematics** (e.g. `jet1_pt`, the leading jet pT). The root cause: the highest b-score jet has a different pT rank distribution than genuine b-jets. The highest b-score jet favors the subleading jet, while genuine b-jets are more uniformly distributed across ranks.

### Problem with independent per-rank rolls

An intermediate approach computed per-rank marginal probabilities P(b-tagged | rank=k) from genuine events and rolled independently for each jet. This correctly captures the marginal rank distribution but **the marginals are not disjoint** — they don't partition the b-jet multiplicity. The result: independent rolls produce ~34% double-b events vs ~7% in genuine data, grossly inflating bjet1_pt.

## Approach: Conditional Rank-based Promotion + Calibration

The current implementation (`python/dibosonRankPromote.py`) uses a **two-step conditional decomposition** that properly partitions the b-jet multiplicity, followed by two layers of calibration weights. **Probability tables and weights are computed separately for Run2 (CHS jets) and Run3 (PUPPI jets)**, since the two jet algorithms produce different b-tag rank distributions.

### Step 1: Sample n_bjets

From genuine events, compute P(n_bjets | nJets) per nJets group (Run2: 1–5 where 5 means >=5; Run3: 1–3 where 3 means >=3). For each 0-tag event, draw whether to promote 1 or 2 jets.

**Run2 (CHS jets) — 5 groups:**

| nJets | P(1b) | P(2b+) |
|------:|------:|-------:|
| 1 | 1.000 | 0.000 |
| 2 | 0.952 | 0.048 |
| 3 | 0.910 | 0.090 |
| 4 | 0.840 | 0.160 |
| >=5 | 0.830 | 0.170 |

**Run3 (PUPPI jets) — 3 groups:**

| nJets | P(1b) | P(2b+) |
|------:|------:|-------:|
| 1 | 1.000 | 0.000 |
| 2 | 0.966 | 0.034 |
| >=3 | 0.928 | 0.072 |

Run3 PUPPI jets have systematically higher P(1b) and lower P(2b+) than Run2 CHS jets.

### Step 2a: Single b-jet — sample rank

If n_bjets=1, sample which pT rank to promote from P(rank | n_bjets=1, nJets group). Run2 uses groups 1–5 (>=5), Run3 uses groups 1–3 (>=3). Run2 rank distributions are roughly uniform across available ranks. Run3 differs notably.

### Step 2b: Double b-jet — sample rank pair

If n_bjets>=2, sample a rank pair (r1, r2) from P(pair | n_bjets>=2, nJets group). The pair code is encoded as r1*10 + r2 with r1 < r2. Same nJets grouping as Step 2a.

### Calibration Layer 1: nJets reweighting

The nJets distribution differs between genuine and 0-tag events (0-tag events have more low-nJets events because high-nJets events are more likely to contain genuine b-jets). A per-nJets shape weight is computed as the ratio of normalized distributions: `w(nJets) = genuine_norm(nJets) / promoted_norm(nJets)`.

**Run2:**

| nJets | Weight |
|------:|-------:|
| 1 | 1.00 |
| 2 | 0.80 |
| 3 | 1.28 |
| 4 | 1.87 |
| 5 | 2.84 |
| 6 | 3.49 |
| 7+ | 1.76 |

**Run3:**

| nJets | Weight |
|------:|-------:|
| 1 | 1.00 |
| 2 | 0.96 |
| 3 | 1.81 |
| 4 | 1.82 |
| 5+ | 1.00 |

Run3 nJets weights are milder (closer to 1.0), consistent with a narrower b-tag response from PUPPI.

### Calibration Layer 2: Inclusive b-jet (pT, eta) reweighting

After nJets reweighting, a residual mismatch remains in the b-jet pT spectrum. This is because rank-based promotion correctly assigns *which* jet to promote by pT rank, but the pT spectrum *within* a given rank differs between genuine and 0-tag events (genuine events contain b-jets from heavy-flavor content, which correlates with different jet pT spectra).

To correct this, we apply a 2D `(pT, |eta|)` calibration weight to each promoted b-jet. The weights are computed as the ratio of normalized 2D histograms from genuine vs nJets-reweighted promoted events, using inclusive b-jets (bjet1 + bjet2 summed). Since ~90% of events have only a single b-jet, inclusive treatment is well-justified.

**Binning** (same as MC b-tag efficiency measurement):
- |eta|: [0.0, 0.8, 1.6, 2.1, 2.5] — 4 bins
- pT: [20, 30, 50, 70, 100, 140, 200, 300, inf) — 8 bins

The (pT, eta) calibration is computed **on top of nJets-reweighted** promoted events, so it captures only the residual mismatch after the nJets shape correction. For 2b events, the calibration weight is the product of per-b-jet weights.

**Run2 weights:**

| |eta| \ pT | [20,30) | [30,50) | [50,70) | [70,100) | [100,140) | [140,200) | [200,300) | [300,inf) |
|:-----------|--------:|--------:|--------:|---------:|----------:|----------:|----------:|----------:|
| [0.0, 0.8) | 1.118 | 0.970 | 0.804 | 1.006 | 0.828 | 0.853 | 1.040 | 1.360 |
| [0.8, 1.6) | 1.075 | 0.993 | 0.867 | 0.965 | 0.837 | 0.893 | 1.217 | 1.341 |
| [1.6, 2.1) | 1.228 | 0.908 | 0.753 | 0.836 | 0.950 | 1.037 | 1.157 | 1.885 |
| [2.1, 2.5) | 1.365 | 0.913 | 0.791 | 0.672 | 1.018 | 1.133 | 0.583 | 1.483 |

**Run3 weights:**

| |eta| \ pT | [20,30) | [30,50) | [50,70) | [70,100) | [100,140) | [140,200) | [200,300) | [300,inf) |
|:-----------|--------:|--------:|--------:|---------:|----------:|----------:|----------:|----------:|
| [0.0, 0.8) | 0.873 | 0.794 | 0.861 | 0.918 | 0.972 | 0.968 | 0.886 | 1.465 |
| [0.8, 1.6) | 1.037 | 0.889 | 1.015 | 0.904 | 0.920 | 1.007 | 0.956 | 1.386 |
| [1.6, 2.1) | 1.319 | 1.073 | 1.405 | 1.118 | 1.321 | 1.043 | 0.625 | 1.368 |
| [2.1, 2.5) | 1.433 | 1.109 | 0.938 | 1.301 | 0.872 | 1.556 | 0.560 | 1.896 |

All weights are in a reasonable range (0.56–1.90). The last pT bin extends to infinity to avoid extreme weights from low-statistics overflow bins.

### Total per-event weight

```
w_total = evt_weight * nJets_weight(nJets) * Product_over_bjets[ pteta_weight(pT_i, |eta_i|) ]
```

where `evt_weight = genWeight * puWeight * prefireWeight`.

### Implementation details

- Separate `conditionalPromoteRun2` / `conditionalPromoteRun3` C++ functions with independent CDF arrays
- Separate `getNjetsWeightRun2` / `getNjetsWeightRun3` lookup functions
- Separate `getBjetPtEtaWeightRun2` / `getBjetPtEtaWeightRun3` C++ functions using `findBin()` helper
- (pT, eta) weight applied jet-by-jet inside `getBjetPtEtaWeight{suffix}(MultiBjetKin)` — for 2b events the weight is the product of both b-jet weights
- Deterministic per-event: TRandom3 seeded from `rdfentry_` ensures reproducibility
- 100% acceptance: every 0-tag event with nJets > 0 is promoted (no discards)
- Inclusive and channel plots are sums of Run2 + Run3 histograms (each weighted with their own tables)
- 4-way validation plots: genuine vs raw promoted vs promoted (nJets rw) vs promoted (+ pT,eta cal.)

### Note on nJets rejection sampling

An alternative approach using nJets rejection sampling (accept/reject events to match the genuine nJets distribution) was tried but abandoned. Run2 had only ~11% acceptance rate (89% of events rejected), reducing 538k Run2 events to ~59k — barely more than the 64k genuine events, defeating the purpose of augmentation. Additionally, the different acceptance rates between Run2 (11%) and Run3 (55%) distorted the Run2:Run3 yield ratio in inclusive plots (genuine is ~8:1 from luminosity, but post-rejection became ~1:1). Physics weights (nJets reweight) are the standard approach in CMS and preserve all events.

## Consequence

**n_bjets composition now matches genuine in both eras:**

| | Run2 Genuine | Run2 Promoted | Run3 Genuine | Run3 Promoted |
|--------:|-----------:|-----------:|-----------:|-----------:|
| 1b | 92.2% | 93.4% | 96.3% | 96.4% |
| 2b | 7.6% | 6.6% | 3.7% | 3.6% |

This resolves the ~34% double-b excess from the independent-roll approach. The Run2/Run3 split correctly captures the different b-tagging characteristics of CHS vs PUPPI jets.

Lepton kinematics, jet eta distributions, n_bjets, and b-jet pT/eta all show good agreement between genuine and promoted events after the full calibration chain (nJets rw + pT,eta cal.).

## Output

- Plots: `DataAugment/diboson/plots/rank_promote/*.png`
- Tables: `DataAugment/diboson/plots/rank_promote/conditional_tables.json` (Run2 and Run3 stored separately, includes nJets weights and pT,eta calibration weights)
- Script: `python/dibosonRankPromote.py`
