# Nonprompt Data Augmentation for ParticleNetMD

## Motivation

Nonprompt (TTLL) events have very low statistics after Tight+Bjet selection because ~87% of events fail the tight lepton ID cut (the nonprompt lepton is rejected). This makes nonprompt the limiting class for ParticleNet training:

| Class | Training events (3 folds) |
|-------|---------------------------:|
| Signal (per mass point) | ~120k (capped) |
| ttX | ~120k (capped) |
| Diboson (genuine + promoted) | ~120k (capped) |
| **Nonprompt (Tight+Bjet)** | **~51k (uncapped, limiting)** |

## Approach: Loose-to-Tight Lepton Promotion

Analogous to the diboson b-jet promotion strategy (see [`diboson.md`](diboson.md)), we use events where at least one lepton fails tight ID but passes the loose preselection, and reweight them to approximate the tight selection.

### Key insight

ParticleNet node features are `[E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]` — no isolation or impact parameter information. From the network's perspective, a loose muon and a tight muon with the same kinematics are **identical**. The only difference is in event-level kinematic distributions (pT spectrum, eta, etc.), which can be corrected by reweighting.

### ptCorr proxy

To correct the lepton pT spectrum, we use `ptCorr = pT * (1 + miniIso)` as a proxy for the tight lepton pT. Tight leptons have low miniIso, so ptCorr ~ pT. For loose-not-tight leptons, miniIso is larger, so ptCorr > pT — this approximates the pT the lepton would have if isolated (tight).

**Requirement:** The `miniIso` branch must be added to the `EvtTreeProducer` output and the TTLL samples must be re-skimmed.

### Training strategy: blind tight events

- **Training:** Use only loose-not-tight + Bjet events (promoted with reweighting)
- **Validation/Test:** Keep tight+Bjet events blinded for unbiased evaluation
- This prevents information leakage between the augmented training set and evaluation

## Sample Selection

9 samples covering all 8 eras (Run2 + Run3), chosen for physical motivation:

| Category | Samples | Motivation |
|----------|---------|------------|
| Nominal | `TTLL_powheg` | Baseline |
| Top mass (±1 GeV) | `TTLL_mtop171p5_powheg`, `TTLL_mtop173p5_powheg` | Bracketing m_top = 172.5 GeV |
| UE tune | `TTLL_TuneCP5up_powheg`, `TTLL_TuneCP5down_powheg` | Underlying event variations |
| Color reconnection | `TTLL_TuneCP5CR1_powheg`, `TTLL_TuneCP5CR2_powheg` | Alternative CR models |
| ME-PS matching | `TTLL_hdamp_up_powheg`, `TTLL_hdamp_down_powheg` | hdamp variation affects jet kinematics |

### Excluded samples

| Sample | Reason |
|--------|--------|
| `TTLL_TuneCP5_erdOn_powheg` | Less standard radiation model |
| `TTLL_TuneCP5_RTT_powheg` | Run2 only, specialized retune |
| `TTLL_mtop169p5/175p5_powheg` | Wider m_top bracket not needed |
| `TTLL_mtop166p5/178p5_powheg` | Extreme variations, Run3 only |
| `TTLL_widthx*_powheg` | Top width variations, Run2 only, less relevant for nonprompt kinematics |
| `TTLL_powheg_ext1` | Only 2022+2022EE, would create era imbalance |

## Loose-Not-Tight + Bjet Statistics

**Definition:** LNT+Bjet = Bjet - Tight+Bjet (events with at least one b-jet where at least one lepton fails tight ID). Both channels (Run1E2Mu + Run3Mu) summed.

### Per-Sample Totals

| Sample | Run2 LNT+Bjet | Run3 LNT+Bjet | Total |
|--------|---------------:|---------------:|------:|
| TTLL_powheg | 77,457 | 22,906 | 100,363 |
| TTLL_mtop171p5_powheg | 31,021 | 8,912 | 39,933 |
| TTLL_mtop173p5_powheg | 32,543 | 9,204 | 41,747 |
| TTLL_TuneCP5up_powheg | 29,648 | 10,422 | 40,070 |
| TTLL_TuneCP5down_powheg | 30,235 | 10,134 | 40,369 |
| TTLL_TuneCP5CR1_powheg | 30,482 | 9,025 | 39,507 |
| TTLL_TuneCP5CR2_powheg | 31,186 | 9,509 | 40,695 |
| TTLL_hdamp_down_powheg | 30,735 | 8,518 | 39,253 |
| TTLL_hdamp_up_powheg | 30,216 | 9,099 | 39,315 |
| **Total** | **323,523** | **97,729** | **421,252** |

### Per-Fold Statistics

Targets set by luminosity ratio: Run2 (138 fb⁻¹) : Run3 (62 fb⁻¹) ~ 2.2 : 1.

| Metric | Run2 | Run3 |
|--------|-----:|-----:|
| Grand total | 323,523 | 97,729 |
| Per fold (/5) | 64,705 | 19,546 |
| **Target per fold** | **34,000** | **16,000** |
| % of target | 190% | 122% |

Both targets comfortably met. All 9 samples cover all 8 eras uniformly.

## Validation: LNT Promotion with Fake Rate Weights

**Script:** `python/nonpromptPromotion.py`

Validates LNT+Bjet events reweighted with MC fake rates as augmentation candidates.
Uses era-specific fake rates from correctionlib JSONs (already produced by the fake rate measurement).

### Fake Rate Source

Pre-existing correctionlib JSONs — no extraction needed:
```
SKNanoAnalyzer/data/Run3_v13_Run2_v9/{era}/MUO/fakerate_TopHNT.json  → "fakerate_muon_TT"
SKNanoAnalyzer/data/Run3_v13_Run2_v9/{era}/EGM/fakerate_TopHNT.json  → "fakerate_electron_TT"
```

Binning:
- **Muon:** absEta [0, 0.9, 1.6, 2.4] × ptCorr [10, 12, 14, 17, 20, 30, 50, 100] (3×7 = 21 bins)
- **Electron:** absEta [0, 0.8, 1.479, 2.5] × ptCorr [15, 17, 20, 25, 35, 50, 100] (3×6 = 18 bins)

### ptCorr Formula

From `TriLeptonBase.cc`:
```
ptCorr = pT × (1 + max(0, MiniPFRelIso - 0.1))
```
- Tight leptons (miniIso < 0.1): ptCorr ≈ pT
- LNT leptons (miniIso > 0.1): ptCorr > pT
- **Run3 muon cap:** ptCorr capped at 50 GeV for Run3 eras

### Fake Rate Weight

Standard tight-to-loose formula (from `TriLeptonBase::GetFakeWeight()`):
```
weight = -1 × Π[-f(ptCorr, |η|) / (1 - f(ptCorr, |η|))]  over non-tight leptons
```

Total event weight: `genWeight × puWeight × prefireWeight × fake_weight`

Note: Uses `ElectronScEtaColl` (supercluster eta) for electrons — correct variable for fake rate binning near the EB/EE boundary at |η|~1.479.

### Running

```bash
cd $WORKDIR/ParticleNetMD
python python/nonpromptPromotion.py
```

### Validation Results

Four-way normalized comparison across 18 kinematic variables:

| Legend | Description |
|--------|-------------|
| Genuine (Tight+Bjet) | Target distribution |
| LNT (raw pT) | Uncorrected LNT events |
| LNT (ptCorr) | Kinematic correction only |
| LNT (ptCorr + FR wt) | Fully corrected |

**Key observations:**
- Leading/subleading lepton pT, eta: good agreement across all variants
- Third lepton pT: ptCorr shifts spectrum to higher pT (expected); FR weighting corrects back toward genuine shape, with residual differences at low pT (10–20 GeV) where fake rates are highest
- Jet and b-jet distributions: near-perfect agreement (b-jet selection is identical)

### Statistics

| Metric | Count |
|--------|------:|
| Total Genuine (Tight+Bjet) | 65,586 |
| Total LNT+Bjet | 421,252 |
| Augmentation ratio | 6.4× |

Fake rate ranges (representative):
- 2017 muon: 0.049–0.456
- 2017 electron: 0.111–0.436
- 2022 muon: 0.059–0.340
- 2022 electron: 0.149–0.427

### Output

```
DataAugment/nonprompt/plots/lnt_promote/
├── inclusive_{var}.png      (18 variables)
├── Run2_{var}.png           (18 variables)
├── Run3_{var}.png           (18 variables)
├── Run1E2Mu_{var}.png       (18 variables)
├── Run3Mu_{var}.png         (18 variables)
└── summary.json
```

## TODO

- [ ] Integrate into `saveDataset.py` pipeline
- [ ] Implement blind training strategy (LNT for training, Tight for validation/test)

## Event Counts Reference

Full event counts under progressive selection cuts, summed over both channels (Run1E2Mu + Run3Mu).

### Run2 (2016preVFP + 2016postVFP + 2017 + 2018)

| Sample | Raw | Tight | Bjet | Tight+Bjet |
|--------|----:|------:|-----:|-----------:|
| TTLL_powheg | 122,664 | 16,352 | 89,658 | 12,201 |
| TTLL_mtop171p5_powheg | 49,315 | 6,633 | 35,926 | 4,905 |
| TTLL_mtop173p5_powheg | 51,470 | 6,900 | 37,703 | 5,160 |
| TTLL_TuneCP5up_powheg | 46,921 | 6,235 | 34,217 | 4,569 |
| TTLL_TuneCP5down_powheg | 47,691 | 6,362 | 35,006 | 4,771 |
| TTLL_TuneCP5CR1_powheg | 48,488 | 6,557 | 35,366 | 4,884 |
| TTLL_TuneCP5CR2_powheg | 49,743 | 6,617 | 36,083 | 4,897 |
| TTLL_hdamp_up_powheg | 47,965 | 6,477 | 35,052 | 4,836 |
| TTLL_hdamp_down_powheg | 48,780 | 6,473 | 35,523 | 4,788 |
| **Run2 total** | **513,037** | **68,606** | **374,534** | **51,011** |

### Run3 (2022 + 2022EE + 2023 + 2023BPix)

| Sample | Raw | Tight | Bjet | Tight+Bjet |
|--------|----:|------:|-----:|-----------:|
| TTLL_powheg | 38,672 | 4,930 | 26,374 | 3,468 |
| TTLL_mtop171p5_powheg | 15,374 | 2,014 | 10,318 | 1,406 |
| TTLL_mtop173p5_powheg | 15,578 | 1,969 | 10,536 | 1,332 |
| TTLL_TuneCP5up_powheg | 17,456 | 2,140 | 11,905 | 1,483 |
| TTLL_TuneCP5down_powheg | 17,270 | 2,238 | 11,658 | 1,524 |
| TTLL_TuneCP5CR1_powheg | 15,144 | 1,965 | 10,438 | 1,413 |
| TTLL_TuneCP5CR2_powheg | 15,841 | 1,989 | 10,881 | 1,372 |
| TTLL_hdamp_up_powheg | 15,478 | 1,944 | 10,420 | 1,321 |
| TTLL_hdamp_down_powheg | 14,304 | 1,821 | 9,774 | 1,256 |
| **Run3 total** | **165,117** | **21,010** | **112,304** | **14,575** |
