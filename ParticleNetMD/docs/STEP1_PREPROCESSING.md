# Step 1: Data Preparation, Augmentation, and Validation

Complete guide to preparing PyTorch Geometric datasets for ParticleNetMD training, from EvtTreeProducer ROOT files to validated `.pt` datasets.

## 0. Prerequisites

### Environment

```bash
cd /path/to/ChargedHiggsAnalysisV3
source setup.sh
cd ParticleNetMD
```

### Upstream Input

Event trees from [EvtTreeProducer](../../SKNanoAnalyzer/Analyzers/src/EvtTreeProducer.cc), stored at:
```
$WORKDIR/SKNanoOutput/EvtTreeProducer/{channel}/{era}/{sample}.root
```

- **Channels:** `Run1E2Mu` (1e+2mu), `Run3Mu` (3mu)
- **Run2 eras:** 2016preVFP, 2016postVFP, 2017, 2018
- **Run3 eras:** 2022, 2022EE, 2023, 2023BPix
- **Tree name:** `Events`
- **Key branches:** `MuonPt/Eta/Phi/Mass/Charge/MiniIso/IsTight` (arrays), `ElectronPt/Eta/Phi/Mass/Charge/MiniIso/IsTight`, `JetPt/Eta/Phi/Mass/BtagScore/IsBtagged`, `genWeight`, `puWeight`, `prefireWeight`

### External Data

- **Fake rate tables** (for nonprompt): pre-existing correctionlib JSONs at `$SKNANO_DATA/{era}/{MUO,EGM}/fakerate_TopHNT.json`
- **Conditional promotion tables** (for diboson): generated in Step 2a below

---

## 1. Overview

ParticleNetMD trains a **4-class GNN classifier** (signal vs nonprompt/diboson/ttX) with DisCo loss for mass decorrelation. The dataset pipeline converts ROOT event trees into PyTorch Geometric graph objects.

### Training Classes

| Class | Samples | Selection | Augmentation |
|-------|---------|-----------|--------------|
| Signal | 6 mass points | Tight+Bjet | None |
| Nonprompt | 9 TTLL variants | LNT+Bjet | Fake rate reweighting ([details](../DataAugment/nonprompt.md)) |
| Diboson | WZ + ZZ | Tight+0tag promoted | Rank-based b-jet promotion |
| ttX | TTZ + tZq | Tight+Bjet | None ([details](../DataAugment/ttX.md)) |

### Graph Representation

Each event becomes a graph with particles as nodes:

**Node features (9-dim):** `[E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]`

**Edges:** k=4 nearest neighbors by DeltaR, with `[DeltaR]` as edge attribute.

**Per-event metadata:**
- `weight`: event weight (genW x puW x prefireW x augmentation weight)
- `mass1`, `mass2`: OS muon pair invariant masses (for DisCo decorrelation)
- `graphInput`: 8-dim era one-hot encoding

### Fold Structure

5-fold cross-validation with deterministic assignment (shuffle with seed=42, round-robin `fold = i % 5`):

| Fold role | Fold indices (relative to test fold `k`) |
|-----------|------------------------------------------|
| Training | `(k+1)%5, (k+2)%5, (k+3)%5` |
| Validation | `(k+4)%5` |
| Test | `k` |

Configuration in [`configs/SglConfig.json`](../configs/SglConfig.json): `train_folds: [0,1,2]`, `valid_folds: [3]`, `test_folds: [4]`.

---

## 2. Data Augmentation Prerequisites

Two augmentation-related scripts must run **before** dataset creation.

### Step 2a: Diboson Conditional Promotion Tables (required)

Computes conditional probabilities P(n_bjets, rank | nJets) from genuine Tight+Bjet diboson events, used to promote 0-tag diboson events with physically-motivated b-jet assignments.

**Script:** [`python/dibosonRankPromote.py`](../python/dibosonRankPromote.py)

```bash
python python/dibosonRankPromote.py
```

**What it does:**
1. Loads WZ + ZZ events from EvtTreeProducer output (all eras, both channels)
2. From genuine Tight+Bjet events, learns:
   - P(n_bjets = 1 | nJets group) — single vs multi b-jet probability
   - P(rank | n_bjets=1, nJets) — which jet to promote (by pT rank)
   - P(pair | n_bjets>=2, nJets) — which jet pair to promote
3. Computes two-layer calibration weights:
   - Layer 1: nJets shape reweighting
   - Layer 2: promoted b-jet (pT, |eta|) shape reweighting
4. Generates validation plots (4-way comparison: genuine vs promoted)

**Output:**
```
DataAugment/diboson/plots/rank_promote/
├── conditional_tables.json    ← REQUIRED by saveDataset.py --sample-type diboson
├── Run2/
│   ├── Run1E2Mu/              (validation plots: el_pt, mu1_pt, jet1_pt, ...)
│   └── Run3Mu/
└── Run3/
    ├── Run1E2Mu/
    └── Run3Mu/
```

**nJets groups:**
- Run2 (CHS jets): 5 groups (1, 2, 3, 4, >=5)
- Run3 (PUPPI jets): 3 groups (1, 2, >=3)

**Calibration weight binning:**
- eta: [0.0, 0.8, 1.6, 2.1, 2.5] (4 bins)
- pT: [20, 30, 50, 70, 100, 140, 200, 300, 10000] GeV (8 bins)

### Step 2b: Nonprompt LNT Promotion Validation (recommended)

Validates that loose-not-tight events reweighted with fake rates reproduce the genuine tight distribution.

**Script:** [`python/nonpromptPromotion.py`](../python/nonpromptPromotion.py)

```bash
python python/nonpromptPromotion.py
```

**What it does:**
1. Loads all 9 TTLL variants from EvtTreeProducer (both channels, all eras)
2. For each kinematic variable, produces normalized comparisons of:
   - Genuine (Tight+Bjet) — target
   - LNT (raw pT) — uncorrected
   - LNT (ptCorr) — kinematic correction only
   - LNT (ptCorr + FR weight) — fully corrected
3. Reports augmentation ratio and fake rate ranges

**Output:**
```
DataAugment/nonprompt/plots/lnt_promote/
├── summary.json
├── Run2/
│   ├── Run1E2Mu/              (18 variables: el_pt/eta, mu1/mu2_pt/eta, jets, bjets, ...)
│   └── Run3Mu/                (18 variables: mu1/mu2/mu3_pt/eta, jets, bjets, ...)
└── Run3/
    ├── Run1E2Mu/
    └── Run3Mu/
```

**Key validation checks:**
- LNT (ptCorr + FR wt) should closely match Genuine shape for all variables
- Augmentation ratio should be ~6.6x (418k LNT+Bjet / 63k genuine Tight+Bjet)

---

## 3. Dataset Creation

### Quick Start: Full Pipeline

```bash
./scripts/saveDatasets.sh
```

This runs all sample types in parallel using GNU parallel. Total output: ~200 `.pt` files in `dataset/samples/`.

For details on what this script does, see [`scripts/saveDatasets.sh`](../scripts/saveDatasets.sh).

### Per-Class Details

#### 3a. Signal (`--sample-type signal`)

**Selection:** All leptons pass tight ID + at least 1 b-tagged jet (Tight+Bjet).

**Conversion function:** [`Preprocess.rtfileToDataList()`](../python/Preprocess.py)

**Particle sources:** [`getMuons()`](../python/DataFormat.py) (tight muons), [`getElectrons()`](../python/DataFormat.py) (tight electrons), [`getJets()`](../python/DataFormat.py) (all jets, split into jets + bjets)

**Weight:** `genWeight x puWeight x prefireWeight`

**Mass computation:** [`compute_os_pair_masses(muons)`](../python/Preprocess.py) — sorted OS muon pair invariant masses (mass1 < mass2). For Run1E2Mu (only 1 OS pair): mass2 = -1.

**Samples (6 mass points):**
```
TTToHcToWAToMuMu-MHc100_MA95
TTToHcToWAToMuMu-MHc115_MA87
TTToHcToWAToMuMu-MHc130_MA90
TTToHcToWAToMuMu-MHc145_MA92
TTToHcToWAToMuMu-MHc160_MA85
TTToHcToWAToMuMu-MHc160_MA98
```

**Running single sample:**
```bash
python python/saveDataset.py --sample TTToHcToWAToMuMu-MHc130_MA90 \
  --sample-type signal --channel Run1E2Mu
```

**Bulk (from saveDatasets.sh):**
```bash
parallel run_save {1} signal {2} ::: "${signals[@]}" ::: "${channels[@]}"
```

#### 3b. Nonprompt (`--sample-type nonprompt`)

**Selection:** At least 1 lepton fails tight ID + at least 1 b-tagged jet (LNT+Bjet). This is the complement of Tight+Bjet — events where the nonprompt lepton is retained at loose level but rejected at tight.

**Conversion function:** [`Preprocess.rtfileToDataList_nonprompt()`](../python/Preprocess.py)

**Particle sources:** [`getAllMuons()`](../python/DataFormat.py) and [`getAllElectrons()`](../python/DataFormat.py) — returns all leptons (tight + loose-not-tight) with ptCorr applied to 4-momentum.

**ptCorr formula:**
```
ptCorr = pT x (1 + max(0, MiniPFRelIso - 0.1))
```
- Tight leptons (miniIso < 0.1): ptCorr ~ pT
- LNT leptons (miniIso > 0.1): ptCorr > pT (approximates tight pT)
- For Run3 muons, ptCorr is capped at 50 GeV **for fake rate lookup only** (node features use uncapped ptCorr)

**Fake rate weight:**
```
w = -1 x Product[ -f(ptCorr, |eta|) / (1 - f(ptCorr, |eta|)) ]  over non-tight leptons
```
where f is the era-specific fake rate from correctionlib JSON ([`load_fakerate_tables()`](../python/Preprocess.py)).

**Fake rate binning:**
- Muon: |eta| [0, 0.9, 1.6, 2.4] x ptCorr [10, 12, 14, 17, 20, 30, 50, 100] (3x7 = 21 bins)
- Electron: |scEta| [0, 0.8, 1.479, 2.5] x ptCorr [15, 17, 20, 25, 35, 50, 100] (3x6 = 18 bins)
  - Note: electron fake rate uses supercluster eta (`ElectronScEtaColl`), not tracking eta — the bin edges correspond to ECAL barrel/endcap boundaries.

**Total weight:** `genWeight x puWeight x prefireWeight x fake_rate_weight`

**Training strategy:** LNT events are used for training only; genuine Tight+Bjet events are reserved for validation/test (blinding). See [`nonprompt.md`](../DataAugment/nonprompt.md).

**Samples (9 TTLL variants):**
```
TTLL_powheg                   (nominal)
TTLL_mtop171p5_powheg         (top mass -1 GeV)
TTLL_mtop173p5_powheg         (top mass +1 GeV)
TTLL_TuneCP5up_powheg         (UE tune up)
TTLL_TuneCP5down_powheg       (UE tune down)
TTLL_TuneCP5CR1_powheg        (color reconnection model 1)
TTLL_TuneCP5CR2_powheg        (color reconnection model 2)
TTLL_hdamp_up_powheg          (ME-PS matching up)
TTLL_hdamp_down_powheg        (ME-PS matching down)
```

For sample selection rationale (why these 9, why others are excluded), see [`nonprompt.md`](../DataAugment/nonprompt.md).

**Running single sample:**
```bash
python python/saveDataset.py --sample Skim_TriLep_TTLL_powheg \
  --sample-type nonprompt --channel Run1E2Mu
```

**Bulk (from saveDatasets.sh):**
```bash
parallel run_save {1} nonprompt {2} ::: "${nonprompt[@]}" ::: "${channels[@]}"
```

#### 3c. Diboson (`--sample-type diboson`)

**Selection:** All leptons pass tight ID + no b-tagged jets + at least 1 jet (Tight+0tag+nJets>0). These 0-tag events are then **promoted** to have b-jets via conditional rank-based sampling.

**Conversion function:** [`Preprocess.rtfileToDataList_diboson()`](../python/Preprocess.py)

**Promotion logic ([`promote_event()`](../python/Preprocess.py)):**
1. Sample n_bjets from P(n_bjets | nJets) learned from genuine events
2. If n_bjets=1: sample jet rank from P(rank | n_bjets=1, nJets)
3. If n_bjets>=2: sample jet pair code from P(pair | n_bjets>=2, nJets)
4. Promoted jets are flagged as b-jets in node features (isBjet=1)

Uses deterministic seeding: `TRandom3(entry_seed x 2654435761 + 12345)` for reproducibility.

**Calibration weight ([`compute_calibration_weight()`](../python/Preprocess.py)):**
```
w = njets_weight(nJets) x Product[ pteta_weight(pT_i, |eta_i|) for promoted b-jets ]
```

**Total weight:** `genWeight x puWeight x prefireWeight x calibration_weight`

**Samples and era-split handling:**

ZZ has the same sample name across all eras:
```bash
# ZZ: all eras at once
python python/saveDataset.py --sample Skim_TriLep_ZZTo4L_powheg \
  --sample-type diboson --channel Run1E2Mu
```

WZ uses different generators for Run2 vs Run3, merged into a single output:
```bash
# WZ Run2 pass (creates new files)
python python/saveDataset.py --sample Skim_TriLep_WZTo3LNu_amcatnlo \
  --sample-type diboson --channel Run1E2Mu \
  --eras run2 --output-name Skim_TriLep_WZTo3LNu

# WZ Run3 pass (appends to existing fold files)
python python/saveDataset.py --sample Skim_TriLep_WZTo3LNu_powheg \
  --sample-type diboson --channel Run1E2Mu \
  --eras run3 --output-name Skim_TriLep_WZTo3LNu --append
```

The `--append` flag loads existing `.pt` files and concatenates new events before re-saving with updated fold assignments.

#### 3d. ttX (`--sample-type ttX`)

**Selection:** Tight+Bjet (same as signal).

**Conversion function:** [`Preprocess.rtfileToDataList()`](../python/Preprocess.py) (same as signal)

**Weight:** `genWeight x puWeight x prefireWeight`

**Samples (2 included, 2 excluded):**

| Sample | Included | Reason |
|--------|----------|--------|
| TTZ (TTZToLLNuNu / TTZ_M50) | Yes | ~76% of ttX yield |
| tZq | Yes | ~24% of ttX yield |
| TTHToNonbb | **No** | Very small sumW relative to TTZ and tZq |
| TTWToLNu | **No** | Only 917 Run3 events after Tight+Bjet (~180/fold) |

For full rationale, see [`ttX.md`](../DataAugment/ttX.md).

**Era-split handling for TTZ:**
```bash
# TTZ Run2 (TTZToLLNuNu)
python python/saveDataset.py --sample Skim_TriLep_TTZToLLNuNu \
  --sample-type ttX --channel Run1E2Mu \
  --eras run2 --output-name Skim_TriLep_TTZ

# TTZ Run3 (TTZ_M50) — append
python python/saveDataset.py --sample Skim_TriLep_TTZ_M50 \
  --sample-type ttX --channel Run1E2Mu \
  --eras run3 --output-name Skim_TriLep_TTZ --append
```

**Other ttX (single name, all eras):**
```bash
python python/saveDataset.py --sample Skim_TriLep_tZq --sample-type ttX --channel Run1E2Mu
```

---

## 4. Output Structure

```
dataset/samples/
├── signals/
│   ├── TTToHcToWAToMuMu-MHc100_MA95/
│   │   ├── Run1E2Mu_fold-0.pt
│   │   ├── Run1E2Mu_fold-1.pt
│   │   ├── Run1E2Mu_fold-2.pt
│   │   ├── Run1E2Mu_fold-3.pt
│   │   ├── Run1E2Mu_fold-4.pt
│   │   ├── Run1E2Mu_stats.json
│   │   ├── Run3Mu_fold-0.pt
│   │   └── ...
│   └── ... (5 more mass points)
└── backgrounds/
    ├── Skim_TriLep_TTLL_powheg/         (nonprompt)
    ├── Skim_TriLep_TTLL_mtop171p5_powheg/
    ├── ... (7 more TTLL variants)
    ├── Skim_TriLep_WZTo3LNu/           (diboson, merged WZ)
    ├── Skim_TriLep_ZZTo4L_powheg/      (diboson)
    ├── Skim_TriLep_TTZ/                (ttX, merged TTZ)
    └── Skim_TriLep_tZq/                (ttX)
```

**Total files:** 6 signals x 2 channels x 5 folds + 13 backgrounds x 2 channels x 5 folds = **190 `.pt` files**

### PyTorch Geometric Data Object

Each `.pt` file contains a list of `torch_geometric.data.Data` objects. Per event:

| Attribute | Shape | Description |
|-----------|-------|-------------|
| `x` | (N, 9) | Node features: [E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet] |
| `edge_index` | (2, E) | k-NN graph edges (k=4, DeltaR-based) |
| `edge_attribute` | (E, 1) | [DeltaR] per edge |
| `weight` | scalar | Event weight |
| `mass1` | scalar | Lower OS muon pair mass (for DisCo) |
| `mass2` | scalar | Higher OS muon pair mass (-1 if only 1 pair) |
| `graphInput` | (8,) | Era one-hot: [2016preVFP, ..., 2023BPix] |

### Stats JSON

Each `{channel}_stats.json` logs:
- Timestamp, sample name, sample type, channel
- Per-era event counts and total weights
- Per-fold event counts
- Total events and weight

### Combined Channel

The `Combined` channel is not pre-computed. [`DynamicDatasetLoader`](../python/DynamicDatasetLoader.py) merges `Run1E2Mu` + `Run3Mu` on-the-fly at training time:
```python
data_combined = load("Run1E2Mu", fold) + load("Run3Mu", fold)
```

---

## 5. Dataset Validation

Two-phase validation pipeline using [`python/validateDatasets.py`](../python/validateDatasets.py).

### Step 5a: Fill Phase (slow, parallelized)

Loads `.pt` files, extracts 38 kinematic observables, fills ROOT histograms, and saves to checkpoints.

```bash
python python/validateDatasets.py --workers 30
```

**CLI options:**
- `--channel {Run1E2Mu|Run3Mu|all}` — which channels to process (default: all)
- `--folds 0,1,2,3,4` — which folds to include
- `--workers N` — parallel workers (default: 8; use 30 on NERSC)

**Parallelization:** `multiprocessing.Pool` over (process, channel, fold) = 6 processes x 2 channels x 5 folds = **60 independent jobs**. Each worker initializes ROOT in batch mode and sets `torch.set_num_threads(1)`.

**Output:**
```
DataAugment/validation/histograms/{process}/{channel}_fold{fold}.root
  ├── Run2/
  │   ├── lep1_pt (TH1D)
  │   ├── lep1_eta
  │   ├── ... (38 observables from histkeys_validate.json)
  │   └── metadata (1-bin TH1D: content=event_count, error=weight_sum)
  └── Run3/
      └── ...
```

**Processes validated:**
- 3 signal benchmarks: MHc130_MA90, MHc100_MA95, MHc160_MA85
- 3 background groups: nonprompt (all 9 TTLL merged), diboson (WZ+ZZ), ttX (TTZ+tZq)

**Observable definitions:** [`configs/histkeys_validate.json`](../configs/histkeys_validate.json) — 38 kinematic variables covering leptons (pT/eta/phi), muons, electrons, jets, b-jets, event-level (nJets, nBjets, HT, MET), and mass decorrelation variables (mass1, mass2).

### Step 5b: Plot Phase (fast)

Reads ROOT checkpoint files, merges across folds, and produces overlay plots.

```bash
python python/validateDatasets.py --plotting
```

**Output:**
```
DataAugment/validation/
├── Run2/
│   ├── Run1E2Mu/
│   │   ├── class_overlay/          (all 6 processes overlaid per observable)
│   │   │   ├── lep1_pt.png
│   │   │   ├── ele1_eta.png
│   │   │   └── ...
│   │   └── fold_overlay/           (per-process, 5 folds overlaid)
│   │       ├── nonprompt/
│   │       │   ├── lep1_pt.png
│   │       │   └── ...
│   │       └── ...
│   └── Run3Mu/
│       └── ...
├── Run3/
│   └── ...
└── {Run}/{Channel}/summary.json    (event counts, weight sums per process/fold)
```

### Validation Checks

1. **Class overlay plots**: Compare kinematic shapes across signal and background classes
2. **Fold overlay plots**: Verify statistical consistency across 5 folds (no fold-dependent biases)
3. **Diboson bjet1_pt**: Verify promoted b-jet pT distribution is physically reasonable
4. **Nonprompt ele1_eta**: Check that endcap population (~20%) matches other classes
5. **mass1, mass2**: Verify OS muon pair mass distributions peak near Z mass

---

## 6. Training Budget

Training target: **40,000 events per fold per class** ([`max_events_per_fold_per_class`](../configs/SglConfig.json)). Run2/Run3 ratio: ~34k/16k (luminosity 138/62 ~ 2.2:1).

| Class | Run2/fold | % of 34k | Run3/fold | % of 16k | Total/fold | Status |
|-------|----------:|---------:|----------:|---------:|-----------:|--------|
| Signal (typical) | ~55k | 162% | ~22k | 138% | ~77k | Capped at 40k |
| Nonprompt (LNT+Bjet) | 63,403 | 186% | 20,217 | 126% | 83,620 | Capped at 40k |
| Diboson (promoted 0-tag) | 107,647 | 317% | 19,798 | 124% | 127,445 | Capped at 40k |
| ttX (Tight+Bjet) | 167,281 | 492% | 24,566 | 154% | 191,847 | Capped at 40k |

All classes comfortably meet the target. Run3 is the tighter constraint but still 124-154% headroom. Full per-sample statistics in [`DataAugment/dataset.md`](../DataAugment/dataset.md).

---

## Quick Reference: Full Pipeline

```bash
# 0. Setup
cd /path/to/ChargedHiggsAnalysisV3
source setup.sh
cd ParticleNetMD

# 1. Diboson promotion tables (required)
python python/dibosonRankPromote.py

# 2. Nonprompt validation (recommended)
python python/nonpromptPromotion.py

# 3. Create all datasets
./scripts/saveDatasets.sh

# 4. Validate datasets
python python/validateDatasets.py --workers 30    # fill phase
python python/validateDatasets.py --plotting       # plot phase

# 5. Check output
ls dataset/samples/signals/*/Run1E2Mu_fold-0.pt | wc -l    # expect 6
ls dataset/samples/backgrounds/*/Run1E2Mu_fold-0.pt | wc -l # expect 13
ls plots/Validation/Run2/Run1E2Mu/class_overlay/*.png | wc -l  # expect 38
```

## References

- [Dataset event counts](../DataAugment/dataset.md) — full per-sample statistics
- [Nonprompt augmentation](../DataAugment/nonprompt.md) — LNT promotion details, sample selection
- [ttX sample selection](../DataAugment/ttX.md) — TTW exclusion rationale
- [Training configuration](../configs/SglConfig.json) — background_groups, folds, hyperparameters
