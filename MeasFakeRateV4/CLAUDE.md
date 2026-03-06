# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Module Overview

MeasFakeRateV4 is the **current baseline** for fake rate measurements in the charged Higgs analysis. It measures the probability for non-prompt leptons to pass tight ID requirements using the tight-to-loose method. Results are used downstream in TriLepton/SignalRegionStudyV2 for nonprompt background estimation.

**Key improvements over V3:**

- Consolidated lepton-type file structure (e.g., `MeasFakeEl_RunSyst` instead of trigger-specific paths)
- Enhanced error propagation with bin-by-bin fractional uncertainties
- Adaptive rebinning in closure tests (30% fractional error threshold)
- Better multi-era support with explicit Run2/Run3 era aggregation
- HEM veto support for 2018 electron analysis

**Eras:** 2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix, Run2, Run3
**Measures:** electron, muon

## Environment Setup

```bash
source setup.sh  # Run from ChargedHiggsAnalysisV3 root
```

## Key Commands

### Run Full Pipeline

```bash
cd MeasFakeRateV4
bash scripts/measFakeRate.sh 2018 electron  # Full pipeline for one era/measure
bash doThis.sh                              # Validation plots for all eras/measures
```

### Individual Steps

```bash
# Step 1: Extract histogram integrals from ROOT to JSON
python python/parseIntegral.py --era 2018 --measure electron

# Step 2: Calculate fake rates (data-driven and MC)
python python/measFakeRate.py --era 2018 --measure electron           # Data
python python/measFakeRate.py --era 2018 --measure electron --isMC    # Inclusive MC
python python/measFakeRate.py --era 2018 --measure electron --isQCD   # QCD MC
python python/measFakeRate.py --era 2018 --measure electron --isTT    # TT MC

# Step 3: Plot fake rates
python python/plotFakeRate.py --era 2018 --measure electron

# Step 4: Validation plots
python python/plotValidation.py --era 2018 --measure electron

# Step 5: Systematic variations
python python/plotSystematics.py --era 2018 --measure electron

# Closure tests
bash scripts/plotClosure.sh 2018

# Export to SKNanoAnalyzer
bash scripts/parseFakeRateToSKNano.sh
```

## Directory Structure

```
MeasFakeRateV4/
├── doThis.sh                       # Entry: validation plots for all eras/measures
├── configs/
│   ├── samplegroup.json            # Sample lists per era/measure (data, W, Z, TT, ST, VV, QCD)
│   ├── systematics.json            # Physics systematic variations per run/lepton type
│   └── histkeys.json               # Histogram configs for closure tests and validation
├── scripts/
│   ├── measFakeRate.sh             # Main pipeline (parseIntegral → measFakeRate → plots)
│   ├── plotValidation.sh           # Data-MC validation (parallel, all eras)
│   ├── plotClosure.sh              # Closure test plots (parallel, all channels)
│   └── parseFakeRateToSKNano.sh    # Export results to SKNanoAnalyzer format
├── python/
│   ├── common.py                   # findbin(), run period helpers, bin definitions
│   ├── parseIntegral.py            # ROOT histograms → JSON integrals
│   ├── measFakeRate.py             # Core fake rate calculation (data + MC modes)
│   ├── plotFakeRate.py             # 2D fake rate histogram → 1D projections
│   ├── plotValidation.py           # Data vs MC comparison plots
│   ├── plotClosure.py              # Closure test with chi-squared validation
│   └── plotSystematics.py          # Systematic uncertainty plots per eta bin
├── results/{era}/
│   ├── JSON/{measure}/             # Histogram integrals + prompt scales (JSON)
│   └── ROOT/{measure}/             # 2D fake rate histograms (ROOT)
└── plots/{era}/{measure}/          # Output validation and systematic plots
```

## pT-η Bin Definitions (from common.py)

**Electrons:** pT bins = [15, 17, 20, 25, 35, 50, 100, 200] GeV; |η| bins = [0, 0.8, 1.479, 2.5]

**Muons:** pT bins = [10, 12, 14, 17, 20, 30, 50, 100, 200] GeV; |η| bins = [0, 0.9, 1.6, 2.4]

Bin names are like `ptcorr_15to17_EB1` (electron) or `ptcorr_10to12_IB` (muon).

## Configuration Files

### configs/samplegroup.json

Per-era/measure sample definitions:

- **data:** Era-specific data periods
- **W, Z, TT, ST, VV:** Standard MC backgrounds
- **QCD_EMEnriched / QCD_bcToE / QCD_MuEnriched:** QCD subsamples (era-specific)
- **noHEMVeto variant:** Alternate electron selection for 2018 HEM veto studies

### configs/systematics.json

Physics systematics by run period and lepton type:

- **Run2:** L1Prefire, PileupReweight, PileupJetIDSF, Jet/Muon/Electron energy variations
- **Run3:** Excludes L1Prefire (not applicable)

### configs/histkeys.json

Histogram configurations for closure tests and validation:

- Pair masses, Z mass, pT, eta, MET distributions
- Per-histogram: xTitle, yTitle, xRange, rebin factor

## Python Script Details

### parseIntegral.py

Extracts MT histogram integrals from ROOT files.

- **Input:** `SKNanoOutput/MeasFakeRateV4/{lepton_type}_RunSyst/{era}/{sample}.root`
  - Histogram path: `{ptcorr_bin}/QCDEnriched/{trig_prefix}/{wp}/{syst}/MT`
- **Output:** `results/{era}/JSON/{measure}/{sample}_{wp}.json`
- **Supports:** Multiple systematics (Central, MotherJetPt_Up/Down, RequireHeavyTag)

### measFakeRate.py

Core fake rate calculation.

- **Data mode:** `fake = data - prompt × scale`; prompt normalization from Z-enriched region
- **MC modes:** `--isMC` (inclusive), `--isQCD`, `--isTT` (mutually exclusive)
- **Output:** 2D histograms (η × pT) in `results/{era}/ROOT/{measure}/`
- **Prompt scale:** Derived from `ZEnriched/{trig_prefix}/{wp}/Central/ZCand/mass`

### plotClosure.py

Validates fake rates via closure test on tri-lepton signal/sideband.

- **Features:** Adaptive rebinning (30% fractional uncertainty threshold), chi-squared test (shape-only), systematic optimization
- **Output:** JSON with chi2/ndf, p-value, bin deviation metrics
- **Supports:** Multiple systematics (Central, TT, bjet, cjet, ljet)

### plotSystematics.py

Plots fractional systematic variations per eta bin.

- **Systematics:** PromptNorm_Up/Down, MotherJetPt_Up/Down, RequireHeavyTag

## Input Data Paths

```
$WORKDIR/SKNanoOutput/MeasFakeRateV4/{lepton_type}_RunSyst/{era}/{Sample}.root
```

e.g., `MeasFakeEl_RunSyst/2018/`, `MeasFakeMu_RunSyst/2022/`

Histogram structure: `{ptcorr_bin}/{region}/{trig_prefix}/{wp}/{syst}/{histname}`

## Output Paths

```
MeasFakeRateV4/results/{era}/JSON/{measure}/    # Intermediate integrals
MeasFakeRateV4/results/{era}/ROOT/{measure}/    # 2D fake rate histograms
MeasFakeRateV4/plots/{era}/{measure}/           # Validation plots
```

## Key Differences from MeasFakeRateV3

| Aspect | V3 | V4 (current baseline) |
|---|---|---|
| File naming | `MeasFakeEl8_RunSyst` (trigger-specific) | `MeasFakeEl_RunSyst` (lepton-type unified) |
| Closure tests | Basic | Adaptive rebinning, chi-squared optimization |
| Era aggregation | Limited | Full Run2/Run3 merged analysis support |
| HEM veto | Not supported | `--noHEMVeto` flag for 2018 electrons |
| Error propagation | Basic | Bin-by-bin fractional uncertainty |

## Common Issues

**Wrong input path:** V4 uses `{lepton_type}_RunSyst` not `{trigger}_RunSyst` as in V3. Double-check file path conventions when comparing results.

**HEM veto for 2018:** The `--noHEMVeto` flag selects an alternate electron sample with relaxed HEM jet veto. Results stored in `results/{era}/JSON/electron/noHEMVeto/`.

**Prompt scale errors:** If Z-enriched region has insufficient statistics, the prompt scale calculation will fail. Check the Z mass window histogram.

**Closure test rebinning:** The adaptive rebinning stops at 30% fractional error. If the closure plot has very few bins, statistics are too low in some pT regions.
