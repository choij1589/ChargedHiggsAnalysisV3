# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Module Overview

TriLepton is the tri-lepton analysis module for charged Higgs searches. It performs data-MC comparisons across 9 channels (signal regions + control regions) with systematic uncertainties, signal overlays, and blinding. It also measures conversion scale factors (ConvSF) and WZ Njet scale factors (WZNjSF) for use downstream.

**Channels:** SR1E2Mu, SR3Mu, ZFake1E2Mu, ZFake3Mu, ZG1E2Mu, ZG3Mu, WZ1E2Mu, WZ3Mu, TTZ2E1Mu
**Eras:** 2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix, Run2, Run3
**Signal mass points:** 38 charged Higgs mass points (MHc70–160 GeV, MA15–155 GeV)

## Environment Setup

```bash
source setup.sh  # Run from ChargedHiggsAnalysisV3 root
```

## Key Commands

### Run Full Module
```bash
cd TriLepton && bash doThis.sh          # All plotting
cd TriLepton && bash doSampleBreakdown.sh  # Event yield extraction
```

### Individual Scripts
```bash
# Data-MC comparison plots
bash scripts/drawPlots.sh 2018 SR1E2Mu
bash scripts/drawPlots.sh Run2 ZFake3Mu

# Single histogram plot (SR regions are blinded by default in drawPlots.sh)
python python/plot.py --era 2018 --channel SR1E2Mu --histkey pair/mass --blind
python python/plot.py --era Run2 --channel WZ1E2Mu --histkey jets/size --exclude WZSF

# Event yield extraction
python python/sampleBreakdown.py --era 2018 --channel SR1E2Mu --blind
python python/sampleBreakdown.py --era Run2 --channel ZFake1E2Mu --onZ

# Scale factor measurements
python python/measConvSF.py --era 2018 --channel ZG1E2Mu
python python/measWZNjSF.py --era Run2 --channel WZCombined
bash scripts/measConvSF.sh  # All eras in parallel
bash scripts/parseCvSFToCommonData.sh  # Copy to Common/Data/ConvSF
```

## Directory Structure

```
TriLepton/
├── doThis.sh                      # Main: drawPlots for all eras/channels
├── doSampleBreakdown.sh           # Event yield extraction with errors
├── configs/
│   ├── histkeys.json              # 100+ kinematic histogram definitions
│   ├── histkeys.score.json        # ML score distributions per signal mass point
│   ├── samplegroup.json           # Sample lists by era/channel
│   ├── systematics.json           # Systematic variations (same as DiLepton)
│   ├── nonprompt.json             # Nonprompt rate thresholds by Run/channel
│   └── signals.json               # 38 signal mass points
├── python/
│   ├── plot.py                    # Data-MC with signal overlay, blinding, K-factors
│   ├── sampleBreakdown.py         # Event yields with stat/syst error breakdown
│   ├── measConvSF.py              # Conversion SF from ZG control regions
│   ├── measWZNjSF.py              # WZ Njet SF from WZ control regions
│   └── utils.py                   # Path building utilities
├── scripts/
│   ├── drawPlots.sh               # Batch runner with GNU parallel
│   ├── measConvSF.sh              # Conversion SF for all eras (parallel)
│   └── parseCvSFToCommonData.sh   # Copy ConvSF JSON to Common/Data/
├── results/
│   ├── {era}/{channel}/           # sample_breakdown_*.json files
│   ├── ZG{channel}/{era}/ConvSF.json
│   └── WZCombined/{Run}/WZNjetsSF.json
└── plots/                         # Output plots
```

## Configuration Files

### configs/histkeys.json
100+ histogram plotting parameters covering:
- muons/electrons 1-3 (pt/eta/phi)
- pair kinematics (mass, pt, eta, phi — including on-Z and lowM variants)
- ZCand/mass, convLep/pt
- jets/bjets 1-4, MET
- distance metrics (dR_* variables)

### configs/samplegroup.json
Sample lists by era and lepton channel (1E2Mu, 2E1Mu, 3Mu):
- **data/nonprompt:** Data streams (matrix method uses same data for nonprompt)
- **conv:** DYJets, TTG, WWG (conversion background)
- **ttX:** TTWToLNu, TTZToLLNuNu, TTHToNonbb, tZq
- **diboson:** WZTo3LNu_amcatnlo, ZZTo4L_powheg
- **others:** tHq, WWW, WWZ, WZZ, ZZZ, TTTT, VBFHToZZTo4L, ggHToZZTo4L

### configs/nonprompt.json
MVA score thresholds for nonprompt lepton selection in the matrix method:
```json
{
  "Run2": {"Run1E2Mu": 0.25, "Run2E1Mu": 0.3, "Run3Mu": 0.3},
  "Run3": {"Run1E2Mu": 0.3, "Run2E1Mu": 0.35, "Run3Mu": 0.35}
}
```

### configs/signals.json
38 signal mass points: MHc70_MA15 through MHc160_MA155 (MHc85_MA21 missing from 2016preVFP).

## Python Script Details

### plot.py
Full data-MC comparison with signal overlay.
- **Input:** `SKNanoOutput/PromptAnalyzer/` (prompt MC) + `MatrixAnalyzer/` (nonprompt)
- **Features:**
  - Fixed background stacking order: others → conv → diboson → ttX → nonprompt
  - ConvSF and WZNjSF reweighting from JSON files
  - Blinding for SR1E2Mu and SR3Mu (`--blind`)
  - Signal overlay with configurable mass points and scale factor (default 10×)
  - K-factor application from `Common/Data/KFactors.json`
- **Arguments:** `--era`, `--channel`, `--histkey`, `--exclude` (WZSF or ConvSF), `--blind`, `--signals`, `--signal-scale`, `--debug`

### sampleBreakdown.py
Extracts event yields with detailed error breakdown.
- **Histogram selection by channel:**
  - SR/ZFake 1E2Mu: `pair/mass` (or `pair_onZ/mass` with `--onZ`)
  - SR/ZFake 3Mu: `pair_lowM/mass` (or `pair_lowM_onZ/mass` with `--onZ`)
  - ZG/WZ control regions: `ZCand/mass` (always)
- **Output JSON:**
  ```json
  {"sample_name": {"events": 123.45, "stat_error": 2.3, "syst_error": 5.6, "total_error": 6.1}}
  ```

### measConvSF.py
Measures electron conversion scale factors from ZG→e+γ events.
- **Channels:** ZG1E2Mu, ZG3Mu, ZGCombined
- **Output:** `results/ZG{channel}/{era}/ConvSF.json` (correctionlib format)

### measWZNjSF.py
Measures WZ cross-section ratio as a function of jet multiplicity.
- **Channels:** WZ1E2Mu, WZ3Mu, WZCombined
- **Note:** Different max_nj for Run2 (5) vs Run3 (3); different WZ sample names per run
- **Output:** `results/WZCombined/{Run}/WZNjetsSF.json`

### utils.py
Path building: `build_sknanoutput_path()` detects channel type and selects the correct analyzer (PromptAnalyzer vs MatrixAnalyzer) and flag suffix.

## Scale Factor Distribution

After measuring conversion SFs, distribute to the common data directory:
```bash
bash scripts/parseCvSFToCommonData.sh
# Copies: TriLepton/results/ZG{channel}/{era}/ConvSF.json
#     to: Common/Data/ConvSF/Run2|Run3/{channel}/{era}.json
```

## Input Data Paths

```
$WORKDIR/SKNanoOutput/PromptAnalyzer/Run{channel}_RunSyst/{era}/{Sample}.root
$WORKDIR/SKNanoOutput/MatrixAnalyzer/Run{channel}_RunNoWZSF/{era}/{Sample}.root
```

Histogram structure: `{Channel}/[Systematic]/[Subpath]`

## Common Issues

**Signal regions are blinded by default** in `drawPlots.sh`. Use `--blind` flag explicitly when calling `plot.py` directly.

**WZNjSF max_nj difference:** Run2 supports up to 5 jets; Run3 supports up to 3. The WZ sample name also differs (amcatnlo vs powheg).

**ConvSF must be distributed** before TriLepton plots accurately reflect conversion backgrounds. Run `parseCvSFToCommonData.sh` after measuring ConvSF.

**Missing `--onZ` for on-Z selections:** The `--onZ` flag in `sampleBreakdown.py` only applies to SR and ZFake channels; ZG/WZ always use `ZCand/mass`.
