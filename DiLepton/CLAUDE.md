# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Module Overview

DiLepton is the di-lepton analysis module for charged Higgs searches. It performs data-MC comparisons for dimuon (DIMU) and electron-muon (EMU) channels with full systematic uncertainty evaluation. It also measures b-tagging scale factors and jet veto efficiency maps.

**Channels:** DIMU (μμ), EMU (eμ)
**Eras:** 2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix, Run2, Run3

## Environment Setup

```bash
source setup.sh  # Run from ChargedHiggsAnalysisV3 root
```

## Key Commands

### Run Full Module
```bash
cd DiLepton && bash doThis.sh
```

### Individual Scripts
```bash
# Data-MC comparison plots (parallel, ~12 cores)
bash scripts/drawPlots.sh 2018 EMU
bash scripts/drawPlots.sh Run2 DIMU

# B-tagging scale factor plots
python python/plotBtagSF.py --era 2018 --channel EMU

# Jet veto efficiency maps (individual eras only, not Run2/Run3)
python python/plotJetVetoMap.py --era 2018 --channel DIMU

# Single histogram plot
python python/plot.py --era 2018 --channel EMU --histkey muons/1/pt
python python/plot.py --era Run2 --channel DIMU --histkey pair/mass --exclude L1Prefire
```

## Directory Structure

```
DiLepton/
├── doThis.sh                    # Main execution: drawPlots + BtagSF + JetVetoMap
├── configs/
│   ├── histkeys.json            # 449 histogram definitions (kinematic variables)
│   ├── samplegroup.json         # Sample grouping per era/channel
│   └── systematics.json         # Systematic variations per Run/channel
├── python/
│   ├── plot.py                  # Main data-MC comparison with systematic envelopes
│   ├── plotSimple.py            # Simplified plotting from SimpleDiLepton output
│   ├── plotBtagSF.py            # B-tagging SF extraction and plotting
│   └── plotJetVetoMap.py        # 2D jet eta-phi efficiency maps
├── scripts/
│   └── drawPlots.sh             # Batch runner with GNU parallel (12 cores)
├── plots/{era}/{channel}/       # Output plots organized by era/channel
│   └── [Central|No{Systematic}]/
└── results/                     # Result files
```

## Configuration Files

### configs/histkeys.json
Defines 449 kinematic histogram plotting parameters:
```json
{
  "muons/1/pt": {
    "xTitle": "p_{T}(#mu1)",
    "yTitle": "Events",
    "xRange": [0.0, 300.0],
    "rRange": [0.5, 1.5],
    "rebin": 5,
    "logy": false
  }
}
```
**Coverage:** muons 1-2, electrons, jets 1-5, b-jets 1-3, pair kinematics, MET, pileup variables.

### configs/samplegroup.json
Sample organization by era and channel:
- **data:** Era-specific data streams (DoubleMuon, MuonEG, etc.)
- **W:** WJets
- **Z:** DYJets, DYJets10to50
- **TT:** TTLL_powheg, TTLJ_powheg
- **ST:** 5 single-top samples
- **VV:** WW, WZ, ZZ

### configs/systematics.json
Systematic variations organized by Run period and channel:

**Run2 DIMU:** L1Prefire, PileupReweight, PileupJetIDSF, MuonIDSF, DblMuTrigSF, JetEn, JetRes, MuonEn, ElectronEn, ElectronRes, UnclusteredEn

**Run2 EMU:** Above + ElectronIDSF, EMuTrigSF, BtagSF_HFcorr/HFuncorr/LFcorr/LFuncorr

**Run3:** Same but without L1Prefire

## Python Script Details

### plot.py
Main analysis script for data-MC comparison.
- **Input:** `$WORKDIR/SKNanoOutput/DiLepton/RunDiMu_RunSyst/` and `RunEMu_RunSyst/`
- **Features:** Loads histograms across multiple eras, computes bin-by-bin systematic envelopes, optional `--exclude` for studying specific systematics
- **Output:** `plots/{era}/{channel}/[Central|No{Systematic}]/`
- **Arguments:** `--era`, `--channel`, `--histkey`, `--exclude`, `--debug`

### plotBtagSF.py
Extracts and plots b-tagging efficiency scale factors.
- **Input:** `weights/btagSF` histogram path in SKNanoOutput
- **Output:** `plots/{era}/{channel}/weights/btagSF.png`

### plotJetVetoMap.py
Creates 2D eta-phi efficiency maps for jet veto validation.
- **Constraint:** Only works with individual eras, not Run2/Run3 combined.
- **Output:** `plots/{era}/{channel}/JetVetoMap/efficiency.json` (also PNG)

### plotSimple.py
Alternative to plot.py reading from `SimpleDiLepton/` output (no systematics).

## Shell Script Details

### scripts/drawPlots.sh
Batch runner with GNU parallel (-j 12):
- **EMU:** ~42 histogram types × 7 systematic exclusions ≈ 294 parallel jobs
  - Exclusions: Central, L1Prefire, PileupReweight, MuonIDSF, ElectronIDSF, EMuTrigSF, BtagSF_HFcorr
- **DIMU:** ~44 histogram types × 5 systematic exclusions ≈ 220 parallel jobs
  - Exclusions: Central, L1Prefire, PileupReweight, MuonIDSF, DblMuTrigSF

## Input Data Paths

```
$WORKDIR/SKNanoOutput/DiLepton/RunDiMu_RunSyst/{era}/{Sample}.root
$WORKDIR/SKNanoOutput/DiLepton/RunEMu_RunSyst/{era}/{Sample}.root
```

Histogram structure: `{Channel}/{SystematicName}/{HistogramPath}`

## Common Issues

**Missing input ROOT files:** Check `$WORKDIR/SKNanoOutput/DiLepton/` exists and contains era-specific subdirectories.

**plotJetVetoMap.py fails for Run2/Run3:** This script only supports individual eras (2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix).

**Parallel job failures:** Check `logs/` directory for per-era/channel error messages.
