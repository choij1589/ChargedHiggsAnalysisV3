# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Module Overview

MeasTrigEff measures trigger efficiencies for di-lepton and multi-lepton trigger paths used in the charged Higgs analysis. It uses the reference trigger method (tag-and-probe) to measure:
- EMu leg efficiencies (Mu8El23, Mu23El12 paths)
- Double muon pairwise filter efficiencies (DblMuDZ, DblMuDZM, DblMuM, EMuDZ)
- Closure tests for di-lepton trigger combinations

Results feed into trigger scale factor corrections applied in DiLepton and TriLepton analyses.

**Eras:** 2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix

## Environment Setup

```bash
source setup.sh  # Run from ChargedHiggsAnalysisV3 root
```

## Key Commands

### Individual Steps (no single doThis.sh)
```bash
cd MeasTrigEff

# Step 1: Measure trigger efficiencies
bash scripts/measTrigEff.sh 2018

# Step 2: Closure test
bash scripts/closTrigEff.sh 2018

# Step 3: Plot ID and trigger efficiencies
bash scripts/plotIDnTrigEff.sh 2018

# Step 4: Export results to SKNanoAnalyzer
bash scripts/parseResultsToSKNano.sh
```

### Individual Python Scripts
```bash
# EMu leg efficiency (Mu8El23, Mu23El12)
python python/measEMuLegEff.py --era 2018 --hltpath Mu8El23 --leg muon
python python/measEMuLegEff.py --era 2018 --hltpath Mu23El12 --leg electron

# Pairwise filter efficiency (double muon)
python python/measPairwiseEff.py --era 2018 --filter EMuDZ
python python/measPairwiseEff.py --era 2016postVFP --filter DblMuDZ

# Closure test
python python/closTrigEff.py --era 2018 --channel RunEMu

# Plotting
python python/plotMuIDEff.py --era 2018
python python/plotEleIDEff.py --era 2018
python python/plotDblMuLegEff.py --era 2018
python python/plotEMuLegEff.py --era 2018 --hltpath Mu8El23
```

## Directory Structure

```
MeasTrigEff/
├── scripts/
│   ├── measTrigEff.sh           # Measures EMu leg + pairwise filter efficiencies
│   ├── plotIDnTrigEff.sh        # Plots ID and trigger efficiencies
│   ├── closTrigEff.sh           # Closure test validation
│   └── parseResultsToSKNano.sh  # Export results
├── python/
│   ├── measPairwiseEff.py       # Double muon pairwise filter efficiency
│   ├── measEMuLegEff.py         # EMu leg efficiency (Mu8El23, Mu23El12)
│   ├── closTrigEff.py           # Closure test for di-lepton triggers
│   ├── plotMuIDEff.py           # Muon ID efficiency plots (2D eta × pT)
│   ├── plotEleIDEff.py          # Electron ID efficiency plots
│   ├── plotDblMuLegEff.py       # Double muon leg efficiency plots
│   └── plotEMuLegEff.py         # EMu leg efficiency plots
├── results/{era}/
│   ├── json/                    # Pairwise filter efficiencies, leg efficiencies (JSON)
│   └── ROOT/                    # Efficiency histograms per HLT path (ROOT)
└── plots/                       # Output efficiency plots
```

## Measurement Methods

### EMu Leg Efficiency (measEMuLegEff.py)
Measures trigger leg efficiency for EMu paths using tag-and-probe:
- **HLT paths:** Mu8El23, Mu23El12
- **Legs:** muon or electron
- **Input histograms:** `TrigEff_{hltpath}_{leg}_DENOM/NUMER/fEta_Pt`
- **Variations:** Data_Central, Data_AltTag, MC_Central, MC_AltMC, MC_AltTag
- **Output:** ROOT files with 2D efficiency histograms (η × pT)

### Pairwise Filter Efficiency (measPairwiseEff.py)
Measures pairwise filter efficiencies for double muon triggers:
- **Filters:** EMuDZ (all eras), DblMuDZM/DblMuM (2016postVFP+), DblMuDZ (2016postVFP only)
- **Input:** `MeasDblMuPairwise` and `MeasEMuPairwise` flags in SKNanoOutput
- **Output:** `results/{era}/json/{filter}.json` with efficiency ± error and MC/AltMC variations

### Closure Test (closTrigEff.py)
Validates trigger efficiency corrections for di-lepton channels:
- **Channels:** RunEMu, RunDiMu, Run1E2Mu (Run2 only), Run3Mu (Run3 only)
- **Processes:** DYJets, TTLL_powheg, WZTo3LNu, TTZ
- **Input:** `ClosDiLepTrigs/{channel}/{era}/{process}.root`
- **Output:** `results/{era}/json/ClosDiLepTrigs/{channel}/{process}.json`

## Input Data Paths

```
$WORKDIR/SKNanoOutput/MeasTrigEff/{flag}/{era}/{Sample}.root
```

Trigger efficiency histograms: `TrigEff_{hltpath}_{leg}_DENOM/NUMER/fEta_Pt`

## Output Paths

```
MeasTrigEff/results/{era}/json/           # JSON efficiency results
MeasTrigEff/results/{era}/ROOT/           # ROOT efficiency histograms
MeasTrigEff/plots/                        # Efficiency comparison plots
```

## Era-Specific Notes

**2016postVFP and later:** DblMuDZM and DblMuM pairwise filters available (in addition to EMuDZ).

**2016postVFP only:** DblMuDZ filter.

**Run3 eras:** Closure test channels Run1E2Mu/Run3Mu not available (only RunEMu and RunDiMu).

**L1Prefire:** Not applicable for Run3 (2022+); only Run2 includes L1Prefire systematic.

## Common Issues

**Filter not available for era:** Check era-specific filter availability. `measTrigEff.sh` has era-conditional logic for which filters to measure.

**Missing closure test process:** Closure processes vary by era (Run2 uses different samples than Run3). Check `closTrigEff.sh` for era-specific process lists.

**2D histogram dimensions:** Efficiency plots use η × pT 2D format; ensure histogram axis binning matches between DENOM and NUMER histograms.
