# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Module Overview

TriggerStrategy evaluates the impact of different trigger strategies on event acceptance in the SR1E2Mu (1 electron + 2 muon) signal region. It compares the baseline EMu trigger strategy against adding single-lepton or double-muon triggers, quantifying acceptance gains for both SM backgrounds and charged Higgs signal.

**Era:** 2018 (fixed)
**Region:** SR1E2Mu

## Environment Setup

```bash
source setup.sh  # Run from ChargedHiggsAnalysisV3 root
```

## Key Commands

```bash
cd TriggerStrategy && bash doThis.sh  # Run all samples
# Or run individually:
python python/plot.py --sample TTLL_powheg
python python/plot.py --sample TTToHcToWAToMuMu-MHc130_MA90
```

## Directory Structure

```
TriggerStrategy/
├── doThis.sh                    # Run plot.py for 5 SM + 4 signal samples
├── python/
│   └── plot.py                  # Compare trigger strategies, output plots + JSON
├── integrals/                   # JSON event yield statistics per sample
│   ├── TTLL_powheg.json
│   ├── DYJets.json
│   └── TTToHcToWAToMuMu-MHc*.json
└── plots/                       # PNG comparison plots per sample
```

## Trigger Strategies Compared

| Strategy | Flag Directory | Description |
|---|---|---|
| Baseline | `RunEMuTrigs` | Standard electron-muon triggers only |
| +SglEl | `RunEMuTrigsWithSglElTrigs` | + single-electron triggers |
| +SglMu | `RunEMuTrigsWithSglMuTrigs` | + single-muon triggers |
| +DblMu | `RunEMuTrigsWithDblMuTrigs` | + double-muon triggers |

## python/plot.py

- **Input:** `$WORKDIR/SKNanoOutput/TriggerStudy/Run{flag}/{era}/{sample}.root`
  - Histogram: `SR1E2Mu/Central/electrons/1/pt` (leading electron pT in SR1E2Mu)
- **Signal scale factors** (filter efficiency correction):
  - MHc70_MA15: ×0.6113, MHc100_MA60: ×0.684, MHc130_MA90: ×0.687, MHc160_MA155: ×0.5187
- **Output:**
  - `plots/{sample}.png`: Kinematic comparison with ratio panel (alternative/baseline)
  - `integrals/{sample}.json`: Event yields with uncertainties for all 4 strategies

## integrals/ JSON Format

```json
{
  "sample": "TTLL_powheg",
  "era": "2018",
  "histkey": "electrons/1/pt",
  "integrals": {
    "EMuTrigs": {"total_integral": 193.69, "total_integral_error": 2.66, ...},
    "EMuTrigsWithSglElTrigs": {...},
    "EMuTrigsWithSglMuTrigs": {...},
    "EMuTrigsWithDblMuTrigs": {...}
  }
}
```

## Samples Processed

**SM backgrounds:** TTLL_powheg, DYJets, TTZToLLNuNu, WZTo3LNu_amcatnlo, ZZTo4L_powheg

**Signals:** MHc70_MA15, MHc100_MA60, MHc130_MA90, MHc160_MA155

## Common Issues

**Fixed era (2018):** This module only studies the 2018 era. It is a diagnostic/optimization study, not part of the main analysis pipeline.

**Signal scaling:** The `plot.py` script hardcodes filter efficiency scale factors per mass point. These must be updated if signal samples change.
