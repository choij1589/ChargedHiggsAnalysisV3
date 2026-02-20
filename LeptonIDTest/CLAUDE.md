# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Module Overview

LeptonIDTest is a comprehensive lepton (electron and muon) identification efficiency, fake rate, and ID optimization study module. It:
- Fills ID variable histograms from simulation
- Plots efficiency curves as a function of ID cut values
- Measures fake rates for different jet flavors
- Optimizes tight/loose ID working points (especially for Run3)

**Focus:** Run3 eras for ID optimization; Run2+Run3 for efficiency/fake rate studies.

## Environment Setup

```bash
source setup.sh  # Run from ChargedHiggsAnalysisV3 root
```

## Key Commands

### Run Full Module
```bash
cd LeptonIDTest && bash doThis.sh
# Runs: fillHists.sh → optimize.sh → plot.sh (for each Run3 era)
```

### Individual Steps
```bash
# Step 1: Fill ID histograms (parallel, 4 cores)
bash scripts/fillHists.sh

# Step 2: Optimize tight ID (parallel, 8 cores)
bash scripts/optimize.sh

# Step 3: Generate all plots for one era
bash scripts/plot.sh 2022EE

# Single script calls
python python/fill_electrons.py --era 2022
python python/fill_muons.py --era 2022
python python/plot_efficiency.py --era 2022 --object electron --region InnerBarrel --plotvar sip3d
python python/plot_fakerate.py --era 2022 --object electron
python python/plot_idvar.py --era 2022 --object electron --region InnerBarrel --histkey sip3d
python python/plot_trigvar.py --era 2022 --object electron --histkey sieie
python python/optimize_tightID.py --era 2022 --object muon
python python/optimize_looseID.py --era 2022 --object electron
python python/filter_optimization.py --fix miniiso --value 0.4 --object muon --era 2022
```

## Directory Structure

```
LeptonIDTest/
├── doThis.sh                    # Main: fillHists + optimize + plot for Run3 eras
├── README.md                    # Optimization results summary
├── configs/
│   ├── histkeys.json            # 15 electron ID/trigger variable histogram definitions
│   └── efficiency.json          # 3 variables for efficiency plots (sip3d, mvaNoIso, miniIso)
├── python/
│   ├── fill_electrons.py        # Fill electron ID histograms from simulation
│   ├── fill_muons.py            # Fill muon ID histograms from simulation
│   ├── plot_efficiency.py       # Efficiency vs cut value (CDF method)
│   ├── plot_fakerate.py         # Fake rate = tight/loose ratio by jet flavor
│   ├── plot_idvar.py            # ID variable distributions by lepton type
│   ├── plot_trigvar.py          # Trigger variable distributions by detector region
│   ├── optimize_tightID.py      # Brute-force miniIso × sip3d scan
│   ├── optimize_looseID.py      # Loose ID optimization (minimize flavor FR differences)
│   └── filter_optimization.py  # Display optimization results in tabular format
├── scripts/
│   ├── fillHists.sh             # Parallel histogram filling (4 cores)
│   ├── optimize.sh              # Parallel optimization (8 cores)
│   └── plot.sh                  # Generate all plots for a given era (50+ parallel jobs)
├── histograms/                  # Output ROOT files per era
│   └── {electron|muon}.{era}.root
├── optimization/                # Optimization results
│   └── tightID_optimization_{object}_{era}.csv
└── plots/                       # Output plots
    └── {era}/{object}/{type}/{region}/{...}
```

## Lepton Classification

Leptons are classified by `lepType` and `nearestJetFlavour`:
- **prompt:** lepType ∈ {1, 2, 6}
- **fromTau:** lepType == 3
- **conv:** lepType ∈ {4, 5, -5, -6} (electron conversion)
- **fromL/fromC/fromB:** lepType < 0, distinguished by jet flavor (1/4/5)
- **fromPU:** lepType < 0, jetFlavour == -1

## Detector Regions

**Electrons** (by supercluster η):
- InnerBarrel: |η_SC| < 0.8
- OuterBarrel: 0.8 ≤ |η_SC| < 1.479
- Endcap: |η_SC| ≥ 1.479

**Muons** (by η):
- InnerBarrel: |η| < 0.9
- OuterBarrel: 0.9 ≤ |η| < 1.6
- Endcap: |η| ≥ 1.6

## ID Working Points

### Electrons
| ID Tier | Cuts |
|---------|------|
| Trigger | Baseline η-dependent cuts (sieie, dEta, dPhi, HoverE, isolation) |
| Loose ID | Above + mvaNoIso/WP90 + sip3d + miniIso cuts |
| Tight ID | isMVANoIsoWP90 + sip3d < 4 + miniIso < 0.1 |

**Note:** mvaNoIso thresholds differ between Run2 (0.985/0.96/0.85 per region) and Run3 (0.8/0.5/-0.8).

### Muons
| ID Tier | Cuts |
|---------|------|
| Baseline | isPOGMedium + trackIso < 0.4 |
| Loose ID | sip3d < 6 (Run2) or 8 (Run3) + miniIso < 0.6 (Run2) or 0.4 (Run3) |
| Tight ID | sip3d < 3 + miniIso < 0.1 |

## pT-corrected Binning

`pt_corr = pt × (1 + max(0, miniIso - 0.1))`

**Electron pT bins:** [15, 17, 20, 25, 35, 50, 70] GeV
**Muon pT bins:** [10, 15, 20, 30, 50, 70] GeV

## Configuration Files

### configs/histkeys.json
15 electron ID/trigger variables:
- **ID:** isMVANoIsoWP90, isPOGM, sip3d, miniIso, lostHits, convVeto, dz, mvaNoIso
- **Trigger:** sieie, deltaEtaInSC, deltaPhiInSeed, hoe, ecalPFClusterIso, hcalPFClusterIso, trackIso

### configs/efficiency.json
3 variables for efficiency curves: sip3d [0,10], mvaNoIso [-1,1], miniIso [0,1]

## Optimization Details

### optimize_tightID.py
Scans miniIso × sip3d combinations:
- **miniIso scan:** [0.05, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13]
- **sip3d scan:** [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
- **Output:** CSV with efficiency ± 95% C.L. binomial errors

### optimize_looseID.py
Minimizes jet flavor fake rate differences:
- **Muons:** Minimize |FR(b) - FR(c)| across pT bins
- **Electrons:** Minimize |FR(l)-FR(b)| + |FR(l)-FR(c)| + |FR(b)-FR(c)|

## Input Data Paths

```
$WORKDIR/SKNanoOutput/ParseEleIDVariables/{era}/TTLJ_powheg.root
$WORKDIR/SKNanoOutput/ParseEleIDVariables/{era}/TTLL_powheg.root
$WORKDIR/SKNanoOutput/ParseMuIDVariables/{era}/TTLJ_powheg.root
$WORKDIR/SKNanoOutput/ParseMuIDVariables/{era}/TTLL_powheg.root
```

## Common Issues

**Run2 vs Run3 mvaNoIso thresholds differ:** The `fill_electrons.py` script applies different mvaNoIso thresholds for Run2 and Run3. If studying cross-era performance, be aware of this difference.

**Optimization results interpretation:** Use `filter_optimization.py` to display CSV results in readable form. The optimization metric is the sum of flavor fake rate differences; lower is better.

**doThis.sh targets Run3 only:** By default `doThis.sh` processes 2022, 2022EE, 2023, 2023BPix eras. For Run2 histograms, run `fill_electrons.py`/`fill_muons.py` directly.
