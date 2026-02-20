# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Module Overview

MeasJetTagEff measures b-tagging (DeepJet) efficiency for different jet flavors (light=0, c=4, b=5) as a function of η and pT. Produces projection plots showing efficiency dependence on kinematic variables for all supported eras.

**Eras:** 2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix

## Environment Setup

```bash
source setup.sh  # Run from ChargedHiggsAnalysisV3 root
```

## Key Commands

```bash
cd MeasJetTagEff

# Plot b-tagging efficiency for a single era
python python/plot.py --era 2018

# Run for all eras
for era in 2016preVFP 2016postVFP 2017 2018 2022 2022EE 2023 2023BPix; do
    python python/plot.py --era $era
done
```

Note: There is no `doThis.sh` in this module. Run `plot.py` directly for each era.

## Directory Structure

```
MeasJetTagEff/
├── python/
│   └── plot.py          # Main (and only) plotting script
└── plots/
    └── {era}/           # Output PNG plots per era
```

## python/plot.py

Generates efficiency vs η and pT projection plots:

- **Input:** `$WORKDIR/SKNanoOutput/MeasJetTagEff/{era}/TTLL_powheg.root`
- **Histogram path:** `tagging#b##era#{era}##flavor#{flavor}##systematic#central#{type}`
  - `flavor`: 0 (light), 4 (c), 5 (b)
  - `type`: `den` (denominator) or `num` (numerator, DeepJet M working point)
- **Functions:**
  - `create_eta_projections()`: 1D efficiency vs η for each pT bin
  - `create_pt_projections()`: 1D efficiency vs pT for each η bin
  - `create_eta_summary()`: pT-averaged efficiency vs η
  - `create_pt_summary()`: η-averaged efficiency vs pT
- **Output:** `plots/{era}/`
  - `eta_projection_pt_{bin}.png`
  - `pt_projection_eta_{bin}.png`
  - `eff_vs_eta_summary.png`
  - `eff_vs_pt_summary.png`

## Input Data Path

```
$WORKDIR/SKNanoOutput/MeasJetTagEff/{era}/TTLL_powheg.root
```

## Common Issues

**Missing input file:** Only `TTLL_powheg.root` is used as input. Ensure it exists for the requested era.

**Histogram naming:** The histogram path uses `#` as delimiter — this is the convention from SKNanoAnalyzer output. Do not confuse with standard ROOT directory separators.
