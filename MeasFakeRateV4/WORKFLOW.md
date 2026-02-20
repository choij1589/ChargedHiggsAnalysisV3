# MeasFakeRateV4 Workflow

## Directory Structure

```
MeasFakeRateV4/
├── doThis.sh                 # Entry point
├── scripts/
│   ├── measFakeRate.sh       # Main measurement pipeline
│   ├── plotClosure.sh        # Closure test validation
│   └── parseFakeRateToSKNano.sh  # Export results to SKNanoAnalyzer
├── python/
│   ├── common.py             # Utility functions (findbin)
│   ├── parseIntegral.py      # Extract histogram integrals to JSON
│   ├── measFakeRate.py       # Calculate fake rates (data & MC)
│   ├── plotFakeRate.py       # Plot fake rate histograms
│   ├── plotNormalization.py  # Z-peak normalization validation
│   ├── plotValidation.py     # Data vs MC validation plots
│   ├── plotClosure.py        # Closure test visualization
│   └── plotSystematics.py    # Systematic uncertainty plots
├── configs/
│   ├── samplegroup.json      # Sample groupings by era
│   ├── systematics.json      # Systematic variation definitions
│   └── histkeys.json         # Histogram axis labels and rebinning
├── results/{ERA}/
│   ├── JSON/{measure}/       # Integrated histogram values
│   └── ROOT/{measure}/       # Fake rate ROOT histograms
└── plots/{ERA}/{measure}/    # Output plots
```

---

## Implementation Checklist

### Phase 1: Setup

- [ ] Create directory structure (configs/, python/, scripts/, results/, plots/)
- [ ] Copy and adapt `configs/samplegroup.json` from MeasFakeRateV3
- [ ] Copy and adapt `configs/systematics.json` from MeasFakeRateV3
- [ ] Copy and adapt `configs/histkeys.json` from MeasFakeRateV3

### Phase 2: Core Python Scripts

- [ ] `python/common.py` - Utility functions
  - [ ] `findbin()` function for bin name mapping
  - [ ] Bin definitions for electrons and muons

- [ ] `python/parseIntegral.py` - Extract histogram integrals
  - [ ] Parse QCDEnriched/MT histograms from ROOT files
  - [ ] Handle loose and tight working points
  - [ ] Extract statistical errors via IntegralAndError
  - [ ] Process systematic variations
  - [ ] Output JSON files per sample/WP

- [ ] `python/measFakeRate.py` - Calculate fake rates
  - [ ] Data-driven measurement: `fake = data - prompt_MC × scale`
  - [ ] Prompt normalization from Z-enriched region
  - [ ] Error propagation (statistical)
  - [ ] Systematic variations (PromptNorm, MotherJetPt, RequireHeavyTag)
  - [ ] MC-based measurement (QCD, TTJJ) with flavor breakdown
  - [ ] Output 2D histograms (eta × pT) to ROOT files

- [ ] `python/plotFakeRate.py` - Plot fake rate results
  - [ ] Project 2D histograms to 1D pT per eta region
  - [ ] CMS style formatting
  - [ ] Support both data and MC fake rates

- [ ] `python/plotNormalization.py` - Z-peak normalization plots
  - [ ] Data vs MC comparison in Z-enriched region
  - [ ] Per HLT path, working point, and selection

- [ ] `python/plotValidation.py` - Validation plots
  - [ ] MT, pT, eta distributions (data vs MC)
  - [ ] Inclusive and Z-enriched regions

- [ ] `python/plotClosure.py` - Closure test visualization
  - [ ] Adaptive rebinning for statistical validity
  - [ ] Compare observed vs expected
  - [ ] Multiple kinematic distributions

- [ ] `python/plotSystematics.py` - Systematic uncertainty plots
  - [ ] Fractional systematic variations per eta bin
  - [ ] PromptNorm, MotherJetPt, RequireHeavyTag variations

### Phase 3: Shell Scripts

- [ ] `scripts/measFakeRate.sh` - Main pipeline orchestration
  - [ ] Step 1: parseIntegral.py
  - [ ] Step 2: measFakeRate.py (data)
  - [ ] Step 3: measFakeRate.py --isMC
  - [ ] Step 4-5: plotFakeRate.py (data & MC)
  - [ ] Step 6-8: plotNormalization.py (parallel)
  - [ ] Step 9-10: plotValidation.py (parallel)
  - [ ] Step 11: plotSystematics.py

- [ ] `scripts/plotClosure.sh` - Closure test orchestration
  - [ ] Loop over channels (Run1E2Mu, Run3Mu)
  - [ ] Loop over histogram keys
  - [ ] Loop over systematic variations

- [ ] `scripts/parseFakeRateToSKNano.sh` - Export results
  - [ ] Copy electron fake rate to SKNanoAnalyzer
  - [ ] Copy muon fake rate to SKNanoAnalyzer

- [ ] `doThis.sh` - Entry point
  - [ ] Loop over eras (Run2 + Run3)
  - [ ] Call measFakeRate.sh for electron and muon
  - [ ] Call plotClosure.sh

### Phase 4: Validation

- [ ] Run full pipeline on one era (e.g., 2018)
- [ ] Verify normalization plots (Z-peak agreement)
- [ ] Verify validation plots (data vs MC)
- [ ] Verify systematic uncertainty plots
- [ ] Run closure tests
- [ ] Compare results with MeasFakeRateV3

---

## Bin Definitions

### Electrons
- **pT bins**: [15, 17, 20, 25, 35, 50, 100, 200] GeV
- **eta bins**: [0, 0.8, 1.479, 2.5] (EB1, EB2, EE)

### Muons
- **pT bins**: [10, 12, 14, 17, 20, 30, 50, 100, 200] GeV
- **eta bins**: [0, 0.9, 1.6, 2.4] (EB1, EB2, EE)

---

## Systematic Variations

| Systematic | Description |
|------------|-------------|
| Central | Nominal measurement |
| PromptNorm_Up/Down | ±15% variation on prompt scale |
| MotherJetPt_Up/Down | Mother jet pT selection variation |
| RequireHeavyTag | b-tagging requirement variation |

---

## Era Support

- **Run2**: 2016preVFP, 2016postVFP, 2017, 2018
- **Run3**: 2022, 2022EE, 2023, 2023BPix
