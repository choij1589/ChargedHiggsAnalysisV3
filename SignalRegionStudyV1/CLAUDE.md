# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Module Overview

SignalRegionStudyV1 is the final limit extraction module for charged Higgs searches. It processes templates from upstream modules (DiLepton, MeasFakeRate) and performs statistical inference using HiggsCombine.

**Target physics:** Charged Higgs (MHc 70-160 GeV, MA 15-155 GeV)
**Channels:** SR1E2Mu (e+2mu), SR3Mu (3mu), Combined
**Methods:** Baseline (simple selection), ParticleNet (ML classifier)

## Environment Setup

```bash
source setup.sh   # CRITICAL: already done before launching claude
```

## Key Commands

### Full Pipeline
```bash
./doThis.sh       # Runs entire analysis pipeline
```

### Individual Steps

**Preprocessing (raw data with systematics):**
```bash
preprocess.py --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90
```

**Template generation:**
```bash
makeBinnedTemplates.py --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --method Baseline --binning uniform
```

**Template validation:**
```bash
checkTemplates.py --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --method Baseline --binning uniform
```

**Datacard generation:**
```bash
printDatacard.py --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --method Baseline --binning uniform
```

**Combine datacards:**
```bash
combineDatacards.py --masspoint MHc130_MA90 --method Baseline --binning extended --mode channel  # SR1E2Mu + SR3Mu -> Combined
combineDatacards.py --masspoint MHc130_MA90 --method Baseline --binning extended --mode era      # 4 eras -> Run2
```

**Run limits:**
```bash
./scripts/runAsymptotic.sh --era Run2 --channel Combined --masspoint MHc130_MA90 --method Baseline --binning extended
./scripts/runHybridNew.sh --era Run2 --channel Combined --masspoint MHc130_MA90 --method Baseline --binning extended --condor --auto-grid
```

**Combined limits (channels + eras):**
```bash
./scripts/runCombinedAsymptotic.sh --masspoint MHc130_MA90 --method Baseline --binning extended
./scripts/runCombinedAsymptotic.sh --masspoint MHc130_MA90 --method ParticleNet --binning extended --partial-unblind
```

**Systematic impacts:**
```bash
./Impact.sh
./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint MHc130_MA90 --method Baseline --binning extended
./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint MHc130_MA90 --method ParticleNet --binning extended --partial-unblind
```

**Signal injection:**
```bash
./signalInj.sh
./scripts/runSignalInjection.sh --era Run2 --channel Combined --masspoint MHc130_MA90 --method Baseline --binning extended
```

**Collect and plot limits:**
```bash
collectLimits.py --era Run2 --channel Combined --masspoint MHc130_MA90 --method Baseline --binning extended
plotLimits.py --era Run2 --channel Combined --masspoint MHc130_MA90 --method Baseline --binning extended
```

## Workflow Pipeline

```
SKNanoOutput (raw ROOT files)
    ↓
preprocess.py           →  preprocessed trees with systematics (samples/)
    ↓
makeBinnedTemplates.py  →  shapes.root (histograms + systematics)
    ↓
checkTemplates.py       →  validation plots
    ↓
printDatacard.py        →  datacard.txt
    ↓
combineDatacards.py     →  combined datacard (channels/eras merged)
    ↓
runAsymptotic.sh / runHybridNew.sh  →  limit trees
    ↓
collectLimits.py        →  limits JSON
    ↓
plotLimits.py           →  Brazilian band plots
```

## Output Structure

```
samples/{era}/{channel}/{masspoint}/
└── preprocessed.root    # Trees with systematic variations

templates/{era}/{channel}/{masspoint}/{method}/{binning}/
├── shapes.root      # Histograms for Combine
├── datacard.txt     # HiggsCombine datacard
└── validation/      # Diagnostic plots
```

## Configuration Files

### configs/samplegroups.json
Maps data/MC samples per era and channel:
```json
{
  "2018": {
    "SR1E2Mu": {
      "data": ["MuonEG_B", "MuonEG_C", ...],
      "WZ": ["WZTo3LNu_amcatnlo"],
      "nonprompt": ["DY", "TTToFullLep", ...]
    }
  }
}
```

### configs/systematics.{era}.json
Defines systematic uncertainties per channel:
- **Preprocessed shape:** Variations stored in input file (e.g., `PileupReweight_Up/Down`)
- **Valued shape:** Percentage modifier (e.g., `1.02` = ±2%)
- **Valued lnN:** Log-normal normalization (e.g., `lumi_2018: 1.0084`)

## Key Parameters

**Mass points (Baseline):** MHc70_MA15, MHc100_MA60, MHc130_MA90, MHc160_MA155
**Mass points (ParticleNet):** MHc130_MA90
**Binning:** uniform (15 bins), extended (15 + tail bins)
**Eras:** 2016preVFP, 2016postVFP, 2017, 2018, Run2 (combined)

## Architecture Notes

### ParticleNet Processing Order
SR3Mu ParticleNet depends on SR1E2Mu fit results (Amass peak parameters). Always process SR1E2Mu first:
```bash
parallel -j 4 preprocess_particleNet SR1E2Mu {} {} ::: "${ERAs[@]}" ::: "${MASSPOINTs[@]}"
parallel -j 4 preprocess_particleNet SR3Mu {} {} ::: "${ERAs[@]}" ::: "${MASSPOINTs[@]}"
```

### Systematic Handling
Three approaches in `makeBinnedTemplates.py`:
1. **Preprocessed:** Extract Up/Down variations from input ROOT file branches
2. **Valued shape:** Apply symmetric percentage to all bins on-the-fly
3. **Valued lnN:** Normalization-only uncertainty (text in datacard)

### Unblinding
- Default: MC sum used as `data_obs` (blinded)
- `--unblind`: Use real data
- `--partial-unblind`: Data only in low ParticleNet score region (score < 0.3)

Scripts supporting `--partial-unblind`: `makeBinnedTemplates.py`, `checkTemplates.py`, `plotParticleNetScore.py`, `printDatacard.py`, `combineDatacards.py`, `runAsymptotic.sh`, `runImpacts.sh`, `runCombinedAsymptotic.sh`

Output directory: `templates/{era}/{channel}/{masspoint}/{method}/{binning}_partial_unblind/`

### HiggsCombine Methods
- **AsymptoticLimits:** Fast CLs+b approximation
- **HybridNew:** Toy-based CLs with HTCondor parallelization

## Common Issues

**Missing fit results for SR3Mu ParticleNet:** Run SR1E2Mu first to generate Amass fit parameters.

**HTCondor jobs stuck:** Check `./scripts/condor/` for submission status; use `--condor` flag for toy generation.

**Non-positive histograms:** `makeBinnedTemplates.py` applies `ensure_positive_integral()` automatically.
