# Systematic Uncertainties in SignalRegionStudy

This document describes how systematic uncertainties are handled for each sample category in the preprocessing step (`python/preprocess.py`).

## Overview

Systematic uncertainties are categorized into three types:
1. **Prompt systematics**: Experimental uncertainties (reconstruction, identification, energy scale/resolution)
2. **Theory systematics**: PDF, scale, parton shower variations (not yet implemented)
3. **Normalization systematics**: Flat rate uncertainties applied in final analysis

## Sample Categories and Their Systematic Treatment

### 1. Signal Samples
**Samples**: `TTToHcToWAToMuMu-{masspoint}`
**Input**: `PromptTreeProducer/{channel}_RunSyst/{era}/`
**Systematics Applied**:
- All prompt systematics (shape variations)
- Each systematic creates a separate TTree in output file

**Output Trees**:
- Central
- L1Prefire_Up, L1Prefire_Down (Run2 only)
- PileupReweight_Up, PileupReweight_Down
- MuonIDSF_Up, MuonIDSF_Down
- ElectronIDSF_Up, ElectronIDSF_Down (SR1E2Mu only)
- EMuTrigSF_Up, EMuTrigSF_Down (SR1E2Mu) / DblMuTrigSF_Up, DblMuTrigSF_Down (SR3Mu)
- ElectronRes_Up, ElectronRes_Down (SR1E2Mu only)
- ElectronEn_Up, ElectronEn_Down (SR1E2Mu only)
- JetRes_Up, JetRes_Down
- JetEn_Up, JetEn_Down
- MuonEn_Up, MuonEn_Down
- UnclusteredEn_Up, UnclusteredEn_Down
- BtagSF_HFcorr_Up, BtagSF_HFcorr_Down
- BtagSF_HFuncorr_Up, BtagSF_HFuncorr_Down
- BtagSF_LFcorr_Up, BtagSF_LFcorr_Down
- BtagSF_LFuncorr_Up, BtagSF_LFuncorr_Down

**Special Processing**:
- Signal cross-section scaled by 1/3 (to 5 fb)

---

### 2. Nonprompt Samples
**Samples**: Data-driven fake estimate from MatrixTreeProducer
**Input**: `MatrixTreeProducer/{channel}/{era}/Skim_TriLep_{datastream}.root`
**Systematics Applied**:
- Weight-based shape systematic
- **Run2**: ±25% uncertainty
- **Run3**: ±15% uncertainty

**Systematic Treatment**:
- Preprocessed with weight variations (Central, Nonprompt_Up, Nonprompt_Down)
- Shape templates created in `makeBinnedTemplates.py` by reading variation trees
- Applied as uncorrelated shape systematic in datacard with era suffix

**Output Trees**:
- Central (weight × 1.0)
- Nonprompt_Up (weight × 1.25 for Run2, 1.15 for Run3)
- Nonprompt_Down (weight × 0.75 for Run2, 0.85 for Run3)

**Special Processing**:
- Uses MatrixTreeProducer output (only Central tree available)
- Weight scaling applied via `weightScale` parameter in preprocessing
- Individual data period files are hadded together

---

### 3. Conversion Samples
**Samples**: DYJets, DYJets10to50, TTG, WWG
**Input**: `PromptTreeProducer/{channel}_RunSyst/{era}/`
**Systematics Applied**:
- All prompt experimental systematics (shape variations)
- ConvSF applied as lnN normalization uncertainty in datacard

**Systematic Treatment**:
- Prompt systematics: Preprocessed variations used for shape templates
- ConvSF uncertainty: Applied as uncorrelated lnN (±20%) in datacard generation
- No ConvSF shape variations created

**Output Trees**:
- Same systematic trees as signal samples (see Section 1)

**Special Processing**:
- ConvSF loaded from TriLepton ZG control region measurement
- ConvSF central value (~0.8-0.9) applied to all systematic variations
- ConvSF uncertainty envelope from ZG measurement used for normalization systematic
- Individual samples hadded into single `conversion.root`

---

### 4. Diboson Samples
**Samples**: WZTo3LNu (amcatnlo/powheg), ZZTo4L_powheg
**Input**: `PromptTreeProducer/{channel}_RunSyst/{era}/`
**Systematics Applied**:
- All prompt systematics (same as signal)

**Output Trees**:
- Same systematic trees as signal samples (see above)

**Special Processing**:
- Samples renamed to short aliases (WZTo3LNu → WZ, ZZTo4L → ZZ)
- Individual samples hadded into single `diboson.root`

---

### 5. ttX Samples
**Samples**: TTWToLNu, TTZToLLNuNu, TTHToNonbb, tZq
**Input**: `PromptTreeProducer/{channel}_RunSyst/{era}/`
**Systematics Applied**:
- All prompt systematics (same as signal)

**Output Trees**:
- Same systematic trees as signal samples (see above)

**Special Processing**:
- Samples renamed to short aliases (TTWToLNu → ttW, TTZToLLNuNu → ttZ, TTHToNonbb → ttH)
- Individual samples hadded into single `ttX.root`

---

### 6. Others (Rare) Samples
**Samples**: WWW, WWZ, WZZ, ZZZ, ggHToZZTo4L, VBFHToZZTo4L, tHq, TTTT
**Input**: `PromptTreeProducer/{channel}_RunSyst/{era}/`
**Systematics Applied**:
- All prompt systematics (same as signal)

**Output Trees**:
- Same systematic trees as signal samples (see above)

**Special Processing**:
- All samples hadded into single `others.root`
- Each tree name is "others" (not individual sample names)

---

### 7. Data Samples
**Samples**: Data periods (MuonEG, DoubleMuon, Muon depending on era/channel)
**Input**: `PromptTreeProducer/{channel}/{era}/Skim_TriLep_{datastream}.root`
**Systematics Applied**:
- None (data has no systematic variations)

**Output Trees**:
- Central only

**Special Processing**:
- Individual data period files are hadded together
- Used for future unblinding
- Currently processed but not used in blinded analysis

---

## Systematic Uncertainty Definitions

### Prompt Systematics (Experimental)

| Systematic | Description | Channels |
|------------|-------------|----------|
| **L1Prefire** | L1 trigger prefiring correction | Run2 only |
| **PileupReweight** | Pileup reweighting | All |
| **MuonIDSF** | Muon identification scale factor | All |
| **ElectronIDSF** | Electron identification scale factor | SR1E2Mu |
| **EMuTrigSF** | e-μ trigger scale factor | SR1E2Mu |
| **DblMuTrigSF** | Double muon trigger scale factor | SR3Mu |
| **ElectronRes** | Electron energy resolution | All |
| **ElectronEn** | Electron energy scale | All |
| **JetRes** | Jet energy resolution | All |
| **JetEn** | Jet energy scale | All |
| **MuonEn** | Muon energy scale | All |
| **UnclusteredEn** | Unclustered energy scale | All |
| **BtagSF_HFcorr** | b-tagging SF (heavy flavor correlated) | All |
| **BtagSF_HFuncorr** | b-tagging SF (heavy flavor uncorrelated) | All |
| **BtagSF_LFcorr** | b-tagging SF (light flavor correlated) | All |
| **BtagSF_LFuncorr** | b-tagging SF (light flavor uncorrelated) | All |

### Theory Systematics (Not Yet Implemented)
- AlpS (αS variation)
- AlpSfact (factorization αS variation)
- PDFReweight (PDF uncertainties, 100 variations)
- ScaleVar (scale variations, indices 0,1,2,3,4,6,8)
- PSVar (parton shower variations, 4 variations)

### Normalization Systematics (Applied in Final Analysis)

| Category | Uncertainty | Notes |
|----------|-------------|-------|
| **Nonprompt** | ±25% (Run2), ±15% (Run3) | Applied as tree variations (Nonprompt_Up/Down) |
| **Conversion (ConvSF)** | Envelope from ZG measurement | Applied as flat rate uncertainty in final fit |
| **Diboson (WZSF)** | ±20% | Applied as flat rate uncertainty in final fit |

---

## File Organization

**Input Files**:
```
$WORKDIR/SKNanoOutput/
├── PromptTreeProducer/
│   ├── {channel}_RunSyst/{era}/          # Signal, prompt MC with systematics
│   └── {channel}/{era}/                  # Data (no systematics)
└── MatrixTreeProducer/
    └── {channel}/{era}/                  # Nonprompt (fake estimate)
```

**Output Files**:
```
$WORKDIR/SignalRegionStudy/samples/{era}/{channel}/{masspoint}/{method}/
├── {masspoint}.root                      # Signal
├── nonprompt.root                        # Nonprompt (hadded)
├── conversion.root                       # Conversion (hadded)
├── diboson.root                          # Diboson (hadded)
├── ttX.root                              # ttX samples (hadded)
├── others.root                           # Rare backgrounds (hadded)
└── data.root                             # Data (hadded, for future unblinding)
```

Each ROOT file contains multiple TTrees (one per systematic variation).

---

## Implementation Notes

1. **Memory Management**: Intermediate files (individual data periods, individual rare samples) are removed after hadding to save disk space

2. **ConvSF Loading**:
   - Loaded from `$WORKDIR/TriLepton/results/ZG{channel}/{era}/ConvSF.json`
   - Must exist (no fallback values)
   - Envelope uncertainty computed from all systematic variations in ZG measurement

3. **Systematic Tree Naming**:
   - Systematics use exact names from input trees (e.g., "L1Prefire_Up")
   - Exception: Nonprompt uses custom names ("Nonprompt_Up", "Nonprompt_Down")

4. **Training Sample Flag**:
   - Signal samples with 80 < mA < 100 GeV marked as `isTrainedSample=True`
   - Used for future MVA training (score branches commented out in current implementation)

5. **Error Handling**:
   - Missing ConvSF file → RuntimeError
   - Missing input files → Warning logged, continue processing
   - Missing systematic trees → Error from C++ Preprocessor

---

## Configuration File

All systematic definitions are centrally managed in **`configs/systematics.json`**:

**Structure**:
- `source`: "preprocessed" (read from files), "manual" (created by scaling), "value" (normalization only)
- `type`: "shape" (distribution variations) or "lnN" (flat rate)
- `correlation`: "correlated" (across eras) or "uncorrelated" (era-specific)
- `applies_to`: List of processes affected by this systematic

**Single Source of Truth**:
- Preprocessing (`preprocess.py`): Uses "preprocessed" systematics
- Template creation (`makeBinnedTemplates.py`): Creates "manual" systematics
- Datacard generation (`printDatacard.py`): Applies all systematic types

See `configs/systematics.json` for complete current systematic list.

## References

- Systematic configurations: `configs/systematics.json`
- Sample configurations: `configs/samplegroups.json`
- C++ implementation: `src/Preprocessor.cc`, `include/Preprocessor.h`
- Main preprocessing script: `python/preprocess.py`
