# SignalRegionStudyV2 TODO

## Overview

SignalRegionStudyV2 builds on the established framework from V1, which provides a complete limit extraction pipeline for charged Higgs searches. The key new feature in V2 is the **Run2 + Run3 combined limit extraction**.

### Key Differences from V1
- **CMSSW**: Uses shared `Common/CMSSW_14_1_0_pre4/` instead of local copy
- **Run3 Signal Scaling**: Run3 signal samples don't exist yet - signal weights are scaled from 2018 based on TTbar cross section changes and luminosity
- **Dual Mode**: Support both cases - with and without Run3 signal samples

### Reference: V1 Completed Work
- Full 8-step pipeline: preprocess → templates → datacards → combine → limits
- 36 systematic uncertainties per channel
- Run2 eras: 2016preVFP, 2016postVFP, 2017, 2018
- Channels: SR1E2Mu, SR3Mu
- Methods: Baseline (4 mass points), ParticleNet (3-6 mass points)
- Limit extraction: AsymptoticLimits and HybridNew

---

## Phase 1: Directory Structure & Core Infrastructure

- [x] Create directory structure
  - [x] `python/` - Analysis scripts
  - [x] `scripts/` - Shell scripts for Combine execution
  - [x] `automize/` - Batch automation helpers
  - [x] `configs/` - JSON configuration files
  - [x] `samples/` - Preprocessed ROOT files (empty, output directory)
  - [x] `templates/` - Output histograms and datacards (created by makeBinnedTemplates.py)
  - [ ] `datacards/` - Combined datacards repository
  - [ ] `results/` - Final limit outputs (json/, plots/)
  - [ ] `docs/` - Documentation

- [ ] Verify CMSSW + HiggsCombine environment
  - [ ] Uses `Common/CMSSW_14_1_0_pre4/` (shared, not local copy)
  - [ ] Verify Combine tools work (`text2workspace.py`, `combine`)

---

## Phase 2: Configuration Files

- [x] Create `configs/samplegroups.json`
  - [x] Define data streams per era (Run2 + Run3)
  - [x] Define MC sample groups (WZ, ZZ, ttW, ttZ, ttH, tZq, conversion, nonprompt, others)
  - [x] Signal mass points configuration

- [x] Create `configs/systematics.{era}.json` for each era
  - [x] Run2: 2016preVFP, 2016postVFP, 2017, 2018
  - [x] Run3: 2022, 2022EE, 2023, 2023BPix
  - [x] ~30 systematics: lumi, pileup, lepton ID/SF, JES/JER, btag, theory, etc.
  - [x] Run3 configs: removed L1 prefiring, updated naming (13TeV → 13p6TeV)

- [x] Create `configs/scaling.json` for Run3 signal scaling
  - [x] TTbar cross section: Run2 (833 pb), Run3 (923 pb)
  - [x] Luminosity per era
  - [x] Source era for Run3 scaling: 2018

---

## Phase 3: Python Analysis Scripts

### Core Scripts
- [x] `preprocess.py` - Preprocess signal/background samples with systematics
  - [x] Run2: Standard processing from SKNanoOutput
  - [x] Run3 signal: `--scale-from-run2` flag to scale weights from 2018
  - [x] Run3 backgrounds: From SKNanoOutput (real Run3 MC)
  - [x] Signal normalization: weights divided by 3.0 (normalized to 5 fb)
  - [x] Systematic variations: preprocessed shapes, multi-variation (PDF, Scale)
- [x] `template_utils.py` - Utility functions for template generation
  - [x] `save_json()`, `parse_variations()`, `get_output_tree_name()`
  - [x] `ensure_positive_integral()`, `build_particlenet_score()`
  - [x] `create_filtered_rdf()`, `create_scaled_hist()`
- [x] `makeBinnedTemplates.py` - Generate binned histograms with A-mass fitting
  - [x] A mass fitting using AmassFitter (SR1E2Mu direct, SR3Mu loads from SR1E2Mu)
  - [x] Binning: extended (default, 19 bins ±10σ) or uniform (15 bins ±5σ)
  - [x] Background validation and merging to "others"
  - [x] Blinding: data_obs = MC sum; Unblinding: data_obs = real data
  - [x] ParticleNet threshold optimization (method=ParticleNet)
- [x] `checkTemplates.py` - Validate templates (non-zero, positive bins)
- [x] `printDatacard.py` - Generate HiggsCombine datacards
- [x] `combineDatacards.py` - Merge datacards across channels and eras
- [ ] `collectLimits.py` - Extract limit values from Combine output
- [ ] `plotLimits.py` - Create Brazilian band limit plots

### Optional/Diagnostic Scripts
- [x] `plotParticleNetScore.py` - ParticleNet score distribution plots
- [x] `plotHybridNewGrid.py` - HybridNew toy MC grid visualization (CLs vs r)
- [x] `plotTestStatDist.py` - Test statistic distributions (S+B vs B-only)
- [ ] `plotPullDist.py` - Nuisance parameter pulls
- [ ] `extractInjectionResults.py` - Signal injection test results
- [ ] `plotBiasTest.py` - Bias test analysis

---

## Phase 4: Shell Scripts for Combine

- [x] `runAsymptotic.sh` - Run AsymptoticLimits method
- [x] `runHybridNew.sh` - Run HybridNew method with HTCondor DAG workflow (Condor-only)
- [x] ~~`runCombinedAsymptotic.sh`~~ - Integrated into `automize/makeBinnedTemplates.sh`
- [ ] `runImpacts.sh` - Nuisance parameter impact calculation
- [ ] `runFitDiagnostics.sh` - Post-fit diagnostics extraction
- [ ] `runSignalInjection.sh` - Signal injection studies
- [ ] `rsync_templates.sh` - Template synchronization

---

## Phase 5: Automation Scripts

- [x] `automize/preprocess.sh` - Batch preprocessing (GNU parallel)
  - [x] `--mode run2` - Process Run2 only
  - [x] `--mode run3-scaled` - Process Run3 with signal scaled from 2018
  - [x] `--mode run3` - Process Run3 with real signal MC (when available)
  - [x] `--mode all` - Process both Run2 and Run3 (scaled)
- [x] `automize/makeBinnedTemplates.sh` - Batch template generation (GNU parallel)
  - [x] `--mode run2/run3/all` - Select era group
  - [x] `--era <era>` - Single era mode (e.g., `--era 2018`)
  - [x] `--method Baseline/ParticleNet` - Select analysis method
  - [x] `--binning extended/uniform` - Select binning strategy
  - [x] `--unblind`, `--partial-unblind` - Unblinding options
  - [x] Validation always runs (checkTemplates.py)
  - [x] `--plot-score` - Run plotParticleNetScore.py (default for ParticleNet)
  - [x] `--validationOnly` - Skip template generation, only run validation/plotting
  - [x] `--no-plot-score` - Disable ParticleNet score plotting
  - [x] `--printDatacard`, `--combineDatacards`, `--runAsymptotic` - Datacard and limit steps (default: enabled)
  - [x] Era-by-era pipeline: templates → validation → datacards → channel combine → asymptotic
  - [x] Automatic era combination (Run2, Run3, All) after individual era processing
  - [x] Parallelization: era-level (`--njobs-era`) and mass-point-level
- [x] `automize/hybridnew.sh` - Batch HybridNew execution
  - [x] `--mode all` - Process combined Run2+Run3
  - [x] `--mode run2` - Process Run2 only
  - [x] `--mode run3` - Process Run3 only
  - [x] `--era <era>` - Single era mode
  - [x] `--auto-grid` - Auto-tune r-range from Asymptotic results (default)
  - [x] `--partial-extract` - Extract limits from incomplete condor jobs
  - [x] `--plot` - Generate CLs vs r and test statistic plots
- [ ] `automize/impact.sh` - Batch impact calculation
- [ ] `automize/signalInj.sh` - Batch signal injection
- [ ] `automize/partialUnblind.sh` - Partial unblinding templates

---

## Phase 6: Run the Pipeline (Run2)

### Per Era: 2016preVFP, 2016postVFP, 2017, 2018
- [ ] Preprocess samples
- [ ] Generate templates (Baseline method)
- [ ] Generate templates (ParticleNet method)
- [ ] Validate templates
- [ ] Generate datacards
- [ ] Run asymptotic limits

### Combined Run2
- [ ] Combine datacards across channels (SR1E2Mu + SR3Mu)
- [ ] Combine datacards across eras
- [ ] Run combined asymptotic limits
- [ ] Run HybridNew for validation
- [ ] Impact studies
- [ ] Generate limit plots

---

## Phase 7: Run the Pipeline (Run3)

### Mode A: Without Run3 Signal Samples (Scaled from Run2)
- [ ] Preprocess with `--scale-from-run2`
  - [ ] Signal: weights scaled from 2018 by `(xsec_Run3/xsec_Run2) × (lumi_Run3/lumi_2018)`
  - [ ] Backgrounds: from Run3 MC (SKNanoOutput)
  - [ ] Data: from Run3 collision data
- [ ] Generate templates with scaled signal
- [ ] Generate datacards
- [ ] Run asymptotic limits

### Mode B: With Run3 Signal Samples (When Available)
- [ ] Preprocess Run3 signal samples (standard mode)
- [ ] Generate templates with real Run3 signal
- [ ] Compare with Mode A for validation

### Per Era: 2022, 2022EE, 2023, 2023BPix
- [ ] Preprocess samples
- [ ] Generate templates (with scaled or real signal)
- [ ] Validate templates
- [ ] Generate datacards
- [ ] Run asymptotic limits

### Combined Run3
- [ ] Combine datacards across channels and eras
- [ ] Run combined limits
- [ ] Impact studies

---

## Phase 8: Full Combination (Run2 + Run3)

- [ ] Combine Run2 and Run3 datacards
- [ ] Handle correlated vs uncorrelated systematics across runs
- [ ] Run full combination limits
- [ ] HybridNew validation
- [ ] Final impact studies
- [ ] Publication-ready limit plots

---

## Phase 9: Documentation

- [ ] Create `CLAUDE.md` - Module guidance for AI assistant
- [ ] Create `docs/WORKFLOW.md` - Pipeline overview
- [ ] Create `docs/systematics.md` - Systematic uncertainties documentation
- [ ] Create `docs/limits.md` - Limit results summary
- [ ] Document Run3 signal scaling methodology

---

## Technical Notes

### Mass Points
- Baseline: MHc70_MA15, MHc100_MA60, MHc130_MA90, MHc160_MA155
- ParticleNet: MHc100_MA95, MHc130_MA90, MHc160_MA85 (+ others)

### Binning Strategies
- Uniform: 15 bins in [-5sigma, +5sigma] around A mass
- Extended: 15 uniform + tail bins at [-10,-7] and [+7,+10] sigma

### Unblinding Modes
- Blinded: MC sum as data_obs
- Unblind: Real data as data_obs
- Partial unblind: Data only where ParticleNet score < 0.3

### Run3 Signal Scaling

**Implementation**: Scaling is done at preprocessing level by scaling event weights.

**Formula**:
```
weight_Run3 = weight_Run2 × (xsec_ttbar_Run3 / xsec_ttbar_Run2) × (lumi_Run3 / lumi_2018)
```

**What gets scaled**:
- Signal weights: scaled by the formula above
- Kinematic variables (mass, MT): preserved unchanged
- ParticleNet scores: preserved unchanged
- All systematic tree variations: same scale factor applied

**What comes from Run3**:
- Backgrounds (WZ, ZZ, ttW, ttZ, etc.): Real Run3 MC from SKNanoOutput
- Nonprompt: Run3 data-driven (MatrixTreeProducer)
- Data: Run3 collision data

**Cross sections**:
- `xsec_ttbar_Run2` = 833 pb (13 TeV)
- `xsec_ttbar_Run3` = 923 pb (13.6 TeV)

**Source era**: 2018 (59.8 fb⁻¹)

### Scale Factors from 2018
| Target Era | Luminosity | Scale Factor |
|------------|------------|--------------|
| 2022       | 8.0 fb⁻¹   | 0.148 |
| 2022EE     | 27.0 fb⁻¹  | 0.500 |
| 2023       | 17.8 fb⁻¹  | 0.330 |
| 2023BPix   | 9.5 fb⁻¹   | 0.176 |

### Eras and Luminosity
| Era | Energy | Luminosity (fb⁻¹) |
|-----|--------|-------------------|
| 2016preVFP | 13 TeV | 19.5 |
| 2016postVFP | 13 TeV | 16.8 |
| 2017 | 13 TeV | 41.5 |
| 2018 | 13 TeV | 59.8 |
| 2022 | 13.6 TeV | 8.0 |
| 2022EE | 13.6 TeV | 27.0 |
| 2023 | 13.6 TeV | 17.8 |
| 2023BPix | 13.6 TeV | 9.5 |
