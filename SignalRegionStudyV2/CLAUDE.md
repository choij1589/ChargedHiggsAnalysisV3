# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Module Overview

SignalRegionStudyV2 is the **current baseline** for charged Higgs signal region analysis and limit extraction. It implements a complete statistical analysis pipeline using HiggsCombine, covering preprocessing, template generation, datacard production, and limit calculation.

**Key features vs V1:**
- Run3 signal scaling from 2018 (cross-section and luminosity ratios)
- Multi-channel/era combination (SR1E2Mu + SR3Mu → Combined; Run2, Run3, All)
- Extended 19-bin binning scheme (±10σ capturing mass resolution sidebands)
- ParticleNet MVA method with optimized score threshold
- Full HTCondor DAGMan workflow with dependency management

**Target physics:** Charged Higgs (MHc 70–160 GeV, MA 15–155 GeV)
**Channels:** SR1E2Mu (e+2μ), SR3Mu (3μ), Combined, TTZ2E1Mu (validation)
**Methods:** Baseline (cut-based), ParticleNet (MVA score-based)

## Environment Setup

**CRITICAL:** Always source the module-local `setup.sh`, NOT the root-level one:
```bash
cd SignalRegionStudyV2
source setup.sh    # Sets up ROOT/CMSSW and PYTHONPATH
```

Use `python3` (not `python`) for all Python scripts in this module.

## Key Commands

### Individual Pipeline Steps
```bash
# Step 1: Preprocess samples (creates trees with systematic variations)
python3 python/preprocess.py --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90

# Step 2: Generate binned templates
python3 python/makeBinnedTemplates.py --era 2018 --channel SR1E2Mu \
  --masspoint MHc130_MA90 --method Baseline --binning extended

# Step 3: Validate templates
python3 python/checkTemplates.py --era 2018 --channel SR1E2Mu \
  --masspoint MHc130_MA90 --method Baseline --binning extended

# Step 4: Generate datacard
python3 python/printDatacard.py --era 2018 --channel SR1E2Mu \
  --masspoint MHc130_MA90 --method Baseline --binning extended

# Step 5: Combine channels (SR1E2Mu + SR3Mu → Combined)
python3 python/combineDatacards.py --mode channel --era 2018 \
  --masspoint MHc130_MA90 --method Baseline --binning extended

# Step 5b: Combine eras (e.g., all Run2 eras → Run2)
python3 python/combineDatacards.py --mode era --eras 2016preVFP,2016postVFP,2017,2018 \
  --channel Combined --masspoint MHc130_MA90 --method Baseline

# Step 6: Run asymptotic limits
bash scripts/runAsymptotic.sh --era 2018 --channel Combined \
  --masspoint MHc130_MA90 --method Baseline --binning extended

# Step 7: Collect and plot limits
python3 python/collectLimits.py --era Run2 --method Baseline
python3 python/plotLimits.py --era Run2 --method Baseline --limit_type Asymptotic
```

### Batch Processing (HTCondor DAGMan)
```bash
# Preprocess all Run2 samples
./automize/preprocess.sh --mode run2

# Full template pipeline (all eras, all mass points)
./automize/makeBinnedTemplates.sh --mode all --method Baseline --binning extended

# ParticleNet method with partial unblinding
./automize/makeBinnedTemplates.sh --mode all --method ParticleNet \
  --partial-unblind --binning extended

# Full unblinding (templates + asymptotic)
./automize/makeBinnedTemplates.sh --mode all --method Baseline --unblind
./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --unblind

# Single era for quick testing
./automize/makeBinnedTemplates.sh --era 2018 --method Baseline

# Dry-run (inspect DAG without submitting)
./automize/makeBinnedTemplates.sh --mode all --dry-run
```

## Workflow Pipeline

```
SKNanoOutput (raw ROOT files)
    ↓
preprocess.py           → samples/{era}/{channel}/{masspoint}/*.root  (trees with systematics)
    ↓
makeBinnedTemplates.py  → templates/.../shapes.root + datacard.txt + signal_fit.json + binning.json
    ↓
checkTemplates.py       → templates/.../validation/ (diagnostic plots)
    ↓
printDatacard.py        → templates/.../datacard.txt (HiggsCombine datacard)
    ↓
combineDatacards.py     → templates/{Combined or Run2}/{masspoint}/.../datacard.txt
    ↓
runAsymptotic.sh / runHybridNew.sh  → higgsCombine.*.root (limit trees)
    ↓
collectLimits.py        → results/json/limits.*.json
    ↓
plotLimits.py           → results/plots/limit.*.png (Brazilian band plots)
```

## Directory Structure

```
SignalRegionStudyV2/
├── setup.sh                          # Module environment setup (source this!)
├── python/
│   ├── preprocess.py                 # Preprocess samples → trees with systematics
│   ├── makeBinnedTemplates.py        # Generate histogram templates for HiggsCombine
│   ├── template_utils.py             # Shared: binning, ParticleNet scoring, syst handling
│   ├── checkTemplates.py             # Validate templates + diagnostic plots
│   ├── printDatacard.py              # Generate HiggsCombine datacard
│   ├── combineDatacards.py           # Combine datacards across channels/eras
│   ├── collectLimits.py              # Collect limit values from ROOT outputs
│   ├── plotLimits.py                 # Plot exclusion limits (Brazilian bands)
│   ├── plotParticleNetScore.py       # Plot ParticleNet score distributions
│   ├── plotBiasTest.py               # Post-fit bias test plots
│   ├── filterImpacts.py              # Filter systematic impacts
│   └── extractInjectionResults.py    # Extract signal injection results
├── configs/
│   ├── samplegroups.json             # Sample lists per era/channel
│   ├── masspoints.json               # Central mass point arrays (all subsets)
│   ├── scaling.json                  # Run3 signal scaling factors
│   ├── dagman.config                 # DAGMan throttling (MAX_JOBS_SUBMITTED=100)
│   └── systematics.{era}.json        # Systematic uncertainties per era (all 8 eras)
├── automize/                         # Batch processing scripts (all condor-only)
│   ├── load_masspoints.sh            # Sources masspoints.json → bash arrays (single python3 call)
│   ├── preprocess.sh                 # Batch preprocess (HTCondor DAGMan, condor-only)
│   ├── makeBinnedTemplates.sh        # DAGMan workflow orchestration
│   ├── hybridnew.sh                  # Batch HybridNew limit calculation (--test for subset)
│   ├── impact.sh                     # Batch systematic impacts
│   ├── signalInjection.sh            # Batch signal injection
│   └── plotLimits.sh                 # Batch limit plotting
├── scripts/
│   ├── runAsymptotic.sh              # Single asymptotic limit calculation
│   ├── runHybridNew.sh               # Single HybridNew calculation
│   ├── runSignalInjection.sh         # Single signal injection
│   ├── runImpacts.sh                 # Single impacts calculation
│   ├── preprocess_wrapper.sh         # HTCondor wrapper for preprocess jobs
│   ├── makeBinnedTemplates_wrapper.sh # HTCondor wrapper for template DAG steps
│   ├── plotScore_wrapper.sh          # HTCondor wrapper for score plotting
│   └── rsync_templates.sh            # Template synchronization
├── templates/                        # Generated histogram templates (output)
│   └── {era}/{channel}/{masspoint}/{method}/{binning}/
│       ├── shapes.root               # Main template for HiggsCombine
│       ├── datacard.txt              # HiggsCombine datacard
│       ├── signal_fit.json           # A-mass fit parameters {mA, width, sigma}
│       ├── binning.json              # Bin edges and mass window
│       ├── background_weights.json   # ParticleNet class weights (if ParticleNet)
│       ├── threshold.json            # Optimized score threshold (if ParticleNet)
│       ├── process_list.json         # {separate_processes, merged_to_others}
│       ├── lowstat.json              # Low-stat process fallbacks
│       ├── validation/               # Diagnostic plots
│       └── combine_output/           # HiggsCombine outputs (asymptotic, hybridnew, impacts)
├── samples/                          # Preprocessed samples (output of preprocess.py)
│   └── {era}/{channel}/{masspoint}/{sample}.root
├── results/
│   ├── json/                         # Limit values JSON
│   └── plots/                        # Limit plots (Brazilian bands)
└── docs/
    └── LOWSTAT.md                    # Low-statistics handling documentation
```

## Configuration Files

### configs/samplegroups.json
Sample names per era and channel:
- **data / nonprompt:** Data streams (MatrixAnalyzer for nonprompt)
- **WZ, ZZ, ttW, ttZ, ttH, tZq:** Individual backgrounds
- **conversion:** DYJets, TTG, WWG
- **others:** Rare SM processes (WWW, WWZ, etc.)

### configs/scaling.json
Run3 signal scaling from 2018:
```json
{
  "ttbar_xsec": {"Run2": 833.0, "Run3": 923.0},
  "luminosity": {"2018": 59.8, "2022": 8.0, ...},
  "source_era_for_run3": "2018"
}
```
Scale factor = (σ_Run3 / σ_Run2) × (L_target / L_source)

### configs/systematics.{era}.json
Systematic uncertainties per channel, organized as:
```json
{
  "SR1E2Mu": {
    "SystName": {
      "source": "preprocessed|valued",
      "type": "shape|lnN",
      "variations": ["SystName_Up", "SystName_Down"],
      "value": 1.05,
      "group": ["signal", "WZ", ...]
    }
  }
}
```
**Types:** `preprocessed` (from ROOT tree branches), `valued` (percentage modifier), `multi_variation` (PDF/Scale envelopes)

## Mass Points and Methods

### Baseline Method (35 mass points)
Cut-based selection, no MVA. All mass points in `configs/masspoints.json` → `baseline` key.

### ParticleNet Method (3 mass points)
MVA-based with optimized score threshold. Trained mass points in `configs/masspoints.json` → `particlenet` key:
- MHc100_MA95, MHc130_MA90, MHc160_MA85

**ParticleNet score formula:**
```
score_PN = s_signal / (s_signal + w_nonprompt×s_nonprompt + w_diboson×s_diboson + w_ttX×s_ttX)
```
Background weights from cross-section ratios in mass window.

**Threshold optimization:** Scans 101 thresholds (0–1.0), maximizes Asimov significance Z = √[2((S+B)ln(1+S/B)−S)].

### Mass Point Subsets
All subsets defined in `configs/masspoints.json` and loaded by `automize/load_masspoints.sh`:
| Key | Count | Purpose |
|-----|-------|---------|
| `baseline` | 35 | Full Baseline method run |
| `particlenet` | 3 | Full ParticleNet method run |
| `run3_real_mc` | 5 | Mass points with real Run3 MC (no scaling needed) |
| `partial_unblind` | 3 | Partial-unblind subset |
| `impact.baseline` | 4 | Impact plot subset (Baseline) |
| `impact.particlenet` | 3 | Impact plot subset (ParticleNet) |
| `signal_injection.baseline` | 4 | Signal injection subset (Baseline) |
| `signal_injection.particlenet` | 3 | Signal injection subset (ParticleNet) |
| `hybridnew.baseline` | 4 | HybridNew vs Asymptotic comparison subset |
| `hybridnew.particlenet` | 3 | HybridNew vs Asymptotic comparison subset (PN) |

### Blinding Modes (Three-Way Exclusive)

| Flag | Template dir suffix | data_obs | Impact plot |
|------|---------------------|----------|-------------|
| (none) | `extended` | Asimov (sum MC) | expected only |
| `--partial-unblind` | `extended_partial_unblind` | real data, `score_PN < 0.3` | `--blind` (hides r) |
| `--unblind` | `extended_unblind` | real data, full / `score_PN >= best_threshold` | shows observed r |

All scripts accept the flag except signal injection (`runSignalInjection.sh` / `automize/signalInjection.sh`), which is Asimov-only by design. `--unblind` and `--partial-unblind` are mutually exclusive in all scripts.

**collectLimits/plotLimits unblind:** adds `.unblind` suffix to JSON/plot filenames:
```bash
python3 python/collectLimits.py --era Run2 --method Baseline --unblind
# → results/json/limits.Run2.Asymptotic.Baseline.unblind.json
python3 python/plotLimits.py --era Run2 --method Baseline --limit_type Asymptotic --unblind
# → results/plots/limit.Run2.Asymptotic.Baseline.unblind.png
```

## Binning Schemes

**Extended (default, 19 bins):** ±10σ coverage; uniform 15-bin core (±5σ) + 4 extra tail bins at ±7σ/±10σ.

**Uniform (15 bins):** Simple uniform bins covering ±5σ only.

Bin edges calculated from A-mass fit parameters (mA, width, sigma) via `template_utils.py`.

## A-Mass Fitting

- **SR1E2Mu:** RooFit Voigt function fit to signal mass1 distribution → (mA, width, sigma)
- **SR3Mu:** Loads fit results from SR1E2Mu (must process SR1E2Mu first!)
- Output: `signal_fit.json` and `fit_result.root`

## Systematic Handling (template_utils.py)

Three systematic types:
1. **Preprocessed shape:** Up/Down variations as separate ROOT TTrees in input file
2. **Valued shape:** Apply symmetric percentage modifier on-the-fly
3. **Multi-variation (PDF/Scale):** Envelope from 100 PDF or 9 scale variations; bin-by-bin max/min

**Low-statistics handling:**
- Backgrounds with relative stat error > 30% flagged as low-stat
- Low-stat processes: shape systematics removed, lnN fallback applied
- Configuration saved in `lowstat.json`; see `docs/LOWSTAT.md` for details

**autoMCStats:** Threshold is 5 (in `printDatacard.py`). Bins with ≥5 effective MC events get Barlow-Beeston-lite nuisances. Lower threshold = fewer nuisances = faster fits.

## Run3 Signal Handling

- By default uses real Run3 MC from SKNanoOutput (if available)
- Use `--scale-from-run2` in `preprocess.py` to scale 2018 MC to Run3 luminosity
- Automatic detection in `makeBinnedTemplates.py` via tree name inspection

## HTCondor DAGMan Workflow

`automize/makeBinnedTemplates.sh` builds and submits a DAG that respects pipeline dependencies:

**Key dependencies:**
1. SR1E2Mu template → SR3Mu template (fit results needed)
2. Both channel templates → Datacards
3. Datacards → Validation → Channel combination
4. Per-era asymptotics → Era combination → Combined asymptotics

**All `automize/` scripts are condor-only.** The `--condor` flag is accepted for backwards compatibility but is a no-op.

**Control options:**
- `--start-from combine_era`: Skip to era combination (per-era work done)
- `--dry-run`: Print DAG without submitting
- `--mode all`: Process all eras and mass points

**HybridNew `--test` flag:** Use subset mass points (`hybridnew.baseline`/`hybridnew.particlenet`) to compare Asymptotic vs HybridNew before running full set:
```bash
./automize/hybridnew.sh --mode all --method Baseline --auto-grid --test
```

**HybridNew auto-grid:** `--auto-grid` reads Asymptotic results and sets the r-scan range to `[0.8×exp-2σ, 1.2×exp+2σ]` with ~20 points:
```python
rmin = max(0.01, exp_minus2 * 0.8)
rmax = exp_plus2 * 1.2
rstep = round((rmax - rmin) / 20, 3)
```
- **Job count:** ~7,000 toy jobs for full Baseline run (35 mass points × ~20 r-points × 10 jobs/point)
- **Throttling:** `DAGMAN_MAX_JOBS_SUBMITTED=100` is per DAGMan process (one per mass point); 35 simultaneous DAGs → up to 3,500 queued jobs
- **Toys per job:** 50 (default in `automize/hybridnew.sh`); `runHybridNew.sh` standalone default is 100

## KNU Tier2/Tier3 Storage and HTCondor

### Storage Element (SE) Access

**SE_UserHome path:** `/pnfs/knu.ac.kr/data/cms/store/user/{CERN_ID}` (uses CERN ID, not KNU ID)

**Access protocols:**
| Protocol | Path Format | Notes |
|----------|-------------|-------|
| xrootd | `root://cluster142.knu.ac.kr//store/user/{userid}/...` | Recommended for HTCondor jobs |
| dcap | `dcap://cluster142.knu.ac.kr//pnfs/knu.ac.kr/data/cms/store/user/{userid}/...` | Read-only, no auth, KNU internal |
| NFS | `/pnfs/knu.ac.kr/...` | Available in HTCondor (since 2023.06.23), no overwrite/append |

**Important:** NFS does NOT support overwriting or appending files. Use xrootd for write operations.

### HTCondor at KNU

**UI servers:**
- Tier-2: `kcms-t2.knu.ac.kr` (or `cms.knu.ac.kr`, `cms01.knu.ac.kr`)
- Tier-3: `kcms-t3.knu.ac.kr` (or `cms02.knu.ac.kr`, `cms03.knu.ac.kr`)

**Job monitoring:**
- `condor_q` — check job status
- `condor_tail -f <job_id>` — real-time stdout/stderr
- `condor_ssh_to_job <job_id>` — SSH into running job for debugging

**Reference:** [KNU T2/T3 Wiki](http://t2-cms.knu.ac.kr/wiki/index.php/HTCondor)

## Input Data Paths

```
# Signal (Run2)
$WORKDIR/SKNanoOutput/PromptAnalyzer/Run1E2Mu_RunSyst_RunTheoryUnc/{era}/{signal}.root
$WORKDIR/SKNanoOutput/PromptAnalyzer/Run3Mu_RunSyst_RunTheoryUnc/{era}/{signal}.root

# Backgrounds
$WORKDIR/SKNanoOutput/PromptAnalyzer/Run1E2Mu_RunSyst/{era}/{sample}.root
$WORKDIR/SKNanoOutput/PromptAnalyzer/Run3Mu_RunSyst/{era}/{sample}.root

# Nonprompt (data-driven, MatrixAnalyzer)
$WORKDIR/SKNanoOutput/MatrixAnalyzer/Run1E2Mu/{era}/{sample}.root

# Data
$WORKDIR/SKNanoOutput/PromptAnalyzer/Run1E2Mu/{era}/{data}.root
```

## Common Issues

**"WORKDIR not set":** Source `setup.sh` inside the SignalRegionStudyV2 directory.

**SR3Mu fit not found:** SR1E2Mu must be processed first — it generates `signal_fit.json` needed by SR3Mu template generation.

**Run3 signal missing:** Check `configs/scaling.json` for correct scaling factors. Use `--scale-from-run2` if Run3 MC is not available.

**Low-stat process merging:** Expected behavior for rare backgrounds. Check `process_list.json` for the final background configuration after merging.

**ParticleNet scores missing / wrong mass point:** Only the 3 trained mass points are supported for ParticleNet method (see `configs/masspoints.json` → `particlenet`). Other mass points must use Baseline.

**HTCondor job failures:** Check `condor/jobs_*/logs/` for detailed error messages. Verify `setup.sh` is sourced in wrapper scripts.

**Zero/negative histogram integrals:** Check mass window selection, sample weights (especially signal normalization ÷3.0), and ConvSF loading from TriLepton results.

**ConvSF fallback:** If `TriLepton/results/{ZG1E2Mu,ZG3Mu}/{era}/ConvSF.json` is not found, `preprocess.py` silently falls back to `ConvSF = 1.0 ± 0.3`. Check the log output for "Using default ConvSF" warnings.

**combineCards.py not found:** `combineDatacards.py` requires `combineCards.py` from HiggsCombine to be in `$PATH`. Ensure the CMSSW environment with HiggsCombine is set up before running datacard combination steps.
