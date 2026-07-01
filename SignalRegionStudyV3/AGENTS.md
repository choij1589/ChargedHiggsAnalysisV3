# AGENTS.md

This file is the Codex-facing project guide for `SignalRegionStudyV3`. It is
checked against the live code and configs; use `docs/*` for deeper background
on the statistical procedures and review history.

## Project Scope

`SignalRegionStudyV3` is the self-contained full-unblind signal-region workflow
for charged Higgs analysis. It covers preprocessing, binned template production,
datacard generation, channel/era combination, Goodness-of-Fit, impacts,
FitDiagnostics, prefit/postfit plotting, signal injection, TTZ control-region
validation, and final limit extraction.

Target physics:

- Charged Higgs mass points: `MHc 70-160 GeV`, `MA 15-155 GeV`
- Channels: `SR1E2Mu`, `SR3Mu`, `Combined`, and `TTZ2E1Mu` validation
- Methods: `Baseline` cut-based and `ParticleNet` MVA
- Main combinations: channel combination `SR1E2Mu + SR3Mu -> Combined`, then
  era combinations `Run2`, `Run3`, and `All`

## Environment Rules

Always source the module-local setup from this directory, not a root-level
setup script:

```bash
cd SignalRegionStudyV3
source setup.sh
```

Use `python3` for all Python scripts in this module.

`combineCards.py`, `text2workspace.py`, `combine`, and `combineTool.py` require
the CMSSW/HiggsCombine environment from `setup.sh` to be active and available in
`$PATH`.

Do not add runtime dependencies on other signal-region study directories.
V3 code, configs, wrappers, templates, and sample paths should resolve through
`SignalRegionStudyV3` only.

## Core Pipeline

The standard workflow is:

```text
SKNanoOutput raw ROOT files
  -> python/preprocess.py
  -> python/makeBinnedTemplates.py
  -> python/checkTemplates.py
  -> python/printDatacard.py
  -> scripts/runAsymptotic.sh or scripts/runHybridNew.sh
  -> python/collectLimits.py
  -> python/plotLimits.py
```

Single-point example:

```bash
python3 python/preprocess.py --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90
# Repeat preprocessing for all suberas/channels needed by the requested Run period.

python3 python/makeBinnedTemplates.py --era Run2 --channel Combined \
  --masspoint MHc130_MA90 --method Baseline --binning extended

python3 python/checkTemplates.py --era Run2 --channel Combined \
  --masspoint MHc130_MA90 --method Baseline --binning extended

python3 python/printDatacard.py --era Run2 --channel Combined \
  --masspoint MHc130_MA90 --method Baseline --binning extended

bash scripts/runAsymptotic.sh --era Run2 --channel Combined \
  --masspoint MHc130_MA90 --method Baseline --binning extended

python3 python/collectLimits.py --era Run2 --method Baseline
python3 python/plotLimits.py --era Run2 --method Baseline --limit_type Asymptotic
```

V3 templates/datacards are built directly as Run-period component likelihoods.
The legacy subera-category combination code is not part of V3; use
SignalRegionStudyV2 for that reference workflow.

## Batch Workflow

The main production automation scripts are HTCondor-oriented. In particular,
`automize/preprocess.sh`, `automize/makeBinnedTemplates.sh`, `automize/impact.sh`,
`automize/gof.sh`, and `automize/signalInjection.sh` treat `--condor` as a
legacy no-op because Condor is the default. Check each script's `--help` before
assuming that behavior for newer or specialized scripts.

Common commands:

```bash
./automize/preprocess.sh --mode run2

./automize/makeBinnedTemplates.sh --mode all --method Baseline --binning extended

./automize/makeBinnedTemplates.sh --mode all --method ParticleNet \
  --partial-unblind --binning extended

./automize/makeBinnedTemplates.sh --mode all --method Baseline --unblind

./automize/makeBinnedTemplates.sh --era Run2 --method Baseline
./automize/makeBinnedTemplates.sh --mode all --dry-run
```

For `automize/makeBinnedTemplates.sh`, DAG dependencies are: Run-period
component templates, datacards, validation, asymptotics, optional
FitDiagnostics/postfit plotting, and optional ParticleNet score plots. There is
no channel-combination or era-combination combineCards.py stage in the default
V3 workflow.

HybridNew guidance:

- Use `./automize/hybridnew.sh --mode all --method Baseline --auto-grid --test`
  before a full run.
- In `automize/hybridnew.sh`, `--mode all` processes only the already-combined
  `All` era; use `--mode run2` or `--mode run3` for those combined eras.
- `--auto-grid` reads asymptotic limits and scans roughly
  `[0.8 * exp-2sigma, 1.2 * exp+2sigma]` with about 20 points.
- Full Baseline HybridNew can enqueue thousands of toy jobs; check throttling in
  `configs/dagman.config` and the generated DAGs.

## Configuration

Key files:

- `configs/samplegroups.json`: sample groups per era/channel.
- `configs/masspoints.json`: central mass-point lists and subsets.
- `configs/systematics.{era}.json`: systematic definitions per era.
- `configs/dagman.config`: DAGMan throttling.

`configs/masspoints.json` subset keys currently include:

- `baseline`: integrated Baseline blind/unblind run, 39 mass points.
- `particlenet`: ParticleNet run, 3 trained points.
- `limits`: curated Baseline plotting list, 20 mass points.
- `impact.baseline`, `impact.particlenet`: blinded impact-plot subsets.
  With `--unblind`, impact automation uses all `baseline`/`particlenet` points.
- `signal_injection.baseline`, `signal_injection.particlenet`: blinded signal
  injection subsets.

Optional keys such as `partial_unblind`, `hybridnew`, and `gof` may be added for
study-specific subsets. If they are absent, automation falls back to
`baseline`/`particlenet`.

ParticleNet is only supported for the trained mass points listed under
`particlenet`. Other mass points should use `Baseline`.

## Blinding Modes

The template directory suffix and data treatment depend on exactly one mode:

| Mode | Directory suffix | `data_obs` | Impact plots |
| --- | --- | --- | --- |
| default | `extended` | Asimov, sum of MC | expected only |
| `--partial-unblind` | `extended_partial_unblind` | real data with `score_PN < 0.3` | r hidden in impact plots |
| `--unblind` | `extended_unblind` | real data, full or `score_PN >= best_threshold` | observed r shown |

In the template, datacard, limit, impact, pull-plot, and validation scripts that
accept both flags, `--unblind` and `--partial-unblind` are mutually exclusive.
Signal injection is Asimov-only by design.

For unblinded limit collection/plotting, pass `--unblind`; outputs get a
`.unblind` suffix under the selected limit mode, e.g.
`results/json/BR/Run2/limits.Run2.Asymptotic.Baseline.unblind.json` and
`results/plots/BR/Run2/limit.Run2.Asymptotic.Baseline*.unblind.png`.

## Binning and Signal Fits

Default binning is `extended`. In V3 this name means the adaptive coarser
binning used for full unblinding: start from 15 core bins plus two sideband bins
over +/-10 sigma, then scan down to 5 core bins if needed for low-stat quality.
Output suffixes remain `extended`, `extended_unblind`, and
`extended_partial_unblind`.

Signal A-mass fits use an unbinned Double Crystal Ball fit in
`python/makeBinnedTemplates.py`. In Run-period component mode each merged
category gets its own fit and binning, e.g. `SR1E2Mu_Run2`, `SR3Mu_Run2`,
`SR1E2Mu_Run3`, and `SR3Mu_Run3` for `All/Combined`.

## Low-Statistics Treatment

See `docs/LOWSTAT.md` before changing low-stat or autoMCStats behavior.

Important invariants:

- Keep the static separate MC background list across eras:
  `nonprompt`, `WZ`, `ZZ`, `ttW`, `ttZ`, `ttH`, `tZq`, `conversion`.
- Do not merge low-stat processes into `others`; missing input files are the
  legitimate reason for `dropped_missing` in `process_list.json`.
- Adaptive binning must test raw per-process histograms, without floor/cap
  hygiene, so low effective-stat bins force coarser binning.
- Adaptive binning quality must match Combine autoMCStats:
  `n_eff = round(y^2 / sigma^2) >= 5` on total background per bin.
- Apply `ensure_positive_integral` only after final binning is chosen.
- Signal and `others` use `floor_mode="floor"` with `1e-6`; individual
  backgrounds use `floor_mode="zero"` so empty process bins are skipped by
  autoMCStats.
- Low-stat backgrounds with relative stat error above 30% use `shape?` fallback
  plus lnN values, and their removed shape histograms are preserved in
  `shapes_original.root`.
- `lowstat.json` records the fallback behavior and is consumed by
  `checkTemplates.py`.

Thresholds to preserve unless deliberately retuning the statistical model:

- `BIN_FLOOR_VALUE = 1e-6`
- `AUTOMC_THRESHOLD = 5`
- `SHAPE_REL_ERR_THRESHOLD = 0.30`
- `MAX_LNN_VALUE = 2.0`
- `SYST_MERGE_THRESHOLD = 2.0`

## Systematics and Datacards

Systematic sources include:

- `preprocessed shape`: Up/Down TTrees in input ROOT files.
- `valued shape`: symmetric percentage modifiers applied in code.
- `multi_variation`: PDF/scale envelopes.
- `lnN`: normalization-only nuisances.

For merged `others`, only apply a systematic when `"others"` is explicitly in
its `group`. Per-process normalization nuisances should remain attached to the
matching process groups, while `CMS_B2G25013_Norm_others` covers the merged
bucket and is correlated across Run2/Run3.

Combine correlates nuisances by name, so keep per-era process columns stable
and only decorrelate nuisance names when the statistical model requires it.

## Run3 Signal Samples

Run3 processing uses real Run3 MC from `SKNanoOutput`. There is no Run2-to-Run3
signal scaling fallback. If a Run3 signal sample is missing, `preprocess.py`
should raise `FileNotFoundError`; do not paper over missing input samples.

## Partial-Unblind Impacts

See `docs/PARTIAL_UNBLIND.md` before diagnosing one-sided impacts in
`All/Combined/*/ParticleNet/extended_partial_unblind`.

The observed high one-sided fraction in the full Run2+Run3 partial-unblind
combination can be physical rather than a minimizer failure: the data deficit is
sideband-dominated, the signal is narrow and peaked, best-fit `r` is near zero,
and the combined fit has many weakly constraining nuisance parameters.

Expected handling:

- `scripts/runImpacts.sh --partial-unblind` passes `--blind` to the impact
  plotting step automatically. For staged full-unblind plots where the observed
  r should stay hidden, use `--blind-result`.
- Compare Run2 and Run3 impact plots separately when presenting diagnostics.
- Check expected/Asimov impacts (`impacts_r1`) when looking for normal two-sided
  behavior.
- Focus on the top-ranked nuisance parameters before treating many tiny
  one-sided impacts as a problem.

## Interpolated Signal Templates

See `docs/INTERPOLATE.md` for the full method and validation.

Current status:

- Central Double-Gaussian interpolated template generation exists in
  `python/generate_interpolated_templates.py`.
- Systematic relative-variation interpolation and full datacard integration are
  planned, not complete.

Generated intermediate mass points:

- `MHc85`: `MA25`, `MA35`, `MA45`, `MA55`
- `MHc115`: `MA40`, `MA55`, `MA70`
- `MHc145`: `MA50`, `MA65`, `MA80`

Usage:

```bash
python3 python/generate_interpolated_templates.py --mhc all
python3 python/generate_interpolated_templates.py --mhc 115 --era 2018 --channel SR1E2Mu
python3 python/test_interpolation.py
```

Do not treat central-only interpolated templates as full production templates
until shape-systematic interpolation is implemented.

## Combine Review and Statistical Tests

The long-form review notes live in `docs/CombineCommandsForOR.md` and
`docs/B2GStatRecommendations.md`. Use them when preparing Stat/Combine review
material.

Review principles to preserve:

- Stay blinded in or near the signal region until the analysis is approved to
  unblind. Use control/measurement regions, sidebands, or channel masks.
- Use saturated-model Goodness-of-Fit for binned Poisson templates.
- B2G expects nuisance pulls, nuisance impacts, Goodness-of-Fit, and signal
  injection tests; F-tests only if relevant.
- For impact plots, avoid hard physics bounds such as `r >= 0` when they would
  bias variations near `r = 0`; use appropriate `--rMin` and `--rMax`.
- If an impact plot uncertainty hits simple bounds such as `0`, `-1`, `2`, or
  `20`, suspect an insufficient `r` range before interpreting the result.
- For blinded toy generation with an unmasked SR, never use
  `--toysFrequentist` by itself. Use `--toysFrequentist --bypassFrequentistFit`
  with a data-like model from the masked measurement/control region.
- In signal-injection tests, do not perform the initial frequentist fit to data
  with nonzero injected signal; that biases the background model.
- Fit generated toys with wide enough `--rMin` and `--rMax` to contain the
  expected fit result and uncertainty.
- When checking systematic names for datacard CI, use the CAT systematics
  workflow and remember that the rename script may not preserve `nuisance edit`
  lines; add them back manually.

## KNU Tier2/Tier3 Notes

Storage Element user home:

```text
/pnfs/knu.ac.kr/data/cms/store/user/{CERN_ID}
```

Use CERN ID, not KNU ID.

Access protocols:

- xrootd: `root://cluster142.knu.ac.kr//store/user/{userid}/...`
- dcap: `dcap://cluster142.knu.ac.kr//pnfs/knu.ac.kr/data/cms/store/user/{userid}/...`
- NFS: `/pnfs/knu.ac.kr/...`

NFS does not support overwrite or append. Use xrootd for writes.

Useful HTCondor commands:

```bash
condor_q
condor_tail -f <job_id>
condor_ssh_to_job <job_id>
```

## Input Paths

Expected input layout under `$WORKDIR`:

```text
SKNanoOutput/PromptAnalyzer/{Run1E2Mu,Run3Mu}_RunSyst_RunTheoryUnc/{era}/TTToHcToWAToMuMu-{masspoint}.root
SKNanoOutput/PromptAnalyzer/{Run1E2Mu,Run3Mu,Run2E1Mu}_RunSyst/{era}/Skim_TriLep_{sample}.root
SKNanoOutput/MatrixAnalyzer/{Run1E2Mu,Run3Mu,Run2E1Mu}/{era}/Skim_TriLep_{sample}.root
SKNanoOutput/PromptAnalyzer/{Run1E2Mu,Run3Mu,Run2E1Mu}/{era}/Skim_TriLep_{data}.root
```

`TTZ2E1Mu` maps to the `Run2E1Mu` input channel and reuses the `SR1E2Mu`
systematics/samplegroup config. Signal processing is only done for `SR1E2Mu`
and `SR3Mu`.

TTZ2E1Mu CR GoF uses the same Run-period component template construction as
the SR workflow. Use `python/makeCRTemplates.py --era Run2|Run3|All`,
`python/printCRDatacard.py --era Run2|Run3|All`, or
`./automize/ttz_cr_gof.sh`; do not rebuild the CR by combining per-subera
datacards.

## Troubleshooting

- `WORKDIR not set`: source module-local `setup.sh` in `SignalRegionStudyV3`.
- `combineCards.py not found`: source the HiggsCombine/CMSSW environment.
- Run3 signal missing: add the real Run3 MC sample or skip that combination.
- ParticleNet score/mass mismatch: use only mass points in
  `configs/masspoints.json` under `particlenet`.
- HTCondor failures: inspect `condor/jobs_*/logs/` and wrapper setup.
- Zero/negative histogram integrals: check mass-window selection, sample
  weights, signal normalization, and ConvSF loading.
- ConvSF loading: `preprocess.py` reads `Common/Data/ConvSF.json` directly.
  There is no fallback in the current code; a missing or incomplete file should
  be treated as an input/configuration error.

## Editing Guidance for Agents

- Prefer existing helper APIs in `python/template_utils.py` and local pipeline
  patterns over new ad hoc logic.
- Keep generated-output directories (`templates/`, `samples/`, `results/`,
  `condor/`) out of source edits unless the task explicitly requires artifacts.
- Preserve blinding semantics when adding options or scripts.
- Preserve static process/nuisance naming unless intentionally changing the
  statistical model.
- Update the relevant doc in `docs/` when changing low-stat, interpolation,
  blinding, or Combine-review behavior.
