# CLAUDE.md — SignalRegionStudyV4

Simplified, self-contained signal-region limit chain for the charged Higgs
search (B2G-25-013), and the base for the next-generation work (parametric
signal models, mA interpolation). This file is the single module guide;
`docs/*` hold deep dives.

## Project Scope

- Core chain only: preprocess → binned templates → run-period merge →
  datacard → validation → asymptotic limits → FitDiagnostics/postfit/pulls →
  collect/plot limits. See `docs/FUNCTIONALITY_SCOPE.md` for the explicit
  active/not-ported lists.
- Methods: `Baseline`, `ParticleNet`. Channels: `SR1E2Mu`, `SR3Mu`,
  `Combined`. Run-period targets: `Run2`, `Run3`, `All`.
- Mass points from `configs/masspoints.json` (`baseline` 78, `particlenet`
  22, `limits` curated plotting subset).
- **Self-containment rule**: zero code references, imports, or symlinks into
  any other SignalRegionStudy directory. The only V3 touchpoint is the
  validation-only comparator `python/compareToV3.py` (explicit `--v3-dir`,
  reads frozen V3 *outputs* only).

## Environment Rules

```bash
source setup.sh   # module-local; REQUIRED before any work (not the repo root setup.sh)
```

- CMSSW_14_1_0_pre4 + Combine v10 (ROOT 6.30) only. Newer ROOT (6.36)
  segfaults template making.
- Use `python3`, never `python`.
- Never run heavy analysis/ROOT work on the login node — submit to condor.
  The `automize/` drivers and `scripts/compare_wrapper.sh` exist for this.
- A valid grid proxy (`voms-proxy-init --voms=cms`) is needed for
  preprocessing (xrdcp output to pnfs).

## Path & Layout Contract

```
samples/{era}/SR1E2Mu/                       shared bkg/nonprompt/data + ALL signals ({masspoint}.root)
samples/{era}/SR3Mu_{lowM,highM}/            shared bkg/nonprompt/data + ALL signals in BOTH
                                             pairing variants (interpolation needs both)
samples/{era}/{channel}/{masspoint}/         ParticleNet per-masspoint dirs (scores +
                                             NoHistMode skims; incl. TTZ2E1Mu)
templates/{masspoint}/{method}/{era}/{channel}/
results/json/{BR,xsec}/{era}/limits.{era}[.{channel}].Asymptotic.{method}.json
fits/MHc{X}/                                 mA-interpolation fit artifacts (dcb_fits, polynomials,
                                             yield model, shape_deltas, validation plots); global
                                             surface panels in fits/{params,yield}/
closure/interpolation/{MHc{X},loo/}          interpolation closures, LOO sweep, pooled
                                             uncertainty diagnostics — see docs/interpolation/
```

The SR3Mu pairing rule: `highM` (higher-mass dimuon pairing) iff
`mHc >= 100 && mA >= 60` — a 2D condition, not a pure mA threshold
(MHc160_MA15 is lowM, MHc70_MA60 is lowM). `srspaths.pairing_variant()`
is the only place it is defined for path resolution;
`preprocess.pairing_for()` mirrors it for production.

- Unblind (real data) is the default everywhere. `--blind` gives Asimov
  data_obs and writes to a `{method}_blind` method segment.
- Path construction lives in exactly two places: `python/srspaths.py`
  (python) and `scripts/env.sh` (shell). Never construct these paths
  anywhere else.
- Site constants (pnfs/xrootd bases, CMSSW release) are defined once in
  `scripts/env.sh`.

## Procedures (in chain order)

### 1. Preprocess

Skims SKNanoOutput into flat per-process trees (one TTree per systematic;
branches `mass`, `mass1`, `mass2`, `pT`, `weight`, and `score_{mp}_*` in
ParticleNet per-masspoint dirs). Vectorized: RDataFrame Define+Snapshot.

```bash
./automize/preprocess.sh [--masspoint MHc130_MA90] [--skip-backgrounds] \
                         [--backgrounds-only] [--dry-run]
```

- Shared-background DAG (24 nodes: 8 eras × {SR1E2Mu, SR3Mu:lowM,
  SR3Mu:highM}) — run ONCE for the whole analysis; backgrounds, nonprompt
  and data are mass-independent.
- Per-masspoint DAG: 16 shared-signal nodes (SR3Mu writes both pairing
  variants), plus 24 full ParticleNet nodes (incl. TTZ2E1Mu) for
  ParticleNet-trained points.
- An interpolated mass point needs only its signal: the shared backgrounds
  are already in place.

Outputs land on pnfs via xrdcp. Owning code: `python/preprocess.py`
(modes: default per-masspoint ParticleNet / `--shared-backgrounds` /
`--shared-signal`), `scripts/preprocess_wrapper.sh`.
Deep dive (branches, weights, pairing rule, layout): `docs/SAMPLES.md`.

### 2. Binned templates

DCB signal fit per merged run-period category, adaptive binning (15→5 core
bins), all systematic variations, low-stat handling (`docs/LOWSTAT.md`).

```bash
python3 python/makeBinnedTemplates.py --era Run2 --channel SR1E2Mu \
    --masspoint MHc130_MA90 --method Baseline
```

Condor: part of the `makeBinnedTemplates.sh` DAG (template step, 6 GB).
Outputs: `shapes.root`, `binning.json`, `categories.json`,
`process_list.json`, `background_validation.json`, signal-fit PNGs
(+ `threshold*.json`, `background_weights*.json` for ParticleNet).

### 3. Run-period merge

File-level union of per-channel/per-period template dirs into Combined/All
targets. Reads `shapes_original.root` from components when present (merge
race guard — see `docs/LOWSTAT.md` S3).

```bash
python3 python/mergeRunPeriodTemplates.py --era All --channel Combined \
    --masspoint MHc130_MA90 --method Baseline --sources Run2:SR1E2Mu,Run2:SR3Mu,Run3:SR1E2Mu,Run3:SR3Mu
```

### 4. Datacard

Multi-category component-process datacard; prunes low-stat shapes to lnN
(`shape?`), writes `lowstat.json`, `shapes_original.root` archive.

```bash
python3 python/printDatacard.py --era All --channel Combined \
    --masspoint MHc130_MA90 --method Baseline
```

### 5. Validation

Closure, binning consistency, shape? Up/Down presence, text2workspace
smoke, diagnostic plots into `validation/`.

```bash
python3 python/validateRunPeriodTemplates.py --era All --channel Combined \
    --masspoint MHc130_MA90 --method Baseline
```

### 6. Asymptotic limits

```bash
./scripts/runAsymptotic.sh --era All --channel Combined \
    --masspoint MHc130_MA90 --method Baseline
```

Frozen Combine command: `combine -M AsymptoticLimits datacard.txt
-n .{mp}.{method} -m 120 --rAbsAcc 0.0001 --rRelAcc 0.01`.
Output: `combine_output/asymptotic/higgsCombine.{mp}.{method}.AsymptoticLimits.mH120.root`.

### 7. FitDiagnostics, postfit mass plots, pulls

Run for `All/Combined` in `--mode all` batch runs.

```bash
./scripts/runFitDiagnostics.sh --era All --channel Combined --masspoint MHc130_MA90 --method Baseline
python3 python/plotPostfitMass.py --era All --masspoint MHc130_MA90 --method Baseline
./scripts/runPullPlots.sh --era All --channel Combined --masspoint MHc130_MA90 --method Baseline --pull-fit both
```

### 8. ParticleNet score plots

Part of the ParticleNet DAG (`plot_score` nodes, 4 GB memory request);
also runnable directly via `python3 python/plotParticleNetScore.py`.

### 9. Collect and plot limits

```bash
python3 python/collectLimits.py --era All --channel Combined --method Baseline
python3 python/plotLimits.py --era All --method Baseline --mhc 130
```

`collectLimits.py` skips-and-logs missing mass points — a partial grid
silently produces a partial JSON, so check the parsed/total count it
reports. `--masspoint` + `--output` give a side-effect-free single-point
collect.

## Batch Workflow (condor)

```bash
# Full chain for one method (per-masspoint DAGs; {Run2,Run3,All} x {SR1E2Mu,SR3Mu,Combined}):
./automize/makeBinnedTemplates.sh --mode all --method Baseline    --fitdiag --pull-fit both
./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --fitdiag --pull-fit both

# Useful options: --masspoint MP, --masspoint-set KEY, --start-from STEP,
#                 --blind, --dry-run
```

DAG topology: 4 template leaves ({Run2,Run3}×{SR1E2Mu,SR3Mu}) → merges →
datacard → validate → asymptotic per target/channel; fitdiag/postfit/pulls
on All/Combined only; plot_score for ParticleNet. Shared generation helpers
live in `automize/dag_lib.sh`; mass-point arrays come from
`automize/load_masspoints.sh` (name-keyed parsing).

## Configuration

- `configs/masspoints.json` — keys `baseline`, `particlenet`, `limits` only.
- `configs/systematics.{era}.json` — verbatim from V3 (8 files). A
  base+override refactor is deferred: datacard bytes are frozen by the
  reproduction contract.
- `configs/samplegroups.json`, `configs/histkeys.json`, `configs/dagman.config`.

## Blinding

Unblind is the default; there is no partial-unblind in V4. `--blind`
(makeBinnedTemplates/merge/datacard/validate/runAsymptotic/collectLimits
and the driver) produces Asimov data_obs under `{method}_blind`.

## Low-Statistics & Systematics Invariants

Frozen constants (see `docs/LOWSTAT.md` for the full catalog):
`BIN_FLOOR_VALUE=1e-6`, `AUTOMC_THRESHOLD=5`, `SHAPE_REL_ERR_THRESHOLD=0.30`,
`MAX_LNN_VALUE=2.0`, `SYST_MERGE_THRESHOLD=2.0`, autoMCStats threshold 5.
Nuisance naming/correlation rules and the run-period component model are in
`docs/RUN_PERIOD_COMPONENT_TEMPLATES.md`.

## Run3 Signal Samples

Mass points without real Run3 MC fail preprocessing with
`FileNotFoundError` — that is intentional; missing inputs must surface
explicitly, never silently.

## Reproduction Test

`docs/REPRODUCTION.md`. V4's chain was verified to reproduce V3 exactly for
MHc130_MA90 (datacards bitwise, limits identical to the last bit). V3 is
never re-run; `python/compareToV3.py --v3-dir <path>` compares against its
frozen outputs.

## mA Interpolation Chain

Parametric signal templates at fixed mHc and arbitrary mA. Chain:
per-point DCB fits `fitInterpShapes.py` (floating/frozen-n passes) →
shape surfaces `fitInterpPolynomials.py` → shape closure
`closInterpShapes.py`; window yields `measInterpYields.py` → yield model
`fitInterpYieldModel.py` → yield closure `closInterpYields.py`;
shape-systematic deltas `measInterpShapeDeltas.py` →
`fitInterpShapeDeltas.py`; derived nuisance sizes
`exportInterpUncertainties.py`. Constants in
`python/interpolation_config.py`; `configs/interpolation.json` now holds
only `known_missing_samples` (the mHc list comes from `mhc_grid()`).

**Production model (frozen 2026-08-12, no variant flags):** pure DCB for
SR1E2Mu and SR3Mu_lowM, DCB+Chebychev2 for SR3Mu_highM alone
(`channel_has_bkg`). Every shape parameter, plus the yield model's `G`
and `k_era`, is ONE surface in (mHc, mA) fitted across all seven studies
and sliced at the study's mHc — the slice of a surface is a polynomial,
so the stored records and every downstream consumer are unchanged.
Interpolation is **in mA only**.

**Two cross-study barriers**: `polynomials` reads every study's
`dcb_fits.json` and `yield_model` every study's `yields.json`. A rebuild
therefore runs in three passes, each fully complete before the next:

```bash
./automize/interpolation.sh --all --stop-after fit-frozen
./automize/interpolation.sh --all --start-from polynomials --stop-after yields
./automize/interpolation.sh --all --start-from yield-model
./automize/interpolation.sh --loo --all
python3 python/exportInterpUncertainties.py --loo --all --pooled --write-config
python3 python/plotInterpSurfaces.py --all      # global surface plots
python3 python/plotInterpNuisances.py           # nuisance-rule plots
```

Outputs under `fits/MHc{X}/` (per-study fit artifacts + plots, global
surface panels in `fits/{params,yield}/`) and `closure/interpolation/`
(per-study closures, `loo/MHc{X}_MA{Y}/` leave-one-out dirs, pooled
diagnostics, nuisance plots) — git-tracked production trees, committed
after a verified run. Merge any sharded stage with
`python3 python/mergeInterpResults.py --mhc N --stage
{fits-floating,fits,closure,yields,yield-closure,shape-deltas}`.

Uncertainties are **derived from the leave-one-out closures**, not
assumed, by one rule for all three families: the rms WITHIN each mHc
study, then the MAX across studies holding at least 2 mass points,
floored by the cell's pooled rms and then by an absolute floor (scale
0.02, res 0.01, norm 0.01). Nothing carries an mHc dependence — the rule
already pools studies and both models are global surfaces. norm VALUES
alone are binned in mA at 15/80/100/155 (the target mA selects the bin);
all three nuisance NAMES are period-level,
`CMS_interp_{scale,res,norm}_{ch}_{13TeV|13p6TeV}` — the LOO residual is
common-mode across a period's eras (r = +0.99 Run2 / +0.80 Run3), so one
nuisance spans the era columns, carrying each era's own lnN value.
Values land in `configs/interpolation_uncertainties.json`, keyed by STUDY
channel (lowM/highM) while the nuisance names use the production channel
SR3Mu — safe because one datacard holds one mass point.

Full record in `docs/interpolation/`: `WORKFLOW.md` (runbook + decision
gates), `EXPERIMENTS.md` (every model decision as motivation/setup/
results/conclusion), `UNCERTAINTY.md` (nuisance rule, values, evidence).

## Future Phases

Next: graduate the interpolation template producer behind a
`srspaths.template_dir` method segment, consuming
`configs/interpolation_uncertainties.json` and declaring the nuisances via
`printDatacard.py`'s `extra_systematics*.json` hook. Its mA scan grid is
frozen in `configs/grid.json` (1907 points over 7 mHc; banded steps of
the order of the dimuon σ_eff, all MC points included for direct-MC
comparison; p-notation names like `MA90p5` — `makeInterpGrid.py`,
`srspaths.grid_config()`). Nothing in V4 may
hard-assume the template payload is binned histograms beyond the existing
per-step contracts.

Known limitations of the frozen model, all recorded in
docs/interpolation/WORKFLOW.md:

- **Low-mA grid density is the binding constraint.** Below mA ≈ 45 the
  below-Z norm envelopes are 7-19%, driven by MHc115 (15 → 27 → 42) and
  MHc130 (15 → 30 → 55). No functional form fixes this — every basis
  tried fails there identically. One extra MC point in each of those gaps
  would do more than any further model work.
- **No mHc interpolation.** The surfaces are better-constrained models AT
  the seven measured mHc; predicting an unmeasured study from the other
  six is a 4% median / 18% p90 yield error.
- **Closure pulls are uncalibrated** — the assumed 1% Run2 G error is far
  below the ~5% surface residual. The exported envelopes use relative
  residuals and are unaffected.

## Troubleshooting

- `plot_score`/comparison jobs held with "cgroup memory limit": bump
  `RequestMemory` (4096) and `condor_release` — the DAG generator already
  requests 4 GB for plot_score.
- Template-stage numeric drift at ≤1e-12 relative on Run3 categories is
  known cross-worker Minuit/numpy noise (see `docs/REPRODUCTION.md`).
- Sanity check: `python3 -m compileall -q python/` and `bash -n` over
  `scripts/ automize/`.
