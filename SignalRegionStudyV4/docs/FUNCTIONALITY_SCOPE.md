# FUNCTIONALITY_SCOPE.md — What SignalRegionStudyV4 Owns

SignalRegionStudyV4 is a self-contained, simplified port of the production
limit-setting chain. It carries only the workflows needed to go from
SKNanoOutput skims to asymptotic limits, and is the base for the
next-generation work (parametric signal models, mA interpolation).

Self-containment rule: V4 has zero code references, imports, or symlinks into
any other SignalRegionStudy directory. Everything V4 runs lives under V4.
The single exception is `python/compareToV3.py`, a validation-only comparator
that reads frozen V3 *output artifacts* through an explicit `--v3-dir`
argument; it is not part of any production workflow.

## Active In V4

- Preprocessing of signal/background/data skims into flat per-process trees
  with systematic variations (`python/preprocess.py`, `automize/preprocess.sh`)
- Run-period component binned templates with DCB signal fits and adaptive
  binning (`python/makeBinnedTemplates.py`)
- Run-period template merging (`python/mergeRunPeriodTemplates.py`)
- Template validation (`python/checkTemplates.py`,
  `python/validateRunPeriodTemplates.py`)
- Multi-category datacards (`python/printDatacard.py`)
- Asymptotic limits (`scripts/runAsymptotic.sh`)
- FitDiagnostics, postfit mass plots, nuisance pulls
  (`scripts/runFitDiagnostics.sh`, `python/plotPostfitMass.py`,
  `scripts/runPullPlots.sh`)
- ParticleNet score plots (`python/plotParticleNetScore.py`)
- Limit collection and plotting (`python/collectLimits.py`,
  `python/plotLimits.py`)
- Reproduction comparison against V3 (`python/compareToV3.py`,
  `scripts/compare_wrapper.sh`)
- Methods: `Baseline` and `ParticleNet`
- Mass points: `configs/masspoints.json` keys `baseline` (78), `particlenet`
  (22), `limits` (curated plotting subset)

## Not In V4

These exist only in older modules and are intentionally not ported. V4 code
and docs make no reference to them:

- Goodness-of-Fit tests, impacts, HybridNew, signal injection / bias tests
- Look-elsewhere-effect (LEE) toys
- TTZ control-region templates and GoF
- Paper-figure scripts
- Cut-and-count (CnC) datacards
- PTOptimized method
- Interpolated signal templates (V3-style double-Gaussian central-only;
  superseded by the planned V4 interpolation work)
- Partial-unblind blinding mode, `uniform` binning, `preserve_shape`
  nuisance handling
- Template/datacard transfer helpers (rsync, copyDatacards)

## Layout Contract

- Template layout: `templates/{masspoint}/{method}/{era}/{channel}/` —
  mass point first, no binning-suffix level.
- The only binning scheme is `extended` (adaptive, 15 to 5 core bins driven
  by the DCB fit resolution); it needs no name in the path.
- Unblind (real data) is the default. A blinded (Asimov) run writes
  `{method}_blind` as the method segment (e.g. `Baseline_blind`), so blind
  and unblind artifacts can never collide.
- Filenames carry no binning/unblind tokens:
  `higgsCombine.{mp}.{method}.AsymptoticLimits.mH120.root`,
  `limits.{era}[.{channel}].Asymptotic.{method}.json`.
- Path construction exists in exactly two places:
  `python/srspaths.py` and `scripts/env.sh`. Do not add a third.
