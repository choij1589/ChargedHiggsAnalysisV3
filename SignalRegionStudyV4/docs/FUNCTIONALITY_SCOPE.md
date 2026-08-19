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
- Limit collection and plotting, in both `BR` and `xsec` units
  (`python/collectLimits.py`, `python/plotLimits.py`, and the
  (mA, mHc) map `python/plotLimits2D.py`)
- interp-signal template production over the scan grid
  (`makeBinnedTemplates.py --signal-source interp-signal`,
  `python/param_signal.py`, `automize/interpTemplates.sh`)
- Goodness-of-fit and impacts, per interp-signal group seed
  (`scripts/runGoF.sh`, `scripts/runImpacts.sh`,
  `python/filterImpacts.py`, `python/plotGoFPValues.py`,
  `automize/interpGofImpacts.sh`; the scripts also accept
  `--signal-source mc-signal` ad hoc)
- FitDiagnostics + prefit/postfit + pulls per interp-signal group seed,
  both methods (`automize/interpFitDiag.sh`; `plotPostfitMass.py
  --signal-source interp-signal` refills the parametric signal from the
  param_signal DCB sidecar), with stitched per-mHc summaries
  (`python/plotPostfitSummary.py`, ported from V3)
- ParticleNet interp-signal score plots per seed
  (`plotParticleNetScore.py --signal-source interp-signal`,
  `automize/pnetScorePlots.sh`, `python/collectPnetScorePlots.py`)
- Template-point artifact bundle (`python/collectTemplatePlots.py`) —
  promotes GoF, impacts (full/filtered/summary), nuisance pulls
  (full/filtered), prefit/postfit mass plots, the ParticleNet score
  panels and the point's per-channel limits of the curated representative
  mass points out of the gitignored template dirs into
  `results/templates/{method}/{masspoint}/`
- Per-channel limit plots: `plotLimits.py` / `plotLimits2D.py` are run for
  `SR1E2Mu` and `SR3Mu` as well as `Combined`, each reading its own
  channel's collected JSON
- Observed local significance on selected points
  (`scripts/runSignificance.sh`, `automize/significance.sh`,
  `python/collectSignificance.py`) — uncapped `combine -M Significance`,
  All × 3 channels, collected into `results/json/significance.*.json`
- Look-elsewhere effect / global significance
  (`automize/significance.sh --grid|--pnet-grid`,
  `python/estimateLEE.py`, `python/plotLEE.py`) — the **asymptotic
  Gross-Vitells** trials estimate from the upcrossing rate of the observed
  Z(mA) scan, scoped to the mA scan alone per (arm, channel, mHc). This is
  NOT V3's toy campaign, which is still not ported and would not be the
  right object on a lattice sampled at the mass resolution; see `docs/LEE.md`
- Grouped uncertainty breakdown of `sigma(r)`
  (`scripts/runBreakdown.sh`, `automize/breakdown.sh`,
  `python/nuisanceGroups.py`, `python/collectBreakdown.py`,
  `python/plotBreakdown.py`) — cumulative `MultiDimFit
  --freezeNuisanceGroups` scans at the curated template points, groups
  defined in `configs/nuisance_groups.json`. V3 has an unrun
  implementation of this; V4 establishes the number and derives the scan
  range per point rather than fixing it. See `docs/BREAKDOWN.md`
- Paper figures, ported from V3 2026-08-18 (`python/plotPaperLRModified.py`
  — SR and TTZ CR LR_modified panels plus the standalone legend;
  `python/plotPaperTemplates.py` — prefit/B-only/S+B mass templates per
  Run-period category; `python/plotPaperPostfitSummary.py` — b-only mA
  summary in the three paper mA regions). Wording, colours and legend
  machinery are defined once in `plotPaperLRModified.py` and imported by
  the other two, so the figure sets cannot drift apart. Output:
  `results/plots/paper/`
- Reproduction comparison against V3 (`python/compareToV3.py`,
  `scripts/compare_wrapper.sh`)
- Methods: `Baseline` and `ParticleNet`
- Mass points: `configs/masspoints.json` keys `baseline` (78), `particlenet`
  (22), `limits` (curated plotting subset)

## Not In V4

These exist only in older modules and are intentionally not ported. V4 code
and docs make no reference to them:

- HybridNew, signal injection / bias tests
- Look-elsewhere-effect (LEE) **toys** — V4 corrects for trials
  asymptotically instead (see Active above); the toy campaign itself is not
  ported
- TTZ control-region templates and GoF
- Cut-and-count (CnC) datacards
- PTOptimized method
- Interpolated signal templates (V3-style double-Gaussian central-only;
  superseded by the planned V4 interpolation work)
- Partial-unblind blinding mode, `uniform` binning, `preserve_shape`
  nuisance handling
- Template/datacard transfer helpers (rsync, copyDatacards)

## Layout Contract

- Sample layout (shared, the V4 default):
  `samples/{era}/SR1E2Mu/` and `samples/{era}/SR3Mu_{lowM,highM}/` hold the
  mass-independent backgrounds/nonprompt/data once, plus every signal as
  `{masspoint}.root` (SR3Mu: in both pairing variants, for interpolation).
  ParticleNet keeps per-masspoint dirs `samples/{era}/{channel}/{masspoint}/`
  (per-masspoint scores, NoHistMode skims, TTZ2E1Mu).
  `highM` ⇔ `mHc >= 100 && mA >= 60` (the SR3Mu pairing rule).

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
