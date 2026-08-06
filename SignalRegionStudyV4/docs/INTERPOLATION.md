# INTERPOLATION.md — Parametric Signal Templates via DCB Interpolation

Purpose, method, and status of the mA-interpolation development
(B2G-25-013 next-generation signal modeling). Study code lives in
`test/interpolation/` (untracked scratch by convention); this document is
the durable record.

## Purpose

The analysis needs exclusion limits as a continuous function of mA, but
signal MC exists only at discrete mass points. The goal is to build
**binned signal templates at arbitrary mA (fixed mHc) by integrating a
DCB fit function** whose parameters are interpolated between the MC
points:

1. Fit the production DCB (double-sided `RooCrystalBall`) to signal MC at
   each available mA, per merged run-period category
   (SR1E2Mu/SR3Mu × Run2/Run3).
2. Parametrize each DCB parameter (`x0, sigmaL, sigmaR, alphaL, nL,
   alphaR, nR`) as a low-order polynomial of mA.
3. For any target mA, evaluate the polynomials, build the DCB, and
   integrate it over the adaptive bin edges to produce the binned signal
   template. Backgrounds stay binned as today; the interpolated point
   plugs into the existing chain behind a `srspaths.template_dir` method
   segment (see CLAUDE.md "Future Phases").

The current feasibility test uses **mHc = 145 GeV** (12 MC points).

## Method

### Fit / validation split (fixed by design)

| role | mA points |
|------|-----------|
| polynomial fit inputs | 35, 60, 80, 90, 100, 140 |
| held-out validation   | 45, 85, 92, 95, 120 (interpolation), **15 (extrapolation)** |

MA15 is validation-only on purpose: the DCB fit window is floored at
12 GeV (`fit_dcb`), clipping the left tail, and it lies below the fit
range — it probes extrapolation and is not a pass/fail gate.

### Fits with uncertainties

`test/interpolation/dcb_fit_utils.py::fit_dcb_with_errors` is a verbatim
mirror of the production two-stage fit
(`python/makeBinnedTemplates.py::fit_dcb`) that additionally captures the
`RooFitResult`: per-parameter errors, Minuit status, covQual, at-bound
flags, dataset stats. The production function is untouched (datacard
bytes are frozen by the reproduction contract). Central values reproduce
production fits exactly.

Fit quality gate: `status != 0`, `0 <= covQual < 2`, a shape parameter
pinned at its bound, or a zero/NaN error ⇒ excluded from polynomial fits
and pulls (always still recorded). Note `covQual = -1` is common under
`SumW2Error(True)` and is not treated as a failure.

### Polynomial parametrization

Error-weighted least squares (`numpy.polyfit`, `w = 1/err`,
`cov="unscaled"`) over the fit points only; order selection walks upward
and accepts a higher order only on an F-test p < 0.05:

| parameter | orders tried | expectation |
|-----------|--------------|-------------|
| x0 | 1, 2 | ≈ mA, slope ≈ 1 |
| sigmaL, sigmaR | 1, 2 | ~linear growth (~1% of mA) |
| alphaL, alphaR | 0, 1 | slowly varying, 1.4–1.7 |
| nL, nR | 0, 1 | noisy; anti-correlated with alpha |

The alpha–n degeneracy (earlier tail transition ⇔ slower fall-off) makes
the n values scatter without the shape changing much; the shape-level
closure χ² is the meaningful tail metric, not the n pulls.

### Closure

At each held-out mA: evaluate polynomials → predicted DCB; compare with
(a) per-parameter pulls against the direct fit, (b) overlay canvas
MC + direct DCB + interpolated DCB, (c) χ²/ndf of each shape vs the MC
histogram (pdf normalized to the MC integral in the fit window).

Success criteria: |pull| < 2 for x0/sigmaL/sigmaR at interpolation
points (tails ≤ ~3); interpolated-vs-MC χ²/ndf ≲ 1.5× the direct fit's;
polynomial χ²/ndf ∈ ~[0.3, 3].

### SR3Mu pairing variants

The shared sample layout stores every signal in **both** SR3Mu pairing
variants (`SR3Mu_lowM`, `SR3Mu_highM`) precisely because interpolation
needs them: at mHc = 145 the production pairing rule
(`srspaths.pairing_variant`: highM iff mHc ≥ 100 and mA ≥ 60) switches
variants at mA = 60, which would put a shape discontinuity into a single
parameter-vs-mA scan. The plan is to parametrize each variant separately
across the full mA range and let the production rule pick the variant
when an interpolated template is built.

## Test chain (`test/interpolation/`)

```
interp_config.py     constants (mass split, poly orders, quality cuts)
dcb_fit_utils.py     fit_dcb_with_errors + canvas parameter labels
verify_samples.py    pnfs integrity gate (open every file; anti-truncation)
fit_all_points.py    stage 1: fits -> results/dcb_fits.json + plots/fits/
fit_polynomials.py   stage 2: polynomials -> results/polynomials.json + plots/params/
closure_test.py      stage 3: closure -> results/closure.json + plots/closure/
```

Stage 1 supports incremental `--masspoints/--categories` reruns (merges
into the existing JSON). Signal-fit canvases carry the fitted parameters
± errors; closure canvases carry a direct-vs-interpolated parameter table.

## Status (2026-08-07)

**Done before the shared-layout refactor** (old per-masspoint layout,
every file verified on pnfs at the time):

- 5 mass points fitted — MA15, 35, 45, 60, 95 × 4 categories = 20 fits,
  **all quality good** (covQual 2–3), in
  `test/interpolation/results/dcb_fits.json` (+ labeled canvases).
- Early trends (strongly supportive of the approach):
  - `x0` on nominal mA within 0.02 GeV everywhere;
  - `sigmaL/R` ≈ 1% of mA, cleanly linear (0.14 → 0.37 → 0.49 GeV at
    15/35/45);
  - `alphaL/R` stable at 1.4–1.7; `nL/R` noisy as expected.
- MA100 had 2 transient node failures (input-open, xrdcp destination
  timeout); its rescue DAG was mid-flight when work was paused.

**Invalidated by the refactor**: the pnfs sample area was rebuilt into
the shared layout; the old-layout MHc145 sample dirs no longer exist.
The recorded fits stay valid as results, but nothing MHc145 can be
re-derived or continued until the signals are re-preprocessed.

**Adaptation needed in `test/interpolation/` before resuming** (scripts
still assume the pre-refactor contracts):

1. `srspaths.sample_dir` now takes `(era, channel, masspoint, method)`
   and resolves Baseline to the shared dirs — update `fit_all_points.py`
   and `closure_test.py` call sites (method `"Baseline"`), and decide the
   SR3Mu variant handling per the pairing plan above.
2. `verify_samples.py` must check the shared layout (signal files
   `{mp}.root` inside `SR1E2Mu` / `SR3Mu_{lowM,highM}`) instead of
   per-masspoint dirs.
3. `link_samples.sh` is obsolete (`samples/` is already a pnfs symlink).
4. Preprocessing driver interface changed:
   `./automize/preprocess.sh [--masspoint MP] [--skip-backgrounds] ...`
   — an interpolation mass point needs only its 16 shared-signal nodes;
   shared backgrounds are already in place.

**Resume checklist**:

1. Re-preprocess the 12 MHc145 signals (signal-only DAGs, new layout).
2. Verify with the adapted `verify_samples.py` (open every file — the
   concurrent-xrdcp truncation hazard is real; see
   `pnfs` history in the repo memory / docs/SAMPLES.md).
3. Re-run stage 1 for all 12 points (re-fit the 5 known points as a
   cross-check against the recorded values — same fit config, same MC
   events ⇒ identical parameters expected).
4. Run stage 2 (polynomials) and stage 3 (closure); judge against the
   success criteria above.
5. If closure holds: promote — template producer that integrates the
   interpolated DCB over adaptive bins, behind a new method segment
   (`test/interpolation` code graduates into `python/` at that point).
