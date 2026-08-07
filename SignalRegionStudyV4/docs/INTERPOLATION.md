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
   each available mA, per merged run-period category — 6 in total:
   SR1E2Mu / SR3Mu_lowM / SR3Mu_highM × Run2 / Run3 (SR3Mu pairing
   variants are interpolated separately, see below).
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

### SR3Mu pairing variants (decision)

**lowM and highM are interpolated separately.** The shared sample layout
stores every signal in **both** SR3Mu pairing variants
(`SR3Mu_lowM`, `SR3Mu_highM`) — even for mass points whose production
templates use only one — precisely because interpolation needs both: at
mHc = 145 the production pairing rule (`srspaths.pairing_variant`:
highM iff mHc ≥ 100 and mA ≥ 60) switches variants at mA = 60, which
would put a shape discontinuity into a single parameter-vs-mA scan.

The study therefore fits **6 categories** over the full mA range:

```
SR1E2Mu      × {Run2, Run3}
SR3Mu_lowM   × {Run2, Run3}
SR3Mu_highM  × {Run2, Run3}
```

Each variant gets its own parameter polynomials; when an interpolated
template is built for production, the pairing rule picks which variant's
polynomials to use at that (mHc, mA).

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

**Adaptations done (2026-08-07)** — `test/interpolation/` now targets the
shared layout: the 6-category scheme above
(`interp_config.STUDY_CHANNELS`), signal paths via
`interp_config.signal_path` (built on `srspaths.shared_channel_dirname`),
`verify_samples.py` rewritten for shared dirs (opens every signal file;
shared backgrounds existence-checked), `link_samples.sh` removed
(obsolete — `samples/` is a pnfs symlink). Pre-refactor fits archived as
`results/dcb_fits.prerefactor.json` (old `SR3Mu_*` categories map to
`SR3Mu_lowM_*` for MA15/35/45 and `SR3Mu_highM_*` for MA60/95 — the
production pairing at the time of that preprocessing).

**In flight**: the 12 signal-only DAGs were submitted 2026-08-07
(`./automize/preprocess.sh --masspoint MHc145_MA{X} --skip-backgrounds`;
16 nodes each, 40 for the ParticleNet points 85/90/92/95).

**Remaining steps**:

1. Verify with `verify_samples.py --all` (open every signal file — the
   concurrent-xrdcp truncation hazard is real; see docs/SAMPLES.md);
   rescue failed DAGs (`condor_submit_dag dag.dag` picks up rescue001).
2. Run stage 1 for all 12 points × 6 categories (72 fits); cross-check
   the archived pre-refactor values (same fit config, same MC events ⇒
   identical parameters expected for the mapped categories).
3. Run stage 2 (polynomials) and stage 3 (closure); judge against the
   success criteria above.
4. If closure holds: promote — template producer that integrates the
   interpolated DCB over adaptive bins, behind a new method segment
   (`test/interpolation` code graduates into `python/` at that point).
