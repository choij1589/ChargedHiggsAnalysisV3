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

### Fit / validation split

| role | mA points |
|------|-----------|
| polynomial fit inputs | **15**, 35, 60, 80, 90, 100, 140 |
| held-out validation   | 45, 85, 92, 95, 120 (15 doubles as an in-sample check) |

MA15 was initially validation-only (extrapolation probe): with fit
points starting at 35, the extrapolated shape at 15 failed badly
(χ²/ndf up to ~600 with mixed sigma orders; still 5–46 after fixing
sigmas to pol2). Moving MA15 into the fit set (2026-08-07) fixed the low
end — χ²/ndf ≤ 7.5 in every category where MA15 is physical — at the
cost of a mild systematic degradation of the lowM tails at mid/high mA
(nL stretches to cover MA15's bias). Residual caveat: the DCB fit window
is floored at 12 GeV (`fit_dcb`), clipping MA15's left tail; that bias
lives in the direct MA15 fit parameters themselves, so no polynomial
choice removes it (lowering the floor in a test-local fit would, at the
cost of departing from the production fit configuration). SR3Mu_highM
never fits MA15 (no physical peak — the quality gate drops it), so its
polynomials are unaffected.

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

| parameter | orders | expectation |
|-----------|--------|-------------|
| x0 | 1, 2 (F-test) | ≈ mA, slope ≈ 1 |
| sigmaL, sigmaR | **2 (fixed)** | ~1% of mA, mild curvature |
| alphaL, alphaR | 1, 2 (F-test) | slowly varying, 1.4–1.7 |
| nL, nR | 1, 2 (F-test) | noisy; anti-correlated with alpha |

Order-choice history (2026-08-07): tails originally 0/1 — failed closure
at SR3Mu_lowM MA45 and SR3Mu_highM MA92/95 (tail parameters trend
strongly with mA in the SR3Mu variants); raised to 1/2, which resolved
MA45 and halved the remaining excesses. Sigmas were then **fixed to a
common pol2** (both widths arise from the same two-muon resolution
convolution — no reason for different functional forms per side, and the
F-test's per-side pol1/pol2 split was selection noise): improved sigma
fit χ² everywhere and, as a bonus, tamed the low-mA extrapolation
(MA15 χ²/ndf e.g. 59→5 in SR1E2Mu_Run2).

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

**Full chain ran 2026-08-07** (shared layout, 6 categories):

- Preprocessing: all 12 points verified on pnfs. One blocker outside this
  repo: the **standard** skim
  `Run1E2Mu_RunSyst_RunTheoryUnc/2023/TTToHcToWAToMuMu-MHc145_MA100.root`
  is corrupt (truncated, unreadable) ⇒ SR1E2Mu_Run3 has no MA100 fit
  point until SKNano regenerates it. Separately, 41 NoHistMode files are
  deficient (affects ParticleNet production only — see docs/SAMPLES.md).
- Stage 1: 71/72 fits; all good except SR3Mu_highM at MA15 (no physical
  peak — highM pairing picks the combinatoric dimuon at low mA;
  structural, not a fit problem). Pre-refactor cross-check: parameters
  agree to ≤3e-4 (the one exception traced to the deficient MA95
  NoHistMode input of the old layout).
- Stage 2: x0 pol1 with slope ≈ 1 everywhere (χ²/ndf 1–6); sigmas
  common pol2 (χ²/ndf 0.5–7.6).
- Stage 3 closure, final configuration (MA15 in the fit set, pol2
  sigmas, pol1/pol2 tails):
  - **SR1E2Mu: passes** — interp χ²/ndf 1.6–8.2 vs direct 1.4–6.6
    across 15–120; at several points the interpolated shape matches MC
    *better* than the direct fit (e.g. Run3 MA45: 5.2 vs 6.6).
  - **MA15: usable** — 2.2/2.7 (SR1E2Mu), 7.5/5.8 (lowM Run2/Run3);
    the lowM residual is the 12 GeV window-floor bias (see above).
  - **SR3Mu_lowM / highM: mostly passes** — remaining hot spots are
    purely tail-degeneracy driven: alphaR at highM MA92/95
    (interp/direct ≈ 1.7–3.0, pulls −3…−6) and nL in lowM at 92–120
    (≈ 2–3×, pulls up to ±8). Next lever: constrain the α–n degeneracy
    (fix n per category, interpolate α alone), not higher orders.

**Next**: decide the tail-degeneracy treatment for SR3Mu (fix n per
category, or add fit points near 92–95), then promote — a template
producer that integrates the interpolated DCB over adaptive bins behind a
new method segment (`test/interpolation` code graduates into `python/`
at that point).
