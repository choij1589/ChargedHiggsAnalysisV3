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

**Scope: this is a Baseline study.** All fits read the shared-layout
signals built from the standard skims — including for ParticleNet-trained
mass points, which enter purely as Baseline signal shapes. A dedicated
ParticleNet interpolation study follows after the Baseline one is done
(the PN per-masspoint sample dirs produced alongside the shared signals
are unused here but will serve that study).

Feasibility tests: **mHc = 145 GeV** (12 MC points) and
**mHc = 160 GeV** (23 MC points).

## Method

### Fit / validation split

| mHc | role | mA points |
|-----|------|-----------|
| 145 | polynomial fit inputs | **15**, 35, 60, 80, 90, 100, 140 |
| 145 | held-out validation   | 45, 85, 92, 95, 120 (15 doubles as an in-sample check) |
| 160 | polynomial fit inputs | 15, 20, 30, 40, 50, 70, 90, 115, 135, 155 |
| 160 | closure               | all 23 points (fit points as in-sample checks) |

Per-study definitions live in `interp_config.STUDIES`; every stage script
takes `--mhc {145,160}` and writes to `results/MHc{X}/`, `plots/MHc{X}/`.

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

**MHc160 study ran 2026-08-07** (23 points, fit set
15/20/30/40/50/70/90/115/135/155, closure on all points; Baseline
samples throughout):

- Preprocessing: all 23 points verified on pnfs; 138/138 fits (1
  expected bad: SR3Mu_highM MA15). PN-node failures during preprocessing
  were all irrelevant to Baseline: the DAG generator adds TTZ2E1Mu nodes
  for PN points outside preprocess.py's 83 < mA < 100 window
  (MA70/80/105/115 — driver inconsistency worth fixing), and the
  MA98 score branches are missing from Run3Mu/2017 and Run3-era
  TTZ NoHistMode inputs (noted for the future PN study).
- Closure (median interp/direct χ² ratio per category):
  - **SR1E2Mu: passes across the full 15–155 range** — median ratio
    1.03 (Run2) / 1.10 (Run3); worst held-out point 3.8 vs 1.9.
  - **SR3Mu_highM: median ratio 1.3–1.4**, but nR/alphaR-driven
    failures cluster at mA 95–105 (χ²/ndf up to 32 vs 6) — same tail
    pattern as MHc145's 92/95 hot spot, in the region where the W*
    softens (mA → mHc − 50).
  - **SR3Mu_lowM: the weak spot** — median ratio 2.2–2.6, nL-driven,
    worst at high mA (120/125: up to 35 vs 2.6) and even at some
    in-sample fit points (MA115: 35.6 vs 3.4): a pol2 cannot track the
    lowM tail evolution over 15–155.

**Range-restricted SR3Mu parametrization (2026-08-07, adopted)**:
each variant is fit and closure-tested only inside its
production-relevant mA range — `interp_config.CHANNEL_MA_RANGES`:
lowM ∈ [15, 70], highM ∈ [50, 155] (overlap covers the mA = 60
production boundary; SR1E2Mu unrestricted). Out-of-range points are
excluded entirely from that variant's fits, plots, and closure. With few
in-range points the order ladder falls back to the highest feasible
order (MHc145 lowM has 3 fit points → pol1).

Results of the restriction:

- **SR3Mu_lowM: solved.** Median interp/direct ratio 0.93–1.25 across
  both mHc studies (was 2.2–2.6 full-range at MHc160); worst in-range
  point 5.2 vs 4.2. The nL polynomial χ²/ndf dropped from 304/7 to 7.4/3
  (MHc160 Run2).
- **SR3Mu_highM: NOT fixed by range restriction** — median ratios
  2.2–3.4, still nR/alphaR-driven, concentrated at mA ≈ 92–105 (both
  mHc) and 120–125 (MHc160). The highM tail parameters genuinely vary
  non-polynomially inside [50, 155]: nR polynomial χ²/ndf stays at
  ~120–130/3–4. This is the α–n degeneracy, not an order or range
  problem.

**Fixed-n variant (2026-08-07, adopted for ALL categories)**: the α–n
degeneracy made the floating tail parameters valley-hop between mass
points, defeating any smooth parametrization (highM nR polynomial
χ²/ndf ~120 even range-restricted). Fix: freeze nL/nR per category to
the **median of the good floating-n in-range fits**
(`interp_config.fixed_n_values`), refit every mass point with only
x0/σL/σR/αL/αR floating (`dcb_fit_utils.fit_dcb_fixed_n`; 5 free
parameters), and interpolate those 5 (the frozen n enter the JSON as
constant order-0 "polynomials"). Run with `--fixed-n` on all three
stages; outputs under `results/MHc{X}/fixedn/`, `plots/MHc{X}/fixedn/`.

The derived n are remarkably consistent between mHc 145 and 160
(SR1E2Mu nL≈2.0 nR≈6–8; highM nL≈1.9 nR≈0.65; lowM nL≈1–1.3 nR≈5–6),
confirming n is a stable per-category property. With n frozen the alphas
become genuinely smooth functions of mA — highM alphaR rises to a
maximum near mA≈115 and falls again, which needs pol3 (allowed for
alphas; F-test selects it where the point count supports it). All
210 fixed-n fits converge with quality good (including highM MA15,
unstable when n floated).

**Final closure, fixed-n + range restriction (held-out points, median
interp/direct χ² ratio | worst point)**:

| category | MHc145 | MHc160 |
|---|---|---|
| SR1E2Mu_Run2 | 1.00 \| 2.4 vs 1.9 | 0.97 \| 3.1 vs 2.7 |
| SR1E2Mu_Run3 | 1.11 \| 5.5 vs 4.7 | 0.98 \| 2.1 vs 1.9 |
| SR3Mu_highM_Run2 | 1.04 \| 6.2 vs 5.7 | 1.06 \| 11.2 vs 7.8 |
| SR3Mu_highM_Run3 | 1.08 \| 3.6 vs 2.6 | 1.03 \| 2.8 vs 2.0 |
| SR3Mu_lowM_Run2 | 1.03 \| 4.5 vs 4.4 | 1.00 \| 5.5 vs 5.0 |
| SR3Mu_lowM_Run3 | 0.87 \| 2.1 vs 2.5 | 0.99 \| 4.7 vs 4.3 |

No held-out point exceeds 1.5× the direct fit's χ²/ndf. The residual
absolute χ² (e.g. highM 95: 13 vs 12) is the DCB model's own MC
mismatch, not interpolation error. Note the α *parameter pulls* vs the
individually-refit alphas can still be large (−5…−12) — with n frozen
the per-point α rides the remnant micro-degeneracy — but the
shape-level closure is the meaningful metric and it passes.

**Adopted method (summary)**: per category — fixed nL/nR (median),
range-restricted mass points (lowM [15,70], highM [50,155], SR1E2Mu
unrestricted), polynomials: x0 pol1/2, σL/σR common pol2, αL/αR
pol1/2/3 (F-test); interpolated template = polynomial x0/σ/α + frozen n.

**Full-range check with fixed n (2026-08-07, `--full-range` variant)**:
could the SR3Mu variants share one [15,155] parametrization? highM: yes
— full-range closure is equal or better (median ratios 1.01–1.06), its
fixed-n parameters evolve smoothly over the whole range, so the [50,155]
restriction is optional there. lowM: no — including the high-mA points
degrades the production-relevant low-mA closure (MHc145 Run2 in-range
median 1.03 → 1.51; alphaL polynomial χ²/ndf up to 424/7): the min-mass
pairing's shape genuinely changes character at high mA. Keep lowM
restricted to [15,70].

**Next**: promote — a template producer that integrates the
interpolated fixed-n DCB over adaptive bins behind a new method segment
(`test/interpolation` code graduates into `python/` at that point), plus
the yield (signal-efficiency) interpolation which this shape study has
not covered.
