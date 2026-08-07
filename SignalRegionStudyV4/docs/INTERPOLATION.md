# INTERPOLATION.md — Parametric Signal Templates via DCB Interpolation

Purpose, adopted method, results, and decision history of the
mA-interpolation development (B2G-25-013 next-generation signal
modeling). Study code lives in `test/interpolation/` (untracked scratch
by convention); this document is the durable record.

## Purpose

The analysis needs exclusion limits as a continuous function of mA, but
signal MC exists only at discrete mass points. Goal: **binned signal
templates at arbitrary mA (fixed mHc) by integrating a fitted signal
model** whose parameters are interpolated between the MC points.
Backgrounds stay binned as today; the interpolated point plugs into the
existing chain behind a `srspaths.template_dir` method segment (see
CLAUDE.md "Future Phases").

**Scope: this is a Baseline study.** All fits read the shared-layout
signals built from the standard skims — including for ParticleNet-trained
mass points. A dedicated ParticleNet interpolation study follows after
the Baseline one.

Feasibility studies: **mHc = 145 GeV** (12 MC points) and
**mHc = 160 GeV** (23 MC points).

## Adopted method

Per merged run-period category — 6 in total,
{SR1E2Mu, SR3Mu_lowM, SR3Mu_highM} × {Run2, Run3}:

**Signal model** (`test/interpolation/dcb_fit_utils.py`):

- SR1E2Mu: pure double-sided Crystal Ball (DCB) — one dimuon pairing,
  no combinatorics.
- SR3Mu_lowM / SR3Mu_highM:
  `S(m) = fsig · DCB(m) + (1 − fsig) · Chebychev₂(m; c1, c2)` — the
  pairing picks the wrong (combinatoric) dimuon in a mass-dependent
  fraction of events, producing a near-flat pedestal under the peak that
  only a shape that *can be flat* describes.
- **Frozen tails**: nL/nR fixed per category to the median of the good
  floating-n fits (two-pass: floating-n campaign first, then the
  frozen-n refit). This breaks the α–n degeneracy — with n floating,
  each mass point lands elsewhere in the α–n valley and no smooth
  parametrization can follow.
- **Background drop**: points where fsig >
  `FSIG_DROP_THRESHOLD` (0.995) refit as pure DCB — unconstrained
  background parameters otherwise inflate every error and de-weight the
  low-mA anchors of the parametrizations. Such points anchor the fsig
  logistic at 1.0 ± `FSIG_ANCHOR_ERROR` (0.002); their c1/c2 carry zero
  error and drop out. The same threshold governs the closure-side model
  build.
- Two-stage fit structure (wide pre-fit sets a ±10σ window) identical to
  the production `fit_dcb`; `fit_dcb_with_errors` remains a verbatim
  mirror of production (frozen reproduction contract),
  `fit_dcb_bkg` is the study's generalized fit.

**Parametrization vs mA** (`fit_polynomials.py`; forms/orders in
`interp_config.POLY_ORDERS`):

| parameter | form |
|---|---|
| x0 | pol1 (fixed) |
| sigmaL, sigmaR | pol2 (fixed; common form — both widths arise from the same two-muon resolution convolution) |
| alphaL, alphaR | pol2 (fixed) |
| nL, nR | frozen per category (constant records) |
| fsig | 4-parameter logistic `base + amp/(1+exp(−(m−m0)/w))` — sigmoid turn-on/off around the pairing boundary; polynomial fallback below 5 points |
| c1, c2 | pol2 (fixed) |

Error-weighted fits over the study's fit points (good-quality fits
only); **uniform mA range** for both SR3Mu variants — production uses
highM only above the mA = 60 pairing boundary, but at lower mHc (e.g.
MHc130) that region holds too few MC points, so every mass point
constrains both variants. Single-entry order lists fix the order;
multi-entry lists engage an F-test ladder (step up accepted at p < 0.05).

**Mass-point splits** (`interp_config.STUDIES`):

| mHc | fit points (mA) | closure points |
|---|---|---|
| 145 | 15, 35, 60, 80, 90, 100, 140 | 15, 45, 85, 92, 95, 120 |
| 160 | 15, 20, 30, 40, 50, 70, 90, 115, 135, 155 | all 23 points |

(Closure points also in the fit set act as in-sample checks. MA15's fit
window is floored at 12 GeV — clipped left tail; it was moved *into*
the fit set because extrapolating below the lowest fit point fails.)

**Chain** (variant `cheb_fixedn`; one condor job per mass point for
stage 1): see `test/interpolation/README.md` for the exact commands.
Quality gate per fit: Minuit status 0, covQual not in [0, 2), no shape
parameter at a bound, positive errors (frozen parameters exempt);
covQual = −1 (common under SumW2Error) is not a failure. Bad fits are
excluded from parametrizations and pulls, always logged.

## Results (final configuration)

Closure = interpolated shape vs MC, χ²/ndf, compared to the per-point
direct fit's:

- **Direct-fit model quality**: χ²/ndf medians 1.4–2.8 per category
  (max ≲ 5) across both mHc — the Chebychev background flattened the
  lowM plateau region that a single DCB (ratio swings ±20–50%) and an
  exponential could not describe.
- **Interpolation closure**: interp medians 1.7–4.2 vs direct 1.4–2.8
  over all 12 category × mHc combinations; the interpolated shapes are
  statistically close to per-point fits everywhere in the studied
  ranges, including MA15 (in-sample) and the dense highM 90–105 region.
- SR3Mu_highM at MA15–25 has no physical peak (the high-mass pairing
  picks the combinatoric dimuon at low mA) — those direct fits fail the
  quality gate by construction and are excluded; production never uses
  highM there.

**Hand-off artifact**: `results/MHc{X}/cheb_fixedn/polynomials.json`
(+ frozen n in the same variant's `dcb_fits.json` meta). Remaining
steps to production: yield (signal-efficiency) interpolation, then a
template producer integrating the interpolated model over adaptive bins
(`test/interpolation` code graduates into `python/` at that point).

## Sample production status (2026-08-07)

- MHc145 (12 points) and MHc160 (23 points) signals preprocessed into
  the shared layout and verified file-by-file on pnfs
  (`verify_samples.py`, the anti-truncation gate — concurrent-xrdcp
  truncation is a real, observed failure mode).
- One **corrupt standard skim** blocks SR1E2Mu/2023 for MHc145_MA100:
  `Run1E2Mu_RunSyst_RunTheoryUnc/2023/TTToHcToWAToMuMu-MHc145_MA100.root`
  (truncated at 9 MB; needs SKNano regeneration). That category-point is
  skipped explicitly.
- 41 deficient NoHistMode files (ParticleNet inputs only — does not
  affect this Baseline study): see docs/SAMPLES.md.

## Decision history (chronological; each superseded by the next where applicable)

| decision | key evidence |
|---|---|
| DCB parameters interpolable in principle; x0 ≈ mA (pol1, slope 1), σ ≈ 1% of mA | first MHc145 pass, 20 fits |
| MA15 moved from validation into the fit set | extrapolation below 35 GeV failed (χ²/ndf up to ~600) |
| Tail orders raised 0/1 → 1/2, then σL/σR fixed to common pol2 | lowM MA45 / highM MA92–95 closure failures; F-test's per-side pol1/pol2 split was selection noise; pol2 σ also tamed the MA15 edge |
| SR3Mu range restriction (lowM [15,70], highM [50,155]) tried, then **reverted to uniform range** | restriction fixed lowM tails but starves low-mHc highM of points (MHc130 use case); uniform range viable once fsig got its logistic form |
| **Fixed n per category** (median of floating fits) | α–n valley-hopping defeated every parametrization (nR polynomial χ²/ndf ~120); with n frozen, α becomes smooth; per-point fit cost only ~0.2 in χ²/ndf. Simultaneous-fit n cross-checked: agrees with medians (nL within 0.6), lowM nR runs to the flat-likelihood n≈50 bound; no closure winner → median kept |
| **Exponential background added (SR3Mu), then replaced by Chebychev₂** | single DCB cannot fit peak+pedestal (lowM direct χ²/ndf up to 44); expo halved it but cannot be flat; cheb2 halves it again (lowM medians 3.8→2.4) and flattens the plateau ratio to ±10% |
| Background dropped when fsig → 1 | unconstrained background parameters inflated errors and broke low-mA anchors (interp χ²/ndf ~115 before, ≤ ~6 after) |
| **Fixed orders** (x0 pol1, α pol2, c1/c2 pol2) replacing per-parameter F-test choice | closure statistically equivalent (one soft spot: MHc160 lowM_Run3 αL prefers pol1); determinism and inter-category consistency preferred |

Superseded variants' outputs are retained under
`results/MHc{X}/{fixedn,expo*,cheb,…}` and `plots/archive/`; the
simultaneous-n machinery was retired after its cross-check concluded
(conclusion recorded above).
