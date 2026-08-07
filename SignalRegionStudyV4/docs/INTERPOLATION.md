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
steps to production: the template producer integrating the interpolated
model over adaptive bins (`test/interpolation` code graduates into
`python/` at that point).

## Yield interpolation (adopted method)

Shapes are shared between the eras of a run period; **yields are
interpolated per sub-era** — the datacard's signal columns are per-era
components (`signal_2017`, …) whose `rate = -1` reads the nominal
histogram integral, so the per-era normalization is the one remaining
mA-dependent quantity (the `weight` branch is already era-lumi × 5 fb
reference; the BR/xsec conversion in `collectLimits.py` is a global
constant).

**Yield definition**: Σw(Central) inside the production mass window
`[max(x0 − 10σ_eff, 12), x0 + 10σ_eff]`, with x0 and
σ_eff = √(0.5(σL²+σR²)) evaluated from the **interpolated** shape
parametrizations (`interp_config.interp_window`) — smooth in mA,
computable at any mA without MC, and exactly the number the parametric
template will be normalized to.

**Model** (per sub-era × study channel; `fit_yield_curves.py`): the
window yield is fit as the product of two log-space polynomials,

    N_win(mA) = N_total(mA) × f_window(mA)

- `N_total` — full-tree Σw = Baseline-selection acceptance × lumi
  (identical for both SR3Mu pairings). Smooth in mA but carrying
  **per-sample normalization scatter** beyond MC stat (see below), so it
  gets orders [1,2,3] with per-period error floors
  (`REL_YIELD_ERR_FLOOR`: Run2 2%, Run3 8%) — the fit averages through
  the noise.
- `f_window = N_win/N_total` — the window-capture fraction. The
  normalization noise cancels in the ratio, leaving near-noise-free
  points that carry all the sharp mA structure (the sigmoid-like peak
  migration through the window around the pairing boundary for
  SR3Mu_highM; the lowM minimum-then-rise as mA → mHc). Orders
  [2,3,4,5], binomial-like errors floored at 0.5% (log space).

A direct log-poly fit of N_win was tried first and failed both ways at
once (Run3 residuals to 40%, the highM turn-on unfittable) — the
decomposition separates the noisy-but-smooth factor from the
precise-but-sharp one. F-test ladder (p < 0.05) selects the order;
prediction errors combine both bands in quadrature.

**Validation** (`yield_closure.py`, all study points): fit points =
self-consistency test, held-out points = interpolation test; plus a
**template-level absolute-normalization check** per merged category —
the interpolated shape model normalized to the summed per-era predicted
yields (no rescaling to MC) against the 100-bin MC histogram.

Results over the production-relevant regions (lowM: mA < 60,
highM: mA ≥ 60, SR1E2Mu: all — mHc ≥ 100 pairing rule):

| | held-out median \|rel\| | held-out max \|rel\| |
|---|---|---|
| Run2 (both mHc, all channels) | 0.8–3.2% | ≤ 9.3% |
| Run3 (both mHc, all channels) | 3.1–8.9% | ≤ 25% |

Template-level χ²/ndf medians 2.1–3.5 (Run2) and 2.1–5.8 (Run3) —
comparable to the shape-only closure (1.7–4.2), i.e. predicting the
normalization does not degrade the template agreement.

**Run3 per-sample normalization scatter (upstream finding)**: Run3
signal samples scatter ±10–20% around any smooth acceptance curve,
nearly identically in SR1E2Mu and SR3Mu for the same (era, mass point)
— a sample-level effect, not channel physics. Raw-skim sizes/entries
vary ×4 between adjacent mass points with the mean per-event weight
tracking 1/N_generated, so the bookkeeping only partially compensates.
Preprocessing is faithful; the issue is upstream (SKNano / sample
production) and equally affects the **current production binned
templates** built from the same samples. Run2 samples are clean at the
1–2% level. The Run3 closure residuals above are dominated by this
scatter, not by interpolation error — the smooth fitted curve is
arguably a better acceptance estimate than any single sample.

**Nuisance structure (design, to be exercised in the template phase)**:
correlation bookkeeping is unchanged — it lives in nuisance names
(era-suffixed = uncorrelated; `_13TeV`/`_13p6TeV` = correlated within a
run period; unsuffixed = fully correlated) and the datacard mechanism
needs no change for parametric signals. Shape systematics will be
treated as **shifts of the fit function** (parameter-level variations
derived from the systematic trees, which exist in every shared signal
file), with per-era varied yields; `valued lnN` (lumi) and
`valued shape` (trigger) stay mA-independent config constants.

**Hand-off artifact**: `results/MHc{X}/yields/yield_polynomials.json`
(per era × channel `total`/`fraction` logpoly records; prediction =
product of the two `eval_param` evaluations).

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
- Run3 signal samples: ±10–20% per-sample yield normalization scatter,
  upstream of preprocessing (see "Run3 per-sample normalization
  scatter" above) — needs follow-up on the SKNano/sample-production
  side; affects production templates as well as this study.

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
| Yield: direct log-poly fit of N_win replaced by the **total × window-fraction decomposition** | direct fit failed twice over: Run3 residuals to 40% (per-sample normalization scatter dragging the curve) and the highM sigmoid turn-on unfittable by any low-order polynomial; the ratio f_window cancels the normalization noise and isolates the sharp structure |
| Yield: per-period error floors (Run2 2%, Run3 8%) on N_total points | Run3 samples scatter ±10–20% around any smooth curve, channel-correlated, traced to raw skims (upstream); MC-stat-only weights made every fit chase sample noise (χ²/ndf to 400) |

Superseded variants' outputs are retained under
`results/MHc{X}/{fixedn,expo*,cheb,…}` and `plots/archive/`; the
simultaneous-n machinery was retired after its cross-check concluded
(conclusion recorded above).
