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
  parametrization at 1 (pinned at logit(1 − 10⁻³) with a fixed
  logit-space error `FSIG_LOGIT_ANCHOR_SIGMA`); their c1/c2 carry zero
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
| alphaL, alphaR | **up-only F-test ladder [2,3]**: pol2 minimum, pol3 where the data demand it (p < 0.05) |
| nL, nR | frozen per category (constant records) |
| fsig | polynomial in **logit space** (F-test ladder pol2–5): bounded in (0,1) like the earlier logistic but able to **turn over** — the true fsig rises past the plateau and falls again as mA → mHc, where the two OS pairings converge and the combinatoric pair re-enters the window. Anchor points (fsig = 1) pinned at logit(1 − 10⁻³); linear-space polynomial fallback below 5 points |
| c1, c2 | **up-only F-test ladder [2,3]** (as α) |

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

**Chain** (variant `cheb_fixedn_logitfsig_ftest`; its direct fits are
read from the `cheb_fixedn` sibling since the fsig/order parametrization
does not affect the per-point fits; one condor job per mass point for
stage 1):
see `test/interpolation/README.md` for the exact commands.
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
- **Interpolation closure**: interp medians 1.5–5.0 vs direct 1.4–2.8
  over all 12 category × mHc combinations (production-relevant regions:
  medians 1.6–5.0, SR3Mu worst ≤ 6.3); the interpolated shapes are
  statistically close to per-point fits everywhere in the studied
  ranges. The logit-space fsig removed the systematic misfit the
  logistic left at its plateau corner and at the mA → mHc endpoint
  (highM MA92–125: e.g. MHc145 Run3 4.3→2.6 / 4.7→2.4, MHc160 Run2
  worst 10.7→6.1; endpoint fsig pulls +9/+10σ → ≈0). The up-only
  order ladders further improved the denser MHc160 grid (held-out
  production-relevant median 2.66→2.46, worst 7.3→6.3) while leaving
  MHc145 unchanged.
- SR3Mu_highM at MA15–25 has no physical peak (the high-mass pairing
  picks the combinatoric dimuon at low mA) — those direct fits fail the
  quality gate by construction and are excluded; production never uses
  highM there.

**Hand-off artifact**:
`results/MHc{X}/cheb_fixedn_logitfsig_ftest/polynomials.json`
(+ frozen n in the `cheb_fixedn` sibling's `dcb_fits.json` meta).
Remaining steps to production: the template producer integrating the
interpolated model over adaptive bins (`test/interpolation` code
graduates into `python/` at that point).

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

**Model** (`fit_yield_model.py`; physics-structured — each factor is a
quantity with a physical reason to be simple):

    N_win(era, mA) = k_era × G_period(mA) × f_category(mA)

- `G_period` — shared baseline-selection yield shape per total-channel
  (SR1E2Mu / SR3Mu — the two pairings have identical totals) × run
  period, log-space poly (orders [3,4]) on the **period-summed**
  full-tree totals. At fixed mHc the b-jet efficiency is flat in mA;
  what remains is smooth mHc–mA kinematics: a rise to a maximum near
  mA ≈ mHc − mW and an asymmetric fall as the W* phase space closes —
  hence the cubic minimum. Summing the four eras halves the per-sample
  normalization scatter (see below).
- `k_era` — constant era share. Era shares were measured flat in mA
  (Run2 rms ≤ 1.3%; the Run3 share deviations are the sample scatter,
  which a constant deliberately refuses to chase).
- `f_category` — window fraction per category, **era-independent**
  (measured era spread ≤ 0.9% abs), fitted on period-merged points:
  - SR1E2Mu: near-constant peak containment (pol0/1; the ±10σ window
    always holds the peak — measured 0.93–0.97 with a slow rise from
    improving tail containment).
  - SR3Mu: pure pairing combinatorics, **derived from the shape fit's
    fsig** instead of fitted freely. Exactly one of the two OS pairings
    is the true A→μμ pair, so p_low + p_high = 1 and
    f_low·fsig_low + f_high·fsig_high = S, the shared containment
    (measured 0.93–1.03 while f_high itself spans a factor ~80).
    S gets a low-order poly, the pairing probability p_high a
    logit-space poly (bounded in [0,1]), and
    f_variant = S · p_variant / fsig_variant with fsig evaluated from
    the adopted shape polynomials — the sigmoid the old fraction fits
    struggled with is exactly what fsig already measures.

The redesign replaced a first-generation per-era log-poly product
(`N_total × f_window` per sub-era × channel, orders [1,2,3] × [2,3,4,5];
`fit_yield_curves.py`, kept as comparison baseline). Deciding
experiments in `yield_model_experiments.py` (E1–E4): SR1E2Mu f flat
(pol1 χ²/ndf ≈ 4/8), S ≈ const while the fractions vary by ×80, Run2
era shares flat at ~1%, and held-out closure equal or better everywhere
with ~10× fewer parameters — most dramatically SR3Mu_highM (MHc145
held-out Run2 max 9.3% → 1.5%). F-test ladder (p < 0.05) selects orders
within each sub-model; prediction errors combine the component bands in
quadrature.

**Validation** (`yield_closure.py`, all study points): fit points =
self-consistency test, held-out points = interpolation test; plus a
**template-level absolute-normalization check** per merged category —
the interpolated shape model normalized to the summed per-era predicted
yields (no rescaling to MC) against the 100-bin MC histogram.

Results over the production-relevant regions (lowM: mA < 60,
highM: mA ≥ 60, SR1E2Mu: all — mHc ≥ 100 pairing rule):

| | held-out median \|rel\| | held-out max \|rel\| |
|---|---|---|
| Run2 (both mHc, all channels) | 0.4–1.8% | ≤ 8.8% (SR3Mu ≤ 6.0%) |
| Run3 (both mHc, all channels) | 2.2–5.2% | ≤ 29% (sample scatter) |

Template-level χ²/ndf medians 2.0–3.5 (Run2) and 1.7–5.8 (Run3) —
comparable to the shape-only closure, i.e. predicting the
normalization does not degrade the template agreement. The remaining
Run2 held-out misses beyond ~4% are localized at sparse fit-grid gaps
in the steep phase-space fall (e.g. mA=120 between fit points 100/140
at MHc145) — a grid-density limitation common to both models, and a
worst case relative to production, where every simulated point anchors
the fit.

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

**Hand-off artifact**: `results/MHc{X}/yields/yield_model.json`
(per-period `fractions` {f_sr1e2mu, S, p_high_logit} and `totals`
{G, k_era} records; prediction via `fit_yield_model.predict_yield`,
which also needs the shape polynomials for fsig). The legacy per-era
product records remain in `yield_polynomials.json` for comparison.

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
| Yield: per-era log-poly product replaced by the **physics-structured model** k_era · G_period · f, with f_SR3Mu = S·p/fsig | physics argument (b-jet eff. flat in mA; ±10σ window ⇒ SR1E2Mu f const; SR3Mu combinatorics already measured by the Chebychev fraction) confirmed by experiments E1–E4: f era-independent and flat for SR1E2Mu (pol1 χ²/ndf 4/8), S = f_l·fsig_l + f_h·fsig_h flat within ±4% while f_high spans ×80, Run2 era shares flat at ~1%; closure equal or better with ~10× fewer parameters, highM held-out max 9.3% → 1.5% (MHc145 Run2) |
| Yield: G orders forced to cubic minimum [3,4] | log N_total is a rise + asymmetric fall (W* phase space closing toward mA → mHc); from-pol2 F-test ladders stalled and left 5–9% held-out misses at sparse steep-fall grid gaps |
| **fsig logistic replaced by a logit-space polynomial** (variant `cheb_fixedn_logitfsig`; F-test pol2–5, anchors pinned at logit(1 − 10⁻³)) | the true fsig **turns over** — it rises past the logistic's plateau and falls as mA → mHc (pairings converge, combinatoric pair re-enters the window); the monotonic logistic left ±4–10σ fsig pulls at the plateau corner and endpoint, i.e. the MA92–125 closure soft spot. Logit-poly: shape closure equal or better in 7/8 SR3Mu categories (MHc145 highM Run3 MA92/95: 4.3→2.6 / 4.7→2.4; MHc160 Run2 plateau halved), endpoints fixed; direct fits are fsig-form-independent and stay in `cheb_fixedn` |
| Full F-test ladders [0..3] for α/c1/c2 rejected | on the sparse MHc145 grid (7 anchors) the F-test let α stall at pol0/1 — in-sample parsimony, but held-out closure degraded (median 2.52→2.77, worst 5.8→7.8); all MHc160 gains came from orders moving *up* |
| **Up-only F-test ladders [2,3] for α/c1/c2 adopted** (variant `cheb_fixedn_logitfsig_ftest`; fixed order = ladder minimum) | strict win: MHc145 unchanged, MHc160 held-out production-relevant median 2.66→2.46, worst 7.3→6.3 (pol3 taken by 7 category-parameters, mostly c1/α in highM); the F-test is safe only as an *upgrade* on the physics-motivated minimum — it cannot see interpolation quality |
| Widened yield ladders (S/G/f_SR1E2Mu +1 order) rejected | F-test took the extra order in 3 places with large in-sample χ² drops (MHc160 Run3 S 22.6/6→1.9/4) but held-out yield closure unchanged (±0.3%) — residuals are per-sample normalization scatter, not model stiffness; G never took pol5 |

Superseded variants' outputs are retained under
`results/MHc{X}/{fixedn,expo*,cheb,…}` and `plots/archive/`; the
simultaneous-n machinery was retired after its cross-check concluded
(conclusion recorded above).
