# INTERPOLATION.md — Parametric Signal Templates via DCB Interpolation

Purpose, adopted method, results, and decision history of the
mA-interpolation development (B2G-25-013 next-generation signal
modeling). The chain now lives in production (`python/`, `scripts/`,
`automize/`, outputs in `tests/interpolation/`); the method-development
study is frozen under `test/archive/interpolation/` (untracked scratch by
convention). This document is the durable record.

## Production model (frozen 2026-08-12)

The state a reader should assume everywhere below unless a section says
otherwise. Full evidence for each line is in the decision history.

| piece | model |
|---|---|
| SR1E2Mu, SR3Mu_lowM | pure DCB, frozen per-category nL/nR |
| SR3Mu_highM | fsig·DCB + (1−fsig)·Chebychev₂ |
| shape vs mA | one (mHc, mA) surface per parameter across all seven studies, sliced at the study's mHc; fsig in logit space |
| yield | N = k_era · G_period(mA) · f_category(mA); G and k_era are (mHc, mA) surfaces, f is per-study with f_SR3Mu = S·p on raw fractions |
| uncertainties | rms within each study, then max across studies; norm binned in mA at 15/80/100/155; no mHc dependence in any family |
| interpolation range | **mA only**, at the seven measured mHc |

Two cross-study barriers follow from the surfaces: `polynomials` reads
every study's fits and `yield_model` every study's yields, so a rebuild
runs in three passes (`automize/interpolation.sh --stop-after`).

## Graduated layout (2026-08-11)

The method-development study below (`test/interpolation/`, now frozen
under `test/archive/interpolation/`) concluded, and the chain graduated
into production:

- Code: `python/interpolation_config.py` (successor of `interp_config.py`
  — mass-point splits move to `configs/interpolation.json`, everything
  study-specific like `param_template_config.py`'s delta policy is
  absorbed), `python/dcb_fit_utils.py`, and the per-stage entry points
  `fitInterpShapes.py` → `fitInterpPolynomials.py` → `closInterpShapes.py`
  (shape chain), `measInterpYields.py` → `fitInterpYieldModel.py` →
  `closInterpYields.py` (yield chain), `measInterpShapeDeltas.py` →
  `fitInterpShapeDeltas.py` (shape-delta chain), one consolidated
  `mergeInterpResults.py --stage {fits-floating,fits,closure,yields,
  yield-closure,shape-deltas}`, `verifyInterpSamples.py`, and the new
  `exportInterpUncertainties.py` (below) + `interp_plot_utils.py`
  (cmsstyle plots — matplotlib is gone from the graduated chain).
- Execution: `scripts/interpolation_wrapper.sh` (condor leaf dispatch) +
  `automize/interpolation.sh --mhc N|--all [--start-from STEP] [--local]
  [--dry-run]` (one DAG per mHc; `--local` runs serially without condor).
- Outputs: `tests/interpolation/MHc{X}/` (fits, polynomials, closure,
  yields, shape_deltas, uncertainties.json, plots/) — git-tracked, JSON
  alongside cmsstyle PNGs.
- **Variant machinery is gone.** The adopted method (Chebychev2
  background, frozen median n, logit-space fsig, up-only F-test ladders)
  is hard-coded; `--expo-bkg`/`--bern-bkg`/`--cheb-bkg`/`--logit-fsig`/
  `--ftest-orders` and the `{variant}` directory level no longer exist.
  The two-pass floating-n → frozen-n structure is kept
  (`fitInterpShapes.py --pass floating|frozen`).
- **Study grid extended to 7 mHc values** (was 2): 70, 85, 100, 115, 130,
  145, 160, with fit-anchor/held-out splits in
  `configs/interpolation.json["fit_points"]` (held-out = full baseline
  grid − fit anchors, computed, never stored). **The MHc145/160 fit
  anchors changed** from the values below (145: 80,90 → 85,92; 160:
  90 → 85,95) — both studies are re-run under the new splits, so their
  `polynomials.json`/`yield_model.json`/closure numbers postdate this
  section and supersede the "Results" section below where they differ.
- **New**: `exportInterpUncertainties.py --mhc N [--all]` derives the
  interpolation nuisance sizes directly from the held-out closure
  residuals (max envelope, conservative by design) instead of the fixed
  arm-C constants (`INTERP_SCALE_NSIGMA`/`INTERP_RES_REL`/
  `INTERP_NORM_LNN` below) — scale/res per (channel, run period),
  decorrelated-per-era norm (`CMS_interp_norm_{channel}_{era}`, replacing
  the arm-C `_13TeV`/`_13p6TeV` norm token). Output:
  `tests/interpolation/MHc{X}/uncertainties.json` (per-point detail) and
  the consolidated `configs/interpolation_uncertainties.json` consumed by
  the (future) production template producer.
- **Not yet graduated**: the parametric-template producer
  (`make_param_templates.py`) and the one-off arm-comparison validation
  (`compare_arms.py`, `param_template_config.py`'s `ARMS` table) — a
  separate follow-up phase once the exported uncertainties are validated
  against the new grid. The decision history and variant comparisons
  below remain the durable record of *how* the method was chosen; they
  are not reproducible from the graduated code (variant flags removed).

Verified at graduation (static only — no physics rerun yet): every entry
point imports under CMSSW (numpy 1.24.3 / scipy 1.10.0 / ROOT 6.30/07 /
cmsstyle), all 7 splits satisfy fit ⊂ grid and held_out = grid − fit, and
`--mhc 160 --dry-run` emits a 148-node DAG (6 fan-out stages × 23 mass
points + 10 spine nodes) with the expected fan-in/fan-out edges.

## Next steps (as of 2026-08-11)

Ordered; each depends on the one before it.

1. **Samples for the five new mHc values.** MHc70/85/100/115/130 signals
   were never preprocessed for this study (only 145/160 were). Run
   `./automize/preprocess.sh --masspoint MHc{X}_MA{Y} --skip-backgrounds`
   (shared backgrounds are already in place, and both SR3Mu pairing
   variants are written per point), then gate on
   `python3 python/verifyInterpSamples.py --all --mhc {X}` — the
   anti-truncation check is not optional, concurrent-xrdcp truncation is
   an observed failure mode.
2. **Run the seven studies.** `./automize/interpolation.sh --all`
   (or `--mhc N` per point; `--local` for serial execution if condor is
   unavailable on the current server). MHc145/160 **must** be re-run:
   their fit anchors changed, so the archived `polynomials.json` /
   `yield_model.json` no longer match `configs/interpolation.json`.
   Regression check on MHc160: closure medians should land near the
   archived record (production-relevant medians 1.6–5.0) — compare
   medians, not bitwise, since covQual can flip on marginal fits across
   machines.
3. **Review the derived uncertainties.** `exportInterpUncertainties.py
   --all` writes `configs/interpolation_uncertainties.json`. Check the
   `n_points` field (a per-era norm envelope resting on 3 points is thin)
   and any `> 0.10` warnings: for Run3 those are expected (the upstream
   per-sample scatter, absorbed deliberately by the max envelope), but on
   Run2 they flag a sparse fit-grid gap in the steep phase-space fall and
   may argue for adding a fit anchor there.
4. **Graduate the template producer** (`make_param_templates.py` →
   `python/`), behind a new `srspaths.template_dir` method segment,
   reading the exported uncertainties instead of the hard-coded
   `INTERP_*` constants, and declaring them through `printDatacard.py`'s
   existing `extra_systematics*.json` mechanism (already a no-op for
   existing methods). **In production the shape/yield/delta models are
   refit over the FULL mass-point grid** — the fit/held-out splits exist
   only for this closure study and for deriving the uncertainties, so no
   model export is needed from the study chain.
5. **Re-validate the limit closure** on the new grid: repeat the four-arm
   comparison (direct MC vs parametric, window/binning, nuisances) at a
   held-out point, now with derived rather than assumed nuisance sizes.
6. **ParticleNet interpolation study** — the whole chain above is a
   Baseline study; the ParticleNet variant follows once Baseline is
   production-ready.

Open upstream items (not blockers, but they bound the achievable closure):
the corrupt `MHc145_MA100` / 2023 / SR1E2Mu raw skim needs SKNano
regeneration (carried as `known_missing_samples`), and the Run3
per-sample normalization scatter (±10–20%, channel-correlated) still
wants a fix on the sample-production side — it inflates the derived Run3
norm nuisances and equally affects the current production templates.

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
- SR3Mu_lowM: **pure DCB** as well (adopted 2026-08-12). Both pairings
  pick a combinatoric dimuon, but lowM's continuum is only a few percent
  and is absorbed by the DCB tails; modelling it explicitly required a
  per-point drop/pin rule that made the category *mixed* and broke the
  parametrizations (see the decision history).
- SR3Mu_highM:
  `S(m) = fsig · DCB(m) + (1 − fsig) · Chebychev₂(m; c1, c2)` — here the
  pairing picks the wrong (combinatoric) dimuon in 5–45% of events,
  producing a near-flat pedestal under the peak that only a shape that
  *can be flat* describes.
- **Frozen tails**: nL/nR fixed per category to the median of the good
  floating-n fits (two-pass: floating-n campaign first, then the
  frozen-n refit). This breaks the α–n degeneracy — with n floating,
  each mass point lands elsewhere in the α–n valley and no smooth
  parametrization can follow.
- **No background drop.** `FSIG_DROP_THRESHOLD` (0.995) survives only as
  a warn-only sanity check: a highM fit that reaches it has no resolvable
  background and something has changed. It never fired across all 156
  highM fits of the seven studies. The old rule — refit such a point as a
  pure DCB and pin fsig = 1 — is what made lowM a mixed category.
- Two-stage fit structure (wide pre-fit sets a ±10σ window) identical to
  the production `fit_dcb`; `fit_dcb_with_errors` remains a verbatim
  mirror of production (frozen reproduction contract),
  `fit_dcb_bkg` is the study's generalized fit.

**Parametrization: one (mHc, mA) surface per parameter** (adopted
2026-08-12, `fitInterpPolynomials.py`). Every shape parameter is fitted
as a single error-weighted surface across **all seven mHc studies** and
sliced at the study's mHc:

| aspect | choice |
|---|---|
| basis | total-degree-truncated tensor polynomial in the scaled coordinates ((mHc−115)/45, (mA−70)/70), mHc degree ≤ 2, total degree ≤ 4 (`SHAPE_SURFACE_DEGREES`) — 12 coefficients |
| space | linear for x0/σ/α/c1/c2; **logit** for fsig (bounded in (0,1) and able to turn over as mA → mHc, where the pairings converge and the combinatoric pair re-enters the window) |
| nL, nR | frozen per category (constant records), median of that study's floating-n fits |
| weights | 1/error, good-quality fits only, `MIN_REL_PARAM_ERROR` guard against collapsed Hesse errors |

Interpolation remains **in mA only**: the surface is a better-constrained
model *at* the seven measured mHc, not a licence to interpolate between
them (leaving a whole study out and predicting it from the other six is a
4% median / 18% p90 yield error).

Because the slice of a polynomial surface at fixed mHc is itself a
polynomial in mA, `polynomials.json` keeps the plain coeffs+cov record and
**nothing downstream had to change** — `eval_param`, `interp_window`, both
closures, the shape-delta chain and the template producer all still work
on the sliced records. `joint_design`/`slice_surface`/`fit_surface` live in
`interpolation_config` and are shared with the yield chain.

**Uniform mA range** for both SR3Mu variants: production uses highM only
above the mA = 60 pairing boundary, but at lower mHc (e.g. MHc130) that
region holds too few MC points, so every mass point constrains both.

Practical consequence: `polynomials` reads every study's `dcb_fits.json`,
a cross-study barrier the per-mHc DAGs do not express. Use
`automize/interpolation.sh --stop-after` to run the chain in passes.

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
  - SR3Mu: pure pairing combinatorics. Exactly one of the two OS pairings
    is the true A→μμ pair, so p_low + p_high = 1 and S = f_low + f_high
    is the shared containment (roughly constant while f_high itself spans
    a factor ~80). S gets a low-order poly, the pairing probability
    p_high a logit-space poly (bounded in [0,1]), and
    f_variant = S · p_variant.

    The decomposition is taken on the **raw measured fractions**. It was
    originally derived from the shape fit's fsig (f_v = S·p_v/fsig_v);
    both forms are exact reparametrizations of (f_low, f_high), the
    smoothness test between them is a wash, and the /fsig version
    coupled the yield model to the shape chain — which pure-DCB lowM
    cannot support, since lowM has no fsig at all (2026-08-12).

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

## Parametric templates and limit closure (2026-08-10)

The models above are only useful if a datacard built from them gives the
same limit as one built from signal MC. Tested at **MHc160_MA90**,
`--era All --channel Combined`, unblind, with every model input refitted
**without** mA=90 (`_ex90` shape variant + `--exclude-ma 90` yield model),
so the template at the target is a genuine interpolation. Four arms
(`test/interpolation/param_template_config.py` holds the table):

| arm | method segment | signal | window/binning | interp. nuisances |
|---|---|---|---|---|
| A | `Baseline` | direct MC | direct DCB fit | — |
| B0 | `BaselineParamB0` | parametric | arm A's | — |
| B | `BaselineParam` | parametric | interpolated | — |
| C | `BaselineParamNuis` | parametric | interpolated | yes |

Backgrounds, nonprompt and data come from the production code path in all
four arms — only the signal columns differ.

**Systematics without signal MC**: every signal shape systematic is
compressed on the *other* mass points of the same mHc into three
dimensionless deltas relative to Central — `dm` (core-window mean),
`dsig` (core-window rms), `dN` (window sum) — each parametrized in mA
with an up-only pol0/pol1 ladder that excludes the target point, and
applied as `x0 → x0(1+dm)`, `σ_{L,R} → σ(1+dsig)`, `N → N(1+dN)`. Windows
come from the interpolated parameters, so the recipe is identical at
donor and target points. Weight-only trees hold the same events as
Central, so their deltas are paired differences; kinematic trees fall
back to the uncorrelated error. Closure of the interpolated delta against
the delta measured on the target's own MC (6576 series): median |Δ|
1.4e-7 (dm), 7.3e-6 (dsig), 1.7e-5 (dN); worst cases are JES `dN`
(−6.3% vs −4.5% measured) and ps_isr/fsr.

**Interpolation nuisances** (arm C), decorrelated per (channel, run
period) because the shape polynomials, G and f/S/p are fitted per
(channel, period): `CMS_interp_scale_*` (x0 ± 0.05 σ_eff),
`CMS_interp_res_*` (σ ± 4%), `CMS_interp_norm_*` (lnN 1.020 Run2 /
1.050 Run3) — 12 lines, each hitting the four signal columns of its
category. The Run3 yield-error floor is deliberately excluded from the
norm term: it encodes the upstream per-sample scatter, which the
direct-MC reference carries with no penalty of its own.

**Results** (`results/MHc160/param_templates/arm_comparison.json`;
limits from a tight-tolerance rerun, `--rRelAcc 0.0005`, because the
frozen production command stops at 1%):

| | A | B0 | B | C |
|---|---|---|---|---|
| expected median r | 0.9590 | 0.9258 | 0.9258 | 0.9258 |
| observed r | 0.6378 | 0.6170 | 0.6498 | 0.6519 |
| signal yield (all categories) | 89.98 | 93.13 | 92.89 | 92.89 |

- **A ↔ B0 (the signal-template method)**: expected limit −3.5%,
  observed −3.3%, driven entirely by normalization: per-category signal
  yields are **+1.0% / +2.3%** (Run2 SR1E2Mu / SR3Mu) and **+12.1% /
  +6.6%** (Run3), while the *shapes* agree to an L1 distance of ≤ 3.1%
  of the normalized template. The Run3 offsets sit inside the known
  ±10–20% Run3 per-sample scatter, i.e. the interpolated curve differs
  from a single noisy MC point, which is the expected — arguably
  desirable — behaviour. Per-bin ratios above 10% are confined to the
  Run3 categories and to near-empty ±10σ tail bins (max 39.9% in a bin
  holding 0.02 events).
- **B0 ↔ B (window/binning)**: the interpolated window reproduces the
  direct-fit one closely (x0 to ~0.01 GeV, σ_eff to ~2%) and gives the
  same 17 bins in every category; the expected limit is unchanged
  (identical to combine's asymptotic r grid, ~0.2%) and the signal yield
  moves ≤ 0.5%. The **observed** limit moves +5.3%: with a real dataset,
  a sub-percent shift of the bin edges reshuffles which events land in
  which bin, and the observed limit is sensitive to that at the few-%
  level while the expected limit is not.
- **B ↔ C (interpolation nuisances)**: expected limit unchanged at the
  ~0.2% grid resolution, observed +0.3%, despite genuinely non-trivial
  shape variations (up to 8% per bin for scale, 17% for resolution).
  The analysis is background-statistics limited, so signal-template
  nuisances of this size are free.

**Conclusion**: a parametric signal template at a held-out mass point
reproduces the direct-MC limit to a few percent, and the residual is a
normalization difference of the size of the known Run3 sample scatter,
not a shape or method failure. The interpolation nuisances cost nothing.

**Chain**: `submit_shape_deltas.sh` → `merge_shape_deltas.py` →
`fit_shape_deltas.py` (→ `delta_model.json`), then `submit_arm.sh --arm
{A,B0,B,C}` (one DAG per arm: four template leaves →
`mergeRunPeriodTemplates.py` → `printDatacard.py` → `runAsymptotic.sh`)
and `compare_arms.py`. Templates are built one (period, channel)
category per job — four categories in one process exceeds 8 GB, and the
production DAG splits them for the same reason.

**Production touchpoints** (both no-ops for existing methods):
`printDatacard.py` merges an optional `extra_systematics*.json` from the
template directory into the per-(subera, channel) systematics blocks,
which is how a template producer declares nuisances no era config can
carry; `mergeRunPeriodTemplates.py` accepts any `--method` segment.

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
| **Parametric templates validated against direct MC at a held-out point** (MHc160_MA90, four-arm limit comparison) | expected limit −3.5% (A→B0), unchanged for the window/binning shift and for the interpolation nuisances; the −3.5% is a pure normalization effect (yields +1.0/+2.3% Run2, +12.1/+6.6% Run3 — inside the known Run3 sample scatter) with shape L1 ≤ 3.1%. Systematics are carried as fit-function shifts (dm, dsig, dN interpolated in mA); their closure against the target's own MC has median |Δ| ≤ 2e-5 |
| Positive-definite Bernstein background (deg-2, b₁,b₂ ≥ 0; `bern_fixedn_logitfsig_ftest`) rejected | motivated by rare negative-Chebychev tails (all in production-irrelevant corners; the visible symptom is only a plot-time χ² = nan). Bernstein converges more often (bad direct fits 25→7 at MHc160 floating-n) but describes the combinatoric shape worse where it matters: direct-fit medians degrade in 7/8 SR3Mu categories (lowM_Run2 2.4→4.6 at MHc160) because the positivity bound is active where the data want the Chebychev's freedom, and the b₁=b₂=0 boundary pileup degrades the coefficient-vs-mA interpolation (production-relevant held-out closure MHc160 median 2.46→2.85, worst 6.3→19.8). Negative Chebychev tails are instead rendered inert by the existing `BIN_FLOOR_VALUE` floor at template-construction time |

| **SR3Mu_lowM refit as a pure DCB** (`puredcb`; `nodrop` tested and rejected) | the drop/pin rule made lowM MIXED — 25 of 156 fits pinned at fsig=1 beside free fsig≈0.9 points, c1/c2 with partial mA support — and its fsig/c1/c2 parametrizations oscillated and extrapolated (MHc145_MA35 lowM_Run3: predicted fsig 0.916 where the direct fit is a pure DCB, c1 = −6, χ²/ndf 154). Pure DCB: worst production χ²/ndf 24.5, and χ²_interp ≈ χ²_direct everywhere (24.5 vs 24.3), i.e. interpolation adds nothing on top of the fit model; peak fidelity equal (scale ≤ 0.12 σ_eff, res ≤ 3.6%). Cost: the unmodelled few-% continuum lifts median χ² 2–3 → 4–15. `nodrop` (never pin, honest logit errors) FAILED: at fsig≈1 the logit error is ~1/(p(1−p)) so the polynomial is unconstrained exactly there — MHc130 lowM_Run2 mA15 predicted fsig = 0.05 against a true 0.99999, χ² 45468. **fsig is not an interpolable quantity**: its point fits zigzag 5–13% with claimed errors ±0.3%, DCB-tail degeneracy artifacts |
| **Yield G(mA) and k_era become (mHc, mA) surfaces** (`joint`) | factorizing the LOO residual showed every large production-pairing miss is a G failure (\|dG\| to 23%, \|df\| ≤ 8%), and that NO alternative 1D basis fixes it — pol up to 6, log-mA, cubic spline, PCHIP and linear all tie or lose against pol[3,4]. It is grid sparsity: MHc100 has 8 points and MHc115 jumps 15 → 27 → 42, so dropping one point swings the cubic ±20% with alternating signs. One surface across all studies (12 parameters against ~35 for seven cubics) takes LOO failures above 10% from 8.6% to 3.5% and the p90 at mA ≤ 45 from 16.1% to 7.7%. k_era likewise: the shares carry a real smooth mHc drift (2018/SR1E2Mu 0.4280 → 0.4346 across mHc) and pooling averages the per-sample noise over 78 points instead of 6–23 — envelopes above 10% 51/152 → 43/152 with 3 coefficients in place of 28 constants. A per-study pol0/pol1 in mA was tried first and is a wash (73 → 75): the gain is the pooling, not the mA freedom |
| **SR3Mu yield pairing decomposition drops its /fsig division** | S = f_low + f_high and p_high = f_high/S on the raw measured fractions. Both forms are exact reparametrizations of (f_low, f_high); the smoothness test is a wash (helps 0, hurts 1, mixed 13 of 14 datasets) and the one loss is MHc145 Run3, the origin of the +95% lowM yield-closure blow-up. It also decouples the yield model from the shape chain, which pure-DCB lowM cannot support (lowM has no fsig at all) |
| **Shape parametrizations become (mHc, mA) surfaces too** | same protocol, applied to x0/σL/σR: worst-case scale error halves (0.344 → 0.172 σ_eff), p90 0.064 → 0.045, res p90 0.033 → 0.026. Two controls establish that the gain is the cross-mHc constraint and not extra freedom in mA: giving the 1D per-study fit wider mA orders leaves the scale max at 0.344 and nearly doubles the res max (0.115 → 0.264), while flattening the mHc dependence out of the surface is also worse (scale p90 0.065). A separate check shows the LOO test is fair: denying the surface every point within ±5 GeV in other studies changes nothing (>10% rate 3.5% either way), so it is not reading off a near-duplicate; removing the target's own study entirely degrades it badly (median 4.1%, p90 17.7%), so it does need that study's points. `POLY_ORDERS` and the per-parameter F-test ladders retire with this |
| **One uncertainty rule for scale/res/norm: rms within each study, then max across studies** | the LOO residuals are unbiased (\|mean\| < 2% in every cell) and Gaussian-like — max/rms lands at 1.8–3.6 against the √(2 ln N) = 1.8–2.9 expected — so the previous plain max was a ~3σ order statistic used as a 1σ lnN width, scaling with how many points a cell happened to hold rather than with model accuracy (one mass point, MHc115_MA27, set 15 of 16 below-Z cells). The plain pooled rms is the right 1σ but hides a real effect: below the Z the per-study rms genuinely varies with mHc (observed spread 69–77% against 41% expected), tracking low-mA grid density — MHc160 samples it every 5 GeV and closes to 2.0%, MHc115/130 have 15–25 GeV gaps and close to 11–18%. rms-inside-a-study means no single mass point sets the value; max-across-studies covers the sparse ones instead of averaging them away. Studies with < 2 mass points in a cell are skipped (a one-point study leaks its outlier straight back in); the pooled rms is the floor |
| **norm binned in mA at 15/80/100/155, and no family carries an mHc dependence** | binning norm is justified by that consistent, physically traceable below-Z effect; scale/res are NOT binned, because scale's region-to-region variation has no consistent pattern across channels and res sits at its floor everywhere. Dropping mHc is forced once mA is binned — most split cells hold one or two points and some hold none at all — and legitimate because both models are now global surfaces. Effect: the average nuisance a mass point sees falls 20.7% → 15.6%, with SR3Mu_lowM on-Z going 29.1% → 5.3%. `UNCERTAINTY_RES_FLOOR` lowered 0.02 → 0.01, where it was *setting* four of six res cells (measured 0.0124/0.0148/0.0179/0.0191 all pushed to 0.0200) rather than catching degenerate ones; all three floors are now pure safety nets |

Superseded variants' outputs are retained under
`results/MHc{X}/{fixedn,expo*,cheb,…}` and `plots/archive/`; the
simultaneous-n machinery was retired after its cross-check concluded
(conclusion recorded above).
