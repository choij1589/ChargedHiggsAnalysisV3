# Interpolation — Experiment Record

Every model decision behind the frozen production model
([WORKFLOW.md](WORKFLOW.md)), as motivation → setup → results → conclusion.
Chronological within each section; where a later experiment supersedes an
earlier one it says so. The uncertainty-rule experiments are in
[UNCERTAINTY.md](UNCERTAINTY.md).

Early experiments (S1–S9, P1–P4, Y1–Y6) ran on the two feasibility studies
mHc = 145 (12 MC points) and 160 (23 points), with fit-anchor/held-out
splits; later ones (S10–S11, P5, Y7–Y8, V2–V3) on all seven mHc studies
over the full 78-point baseline grid with leave-one-out closure. The
variant machinery that produced the early comparisons was removed at
adoption — those results are recorded here, not reproducible from the
graduated code.

---

## S. Signal shape model

### S1 — DCB parameters are interpolable in principle

**Motivation.** The whole idea rests on the fitted shape parameters being
smooth in mA.
**Setup.** First MHc145 pass: 20 per-point DCB fits, parameters plotted
vs mA.
**Results.** x0 ≈ mA (pol1, slope 1); σ ≈ 1% of mA; tails noisier.
**Conclusion.** Feasible; proceed.

### S2 — MA15 moved from validation into the fit set

**Motivation.** Test extrapolation below the lowest fit anchor.
**Setup.** MA15 held out, predicted from the fit at mA ≥ 35.
**Results.** χ²/ndf up to ~600.
**Conclusion.** Never extrapolate below the lowest fit point; MA15 anchors
the fit. (Its fit window is floored at 12 GeV — clipped left tail.)

### S3 — Tail orders raised, then σL/σR fixed to a common pol2

**Motivation.** lowM MA45 / highM MA92–95 closure failures.
**Setup.** F-test per side vs forced common orders.
**Results.** The F-test's per-side pol1/pol2 split was selection noise;
common pol2 σ also tamed the MA15 edge.
**Conclusion.** σL/σR share a pol2. (Superseded by S11 — all parameters
are now surface slices.)

### S4 — SR3Mu mA-range restriction tried, then reverted

**Motivation.** lowM tails misbehave at high mA where the pairing is
production-irrelevant.
**Setup.** lowM fitted on [15,70] and highM on [50,155] only, vs the
uniform full range.
**Results.** Restriction fixed the lowM tails but starves low-mHc highM of
points (MHc130 has few mA ≥ 60 samples).
**Conclusion.** Uniform range for both pairings — every mass point
constrains both variants (this is why preprocessing writes BOTH pairings
for every point). Viable once fsig got a bounded form (S8).

### S5 — Frozen per-category tails nL/nR

**Motivation.** No smooth parametrization could follow n: α–n
valley-hopping put each mass point elsewhere in the degenerate valley
(nR polynomial χ²/ndf ~120).
**Setup.** Two-pass structure: floating-n campaign, then refit with nL/nR
frozen to the per-category median of the good floating fits. Cross-check:
a simultaneous fit of n across mass points.
**Results.** With n frozen, α becomes smooth; per-point fit cost only ~0.2
in χ²/ndf. The simultaneous fit agrees with the medians (nL within 0.6;
lowM nR runs to the flat-likelihood n≈50 bound); no closure winner.
**Conclusion.** Median-frozen n adopted; the two-pass structure
(`fitInterpShapes.py --pass floating|frozen`) is permanent.

### S6 — Combinatoric background: none → exponential → Chebychev₂

**Motivation.** In SR3Mu the pairing picks the wrong (combinatoric) dimuon
in 5–45% of events (highM), producing a near-flat pedestal a single DCB
cannot fit (lowM direct χ²/ndf up to 44; ratio swings ±20–50%).
**Setup.** Added `fsig·DCB + (1−fsig)·bkg` with bkg = exponential, then
Chebychev₂.
**Results.** Expo halved the misfit but cannot be flat; cheb₂ halved it
again (lowM medians 3.8 → 2.4) and flattened the plateau ratio to ±10%.
**Conclusion.** Chebychev₂ adopted for SR3Mu. (lowM later dropped it
entirely — S10.)

### S7 — Background dropped when fsig → 1 (later removed)

**Motivation.** Points with no resolvable background got unconstrained
background parameters, inflating errors and breaking low-mA anchors
(interp χ²/ndf ~115).
**Setup.** Refit such points as pure DCB, pin fsig = 1
(`FSIG_DROP_THRESHOLD` = 0.995).
**Results.** Interp χ²/ndf ≤ ~6 after.
**Conclusion.** Adopted at the time — but this rule is what later made
lowM a *mixed* category and was removed by S10. `FSIG_DROP_THRESHOLD`
survives only as a warn-only sanity check (it never fired across all 156
highM fits of the seven studies).

### S8 — fsig: logistic replaced by a logit-space polynomial

**Motivation.** The true fsig **turns over** — it rises past the
logistic's plateau and falls as mA → mHc, where the pairings converge and
the combinatoric pair re-enters the window. The monotonic logistic left
±4–10σ fsig pulls at the plateau corner and endpoint (the MA92–125
closure soft spot).
**Setup.** Variant `cheb_fixedn_logitfsig`: polynomial in logit(fsig),
F-test pol2–5. Direct fits are fsig-form-independent (read from the
`cheb_fixedn` sibling).
**Results.** Shape closure equal or better in 7/8 SR3Mu categories
(MHc145 highM Run3 MA92/95: 4.3→2.6 / 4.7→2.4; MHc160 Run2 plateau
halved); endpoint pulls +9/+10σ → ≈ 0.
**Conclusion.** Logit-space fsig adopted (still in use — the surface fits
fsig in logit space).

### S9 — F-test order ladders: full ladders rejected, up-only adopted

**Motivation.** Fixed orders (x0 pol1, α pol2, c1/c2 pol2) were chosen
over per-parameter F-tests for determinism; revisit with more data.
**Setup.** (a) Full ladders [0..3] for α/c1/c2; (b) up-only ladders [2,3]
with the physics-motivated minimum as the floor
(`cheb_fixedn_logitfsig_ftest`).
**Results.** (a) On the sparse MHc145 grid the F-test let α stall at
pol0/1 — in-sample parsimony, held-out closure degraded (median
2.52→2.77, worst 5.8→7.8). (b) Strict win: MHc145 unchanged, MHc160
held-out production-relevant median 2.66→2.46, worst 7.3→6.3.
**Conclusion.** The F-test is safe only as an *upgrade* on a
physics-motivated minimum — it cannot see interpolation quality.
(Superseded by S11: the surface degree is fixed, no ladders.)

### S10 — SR3Mu_lowM refit as a pure DCB (`puredcb`; `nodrop` rejected)

**Motivation.** The S7 drop/pin rule made lowM MIXED: 25 of 156 fits
pinned at fsig = 1 beside free fsig ≈ 0.9 points, c1/c2 with partial mA
support. Its fsig/c1/c2 parametrizations oscillated and extrapolated —
MHc145_MA35 lowM_Run3: predicted fsig 0.916 where the direct fit is a
pure DCB, c1 = −6, χ²/ndf 154.
**Setup.** Two variants on all seven studies: `puredcb` (lowM loses the
background entirely — its continuum is a few percent, absorbed by the DCB
tails) and `nodrop` (keep the background everywhere, never pin, honest
logit errors).
**Results.** `nodrop` FAILED: at fsig ≈ 1 the logit error is ~1/(p(1−p)),
so the polynomial is unconstrained exactly there — MHc130 lowM_Run2 mA15
predicted fsig 0.05 against a true 0.99999, χ² 45468. `puredcb`: worst
production χ²/ndf 24.5, χ²_interp ≈ χ²_direct everywhere (24.5 vs 24.3),
peak fidelity equal (scale ≤ 0.12 σ_eff, res ≤ 3.6%). Cost: the
unmodelled few-% continuum lifts lowM median χ² 2–3 → 4–15.
**Conclusion.** `puredcb` ADOPTED (2026-08-12). **fsig is not an
interpolable quantity** in a category where it saturates: its point fits
zigzag 5–13% with claimed errors ±0.3% (DCB-tail degeneracy artifacts).
Background only where the continuum is structural: `channel_has_bkg()` ==
SR3Mu_highM.

### S11 — Shape parametrizations become (mHc, mA) surfaces

**Motivation.** Y7 showed the yield's G improves dramatically as a joint
surface; the shape parameters face the same sparse-grid problem.
**Setup.** Each shape parameter fitted as one error-weighted
total-degree-truncated tensor polynomial in scaled ((mHc−115)/45,
(mA−70)/70), mHc degree ≤ 2, total ≤ 4 (12 coefficients,
`SHAPE_SURFACE_DEGREES`), across all seven studies, sliced at each
study's mHc; fsig in logit space. LOO protocol on x0/σL/σR. Two controls:
(a) 1D per-study fits with wider mA orders; (b) the surface with its mHc
dependence flattened out.
**Results.** Worst-case scale error halves (0.344 → 0.172 σ_eff), p90
0.064 → 0.045; res p90 0.033 → 0.026. Control (a): scale max stays 0.344
and res max nearly doubles (0.115 → 0.264) — extra mA freedom is not the
gain. Control (b): worse (scale p90 0.065) — the mHc dependence is real.
Fairness checks: denying the surface every point within ±5 GeV in other
studies changes nothing (>10% rate 3.5% either way — it is not reading
off a near-duplicate); removing the target's whole study degrades badly
(median 4.1%, p90 17.7% — it does need that study's points, i.e. this is
**not** an mHc-interpolation licence).
**Conclusion.** Adopted. Because the slice of a surface at fixed mHc is a
polynomial in mA, `polynomials.json` keeps the plain coeffs+cov record
and nothing downstream changed. `POLY_ORDERS` and the per-parameter
ladders retire. This introduces the `polynomials` cross-study barrier
([WORKFLOW.md](WORKFLOW.md)).

### S12 — Positive-definite Bernstein background rejected

**Motivation.** Rare negative-Chebychev tails (all in
production-irrelevant corners; visible symptom only a plot-time
χ² = nan).
**Setup.** Degree-2 Bernstein with b₁,b₂ ≥ 0
(`bern_fixedn_logitfsig_ftest`) vs Chebychev₂, MHc160.
**Results.** Bernstein converges more often (bad direct fits 25 → 7 at
floating-n) but describes the combinatoric shape worse where it matters:
direct-fit medians degrade in 7/8 SR3Mu categories (lowM_Run2 2.4 → 4.6)
because the positivity bound is active where the data want the
Chebychev's freedom; the b₁=b₂=0 boundary pileup degrades the
coefficient-vs-mA interpolation (production-relevant held-out closure
median 2.46 → 2.85, worst 6.3 → 19.8).
**Conclusion.** Rejected. Negative Chebychev tails are rendered inert by
the existing `BIN_FLOOR_VALUE` floor at template-construction time.

---

## Y. Yield model

Yields are interpolated per sub-era (the datacard's signal columns are
per-era components whose `rate = -1` reads the histogram integral).
Yield definition: Σw(Central) inside the production window
`[max(x0 − 10σ_eff, 12), x0 + 10σ_eff]` with x0, σ_eff from the
**interpolated** shape model (`interp_window`) — smooth in mA, computable
without MC, and exactly the number the parametric template is normalized
to.

### Y1 — Direct per-era log-poly fit of N_win rejected

**Motivation.** Simplest possible model: one log-space polynomial per
(sub-era, channel).
**Setup.** Orders [1,2,3] × [2,3,4,5] F-test ladder.
**Results.** Failed twice over: Run3 residuals to 40% (per-sample
normalization scatter dragging the curve — see Y6), and the highM sigmoid
turn-on unfittable by any low-order polynomial.
**Conclusion.** Factorize: the ratio f_window = N_win/N_total cancels the
normalization noise and isolates the sharp structure.

### Y2 — Per-period error floors on N_total points

**Motivation.** MC-stat-only weights made every fit chase sample noise
(χ²/ndf to 400).
**Setup.** Run2 2%, Run3 8% floors on the N_total points.
**Results / conclusion.** Fits stabilized; adopted. (The floors encode
the Y6 scatter, not the model.)

### Y3 — Physics-structured model N = k_era · G_period(mA) · f_category(mA)

**Motivation.** Each factor should be a quantity with a physical reason
to be simple: at fixed mHc the b-jet efficiency is flat in mA, the ±10σ
window always holds the SR1E2Mu peak, and the SR3Mu split is pure pairing
combinatorics (exactly one OS pairing is the true A→μμ pair, so
p_low + p_high = 1).
**Setup.** Experiments E1–E4: f measured per era and per period; S ≡
f_low + f_high and p_high ≡ f_high/S tested for smoothness; era shares
k_era measured vs mA. Model: G_period log-space poly on period-summed
totals (summing four eras halves the sample scatter), f era-independent,
f_SR3Mu = S·p with p_high a logit-space poly.
**Results.** f is era-independent (spread ≤ 0.9% abs) and flat for
SR1E2Mu (pol1 χ²/ndf ≈ 4/8); S is constant within ±4% while f_high spans
×80; Run2 era shares flat at ~1% (rms ≤ 1.3%). Held-out closure equal or
better everywhere with ~10× fewer parameters — most dramatically
SR3Mu_highM: MHc145 held-out Run2 max 9.3% → 1.5%.
**Conclusion.** Adopted, and still the production structure (with Y7/Y8
refinements).

### Y4 — G orders forced to a cubic minimum [3,4]

**Motivation.** log N_total is a rise to a maximum near mA ≈ mHc − mW and
an asymmetric fall as the W* phase space closes — genuinely cubic-like.
**Setup.** From-pol2 F-test ladders vs forced [3,4] minimum.
**Results.** The ladders stalled at pol2 and left 5–9% held-out misses at
sparse steep-fall grid gaps.
**Conclusion.** Cubic minimum adopted. (Superseded by Y7 — G is now a
surface.)

### Y5 — Widened yield ladders rejected

**Motivation.** Check the model is not order-starved.
**Setup.** S/G/f_SR1E2Mu ladders +1 order.
**Results.** The F-test took the extra order in 3 places with large
in-sample χ² drops (MHc160 Run3 S: 22.6/6 → 1.9/4) but held-out closure
unchanged (±0.3%); G never took pol5.
**Conclusion.** Residuals are per-sample normalization scatter, not model
stiffness. Rejected.

### Y6 — Run3 per-sample normalization scatter (upstream finding)

**Motivation.** Persistent 10–20% Run3 closure residuals that no model
change touched.
**Setup.** Residuals compared across channels at fixed (era, mass point);
raw-skim bookkeeping inspected.
**Results.** Run3 signal samples scatter ±10–20% around any smooth
acceptance curve, nearly identically in SR1E2Mu and SR3Mu for the same
(era, point) — a sample-level effect, not channel physics. Raw-skim
sizes/entries vary ×4 between adjacent mass points with the mean
per-event weight tracking 1/N_generated, so the bookkeeping only
partially compensates. Preprocessing is faithful; the issue is upstream
(SKNano / sample production) and equally affects the production binned
templates. Run2 is clean at 1–2%.
**Conclusion.** Not fixable here. The smooth fitted curve is arguably a
better acceptance estimate than any single noisy sample; the scatter is
absorbed into the Run3 norm nuisances
([UNCERTAINTY.md](UNCERTAINTY.md)) and this finding is why they exceed
10%. *(Mechanism re-diagnosed 2026-08-19, see Y9: the weight bookkeeping
is exact — "only partially compensates" is disproven — and the scatter
is genuine acceptance structure. The absorption stays correct.)*

### Y7 — G(mA) and k_era become (mHc, mA) surfaces (`joint`)

**Motivation.** Factorizing the LOO residual showed every large
production-pairing miss is a G failure (|dG| to 23%, |df| ≤ 8%) — and NO
alternative 1D basis fixes it: pol up to 6, log-mA, cubic spline, PCHIP
and linear all tie or lose against pol[3,4]. It is grid sparsity: MHc100
has 8 points, MHc115 jumps 15 → 27 → 42, so dropping one point swings the
cubic ±20% with alternating signs.
**Setup.** One G surface across all studies (total-degree-4,
12 parameters vs ~35 for seven cubics; `JOINT_G_DEGREES`), sliced per
study. k_era: a plane in (mHc, mA) pooled across studies
(`JOINT_K_DEGREES` = (1,1), 3 coefficients vs 28 constants), shares
renormalized to sum to 1. Control: per-study pol0/pol1 k_era in mA.
**Results.** LOO yield failures above 10%: 8.6% → 3.5% of cells; p90 at
mA ≤ 45: 16.1% → 7.7%. k_era shares carry a real smooth mHc drift
(2018/SR1E2Mu 0.4280 → 0.4346 across mHc); envelopes above 10%
51/152 → 43/152. The control shows the gain is the pooling, not the mA
freedom (73 → 75, a wash).
**Conclusion.** Adopted. Introduces the `yield_model` cross-study
barrier.

### Y8 — SR3Mu pairing decomposition drops its /fsig division

**Motivation.** The original decomposition f_v = S·p_v/fsig_v coupled the
yield model to the shape chain — which pure-DCB lowM (S10) cannot
support: lowM has no fsig at all.
**Setup.** S = f_low + f_high and p_high = f_high/S on the RAW measured
fractions; both forms are exact reparametrizations of (f_low, f_high).
Smoothness compared on 14 datasets.
**Results.** A wash (helps 0, hurts 1, mixed 13); the one loss is MHc145
Run3, the origin of a +95% lowM yield-closure blow-up under /fsig.
**Conclusion.** Raw-fraction decomposition adopted; yield chain decoupled
from the shape chain.

### Y9 — Y6 mechanism re-diagnosed: bookkeeping exact, scatter is real acceptance structure (2026-08-19)

**Motivation.** Y6 attributed the 10–20% Run3 closure scatter to sample
production bookkeeping that "only partially compensates" the ×4
raw-skim size variations. That hypothesis was never verified against
the produced files. AN review (v15) forced the question.

**Setup.** Four independent audits against the actual production, all on
the `Run3_v13_Run2_v9` inputs the analysis uses:

1. *File-level bookkeeping*: for every signal sample of all four Run3
   eras (4 × 78), sum `Runs.genEventCount` / `genEventSumw` /
   `genEventSumw2` over every file in the ForSNU `path` list (uproot on
   the storage host, `/gv0/DATA/SKNano/Run3NanoAODv13p1/`) and compare
   with the stored `nmc` / `sumW`; effective sample size
   n_eff = (Σw)²/Σw².
2. *filterEff reproducibility*: cross-era max/min per mass point vs the
   binomial expectation from ~300k generated events.
3. *Production-pipeline correlation*: LOO |residual| split by sample
   type — central-style crab (`..._13p6TeV_madgraph-pythia8`) vs local
   `_cff` productions (30 of 78 points, multi-batch merges included).
4. *Gen-level content*: fraction of post-filter events with ≥3 status-1
   e/μ (pT > 8, |η| < 2.6) plus the gen dimuon mass median, for the
   mA = 15 line across mHc in 2022/2023BPix **and 2018** (Run2
   reference); per-point LOO residuals decomposed into period averages
   (production pairing only).

**Results.**

- Bookkeeping is EXACT everywhere: `nmc` matches to 0, `sumW` to
  ≲1e-8 (float summation order), zero missing/broken files, and
  n_eff/nmc > 0.9999 in all 312 (era, sample) cells — including every
  multi-batch merge. Total normalization is lumi × xsec × filterEff to
  machine precision; Y6's "bookkeeping only partially compensates" is
  **disproven**.
- filterEff cross-era spread: max 0.61%, median 0.23%, zero points
  above 3× the statistical expectation.
- Production pipeline: mean |rel| 5.0% (cff) vs 6.3% (crab) — no
  correlation; the pipeline mix is not the cause.
- Gen content: the worst point, MHc100_MA15 (−28% Run3 closure), has
  IDENTICAL post-filter lepton content in 2018 / 2022 / 2023BPix
  (proxy 0.7176 / 0.7178 / 0.7170) and the correct mass (medM 15.02).
  Its ~8% dip off the mA15-line proxy trend is present in BOTH periods:
  physical structure at the W-on-shell threshold (mHc − mA = 85 GeV),
  not a Run3 production defect.
- Residual decomposition (156 production-pairing points): rms 5.5%
  (Run2) vs 6.4% (Run3), Run2–Run3 correlation **r = +0.78**,
  rms(Run3 − Run2) = 4.1%. The largest residuals are SHARED with the
  same sign — MHc100_MA15 (−31%/−28%), MHc130_MA15 (+26%/+22%),
  MHc115_MA27 (−17%/−15%) — all in the low-mA turn-on / W-threshold
  region. The Run3-specific tail (MHc145_MA120 SR1E2Mu: −0.5% Run2 vs
  −14.9% Run3; MHc160_MA105: −0.1% vs −8.4%; also 70/65, 100/90,
  115/100) clusters in the far off-shell-W region (mHc − mA < m_W),
  consistent with the 13.6 TeV gridpacks (newer MadGraph) producing
  genuinely different W* kinematics than the Run2 ones, on top of the
  harsher Run3 reco selection (PUPPI + b-tag; A×ε roughly half of
  Run2's, visibly jagged in the AN's SigEff table).

**Conclusion.** The Run3 scatter is genuine acceptance structure that a
smooth surface cannot follow: mostly threshold/turn-on physics shared
with Run2, plus a Run3-specific off-shell-W component — NOT a
normalization error. Everything downstream of Y6 stands: the residual
is a property of the reference samples, equally affects direct-MC
templates, and is correctly absorbed into the norm nuisances. Only the
mechanism wording changes ("per-sample normalization scatter" →
"per-sample acceptance scatter"). Ruled out for the record: sumW/nmc
bookkeeping, weight heterogeneity (n_eff), filterEff bookkeeping,
production pipeline mix, jet veto maps (era-specific, ~3%, wrong
signature vs the era-common residuals).

---

## V. Validation

### V1 — Parametric templates vs direct MC at a held-out point (four arms)

**Motivation.** The models are only useful if a datacard built from them
gives the same limit as one built from signal MC.
**Setup.** MHc160_MA90, `--era All --channel Combined`, unblind; every
model input refitted WITHOUT mA = 90, so the target is a genuine
interpolation. Four arms sharing backgrounds/nonprompt/data from the
production code path — only the signal columns differ:

| arm | signal | window/binning | interp. nuisances |
|---|---|---|---|
| A | direct MC | direct DCB fit | — |
| B0 | parametric | arm A's | — |
| B | parametric | interpolated | — |
| C | parametric | interpolated | yes |

Signal shape systematics without target MC: each systematic compressed on
the *other* mass points of the same mHc into three dimensionless deltas
vs Central — dm (core-window mean), dsig (core-window rms), dN (window
sum) — parametrized in mA excluding the target, applied as
x0 → x0(1+dm), σ → σ(1+dsig), N → N(1+dN). Weight-only trees are paired
differences; kinematic trees fall back to the uncorrelated error.
**Results.** Limits (tight-tolerance rerun, `--rRelAcc 0.0005`):

| | A | B0 | B | C |
|---|---|---|---|---|
| expected median r | 0.9590 | 0.9258 | 0.9258 | 0.9258 |
| observed r | 0.6378 | 0.6170 | 0.6498 | 0.6519 |
| signal yield | 89.98 | 93.13 | 92.89 | 92.89 |

- A↔B0 (the method): expected −3.5%, observed −3.3% — entirely
  normalization (+1.0/+2.3% Run2, +12.1/+6.6% Run3, inside the known Y6
  scatter); shape L1 distance ≤ 3.1%.
- B0↔B (window/binning): interpolated window reproduces the direct one
  (x0 to ~0.01 GeV, σ_eff to ~2%), same 17 bins everywhere; expected
  unchanged, observed +5.3% (real-data sensitivity to sub-percent bin-edge
  shifts).
- B↔C (nuisances): expected unchanged, observed +0.3%, despite per-bin
  shape variations up to 8% (scale) / 17% (res) — the analysis is
  background-statistics limited.
- Delta closure against the target's own MC (6576 series): median |Δ|
  1.4e-7 (dm), 7.3e-6 (dsig), 1.7e-5 (dN); worst JES dN −6.3% vs −4.5%.
**Conclusion.** A parametric signal template at a held-out point
reproduces the direct-MC limit to a few percent; the residual is a
normalization difference of the size of the known Run3 scatter, not a
shape or method failure. Interpolation nuisances cost nothing.
Templates must be built one (period, channel) category per condor job
(four in one process exceeds 8 GB).

### V2 — The LOO protocol is fair, and mHc interpolation is out of scope

**Motivation.** Two opposite worries about the surface models: that LOO
flatters them (reading off a near-duplicate point from another study), or
that the surface licences interpolating in mHc.
**Setup.** (a) LOO denying the surface every point within ±5 GeV in mA in
OTHER studies; (b) leave-one-STUDY-out: predict a whole mHc study from
the other six.
**Results.** (a) No change: >10% failure rate 3.5% either way. (b) Median
4.1%, p90 17.7% yield error.
**Conclusion.** The LOO numbers are honest, and interpolation stays
**in mA only**.

### V3 — Uncertainty rule and its evidence

See [UNCERTAINTY.md](UNCERTAINTY.md) — the rms-then-max rule, the mA
binning of norm, the floors, and the residual-correlation study that
justifies pooling by rms.
