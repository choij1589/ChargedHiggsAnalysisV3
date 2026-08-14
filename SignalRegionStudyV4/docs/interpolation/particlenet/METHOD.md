# ParticleNet Interpolation — Method

How parametric signal templates are built for the **ParticleNet** method, and
the decision gates that must pass before the model is used. The ParticleNet
arm is a thin layer on top of the frozen Baseline interpolation, not a
parallel chain — read [../WORKFLOW.md](../WORKFLOW.md) first; this file
records only the delta. Nuisances: [UNCERTAINTY.md](UNCERTAINTY.md).

Status: **feasibility complete and the model frozen (2026-08-14)**. Template
production (the analogue of `automize/interpTemplates.sh`) is not built yet.

## Why a separate layer is needed

A ParticleNet template carries a score cut. The cut is defined per
template-sharing group, and it changes both the signal mass shape and the
normalization — neither of which the Baseline chain models. Two questions had
to be answered before any of it could be reused:

1. Does the Baseline signal shape survive the cut? (The nets are trained with
   the dimuon mass decorrelated, so it should.)
2. Can the normalization be factorized into (Baseline window yield) x
   (threshold efficiency), so only the efficiency needs a new model?

Both hold. The measured sizes are in [UNCERTAINTY.md](UNCERTAINTY.md).

## Production model (frozen 2026-08-14)

| piece | model |
|---|---|
| seeds | mA = 85, 90, 95 per mHc — the mass points the nets are trained at |
| groups | nearest seed, +-2.5 GeV; 3 groups per mHc, 15 in total |
| grid | 0.5 GeV steps, mA in **[82.5, 97.5]**; 31 points per mHc, 155 in total |
| outside the reach | Baseline templates only — the two arms coexist as separate methods |
| score | `s_sig / (s_sig + w_np*s_np + w_db*s_db + w_ttX*s_ttX)`, weights from the SEED's mass window |
| working point | fixed **eps_B = 20%**, one threshold per (channel, run period, seed) |
| backgrounds | built once per group by the seed, shared by every member |
| shape | Baseline surfaces (`fits/{mHc}/polynomials.json`) reused **verbatim** |
| yield | `N = k_era · G_period(mA) · f_category(mA) · eps_seed(mA)` |
| eps(mA) | quadratic through the seed net's three anchors at mA = 85/90/95 |
| interpolation | **mA only**, at the five ParticleNet mHc (100–160 in steps of 15) |
| everything else | identical to the Baseline method (low-stat rules, run-period components, blinding) |

## Group construction

A grid point joins its **nearest seed**. With seeds 5 GeV apart and a 0.5 GeV
lattice, the boundaries fall exactly halfway: mA = 87.5 and 92.5 are
equidistant from two seeds and are assigned to the **lower** seed, so grouping
is reproducible rather than dependent on floating-point comparison order.

The reach is set by the trained grid, not by choice: seeds exist only at
85/90/95, and +-2.5 GeV around them tiles [82.5, 97.5] exactly with no gap.
An earlier +-2 GeV proposal would have left 87.5 and 92.5 uncovered.

**Every seed-member pair used in production has |dmA| <= 2.5.** This is not a
detail — the seed's mass window is `[max(x0 - 10σ, 12), x0 + 10σ]` around the
SEED's peak, so a member far from its seed has its own peak clipped by that
window. Scanning the full seed x member matrix shows an ~85% yield residual at
(seed 85, member 95); it is pure window geometry, and that pair never occurs.
Any diagnostic that reports the unrestricted matrix will look alarming and
mean nothing.

## Score and working point

The score is a background-composition-weighted likelihood ratio: the three
background class scores are weighted by the expected per-class weight
fractions measured **inside the seed's mass window**, then the signal score is
divided by the total. Implemented by
`template_utils.build_particlenet_score` and
`makeBinnedTemplates.getCategoryBackgroundWeights`; the weights are computed
once by the seed and shared, so every member of a group is cut on exactly the
same quantity.

The threshold is the score giving **eps_B = 20%** in that category, replacing
the Baseline ParticleNet rule of maximizing the Asimov Z on a 101-point grid.
Measured over 68 categories (5 mHc x 2 channels x 2 run periods x 3–4 seeds):

| WP | eps_S med | eps_B med | Z/Z_nocut med | Z/Z_nocut min | **min B** | thr spread in a category | eps_S spread over members |
|---|---|---|---|---|---|---|---|
| argmax Z (old) | 0.511 | 0.145 | 1.314 | 1.101 | **1.6** | 0.095 | 0.0373 |
| eps_B = 10% | 0.409 | 0.100 | 1.263 | 1.021 | 16.7 | 0.038 | 0.0546 |
| **eps_B = 20%** | **0.588** | 0.200 | **1.288** | 1.063 | **33.6** | 0.051 | **0.0334** |
| eps_B = 30% | 0.700 | 0.300 | 1.263 | 1.071 | 50.4 | 0.051 | 0.0251 |

eps_B = 20% costs **~2% of the median significance ratio** and buys three
things:

- **It removes a real pathology.** The argmax-Z rule put 5 of 68 categories on
  1.6–8.8 background events — e.g. MHc115/SR3Mu_Run2 seed 95 at thr = 0.930,
  eps_B = 0.29%, B = 1.6. Its Z = 2.25 "optimum" beats the eps_B=20% Z = 1.885
  only by cutting into a regime where MC statistics dominate and autoMCStats
  would own the datacard. 4 of the 5 were MHc100.
- **It stabilizes the threshold**, 0.095 -> 0.051 spread across a category's
  seeds, because a fixed-eps_B cut is defined by the backgrounds alone — which
  are shared and mass-independent within a group — instead of chasing per-seed
  signal fluctuations.
- **It makes eps easier to interpolate.** The same categories that carried the
  low-B spikes carried the worst efficiency-interpolation error, and freezing
  the WP improved the worst study (MHc100) most: `r_eps` max 0.143 -> 0.074.

Note eps_S spread over a group's members is **not monotonic** in the WP:
eps_B = 10% is the worst of all four, because a tight cut probes the score
tail where eps_S varies most with mA. Tighter is not safer here.

## Shapes

The Baseline shape surfaces are reused **unchanged** — no post-cut refit, no
sigma correction. The nets are mass-decorrelated, and the measured cost of the
assumption is small enough to carry as a nuisance rather than a correction
(see [UNCERTAINTY.md](UNCERTAINTY.md), Gates U1/U2).

The DCB is therefore evaluated exactly as in the Baseline arm; only the
normalization is rescaled by eps.

## Yields

```
N(era, mA) = k_era · G_period(mA) · f_category(mA) · eps_seed(era, mA)
             └──────── Baseline yield_model.json ────────┘  └── new ──┘
```

`eps_seed` is measured under the **seed's** net at the **seed's** threshold,
and interpolated in mA by a quadratic through the three anchors at 85/90/95.
Deliberately the simplest basis that can work: with three points 5 GeV apart
there is nothing to gain from a richer form, and a failure should be visible
rather than absorbed.

The residual splits into two independent terms, which must not be conflated:

- `r_model` — the Baseline yield model transferring into the ParticleNet
  sample dirs. Already covered by the Baseline `CMS_interp_norm`.
- `r_eps` — the efficiency interpolation. The **only** genuinely new error,
  and the only one the ParticleNet eff nuisance may cover.

## Closure against MC

The end-to-end check: build the interpolated template exactly as production
would (Baseline shape surface sliced at the target mA, absolutely normalized
by `k_era · G · f · eps`) and overlay it on the direct MC of the same
ParticleNet mass point, above the same threshold. Absolute normalization, not
shape-only — the point is to test the yield model at the same time. The ratio
panel is **interp / MC**, and the band on the interpolated template is the
assigned uncertainty of [UNCERTAINTY.md](UNCERTAINTY.md): `CMS_interp_norm`
(+) `CMS_interp_eff_pnet` on the normalization, and `CMS_interp_res_pnet` as a
per-bin shape variation.

`eps` is the **production** quadratic through all three anchors — no
leave-one-out. LOO belongs to uncertainty derivation, where a point must not
inform its own error; the closure plot asks a different question, namely
whether the frozen model reproduces MC.

| set | n | interp/MC median | \|dev\| p90 | max | χ²/ndf MC-stat only | χ²/ndf + assigned |
|---|---|---|---|---|---|---|
| anchors (mA = 85/90/95) | 60 | 0.9990 | 0.057 | 0.091 | 4.97 (p90 9.27) | **1.80** (p90 3.20, max 4.13) |
| validation (MA87, MA92) | 8 | 0.9760 | 0.076 | 0.082 | 6.88 (p90 13.59) | **2.37** (p90 2.77, max 3.09) |

The two sets test different things, and the anchors test **less** than they
look like they do: a 3-point quadratic passes exactly through its anchors, so
at mA = 85/90/95 `eps_pred == eps_measured` and the closure collapses onto the
Baseline yield model (`r_model`) alone. Their agreement says the Baseline model
transfers cleanly into the ParticleNet sample dirs; it says nothing about the
new layer.

The eight MA87/MA92 points are the only genuine test of the eps interpolation
(ndf = 30 each):

| point | interp/MC | χ²/ndf (+assigned) |
|---|---|---|
| MHc115/SR1E2Mu_Run2/MA87 | 0.9969 | 2.63 |
| MHc115/SR3Mu_Run2/MA87 | 0.9949 | 3.09 |
| MHc145/SR1E2Mu_Run2/MA92 | 0.9856 | 2.27 |
| MHc145/SR3Mu_Run2/MA92 | 1.0133 | 2.51 |
| MHc115/SR1E2Mu_Run3/MA87 | 0.9260 | 2.09 |
| MHc115/SR3Mu_Run3/MA87 | 0.9364 | 2.46 |
| MHc145/SR1E2Mu_Run3/MA92 | 0.9185 | 1.56 |
| MHc145/SR3Mu_Run3/MA92 | 0.9663 | 1.55 |

Run2 closes to within 1.5% on all four. The 3–8% Run3 deficit is the inherited
Baseline `r_model` (6.6% median in Run3, from the upstream per-sample yield
scatter recorded in `docs/interpolation/WORKFLOW.md`), not the eps layer, whose
own contribution is ~0.5–1% — which is why it is `CMS_interp_norm` that covers
it and not the ParticleNet `eff` nuisance.

The χ² columns are the reason the band matters: 4.97 -> 1.80 and 6.88 -> 2.37
once the assigned uncertainty is included. Values near 2 over 30 bins with **no
fitted parameters** are what a fixed model overlaid on independent MC should
give; anchors and validation landing at the same place is the consistency
check.

## Procedure

Deriving and re-verifying the model. Study code lives in
`test/pnet_interp/` (gitignored scratch); production template building is not
implemented yet.

```bash
# Samples: one dir per mHc holding every trained mA with EVERY net's score
# branches, plus one shared background/nonprompt/data set. 3-4x cheaper than
# the per-masspoint layout, which duplicates backgrounds once per mA.
python3 python/preprocess.py --era 2017 --channel SR1E2Mu \
        --shared-scores --mhc MHc115 [--central-only]

# Working points (must run first -- the other two read its thresholds).
python3 test/pnet_interp/thresholdWP.py --mhc MHc115

# Shape and yield checks at the frozen WP.
python3 test/pnet_interp/shapeReuse.py  --mhc MHc115 --wp 'epsB=20%'
python3 test/pnet_interp/yieldInterp.py --mhc MHc115 --wp 'epsB=20%'

# Merged report, then the derived nuisances.
python3 test/pnet_interp/summarize.py
python3 test/pnet_interp/exportPnetUncertainties.py

# End-to-end closure: interpolated template vs direct MC, absolutely
# normalized, band = the assigned nuisances. 68 plots + one shard per mHc.
python3 test/pnet_interp/closPnetTemplates.py --mhc MHc115
```

On condor: `test/pnet_interp/submit.sh {preprocess-ttz|studies}`.
`--central-only` writes the nominal tree alone and drops a `CENTRAL_ONLY`
marker — study use only, **never** an input to a datacard.

## Decision gates

**Gate 0 (inputs)** — the per-mHc `*_NoHistMode` skims must be complete, and
every file must carry the score branches of **every** trained mA of its mHc
(that is what makes cross-scoring possible without re-production). Verify with
`test/pnet_interp/auditScoreBranches.py`. Also confirm standard vs NoHistMode
entry counts agree; 41 deficient files were found on 2026-08-07 and are fixed,
but the check is cheap and the failure mode is silent under-counting.

**Gate A (after each condor stage)** — "the jobs finished" is not "they
worked":

```bash
grep -h "EXITING WITH STATUS" test/pnet_interp/logs/*.out | sort | uniq -c
```

Then confirm no shard is stale: every entry must carry the working point you
intended. Mixing an old shard into a summary is the easiest mistake to make
here, because job completion alone will not reveal it.

```bash
python3 -c "import json,glob;print({f: next(iter(json.load(open(f))['results'].values())).get('wp') for f in glob.glob('test/pnet_interp/*.MHc*.json')})"
```

**Gate B (working points)** — from `summarize.py`: **no category may sit on a
low-statistics spike.** The `B min` column must stay well above the handful of
events where MC statistics take over; eps_B = 20% gives 33.6. A category
appearing in the "low-statistics spikes" list is a bug in the threshold rule,
not an unlucky point.

**Gate C (shape closure)** — `|d_scale|` and `|d_res|` per (channel, period)
against the Baseline `CMS_interp_scale/res` in the same cell, and against the
refit statistical error. Interpretation is Gate U1 in
[UNCERTAINTY.md](UNCERTAINTY.md).

**Gate D (yield closure)** — the decomposition must show `r_eps` small and
`r_model` carrying the bulk; if `r_eps` approaches `r_model`, the efficiency
model — not the Baseline arm — is the thing to fix. Check `r_eps by mHc`: a
single study driving every cell means the grid is talking, not the model.

**Gate E (template closure)** — `closPnetTemplates.py` over all five mHc. Two
things must hold: the **validation** points (MA87/MA92) close in normalization
at the few-percent level, and χ²/ndf **with the assigned uncertainty** comes
down to O(1–3) from the MC-stat-only value. A validation point that stays high
after the band is applied means the assigned nuisance does not cover the model
error at that point. Read the anchors as a Baseline check, not a ParticleNet
one — eps is exact there by construction.

## Known limitations

- **The reach is [82.5, 97.5] and nothing widens it but new trainings.** The
  nets exist at three mA per mHc; outside that window only Baseline templates
  exist.
- **No mHc interpolation.** Five measured mHc, each interpolated in mA only —
  the same restriction as the Baseline arm, for the same reason.
- **MHc100 is the weakest study**, with the largest `r_eps` (median 0.025 vs
  0.006–0.011 elsewhere) and 4 of the 5 historical low-B spikes. It has the
  smallest signal yields of the five.
- **Only two blind validation points exist** — MHc115_MA87 and MHc145_MA92 are
  the only trained mA off the 85/90/95 lattice, so the genuine
  out-of-sample test rests on them; the other 15 anchors are tested
  leave-one-out, which at the 85 and 95 endpoints is extrapolation rather than
  interpolation.
- **TTZ2E1Mu is preprocessed but is not a datacard channel** in V4 — it feeds
  `plotParticleNetScore.py` and `plotPostfitMass.py` only.
