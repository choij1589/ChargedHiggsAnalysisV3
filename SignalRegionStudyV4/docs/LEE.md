# Look-Elsewhere Effect — Global Significance of the Scan Maxima

How V4 turns the local significances of `docs/SIGNIFICANCE.md` into global
ones, why the method is not the one V3 used, and what the numbers are.

Measured on the frozen production scan (unblind, era `All`, signal source
`interp-signal`): Baseline 2467 points over 7 mHc, ParticleNet 150 points
over 5 mHc, each at All × {Combined, SR1E2Mu, SR3Mu}.

Source of record: `results/json/lee.All.interp-signal.json`, written by
`python/estimateLEE.py` from `results/json/significance.All.interp-signal.json`.
Diagnostic panels: `results/plots/lee/`.

---

## 1. Why the V3 method does not port

V3 ran a brute-force coherent-toy campaign (`SignalRegionStudyV3/docs/LEE.md`):
a fine-binned background density, 1000 event-level Poisson pseudo-datasets,
each projected into every mass point's frozen binning and fitted with
`combine -M Significance`, then the distribution of `Z_max` over the trials
set. 35 points × 1000 toys = 35k fits ≈ 85 CPU-h. Result:
`p_global = 0.130 ± 0.011`, `Z_global = 1.13` for a local `Z = 2.06`, Baseline
only, All/Combined only, trials set restricted to `mHc < 100 OR mA < 60`.

That was the right object **for that grid**. V3's 78-point MC grid is spaced
far above the dimuon mass resolution — at mHc 160 the mA values are
15, 17, 20, 25, 30, … while σ_eff ≈ 0.16 GeV at mA = 17.5 — so every tested
point is a statistically independent test, the tested set *is* the trials set,
and enumerating it is both correct and affordable.

V4's grid is the opposite. `configs/grid.json` fixes the lattice **at** the
resolution by construction: bands of 0.1 / 0.25 / 0.5 / 1.0 GeV chosen so that
`step / σ_eff,min` stays below 1 everywhere (worst 0.86–0.89, at mA = 30). Two
consequences:

1. **Counting points is meaningless.** Neighbouring scan points share the same
   events and differ by well under one resolution width. Halving the lattice
   spacing would double the "number of tests" without adding a single
   independent one. Any trials estimate proportional to the point count is an
   artifact of the lattice, not a property of the search.
2. **The toy campaign is unaffordable.** 2467 points × 3 channels × 1000 toys
   ≈ 7.4M `Significance` fits ≈ 68,000 CPU-h — 800× V3's campaign, for a scan
   whose whole template production cost ~360 core-hours.

What survives the lattice change is how often the scan statistic **crosses a
level**. That is a property of the scanned range and of σ(mA), and it is
exactly what the asymptotic (Gross–Vitells) treatment uses.

## 2. Method

For a search with one parameter present only under the alternative (here mA),
the number of upcrossings of level `u` by the field `q(mA) = Z(mA)²` obeys

```
<N_u(u)> = N_0 · exp(−u/2)
```

and the global p-value of an observed maximum `Z_max` (`u_obs = Z_max²`) is

```
p_global ≈ p_local + <N_u(u_obs)>          (first order)
p_global  = 1 − (1 − p_local)·exp(−<N_u>)   (used here)
```

The second form is what `estimateLEE.py` stores as `p_global`: it counts the
*probability* of at least one excursion rather than the *expected number* of
them, agrees with the first order-by-order, and stays a probability when
`<N_u> ~ 1` — which happens for the weaker column maxima, where the linear form
runs past 1. Both are written to the JSON (`p_global`, `p_global_linear`).

Both tails are corrected, not just the excess: the deficit is the maximum of
the reflected curve −Z, and a downward fluctuation is an upcrossing of that
reflected field, so the identical calculation applies. `estimateLEE.py` stores
an `excess` and a `deficit` block per column with the same fields.

`N_0` is measured from the observed scan by counting upcrossings at a ladder of
low thresholds, Z = 0, 0.5, 1.0, 1.5, 2.0, and taking the median of
`N_u(z)·exp(z²/2)`. Low thresholds are used because the crossing sample is
largest there and the law is being extrapolated *upward* to `Z_max`; `N_u(0)` is
scale-invariant and anchors the ladder. The threshold-to-threshold spread is
carried through into `z_global_range`.

**Cross-checks written alongside every column:**

- `p_global ≥ p_local` — a trials correction can only decrease significance.
- `p_global ≤ 1.2 × Šidák`, where the Šidák bound
  `1 − (1 − p_local)^N_res` uses `N_res = ∫ dmA / σ_eff(mA)`, the number of
  resolution elements in the scanned range (≈150–224 per Baseline column,
  from the `fits/MHc{X}/polynomials.json` surfaces). Šidák treats every element
  as an independent test, so a correlated scan must land below it; the 20%
  tolerance is far inside the ladder's own spread (a factor ~2 between
  its highest and lowest threshold estimate, median over the 36 columns).
- The `upcrossings.*` panels show `N_u` against `u` on a semilog axis. A
  straight line is the asymptotic law holding; curvature would be the warning
  that the extrapolation to `Z_max` is not supported.

## 3. Scope — what the global p-value covers

**Frozen decision: the mA scan alone, evaluated per (arm, channel, mHc)
column.** A quoted `Z_global` corrects for having scanned mA in that one
column, and for nothing else.

This is the natural unit here because mA is the only continuous search
direction — the model does not interpolate in mHc, and the arms and channels
are published as separate results. It is *not* the whole search, and the
difference matters, so the extra multiplicity is bounded rather than ignored,
from the measured correlations of the scan curves:

| axis | measured correlation of Z(mA) | effective extra trials |
|---|---|---|
| the 7 mHc columns, same channel | +0.84 … +1.00 (SR1E2Mu +0.98–1.00) | ≈ 1 |
| SR1E2Mu vs SR3Mu, same mHc | −0.13 | ≈ 2 |
| Combined vs SR1E2Mu / SR3Mu | +0.78 / +0.50 | — |
| Baseline vs ParticleNet, in the PN window | +0.3 … +0.8 | ≈ 1 extra |

The seven mHc columns share one dataset and, at fixed mA, nearly the same
signal shape: their maxima at mA ≈ 18 are one observation, not seven. The two
exclusive channels are genuinely independent. Folding all of it in — one number
for the entire search — would cost the Baseline maximum roughly a further
0.3–0.4σ and the ParticleNet maximum about 1.4σ. Those figures are the honest
upper bound on the correction; the tabulated numbers below are the per-scan
ones.

## 4. Results

Measured 2026-08-18 on the complete exact scan: 7851/7851 (point, channel)
`Significance` fits, zero failures.

| | arm | point | channel | Z local | p local | N_0 | trials factor | p global | **Z global** |
|---|---|---|---|---|---|---|---|---|---|
| largest excess | Baseline | `MHc145_MA17p7` | SR1E2Mu | **+3.75** | 8.8e-05 | 21 | 209 | 0.018 | **+2.09** |
| largest excess | ParticleNet | `MHc115_MA82p5` | SR3Mu | **+2.92** | 1.7e-03 | 2 | 15 | 0.027 | **+1.93** |
| largest in Combined | Baseline | `MHc145_MA17p7` | Combined | +2.93 | 1.7e-03 | 25 | 171 | 0.293 | +0.54 |
| largest in Combined | ParticleNet | `MHc130_MA82p5` | Combined | +2.79 | 2.7e-03 | 2 | 13 | 0.034 | +1.82 |

`Z global` carries a ±0.1–0.4 band from the `N_0` threshold ladder
(`z_global_range` in the JSON; ±0.08 for the Baseline maximum, ±0.2 for the
ParticleNet one).

**The scan found larger local maxima than the quoted points.** The extremes in
`docs/SIGNIFICANCE.md` were picked by a band-pull sweep and fitted at a handful
of points; the exact scan fits everywhere and lands on neighbouring lattice
points that were never fitted:

| | previously quoted | exact scan |
|---|---|---|
| largest excess | `MHc145_MA17p5` SR1E2Mu, +3.02 | `MHc145_MA17p7` SR1E2Mu, **+3.75** |
| largest deficit | `MHc160_MA39` SR3Mu, −2.61 | `MHc100_MA44p5` SR1E2Mu, **−3.70** |
| largest Combined excess | `MHc115_MA18`, +2.41 | `MHc145_MA17p7`, **+2.93** |
| largest Combined deficit | `MHc160_MA39`, −2.33 | `MHc160_MA39p5`, **−2.85** |

Both directions moved by a similar amount, which is what a trials-dominated
scan looks like: sampling a Gaussian field more finely raises the observed
maximum *and* deepens the observed minimum, and the trials correction pays for
both. The mA ≈ 17.7 excess is unchanged in character — eμμ, present in all
seven mHc columns (Z local +3.56 to +3.75), diluted in Combined by μμμ, and
the μμμ maximum sits elsewhere, at mA ≈ 23.3–23.4.

Comparison with V3: local 2.06 → global 1.13 there, local 3.75 → global 2.09
here. V4's finer scan buys a larger local maximum and pays a much larger trials
factor (209 against V3's ≈6.5), and the two roughly cancel: **the global result
is comparable to V3's, not qualitatively better, despite a local significance
1.7σ higher.** That cancellation is the whole point of doing this
lattice-independently.

## 4b. Per (arm, mHc), Combined channel — both tails

The channel the analysis actually publishes is `Combined`, so this is the
table to read for "what does each mHc study show, after trials". One mA
scan per row; the trials correction is the one for that scan alone (§3).
Deficits are corrected the same way, by running the identical calculation
on the reflected curve −Z: a deficit is an upcrossing of the reflected
field. `equiv.` is the equivalent significance of the global p; where the
global p exceeds 0.5 there is nothing left to quote.

**Excesses**

| arm | mHc | mA* | Z local | p local | N_0 | trials | p global | equiv. global |
|---|---|---|---|---|---|---|---|---|
| Baseline | 70 | 17.2 | **+2.59** | 4.8e-03 | 15 | 84 | 0.407 | **0.23σ** |
| Baseline | 85 | 17.7 | **+2.61** | 4.5e-03 | 17 | 96 | 0.434 | **0.17σ** |
| Baseline | 100 | 17.7 | **+2.69** | 3.6e-03 | 19 | 113 | 0.408 | **0.23σ** |
| Baseline | 115 | 17.7 | **+2.74** | 3.1e-03 | 20 | 125 | 0.380 | **0.31σ** |
| Baseline | 130 | 17.7 | **+2.72** | 3.3e-03 | 22 | 129 | 0.421 | **0.20σ** |
| Baseline | 145 | 17.7 | **+2.93** | 1.7e-03 | 25 | 171 | 0.293 | **0.54σ** |
| Baseline | 160 | 17.7 | **+2.88** | 2.0e-03 | 25 | 163 | 0.325 | **0.45σ** |
| ParticleNet | 100 | 82.5 | **+2.38** | 8.7e-03 | 1 | 8 | 0.069 | **1.48σ** |
| ParticleNet | 115 | 82.5 | **+2.75** | 3.0e-03 | 2 | 13 | 0.040 | **1.75σ** |
| ParticleNet | 130 | 82.5 | **+2.79** | 2.7e-03 | 2 | 13 | 0.034 | **1.82σ** |
| ParticleNet | 145 | 82.5 | **+1.88** | 3.0e-02 | 1 | 7 | 0.201 | **0.84σ** |
| ParticleNet | 160 | 82.5 | **+2.24** | 1.3e-02 | 2 | 11 | 0.138 | **1.09σ** |

**Deficits**

| arm | mHc | mA* | Z local | p local | N_0 | trials | p global | equiv. global |
|---|---|---|---|---|---|---|---|---|
| Baseline | 70 | 44.75 | **-2.23** | 1.3e-02 | 17 | 59 | 0.757 | **— (p > 0.5)** |
| Baseline | 85 | 72 | **-2.42** | 7.7e-03 | 23 | 91 | 0.702 | **— (p > 0.5)** |
| Baseline | 100 | 44.75 | **-2.33** | 1.0e-02 | 23 | 79 | 0.787 | **— (p > 0.5)** |
| Baseline | 115 | 104 | **-2.58** | 4.9e-03 | 25 | 120 | 0.589 | **— (p > 0.5)** |
| Baseline | 130 | 39.5 | **-2.52** | 5.9e-03 | 28 | 118 | 0.694 | **— (p > 0.5)** |
| Baseline | 145 | 104 | **-2.60** | 4.7e-03 | 31 | 141 | 0.660 | **— (p > 0.5)** |
| Baseline | 160 | 39.5 | **-2.85** | 2.2e-03 | 30 | 184 | 0.402 | **0.25σ** |
| ParticleNet | 100 | 94.5 | **-1.02** | 1.5e-01 | 2 | 5 | 0.742 | **— (p > 0.5)** |
| ParticleNet | 115 | 93.5 | **-0.79** | 2.1e-01 | 2 | 4 | 0.835 | **— (p > 0.5)** |
| ParticleNet | 130 | 92.5 | **-0.90** | 1.8e-01 | 2 | 4 | 0.803 | **— (p > 0.5)** |
| ParticleNet | 145 | 92.5 | **-1.28** | 1.0e-01 | 2 | 6 | 0.628 | **— (p > 0.5)** |
| ParticleNet | 160 | 97.5 | **-2.21** | 1.3e-02 | 2 | 13 | 0.170 | **0.95σ** |

Reading it:

- **No Baseline Combined study survives its own trials factor.** The
  mA ≈ 17.7 excess is the maximum of every one of the seven columns at
  +2.59 to +2.93 local, and every one lands at 0.2–0.5σ global
  (p = 0.29–0.43). The trials factor is 84–184, tracking the scanned
  range: the widest column (mHc 160, mA 15–155) pays the most.
- **The ParticleNet columns are the only ones with anything left**, and
  only because their window is 15 GeV wide instead of 50–140: trials
  factors of 7–13 against the Baseline's 84–184. mHc 130 at mA = 82.5 is
  +2.79 local → **1.82σ global** (p = 0.034), mHc 115 → 1.75σ, mHc 100 →
  1.48σ. The same mass in the Baseline arm is unremarkable, so this is a
  ParticleNet-selection feature, not a mass feature.
- **Every Combined deficit is gone after trials.** The deepest, −2.85 at
  mHc 160, mA = 39.5, has p_global = 0.40. The mA ≈ 39.5 and 44.75 dips
  recur across columns exactly as the excess does, and for the same
  reason: the columns are one dataset seen seven times (§3).
- Excess and deficit extremes are the same size in every column (+2.6/−2.2
  to +2.9/−2.9). A signal would break that symmetry; trials do not.

The equivalent per-arm headline, taking the largest of each tail:

| arm | tail | point | Z local | p global | equiv. global |
|---|---|---|---|---|---|
| Baseline | excess | mHc 145, mA 17.7 | +2.93 | 0.293 | 0.54σ |
| Baseline | deficit | mHc 160, mA 39.5 | −2.85 | 0.402 | 0.25σ |
| ParticleNet | excess | mHc 130, mA 82.5 | +2.79 | 0.034 | 1.82σ |
| ParticleNet | deficit | mHc 160, mA 97.5 | −2.21 | 0.170 | 0.95σ |

Regenerate with `python3 python/estimateLEE.py --channels Combined`
(writes `results/json/lee.All.interp-signal.Combined.json`; a restricted
run gets its own filename so it cannot truncate the full record).

## 5. All 36 columns

**Baseline**

| channel | mHc | points | mA* | Z local | p local | N_0 | trials | p global | Z global |
|---|---|---|---|---|---|---|---|---|---|
| Combined | 70 | 281 | 17.2 | +2.59 | 4.8e-03 | 15 | 84 | 0.407 | **+0.23** [−0.38, +0.37] |
| Combined | 85 | 311 | 17.7 | +2.61 | 4.5e-03 | 17 | 96 | 0.434 | **+0.17** [−0.06, +0.46] |
| Combined | 100 | 341 | 17.7 | +2.69 | 3.6e-03 | 19 | 113 | 0.408 | **+0.23** [−0.13, +0.52] |
| Combined | 115 | 361 | 17.7 | +2.74 | 3.1e-03 | 20 | 125 | 0.380 | **+0.31** [+0.00, +0.54] |
| Combined | 130 | 376 | 17.7 | +2.72 | 3.3e-03 | 22 | 129 | 0.421 | **+0.20** [−0.25, +0.35] |
| Combined | 145 | 391 | 17.7 | +2.93 | 1.7e-03 | 25 | 171 | 0.293 | **+0.54** [+0.42, +0.82] |
| Combined | 160 | 406 | 17.7 | +2.88 | 2.0e-03 | 25 | 163 | 0.325 | **+0.45** [+0.32, +0.61] |
| SR1E2Mu | 70 | 281 | 17.7 | +3.61 | 1.5e-04 | 14 | 131 | 0.020 | **+2.05** [+1.52, +2.09] |
| SR1E2Mu | 85 | 311 | 17.7 | +3.61 | 1.5e-04 | 15 | 143 | 0.022 | **+2.02** [+1.62, +2.10] |
| SR1E2Mu | 100 | 341 | 17.7 | +3.56 | 1.9e-04 | 16 | 151 | 0.028 | **+1.91** [+1.52, +1.94] |
| SR1E2Mu | 115 | 361 | 17.7 | +3.61 | 1.5e-04 | 17 | 163 | 0.025 | **+1.96** [+1.62, +2.02] |
| SR1E2Mu | 130 | 376 | 17.7 | +3.60 | 1.6e-04 | 18 | 173 | 0.028 | **+1.92** [+1.59, +1.97] |
| SR1E2Mu | 145 | 391 | 17.7 | +3.75 | 8.8e-05 | 21 | 209 | 0.018 | **+2.09** [+2.02, +2.18] |
| SR1E2Mu | 160 | 406 | 17.7 | +3.56 | 1.8e-04 | 21 | 198 | 0.037 | **+1.79** [+1.53, +1.91] |
| SR3Mu | 70 | 281 | 23.3 | +2.59 | 4.8e-03 | 15 | 85 | 0.404 | **+0.24** [−0.07, +0.59] |
| SR3Mu | 85 | 311 | 23.3 | +2.75 | 3.0e-03 | 15 | 96 | 0.289 | **+0.56** [+0.37, +0.68] |
| SR3Mu | 100 | 341 | 23.3 | +2.52 | 5.9e-03 | 20 | 96 | 0.566 | **−0.17** [−0.22, +0.61] |
| SR3Mu | 115 | 361 | 23.4 | +2.58 | 4.9e-03 | 22 | 111 | 0.545 | **−0.11** [−0.16, +0.19] |
| SR3Mu | 130 | 376 | 23.4 | +2.78 | 2.7e-03 | 22 | 137 | 0.374 | **+0.32** [+0.18, +0.46] |
| SR3Mu | 145 | 391 | 23.4 | +2.26 | 1.2e-02 | 20 | 67 | 0.800 | **−0.84** [−1.15, −0.54] |
| SR3Mu | 160 | 406 | 23.4 | +2.70 | 3.4e-03 | 20 | 117 | 0.403 | **+0.25** [−0.02, +0.60] |

**ParticleNet**

| channel | mHc | points | mA* | Z local | p local | N_0 | trials | p global | Z global |
|---|---|---|---|---|---|---|---|---|---|
| Combined | 100 | 26 | 82.5 | +2.38 | 8.7e-03 | 1 | 8 | 0.069 | **+1.48** [+1.45, +1.51] |
| Combined | 115 | 31 | 82.5 | +2.75 | 3.0e-03 | 2 | 13 | 0.040 | **+1.75** [+1.67, +1.90] |
| Combined | 130 | 31 | 82.5 | +2.79 | 2.7e-03 | 2 | 13 | 0.034 | **+1.82** [+1.72, +1.95] |
| Combined | 145 | 31 | 82.5 | +1.88 | 3.0e-02 | 1 | 7 | 0.201 | **+0.84** [+0.62, +0.90] |
| Combined | 160 | 31 | 82.5 | +2.24 | 1.3e-02 | 2 | 11 | 0.138 | **+1.09** [+0.66, +1.34] |
| SR1E2Mu | 100 | 26 | 87 | +1.77 | 3.8e-02 | 2 | 10 | 0.379 | **+0.31** [−0.04, +0.78] |
| SR1E2Mu | 115 | 31 | 88 | +1.76 | 4.0e-02 | 1 | 7 | 0.287 | **+0.56** [−0.01, +0.76] |
| SR1E2Mu | 130 | 31 | 82.5 | +1.36 | 8.7e-02 | 1 | 4 | 0.387 | **+0.29** [+0.29, +0.29] |
| SR1E2Mu | 145 | 31 | 87.5 | +1.07 | 1.4e-01 | 1 | 4 | 0.548 | **−0.12** [−0.42, −0.03] |
| SR1E2Mu | 160 | 31 | 90.5 | +1.54 | 6.1e-02 | 3 | 9 | 0.578 | **−0.20** [−0.34, +0.17] |
| SR3Mu | 100 | 26 | 82.5 | +2.30 | 1.1e-02 | 1 | 8 | 0.088 | **+1.35** [+1.17, +1.41] |
| SR3Mu | 115 | 31 | 82.5 | +2.92 | 1.7e-03 | 2 | 15 | 0.027 | **+1.93** [+1.71, +2.11] |
| SR3Mu | 130 | 31 | 83 | +2.41 | 8.0e-03 | 2 | 16 | 0.125 | **+1.15** [+0.94, +1.54] |
| SR3Mu | 145 | 31 | 82.5 | +1.58 | 5.7e-02 | 2 | 9 | 0.491 | **+0.02** [−0.03, +0.07] |
| SR3Mu | 160 | 31 | 82.5 | +2.29 | 1.1e-02 | 2 | 11 | 0.122 | **+1.17** [+0.99, +1.41] |

`N_0` runs 14–25 per Baseline column and 1–3 per ParticleNet column, tracking
the scanned range: a Baseline column covers 50–140 GeV in mA, a ParticleNet one
15 GeV. Every column passes both consistency checks
(`p_local ≤ p_global ≤ 1.2 × Šidák`), and the `upcrossings.*` panels show `N_u`
falling as a straight line on the semilog axis over the whole ladder — the
asymptotic law is followed, which is what licenses the extrapolation to
Z ≈ 3.75.

**AN wording.** *The largest local excess of the Baseline scan is at
mHc = 145 GeV, mA = 17.7 GeV in the eμμ channel, with a local significance of
3.75σ. Correcting for the look-elsewhere effect over the scanned mA range at
that mHc and in that channel — from the upcrossing rate of the observed
significance scan, following Gross and Vitells — gives a global p-value of
0.018, i.e. a global significance of 2.09σ. The largest local excess of the
ParticleNet arm, 2.92σ at mHc = 115 GeV, mA = 82.5 GeV in μμμ, corresponds to a
global significance of 1.93σ over its own 15 GeV mA window. Neither correction
accounts for the seven mHc columns, the three channels, or the two arms that
were also scanned; those are highly correlated (§3) but would reduce the global
significances further.*

## 5b. Cross-check: the band-pull proxy

Rerunning the whole calculation off the limit JSONs (`--statistic bandpull`,
no `Significance` fits at all) calibrates to `Z ≈ 1.208 × pull` with residual
rms 0.205 over all 7851 points, and reproduces `Z_global` with mean offset
−0.34 and rms 0.38 (max 0.91). The proxy is systematically **optimistic** in the
tails — it puts the Baseline maximum at 4.32 where the fit gives 3.75 — because
a linear pull-to-Z calibration overshoots where the band is most asymmetric. It
locates features correctly (the mA = 17.7 argmax is the same) and is kept as a
ranking tool and a smoke test, but it is not a substitute for the fits: the
scan campaign is what the quoted numbers come from.

Note also that the calibration slope moved from 1.087 to 1.208 when the fit
sample grew from the 45 hand-picked points to all 7851 — the hand-picked set was
biased towards the extremes, where the proxy is worst.


## 6. Caveats

- **`N_0` is estimated from the observed realisation.** The formally correct
  estimator averages upcrossings over background-only toys. A single
  realisation is noisy — the threshold ladder spans a factor ~2 between its
  highest and lowest `N_0` estimate (median over the 36 columns; worst 3.6) —
  and is biased *low* if a real signal is present, which would make `p_global`
  an underestimate.  `Z_global` depends on `log N_0`, so that factor 2 only
  moves it by ~±0.1–0.2 (`z_global_range`), but the bias is not bounded by it. This is the one place where V3's toy
  machinery would still buy something, and the affordable form of it is *not*
  V3's brute force but a linearised matched filter,
  `q0 = (sᵀV⁻¹(n−b))² / (sᵀV⁻¹s)`, which evaluates a whole 2467-point scan of one
  pseudo-dataset as a matrix product. Validated against all 45 exact combine
  significances with `V = diag(b)` (no systematics): slope 0.852, residual rms
  0.235, max residual 0.78. Not built; the grid scan now provides thousands of
  exact values to calibrate it against if it is ever wanted.
- **The maximum was selected over more than one scan.** See §3 — the scope is a
  decision, not a derivation.
- **Asymptotic validity.** The Gross–Vitells result is a Gaussian-field
  asymptotic. The categories entering these fits hold few events at low mA, so
  the `q(mA) → χ²₁` step is the weakest assumption in the chain. The
  `upcrossings.*` panels are the direct diagnostic: over Z = 0…2 the exponential
  law is followed closely, which is the evidence that the extrapolation to
  Z ≈ 3 is reasonable — not a proof that it is exact.
- **The band-pull proxy is not a substitute.** `--statistic bandpull` rebuilds
  the whole calculation from the limit JSONs with no `Significance` fits, using
  `Z ≈ slope × pull` calibrated on the exactly-fitted points. It is ~0.3σ
  noisier and is kept as a cross-check, not as a result.
- **The asymptotic law is followed upward, but not always downward.** On the
  excess side the `N_0` ladder is flat within its spread in all 36 columns. On
  the deficit side it *rises* with threshold in the seven Baseline SR1E2Mu
  columns, and there the Gross-Vitells `p_global` comes out 30-50% ABOVE the
  Šidák independent-trials bound — impossible for a correlated field, so the
  correction is an over-correction there and Šidák is the tighter statement
  (`estimateLEE.py` prints the pair and explains it; only an excess-tail
  failure is fatal). The likely cause is that the categories hold few events at
  low mA, where a downward fluctuation is bounded by the Poisson floor and the
  uncapped Z is furthest from its χ²₁ asymptotic. Every Combined-channel tail
  passes, so the published channel is unaffected.

## 7. Reproducing

```bash
# 1. Exact local Z at every scan point (7401 + 450 condor nodes, ~33 s each).
./automize/significance.sh --grid
./automize/significance.sh --pnet-grid
python3 python/collectSignificance.py --grid --pnet-grid

# 2. The trials estimate and its diagnostics (no jobs).
python3 python/estimateLEE.py
python3 python/plotLEE.py --all

# Cross-check with no Significance fits at all:
python3 python/estimateLEE.py --statistic bandpull
```

`automize/significance.sh` keeps its explicit-point mode for the *quoted*
points (`--template-points`, `--point METHOD:MP`); `--grid` is the LEE input and
does not replace it. `collectSignificance.py` merges into the existing record
and reports parsed/total, exiting non-zero on a partial collection — a short DAG
must not silently become a truncated curve.
