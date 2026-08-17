# Observed Excesses and Deficits

Where the observed limit departs most from the expected one over the
interp-signal scan, and what the observed local significance is at those
points. Measured 2026-08-18 on the frozen production scan (Baseline 2467
points over 7 mHc, ParticleNet 150 points over 5 mHc; unblind, era `All`).

Source of record: `results/json/significance.All.interp-signal.json`
(git-tracked, written by `python/collectSignificance.py`). The limit
inputs are the same `results/json/{BR,xsec}/All/limits.*.json` the
published curves are drawn from — nothing here is re-derived from Combine
by hand.

**Everything below is LOCAL.** No look-elsewhere correction is applied
anywhere in this document, and the trials factor on this scan is large
(see [Trials](#trials-what-these-numbers-are-not)). None of these are
claims of a signal.

## Method

Three quantities, computed in this order.

### 1. obs/exp and the band pull — the sweep

Over every scan point, per channel and per arm:

- `obs/exp0` — observed limit over median expected. Cheap, needs no new
  fits, and is what the published Brazilian bands already show.
- **band pull** — `(obs − exp0)` divided by the half-width of the
  expected band *on the side the observation falls*:

  ```
  pull = (obs − exp0) / (exp+1 − exp0)   if obs ≥ exp0
       = (obs − exp0) / (exp0 − exp−1)   otherwise
  ```

  This is a normalized departure, not a significance: it says how far
  out in the expected band the observation sits. It is used only to
  *rank* points and pick which ones deserve a fit.

### 2. Observed local significance — the fit

At the ranked points, `combine -M Significance` on the same datacard the
limit came from, at `All` × {`Combined`, `SR1E2Mu`, `SR3Mu`}:

```
combine -M Significance datacard.txt -n .{mp}.{method} -m 120 \
        --uncapped 1 --rMin -5
```

Frozen in `scripts/runSignificance.sh`; driver `automize/significance.sh`
(one condor node per point × channel); collected by
`python/collectSignificance.py`.

**`--uncapped 1` is the whole point of the configuration.** Combine's
default caps the significance at zero for a downward fluctuation, which
would report every deficit point as the same `0` and make a deficit scan
impossible. Uncapped, a deficit keeps its sign and the two conventions
agree wherever Z > 0. The quoted `p` is the one-sided tail
`0.5·erfc(Z/√2)`, so a negative Z gives p > 0.5 by construction — read
those as "how far below expectation", not as a p-value for a signal.

### 3. Saturated GoF — is it the data or the model?

A large Z says the signal+background fit prefers r > 0 at that mass. It
does not say the background-only model was adequate to begin with. The
background-only saturated goodness-of-fit already run at every group seed
(`automize/interpGofImpacts.sh`, summarized by `plotGoFPValues.py`)
answers that separately, and is quoted alongside every excess below.

### Why the point list is explicit

The significance is **not** a scan step. The fit is cheap, but choosing
which points to quote is a judgement call, so `automize/significance.sh`
never has a default point list: `--template-points` takes the curated
bundle set (read from `collectTemplatePlots.DEFAULT_POINTS`, so the two
cannot drift), and every other point is named with `--point METHOD:MP`.
`collectSignificance.py` merges into the existing JSON rather than
replacing it, so a later run on different points adds to the record.

The extremes recorded here are a property of **the data**, not of the
code. Re-derive them from the sweep after any reprocessing; do not reuse
the point list blindly.

## Summary

The extremes of the scan, one row per "largest by that measure". No
single point is the largest by all of them — the biggest single-channel
departure and the biggest combined one are different features in
different arms — which is why this is a table and not a headline number.

| measure | arm | point | channel | obs/exp | Z (local) |
|---|---|---|---|---|---|
| largest in any single channel | Baseline | `MHc145_MA17p5` | SR1E2Mu | 3.03× | **+3.02** |
| largest in Combined | ParticleNet | `MHc130_MA82p5` | Combined | 2.58× | **+2.79** |
| largest Baseline Combined | Baseline | `MHc115_MA18` | Combined | 2.40× | +2.41 |
| largest in SR3Mu | Baseline | `MHc85_MA23p4` | SR3Mu | 2.31× | +2.70 |
| largest deficit, Combined | Baseline | `MHc160_MA39` | Combined | 0.47× | **−2.33** |
| largest deficit, single channel | Baseline | `MHc160_MA39` | SR3Mu | 0.48× | −2.61 |
| largest deficit, ParticleNet | ParticleNet | `MHc160_MA97p5` | Combined | 0.52× | −2.21 |

Three distinct excess features appear, in different channels and at
different mA — they are not one effect seen three ways:

- **mA ≈ 17–18, eμμ**, in every mHc column, with poor background-only
  GoF in that channel. The largest single-channel value in the scan.
- **mA ≈ 23, μμμ**, opposite channel, GoF unremarkable.
- **mA ≈ 82.5, ParticleNet**, both channels positive, at the bottom edge
  of the ParticleNet mA window.

## The mA ≈ 17–18 GeV excess — eμμ, every mHc column

The largest single-channel departure in the scan, and it is the same
feature in every mHc column — not seven separate ones. Combined obs/exp
runs 2.28–2.42 with the maximum at mA between 17.1 and 18.2 in all seven
studies.

| point | Combined | SR1E2Mu | SR3Mu |
|---|---|---|---|
| `MHc145_MA17p5` | 2.14×, Z = +2.25 (p = 0.012) | 3.03×, **Z = +3.02** (p = 0.001) | 0.82×, Z = −0.76 |
| `MHc160_MA17p5` | 2.16×, Z = +2.23 (p = 0.013) | 2.96×, Z = +2.85 (p = 0.002) | 0.91×, Z = −0.44 |
| `MHc115_MA18` | 2.40×, Z = +2.41 (p = 0.008) | 2.78×, Z = +2.61 (p = 0.005) | 1.41×, Z = +0.68 |

Saturated background-only GoF p-values at the same seeds:

| seed | Combined | SR1E2Mu | SR3Mu |
|---|---|---|---|
| `MHc145_MA17p5` | 0.064 | **0.004** | 0.576 |
| `MHc160_MA17p5` | 0.104 | **0.014** | 0.656 |
| `MHc115_MA18` | 0.086 | 0.080 | 0.286 |

**It is an eμμ effect.** SR3Mu is at or below zero at two of the three
points, so the combination is *diluted* by the μμμ channel rather than
reinforced — Combined (≈ +2.3) sits below SR1E2Mu (≈ +2.6 to +3.0). The
GoF agrees and localizes it the same way: the background-only model
describes the eμμ data poorly there (p = 0.004–0.014) while μμμ is
unremarkable (p = 0.29–0.66).

Band-pull windows with pull > +2 in Combined, per study — the feature is
~1 GeV wide in every column:

| mHc | mA windows (GeV) |
|---|---|
| 70 | 17.2–18.2 |
| 85 | 17.8–18.2 |
| 100 | 17.7–18.2, 62.5–64 |
| 115 | 17.7–18.2, 64 |
| 130 | 17.7–18.2, 50.25–50.5, 64 |
| 145 | 17.1–18.2, 64 |
| 160 | 17.1–18.2, 62.5–63 |

A secondary, much weaker feature sits at mA ≈ 62–64 for mHc ≥ 100.

### Caveats specific to this excess

- **It sits in the interpolation's weakest region.** Below mA ≈ 45 the
  norm envelopes are 7–19%, driven by the MHc115 (15 → 27 → 42) and
  MHc130 (15 → 30 → 55) MC gaps — the binding limitation recorded in
  `docs/interpolation/WORKFLOW.md`. The parametric signal is least
  constrained exactly where the excess is.
- **It is a handful of events.** At mA ≈ 18 the fitted DCB width is
  σ ≈ 0.15–0.22 GeV per category (`fits/MHc160/dcb_fits.json`), so the
  ~1 GeV window is about ±3σ, and the 0.5 GeV scan lattice puts ~3 points
  per σ. The many scan points in the window are the same events seen
  repeatedly.

## The other two excess features

### mA ≈ 23 GeV, μμμ — `MHc85_MA23p4`

| | Combined | SR1E2Mu | SR3Mu |
|---|---|---|---|
| obs/exp | 1.71× | 0.66× | 2.31× |
| Z (local) | +1.64 (p = 0.050) | −1.20 | **+2.70** (p = 0.003) |
| GoF (seed `MHc85_MA23p5`) | 0.248 | 0.486 | 0.168 |

The mirror image of the mA ≈ 18 feature: the excess is in μμμ and eμμ is
*below* expectation, so the combination again lands well under the
driving channel. The two are at different masses (23.4 vs 17.5, i.e.
~25σ apart in the fitted resolution) and in disjoint channels, so they
are independent observations, not one effect. Unlike the eμμ one, the
background-only fit here is fine (GoF 0.17–0.49).

### mA = 82.5 GeV, ParticleNet — `MHc130_MA82p5`

| | Combined | SR1E2Mu | SR3Mu |
|---|---|---|---|
| obs/exp | 2.58× | 1.56× | 2.43× |
| Z (local) | **+2.79** (p = 0.003) | +1.36 | +2.39 (p = 0.008) |
| GoF (seed `MHc130_MA85`) | 0.428 | 0.460 | 0.334 |

The largest *Combined* excess of the study. It is the only feature where
both channels contribute comparably (+2.39 and +1.36), so the combination
exceeds either one instead of being diluted the way the mA ≈ 18 and
mA ≈ 23 features are. Two caveats specific to it:

- It sits at **mA = 82.5, the bottom edge of the ParticleNet arm's
  reach** ([82.5, 97.5], `configs/pnet_grid.json`), i.e. at the extreme
  of the ε interpolation, 2.5 GeV below the seed net's lowest anchor.
- **The score cut is what makes it large.** The Baseline arm covers the
  same mass from the same data with no score cut and sees 1.31×
  (pull +0.63) in Combined, 1.52× in SR3Mu — a mild upward fluctuation,
  not an extremum of its scan. The two arms are not independent
  measurements; ParticleNet is a subset selection of the same events.

GoF is unremarkable, so this is not background mismodelling.

## The largest deficit: mA ≈ 39 GeV

| point | Combined | SR1E2Mu | SR3Mu |
|---|---|---|---|
| `MHc160_MA39` | 0.47×, **Z = −2.33** | 0.65×, Z = −1.12 | 0.48×, Z = −2.61 |
| `MHc100_MA44` | 0.84×, Z = −0.58 | 0.51×, Z = −1.87 | 1.87×, Z = +1.76 |

The genuine deficit is `MHc160_MA39`, driven by μμμ. `MHc100_MA44` is the
*sweep's* SR1E2Mu minimum (0.51×) but not a deficit in the combination:
its two channels pull opposite ways (−1.87 vs +1.76) and Combined lands
at −0.58. This is why the sweep alone is not enough — obs/exp ranks
single channels, the fit shows what survives combination.

Neither point shows a modelling problem: GoF p = 0.404/0.580/0.240 at
`MHc160_MA39` and 0.552/0.184/0.790 at `MHc100_MA44`. That is expected —
a downward fluctuation of the limit is not a failure of the
background-only fit.

The ParticleNet arm's deficit is `MHc160_MA97p5` — 0.52×, Z = −2.21
(SR1E2Mu −1.82, SR3Mu −1.15; GoF 0.292/0.598/0.132 at seed
`MHc160_MA95`). Like the arm's excess it sits at an **edge of the
ParticleNet mA window**, this time the top one (97.5), 2.5 GeV above the
highest anchor. Baseline at the same mass is 0.83× (pull −0.56), so here
too the score cut sharpens a mild fluctuation into the arm's extremum.

That both ParticleNet extremes land exactly on the two window edges is
worth keeping in view: the edges are where the ε interpolation is
furthest from its anchors, and they are only 15 GeV apart, so the arm has
few effectively independent cells and its extremes have nowhere else to
sit.

Band-pull windows with pull < −1.5 in Combined, per study. A dip at
mA ≈ 39–39.5 is present in **all seven** columns; the mA ≈ 72 and
mA ≈ 103–104 dips appear only where those masses are in range:

| mHc | mA windows (GeV) |
|---|---|
| 70 | 39–39.5 |
| 85 | 26.3, 39.25, 71.5–73, 77–77.5 |
| 100 | 39.25–39.5, 76.5–77 |
| 115 | 26.3, 39.25–39.5, 72–72.5, 76.5, 103–104 |
| 130 | 22.5–22.6, 26.3–26.5, 39–39.5, 72–72.5, 103–104 |
| 145 | 39–39.5, 72–72.5, 103–104, 136–137 |
| 160 | 22.3–22.5, 38.75–39.5, 71–72.5, 103–104, 136–137 |

## Template points — the reference set

The curated bundle points (`results/templates/`, see the module
`CLAUDE.md` §11) are all unremarkable, |Z| ≤ 1.5, and are recorded here
as the null reference the extremes are read against:

| arm | point | Combined | SR1E2Mu | SR3Mu |
|---|---|---|---|---|
| Baseline | `MHc70_MA15` | −0.40 | −0.20 | −0.39 |
| Baseline | `MHc100_MA60` | −0.77 | −0.38 | −0.68 |
| Baseline | `MHc130_MA90` | −1.11 | −0.31 | −1.20 |
| Baseline | `MHc160_MA155` | −1.50 | −1.27 | −0.80 |
| ParticleNet | `MHc160_MA85` | +0.36 | +0.11 | +0.44 |
| ParticleNet | `MHc130_MA90` | −0.57 | −0.12 | −0.66 |
| ParticleNet | `MHc100_MA95` | −0.90 | −1.05 | +0.03 |

Each point's own copy is in its bundle at
`results/templates/{method}/{masspoint}/significance.json`, lifted out of
the collected JSON so a bundle can never disagree with this document.

## Trials: what these numbers are not

Z = +3.02 is p = 0.0013 **at one pre-selected mass point**. It was not
pre-selected: it is the maximum of a sweep over 2467 Baseline scan points
in three channels. Three facts bound how much that matters:

1. The scan lattice is 0.5 GeV while the signal resolution at low mA is
   σ ≈ 0.15–0.22 GeV, so neighbouring points are strongly correlated —
   the effective number of independent mA cells is far below 2467, but it
   is not one either.
2. The seven mHc columns share one dataset and, at fixed mA, nearly the
   same signal shape, so the seven "independent" maxima at mA ≈ 18 are
   one observation, not seven.
3. Two arms and three channels were scanned and the reported maxima are
   over all of them. The arms are not independent either: ParticleNet is
   a score-cut subset of the same events the Baseline arm uses, over the
   sub-range mA ∈ [82.5, 97.5].

Converting this to a global p-value needs a proper look-elsewhere
treatment (toy-based, or the standard asymptotic trials estimate over the
scanned range) and **has not been done**. Until it is, the honest
statement is: the largest local departures in the scan are ~3σ — eμμ at
mA ≈ 17.5 in the region where the interpolation is least constrained, and
the ParticleNet combination at the low edge of its window — and the
analysis has no global significance.

## Reproducing

```bash
# 1. rank: sweep obs/exp over the collected limit JSONs (no fits)
#    -> pick the points to quote
# 2. fit those points
POINTS="--point Baseline:MHc145_MA17p5 --point Baseline:MHc160_MA17p5 \
        --point Baseline:MHc115_MA18   --point Baseline:MHc85_MA23p4 \
        --point Baseline:MHc160_MA39   --point Baseline:MHc100_MA44 \
        --point ParticleNet:MHc130_MA82p5 --point ParticleNet:MHc160_MA97p5"
./automize/significance.sh --template-points $POINTS
python3 python/collectSignificance.py --template-points $POINTS
# 3. fold into the template bundles
python3 python/collectTemplatePlots.py
```

Related: `docs/interpolation/WORKFLOW.md` (the low-mA grid limitation),
`docs/interpolation/UNCERTAINTY.md` (interpolation nuisances),
`docs/FUNCTIONALITY_SCOPE.md` (what is and is not in V4).
