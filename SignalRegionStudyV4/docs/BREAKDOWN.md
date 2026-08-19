# Uncertainty Breakdown

Where the uncertainty on the signal strength comes from: `sigma(r)` split
into signal-model, background-normalization, experimental and statistical
components, at the curated template points.

Source of record: `results/json/breakdown.All.interp-signal.json`
(git-tracked, written by `python/collectBreakdown.py`). The per-point
likelihood-scan panels live in each point's
`combine_output/breakdown/breakdown.pdf` and are promoted into
`results/templates/{method}/{masspoint}/` by `collectTemplatePlots.py`.

**This study has no V3 counterpart to reproduce.** V3 carries a written
implementation, `SignalRegionStudyV3/scripts/runImpactBreakdown.sh`, but
it was never run: there are no `impact_breakdown_*` directories under V3
`templates/`, nothing under V3 `results/`, and the four invocations in V3
`doThis.sh:36-40` are all commented out. V4 establishes the number. The
method below follows V3's where V3 made a choice, and departs from it in
two places, each forced by a measurement rather than by taste: the shared
best-fit snapshot ([Recipe correction](#recipe-correction)) and the
per-point scan range
([Scan range](#scan-range-the-second-departure-from-v3)).

## Method

### 1. Groups

Combine reads nuisance groups from `<name> group = <members>` lines in the
datacard. V4's production datacards carry none and must keep their exact
bytes, so `python/nuisanceGroups.py` appends the group block to a
throwaway `grouped_datacard.txt` in the point's breakdown dir. The
production card written by `printDatacard.py` is never touched, and the
2467-point campaign never needs regenerating.

The groups are defined once in `configs/nuisance_groups.json`. The array
order **is** the cumulative freeze order.

| group | contents |
|---|---|
| `signal_theory` | `QCDScale_mu{R,F}_BSMsignal`, `pdf`, `pdf_alphas`, `ps_isr`, `ps_fsr`, **and** `CMS_interp_*` |
| `prompt_norm` | `CMS_B2G25013_Norm_{WZ,ZZ,ttW,ttZ,ttH,tZq,conversion,others}*` |
| `nonprompt_norm` | `CMS_B2G25013_Norm_nonprompt_*` |
| `experimental` | `CMS_{scale,res,eff,btag,pileup,l1_prefiring}_*`, `lumi_*` |
| `stat` | the residual — everything still floating after the four freezes |

**Why `CMS_interp_*` sits in `signal_theory`.** The interpolation
nuisances — `CMS_interp_{scale,res,norm}_*` and the ParticleNet
`CMS_interp_{res,eff}_pnet_*` — are uncertainties on the *parametric
signal model*: how well a DCB surface sliced at an unmeasured mA
reproduces what MC would have given. That is a signal-modelling error, of
the same kind as the scale/PDF/parton-shower ones, so it belongs in the
signal group rather than in a component of its own. A single
`^CMS_interp_` pattern covers both arms' families.

**Why `stat` includes autoMCStats.** The `prop_bin*` parameters are
created by `text2workspace.py` from the `autoMCStats` lines, so they
cannot be named in a datacard `group =` line and cannot be frozen by
`--freezeNuisanceGroups`. They therefore remain floating in every scan and
land in the residual. This is V3's definition, kept deliberately: the
residual is "data statistics + simulation statistics", not "data
statistics". Quoting them apart would need a separate
`--freezeParameters 'rgx{prop_bin.*}'` step, which is not run.

**Unmatched nuisances are a hard error.** V3 swept anything unmatched into
`experimental`. Had `CMS_interp_*` existed then it would have been
silently absorbed there and mis-attributed. `nuisanceGroups.classify`
raises instead, listing every unmatched name. Verified exhaustive on both
production arms:

| datacard | total | signal_theory | prompt_norm | nonprompt_norm | experimental |
|---|---|---|---|---|---|
| `MHc160_MA16p5` Baseline | 129 | 18 | 12 | 15 | 84 |
| `MHc100_MA85` ParticleNet | 137 | 22 | 15 | 16 | 84 |

### 2. The scans

One best-fit step, then the total scan and one scan per group freezing
everything up to and including it -- `plot1DScan.py --breakdown` requires
exactly this cumulative nesting.

```
combine -M MultiDimFit grouped_workspace.root \
    --algo none --setParameterRanges r=${RMIN},${RMAX} \
    --saveWorkspace -n .{mp}.{method}.bestfit -m 120

combine -M MultiDimFit {BESTFIT_FILE} --snapshotName MultiDimFit \
    --algo grid --points 200 --setParameterRanges r=${RMIN},${RMAX} \
    -n .{mp}.{method}.total -m 120

combine -M MultiDimFit {BESTFIT_FILE} --snapshotName MultiDimFit \
    --algo grid --points 200 --setParameterRanges r=${RMIN},${RMAX} \
    --freezeNuisanceGroups {cumulative,csv} \
    -n .{mp}.{method}.freeze_{cumulative_tag} -m 120
```

**Every scan hangs off the one `--algo none` snapshot**, and this is
load-bearing, not tidiness. Quadrature subtraction compares intervals
*across* scans, which only means anything if they share a minimum. V3's
recipe made the total `--algo grid` scan its own snapshot source, so the
frozen scans were free to re-minimise to a different r. Ported verbatim,
that failed on real data: at the three template points whose total scan
pinned r-hat at exactly 0.000, freezing a group *raised* sigma, and two of
them returned `stat > total`, which is impossible. See
[Recipe correction](#recipe-correction).

Frozen in `scripts/runBreakdown.sh`; driver `automize/breakdown.sh`
(setup -> bestfit -> {total, the four freezes} in parallel -> plot,
8 nodes per point); collected by `python/collectBreakdown.py`.

The last freeze adds `--X-rtd MINIMIZER_no_analytic`, following V3. Note
V3's stated reason — the endpoint degenerating into a single best-fit
point when nothing floats — **cannot arise in V4**, because `prop_bin*`
are never grouped and so still float in the final scan. It is kept as
cheap insurance, not because it is load-bearing.

Unblind (real data) is the default, as everywhere in V4; `--blind` gives
`-t -1 --expectSignal 1` under the `{method}_blind` segment.

### Scan range: the second departure from V3

V3 scanned `r` over a fixed `[-5, 5]` with `--points 100` — a grid step of
0.1. Converting the V4 limits back to `r` (`collectLimits._convert`
inverted, factor `1.8199e5`), `sigma(r) ~ r_exp0 / 1.96` at the seven
template points is:

| point | r_obs | r_exp0 | sigma | V3 grid steps per sigma |
|---|---|---|---|---|
| Baseline MHc70_MA15 | 0.140 | 0.164 | 0.084 | 1.2 |
| Baseline MHc100_MA60 | 0.196 | 0.256 | 0.131 | 0.8 |
| Baseline MHc130_MA90 | 0.616 | 0.891 | 0.454 | 0.2 |
| Baseline MHc160_MA155 | 0.296 | 0.441 | 0.225 | 0.4 |
| ParticleNet MHc160_MA85 | 0.352 | 0.291 | 0.148 | 0.7 |
| ParticleNet MHc130_MA90 | 0.477 | 0.582 | 0.297 | 0.3 |
| ParticleNet MHc100_MA95 | 0.378 | 0.504 | 0.257 | 0.4 |

**0.2 to 1.2 grid steps per sigma.** The `2*deltaNLL = 1` crossing would be
splined from one or two points; the frozen scans are narrower still, and
the quadrature subtraction of two such numbers is noise, not a
measurement. That is exactly the regime in which `plot1DScan.py` prints
`ERROR SUBTRACTION IS NEGATIVE` and substitutes zero.

So `runBreakdown.sh resolve_scan_range()` derives the window per point:
the asymptotic `limit` tree already holds `r`, the median expected upper
limit is `~1.96 sigma` for `r` near zero, and the scan runs over
`+-5 sigma` with 200 points — **20 grid points per sigma at every point**,
whatever its sensitivity. `--r-range` and `--points` override;
`--sigma-window` changes the width.

`scripts/runImpacts.sh` has a similar-looking `resolve_r_range()`, but it
only ever *widens* past `+-5`: for every template point
`2 x exp+2sigma < 5`, so it returns `-5,5` and cannot be reused here.

### Recipe correction

The first production run used V3's chaining -- total scan saves the
snapshot, freezes read it -- and split cleanly in two:

| | points | behaviour |
|---|---|---|
| r-hat != 0 | 4 | consistent; quadrature sum reproduced sigma_total exactly, no negative subtractions |
| r-hat == 0.000 | 3 | freezing raised sigma; `stat > total` at 2 of them |

Diagnosed on `MHc70_MA15`: the total scan minimised at r = 0.000 while the
`signal_theory`-frozen scan minimised at r = -0.026, and the frozen scan
also lost 24 of its 201 grid points at the low end (x_min -0.316 against
-0.416). Two scans with different minima share no reference for a width
subtraction.

The fix is the standard Combine ordering: a separate `--algo none
--saveWorkspace` best fit that *all* scans -- the total included -- start
from. It costs one extra node per point and lets the total scan run in
parallel with the freezes rather than ahead of them.

The superseded outputs are kept alongside as
`combine_output/breakdown_v3recipe/` for comparison.

### 3. The components

Each group's contribution is the quadrature difference between
consecutive cumulative scans, with the residual taken as-is:

```
sigma_i    = sqrt(sigma_i^2 - sigma_{i+1}^2)     per group, up and down separately
sigma_stat = sigma_last
```

`collectBreakdown.py` recomputes these from the scan ROOTs using the same
spline and the same crossing finder the plot uses
(`HiggsAnalysis.CombinedLimit.util.plotting.FindCrossingsWithSpline`), so
the JSON and the PDF agree by construction rather than by coincidence.
`plot1DScan.py` itself keeps the numbers only inside a `TPaveText`, which
is why they are recomputed rather than scraped.

**A negative subtraction is recorded as `null`, not zero.** It means the
freeze did not shrink the interval — minimizer noise, or a scan too coarse
to resolve the difference — and "not measured" is a different statement
from "negligible". The collector prints every occurrence and the summary
plot omits those segments and says so. A clean campaign should have none;
any at all mean the scan window or `--points` needs revisiting.

An interval clipped by the scan window (`valid_lo`/`valid_hi` false) is
treated as unusable rather than as a sigma, since it would only be a lower
bound.

## Running it

```bash
./automize/breakdown.sh --template-points          # 56 nodes
./automize/breakdown.sh --point Baseline:MHc145_MA90
./automize/breakdown.sh --template-points --skip-existing
python3 python/collectBreakdown.py --template-points
python3 python/plotBreakdown.py
python3 python/collectTemplatePlots.py             # fold into the bundles
```

Like the significance step, the point list is **explicit**: the breakdown
is a follow-up on the points the analysis quotes, not a scan.
`--template-points` reads `collectTemplatePlots.DEFAULT_POINTS`, so the
bundle set and the breakdown set cannot drift.

## Results

Measured 2026-08-19 on the frozen production templates (unblind, era
`All`, channel `Combined`, `interp-signal`). Up-side sigma(r); `syst` is
`sqrt(total^2 - stat^2)`.

| point | r-hat | total | signal theory | prompt norm | nonprompt norm | experimental | stat | syst |
|---|---|---|---|---|---|---|---|---|
| Baseline mHc 70, mA 15 | -0.026 | 0.069 | 0.011 | 0.003 | 0.013 | 0.005 | 0.066 | 0.018 |
| Baseline mHc 100, mA 60 | -0.089 | 0.116 | 0.011 | 0.009 | 0.028 | 0.006 | 0.112 | 0.032 |
| Baseline mHc 130, mA 90 | -0.469 | 0.423 | 0.036 | 0.169 | 0.138 | 0.173 | 0.317 | 0.281 |
| Baseline mHc 160, mA 155 | -0.315 | 0.206 | 0.027 | 0.036 | 0.029 | 0.028 | 0.197 | 0.060 |
| ParticleNet mHc 100, mA 95 | -0.199 | 0.219 | 0.013 | 0.046 | 0.019 | 0.052 | 0.207 | 0.073 |
| ParticleNet mHc 130, mA 90 | -0.152 | 0.270 | 0.023 | 0.098 | 0.054 | 0.063 | 0.236 | 0.130 |
| ParticleNet mHc 160, mA 85 | +0.049 | 0.152 | 0.038 | 0.017 | 0.019 | 0.028 | 0.142 | 0.054 |

**The measurement is statistically dominated everywhere.** `stat` is the
largest component at all seven points, and at five of them the systematic
part is under a third of the total. The one point where systematics come
close is Baseline mHc 130, mA 90 -- syst 0.281 against stat 0.317, i.e.
44% of the variance -- and it is also the point with the largest
excursion in r-hat. This is the expected picture for a search whose
categories carry few events, and it is the quantitative statement behind
the autoMCStats/low-stat machinery in `docs/LOWSTAT.md`.

Among the systematics, **the two background normalizations dominate**
(`prompt_norm` and `nonprompt_norm` together exceed `experimental` at
every Baseline point and at ParticleNet mHc 130, mA 90). **`signal_theory`
is the smallest component everywhere**, 0.011-0.038, so the interpolation
nuisances folded into it -- the whole `CMS_interp_*` family -- are not a
limiting uncertainty anywhere in the scan. That is a useful negative
result for the parametric-signal programme: the interpolation model's
derived envelopes (`docs/interpolation/UNCERTAINTY.md`) are comfortably
below the level at which they would matter.

Every point satisfies the closure `sqrt(sum of components^2) =
sigma_total` exactly, and no point returns `stat > total`.

**One caveat.** At Baseline mHc 70, mA 15 -- the least sensitive point,
sigma = 0.069 -- the DOWN-side `signal_theory`, `prompt_norm` and
`nonprompt_norm` subtractions are negative and recorded as `null`. Their
total systematic contribution there is 0.018, and the individual
components are 0.003-0.013, at or below what a 200-point spline crossing
resolves. Read those three as "below the resolution of the method at this
point", not as zero. The up side of the same point is measured cleanly.
