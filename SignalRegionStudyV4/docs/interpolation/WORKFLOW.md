# Interpolation — Workflow

Concise runbook for the mA-interpolation chain: what runs, in what order,
and the decision gates between the stages. Deep dives live next door:

- [EXPERIMENTS.md](EXPERIMENTS.md) — how every model choice was made
  (motivation / setup / results / conclusion per experiment).
- [UNCERTAINTY.md](UNCERTAINTY.md) — which nuisances are derived, by what
  rule, their current values, and the evidence behind the rule.

`doThis.sh` at module root carries the same commands with inline warnings;
this file is the reference, that one is the copy-paste surface.

## Production model (frozen 2026-08-12)

| piece | model |
|---|---|
| SR1E2Mu, SR3Mu_lowM | pure DCB, frozen per-category nL/nR |
| SR3Mu_highM | fsig·DCB + (1−fsig)·Chebychev₂ |
| shape vs mA | one (mHc, mA) surface per parameter across all seven studies, sliced at the study's mHc; fsig in logit space |
| yield | N = k_era · G_period(mA) · f_category(mA); G and k_era are (mHc, mA) surfaces, f is per-study with f_SR3Mu = S·p on raw fractions |
| uncertainties | rms within each study, then max across studies; norm binned in mA at 15/80/100/155; no mHc dependence ([UNCERTAINTY.md](UNCERTAINTY.md)) |
| interpolation range | **mA only**, at the seven measured mHc (70–160 in steps of 15) |

Constants: `python/interpolation_config.py`. Study grid: the full baseline
grid (78 mass points), from `configs/masspoints.json` via `mhc_grid()`;
`configs/interpolation.json` holds only `known_missing_samples`.

## Chain structure — why it runs in passes

Every mA dependence — each shape parameter, and the yield model's G and
k_era — is ONE surface in (mHc, mA) fitted across all seven studies and
sliced at each study's mHc. That creates two cross-study barriers the
per-mHc DAGs do not express:

- `polynomials` reads EVERY study's `fits/dcb_fits.json`
- `yield_model` reads EVERY study's `yields/yields.json`

A single-mHc run cannot get past `polynomials`: it raises
`FileNotFoundError` naming the study it is missing. That is deliberate —
the alternative is a surface silently fitted on a subset. Run the passes
below, each fully finished before the next is submitted.

Per-mHc stages, in DAG order:

```
fit-floating → fit-frozen ─┐ (barrier: all 7 studies)
                           ├→ polynomials → closure
                           │              → yields ─┐ (barrier: all 7)
                           │                        ├→ yield-model → yield-closure
                           │                        │              → deltas
```

## Procedure

```bash
source setup.sh          # module-local; REQUIRED

# Gate 0 — inputs. Opens every signal file on pnfs; fails on zombie /
# truncated / empty Central tree. NOT optional: concurrent xrdcp has
# silently truncated samples before (docs/SAMPLES.md).
for mhc in 70 85 100 115 130 145 160; do
    python3 python/verifyInterpSamples.py --all --mhc $mhc
done

# Pass 1 — per-point DCB fits, floating then frozen-n (~170 nodes).
./automize/interpolation.sh --all --stop-after fit-frozen

# Pass 2 — shape surfaces, shape closure, window yields (~180 nodes).
./automize/interpolation.sh --all --start-from polynomials --stop-after yields

# Pass 3 — yield model, yield closure, shape-systematic deltas (~190 nodes).
./automize/interpolation.sh --all --start-from yield-model

# LOO sweep — 78 nodes, each refitting BOTH surfaces without one point and
# closing shape+yield there. May run in parallel with pass 3 (needs only
# fits and yields).
./automize/interpolation.sh --loo --all

# Uncertainties + production config (JSON-only, login node fine).
python3 python/exportInterpUncertainties.py --loo --all --pooled --write-config

# Global plots: surfaces and the nuisance rule.
python3 python/plotInterpSurfaces.py --all
python3 python/plotInterpNuisances.py
```

## Decision gates

**Gate 0 (before pass 1)** — `verifyInterpSamples.py` exits 0 for all seven
mHc. A truncated file otherwise surfaces much later as a confusing fit
failure.

**Gate A (after each condor pass)** — "the DAG finished" is not the same as
"it worked"; a failing terminal node is easy to miss:

```bash
grep -h "EXITING WITH STATUS" condor/jobs_interp_<ts>/MHc*/dag.dag.dagman.out
```

All seven must report STATUS 0 before the next pass is submitted — the
passes ARE the barriers.

**Gate B (after pass 2)** — shape closure. `chi2_interp ≈ chi2_direct` per
category (interpolation adds nothing on top of the fit model); bad direct
fits are excluded and logged, and `SR3Mu_highM` at MA15–25 failing by
construction is expected (no physical peak — production never uses highM
there). Look at `plots/` per study and `plotInterpSurfaces.py` output:
every parameter panel should show the seven sliced curves tracking each
study's points.

**Gate C (after LOO + export)** — review
`configs/interpolation_uncertainties.json` in this order:

1. `per_study_detail[cell].driver` — which study set the value, and
   `per_study_rms` for how far the others sit below it. A cell driven by a
   study with a sparse low-mA grid (MHc115, MHc130) is the grid talking,
   not the model.
2. `studies_below_min_points` — studies skipped for holding < 2 points.
3. `n_points` — a cell resting on very few points is flagged in `warnings`.
4. Any value sitting EXACTLY on its floor (scale 0.02, res 0.01, norm
   0.01): the floor is then setting the number rather than catching a
   degenerate cell, and wants revisiting. With the frozen model none is
   active.
5. `warnings` for empty-but-reachable mA bins; those inherit the channel's
   worst populated bin (never the bare floor) and say so.

Run3 norm cells above 10% are expected — the known upstream per-sample
yield scatter, which is the REFERENCE's error, not the model's
([EXPERIMENTS.md](EXPERIMENTS.md), "Run3 normalization scatter").

## Recovery

```bash
# Resume a pass at a named step (fit-floating, fit-frozen, polynomials,
# closure, yields, yield-model, yield-closure, deltas). closure and yields
# share a level, so either re-runs both. Nodes outside
# [--start-from, --stop-after] are emitted but marked DONE, so the
# dependency graph stays intact:
./automize/interpolation.sh --mhc 160 --start-from yield-model

# A sharded stage interrupted mid-flight: merge what its jobs did produce
# before re-running anything downstream:
python3 python/mergeInterpResults.py --mhc 160 --stage closure

# Failed DAG: DAGMan writes dag.dag.rescue001. Resubmit with plain
# condor_submit_dag from the per-mHc dir — NOT ./submit_all.sh, whose -f
# interferes with the rescue:
cd condor/jobs_interp_<ts>/MHc145 && condor_submit_dag dag.dag
```

## Outputs

```
fits/MHc{X}/                     fit artifacts: dcb_fits{,_floating}.json, parts/,
                                 polynomials.json, yields/{yields,yield_model}.json,
                                 shape_deltas/, plots/{fits,params,yields,deltas}/
fits/params/  fits/yield/        global surface panels (shape params; G, k_era)
closure/interpolation/MHc{X}/    closure.json, yield_closure.json,
                                 loo_uncertainties.json, parts/, plots/{closure,yields}/
closure/interpolation/loo/MHc{X}_MA{Y}/   leave-one-out: both surfaces refit
                                 without the point, closure + yield closure at it
closure/interpolation/loo_uncertainties.pooled.json   pooled envelope + detail
closure/interpolation/plots/nuisance/                 nuisance-rule plots
configs/interpolation_uncertainties.json              production config
```

`fits/`, `closure/` and the config are **git-tracked production outputs**
(JSONs and PNGs both) — commit them after a verified re-run. The retired
scratch trees live under `archive/test{,s}/` and stay untracked.

## Known input issues

- `MHc145_MA100` / 2023 / SR1E2Mu: corrupt raw skim (truncated at 9 MB;
  needs SKNano regeneration). Carried as `known_missing_samples` in
  `configs/interpolation.json`; the category-point is skipped explicitly.
- Run3 signal samples scatter ±10–20% per-sample around any smooth
  acceptance curve (upstream of preprocessing; affects the production
  binned templates identically). Run2 is clean at 1–2%.

## Known limitations of the frozen model

- **Low-mA grid density is the binding constraint.** Below mA ≈ 45 the
  below-Z norm envelopes are driven by MHc115 (15 → 27 → 42) and MHc130
  (15 → 30 → 55). No functional form fixes this — every basis tried fails
  there identically. One extra MC point in each of those gaps would do
  more than any further model work.
- **No mHc interpolation.** The surfaces are better-constrained models AT
  the seven measured mHc; predicting an unmeasured study from the other
  six is a 4% median / 18% p90 yield error.
- **Closure pulls are uncalibrated** — the assumed 1% Run2 G error is far
  below the ~5% surface residual. The exported envelopes use relative
  residuals and are unaffected.
