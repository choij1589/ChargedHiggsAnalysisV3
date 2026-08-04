# PTOptimized — Dimuon-pT Threshold Method

A third template method alongside `Baseline` and `ParticleNet`, built for the
**ARC review** to test whether a cut on the reconstructed dimuon pT improves the
expected significance and limits at **low mA**, where `ParticleNet` was never
trained.

`PTOptimized` is deliberately the exact structural analogue of `ParticleNet`:
scan a threshold on a discriminant, keep the value that maximises the Asimov
significance per category, then build templates with that cut applied. The only
difference is the discriminant — the stored dimuon pT instead of the network
score.

This document doubles as the **status board** for the study (see
[Status board](#status-board)). Keep it updated as steps land.

---

## Scope

| Item | Value |
|---|---|
| Mass points | **all `mA <= 60`, every `mHc`** — 35 points, config key `ptoptimized` |
| Methods compared | `Baseline` vs `PTOptimized` (both must be run) |
| Channels | `SR1E2Mu`, `SR3Mu`, `Combined` |
| Eras | `Run2`, `Run3`, `All` |
| Binning | `extended` (unblind) |
| Purpose | ARC review study only — **not** part of the production limit chain |

The 35 points, from `configs/masspoints.json` key `ptoptimized`:

| mHc | mA values | n |
|---|---|---|
| 70  | 15, 18, 30, 40, 55 | 5 |
| 85  | 15, 21, 30, 42, 55 | 5 |
| 100 | 15, 24, 40, 60 | 4 |
| 115 | 15, 27, 42, 57 | 4 |
| 130 | 15, 30, 55 | 3 |
| 145 | 15, 35, 45, 60 | 4 |
| 160 | 15, 17, 20, 25, 30, 35, 40, 45, 50, 60 | 10 |

**Baseline must be run for these points too** — it is the comparison arm. Only
the 10 `MHc160` points already have Baseline templates (from the 2026-07-31
MHc160 reproduction); the other **25 have no Baseline templates** and must be
built. Preprocessed samples for all 35 exist on pnfs (verified: 560/560 sample
directories complete).

---

## Method

### Discriminant

The dimuon pT of the **same pairing** whose mass enters the template. This is
critical: `_select_mass()` picks `mass1` for `1E2Mu`/`2E1Mu`, and
`min`/`max(mass1, mass2)` for `3Mu` depending on `mHc >= 100 and mA >= 60`.
That boundary falls **inside this study's mass range**, separating `MA60` from
`MA<=50`, so a naive "always use `pT1`" would mismatch mass and pT for part of
the `SR3Mu` set.

`preprocess.py` therefore selects the pair once and stores both:

```python
def _select_pair(self, mass1, mass2, pT1, pT2):
    if "1E2Mu" in self.channel or "2E1Mu" in self.channel:
        return mass1, pT1
    elif "3Mu" in self.channel:
        if self.mHc >= 100 and self.mA >= 60:
            return (mass1, pT1) if mass1 >= mass2 else (mass2, pT2)
        return (mass1, pT1) if mass1 <= mass2 else (mass2, pT2)
```

`_select_mass()` is retained and delegates to `_select_pair()`, so the mass
selection is provably unchanged.

### Threshold scan

| Property | Value |
|---|---|
| Cut | `pT >= threshold` |
| Step | `PT_SCAN_STEP = 5.0` GeV |
| Ceiling | 99th percentile of the combined signal+background pT (`PT_SCAN_MAX_PERCENTILE`) |
| Figure of merit | Asimov significance, `evalSensitivity()` — shared with ParticleNet |
| Granularity | one threshold per category (`SR1E2Mu_Run2`, `SR3Mu_Run2`, ...) |
| Record | `threshold.{category}.json` in the template dir |

pT is unbounded, so the ceiling is derived from the data rather than using
ParticleNet's fixed `linspace(0, 1, 101)`. The scan range, step, and point count
are written into the payload so a threshold pinned at the scan edge is visible
rather than silent.

### Cut direction — physics motivation

The cut is fixed to `pT >= threshold`. This is not just the ParticleNet
analogue: at **low mA the A is produced with a large boost**, since the mass gap
`mHc - mA` is large and the A recoils hard. Its decay muons therefore form a
high-pT dimuon system, while the in-window background is comparatively soft.
Cutting from below on the dimuon pT is the physically motivated direction, and
is why the study targets `mA <= 60` specifically.

An empirical caveat to watch, measured inside the mass window (2018, SR1E2Mu):

| Mass point | Signal pT (median) | Background pT (median) | Favours |
|---|---|---|---|
| `MHc160_MA30` | 69.5 | 60.2 | `pT >= t` — gain expected |
| `MHc160_MA60` | 57.5 | 77.0 | `pT <= t` — **no gain expected** |

The boost argument holds at `MA30` as expected, but at the `mA = 60` edge of the
range the measured in-window signal was *softer* than the background — the
mass gap is smaller there and the window sits nearer the Z peak, where the
background is itself boosted. For such points the optimiser will settle at
`threshold ~ 0` and report ~0% improvement. That is a legitimate null result,
not a bug: check `improvement` in `threshold.{category}.json` before concluding
anything is broken, and expect the gain to fall off as mA approaches 60.

Scanning both directions was considered and rejected: the physics motivates one
direction, and a two-sided scan would double the trials for no motivated reason.

---

## Implementation

### `python/preprocess.py`

- reads `pT1`, `pT2` (present in **every** input type: signal, prompt
  backgrounds, all systematic trees, data, and MatrixAnalyzer nonprompt)
- writes a new `pT` output branch
- `_select_pair()` keeps mass and pT on the same pairing

### `python/makeBinnedTemplates.py`

Threshold cuts dispatch on a module-level `DISCRIMINANT`, set once per process
in `build_run_period_templates()`:

```python
DISCRIMINANT = "pT" if args.method == "PTOptimized" else "PN"
```

`makeBinnedTemplates.py` runs one method per process, so there is no
cross-contamination, and the many `getHist` / `getDataHist` /
`validateBackgroundStatistics` call sites keep their existing signatures. The
pT branch is checked *before* the `elif` that handles ParticleNet, so the
`Baseline` and `ParticleNet` paths execute exactly as before.

New functions:

| Function | Role |
|---|---|
| `loadPTDataset()` | load `(pT, weight)` inside the mass window |
| `getOptimizedPTThreshold()` | 5 GeV scan, returns best threshold + sensitivities |
| `optimizeCategoryPTThreshold()` | per-category driver, writes `threshold.{category}.json` |
| `_require_pt_branch()` | hard error if samples predate the pT branch |

`--method` in `makeBinnedTemplates.py` is a free-form string, so no argparse
whitelist change was needed; `main()` gained an explicit check for the three
known methods.

### Not implemented (intentional)

- `collectLimits.py` / `plotLimits.py` have **no** `PTOptimized` branch. This
  study is ARC-only, so limits are read directly from each mass point's
  `combine_output/asymptotic/` ROOT file rather than collected into
  `results/json`. This also keeps the source-of-truth limit JSONs untouched.
- No `PTOptimized` entry in the blinding/partial-unblind paths.

---

## Prerequisite: re-preprocessing

**The pT branch did not exist in the preprocessed samples.** It is present in the
raw SKNano inputs as `pT1`/`pT2` but was dropped by `preprocess.py`. Every mass
point in this study must be re-preprocessed before `PTOptimized` templates can
be built, otherwise `_require_pt_branch()` raises.

- 35 mass points x 8 eras x 2 SR channels = **560 preprocess jobs**
- Adding a branch is additive: existing `Baseline` templates stay valid, and
  `Baseline` does not read `pT`, so it does **not** need rebuilding
- Must not run while template jobs for the same mass points are in flight —
  `preprocess.py` calls `ensure_directory(..., clean=True)` on the shared
  pnfs sample directory

---

## Running

```bash
cd SignalRegionStudyV3
source setup.sh

# 1. re-preprocess all 35 points (needs a valid grid proxy: xrdcp writes)
#    submitted as a targeted condor cluster, not automize/preprocess.sh
#    35 points x 8 eras x 2 SR channels = 560 jobs

# 2a. Baseline arm. --masspoint-set overrides the list that --method would pick,
#     so Baseline runs over exactly the ptoptimized points.
./automize/makeBinnedTemplates.sh --mode all --method Baseline \
    --masspoint-set ptoptimized --binning extended --unblind

# 2b. PTOptimized arm (defaults to the ptoptimized list)
./automize/makeBinnedTemplates.sh --mode all --method PTOptimized \
    --binning extended --unblind

# single point, interactively
python3 python/makeBinnedTemplates.py --era Run2 --channel SR1E2Mu \
    --masspoint MHc160_MA30 --method PTOptimized --binning extended --unblind
```

`--method PTOptimized` selects `MASSPOINTs_PTOPTIMIZED` automatically.
`--masspoint-set baseline|particlenet|ptoptimized` overrides that for any
method — it exists so the Baseline comparison arm covers the identical list.

Inspect the chosen cut and the gain:

```bash
cat templates/All/Combined/MHc160_MA30/PTOptimized/extended_unblind/threshold.*.json
```

---

## Validation log

| Check | Result |
|---|---|
| Mass selection unchanged by `_select_pair` refactor | 200k random cases, **0 mismatches** |
| pT paired with the selected mass | 200k random cases, **0 violations** |
| pT branch written for all processes | signal, WZ, ZZ, conversion, nonprompt, data, others — all present |
| `mass` identical to pre-existing pnfs samples | **bit-identical** for all 7 processes (2016preVFP/SR1E2Mu/MHc160_MA30) |
| Systematic trees carry pT | **141/141** |
| Optimizer on real data | `MHc160_MA30`, SR1E2Mu, 2016preVFP: threshold **10 GeV**, sensitivity 1.871 -> 1.983, **+6.00%** |
| Stability vs scan grid | +6.02% (fine grid) vs +6.00% (5 GeV) — optimum is not a grid artefact |
| Baseline regression under the patch | **PASS** — `Run2/SR1E2Mu/MHc160_MA30` built with patched vs pristine `makeBinnedTemplates.py`: 1241 histograms, 0 binning mismatches, max abs difference over all bins **and** errors `0.000e+00`; `binning.json`, `categories.json`, `process_list.json`, `background_validation.json` byte-identical |
| pnfs samples for all 35 points | 560/560 directories complete before re-preprocessing |

---

## Status board

Update this section as steps complete.

| # | Step | State | Notes |
|---|---|---|---|
| 1 | `preprocess.py` stores `pT` | **DONE** | validated, mass bit-identical |
| 2 | `makeBinnedTemplates.py` PTOptimized patch | **DONE (sandbox)** | 8 hunks, compiles, idempotent |
| 3 | 5 GeV scan step | **DONE** | `PT_SCAN_STEP = 5.0` |
| 4 | `ptoptimized` key in `configs/masspoints.json` | **DONE** | 35 points, all `mA <= 60` |
| 5 | `--masspoint-set` + PTOptimized in `automize/makeBinnedTemplates.sh` | **DONE** | `load_masspoints.sh` exposes `MASSPOINTs_PTOPTIMIZED` |
| 6 | Baseline regression test under patch | **DONE** | 1241 histograms **bit-identical**, max diff 0.000e+00 |
| 7 | Apply patch to live tree | **DONE** | identical to sandbox-validated file |
| 8 | Re-preprocess 35 points (560 jobs) | **DONE** | needed 2 repair rounds (43 + 7 dirs); final verify 560/560 complete with pT |
| 9 | Baseline arm for the 35 points | **DONE 35/35** | all clean after corrupt-input repair |
| 10 | `PTOptimized` templates/datacards/asymptotics | **DONE 35/35** | all clean |
| 11 | Baseline vs PTOptimized comparison for ARC | **DONE** | table below; regenerate via `condor/jobs_pt_compare` |

**STUDY COMPLETE.** Both arms 35/35 clean, all 6160 sample files validated readable.

**Sentinel check (resolved):** `pT >= 0` is a true no-op at event level — zero
events with `pT < 0` in any sample. Verified directly for `MHc130_MA55`
(threshold 0 in all four categories): `shapes.root` is **bit-identical** between
the two arms (3844 histograms, none differing, none added or removed), and
`binning.json`, `process_list.json`, `background_validation.json`,
`background_weights.json` all match.

**Two interpretation caveats found while answering "why did X not move / why did
Y move".** Both are properties of the pipeline, not of the pT cut:

1. **The median expected limit is quantised to 1/1024 in `r`.** Every one of the
   70 median expected limits in this study is an exact integer multiple of
   `1/1024 = 0.00098` (e.g. `MHc100_MA24` is `183/1024 = 0.1787109375` in *both*
   arms). That is Combine's `AsymptoticLimits` bisection resolution. So
   **`ratio = 1.000` means "moved by less than 0.1%", not "nothing changed"**.
   `MHc100_MA24` is the clearest example: the cut really was applied (1928 of
   3448 histograms differ; `data_obs` goes 15->14 in `SR1E2Mu_Run2` and 7->4 in
   `SR1E2Mu_Run3`), the observed limit moves `0.24593 -> 0.25901` (+5.3%), and
   the non-quantised `+-1sigma`/`+-2sigma` bands move too — only the median
   happens to land on the same grid node.

2. **Threshold 0 does not guarantee an identical datacard.** 7 of the 13
   all-zero-threshold points have differing datacards (`MHc85_MA42`,
   `MHc85_MA55`, `MHc100_MA15`, `MHc100_MA60`, `MHc130_MA55`, `MHc145_MA45`,
   `MHc160_MA50`); the other 6 are identical. The difference is confined to the
   low-stat `shape?` -> lnN fallback bookkeeping: for `MHc130_MA55`, 19
   (bin, process) columns carry an lnN value in the PTOptimized card where
   Baseline has `-`, and 3 go the other way — all of them sparse processes
   (`others`, `ZZ`, `ttZ`, `conversion`) in individual suberas.
   `shapes_original.root` confirms it: 564 systematic-variation histograms are
   archived only in the PT arm and 80 only in Baseline, while every commonly
   archived histogram has an identical integral.

   So `MHc130_MA55` (1.007) and `MHc160_MA50` (0.993) are **not** limit-fit
   numerical noise, as an earlier revision of this doc claimed. They are genuine
   differences in the nuisance model reaching Combine, with identical input
   histograms. Root cause of the divergent bookkeeping is not yet identified —
   something in the PTOptimized path retains more low-stat systematic variations
   before the `SHAPE_REL_ERR_THRESHOLD = 0.30` test. Effect size is <= 0.7% and
   in both directions, so it does not change any conclusion below, but it should
   be fixed before these cards are reused for anything quantitative.

Counting the two affected points as unchanged, the honest tally is
**7 better / 7 worse / 21 unchanged**.

**Context — how this compares to ParticleNet** (MHc160, mA 70-115, disjoint mA
range, so the two studies are complementary rather than competing):

| | ParticleNet | PTOptimized |
|---|---|---|
| points improved | **8/8** | 7/35 |
| mean limit ratio | **0.76** | ~1.00 |
| best | 0.654 | 0.933 |
| typical Asimov gain | 20-42% | 0-11% |

ParticleNet delivers ~24% average limit improvement; PTOptimized is marginal and
confined to `mHc >= 130`. On this evidence pT is **not** a substitute for an MVA
at low mA — the honest ARC statement is that a simple kinematic cut recovers
only a small fraction of what the network achieves in its own mass range.

### Result (All/Combined, median expected limit ratio PTOpt/Baseline)

**8 better / 8 worse / 19 identical.** Best improvement ~6.7%.

| mass point | mHc | mA | thr (GeV) | avg gain % | r95 base | r95 PTOpt | ratio |
|---|---|---|---|---|---|---|---|
| MHc160_MA25 | 160 | 25 | 35-45 | 8.98 | 0.1611 | 0.1504 | **0.933** |
| MHc160_MA35 | 160 | 35 | 25-30 | 3.40 | 0.2402 | 0.2246 | **0.935** |
| MHc160_MA30 | 160 | 30 | 20-40 | 7.86 | 0.1953 | 0.1826 | **0.935** |
| MHc160_MA15 | 160 | 15 | 30-40 | 3.82 | 0.1279 | 0.1221 | **0.954** |
| MHc145_MA15 | 145 | 15 | 15-40 | 1.96 | 0.1221 | 0.1201 | 0.984 |
| MHc145_MA35 | 145 | 35 | 0-10 | 0.03 | 0.2402 | 0.2363 | 0.984 |
| MHc130_MA15 | 130 | 15 | 0-25 | 0.81 | 0.1289 | 0.1279 | 0.992 |
| MHc160_MA50 | 160 | 50 | 0-0 | 0.00 | 0.2852 | 0.2832 | 0.993 |
| MHc115_MA27 | 115 | 27 | 0-15 | 0.14 | 0.2383 | 0.2480 | 1.041 (worse) |
| MHc70_MA40 | 70 | 40 | 0-25 | 0.11 | 0.2773 | 0.2871 | 1.035 (worse) |
| **MHc160_MA20** | 160 | 20 | 30-65 | **11.54** | 0.1494 | 0.1543 | **1.033 (worse)** |
| MHc85_MA21 | 85 | 21 | 0-25 | 3.49 | 0.1689 | 0.1738 | 1.029 (worse) |
| MHc160_MA40 | 160 | 40 | 0-25 | 0.61 | 0.2285 | 0.2324 | 1.017 (worse) |

**Three conclusions:**

1. **Asimov gain does not predict limit improvement, and can invert it.**
   `MHc160_MA20` has the largest significance gain in the set (+11.5%) yet a
   3.3% *worse* limit. `evalSensitivity` maximises `S/sqrt(B)` on raw event
   counts with **no systematics**, while the limit comes from the full binned
   fit — cutting events forces coarser adaptive binning and inflates the
   autoMCStats per-bin nuisances, which can cost more than the cut buys. If the
   method is pursued further, the optimisation objective should be the expected
   limit, not `S/sqrt(B)`.
2. **The effect is driven by the mass gap, not mA alone.** Every improvement is
   at `mHc >= 130`; at `mHc = 70-115` the method is neutral or slightly harmful.
   That matches the boost motivation (`mHc - mA` large => boosted A).
3. **Graceful degradation.** Where pT does not help, the optimiser selects
   threshold 0 and the limit is *exactly* Baseline (ratio 1.000).

### Data-integrity note (important for interpreting any rerun)

Re-preprocessing suffered **silent xrdcp write failures under concurrency**
(560-way). Three rounds of repair were needed:

- round 1: 43 sample dirs each missing one random file
- round 2: 7 more, introduced by the repair itself
- round 3: 3 files present but **unreadable** (truncated)

A missing background file does *not* crash template-making —
`validateBackgroundStatistics` marks it `missing_file` and drops the process —
so the comparison would have been silently biased. An existence check is **not
sufficient**: `condor/jobs_pt_validate/` opens all 6160 sample files and checks
for the `Central` tree. Run it before trusting any PTOptimized numbers, and
throttle preprocessing (`max_materialize`) to limit the failure rate.

### Preview result (Run2, SR1E2Mu, MHc160) — before the full run

Produced with the real pipeline machinery (DCB mass window + 5 GeV scan):

| mA | window (GeV) | best pT cut | Z baseline | Z optimised | gain |
|---|---|---|---|---|---|
| 15 | 13.6-16.4 | 35 GeV | 5.031 | 5.292 | +5.2% |
| 30 | 26.9-33.0 | 30 GeV | 4.053 | 4.357 | +7.5% |
| 45 | 40.2-49.7 | none | 3.070 | 3.070 | 0.00% |
| 60 | 53.5-66.5 | none | 2.787 | 2.787 | 0.00% |

**Sharp turn-off:** the method gains ~5-7% expected significance at `mA <= 30`
and exactly nothing from `mA >= 45`, where the optimiser drives the threshold to
0 (no cut improves the Asimov significance). Consistent with the boost
motivation — the discrimination comes from the `mHc - mA` gap. Expect roughly
half the 35-point set to return a legitimate null result.

### Decisions on record

- Cut direction fixed to `pT >= threshold` (user decision), motivated by the
  large boost of a light A — accepted null result near the `mA = 60` edge.
- Scope is **all `mA <= 60` regardless of `mHc`** (user decision), not MHc160
  only; Baseline must be run over the same list.
- Scan step 5 GeV (user decision).
- No `collectLimits.py` integration (user decision) — ARC review only.
- No new git worktree; work happens in the main tree.

### Related docs

- `docs/LOWSTAT.md` — low-stat and autoMCStats behaviour, unchanged by this method
- `docs/RUN_PERIOD_COMPONENT_TEMPLATES.md` — the Run-period category construction
  that `PTOptimized` reuses unmodified
