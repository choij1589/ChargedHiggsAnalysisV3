# LEE.md — Global p-value (Look-Elsewhere Effect) for the Maximum Excess

Status: procedure implemented through global p-value collection.

This document specifies why and how we compute the global p-value for the
largest local excess of the full-unblind Baseline scan, requested by the
reviewer. It is written so an agent can plan and execute the implementation
without further design decisions.

## 1. Problem Statement

The unblind Baseline scan (era `All`, channel `Combined`, binning
`extended_unblind`) shows its largest local excess at `MHc70_MA18`:

- Local observed significance: `Z_obs = +2.056` (uncapped),
  `p_local = 0.020` (one-sided).
- Computed with `combine -M Significance workspace.root --uncapped=1
  --rMin=-20 --rMax=20 -m 120` via `scripts/runSignificance.sh`.
- Summary of local extremes: `results/json/significance.json`.

The reviewer asks for the **global** p-value of this maximum excess:

```
p_global = P( max over trials set of Z(m)  >=  Z_obs | background-only )
```

## 2. Why the Combine Documentation Recipe Does Not Apply

The documented LEE procedure
(<https://cms-analysis.github.io/HiggsAnalysis-CombinedLimit/latest/part3/nonstandard/#look-elsewhere-effect-for-one-parameter>)
assumes a *parametric* search: one workspace in which the resonance mass is a
continuous parameter, so `Z(m)` can be scanned finely and the Gross–Vitells
upcrossing formula applied.

V3 is template-based: each mass point is an independent workspace with its own
signal template, its own ±10σ mass window, and its own adaptive binning. There
is no continuous `Z(m)`, and combine's internal toy generation cannot be used
per-workspace, because toys thrown independently per mass point are
uncorrelated across the scan — a global p-value requires every toy to be a
**single pseudo-dataset evaluated coherently at all mass points**.

Therefore we use brute-force background-only toys generated at the
event/observable level and projected into every mass point's frozen binning.
This is feasible precisely because the excess is small (~2σ): the global
p-value is large, so a few hundred toys measure it precisely, and no
asymptotic (Gross–Vitells) extrapolation is needed.

## 3. Trials Set Definition and Justification

**Trials set S′ = all Baseline mass points with `MHc < 100 OR MA < 60`
(35 of the 78 scanned points).**

Rationale:

1. **Observable boundary, not an arbitrary cut.** In `python/preprocess.py`
   (`_select_mass`), the SR3Mu fitted observable is `max(mass1, mass2)` iff
   `MHc >= 100 AND MA >= 60`, else `min(mass1, mass2)`. S′ is exactly the
   complement: the maximal subset of the scan on which the observables are
   uniform (`mass1` in SR1E2Mu, `min(mass1, mass2)` in SR3Mu). This makes toy
   generation one-dimensional per category and the set definition principled.
2. **Conservative bound.** For any subset S′ ⊂ S,
   `max over S′ <= max over S`, hence `p_global(S′) <= p_global(S)`. Quoting
   `p_global(S′)` *overstates* the global significance; if it is already
   large, the full-scan value is necessarily larger. State this explicitly
   when quoting the result.
3. The set contains the observed maximum and its closest competitors
   (`MHc160_MA17`, `MHc160_MA50`), so it cannot be accused of removing rival
   fluctuations.

Observed significances over S′ (from `scripts/runSignificance.sh`, all 35 run;
outputs in `templates/All/Combined/{MP}/Baseline/extended_unblind/
combine_output/significance/`):

- Maximum: `MHc70_MA18  Z = +2.056`  ← the quoted excess
- Next: `MHc160_MA17 +1.787`, `MHc160_MA50 +1.773`, `MHc100_MA24 +1.008`
- Minimum: `MHc115_MA27 -1.358`

The 35 points (6+7 for MHc70/85 full MA range; MA < 60 only for MHc ≥ 100):

```
MHc70:  MA15 MA18 MA30 MA40 MA55 MA65
MHc85:  MA15 MA21 MA30 MA42 MA55 MA70 MA80
MHc100: MA15 MA24 MA40
MHc115: MA15 MA27 MA42 MA57
MHc130: MA15 MA30 MA55
MHc145: MA15 MA35 MA45
MHc160: MA15 MA17 MA20 MA25 MA30 MA35 MA40 MA45 MA50
```

None of these are ParticleNet-trained points, so all use the standard (non
`NoHistMode`) preprocessed inputs and share identical background event lists.
ParticleNet is a separate 17-point grid and is **not** covered here; if a
global p-value is requested for it, run the same machinery on its own grid.

## 4. Statistical Conventions (fix these; do not re-decide)

1. **Generation model**: nominal pre-fit background with per-process block
   flooring (Step 1), nominal nuisance values. Each toy *fit* profiles all
   nuisances exactly as the data fit did ("nominal toys, profiled fits") —
   standard for a sub-3σ LEE quote. The flooring scheme is chosen to track
   the frozen datacard backgrounds; residual per-point expectation offsets
   of a few % (up to ~±8%) are irreducible because the 35 datacards are
   mutually inconsistent at that level, so toy Z means within ±0.3 of zero
   are accepted and p_global carries a corresponding mild model dependence.
2. **Frozen statistical model**: datacards, adaptive binning, lowstat/shape
   fallbacks, autoMCStats settings are kept exactly as in production; only
   `data_obs` changes per toy. This conditioning is correct because the
   binning was derived from MC, not from data.
3. **Test statistic**: identical to data — `combine -M Significance
   --uncapped=1 --rMin=-20 --rMax=20 -m 120`, one-sided excess counting
   (`Z_t_max >= Z_obs`). `Z_t_max` is computed from the coherent 35-point toy
   scan; `Z_obs` is read only for the requested observed maximum mass point.
4. **Estimator**: `p_global = (1 + N_exceed) / (N_toys + 1)` (unbiased
   frequentist convention), binomial uncertainty `sqrt(p(1-p)/N_toys)`.
   `N_toys = 1000` (precision ±0.016 at p ≈ 0.5; ±0.01 at p ≈ 0.15).
5. `Z_obs` is read from the observed Significance ROOT output for
   `--masspoint` (currently `MHc70_MA18`), never hardcoded.

## 5. Exact Procedure

Environment for every step: `cd SignalRegionStudyV3 && source setup.sh`
(module-local; provides combine and PyROOT via cmsenv).

### Step 1 — Build the background generation model (run once)

Implemented scripts:

```bash
./automize/LEE.sh --step 1 --masspoint MHc70_MA18
./automize/LEE.sh --step 1 --masspoint MHc70_MA18 --local
```

Condor is the default mode; `--condor` is accepted as an explicit no-op.  The
worker script is `scripts/prepareLEEModel.sh`, which sources the module-local
`setup.sh` and runs `python/prepareLEEModel.py`.

The allowed LEE mass points are stored in `configs/masspoints.json` under the
top-level `LEE` key. `--masspoint` must be a member of that list. The default
and current production generation model point is `MHc70_MA18`.

- Fit categories of the All/Combined datacard and their suberas:
  - `SR1E2Mu_Run2`, `SR3Mu_Run2` ← 2016preVFP, 2016postVFP, 2017, 2018
  - `SR1E2Mu_Run3`, `SR3Mu_Run3` ← 2022, 2022EE, 2023, 2023BPix
- Inputs per (subera, channel): every background file in
  `samples/{subera}/{channel}/{masspoint}/` **except** `data.root` and the
  signal file `{masspoint}.root` (expected: nonprompt, WZ, ZZ, ttW, ttZ, ttH,
  tZq, conversion, others). Tree `Central`, branches
  `mass1, mass2, weight`. Fail fast (`FileNotFoundError`) on missing
  files — no silent skips.
- Observable per event: SR1E2Mu → `mass1`; SR3Mu → `min(mass1, mass2)`.
  Mirror `_select_mass` (all 35 points are in the min-rule region); do not
  reuse the per-masspoint `mass` branch blindly.
- Per (subera, process) input file, fill a fine-binned TH1D of the observable
  with net weights: **bin width 0.1 GeV, range [10, 100] GeV** (covers all 35
  windows; assert this against `mass_min`/`mass_max` of every point's
  `binning.json`).
- **Per-process block flooring** (`--floor-block-width`, default 0.5 GeV):
  within each block, clip the process block total at zero and redistribute it
  across the block's fine bins proportionally to the positive fine-bin
  content. Sum the floored process histograms into the category model. This
  mimics the per-process negative-bin flooring that `makeBinnedTemplates.py`
  applies to the datacard templates; a net-sum floor undershoots the frozen
  datacard backgrounds by up to ~10% and biases toy Z values negative (this
  was the v1 campaign failure mode).
- **Consistency check** (recorded in `bkg_model.json` under `consistency`):
  for every trial point and category, the model is projected onto the point's
  mass window (fractional fine-bin overlap) and compared to the datacard
  background (sum of all process histograms in `shapes.root` excluding
  `data_obs`, `signal_*`, and Up/Down variations, deduplicating ROOT key
  cycles). A WARNING is logged for any |1 − ratio| > 0.10. Residual
  deviations of a few % are irreducible: the datacards themselves are
  mutually inconsistent at that level because per-process flooring in each
  point's own coarse bins injects binning-dependent yield.
- Output: `LEE/{masspoint}/model/bkg_model.root` — four TH1Ds named by category,
  plus a JSON sidecar with total yields `B_c`, provenance (input files, raw and
  post-floor sums), the flooring scheme, and the consistency ratios.

Fine-binned Poisson sampling (Step 2) is used instead of unbinned event
resampling because weighted events include negative weights: sampling events
with probability proportional to weight is undefined for negative weights, so
a tree/unbinned generation model is not usable. The fine-binned expectation
with per-process block flooring is the sampling density. Note also that the
frozen analysis bin *edges* are arbitrary floats from the DCB fits (not
aligned to a 0.1 GeV grid), so toys must be event-level (TTrees with uniform
in-bin positions); a pure histogram rebin cannot reproduce the frozen binning.

### Step 2 — Generate toys

Implemented scripts:

```bash
./automize/LEE.sh --step 2 --masspoint MHc70_MA18 --ntoys 1000
./automize/LEE.sh --step 2 --masspoint MHc70_MA18 --toy 1 --local
```

Condor is the default mode and creates one DAG job per toy. The worker script
is `scripts/generateLEEToys.sh`, which sources the module-local `setup.sh` and
runs `python/generateLEEToys.py`.

Each toy is generated from the nominal Step 1 background model only. Systematic
nuisance parameters are **not** fluctuated during generation; they are profiled
later when the toy is fit in Step 3.

- Seed the RNG deterministically: `seed = 12345 + T`.
- Per category: for each fine bin `i`, draw `n_i ~ Poisson(mu_i)`; place the
  `n_i` toy events uniformly within the fine bin. (Equivalent to Poisson-total
  + multinomial; uniform placement removes edge-alignment artifacts against
  the analysis bin edges.)
- Output per toy:
  - `LEE/{masspoint}/toys/toy_{T:04d}.root`
  - `LEE/{masspoint}/toys/toy_{T:04d}.json`
- The ROOT file contains four TTrees named by category:
  `SR1E2Mu_Run2`, `SR3Mu_Run2`, `SR1E2Mu_Run3`, `SR3Mu_Run3`.
- Each tree stores branches `mass/D` and `weight/D`, with `weight = 1.0`.
- Existing complete toy outputs are skipped unless `--force` is used.
- These TTrees are the coherent pseudo-dataset shared by all 35 points; Step 3
  projects them into each point's frozen binning and fits them.

### Step 3 — Project into each mass point and fit in batch

Implemented scripts:

```bash
./automize/LEE.sh --step 3 --masspoint MHc70_MA18 --toy 1 --local
./automize/LEE.sh --step 3 --masspoint MHc70_MA18 --ntoys 1000
```

Condor is the default mode and creates one DAG job per toy; this is the batch
execution step. The worker script is `scripts/fitLEEToy.sh`, which sources the
module-local `setup.sh` and runs `python/fitLEEToy.py`.

One Step 3 job reads one coherent toy pseudo-dataset from
`LEE/{masspoint}/toys/toy_{T:04d}.root`, loops over every mass point in
`configs/masspoints.json:LEE`, and evaluates the frozen production likelihood.

- Read the point's per-category `bin_edges` (and `mass_min`/`mass_max`) from
  `templates/All/Combined/{MP}/Baseline/extended_unblind/binning.json`.
  **Never recompute binning.**
- Fill each category's toy events into a TH1D with exactly those edges;
  events outside the window fall out of range and are dropped (as in data).
- Stage a temporary fit directory (use `$TMPDIR` / job scratch):
  1. Copy `datacard.txt` and `shapes.root` from the template directory.
  2. In the copied `shapes.root`, overwrite `{category}/data_obs` with the
     toy histogram (open UPDATE; write with `TObject::kOverwrite`; ensure the
     old cycle is replaced, e.g. delete `data_obs;*` first).
  3. In the copied `datacard.txt`, replace the numbers on the `observation`
     line with `-1` for all four bins. The production datacard hardcodes the
     observed yields, and text2workspace validates them against `data_obs`;
     `-1` makes combine take the rate from the shapes file.
- Run in the staged directory:

  ```
  combine -M Significance datacard.txt --uncapped=1 --rMin=-20 --rMax=20 -m 120 -n .lee_toy
  ```

  Parse `Z` from the `limit` tree of the output file. On failure or
  non-convergence, retry once with `--rMin=-40 --rMax=40`; if it still fails,
  record `null` (collector accounts for these; the failure rate must stay
  well below 1%, otherwise stop and investigate).
- Clean the staging directory after each point.
- Output per toy: `LEE/{masspoint}/fits/toy_{T:04d}.json` with
  `{ "seed": ..., "Z": {masspoint: Z or null}, "Z_max": ..., "Z_min": ... }`,
  plus per-masspoint projection counts and fit-attempt metadata.
- Existing complete fit outputs are skipped unless `--force` is used.

### Step 4 — Collect and quote the global p-value

Implemented scripts:

```bash
./automize/LEE.sh --step 4 --masspoint MHc70_MA18 --ntoys 1000
./automize/LEE.sh --step 4 --masspoint MHc70_MA18 --ntoys 1000 --local
```

Condor is the default mode and submits one lightweight collection job. The
worker script is `scripts/collectLEE.sh`, which sources the module-local
`setup.sh` and runs `python/collectLEE.py`.

- `--masspoint` is the fixed observed maximum-excess mass point and the LEE
  output namespace. For the current result this is `MHc70_MA18`.
- Read `Z_obs` only from
  `templates/All/Combined/{masspoint}/Baseline/extended_unblind/
  combine_output/significance/higgsCombine*.root`. Do not rescan all observed
  mass points in Step 4; the maximum point is already defined by `--masspoint`.
- Load Step 3 toy fit JSONs from `LEE/{masspoint}/fits/toy_{T:04d}.json`.
  Each expected toy must be complete: 35 finite `Z` values matching
  `configs/masspoints.json:LEE`, `failure_count == 0`, and no failed mass
  points. Missing or incomplete toys block the official quote.
- Per toy, compute `Z_t_max` as the maximum fitted `Z` across the 35 trial
  points stored in that toy JSON.
- Compute `p_global = (1 + #{ t : Z_t_max >= Z_obs }) / (N + 1)` and the
  binomial uncertainty `sqrt(p(1-p)/N)`.
- `Z_global = Phi^-1(1 - p_global)` is quoted only if `p_global < 0.5`;
  otherwise quote the p-value alone.
- Outputs:
  - `results/lee/global_pvalue.json` — p_global, uncertainty, N_toys,
    exceedance count, Z_obs, and trials-set definition.
  - `results/lee/global_pvalue.txt` — compact human-readable summary.
  - `results/lee/plots/zmax_distribution.(png|pdf)` — distribution of
    `Z_t_max` with a line at `Z_obs` (median of `Z_t_max` is the expected
    background-only maximum).

### Step 5 — Validate the toy chain and quote inputs

Implemented scripts:

```bash
./automize/LEE.sh --step 5 --masspoint MHc70_MA18 --ntoys 1000
./automize/LEE.sh --step 5 --masspoint MHc70_MA18 --ntoys 1000 --local
```

Condor is the default mode and submits one lightweight validation job. The
worker script is `scripts/validateLEE.sh`, which sources the module-local
`setup.sh` and runs `python/validateLEE.py`.

- Missing or corrupt required inputs are fatal. Statistical validation
  thresholds are written as `pass`, `warn`, or `fail` statuses in the report;
  they do not hide the diagnostic plots behind a Condor failure.
- Inputs:
  - Step 1 model: `LEE/{masspoint}/model/bkg_model.{root,json}`
  - Step 2 toys: `LEE/{masspoint}/toys/toy_{T:04d}.{root,json}`
  - Step 3 fits: `LEE/{masspoint}/fits/toy_{T:04d}.json`
  - Step 4 result: `results/lee/global_pvalue.json`
- Outputs:
  - `results/lee/validation/validation.json`
  - `results/lee/validation/validation.txt`
  - `results/lee/validation/plots/z_calibration_mean.(png|pdf)`
  - `results/lee/validation/plots/z_calibration_width.(png|pdf)`
  - `results/lee/validation/plots/toy_closure_{category}.(png|pdf)`
  - `results/lee/validation/plots/toy_yields.(png|pdf)`

Validation checks:

1. **Completeness**: require all expected toy ROOT/JSON files and fit JSON
   files; require each fit JSON to contain 35 finite `Z` values matching
   `configs/masspoints.json:LEE`, with `failure_count == 0`.
2. **Per-point calibration**: across toys, compute each mass point's fitted
   `Z` mean and width. `fail` if `|mean| > 0.30`; `warn` if `|mean| > 0.15`
   or the width is outside `[0.80, 1.10]`. Residual mean offsets are expected
   because the datacards are mutually inconsistent at the few-% level
   (per-process coarse-bin flooring); widths below 1 are the expected
   over-coverage of nominal toys fit with profiled nuisances.
3. **Toy closure**: compare toy total yields with Step 1 expected yields per
   category, and overlay average toy spectra on the Step 1 generation model.
4. **Model sanity**: report Step 1 flooring fractions (`fail > 0.20`,
   `warn > 0.10` — the per-process block floor intentionally adds yield),
   and compare background `Central` tree entries and weight sums between
   `--masspoint` and `--reference-masspoint` (default `MHc160_MA50`) for the
   same subera/channel/process files.
5. **Model/datacard consistency**: gate on the Step 1 `consistency` ratios
   (model window integral / datacard background): `fail` if any
   `|1 - ratio| > 0.15`, `warn` above `0.10`. This is the check that catches
   a biased generation model before any toys are fit.
6. **Stability and cross-check**: recompute p_global from the first half of
   toys and from all toys; require compatibility within the combined binomial
   uncertainty. Also report the Šidák independent-trials value,
   `1 - (1 - p_local)^35`, which should be above the toy p-value for the
   correlated scan.

## 6. How to Quote the Result (AN wording)

> The largest local excess of the Baseline scan is observed at MHc70_MA18
> with a local significance of 2.06σ (p_local = 0.020). The global
> probability for a background fluctuation at least as large anywhere in the
> low-mass scan region (MHc < 100 GeV or mA < 60 GeV, 35 mass points — the
> region sharing a common SR3Mu observable definition) is p_global = X ± Y,
> estimated from N background-only pseudo-experiments evaluated coherently at
> all mass points with the statistical model frozen to the one used on data.
> Since this restricted trials set can only decrease the global p-value, the
> full-scan global p-value is necessarily larger; the excess is consistent
> with a background fluctuation.

## 7. Cost Summary

| Item | Size |
| --- | --- |
| Observed scan (done) | 35 fits |
| Toy fits | 35 × 1000 = 35k fits, ~100–300 CPU-h |
| Condor jobs | 1000 (one per toy, ~10–20 min each) |
| Precision on p_global | ±0.016 at p = 0.5; ±0.011 at p = 0.15 |
