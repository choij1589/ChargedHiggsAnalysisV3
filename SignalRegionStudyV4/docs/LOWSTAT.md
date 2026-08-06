# Low-Statistics Background Treatment

Low-stat backgrounds can produce non-positive bins, >100% per-bin stat error,
and shape variations dominated by MC noise. This doc catalogs every mitigation
in the pipeline, grouped by the level at which it acts. For the exact order
they execute, see the "Ordering" block near the bottom. Every constant here is
frozen — the reproduction test (`docs/REPRODUCTION.md`) pins the resulting
datacards bitwise.

## Thresholds

| Constant | Value | File | Meaning |
|---|---|---|---|
| `BIN_FLOOR_VALUE`  | `1e-6` | `template_utils.py` | per-bin content/error floor for signal and `others` (not individual bkg) |
| `AUTOMC_THRESHOLD` | `5`    | `template_utils.py` | adaptive binning quality: n_eff per bin, matches autoMCStats threshold |
| `min_total_events` | `1`    | `makeBinnedTemplates.py` | diagnostic: logs WARNING when raw bkg yield falls below this |
| `SYST_MERGE_THRESHOLD` | `2.0` | `makeBinnedTemplates.py` | merge bin if total-bkg `sigma_syst / nominal > 2.0` |
| `SHAPE_REL_ERR_THRESHOLD` | `0.30` | `printDatacard.py` | switch shape -> `shape?` + lnN above 30% relative stat error |
| `MAX_LNN_VALUE` | `2.0` | `printDatacard.py` | cap on lnN fallback (100% uncertainty) |
| `MIN_YIELD_THRESHOLD` | `1e-6` | `printDatacard.py` | yield below which per-syst fallback is skipped (`-`) |
| `autoMCStats` threshold | `5` | `printDatacard.py` | Barlow-Beeston-lite per-bin stat nuisances |

---

## Process level

### P1. Always-separate process list (static across all eras)

**File**: `python/makeBinnedTemplates.py:category_background_processes()` and
`validateBackgroundStatistics()`

Every MC background is kept as its own subera component column inside the merged
Run-period categories. The list is derived from `configs/samplegroups.json`
top-level MC keys:

```
["nonprompt", "WZ", "ZZ", "ttW", "ttZ", "ttH", "tZq", "conversion"]
```

`validateBackgroundStatistics` still measures each process's raw weighted yield
under the mass window (and PN score cut for ParticleNet method). Results:

- `total_events >= min_total_events` → **present** (logged INFO)
- `total_events < min_total_events` → **low_stat** (logged WARNING; process is
  still kept separate; `ensure_positive_integral` + S1 shape→lnN fallback
  handle the low-stat bins)
- `total_events <= 0` → **empty** (logged WARNING; kept separate; floor + S1
  fallback apply)
- sample file not found → **missing_file** → added to `dropped_missing` (logged
  ERROR); only legitimate case for a column to differ between eras

The Run-period builder records the active subera components in
`process_list.json`; missing input files are the only reason for omitting a
component:
```json
{
  "construction": "run_period_components",
  "process_components": [
    {"category": "SR1E2Mu_Run2", "name": "WZ_2017", "base_process": "WZ"}
  ],
  "physics_groups": {
    "WZ": ["WZ_2016preVFP", "WZ_2016postVFP", "WZ_2017", "WZ_2018"]
  },
  "dropped_missing": []
}
```

**Why static?** Combine correlates nuisance parameters by name. If ttZ is a
separate component in one subera but merged into `others` in another, the
intended nuisance mapping becomes ambiguous. With component processes, every
available subera contribution keeps its own column and era-specific nuisance
attachment while the Poisson category is merged at the Run-period level.

### P2. Dropping zero/negative-yield processes from the datacard

**File**: `python/printDatacard.py:__init__()`

After templates exist, any process whose final nominal yield is `<= 0` is
dropped from the datacard with a WARNING. In practice `ensure_positive_integral`
(B1) guarantees every histogram has positive integral before `shapes.root` is
written, so P2 is a safety net rather than an expected trigger.

### P3. Per-systematic filter for the merged `others` template

**File**: `python/makeBinnedTemplates.py` (`others` systematic-building loop)

When the merged `others` template is being built, a systematic is only
applied if `"others"` is explicitly in its `group`. All instrumental systs
(pileup, btag, JES, lepton ID/res/scale, MET) list `"others"` in their group,
so they pass through. Per-process normalization systs do **not**:
`CMS_B2G25013_Norm_WZ_13p6TeV` has `group=["WZ"]` and would fail to build on
the merged template anyway (the syst tree only exists in `WZ.root`).

Those per-process norms are then also filtered out in the datacard by
`printDatacard.py:generate_systematic_lines` (same `group` semantics: no
relevant process → line dropped). The merged bucket's normalization uncertainty
is covered by a single flat lnN, `CMS_B2G25013_Norm_others` (50%,
`group=["others"]`), which is conservative and correlated across Run2/Run3.

---

## Bin level

### B1. Per-bin floor/zero (`ensure_positive_integral`) — post-selection only

**File**: `python/template_utils.py:ensure_positive_integral(hist, floor_mode)`

Called on **every final histogram after adaptive binning has been locked in**
— nominal, all syst Up/Down variations (preprocessed shape, valued shape,
envelope), signal, backgrounds, merged others — and paired immediately with
`cap_stat_errors` (see B3). The `floor_mode` parameter controls how
non-positive bins are handled:

- **`floor_mode="floor"`** (signal and `others`): set `content = error =
  BIN_FLOOR_VALUE = 1e-6`. Guarantees positive bins for Combine's vertical
  morphing and ensures total background is never zero in any bin.

- **`floor_mode="zero"`** (individual background processes): set `content = 0,
  error = 0`. Combine sees `sigma_p = 0` for this process in this bin and
  **skips it** in autoMCStats — no phantom NP is created.

In both modes, negative content is logged as a WARNING.

**Not applied inside the adaptive-binning trial loop.** The loop must see
honest per-process stats so that low-n_eff bins force coarser binning (see B2).
Applying the floor pre-check would hide dead bins and accept fine binnings
that trigger per-process autoMCStats NPs.

### B2. Adaptive binning loop

**File**: `python/makeBinnedTemplates.py` (adaptive binning optimization block)

The extended binning scheme is nominally `15 core + 2 sideband = 17 bins`
(see `calculate_adaptive_bins` in `template_utils.py`). The scan is done
per merged Run-period category using the summed category background over all
subera component processes, looping `n_core in [15, 14, ..., 5]`.

For each candidate:
1. Build candidate edges with `calculate_adaptive_bins(x0, sigma_eff, n_core)`.
2. Fill `Central` histograms for every separate process and the `others`
   merged template. **No hygiene is applied** — the test histograms are raw
   `getHist` / `getHistMerged` output. Honest per-process stats drive the
   binning decision.
3. `check_binning_quality(test_hists)` builds `h_total` (sum over all
   processes via `TH1::Add`, which propagates errors in quadrature) and
   checks a **single criterion** matching Combine's autoMCStats algorithm:
   - `n_eff = round(y^2 / sigma^2) >= AUTOMC_THRESHOLD (=5)`
   - where `y` = total background content, `sigma^2` = sum of squared
     errors per bin.
   - Bins with non-positive content also fail (n_eff undefined).
   - This directly guarantees all bins get BB-lite treatment (1 NP per bin)
     rather than per-process treatment (N_proc NPs per bin).
4. If clean: `break` and adopt this `n_core`.
5. Otherwise log up to five diagnostic messages and try the next smaller `n_core`.

If the `for` loop exhausts without a `break`, the coarsest candidate is kept
and a fallback is flagged (see B3).

**Key invariant**: `ensure_positive_integral` and `cap_stat_errors` are applied
only *after* the loop has picked a binning, as numerical hygiene on the final
shapes that go to `shapes.root`. They never influence which `n_core` is
selected.

### B3. Bin floor when adaptive binning is exhausted (`apply_floor`)

**File**: `python/makeBinnedTemplates.py` (`apply_floor` safety-net block)

When the adaptive loop gave up, `apply_floor = True` triggers a per-bin sweep
over **every histogram** in each separate process and in `others` — both the
nominal and every `{syst}Up/Down` variation. The floor is **process-dependent**:

- **`others`**: `content <= 0` → set `content = error = BIN_FLOOR_VALUE = 1e-6`.
  Ensures no bin is completely empty for the total background.
- **Individual backgrounds**: `content <= 0` → set `content = 0, error = 0`.
  Combine sees `sigma_p = 0` and skips autoMCStats for this process in this bin.
- **Both**: `err / content > 1.0` → cap `error = content` (100% error).

Recorded in `binning.json` as `floor_applied: true`.

### B4. Post-binning systematic-driven bin merging

**File**: `python/makeBinnedTemplates.py`, calling
`python/template_utils.py:apply_syst_driven_merging()`

Motivation: even after adaptive binning succeeds and templates are filled with
all systematic variations, individual bins can still have a *systematic*
envelope that exceeds the nominal — for example a sudden-dip bin where
`JES_Up` goes to zero while `JES_Down` doubles. A bin with
`sigma_syst >> nominal` is not meaningfully constrained and can pull the fit.

Algorithm (iterative, operates on numpy content/variance snapshots of every
template to avoid repeated TH1 rebinning):

1. Build per-bin total-background `nominal`, `sigma_syst` (envelope from
   `max(|Up-nom|, |Down-nom|)` summed in quadrature over all collected
   systematic names), and `stat_err` arrays via
   `_total_bkg_syst_from_snapshot`.
2. Compute `rel = sigma_syst / nominal`; find `worst = argmax(rel)`.
3. If `rel[worst] <= SYST_MERGE_THRESHOLD = 2.0`, converged — break.
4. Otherwise pick merge neighbour via `select_merge_neighbor`: whichever
   neighbour has the higher **relative stat error** (`stat_err / nominal`),
   i.e. prefer merging into the statistically weakest side so we do not
   hurt a well-measured neighbour.
5. Merge bins `(lo, lo+1)` in the snapshot (`_merge_snapshot` adds contents
   and variances bin-wise across every process and every systematic
   variation), delete the dropped edge from `current_edges`, log the merge.
6. Repeat until convergence or `nbins <= min_nbins = 3`.

At convergence, every histogram in `templates` (signal nominal, all syst
Up/Down, backgrounds, data_obs) is rebinned **once** to `current_edges` via
`rebin_hist_with_edges`, which is a thin wrapper around `TH1.Rebin` that keeps
histograms detached (`SetDirectory(0)`).

Merged bin edges, the merge count, and the threshold are recorded in
`binning.json` (`syst_merge_applied`, `n_bins_merged`, `syst_merge_threshold`).

---

## Systematic / Datacard level

### S1. Shape -> `shape?` fallback for low-stat processes

**File**: `python/printDatacard.py`

For every background process compute
```
rel_err = sqrt(sum of bin_error^2) / integral
```
on the final nominal histogram. Processes with `rel_err > SHAPE_REL_ERR_THRESHOLD = 0.30`
(30%) are flagged as low-stat. For these:

1. Per-systematic lnN values are pre-computed in `precompute_lnn_fallbacks()`:
   ```
   rate_effect = max(|Up_int - Cen_int|, |Down_int - Cen_int|) / Cen_int
   lnn_value   = min(1.0 + rate_effect, MAX_LNN_VALUE=2.0)
   ```
   - effects smaller than 0.1% are written as `-` (no nuisance)
   - yields below `MIN_YIELD_THRESHOLD = 1e-6` are skipped (`-`)
2. The Up/Down shape histograms for that (process, syst) are physically
   **removed** from `shapes.root` by `rewrite_shapes_root()` (see S3).
3. The datacard line uses Combine's `shape?` type: Combine will use the shape
   histogram if it exists, otherwise it falls back to the lnN value in that
   column. Because step 2 removed the histograms, the lnN branch is taken.

`rel_err` is used rather than absolute yield because it directly measures
template statistical quality: a small-yield process with many MC entries is
well-determined, whereas similar yield with few entries is not.

### S2. autoMCStats (Barlow-Beeston-lite)

**File**: `python/printDatacard.py:generate_automc_line()`

Threshold is `5` (effective MC entries per bin). Bins at or above 5 effective
entries get a single Barlow-Beeston-lite nuisance; below-threshold bins get
full per-process stat nuisances.

### S3. Non-destructive `shapes.root` rewrite

**File**: `python/printDatacard.py:rewrite_shapes_root()`

When S1 removes shape histograms:

1. The original `shapes.root` is renamed to `shapes_original.root`. If
   `shapes_original.root` already exists, it is the pre-prune archive from an
   earlier pass and is **kept**; the already-pruned `shapes.root` is discarded
   instead. Overwriting it would destroy the only unpruned copy.
2. A new `shapes.root` is written containing only the kept histograms.
3. The originals are preserved on disk for debugging and for re-runs that
   need the full set.

**Invariant relied on elsewhere:** `makeBinnedTemplates.py` `rmtree`s the output
directory before rebuilding, so `shapes_original.root` exists **iff** it is the
pre-prune snapshot of the current `shapes.root`; it can never be stale relative
to it.

**Why this matters for run-period merges.** This rewrite mutates `shapes.root`
in place, and component datacard jobs run concurrently with combined merge jobs
in the DAG. `mergeRunPeriodTemplates.py` therefore reads
`shapes_original.root` when it exists (see `source_shapes_path()`), not
`shapes.root`. Reading `shapes.root` makes the merged category inherit whatever
pruning state the component happened to be in when the merge ran, which is a
race: the merged category then cannot see the Up/Down histograms it needs to
compute its own lnN fallbacks, so `precompute_lnn_fallbacks()` returns `"-"` and
the nuisance is **dropped outright instead of degrading to lnN**. A merged
category must always start from unpruned component shapes and re-derive its own
low-stat decisions.

### S4. `lowstat.json` metadata

**File**: `python/printDatacard.py:write_lowstat_json()`

After pre-computing fallbacks, a JSON file is written to
`{template_dir}/lowstat.json`:

```json
{
  "threshold": 0.30,
  "processes": ["nonprompt", "conversion"],
  "fallbacks": {
    "nonprompt": {
      "CMS_pileup_13TeV": "1.050",
      "CMS_l1_prefiring_2016preVFP": "-"
    },
    "conversion": {
      "CMS_pileup_13TeV": "-"
    }
  }
}
```

`checkTemplates.py` consumes this file to:

1. Suppress "Missing histogram" warnings for (process, systematic) pairs that
   were intentionally removed in S3.
2. Re-introduce the lnN effect into the plotted uncertainty band via
   `error_contribution = nominal_bin_content * (lnN_value - 1.0)`, added in
   quadrature with the remaining shape-based errors.

---

## Ordering of techniques in one pipeline run

```
preprocess.py           (no low-stat treatment; fails hard on bad input)
makeBinnedTemplates.py
  P1  category_background_processes   -> process_list.json
      (subera component list; dropped_missing only for absent sample files)
  B2  adaptive binning loop (n_core 15 -> 5)
      (raw per-process test hists; no floor/cap; check_binning_quality
       requires n_eff = round(y²/σ²) >= 5 on total background per bin)
  B1  ensure_positive_integral(floor_mode) + cap_stat_errors on every final TH1
      (signal + others: floor_mode="floor"; individual bkg: floor_mode="zero")
  P3  "others" systematic filter when building merged bucket
  B3  apply_floor if loop exhausted   -> binning.json.floor_applied
      (process-dependent: floor for others, zero for individual bkg)
  B4  apply_syst_driven_merging       -> binning.json.syst_merge_*
      write shapes.root + signal_fit.json + binning.json
printDatacard.py
  P2  drop <=0 yield processes (safety net; B1 prevents this in normal runs)
  S1  precompute_lnn_fallbacks (rel_err > 30%)
  S3  rewrite_shapes_root (shapes_original.root + filtered shapes.root)
  S2  autoMCStats line (threshold=5)
  S4  write lowstat.json
checkTemplates.py
      loads lowstat.json, suppresses missing-histogram warnings,
      adds lnN error contribution to plotted bands
```

---

## Verification

```bash
# Templates: runs B1-B4 + P1
python3 python/makeBinnedTemplates.py --era 2016postVFP --channel SR3Mu \
    --masspoint MHc130_MA90 --method Baseline

# Datacard: runs P2, S1-S4
python3 python/printDatacard.py --era 2016postVFP --channel SR3Mu \
    --masspoint MHc130_MA90 --method Baseline --debug

# Artifacts
ls templates/2016postVFP/SR3Mu/MHc130_MA90/Baseline/extended/
#  -> shapes.root  shapes_original.root  datacard.txt
#     binning.json  signal_fit.json  process_list.json  lowstat.json

cat templates/.../binning.json          # floor_applied / syst_merge_applied
cat templates/.../process_list.json     # separate_processes, dropped_missing
cat templates/.../lowstat.json          # which S1 fallbacks kicked in
```

Check:

- `binning.json` — `floor_applied` only true if adaptive loop exhausted all
  n_core options; `syst_merge_applied` / `n_bins_merged` reflect B4 activity.
- `process_list.json` — `separate_processes` has the same 8 entries across all
  eras; `dropped_missing` is `[]` unless a sample file is physically absent for
  this era; `merged_to_others` is always `[]`.
- Datacard — WZ normalization remains split as
  `CMS_B2G25013_Norm_WZ_{13TeV,13p6TeV}`, SR1E2Mu conversion remains
  era-specific, and the other prompt/theory normalization names are shared
  across Run2/Run3; `CMS_B2G25013_Norm_others` (50% lnN) is present whenever
  `others` has a column.
- `shapes_original.root` exists alongside `shapes.root` iff S3 ran (at least
  one low-stat process with rel_err > 30%).
- `lowstat.json` lists the S1-flagged processes and their per-systematic lnN
  fallback values.
- `checkTemplates.py` logs "Loaded lowstat.json" and produces no spurious
  "Missing" issues for intentionally-removed (process, syst) pairs.
