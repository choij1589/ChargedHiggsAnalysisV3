# Low-Statistics Background Treatment

Low-stat backgrounds can produce non-positive bins, >100% per-bin stat error,
and shape variations dominated by MC noise. This doc catalogs every mitigation
in the pipeline, grouped by the level at which it acts. For the exact order
they execute, see the "Ordering" block near the bottom.

## Thresholds

| Constant | Value | File | Meaning |
|---|---|---|---|
| `min_total_events` | `1`    | `makeBinnedTemplates.py` | minimum raw bkg yield to keep a process separate |
| `BIN_FLOOR_VALUE`  | `1e-6` | `makeBinnedTemplates.py` | content/error floor when adaptive binning is exhausted |
| `SYST_MERGE_THRESHOLD` | `2.0` | `makeBinnedTemplates.py` | merge bin if total-bkg `sigma_syst / nominal > 2.0` |
| `SHAPE_REL_ERR_THRESHOLD` | `0.30` | `printDatacard.py` | switch shape -> `shape?` + lnN above 30% relative stat error |
| `MAX_LNN_VALUE` | `2.0` | `printDatacard.py` | cap on lnN fallback (100% uncertainty) |
| `MIN_YIELD_THRESHOLD` | `1e-6` | `printDatacard.py` | yield below which per-syst fallback is skipped (`-`) |
| `autoMCStats` threshold | `5` | `printDatacard.py` | Barlow-Beeston-lite per-bin stat nuisances |

---

## Process level

### P1. Process merging into `others`

**File**: `python/makeBinnedTemplates.py:validateBackgroundStatistics()` and
`determineProcessList()`

Each background is validated on the `Central` tree under the mass window
(and the PN score cut for ParticleNet). The raw weighted integral is compared
against `min_total_events = 1`:

- `total_events >= 1` -> **keep separate** (own histogram + own datacard column)
- `total_events <  1` -> **merge into `others`** (via `getHistMerged`)

Only `nonprompt` is hard-wired as always-separate, because its transfer-factor
uncertainty attaches to its own column. `conversion` is **not** on that list:
ConvSF is baked in as a per-sample K-factor at preprocess time
(`preprocess.py:process_tree`), so merging is numerically safe, and the
`CMS_B2G25013_Norm_conversion_*` lnN is automatically skipped by the `group`
filter in `printDatacard.py` when there is no conversion column.

The final separate vs merged list is persisted to `process_list.json`.

### P2. Dropping zero/negative-yield processes from the datacard

**File**: `python/printDatacard.py:__init__()`

After templates exist, any process whose final nominal yield is `<= 0` is
dropped from the datacard entirely with a warning.

### P3. Per-systematic filter for the merged `others` template

**File**: `python/makeBinnedTemplates.py` (others-syst loops, lines ~1166-1197)

When the merged `others` template is being built, a systematic is only
applied if `"others"` is explicitly in its `group`. All instrumental systs
(pileup, btag, JES, lepton ID/res/scale, MET) list `"others"` in their group,
so they pass through. Per-process normalization systs do **not**:
`CMS_B2G25013_Norm_WZ_13p6TeV` has `group=["WZ"]` and would fail to build on
the merged template anyway (the syst tree only exists in `WZ.root`).

Those per-process norms are then also filtered out in the datacard by
`printDatacard.py:generate_systematic_lines` (same `group` semantics: no
relevant process -> line dropped). The merged bucket's normalization
uncertainty is covered by a single flat lnN per era,
`CMS_B2G25013_Norm_others_{13TeV,13p6TeV}` (50%, `group=["others"]`), which
is conservative and consistent across Run2/Run3.

---

## Bin level

### B1. Negative bin clamp (`ensure_positive_integral`)

**File**: `python/template_utils.py:ensure_positive_integral()`

Called on every histogram immediately after it is filled (nominal, Up, Down,
signal, backgrounds, merged others, data_obs for MC-Asimov). For each bin:

- `content < 0` -> `content = 0`, `error = 0`, warn with histogram name and
  bin index
- after the bin sweep, if `Integral() <= 0` -> seed the middle bin with
  `content = error = 1e-10` so Combine can still normalize the template

This is the floor of last resort to keep Combine from crashing on non-positive
templates; it does **not** attempt to preserve shape.

### B2. Adaptive binning loop

**File**: `python/makeBinnedTemplates.py` (lines ~921-967)

The Extended binning scheme is nominally `15 core + 2 sideband = 17 bins`
(see `calculate_adaptive_bins` in `template_utils.py`). For a low-stat era or
channel, this is often too fine — empty total-bkg bins or per-process bins
with `stat_err / content > 1.0` break Combine fits.

Loop: for `n_core in [15, 13, 11, 9, 7, 5]`:
1. Build candidate edges with `calculate_adaptive_bins(x0, sigma_eff, n_core)`.
2. Fill `Central` histograms for every separate process and the `others`
   merged template.
3. Call `check_binning_quality(background_hists)` which reports:
   - any bin with `total_bkg <= 0`
   - any process with a bin where `content < 0` or `err / content > 1.0`
4. If clean: `break` and adopt this `n_core`.
5. Otherwise log up to five diagnostic messages and try the next smaller
   `n_core`.

If the `for` loop exhausts without a `break` (i.e. even 5 core bins fail), the
5-bin candidate is kept and a fallback is flagged (see B3).

### B3. Bin floor when adaptive binning is exhausted (`apply_floor`)

**File**: `python/makeBinnedTemplates.py` (lines ~1206-1226)

When the adaptive loop gave up, `apply_floor = True` triggers a per-bin
sweep over **every histogram** in each separate process and in `others` —
both the nominal and every `{syst}Up/Down` variation — so syst-band bins
cannot reintroduce zeros after the nominal is floored. Per bin:

- `content <= 0`: set `content = BIN_FLOOR_VALUE = 1e-6` and `error = 1e-6`
  (effectively 100% relative error)
- `err / content > 1.0`: cap `error = content` (exactly 100% error)

Last-resort patch: does not change the number of bins, but guarantees every
bin is positive with a well-defined error across nominal and syst templates.
Recorded in `binning.json` as `floor_applied: true`.

### B4. Post-binning systematic-driven bin merging

**File**: `python/makeBinnedTemplates.py` (lines ~1228-1246) calling
`python/template_utils.py:apply_syst_driven_merging()`

Motivation: even after adaptive binning succeeds and templates are filled with
all systematic variations, individual bins can still have a *systematic*
envelope that exceeds the nominal — for example a sudden-dip bin where
`JES_Up` goes to zero while `JES_Down` doubles. The B2G stat review flagged
this as an under-coverage concern: a bin with `sigma_syst >> nominal` is not
meaningfully constrained and can pull the fit.

Algorithm (iterative, operates on numpy content/variance snapshots of every
template to avoid repeated TH1 rebinning):

1. Build per-bin total-background `nominal`, `sigma_syst` (envelope from
   `max(|Up-nom|, |Down-nom|)` summed in quadrature over all collected
   systematic names), and `stat_err` arrays via
   `_total_bkg_syst_from_snapshot`.
2. Compute `rel = sigma_syst / nominal`; find `worst = argmax(rel)`.
3. If `rel[worst] <= SYST_MERGE_THRESHOLD = 2.0`, converged - break.
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
full per-process stat nuisances. Lowering this value would create more
nuisances and slow down fits; 5 is the pragmatic default.

### S3. Non-destructive `shapes.root` rewrite

**File**: `python/printDatacard.py:rewrite_shapes_root()`

When S1 removes shape histograms:

1. The original `shapes.root` is renamed to `shapes_original.root` (if a
   previous rewrite already happened, the stale original is removed first).
2. A new `shapes.root` is written containing only the kept histograms.
3. The originals are preserved on disk for debugging and for re-runs that
   need the full set.

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
  P1  determineProcessList            -> process_list.json
  B2  adaptive binning loop (n_core 15 -> 5)
  B1  ensure_positive_integral on every filled TH1
  P3  "others" systematic filter when building merged bucket
  B3  apply_floor if loop exhausted   -> binning.json.floor_applied
      (patches nominal + every syst variation)
  B4  apply_syst_driven_merging       -> binning.json.syst_merge_*
      write shapes.root + signal_fit.json + binning.json
printDatacard.py
  P2  drop <=0 yield processes
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
    --masspoint MHc130_MA90 --method Baseline --binning extended

# Datacard: runs P2, S1-S4
python3 python/printDatacard.py --era 2016postVFP --channel SR3Mu \
    --masspoint MHc130_MA90 --method Baseline --binning extended --debug

# Artifacts
ls templates/2016postVFP/SR3Mu/MHc130_MA90/Baseline/extended/
#  -> shapes.root  shapes_original.root  datacard.txt
#     binning.json  signal_fit.json  process_list.json  lowstat.json

cat templates/.../binning.json          # floor_applied / syst_merge_applied
cat templates/.../process_list.json     # separate vs merged
cat templates/.../lowstat.json          # which fallbacks kicked in

# Validation (runs the checkTemplates integration for lowstat.json)
python3 python/checkTemplates.py --era 2016postVFP --channel SR3Mu \
    --masspoint MHc130_MA90 --method Baseline --binning extended
```

Check:

- `binning.json` — `floor_applied` only true if adaptive loop exhausted;
  `syst_merge_applied` / `n_bins_merged` reflect B4 activity.
- `process_list.json` — `nonprompt` always in `separate_processes`;
  `conversion` is there when its yield `>= 1`, otherwise in `merged_to_others`.
- Datacard — no `CMS_B2G25013_Norm_{WZ,ZZ,ttW,...,conversion}_*` line when
  the corresponding process is absent (merged into `others`);
  `CMS_B2G25013_Norm_others_*` (50% lnN) is present whenever `others` has
  a column.
- `shapes_original.root` exists alongside `shapes.root` iff S3 ran (at least
  one low-stat process).
- `lowstat.json` lists low-stat processes and per-systematic fallback values.
- `checkTemplates.py` logs "Loaded lowstat.json" and produces no spurious
  "Missing" issues for low-stat pairs; error bands in the stack plot include
  the lnN contribution.
