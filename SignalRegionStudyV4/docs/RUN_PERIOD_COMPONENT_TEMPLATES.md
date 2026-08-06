# Run-Period Component Templates

SignalRegionStudyV4 uses Run-period component templates as its only Combine
model, inherited unchanged from the V3 construction (the reproduction test in
`docs/REPRODUCTION.md` pins it bitwise).

## Model Layout

For `All/Combined`, the fit categories are:

- `SR1E2Mu_Run2`
- `SR3Mu_Run2`
- `SR1E2Mu_Run3`
- `SR3Mu_Run3`

Each category contains subera component processes:

```text
signal_2016preVFP
signal_2016postVFP
signal_2017
signal_2018
nonprompt_2016preVFP
WZ_2017
ttZ_2018
others_2018
...
```

The Poisson terms are merged at Run-period category level, while the component
processes preserve subera-specific nuisance treatment.

## Binning

Each merged category has its own signal fit and its own adaptive binning.

For `All/Combined`, this means four independent DCB fits:

```text
SR1E2Mu_Run2  -> summed/appended Run2 SR1E2Mu signal
SR3Mu_Run2    -> summed/appended Run2 SR3Mu signal
SR1E2Mu_Run3  -> summed/appended Run3 SR1E2Mu signal
SR3Mu_Run3    -> summed/appended Run3 SR3Mu signal
```

For each category:

1. Fit the combined category signal distribution with the DCB model.
2. Define candidate edges from the fitted `x0` and `sigma_eff`:

   ```text
   mass_min = max(x0 - 10*sigma_eff, 12)
   mass_max = x0 + 10*sigma_eff
   edges = x0 + [-10, -5 ... +5, +10] * sigma_eff
   ```

3. Scan `n_core = 15, 14, ..., 5`.
4. For each candidate, sum the expected background over all subera component
   processes in the category.
5. Accept the first candidate where every bin has
   `round(y^2 / sigma^2) >= 5`.
6. If 5 core bins still fails, keep 5 core bins plus two sideband bins and use
   the floor/zero low-stat handling described in `docs/LOWSTAT.md`.

All component processes inside one category use exactly the same final binning.

## Files

Template generation:

```bash
python3 python/makeBinnedTemplates.py --era All --channel Combined \
  --masspoint MHc130_MA90 --method Baseline --unblind
```

The output directory:

```text
templates/{Run2,Run3,All}/{SR1E2Mu,SR3Mu,Combined}/{masspoint}/{method}/{extended,extended_unblind}/
```

Important outputs:

- `shapes.root`: one ROOT directory per merged category.
- `datacard.txt`: multi-category component-process datacard.
- `categories.json`: category, subera, and component metadata.
- `process_list.json`: component process list and public physics-group mapping.
- `binning.json`: per-category fit results and final bin edges.
- `lowstat.json`: low-stat shape-to-`lnN` fallback metadata when applicable.

## Nuisances

Subera-specific nuisances apply only to matching component processes. For
example, a 2017 nuisance affects `*_2017` columns and not `*_2018` columns.

Globally correlated nuisances remain correlated by exact nuisance name across
all matching component processes.

All signal components use non-positive process IDs in the datacard so the
common POI `r` scales every `signal_<subera>` process.

Each merged category receives its own `autoMCStats 5` line. Combine evaluates
the threshold on the merged category/bin total; this is checked from
`text2workspace.py` logs and workspace nuisance names during validation.

## Validation

Run:

```bash
python3 python/validateRunPeriodTemplates.py --era All --channel Combined \
  --masspoint MHc130_MA90 --method Baseline --unblind
```

Required checks:

- `data_obs` closure against summed expected background in blinded mode.
- One common binning per category across all components and variations.
- Active `shape? = 1` entries have both Up and Down histograms.
- Component-to-physics-group metadata exists for public plotting.
- `text2workspace.py datacard.txt -o workspace.root` succeeds without missing
  shape or suspicious non-positive-yield warnings.
