# Statistical Object Review (OR) — To-Do Board

Analysis: **B2G-25-013** | Stage: Object Review (Stat/Combine review)

> **Status legend:** ✅ Done &nbsp; ⚠️ Partial / issue noted &nbsp; ❌ Not done &nbsp; N/A Not applicable

---

## B2G Required Statistical Tests

| # | Test | Status | Notes |
|---|------|--------|-------|
| 1 | **NP impacts** | ✅ | Expected (r=1) for Run2/Run3/All × 4 Baseline + 3 PN mass points; observed (partial-unblind) for 3 PN mass points × 3 era groups. Filtered + summary PDFs produced. |
| 2 | **NP pulls** | ❌ | `scripts/runPullPlots.sh` implemented (`diffNuisances.py` → text table + PDF canvas). Wired into `makeBinnedTemplates.sh --fitdiag` DAG as `plotpulls` step (parallel with `plotpostfit`, both after `fitdiag`). Run after templates ready. |
| 3 | **Goodness-of-Fit (GoF)** | ❌ | Scripts implemented (`scripts/runGoF.sh` + `automize/gof.sh`). Run after templates ready. PN: `extended_partial_unblind` (real sideband data, `--toysFrequentist`). Baseline: Asimov (`--toysFrequentist --bypassFrequentistFit`). Commands in `doThis.sh`. |
| 4 | **Signal injection (bias test)** | ✅ | Done for 21 configs (Run2/Run3/All × 4 Baseline + 3 PN mass points), 4 r-values (r=0, exp-1σ, exp-median, exp+1σ), 500 toys. `bias_test.pdf` + `pull_dist.pdf` produced. `GenerateOnly` step now uses `--toysFrequentist --bypassFrequentistFit` (fixed in `scripts/runSignalInjection.sh`). **Re-run needed** on new templates. |
| 5 | **F-test** | N/A | Template-based (histogram) analysis — no parametric background fit. |

---

## Additional Recommended Checks (B2G health checks)

| # | Check | Status | Notes |
|---|-------|--------|-------|
| 6 | **Post-fit distributions** | ❌ | Already wired into `makeBinnedTemplates.sh --fitdiag` → `plotpostfit` DAG step. Commands added to `doThis.sh`. PN: `--partial-unblind` (real sideband data). Baseline: Asimov (blinded). Run after templates ready. |
| 7 | **Impacts at r=0** | ❌ | OR doc requires impacts for both r=0 and r≈expected. `doThis.sh` now includes `--expect-signal 0` lines (added). Run after templates are ready. |
| 8 | **±1σ systematic variation plots** | ❌ | `python/checkTemplates.py` produces template diagnostic plots; verify these are reviewed for each channel/era. |
| 9 | **Asymptotic limits** | ✅ | All eras (individual + Run2/Run3/All), both methods (Baseline 35mp, ParticleNet 3mp), blinded + unblinded. Brazilian band plots in `results/plots/`. |
| 10 | **HybridNew limits** | ✅ | All/Combined for both methods. Results in `results/json/limits.All.HybridNew.*.json`. |

---

## Runbook — Commands in Execution Order

> All commands assume `cd SignalRegionStudyV2 && source setup.sh`.
> Commands are **commented out** in `doThis.sh`; uncomment and run sequentially.

### Step 0 — Preprocess ✅ (completed 2026-03-17)

```bash
./automize/preprocess.sh --mode all
```

### Step 1 — Templates + Asymptotic limits (blinded + partial-unblind)

```bash
# Baseline (blinded, Asimov data_obs)
./automize/makeBinnedTemplates.sh --mode all --method Baseline

# ParticleNet (blinded)
./automize/makeBinnedTemplates.sh --mode all --method ParticleNet

# ParticleNet (partial-unblind: real sideband score_PN < 0.3)
./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --partial-unblind
```

### Step 2 — NP Impacts [tests #1, #7]

```bash
# Baseline (r=1 expected, r=0 background-only)
./automize/impact.sh --mode all --method Baseline --expect-signal 1
./automize/impact.sh --mode all --method Baseline --expect-signal 0

# ParticleNet (r=1 expected, r=0 background-only)
./automize/impact.sh --mode all --method ParticleNet --expect-signal 1
./automize/impact.sh --mode all --method ParticleNet --expect-signal 0

# ParticleNet (partial-unblind: observed r)
./automize/impact.sh --mode all --method ParticleNet --partial-unblind
```

### Step 3 — Goodness-of-Fit [test #3]

```bash
# ParticleNet with real sideband data (preferred for OR)
./automize/gof.sh --mode all --method ParticleNet --partial-unblind

# Baseline with Asimov data
./automize/gof.sh --mode all --method Baseline

# After HTCondor jobs finish — collect and plot
./automize/gof.sh --mode all --method ParticleNet --partial-unblind --plot-only
./automize/gof.sh --mode all --method Baseline --plot-only
```

### Step 4 — FitDiagnostics + Post-fit plots + NP pulls [tests #2, #6]

```bash
# ParticleNet partial-unblind (fitdiag → plotpostfit + plotpulls)
./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --partial-unblind \
    --fitdiag --start-from combine --no-runAsymptotic

# Baseline Asimov (fitdiag → plotpostfit + plotpulls)
./automize/makeBinnedTemplates.sh --mode all --method Baseline \
    --fitdiag --start-from combine --no-runAsymptotic
```

> Output per mass point:
> - `combine_output/fitdiag/fitDiagnostics.*.root`
> - `combine_output/fitdiag/postfit_*.{root,pdf}` (plotpostfit)
> - `combine_output/fitdiag/nuisance_pulls.{txt,pdf}` (plotpulls via `diffNuisances.py`)

### Step 5 — Signal injection (bias test) [test #4]

```bash
# Run (re-run needed on new templates — fix applied: --toysFrequentist added)
./automize/signalInjection.sh --mode all --method Baseline
./automize/signalInjection.sh --mode all --method ParticleNet

# After jobs finish — plot
./automize/signalInjection.sh --mode all --method Baseline --plot-only
./automize/signalInjection.sh --mode all --method ParticleNet --plot-only
```

### Step 6 — HybridNew limits [test #10]

```bash
# Test run (subset mass points, verify grid range)
./automize/hybridnew.sh --mode all --method Baseline --test --auto-grid
./automize/hybridnew.sh --mode all --method ParticleNet --test --auto-grid

# Full run
./automize/hybridnew.sh --mode all --method Baseline --auto-grid
./automize/hybridnew.sh --mode all --method ParticleNet --auto-grid
```

### Step 7 — Full unblinding (after OR approval)

```bash
# Templates + Asymptotic
./automize/makeBinnedTemplates.sh --mode all --method Baseline --unblind
./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --unblind

# Impact plots
./automize/impact.sh --mode all --method Baseline --unblind
./automize/impact.sh --mode all --method ParticleNet --unblind

# FitDiagnostics
./automize/makeBinnedTemplates.sh --mode all --method Baseline --unblind \
    --fitdiag --start-from combine --no-runAsymptotic
./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --unblind \
    --fitdiag --start-from combine --no-runAsymptotic

# HybridNew
./automize/hybridnew.sh --mode all --method Baseline --unblind --auto-grid
./automize/hybridnew.sh --mode all --method ParticleNet --unblind --auto-grid
```

---

## Current Iteration Status

> All 8 systematics configs (`configs/systematics.*.json`) have been modified. Preprocessing completed 2026-03-17 (584/584 jobs, 0 failures). **All ANv6 results in `BackUps/ANv6/` are from the previous iteration.** The full pipeline must now be re-run:

```
makeBinnedTemplates → impact → signalInjection → [GoF (new)]
```

### Priority order for OR
> All items below are **waiting for templates** (preprocessing done 2026-03-17; ready to run `makeBinnedTemplates`).

1. ❌ **GoF test** — scripts ready; run `automize/gof.sh` after templates
2. ❌ **Post-fit plots** — run `makeBinnedTemplates.sh --fitdiag` after templates
3. ❌ **Impacts at r=0** — run `automize/impact.sh --expect-signal 0` after templates
4. ⚠️ **Signal injection re-run** — `--toysFrequentist` fix applied; re-run on new templates
5. ❌ **NP pull plots** — `runPullPlots.sh` ready; runs automatically with `--fitdiag`

---

## Key Script Reference

| Task | Script |
|------|--------|
| Asymptotic limits | `scripts/runAsymptotic.sh` / `automize/makeBinnedTemplates.sh` |
| HybridNew limits | `scripts/runHybridNew.sh` / `automize/hybridnew.sh` |
| FitDiagnostics | `scripts/runFitDiagnostics.sh` |
| Post-fit plots | `python/plotPostfit.py` (run after FitDiagnostics) |
| Impact plots | `scripts/runImpacts.sh` / `automize/impact.sh` |
| Signal injection | `scripts/runSignalInjection.sh` / `automize/signalInjection.sh` |
| GoF test | `scripts/runGoF.sh` / `automize/gof.sh` |
| Pull plots | `scripts/runPullPlots.sh` (via `makeBinnedTemplates.sh --fitdiag` DAG) |
