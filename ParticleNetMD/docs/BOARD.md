# Project Status Board — ParticleNetMD

_Last updated: 2026-02-26_

---

## Active Jobs

- **Lambda sweep pilot** (MHc130_MA90, Combined, cuda:0) — re-running with new directory structure
- **GA pilot** (MHc130_MA90, Combined, cuda:1) — re-running with new directory structure
- **Full lambda sweep** (`sbatch submit_sweep.slurm`) — 3 signals × 8 λ, submitted
- **Full GAOptim** (`sbatch submit.slurm`) — 3 signals × Combined, submitted

---

## Task Board

### In Progress

- Lambda sweep pilot re-run → `LambdaSweep/Combined/MHc130_MA90/pilot/`
- GA pilot re-run → `GAOptim/Combined/MHc130_MA90/pilot/`
- Full lambda sweep (SLURM) → `LambdaSweep/Combined/{signal}/fold-4/`
- Full GAOptim (SLURM) → `GAOptim/Combined/{signal}/fold-4/`

### Up Next

- Post-GA processing (full run): `evaluateGAModels.py`, `visualizeGAIteration.py`, `summarizeGALoss.py` for all 3 signals
- Lambda sweep comparison (full run): `compareDecorrelation.py` + `visualizeMultiClass.py` for all 3 signals

### Done

- [done] Dataset creation (all 8 eras, Run1E2Mu + Run3Mu + Combined)
- [done] Data augmentation validation (diboson rank-promote, nonprompt LNT)
- [done] `python/` → `python/lib/` refactor (entry scripts vs library modules)
- [done] `evaluateGAModels.py` path fix (pilot/ prefix, removed /multiclass/)
- [done] `visualizeGAIteration.py` path fix (same)
- [done] `submit.slurm` defaults set (3 signals × Combined, cuda:0,1,2)
- [done] `GAConfig.py` / `SglConfig.py` path fix (`../configs` → `../../configs` after lib/ refactor)
- [done] GAOptim pilot — MHc130_MA90, Combined (4/4 iters, completed 2026-02-25 23:50)
- [done] GAOptim pilot evaluation — `evaluateGAModels.py` iters 0–3 (all 16 models flagged overfitted in pilot mode; expected)
- [done] Lambda sweep training — 8 λ × 3 signals, Combined, fold-4, full dataset (completed 2026-02-26 00:23)
- [done] `compareDecorrelation.py` fix — OOM from O(n²) DisCo on 1M events; fixed with 5000-event subsample; `load_tree_to_arrays` replaced with `RDataFrame.AsNumpy()`
- [done] Lambda sweep comparison plots — `compareDecorrelation.py` run for all 3 signals; `disco_lambda=0.1` confirmed as production value
- [done] Per-lambda visualization — `visualizeMultiClass.py` run for all 24 (3 signals × 8 λ) fold-4 combinations; 11 PNGs each
- [done] **Output directory refactor** — consolidated `results/` + `plots/` into self-contained `DataAugment/`, `LambdaSweep/`, `GAOptim/`; short signal names in output paths; removed `multiclass/` from paths
- [done] **Dataset validation re-run** (Step 1) — 60 jobs, histograms in `DataAugment/validation/histograms/`, 38 class overlay PNGs

---

## Workflow Reference

See [`WORKFLOW.md`](WORKFLOW.md) for full details.

```
saveDatasets.sh → launchLambdaSweep.sh → compareDecorrelation.py
               → launchGAOptim.sh (submit.slurm) → evaluateGAModels.py → visualizeGAIteration.py
```
