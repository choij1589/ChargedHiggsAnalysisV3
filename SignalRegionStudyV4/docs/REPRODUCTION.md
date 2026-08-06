# REPRODUCTION.md — Exact-Reproduction Test Against V3

V4's acceptance criterion is that its chain reproduces the frozen V3 results
exactly for one mass point, `MHc130_MA90` (present in both the `baseline` and
`particlenet` sets), for both methods. V3 is never re-run: the references are
V3's checked-in `results/json`, its `templates/` tree, and its preprocessed
samples on pnfs.

`python/compareToV3.py` is the comparator and the only V4 file allowed to
read V3 artifacts. `--v3-dir` is required and has no default; it must point
at the V3 module directory. The comparator is not part of any production
workflow and can be retired once the port is blessed.

## Procedure

```bash
source setup.sh

# Step 1: preprocess the test point (24 condor jobs:
#         8 eras x {SR1E2Mu, SR3Mu, TTZ2E1Mu})
./automize/preprocess.sh --mode all --masspoint MHc130_MA90

# Step 2: compare samples (condor job; heavy I/O stays off the login node)
#         submit scripts/compare_wrapper.sh with:
#         --masspoint MHc130_MA90 --v3-dir <V3 dir> --stage samples

# Step 3: templates -> datacards -> validation -> limits -> fitdiag, both methods
./automize/makeBinnedTemplates.sh --mode all --method Baseline    \
    --binning extended --unblind --fitdiag --pull-fit both --masspoint MHc130_MA90
./automize/makeBinnedTemplates.sh --mode all --method ParticleNet \
    --binning extended --unblind --fitdiag --pull-fit both --masspoint MHc130_MA90

# Step 4: compare templates and limits (light; login node is fine)
python3 python/compareToV3.py --masspoint MHc130_MA90 --v3-dir <V3 dir> --stage templates
python3 python/compareToV3.py --masspoint MHc130_MA90 --v3-dir <V3 dir> --stage limits

# Step 5: side-effect-free single-point collect (does not touch grid JSONs)
python3 python/collectLimits.py --era All --method Baseline --unblind \
    --masspoint MHc130_MA90 --output results/repro/limits.single.All.Baseline.json
```

Reports land in `results/repro/<masspoint>.<stage>.json`; exit code is
nonzero on any failure.

## What Is Compared, And How Tightly

| Artifact | Comparison | Tolerance |
|---|---|---|
| `samples/{era}/{ch}/{mp}/*.root` | every tree: entry count + branch set (metadata); `Central` tree: per-branch sum and sum-of-squares | counts exact; sums rtol 1e-9 |
| `datacard.txt` (18 dirs: {Run2,Run3,All} x {SR1E2Mu,SR3Mu,Combined} x 2 methods) | bitwise | zero diff |
| `binning.json`, `categories.json`, `process_list.json`, `lowstat.json`, `background_validation.json`, `threshold*.json`, `background_weights*.json` | parsed-JSON deep equality | exact |
| `shapes.root`, `shapes_original.root` | bin edges exact; per-bin content and error | rtol 1e-12 |
| AsymptoticLimits (6 entries, BR-converted) vs V3 `results/json/BR/...` | value compare | rtol 1e-6; warn above 1e-10 |
| `workspace.root`, `validation/summary.json` | existence, nonzero size | — |

The samples metadata check opens every tree of every file — this is the
guard against silently lost or truncated pnfs transfers.

### Known, accepted difference: the `pT` branch

V3's frozen pnfs samples were produced (2026-07-06) before the `pT` branch
was added to V3's own `preprocess.py` for the PTOptimized study. V4 ports
the current code, so every V4 tree carries an extra `pT` branch that the V3
reference lacks. The branch is unused by the Baseline and ParticleNet
template paths, entry counts and all common-branch contents match exactly,
and the comparator whitelists it via `--allow-extra-branches pT` (the
default). Any other branch-set difference still fails the check.

## Drift Triage

Order matters: samples must PASS before template mismatches are meaningful.
The adaptive binning is discretely sensitive — a 1-ulp drift in the DCB fit
can flip a bin edge and produce a structurally different datacard, so
`binning.json` is the first artifact to inspect on any template mismatch.

1. Confirm environment: the comparator prints the ROOT version; everything
   must run under `Common/CMSSW_14_1_0_pre4` (ROOT 6.30). Newer ROOT
   versions are known to segfault template making.
2. Check the executing worker host in `condor/jobs_dag_*/*/logs/*.out`;
   resubmit the failing node once before debugging code.
3. Bisect the offending V4 file against its V3 original (Stage-A files are
   byte-identical modulo the module-name rename; Stage-B diffs are
   deletions-only).
