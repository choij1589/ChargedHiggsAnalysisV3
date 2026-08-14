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
# (V4 defaults: extended binning, unblind)
./automize/makeBinnedTemplates.sh --mode all --method Baseline    \
    --fitdiag --pull-fit both --masspoint MHc130_MA90
./automize/makeBinnedTemplates.sh --mode all --method ParticleNet \
    --fitdiag --pull-fit both --masspoint MHc130_MA90

# Step 4: compare templates and limits (light; login node is fine)
python3 python/compareToV3.py --masspoint MHc130_MA90 --v3-dir <V3 dir> --stage templates
python3 python/compareToV3.py --masspoint MHc130_MA90 --v3-dir <V3 dir> --stage limits

# Step 5: side-effect-free single-point collect (does not touch grid JSONs)
python3 python/collectLimits.py --era All --method Baseline \
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
| AsymptoticLimits (6 entries, BR-converted) vs the ROOT outputs inside V3's frozen template dirs | value compare | rtol 1e-6; warn above 1e-10 |
| V3's own template ROOT vs V3 `results/json/BR/...` | reference self-consistency | WARN only (see below) |
| `validation/summary.json` | existence, nonzero size | — |

`workspace.root` is deliberately not checked: V4 produces it only in the
GoF/impacts workflows on the interp-signal seeds, whose datacards have no
V3 counterpart.

The samples metadata check opens every tree of every file — this is the
guard against silently lost or truncated pnfs transfers.

## Verdict For MHc130_MA90 (2026-08-06)

- samples: 512 PASS / 0 FAIL (with the `pT` exception below)
- templates: 212 PASS / 0 FAIL — all 18 datacards bitwise-identical;
  Baseline metadata exact to the last digit; ParticleNet Run3 fit metadata
  within 4e-13 relative (cross-worker noise, below)
- limits: 18 PASS / 0 FAIL — every limit value exactly identical
  (max relative deviation 0.0) to V3's frozen template outputs
- 10 WARN: V3-internal `results/json` staleness (below)

## Known, Accepted Differences

### The `pT` branch

V3's frozen pnfs samples were produced (2026-07-06) before the `pT` branch
was added to V3's own `preprocess.py` for the PTOptimized study. V4 ports
the current code, so every V4 tree carries an extra `pT` branch that the V3
reference lacks. The branch is unused by the Baseline and ParticleNet
template paths, entry counts and all common-branch contents match exactly,
and the comparator whitelists it via `--allow-extra-branches pT` (the
default). Any other branch-set difference still fails the check.

### Cross-worker fit noise (ParticleNet Run3 categories)

V4's Baseline templates reproduced V3 bitwise even on different worker
hosts, but the ParticleNet Run3 categories (low-stat, score-cut fits near
the numerical noise floor) show Minuit/numpy noise of at most 4e-13
relative in fit-derived metadata (bin edges, threshold sensitivities).
Datacards are bitwise-identical and limit values exactly match regardless.
The comparator therefore uses rtol 1e-9 for numeric JSON leaves and bin
edges (`--json-rtol`, `--edge-rtol`) — still far below any real change,
since binning revisions move edges at the percent level.

### V3 `results/json` staleness (WARN, not a V4 failure)

V3's checked-in `results/json` (frozen 2026-07-31) predates V3's final
MHc130_MA90 template rebuild (2026-08-03/04, which included the
merge-race fix from commit `da959af4fb`). For every merged target
(Run2/Run3/All Combined, All SR1E2Mu/SR3Mu) the JSON `obs` values disagree
with V3's own template ROOT outputs by up to ~0.4%; the unmerged targets
agree. V4 exactly reproduces the template ROOT outputs — the current V3
chain — so the comparator's primary limit check uses those, and reports
the JSON disagreement as WARN (`V3 json stale`). Re-collecting V3's
limits from its rebuilt templates would resolve the inconsistency; that
is a V3-side decision.

## Extended Verification (2026-08-07)

Five more points, all on the shared-sample layout, vs V3 frozen outputs:

| Point | Method | Samples | Templates | Limits |
|---|---|---|---|---|
| MHc70_MA15 (lowM) | Baseline | 352/0 | 81/0 (datacards bitwise) | 9/0 |
| MHc100_MA60 | Baseline | 352/0 | 81/0 | 9/0 |
| MHc160_MA155 | Baseline | 352/0 | 81/0 | 9/0 |
| MHc100_MA95 | ParticleNet | 512/0 | 131/0 | 9/0 |
| MHc160_MA85 | ParticleNet | see below | see below | see below |

### MHc160_MA85: stale V3 reference (upstream skim regeneration)

The MHc160 NoHistMode input skims in SKNanoOutput were regenerated after
V3's samples were frozen (2026-07-06): ParticleNet scores differ at ~1e-8
relative (re-inference wobble) and 2022 `others` gained 3 events. Proof
that V4 is faithful: V4's sample content equals **today's** input skim
exactly, while V3's frozen sample carries the old values.

Downstream impact: datacards STILL bitwise identical; a few Run3 shape
bins shift at the ~1% level (the 3 events), threshold sensitivities at
~1e-5, and only the observed limits move (<= 1e-5 relative; all expected
quantiles within tolerance). Not a chain defect — the V3 reference is
stale for this mass point. Any future full-grid production supersedes the
V3 numbers here by construction.

### DCB tail-parameter noise

Cross-worker Minuit noise is <= 4e-13 on x0/sigma_eff but up to ~1.5e-9 on
the DCB tail parameters (alphaL/nL), which do not feed the binning. The
comparator's `--json-rtol` default is 1e-8 (the measured envelope, still
six orders below any real change).

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
