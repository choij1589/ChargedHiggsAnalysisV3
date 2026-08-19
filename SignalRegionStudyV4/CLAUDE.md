# CLAUDE.md — SignalRegionStudyV4

Simplified, self-contained signal-region limit chain for the charged Higgs
search (B2G-25-013), and the base for the next-generation work (parametric
signal models, mA interpolation). This file is the single module guide;
`docs/*` hold deep dives.

## Project Scope

- Core chain only: preprocess → binned templates → run-period merge →
  datacard → validation → asymptotic limits → FitDiagnostics/postfit/pulls →
  collect/plot limits. See `docs/FUNCTIONALITY_SCOPE.md` for the explicit
  active/not-ported lists.
- Methods: `Baseline`, `ParticleNet`. Channels: `SR1E2Mu`, `SR3Mu`,
  `Combined`. Run-period targets: `Run2`, `Run3`, `All`.
- Mass points from `configs/masspoints.json` (`baseline` 78, `particlenet`
  22, `limits` curated plotting subset).
- **Self-containment rule**: zero code references, imports, or symlinks into
  any other SignalRegionStudy directory. The only V3 touchpoint is the
  validation-only comparator `python/compareToV3.py` (explicit `--v3-dir`,
  reads frozen V3 *outputs* only).

## Environment Rules

```bash
source setup.sh   # module-local; REQUIRED before any work (not the repo root setup.sh)
```

- CMSSW_14_1_0_pre4 + Combine v10 (ROOT 6.30) only. Newer ROOT (6.36)
  segfaults template making.
- Use `python3`, never `python`.
- Never run heavy analysis/ROOT work on the login node — submit to condor.
  The `automize/` drivers and `scripts/compare_wrapper.sh` exist for this.
- A valid grid proxy (`voms-proxy-init --voms=cms`) is needed for
  preprocessing (xrdcp output to pnfs).

## Path & Layout Contract

```
samples/{era}/SR1E2Mu/                       shared bkg/nonprompt/data + ALL signals ({masspoint}.root)
samples/{era}/SR3Mu_{lowM,highM}/            shared bkg/nonprompt/data + ALL signals in BOTH
                                             pairing variants (interpolation needs both)
samples/{era}/{channel}/{masspoint}/         ParticleNet per-masspoint dirs (scores +
                                             NoHistMode skims; incl. TTZ2E1Mu)
templates/{masspoint}/{method}/{era}/{channel}/
results/json/{BR,xsec}/{era}/limits.{era}[.{channel}].Asymptotic.{method}.json
results/templates/{method}/{masspoint}/      promoted diagnostics of the curated template
                                             points: gof/ impacts/ pulls/ mass/ scores/
fits/MHc{X}/                                 mA-interpolation fit artifacts (dcb_fits, polynomials,
                                             yield model, shape_deltas, validation plots); global
                                             surface panels in fits/{params,yield}/
closure/interpolation/{MHc{X},loo/}          interpolation closures, LOO sweep, pooled
                                             uncertainty diagnostics — see docs/interpolation/
```

The SR3Mu pairing rule: `highM` (higher-mass dimuon pairing) iff
`mHc >= 100 && mA >= 60` — a 2D condition, not a pure mA threshold
(MHc160_MA15 is lowM, MHc70_MA60 is lowM). `srspaths.pairing_variant()`
is the only place it is defined for path resolution;
`preprocess.pairing_for()` mirrors it for production.

- Unblind (real data) is the default everywhere. `--blind` gives Asimov
  data_obs and writes to a `{method}_blind` method segment.
- Path construction lives in exactly two places: `python/srspaths.py`
  (python) and `scripts/env.sh` (shell). Never construct these paths
  anywhere else.
- Site constants (pnfs/xrootd bases, CMSSW release) are defined once in
  `scripts/env.sh`.

## Procedures (in chain order)

### 1. Preprocess

Skims SKNanoOutput into flat per-process trees (one TTree per systematic;
branches `mass`, `mass1`, `mass2`, `pT`, `weight`, and `score_{mp}_*` in
ParticleNet per-masspoint dirs). Vectorized: RDataFrame Define+Snapshot.

```bash
./automize/preprocess.sh [--masspoint MHc130_MA90] [--skip-backgrounds] \
                         [--backgrounds-only] [--dry-run]
```

- Shared-background DAG (24 nodes: 8 eras × {SR1E2Mu, SR3Mu:lowM,
  SR3Mu:highM}) — run ONCE for the whole analysis; backgrounds, nonprompt
  and data are mass-independent.
- Per-masspoint DAG: 16 shared-signal nodes (SR3Mu writes both pairing
  variants), plus 24 full ParticleNet nodes (incl. TTZ2E1Mu) for
  ParticleNet-trained points.
- An interpolated mass point needs only its signal: the shared backgrounds
  are already in place.

Outputs land on pnfs via xrdcp. Owning code: `python/preprocess.py`
(modes: default per-masspoint ParticleNet / `--shared-backgrounds` /
`--shared-signal`), `scripts/preprocess_wrapper.sh`.
Deep dive (branches, weights, pairing rule, layout): `docs/SAMPLES.md`.

### 2. Binned templates

DCB signal fit per merged run-period category, adaptive binning (15→5 core
bins), all systematic variations, low-stat handling (`docs/LOWSTAT.md`).

```bash
python3 python/makeBinnedTemplates.py --era Run2 --channel SR1E2Mu \
    --masspoint MHc130_MA90 --method Baseline
```

Condor: part of the `makeBinnedTemplates.sh` DAG (template step, 6 GB).
Outputs: `shapes.root`, `binning.json`, `categories.json`,
`process_list.json`, `background_validation.json`, signal-fit PNGs
(+ `threshold*.json`, `background_weights*.json` for ParticleNet).

### 3. Run-period merge

File-level union of per-channel/per-period template dirs into Combined/All
targets. Reads `shapes_original.root` from components when present (merge
race guard — see `docs/LOWSTAT.md` S3).

```bash
python3 python/mergeRunPeriodTemplates.py --era All --channel Combined \
    --masspoint MHc130_MA90 --method Baseline --sources Run2:SR1E2Mu,Run2:SR3Mu,Run3:SR1E2Mu,Run3:SR3Mu
```

### 4. Datacard

Multi-category component-process datacard; prunes low-stat shapes to lnN
(`shape?`), writes `lowstat.json`, `shapes_original.root` archive.

```bash
python3 python/printDatacard.py --era All --channel Combined \
    --masspoint MHc130_MA90 --method Baseline
```

### 5. Validation

Closure, binning consistency, shape? Up/Down presence, text2workspace
smoke, diagnostic plots into `validation/`.

```bash
python3 python/validateRunPeriodTemplates.py --era All --channel Combined \
    --masspoint MHc130_MA90 --method Baseline
```

### 6. Asymptotic limits

```bash
./scripts/runAsymptotic.sh --era All --channel Combined \
    --masspoint MHc130_MA90 --method Baseline
```

Frozen Combine command: `combine -M AsymptoticLimits datacard.txt
-n .{mp}.{method} -m 120 --rAbsAcc 0.0001 --rRelAcc 0.01`.
Output: `combine_output/asymptotic/higgsCombine.{mp}.{method}.AsymptoticLimits.mH120.root`.

**Observed significance** is a follow-up on the same datacards, not a scan
step — the fit is cheap but the point list is a judgement call, so it is
always explicit:

```bash
./automize/significance.sh --template-points --point Baseline:MHc160_MA17p5
python3 python/collectSignificance.py --template-points --point Baseline:MHc160_MA17p5
```

Frozen Combine command: `combine -M Significance datacard.txt
-n .{mp}.{method} -m 120 --uncapped 1 --rMin -5`. **Uncapped** is the
point: the capped default floors a downward fluctuation at Z = 0, which
would report every deficit as the same "0"; uncapped, a deficit keeps its
sign and the two agree wherever Z > 0. One node per (point, channel) at
All × 3 channels; `collectSignificance.py` merges into
`results/json/significance.{era}.{source}.json`, so a later run on
different points adds to it rather than replacing it. Z and its one-sided
p are **local**; the trials correction is a separate step (§6b). The
measured excesses and deficits, the ranking method behind the point list,
and the GoF cross-check are recorded in `docs/SIGNIFICANCE.md`.

### 6b. Look-elsewhere effect

The trials correction **is** a scan step — it counts how often the
observed Z(mA) curve crosses a level, so it needs Z everywhere, not at
the quoted points:

```bash
./automize/significance.sh --grid          # 2467 x 3 nodes, ~33 s each
./automize/significance.sh --pnet-grid     #  150 x 3
python3 python/collectSignificance.py --grid --pnet-grid
python3 python/estimateLEE.py              # -> results/json/lee.{era}.{source}.json
python3 python/plotLEE.py --all            # -> results/plots/lee/
```

**Asymptotic (Gross-Vitells), not V3's toys**, and that follows from the
grid: V3 tested 35 points spaced far above the resolution, so each was an
independent test and a toy campaign over the tested set was the right
object. V4's lattice sits AT the resolution (step/σ_eff = 0.86–0.89), so
the point count is not a trials count — a finer lattice would inflate it
without adding one independent test — and the same campaign would cost
~68,000 CPU-h. What is invariant is the upcrossing rate:
`<N_u(u)> = N_0 e^(−u/2)`, `p_global ≈ p_local + <N_u(Z²)>`, with `N_0`
measured from the observed scan over a ladder of low thresholds.

**Frozen scope**: the mA scan ALONE, per (arm, channel, mHc) column. The
extra multiplicity from also scanning 7 mHc columns and 3 channels is
bounded in `docs/LEE.md` from the measured curve correlations, not folded
in. `--statistic bandpull` reruns everything off the limit JSONs with no
fits at all, as a cross-check. Full record: `docs/LEE.md`.

### 7. FitDiagnostics, postfit mass plots, pulls

Run for `All/Combined` in `--mode all` batch runs.

```bash
./scripts/runFitDiagnostics.sh --era All --channel Combined --masspoint MHc130_MA90 --method Baseline
python3 python/plotPostfitMass.py --era All --masspoint MHc130_MA90 --method Baseline
./scripts/runPullPlots.sh --era All --channel Combined --masspoint MHc130_MA90 --method Baseline --pull-fit both
```

For the interp arms this runs per GROUP SEED (members share the seed's
backgrounds bitwise) via `./automize/interpFitDiag.sh --all
[--method ParticleNet]`; `plotPostfitMass.py --signal-source interp-signal`
refills the parametric signal from the `param_signal` DCB sidecar.
Stitched per-mHc panels: `./automize/postfitSummary.sh --all` then
`--all --method ParticleNet` → `results/plots/postfit_summary/`. The
summary stitches EVERY group seed of the study (Baseline 64-95,
ParticleNet those 3 on top); building a seed's fine-mass hists is the
whole cost and is per-seed independent, so the DAG fans it out
(`postfit-cache`, one node per seed, 4 GB) and the `postfit-summary` node
(32 GB) then stitches from cache. Run Baseline FIRST — a ParticleNet
panel reads the Baseline seeds' caches, and each arm warms only its own.
The production `interp-signal` source carries no filename token (the ad
hoc `mc-signal` run is the one tagged).

### 7b. Uncertainty breakdown

`sigma(r)` split into signal-model, background-normalization,
experimental and statistical components, at the curated template points.
Cumulative `MultiDimFit --algo grid` scans with `--freezeNuisanceGroups`;
each group's contribution is the quadrature difference between
consecutive scans.

```bash
./automize/breakdown.sh --template-points     # 7 points x 8 nodes = 56
python3 python/collectBreakdown.py --template-points
python3 python/plotBreakdown.py
```

Frozen Combine commands: one best fit `combine -M MultiDimFit
grouped_workspace.root --algo none --setParameterRanges r={RMIN},{RMAX}
--saveWorkspace -n .{mp}.{method}.bestfit -m 120`, then EVERY scan off
that snapshot -- `combine -M MultiDimFit {bestfit.root} --snapshotName
MultiDimFit --algo grid --points 200 --setParameterRanges
r={RMIN},{RMAX} [--freezeNuisanceGroups {cumulative,csv}] -n
.{mp}.{method}.{total|freeze_{tag}} -m 120`.

**The shared snapshot is load-bearing.** Quadrature subtraction compares
intervals across scans, so they must share a minimum. V3 let the total
grid scan be its own snapshot source; ported verbatim that failed on real
data at the three template points whose total scan pinned r-hat at
0.000 -- freezing RAISED sigma, and two returned `stat > total`. See
docs/BREAKDOWN.md "Recipe correction".

Groups live in `configs/nuisance_groups.json`, whose array order IS the
cumulative freeze order: `signal_theory` (theory **and** `CMS_interp_*` —
the interpolation nuisances are signal-model errors), `prompt_norm`,
`nonprompt_norm`, `experimental`, then the residual `stat`. The residual
deliberately carries the autoMCStats `prop_bin*` parameters, which
`text2workspace` generates and which therefore cannot be named in a
`group =` line. An unmatched nuisance is a **hard error**, not a
catch-all, so a new systematic family surfaces instead of being
mis-attributed.

The datacard is never modified: `python/nuisanceGroups.py` appends the
group block to a throwaway `grouped_datacard.txt`, so the production
campaign never needs regenerating.

**The scan range is per point, not fixed** — the one real departure from
V3. V3's `r=-5,5 --points 100` is a step of 0.1 while `sigma(r)` here is
0.08-0.45, i.e. 0.2-1.2 grid steps per sigma, which cannot resolve the
`2*deltaNLL=1` crossing. `resolve_scan_range()` reads the point's own
asymptotic ROOT (`limit` tree already in `r`), takes
`sigma ~ exp0/1.96`, and scans `+-5 sigma` with 200 points: 20 grid
points per sigma at every point. Note `runImpacts.sh resolve_r_range()`
only ever WIDENS past +-5 and cannot be reused.

`collectBreakdown.py` recomputes the components from the scan ROOTs with
the same spline `plot1DScan.py` uses, so the JSON and the PDF cannot
disagree; a negative quadrature subtraction is recorded as `null`, never
zero. Full record: `docs/BREAKDOWN.md`.

### 8. ParticleNet score plots

Part of the ParticleNet DAG (`plot_score` nodes, 4 GB memory request);
also runnable directly via `python3 python/plotParticleNetScore.py`.
Interp arm: `--signal-source interp-signal` reads the per-mHc
shared-scores dirs and the frozen eps_B=20% WP — driver
`./automize/pnetScorePlots.sh --all` (every seed x {Run2,Run3,All} x
{SR1E2Mu,SR3Mu,Combined} + TTZ2E1Mu CR), collector
`python3 python/collectPnetScorePlots.py` → `results/plots/scores/`.

### 9. Collect and plot limits

```bash
python3 python/collectLimits.py --era All --channel Combined --method Baseline
python3 python/plotLimits.py --era All --method Baseline --mhc 130
```

`collectLimits.py` skips-and-logs missing mass points — a partial grid
silently produces a partial JSON, so check the parsed/total count it
reports. `--masspoint` + `--output` give a side-effect-free single-point
collect.

Everything is produced **per channel as well as combined**: both plotters
take `--channel {Combined,SR1E2Mu,SR3Mu}`, which selects the input JSON
and the in-plot final-state label. `Combined` carries no filename token,
`SR1E2Mu`/`SR3Mu` insert one after the era
(`limit.All.SR3Mu.Asymptotic....`). A single-channel ParticleNet panel
reads the Baseline JSON of the SAME channel for the off-window regions,
so the arms are never mixed across channels. The 2D colour scale stays
the mode's fixed `DEFAULT_ZRANGE` for every channel so the three maps are
read against each other; the single channels are weaker and <1% of their
cells sit above the top of the scale.

`plotLimits2D.py` draws the 2D map over the (mHc, mA) plane — mHc on x,
mA on y, colour = the limit — as ONE VERTICAL COLUMN PER MEASURED mHc,
each filled by linear interpolation along its own mA curve. Nothing is
interpolated between columns, because the model does not interpolate in
mHc; cells beyond a column's mA reach are left unpainted, which draws the
kinematic boundary mA <= mHc - 5 and leaves the upper-left corner white
for the information text. The colour range is FIXED per mode
(`DEFAULT_ZRANGE`: BR 5e-7 to 1e-5, xsec its image under the same
sigma_ttbar factor) so every map of the campaign is read on one scale;
`--zrange ZMIN ZMAX` overrides it. `--method ParticleNet` stitches the
ParticleNet arm into its mA window on the columns that have one (mHc =
70, 85 stay Baseline) and dashes in the on-Z/off-Z window edges per
column; `--quantity {exp0,obs}` picks expected or observed.
`--interpolate-mhc` (filename token `.smooth`) instead hands the scan
points to a `TGraph2D` and lets ROOT's Delaunay triangulation fill the
plane: every column's top point lies on mA = mHc - 5, so the hull edge
is that straight line and the staircase disappears. It is a rendering
choice — the model still has no mHc interpolation — so both styles are
produced and the column one stays the default.

Both scripts take `--mode {BR,xsec}` (default `BR`) and the production
carries both units: `BR` is `B_sig`, `xsec` is
`sigma(pp->ttbar) x B_sig` in fb, i.e. the same limit times
`sigma_ttbar(13 TeV) = 833.9 pb`. Each is collected from the Combine
output in its own pass — never rescale one JSON into the other — and
lands in `results/{json,plots}/{BR,xsec}/`. `doThis.sh` loops both modes.

## Batch Workflow (condor)

```bash
# Full chain for one method (per-masspoint DAGs; {Run2,Run3,All} x {SR1E2Mu,SR3Mu,Combined}):
./automize/makeBinnedTemplates.sh --mode all --method Baseline    --fitdiag --pull-fit both
./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --fitdiag --pull-fit both

# Useful options: --masspoint MP, --masspoint-set KEY, --start-from STEP,
#                 --blind, --dry-run
```

DAG topology: 4 template leaves ({Run2,Run3}×{SR1E2Mu,SR3Mu}) → merges →
datacard → validate → asymptotic per target/channel; fitdiag/postfit/pulls
on All/Combined only; plot_score for ParticleNet. Shared generation helpers
live in `automize/dag_lib.sh`; mass-point arrays come from
`automize/load_masspoints.sh` (name-keyed parsing).

### 10. Paper figures

Ported from V3 2026-08-18. Vector PDFs in the publication style, all
three sharing one set of wording/colour/legend constants defined in
`plotPaperLRModified.py` (the other two import them, so the figure sets
cannot drift apart). Output root `results/plots/paper/`.

```bash
python3 python/plotPaperLRModified.py --region all --masspoint all
python3 python/plotPaperTemplates.py --masspoint MHc130_MA90 --method ParticleNet
python3 python/plotPaperPostfitSummary.py            # mHc160, ParticleNet, b-only
```

All three default to `--signal-source interp-signal` (the V4 production
arm) and resolve every path through `srspaths`. The legend is drawn
inside each panel, in two columns, with the signal entry carrying the
exact mass point; `--standalone-legend` instead publishes it once as its
own panel. `plotPaperPostfitSummary.py` reads the fine-mass caches
written by the postfit-summary step, so run that first.

### 11. Template-point artifact bundle

The per-mass-point fit diagnostics live in gitignored template dirs, so
the curated **template points** — one bundle per corner of each arm's
reach — are promoted into the tracked tree, the same rule
`collectPnetScorePlots.py` applies to the LR panels:

```bash
python3 python/collectTemplatePlots.py          # -> results/templates/
```

Baseline `MHc70_MA15, MHc100_MA60, MHc130_MA90, MHc160_MA155`;
ParticleNet `MHc160_MA85, MHc130_MA90, MHc100_MA95` (all seven are group
seeds, which is where GoF/impacts/fitdiag run). Per point: GoF at All ×
3 channels, impacts full/filtered/summary, nuisance pulls full/filtered,
the prefit / postfit_b / postfit_s / prefit-vs-postfit mass plots, and —
ParticleNet only — the score panels incl. the TTZ2E1Mu CR. Plus
`limits.json` and `significance.json`: that point's own limit (per
channel, both units) and observed local significance, lifted out of the
campaign JSONs rather than re-read from Combine, so the bundle can never
disagree with the published values. `--point
METHOD:MASSPOINT` (repeatable) collects a different set; the script
exits 1 listing every missing source, so a partial campaign cannot pass
silently.

### 12. Template closure (MC vs interpolation)

The production limits come from parametric templates everywhere, so at the
mass points that HAVE signal MC the two can be compared directly. This is
that comparison, drawn on the **final production adaptive binning**:

```bash
./automize/templateClosure.sh --method Baseline        # 78 pts x 4 cats = 312 nodes
./automize/templateClosure.sh --method ParticleNet     # 17 pts x 4 cats =  68
python3 python/collectTemplateClosure.py               # -> results/plots/closure/
```

Scope is `configs/masspoints.json` `baseline` (78) and `particlenet` (17)
— exactly the `mc_points` of the two scan grids — times
{Run2, Run3} x {SR1E2Mu, SR3Mu}. One node per category, no DAG edges.

The interpolated template is read straight out of the interp-signal
`shapes.root` and summed over the run period's sub-era components; the
signal MC is filled onto **those same bin edges** with
`binned_template_core.getHist`, the function `makeBinnedTemplates` itself
uses for the MC signal component — so it is the production MC template, not
a rebin, and no `mc-signal` campaign is needed. Verified at
Baseline MHc130_MA90 / Run2 / SR1E2Mu: 33.596 against the stored mc-signal
template's 33.600, and N_interp = 33.017 against the `param_signal` sidecar's
own 33.018.

Uncertainties are drawn **separately**, which is the point of the figure:
the red band on the prediction is the quadrature of every `CMS_interp_*`
nuisance the datacard carries (`scale`/`res` as shape templates, `norm` —
and `eff_pnet` on the ParticleNet arm — as lnN), while the MC keeps its own
statistical bars. Each family is period-level, i.e. ONE nuisance spanning
the period's sub-eras, so its sub-era shifts add linearly and only the
families add in quadrature. The nuisance set is **discovered** from
`extra_systematics.json` and cross-checked against the `shapes.root` keys:
a family in one and not the other is a hard error, never a silent drop.

Two deliberate departures from the production signal build, both recorded
in the script docstring: `cap_stat_errors` is not applied (the honest MC
stat error is what is being drawn against), and the nuisance set is
discovered rather than hardcoded.

These points are **in-sample** — the surfaces were fitted using them — so
this is a production-model closure. `closure/interpolation/loo/` remains the
out-of-sample statement.

Two chi2 are recorded per category **in the JSON** (the panel itself
carries no fit numbers): against MC stat alone, and against MC stat (+) the
assigned band. The first is large by construction (signal MC
stat is ~0.5% in the peak, so a percent-level shape residual is a multi-sigma
pull); the second is the one that answers whether the closure sits inside
what the analysis quotes.

Per-category outputs land in the point's own template dir,
`templates/{mp}/{method}/interp-signal/{era}/{channel}/closure/closure.{cat}.{png,pdf,json}`
(members nest under their seed); the JSON carries the per-bin MC/interp
arrays so the chi2 can be re-derived without reopening ROOT.

The panel follows the paper figures: the caption block is the bold region
tag, the final state below it (`e#mu#mu` / `#mu#mu#mu`), then the mass point
in `plotPaperLRModified.format_signal_label`'s own wording -- imported from
that module rather than restated, so this figure cannot drift from the
published ones. The ratio panel treats the interpolated template as the
prediction: `MC / interp.`, its uncertainty the shaded band at unity and
the MC the points that either sit in it or do not.
`collectTemplateClosure.py` promotes the figures only, to
`results/plots/closure/{method}/{masspoint}/`, and exits 1 listing every
missing source.

**Campaign result (2026-08-19, 380 categories, all nodes clean).**
Baseline: N_interp/N_MC median 0.998, |dev| p90 7.8%; chi2/ndf (+unc)
median 0.60, p90 1.58. ParticleNet: median 0.999, |dev| p90 6.4%;
chi2/ndf median 0.78, p90 2.11. Splitting the Baseline arm at the known
grid-density boundary: |dev| median 2.0% / p90 6.6% above mA = 45, versus
3.6% / 13.8% below it.

Only **6 of 380** categories exceed chi2/ndf (+unc) = 3, and **five sit at
mA = 15** — the documented low-mA grid-density limit, now quantified:

| point | category | N_interp/N_MC | chi2/ndf (+unc) |
|---|---|---|---|
| MHc100_MA15 | SR1E2Mu_Run2 | 1.454 | 8.97 |
| MHc100_MA15 | SR1E2Mu_Run3 | 1.393 | 8.95 |
| MHc130_MA15 | SR3Mu_Run3 | 0.828 | 5.82 |
| MHc160_MA98 | SR3Mu_Run2 | 1.015 | 4.86 |
| MHc130_MA15 | SR3Mu_Run2 | 0.798 | 4.33 |
| MHc145_MA15 | SR3Mu_Run3 | 0.906 | 3.35 |

**MHc100_MA15 is the worst point of the study and is not covered**: +37% to
+45% in all four categories against a `belowZ` norm nuisance of 9%. This is
NOT new and NOT an artifact of this comparison — the frozen
`closure/interpolation/MHc100/yield_closure.json` already records exactly
these ratios (1.454 / 1.393 / 1.373 / 1.321) at 15-20 sigma per-era pulls.
Reproducing them through a completely separate path (production
`shapes.root` + `getHist` on the production adaptive binning) is the
strongest available check that this closure is measuring what it claims.
What is new is that the module docs named MHc115 and MHc130 as the binding
low-mA cases; MHc100_MA15 is worse than either. Its `mc_points` gap
(15 -> 24) is not unusually wide, so the cause is the global surface, not
grid density alone. MHc160_MA98 is the one non-mA=15 entry and is a pure
shape effect (ratio 1.015).

Every other point closes inside the assigned band.

## Configuration

- `configs/masspoints.json` — keys `baseline`, `particlenet`, `limits` only.
- `configs/systematics.{era}.json` — verbatim from V3 (8 files). A
  base+override refactor is deferred: datacard bytes are frozen by the
  reproduction contract.
- `configs/samplegroups.json`, `configs/histkeys.json`, `configs/dagman.config`.

## Blinding

Unblind is the default; there is no partial-unblind in V4. `--blind`
(makeBinnedTemplates/merge/datacard/validate/runAsymptotic/collectLimits
and the driver) produces Asimov data_obs under `{method}_blind`.

## Low-Statistics & Systematics Invariants

Frozen constants (see `docs/LOWSTAT.md` for the full catalog):
`BIN_FLOOR_VALUE=1e-6`, `AUTOMC_THRESHOLD=5`, `SHAPE_REL_ERR_THRESHOLD=0.30`,
`MAX_LNN_VALUE=2.0`, `SYST_MERGE_THRESHOLD=2.0`, autoMCStats threshold 5.
Nuisance naming/correlation rules and the run-period component model are in
`docs/RUN_PERIOD_COMPONENT_TEMPLATES.md`.

## Run3 Signal Samples

Mass points without real Run3 MC fail preprocessing with
`FileNotFoundError` — that is intentional; missing inputs must surface
explicitly, never silently.

## Reproduction Test

`docs/REPRODUCTION.md`. V4's chain was verified to reproduce V3 exactly for
MHc130_MA90 (datacards bitwise, limits identical to the last bit). V3 is
never re-run; `python/compareToV3.py --v3-dir <path>` compares against its
frozen outputs.

## mA Interpolation Chain

Parametric signal templates at fixed mHc and arbitrary mA. Chain:
per-point DCB fits `fitInterpShapes.py` (floating/frozen-n passes) →
shape surfaces `fitInterpPolynomials.py` → shape closure
`closInterpShapes.py`; window yields `measInterpYields.py` → yield model
`fitInterpYieldModel.py` → yield closure `closInterpYields.py`;
shape-systematic deltas `measInterpShapeDeltas.py` →
`fitInterpShapeDeltas.py`; derived nuisance sizes
`exportInterpUncertainties.py`. Constants in
`python/interpolation_config.py`; `configs/interpolation.json` now holds
only `known_missing_samples` (the mHc list comes from `mhc_grid()`).

**Production model (frozen 2026-08-12, no variant flags):** pure DCB for
SR1E2Mu and SR3Mu_lowM, DCB+Chebychev2 for SR3Mu_highM alone
(`channel_has_bkg`). Every shape parameter, plus the yield model's `G`
and `k_era`, is ONE surface in (mHc, mA) fitted across all seven studies
and sliced at the study's mHc — the slice of a surface is a polynomial,
so the stored records and every downstream consumer are unchanged.
Interpolation is **in mA only**.

**Two cross-study barriers**: `polynomials` reads every study's
`dcb_fits.json` and `yield_model` every study's `yields.json`. A rebuild
therefore runs in three passes, each fully complete before the next:

```bash
./automize/interpolation.sh --all --stop-after fit-frozen
./automize/interpolation.sh --all --start-from polynomials --stop-after yields
./automize/interpolation.sh --all --start-from yield-model
./automize/interpolation.sh --loo --all
python3 python/exportInterpUncertainties.py --loo --all --pooled --write-config
python3 python/plotInterpSurfaces.py --all      # global surface plots
python3 python/plotInterpNuisances.py           # nuisance-rule plots
python3 python/plotInterpResiduals.py --all     # signed LOO residual scatters
```

`./automize/interpolation.sh --replot --all` redraws the yield-closure
PNGs alone (7 study nodes + 78 LOO nodes, `closInterpYields.py
--plots-only`). Those JSON payloads carry a timestamp and the argv, so a
plain re-run dirties the frozen production files even when every number is
identical; `--plots-only` skips the write. Use it when the PLOTTING code
changes and the numbers do not.

Outputs under `fits/MHc{X}/` (per-study fit artifacts + plots, global
surface panels in `fits/{params,yield}/`) and `closure/interpolation/`
(per-study closures, `loo/MHc{X}_MA{Y}/` leave-one-out dirs, pooled
diagnostics, nuisance plots) — git-tracked production trees, committed
after a verified run. Merge any sharded stage with
`python3 python/mergeInterpResults.py --mhc N --stage
{fits-floating,fits,closure,yields,yield-closure,shape-deltas}`.

Uncertainties are **derived from the leave-one-out closures**, not
assumed, by one rule for all three families: the rms WITHIN each mHc
study, then the MAX across studies holding at least 2 mass points,
floored by the cell's pooled rms and then by an absolute floor (scale
0.02, res 0.01, norm 0.01). Nothing carries an mHc dependence — the rule
already pools studies and both models are global surfaces. norm VALUES
alone are binned in mA at 15/80/100/155 (the target mA selects the bin);
all three nuisance NAMES are period-level,
`CMS_interp_{scale,res,norm}_{ch}_{13TeV|13p6TeV}` — the LOO residual is
common-mode across a period's eras (r = +0.99 Run2 / +0.80 Run3), so one
nuisance spans the era columns, carrying each era's own lnN value.
Values land in `configs/interpolation_uncertainties.json`, keyed by STUDY
channel (lowM/highM) while the nuisance names use the production channel
SR3Mu — safe because one datacard holds one mass point.

Full record in `docs/interpolation/`: `WORKFLOW.md` (runbook + decision
gates), `EXPERIMENTS.md` (every model decision as motivation/setup/
results/conclusion), `UNCERTAINTY.md` (nuisance rule, values, evidence).

## Interp-Signal Template Production

Parametric-signal templates at every scan-grid point (2467 over 7 mHc;
`configs/grid.json`, band edges [15, 30, 60, 100, –], p-notation names
like `MA90p5`). `makeBinnedTemplates.py --signal-source interp-signal`:
a group SEED builds the shared backgrounds with its own interpolated
mean/sigma; members inject only their parametric signal
(`python/param_signal.py`; shared machinery in
`python/binned_template_core.py`). 572 groups frozen in `grid.json`
(seed lattice 0.5/1/2/4 GeV; never spanning the mA=60 pairing
boundary); members nest at
`templates/{seed}/Baseline/interp-signal/{era}/{channel}/points/{member}`.
Interp nuisances enter via the `extra_systematics*.json` hook, sized
from `configs/interpolation_uncertainties.json`.

```bash
./automize/interpTemplates.sh --all            # full scan (7 DAGs, ~27k nodes;
                                               #  heavy tier ~360 core-hours)
./automize/interpTemplates.sh --mhc 160 --group MHc160_MA90   # one group
python3 python/collectLimits.py --era All --method Baseline --signal-source interp-signal
```

Per point: All × {SR1E2Mu, SR3Mu, Combined} merge → datacard →
asymptotic; one full validation per group at the seed. Verified
2026-08-13 (MHc160_MA90 group): refactor bit-identical on the MC path,
member backgrounds bitwise shared, limits smooth in mA — see
docs/interpolation/WORKFLOW.md "Template production".

GoF + impacts per group seed (mirrors V3; first V4 `text2workspace.py`
use): `./automize/interpGofImpacts.sh --all` — saturated background-only
GoF (500 toys/5 batches) at All × {Combined,SR1E2Mu,SR3Mu}, combineTool
impacts (robustFit, prop_bin-filtered) at All/Combined; summaries via
`python3 python/plotGoFPValues.py --all`. Workers
`scripts/runGoF.sh`/`runImpacts.sh` also accept mc-signal ad hoc.

## ParticleNet Interpolation

A thin layer on the Baseline interpolation, **model frozen 2026-08-14,
production complete 2026-08-15**; full record in
`docs/interpolation/particlenet/` (`METHOD.md`, `UNCERTAINTY.md`). Seeds
at the trained mA = 85/90/95 per mHc, groups of +-2.5 GeV on the 0.5 GeV
lattice, clipped per mHc to the Baseline MC range — the arm reaches
**mA in [82.5, 97.5]** (MHc100 stops at 95): **150 points, 15 groups,
5 mHc** (`configs/pnet_grid.json`, `makePnetGrid.py`); Baseline
templates cover everything outside. Backgrounds are shared per group by
the seed, which also fixes the net and the threshold — a **fixed
eps_B = 20%** working point, replacing the argmax-Asimov-Z rule that
left 5 of 68 categories on 1.6-8.8 background events.

Shapes reuse the Baseline surfaces verbatim (the nets are
mass-decorrelated); the yield is
`k_era * G_period(mA) * f_category(mA) * eps_seed(mA)` with eps a
quadratic through the seed net's three anchors. Two new nuisances only,
`CMS_interp_{res,eff}_pnet_{ch}_{13TeV|13p6TeV}` — there is deliberately
no scale family (the shift is refit statistics, already carried by
autoMCStats), and `eff` covers the eps interpolation ALONE, since the
Baseline `CMS_interp_norm` already covers the yield model.

The production chain (doThis.sh Step 5; frozen artifacts in
`fits/pnet/MHc{X}/{threshold_wp,eps_model}.json`, closures in
`closure/pnet/`, nuisances in
`configs/pnet_interpolation_uncertainties.json`):

```bash
./automize/preprocess.sh --pnet-scores        # 120 jobs, full-syst shared-scores dirs
python3 python/verifyInterpSamples.py --pnet --mhc N   # anti-truncation gate
./automize/pnetInterpolation.sh --all         # 27-node study DAG (one DAG, export barrier)
./automize/interpTemplates.sh  --all --method ParticleNet   # 1610 nodes, 150 points
./automize/interpGofImpacts.sh --all --method ParticleNet   # 330 nodes, 15 seeds
```

Templates land at `templates/{seed}/ParticleNet/interp-signal/...` —
the directory rule is literally Baseline → ParticleNet; every consumer
resolves the group seed method-aware
(`interpolation_config.group_seed(mp, method)`). Sensitivity gain over
Baseline: Combined exp0 median 1.38x (up to ~1.5x at the Z peak);
mc-vs-interp at the 17 trained points vs V3's frozen mc limits: median
1.024. `preprocess.py --shared-scores --mhc MHcX` writes the per-mHc
sample dir (all trained mA, one shared background set, every net's
score branches); `--central-only` is study-only and marks its output
`CENTRAL_ONLY` (the verify gate fails on it).

## Future Phases

A dedicated mc-vs-interp limit comparison plotter over `mc_points` (the
numeric comparison is in docs/interpolation/particlenet/METHOD.md
"Production campaign record"). Nothing in V4 may hard-assume the
template payload is binned histograms beyond the existing per-step
contracts.

Known limitations of the frozen model, all recorded in
docs/interpolation/WORKFLOW.md:

- **Low-mA grid density is the binding constraint.** Below mA ≈ 45 the
  below-Z norm envelopes are 7-19%, driven by MHc115 (15 → 27 → 42) and
  MHc130 (15 → 30 → 55). No functional form fixes this — every basis
  tried fails there identically. One extra MC point in each of those gaps
  would do more than any further model work.
- **No mHc interpolation.** The surfaces are better-constrained models AT
  the seven measured mHc; predicting an unmeasured study from the other
  six is a 4% median / 18% p90 yield error.
- **Closure pulls are uncalibrated** — the assumed 1% Run2 G error is far
  below the ~5% surface residual. The exported envelopes use relative
  residuals and are unaffected.

## Troubleshooting

- `plot_score`/comparison jobs held with "cgroup memory limit": bump
  `RequestMemory` (4096) and `condor_release` — the DAG generator already
  requests 4 GB for plot_score.
- Template-stage numeric drift at ≤1e-12 relative on Run3 categories is
  known cross-worker Minuit/numpy noise (see `docs/REPRODUCTION.md`).
- Sanity check: `python3 -m compileall -q python/` and `bash -n` over
  `scripts/ automize/`.
