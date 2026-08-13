#!/bin/bash
# SignalRegionStudyV4 runbook.
#
# Rewritten 2026-08-12 (interpolation model frozen). It carries ONLY the steps
# verified against the current SKNanoOutput production (the one that added
# Events_ElectronRecoSF_{Up,Down}): preprocessing and the mA interpolation
# chain. Everything else - binned templates, datacards, validation, limits,
# plots, the V3 comparison - is parked verbatim in archive/doThis.sh and comes
# back one step at a time, only after it has been re-run and checked against
# this production.
#
# Unblind is the default.
# Layout: templates/{masspoint}/{method}/{mc-signal,interp-signal}/{era}/{channel}
#         (interp-signal group members nest under their seed: .../points/{member}).

# ---------------------------------------------------------------------------
# Step 0: Preprocess  (docs/SAMPLES.md)
# ---------------------------------------------------------------------------
# Shared backgrounds are mass-independent and produced once (24 jobs);
# per-masspoint DAGs add the shared signals, writing BOTH SR3Mu pairing
# variants because the interpolation chain needs both. Needs a grid proxy
# (voms-proxy-init --voms=cms) - the wrapper xrdcp's its output to pnfs.
#
# --skip-particlenet emits shared backgrounds + shared signals only, which is
# what the Baseline chain and the interpolation study consume. Use it whenever
# the per-mHc NoHistMode skims that the ParticleNet nodes read are incomplete
# upstream, otherwise those nodes are submitted only to fail on missing inputs
# and their noise buries the real failures. 79 DAGs / 1272 jobs.
./automize/preprocess.sh --skip-particlenet

# Full production including the ParticleNet per-masspoint dirs (requires the
# _MHc{X}_*NoHistMode input skims for every ParticleNet-trained point):
#./automize/preprocess.sh

# Later signal-only additions (shared backgrounds already on pnfs):
#./automize/preprocess.sh --masspoint MHc130_MA90 --skip-backgrounds

# Anti-truncation gate. NOT optional, and it blocks everything downstream:
# concurrent xrdcp to pnfs has silently truncated samples before, and a
# truncated file otherwise surfaces only much later as a confusing fit
# failure. Opens every signal file and fails on zombie / kRecovered / missing
# or empty Central tree / sum(weight) <= 0. Exits 1 on any problem.
for mhc in 70 85 100 115 130 145 160; do
    python3 python/verifyInterpSamples.py --all --mhc $mhc
done

# BEFORE overwriting samples/ with a new production, preserve the old tree.
# The SKNanoOutput inputs behind it get replaced upstream, so an overwritten
# samples/ is NOT regenerable. Rename on the same dCache filesystem - a
# namespace operation, instant, no data copied - and let the campaign rebuild
# samples/ (preprocess_wrapper.sh mkdir -p's its destination):
#cd /pnfs/knu.ac.kr/data/cms/store/user/choij/SignalRegionStudyV4
#mv samples samples_backup_$(date +%Y%m%d)

# ---------------------------------------------------------------------------
# Step 1: mA interpolation chain  (docs/interpolation/WORKFLOW.md)
# ---------------------------------------------------------------------------
# Parametric signal templates at fixed mHc, arbitrary mA, over the FULL
# baseline grid (78 mass points across seven mHc). Consumes Step 0's shared
# signals only (no backgrounds) and needs both SR3Mu pairing variants for
# every point. Outputs: fits/ (models + validation plots) and
# closure/interpolation/ (closures, LOO, nuisance diagnostics) — both
# git-tracked production trees.
#
# ===========================================================================
# THE CHAIN IS NO LONGER SEVEN INDEPENDENT PER-mHc RUNS.
#
# Every mA dependence - each shape parameter, and the yield model's G and
# k_era - is ONE surface in (mHc, mA) fitted across all seven studies and
# sliced at each study's mHc. So:
#
#   * `polynomials` reads EVERY study's fits/dcb_fits.json
#   * `yield_model` reads EVERY study's yields/yields.json
#
# A single-mHc run cannot get past `polynomials`: it raises FileNotFoundError
# naming the study it is missing. That is deliberate - the alternative is a
# surface silently fitted on a subset. Run the passes below instead, each one
# fully finished before the next is submitted.
# ===========================================================================

# Pass 1 - per-point DCB fits, floating then frozen-n (~170 nodes).
./automize/interpolation.sh --all --stop-after fit-frozen

# Pass 2 - shape surfaces, shape closure, window yields (~180 nodes).
# The surfaces are fitted here, which is why pass 1 must be complete.
./automize/interpolation.sh --all --start-from polynomials --stop-after yields

# Pass 3 - yield model, yield closure, shape-systematic deltas (~190 nodes).
./automize/interpolation.sh --all --start-from yield-model

# Leave-one-out sweep: 78 nodes, each refitting BOTH surfaces without one
# point and closing shape+yield at that point. This is where the uncertainties
# come from - the full-grid closures above are in-sample. It only needs the
# fits and yields, so it may run in parallel with pass 3.
./automize/interpolation.sh --loo --all

# Derived nuisance sizes + the production config. JSON-only, seconds, login
# node is fine.
python3 python/exportInterpUncertainties.py --loo --all --pooled --write-config

# Global plots: the surfaces (seven slices per panel with each study's points)
# and the nuisance rule (per-study rms vs mHc, adopted value, pooled rms).
python3 python/plotInterpSurfaces.py --all
python3 python/plotInterpNuisances.py

# --- recovery ---------------------------------------------------------------
# Resume a pass at a named step (fit-floating, fit-frozen, polynomials,
# closure, yields, yield-model, yield-closure, deltas). closure and yields
# share a level, so either re-runs both. Nodes outside
# [--start-from, --stop-after] are emitted but marked DONE, so the dependency
# graph stays intact:
#./automize/interpolation.sh --mhc 160 --start-from yield-model

# If a sharded stage was interrupted, merge what its jobs did produce before
# re-running anything downstream of it:
#python3 python/mergeInterpResults.py --mhc 160 --stage closure

# On a failed DAG, DAGMan writes dag.dag.rescue001. Resubmit with plain
# condor_submit_dag from inside the per-mHc dir - NOT ./submit_all.sh, which
# passes -f and interferes with the rescue:
#cd condor/jobs_interp_<ts>/MHc145 && condor_submit_dag dag.dag

# Check a campaign really succeeded - "the DAG finished" is not the same as
# "it worked", and an export node failing at the end is easy to miss:
#grep -h "EXITING WITH STATUS" condor/jobs_interp_<ts>/MHc*/dag.dag.dagman.out

# --- reviewing the result ---------------------------------------------------
# configs/interpolation_uncertainties.json. One rule for all three families:
# the rms WITHIN each mHc study, then the MAX across studies holding >= 2 mass
# points, floored by the pooled rms and then by an absolute floor (scale 0.02,
# res 0.01, norm 0.01). Nothing carries an mHc dependence; norm alone is
# binned in mA at 15/80/100/155.
#
# What to look at, in order:
#   * per_study_detail[cell].driver - which study set the value, and
#     per_study_rms for how far the others sit below it. A cell driven by one
#     study with a sparse low-mA grid (MHc115, MHc130) is the grid talking,
#     not the model.
#   * studies_below_min_points - studies skipped for holding < 2 points.
#   * n_points - a cell resting on very few points is flagged in warnings.
#   * any value sitting EXACTLY on its floor: the floor is then setting the
#     number rather than catching a degenerate cell, and wants revisiting.
#     With the frozen model none of them is active.
#   * warnings for empty-but-reachable mA bins; those inherit the channel's
#     worst populated bin (never the bare floor) and say so.
# Run3 norm cells above 10% are expected - the known upstream per-sample yield
# scatter, which is the REFERENCE's error, not the model's.

# ---------------------------------------------------------------------------
# Step 2: interp-signal template scan  (docs/interpolation/WORKFLOW.md,
#         "Template production")
# ---------------------------------------------------------------------------
# Parametric-signal templates + datacards + asymptotic limits at every
# grid.json point (2467 over 7 mHc). Grouping is FROZEN in grid.json: each
# group's seed builds the shared backgrounds (4 heavy jobs, 6 GB), members
# inject only their signal. Per point: All x {SR1E2Mu,SR3Mu,Combined}.
# Requires Step 1's fits/ tree (polynomials, yield_model, delta_model) and
# configs/interpolation_uncertainties.json.
./automize/interpTemplates.sh --all

# Single group (e.g. verification, or re-running one seed's neighbourhood):
#./automize/interpTemplates.sh --mhc 160 --group MHc160_MA90

# Gate: every DAG must exit 0 -
#grep -h "EXITING WITH STATUS" condor/jobs_interp_templates_<ts>/MHc*/dag.dag.dagman.out

# Collect the scan into results/json (source token in the filename keeps it
# apart from the mc-signal collection), then plot:
python3 python/collectLimits.py --era All --method Baseline --signal-source interp-signal
python3 python/collectLimits.py --era All --channel SR1E2Mu --method Baseline --signal-source interp-signal
python3 python/collectLimits.py --era All --channel SR3Mu   --method Baseline --signal-source interp-signal
#python3 python/plotLimits.py --era All --method Baseline --signal-source interp-signal --mhc 160

# Direct-MC comparison at the 78 mc_points: run the standard mc-signal chain
# at the same point and compare limits -
#./automize/makeBinnedTemplates.sh --masspoint MHc160_MA90 --method Baseline
