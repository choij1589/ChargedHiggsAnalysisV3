#!/bin/bash
# SignalRegionStudyV4 runbook.
#
# Rewritten from scratch 2026-08-12. It carries ONLY the two steps that are
# currently verified against the new SKNanoOutput production (the one that
# added Events_ElectronRecoSF_{Up,Down}): preprocessing and the mA
# interpolation chain. Everything else - binned templates, datacards,
# validation, limits, plots, the V3 comparison - is parked verbatim in
# archive/doThis.sh and comes back one step at a time, only after it has been
# re-run and checked against this production.
#
# Unblind is the default. Layout: templates/{masspoint}/{method}/{era}/{channel}.

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
# Step 1: mA interpolation chain  (docs/INTERPOLATION.md)
# ---------------------------------------------------------------------------
# Parametric signal templates at fixed mHc, arbitrary mA. Per-mHc condor DAG:
#   shape fits (floating -> frozen) -> polynomials -> shape closure
#                                   -> window yields -> yield model -> yield closure
#                                   -> shape-syst deltas -> delta model
# with both closures feeding the derived-uncertainty export. Runs over the FULL
# baseline grid for that mHc (78 mass points across all seven), not just the
# fit anchors - the anchors in configs/interpolation.json are a closure-study
# device; in production every model is refit over the full grid.
# Consumes Step 0's shared signals only (no backgrounds), and needs both SR3Mu
# pairing variants for every point. Outputs: tests/interpolation/MHc{X}/.
#
# Smoke first. MHc145 is the smallest grid that also exercises the
# known-missing-sample branch (82 nodes, ~30 min wall). Gate on it before
# committing to the full campaign: polynomials.json must carry nL and nR in
# every category, and closure.json must have non-empty per-category records -
# an empty closure.json is the silent failure mode, it still exits 0.
#./automize/interpolation.sh --mhc 145

# Full grid (538 nodes across seven mHc), then the derived nuisance sizes:
#./automize/interpolation.sh --all
#python3 python/exportInterpUncertainties.py --all

# Resume a partially-finished study at a named step (fit-floating, fit-frozen,
# polynomials, closure, yields, yield-model, yield-closure, deltas, export).
# Note closure and yields share a level, so either re-runs both:
#./automize/interpolation.sh --mhc 160 --start-from polynomials

# If a sharded stage was interrupted, merge what its jobs did produce before
# re-running anything downstream of it:
#python3 python/mergeInterpResults.py --mhc 160 --stage closure

# On a failed DAG, DAGMan writes dag.dag.rescue001. Resubmit with plain
# condor_submit_dag from inside the per-mHc dir - NOT ./submit_all.sh, which
# passes -f and interferes with the rescue:
#cd condor/jobs_interp_<ts>/MHc145 && condor_submit_dag dag.dag

# Reviewing configs/interpolation_uncertainties.json: check the n_points field
# (a per-era norm envelope resting on 3 points is thin) and any > 0.10
# warnings. On Run3 those are expected - the known upstream per-sample yield
# scatter, absorbed deliberately by the max envelope. On Run2 they instead
# flag a sparse fit-grid gap in the steep phase-space fall, and may argue for
# adding a fit anchor there in configs/interpolation.json.
