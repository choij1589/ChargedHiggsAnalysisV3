#!/bin/bash
# SignalRegionStudyV4 runbook.
#
# The production path is unblind by default: no --binning/--unblind flags
# exist anymore. Layout: templates/{masspoint}/{method}/{era}/{channel}.

# Step 0: Preprocess. Shared backgrounds are produced once (24 jobs);
# per-masspoint DAGs add shared signals (+ ParticleNet dirs for trained
# points). See docs/SAMPLES.md.
./automize/preprocess.sh
# Baseline-only production (shared backgrounds + shared signals, 79 DAGs /
# 1272 jobs). Required whenever the per-mHc NoHistMode skims the ParticleNet
# nodes read are incomplete upstream: without it those nodes are submitted
# and fail on missing inputs, masking real failures.
#./automize/preprocess.sh --skip-particlenet
# Later signal-only additions (backgrounds already on pnfs):
#./automize/preprocess.sh --masspoint MHc130_MA90 --skip-backgrounds
#
# Anti-truncation gate. NOT optional: concurrent xrdcp to pnfs has silently
# truncated samples before, and a truncated file fails only much later. Run
# after every preprocessing campaign, before anything consumes the samples.
#for mhc in 70 85 100 115 130 145 160; do
#    python3 python/verifyInterpSamples.py --all --mhc $mhc
#done

# Step 1: Templates, datacards, validation, asymptotic limits, and
# All/Combined FitDiagnostics + postfit + pull plots. With --mode all the
# DAG covers {Run2, Run3, All} x {SR1E2Mu, SR3Mu, Combined} per mass point.
./automize/makeBinnedTemplates.sh --mode all --method Baseline    --fitdiag --pull-fit both
./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --fitdiag --pull-fit both

# Step 2: Collect limits and draw Brazilian plots.
#for method in Baseline ParticleNet; do
#    for era in Run2 Run3 All; do
#        for ch in Combined SR1E2Mu SR3Mu; do
#            python3 python/collectLimits.py --era $era --channel $ch --method $method
#        done
#    done
#done
#python3 python/plotLimits.py --era All --method Baseline    --mhc 160
#python3 python/plotLimits.py --era All --method ParticleNet --stack_baseline

# Step 3: mA interpolation chain (parametric signal templates at fixed mHc).
# Per-mHc condor DAG: shape fits -> polynomials -> shape closure; window
# yields -> yield model -> yield closure; shape-syst deltas -> delta model;
# both closures then feed the derived-uncertainty export. Needs the Step 0
# shared signals for the whole baseline grid (both SR3Mu pairing variants).
# Outputs land in tests/interpolation/MHc{X}/. See docs/INTERPOLATION.md.
#./automize/interpolation.sh --all
# One mHc, or resume a partially-finished study at a named step
# (fit-floating, fit-frozen, polynomials, closure, yields, yield-model,
#  yield-closure, deltas, export); --local runs serially without condor:
#./automize/interpolation.sh --mhc 160 --start-from polynomials
#
# If a sharded stage is interrupted, merge what its jobs produced before
# re-running the stages downstream of it:
#python3 python/mergeInterpResults.py --mhc 160 --stage closure
#
# Derived nuisance sizes from the closure tests (max envelope over held-out
# points) -> configs/interpolation_uncertainties.json. Review the n_points
# field and any >0.10 warnings before trusting the result: on Run3 those are
# the known upstream per-sample scatter, on Run2 they flag a sparse fit-grid
# gap that may argue for an extra anchor in configs/interpolation.json.
#python3 python/exportInterpUncertainties.py --all

# One-time reproduction validation against frozen V3 outputs
# (see docs/REPRODUCTION.md; the samples stage runs on condor via
# scripts/compare_wrapper.sh):
#python3 python/compareToV3.py --masspoint MHc130_MA90 \
#    --v3-dir ../SignalRegionStudyV3 --stage all
