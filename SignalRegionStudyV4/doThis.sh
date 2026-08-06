#!/bin/bash
# SignalRegionStudyV4 runbook.
#
# The production path is unblind by default: no --binning/--unblind flags
# exist anymore. Layout: templates/{masspoint}/{method}/{era}/{channel}.

# Step 0: Preprocess the integrated baseline + ParticleNet mass-point set.
./automize/preprocess.sh --mode all

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

# One-time reproduction validation against frozen V3 outputs
# (see docs/REPRODUCTION.md; the samples stage runs on condor via
# scripts/compare_wrapper.sh):
#python3 python/compareToV3.py --masspoint MHc130_MA90 \
#    --v3-dir ../SignalRegionStudyV3 --stage all
