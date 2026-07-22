#!/bin/bash

# SignalRegionStudyV3 full-unblind runbook.
#
# V3 keeps explicit blinding options, but this file documents the primary
# full-unblind path. In V3, --binning extended is the adaptive coarser binning
# used for the final unblind workflow.

# Step 0: Preprocess the integrated baseline + ParticleNet mass-point set.
# Preprocessing has no blinding mode.
#./automize/preprocess.sh --mode all

# Step 1: Templates, datacards, validation, asymptotic limits, and
# All+Combined FitDiagnostics/postfit/pull plots. With --mode all, the template
# workflow always submits downstream datacard/validation/asymptotic targets for:
#   {Run2, Run3, All} x {SR1E2Mu, SR3Mu, Combined}
#./automize/makeBinnedTemplates.sh --mode all --method Baseline    --binning extended --unblind  --fitdiag --pull-fit both
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --binning extended --unblind --fitdiag --pull-fit both
#./automize/makeBinnedTemplates.sh --mode all --method Baseline    --binning extended --unblind  --fitdiag --pull-fit b
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --binning extended --unblind --fitdiag --pull-fit b

# Step 2: Goodness-of-Fit tests.
#./automize/gof.sh --mode all --method Baseline    --binning extended --unblind --ntoys 1000 --nbatches 10
#./automize/gof.sh --mode all --method ParticleNet --binning extended --unblind --ntoys 1000 --nbatches 10
#./automize/gof.sh --mode all --method Baseline    --binning extended --unblind --plot-only
#./automize/gof.sh --mode all --method ParticleNet --binning extended --unblind --plot-only
#python3 python/plotGoFPValues.py --mhc 70 85 100 115 130 145 160 --eras {Run2,Run3,All} --channel Combined --methods Baseline ParticleNet
#python3 python/plotGoFPValues.py --mhc 70 85 100 115 130 145 160 --eras All --channels {Combined,SR1E2Mu,SR3Mu} --methods Baseline ParticleNet

# Step 3: Impacts. Use --blind-result for the first unblind review pass.
#./automize/impact.sh --mode all --method Baseline    --binning extended --unblind
#./automize/impact.sh --mode all --method ParticleNet --binning extended --unblind
#./automize/impact.sh --mode all --method Baseline    --binning extended --unblind --blind-result
#./automize/impact.sh --mode all --method ParticleNet --binning extended --unblind --blind-result

# Grouped impact breakdown for selected review mass points.
./scripts/runImpactBreakdown.sh --era All --channel Combined --masspoint MHc70_MA15    --method Baseline    --binning extended --unblind --condor
./scripts/runImpactBreakdown.sh --era All --channel Combined --masspoint MHc100_MA60   --method Baseline    --binning extended --unblind --condor
./scripts/runImpactBreakdown.sh --era All --channel Combined --masspoint MHc130_MA90   --method ParticleNet --binning extended --unblind --condor
./scripts/runImpactBreakdown.sh --era All --channel Combined --masspoint MHc160_MA155  --method Baseline    --binning extended --unblind --condor

# Step 4: Prefit/postfit mass plots and full-mA summary plots.
#./automize/plotPostfitMass.sh --method Baseline    --binning extended --unblind --condor
#./automize/plotPostfitMass.sh --method ParticleNet --binning extended --unblind --condor
#./automize/plotPostfitSummary.sh --mhc 70 85 100 115 130 145 160 \
#    --methods Baseline ParticleNet --eras Run2 Run3 All \
#    --channels SR1E2Mu SR3Mu Combined --binning extended --unblind \
#    --bin-width 1 --signal-line median --fit-type both \
#    --output-dir results/plots/postfit_summary/extended --condor

#python3 python/plotPostfitSummary.py --mhc 160 \
#    --methods Baseline ParticleNet --eras All --channels Combined \
#    --binning extended --nuisance fallback_lnn --fit-type b \
#    --bin-width 1 --output-dir results/plots/postfit_summary/extended \
#    --signal-line median --signal-mass 30 60 90 120 \
#    --unblind --signal-region-style

# Paper LR_modified ParticleNet score plots for SR and TTZ CR.
# Outputs:
#   results/paper/SR/LR_modified_{masspoint}.pdf
#   results/paper/TTZCR/LR_modified_{masspoint}.pdf
#python3 python/plotPaperLRModified.py --region all --masspoint all --output-root results/paper

# Paper b-only postfit SR mass summary plots.
# Outputs: results/paper/Postfit/postfit_b_mHc160_{channel}_{region}.pdf
#python3 python/plotPaperPostfitSummary.py

# Signal injection and bias tests.
#./automize/signalInjection.sh --mode all --method Baseline    --binning extended
#./automize/signalInjection.sh --mode all --method ParticleNet --binning extended
#./automize/signalInjection.sh --mode all --method Baseline    --binning extended --plot-only
#./automize/signalInjection.sh --mode all --method ParticleNet --binning extended --plot-only

#
#TTZ control-region GoF validation.
#./automize/ttz_cr_gof.sh --ntoys 1000

# Backfill limits from existing templates when templates already exist but
# datacards/asymptotic outputs are missing for the full target grid.
#./automize/makeBinnedTemplates.sh --mode all --method Baseline    --binning extended --unblind \
#    --start-from datacard --no-fitdiag
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --binning extended --unblind \
#    --start-from datacard --no-fitdiag --no-plot-score

# Backfill starting for fitdiag with S+B fit
#./automize/makeBinnedTemplates.sh --mode all --method Baseline    --binning extended --unblind \
#    --fitdiag --start-from fitdiag --pull-fit both
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --binning extended --unblind \
#    --fitdiag --start-from fitdiag --pull-fit both

# ParticleNet score plots, including TTZ2E1Mu validation score distributions,
# are now part of the makeBinnedTemplates ParticleNet DAG above:
#   {Run2, Run3, All} x {SR1E2Mu, SR3Mu, Combined}
# The per-channel SR1E2Mu/SR3Mu score jobs also produce scores/TTZ2E1Mu.

# HybridNew observed limits.
#./automize/hybridnew.sh --mode all --method Baseline    --binning extended --unblind --auto-grid
#./automize/hybridnew.sh --mode all --method ParticleNet --binning extended --unblind --auto-grid

# Collect review artifacts by mass point.
#./automize/collectUnblindResults.sh --method all --binning extended
