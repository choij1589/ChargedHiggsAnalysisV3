#!/bin/bash

# Step 0: Preprocess
#./automize/preprocess.sh --mode all

# Step 1: Templates + Asymptotic limits (blinded + partial-unblind)
#./automize/makeBinnedTemplates.sh --mode all --method Baseline
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --partial-unblind

# Step 2: NP Impacts (r=0 background-only + r=median expected; OR requires both)
#./automize/impact.sh --mode all --method Baseline --expect-signal 0
#./automize/impact.sh --mode all --method Baseline --auto-expect-signal
#./automize/impact.sh --mode all --method ParticleNet --expect-signal 0
#./automize/impact.sh --mode all --method ParticleNet --auto-expect-signal
#./automize/impact.sh --mode all --method ParticleNet --partial-unblind

# Step 3: Goodness-of-Fit test
#./automize/gof.sh --mode all --method Baseline
#./automize/gof.sh --mode all --method ParticleNet --partial-unblind
#./automize/ttz_cr_gof.sh --ntoys 1000
# After HTCondor jobs finish collect and plot:
#./automize/gof.sh --mode all --method Baseline --plot-only
#./automize/gof.sh --mode all --method ParticleNet --partial-unblind --plot-only

# Step 4: FitDiagnostics + post-fit plots + NP pull plots
#./automize/makeBinnedTemplates.sh --mode all --method Baseline --fitdiag --start-from combine --no-runAsymptotic
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --partial-unblind --fitdiag --start-from combine --no-runAsymptotic
# After fitdiag, post-fit mass plots with re-filling from the unbinned tree
#./automize/plotPostfitMass.sh --method ParticleNet --partial-unblind --condor
#./automize/plotPostfitMass.sh --method ParticleNet --partial-unblind --plot-only

# Step 5: Signal injection (bias test); re-run needed after new templates
#./automize/signalInjection.sh --mode all --method Baseline
#./automize/signalInjection.sh --mode all --method ParticleNet
# After HTCondor jobs finish plot:
#./automize/signalInjection.sh --mode all --method Baseline --plot-only
#./automize/signalInjection.sh --mode all --method ParticleNet --plot-only

# Step 6: HybridNew limits (test subset first, then full run)
#./automize/hybridnew.sh --mode all --method Baseline --test --auto-grid
#./automize/hybridnew.sh --mode all --method ParticleNet --test --auto-grid
#./automize/hybridnew.sh --mode all --method Baseline --auto-grid
#./automize/hybridnew.sh --mode all --method ParticleNet --auto-grid

# Step 7: Step 1 unblinding
#./automize/preprocess.sh --mode all --unblind

# Unblind data, but check GoF and impact first with --blind option
#./automize/makeBinnedTemplates.sh --mode all --method Baseline    --unblind
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --unblind
#./automize/gof.sh   --mode all --method Baseline    --unblind --ntoys 1000 --nbatches 10
#./automize/gof.sh   --mode all --method ParticleNet --unblind --ntoys 1000 --nbatches 10
#./automize/impact.sh --mode all --method Baseline    --unblind --blind-result
#./automize/impact.sh --mode all --method ParticleNet --unblind --blind-result

# Plot b-only GoF p-values vs mA for each mHc.
#python3 python/plotGoFPValues.py --mhc 70 160

# Plot full-mA prefit and B-only postfit summaries for each mHc.
#./automize/plotPostfitSummary.sh \
#    --mhc 70 160 \
#    --methods Baseline ParticleNet \
#    --eras Run2 Run3 All \
#    --channels SR1E2Mu SR3Mu Combined \
#    --binning extended \
#    --unblind \
#    --bin-width 1 \
#    --fit-type b \
#    --condor


# FitDiagnostics (prerequisite for post-fit mass plots)
# Current unblinding pull plots default to b-only. For the later S+B review,
#./automize/makeBinnedTemplates.sh --mode all --method Baseline    --unblind --fitdiag --start-from combine --no-runAsymptotic
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --unblind --fitdiag --start-from combine --no-runAsymptotic
# Post-fit mass plots (run after fitDiag DAGs above finish)
#./automize/plotPostfitMass.sh --method Baseline    --unblind --condor
#./automize/plotPostfitMass.sh --method ParticleNet --unblind --condor
# HybridNew
#./automize/hybridnew.sh --mode all --method Baseline --unblind --auto-grid
#./automize/hybridnew.sh --mode all --method ParticleNet --unblind --auto-grid

# Extended-coarser-binning unblind test: full isolated comparison
# Outputs go to extended_coarser_binning_unblind
#./automize/makeBinnedTemplates.sh --mode all --method Baseline    --binning extended_coarser_binning --unblind
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --binning extended_coarser_binning --unblind
./automize/gof.sh --mode all --method Baseline    --binning extended_coarser_binning --unblind --ntoys 1000 --nbatches 10
#./automize/gof.sh --mode all --method ParticleNet --binning extended_coarser_binning --unblind --ntoys 1000 --nbatches 10
#./automize/impact.sh --mode all --method Baseline    --binning extended_coarser_binning --unblind --blind-result
#./automize/impact.sh --mode all --method ParticleNet --binning extended_coarser_binning --unblind --blind-result
#./automize/makeBinnedTemplates.sh --mode all --method Baseline    --binning extended_coarser_binning --unblind --fitdiag --start-from combine --no-runAsymptotic
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --binning extended_coarser_binning --unblind --fitdiag --start-from combine --no-runAsymptotic
#python3 python/plotGoFPValues.py --mhc 70 160 --channels Combined SR1E2Mu SR3Mu --eras All --methods Baseline ParticleNet --suffix extended_coarser_binning_unblind --output-dir results/plots/gof_pvalues/extended_coarser_binning
#./automize/plotPostfitSummary.sh \
#    --mhc 70 160 \
#    --methods Baseline ParticleNet \
#    --eras Run2 Run3 All \
#    --channels SR1E2Mu SR3Mu Combined \
#    --binning extended_coarser_binning \
#    --unblind \
#    --bin-width 1 \
#    --fit-type b \
#    --output-dir results/plots/postfit_summary/extended_coarser_binning \
#    --condor
#./automize/plotPostfitMass.sh --method Baseline    --binning extended_coarser_binning --unblind --condor
#./automize/plotPostfitMass.sh --method ParticleNet --binning extended_coarser_binning --unblind --condor

# Preserve-shape nuisance test: full isolated unblind comparison
# Outputs go to extended_unblind_preserve_shape
#./automize/makeBinnedTemplates.sh --mode all --method Baseline --binning extended --unblind --nuisance preserve_shape
#./automize/gof.sh --mode all --method Baseline --binning extended --unblind --nuisance preserve_shape --ntoys 1000 --nbatches 10
#./automize/impact.sh --mode all --method Baseline --binning extended --unblind --nuisance preserve_shape --blind-result
#./automize/makeBinnedTemplates.sh --mode all --method Baseline --binning extended --unblind --nuisance preserve_shape --fitdiag --start-from combine --no-runAsymptotic
# Post-fit mass plots: default wrapper now saves all era scopes
# (individual eras, Run2, Run3, All), all channel scopes
# (SR1E2Mu, SR3Mu, Combined), and prefit/B-only/S+B postfit.
#./automize/plotPostfitMass.sh --method Baseline --binning extended --unblind --nuisance preserve_shape --condor
#python3 scripts/collectGoFComparison.py --method Baseline --binning extended


# Step 8: Full unblinding after approval to show S+B/observed results
# Re-run observed impacts without --blind-result.
#./automize/impact.sh --mode all --method Baseline    --unblind
#./automize/impact.sh --mode all --method ParticleNet --unblind

# Re-run FitDiagnostics pull plots with both b-only and S+B pulls visible.
# This keeps the current b-only nuisance_pulls.{txt,root,pdf} outputs and
# additionally writes nuisance_pulls_both.{txt,root,pdf}.
#./automize/makeBinnedTemplates.sh --mode all --method Baseline    --unblind --fitdiag --start-from combine --no-runAsymptotic --pull-fit both
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --unblind --fitdiag --start-from combine --no-runAsymptotic --pull-fit both

# Re-render post-fit mass plots after the S+B FitDiagnostics pass.
#./automize/plotPostfitMass.sh --method Baseline    --unblind --condor
#./automize/plotPostfitMass.sh --method ParticleNet --unblind --condor

# Final observed limits.
#./automize/hybridnew.sh --mode all --method Baseline    --unblind --auto-grid
#./automize/hybridnew.sh --mode all --method ParticleNet --unblind --auto-grid

# Collect the full-unblind review artifacts by masspoint.
#./automize/collectUnblindResults.sh --method all
