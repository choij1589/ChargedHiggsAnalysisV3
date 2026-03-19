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
./automize/impact.sh --mode all --method ParticleNet --partial-unblind

# Step 3: Goodness-of-Fit test
#./automize/gof.sh --mode all --method Baseline
#./automize/gof.sh --mode all --method ParticleNet --partial-unblind
# After HTCondor jobs finish collect and plot:
#./automize/gof.sh --mode all --method Baseline --plot-only
#./automize/gof.sh --mode all --method ParticleNet --partial-unblind --plot-only

# Step 4: FitDiagnostics + post-fit plots + NP pull plots
#./automize/makeBinnedTemplates.sh --mode all --method Baseline --fitdiag --start-from combine --no-runAsymptotic
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --partial-unblind --fitdiag --start-from combine --no-runAsymptotic

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

# Step 7: Full unblinding (after OR approval)
# Templates + Asymptotic
#./automize/makeBinnedTemplates.sh --mode all --method Baseline --unblind
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --unblind
# Impact plots
#./automize/impact.sh --mode all --method Baseline --unblind
#./automize/impact.sh --mode all --method ParticleNet --unblind
# FitDiagnostics
#./automize/makeBinnedTemplates.sh --mode all --method Baseline --unblind --fitdiag --start-from combine --no-runAsymptotic
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --unblind --fitdiag --start-from combine --no-runAsymptotic
# HybridNew
#./automize/hybridnew.sh --mode all --method Baseline --unblind --auto-grid
#./automize/hybridnew.sh --mode all --method ParticleNet --unblind --auto-grid
