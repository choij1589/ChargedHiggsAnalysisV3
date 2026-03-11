#!/bin/bash
#./automize/preprocess.sh --mode all

# Generating templates and Asymptotic run
#./automize/makeBinnedTemplates.sh --mode all --method Baseline
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --partial-unblind

# Impact plots
#./automize/impact.sh --mode all --method Baseline --expect-signal 1
#./automize/impact.sh --mode all --method ParticleNet --expect-signal 1
#./automize/impact.sh --mode all --method ParticleNet --partial-unblind

# Signal injection test
#./automize/signalInjection.sh --mode all --method Baseline
#./automize/signalInjection.sh --mode all --method ParticleNet
#./automize/signalInjection.sh --mode all --method Baseline --plot-only
#./automize/signalInjection.sh --mode all --method ParticleNet --plot-only

# HybridNew jobs - test run
#./automize/hybridnew.sh --mode all --method Baseline --test --auto-grid
#./automize/hybridnew.sh --mode all --method ParticleNet --test --auto-grid
./automiez/hybridnew.sh --mode all --method Baseline --test --partial-extract
./automize/hybridnew.sh --mode all --method ParticleNet --test --plot
./automize/hybridnew.sh --mode all --method ParticleNet --test --partial-extract
./automize/hybridnew.sh --mode all --method ParticleNet --test --plot

# HybridNew jobs - full run
#./automize/hybridnew.sh --mode all --method Baseline --auto-grid
#./automize/hybridnew.sh --mode all --method ParticleNet --auto-grid
#./automiez/hybridnew.sh --mode all --method Baseline --partial-extract
#./automize/hybridnew.sh --mode all --method ParticleNet --plot
#./automize/hybridnew.sh --mode all --method ParticleNet --partial-extract
#./automize/hybridnew.sh --mode all --method ParticleNet --plot

# HybridNew jobs - partail unblind (Is it really necessary?)
#./automize/hybridnew.sh --mode all --method ParticleNet --partial-unblind --auto-grid
#./automize/hybridnew.sh --mode all --method ParticleNet --partial-unblind --partial-extract
#/automize/hybridnew.sh --mode all --method ParticleNet --partial-unblind --plot

# Full unblinding - templates and Asymptotic
#./automize/makeBinnedTemplates.sh --mode all --method Baseline --unblind
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --unblind

# Full unblinding - Impact plots
#./automize/impact.sh --mode all --method Baseline --unblind
#./automize/impact.sh --mode all --method ParticleNet --unblind

# Full unblinding - HybridNew
#./automize/hybridnew.sh --mode all --method Baseline --unblind --auto-grid
#./automize/hybridnew.sh --mode all --method ParticleNet --unblind --auto-grid
#./automize/hybridnew.sh --mode all --method Baseline --unblind --partial-extract
#./automize/hybridnew.sh --mode all --method Baseline --unblind --unblind
#./automize/hybridnew.sh --mode all --method ParticleNet --unblind --partial-extract
#./automize/hybridnew.sh --mode all --method ParticleNet --unblind --plot

# Full unblinding - FitDiagnostics
#./automize/makeBinnedTemplates.sh --mode all --method Baseline --binning extended --unblind --fitdiag --start-from combine --no-runAsymptotic
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --binning extended --unblind --fitdiag --start-from combine --no-runAsymptotic
