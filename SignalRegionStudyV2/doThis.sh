#!/bin/bash
#./automize/preprocess.sh --mode all --condor
#./automize/makeBinnedTemplates.sh --mode all --method Baseline 
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet
#./automize/plotLimits.sh --stack-baseline

./automize/impact.sh --mode all --method Baseline --expect-signal 1 --condor
./automize/impact.sh --mode all --method ParticleNet --expect-signal 1 --condor
./automize/signalInjection.sh --mode all --method Baseline --condor
./automize/signalInjection.sh --mode all --method ParticleNet --condor
