#!/bin/bash
./automize/preprocess.sh --mode all
./automize/makeBinnedTemplates.sh --mode all --method Baseline
./automize/makeBinnedTemplates.sh --mode all --method ParticleNet
./automize/plotLimits.sh --stack-baseline
