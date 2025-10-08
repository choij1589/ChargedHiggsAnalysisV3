#!/bin/bash
CHANNEL=$1


# Parse arguments for --separate_bjets flag
FLAGS=""
[[ "$*" == *"--separate_bjets"* ]] && FLAGS+=" --separate_bjets"

for signal in MHc160_MA85 MHc130_MA90 MHc100_MA95; do
    echo "Processing $signal..."
    python3 python/visualizeBinary.py --signal $signal --background nonprompt --channel ${CHANNEL} --fold 3${FLAGS}
    python3 python/visualizeBinary.py --signal $signal --background diboson --channel ${CHANNEL} --fold 3${FLAGS}
    python3 python/visualizeBinary.py --signal $signal --background ttZ --channel ${CHANNEL} --fold 3${FLAGS}
    python3 python/visualizeMultiClass.py --signal $signal --channel ${CHANNEL} --fold 3${FLAGS}
done
