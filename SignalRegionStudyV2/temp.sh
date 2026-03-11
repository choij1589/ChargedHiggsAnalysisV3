#!/bin/bash
for mp in MHc100_MA95 MHc130_MA90 MHc160_MA85; do
    bash scripts/runAsymptotic.sh --era All --channel Combined \
        --masspoint $mp --method ParticleNet --binning extended --partial-unblind &
done
