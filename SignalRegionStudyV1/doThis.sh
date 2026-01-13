#!/bin/bash
./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint MHc100_MA60 --method Baseline --binning extended --condor
./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint MHc100_MA95 --method ParticleNet --binning extended --partial-unblind --condor
./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint MHc160_MA85 --method ParticleNet --binning extended --partial-unblind --condor
./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint MHc130_MA90 --method ParticleNet --binning uniform  --condor
