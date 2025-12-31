#!/bin/bash
#./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint MHc70_MA15 --method Baseline --binning uniform
./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint MHc70_MA15 --method Baseline --binning extended
./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint MHc100_MA60 --method Baseline --binning uniform
./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint MHc100_MA60 --method Baseline --binning extended
./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint MHc130_MA90 --method Baseline --binning uniform
./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint MHc130_MA90 --method Baseline --binning extended
./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint MHc160_MA155 --method Baseline --binning uniform
./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint MHc160_MA155 --method Baseline --binning extended
./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint MHc130_MA90 --method ParticleNet --binning uniform
./scripts/runImpacts.sh --era Run2 --channel Combined --masspoint MHc130_MA90 --method ParticleNet --binning extended
