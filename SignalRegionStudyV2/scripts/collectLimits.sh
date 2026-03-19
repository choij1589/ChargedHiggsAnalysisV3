#!/bin/bash

ERAs=("2016preVFP" "2016postVFP" "2017" "2018"
      "2022" "2022EE" "2023" "2023BPix"
      "Run2" "Run3" "All")

for era in "${ERAs[@]}"; do
    python3 python/collectLimits.py --era "$era" --method Baseline --limit_type Asymptotic
    python3 python/collectLimits.py --era "$era" --method ParticleNet --limit_type Asymptotic
    python3 python/plotLimits.py --era $era --method Baseline --limit_type Asymptotic --blind
    python3 python/plotLimits.py --era $era --method ParticleNet --limit_type Asymptotic --stack_baseline --blind
done

