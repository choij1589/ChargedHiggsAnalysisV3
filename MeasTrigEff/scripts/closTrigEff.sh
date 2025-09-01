#!/bin/bash
ERA=$1

export PATH=$PWD/python:$PATH
closTrigEff.py --era $ERA --channel RunEMu --process DYJets
closTrigEff.py --era $ERA --channel RunEMu --process TTLL_powheg
closTrigEff.py --era $ERA --channel RunDiMu --process DYJets
closTrigEff.py --era $ERA --channel RunDiMu --process TTLL_powheg
if [[ $ERA == "201"* ]]; then
  closTrigEff.py --era $ERA --channel Run1E2Mu --process WZTo3LNu_amcatnlo
  closTrigEff.py --era $ERA --channel Run1E2Mu --process TTZToLLNuNu
  closTrigEff.py --era $ERA --channel Run3Mu --process WZTo3LNu_amcatnlo
  closTrigEff.py --era $ERA --channel Run3Mu --process TTZToLLNuNu
elif [[ $ERA == "202"* ]]; then
  closTrigEff.py --era $ERA --channel Run1E2Mu --process WZTo3LNu_powheg
  closTrigEff.py --era $ERA --channel Run1E2Mu --process TTZ_M50
  closTrigEff.py --era $ERA --channel Run1E2Mu --process TTZ_M4to50
  closTrigEff.py --era $ERA --channel Run3Mu --process WZTo3LNu_powheg
  closTrigEff.py --era $ERA --channel Run3Mu --process TTZ_M50
  closTrigEff.py --era $ERA --channel Run3Mu --process TTZ_M4to50
else
  echo "Unknown era: $ERA"
  exit 1
fi
