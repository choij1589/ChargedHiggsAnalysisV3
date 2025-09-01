#!/bin/bash
export PATH=$PWD/python:$PATH

plot.py --sample TTLL_powheg
plot.py --sample DYJets
plot.py --sample TTZToLLNuNu
plot.py --sample WZTo3LNu_amcatnlo
plot.py --sample ZZTo4L_powheg
plot.py --sample TTToHcToWAToMuMu-MHc70_MA15
plot.py --sample TTToHcToWAToMuMu-MHc100_MA60
plot.py --sample TTToHcToWAToMuMu-MHc130_MA90
plot.py --sample TTToHcToWAToMuMu-MHc160_MA155
