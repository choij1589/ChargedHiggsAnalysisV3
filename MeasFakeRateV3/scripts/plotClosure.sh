#!/bin/bash
ERA=$1

export PATH="$PWD/python:$PATH"
CHANNELs=("Run1E2Mu" "Run3Mu")
HISTKEYs_El=("pair/mass" "electrons/1/pt" "muons/1/pt" "muons/2/pt" "electrons/1/scEta" "muons/1/eta" "muons/2/eta" "nonprompt/pt" "nonprompt/eta")
HISTKEYs_Mu=("pair_lowM/mass" "pair_highM/mass" "ZCand/mass" "nZCand/mass" "muons/1/pt" "muons/2/pt" "muons/3/pt" "muons/1/eta" "muons/2/eta" "muons/3/eta" "nonprompt/pt" "nonprompt/eta")

plot_closure() {
    local era=$1
    local channel=$2
    local histkey=$3
    plotClosure.py --era $era --channel $channel --histkey $histkey
}

export -f plot_closure

parallel plot_closure ::: $ERA ::: Run1E2Mu ::: ${HISTKEYs_El[@]}
parallel plot_closure ::: $ERA ::: Run3Mu ::: ${HISTKEYs_Mu[@]}
