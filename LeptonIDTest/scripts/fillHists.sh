#!/bin/bash
export PATH=$PWD/python:$PATH

ERAs=("2017" "2018" "2022EE" "2023" "2023BPix")

fill_hists() {
    local era=$1
    fill_electrons.py --era $era
    fill_muons.py --era $era
}
export -f fill_hists
parallel -j 4 fill_hists ::: ${ERAs[@]}
