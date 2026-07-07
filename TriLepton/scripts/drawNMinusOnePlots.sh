#!/bin/bash
ERA=$1
CHANNEL=$2
export PATH=$PWD/python:$PATH
export ERA
export CHANNEL

draw_nminusone_plot() {
    local histkey=$1
    plotNMinusOne.py --era "$ERA" --channel "$CHANNEL" --histkey "$histkey"
}

export -f draw_nminusone_plot

if [[ $CHANNEL == "SR1E2Mu" ]]; then
    histkeys=(
        "os_mumu/mu_charge_sum"
        "os_mumu/pair/mass"
        "njet_ge2/n_jets"
        "njet_ge2/pair/mass"
        "nbjet_ge1/n_bjets"
        "nbjet_ge1/pair/mass"
        "baseline/jets/size"
        "baseline/bjets/size"
        "baseline/pair/mass"
    )
    parallel draw_nminusone_plot ::: ${histkeys[@]}
elif [[ $CHANNEL == "SR3Mu" ]]; then
    histkeys=(
        "charge_abs1/charge_sum"
        "charge_abs1/pair_lowM/mass"
        "charge_abs1/pair_highM/mass"
        "njet_ge2/n_jets"
        "njet_ge2/pair_lowM/mass"
        "njet_ge2/pair_highM/mass"
        "nbjet_ge1/n_bjets"
        "nbjet_ge1/pair_lowM/mass"
        "nbjet_ge1/pair_highM/mass"
        "baseline/jets/size"
        "baseline/bjets/size"
        "baseline/pair_lowM/mass"
        "baseline/pair_highM/mass"
    )
    parallel draw_nminusone_plot ::: ${histkeys[@]}
else
    echo "N-1 plots are only available for SR1E2Mu and SR3Mu"
    exit 1
fi
