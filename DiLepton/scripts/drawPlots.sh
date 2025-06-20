#!/bin/bash
ERA=$1
CHANNEL=$2
export PATH="${PWD}/python:${PATH}"
export ERA
export CHANNEL

draw_plot() {
    local histkey=$1
    plot.py --era "$ERA" --channel "$CHANNEL" --histkey "$histkey"
}

if [[ $CHANNEL == *"EMu"* ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi"
        "electrons/1/pt" "electrons/1/eta" "electrons/1/phi"
        "METv/pt" "METv/phi"
        "jets/size" "bjets/size"
        "jets/1/pt" "jets/1/eta" "jets/1/phi"
        "jets/2/pt" "jets/2/eta" "jets/2/phi"
        "jets/3/pt" "jets/3/eta" "jets/3/phi"
        "jets/4/pt" "jets/4/eta" "jets/4/phi"
        "jets/5/pt" "jets/5/eta" "jets/5/phi"
        "bjets/1/pt" "bjets/1/eta" "bjets/1/phi"
        "bjets/2/pt" "bjets/2/eta" "bjets/2/phi"
        "bjets/3/pt" "bjets/3/eta" "bjets/3/phi"
    )
    export -f draw_plot
    parallel draw_plot ::: "${histkeys[@]}"
elif [[ $CHANNEL == *"DiMu"* ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi"
        "muons/2/pt" "muons/2/eta" "muons/2/phi"
        "pair/mass" "pair/pt" "pair/eta" "pair/phi"
        "METv/pt" "METv/phi"
        "jets/size" "bjets/size"
        "jets/1/pt" "jets/1/eta" "jets/1/phi"
        "jets/2/pt" "jets/2/eta" "jets/2/phi"
        "jets/3/pt" "jets/3/eta" "jets/3/phi"
        "jets/4/pt" "jets/4/eta" "jets/4/phi"
        "jets/5/pt" "jets/5/eta" "jets/5/phi"
        "bjets/1/pt" "bjets/1/eta" "bjets/1/phi"
        "bjets/2/pt" "bjets/2/eta" "bjets/2/phi"
        "bjets/3/pt" "bjets/3/eta" "bjets/3/phi"
    )
    export -f draw_plot
    parallel draw_plot ::: "${histkeys[@]}"
fi
   
