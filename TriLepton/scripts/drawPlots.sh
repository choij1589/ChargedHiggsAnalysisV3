#!/bin/bash
ERA=$1
CHANNEL=$2
export PATH=$PWD/python:$PATH
export ERA
export CHANNEL

draw_plot() {
    local histkey=$1
    # Check if this is a score plot (starts with "MHc")
    if [[ "$histkey" == MHc* ]]; then
        local mass_point=$(echo "$histkey" | cut -d'/' -f1)
        plot.py --era "$ERA" --channel "$CHANNEL" --histkey "$histkey" --signals "$mass_point"
    else
        plot.py --era "$ERA" --channel "$CHANNEL" --histkey "$histkey"
    fi
}

draw_plot_exclude_WZSF() {
    local histkey=$1
    # Check if this is a score plot (starts with "MHc")
    if [[ "$histkey" == MHc* ]]; then
        local mass_point=$(echo "$histkey" | cut -d'/' -f1)
        plot.py --era "$ERA" --channel "$CHANNEL" --histkey "$histkey" --exclude "WZSF" --signals "$mass_point"
    else
        plot.py --era "$ERA" --channel "$CHANNEL" --histkey "$histkey" --exclude "WZSF"
    fi
}

draw_plot_exclude_ConvSF() {
    local histkey=$1
    # Check if this is a score plot (starts with "MHc")
    if [[ "$histkey" == MHc* ]]; then
        local mass_point=$(echo "$histkey" | cut -d'/' -f1)
        plot.py --era "$ERA" --channel "$CHANNEL" --histkey "$histkey" --exclude "ConvSF" --signals "$mass_point"
    else
        plot.py --era "$ERA" --channel "$CHANNEL" --histkey "$histkey" --exclude "ConvSF"
    fi
}

draw_plot_blind() {
    local histkey=$1
    # Check if this is a score plot (starts with "MHc")
    if [[ "$histkey" == MHc* ]]; then
        local mass_point=$(echo "$histkey" | cut -d'/' -f1)
        plot.py --era "$ERA" --channel "$CHANNEL" --histkey "$histkey" --blind --signals "$mass_point"
    else
        plot.py --era "$ERA" --channel "$CHANNEL" --histkey "$histkey" --blind
    fi
}
export -f draw_plot
export -f draw_plot_blind
export -f draw_plot_exclude_WZSF
export -f draw_plot_exclude_ConvSF

if [[ $CHANNEL == "SR1E2Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi" "muons/1/charge" "muons/1/px" "muons/1/py" "muons/1/pz" "muons/1/energy" "muons/1/min_dR_bjets"
        "muons/2/pt" "muons/2/eta" "muons/2/phi" "muons/2/charge" "muons/2/px" "muons/2/py" "muons/2/pz" "muons/2/energy" "muons/2/min_dR_bjets"
        "electrons/1/pt" "electrons/1/eta" "electrons/1/phi" "electrons/1/charge" "electrons/1/px" "electrons/1/py" "electrons/1/pz" "electrons/1/energy" "electrons/1/min_dR_bjets"
        "jets/size" "bjets/size"
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass" "jets/1/charge" "jets/1/px" "jets/1/py" "jets/1/pz" "jets/1/energy"
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass" "jets/2/charge" "jets/2/px" "jets/2/py" "jets/2/pz" "jets/2/energy"
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass" "jets/3/charge" "jets/3/px" "jets/3/py" "jets/3/pz" "jets/3/energy"
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass" "jets/4/charge" "jets/4/px" "jets/4/py" "jets/4/pz" "jets/4/energy"
        "bjets/1/pt" "bjets/1/eta" "bjets/1/phi" "bjets/1/mass" "bjets/1/charge"
        "bjets/2/pt" "bjets/2/eta" "bjets/2/phi" "bjets/2/mass" "bjets/2/charge"
        "bjets/3/pt" "bjets/3/eta" "bjets/3/phi" "bjets/3/mass" "bjets/3/charge"
        "METv/pt" "METv/phi" "METv/px" "METv/py" "METv/energy"
        "pair/pt" "pair/eta" "pair/phi" "pair/mass"
        "dR_ele_mu1" "dR_ele_mu2" "dR_min_ele_mu" "dR_mu1_mu2"
        "MHc100_MA95/score_nonprompt" "MHc100_MA95/score_diboson" "MHc100_MA95/score_ttZ" "MHc100_MA95/score_signal"
        "MHc100_MA95/LR_nonprompt" "MHc100_MA95/LR_diboson" "MHc100_MA95/LR_ttZ" "MHc100_MA95/LR_totalBkg"
        "MHc115_MA87/score_nonprompt" "MHc115_MA87/score_diboson" "MHc115_MA87/score_ttZ" "MHc115_MA87/score_signal"
        "MHc115_MA87/LR_nonprompt" "MHc115_MA87/LR_diboson" "MHc115_MA87/LR_ttZ" "MHc115_MA87/LR_totalBkg"
        "MHc130_MA90/score_nonprompt" "MHc130_MA90/score_diboson" "MHc130_MA90/score_ttZ" "MHc130_MA90/score_signal"
        "MHc130_MA90/LR_nonprompt" "MHc130_MA90/LR_diboson" "MHc130_MA90/LR_ttZ" "MHc130_MA90/LR_totalBkg"
        "MHc145_MA92/score_nonprompt" "MHc145_MA92/score_diboson" "MHc145_MA92/score_ttZ" "MHc145_MA92/score_signal"
        "MHc145_MA92/LR_nonprompt" "MHc145_MA92/LR_diboson" "MHc145_MA92/LR_ttZ" "MHc145_MA92/LR_totalBkg"
        "MHc160_MA85/score_nonprompt" "MHc160_MA85/score_diboson" "MHc160_MA85/score_ttZ" "MHc160_MA85/score_signal"
        "MHc160_MA85/LR_nonprompt" "MHc160_MA85/LR_diboson" "MHc160_MA85/LR_ttZ" "MHc160_MA85/LR_totalBkg"
        "MHc160_MA98/score_nonprompt" "MHc160_MA98/score_diboson" "MHc160_MA98/score_ttZ" "MHc160_MA98/score_signal"
        "MHc160_MA98/LR_nonprompt" "MHc160_MA98/LR_diboson" "MHc160_MA98/LR_ttZ" "MHc160_MA98/LR_totalBkg"
    )
    parallel draw_plot_blind ::: ${histkeys[@]}
elif [[ $CHANNEL == "TTZ2E1Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi" "muons/1/charge" "muons/1/px" "muons/1/py" "muons/1/pz" "muons/1/energy" "muons/1/min_dR_bjets"
        "electrons/1/pt" "electrons/1/eta" "electrons/1/phi" "electrons/1/charge" "electrons/1/px" "electrons/1/py" "electrons/1/pz" "electrons/1/energy" "electrons/1/min_dR_bjets"
        "electrons/2/pt" "electrons/2/eta" "electrons/2/phi" "electrons/2/charge" "electrons/2/px" "electrons/2/py" "electrons/2/pz" "electrons/2/energy" "electrons/2/min_dR_bjets"
        "jets/size" "bjets/size"
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass" "jets/1/charge" "jets/1/px" "jets/1/py" "jets/1/pz" "jets/1/energy"
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass" "jets/2/charge" "jets/2/px" "jets/2/py" "jets/2/pz" "jets/2/energy"
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass" "jets/3/charge" "jets/3/px" "jets/3/py" "jets/3/pz" "jets/3/energy"
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass" "jets/4/charge" "jets/4/px" "jets/4/py" "jets/4/pz" "jets/4/energy"
        "bjets/1/pt" "bjets/1/eta" "bjets/1/phi" "bjets/1/mass" "bjets/1/charge"
        "bjets/2/pt" "bjets/2/eta" "bjets/2/phi" "bjets/2/mass" "bjets/2/charge"
        "bjets/3/pt" "bjets/3/eta" "bjets/3/phi" "bjets/3/mass" "bjets/3/charge"
        "METv/pt" "METv/phi" "METv/px" "METv/py" "METv/energy"
        "pair/pt" "pair/eta" "pair/phi" "pair/mass"
        "dR_ele1_ele2" "dR_ele1_mu" "dR_ele2_mu" "dR_min_ele_mu"
        "MHc100_MA95/score_nonprompt" "MHc100_MA95/score_diboson" "MHc100_MA95/score_ttZ" "MHc100_MA95/score_signal"
        "MHc100_MA95/LR_nonprompt" "MHc100_MA95/LR_diboson" "MHc100_MA95/LR_ttZ" "MHc100_MA95/LR_totalBkg"
        "MHc115_MA87/score_nonprompt" "MHc115_MA87/score_diboson" "MHc115_MA87/score_ttZ" "MHc115_MA87/score_signal"
        "MHc115_MA87/LR_nonprompt" "MHc115_MA87/LR_diboson" "MHc115_MA87/LR_ttZ" "MHc115_MA87/LR_totalBkg"
        "MHc130_MA90/score_nonprompt" "MHc130_MA90/score_diboson" "MHc130_MA90/score_ttZ" "MHc130_MA90/score_signal"
        "MHc130_MA90/LR_nonprompt" "MHc130_MA90/LR_diboson" "MHc130_MA90/LR_ttZ" "MHc130_MA90/LR_totalBkg"
        "MHc145_MA92/score_nonprompt" "MHc145_MA92/score_diboson" "MHc145_MA92/score_ttZ" "MHc145_MA92/score_signal"
        "MHc145_MA92/LR_nonprompt" "MHc145_MA92/LR_diboson" "MHc145_MA92/LR_ttZ" "MHc145_MA92/LR_totalBkg"
        "MHc160_MA85/score_nonprompt" "MHc160_MA85/score_diboson" "MHc160_MA85/score_ttZ" "MHc160_MA85/score_signal"
        "MHc160_MA85/LR_nonprompt" "MHc160_MA85/LR_diboson" "MHc160_MA85/LR_ttZ" "MHc160_MA85/LR_totalBkg"
        "MHc160_MA98/score_nonprompt" "MHc160_MA98/score_diboson" "MHc160_MA98/score_ttZ" "MHc160_MA98/score_signal"
        "MHc160_MA98/LR_nonprompt" "MHc160_MA98/LR_diboson" "MHc160_MA98/LR_ttZ" "MHc160_MA98/LR_totalBkg"
    )
    parallel draw_plot ::: ${histkeys[@]}
elif [[ $CHANNEL == "SR3Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi" "muons/1/charge" "muons/1/px" "muons/1/py" "muons/1/pz" "muons/1/energy" "muons/1/min_dR_bjets"
        "muons/2/pt" "muons/2/eta" "muons/2/phi" "muons/2/charge" "muons/2/px" "muons/2/py" "muons/2/pz" "muons/2/energy" "muons/2/min_dR_bjets"
        "muons/3/pt" "muons/3/eta" "muons/3/phi" "muons/3/charge" "muons/3/px" "muons/3/py" "muons/3/pz" "muons/3/energy" "muons/3/min_dR_bjets"
        "jets/size" "bjets/size"
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass" "jets/1/charge" "jets/1/px" "jets/1/py" "jets/1/pz" "jets/1/energy"
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass" "jets/2/charge" "jets/2/px" "jets/2/py" "jets/2/pz" "jets/2/energy"
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass" "jets/3/charge" "jets/3/px" "jets/3/py" "jets/3/pz" "jets/3/energy"
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass" "jets/4/charge" "jets/4/px" "jets/4/py" "jets/4/pz" "jets/4/energy"
        "bjets/1/pt" "bjets/1/eta" "bjets/1/phi" "bjets/1/mass" "bjets/1/charge"
        "bjets/2/pt" "bjets/2/eta" "bjets/2/phi" "bjets/2/mass" "bjets/2/charge"
        "bjets/3/pt" "bjets/3/eta" "bjets/3/phi" "bjets/3/mass" "bjets/3/charge"
        "METv/pt" "METv/phi" "METv/px" "METv/py" "METv/energy"
        "pair_lowM/pt" "pair_lowM/eta" "pair_lowM/phi" "pair_lowM/mass"
        "pair_highM/pt" "pair_highM/eta" "pair_highM/phi" "pair_highM/mass"
        "dR_pair_ss1_os" "dR_pair_ss2_os" "dR_pair_ss1_ss2"
        "MHc100_MA95/score_nonprompt" "MHc100_MA95/score_diboson" "MHc100_MA95/score_ttZ" "MHc100_MA95/score_signal"
        "MHc100_MA95/LR_nonprompt" "MHc100_MA95/LR_diboson" "MHc100_MA95/LR_ttZ" "MHc100_MA95/LR_totalBkg"
        "MHc115_MA87/score_nonprompt" "MHc115_MA87/score_diboson" "MHc115_MA87/score_ttZ" "MHc115_MA87/score_signal"
        "MHc115_MA87/LR_nonprompt" "MHc115_MA87/LR_diboson" "MHc115_MA87/LR_ttZ" "MHc115_MA87/LR_totalBkg"
        "MHc130_MA90/score_nonprompt" "MHc130_MA90/score_diboson" "MHc130_MA90/score_ttZ" "MHc130_MA90/score_signal"
        "MHc130_MA90/LR_nonprompt" "MHc130_MA90/LR_diboson" "MHc130_MA90/LR_ttZ" "MHc130_MA90/LR_totalBkg"
        "MHc145_MA92/score_nonprompt" "MHc145_MA92/score_diboson" "MHc145_MA92/score_ttZ" "MHc145_MA92/score_signal"
        "MHc145_MA92/LR_nonprompt" "MHc145_MA92/LR_diboson" "MHc145_MA92/LR_ttZ" "MHc145_MA92/LR_totalBkg"
        "MHc160_MA85/score_nonprompt" "MHc160_MA85/score_diboson" "MHc160_MA85/score_ttZ" "MHc160_MA85/score_signal"
        "MHc160_MA85/LR_nonprompt" "MHc160_MA85/LR_diboson" "MHc160_MA85/LR_ttZ" "MHc160_MA85/LR_totalBkg"
        "MHc160_MA98/score_nonprompt" "MHc160_MA98/score_diboson" "MHc160_MA98/score_ttZ" "MHc160_MA98/score_signal"
        "MHc160_MA98/LR_nonprompt" "MHc160_MA98/LR_diboson" "MHc160_MA98/LR_ttZ" "MHc160_MA98/LR_totalBkg"
    )
    parallel draw_plot_blind ::: ${histkeys[@]}
elif [[ $CHANNEL == "ZFake1E2Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi" "muons/1/charge" "muons/1/px" "muons/1/py" "muons/1/pz" "muons/1/energy"
        "muons/2/pt" "muons/2/eta" "muons/2/phi" "muons/2/charge" "muons/2/px" "muons/2/py" "muons/2/pz" "muons/2/energy"
        "electrons/1/pt" "electrons/1/eta" "electrons/1/phi" "electrons/1/charge" "electrons/1/px" "electrons/1/py" "electrons/1/pz" "electrons/1/energy"
        "jets/size" "bjets/size"
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass" "jets/1/charge" "jets/1/px" "jets/1/py" "jets/1/pz" "jets/1/energy"
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass" "jets/2/charge" "jets/2/px" "jets/2/py" "jets/2/pz" "jets/2/energy"
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass" "jets/3/charge" "jets/3/px" "jets/3/py" "jets/3/pz" "jets/3/energy"
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass" "jets/4/charge" "jets/4/px" "jets/4/py" "jets/4/pz" "jets/4/energy"
        "METv/pt" "METv/phi" "METv/px" "METv/py" "METv/energy"
        "ZCand/pt" "ZCand/eta" "ZCand/phi" "ZCand/mass"
        "nonprompt/pt" "nonprompt/eta"
        "dR_ele_mu1" "dR_ele_mu2" "dR_min_ele_mu" "dR_mu1_mu2"
        "MHc100_MA95/score_nonprompt" "MHc100_MA95/score_diboson" "MHc100_MA95/score_ttZ" "MHc100_MA95/score_signal"
        "MHc100_MA95/LR_nonprompt" "MHc100_MA95/LR_diboson" "MHc100_MA95/LR_ttZ" "MHc100_MA95/LR_totalBkg"
        "MHc115_MA87/score_nonprompt" "MHc115_MA87/score_diboson" "MHc115_MA87/score_ttZ" "MHc115_MA87/score_signal"
        "MHc115_MA87/LR_nonprompt" "MHc115_MA87/LR_diboson" "MHc115_MA87/LR_ttZ" "MHc115_MA87/LR_totalBkg"
        "MHc130_MA90/score_nonprompt" "MHc130_MA90/score_diboson" "MHc130_MA90/score_ttZ" "MHc130_MA90/score_signal"
        "MHc130_MA90/LR_nonprompt" "MHc130_MA90/LR_diboson" "MHc130_MA90/LR_ttZ" "MHc130_MA90/LR_totalBkg"
        "MHc145_MA92/score_nonprompt" "MHc145_MA92/score_diboson" "MHc145_MA92/score_ttZ" "MHc145_MA92/score_signal"
        "MHc145_MA92/LR_nonprompt" "MHc145_MA92/LR_diboson" "MHc145_MA92/LR_ttZ" "MHc145_MA92/LR_totalBkg"
        "MHc160_MA85/score_nonprompt" "MHc160_MA85/score_diboson" "MHc160_MA85/score_ttZ" "MHc160_MA85/score_signal"
        "MHc160_MA85/LR_nonprompt" "MHc160_MA85/LR_diboson" "MHc160_MA85/LR_ttZ" "MHc160_MA85/LR_totalBkg"
        "MHc160_MA98/score_nonprompt" "MHc160_MA98/score_diboson" "MHc160_MA98/score_ttZ" "MHc160_MA98/score_signal"
        "MHc160_MA98/LR_nonprompt" "MHc160_MA98/LR_diboson" "MHc160_MA98/LR_ttZ" "MHc160_MA98/LR_totalBkg"
    )
    parallel draw_plot ::: ${histkeys[@]}
elif [[ $CHANNEL == "ZFake3Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi" "muons/1/charge" "muons/1/px" "muons/1/py" "muons/1/pz" "muons/1/energy"
        "muons/2/pt" "muons/2/eta" "muons/2/phi" "muons/2/charge" "muons/2/px" "muons/2/py" "muons/2/pz" "muons/2/energy"
        "muons/3/pt" "muons/3/eta" "muons/3/phi" "muons/3/charge" "muons/3/px" "muons/3/py" "muons/3/pz" "muons/3/energy"
        "jets/size" "bjets/size"
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass" "jets/1/charge" "jets/1/px" "jets/1/py" "jets/1/pz" "jets/1/energy"
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass" "jets/2/charge" "jets/2/px" "jets/2/py" "jets/2/pz" "jets/2/energy"
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass" "jets/3/charge" "jets/3/px" "jets/3/py" "jets/3/pz" "jets/3/energy"
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass" "jets/4/charge" "jets/4/px" "jets/4/py" "jets/4/pz" "jets/4/energy"
        "METv/pt" "METv/phi" "METv/px" "METv/py" "METv/energy"
        "ZCand/pt" "ZCand/eta" "ZCand/phi" "ZCand/mass"
        "nZCand/pt" "nZCand/eta" "nZCand/phi" "nZCand/mass"
        "nonprompt/pt" "nonprompt/eta"
        "dR_pair_ss1_os" "dR_pair_ss2_os" "dR_pair_ss1_ss2"
        "MHc100_MA95/score_nonprompt" "MHc100_MA95/score_diboson" "MHc100_MA95/score_ttZ" "MHc100_MA95/score_signal"
        "MHc100_MA95/LR_nonprompt" "MHc100_MA95/LR_diboson" "MHc100_MA95/LR_ttZ" "MHc100_MA95/LR_totalBkg"
        "MHc115_MA87/score_nonprompt" "MHc115_MA87/score_diboson" "MHc115_MA87/score_ttZ" "MHc115_MA87/score_signal"
        "MHc115_MA87/LR_nonprompt" "MHc115_MA87/LR_diboson" "MHc115_MA87/LR_ttZ" "MHc115_MA87/LR_totalBkg"
        "MHc130_MA90/score_nonprompt" "MHc130_MA90/score_diboson" "MHc130_MA90/score_ttZ" "MHc130_MA90/score_signal"
        "MHc130_MA90/LR_nonprompt" "MHc130_MA90/LR_diboson" "MHc130_MA90/LR_ttZ" "MHc130_MA90/LR_totalBkg"
        "MHc145_MA92/score_nonprompt" "MHc145_MA92/score_diboson" "MHc145_MA92/score_ttZ" "MHc145_MA92/score_signal"
        "MHc145_MA92/LR_nonprompt" "MHc145_MA92/LR_diboson" "MHc145_MA92/LR_ttZ" "MHc145_MA92/LR_totalBkg"
        "MHc160_MA85/score_nonprompt" "MHc160_MA85/score_diboson" "MHc160_MA85/score_ttZ" "MHc160_MA85/score_signal"
        "MHc160_MA85/LR_nonprompt" "MHc160_MA85/LR_diboson" "MHc160_MA85/LR_ttZ" "MHc160_MA85/LR_totalBkg"
        "MHc160_MA98/score_nonprompt" "MHc160_MA98/score_diboson" "MHc160_MA98/score_ttZ" "MHc160_MA98/score_signal"
        "MHc160_MA98/LR_nonprompt" "MHc160_MA98/LR_diboson" "MHc160_MA98/LR_ttZ" "MHc160_MA98/LR_totalBkg"
    )
    parallel draw_plot ::: ${histkeys[@]}
elif [[ $CHANNEL == "ZG1E2Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi"
        "muons/2/pt" "muons/2/eta" "muons/2/phi"
        "electrons/1/pt" "electrons/1/eta" "electrons/1/phi"
        "jets/size" "bjets/size" 
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass" 
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass" 
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass" 
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass" 
        "METv/pt" "METv/phi"
        "ZCand/pt" "ZCand/eta" "ZCand/phi" "ZCand/mass"
        "convLep/pt" "convLep/eta" "convLep/phi"
    )
    parallel draw_plot ::: ${histkeys[@]}
    parallel draw_plot_exclude_ConvSF ::: ${histkeys[@]}
elif [[ $CHANNEL == "ZG3Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi"
        "muons/2/pt" "muons/2/eta" "muons/2/phi"
        "muons/3/pt" "muons/3/eta" "muons/3/phi"
        "jets/size" 
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass" 
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass" 
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass" 
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass" 
        "METv/pt" "METv/phi"
        "ZCand/pt" "ZCand/eta" "ZCand/phi" "ZCand/mass"
        "convLep/pt" "convLep/eta" "convLep/phi"
    )
    parallel draw_plot ::: ${histkeys[@]}
    parallel draw_plot_exclude_ConvSF ::: ${histkeys[@]}
elif [[ $CHANNEL == "WZ1E2Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi"
        "muons/2/pt" "muons/2/eta" "muons/2/phi"
        "electrons/1/pt" "electrons/1/eta" "electrons/1/phi"
        "jets/size" "bjets/size"
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass"
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass"
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass"
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass"
        "METv/pt" "METv/phi"
        "ZCand/pt" "ZCand/eta" "ZCand/phi" "ZCand/mass"
    )
    parallel draw_plot ::: ${histkeys[@]}
    parallel draw_plot_exclude_WZSF ::: ${histkeys[@]}
elif [[ $CHANNEL == "WZ3Mu" ]]; then
    histkeys=(
        "muons/1/pt" "muons/1/eta" "muons/1/phi"
        "muons/2/pt" "muons/2/eta" "muons/2/phi"
        "muons/3/pt" "muons/3/eta" "muons/3/phi"
        "jets/size" "bjets/size"
        "jets/1/pt" "jets/1/eta" "jets/1/phi" "jets/1/mass"
        "jets/2/pt" "jets/2/eta" "jets/2/phi" "jets/2/mass"
        "jets/3/pt" "jets/3/eta" "jets/3/phi" "jets/3/mass"
        "jets/4/pt" "jets/4/eta" "jets/4/phi" "jets/4/mass"
        "METv/pt" "METv/phi"
        "ZCand/pt" "ZCand/eta" "ZCand/phi" "ZCand/mass"
        "nZCand/pt" "nZCand/eta" "nZCand/phi" "nZCand/mass"
    )
    parallel draw_plot ::: ${histkeys[@]}
    parallel draw_plot_exclude_WZSF ::: ${histkeys[@]}
fi

