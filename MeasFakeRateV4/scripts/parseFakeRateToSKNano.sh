#!/bin/bash
ERA=("2016preVFP" "2016postVFP" "2017" "2018" "2022" "2022EE" "2023" "2023BPix")

for era in "${ERA[@]}"; do
    mkdir -p ../SKNanoAnalyzer/data/Run3_v13_Run2_v9/${era}/EGM/root
    ## parse electron fake rate
    cp results/${era}/ROOT/electron/fakerate.root ../SKNanoAnalyzer/data/Run3_v13_Run2_v9/${era}/EGM/root/fakerate_TopHNT.root
    cp results/${era}/ROOT/electron/fakerate_MC.root ../SKNanoAnalyzer/data/Run3_v13_Run2_v9/${era}/EGM/root/fakerate_MC_TopHNT.root

    mkdir -p ../SKNanoAnalyzer/data/Run3_v13_Run2_v9/${era}/MUO/root
    ## parse muon fake rate
    cp results/${era}/ROOT/muon/fakerate.root ../SKNanoAnalyzer/data/Run3_v13_Run2_v9/${era}/MUO/root/fakerate_TopHNT.root
    cp results/${era}/ROOT/muon/fakerate_MC.root ../SKNanoAnalyzer/data/Run3_v13_Run2_v9/${era}/MUO/root/fakerate_MC_TopHNT.root
done