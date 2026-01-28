#!/bin/bash
ERA=$1

mkdir -p ../SKNanoAnalyzer/data/Run3_v13_Run2_v9/${ERA}/EGM/root
## parse electron fake rate
cp results/${ERA}/ROOT/electron/fakerate.root ../SKNanoAnalyzer/data/Run3_v13_Run2_v9/${ERA}/EGM/root/fakerate_TopHNT.root
cp results/${ERA}/ROOT/electron/fakerate_MC.root ../SKNanoAnalyzer/data/Run3_v13_Run2_v9/${ERA}/EGM/root/fakerate_MC_TopHNT.root

mkdir -p ../SKNanoAnalyzer/data/Run3_v13_Run2_v9/${ERA}/MUO/root
## parse muon fake rate
cp results/${ERA}/ROOT/muon/fakerate.root ../SKNanoAnalyzer/data/Run3_v13_Run2_v9/${ERA}/MUO/root/fakerate_TopHNT.root
cp results/${ERA}/ROOT/muon/fakerate_MC.root ../SKNanoAnalyzer/data/Run3_v13_Run2_v9/${ERA}/MUO/root/fakerate_MC_TopHNT.root
