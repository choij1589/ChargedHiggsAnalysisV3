#!/bin/bash
ERA=$1

cp ~/Sync/workspace/SKNanoAnalyzer/data/Run3_v13_Run2_v9/$ERA/EGM/root/efficiency_TopHNT.root results/$ERA/ROOT/efficiency_EleID.root
cp ~/Sync/workspace/SKNanoAnalyzer/data/Run3_v13_Run2_v9/$ERA/EGM/root/efficiency_Mu8El23_El23Leg.root results/$ERA/ROOT/Mu8El23_electron.root
cp ~/Sync/workspace/SKNanoAnalyzer/data/Run3_v13_Run2_v9/$ERA/EGM/root/efficiency_Mu23El12_El12Leg.root results/$ERA/ROOT/Mu23El12_electron.root

cp ~/Sync/workspace/SKNanoAnalyzer/data/Run3_v13_Run2_v9/$ERA/MUO/root/efficiency_TopHNT.root results/$ERA/ROOT/efficiency_MuID.root
cp ~/Sync/workspace/SKNanoAnalyzer/data/Run3_v13_Run2_v9/$ERA/MUO/root/efficiency_Mu17Leg1.root results/$ERA/ROOT/efficiency_DLT_Mu17Leg.root
cp ~/Sync/workspace/SKNanoAnalyzer/data/Run3_v13_Run2_v9/$ERA/MUO/root/efficiency_Mu8Leg2.root results/$ERA/ROOT/efficiency_DLT_Mu8Leg.root
cp ~/Sync/workspace/SKNanoAnalyzer/data/Run3_v13_Run2_v9/2016preVFP/MUO/root/efficiency_Mu8El23_Mu8Leg.root results/$ERA/ROOT/Mu8El23_muon.root
cp ~/Sync/workspace/SKNanoAnalyzer/data/Run3_v13_Run2_v9/2016preVFP/MUO/root/efficiency_Mu23El12_Mu23Leg.root results/$ERA/ROOT/Mu23El12_muon.root

