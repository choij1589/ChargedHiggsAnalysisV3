#!/bin/bash
ERA=$1

EGM_DIR="../SKNanoAnalyzer/data/Run3_v13_Run2_v9/${ERA}/EGM/root"
MUO_DIR="../SKNanoAnalyzer/data/Run3_v13_Run2_v9/${ERA}/MUO/root"

mkdir -p ${EGM_DIR}
cp results/${ERA}/ROOT/efficiency_EleID.root ${EGM_DIR}/efficiency_TopHNT.root
cp results/${ERA}/ROOT/Mu8El23_electron.root ${EGM_DIR}/efficiency_Mu8El23_El23Leg.root
cp results/${ERA}/ROOT/Mu23El12_electron.root ${EGM_DIR}/efficiency_Mu23El12_El12Leg.root

mkdir -p ${MUO_DIR}
cp results/${ERA}/ROOT/efficiency_MuID.root ${MUO_DIR}/efficiency_TopHNT.root
cp results/${ERA}/ROOT/efficiency_DLT_Mu17Leg.root ${MUO_DIR}/efficiency_Mu17Leg1.root
cp results/${ERA}/ROOT/efficiency_DLT_Mu8Leg.root ${MUO_DIR}/efficiency_Mu8Leg2.root
cp results/${ERA}/ROOT/Mu23El12_muon.root ${MUO_DIR}/efficiency_Mu23El12_Mu23Leg.root
cp results/${ERA}/ROOT/Mu8El23_muon.root ${MUO_DIR}/efficiency_Mu8El23_Mu8Leg.root
echo "Done"