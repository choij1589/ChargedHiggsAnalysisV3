#!/bin/bash
export PATH="$PWD/python:$PATH"
ERA=$1

REGIONs=("InnerBarrel" "OuterBarrel" "Endcap")
HISTKEYs_EleID=("isMVANoIsoWP90" "convVeto" "lostHits" "sip3d" "mvaNoIso" "dz" "miniIso")
HISTKEYs_EleTrig=("sieie" "deltaEtaInSC" "deltaPhiInSeed" "hoe" "ecalPFClusterIso" "hcalPFClusterIso" "trackIso")
HISTKEYs_MuonID=("isPOGM" "sip3d" "dz" "miniIso")
HISTKEYs_MuonTrig=("trackIso")
PLOTVARs_Ele=("miniIso" "sip3d" "mvaNoIso")
PLOTVARs_Muon=("miniIso" "sip3d")

plot_idvar() {
    local era=$1
    local object=$2
    local region=$3
    local histkey=$4
    plot_idvar.py --era $era --object $object --region $region --histkey $histkey
}
plot_trigvar() {
    local era=$1
    local object=$2
    local histkey=$3
    plot_trigvar.py --era $era --object $object --histkey $histkey
}
plot_efficiency() {
    local era=$1
    local object=$2
    local region=$3
    local plotvar=$4
    plot_efficiency.py --era $era --object $object --region $region --plotvar $plotvar
}
plot_fakerate() {
    local era=$1
    local object=$2
    plot_fakerate.py --era $era --object $object
}

export -f plot_idvar
export -f plot_trigvar
export -f plot_efficiency
export -f plot_fakerate

parallel plot_idvar ::: $ERA ::: electron ::: ${REGIONs[@]} ::: ${HISTKEYs_EleID[@]}
parallel plot_idvar ::: $ERA ::: muon ::: ${REGIONs[@]} ::: ${HISTKEYs_MuonID[@]}
parallel plot_trigvar ::: $ERA ::: electron ::: ${HISTKEYs_EleTrig[@]}
parallel plot_trigvar ::: $ERA ::: muon ::: ${HISTKEYs_MuonTrig[@]}
parallel plot_efficiency ::: $ERA ::: electron ::: ${REGIONs[@]} ::: ${PLOTVARs_Ele[@]}
parallel plot_efficiency ::: $ERA ::: muon ::: ${REGIONs[@]} ::: ${PLOTVARs_Muon[@]}
parallel plot_fakerate ::: $ERA ::: electron
parallel plot_fakerate ::: $ERA ::: muon