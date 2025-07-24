#!/bin/bash
ERA=$1

export PATH="$WORKDIR/MeasTrigEff/python":$PATH
measEMuLegEff.py --era $ERA --hltpath Mu8El23 --leg muon
measEMuLegEff.py --era $ERA --hltpath Mu23El12 --leg muon
measEMuLegEff.py --era $ERA --hltpath Mu8El23 --leg electron
measEMuLegEff.py --era $ERA --hltpath Mu23El12 --leg electron

if [[ $ERA == "2016preVFP" ]]; then
    echo "No pairwise filter efficiency measurement for $ERA."
elif [[ $ERA == "2016postVFP" ]]; then
    measPairwiseEff.py --era $ERA --filter EMuDZ
    measPairwiseEff.py --era $ERA --filter DblMuDZ
elif [[ $ERA == "2016postVFP" ]]; then
    measPairwiseEff.py --era $ERA --filter EMuDZ
    measPairwiseEff.py --era $ERA --filter DblMuDZ
    measPairwiseEff.py --era $ERA --filter DblMuDZM
    measPairwiseEff.py --era $ERA --filter DblMuM
else
    measPairwiseEff.py --era $ERA --filter EMuDZ
    measPairwiseEff.py --era $ERA --filter DblMuDZM
    measPairwiseEff.py --era $ERA --filter DblMuM
fi