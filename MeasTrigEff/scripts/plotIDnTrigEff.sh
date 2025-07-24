#!/bin/bash
ERA=$1

export PATH="$WORKDIR/MeasTrigEff/python":$PATH

plotMuIDEff.py --era $ERA
plotEleIDEff.py --era $ERA
plotDblMuLegEff.py --era $ERA --leg mu17
plotDblMuLegEff.py --era $ERA --leg mu8
plotEMuLegEff.py --era $ERA --hltpath Mu8El23 --object muon
plotEMuLegEff.py --era $ERA --hltpath Mu23El12 --object muon
plotEMuLegEff.py --era $ERA --hltpath Mu8El23 --object electron
plotEMuLegEff.py --era $ERA --hltpath Mu23El12 --object electron