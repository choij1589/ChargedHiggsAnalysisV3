#!/bin/bash
ERA=$1
EXTRA_ARGS=${2:-}   # optional, e.g. "--private-work"

export PATH="$WORKDIR/MeasTrigEff/python":$PATH

plotMuIDEff.py --era $ERA ${EXTRA_ARGS}
plotEleIDEff.py --era $ERA ${EXTRA_ARGS}
plotDblMuLegEff.py --era $ERA --leg mu17 ${EXTRA_ARGS}
plotDblMuLegEff.py --era $ERA --leg mu8 ${EXTRA_ARGS}
plotEMuLegEff.py --era $ERA --hltpath Mu8El23 --object muon ${EXTRA_ARGS}
plotEMuLegEff.py --era $ERA --hltpath Mu23El12 --object muon ${EXTRA_ARGS}
plotEMuLegEff.py --era $ERA --hltpath Mu8El23 --object electron ${EXTRA_ARGS}
plotEMuLegEff.py --era $ERA --hltpath Mu23El12 --object electron ${EXTRA_ARGS}