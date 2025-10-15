#!/bin/bash
export ERA=$1                  # 2016preVFP 2016postVFP 2017 2018 FullRun2
export CHANNEL=$2              # SR1E2Mu SR3Mu Combined
export MASSPOINT=$3
export METHOD=$4

# WORKDIR should already be set by parent shell (doThis.sh sources setup.sh)
# If running standalone, you must source setup.sh before calling this script
if [ -z "$WORKDIR" ]; then
    echo "ERROR: WORKDIR not set. Please run 'source setup.sh' before calling this script."
    exit 1
fi

export BASEDIR=$WORKDIR/SignalRegionStudy/templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD

if [ $CHANNEL == "Combined" ]; then
    if [ -d $BASEDIR ]; then 
        rm -rf $BASEDIR; 
    fi
    mkdir -p $BASEDIR && cd $BASEDIR
    combineCards.py \
        ch1e2mu=$WORKDIR/SignalRegionStudy/templates/$ERA/SR1E2Mu/$MASSPOINT/Shape/$METHOD/datacard.txt \
        ch3mu=$WORKDIR/SignalRegionStudy/templates/$ERA/SR3Mu/$MASSPOINT/Shape/$METHOD/datacard.txt >> datacard.txt
    cd -
fi

if [ $ERA == "FullRun2" ]; then
    if [ -d $BASEDIR ]; then 
        rm -rf $BASEDIR; 
    fi
    mkdir -p $BASEDIR && cd $BASEDIR
    combineCards.py \
        era16a=$WORKDIR/SignalRegionStudy/templates/2016preVFP/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt \
        era16b=$WORKDIR/SignalRegionStudy/templates/2016postVFP/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt \
        era17=$WORKDIR/SignalRegionStudy/templates/2017/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt \
        era18=$WORKDIR/SignalRegionStudy/templates/2018/$CHANNEL/$MASSPOINT/Shape/$METHOD/datacard.txt >> datacard.txt
    cd -
fi

cd $BASEDIR

# Run the combine
text2workspace.py datacard.txt -o workspace.root
combine -M FitDiagnostics workspace.root
combine -M AsymptoticLimits workspace.root -t -1
combineTool.py -M Impacts -d workspace.root -m 125 --doInitialFit --robustFit 1 -t -1 --setParameterRanges r=-1,1
combineTool.py -M Impacts -d workspace.root -m 125 --robustFit 1 -t -1 --doFits --setParameterRanges r=-1,1
combineTool.py -M Impacts -d workspace.root -m 125 -o impacts.json
plotImpacts.py -i impacts.json -o impacts
cd $WORKDIR
