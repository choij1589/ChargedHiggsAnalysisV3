#!/bin/bash
export ERA=$1
export CHANNEL=$2
export MASSPOINT=$3             
export METHOD=$4
export PATH=$PWD/python:$PATH
export LD_LIBRARY_PATH=$WORKDIR/SignalRegionStudy/lib:$LD_LIBRARY_PATH

ERAs=("2016preVFP" "2016postVFP" "2017" "2018")
CHANNELs=("SR1E2Mu" "SR3Mu" "Combined")


# Sample and template directories use SR channel
SAMPLEDIR=$PWD/samples/$ERA/$CHANNEL/$MASSPOINT/$METHOD
BASEDIR=$PWD/templates/$ERA/$CHANNEL/$MASSPOINT/Shape/$METHOD

if [ -d $SAMPLEDIR ]; then
    echo "WARNING: Directory $SAMPLEDIR already exists"
    rm -rf $SAMPLEDIR
fi
if [ -d $BASEDIR ]; then
    echo "WARNING: Directory $BASEDIR already exists"
    rm -rf $BASEDIR
fi

# Preprocess with SR channel name (script handles internal conversion)
preprocess.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method $METHOD

# Template creation with SR channel name
if [ $METHOD == "Baseline" ]; then
    makeBinnedTemplates.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method $METHOD
else
    makeBinnedTemplates.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method $METHOD --update
fi

printDatacard.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method $METHOD
checkTemplates.py --era $ERA --channel $CHANNEL --masspoint $MASSPOINT --method $METHOD
