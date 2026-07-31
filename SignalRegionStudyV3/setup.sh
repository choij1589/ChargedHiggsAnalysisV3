#!/bin/bash
export WORKDIR=$PWD/..
cd $WORKDIR/Common/CMSSW_16_0_0/src
source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsenv
cd -

export PATH=$PWD/python:$PATH
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$WORKDIR/Common/Tools:$PYTHONPATH
