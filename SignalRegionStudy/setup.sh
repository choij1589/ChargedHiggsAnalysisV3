#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export WORKDIR=$PWD/..
cd $WORKDIR/Common/CMSSW_14_1_0_pre4/src
cmsenv
cd -

export PATH=$PWD/python:$PATH
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$WORKDIR/Common/Tools:$PYTHONPATH
