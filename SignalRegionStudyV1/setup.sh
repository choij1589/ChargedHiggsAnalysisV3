#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
cd CMSSW_14_1_0_pre4/src
cmsenv
cd -

export WORKDIR=$PWD/..
export PATH=$PWD/python:$PATH
export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$WORKDIR/Common/Tools:$PYTHONPATH
