#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd /u/user/choij/scratch/ChargedHiggsAnalysisV3/SignalRegionStudyV1/CMSSW_14_1_0_pre4/src
eval $(scramv1 runtime -sh)
cd ${_CONDOR_SCRATCH_DIR}

hadd -f hybridnew_grid.root higgsCombine.*.root
