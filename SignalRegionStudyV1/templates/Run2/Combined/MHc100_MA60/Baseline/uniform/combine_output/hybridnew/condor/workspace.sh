#!/bin/bash
source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el8_amd64_gcc12
cd /home/choij/Sync/workspace/ChargedHiggsAnalysisV3/SignalRegionStudyV1/CMSSW_14_1_0_pre4/src
eval $(scramv1 runtime -sh)
cd ${_CONDOR_SCRATCH_DIR}
text2workspace.py datacard.txt -o workspace.root
