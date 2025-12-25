#!/bin/bash
R_VALUE=$1
SEED=$2

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el8_amd64_gcc12
cd /home/choij/Sync/workspace/ChargedHiggsAnalysisV3/SignalRegionStudyV1/CMSSW_14_1_0_pre4/src
eval $(scramv1 runtime -sh)
cd ${_CONDOR_SCRATCH_DIR}

combine -M HybridNew workspace.root \
    --LHCmode LHC-limits \
    --singlePoint ${R_VALUE} \
    --saveToys \
    --saveHybridResult \
    -T 500 \
    -s ${SEED} \
    -n .r${R_VALUE}.seed${SEED}
