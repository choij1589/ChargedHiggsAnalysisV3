#!/bin/bash
R_VALUE=$1
SEED=$2

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd /u/user/choij/scratch/ChargedHiggsAnalysisV3/SignalRegionStudyV1/CMSSW_14_1_0_pre4/src
eval $(scramv1 runtime -sh)
cd ${_CONDOR_SCRATCH_DIR}

combine -M HybridNew workspace.root \
    --LHCmode LHC-limits \
    --singlePoint ${R_VALUE} \
    --saveToys \
    --saveHybridResult \
    --expectSignal 0 \
    -T 500 \
    -t -1 \
    -s ${SEED} \
    -n .r${R_VALUE}.seed${SEED}
