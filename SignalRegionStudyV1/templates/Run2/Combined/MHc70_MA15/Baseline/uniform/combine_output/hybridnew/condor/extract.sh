#!/bin/bash
QUANTILE=$1
NAME=$2

source /cvmfs/cms.cern.ch/cmsset_default.sh
export SCRAM_ARCH=el9_amd64_gcc12
cd /u/user/choij/scratch/ChargedHiggsAnalysisV3/SignalRegionStudyV1/CMSSW_14_1_0_pre4/src
eval $(scramv1 runtime -sh)
cd ${_CONDOR_SCRATCH_DIR}

if [[ "${QUANTILE}" != "observed" ]]; then
    # Expected limit
    combine -M HybridNew workspace.root \
        --LHCmode LHC-limits \
        --readHybridResults \
        --grid=hybridnew_grid.root \
        --expectedFromGrid ${QUANTILE} \
        -n .${NAME} \
        -m 120 \
        2>&1 | tee combine_${NAME}.out
else
    # Observed limit
    combine -M HybridNew workspace.root \
        --LHCmode LHC-limits \
        --readHybridResults \
        --grid=hybridnew_grid.root \
        -n .${NAME} \
        -m 120 \
        2>&1 | tee combine_${NAME}.out
fi
