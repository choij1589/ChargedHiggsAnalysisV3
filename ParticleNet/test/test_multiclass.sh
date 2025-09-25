#!/bin/bash
# Test script to verify multi-class dataset preparation

# Source the setup script first
cd /home/choij/Sync/workspace/ChargedHiggsAnalysisV3
source setup.sh
cd ParticleNet

export PATH=$PWD/python:$PATH

# Test with one signal in multi-class mode
echo "Testing multi-class dataset preparation in pilot mode..."
saveDataset.py --signal MHc-100_MA-95 --channel Run1E2Mu --multiclass --pilot --debug

if [ $? -eq 0 ]; then
    echo "Test passed! Multi-class dataset preparation works correctly."
    echo "Checking output..."
    ls -la dataset/Run1E2Mu__multiclass__pilot__/
else
    echo "Test failed. Please check the error messages above."
fi
