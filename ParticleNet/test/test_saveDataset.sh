#!/bin/bash
# Test script to verify dataset preparation with a single configuration

# Source the setup script first
cd /home/choij/Sync/workspace/ChargedHiggsAnalysisV3
source setup.sh
cd ParticleNet

export PATH=$PWD/python:$PATH

# Test with one signal, one background, one channel in pilot mode
echo "Testing dataset preparation in pilot mode..."
saveDataset.py --signal MHc-100_MA-95 --background ttZ --channel Run1E2Mu --pilot --debug

if [ $? -eq 0 ]; then
    echo "Test passed! Dataset preparation works correctly."
    echo "Checking output..."
    ls -la dataset/Run1E2Mu__pilot__/
else
    echo "Test failed. Please check the error messages above."
fi
