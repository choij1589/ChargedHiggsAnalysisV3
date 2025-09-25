#!/bin/bash
# Test script to verify C++ compilation with ROOT environment

# Source the setup script first
cd /home/choij/Sync/workspace/ChargedHiggsAnalysisV3
source setup.sh
cd ParticleNet

echo "Testing C++ compilation with RVec..."
make clean
make

if [ $? -eq 0 ]; then
    echo "Compilation successful!"
    echo "Checking library..."
    ls -la libs/
else
    echo "Compilation failed. Please check the error messages above."
fi
