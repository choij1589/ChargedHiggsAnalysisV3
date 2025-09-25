#!/bin/bash
# Build and test the DataFormat library

# Source the setup script first
cd /home/choij/Sync/workspace/ChargedHiggsAnalysisV3
source setup.sh
cd ParticleNet

echo "Building test program..."
make test

if [ $? -eq 0 ]; then
    echo -e "\nRunning test program..."
    export LD_LIBRARY_PATH=$PWD/libs:$LD_LIBRARY_PATH
    ./test/test_dataformat
else
    echo "Build failed!"
    exit 1
fi
