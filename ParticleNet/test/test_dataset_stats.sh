#!/bin/bash
# Test script to verify dataset statistics logging

# Source the setup script first
cd /home/choij/Sync/workspace/ChargedHiggsAnalysisV3
source setup.sh
cd ParticleNet

echo "Testing dataset statistics logging in pilot mode..."
python3 python/saveDataset.py --signal MHc-100_MA-95 --background ttZ --channel Run1E2Mu --pilot

if [ $? -eq 0 ]; then
    echo -e "\nChecking generated files..."
    echo "Dataset files:"
    ls -la dataset/Run1E2Mu__pilot__/MHc-100_MA-95_vs_ttZ_*
    
    echo -e "\nStatistics file:"
    if [ -f dataset/Run1E2Mu__pilot__/MHc-100_MA-95_vs_ttZ_stats.json ]; then
        echo "Found stats file in dataset directory"
        echo "First 20 lines of stats:"
        head -n 20 dataset/Run1E2Mu__pilot__/MHc-100_MA-95_vs_ttZ_stats.json
    fi
    
    echo -e "\nLog files:"
    ls -la logs/dataset_stats_Run1E2Mu_MHc-100_MA-95_vs_ttZ_*.json
else
    echo "Test failed. Please check the error messages above."
fi
