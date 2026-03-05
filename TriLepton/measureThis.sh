#!/bin/bash
./scripts/measConvSF.sh
./scripts/parseCvSFToCommonData.sh

python python/measWZNjSF.py --era Run3 --channel WZCombined
cp results/WZCombined/Run3/WZNjetsSF.json ~/Sync/workspace/SKNanoAnalyzer/data/Run3_v13_Run2_v9/Run3/WZSF/
