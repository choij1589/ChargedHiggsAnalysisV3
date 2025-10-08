#!/usr/bin/env python
"""
Examples of using the flexible multi-class training system.

The system now supports any number of background classes, not just 3.
"""

import os

# Example 1: Standard 3-background training (signal + 3 backgrounds = 4 classes)
print("Example 1: Standard 3-background training")
cmd1 = """
python trainMultiClass.py \\
    --signal TTToHcToWAToMuMu-MHc130_MA100 \\
    --channel Run1E2Mu \\
    --fold 0 \\
    --backgrounds Skim_TriLep_TTLL_powheg Skim_TriLep_WZTo3LNu_amcatnlo Skim_TriLep_TTZToLLNuNu
"""
print(cmd1)

# Example 2: 4-background training with tZq (signal + 4 backgrounds = 5 classes)
print("\nExample 2: 4-background training including tZq")
cmd2 = """
python trainMultiClass.py \\
    --signal TTToHcToWAToMuMu-MHc130_MA100 \\
    --channel Run1E2Mu \\
    --fold 0 \\
    --backgrounds Skim_TriLep_TTLL_powheg Skim_TriLep_WZTo3LNu_amcatnlo Skim_TriLep_TTZToLLNuNu Skim_TriLep_tZq
"""
print(cmd2)

# Example 3: 2-background training for quick tests
print("\nExample 3: 2-background training (minimal)")
cmd3 = """
python trainMultiClass.py \\
    --signal TTToHcToWAToMuMu-MHc130_MA100 \\
    --channel Run1E2Mu \\
    --fold 0 \\
    --backgrounds Skim_TriLep_TTLL_powheg Skim_TriLep_TTZToLLNuNu \\
    --pilot
"""
print(cmd3)

# Example 4: Full training with all available backgrounds
print("\nExample 4: All backgrounds (7 classes total)")
cmd4 = """
python trainMultiClass.py \\
    --signal TTToHcToWAToMuMu-MHc130_MA100 \\
    --channel Run1E2Mu \\
    --fold 0 \\
    --backgrounds Skim_TriLep_TTLL_powheg Skim_TriLep_DYJets Skim_TriLep_DYJets10to50 \\
                  Skim_TriLep_WZTo3LNu_amcatnlo Skim_TriLep_TTZToLLNuNu \\
                  Skim_TriLep_TTWToLNu Skim_TriLep_tZq
"""
print(cmd4)

# Example 5: Using the shell script for batch training
print("\nExample 5: Batch training with shell script")
print("# Standard 3-background")
cmd5a = """
./scripts/trainMultiClass.sh \\
    --channel Run1E2Mu \\
    --backgrounds "Skim_TriLep_TTLL_powheg Skim_TriLep_WZTo3LNu_amcatnlo Skim_TriLep_TTZToLLNuNu"
"""
print(cmd5a)

print("\n# 4-background with tZq")
cmd5b = """
./scripts/trainMultiClass.sh \\
    --channel Run3Mu \\
    --backgrounds "Skim_TriLep_TTLL_powheg Skim_TriLep_WZTo3LNu_amcatnlo Skim_TriLep_TTZToLLNuNu Skim_TriLep_tZq" \\
    --model ParticleNetV2 \\
    --loss-type sample_normalized
"""
print(cmd5b)

# Background mapping reference
print("\n" + "="*60)
print("BACKGROUND SAMPLE MAPPING REFERENCE")
print("="*60)
print("""
Available background samples and their physics categories:
- Skim_TriLep_TTLL_powheg       -> nonprompt (non-prompt leptons)
- Skim_TriLep_DYJets            -> prompt (Drell-Yan)
- Skim_TriLep_DYJets10to50      -> prompt (low-mass Drell-Yan)
- Skim_TriLep_WZTo3LNu_amcatnlo -> diboson
- Skim_TriLep_TTZToLLNuNu       -> ttZ
- Skim_TriLep_TTWToLNu          -> ttW
- Skim_TriLep_tZq               -> rare_top

Common combinations:
- 3 backgrounds (standard): TTLL, WZ, TTZ
- 4 backgrounds (+tZq): TTLL, WZ, TTZ, tZq
- 5 backgrounds (+ttW): TTLL, WZ, TTZ, ttW, tZq
- All prompt merged: Use DYJets only (not DYJets10to50)
""")

# Model naming convention
print("\n" + "="*60)
print("MODEL NAMING CONVENTION")
print("="*60)
print("""
Models are named with background count indicator:
- ParticleNet-nNodes128-Adam-initLR0p0010-decay0p00010-StepLR-weighted_ce-3bg
- ParticleNet-nNodes128-Adam-initLR0p0010-decay0p00010-StepLR-weighted_ce-4bg
                                                                           ^^^
The suffix indicates number of background classes (3bg = 3 backgrounds, etc.)

Output paths remain the same:
results/multiclass/<CHANNEL>/<SIGNAL>/fold-<N>/
├── models/<model_name>-<N>bg.pt
├── CSV/<model_name>-<N>bg.csv
└── trees/<model_name>-<N>bg.root
""")

# Visualization updates
print("\n" + "="*60)
print("VISUALIZATION UPDATES")
print("="*60)
print("""
The visualization scripts now automatically detect the number of classes:
- Reads score branches from ROOT file dynamically
- Adjusts subplot layouts based on number of classes
- Generates appropriate class names from branch names

Usage remains the same:
./scripts/visualizeResults.sh <CHANNEL> <SIGNAL> <FOLD>

The plots will automatically adapt to show all classes present in the data.
""")

print("\n" + "="*60)
print("TIPS")
print("="*60)
print("""
1. Start with fewer backgrounds for debugging/testing
2. Use --pilot flag for quick tests with smaller datasets
3. Background order doesn't matter - they're assigned labels 1,2,3... in the order provided
4. Signal is always class 0
5. Visualization automatically adapts to any number of classes
6. Consider computational resources - more classes = larger models and longer training
""")
