# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChargedHiggsAnalysisV3 is a physics analysis framework for charged Higgs particle searches in CMS experiments, merging Run2 and Run3 analyses. The project uses ROOT/Python for analyzing particle collision data, focusing on lepton identification, trigger efficiency measurements, fake rate studies, and systematic uncertainty evaluation.

## Environment Setup

```bash
source setup.sh
```

## Project Architecture

### Common/Tools/ - Shared Framework
Core utilities used across all analysis modules:

- `DataFormat.py`: Physics object hierarchy with Particle base class extending ROOT.TLorentzVector
  - `Lepton` class with lepton type classification (prompt, fromTau, conv, fromC, fromB, fake)
  - `Electron`/`Muon` subclasses with ID/trigger variables
  - Kinematic utility methods (mT calculation, delta phi, etc.)

- `plotter.py`: Standardized plotting framework
  - `ComparisonCanvas`: Data vs Monte Carlo with ratio panels
  - `KinematicCanvas`: Multi-sample kinematic distributions
  - CMS style formatting with proper luminosity labels

### Analysis Module Structure
Each analysis follows consistent organization:
```
ModuleName/
├── configs/*.json           # Histogram, systematics, sample configuration
├── python/                  # Analysis scripts using Common/Tools
├── scripts/                 # Shell scripts for batch processing with GNU parallel
├── plots/                   # Output plots directory
├── results/                 # Output ROOT/json files
└── doThis.sh                # Main execution script for the module
```

### Key Analysis Modules

**ExampleRun/**: Z mass validation analysis - simple reference implementation
**DiLepton/**: Full di-lepton analysis (DiMu/EMu channels) with systematic uncertainties
**TriLepton/**: Tri-lepton analysis, with CvSF and WZNjSF measurements
**LeptonIDTest/**: Lepton identification efficiency, fake rate and optimization studies
**MeasTrigEff/**: Trigger efficiency measurements using reference trigger method
**MeasFakeRate/**: Fake rate measurements using tight-to-loose method (Default for current fake rate)
**MeasFakeRateV2/**: Same as MeasFakeRate but not using absolute eta bins for Run3
**MeasFakeRateV3/**: Same as MeasFakeRate but using finer low ptCorr bins for muons and electrons -> Current baseline.
**MeasJetTagEff/**: Jet tagging efficiency measurements
**TriggerStrategy/**: Trigger strategy optimization studies
**SignalRegionStudy/**: Final limit extraction using Combine (C++/CMake) -> Current baseline.
**ParticleNet/**: ML pipeline for jet classification using ParticleNet architecture.

## Key Development Commands

### Environment and Setup
```bash
source setup.sh                    # REQUIRED before any analysis work
```

### Build / Lint / Test Commands

**Repo-wide sanity check** (no unified test runner):
```bash
python -m compileall -q .                              # Syntax check (fast)
python -c "import runpy; runpy.run_path('path/to/script.py')"  # Import check for specific script
```

**SignalRegionStudy (C++/ROOT library)**:
```bash
bash SignalRegionStudy/scripts/build.sh               # Build (wipes build/ and lib/)
python -c "import ROOT; ROOT.gSystem.Load('SignalRegionStudy/lib/libSignalRegionStudy.so')"  # Smoke test
```

**ParticleNet**:
```bash
ParticleNet/scripts/saveDatasets.sh --pilot           # Dataset creation (small test)
ParticleNet/scripts/trainMultiClass.sh                # Training (multiclass)
bash ParticleNet/test/test_saveDataset.sh             # Dataset prep test
python ParticleNet/test/test_shared_memory.py         # Unit-style test
```

**Analysis modules** (DiLepton/TriLepton/MeasFakeRate*/etc.):
```bash
cd <ModuleName> && bash doThis.sh                     # Module entrypoint
```

## Data Organization

### Input Data Structure
Analysis expects ROOT files in standardized paths:
```
$WORKDIR/SKNanoOutput/{ModuleName}/{Era}/{Sample}.root
```

### Systematic Variations
DiLepton analysis stores systematics as:
```
{Channel}/{SystematicName}/{HistogramPath}
```
Common systematics: L1Prefire, PileupReweight, MuonIDSF, ElectronIDSF, TrigSF, BtagSF

### Era Support
- **Run2**: 2016preVFP, 2016postVFP, 2017, 2018
- **Run3**: 2022, 2022EE, 2023, 2023BPix

Each era has specific data stream configurations and luminosity values.

### Centralized Configuration: Luminosity
**IMPORTANT**: All luminosity and center-of-mass energy values are stored in `Common/Data/Luminosity.json`.

```json
{
    "Run2": {"2016preVFP": 19.5, ..., "combined": 137.6, "energy_TeV": 13},
    "Run3": {"2022": 8.0, ..., "combined": 62.5, "energy_TeV": 13.6},
    "All": {"combined": 200.1}
}
```

**Usage in Python**:
```python
# Option 1: Import from plotter (recommended)
from plotter import LumiInfo, EnergyInfo, get_CoM_energy

# Option 2: Load JSON directly
import json
import os
_LUMI_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Common", "Data", "Luminosity.json")
with open(_LUMI_JSON_PATH, "r") as f:
    _LUMI_CONFIG = json.load(f)
```

**NEVER hardcode luminosity values** (e.g., `CMS.SetLumi(138)` or `"Run2": 138`). Always load from JSON.

## Code Architecture Patterns

### Analysis Script Structure
All Python analysis scripts follow common patterns:
1. Command line argument parsing (era, channel, histkey)
2. JSON configuration loading from `configs/histkeys.json`
3. ROOT file handling with proper memory management (`SetDirectory(0)`)
4. Histogram processing with bin-by-bin systematic uncertainties
5. Output to era/channel organized directories

### Shell Script Patterns
- Path setup: `export PATH="${PWD}/python:${PATH}"`
- Environment variable export for parallel processing
- GNU parallel for multi-threaded execution
- Era-specific logic for different run periods

## Development Notes

- All scripts assume ROOT environment and Common/Tools are available via setup.sh
- Use `ROOT.gROOT.SetBatch(True)` for non-interactive plotting
- Histogram memory management: always call `SetDirectory(0)` after reading
- Systematic uncertainties calculated using envelope method
- GNU parallel extensively used for multi-core processing
- JSON configuration files control histogram properties and systematic ranges
- When creating files with directory paths, always use `os.makedirs(os.path.dirname(file_path), exist_ok=True)` before file creation
- Do not make silent fallbacks. Raise error when the behaviour of the function call in not expected.

## Code Style and Conventions

### Python
- 4-space indentation, line length roughly <= 120
- Import order: (1) standard library, (2) third-party (ROOT, numpy, torch), (3) local repo modules
- Naming: `snake_case` for functions/variables, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants
- Validate required environment (`WORKDIR`, input paths) early and fail fast
- Use `ValueError` for invalid user inputs, `RuntimeError` for unexpected states
- When reading ROOT objects/histograms, handle "missing object" as an error unless explicitly allowed
- Close ROOT files promptly to avoid file descriptor/memory issues
- Do not hardcode sample lists, systematic names, era lists, or **luminosity values** if a config already exists (see `Common/Data/Luminosity.json`)

### Shell Scripts
- Start with `#!/bin/bash`
- For new scripts, prefer strict mode: `set -euo pipefail`
- Quote variables (`"$WORKDIR"`) and paths
- Use arrays for lists of eras/channels

### CMake/C++ (SignalRegionStudy)
- Target standard: C++17
- Keep CMake changes minimal; the build is environment-sensitive (ROOT/CMSSW)

## Version Control Safety

Before modifying any files, commit all uncommitted changes with a descriptive message so the user can roll back. Check `git status` first — if there are unstaged or staged changes, commit them in a separate commit before starting your work.

## Practical Workflow

1. `source setup.sh`
2. Identify the module impacted (ParticleNet vs analysis module vs SignalRegionStudy)
3. Make the smallest correct change
4. Run the narrowest validation:
   - Python: `python -m compileall -q <changed_dirs>`
   - ParticleNet: one relevant script under `ParticleNet/test/`
   - SignalRegionStudy: `bash SignalRegionStudy/scripts/build.sh` (only if needed)
