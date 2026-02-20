# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChargedHiggsAnalysisV3 is a ROOT/Python framework for charged Higgs searches in CMS, merging Run2 and Run3 analyses. It covers lepton ID, trigger efficiency, fake rate measurements, and statistical limit extraction.

## Environment Setup

```bash
source setup.sh   # REQUIRED before any work
```

## Common/Tools/ — Shared Framework

- `DataFormat.py`: Particle/Lepton/Electron/Muon classes (extends ROOT.TLorentzVector); lepton type classification (prompt, fromTau, conv, fromC, fromB, fake)
- `plotter.py`: `ComparisonCanvas` (data vs MC + ratio), `KinematicCanvas` (multi-sample); CMS style, luminosity labels

## Key Analysis Modules

Each module has its own `CLAUDE.md` with commands and details.

| Module | Description |
|--------|-------------|
| `ExampleRun/` | Z mass validation — simple reference |
| `DiLepton/` | Di-lepton analysis (DiMu/EMu) with systematics |
| `TriLepton/` | Tri-lepton analysis; ConvSF and WZNjSF measurements |
| `LeptonIDTest/` | Lepton ID efficiency, fake rate, optimization studies |
| `MeasTrigEff/` | Trigger efficiency (tag-and-probe) |
| `MeasFakeRateV4/` | **Current baseline** fake rate (tight-to-loose) |
| `MeasFakeRate/`, `V2/`, `V3/` | Legacy fake rate modules |
| `MeasJetTagEff/` | b-tagging (DeepJet) efficiency |
| `TriggerStrategy/` | Trigger acceptance comparison study |
| `SignalRegionStudyV2/` | **Current baseline** signal region + limit extraction |
| `SignalRegionStudy/` | Legacy limit extraction (C++/CMake) |
| `ParticleNet/` | GNN classifier for signal/background discrimination |
| `ParticleNetMD/` | Mass-decorrelated variant (DisCo loss) |
| `GenKinematics/` | Generator-level kinematics plots |
| `SignalKinematics/` | Signal pair selection and discrimination studies |

## Data

**Input path pattern:**
```
$WORKDIR/SKNanoOutput/{ModuleName}/{Era}/{Sample}.root
```

**Eras — Run2:** 2016preVFP, 2016postVFP, 2017, 2018
**Eras — Run3:** 2022, 2022EE, 2023, 2023BPix

**Luminosity — NEVER hardcode.** Always load from `Common/Data/Luminosity.json`:
```python
from plotter import LumiInfo, EnergyInfo   # preferred
```

## Sanity Check

```bash
python -m compileall -q .   # syntax check (no unified test runner)
```

## Code Conventions

**Python:** 4-space indent; `snake_case` functions/vars, `PascalCase` classes, `UPPER_SNAKE` constants; fail fast on bad env/inputs (`ValueError`/`RuntimeError`); never silent fallbacks; `SetDirectory(0)` after reading ROOT histograms; close ROOT files promptly; no hardcoded lumi/era/sample lists.

**Shell:** `#!/bin/bash`; strict mode `set -euo pipefail` for new scripts; quote all variables and paths; `export PATH="${PWD}/python:${PATH}"` in module scripts.

**Directories:** `os.makedirs(os.path.dirname(path), exist_ok=True)` before writing files.

## Version Control Safety

Check `git status` before starting. If there are uncommitted changes, commit them first so the user can roll back.

## Workflow

1. `source setup.sh`
2. Find the module; read its `CLAUDE.md`
3. Make the smallest correct change
4. Validate: `python -m compileall -q <changed_dirs>`
