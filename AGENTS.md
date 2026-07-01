# AGENTS.md

Guidance for coding agents working in `ChargedHiggsAnalysisV3`.

## Project Overview

`ChargedHiggsAnalysisV3` is a ROOT/Python framework for charged Higgs searches
in CMS, merging Run 2 and Run 3 analyses. It covers lepton identification,
trigger efficiency, fake-rate measurements, signal-region studies, ParticleNet
training, and statistical limit extraction.

## Environment Setup

Source the repository setup before running analysis code from the workspace
root:

```bash
source setup.sh
```

Some modules provide their own setup wrappers or stricter environment rules.
When a module contains its own `AGENTS.md`, follow that file for work inside the
module.

## Shared Framework

`Common/Tools/` contains shared analysis utilities:

- `HistoUtils.py`: histogram helpers shared across modules.
- `plotter.py`: `ComparisonCanvas` for data/MC ratio plots and
  `KinematicCanvas` for multi-sample plots, including CMS style and luminosity
  labels.
- `cpp/`: compiled C++ helpers loaded by Python.

## Key Analysis Modules

| Module | Description |
| --- | --- |
| `ExampleRun/` | Z mass validation and simple reference workflow. |
| `DiLepton/` | Dilepton analysis (`DiMu`, `EMu`) with systematics. |
| `TriLepton/` | Trilepton analysis, conversion scale factors, and WZ+jets scale-factor measurements. |
| `LeptonIDTest/` | Lepton ID efficiency, fake-rate, and optimization studies. |
| `MeasTrigEff/` | Trigger-efficiency tag-and-probe measurement. |
| `MeasFakeRateV4/` | Current baseline tight-to-loose fake-rate measurement. |
| `MeasFakeRate/`, `MeasFakeRateV2/`, `MeasFakeRateV3/` | Legacy fake-rate modules. |
| `MeasJetTagEff/` | DeepJet b-tagging efficiency measurement. |
| `TriggerStrategy/` | Trigger acceptance comparison studies. |
| `SignalRegionStudyV2/` | Current baseline signal-region and limit-extraction workflow. |
| `SignalRegionStudyV1/` | Legacy signal-region module; prefer `SignalRegionStudyV2` unless maintaining old results. |
| `SignalRegionStudyV3/` | In-progress full-unblind signal-region workflow. |
| `SignalRegionStudy/` | Legacy C++/CMake limit-extraction workflow. |
| `ParticleNet/` | GNN classifier for signal/background discrimination. |
| `ParticleNetMD/` | Mass-decorrelated ParticleNet variant using DisCo loss. |
| `GenKinematics/` | Generator-level kinematic plots. |
| `SignalKinematics/` | Signal pair-selection and discrimination studies. |

## Data Layout

Input ROOT files generally follow:

```text
$WORKDIR/SKNanoOutput/{ModuleName}/{Era}/{Sample}.root
```

Run 2 eras:

```text
2016preVFP, 2016postVFP, 2017, 2018
```

Run 3 eras:

```text
2022, 2022EE, 2023, 2023BPix
```

Never hardcode integrated luminosities. Load them from
`Common/Data/Luminosity.json`; the preferred access path is through
`LumiInfo`/`EnergyInfo` from `plotter.py` when available:

```python
from plotter import LumiInfo, EnergyInfo
```

## Validation

There is no single project-wide test runner. For Python changes, run a focused
syntax check on the changed package or directory:

```bash
python -m compileall -q <changed_dirs>
```

For broad syntax checks, use:

```bash
python -m compileall -q .
```

Many physics workflows require ROOT files, CMSSW tools, HiggsCombine, or batch
systems that may not be available in a generic shell. Prefer focused validation
that matches the touched module and report any unavailable external dependency.

## Python Conventions

- Use 4-space indentation.
- Use `snake_case` for functions and variables, `PascalCase` for classes, and
  `UPPER_SNAKE_CASE` for constants.
- Fail fast on invalid environment, input files, eras, channels, or samples with
  clear `ValueError`, `RuntimeError`, or `FileNotFoundError` exceptions.
- Do not add silent fallbacks for missing physics inputs.
- After reading ROOT histograms that must outlive their source file, call
  `SetDirectory(0)`.
- Close ROOT files promptly.
- Avoid hardcoded luminosity, era, or sample lists when a JSON config or shared
  helper already exists.
- Create output directories before writing files:

```python
os.makedirs(os.path.dirname(path), exist_ok=True)
```

## Shell Conventions

- Use `#!/bin/bash` for new Bash scripts.
- Use `set -euo pipefail` for new scripts unless a legacy calling pattern
  requires different behavior.
- Quote shell variables and paths.
- In module scripts that need local Python helpers, export the module `python`
  directory onto `PATH`:

```bash
export PATH="${PWD}/python:${PATH}"
```

## Workflow

1. Check `git status --short` before editing and preserve unrelated user
   changes.
2. Source the needed environment.
3. Identify the relevant module and read its local `AGENTS.md` if present.
4. Make the smallest correct change consistent with local patterns.
5. Validate with focused commands, usually `python -m compileall -q
   <changed_dirs>` for Python edits.
6. Report modified files, validation performed, and any dependency or data
   limitations.

