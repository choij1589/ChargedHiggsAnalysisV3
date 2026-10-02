# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ChargedHiggsAnalysisV3 is a ROOT/Python framework for charged Higgs searches in CMS, merging Run2 and Run3 analyses. It covers lepton ID, trigger efficiency, fake rate measurements, and statistical limit extraction.

## Environment Setup

```bash
source setup.sh   # REQUIRED before any work
```

## Common/Tools/ — Shared Framework

- `HistoUtils.py`: Histogram utility helpers shared across modules
- `plotter.py`: `ComparisonCanvas` (data vs MC + ratio), `KinematicCanvas` (multi-sample); CMS style, luminosity labels
- `cpp/`: Compiled C++ helpers loaded by Python

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
| `SignalRegionStudyV4/` | **Current baseline** signal-region limit chain (simplified V3 port; parametric-signal & interpolation development) |
| `SignalRegionStudyV3/` | Full-featured unblind signal-region study (frozen reference for the V4 reproduction) |
| `SignalRegionStudyV2/` | Older signal region + limit extraction |
| `SignalRegionStudyV1/` | Legacy signal region module (deprecated) |
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

## Plotting Style

The look of every data/MC figure is fixed by the TriLepton and DiLepton private-work (PW)
plots. Only the label changes between modes. Captions, legends, colours and the layout do
not, so copy them from the reference implementation and don't re-derive them.

**Reference implementation:** `TriLepton/python/paper_plotting.py` (constants at the top)
and `DiLepton/python/plot.py --private-work`. **Reference outputs:**
`TriLepton/plots/PW/All/SR1E2Mu/pair_mass.pdf` (non-η layout) and
`TriLepton/plots/PW/All/SR1E2Mu/electrons_1_eta.pdf` (η layout).

### Label modes (one flag per script, mutually exclusive)

| Flag | Label | Output |
|------|-------|--------|
| `--paper` (default) | "CMS", no extra text, custom position `cmsPosX/Y = (0.20, 0.865)`, `cmsLabelSize = 0.070` | `plots/Paper/...` |
| `--preliminary` | "CMS" + "Preliminary" ("Simulation Preliminary" for MC-only) | `*_preliminary` / `plots/Preliminary/...` |
| `--private-work` | "Private work (CMS data)" (MC-only: "(CMS simulation)"), set with `CMS.SetCmsText(text, font=52, size=0.75*0.76)`, `extraText = ""`, placed by cmsstyle at iPos=11 (drop `cmsPos*`) | `plots/PW/{Run2,Run3,All}/{channel}/*.pdf` |
| `--wip` | "CMS" + "Work in progress" ("Simulation Work in progress" for MC-only) | `*_WIP` / `plots/WIP/...` |

- `CMS.SetCmsText` is global state. Reset it to `CMS.SetCmsText("CMS", font=61, size=0.75)` in non-PW modes, as `ParticleNetMD/python/plotPaperROCs.py` does with `configure_cms_style`.
- Thesis figures use PW. Every script spells the flag `--private-work`.
- iPos=11 placement is for the PW label only. The "CMS" mark in `--paper` keeps the custom `cmsPosX/Y` position.

### Shared layout (all modes; `ComparisonCanvas` config keys)

- **Header:**
  - Per-run luminosity from `LumiInfo`/`EnergyInfo`.
  - For Run2+Run3 (`--era All`), set `CoM = "<Run3> TeV"` and `run_label = "<L2> fb^{#minus1} (<E2> TeV) + <L3> fb^{#minus1}"`.
- **Ratio:**
  - `rTitle = "Obs. / Pred."`.
  - The uncertainty band is labelled `systSrc = "Stat.+Syst."` (hatch 3004, colour 12).
- **Data points:** `dataErrorX0 = True` for equal-width bins, with `maxDigits = 3`.
- **Legend (data + backgrounds):**
  - `legend = (0.650, 0.545, 0.970, 0.870)`, `legendTextSize = 0.036`, one column.
- **Signal legend:**
  - Box `(0.650, 0.270, 0.970, 0.515)`, text size 0.034, one column.
  - Header "(m_{H^{+}}, m_{A}) [GeV]". Line width 3, colours `#d81b60, #2ca02c, #000000, #0099b4`.
- **η plots only:**
  - The signal legend keeps its size and moves to the left of the background legend, bottoms aligned, shifted by +0.07 (`PRIVATE_WORK_ETA_SIGNAL_LEGEND`).
  - Headroom is ×1.4. Don't apply this to p_T or other plots.
- **Caption:**
  - Bold region over the lepton flavours (e.g. "SR" / "e#mu#mu", "DiLepton CR" / "#mu#mu"), `channelSize = 0.063`.
  - Position `(0.22, 0.75)` with the iPos=11 label; `(0.20, 0.665)` under the paper label.
- **Headroom:** `yHeadroom` is 1.6 for PW and 1.35 for the paper; control regions get ×1.2.
- **Backgrounds:**
  - The stack order (bottom→top) is fixed, and colours come from `PALETTE_LONG` by role:
    - [0] blue: dominant background (Nonprompt / Drell-Yan), on top
    - [1] orange: Diboson
    - [2] red: ttX / t#bar{t}
    - [3] grey: conversion / W
    - [4] purple: others / single t
  - Pass `config["colors"]` aligned with the surviving categories.
- **Axes:**
  - Units go in the x title, e.g. `p_{T} [GeV]`, `m(#mu^{+}#mu^{-}) [GeV]` (lowercase m), `p_{T}^{miss} [GeV]`.
  - Multiplicities use N_j 0–8 and N_b 0–5 with integer labels only: `SetNdivisions(n, False)` on every `hframe*` frame in both pads.
- **Format:** PDFs for PW, paper, preliminary and WIP; PNG only for the regular per-era validation plots.

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
