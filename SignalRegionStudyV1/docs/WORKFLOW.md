# Directory Structure
SignalRegionStudyV1/
- CMSSW\_14\_1\_0\_pre4/src     # Combine
- datacards/                    # Main cards and root files                
- python/                       # Core analysis scripts
- scripts/                      # Shell automation (build, combine execution)
- configs/                      # samplegroups.json, systematics.json
- templates/                    # Output histograms & datacards
- samples/                      # Preprocessed ROOT files
- results/                      # Final limit JSON files
- doThis.sh                     # Main orchestration

# Workflow Pipeline (8 Steps)
| Step | Script                 | Purpose                                        |
|------|------------------------|------------------------------------------------|
| 1    | preprocess.py          | Raw data → preprocessed trees with systematics |
| 2    | makeBinnedTemplates.py | Create binned histograms, fit signal mass peak |
| 3    | checkTemplates.py      | Validate templates (non-zero, positive bins)   |
| 4    | printDatacard.py       | Generate HiggsCombine datacards                |
| 5    | combineCards.py        | Merge channels (SR1E2Mu + SR3Mu) and eras      |
| 6    | combineTool.py         | Run Combine fits + impact studies              |
| 7    | collectLimits.py       | Extract limits from Combine output             |
| 8    | plotLimits.py          | Create publication-quality limit plots         |

# Execution Flow
doThis.sh
  └─> runCombineWrapper.sh (per mass point)
      └─> prepareCombine.sh (steps 1-4)
      └─> runCombine.sh (steps 5-6)

## Key Parameters
- Eras: 2016preVFP, 2016postVFP, 2017, 2018, FullRun2
- Channels: SR1E2Mu, SR3Mu, Combined
- Methods: Baseline (26 mass points), ParticleNet (6 optimized points)
- Mass points: MHc 70-160 GeV, MA 15-155 GeV
