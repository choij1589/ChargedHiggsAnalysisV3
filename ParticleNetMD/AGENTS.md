# AGENTS.md

Guidance for coding agents working in `ParticleNetMD`.

## Project Scope

`ParticleNetMD` trains a mass-decorrelated ParticleNet graph neural network for the charged Higgs analysis. The model is a 4-class classifier:

- `signal`
- `nonprompt`
- `diboson`
- `ttX`

The defining feature is DisCo distance-correlation regularization against OS muon pair masses:

```text
L_total = L_CE + disco_lambda * (DisCo(score, mass1) + DisCo(score, mass2))
```

`mass1` and `mass2` are stored in each graph during dataset creation. `mass2 = -1` for `Run1E2Mu` events with only one OS pair and is ignored by the DisCo term.

## Environment

Set up the parent analysis environment before running scripts:

```bash
cd ..
source setup.sh
cd ParticleNetMD
```

Important environment variables:

- `WORKDIR` must point at the `ChargedHiggsAnalysisV3` workspace root.
- `SKNANO_DATA` is used for correctionlib fake-rate JSONs. If unset, `saveDataset.py` falls back to `../SKNanoAnalyzer/data/Run3_v13_Run2_v9` relative to `WORKDIR`.

Input ROOT files are expected under:

```text
$WORKDIR/SKNanoOutput/EvtTreeProducer/{channel}/{era}/{sample}.root
```

Supported dataset creation channels are `Run1E2Mu` and `Run3Mu`. The `Combined` channel is created dynamically at training time by merging those two channels.

## Current Workflow

Run augmentation prerequisites before creating full datasets:

```bash
python python/dibosonRankPromote.py
python python/nonpromptPromotion.py
```

Create datasets:

```bash
./scripts/saveDatasets.sh
```

`saveDatasets.sh` uses GNU parallel and honors `JOBS`:

```bash
JOBS=64 ./scripts/saveDatasets.sh
```

For a single sample:

```bash
python3 python/saveDataset.py \
  --sample TTToHcToWAToMuMu-MHc130_MA90 \
  --sample-type signal \
  --channel Run1E2Mu
```

Validate datasets in two phases:

```bash
python python/validateDatasets.py --workers 30
python python/validateDatasets.py --plotting
```

Train one model:

```bash
python python/trainMultiClass.py --signal MHc130_MA90 --channel Combined
```

Useful training flags:

- `--config configs/SglConfig-custom.json`
- `--disco-lambda 0.05`
- `--pilot`

Run the lambda sweep:

```bash
bash scripts/runLambdaSweep.sh
bash scripts/runLambdaSweep.sh --pilot
```

Compare lambda sweep results:

```bash
for sig in MHc130_MA90 MHc100_MA95 MHc160_MA85; do
  python python/compareDecorrelation.py --signal "${sig}" --channel Combined
done
```

Launch GA optimization:

```bash
./scripts/launchGAOptim.sh --signal MHc130_MA90 --channel Combined --device cuda:0
```

Multiple signal/channel pairs:

```bash
./scripts/launchGAOptim.sh \
  --config MHc130_MA90:Combined,MHc100_MA95:Combined \
  --device cuda:0,cuda:1
```

Resume GA from iteration `N`:

```bash
./scripts/launchGAOptim.sh --config MHc130_MA90:Combined --device cuda:0 --resume-from 2
```

Visualize GA iteration results:

```bash
./scripts/visualizeGAIteration.sh MHc130_MA90 Combined 3 cuda:0 --parallel --jobs 8
```

Summarize GA loss and select the best model:

```bash
python python/summarizeGALoss.py --signal MHc130_MA90 --channel Combined
```

Best GA model output:

```text
GAOptim/{channel}/{signal}/fold-4/best_model/model.pt
```

## Configuration

Primary configs:

- `configs/SglConfig.json`: single-model and lambda-sweep training defaults.
- `configs/GAConfig.json`: GA search space and GA training defaults.
- `configs/histkeys_validate.json`: observables for dataset validation.

Current `SglConfig.json` defaults include:

- `training_parameters.max_epochs`: `81`
- `training_parameters.batch_size`: `512`
- `training_parameters.dropout_p`: `0.4`
- `training_parameters.loss_type`: `disco`
- `training_parameters.train_folds`: `[0, 1, 2]`
- `training_parameters.valid_folds`: `[3]`
- `training_parameters.test_folds`: `[4]`
- `training_parameters.max_events_per_fold_per_class`: `50000`
- `training_parameters.augment_phi_rotation`: `true`
- `disco_parameters.disco_lambda`: `0.05`
- `model_config.nNodes`: `256`
- `optimization_config.optimizer`: `Adam`
- `optimization_config.initLR`: `0.0005`
- `optimization_config.weight_decay`: `4e-05`
- `optimization_config.scheduler`: `ExponentialLR`
- `output_config.results_dir`: `LambdaSweep`

Current `GAConfig.json` defaults include:

- `ga_parameters.population_size`: `16`
- `ga_parameters.max_iterations`: `4`
- `ga_parameters.fitness_metric`: `loss/valid`
- `ga_parameters.overfitting_penalty_weight`: `0.3`
- `disco_parameters.disco_lambda`: `0.1`
- `training_parameters.max_epochs`: `120`
- `training_parameters.batch_size`: `512`
- `training_parameters.dropout_p`: `0.4`
- `output_config.results_dir`: `GAOptim`

GA searches over:

- `nNodes`: `96`, `128`, `192`, `256`
- `optimizer`: `RMSprop`, `Adam`, `Adadelta`
- `scheduler`: `ExponentialLR`, `CyclicLR`, `ReduceLROnPlateau`
- `initLR`: log-uniform `[1e-4, 1e-2]`
- `weight_decay`: log-uniform `[1e-5, 1e-3]`

## Samples and Classes

Configured signal mass points in `scripts/saveDatasets.sh`:

- `TTToHcToWAToMuMu-MHc100_MA95`
- `TTToHcToWAToMuMu-MHc115_MA87`
- `TTToHcToWAToMuMu-MHc130_MA90`
- `TTToHcToWAToMuMu-MHc145_MA92`
- `TTToHcToWAToMuMu-MHc160_MA85`
- `TTToHcToWAToMuMu-MHc160_MA98`

Configured grouped backgrounds:

- `nonprompt`: 9 TTLL variants
- `diboson`: `WZTo3LNu`, `ZZTo4L_powheg`
- `ttX`: `TTZ`, `tZq`

Era-dependent sample names are normalized at dataset creation:

- WZ: Run2 `WZTo3LNu_amcatnlo`, Run3 `WZTo3LNu_powheg`, output `Skim_TriLep_WZTo3LNu`
- TTZ: Run2 `TTZToLLNuNu`, Run3 `TTZ_M50`, output `Skim_TriLep_TTZ`

Dataset files live under:

```text
dataset/samples/signals/{sample}/{channel}_fold-{fold}.pt
dataset/samples/backgrounds/{sample}/{channel}_fold-{fold}.pt
```

Before launching long training jobs, verify that the needed signal and background fold files exist. The checked-out local dataset directory may be partial.

## Core Code Map

Entry points:

- `python/saveDataset.py`: ROOT to PyTorch Geometric conversion.
- `python/trainMultiClass.py`: single-model training.
- `python/launchLambdaSweep.py`: multi-lambda training launcher.
- `python/visualizeMultiClass.py`: single-model plots.
- `python/compareDecorrelation.py`: cross-lambda comparison plots.
- `python/launchGAOptim.py`: GA driver.
- `python/evaluateGAModels.py`: GA model KS overfitting checks.
- `python/visualizeGAIteration.py`: GA iteration plots.
- `python/summarizeGALoss.py`: GA loss summary and best-model extraction.
- `python/validateDatasets.py`: validation histogram fill and plot phases.
- `python/dibosonRankPromote.py`: diboson promotion tables.
- `python/nonpromptPromotion.py`: nonprompt fake-rate validation.

Library modules:

- `python/lib/MultiClassModels.py`: 4-class ParticleNet model with 3 DynamicEdgeConv layers, `k=4`.
- `python/lib/WeightedLoss.py`: weighted CE and DisCo loss.
- `python/lib/TrainingOrchestrator.py`: training loop and decomposed loss logging.
- `python/lib/DataPipeline.py`: train/valid/test split loading.
- `python/lib/DynamicDatasetLoader.py`: `Combined` channel loading and grouped backgrounds.
- `python/lib/Preprocess.py`: graph construction, mass extraction, fake-rate handling, diboson promotion.
- `python/lib/SglConfig.py` and `python/lib/GAConfig.py`: config loaders.
- `python/lib/ResultPersistence.py`: model checkpoints, JSON histories, ROOT trees.
- `python/lib/ROCCurveCalculator.py`: ROC and metrics.

## Data Representation

Node features are 9-dimensional:

```text
[E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]
```

Graph-level era input is an 8-dimensional one-hot vector:

```text
[2016preVFP, 2016postVFP, 2017, 2018, 2022, 2022EE, 2023, 2023BPix]
```

Edges are built from `k=4` nearest neighbors in DeltaR, with DeltaR as the edge attribute.

## Output Locations

Common output directories:

- `DataAugment/diboson/plots/rank_promote/`: diboson conditional tables and validation plots.
- `DataAugment/nonprompt/plots/lnt_promote/`: nonprompt validation plots.
- `LambdaSweep/{channel}/{signal}/fold-4/`: lambda-sweep training outputs.
- `GAOptim/{channel}/{signal}/fold-4/GA-iter{N}/`: GA iteration outputs.
- `GAOptim/{channel}/{signal}/fold-4/best_model/`: best GA model and copied diagnostics.
- `logs/`: dataset stats, sweep logs, GA logs, validation logs.
- `TrainTestLR/`: train/test likelihood-ratio outputs.

Some scripts also create `results/`, `plots/`, or validation subdirectories as needed.

## Agent Working Notes

- Prefer the existing script and config interfaces over ad hoc launch commands.
- Keep `Combined` as a training-time channel only; do not pass it to `saveDataset.py`.
- Run `python python/dibosonRankPromote.py` before diboson dataset creation if `conditional_tables.json` is missing.
- Do not assume all configured datasets exist locally; inspect `dataset/samples/` first.
- Long jobs can be GPU-heavy. Use `--pilot` when available for smoke tests.
- `saveDatasets.sh` does not define a `--pilot` option.
- Preserve existing user changes and generated outputs unless explicitly asked to clean them.
- Use `rg` for code search.

## Reference Docs

- `docs/WORKFLOW.md`: end-to-end workflow.
- `docs/STEP1_PREPROCESSING.md`: data preparation, augmentation, validation.
- `docs/STEP2-DECORRELATION.md`: lambda/decorrelation workflow.
- `docs/STEP3_HYPERPARAM.md`: GA optimization details.
- `docs/BOARD.md`: project task/status notes.
- `DataAugment/*.md`: sample-specific augmentation and dataset notes.
