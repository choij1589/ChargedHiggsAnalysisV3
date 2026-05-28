# SignalRegionStudyV3 Functionality Scope

This copy is intended to be self-contained. Code in this directory must not
import from, symlink to, or build runtime paths through another signal-region
study directory.

## Active In V3

- Environment setup, module docs, and full-unblind runbook.
- Masspoint/config/systematics/DAG configuration.
- Signal-region preprocessing.
- Binned template production and low-stat handling.
- Template validation.
- Datacard printing and channel/era combination.
- Asymptotic limit running, collection, and plotting.
- HybridNew running, merge/extract, and plotting.
- Goodness-of-Fit running and p-value plotting.
- Impacts, filtered impacts, and nuisance pull plots.
- FitDiagnostics.
- Prefit/postfit mass plots and full-mA postfit summary plots.
- ParticleNet score and threshold workflow.
- Signal injection and bias tests.
- TTZ control-region GoF workflow.
- Full-unblind artifact collection.

## Pending Deprecation Decision

Keep these copied from V2 for now, but do not treat them as core V3 workflows
until explicitly approved:

- Cut-and-count workflow.
- Template interpolation workflow.
- Binning scan and rate-ratio studies.
- Drop-Run3 comparison studies.
- Preserve-shape comparison helpers.
- Transfer/export helpers such as `rsync_templates.sh` and `copyDatacards.sh`.
- Historical generated results and plots.

## Binning Contract

In V3, `--binning extended` is the user-facing name for the adaptive coarser
binning used for the full-unblind workflow. Outputs should use `extended*`
suffixes, not the historical long-name suffixes from V2.
