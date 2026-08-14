#!/bin/bash
# SignalRegionStudyV4 runbook — the full production chain, one driver
# command per step (deep dives: docs/ and docs/interpolation/).
#
# Unblind is the default everywhere.
# Layout: templates/{masspoint}/{method}/{mc-signal,interp-signal}/{era}/{channel}
#         (interp-signal group members nest under their seed: .../points/{member}).

# ---------------------------------------------------------------------------
# Step 0: Preprocess  (docs/SAMPLES.md)
# ---------------------------------------------------------------------------
# Shared backgrounds are mass-independent and produced once (24 jobs);
# per-masspoint DAGs add the shared signals, writing BOTH SR3Mu pairing
# variants because the interpolation chain needs both. Needs a grid proxy
# (voms-proxy-init --voms=cms) - the wrapper xrdcp's its output to pnfs.
#
# --skip-particlenet emits shared backgrounds + shared signals only (what the
# Baseline chain and the interpolation study consume). Use it whenever the
# per-mHc NoHistMode skims the ParticleNet nodes read are incomplete
# upstream. 79 DAGs / 1272 jobs.
./automize/preprocess.sh --skip-particlenet
# Full production including ParticleNet per-masspoint dirs:
#./automize/preprocess.sh
# Later signal-only additions (shared backgrounds already on pnfs):
#./automize/preprocess.sh --masspoint MHc130_MA90 --skip-backgrounds

# Anti-truncation gate. NOT optional - concurrent xrdcp to pnfs has silently
# truncated samples before; a truncated file otherwise surfaces much later as
# a confusing fit failure. Exits 1 on any problem.
for mhc in 70 85 100 115 130 145 160; do
    python3 python/verifyInterpSamples.py --all --mhc $mhc
done

# BEFORE overwriting samples/ with a new production, preserve the old tree
# (rename on the same dCache filesystem - instant, no data copied):
#cd /pnfs/knu.ac.kr/data/cms/store/user/choij/SignalRegionStudyV4
#mv samples samples_backup_$(date +%Y%m%d)

# ---------------------------------------------------------------------------
# Step 1: mA interpolation chain  (docs/interpolation/WORKFLOW.md)
# ---------------------------------------------------------------------------
# Fits + surfaces + closures over the FULL 78-point baseline grid. Every mA
# dependence is ONE (mHc, mA) surface across all seven studies, so the chain
# runs in three passes with hard barriers - a single-mHc run cannot pass
# `polynomials` (deliberate FileNotFoundError). Each pass fully finished
# before the next is submitted; gate with
#   grep -h "EXITING WITH STATUS" condor/jobs_interp_<ts>/MHc*/dag.dag.dagman.out
./automize/interpolation.sh --all --stop-after fit-frozen
./automize/interpolation.sh --all --start-from polynomials --stop-after yields
./automize/interpolation.sh --all --start-from yield-model
# Leave-one-out sweep (the uncertainty source; may overlap pass 3):
./automize/interpolation.sh --loo --all
# Derived nuisance sizes -> configs/interpolation_uncertainties.json
# (ceil3-rounded; review order in docs/interpolation/WORKFLOW.md Gate C):
python3 python/exportInterpUncertainties.py --loo --all --pooled --write-config
# Global surface + nuisance-rule plots:
python3 python/plotInterpSurfaces.py --all
python3 python/plotInterpNuisances.py
# Recovery: --start-from/--stop-after a named step; merge interrupted shards
# with mergeInterpResults.py --stage; rescue DAGs with plain condor_submit_dag.

# ---------------------------------------------------------------------------
# Step 2: interp-signal template scan  (docs/interpolation/WORKFLOW.md,
#         "Template production")
# ---------------------------------------------------------------------------
# Parametric-signal templates + datacards + asymptotic limits at every
# configs/grid.json point (2467 over 7 mHc). Grouping is FROZEN in grid.json:
# each group's seed builds the shared backgrounds with its own mean/sigma
# (4 heavy jobs, 6 GB); members inject only their signal. Per point:
# All x {SR1E2Mu, SR3Mu, Combined} merge -> datacard -> asymptotic; one full
# validation per group at the seed.
./automize/interpTemplates.sh --all
# Single mHc or single group (verification / reruns):
#./automize/interpTemplates.sh --mhc 160
#./automize/interpTemplates.sh --mhc 160 --group MHc160_MA90

# Collect the scan (source token keeps it apart from mc-signal) and plot:
python3 python/collectLimits.py --era All --method Baseline --signal-source interp-signal
python3 python/collectLimits.py --era All --channel SR1E2Mu --method Baseline --signal-source interp-signal
python3 python/collectLimits.py --era All --channel SR3Mu   --method Baseline --signal-source interp-signal
for mhc in 70 85 100 115 130 145 160; do
    python3 python/plotLimits.py --era All --method Baseline --signal-source interp-signal --mhc $mhc
done
python3 python/plotLimits.py --era All --method Baseline --signal-source interp-signal --compare-mhc

# ---------------------------------------------------------------------------
# Step 3: GoF + impacts per group seed  (docs/interpolation/WORKFLOW.md)
# ---------------------------------------------------------------------------
# Mirrors V3: saturated GoF, background-only (r frozen at 0), real data +
# frequentist toys (500 toys / 5 batches); impacts via combineTool
# (--robustFit 1, r-range from the seed's own asymptotic, prop_bin filtered
# post-hoc). Per seed: GoF at All x {Combined, SR1E2Mu, SR3Mu}, impacts at
# All/Combined (V3 convention). Runs on the Step 2 datacards only.
./automize/interpGofImpacts.sh --all
# Variants:
#./automize/interpGofImpacts.sh --mhc 160 --group MHc160_MA90
#./automize/interpGofImpacts.sh --all --gof-only
#./automize/interpGofImpacts.sh --all --impacts-only

# p-value summaries per mHc (reads each seed's gof.json):
python3 python/plotGoFPValues.py --all
# Per-seed outputs: combine_output/gof/{gof.json,gof_plot.png} and
# combine_output/impacts_obs/{impacts.pdf,impacts_filtered.pdf} under
# templates/{seed}/Baseline/interp-signal/All/{target}/.

# ---------------------------------------------------------------------------
# Step 4: direct-MC chain and comparison  (docs/FUNCTIONALITY_SCOPE.md)
# ---------------------------------------------------------------------------
# Standard mc-signal chain at the MC points ({Run2,Run3,All} x
# {SR1E2Mu,SR3Mu,Combined} per mass point, FitDiagnostics/postfit/pulls on
# All/Combined). grid.json's mc_points mark where direct-MC vs interp-signal
# comparison is possible.
#./automize/makeBinnedTemplates.sh --mode all --method Baseline    --fitdiag --pull-fit both
#./automize/makeBinnedTemplates.sh --mode all --method ParticleNet --fitdiag --pull-fit both
# Single point (e.g. the verification anchor):
#./automize/makeBinnedTemplates.sh --masspoint MHc160_MA90 --method Baseline

# Collect + plot (mc-signal keeps the legacy filenames):
#for era in Run2 Run3 All; do
#    for ch in Combined SR1E2Mu SR3Mu; do
#        python3 python/collectLimits.py --era $era --channel $ch --method Baseline
#    done
#done
#python3 python/plotLimits.py --era All --method Baseline --mhc 160

# One-time reproduction validation against frozen V3 outputs
# (docs/REPRODUCTION.md; the samples stage runs on condor):
#python3 python/compareToV3.py --masspoint MHc130_MA90 \
#    --v3-dir ../SignalRegionStudyV3 --stage all
