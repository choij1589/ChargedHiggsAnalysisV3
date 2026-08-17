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

# Collect the scan (source token keeps it apart from mc-signal) and plot.
# Both units are produced, as in V3: BR = B_sig, xsec = sigma(pp->ttbar) x
# B_sig in fb -> results/{json,plots}/{BR,xsec}/. The two differ by the
# constant sigma_ttbar(13 TeV), but each is collected from the Combine
# output so the JSONs can never drift apart.
for mode in BR xsec; do
    for ch in Combined SR1E2Mu SR3Mu; do
        python3 python/collectLimits.py --era All --channel $ch --method Baseline \
            --signal-source interp-signal --mode $mode
    done
    for mhc in 70 85 100 115 130 145 160; do
        python3 python/plotLimits.py --era All --method Baseline \
            --signal-source interp-signal --mode $mode --mhc $mhc
    done
    python3 python/plotLimits.py --era All --method Baseline \
        --signal-source interp-signal --mode $mode --compare-mhc
    # 2D map over the (m_H+, m_A) plane: one column per measured m_H+, never
    # interpolated between them (the model interpolates in m_A alone).
    for q in exp0 obs; do
        python3 python/plotLimits2D.py --era All --method Baseline \
            --signal-source interp-signal --mode $mode --quantity $q
        # ...and the same map with m_H+ interpolated too (Delaunay over the
        # scan points), which trades the staircase for the straight
        # kinematic edge m_A = m_H+ - 5. Rendering choice, not a prediction.
        python3 python/plotLimits2D.py --era All --method Baseline \
            --signal-source interp-signal --mode $mode --quantity $q \
            --interpolate-mhc
    done
done

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

# Per-seed outputs: combine_output/gof/{gof.json,gof_plot.png} and
# combine_output/impacts_obs/{impacts.pdf,impacts_filtered.pdf} under
# templates/{seed}/Baseline/interp-signal/All/{target}/.
#
# The per-mHc p-value summary is NOT drawn here: the Baseline panel also
# carries the ParticleNet seeds as open markers, so it needs the Step 5d GoF
# to exist and is drawn there. For a Baseline-only campaign, draw it now with
#   python3 python/plotGoFPValues.py --all --overlay ""

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

# ---------------------------------------------------------------------------
# Step 5: ParticleNet interpolation  (docs/interpolation/particlenet/METHOD.md)
# ---------------------------------------------------------------------------
# The ParticleNet arm of the mA interpolation: model frozen 2026-08-14
# (seeds mA=85/90/95, groups +-2.5 GeV, eps_B=20% WP, Baseline shapes
# reused, reach mA in [82.5, 97.5] at the five trained mHc). Everything
# below is method-aware plumbing over the Step 1-3 machinery.

# 5a. Per-mHc shared-scores samples, FULL systematics (template inputs):
#     120 jobs (5 mHc x 8 eras x {SR1E2Mu, SR3Mu, TTZ2E1Mu}) -> pnfs.
./automize/preprocess.sh --pnet-scores
# Anti-truncation gate (opens EVERY file; fails on CENTRAL_ONLY markers):
for mhc in 100 115 130 145 160; do
    python3 python/verifyInterpSamples.py --pnet --mhc $mhc
done

# 5b. Study chain: ONE 27-node DAG (thresholds -> shapes/yields ->
#     eps model -> export -> template closure -> summary). Tracked outputs:
#     fits/pnet/MHc{X}/{threshold_wp,eps_model}.json,
#     closure/pnet/MHc{X}/ + summary.txt,
#     configs/pnet_interpolation_uncertainties.json. Gate:
#   grep -h "EXITING WITH STATUS" condor/jobs_pnet_interp_<ts>/study/logs/*.out | sort | uniq -c
./automize/pnetInterpolation.sh --all

# 5c. Template scan over configs/pnet_grid.json (150 points / 15 groups;
#     322-node DAG per mHc, 304 for MHc100 whose reach is clipped at its
#     mA = 95 MC endpoint). One-group E2E first, then the campaign:
#./automize/interpTemplates.sh --mhc 115 --group MHc115_MA90 --method ParticleNet
./automize/interpTemplates.sh --all --method ParticleNet
# check parsed/total = 150 on every collect; then per-mHc limit plots
# (--ymax syncs the y-scale to the Baseline counterpart for method
# comparison). Both units again, and the per-mHc plots need the Baseline
# JSON of the same mode for the off-window regions:
for mode in BR xsec; do
    for ch in Combined SR1E2Mu SR3Mu; do
        python3 python/collectLimits.py --era All --channel $ch --method ParticleNet \
            --signal-source interp-signal --mode $mode
    done
    for mhc in 100 115 130 145 160; do
        python3 python/plotLimits.py --era All --method ParticleNet \
            --signal-source interp-signal --mode $mode --mhc $mhc
    done
    # 2D map with the ParticleNet arm stitched into its m_A window on the
    # five trained columns (m_H+ = 70, 85 stay Baseline), window edges
    # dashed in. Same fixed colour scale as the Baseline map above, so the
    # two can be laid side by side.
    for q in exp0 obs; do
        python3 python/plotLimits2D.py --era All --method ParticleNet \
            --signal-source interp-signal --mode $mode --quantity $q
        python3 python/plotLimits2D.py --era All --method ParticleNet \
            --signal-source interp-signal --mode $mode --quantity $q \
            --interpolate-mhc
    done
done

# 5d. GoF + impacts per group seed (66-node DAG per mHc; 15 seeds):
./automize/interpGofImpacts.sh --all --method ParticleNet
# Both arms' GoF now exist, so draw the per-mHc summaries deferred from
# Step 3. ONE panel per mHc carries both arms: Baseline filled, ParticleNet
# open markers (absent at mHc = 70, 85, which have no ParticleNet grid).
# `--method ParticleNet` or `--overlay ""` give single-arm panels ad hoc.
python3 python/plotGoFPValues.py --all

# 5e. ParticleNet score distributions at every interpolation seed
#     (V3 coverage: {Run2,Run3,All} x {SR1E2Mu,SR3Mu,Combined} per seed,
#     TTZ2E1Mu CR auto-emitted by the per-channel jobs; 27 nodes per mHc):
./automize/pnetScorePlots.sh --all
python3 python/collectPnetScorePlots.py   # LR_modified -> results/plots/scores/

# 5f. FitDiagnostics + prefit/postfit + pulls per GROUP SEED, both arms
#     (members share the seed's backgrounds bitwise, so seeds carry the
#     fit validation; All/Combined, V3 convention; 3 nodes per seed):
./automize/interpFitDiag.sh --all --method ParticleNet    # 15 seeds, 45 nodes
./automize/interpFitDiag.sh --all                         # Baseline: 572 seeds, 1716 nodes
# Stitched per-mHc summary panels -> results/plots/postfit_summary/.
# The summary stitches EVERY group seed of the study (Baseline: 64 at mHc70
# to 95 at mHc160; ParticleNet: those 3 on top), not just the seeds of the
# mc_points -- that earlier restriction left the panel mostly empty.
#
# Parallelized at the SEED level: building a seed's fine-mass hists out of
# its fitDiagnostics shapes (160 per seed/channel) is the whole cost and is
# per-seed independent, so the DAG fans that out and the summary node then
# stitches from cache in seconds. Serially it was hours per mHc.
#
# ORDER MATTERS: a ParticleNet panel stitches the Baseline seeds too and
# reads THEIR caches. Each arm warms only its own seeds, so Baseline must
# finish first -- otherwise the ParticleNet panel finds no Baseline cache.
./automize/postfitSummary.sh --all                      # 572 cache + 7 summary
./automize/postfitSummary.sh --all --method ParticleNet # then: 15 cache + 5 summary
# Variants:
#   --mhc 160            one study
#   --summary-only       caches already warm; skip straight to stitching
#   --dry-run
# Cache nodes ask 4 GB, summary nodes 32 GB (at the old 12-13-seed scope
# 4 GB already died with cgroup kills mid-SR3Mu, 2026-08-15).
# Single point on the login node (needs its caches warm):
#   python3 python/plotPostfitSummary.py --mhc 160 --methods Baseline \
#       --eras All --signal-source interp-signal --fit-type both
#
# Paper figures (Step 6) consume these caches, so run them after this.

# ---------------------------------------------------------------------------
# Step 6: paper figures  (docs/FUNCTIONALITY_SCOPE.md)
# ---------------------------------------------------------------------------
# Vector PDFs in the publication style, ported from V3 2026-08-18. Wording,
# colours and legend geometry are defined once in plotPaperLRModified.py and
# imported by the other two, so the figure sets cannot drift apart. All three
# default to --signal-source interp-signal. Output: results/plots/paper/.
#
# Light enough for the login node: the LR panels read the cached score
# histograms plotParticleNetScore.py leaves under templates/.../scores/, and
# the template panels read shapes.root + fitDiagnostics directly.

# 6a. ParticleNet LR_modified, SR and TTZ CR, at the three showcase points
#     (MHc160_MA85, MHc130_MA90, MHc100_MA95) -> paper/{SR,TTZCR}/:
python3 python/plotPaperLRModified.py --region all --masspoint all
# The legend is drawn inside each panel, in two columns, with the signal
# entry carrying the exact mass point. --standalone-legend instead publishes
# it once as its own panel (legend.pdf / legend_nosignal.pdf).

# 6b. Pre-fit / B-only / S+B mass templates, one panel per Run-period
#     category ({Run2,Run3} x {SR1E2Mu,SR3Mu}) -> paper/templates/{mp}/:
python3 python/plotPaperTemplates.py --masspoint MHc130_MA90 --method ParticleNet
# --masspoint all sweeps every point with an All/Combined fitdiag, which
# under interp-signal is all 572 Baseline seeds -- name the points you want.

# 6c. B-only postfit mA summary for mHc160/ParticleNet, split into the three
#     paper mA regions (<85, 85-95, >95) -> paper/Postfit/:
python3 python/plotPaperPostfitSummary.py
# Reads the fine-mass caches from Step 5f, so run that first. Legend is drawn
# inside each panel (top-right, two columns) as in 6a, which pushes the mA
# range and fit stage into the left-hand text block; --standalone-legend
# restores the shared legend panel.

# ---------------------------------------------------------------------------
# Step 7: template-point artifact bundle
# ---------------------------------------------------------------------------
# The per-mass-point fit diagnostics of the campaign live in 572 gitignored
# template dirs. This promotes the curated TEMPLATE POINTS -- one bundle per
# arm corner -- into the tracked tree, the same way collectPnetScorePlots.py
# promotes the LR panels:
#
#   Baseline:    MHc70_MA15, MHc100_MA60, MHc130_MA90, MHc160_MA155
#   ParticleNet: MHc160_MA85, MHc130_MA90, MHc100_MA95
#
# Per point: GoF (All x 3 channels), impacts full/filtered/summary, nuisance
# pulls full/filtered, the prefit / postfit_b / postfit_s / prefit-vs-postfit
# mass plots, and -- ParticleNet only -- the score panels including the
# TTZ2E1Mu CR. ~16 MB, 512 files -> results/templates/{method}/{masspoint}/.
python3 python/collectTemplatePlots.py
# --point METHOD:MASSPOINT (repeatable) collects a different set;
# --eras adds Run2/Run3 (only the score plots exist there).
# Exits 1 listing anything missing, so a partial campaign cannot pass silently.
