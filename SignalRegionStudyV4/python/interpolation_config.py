"""Shared constants and helpers for the mA-interpolation chain (production
successor of the ``test/interpolation`` study; see docs/INTERPOLATION.md for
the method record and decision history).

Adopted configuration (frozen, no variant flags): SR1E2Mu = pure DCB;
SR3Mu_{lowM,highM} = fsig*DCB + (1-fsig)*Chebychev2, with per-category frozen
median nL/nR, uniform mA range, fixed x0/sigma orders, up-only F-test ladders
for alpha/c1/c2, and a logit-space polynomial for fsig (turnover-capable).

This module is a plain sibling of the other python/ libraries: scripts run as
``python3 python/<script>.py``, so Python already puts this directory on
sys.path — no bootstrap step is needed (unlike the archived study, which had
to insert python/ manually).
"""
import math
import os

import numpy as np

import srspaths

# Study channels: SR3Mu is split into its two dimuon-pairing variants and
# each variant is interpolated separately over the full mA range (the
# production pairing rule switches at mA=60; low-mHc points need every mass
# point constraining both variants — see docs/INTERPOLATION.md).
STUDY_CHANNELS = ["SR1E2Mu", "SR3Mu_lowM", "SR3Mu_highM"]


def channel_has_bkg(channel):
    """SR3Mu pairing picks a combinatoric dimuon in a mass-dependent
    fraction of events -> background component; SR1E2Mu has a single
    pairing and none."""
    return channel.startswith("SR3Mu")


# ---- Fit-model variant tests ---------------------------------------------
# Alternative background policies for the SR3Mu_lowM pathology: the adopted
# per-point drop/pin rule (FSIG_DROP_THRESHOLD) makes lowM a MIXED category
# (pinned fsig=1 anchors next to free fsig~0.9 points, c1/c2 with partial
# mA support) whose fsig/c1/c2 parametrizations oscillate and extrapolate
# (e.g. MHc145_MA35 lowM_Run3: predicted fsig 0.916 where the direct fit is
# a pure DCB, c1=-6). Each variant refits ONLY the categories it lists;
# every other category keeps the adopted model, so comparisons read the
# adopted tests/interpolation/MHc{X} outputs for those.
#
#  nodrop  — DCB+cheb2 with the drop/pin rule disabled: fsig fitted freely
#            at every point, c1/c2 measured everywhere (homogeneous
#            parametrization inputs).
#  puredcb — pure DCB (no background component at all), lowM continuum
#            absorbed by the DCB tails.
#
# Outputs land under tests/interpolation/variants/{name}/MHc{X}/
# (srspaths.interpolation_dir(mhc, variant=name)).
FIT_VARIANTS = {
    "nodrop": {"bkg": {"SR3Mu_lowM": "cheb2"}, "allow_drop": False},
    "puredcb": {"bkg": {"SR3Mu_lowM": None}, "allow_drop": True},
}


def variant_config(name):
    if name not in FIT_VARIANTS:
        raise ValueError(f"Unknown fit variant '{name}' "
                         f"(known: {sorted(FIT_VARIANTS)})")
    return FIT_VARIANTS[name]


def variant_channels(name):
    """Study channels a variant refits; all others keep the adopted model."""
    return sorted(variant_config(name)["bkg"])


# Parametrization forms per parameter. Single-entry order lists FIX the
# order (no F-test); multi-entry lists engage the F-test ladder in
# fitInterpPolynomials.select_order (higher order accepted at p < 0.05).
# fsig uses a logit-space polynomial (turnover-capable) with a linear-space
# polynomial fallback below FSIG_LOGISTIC_MIN_POINTS.
POLY_ORDERS = {
    "x0": [1],
    "sigmaL": [2],
    "sigmaR": [2],
    "alphaL": [2, 3],
    "nL": [1, 2],       # frozen in the adopted model; ladder unused
    "alphaR": [2, 3],
    "nR": [1, 2],       # frozen in the adopted model; ladder unused
    "fsig": [1, 2, 3],
    "c1": [2, 3],
    "c2": [2, 3],
}
F_TEST_PVALUE = 0.05
FSIG_LOGISTIC_MIN_POINTS = 5

# Smallest believable relative error on a fitted parameter. A not-fully
# converged Minuit pass (status=3) can leave a collapsed Hesse error -- values
# like 1.5e-13 on a Chebychev coefficient of 0.28 have been observed. Those are
# fit artifacts, not measurements, and since the parametrization is fit with
# weights 1/err a single such point carries weight ~1e13, swamping every real
# anchor and driving the weighted design matrix singular. Anchors whose
# relative error falls below this are skipped per-parameter (the same fit
# stays usable for its well-determined parameters).
MIN_REL_PARAM_ERROR = 1e-5

# Canonical parameter ordering for reports/plots (categories carry only the
# subset present in their fit model: SR1E2Mu has no fsig/c1/c2). nL/nR are
# frozen constants in the adopted model, but they must stay listed here:
# fitInterpPolynomials filters the parameters it records against this list,
# and build_model requires them.
ALL_PARAM_ORDER = ["x0", "sigmaL", "sigmaR", "alphaL", "nL",
                   "alphaR", "nR", "fsig", "c1", "c2"]

# Background-shape parameters (zero weight when fsig -> 1).
BKG_PARAMS = ("c1", "c2")

# fsig >= this  =>  the point carries no background: the fit refits as a
# pure DCB (dcb_fit_utils) and closure builds a pure DCB from the
# prediction. The dropped point still anchors fsig(mA) at 1.0.
FSIG_DROP_THRESHOLD = 0.995
FSIG_ANCHOR_ERROR = 0.002

# fsig fitted as a polynomial in logit space (bounded in (0,1) AND able to
# turn over — the true fsig rises past the naive-logistic plateau and falls
# again as mA -> mHc, where the two OS pairings converge and the combinatoric
# pair re-enters the window). Anchor points (fsig = 1, background dropped)
# are pinned at logit(1 - FSIG_LOGIT_CLIP) with a fixed logit-space error.
FSIG_LOGIT_ORDERS = [2, 3, 4, 5]
FSIG_LOGIT_CLIP = 1e-3
FSIG_LOGIT_ANCHOR_SIGMA = 0.5

# Informational low-statistics flag threshold (unweighted entries in the
# narrow fit window); recorded in dcb_fits.json, not a gate.
LOW_STAT_ENTRIES = 500

# ---- Yield-interpolation -------------------------------------------------
# Shapes are shared between the eras of a run period; yields are
# interpolated PER SUB-ERA, since the datacard's signal columns are per-era
# components whose rate is the nominal histogram integral. The yield
# definition is the production mass window integral, with the window
# computed from the INTERPOLATED shape parameters — see interp_window().
#
# N_win = k_era * G_period(mA) * f_category(mA):
#  - G(mA): shared per-period baseline-selection yield shape (log-space poly
#    on the period-summed totals; b-jet efficiency is flat in mA, the rest
#    is smooth mHc-mA kinematics with an asymmetric fall as the W* phase
#    space closes -> cubic minimum).
#  - f(mA): per-category window fraction. SR1E2Mu: near-constant peak
#    containment (pol0/1). SR3Mu: pure pairing combinatorics, derived from
#    the shape fit's fsig: exactly one OS pairing is the true A->mumu pair,
#    so p_low + p_high = 1 and f_variant = S(mA) * p_variant(mA) / fsig_variant(mA),
#    with S the shared containment and p_high fitted in logit space.
YIELD_F_SR1E2MU_ORDERS = [0, 1]
YIELD_S_ORDERS = [2, 3, 4]
YIELD_P_LOGIT_ORDERS = [2, 3, 4, 5]
YIELD_G_ORDERS = [3, 4]
YIELD_F_ABS_ERR_FLOOR = 0.005   # absolute floor on a merged window fraction

YIELD_ORDERS = {
    "f_sr1e2mu": YIELD_F_SR1E2MU_ORDERS,
    "S": YIELD_S_ORDERS,
    "p_high_logit": YIELD_P_LOGIT_ORDERS,
    "G": YIELD_G_ORDERS,
}

# ---- Yield-model variant tests -------------------------------------------
# The adopted yield model fits G(mA) independently per mHc study. The LOO
# review showed that every large production-pairing residual is a G
# failure (|dG| up to 23%, |df| <= 8%) and that NO alternative 1D basis
# fixes it (pol up to 6, log-mA, spline, pchip, linear all tie or lose):
# the per-mHc grids are simply too coarse at the steep low-mA turn-on
# (MHc100 has 8 points, MHc115 jumps 15 -> 27 -> 42), so dropping one
# point makes the cubic swing +-20% with alternating signs.
#
#  joint — three changes bundled, all aimed at that failure:
#    1. G is fitted as ONE surface in (mHc, mA) across all seven studies
#       and sliced at the study's mHc (the slice of a polynomial surface at
#       fixed mHc is a polynomial in mA, so the stored record keeps the
#       adopted logpoly+cov contract). 12 parameters per (period,
#       total-channel) instead of ~35 for seven independent cubics; the
#       dense MHc160 grid constrains the low-mA shape of the sparse
#       studies. LOO >10% failure rate 8.6% -> 3.5%, p90 at mA<=45
#       16.1% -> 7.7%.
#    2. The SR3Mu pairing decomposition drops the /fsig division:
#       S = f_low + f_high, p_high = f_high/S on the measured fractions.
#       Both forms are exact reparametrizations of (f_low, f_high); the
#       smoothness test is a wash (helps 0, hurts 1, mixed 13 of 14) and
#       the one loss is MHc145 Run3 — the origin of the +95% lowM
#       yield-closure blow-up. Dropping it also decouples the yield model
#       from the shape chain, which the puredcb shape model requires
#       (lowM then has no fsig at all).
#    3. k_era is an F-tested pol0/pol1 in mA (the share carries a real
#       trend: pol1 cuts its RMS by 1.3-2.2x, slope significances up to
#       9 sigma) and its quoted error is the SCATTER, not the standard
#       error of the mean — the adopted std/sqrt(N) understates the
#       single-point predictive error by sqrt(N) = 2.4-4.8.
#
# Outputs land under tests/interpolation/variants/{name}/MHc{X}[_MA{Y}]/;
# the shape chain is untouched, so shape polynomials are read from the
# adopted (or per-point LOO) tree.
YIELD_VARIANTS = {
    "joint": {"joint_G": True, "pairing_fsig": False, "k_era_orders": [0, 1]},
}

# Joint-surface basis: total-degree-truncated tensor polynomial in
# (u, v) = ((mHc - 115)/45, (mA - 70)/70), i <= JOINT_G_MHC_DEGREE,
# j <= JOINT_G_MA_DEGREE, i + j <= JOINT_G_MA_DEGREE.
JOINT_G_MHC_DEGREE = 2
JOINT_G_MA_DEGREE = 4
JOINT_G_MHC_SCALE = (115.0, 45.0)
JOINT_G_MA_SCALE = (70.0, 70.0)


def yield_variant_config(name):
    if name not in YIELD_VARIANTS:
        raise ValueError(f"Unknown yield variant '{name}' "
                         f"(known: {sorted(YIELD_VARIANTS)})")
    return YIELD_VARIANTS[name]


# Log-space error floor for N_total points, per run period: the observed
# per-sample normalization scatter (channel-correlated, tracked to the raw
# skims — upstream sample-production issue, largest in Run3).
REL_YIELD_ERR_FLOOR = {"Run2": 0.02, "Run3": 0.08}
FRACTION_LOGERR_FLOOR = 0.005   # log-space floor for f_window points

# Numerical guards when building a DCB(+Chebychev2) from interpolated
# parameters. c1/c2 are clipped to the direct-fit bounds (dcb_fit_utils
# fits them in [-1.5, 1.5]): an extrapolated background shape outside that
# range goes negative over part of the window and produces a lopsided
# pedestal after RooFit clips it to zero.
PARAM_FLOORS = {
    "sigmaL": 0.01, "sigmaR": 0.01,
    "alphaL": 0.05, "alphaR": 0.05,
    "fsig": 0.05,
    "c1": -1.5, "c2": -1.5,
}
PARAM_CEILINGS = {"fsig": 1.0, "c1": 1.5, "c2": 1.5}

# ---- Shape-systematic delta policy ---------------------------------------
# Every signal shape systematic is compressed into three dimensionless
# deltas measured on MC at the *other* mass points of the same mHc:
#   dm   = <m>_var/<m>_cen - 1     (core window, x0 +- DELTA_CORE_NSIGMA)
#   dsig = rms_var/rms_cen - 1     (core window)
#   dN   = sumw_var/sumw_cen - 1   (full +-10 sigma template window)
# applied as x0 -> x0(1+dm), sigmaL,R -> sigma(1+dsig), N -> N(1+dN).
DELTA_QUANTITIES = ("dm", "dsig", "dN")
DELTA_CORE_NSIGMA = 2.0

# Up-only ladder: the physics prior is "mA-independent relative shift", and
# the F-test may only upgrade it (same rule as the adopted shape ladders).
DELTA_ORDERS = [0, 1]

# Below this mA the pairing variant is mostly combinatoric, so its peak
# moments carry no information; measured anyway, dropped at fit time.
DELTA_MIN_MA = {"SR3Mu_highM": 60}
DELTA_MAX_MA = {"SR3Mu_lowM": 60}

# Error floors: MC-statistical errors on a paired difference can be
# absurdly small, which would let one point dominate the ladder fit.
DELTA_ERR_FLOOR = {"dm": 2e-4, "dsig": 2e-3, "dN": 1e-3}


def delta_ma_range(study_channel):
    return (DELTA_MIN_MA.get(study_channel, 0.0),
            DELTA_MAX_MA.get(study_channel, 1e9))


def delta_key(era, study_channel):
    return f"{era}|{study_channel}"


# ---- Interpolation-uncertainty derivation --------------------------------
# Derived from the held-out closure residuals themselves (exportInterpUncertainties.py),
# not fixed here — these are only the floors and the sample-scatter warn
# threshold applied on top of the per-mHc max-envelope.
UNCERTAINTY_SCALE_FLOOR = 0.02   # min x0 -> x0 +- floor * sigma_eff
UNCERTAINTY_RES_FLOOR = 0.02     # min sigmaL,R -> sigma * (1 +- floor)
UNCERTAINTY_NORM_FLOOR = 0.01    # min lnN - 1
UNCERTAINTY_NORM_WARN = 0.10     # warn (not fail) above this envelope


def period_token(period):
    """Nuisance-name token encoding correlation within a run period — the
    module's existing convention (see configs/systematics.*.json)."""
    return {"Run2": "13TeV", "Run3": "13p6TeV"}[period]


def interp_nuisance_names(prod_channel, period, era=None):
    """Scale/res nuisance names are correlated within a run period; norm is
    decorrelated between eras. Pass era=None for scale/res, an era string
    for norm."""
    tok = period_token(period)
    names = {
        "scale": f"CMS_interp_scale_{prod_channel}_{tok}",
        "res": f"CMS_interp_res_{prod_channel}_{tok}",
    }
    if era is not None:
        names["norm"] = f"CMS_interp_norm_{prod_channel}_{era}"
    return names


def production_channel(study_channel):
    """Study channel (SR1E2Mu, SR3Mu_lowM, SR3Mu_highM) -> datacard channel."""
    return "SR3Mu" if study_channel.startswith("SR3Mu") else study_channel


def study_channel_for(prod_channel, masspoint):
    """Datacard channel + mass point -> study channel, i.e. the SR3Mu
    dimuon-pairing variant production actually uses for this point."""
    if prod_channel != "SR3Mu":
        return prod_channel
    return f"SR3Mu_{srspaths.pairing_variant(masspoint)}"


def study_channels_for(masspoint):
    """The study channels a datacard for this mass point needs."""
    return [study_channel_for(ch, masspoint) for ch in ("SR1E2Mu", "SR3Mu")]


def study(mhc, loo_ma=None):
    """Fit/held-out/all mA split for one mHc: 'all' is the full baseline
    grid, 'fit' is the parametrization anchor set from
    configs/interpolation.json, 'held_out' = all - fit (the interpolation
    test set; points also appear in closure as in-sample checks when they
    are fit anchors).

    loo_ma engages the leave-one-out split instead: 'fit' = full grid
    minus that point, 'held_out' = [loo_ma] — the production-like closure
    used to derive the interpolation uncertainties."""
    mhc = int(mhc)
    prefix = f"MHc{mhc}_MA"
    grid = sorted(
        (int(mp[len(prefix):]) for mp in srspaths.masspoints_config()["baseline"]
         if mp.startswith(prefix))
    )
    if not grid:
        raise ValueError(f"No baseline mass points for mHc={mhc}")
    if loo_ma is not None:
        loo_ma = int(loo_ma)
        if loo_ma not in grid:
            raise ValueError(f"LOO mA={loo_ma} not in the mHc={mhc} baseline grid {grid}")
        return {"all": grid,
                "fit": [ma for ma in grid if ma != loo_ma],
                "held_out": [loo_ma]}
    fit_points = srspaths.interpolation_config()["fit_points"].get(str(mhc))
    if fit_points is None:
        raise ValueError(f"No fit_points defined for mHc={mhc} in configs/interpolation.json")
    fit_points = sorted(fit_points)
    missing = [ma for ma in fit_points if ma not in grid]
    if missing:
        raise ValueError(f"mHc={mhc} fit_points {missing} not in the baseline grid {grid}")
    held_out = [ma for ma in grid if ma not in fit_points]
    return {"all": grid, "fit": fit_points, "held_out": held_out}


def mhc_grid():
    """Every mHc that has baseline mass points — the studies the joint
    yield surface is fitted across."""
    grid = sorted({int(mp[3:mp.index("_MA")])
                   for mp in srspaths.masspoints_config()["baseline"]
                   if mp.startswith("MHc") and "_MA" in mp})
    if not grid:
        raise ValueError("No baseline mass points in configs/masspoints.json")
    return grid


def masspoint_name(mA, mhc):
    return f"MHc{int(mhc)}_MA{mA}"


def mA_of(masspoint):
    return srspaths.masspoint_mhc_ma(masspoint)[1]


def categories():
    """All (channel, period, suberas) categories of this study."""
    import run_period_utils
    out = []
    for channel in STUDY_CHANNELS:
        for period, suberas in run_period_utils.RUN_PERIODS.items():
            out.append((channel, period, list(suberas)))
    return out


def category_key(channel, period):
    import run_period_utils
    return run_period_utils.category_name(channel, period)


def period_of(era):
    """Run period (Run2/Run3) of a sub-era."""
    import run_period_utils
    for period, suberas in run_period_utils.RUN_PERIODS.items():
        if era in suberas:
            return period
    raise ValueError(f"Unknown era: {era}")


def signal_path(era, channel, masspoint):
    """Signal file path in the shared sample layout for a study channel."""
    if channel.startswith("SR3Mu_"):
        pairing = channel.split("_", 1)[1]
        dirname = srspaths.shared_channel_dirname("SR3Mu", pairing=pairing)
    else:
        dirname = srspaths.shared_channel_dirname(channel)
    return os.path.join(srspaths.module_dir(), "samples", era, dirname,
                        f"{masspoint}.root")


def build_signal_chain(suberas, channel, masspoint, ROOT):
    """TChain('Central') over the run period's suberas, or None (with the
    missing path) when a sample is absent."""
    chain = ROOT.TChain("Central")
    for era in suberas:
        path = signal_path(era, channel, masspoint)
        if not os.path.exists(path):
            return None, path
        chain.Add(path)
    return chain, None


def filter_csv(known, csv_arg, kind):
    """Validated comma-separated filter: returns the known list filtered to
    the requested entries, raising on unknown names."""
    if not csv_arg:
        return list(known)
    requested = {x.strip() for x in csv_arg.split(",") if x.strip()}
    unknown = requested - set(known)
    if unknown:
        raise ValueError(f"Unknown {kind}(s): {sorted(unknown)}")
    return [k for k in known if k in requested]


def known_missing_samples():
    return {tuple(entry) for entry in srspaths.interpolation_config()["known_missing_samples"]}


def fixed_n_values(mhc, variant=None):
    """Per-category fixed nL/nR: median of the good floating-n direct fits
    (reads the floating-n dcb_fits.json; a variant reads its own floating
    pass)."""
    import json
    from statistics import median
    path = os.path.join(srspaths.interpolation_dir(mhc, variant=variant),
                        "fits", "dcb_fits_floating.json")
    with open(path) as f:
        fits = json.load(f)["results"]
    out = {}
    for cat_key, cat_fits in fits.items():
        values = {"nL": [], "nR": []}
        for fit in cat_fits.values():
            if fit["quality"] != "good":
                continue
            for n in ("nL", "nR"):
                values[n].append(fit["params"][n]["value"])
        if not values["nL"]:
            raise RuntimeError(f"No good floating-n fits for {cat_key}; cannot derive fixed n")
        out[cat_key] = {n: float(median(v)) for n, v in values.items()}
    return out


def eval_param(info, x):
    """Evaluate a parametrization record from polynomials.json /
    yield_model.json at x (scalar or numpy array). Supports the polynomial
    form (coeffs, numpy convention), the logpoly form (polynomial in log
    space, yields), and the logitpoly form (bounded in (0,1), fsig/p_high)."""
    if info.get("form") == "logpoly":
        return np.exp(np.polyval(np.asarray(info["coeffs"]), x))
    if info.get("form") == "logitpoly":
        return 1.0 / (1.0 + np.exp(-np.polyval(np.asarray(info["coeffs"]), x)))
    return np.polyval(np.asarray(info["coeffs"]), x)


def load_shape_polynomials(mhc, suffix="", loo_ma=None, variant=None):
    """Per-category shape parametrizations (the yield/closure/export
    steps' window and template source). suffix selects an anchor-exclusion
    sibling file, e.g. '_ex90'; loo_ma selects the leave-one-out per-point
    directory instead (tests/interpolation/MHc{X}_MA{Y}/); variant selects
    a fit-model variant tree (FIT_VARIANTS)."""
    import json
    if loo_ma is not None and variant is not None:
        raise ValueError("loo_ma and variant are mutually exclusive")
    base = (srspaths.interpolation_loo_dir(mhc, loo_ma) if loo_ma is not None
            else srspaths.interpolation_dir(mhc, variant=variant))
    path = os.path.join(base, f"polynomials{suffix}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run the shape-interpolation chain first")
    with open(path) as f:
        return json.load(f)["polynomials"], path


def interp_window(cat_polys, mA):
    """Production mass window from the interpolated shape parameters:
    [max(x0 - 10*sigma_eff, 12), x0 + 10*sigma_eff] with
    sigma_eff = sqrt(0.5*(sigmaL^2 + sigmaR^2)) — the same construction as
    makeBinnedTemplates, but evaluable at any mA."""
    x0 = float(eval_param(cat_polys["x0"], mA))
    sL = float(eval_param(cat_polys["sigmaL"], mA))
    sR = float(eval_param(cat_polys["sigmaR"], mA))
    sigma_eff = math.sqrt(0.5 * (sL * sL + sR * sR))
    return max(x0 - 10.0 * sigma_eff, 12.0), x0 + 10.0 * sigma_eff
