"""Shared constants and helpers for the mA-interpolation chain (production
successor of the ``test/interpolation`` study; see docs/interpolation/ for
the method record and decision history).

Adopted configuration (frozen): SR1E2Mu and SR3Mu_lowM = pure DCB;
SR3Mu_highM = fsig*DCB + (1-fsig)*Chebychev2, with per-category frozen median
nL/nR. Every shape parameter and every yield-model component is fitted as ONE
surface in (mHc, mA) across all seven studies and sliced at the study's mHc,
with fsig in logit space. Interpolation is in mA only.

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
# point constraining both variants — see docs/interpolation/EXPERIMENTS.md S4).
STUDY_CHANNELS = ["SR1E2Mu", "SR3Mu_lowM", "SR3Mu_highM"]


def channel_has_bkg(channel):
    """Only SR3Mu_highM carries a background component.

    Both SR3Mu pairings pick a combinatoric dimuon, but the two are very
    different in size: highM's wrong-pairing continuum runs 5-45% and is
    smooth in mA, while lowM's is a few percent and was previously modelled
    with a per-point drop/pin rule that made the category MIXED — pinned
    fsig=1 anchors next to free fsig~0.9 points, c1/c2 with partial mA
    support — whose parametrizations oscillated and extrapolated. Fitting
    lowM as a pure DCB and letting its small continuum sit in the tails
    took the worst production chi2/ndf from 154 to 24.5, with
    chi2_interp ~= chi2_direct everywhere (docs/interpolation/EXPERIMENTS.md S10).
    SR1E2Mu has a single pairing and no combinatoric background at all.
    """
    return channel == "SR3Mu_highM"


# Shape parametrization: every parameter is ONE surface in (mHc, mA) fitted
# across all seven mHc studies and sliced at the study's mHc. Interpolation
# is in mA only — the surface is a better-constrained model AT the measured
# mHc, not a licence to interpolate between them (leaving a whole study out
# and predicting it from the other six is a 4% median / 18% p90 error).
#
# Measured against 1D per-study polynomials on the same LOO protocol, the
# surface halves the worst-case scale error (0.344 -> 0.172 sigma_eff) and
# cuts p90 from 0.064 to 0.045. Two controls show the gain is the cross-mHc
# constraint and not extra freedom in mA: giving the 1D fit wider mA orders
# leaves the scale max at 0.344 and nearly doubles the res max, while
# flattening the mHc dependence out of the surface is also worse.
#
# (mHc degree, total degree), total-degree-truncated as in joint_design().
SHAPE_SURFACE_DEGREES = (2, 4)
F_TEST_PVALUE = 0.05

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

# Background-shape parameters (SR3Mu_highM only).
BKG_PARAMS = ("c1", "c2")

# fsig is fitted as a surface in LOGIT space: bounded in (0,1) and able to
# turn over, since the true fsig rises past the naive-logistic plateau and
# falls again as mA -> mHc, where the two OS pairings converge and the
# combinatoric pair re-enters the window.
FSIG_LOGIT_PARAMS = ("fsig",)

# Sanity threshold only. A highM fit returning fsig above this has no
# resolvable background, which never happened across all 156 highM fits of
# the seven studies; it is reported as a warning rather than silently
# refitting as a pure DCB (that drop/pin rule is what made lowM pathological
# and was removed with the puredcb adoption).
FSIG_DROP_THRESHOLD = 0.995

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
#  - G: per-period total yield, a log-space SURFACE in (mHc, mA) sliced at
#    the study's mHc (JOINT_G_DEGREES).
#  - k_era: era share, a plane in (mHc, mA) (JOINT_K_DEGREES), pooled across
#    studies; the shares are renormalized to sum to one over a period.
#  - f(mA): per-category window fraction, still a per-study 1D fit — it is
#    a containment fraction, close to flat and well measured everywhere.
#    SR1E2Mu: pol0/1. SR3Mu: pure pairing combinatorics, exactly one OS
#    pairing being the true A->mumu pair, so p_low + p_high = 1 and
#    f_variant = S(mA) * p_variant(mA), with S the shared containment and
#    p_high fitted in logit space. The decomposition is taken on the RAW
#    measured fractions: dividing by the shape fit's fsig is an exact
#    reparametrization that buys no smoothness (helps 0, hurts 1, mixed 13
#    of 14 datasets) and coupled the yield model to the shape chain, which
#    pure-DCB lowM cannot support.
YIELD_F_SR1E2MU_ORDERS = [0, 1]
YIELD_S_ORDERS = [2, 3, 4]
YIELD_P_LOGIT_ORDERS = [2, 3, 4, 5]
YIELD_F_ABS_ERR_FLOOR = 0.005   # absolute floor on a merged window fraction

YIELD_ORDERS = {
    "f_sr1e2mu": YIELD_F_SR1E2MU_ORDERS,
    "S": YIELD_S_ORDERS,
    "p_high_logit": YIELD_P_LOGIT_ORDERS,
}

# ---- Joint (mHc, mA) surfaces --------------------------------------------
# Both chains fit their mA dependence as ONE surface across all seven mHc
# studies, sliced at the study's mHc. The reason is the same in both cases:
# the per-mHc grids are too coarse at the steep low-mA turn-on (MHc100 has
# 8 points, MHc115 jumps 15 -> 27 -> 42), so a per-study fit swings by
# +-20% when one point moves, and NO alternative 1D basis fixes it (pol up
# to 6, log-mA, spline, pchip and linear all tie or lose). Borrowing the
# SHAPE across studies does: LOO failures above 10% fall 8.6% -> 3.5% for
# the yield totals, and the worst shape scale error halves.
#
# Basis: total-degree-truncated tensor polynomial in the scaled coordinates
# (u, v) = ((mHc - 115)/45, (mA - 70)/70), keeping i <= mhc_degree and
# i + j <= total_degree. Because the slice of a polynomial surface at fixed
# mHc is itself a polynomial in mA, every stored record keeps the plain
# coeffs+cov contract and nothing downstream needs to know a surface was
# involved (see slice_surface).
JOINT_MHC_SCALE = (115.0, 45.0)
JOINT_MA_SCALE = (70.0, 70.0)

# G: the per-period total-yield surface. k_era: the era-share plane — the
# total-degree truncation keeps 1, mHc and mA, three coefficients per
# (era, total-channel) in place of 28 constants. The shares carry a real
# smooth mHc drift the per-study constants were absorbing independently
# (2018/SR1E2Mu runs 0.4280 -> 0.4346 monotonically from mHc 70 to 160),
# and pooling averages the per-sample noise over 78 points instead of 6-23.
JOINT_G_DEGREES = (2, 4)
JOINT_K_DEGREES = (1, 1)


def joint_design(mhc, mA, degrees):
    """Total-degree-truncated tensor basis in scaled (mHc, mA).

    Returns (design matrix, [(i, j) powers]) — the powers are needed to
    slice the fitted surface at a fixed mHc."""
    dh, da = degrees
    mh0, mhs = JOINT_MHC_SCALE
    ma0, mas = JOINT_MA_SCALE
    u = (np.asarray(mhc, float) - mh0) / mhs
    v = (np.asarray(mA, float) - ma0) / mas
    cols, powers = [], []
    for i in range(dh + 1):
        for j in range(da + 1):
            if i + j > da:
                continue
            cols.append(u ** i * v ** j)
            powers.append((i, j))
    return np.vstack(cols).T, powers


def slice_surface(coeffs, cov, powers, mhc, degrees):
    """Collapse a (mHc, mA) surface at fixed mHc into a plain polynomial in
    mA (numpy descending convention) with a propagated covariance.

    This is what keeps the surfaces invisible downstream: eval_param,
    interp_window, closure and the template producer all keep working on
    the sliced record exactly as they did on a per-study 1D fit."""
    _, da = degrees
    ma0, mas = JOINT_MA_SCALE
    mh0, mhs = JOINT_MHC_SCALE
    u = (float(mhc) - mh0) / mhs
    base = np.array([1.0 / mas, -ma0 / mas])   # v as a polynomial in mA
    kmat = np.zeros((da + 1, len(powers)))
    for k, (i, j) in enumerate(powers):
        vj = np.array([1.0])
        for _ in range(j):
            vj = np.polymul(vj, base)
        kmat[da + 1 - len(vj):, k] = (u ** i) * vj
    beta = kmat @ np.asarray(coeffs)
    return beta, kmat @ np.asarray(cov) @ kmat.T


def fit_surface(mhc, mA, values, errors, degrees, slice_at):
    """Weighted least-squares surface fit + slice, as a record.

    Shared by the shape parametrizations and the yield model: same basis,
    same slicing, same bookkeeping."""
    amat, powers = joint_design(mhc, mA, degrees)
    w = 1.0 / np.asarray(errors, float)
    aw = amat * w[:, None]
    coeffs, *_ = np.linalg.lstsq(aw, np.asarray(values, float) * w, rcond=None)
    resid = amat @ coeffs - np.asarray(values, float)
    # UNSCALED covariance, i.e. the input errors are trusted — the same
    # convention the per-study weighted_polyfit used, so the sliced band
    # means the same thing it always did.
    cov = np.linalg.pinv(aw.T @ aw)
    beta, beta_cov = slice_surface(coeffs, cov, powers, slice_at, degrees)
    here = np.asarray(mhc, float) == float(slice_at)
    return {
        "coeffs": [float(c) for c in beta],
        "cov": [[float(c) for c in row] for row in beta_cov],
        "chosen_order": int(degrees[1]),
        "chi2": float((((resid * w) ** 2)[here]).sum()),
        "ndf": int(here.sum()),
        "joint_surface": {
            "mhc_degree": int(degrees[0]), "total_degree": int(degrees[1]),
            "n_points": int(len(values)), "n_params": int(len(coeffs)),
            "mhc_values": sorted({int(v) for v in np.asarray(mhc)}),
            "chi2_all": float(((resid * w) ** 2).sum()),
            "ndf_all": int(len(values) - len(coeffs)),
            "coeffs": [float(c) for c in coeffs],
            "powers": [[int(i), int(j)] for i, j in powers],
        },
    }


# Log-space error floor for N_total points, per run period: the observed
# per-sample normalization scatter (channel-correlated, tracked to the raw
# skims — upstream sample-production issue, largest in Run3).
REL_YIELD_ERR_FLOOR = {"Run2": 0.02, "Run3": 0.08}

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
# Sizes are derived from the leave-one-out closure residuals themselves
# (exportInterpUncertainties.py), by one rule for all three families: the
# rms WITHIN each mHc study, then the MAX across studies. These constants
# are only the safety floors and the warn threshold applied on top.
#
# The floors are meant to catch degenerate cells, NOT to set values. With
# the adopted model none of them is active: scale lands at 0.022-0.055,
# res at 0.012-0.028 and norm at 0.019-0.235. RES was lowered from 0.02,
# where it was setting four of the six res cells rather than catching
# anything (measured 0.0124/0.0148/0.0179/0.0191 all pushed up to 0.0200).
# A cell sitting exactly on a floor is a signal to revisit it.
UNCERTAINTY_SCALE_FLOOR = 0.02   # min x0 -> x0 +- floor * sigma_eff
UNCERTAINTY_RES_FLOOR = 0.01     # min sigmaL,R -> sigma * (1 +- floor)
UNCERTAINTY_NORM_FLOOR = 0.01    # min lnN - 1
UNCERTAINTY_NORM_WARN = 0.10     # warn (not fail) above this envelope


def period_token(period):
    """Nuisance-name token encoding correlation within a run period — the
    module's existing convention (see configs/systematics.*.json)."""
    return {"Run2": "13TeV", "Run3": "13p6TeV"}[period]


# The norm envelope is binned in mA and POOLED over mHc. Two reasons:
# with the joint (mHc, mA) yield surface the model is one global object,
# so its error is a property of the plane rather than of a study; and a
# per-study max is not an estimator once mA is binned (53% of split cells
# hold <= 2 points, and MHc70/MHc130 hold NONE in [60, 80) — MHc130's grid
# jumps 55 -> 83 and MHc70's only point there is a grid endpoint).
# Edges 15 / 80 / 100 / 155 split the grid into below-Z, on-Z and above-Z,
# with [80, 100) a tight window on the Z pole at 91.2 GeV. 155 is the
# largest mA in the baseline grid. The last bin is closed; an mA outside
# [15, 155] is an error rather than a silent extra bin.
NORM_MA_BINS = (("belowZ", 15.0, 80.0),
                ("onZ", 80.0, 100.0),
                ("aboveZ", 100.0, 155.0))


def norm_ma_bin(mA):
    """mA -> norm-uncertainty bin label."""
    for label, lo, hi in NORM_MA_BINS:
        if lo <= mA < hi or (hi == NORM_MA_BINS[-1][2] and mA == hi):
            return label
    raise ValueError(f"mA={mA} falls outside NORM_MA_BINS {NORM_MA_BINS}")


def interp_nuisance_names(prod_channel, period):
    """All three families are correlated across the eras of a run period and
    decorrelated between periods (the measured cross-era residual
    correlation is +0.99 Run2 / +0.80 Run3 while Run2 x Run3 is only
    partial — docs/interpolation/UNCERTAINTY.md). One nuisance name may
    still carry per-era lnN values in its datacard columns. Neither the era
    nor the norm mA bin appears in the name: the bin only selects the VALUE
    (norm_ma_bin), and one datacard holds one mass point, so only one bin
    per channel can ever occur in a workspace."""
    tok = period_token(period)
    return {
        "scale": f"CMS_interp_scale_{prod_channel}_{tok}",
        "res": f"CMS_interp_res_{prod_channel}_{tok}",
        "norm": f"CMS_interp_norm_{prod_channel}_{tok}",
    }


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
    """Fit/held-out/all mA split for one mHc.

    In production every model is fitted over the FULL baseline grid, so
    'fit' == 'all' and 'held_out' is empty. loo_ma engages the
    leave-one-out split instead: 'fit' = full grid minus that point,
    'held_out' = [loo_ma]. That split is the only closure the chain has,
    and it is what the interpolation uncertainties are derived from."""
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
    return {"all": grid, "fit": list(grid), "held_out": []}


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


def fixed_n_values(mhc):
    """Per-category fixed nL/nR: median of the good floating-n direct fits
    (reads the floating-n dcb_fits.json)."""
    import json
    from statistics import median
    path = os.path.join(srspaths.interpolation_dir(mhc),
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


def load_shape_polynomials(mhc, loo_ma=None):
    """Per-category shape parametrizations (the yield/closure/export steps'
    window and template source). loo_ma selects the leave-one-out per-point
    directory instead (tests/interpolation/MHc{X}_MA{Y}/)."""
    import json
    base = (srspaths.interpolation_loo_dir(mhc, loo_ma) if loo_ma is not None
            else srspaths.interpolation_dir(mhc))
    path = os.path.join(base, "polynomials.json")
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
