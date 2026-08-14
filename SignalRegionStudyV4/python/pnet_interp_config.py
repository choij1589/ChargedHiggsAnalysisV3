"""ParticleNet mA-interpolation: frozen constants, grid/group helpers and
working-point machinery.

The production model was frozen 2026-08-14 and is recorded in
docs/interpolation/particlenet/{METHOD,UNCERTAINTY}.md. This module is the
single home of everything the chain's scripts previously duplicated
(test/pnet_interp study code): the run-period map, the study-channel map,
the anchor set, the uncertainty-rule constants, the WeightedScores
threshold machinery and the frozen-working-point lookup.

ROOT-free on purpose: makePnetGrid.py and the exporters must stay
login-node safe.
"""
import json
import os
from collections import OrderedDict

import numpy as np

import interpolation_config
import srspaths

# ----------------------------------------------------------------- frozen
# The production working point: fixed background efficiency per
# (channel, run period, seed). See METHOD.md "Score and working point".
DEFAULT_WP = "epsB=20%"

# Seeds are the trained mA per mHc; groups reach +-2.5 GeV on a 0.5 GeV
# lattice, so the arm covers mA in [82.5, 97.5] only.
SEED_MA = (85.0, 90.0, 95.0)
GROUP_HALF_WIDTH = 2.5
GRID_STEP = 0.5

# Anchors of the eps(mA) model: the mA trained in EVERY mHc. MA87/MA92
# exist only in MHc115/MHc145 and are blind validation points, never
# anchors.
ANCHOR_MA = (85, 90, 95)

# Production-relevant seed-member pairs: a grid point joins its nearest
# seed, so wider pairs never occur. Diagnostics over the unrestricted
# matrix are actively misleading (the seed's window clips a far member's
# peak -- METHOD.md "Group construction").
MAX_DELTA_MA = 2.5

# Uncertainty rule constants (UNCERTAINTY.md Gate U2, mirroring the
# Baseline arm): rms within study -> max across studies holding >=
# MIN_STUDY_POINTS points -> pooled-rms floor -> absolute floor.
MIN_STUDY_POINTS = 2
UNCERTAINTY_FLOORS = {"res": 0.01, "norm": 0.01}

PERIODS = OrderedDict([
    ("Run2", ["2016preVFP", "2016postVFP", "2017", "2018"]),
    ("Run3", ["2022", "2022EE", "2023", "2023BPix"]),
])
ENERGY = {"Run2": "13TeV", "Run3": "13p6TeV"}
CHANNELS = ("SR1E2Mu", "SR3Mu")

# PN channel -> Baseline study channel (polynomials/yield-model keys).
# Every trained point has mHc >= 100 and mA >= 70, so SR3Mu always
# resolves to the highM pairing. The SAMPLE dirs use the plain channel
# name (srspaths.mhc_sample_dir(era, 'SR3Mu', mhc)).
STUDY_CHANNEL = {"SR1E2Mu": "SR1E2Mu", "SR3Mu": "SR3Mu_highM"}


def mhc_int(mhc):
    """115 from 'MHc115' or 115."""
    return int(str(mhc).replace("MHc", ""))


def mA_of(masspoint):
    """Integer mA of a TRAINED masspoint name (grid points may carry
    p-notation; use interpolation_config.parse_ma for those)."""
    return int(masspoint.split("_MA")[1])


def pn_mhc_list():
    """The mHc with trained nets, as 'MHc{X}' strings ordered by mass,
    from configs/masspoints.json['particlenet']."""
    mps = srspaths.masspoints_config()["particlenet"]
    return [f"MHc{m}" for m in
            sorted({int(mp.split("_")[0].replace("MHc", "")) for mp in mps})]


def trained_masspoints(mhc):
    """Trained mass points of one mHc study, ordered by mA (json-only
    mirror of preprocess.pn_masspoints_for_mhc, importable without ROOT)."""
    prefix = srspaths._mhc_dirname(mhc)
    points = sorted((mp for mp in srspaths.masspoints_config()["particlenet"]
                     if mp.split("_")[0] == prefix), key=mA_of)
    if not points:
        raise ValueError(f"no ParticleNet-trained mass points for {prefix}")
    return points


# ------------------------------------------------------------ grid/groups
_PN_GROUP_SEED_CACHE = None


def pn_group_seed(masspoint):
    """Template-sharing group seed of a ParticleNet grid point
    (configs/pnet_grid.json). Returns the SEED masspoint name; a seed maps
    to itself. Raises for a mass point outside the ParticleNet reach."""
    global _PN_GROUP_SEED_CACHE
    if _PN_GROUP_SEED_CACHE is None:
        _PN_GROUP_SEED_CACHE = {}
        for key, entry in srspaths.pnet_grid_config()["grids"].items():
            mhc = mhc_int(key)
            for grp in entry["groups"]:
                seed_name = interpolation_config.masspoint_name(
                    grp["seed"], mhc)
                for mA in grp["members"]:
                    name = interpolation_config.masspoint_name(mA, mhc)
                    _PN_GROUP_SEED_CACHE[name] = seed_name
    if masspoint not in _PN_GROUP_SEED_CACHE:
        raise KeyError(
            f"{masspoint} is not on the ParticleNet scan grid "
            "(configs/pnet_grid.json; the reach is mA in [82.5, 97.5] at "
            "the trained mHc only)")
    return _PN_GROUP_SEED_CACHE[masspoint]


def pn_nuisance_name(family, prod_channel, period):
    """CMS_interp_{res,eff}_pnet_{ch}_{13TeV|13p6TeV}. Period-level names
    (one nuisance spans a period's era columns), production-channel token
    -- exactly the Baseline convention with a _pnet method qualifier."""
    if family not in ("res", "eff"):
        raise ValueError(f"unknown ParticleNet nuisance family: {family}")
    return f"CMS_interp_{family}_pnet_{prod_channel}_{ENERGY[period]}"


# -------------------------------------------------------- working points
class WeightedScores:
    """Sorted scores with suffix weight sums; O(log n) threshold queries.

    The naive form re-scanned multi-million-entry arrays 101 times per
    seed and dominated the study's runtime."""

    def __init__(self, scores, weights):
        self.n = len(scores)
        if self.n == 0:
            self.s = np.array([])
            self.tail = np.array([0.0])
            self.total = 0.0
            return
        order = np.argsort(scores, kind="mergesort")
        self.s = scores[order]
        w = weights[order]
        # tail[i] = sum(w[i:]), with tail[n] = 0
        self.tail = np.concatenate([np.cumsum(w[::-1])[::-1], [0.0]])
        self.total = float(self.tail[0])

    def sum_above(self, threshold):
        """Sum of weights with score > threshold (strict, as in
        production)."""
        if self.n == 0 or threshold != threshold:      # NaN guard
            return 0.0
        return float(self.tail[np.searchsorted(self.s, threshold,
                                               side="right")])

    def eff_above(self, threshold):
        if self.total == 0:
            return float("nan")
        return self.sum_above(threshold) / self.total

    def threshold_for_eff(self, target_eff):
        """Score cut giving the requested weighted efficiency.

        Walks the suffix sum down from the highest score and takes the
        first crossing. Weights are not all positive (matrix-method
        nonprompt, negative-weight MC), so the suffix sum is not
        guaranteed monotonic; callers are handed the efficiency ACHIEVED,
        never the requested one."""
        if self.n == 0 or self.total <= 0:
            return float("nan")
        target = target_eff * self.total
        below = np.nonzero(self.tail <= target)[0]
        i = int(below[0]) if len(below) else self.n
        return float(self.s[max(i - 1, 0)])


def threshold_wp_path(mhc):
    return os.path.join(srspaths.pnet_fits_dir(mhc), "threshold_wp.json")


def eps_model_path(mhc):
    return os.path.join(srspaths.pnet_fits_dir(mhc), "eps_model.json")


def wp_lookup(label=DEFAULT_WP, mhcs=None):
    """Frozen working point per category, from fits/pnet/MHc*/threshold_wp.json.

    Returns {"{mHc}/{channel}_{period}/seed{mA}": {threshold, eff_bkg,
    bg_weights, mass_window}}. Missing shard files are skipped -- an empty
    result must be treated as fatal by a --wp caller, never as licence to
    fall back to the sensitivity-optimized threshold (the three studies'
    residuals are only comparable on a byte-identical cut)."""
    out = {}
    for mhc in (mhcs or pn_mhc_list()):
        path = threshold_wp_path(mhc)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            payload = json.load(fh)
        for key, entry in payload.get("results", {}).items():
            match = next((w for w in entry["working_points"]
                          if w["label"] == label), None)
            if match is None:
                continue
            out[key] = {
                "threshold": float(match["threshold"]),
                "eff_bkg": float(match["eff_bkg"]),
                "bg_weights": entry["bg_weights"],
                "mass_window": entry["mass_window"],
            }
    return out


def wp_labels(mhcs=None):
    """Working-point labels present in the shards (for error messages)."""
    labels = []
    for mhc in (mhcs or pn_mhc_list()):
        path = threshold_wp_path(mhc)
        if not os.path.exists(path):
            continue
        with open(path) as fh:
            payload = json.load(fh)
        for entry in payload.get("results", {}).values():
            for w in entry["working_points"]:
                if w["label"] not in labels:
                    labels.append(w["label"])
    return labels


# ------------------------------------------------------------- eps model
def eval_eps(record, mA):
    """Threshold efficiency at mA from an eps_model.json record
    ({'anchors': {mA: eps}, 'coeffs': [...]}). The stored coefficients are
    the numpy-convention polynomial through the anchors (quadratic when
    all three exist)."""
    return float(np.polyval(np.asarray(record["coeffs"]), float(mA)))


def fit_eps_anchors(anchors):
    """Polynomial through {mA: eps} anchors: 3 -> quadratic, 2 -> linear,
    1 -> flat. Deliberately the simplest basis that can work; a failure
    should be visible rather than absorbed. Returns (coeffs, degree)."""
    xs = np.array(sorted(anchors), dtype=float)
    ys = np.array([anchors[x] for x in sorted(anchors)], dtype=float)
    if len(xs) == 1:
        return [float(ys[0])], 0
    deg = min(len(xs) - 1, 2)
    return [float(c) for c in np.polyfit(xs, ys, deg)], deg
