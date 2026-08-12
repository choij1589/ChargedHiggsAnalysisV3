"""Single source of truth for SignalRegionStudyV4 path construction.

Every python script in this module must construct sample/template/result
paths through these helpers. The module name appears as a literal exactly
once, here.

Path layout contract (V4, no binning_suffix level):
    {WORKDIR}/{MODULE_NAME}/samples/{era}/{channel}/{masspoint}/{process}.root
    {WORKDIR}/{MODULE_NAME}/templates/{masspoint}/{method}/{era}/{channel}/
    {WORKDIR}/{MODULE_NAME}/results/json/{mode}/{era}/limits....json

Unblind (real data) is the V4 default. A blinded (Asimov) run writes
"{method}_blind" as the method segment, so blind and unblind artifacts can
never collide. Filenames carry no extended/unblind tokens.

WORKDIR-based (not __file__-based) on purpose: the condor template wrapper
builds a scratch workdir in which python/ is a symlink back into the repo,
and resolving __file__ would escape the scratch sandbox.
"""
import json
import os

MODULE_NAME = "SignalRegionStudyV4"

RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]


def workdir():
    try:
        return os.environ["WORKDIR"]
    except KeyError:
        raise RuntimeError(
            "WORKDIR is not set. Source the module-local setup.sh first "
            f"(cd {MODULE_NAME} && source setup.sh)."
        )


def module_dir():
    return os.path.join(workdir(), MODULE_NAME)


def method_segment(method, blind=False):
    """Directory segment for a method. Blinded (Asimov) runs get a
    '_blind' suffix; unblind is the V4 default and carries no token."""
    return f"{method}_blind" if blind else method


def parse_ma_token(token):
    """Inverse of the mA p-notation: '90' -> 90 (int), '90p5' -> 90.5,
    '30p25' -> 30.25. Fractional grid points exist only on the template
    scan grid (configs/grid.json); real MC points are integers."""
    if "p" in token:
        whole, frac = token.split("p")
        return int(whole) + int(frac) / 10 ** len(frac)
    return int(token)


def masspoint_mhc_ma(masspoint):
    parts = masspoint.split("_")
    return (int(parts[0].replace("MHc", "")),
            parse_ma_token(parts[1].replace("MA", "")))


def pairing_variant(masspoint):
    """SR3Mu dimuon-pairing variant for a mass point.

    'highM' (higher-mass pairing) iff mHc >= 100 && mA >= 60 — the pairing
    rule, NOT a pure mA threshold (MHc160_MA15 is lowM, MHc70_MA60 is lowM).
    """
    mhc, ma = masspoint_mhc_ma(masspoint)
    return "highM" if (mhc >= 100 and ma >= 60) else "lowM"


def shared_channel_dirname(channel, masspoint=None, pairing=None):
    """Shared-sample directory name for a channel.

    SR3Mu needs the pairing variant (given explicitly or derived from the
    mass point); SR1E2Mu is mass-independent."""
    if channel == "SR3Mu":
        if pairing is None:
            if masspoint is None:
                raise ValueError("SR3Mu shared dir needs a pairing or a masspoint")
            pairing = pairing_variant(masspoint)
        return f"SR3Mu_{pairing}"
    return channel


def sample_dir(era, channel, masspoint, method):
    """Directory holding the preprocessed inputs for one (era, channel, mp).

    ParticleNet: per-masspoint dirs (per-masspoint score branches and
    MHc-specific input skims). Baseline: shared dirs —
    samples/{era}/SR1E2Mu and samples/{era}/SR3Mu_{lowM,highM} — holding
    the mass-independent backgrounds/data/nonprompt plus every signal as
    {masspoint}.root."""
    if method == "ParticleNet" or channel == "TTZ2E1Mu":
        return os.path.join(module_dir(), "samples", era, channel, masspoint)
    return os.path.join(module_dir(), "samples", era,
                        shared_channel_dirname(channel, masspoint=masspoint))


def template_dir(masspoint, method, era, channel, blind=False):
    return os.path.join(
        module_dir(), "templates", masspoint,
        method_segment(method, blind), era, channel
    )


def asymptotic_root(masspoint, method, era, channel, blind=False):
    return os.path.join(
        template_dir(masspoint, method, era, channel, blind),
        "combine_output", "asymptotic",
        f"higgsCombine.{masspoint}.{method_segment(method, blind)}."
        "AsymptoticLimits.mH120.root"
    )


def limits_json(era, channel, method, mode="BR", blind=False):
    """Collected-limits JSON path. Combined is the default channel and gets
    no channel infix."""
    channel_infix = "" if channel == "Combined" else f".{channel}"
    return os.path.join(
        module_dir(), "results", "json", mode, era,
        f"limits.{era}{channel_infix}.Asymptotic."
        f"{method_segment(method, blind)}.json"
    )


def config_path(name):
    return os.path.join(module_dir(), "configs", name)


def systematics_config(era):
    with open(config_path(f"systematics.{era}.json")) as f:
        return json.load(f)


def samplegroups_config():
    with open(config_path("samplegroups.json")) as f:
        return json.load(f)


def masspoints_config():
    with open(config_path("masspoints.json")) as f:
        return json.load(f)


def interpolation_config():
    with open(config_path("interpolation.json")) as f:
        return json.load(f)


def interpolation_uncertainties_path():
    return config_path("interpolation_uncertainties.json")


def grid_config():
    """configs/grid.json — the frozen template-scan mA grid per mHc
    (regenerate with python/makeInterpGrid.py). Per mHc: 'grid' (full
    scan list, steps below the dimuon mass resolution, MC points
    guaranteed members) and 'mc_points' (the baseline MC grid, where
    direct-MC vs fit-template comparison is possible)."""
    with open(config_path("grid.json")) as f:
        return json.load(f)


# Interpolation production layout (2026-08-13). Two trees, split by what
# the artifact IS, both git-tracked:
#   fits/                 fit-function artifacts (per-study models + their
#                         validation plots; global surface panels at
#                         fits/{params,yield})
#   closure/interpolation/  closure products (per-study closures, the 78
#                         LOO dirs under loo/, the pooled uncertainty
#                         diagnostic and the nuisance-rule plots)
# The old tests/interpolation tree is retired under archive/.

def interpolation_fits_dir(mhc=None):
    """fits/ (no arg) or fits/MHc{X}: dcb_fits{,_floating}.json,
    polynomials.json, yields/{yields,yield_model}.json, shape_deltas/,
    parts/ shards and plots/."""
    base = os.path.join(module_dir(), "fits")
    return os.path.join(base, f"MHc{int(mhc)}") if mhc is not None else base


def interpolation_closure_dir(mhc=None):
    """closure/interpolation/ (no arg) or .../MHc{X}: closure.json,
    yield_closure.json, loo_uncertainties.json, parts/ shards and plots/."""
    base = os.path.join(module_dir(), "closure", "interpolation")
    return os.path.join(base, f"MHc{int(mhc)}") if mhc is not None else base


def interpolation_fit_plots_dir(mhc, kind):
    """Per-study fit-validation plots, kind in {fits, params, yields,
    deltas}: fits/MHc{X}/plots/{kind}."""
    return os.path.join(interpolation_fits_dir(mhc), "plots", kind)


def interpolation_closure_plots_dir(mhc, kind):
    """Per-study closure plots, kind in {closure, yields}:
    closure/interpolation/MHc{X}/plots/{kind}."""
    return os.path.join(interpolation_closure_dir(mhc), "plots", kind)


def interpolation_global_plots_dir(kind):
    """Plots of objects that span every study. The surface panels live with
    the fit artifacts (kind "params" -> fits/params, "yield" ->
    fits/yield); the nuisance-rule summary with the closure products
    (kind "nuisance" -> closure/interpolation/plots/nuisance)."""
    if kind in ("params", "yield"):
        return os.path.join(interpolation_fits_dir(), kind)
    if kind == "nuisance":
        return os.path.join(interpolation_closure_dir(), "plots", kind)
    raise ValueError(f"unknown global plot kind: {kind}")


def interpolation_loo_dir(mhc, ma):
    """Per-point leave-one-out (LOO) output dir: models refit on the full
    mA grid minus this point, closure evaluated at this point only. The
    per-mHc aggregate lives in interpolation_closure_dir(mhc)."""
    return os.path.join(interpolation_closure_dir(), "loo",
                        f"MHc{int(mhc)}_MA{int(ma)}")
