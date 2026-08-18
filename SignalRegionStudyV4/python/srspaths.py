"""Single source of truth for SignalRegionStudyV4 path construction.

Every python script in this module must construct sample/template/result
paths through these helpers. The module name appears as a literal exactly
once, here.

Path layout contract (V4, no binning_suffix level):
    {WORKDIR}/{MODULE_NAME}/samples/{era}/{channel}/{masspoint}/{process}.root
    {WORKDIR}/{MODULE_NAME}/templates/{masspoint}/{method}/{source}/{era}/{channel}/
    {WORKDIR}/{MODULE_NAME}/results/json/{mode}/{era}/limits....json

source is 'mc-signal' (direct-MC templates; the only source for
ParticleNet) or 'interp-signal' (parametric signal from the
mA-interpolation surfaces, Baseline only). interp-signal group members
nest under their group seed's dir:
    .../templates/{seed}/Baseline/interp-signal/{era}/{channel}/points/{member}/

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


def mhc_sample_dir(era, channel, mhc):
    """Per-mHc ParticleNet sample dir (preprocess.py --shared-scores).

    Holds every trained mass point of one mHc plus a single shared copy of
    the backgrounds/nonprompt/data, all keeping the score branches of every
    trained mA — the layout the ParticleNet interpolation study needs, since
    a member point is scored by its SEED's net. Distinct from sample_dir()'s
    per-masspoint dirs by the directory name alone ('MHc115' vs
    'MHc115_MA85')."""
    return os.path.join(module_dir(), "samples", era, channel, mhc)


SIGNAL_SOURCES = ("mc-signal", "interp-signal")


def _check_source(source):
    if source not in SIGNAL_SOURCES:
        raise ValueError(f"unknown signal source {source!r} "
                         f"(expected one of {SIGNAL_SOURCES})")
    return source


def template_dir(masspoint, method, era, channel, blind=False,
                 source="mc-signal"):
    """templates/{mp}/{method_segment}/{source}/{era}/{channel}.

    source: 'mc-signal' (direct-MC signal templates, the default and the
    only source for ParticleNet) or 'interp-signal' (parametric signal
    from the mA-interpolation surfaces; Baseline only). For an
    interp-signal GROUP MEMBER, the masspoint here is the group SEED —
    member outputs nest under it via interp_member_dir."""
    return os.path.join(
        module_dir(), "templates", masspoint,
        method_segment(method, blind), _check_source(source), era, channel
    )


def interp_member_dir(seed_masspoint, member_masspoint, era, channel,
                      blind=False, method="Baseline"):
    """Template dir of an interp-signal group member: nested under the
    seed's dir, which holds the group's shared background templates.
    The seed itself lives directly in template_dir (source-level).
    method selects the arm (Baseline grid.json / ParticleNet
    pnet_grid.json groups)."""
    return os.path.join(
        template_dir(seed_masspoint, method, era, channel, blind,
                     source="interp-signal"),
        "points", member_masspoint)


def asymptotic_root(masspoint, method, era, channel, blind=False,
                    source="mc-signal", seed_masspoint=None):
    """The AsymptoticLimits output ROOT file. Filenames are unchanged
    from the 4-segment era (mc-signal artifacts stay byte-identical);
    the source only moves the directory. For an interp-signal member,
    pass its seed so the path nests correctly."""
    if seed_masspoint is not None and seed_masspoint != masspoint:
        base = interp_member_dir(seed_masspoint, masspoint, era, channel,
                                 blind, method=method)
    else:
        base = template_dir(masspoint, method, era, channel, blind, source)
    return os.path.join(
        base, "combine_output", "asymptotic",
        f"higgsCombine.{masspoint}.{method_segment(method, blind)}."
        "AsymptoticLimits.mH120.root"
    )


def limits_json(era, channel, method, mode="BR", blind=False,
                source="mc-signal"):
    """Collected-limits JSON path. Combined is the default channel and gets
    no channel infix. mc-signal keeps the legacy filename (existing
    results and the V3 comparator stay valid); interp-signal gets a
    source token so the two scans never collide."""
    channel_infix = "" if channel == "Combined" else f".{channel}"
    source_infix = "" if _check_source(source) == "mc-signal" \
        else f".{source}"
    return os.path.join(
        module_dir(), "results", "json", mode, era,
        f"limits.{era}{channel_infix}.Asymptotic."
        f"{method_segment(method, blind)}{source_infix}.json"
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


def _mhc_dirname(mhc):
    """'MHc115' from 115, '115' or 'MHc115' — the pnet helpers accept the
    string form because the study chain keys everything by 'MHc{X}'."""
    text = str(mhc)
    return text if text.startswith("MHc") else f"MHc{int(mhc)}"


def pnet_grid_config():
    """configs/pnet_grid.json — the frozen ParticleNet-interpolation scan
    grid (regenerate with python/makePnetGrid.py). Per mHc: 'grid' (0.5 GeV
    lattice over the reach [82.5, 97.5]), 'mc_points' (the trained mA,
    where direct-MC comparison is possible) and 'groups' (template-sharing
    groups seeded at the trained mA = 85/90/95)."""
    with open(config_path("pnet_grid.json")) as f:
        return json.load(f)


def pnet_uncertainties_path():
    """configs/pnet_interpolation_uncertainties.json — the ParticleNet-layer
    nuisance values (CMS_interp_{res,eff}_pnet_*), written by
    python/exportPnetUncertainties.py from the closure/pnet shards."""
    return config_path("pnet_interpolation_uncertainties.json")


def pnet_fits_dir(mhc=None):
    """fits/pnet/ (no arg) or fits/pnet/MHc{X}: threshold_wp.json (the
    frozen working point per category x seed) and eps_model.json (the
    per-seed threshold-efficiency quadratics). Git-tracked, like the
    Baseline fits/MHc{X} tree."""
    base = os.path.join(module_dir(), "fits", "pnet")
    return os.path.join(base, _mhc_dirname(mhc)) if mhc is not None else base


def pnet_closure_dir(mhc=None):
    """closure/pnet/ (no arg) or closure/pnet/MHc{X}: shape_reuse.json,
    yield_interp.json, template_closure.json and plots/. Git-tracked, like
    closure/interpolation/."""
    base = os.path.join(module_dir(), "closure", "pnet")
    return os.path.join(base, _mhc_dirname(mhc)) if mhc is not None else base


def pnet_closure_plots_dir(kind):
    """Plots of ParticleNet-layer closure products that span every study:
    closure/pnet/plots/{kind}. Currently kind "residual" — the LOO
    residual scatters behind CMS_interp_{res,eff}_pnet."""
    if kind != "residual":
        raise ValueError(f"unknown pnet closure plot kind: {kind}")
    return os.path.join(pnet_closure_dir(), "plots", kind)


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
    fits/yield); the nuisance-rule summary and the LOO residual scatters
    with the closure products (kind "nuisance" ->
    closure/interpolation/plots/nuisance, kind "residual" ->
    closure/interpolation/plots/residual)."""
    if kind in ("params", "yield"):
        return os.path.join(interpolation_fits_dir(), kind)
    if kind in ("nuisance", "residual"):
        return os.path.join(interpolation_closure_dir(), "plots", kind)
    raise ValueError(f"unknown global plot kind: {kind}")


def interpolation_loo_dir(mhc, ma):
    """Per-point leave-one-out (LOO) output dir: models refit on the full
    mA grid minus this point, closure evaluated at this point only. The
    per-mHc aggregate lives in interpolation_closure_dir(mhc)."""
    return os.path.join(interpolation_closure_dir(), "loo",
                        f"MHc{int(mhc)}_MA{int(ma)}")
