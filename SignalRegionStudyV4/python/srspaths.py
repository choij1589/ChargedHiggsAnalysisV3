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


def masspoint_mhc_ma(masspoint):
    parts = masspoint.split("_")
    return (int(parts[0].replace("MHc", "")), int(parts[1].replace("MA", "")))


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


def tests_dir():
    return os.path.join(module_dir(), "tests")


def interpolation_dir(mhc=None):
    base = os.path.join(tests_dir(), "interpolation")
    return os.path.join(base, f"MHc{int(mhc)}") if mhc is not None else base


def interpolation_plots_dir(mhc, kind):
    """kind in {fits, params, closure, yields, deltas}."""
    return os.path.join(interpolation_dir(mhc), "plots", kind)
