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


def sample_dir(era, channel, masspoint):
    return os.path.join(module_dir(), "samples", era, channel, masspoint)


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
