"""Single source of truth for SignalRegionStudyV4 path construction.

Every python script in this module must construct sample/template/result
paths through these helpers (adoption completed in the Stage-B
simplification).  The module name appears as a literal exactly once, here.

Path layout contract:
    {WORKDIR}/{MODULE_NAME}/samples/{era}/{channel}/{masspoint}/{process}.root
    {WORKDIR}/{MODULE_NAME}/templates/{era}/{channel}/{masspoint}/{method}/{suffix}/
    {WORKDIR}/{MODULE_NAME}/results/json/{mode}/{era}/limits....json

WORKDIR-based (not __file__-based) on purpose: the condor template wrapper
builds a scratch workdir in which python/ is a symlink back into the repo,
and resolving __file__ would escape the scratch sandbox.
"""
import json
import os

MODULE_NAME = "SignalRegionStudyV4"

BINNING = "extended"

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


def binning_suffix(unblind):
    """Template-directory suffix. The only python-side construction site;
    shell-side equivalent is srs_binning_suffix() in scripts/env.sh."""
    return f"{BINNING}_unblind" if unblind else BINNING


def sample_dir(era, channel, masspoint):
    return os.path.join(module_dir(), "samples", era, channel, masspoint)


def template_dir(era, channel, masspoint, method, unblind=True):
    return os.path.join(
        module_dir(), "templates", era, channel, masspoint, method,
        binning_suffix(unblind)
    )


def asymptotic_root(era, channel, masspoint, method, unblind=True):
    return os.path.join(
        template_dir(era, channel, masspoint, method, unblind),
        "combine_output", "asymptotic",
        f"higgsCombine.{masspoint}.{method}.{binning_suffix(unblind)}."
        "AsymptoticLimits.mH120.root"
    )


def limits_json(era, channel, method, mode="BR", unblind=True):
    """Collected-limits JSON path. Combined is the default channel and gets
    no channel infix (matches the V3 naming contract)."""
    channel_infix = "" if channel == "Combined" else f".{channel}"
    unblind_suffix = ".unblind" if unblind else ""
    return os.path.join(
        module_dir(), "results", "json", mode, era,
        f"limits.{era}{channel_infix}.Asymptotic.{method}{unblind_suffix}.json"
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
