"""Helpers for Run-period component template construction."""

RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]
RUN_PERIODS = {
    "Run2": RUN2_ERAS,
    "Run3": RUN3_ERAS,
}

SR_CHANNELS = ["SR1E2Mu", "SR3Mu"]
PHYSICS_PROCESS_ORDER = [
    "signal",
    "nonprompt",
    "WZ",
    "ZZ",
    "ttW",
    "ttZ",
    "ttH",
    "tZq",
    "conversion",
    "others",
]


def is_run_period(era):
    return era in RUN_PERIODS or era == "All"


def resolve_run_periods(era):
    """Return ordered ``[(period_name, suberas), ...]`` for an era request."""
    if era == "All":
        return [("Run2", list(RUN2_ERAS)), ("Run3", list(RUN3_ERAS))]
    if era in RUN_PERIODS:
        return [(era, list(RUN_PERIODS[era]))]
    raise ValueError(f"Unsupported V3 run-period target: {era}. Use Run2, Run3, or All.")


def resolve_channels(channel):
    """Return the atomic SR channels requested by a channel argument."""
    if channel == "Combined":
        return list(SR_CHANNELS)
    if channel in SR_CHANNELS:
        return [channel]
    raise ValueError(f"Unsupported run-period template channel: {channel}")


def category_name(channel, period):
    return f"{channel}_{period}"


def component_name(base_process, subera, *, is_signal=False):
    base = "signal" if is_signal else base_process
    return f"{base}_{subera}"


def component_base(process):
    for era in RUN2_ERAS + RUN3_ERAS:
        suffix = f"_{era}"
        if process.endswith(suffix):
            return process[:-len(suffix)]
    return process


def component_subera(process):
    for era in RUN2_ERAS + RUN3_ERAS:
        suffix = f"_{era}"
        if process.endswith(suffix):
            return era
    return None


def physics_group(process):
    base = component_base(process)
    if base == "signal":
        return "signal"
    return base


def is_signal_component(process):
    return component_base(process) == "signal"
