#!/usr/bin/env python3
"""
Build per-sample and per-category cutflow JSON files from analyzer cutflow histograms.

The output is intentionally a pure analyzer cutflow:
- PromptAnalyzer cutflow bins use the weights applied in PromptAnalyzer fillCutflow.
- MatrixAnalyzer cutflow bins use the MatrixAnalyzer cutflow weights.
- KFactor, ConvSF, WZSF, fake-rate normalization, and post-processing scale factors
  are not applied here.
"""

import argparse
import json
import os
import sys
from math import sqrt

from ROOT import gROOT

gROOT.SetBatch(True)

TRILEPTON_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKDIR = os.environ.get("WORKDIR", os.path.dirname(TRILEPTON_DIR))
sys.path.insert(0, f"{WORKDIR}/Common/Tools")

from plotter import get_era_list  # noqa: E402
from HistoUtils import load_histogram, load_era_configs, get_sample_lists  # noqa: E402
from utils import build_sknanoutput_path  # noqa: E402


CHANNELS = [
    "SR1E2Mu", "SR3Mu",
    "ZFake1E2Mu", "ZFake3Mu",
    "ZG1E2Mu", "ZG3Mu",
    "WZ1E2Mu", "WZ3Mu",
    "TTZ2E1Mu",
]

CATEGORIES = ["nonprompt", "conv", "ttX", "diboson", "others"]
PROMPT_CATEGORIES = ["conv", "ttX", "diboson", "others"]

PAPER_SIGNALS = ["MHc70_MA15", "MHc100_MA60", "MHc130_MA90", "MHc160_MA155"]
PN_SIGNALS = ["MHc100_MA95", "MHc130_MA90", "MHc160_MA85"]

CUT_STAGES = [
    ("Initial", 0),
    ("NoiseFilter", 1),
    ("EventVetoMap", 2),
    ("LeptonSelection", 3),
    ("ConversionFilter", 4),
    ("Trigger", 5),
    ("KinematicCuts", 6),
    ("JetRequirements", 7),
    ("Final", 8),
]

BASE_CUT_DEFINITIONS = {
    "Initial": "Analyzer entry point after input MiniAOD/NanoAOD production and skimmed TriLepton tree selection.",
    "NoiseFilter": "Common event noise filters passed.",
    "EventVetoMap": "Event veto map selection passed.",
    "LeptonSelection": "Three-lepton channel object and flavor selection passed.",
    "ConversionFilter": "Conversion veto/filter selection passed.",
    "Trigger": "Analysis trigger selection passed.",
    "KinematicCuts": "Channel kinematic selection passed.",
    "JetRequirements": "Channel jet/b-jet requirement stage passed.",
    "Final": "Final SR/CR channel selection passed.",
}

FINAL_CUT_DEFINITIONS = {
    "SR1E2Mu": "Final SR1E2Mu signal-region selection.",
    "SR3Mu": "Final SR3Mu signal-region selection.",
    "ZFake1E2Mu": "Final ZFake1E2Mu nonprompt control-region selection.",
    "ZFake3Mu": "Final ZFake3Mu nonprompt control-region selection.",
    "ZG1E2Mu": "Final ZG1E2Mu conversion control-region selection.",
    "ZG3Mu": "Final ZG3Mu conversion control-region selection.",
    "WZ1E2Mu": "Final WZ1E2Mu diboson control-region selection.",
    "WZ3Mu": "Final WZ3Mu diboson control-region selection.",
    "TTZ2E1Mu": "Final TTZ2E1Mu ttX control-region selection.",
}

CUTFLOW_WEIGHTS = {
    "PromptAnalyzer": {
        "data": "1.0",
        "mc_and_signal": "MCweight() * ev.GetTriggerLumi(\"Full\")",
        "not_applied": [
            "KFactor",
            "ConvSF",
            "WZNjetsSF",
            "fake-rate normalization",
            "fake weights",
            "lepton reconstruction/ID/isolation scale factors",
            "trigger scale factors",
            "pileup scale factors",
            "b-tagging scale factors",
            "post-processing normalization factors",
        ],
    },
    "MatrixAnalyzer": {
        "nonprompt": "1.0 unit cutflow weight; fake weights are not applied to cutflow bins.",
        "not_applied": [
            "KFactor",
            "ConvSF",
            "WZNjetsSF",
            "fake-rate normalization",
            "post-processing normalization factors",
        ],
    },
}

EXCLUDED_CORRECTIONS = [
    "KFactor",
    "ConvSF",
    "WZNjetsSF",
    "fake-rate normalization",
    "post-processing rate uncertainties",
]


def get_channel_flags(channel):
    if "1E2Mu" in channel:
        return "Run1E2Mu", "1E2Mu"
    if "2E1Mu" in channel:
        return "Run2E1Mu", "2E1Mu"
    if "3Mu" in channel:
        return "Run3Mu", "3Mu"
    raise ValueError(f"Cannot determine run flag for channel: {channel}")


def get_cut_stages(channel):
    if channel.startswith("SR") or channel.startswith("ZFake"):
        return list(CUT_STAGES)
    return [(cut_name, index) for cut_name, index in CUT_STAGES if cut_name != "JetRequirements"]


def load_signal_config(selection):
    if selection == "paper":
        return list(PAPER_SIGNALS)
    if selection == "pn":
        return list(PN_SIGNALS)

    config_path = os.path.join(TRILEPTON_DIR, "configs", "signals.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config.get("signals", [])


def empty_cutflow(cut_stages):
    return {
        cut_name: {"yield": 0.0, "stat_error": 0.0}
        for cut_name, _ in cut_stages
    }


def add_cutflows(cutflows, cut_stages):
    total = empty_cutflow(cut_stages)
    for cutflow in cutflows:
        for cut_name, values in cutflow.items():
            if cut_name not in total:
                continue
            total[cut_name]["yield"] += values["yield"]
            total[cut_name]["stat_error"] = sqrt(
                total[cut_name]["stat_error"]**2 + values["stat_error"]**2
            )
    return total


def hist_bin_value(hist, stage_index):
    if hist is None:
        return 0.0, 0.0, False
    bin_index = stage_index + 1
    if hist.GetNbinsX() < bin_index:
        return 0.0, 0.0, False
    return float(hist.GetBinContent(bin_index)), float(hist.GetBinError(bin_index)), True


def choose_stage_value(flag_hist, channel_hist, stage_index):
    """Choose the channel cutflow when populated, with flag fallback for shared stages."""
    channel_value, channel_error, channel_ok = hist_bin_value(channel_hist, stage_index)
    flag_value, flag_error, flag_ok = hist_bin_value(flag_hist, stage_index)

    if stage_index == 8:
        if channel_ok:
            return channel_value, channel_error, "channel"
        return 0.0, 0.0, "missing_channel_final"

    if channel_ok and (channel_value != 0.0 or channel_error != 0.0):
        return channel_value, channel_error, "channel"
    if flag_ok:
        return flag_value, flag_error, "flag"
    if channel_ok:
        return channel_value, channel_error, "channel_zero"
    return 0.0, 0.0, "missing"


def load_cutflow(file_path, flag, channel, era, cut_stages):
    flag_hist = load_histogram(file_path, f"{flag}/Central/cutflow", era)
    channel_hist = load_histogram(file_path, f"{channel}/Central/cutflow", era)

    cutflow = {}
    sources = {}
    has_any_bin = False
    for cut_name, stage_index in cut_stages:
        value, error, source = choose_stage_value(flag_hist, channel_hist, stage_index)
        cutflow[cut_name] = {"yield": value, "stat_error": error}
        sources[cut_name] = source
        if source not in ("missing", "missing_channel_final"):
            has_any_bin = True

    status = "ok" if has_any_bin else "missing_cutflow"
    return {
        "status": status,
        "file": file_path,
        "cutflow": cutflow,
        "sources": sources,
    }


def merge_sample_records(records, sample_name, analyzer, category, cut_stages, sample_label=None):
    valid_records = [record for record in records if record["status"] == "ok"]
    return {
        "sample": sample_name,
        "label": sample_label or sample_name,
        "category": category,
        "analyzer": analyzer,
        "status": "ok" if valid_records else "missing_cutflow",
        "files": [record["file"] for record in records],
        "cutflow": add_cutflows([record["cutflow"] for record in valid_records], cut_stages),
        "bin_sources": {
            os.path.basename(record["file"]): record["sources"]
            for record in valid_records
        },
    }


def make_signal_path(signal, flag, era):
    return (
        f"{WORKDIR}/SKNanoOutput/PromptAnalyzer/"
        f"{flag}_RunSyst_RunTheoryUnc/{era}/TTToHcToWAToMuMu-{signal}.root"
    )


def get_cut_definitions(channel, cut_stages):
    definitions = dict(BASE_CUT_DEFINITIONS)
    definitions["Final"] = FINAL_CUT_DEFINITIONS[channel]
    return {
        cut_name: definitions[cut_name]
        for cut_name, _ in cut_stages
    }


def make_metadata(args, era_list, signal_list, cut_stages):
    stage_names = [cut_name for cut_name, _ in cut_stages]
    return {
        "schema_version": 1,
        "era": args.era,
        "eras": era_list,
        "channel": args.channel,
        "systematic": "Central",
        "source": "Analyzer Central/cutflow histograms",
        "pre_skim": {
            "included_as_cutflow_bins": False,
            "policy": "Analyzer-only start selected for this implementation.",
            "background_note": (
                "Background files already come from Skim_TriLep outputs. The skimmer "
                "selection is documented here but not reconstructed into extra JSON bins."
            ),
            "signal_note": "Signal cutflow starts at the analyzer Initial bin.",
            "Skim_TriLep_background_selection": [
                "preselected electron pT > 8 and eta acceptance",
                "preselected muon pT > 8 and eta acceptance",
                "NEl + NMu >= 3",
                "leading lepton pT > 15",
            ],
        },
        "benchmark_signals": signal_list,
        "cut_order": stage_names,
        "cut_definitions": get_cut_definitions(args.channel, cut_stages),
        "cutflow_weights": CUTFLOW_WEIGHTS,
        "excluded_corrections": EXCLUDED_CORRECTIONS,
        "notes": [
            "Per-cut statistical errors are read from ROOT bin errors.",
            "Category and total errors are summed in quadrature across samples and eras.",
            "Shared pre-final stages may be read from the run-flag cutflow when the final-channel cutflow is empty.",
            "Final is always read from the final-channel cutflow.",
            "JetRequirements is emitted only for SR/ZFake channels, where the analyzers fill that stage.",
        ],
    }


def add_record(collection, key, record):
    collection[key] = record
    return record


def build_output(args):
    if args.channel not in CHANNELS:
        raise ValueError(f"Invalid channel: {args.channel}")

    flag, channel_flag = get_channel_flags(args.channel)
    era_list = get_era_list(args.era)
    signal_list = load_signal_config(args.signals)
    cut_stages = get_cut_stages(args.channel)

    class ChannelArgs:
        def __init__(self, channel):
            self.channel = channel

    era_samples, _ = load_era_configs(ChannelArgs(channel_flag), era_list)
    _, mc_categories, _ = get_sample_lists(era_samples, CATEGORIES)

    output = {
        "metadata": make_metadata(args, era_list, signal_list, cut_stages),
        "samples": {},
        "categories": {},
        "total_background": empty_cutflow(cut_stages),
        "signals": {},
        "data": None,
    }

    for category in CATEGORIES:
        category_records = []
        samples = sorted(mc_categories.get(category, []))
        for sample in samples:
            records = []
            for era in era_list:
                if category not in era_samples.get(era, {}) or sample not in era_samples[era][category]:
                    continue
                is_nonprompt = category == "nonprompt"
                file_path = build_sknanoutput_path(
                    WORKDIR,
                    args.channel,
                    flag,
                    era,
                    sample,
                    is_nonprompt=is_nonprompt,
                    run_syst=not is_nonprompt,
                )
                records.append(load_cutflow(file_path, flag, args.channel, era, cut_stages))

            analyzer = "MatrixAnalyzer" if category == "nonprompt" else "PromptAnalyzer"
            sample_key = f"{category}:{sample}" if category == "nonprompt" else sample
            record = merge_sample_records(records, sample, analyzer, category, cut_stages, sample_key)
            add_record(output["samples"], sample_key, record)
            if record["status"] == "ok":
                category_records.append(record)

        output["categories"][category] = add_cutflows(
            [record["cutflow"] for record in category_records],
            cut_stages,
        )

    output["total_background"] = add_cutflows(output["categories"].values(), cut_stages)

    for signal in signal_list:
        records = [
            load_cutflow(make_signal_path(signal, flag, era), flag, args.channel, era, cut_stages)
            for era in era_list
        ]
        record = merge_sample_records(records, signal, "PromptAnalyzer", "signal", cut_stages)
        if record["status"] == "ok":
            output["signals"][signal] = record

    if not args.blind:
        data_records = []
        for era in era_list:
            for sample in era_samples.get(era, {}).get("data", []):
                file_path = build_sknanoutput_path(WORKDIR, args.channel, flag, era, sample)
                data_records.append(load_cutflow(file_path, flag, args.channel, era, cut_stages))
        output["data"] = merge_sample_records(
            data_records,
            "data",
            "PromptAnalyzer",
            "data",
            cut_stages,
        )

    return output


def save_output(args, output):
    output_dir = os.path.join(args.output_root, args.era, args.channel)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cutflow_breakdown_Central.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
        f.write("\n")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build pure cutflow breakdown JSON files.")
    parser.add_argument("--era", required=True, help="Era or combined period: 2016preVFP, ..., Run2, Run3")
    parser.add_argument("--channel", required=True, choices=CHANNELS, help="SR/CR channel")
    parser.add_argument(
        "--signals",
        default="paper",
        choices=["paper", "pn", "all"],
        help="Benchmark signal set to include",
    )
    parser.add_argument("--blind", action="store_true", help="Do not include data cutflows")
    parser.add_argument(
        "--include-data",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--output-root",
        default=os.path.join(TRILEPTON_DIR, "results"),
        help="Directory under which era/channel JSON outputs are written",
    )
    args = parser.parse_args()

    output = build_output(args)
    output_path = save_output(args, output)

    print(f"[INFO] Wrote {output_path}")
    print(f"[INFO] Background total Final = {output['total_background']['Final']['yield']:.6g}")
    if output["signals"]:
        print(f"[INFO] Signals included: {', '.join(output['signals'].keys())}")


if __name__ == "__main__":
    main()
