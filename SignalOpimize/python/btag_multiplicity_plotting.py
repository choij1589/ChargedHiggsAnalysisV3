#!/usr/bin/env python3
"""Shared b-tag multiplicity plotting helpers for SignalOpimize."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import ROOT


ROOT.gROOT.SetBatch(True)

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_TOOLS = REPO_ROOT / "Common" / "Tools"
if str(COMMON_TOOLS) not in sys.path:
    sys.path.insert(0, str(COMMON_TOOLS))

from plotter import ComparisonCanvas, KinematicCanvas, get_CoM_energy  # noqa: E402
from plotter import PALETTE_LONG as PALETTE  # noqa: E402


ERA = "2018"
CHANNEL_INPUT = {
    "SR1E2Mu": "Run1E2Mu",
    "SR3Mu": "Run3Mu",
}
CHANNEL_LABELS = {
    "SR1E2Mu": ("SR", "e#mu#mu"),
    "SR3Mu": ("SR", "#mu#mu#mu"),
}
SIGNAL_POINTS = ["MHc70_MA15", "MHc100_MA60", "MHc130_MA90", "MHc160_MA155"]

BKG_COLORS = {
    "nonprompt": PALETTE[0],
    "diboson": PALETTE[1],
    "ttX": PALETTE[2],
    "conv": PALETTE[3],
    "others": PALETTE[4],
}
BKG_ORDER = ["others", "conv", "diboson", "ttX", "nonprompt"]
GROUP_MAP = OrderedDict(
    [
        ("nonprompt", ["nonprompt"]),
        ("diboson", ["WZ", "ZZ"]),
        ("ttX", ["ttW", "ttZ", "ttH", "tZq"]),
        ("conv", ["conversion"]),
        ("others", ["others"]),
    ]
)


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--channel", default="SR1E2Mu", choices=sorted(CHANNEL_INPUT))
    parser.add_argument("--era", default=ERA, choices=[ERA])
    parser.add_argument("--signals", default=SIGNAL_POINTS, nargs="+")
    parser.add_argument("--signal-scale", default=2.0, type=float)
    parser.add_argument(
        "--signal-normalization",
        default=1.0 / 3.0,
        type=float,
        help="Extra normalization applied to signal weights before --signal-scale.",
    )
    parser.add_argument("--unblind", action="store_true", help="Show data and ratio pad.")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-dir", default="plots/2018")
    parser.add_argument("--output-format", default="png", choices=["png", "pdf"])
    parser.add_argument("--nbins", default=10, type=int)
    parser.add_argument("--xmin", default=0.0, type=float)
    parser.add_argument("--xmax", default=10.0, type=float)
    return parser


def configure_logging(debug: bool) -> None:
    logging.basicConfig(level=logging.DEBUG if debug else logging.INFO, format="%(levelname)s: %(message)s")


def load_json(path: Path) -> Dict:
    with path.open() as handle:
        return json.load(handle)


def samplegroups(repo_root: Path) -> Dict:
    return load_json(repo_root / "SignalRegionStudyV3" / "configs" / "samplegroups.json")


def kfactors(repo_root: Path) -> Dict:
    return load_json(repo_root / "Common" / "Data" / "KFactors.json")


def conv_sf(repo_root: Path) -> Dict:
    return load_json(repo_root / "Common" / "Data" / "ConvSF.json")


def fake_norm(repo_root: Path) -> Dict:
    return load_json(repo_root / "Common" / "Data" / "FakeNorm.json")


def make_hist(name: str, nbins: int, xmin: float, xmax: float) -> ROOT.TH1D:
    hist = ROOT.TH1D(name, "", nbins, xmin, xmax)
    hist.Sumw2()
    hist.SetDirectory(0)
    return hist


def clone_hist(hist, name: str):
    clone = hist.Clone(name)
    clone.SetDirectory(0)
    return clone


def branch_names(path: Path, tree_name: str) -> Tuple[str, List[str]]:
    if not path.exists():
        return "missing_file", []
    root_file = ROOT.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie():
        return "open_failed", []
    tree = root_file.Get(tree_name)
    if not tree:
        root_file.Close()
        return "missing_tree", []
    names = [branch.GetName() for branch in tree.GetListOfBranches()]
    root_file.Close()
    return "ok", names


def load_tree_hist(
    path: Path,
    tree_name: str,
    branch: str,
    hist_name: str,
    nbins: int,
    xmin: float,
    xmax: float,
    weight_expr: Optional[str],
) -> Tuple[Optional[ROOT.TH1D], Dict]:
    status, branches = branch_names(path, tree_name)
    audit = {
        "path": str(path),
        "tree": tree_name,
        "status": status,
        "missing_branches": [],
    }
    if status != "ok":
        return None, audit
    required = [branch]
    if weight_expr:
        required.append("weight")
    missing = [name for name in required if name not in branches]
    if missing:
        audit["status"] = "missing_branch"
        audit["missing_branches"] = missing
        return None, audit

    rdf = ROOT.RDataFrame(tree_name, str(path))
    if weight_expr:
        rdf = rdf.Define("__plot_weight", weight_expr)
        result = rdf.Histo1D((hist_name, "", nbins, xmin, xmax), branch, "__plot_weight")
    else:
        result = rdf.Histo1D((hist_name, "", nbins, xmin, xmax), branch)
    hist = clone_hist(result.GetValue(), hist_name)
    return hist, audit


def load_file_hist(path: Path, hist_path: str, hist_name: str) -> Tuple[Optional[ROOT.TH1], Dict]:
    audit = {"path": str(path), "hist": hist_path, "status": "ok", "missing_branches": []}
    if not path.exists():
        audit["status"] = "missing_file"
        return None, audit
    root_file = ROOT.TFile.Open(str(path), "READ")
    if not root_file or root_file.IsZombie():
        audit["status"] = "open_failed"
        return None, audit
    hist = root_file.Get(hist_path)
    if not hist:
        root_file.Close()
        audit["status"] = "missing_hist"
        return None, audit
    out = clone_hist(hist, hist_name)
    root_file.Close()
    return out, audit


def add_hist(total, hist, name: str):
    if hist is None:
        return total
    if total is None:
        return clone_hist(hist, name)
    total.Add(hist)
    return total


def clip_negative_bins(hist) -> None:
    for idx in range(hist.GetNcells()):
        if hist.GetBinContent(idx) < 0.0:
            hist.SetBinContent(idx, 0.0)
            hist.SetBinError(idx, 0.0)


def apply_rate_uncertainty(hist, rel_unc: float) -> None:
    for idx in range(hist.GetNcells()):
        content = hist.GetBinContent(idx)
        stat = hist.GetBinError(idx)
        hist.SetBinError(idx, math.sqrt(stat * stat + (content * rel_unc) ** 2))


def run_period(era: str) -> str:
    if era in {"2016preVFP", "2016postVFP", "2017", "2018"}:
        return "Run2"
    return "Run3"


def apply_background_scales(
    hist,
    sample: str,
    category: str,
    channel: str,
    era: str,
    kfactor_data: Dict,
    conv_data: Dict,
) -> None:
    period = run_period(era)
    if sample in kfactor_data.get(period, {}):
        info = kfactor_data[period][sample]
        hist.Scale(info["kFactor"])
        if "xsecErr" in info:
            apply_rate_uncertainty(hist, info["xsecErr"] - 1.0)

    if category == "conversion":
        channel_key = "1E2Mu" if channel == "SR1E2Mu" else "3Mu"
        era_data = conv_data.get(channel_key, {}).get(era)
        if era_data:
            hist.Scale(era_data["central"])
            apply_rate_uncertainty(hist, era_data["total"])
        else:
            apply_rate_uncertainty(hist, 0.20)

    if category == "others":
        apply_rate_uncertainty(hist, 0.50)


def format_signal_label(signal_mass: str) -> str:
    match = re.fullmatch(r"MHc(\d+)_MA(\d+)", signal_mass)
    if not match:
        return signal_mass
    mhc, ma = match.groups()
    return f"(m_{{H^{{+}}}}, m_{{A}}) = ({mhc}, {ma}) GeV"


def data_path(repo_root: Path, channel: str, sample: str) -> Path:
    flag = CHANNEL_INPUT[channel]
    return repo_root / "SKNanoOutput" / "PromptAnalyzer" / flag / ERA / f"Skim_TriLep_{sample}.root"


def background_path(repo_root: Path, channel: str, category: str, sample: str) -> Tuple[Path, str]:
    flag = CHANNEL_INPUT[channel]
    if category == "nonprompt":
        return (
            repo_root / "SKNanoOutput" / "MatrixAnalyzer" / flag / ERA / f"Skim_TriLep_{sample}.root",
            "Events",
        )
    return (
        repo_root
        / "SKNanoOutput"
        / "PromptAnalyzer"
        / f"{flag}_RunSyst"
        / ERA
        / f"Skim_TriLep_{sample}.root",
        "Events_Central",
    )


def signal_path(repo_root: Path, channel: str, masspoint: str) -> Path:
    flag = CHANNEL_INPUT[channel]
    return (
        repo_root
        / "SKNanoOutput"
        / "PromptAnalyzer"
        / f"{flag}_RunSyst_RunTheoryUnc"
        / ERA
        / f"TTToHcToWAToMuMu-{masspoint}.root"
    )


def build_config(args, x_title: str) -> Dict:
    region, channel_label = CHANNEL_LABELS[args.channel]
    config = {
        "xTitle": x_title,
        "yTitle": "Events",
        "xRange": [args.xmin, args.xmax],
        "rRange": [0.0, 2.0],
        "logy": False,
        "era": args.era,
        "CoM": get_CoM_energy(args.era),
        "rTitle": "Data / Pred",
        "maxDigits": 3,
        "blind": not args.unblind,
        "no_ratio": not args.unblind,
        "overflow": True,
        "iPos": 0,
        "legend": (0.72, 0.55, 0.99, 0.89),
        "legendTextSize": 0.038,
        "signalLegend": (0.32, 0.63, 0.73, 0.87),
        "signalLegendTextSize": 0.034,
        "signalLineWidth": 2,
        "signalFill": False,
        "colors": [BKG_COLORS[name] for name in BKG_ORDER],
        "channel": region,
        "region": channel_label,
        "channelPosY": 0.75,
        "channelPosX": 0.22,
        "chi2_test": False,
    }
    return config


def plot_with_canvas(data_hist, backgrounds: Dict, signals: Dict, config: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plotter = ComparisonCanvas(data_hist, backgrounds, config)
    plotter.drawPadUp()
    if signals:
        plotter.drawSignals(signals)
    plotter.drawPadDown()
    plotter.canv.SaveAs(str(output_path))


def plot_signals_with_kinematic_canvas(signals: Dict, config: Dict, output_path: Path) -> None:
    if not signals:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    signal_config = dict(config)
    signal_config["legend"] = signal_config.get("signalLegend", (0.32, 0.63, 0.73, 0.87))
    signal_config["legendTextSize"] = signal_config.get("signalLegendTextSize", 0.034)
    signal_config["yTitle"] = "Events"
    plotter = KinematicCanvas(signals, signal_config)
    plotter.drawPad()
    plotter.leg.Draw()
    plotter.canv.SaveAs(str(output_path))
