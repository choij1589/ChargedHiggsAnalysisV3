#!/usr/bin/env python3
"""Draw signal pT shape comparisons for fixed mA groups with KinematicCanvas."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

import ROOT


ROOT.gROOT.SetBatch(True)

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMON_TOOLS = REPO_ROOT / "Common" / "Tools"
if str(COMMON_TOOLS) not in sys.path:
    sys.path.insert(0, str(COMMON_TOOLS))

from plotter import KinematicCanvas, PALETTE_LONG, get_CoM_energy  # noqa: E402

from btag_multiplicity_plotting import (  # noqa: E402
    CHANNEL_INPUT,
    ERA,
    add_hist,
    branch_names,
    clip_negative_bins,
    configure_logging,
    format_signal_label,
    signal_path,
)


PLOTS = [
    ("SR1E2Mu", "pT1", "pT", "p_{T} [GeV]"),
    ("SR3Mu", "pT1", "pT1", "p_{T}^{1} [GeV]"),
    ("SR3Mu", "pT2", "pT2", "p_{T}^{2} [GeV]"),
]
MASSPOINTS_BY_MA = OrderedDict(
    [
        ("15", ["MHc70_MA15", "MHc100_MA15", "MHc130_MA15", "MHc160_MA15"]),
        ("30", ["MHc70_MA30", "MHc130_MA30", "MHc160_MA30"]),
        ("55", ["MHc70_MA55", "MHc130_MA55"]),
        ("60", ["MHc100_MA60", "MHc160_MA60"]),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output-dir", default="plots/2018/pt_same_ma")
    parser.add_argument("--output-format", default="png", choices=["png", "pdf"])
    parser.add_argument("--nbins", default=30, type=int)
    parser.add_argument("--xmin", default=0.0, type=float)
    parser.add_argument("--xmax", default=300.0, type=float)
    parser.add_argument("--signal-normalization", default=1.0 / 3.0, type=float)
    parser.add_argument("--absolute", action="store_true", help="Do not normalize histograms to unit area.")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def mhc_value(masspoint: str) -> int:
    match = re.fullmatch(r"MHc(\d+)_MA\d+", masspoint)
    if not match:
        return 999999
    return int(match.group(1))


def make_hist(path: Path, tree_name: str, branch: str, hist_name: str, args) -> Tuple[ROOT.TH1D | None, Dict]:
    status, branches = branch_names(path, tree_name)
    audit = {
        "path": str(path),
        "tree": tree_name,
        "branch": branch,
        "status": status,
        "missing_branches": [],
    }
    if status != "ok":
        return None, audit

    missing = [name for name in [branch, "weight"] if name not in branches]
    if missing:
        audit["status"] = "missing_branch"
        audit["missing_branches"] = missing
        return None, audit

    rdf = ROOT.RDataFrame(tree_name, str(path)).Define(
        "__plot_weight",
        f"weight * {args.signal_normalization:.12g}",
    )
    hist = rdf.Histo1D(
        (hist_name, "", args.nbins, args.xmin, args.xmax),
        branch,
        "__plot_weight",
    ).GetValue()
    out = hist.Clone(hist_name)
    out.SetDirectory(0)
    clip_negative_bins(out)
    return out, audit


def build_config(args, channel: str, x_title: str, ma: str) -> Dict:
    region = "e#mu#mu" if channel == "SR1E2Mu" else "#mu#mu#mu"
    return {
        "xTitle": x_title,
        "yTitle": "Normalized events" if not args.absolute else "Events",
        "xRange": [args.xmin, args.xmax],
        "era": ERA,
        "CoM": get_CoM_energy(ERA),
        "normalize": not args.absolute,
        "overflow": True,
        "logy": False,
        "iPos": 0,
        "maxDigits": 3,
        "legend": (0.50, 0.68, 0.90, 0.88),
        "legendTextSize": 0.032,
        "colors": PALETTE_LONG,
        "channel": "SR",
        "region": f"{region}, m_{{A}} = {ma} GeV",
        "channelPosX": 0.20,
        "channelPosY": 0.78,
        "channelSize": 0.042,
    }


def draw_group(args, repo_root: Path, outdir: Path, channel: str, branch: str, output_name: str, x_title: str, ma: str) -> Dict:
    hists = OrderedDict()
    audit = []
    for masspoint in sorted(MASSPOINTS_BY_MA[ma], key=mhc_value):
        path = signal_path(repo_root, channel, masspoint)
        hist, item = make_hist(path, "Events_Central", branch, f"{channel}_{branch}_{masspoint}", args)
        item.update({"channel": channel, "sample": masspoint, "ma": ma})
        audit.append(item)
        if hist is None:
            continue
        hists[format_signal_label(masspoint)] = add_hist(None, hist, masspoint)

    if not hists:
        return {"ma": ma, "channel": channel, "branch": branch, "status": "empty", "audit": audit}

    plot_dir = outdir / channel
    plot_dir.mkdir(parents=True, exist_ok=True)
    canvas = KinematicCanvas(hists, build_config(args, channel, x_title, ma))
    canvas.drawPad()
    output_path = plot_dir / f"{output_name}_MA{ma}.{args.output_format}"
    canvas.canv.SaveAs(str(output_path))

    audit_path = plot_dir / f"{output_name}_MA{ma}_audit.json"
    with audit_path.open("w") as handle:
        json.dump(audit, handle, indent=2)

    print(output_path)
    print(audit_path)
    return {
        "ma": ma,
        "channel": channel,
        "branch": branch,
        "status": "ok",
        "plot": str(output_path),
        "audit": str(audit_path),
        "samples_drawn": list(hists.keys()),
        "audit_entries": audit,
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.debug)
    repo_root = Path(args.repo_root).resolve()
    outdir = Path(args.output_dir)
    if not outdir.is_absolute():
        outdir = Path.cwd() / outdir

    summary = []
    for ma in MASSPOINTS_BY_MA:
        for channel, branch, output_name, x_title in PLOTS:
            summary.append(draw_group(args, repo_root, outdir, channel, branch, output_name, x_title, ma))

    outdir.mkdir(parents=True, exist_ok=True)
    with (outdir / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(outdir / "summary.json")


if __name__ == "__main__":
    main()
