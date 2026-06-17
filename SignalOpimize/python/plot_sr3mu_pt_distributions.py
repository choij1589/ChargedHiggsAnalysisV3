#!/usr/bin/env python3
"""Draw pT distributions for b-tag reference signals."""

from __future__ import annotations

import json
from pathlib import Path

from btag_multiplicity_plotting import (
    BKG_ORDER,
    CHANNEL_INPUT,
    ERA,
    GROUP_MAP,
    add_hist,
    apply_background_scales,
    background_path,
    base_parser,
    build_config,
    clip_negative_bins,
    configure_logging,
    conv_sf,
    data_path,
    fake_norm,
    format_signal_label,
    kfactors,
    load_tree_hist,
    plot_with_canvas,
    samplegroups,
    signal_path,
)


PLOTS = [
    ("SR1E2Mu", "pT1", "pT"),
    ("SR3Mu", "pT1", "pT1"),
    ("SR3Mu", "pT2", "pT2"),
]
X_TITLES = {
    "pT": "p_{T} [GeV]",
    "pT1": "p_{T}^{1} [GeV]",
    "pT2": "p_{T}^{2} [GeV]",
}
REFERENCE_SIGNALS = ["MHc70_MA15", "MHc100_MA60", "MHc130_MA90", "MHc160_MA155"]


def parse_args():
    parser = base_parser(__doc__)
    parser.set_defaults(
        signals=REFERENCE_SIGNALS,
        signal_scale=10.0,
        output_dir="plots/2018/pt_distributions",
        nbins=30,
        xmin=0.0,
        xmax=300.0,
    )
    parser.set_defaults(output_format="png")
    return parser.parse_args()


def build_branch_plot(args, channel: str, branch: str, output_name: str, outdir: Path) -> None:
    repo_root = Path(args.repo_root).resolve()
    samples = samplegroups(repo_root)[ERA][channel]
    kfactor_data = kfactors(repo_root)
    conv_data = conv_sf(repo_root)
    fake_data = fake_norm(repo_root)
    flag = CHANNEL_INPUT[channel]
    audit = []
    args.channel = channel

    data_hist = None
    for sample in samples["data"]:
        path = data_path(repo_root, channel, sample)
        hist, item = load_tree_hist(
            path,
            "Events_Central",
            branch,
            f"data_{sample}_{branch}",
            args.nbins,
            args.xmin,
            args.xmax,
            weight_expr=None,
        )
        item.update({"kind": "data", "sample": sample, "branch": branch})
        audit.append(item)
        data_hist = add_hist(data_hist, hist, "data")

    backgrounds = {}
    for group_name, categories in GROUP_MAP.items():
        group_hist = None
        for category in categories:
            for sample in samples.get(category, []):
                path, tree_name = background_path(repo_root, channel, category, sample)
                hist, item = load_tree_hist(
                    path,
                    tree_name,
                    branch,
                    f"{group_name}_{sample}_{branch}",
                    args.nbins,
                    args.xmin,
                    args.xmax,
                    weight_expr="weight",
                )
                item.update(
                    {
                        "kind": "background",
                        "process": group_name,
                        "category": category,
                        "sample": sample,
                        "branch": branch,
                    }
                )
                audit.append(item)
                if hist is None:
                    continue
                clip_negative_bins(hist)
                if category == "nonprompt":
                    for idx in range(hist.GetNcells()):
                        hist.SetBinError(idx, hist.GetBinContent(idx) * fake_data.get(flag, {}).get(ERA, 0.30))
                else:
                    apply_background_scales(hist, sample, category, channel, ERA, kfactor_data, conv_data)
                group_hist = add_hist(group_hist, hist, group_name)
        if group_hist is not None:
            backgrounds[group_name] = group_hist

    ordered_backgrounds = {name: backgrounds[name] for name in BKG_ORDER if name in backgrounds}

    signals = {}
    for masspoint in args.signals:
        path = signal_path(repo_root, channel, masspoint)
        hist, item = load_tree_hist(
            path,
            "Events_Central",
            branch,
            f"{masspoint}_{branch}",
            args.nbins,
            args.xmin,
            args.xmax,
            weight_expr=f"weight * {args.signal_normalization * args.signal_scale:.12g}",
        )
        item.update({"kind": "signal", "sample": masspoint, "branch": branch})
        audit.append(item)
        if hist is not None:
            clip_negative_bins(hist)
            signals[format_signal_label(masspoint)] = hist

    plot_dir = outdir / channel
    audit_path = plot_dir / f"{output_name}_audit.json"
    plot_dir.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w") as handle:
        json.dump(audit, handle, indent=2)

    config = build_config(args, X_TITLES[output_name])
    config["signalLegend"] = (0.30, 0.62, 0.73, 0.87)
    output_path = plot_dir / f"{output_name}.{args.output_format}"
    plot_with_canvas(data_hist, ordered_backgrounds, signals, config, output_path)
    print(output_path)
    print(audit_path)


def main() -> None:
    args = parse_args()
    configure_logging(args.debug)

    outdir = Path(args.output_dir)
    if not outdir.is_absolute():
        outdir = Path.cwd() / outdir

    for channel, branch, output_name in PLOTS:
        build_branch_plot(args, channel, branch, output_name, outdir)


if __name__ == "__main__":
    main()
