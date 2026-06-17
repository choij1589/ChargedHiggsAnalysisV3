#!/usr/bin/env python3
"""Draw medium b-jet multiplicity from the existing bjets/size histograms."""

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
    load_file_hist,
    plot_with_canvas,
    samplegroups,
    signal_path,
)


HISTKEY = "bjets/size"


def parse_args():
    parser = base_parser(__doc__)
    parser.set_defaults(output_dir="plots/2018/medium_bjets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.debug)

    repo_root = Path(args.repo_root).resolve()
    outdir = Path(args.output_dir)
    if not outdir.is_absolute():
        outdir = Path.cwd() / outdir

    samples = samplegroups(repo_root)[ERA][args.channel]
    kfactor_data = kfactors(repo_root)
    conv_data = conv_sf(repo_root)
    fake_data = fake_norm(repo_root)
    flag = CHANNEL_INPUT[args.channel]
    hist_path = f"{args.channel}/Central/{HISTKEY}"
    audit = []

    data_hist = None
    for sample in samples["data"]:
        path = data_path(repo_root, args.channel, sample)
        hist, item = load_file_hist(path, hist_path, f"data_{sample}")
        item.update({"kind": "data", "sample": sample})
        audit.append(item)
        data_hist = add_hist(data_hist, hist, "data")

    backgrounds = {}
    for group_name, categories in GROUP_MAP.items():
        group_hist = None
        for category in categories:
            for sample in samples.get(category, []):
                path, _tree_name = background_path(repo_root, args.channel, category, sample)
                hist, item = load_file_hist(path, hist_path, f"{group_name}_{sample}")
                item.update({"kind": "background", "process": group_name, "category": category, "sample": sample})
                audit.append(item)
                if hist is None:
                    continue
                clip_negative_bins(hist)
                if category == "nonprompt":
                    for idx in range(hist.GetNcells()):
                        hist.SetBinError(idx, hist.GetBinContent(idx) * fake_data.get(flag, {}).get(ERA, 0.30))
                else:
                    apply_background_scales(hist, sample, category, args.channel, ERA, kfactor_data, conv_data)
                group_hist = add_hist(group_hist, hist, group_name)
        if group_hist is not None:
            backgrounds[group_name] = group_hist

    ordered_backgrounds = {name: backgrounds[name] for name in BKG_ORDER if name in backgrounds}

    signals = {}
    for masspoint in args.signals:
        path = signal_path(repo_root, args.channel, masspoint)
        hist, item = load_file_hist(path, hist_path, masspoint)
        item.update({"kind": "signal", "sample": masspoint})
        audit.append(item)
        if hist is not None:
            clip_negative_bins(hist)
            hist.Scale(args.signal_normalization * args.signal_scale)
            signals[format_signal_label(masspoint)] = hist

    audit_path = outdir / args.channel / "nB_medium_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open("w") as handle:
        json.dump(audit, handle, indent=2)

    config = build_config(args, "N_{b}^{medium}")
    output_path = outdir / args.channel / f"nB_medium.{args.output_format}"
    plot_with_canvas(data_hist, ordered_backgrounds, signals, config, output_path)
    print(output_path)
    print(audit_path)


if __name__ == "__main__":
    main()
