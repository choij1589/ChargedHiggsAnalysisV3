#!/usr/bin/env python3
"""Collect nominal vs preserve-shape unblind GoF p-values."""
import argparse
import csv
import json
from pathlib import Path


def read_gof(path):
    if not path.exists():
        return None
    with path.open() as handle:
        data = json.load(handle)
    record = data.get("120.0") or next(iter(data.values()))
    return {
        "p": record.get("p"),
        "obs": (record.get("obs") or [None])[0],
        "ntoys": len(record.get("toy", [])),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="Baseline")
    parser.add_argument("--binning", default="extended")
    parser.add_argument("--channel", default="Combined")
    parser.add_argument("--output", default="results/gof_unblind_preserve_shape_comparison.tsv")
    args = parser.parse_args()

    nominal_suffix = f"{args.binning}_unblind"
    preserve_suffix = f"{nominal_suffix}_preserve_shape"
    rows = []

    base = Path("templates")
    for preserve_json in sorted(base.glob(f"*/{args.channel}/*/{args.method}/{preserve_suffix}/combine_output/gof/gof.json")):
        parts = preserve_json.parts
        era = parts[1]
        masspoint = parts[3]
        nominal_json = base / era / args.channel / masspoint / args.method / nominal_suffix / "combine_output/gof/gof.json"
        nominal = read_gof(nominal_json)
        preserve = read_gof(preserve_json)
        rows.append({
            "era": era,
            "method": args.method,
            "masspoint": masspoint,
            "nominal_p": "" if nominal is None else nominal["p"],
            "preserve_shape_p": "" if preserve is None else preserve["p"],
            "delta_p": "" if nominal is None or preserve is None else preserve["p"] - nominal["p"],
            "nominal_ntoys": "" if nominal is None else nominal["ntoys"],
            "preserve_shape_ntoys": "" if preserve is None else preserve["ntoys"],
            "nominal_json": "" if nominal is None else str(nominal_json),
            "preserve_shape_json": str(preserve_json),
        })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        fieldnames = [
            "era", "method", "masspoint", "nominal_p", "preserve_shape_p",
            "delta_p", "nominal_ntoys", "preserve_shape_ntoys",
            "nominal_json", "preserve_shape_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
