#!/usr/bin/env python3
"""Filter a combineTool impacts.json for plotting (V3-equivalent).

Drops autoMCStats bin parameters (prop_bin*/autoMCStat*), sorts the
remaining nuisances by |impact on r| descending, and optionally keeps the
top N. The unfiltered json stays the full record; the filtered one feeds
plotImpacts.py --summary.
"""
import argparse
import json
import re


def impact_r(param):
    return abs(param.get("impact_r", 0.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--top", type=int, default=None,
                        help="keep only the N largest |impact_r|")
    parser.add_argument("--drop", default=r"^(prop_bin|autoMCStat)",
                        help="regex of parameter names to drop")
    args = parser.parse_args()

    with open(args.input) as f:
        payload = json.load(f)

    drop = re.compile(args.drop)
    params = [p for p in payload["params"] if not drop.search(p["name"])]
    params.sort(key=impact_r, reverse=True)
    n_dropped = len(payload["params"]) - len(params)
    if args.top is not None:
        params = params[:args.top]

    payload["params"] = params
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"{args.output}: kept {len(params)} nuisances "
          f"({n_dropped} dropped by /{args.drop}/)")


if __name__ == "__main__":
    main()
