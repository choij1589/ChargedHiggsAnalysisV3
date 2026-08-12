#!/usr/bin/env python3
"""The interpolation-uncertainty rule, made visible.

One PNG per (channel, era) of the norm nuisance: for each mA region, every
study's rms as a point against mHc, the adopted value (the max over those
studies) as a solid line and the cell's pooled rms dashed. It shows at a
glance which study set the nuisance and how far the others sit below it —
the thing a single number in the config cannot say.

Reads tests/interpolation/loo_uncertainties.pooled.json, so it runs after
`exportInterpUncertainties.py --loo --all --pooled`. JSON-only apart from
the ROOT canvases; safe on the login node.

  python3 plotInterpNuisances.py
"""
import argparse
import json
import os

import interp_plot_utils
import srspaths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-grid", action="store_true",
                        help="use the full-mA-grid block instead of the "
                             "production-pairing one (diagnostic)")
    args = parser.parse_args()

    path = os.path.join(srspaths.interpolation_dir(),
                        "loo_uncertainties.pooled.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run exportInterpUncertainties.py "
            "--loo --all --pooled first")
    with open(path) as f:
        payload = json.load(f)
    block = payload if args.full_grid else payload["production_restricted"]
    detail = block["per_study_detail"]

    outdir = srspaths.interpolation_global_plots_dir("nuisance")
    os.makedirs(outdir, exist_ok=True)

    n = 0
    for channel, per_era in sorted(block["norm"].items()):
        for era, per_bin in sorted(per_era.items()):
            regions = {}
            for region, value in sorted(per_bin.items()):
                diag = detail.get(f"{channel}/{era}/{region}", {})
                if not diag.get("per_study_rms"):
                    continue          # unreachable or fallback cell
                regions[region] = {
                    "value": value - 1.0,
                    "per_study_rms": diag["per_study_rms"],
                    "pooled_rms": diag["pooled_rms"],
                    "driver": diag["driver"],
                }
            if not regions:
                continue
            interp_plot_utils.plot_nuisance_cell(channel, era, regions,
                                                 outdir)
            n += 1
    print(f"Wrote {n} nuisance plots into {outdir}")


if __name__ == "__main__":
    main()
