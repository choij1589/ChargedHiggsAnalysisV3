#!/usr/bin/env python3
"""Export the PRODUCTION eps(mA) model: fits/pnet/MHc{X}/eps_model.json.

The ParticleNet-interpolation yield is

    N(era, mA) = k_era * G_period(mA) * f_category(mA) * eps_seed(era, mA)

with eps_seed a polynomial through the seed net's measured threshold
efficiencies at the anchor mA = 85/90/95 (quadratic when all three exist).
closPnetYields.py measures those efficiencies (and derives the eff
nuisance from their LEAVE-ONE-OUT residuals); this script fits the
all-anchor production polynomial per (category, seed, era) and freezes it
as a first-class artifact, so template production evaluates a committed
model instead of re-deriving one from closure shards.

Reads closure/pnet/MHc{X}/yield_interp.json; refuses a working-point
mismatch.

  python3 python/fitPnetEpsModel.py --mhc MHc115
"""
import argparse
import datetime
import json
import os
import sys
from collections import OrderedDict

import pnet_interp_config as pic
import srspaths
from pnet_interp_config import ANCHOR_MA, DEFAULT_WP


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", default="all",
                        help="comma-separated mHc studies, or 'all'")
    parser.add_argument("--wp", default=DEFAULT_WP,
                        help="working point the yield shards must carry")
    args = parser.parse_args()

    mhcs = (pic.pn_mhc_list() if args.mhc == "all"
            else [m.strip() for m in args.mhc.split(",") if m.strip()])

    for mhc in mhcs:
        src = os.path.join(srspaths.pnet_closure_dir(mhc),
                           "yield_interp.json")
        if not os.path.exists(src):
            raise SystemExit(f"{src} missing; run closPnetYields.py first")
        with open(src) as fh:
            results = json.load(fh)["results"]

        model = OrderedDict()
        warnings = []
        for key, entry in results.items():
            if entry.get("wp") != args.wp:
                raise RuntimeError(
                    f"{src}: {key} has wp={entry.get('wp')!r}, expected "
                    f"{args.wp!r} -- refusing to mix working points")
            per_era = OrderedDict()
            for pt_key, pt in entry["points"].items():
                if pt["mA"] not in ANCHOR_MA:
                    continue
                era = pt["era"]
                per_era.setdefault(era, {})[float(pt["mA"])] = pt["eps"]
            eras = OrderedDict()
            for era, anchors in per_era.items():
                if len(anchors) < len(ANCHOR_MA):
                    warnings.append(
                        f"[{key}/{era}] only {len(anchors)} of "
                        f"{len(ANCHOR_MA)} anchors present")
                coeffs, degree = pic.fit_eps_anchors(anchors)
                eras[era] = OrderedDict([
                    ("anchors", OrderedDict(
                        (f"{a:g}", anchors[a]) for a in sorted(anchors))),
                    ("coeffs", coeffs),
                    ("degree", degree),
                ])
            model[key] = OrderedDict([
                ("channel", entry["channel"]),
                ("period", entry["period"]),
                ("seed", entry["seed"]),
                ("seed_mA", entry["seed_mA"]),
                ("mass_window", entry["mass_window"]),
                ("threshold", entry["threshold"]),
                ("eras", eras),
            ])

        payload = OrderedDict([
            ("meta", OrderedDict([
                ("working_point", args.wp),
                ("rule", "polynomial through the seed net's measured "
                         "threshold efficiencies at the anchor mA "
                         f"{list(ANCHOR_MA)} (quadratic when all three "
                         "exist); numpy-convention coefficients, evaluate "
                         "with pnet_interp_config.eval_eps"),
                ("source", src.replace(srspaths.module_dir() + os.sep, "")),
                ("command", " ".join(sys.argv)),
                ("date", datetime.datetime.now().isoformat(
                    timespec="seconds")),
            ])),
            ("model", model),
            ("warnings", warnings),
        ])
        out = pic.eps_model_path(mhc)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump(payload, fh, indent=2)
        n_curves = sum(len(m["eras"]) for m in model.values())
        print(f"Wrote {out}: {len(model)} category-seeds, "
              f"{n_curves} eps curves"
              + (f", {len(warnings)} warning(s)" if warnings else ""))


if __name__ == "__main__":
    main()
