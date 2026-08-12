#!/usr/bin/env python3
"""Merge per-masspoint condor outputs of the interpolation chain.

One entry point for every sharded stage (fits-floating, fits, closure,
yields, yield-closure, shape-deltas); replaces the study's five separate
merge_*.py scripts. Fails loudly on a missing expected masspoint part
unless --allow-missing.

  python3 mergeInterpResults.py --mhc 160 --stage fits-floating
  python3 mergeInterpResults.py --mhc 160 --stage fits
  python3 mergeInterpResults.py --mhc 160 --stage closure
  python3 mergeInterpResults.py --mhc 160 --stage yields
  python3 mergeInterpResults.py --mhc 160 --stage yield-closure
  python3 mergeInterpResults.py --mhc 160 --stage shape-deltas [--allow-missing]
"""
import argparse
import datetime
import glob
import json
import os
import sys

import interp_plot_utils
import interpolation_config
import srspaths
from interpolation_config import masspoint_name


def _collect_parts(parts_dir, prefix, expected, allow_missing):
    """Returns (parts: [(mp, payload), ...] in expected order, missing,
    stray) for parts named '{prefix}.{mp}.json'."""
    parts = []
    missing = []
    for mp in expected:
        path = os.path.join(parts_dir, f"{prefix}.{mp}.json")
        if not os.path.exists(path):
            missing.append(mp)
            continue
        with open(path) as f:
            parts.append((mp, json.load(f)))
    if missing and not allow_missing:
        raise RuntimeError(f"missing part(s) for: {', '.join(missing)} "
                           "(use --allow-missing to override)")
    stray = [os.path.basename(p) for p in
             glob.glob(os.path.join(parts_dir, f"{prefix}.*.json"))
             if os.path.basename(p)[len(prefix) + 1:-5] not in expected]
    return parts, missing, stray


def _write(outpath, payload):
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    with open(outpath, "w") as f:
        json.dump(payload, f, indent=2)


def merge_fits(mhc, fit_pass, allow_missing, expected, study, variant=None):
    basename = "dcb_fits.json" if fit_pass == "frozen" else "dcb_fits_floating.json"
    part_prefix = basename[:-5]
    fits_dir = os.path.join(srspaths.interpolation_dir(mhc, variant=variant),
                            "fits")
    parts, missing, stray = _collect_parts(
        os.path.join(fits_dir, "parts"), part_prefix, expected, allow_missing)

    results, warnings, fixed_n = {}, [], None
    for _mp, part in parts:
        for cat_key, fits in part["results"].items():
            results.setdefault(cat_key, {}).update(fits)
        warnings.extend(part.get("warnings", []))
        if part["meta"].get("fixed_n"):
            fixed_n = part["meta"]["fixed_n"]
    if stray:
        warnings.append(f"stray part files ignored: {', '.join(stray)}")

    payload = {
        "meta": {
            "mhc": mhc, "fit_pass": fit_pass, "variant": variant,
            "fit_ma": study["fit"], "held_out_ma": study["held_out"],
            "fixed_n": fixed_n,
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "results": results,
        "warnings": warnings,
    }
    outpath = os.path.join(fits_dir, basename)
    _write(outpath, payload)
    n_fits = sum(len(v) for v in results.values())
    n_bad = sum(1 for cat in results.values() for fit in cat.values()
                if fit["quality"] != "good")
    print(f"Merged {len(expected) - len(missing)}/{len(expected)} parts -> "
          f"{outpath}: {n_fits} fits ({n_bad} bad).")
    for w in warnings:
        print(f"  warning: {w}")
    return 1 if missing and not allow_missing else 0


def merge_closure(mhc, allow_missing, expected, study, variant=None):
    interp_dir = srspaths.interpolation_dir(mhc, variant=variant)
    parts, missing, stray = _collect_parts(
        os.path.join(interp_dir, "closure", "parts"), "closure",
        expected, allow_missing)

    results, warnings, fit_ma = {}, [], study["fit"]
    for _mp, part in parts:
        for cat_key, per_mp in part["closure"].items():
            results.setdefault(cat_key, {}).update(per_mp)
        warnings.extend(part.get("warnings", []))
        fit_ma = part["meta"].get("fit_ma", fit_ma)
    if stray:
        warnings.append(f"stray part files ignored: {', '.join(stray)}")

    payload = {
        "meta": {
            "mhc": mhc, "all_ma": study["all"], "fit_ma": fit_ma,
            "held_out_ma": study["held_out"], "variant": variant,
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "closure": results,
        "warnings": warnings,
    }
    outpath = os.path.join(interp_dir, "closure.json")
    _write(outpath, payload)
    n = sum(len(v) for v in results.values())
    print(f"Merged {len(expected) - len(missing)}/{len(expected)} parts -> "
          f"{outpath}: {n} closure records.")
    for w in warnings:
        print(f"  warning: {w}")
    return 1 if missing and not allow_missing else 0


def merge_yields(mhc, allow_missing, expected, study):
    yields_dir = os.path.join(srspaths.interpolation_dir(mhc), "yields")
    parts, missing, stray = _collect_parts(
        os.path.join(yields_dir, "parts"), "yields", expected, allow_missing)

    results, warnings, meta_part = {}, [], None
    for _mp, part in parts:
        results.update(part["results"])
        warnings.extend(part.get("warnings", []))
        meta_part = part["meta"]
    if stray:
        warnings.append(f"stray part files ignored: {', '.join(stray)}")
    if meta_part is None:
        raise RuntimeError("No parts found")

    payload = {
        "meta": {
            "mhc": mhc, "fit_ma": study["fit"],
            "shape_polynomials": meta_part["shape_polynomials"],
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "results": results,
        "warnings": warnings,
    }
    outpath = os.path.join(yields_dir, "yields.json")
    _write(outpath, payload)
    n = sum(len(c) for mp in results.values() for c in mp["channels"].values())
    print(f"Merged {len(expected) - len(missing)}/{len(expected)} parts -> "
          f"{outpath}: {n} era yields.")
    for w in warnings:
        print(f"  warning: {w}")
    return 1 if missing and not allow_missing else 0


def summarize_scalar(closure):
    """Per channel x era: worst |pull| and |rel|, split by sample type."""
    stats = {}
    for entry in closure.values():
        kind = "in_sample" if entry["in_sample"] else "held_out"
        for channel, per_era in entry["scalar"].items():
            for era, rec in per_era.items():
                s = stats.setdefault((channel, era), {
                    "in_sample": {"pull": 0.0, "rel": 0.0, "n": 0},
                    "held_out": {"pull": 0.0, "rel": 0.0, "n": 0}})[kind]
                s["pull"] = max(s["pull"], abs(rec["pull"]))
                s["rel"] = max(s["rel"], abs(rec["rel"]))
                s["n"] += 1
    return stats


def merge_yield_closure(mhc, allow_missing, expected, study):
    yields_dir = os.path.join(srspaths.interpolation_dir(mhc), "yields")
    parts, missing, stray = _collect_parts(
        os.path.join(yields_dir, "parts"), "yield_closure", expected,
        allow_missing)

    results, warnings = {}, []
    for _mp, part in parts:
        results.update(part["closure"])
        warnings.extend(part.get("warnings", []))
    if stray:
        warnings.append(f"stray part files ignored: {', '.join(stray)}")
    if not results:
        raise RuntimeError("No parts found")

    payload = {
        "meta": {
            "mhc": mhc, "fit_ma": study["fit"],
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "closure": results,
        "warnings": warnings,
    }
    outpath = os.path.join(yields_dir, "yield_closure.json")
    _write(outpath, payload)
    print(f"Merged {len(expected) - len(missing)}/{len(expected)} parts -> "
          f"{outpath}\n")

    interp_plot_utils.plot_yield_residuals(
        results, mhc, srspaths.interpolation_plots_dir(mhc, "yields"))

    stats = summarize_scalar(results)
    header = (f"{'channel':<14} {'era':<12} "
              f"{'self-consistency (max |pull| / |rel|)':>38} "
              f"{'interpolation (max |pull| / |rel|)':>36}")
    print(header)
    print("-" * len(header))
    for (channel, era), s in sorted(stats.items()):
        cells = []
        for kind in ("in_sample", "held_out"):
            k = s[kind]
            cells.append(f"{k['pull']:.2f} / {100 * k['rel']:.1f}%"
                         f" ({k['n']} pts)" if k["n"] else "n/a")
        print(f"{channel:<14} {era:<12} {cells[0]:>38} {cells[1]:>36}")

    print("\nTemplate-level absolute-normalization closure (chi2/ndf):")
    header2 = (f"{'category':<20} {'median':>8} {'worst':>8} "
               f"{'worst mp':<16}")
    print(header2)
    print("-" * len(header2))
    per_cat = {}
    for mp, entry in results.items():
        for cat_key, rec in entry["template"].items():
            per_cat.setdefault(cat_key, []).append(
                (rec["chi2"] / rec["ndf"], mp))
    for cat_key, vals in sorted(per_cat.items()):
        vals.sort()
        median = vals[len(vals) // 2][0]
        worst_val, worst_mp = vals[-1]
        print(f"{cat_key:<20} {median:>8.2f} {worst_val:>8.2f} "
              f"{worst_mp:<16}")

    for w in warnings:
        print(f"  warning: {w}")
    return 1 if missing and not allow_missing else 0


def merge_shape_deltas(mhc, allow_missing, expected, study):
    deltas_dir = os.path.join(srspaths.interpolation_dir(mhc), "shape_deltas")
    parts, missing, stray = _collect_parts(
        os.path.join(deltas_dir, "parts"), "shape_deltas", expected,
        allow_missing)

    results, warnings, meta_part = {}, [], None
    for _mp, part in parts:
        results.update(part["results"])
        warnings.extend(part.get("warnings", []))
        meta_part = part["meta"]
    if stray:
        warnings.append(f"stray part files ignored: {', '.join(stray)}")
    if meta_part is None:
        raise RuntimeError("No parts found")

    payload = {
        "meta": {
            **{k: v for k, v in meta_part.items() if k != "command"},
            "merged_from": len(parts), "missing": missing,
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "results": results,
        "warnings": warnings,
    }
    outpath = os.path.join(deltas_dir, "shape_deltas.json")
    _write(outpath, payload)
    print(f"Merged {len(parts)}/{len(expected)} -> {outpath}")
    for w in warnings:
        print(f"  warning: {w}")
    return 1 if missing and not allow_missing else 0


STAGES = ("fits-floating", "fits", "closure", "yields", "yield-closure",
          "shape-deltas")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", type=int, required=True,
                        help="mHc study to merge")
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--allow-missing", action="store_true",
                        help="merge whatever parts exist (default: fail on "
                             "missing masspoints)")
    parser.add_argument("--variant", default=None,
                        choices=sorted(interpolation_config.FIT_VARIANTS),
                        help="merge a fit-model variant tree (fits and "
                             "closure stages only)")
    args = parser.parse_args()

    if args.variant is not None and args.stage not in (
            "fits-floating", "fits", "closure"):
        raise ValueError(f"--variant does not apply to stage '{args.stage}' "
                         "(variant tests run fits -> polynomials -> closure only)")
    study = interpolation_config.study(args.mhc)
    expected = [masspoint_name(m, args.mhc) for m in study["all"]]

    if args.stage == "fits-floating":
        rc = merge_fits(args.mhc, "floating", args.allow_missing, expected,
                        study, variant=args.variant)
    elif args.stage == "fits":
        rc = merge_fits(args.mhc, "frozen", args.allow_missing, expected,
                        study, variant=args.variant)
    elif args.stage == "closure":
        rc = merge_closure(args.mhc, args.allow_missing, expected, study,
                           variant=args.variant)
    elif args.stage == "yields":
        rc = merge_yields(args.mhc, args.allow_missing, expected, study)
    elif args.stage == "yield-closure":
        rc = merge_yield_closure(args.mhc, args.allow_missing, expected, study)
    elif args.stage == "shape-deltas":
        rc = merge_shape_deltas(args.mhc, args.allow_missing, expected, study)
    else:
        raise AssertionError(args.stage)
    return rc


if __name__ == "__main__":
    sys.exit(main())
