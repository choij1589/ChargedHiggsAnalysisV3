#!/usr/bin/env python3
"""Leave-one-out closure of the shape-delta parametrisation (stage 2).

A parametric signal template carries its shape systematics as three
dimensionless deltas per (era|channel, systematic, direction) — dm, dsig,
dN — measured at every simulated mass point by measInterpShapeDeltas.py and
parametrised in mA by fitInterpShapeDeltas.py. In production every
simulated point is a fit anchor (meta.held_out_ma == [] everywhere), so
stage 2's own held-out closure is empty and the only genuine interpolation
test of the delta transfer was the single-point end-to-end study at
MHc160_MA90 (docs/interpolation/EXPERIMENTS.md V1).

This script extends that test to the full grid: for every series, drop one
anchor at a time, refit with the SAME production procedure, and evaluate
the refitted polynomial at the dropped point —

    resid = predicted(mA_i) - measured(mA_i)

dimensionless, in the units of the delta itself. It mirrors the LOO
protocol the shape and yield interpolation uncertainties are derived from.

The refit is production's own fitInterpShapeDeltas.fit_series (which in
turn is fitInterpPolynomials.select_order / weighted_polyfit): the up-only
[0, 1] F-test ladder and the two-pass rescaling of the errors to the
first-pass residual RMS. Nothing about the fit is re-implemented here, and
gate A below verifies that by refitting with NO point dropped and demanding
the stored production coefficients back.

The LOO itself reads only fits/MHc{X}/shape_deltas/delta_model.json — the
stored `points` of each series are the (mA, value, error) anchors production
fitted, with the DELTA_ERR_FLOOR already applied. shape_deltas.json is
opened for one thing: the measured `paired` flag that separates the
event-migrating trees from the weight-only ones in the summary.

Outputs fits/MHc{X}/shape_deltas/loo_closure.json per study and the global
fits/loo_closure_summary.json, plus a printed summary.

  python3 closInterpShapeDeltasLOO.py [--mhc 145,160] [--pdf-members]
"""
import argparse
import datetime
import json
import os
import sys

import numpy as np

import interpolation_config
import srspaths
from fitInterpShapeDeltas import fit_series

# Two ways to name the "kinematic" trees, and they do NOT agree, so both
# subsets are summarised:
#
#   * by NAME -- the JES / JER / unclustered set V1 quoted as the worst delta
#     transfer. Kept because the AN quotes this subset.
#   * by MEASUREMENT -- stage 1 records, per series, whether the variation
#     tree holds the same events as Central (`paired`). An unpaired tree is
#     one that re-ran the selection, so its delta carries the full
#     uncorrelated error (measInterpShapeDeltas.deltas). This is the
#     authoritative marker of event migration.
#
# Measured against shape_deltas.json, the name list is wrong both ways:
# CMS_scale_met_unclustered_energy_* is paired everywhere (the MET shift
# moves no event in or out of the mass window), while CMS_scale_m_* is
# unpaired everywhere and CMS_scale_e_*/CMS_res_e_*/ps_isr/ps_fsr are
# unpaired in some categories. The `paired` flag is read from
# shape_deltas.json, the only thing this script needs it for.
KINEMATIC_SYST_PREFIXES = ("CMS_scale_j_", "CMS_res_j_",
                           "CMS_scale_met_unclustered_energy_")

# Gate A: the no-drop refit must return the stored production fit.
GATE_A_RTOL = 1e-9
GATE_A_ATOL = 1e-12

# Gate B: global median |resid| per quantity against the V1 single-point
# end-to-end delta closure (EXPERIMENTS.md V1, 6576 series). LOO drops a
# genuine anchor from a sparser grid, so it may sit above V1 — what is
# checked is that it is not off by orders of magnitude, which would mean a
# bug rather than a measurement.
GATE_B_REFERENCE = {"dm": 1.4e-7, "dsig": 7.3e-6, "dN": 1.7e-5}
GATE_B_FACTOR = 100.0

MIN_LOO_POINTS = 2  # points that must REMAIN after the drop


def is_kinematic(syst):
    return syst.startswith(KINEMATIC_SYST_PREFIXES)


def load_unpaired_map(mhc):
    """{(key, syst, direction): True} for every series unpaired ANYWHERE.

    Stage 1 decides pairing per (mass point, category, systematic,
    direction); a series is called unpaired if any of its own anchors was.
    Missing shape_deltas.json is a hard error: silently falling back to the
    name list would report the subset this script exists to correct.
    """
    path = os.path.join(srspaths.interpolation_fits_dir(mhc),
                        "shape_deltas", "shape_deltas.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — it carries the measured `paired` flag; run "
            f"measInterpShapeDeltas.py --mhc {mhc} first")
    with open(path) as f:
        results = json.load(f)["results"]
    unpaired = {}
    for rec in results.values():
        for key, cat in rec["cats"].items():
            for bucket in ("systs", "pdf_members"):
                for syst, directions in cat.get(bucket, {}).items():
                    for direction, node in directions.items():
                        if node is None:
                            continue
                        ident = (key, syst, direction)
                        unpaired[ident] = (unpaired.get(ident, False)
                                           or not node.get("paired", True))
    return unpaired


def iter_series(model, buckets):
    """(key, bucket, syst, direction, quantity, record) over the model."""
    for key in sorted(model):
        entry = model[key]
        for bucket in buckets:
            for syst in sorted(entry.get(bucket, {})):
                for direction in sorted(entry[bucket][syst]):
                    recs = entry[bucket][syst][direction]
                    for quantity in interpolation_config.DELTA_QUANTITIES:
                        rec = recs.get(quantity)
                        if rec is not None:
                            yield key, bucket, syst, direction, quantity, rec


def check_reproduction(rec, refit, ident):
    """Gate A on one series: same order, same coefficients."""
    if int(refit["order"]) != int(rec["order"]):
        return (f"{ident}: order {refit['order']} from the refit vs "
                f"{rec['order']} stored")
    stored = np.array(rec["coeffs"], float)
    got = np.array(refit["coeffs"], float)
    if stored.shape != got.shape:
        return (f"{ident}: {got.size} coefficients from the refit vs "
                f"{stored.size} stored")
    bad = ~np.isclose(got, stored, rtol=GATE_A_RTOL, atol=GATE_A_ATOL)
    if bad.any():
        return (f"{ident}: coeffs {got[bad].tolist()} from the refit vs "
                f"{stored[bad].tolist()} stored")
    return None


def loo_series(pts):
    """(mA, measured, predicted, refit order) for every droppable anchor."""
    rows = []
    for i in range(len(pts)):
        remaining = pts[:i] + pts[i + 1:]
        if len(remaining) < MIN_LOO_POINTS:
            continue
        refit = fit_series(remaining)
        mA, measured = float(pts[i][0]), float(pts[i][1])
        predicted = float(np.polyval(np.array(refit["coeffs"], float), mA))
        rows.append((mA, measured, predicted, int(refit["order"])))
    return rows


def run_study(mhc, buckets):
    """LOO over one mHc study; returns (payload, flat residual records)."""
    path = os.path.join(srspaths.interpolation_fits_dir(mhc),
                        "shape_deltas", "delta_model.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run fitInterpShapeDeltas.py --mhc {mhc} first")
    with open(path) as f:
        payload = json.load(f)

    meta = payload["meta"]
    if meta["orders"] != interpolation_config.DELTA_ORDERS:
        raise RuntimeError(
            f"{path}: stored order ladder {meta['orders']} differs from the "
            f"current DELTA_ORDERS {interpolation_config.DELTA_ORDERS}; the "
            "refit would not be the production fit")
    if meta["err_floor"] != dict(interpolation_config.DELTA_ERR_FLOOR):
        raise RuntimeError(
            f"{path}: stored error floors {meta['err_floor']} differ from the "
            f"current DELTA_ERR_FLOOR {interpolation_config.DELTA_ERR_FLOOR}")

    unpaired_map = load_unpaired_map(mhc)
    series_out, flat, gate_a_failures = [], [], []
    n_skipped_series, n_skipped_points = 0, 0
    for key, bucket, syst, direction, quantity, rec in iter_series(
            payload["model"], buckets):
        ident = f"MHc{mhc} {key} {syst}/{direction}/{quantity}"
        unpaired = unpaired_map.get((key, syst, direction))
        if unpaired is None:
            raise RuntimeError(
                f"{ident}: fitted in delta_model.json but absent from "
                "shape_deltas.json — the two stages disagree on the series set")
        pts = [tuple(p) for p in rec["points"]]

        # Gate A — the no-drop refit must be the production fit.
        failure = check_reproduction(rec, fit_series(pts), ident)
        if failure:
            gate_a_failures.append(failure)

        rows = loo_series(pts)
        n_skipped_points += len(pts) - len(rows)
        if not rows:
            n_skipped_series += 1
            continue
        resid = [pred - mc for _ma, mc, pred, _o in rows]
        series_out.append({
            "key": key, "bucket": bucket, "syst": syst,
            "direction": direction, "quantity": quantity,
            "kinematic": is_kinematic(syst), "unpaired": unpaired,
            "npoints": len(pts), "prod_order": int(rec["order"]),
            "mA": [r[0] for r in rows],
            "mc": [r[1] for r in rows],
            "model": [r[2] for r in rows],
            "resid": resid,
            "loo_order": [r[3] for r in rows],
        })
        for (mA, mc, pred, _o), d in zip(rows, resid):
            flat.append({"mhc": mhc, "mA": mA, "key": key, "bucket": bucket,
                         "syst": syst, "direction": direction,
                         "quantity": quantity, "kinematic": is_kinematic(syst),
                         "unpaired": unpaired,
                         "mc": mc, "model": pred, "resid": d})

    out = {
        "meta": {
            "mhc": mhc,
            "fit_ma": meta["fit_ma"],
            "orders": meta["orders"],
            "err_floor": meta["err_floor"],
            "buckets": list(buckets),
            "n_series": len(series_out),
            "n_residuals": len(flat),
            "n_skipped_series": n_skipped_series,
            "n_skipped_points": n_skipped_points,
            "gate_a_failures": gate_a_failures,
            "gate_a_passed": not gate_a_failures,
            "source": path,
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "series": series_out,
    }
    return out, flat, gate_a_failures


def summarize(flat, label):
    """Per-quantity median/p95/worst of |resid| over a set of residuals."""
    summary = {}
    for quantity in interpolation_config.DELTA_QUANTITIES:
        rows = [r for r in flat if r["quantity"] == quantity]
        if not rows:
            continue
        mag = np.abs(np.array([r["resid"] for r in rows], float))
        worst = rows[int(np.argmax(mag))]
        summary[quantity] = {
            "label": label,
            "n": len(rows),
            "median_abs": float(np.median(mag)),
            "p95_abs": float(np.percentile(mag, 95)),
            "max_abs": float(mag.max()),
            "worst": worst,
        }
    return summary


def print_summary(summary, title):
    print(f"\n{title}")
    header = (f"  {'quantity':<8} {'n':>7} {'median|d|':>11} {'p95|d|':>11} "
              f"{'max|d|':>11}   worst series")
    print(header)
    print("  " + "-" * (len(header) - 2))
    for quantity, s in summary.items():
        w = s["worst"]
        ident = (f"MHc{w['mhc']} mA={w['mA']:g} {w['key']} {w['syst']} "
                 f"{w['direction']} (model {w['model']:+.4g} vs mc "
                 f"{w['mc']:+.4g})")
        print(f"  {quantity:<8} {s['n']:>7d} {s['median_abs']:>11.2e} "
              f"{s['p95_abs']:>11.2e} {s['max_abs']:>11.2e}   {ident}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mhc", default=None,
                        help="comma-separated mHc studies (default: all)")
    parser.add_argument("--pdf-members", action="store_true",
                        help="also LOO the 100 PDF replica series per key "
                             "(noise-dominated; summarised separately)")
    args = parser.parse_args()

    mhcs = ([int(m) for m in args.mhc.split(",")] if args.mhc
            else interpolation_config.mhc_grid())
    buckets = ("systs", "pdf_members") if args.pdf_members else ("systs",)

    flat_all, gate_a_failures, per_study = [], [], {}
    for mhc in mhcs:
        out, flat, failures = run_study(mhc, buckets)
        outpath = os.path.join(srspaths.interpolation_fits_dir(mhc),
                               "shape_deltas", "loo_closure.json")
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        with open(outpath, "w") as f:
            json.dump(out, f, indent=2)
        m = out["meta"]
        print(f"Wrote {outpath}  ({m['n_series']} series, "
              f"{m['n_residuals']} residuals, {m['n_skipped_points']} points "
              f"skipped, {m['n_skipped_series']} series skipped)")
        flat_all.extend(flat)
        gate_a_failures.extend(failures)
        per_study[mhc] = {k: v for k, v in m.items() if k != "gate_a_failures"}

    named = [r for r in flat_all if r["bucket"] == "systs"]
    pdf = [r for r in flat_all if r["bucket"] == "pdf_members"]

    print(f"\nGate A — no-drop refit reproduces the production fit: "
          f"{'PASS' if not gate_a_failures else 'FAIL'}")
    for failure in gate_a_failures[:20]:
        print(f"  {failure}")
    if gate_a_failures:
        print(f"  ... {len(gate_a_failures)} failures in total")
        raise SystemExit("Gate A failed — the LOO refit is not the production "
                         "fit; fix that before trusting any LOO number.")

    overall = summarize(named, "all named systematics")
    kinematic = summarize([r for r in named if r["kinematic"]],
                          "JES/JER/unclustered by name (the V1 subset)")
    unpaired = summarize([r for r in named if r["unpaired"]],
                         "measured-unpaired (event-migrating) trees")

    print_summary(overall, f"LOO delta closure over {len(mhcs)} mHc studies "
                           f"({len(named)} residuals, named systematics):")
    print_summary(kinematic, "Restricted to JES / JER / unclustered by name "
                             "(the subset V1 quoted):")
    print_summary(unpaired, "Restricted to the trees stage 1 measured as "
                            "UNPAIRED (genuine event migration):")
    if pdf:
        print_summary(summarize(pdf, "pdf replica members"),
                      f"PDF replica members ({len(pdf)} residuals), separately:")

    print("\n  worst 10 series overall (by max |resid|):")
    by_series = {}
    for r in named:
        k = (r["mhc"], r["key"], r["syst"], r["direction"], r["quantity"])
        if abs(r["resid"]) > abs(by_series.get(k, {"resid": 0.0})["resid"]):
            by_series[k] = r
    worst10 = sorted(by_series.values(), key=lambda r: -abs(r["resid"]))[:10]
    header = (f"    {'mHc':>4} {'mA':>5} {'era|channel':<24} "
              f"{'systematic':<44} {'dir':<5} {'qty':<5} "
              f"{'model':>11} {'mc':>11} {'resid':>11}")
    print(header)
    print("    " + "-" * (len(header) - 4))
    for r in worst10:
        print(f"    {r['mhc']:>4d} {r['mA']:>5g} {r['key']:<24} "
              f"{r['syst']:<44} {r['direction']:<5} {r['quantity']:<5} "
              f"{r['model']:>+11.4g} {r['mc']:>+11.4g} {r['resid']:>+11.4g}")

    # Gate B — scale against the V1 single-point end-to-end delta closure.
    gate_b, gate_b_passed = {}, True
    print("\nGate B — median |resid| vs the V1 end-to-end reference:")
    for quantity, ref in GATE_B_REFERENCE.items():
        if quantity not in overall:
            continue
        median = overall[quantity]["median_abs"]
        ratio = median / ref
        ok = 1.0 / GATE_B_FACTOR <= ratio <= GATE_B_FACTOR
        gate_b_passed &= ok
        gate_b[quantity] = {"median_abs": median, "v1_reference": ref,
                            "ratio": ratio, "passed": bool(ok)}
        print(f"  {quantity:<5} median {median:.2e}  V1 {ref:.2e}  "
              f"ratio {ratio:6.2f}x  {'ok' if ok else 'OUT OF SCALE'}")
    print(f"  Gate B: {'PASS' if gate_b_passed else 'FAIL'} "
          f"(tolerance {GATE_B_FACTOR:g}x either way)")

    summary_path = os.path.join(srspaths.interpolation_fits_dir(),
                                "loo_closure_summary.json")
    payload = {
        "meta": {
            "mhc": mhcs,
            "buckets": list(buckets),
            "n_residuals_named": len(named),
            "n_residuals_pdf": len(pdf),
            "gate_a_passed": True,
            "gate_b_passed": bool(gate_b_passed),
            "command": " ".join(sys.argv),
            "date": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "per_study": per_study,
        "overall": overall,
        "kinematic": kinematic,
        "unpaired": unpaired,
        "pdf_members": summarize(pdf, "pdf replica members") if pdf else {},
        "gate_b": gate_b,
        "worst10": worst10,
    }
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {summary_path}")

    if not gate_b_passed:
        raise SystemExit("Gate B failed — the LOO residual scale disagrees "
                         "with the V1 end-to-end closure by more than "
                         f"{GATE_B_FACTOR:g}x; suspect a bug, not a discovery.")


if __name__ == "__main__":
    main()
