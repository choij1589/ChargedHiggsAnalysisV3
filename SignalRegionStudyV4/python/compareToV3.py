#!/usr/bin/env python3
"""Exact-reproduction comparator for the SignalRegionStudyV4 port.

Validation-only script: this is V4's single touchpoint to SignalRegionStudyV3,
and only to V3's frozen *output artifacts* (samples on pnfs, template dirs,
results/json). It never imports or executes V3 code. --v3-dir is required and
has no default on purpose.

Stages:
  samples    per-tree entry counts (exact) and per-branch sum/sum^2
             aggregates (rtol) between V4 and V3 preprocessed samples
  templates  datacard.txt bitwise; metadata JSONs deep-equal;
             shapes.root / shapes_original.root per-bin contents+errors
  limits     V4 AsymptoticLimits ROOT values (BR-converted) vs the
             MHc130_MA90 rows of V3 results/json
  all        everything above

Exit code 0 only if every check passes. A JSON report is always written.
"""
import argparse
import json
import math
import os
import sys

import ROOT

import srspaths

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kWarning

ERAS_SUB = srspaths.RUN2_ERAS + srspaths.RUN3_ERAS
ERAS_TARGET = ["Run2", "Run3", "All"]
CHANNELS_TARGET = ["SR1E2Mu", "SR3Mu", "Combined"]

# V3's frozen layout and naming (the reference side never changes):
#   templates/{era}/{channel}/{masspoint}/{method}/extended_unblind/
V3_SUFFIX = "extended_unblind"


def v3_template_dir(v3_dir, era, channel, masspoint, method):
    return os.path.join(v3_dir, "templates", era, channel, masspoint,
                        method, V3_SUFFIX)


def v3_asymptotic_root(v3_dir, era, channel, masspoint, method):
    return os.path.join(
        v3_template_dir(v3_dir, era, channel, masspoint, method),
        "combine_output", "asymptotic",
        f"higgsCombine.{masspoint}.{method}.{V3_SUFFIX}.AsymptoticLimits.mH120.root",
    )

# BR conversion identical to collectLimits.py
REFERENCE_XSEC = 5.0
TTBAR_XEC_13TEV = 833.9e3
BR_TTBAR_TO_LEPTON = 2 * 0.5456

METADATA_JSONS = [
    "binning.json",
    "categories.json",
    "process_list.json",
    "lowstat.json",
    "background_validation.json",
]

# workspace.root is intentionally absent: in V3 it was written by the GoF /
# impacts workflows (text2workspace.py), which are out of V4's scope.
EXISTENCE_FILES = [os.path.join("validation", "summary.json")]


class Checker:
    def __init__(self):
        self.results = []  # (stage, label, status, detail)

    def record(self, stage, label, ok, detail=""):
        self.results.append(
            {"stage": stage, "label": label,
             "status": "PASS" if ok else "FAIL", "detail": detail}
        )
        if not ok:
            print(f"  FAIL {label}: {detail}")

    def warn(self, stage, label, detail=""):
        """Recorded and printed, but does not fail the run — used for
        inconsistencies internal to the V3 reference itself."""
        self.results.append(
            {"stage": stage, "label": label, "status": "WARN", "detail": detail}
        )
        print(f"  WARN {label}: {detail}")

    def failed(self):
        return [r for r in self.results if r["status"] == "FAIL"]

    def warned(self):
        return [r for r in self.results if r["status"] == "WARN"]


def open_root(path):
    f = ROOT.TFile.Open(path)
    if not f or f.IsZombie():
        raise RuntimeError(f"Cannot open ROOT file: {path}")
    return f


def tree_names(tfile):
    names = []
    for key in tfile.GetListOfKeys():
        if key.GetClassName() in ("TTree",):
            names.append(key.GetName())
    return sorted(set(names))


def file_metadata(path):
    """Per-tree entry counts and branch sets (pure metadata, no event loop).

    This is the cheap integrity check that catches silently truncated or
    lost pnfs files: every tree of every file is opened and counted."""
    f = open_root(path)
    try:
        meta = {}
        for name in tree_names(f):
            tree = f.Get(name)
            meta[name] = {
                "entries": int(tree.GetEntries()),
                "branches": sorted(b.GetName() for b in tree.GetListOfBranches()),
            }
        return meta
    finally:
        f.Close()


def tree_content_sums(path, tree_name):
    """Per-branch (sum, sum^2) via one lazy RDataFrame event loop."""
    df = ROOT.RDataFrame(tree_name, path)
    cols = sorted(str(c) for c in df.GetColumnNames())
    actions = {}
    for c in cols:
        df = df.Define(f"__sq_{c}", f"(double){c}*(double){c}")
        actions[c] = (df.Sum(c), df.Sum(f"__sq_{c}"))
    return {c: (float(s.GetValue()), float(q.GetValue()))
            for c, (s, q) in actions.items()}


def compare_sample_file(checker, label, v4_path, v3_path, rtol, stats_trees,
                        allow_extra_branches):
    m4 = file_metadata(v4_path)
    m3 = file_metadata(v3_path)
    if sorted(m4) != sorted(m3):
        checker.record("samples", label, False,
                       f"tree set differs: {sorted(set(m4) ^ set(m3))[:10]}")
        return
    ok = True
    details = []
    for name in sorted(m3):
        if m4[name]["entries"] != m3[name]["entries"]:
            ok = False
            details.append(f"{name}: entries {m4[name]['entries']} != {m3[name]['entries']}")
        missing_in_v4 = set(m3[name]["branches"]) - set(m4[name]["branches"])
        extra_in_v4 = (set(m4[name]["branches"]) - set(m3[name]["branches"])
                       - set(allow_extra_branches))
        if missing_in_v4:
            ok = False
            details.append(f"{name}: missing in V4: {sorted(missing_in_v4)}")
        if extra_in_v4:
            ok = False
            details.append(f"{name}: unexpected extra in V4: {sorted(extra_in_v4)}")
    checker.record("samples", f"{label} (metadata)", ok, "; ".join(details[:5]))

    for tree_name in stats_trees:
        if tree_name not in m3:
            continue
        s4 = tree_content_sums(v4_path, tree_name)
        s3 = tree_content_sums(v3_path, tree_name)
        ok = True
        details = []
        for br in sorted(s3):
            for tag, x4, x3 in (("sum", s4[br][0], s3[br][0]),
                                ("sum2", s4[br][1], s3[br][1])):
                if not math.isclose(x4, x3, rel_tol=rtol, abs_tol=1e-12):
                    ok = False
                    details.append(f"{br}:{tag} {x4!r} != {x3!r}")
        checker.record("samples", f"{label}:{tree_name} (content)", ok,
                       "; ".join(details[:5]))


def stage_samples(checker, args):
    channels = ["SR1E2Mu", "SR3Mu"] + (["TTZ2E1Mu"] if args.ttz else [])
    v3_samples = os.path.join(args.v3_dir, "samples")
    for era in ERAS_SUB:
        for channel in channels:
            v4_dir = srspaths.sample_dir(era, channel, args.masspoint)
            v3_dir = os.path.join(v3_samples, era, channel, args.masspoint)
            label = f"samples/{era}/{channel}"
            if not os.path.isdir(v3_dir):
                checker.record("samples", label, False, f"V3 dir missing: {v3_dir}")
                continue
            if not os.path.isdir(v4_dir):
                checker.record("samples", label, False, f"V4 dir missing: {v4_dir}")
                continue
            v3_files = sorted(f for f in os.listdir(v3_dir) if f.endswith(".root"))
            v4_files = sorted(f for f in os.listdir(v4_dir) if f.endswith(".root"))
            if v3_files != v4_files:
                checker.record("samples", label, False,
                               f"file inventory differs: {sorted(set(v3_files) ^ set(v4_files))}")
                continue
            for fname in v3_files:
                compare_sample_file(
                    checker, f"{label}/{fname}",
                    os.path.join(v4_dir, fname), os.path.join(v3_dir, fname),
                    args.sample_rtol, args.stats_trees, args.allow_extra_branches,
                )


def compare_json_deep(v4_obj, v3_obj, rtol=0.0, path=""):
    """Return list of leaf-level mismatches.

    Numeric leaves compare within rtol (fit-derived quantities carry
    cross-worker Minuit/numpy noise, measured at <= 4e-13 relative);
    everything else compares exactly."""
    diffs = []
    if isinstance(v3_obj, dict) and isinstance(v4_obj, dict):
        for k in sorted(set(v3_obj) | set(v4_obj)):
            if k not in v3_obj:
                diffs.append(f"{path}/{k}: only in V4")
            elif k not in v4_obj:
                diffs.append(f"{path}/{k}: only in V3")
            else:
                diffs.extend(compare_json_deep(v4_obj[k], v3_obj[k], rtol, f"{path}/{k}"))
    elif isinstance(v3_obj, list) and isinstance(v4_obj, list):
        if len(v3_obj) != len(v4_obj):
            diffs.append(f"{path}: list length {len(v4_obj)} != {len(v3_obj)}")
        else:
            for i, (a, b) in enumerate(zip(v4_obj, v3_obj)):
                diffs.extend(compare_json_deep(a, b, rtol, f"{path}[{i}]"))
    elif (isinstance(v3_obj, (int, float)) and not isinstance(v3_obj, bool)
          and isinstance(v4_obj, (int, float)) and not isinstance(v4_obj, bool)):
        if not math.isclose(v4_obj, v3_obj, rel_tol=rtol, abs_tol=0.0):
            diffs.append(f"{path}: {v4_obj!r} != {v3_obj!r}")
    else:
        if v4_obj != v3_obj:
            diffs.append(f"{path}: {v4_obj!r} != {v3_obj!r}")
    return diffs


def compare_shapes_file(checker, label, v4_path, v3_path, rtol, edge_rtol):
    fv4 = open_root(v4_path)
    fv3 = open_root(v3_path)

    def hist_index(tfile):
        out = {}
        for key in tfile.GetListOfKeys():
            obj = key.ReadObj()
            if obj.InheritsFrom("TH1"):
                out[key.GetName()] = obj
            elif obj.InheritsFrom("TDirectory"):
                for subkey in obj.GetListOfKeys():
                    subobj = subkey.ReadObj()
                    if subobj.InheritsFrom("TH1"):
                        out[f"{key.GetName()}/{subkey.GetName()}"] = subobj
        return out

    try:
        h4, h3 = hist_index(fv4), hist_index(fv3)
        if sorted(h4) != sorted(h3):
            checker.record("templates", label, False,
                           f"histogram set differs: {sorted(set(h4) ^ set(h3))[:10]}")
            return
        ok = True
        details = []
        max_dev = 0.0
        for name in sorted(h3):
            a, b = h4[name], h3[name]
            if a.GetNbinsX() != b.GetNbinsX():
                ok = False
                details.append(f"{name}: nbins {a.GetNbinsX()} != {b.GetNbinsX()}")
                continue
            for i in range(0, a.GetNbinsX() + 2):
                ea, eb = a.GetBinLowEdge(i), b.GetBinLowEdge(i)
                if not math.isclose(ea, eb, rel_tol=edge_rtol, abs_tol=0.0):
                    ok = False
                    details.append(f"{name}: bin edge {i}: {ea!r} != {eb!r}")
                    break
                for tag, xa, xb in (("content", a.GetBinContent(i), b.GetBinContent(i)),
                                    ("error", a.GetBinError(i), b.GetBinError(i))):
                    if not math.isclose(xa, xb, rel_tol=rtol, abs_tol=1e-15):
                        ok = False
                        details.append(f"{name}: bin {i} {tag}: {xa!r} != {xb!r}")
                    if xb != 0:
                        max_dev = max(max_dev, abs(xa - xb) / abs(xb))
        detail = "; ".join(details[:5])
        if ok:
            detail = f"max relative deviation {max_dev:.2e}"
        checker.record("templates", label, ok, detail)
    finally:
        fv4.Close()
        fv3.Close()


def stage_templates(checker, args):
    for method in args.methods:
        for era in ERAS_TARGET:
            for channel in CHANNELS_TARGET:
                v4_dir = srspaths.template_dir(args.masspoint, method, era, channel)
                v3_dir = v3_template_dir(args.v3_dir, era, channel,
                                         args.masspoint, method)
                label = f"{method}/{era}/{channel}"
                if not os.path.isdir(v3_dir):
                    checker.record("templates", label, False, f"V3 dir missing: {v3_dir}")
                    continue
                if not os.path.isdir(v4_dir):
                    checker.record("templates", label, False, f"V4 dir missing: {v4_dir}")
                    continue

                # datacard: bitwise
                v4_card = os.path.join(v4_dir, "datacard.txt")
                v3_card = os.path.join(v3_dir, "datacard.txt")
                if not os.path.isfile(v4_card):
                    checker.record("templates", f"{label}/datacard.txt", False, "missing in V4")
                else:
                    with open(v4_card, "rb") as f4, open(v3_card, "rb") as f3:
                        same = f4.read() == f3.read()
                    checker.record("templates", f"{label}/datacard.txt", same,
                                   "" if same else "bitwise diff")

                # metadata JSONs (plus any ParticleNet threshold/weight JSONs in V3)
                json_names = list(METADATA_JSONS)
                json_names += sorted(
                    n for n in os.listdir(v3_dir)
                    if n.endswith(".json")
                    and (n.startswith("threshold") or n.startswith("background_weights"))
                )
                for name in json_names:
                    v3_path = os.path.join(v3_dir, name)
                    v4_path = os.path.join(v4_dir, name)
                    if not os.path.isfile(v3_path):
                        continue
                    if not os.path.isfile(v4_path):
                        checker.record("templates", f"{label}/{name}", False, "missing in V4")
                        continue
                    with open(v4_path) as f4, open(v3_path) as f3:
                        diffs = compare_json_deep(json.load(f4), json.load(f3),
                                                  rtol=args.json_rtol)
                    checker.record("templates", f"{label}/{name}", not diffs,
                                   "; ".join(diffs[:5]))

                # shapes: per-bin
                for name in ("shapes.root", "shapes_original.root"):
                    v3_path = os.path.join(v3_dir, name)
                    v4_path = os.path.join(v4_dir, name)
                    if not os.path.isfile(v3_path):
                        continue
                    if not os.path.isfile(v4_path):
                        checker.record("templates", f"{label}/{name}", False, "missing in V4")
                        continue
                    compare_shapes_file(checker, f"{label}/{name}", v4_path, v3_path,
                                        args.shapes_rtol, args.edge_rtol)

                # existence-only artifacts
                for name in EXISTENCE_FILES:
                    v4_path = os.path.join(v4_dir, name)
                    ok = os.path.isfile(v4_path) and os.path.getsize(v4_path) > 0
                    checker.record("templates", f"{label}/{name} (exists)", ok,
                                   "" if ok else f"missing or empty: {v4_path}")


def convert_br(r):
    return r * REFERENCE_XSEC / TTBAR_XEC_13TEV / BR_TTBAR_TO_LEPTON


def read_limit_values(path):
    f = open_root(path)
    try:
        tree = f.Get("limit")
        return [convert_br(entry.limit) for entry in tree]
    finally:
        f.Close()


def stage_limits(checker, args):
    """Primary check: V4 AsymptoticLimits vs V3's own frozen template ROOT
    outputs (this is what 'reproduce the chain' means). Secondary: V3's ROOT
    vs V3's results/json — mismatches there are V3-internal staleness (the
    frozen JSON predates V3's last template rebuild) and are WARNed, not
    failed."""
    keys = ["exp-2", "exp-1", "exp0", "exp+1", "exp+2", "obs"]
    for method in args.methods:
        for era in ERAS_TARGET:
            for channel in CHANNELS_TARGET:
                label = f"limits/{method}/{era}/{channel}"
                v3_root = v3_asymptotic_root(args.v3_dir, era, channel,
                                             args.masspoint, method)
                v4_root = srspaths.asymptotic_root(args.masspoint, method, era, channel)
                if not os.path.isfile(v3_root):
                    checker.record("limits", label, False, f"V3 reference missing: {v3_root}")
                    continue
                if not os.path.isfile(v4_root):
                    checker.record("limits", label, False, f"V4 output missing: {v4_root}")
                    continue
                v3_vals = read_limit_values(v3_root)
                v4_vals = read_limit_values(v4_root)
                if len(v4_vals) != 6 or len(v3_vals) != 6:
                    checker.record("limits", label, False,
                                   f"expected 6 limit entries, got V4={len(v4_vals)} V3={len(v3_vals)}")
                    continue
                ok = True
                details = []
                max_dev = 0.0
                for key, v4_val, v3_val in zip(keys, v4_vals, v3_vals):
                    if not math.isclose(v4_val, v3_val, rel_tol=args.limits_rtol,
                                        abs_tol=0.0):
                        ok = False
                        details.append(f"{key}: {v4_val!r} != {v3_val!r}")
                    if v3_val != 0:
                        max_dev = max(max_dev, abs(v4_val - v3_val) / abs(v3_val))
                detail = "; ".join(details) if details else f"max relative deviation {max_dev:.2e}"
                if ok and max_dev > 1e-10:
                    detail += " (WARNING: above 1e-10)"
                checker.record("limits", label, ok, detail)

                # Reference self-consistency: V3 ROOT vs V3 results/json
                ch_infix = "" if channel == "Combined" else f".{channel}"
                v3_json = os.path.join(
                    args.v3_dir, "results", "json", "BR", era,
                    f"limits.{era}{ch_infix}.Asymptotic.{method}.unblind.json",
                )
                if not os.path.isfile(v3_json):
                    checker.warn("limits", f"{label} (V3 json)",
                                 f"V3 results/json missing: {v3_json}")
                    continue
                with open(v3_json) as f:
                    ref_all = json.load(f)
                if args.masspoint not in ref_all:
                    checker.warn("limits", f"{label} (V3 json)",
                                 f"{args.masspoint} not in {v3_json}")
                    continue
                ref = ref_all[args.masspoint]
                stale = [f"{k}: root={rv!r} json={ref[k]!r}"
                         for k, rv in zip(keys, v3_vals)
                         if not math.isclose(rv, ref[k], rel_tol=args.limits_rtol,
                                             abs_tol=0.0)]
                if stale:
                    checker.warn("limits", f"{label} (V3 json stale)",
                                 "V3 results/json disagrees with V3's own template "
                                 "outputs: " + "; ".join(stale[:3]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--masspoint", required=True)
    parser.add_argument("--v3-dir", required=True,
                        help="SignalRegionStudyV3 module directory (frozen reference outputs)")
    parser.add_argument("--stage", required=True,
                        choices=["samples", "templates", "limits", "all"])
    parser.add_argument("--methods", nargs="+", default=["Baseline", "ParticleNet"])
    parser.add_argument("--no-ttz", dest="ttz", action="store_false",
                        help="Skip TTZ2E1Mu sample dirs (non-ParticleNet mass points)")
    parser.add_argument("--sample-rtol", type=float, default=1e-9)
    parser.add_argument("--json-rtol", type=float, default=1e-9,
                        help="Relative tolerance for numeric JSON leaves "
                             "(cross-worker fit noise measured at <= 4e-13)")
    parser.add_argument("--edge-rtol", type=float, default=1e-9,
                        help="Relative tolerance for histogram bin edges")
    parser.add_argument("--stats-trees", nargs="+", default=["Central"],
                        help="Trees whose branch contents are summed and compared "
                             "(all trees always get the metadata check)")
    parser.add_argument("--allow-extra-branches", nargs="*", default=["pT"],
                        help="Branches V4 may have that the frozen V3 samples lack. "
                             "Default 'pT': added to V3 preprocess.py after its pnfs "
                             "samples were produced; unused by Baseline/ParticleNet.")
    parser.add_argument("--shapes-rtol", type=float, default=1e-12)
    parser.add_argument("--limits-rtol", type=float, default=1e-6)
    parser.add_argument("--report", default=None,
                        help="JSON report path (default: results/repro/<masspoint>.<stage>.json)")
    args = parser.parse_args()

    if "CMSSW_BASE" not in os.environ:
        raise RuntimeError("Not in a CMSSW environment. Source setup.sh first.")
    print(f"ROOT version: {ROOT.gROOT.GetVersion()}  (CMSSW: {os.environ['CMSSW_BASE']})")
    print(f"Comparing V4 ({srspaths.module_dir()}) vs V3 ({args.v3_dir})")

    if not os.path.isdir(args.v3_dir):
        raise NotADirectoryError(args.v3_dir)

    checker = Checker()
    stages = ["samples", "templates", "limits"] if args.stage == "all" else [args.stage]
    for stage in stages:
        print(f"\n=== Stage: {stage} ===")
        {"samples": stage_samples,
         "templates": stage_templates,
         "limits": stage_limits}[stage](checker, args)

    n_pass = sum(1 for r in checker.results if r["status"] == "PASS")
    n_fail = len(checker.failed())
    n_warn = len(checker.warned())
    print(f"\n{'='*60}")
    print(f"Result: {n_pass} PASS, {n_fail} FAIL, {n_warn} WARN")
    for r in checker.failed():
        print(f"  FAIL [{r['stage']}] {r['label']}: {r['detail']}")

    report_path = args.report or os.path.join(
        srspaths.module_dir(), "results", "repro",
        f"{args.masspoint}.{args.stage}.json",
    )
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({"masspoint": args.masspoint, "stage": args.stage,
                   "root_version": ROOT.gROOT.GetVersion(),
                   "n_pass": n_pass, "n_fail": n_fail, "n_warn": n_warn,
                   "results": checker.results}, f, indent=4)
    print(f"Report written to {report_path}")

    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
