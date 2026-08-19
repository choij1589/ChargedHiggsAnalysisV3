#!/usr/bin/env python3
"""MC vs interpolation closure of the FINAL binned signal template.

The production limits are set from parametric (interp-signal) templates
everywhere, including at the mass points where real signal MC exists. The
LOO/residual machinery under closure/ validates the interpolation model on
a uniform 100-bin histogram in the fit window; this draws the object the
datacard actually contains:

  - the interpolated signal template, on its own PRODUCTION ADAPTIVE
    BINNING, read straight out of the interp-signal shapes.root, summed
    over the run period's sub-era components;
  - the signal MC filled onto THOSE SAME bin edges. This is not a rebin:
    binned_template_core.getHist is the function makeBinnedTemplates uses
    to build the MC signal component, so calling it with the interp edges
    reproduces the production MC template exactly, on the production
    binning, without a template campaign.

Uncertainties are drawn separately, which is the point of the figure: the
red band on the prediction is the full up/down envelope of every
CMS_interp_* nuisance the datacard carries (scale and res as shape
templates, norm -- and eff_pnet on the ParticleNet arm -- as lnN), while
the MC keeps its own statistical error bars. A discrepancy can then be
attributed to the model or to MC noise.

Two deliberate departures from the production signal build:

  - cap_stat_errors is NOT applied to the MC histogram. Production caps the
    per-bin error at 100% of the content so autoMCStats behaves; here the
    honest MC statistical error is exactly what is being drawn against.
  - the interp nuisance set is DISCOVERED from extra_systematics.json and
    cross-checked against the shapes.root keys, never hardcoded. A new
    family surfaces as a hard error instead of silently vanishing from the
    band.

Scope: the mass points that have MC -- configs/masspoints.json 'baseline'
(78) and 'particlenet' (17), which are exactly the mc_points of the two
scan grids. These points are IN-SAMPLE (the surfaces were fitted using
them), so this is a production-model closure, not an out-of-sample test;
closure/interpolation/loo/ remains the out-of-sample statement.

Outputs (per category, into the point's own interp template dir):
  templates/{mp}/{method}/interp-signal/{era}/{channel}/closure/
      closure.{channel}_{era}.{png,pdf,json}
Group members nest under their seed, as everywhere else.

  python3 python/plotTemplateClosure.py --masspoint MHc130_MA90 \
      --method Baseline --era Run2 --channel SR1E2Mu
"""
import argparse
import json
import os

import numpy as np
import ROOT

import interp_plot_utils
import interpolation_config
import run_period_utils
import srspaths
from binned_template_core import getHist
from template_utils import ensure_positive_integral

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError

INTERP_PREFIX = "CMS_interp_"


def leaf_dir(masspoint, method, era, channel):
    """The point's own interp-signal template dir: seed-level for a group
    seed, nested under the seed for a member (srspaths owns both forms)."""
    seed = interpolation_config.group_seed(masspoint, method)
    if seed == masspoint:
        return seed, srspaths.template_dir(masspoint, method, era, channel,
                                           source="interp-signal")
    return seed, srspaths.interp_member_dir(seed, masspoint, era, channel,
                                            method=method)


def load_json(path, what):
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing -- {what}")
    with open(path) as fh:
        return json.load(fh)


def interp_nuisances(extra_syst, suberas, channel, shape_keys):
    """(shape families, lnN families) of the interpolation nuisances.

    shape: {name: None} -- the Up/Down templates live in shapes.root.
    lnN:   {name: {subera: value}} -- one nuisance carries a column per
           sub-era of the period.

    Every CMS_interp_* entry of extra_systematics.json must be classified,
    and every CMS_interp_* shape key found in shapes.root must be declared
    there. A mismatch is a hard error: it means a nuisance family exists
    that this band does not know how to draw.
    """
    shapes, lnns = {}, {}
    for subera in suberas:
        block = extra_syst["systematics"].get(f"{subera}|{channel}")
        if block is None:
            raise SystemExit(f"extra_systematics: no block for "
                             f"{subera}|{channel}")
        for name, entry in block.items():
            if not name.startswith(INTERP_PREFIX):
                continue
            if entry["type"] == "shape":
                shapes.setdefault(name, None)
            elif entry["type"] == "lnN":
                lnns.setdefault(name, {})[subera] = float(entry["value"])
            else:
                raise SystemExit(f"{name}: unhandled interp nuisance type "
                                 f"{entry['type']!r}")

    declared = set(shapes)
    found = {k for k in shape_keys if k.startswith(INTERP_PREFIX)}
    if found != declared:
        raise SystemExit(
            "interp shape nuisances disagree between shapes.root and "
            f"extra_systematics.json: only in shapes.root {sorted(found - declared)}, "
            f"only in extra_systematics.json {sorted(declared - found)}")
    return shapes, lnns


def read_interp_template(shapes_path, cat, components, shapes, lnns,
                         suberas):
    """Nominal interp template with the assigned uncertainty in its bin
    errors.

    Every CMS_interp_* nuisance is period-level -- ONE nuisance spanning the
    period's sub-eras -- so its sub-era shifts are fully correlated and add
    LINEARLY across them (which summing the per-direction histograms does).
    Different families are independent and add in quadrature.
    """
    fh = ROOT.TFile.Open(shapes_path)
    if not fh or fh.IsZombie():
        raise SystemExit(f"cannot open {shapes_path}")

    def get(name, alt=None):
        """Fetch a histogram, tolerating the member-leaf key quirk.

        A group MEMBER's per-channel shapes.root writes its signal
        variations under keys that omit the underscore between the process
        and the systematic (makeBinnedTemplates.py:955 builds
        f"{sig_comp}{key}" while the seed path writes f"{process}_{key}").
        The HISTOGRAM's own name is correct in both, which is why the merge
        step -- it clones by obj.GetName(), mergeRunPeriodTemplates.py:142
        -- hands Combine the right names and the datacards are unaffected.
        Only these intermediate per-channel member files carry the odd key,
        so accept either form rather than reading a merged file this plot
        does not otherwise need.
        """
        for key in (name, alt):
            if key is None:
                continue
            hist = fh.Get(f"{cat}/{key}")
            if hist:
                hist = hist.Clone(f"c_{key}")
                hist.SetDirectory(0)
                return hist
        raise SystemExit(f"{shapes_path}: missing {cat}/{name}")

    nominal = None
    for comp in components:
        hist = get(comp)
        if nominal is None:
            nominal = hist
            nominal.SetName(f"interp_{cat}")
        else:
            nominal.Add(hist)

    nbins = nominal.GetNbinsX()
    deltas = []
    for name in sorted(shapes):
        totals = {}
        for direction in ("Up", "Down"):
            summed = None
            for comp in components:
                hist = get(f"{comp}_{name}{direction}",
                           f"{comp}{name}{direction}")
                if summed is None:
                    summed = hist
                else:
                    summed.Add(hist)
            totals[direction] = summed
        deltas.append(np.array(
            [0.5 * abs(totals["Up"].GetBinContent(i)
                       - totals["Down"].GetBinContent(i))
             for i in range(1, nbins + 1)]))

    for name in sorted(lnns):
        values = lnns[name]
        shift = np.zeros(nbins)
        for comp, subera in zip(components, suberas):
            if subera not in values:
                raise SystemExit(f"{name}: no lnN value for {subera}")
            hist = get(comp)
            rel = values[subera] - 1.0
            shift += np.array([hist.GetBinContent(i) * rel
                               for i in range(1, nbins + 1)])
        deltas.append(np.abs(shift))

    fh.Close()

    total = (np.sqrt(np.sum(np.square(deltas), axis=0)) if deltas
             else np.zeros(nbins))
    for i in range(1, nbins + 1):
        nominal.SetBinError(i, float(total[i - 1]))
    return nominal


def build_mc_template(masspoint, method, channel, suberas, binning, seed,
                      leaf):
    """Signal MC on the interp template's bin edges, summed over sub-eras.

    Sample-dir resolution mirrors makeBinnedTemplates' interp-signal
    resolver verbatim: the Baseline arm reads the shared per-channel dirs
    (pairing variant applied by srspaths), the ParticleNet arm reads the
    per-mHc shared-scores dir and cuts on the SEED's net at the seed's
    frozen working point -- the net the interp template was built under.
    """
    edges = binning["bin_edges"]
    lo, hi = binning["mass_min"], binning["mass_max"]
    mhc, _ma = srspaths.masspoint_mhc_ma(masspoint)

    threshold, bg_weights, score_key = -999.0, None, None
    if method == "ParticleNet":
        wp = load_json(os.path.join(leaf, "threshold.json"),
                       "ParticleNet working point (run the template step)")
        cat = run_period_utils.category_name(channel, binning["_period"])
        threshold = float(wp["categories"][cat]["threshold"])
        bg_weights = load_json(
            os.path.join(leaf, "background_weights.json"),
            "ParticleNet background weights")["categories"][cat]["weights"]
        score_key = seed

    total = None
    for subera in suberas:
        if method == "ParticleNet":
            basedir = srspaths.mhc_sample_dir(subera, channel, f"MHc{mhc}")
        else:
            basedir = srspaths.sample_dir(subera, channel, masspoint,
                                          "Baseline")
        hist = getHist(basedir, masspoint, edges, lo, hi, "Central",
                       threshold, None, bg_weights, score_key)
        if total is None:
            total = hist
            total.SetName(f"mc_{masspoint}")
        else:
            total.Add(hist)
    # Production floors empty bins so vertical morphing is safe; the stat
    # error cap it also applies is deliberately skipped (see module doc).
    ensure_positive_integral(total)
    return total


def closure_numbers(h_mc, h_interp):
    nbins = h_mc.GetNbinsX()
    mc = np.array([h_mc.GetBinContent(i) for i in range(1, nbins + 1)])
    mc_err = np.array([h_mc.GetBinError(i) for i in range(1, nbins + 1)])
    pr = np.array([h_interp.GetBinContent(i) for i in range(1, nbins + 1)])
    pr_err = np.array([h_interp.GetBinError(i) for i in range(1, nbins + 1)])

    ok = (mc_err > 0) & ((mc > 0) | (pr > 0))
    chi2_stat = float(np.sum(((mc[ok] - pr[ok]) / mc_err[ok]) ** 2))
    chi2_total = float(np.sum(((mc[ok] - pr[ok])
                               / np.hypot(mc_err[ok], pr_err[ok])) ** 2))
    n_mc, n_interp = float(mc.sum()), float(pr.sum())
    return {
        "n_mc": n_mc,
        "n_interp": n_interp,
        "norm_ratio": (n_interp / n_mc) if n_mc > 0 else None,
        "chi2_stat": chi2_stat,
        "chi2_total": chi2_total,
        "ndf": int(ok.sum()),
        # Per-bin arrays so the chi2 above can be re-derived -- and a bad
        # bin located -- without reopening shapes.root.
        "mc": mc.tolist(),
        "mc_stat": mc_err.tolist(),
        "interp": pr.tolist(),
        "interp_unc": pr_err.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--masspoint", required=True)
    parser.add_argument("--method", default="Baseline",
                        choices=["Baseline", "ParticleNet"])
    parser.add_argument("--era", default="Run2", choices=["Run2", "Run3"])
    parser.add_argument("--channel", default="SR1E2Mu",
                        choices=["SR1E2Mu", "SR3Mu"])
    parser.add_argument("--signal-source", default="interp-signal",
                        choices=["interp-signal"],
                        help="frozen: the closure compares MC against the "
                             "parametric production template")
    parser.add_argument("--ratio-range", nargs=2, type=float,
                        default=[0.5, 1.5], metavar=("MIN", "MAX"))
    parser.add_argument("--outdir", default=None,
                        help="override the output dir "
                             "[{template leaf}/closure]")
    args = parser.parse_args()

    seed, leaf = leaf_dir(args.masspoint, args.method, args.era,
                          args.channel)
    cat = run_period_utils.category_name(args.channel, args.era)
    suberas = run_period_utils.RUN_PERIODS[args.era]

    binning = load_json(os.path.join(leaf, "binning.json"),
                        "run the interp-signal template step first")
    if cat not in binning["categories"]:
        raise SystemExit(f"{leaf}/binning.json: no category {cat}")
    cat_binning = dict(binning["categories"][cat])
    cat_binning["_period"] = args.era

    categories = load_json(os.path.join(leaf, "categories.json"),
                           "run the interp-signal template step first")
    processes = categories["categories"][cat]["processes"]
    components = [p["name"] for p in processes if p["is_signal"]]
    comp_suberas = [p["subera"] for p in processes if p["is_signal"]]
    if comp_suberas != list(suberas):
        raise SystemExit(f"{leaf}: signal components {comp_suberas} do not "
                         f"match the {args.era} sub-eras {list(suberas)}")

    extra_syst = load_json(
        os.path.join(leaf,
                     f"extra_systematics.{args.era}.{args.channel}.json"),
        "interp nuisance sidecar (run the interp-signal template step)")

    # Prefer the pre-prune archive when it exists; signal columns are
    # byte-identical in both (printDatacard skips signal), but the archive
    # is the untouched makeBinnedTemplates output.
    shapes_path = os.path.join(leaf, "shapes_original.root")
    if not os.path.exists(shapes_path):
        shapes_path = os.path.join(leaf, "shapes.root")
    if not os.path.exists(shapes_path):
        raise SystemExit(f"{leaf}: no shapes.root")

    fh = ROOT.TFile.Open(shapes_path)
    cat_dir = fh.Get(cat)
    if not cat_dir:
        raise SystemExit(f"{shapes_path}: no directory {cat}")
    prefix = components[0]
    shape_keys = set()
    for key in cat_dir.GetListOfKeys():
        name = key.GetName()
        if not name.startswith(prefix):
            continue
        # Member leaves drop the separating underscore (see get() below).
        stem = name[len(prefix):]
        stem = stem[1:] if stem.startswith("_") else stem
        for direction in ("Up", "Down"):
            if stem.endswith(direction):
                shape_keys.add(stem[:-len(direction)])
                break
    fh.Close()

    shapes, lnns = interp_nuisances(extra_syst, suberas, args.channel,
                                    shape_keys)
    h_interp = read_interp_template(shapes_path, cat, components, shapes,
                                    lnns, suberas)
    h_mc = build_mc_template(args.masspoint, args.method, args.channel,
                             suberas, cat_binning, seed, leaf)

    summary = closure_numbers(h_mc, h_interp)
    summary.update({
        "masspoint": args.masspoint, "method": args.method,
        "category": cat, "group_seed": seed,
        "bin_edges": cat_binning["bin_edges"],
        "interp_shape_nuisances": sorted(shapes),
        "interp_lnN_nuisances": {k: lnns[k] for k in sorted(lnns)},
    })

    outdir = args.outdir or os.path.join(leaf, "closure")
    interp_plot_utils.plot_template_closure(
        cat, args.masspoint, h_mc, h_interp, summary, args.era, outdir,
        ratio_range=tuple(args.ratio_range))
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, f"closure.{cat}.json"), "w") as fh_out:
        json.dump(summary, fh_out, indent=2, sort_keys=True)
        fh_out.write("\n")

    ndf = summary["ndf"]
    print(f"{args.method}/{args.masspoint}/{cat}: "
          f"N_mc={summary['n_mc']:.2f} N_interp={summary['n_interp']:.2f} "
          f"ratio={summary['norm_ratio']:.4f} "
          f"chi2/ndf={(summary['chi2_stat'] / ndf) if ndf else float('nan'):.2f}"
          f" (stat) {(summary['chi2_total'] / ndf) if ndf else float('nan'):.2f}"
          f" (+unc) -> {outdir}")


if __name__ == "__main__":
    main()
