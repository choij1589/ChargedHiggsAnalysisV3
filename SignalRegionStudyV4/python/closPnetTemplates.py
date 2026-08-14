#!/usr/bin/env python3
"""ParticleNet template closure: direct MC vs the interpolated template,
ABSOLUTELY normalized.

The shape and yield closures report residuals as numbers; this draws the
thing they summarize. For every ParticleNet mass point: the signal MC that
survives its group's score cut, overlaid with the template the production
model builds at that mA, scaled to

    N_pred = SUM_era [ N_baseline(era, mA) x eps_prod(era, mA) ]

The absolute normalization is the point -- the Baseline closure plots are
shape-only (RooPlot renormalizes the pdf to the data), which cannot show a
yield failure. Here the ratio panel (interp / MC) tests the shape AND the
normalization together, against the band this analysis actually assigns:
Baseline yield-model error (+) CMS_interp_eff_pnet on the normalization,
CMS_interp_res_pnet as a per-bin shape variation.

Each point is drawn in its PRODUCTION configuration: assigned to its
nearest seed, cut with that seed's frozen threshold, inside that seed's
window, with the eps that production ships -- the frozen eps_model.json
polynomial through ALL THREE anchors. Deliberately NOT leave-one-out (LOO
belongs to uncertainty derivation; a closure plot shows the model that is
actually built). A consequence worth reading correctly: a 3-point
quadratic passes exactly through its anchors, so at mA = 85/90/95 the
normalization closure reduces to the Baseline yield model alone; the
points that genuinely test the ParticleNet layer are MHc115_MA87 and
MHc145_MA92.

Outputs: closure/pnet/MHc{X}/template_closure.json and
closure/pnet/MHc{X}/plots/templates/{channel}_{period}/{masspoint}.png.

  python3 python/closPnetTemplates.py --mhc MHc115 [--nbins 30]
"""
import argparse
import json
import os
from collections import OrderedDict

import numpy as np

import ROOT

import interpolation_config
import pnet_interp_config as pic
import srspaths
from dcb_fit_utils import build_model
from pnet_interp_config import ANCHOR_MA, DEFAULT_WP, PERIODS, STUDY_CHANNEL
from template_utils import build_particlenet_score
from plotter import ComparisonCanvas  # Common/Tools

ROOT.gROOT.SetBatch(True)
ROOT.gErrorIgnoreLevel = ROOT.kError


def nearest_seed(mA, seeds):
    """Production grouping: nearest seed, ties to the LOWER one."""
    return min(seeds, key=lambda s: (abs(s - mA), s))


def load_yields(mhc, wp):
    path = os.path.join(srspaths.pnet_closure_dir(mhc), "yield_interp.json")
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing; run closPnetYields.py first")
    with open(path) as fh:
        payload = json.load(fh)
    for e in payload["results"].values():
        if e.get("wp") != wp:
            raise RuntimeError(f"{path}: wp={e.get('wp')!r} != {wp!r}")
    return payload["results"]


def load_eps_model(mhc, wp):
    path = pic.eps_model_path(mhc)
    if not os.path.exists(path):
        raise SystemExit(f"{path} missing; run fitPnetEpsModel.py first")
    with open(path) as fh:
        payload = json.load(fh)
    if payload["meta"]["working_point"] != wp:
        raise RuntimeError(f"{path}: working point "
                           f"{payload['meta']['working_point']!r} != {wp!r}")
    return payload["model"]


def mc_hist(basedirs, mp, seed, threshold, bg_weights, lo, hi, nbins, name):
    """Signal MC surviving the SEED's cut, summed over a period's sub-eras."""
    hist = ROOT.TH1D(name, "", nbins, lo, hi)
    hist.Sumw2()
    hist.SetDirectory(0)
    formula = build_particlenet_score(seed, bg_weights)
    for basedir in basedirs:
        path = os.path.join(basedir, f"{mp}.root")
        if not os.path.exists(path):
            continue
        tmp = f"{name}_tmp"
        rdf = (ROOT.RDataFrame("Central", path)
               .Filter(f"mass >= {lo} && mass <= {hi}")
               .Define("score_PN", formula)
               .Filter(f"score_PN >= {threshold}"))
        model = ROOT.RDF.TH1DModel(tmp, "", nbins, lo, hi)
        part = rdf.Histo1D(model, "mass", "weight").GetValue()
        hist.Add(part)
    return hist


def _shape_hist(params, mass, nbins, lo, hi, norm, tag):
    """pdf -> TH1 normalized to `norm`, with the bin errors ZEROED.

    RooAbsPdf::createHistogram returns a histogram with no Sumw2, so ROOT
    hands back sqrt(content) bin errors on the UNNORMALIZED pdf values, and
    Scale() then multiplies those by the (large) normalization factor. The
    result is relative "errors" of 2-90 -- Poisson noise on a model curve.
    A pdf has no statistical error; the real uncertainty is assigned
    separately in pred_hist below."""
    pdf, keep = build_model(f"pred_{tag}", mass, params)
    hist = pdf.createHistogram(f"h_{tag}", mass,
                               ROOT.RooFit.Binning(nbins, lo, hi))
    hist.SetDirectory(0)
    integral = hist.Integral()
    if integral > 0:
        hist.Scale(norm / integral)
    for b in range(1, nbins + 1):
        hist.SetBinError(b, 0.0)
    del keep, pdf
    return hist


def pred_hist(cat_polys, mA, lo, hi, nbins, n_pred, rel_norm, res_unc, name):
    """Interpolated template scaled to N_pred, carrying the ASSIGNED band.

    The band is the uncertainty this analysis actually assigns to the
    prediction, so the ratio panel answers the question that matters: does
    the direct MC sit inside the uncertainty we quote?

      normalization  rel_norm  = Baseline yield-model error (+) CMS_interp_eff_pnet
      shape          per bin, from scaling sigmaL/R by (1 +- CMS_interp_res_pnet)
    """
    params = {p: float(interpolation_config.eval_param(rec, mA))
              for p, rec in cat_polys.items()}
    mass = ROOT.RooRealVar("mass_pred", "mass", lo, hi)

    nominal = _shape_hist(params, mass, nbins, lo, hi, n_pred, name)
    var = {}
    for direction, factor in (("up", 1.0 + res_unc), ("dn", 1.0 - res_unc)):
        shifted = dict(params)
        for key in ("sigmaL", "sigmaR"):
            shifted[key] = params[key] * factor
        var[direction] = _shape_hist(shifted, mass, nbins, lo, hi, n_pred,
                                     f"{name}_{direction}")

    for b in range(1, nbins + 1):
        c = nominal.GetBinContent(b)
        shape = 0.5 * abs(var["up"].GetBinContent(b) - var["dn"].GetBinContent(b))
        nominal.SetBinError(b, float(np.hypot(c * rel_norm, shape)))
    return nominal, params


def run_mhc(mhc, channels, args, frozen, unc, summary, warnings):
    group = pic.trained_masspoints(mhc)
    seeds = sorted({pic.mA_of(mp) for mp in group
                    if pic.mA_of(mp) in ANCHOR_MA})
    yields = load_yields(mhc, args.wp)
    eps_model = load_eps_model(mhc, args.wp)
    polys, _path = interpolation_config.load_shape_polynomials(
        pic.mhc_int(mhc))
    plots_base = os.path.join(srspaths.pnet_closure_dir(mhc),
                              "plots", "templates")

    for channel in channels:
        study_ch = STUDY_CHANNEL[channel]
        for period, suberas in PERIODS.items():
            cat = f"{channel}_{period}"
            cat_key = f"{study_ch}_{period}"
            if cat_key not in polys:
                warnings.append(f"[{mhc}/{cat}] no polynomials for {cat_key}")
                continue
            cat_polys = polys[cat_key]
            basedirs = [srspaths.mhc_sample_dir(e, channel, mhc)
                        for e in suberas]

            for mp in group:
                mA = pic.mA_of(mp)
                seed_mA = nearest_seed(mA, seeds)
                seed = f"{mhc}_MA{seed_mA}"
                key = f"{mhc}/{cat}/seed{seed_mA}"
                rec = frozen.get(key)
                entry = yields.get(key)
                eps_rec = eps_model.get(key)
                if rec is None or entry is None or eps_rec is None:
                    warnings.append(f"[{key}] missing WP, yield or eps "
                                    "record")
                    continue
                lo, hi = rec["mass_window"]

                n_mc = n_pred = 0.0
                err_model = 0.0
                for era in suberas:
                    pt = entry["points"].get(f"{mp}/{era}")
                    era_eps = eps_rec["eras"].get(era)
                    if pt is None or era_eps is None:
                        continue
                    # Production eps: the frozen all-anchor polynomial. A
                    # quadratic through three points is exact at those
                    # points, so at an anchor this equals the measured eps.
                    eps_prod = pic.eval_eps(era_eps, mA)
                    n_mc += pt["n_cut"]
                    n_pred += pt["n_base_pred"] * eps_prod
                    # Baseline yield-model error, scaled by the same eps.
                    err_model += (pt["err_base_pred"] * eps_prod) ** 2
                err_model = float(np.sqrt(err_model))
                if n_mc <= 0 or n_pred <= 0:
                    warnings.append(f"[{key}/{mp}] non-positive yields")
                    continue

                tag = f"{mhc}_{cat}_{mp}"
                h_mc = mc_hist(basedirs, mp, seed, rec["threshold"],
                               rec["bg_weights"], lo, hi, args.nbins,
                               f"mc_{tag}")
                # Assigned uncertainty on the prediction: the Baseline
                # yield-model error and the two ParticleNet nuisances.
                eff_unc = unc["norm"][channel][suberas[0]]["value"]
                res_unc = unc["res"][channel][period]["value"]
                rel_norm = float(np.hypot(err_model / n_pred, eff_unc))
                h_pr, params = pred_hist(cat_polys, float(mA), lo, hi,
                                         args.nbins, n_pred, rel_norm,
                                         res_unc, f"pr_{tag}")

                outdir = os.path.join(plots_base, cat)
                os.makedirs(outdir, exist_ok=True)
                config = {
                    "era": period, "channel": channel, "masspoint": mp,
                    "xTitle": "m(#mu#mu) [GeV]", "yTitle": "Events",
                    "rTitle": "interp / MC", "rRange": [0.5, 1.5],
                    "yRange": [0.0, 1.6 * max(h_mc.GetMaximum(),
                                              h_pr.GetMaximum())],
                    "legend": [0.62, 0.62, 0.90, 0.80],
                }
                try:
                    # incl is the numerator of the ratio and the stack is
                    # the denominator, so the interpolated template is
                    # passed as incl: ratio = interp / MC, with the grey
                    # band showing the MC statistical error it is
                    # measured against.
                    canvas = ComparisonCanvas(
                        h_pr, OrderedDict([("direct MC", h_mc)]), config)
                    canvas.drawPadUp()
                    canvas.drawPadDown()
                    canvas.canv.SaveAs(os.path.join(outdir, f"{mp}.png"))
                    canvas.canv.Close()
                except Exception as exc:                     # noqa: BLE001
                    warnings.append(f"[{key}/{mp}] plot failed: {exc}")

                # Bin-level closure, independent of the plot library.
                # Two chi2: against MC stat alone (how well the model
                # reproduces MC), and against MC stat (+) the assigned
                # uncertainty (whether the closure is within what we
                # quote).
                chi2 = chi2_tot = 0.0
                ndf = 0
                for b in range(1, args.nbins + 1):
                    m, p = h_mc.GetBinContent(b), h_pr.GetBinContent(b)
                    e_mc, e_pr = h_mc.GetBinError(b), h_pr.GetBinError(b)
                    if e_mc > 0 and (m > 0 or p > 0):
                        chi2 += ((m - p) / e_mc) ** 2
                        chi2_tot += ((m - p) / np.hypot(e_mc, e_pr)) ** 2
                        ndf += 1
                summary[f"{mhc}/{cat}/{mp}"] = {
                    "mA": mA, "seed_mA": seed_mA,
                    "is_anchor": mA in ANCHOR_MA,
                    "n_mc": n_mc, "n_pred": n_pred,
                    "norm_ratio": n_pred / n_mc,
                    "chi2": chi2, "ndf": ndf,
                    "chi2_per_ndf": (chi2 / ndf) if ndf else None,
                    "chi2_per_ndf_with_syst": (chi2_tot / ndf) if ndf else None,
                    "rel_norm_assigned": rel_norm,
                    "res_unc_assigned": res_unc,
                    "window": [lo, hi],
                }
                print(f"  {mhc}/{cat}/{mp:16s} seed{seed_mA} "
                      f"N_mc={n_mc:8.2f} N_pred={n_pred:8.2f} "
                      f"ratio={n_pred / n_mc:6.4f} "
                      f"chi2/ndf={summary[f'{mhc}/{cat}/{mp}']['chi2_per_ndf']:6.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mhc", default="all",
                        help="comma-separated mHc studies, or 'all'")
    parser.add_argument("--channels", default="SR1E2Mu,SR3Mu")
    parser.add_argument("--nbins", type=int, default=30)
    parser.add_argument("--output", default=None,
                        help="override output path (single --mhc only); "
                             "default closure/pnet/MHc{X}/template_closure.json")
    parser.add_argument("--wp", default=DEFAULT_WP,
                        help="frozen working-point label; must match the "
                             "one the yield shards and eps model carry")
    args = parser.parse_args()

    mhcs = (pic.pn_mhc_list() if args.mhc == "all"
            else [m.strip() for m in args.mhc.split(",") if m.strip()])
    channels = [c.strip() for c in args.channels.split(",") if c.strip()]
    if args.output and len(mhcs) != 1:
        parser.error("--output requires a single --mhc")

    frozen = pic.wp_lookup(args.wp, mhcs)
    if not frozen:
        raise SystemExit("no threshold_wp shards; run measPnetThresholds.py "
                         "first")

    unc_path = srspaths.pnet_uncertainties_path()
    if not os.path.exists(unc_path):
        raise SystemExit(f"{unc_path} missing; run "
                         "exportPnetUncertainties.py first")
    with open(unc_path) as fh:
        unc = json.load(fh)

    all_summary = OrderedDict()
    for mhc in mhcs:
        summary, warnings = OrderedDict(), []
        run_mhc(mhc, channels, args, frozen, unc, summary, warnings)
        out = args.output or os.path.join(
            srspaths.pnet_closure_dir(mhc), "template_closure.json")
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w") as fh:
            json.dump({"results": summary, "warnings": warnings},
                      fh, indent=2)
        if warnings:
            print(f"{mhc}: {len(warnings)} warning(s); first: {warnings[0]}")
        print(f"Wrote {out}")
        all_summary.update(summary)

    print("\n" + "=" * 78)
    print("PNET TEMPLATE CLOSURE -- interpolated / direct MC")
    print("=" * 78)
    for label, sel in (("anchors (eps exact -> tests the Baseline yield model)",
                        lambda r: r["is_anchor"]),
                       ("validation MA87/MA92 (eps genuinely interpolated)",
                        lambda r: not r["is_anchor"])):
        pts = [r for r in all_summary.values() if sel(r)]
        if not pts:
            continue
        nr = np.array([r["norm_ratio"] for r in pts])
        c2 = np.array([r["chi2_per_ndf"] for r in pts if r["chi2_per_ndf"]])
        print(f"\n{label}  [{len(pts)} points]")
        print(f"  norm ratio   median={np.median(nr):.4f}  "
              f"|dev| p90={np.percentile(np.abs(nr - 1), 90):.4f}  "
              f"max={np.abs(nr - 1).max():.4f}")
        print(f"  chi2/ndf     MC stat only : median={np.median(c2):.2f}  "
              f"p90={np.percentile(c2, 90):.2f}  max={c2.max():.2f}")
        ct = np.array([r["chi2_per_ndf_with_syst"] for r in pts
                       if r.get("chi2_per_ndf_with_syst")])
        if len(ct):
            print(f"               + assigned unc: median={np.median(ct):.2f}  "
                  f"p90={np.percentile(ct, 90):.2f}  max={ct.max():.2f}")
    print("\nPlots under closure/pnet/MHc{X}/plots/templates/"
          "{channel}_{period}/")


if __name__ == "__main__":
    main()
