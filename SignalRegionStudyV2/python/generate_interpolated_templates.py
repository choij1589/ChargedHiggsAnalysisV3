#!/usr/bin/env python3
"""
Generate interpolated signal templates using Double Gaussian shape model.

For MHc = 85, 115, 145 GeV where few MA points are available, interpolate
Double Gaussian fit parameters (mu, sigma1, sigma2, frac) and signal rate
from existing mass points to produce templates at new MA values.

Usage:
  python3 python/generate_interpolated_templates.py [--mhc 85] [--era 2018] [--channel SR1E2Mu]
  python3 python/generate_interpolated_templates.py --mhc all   # runs 85, 115, 145

Output per new mass point:
  templates/{era}/{channel}/MHc{X}_MA{Y}/Baseline/extended/
    shapes_dg.root      # signal histogram (Double Gaussian, central only)
    signal_fit_dg.json  # fitted DG params for record
    dg_validation.png   # shape comparison plot

The output shapes_dg.root contains only the central signal histogram.
Systematics are NOT generated here (reserved for future work).
"""

import argparse
import json
import os
from math import sqrt

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ROOT
ROOT.gROOT.SetBatch(True)
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Configuration: existing and new mass points per MHc
# ---------------------------------------------------------------------------

CONFIG = {
    85:  {
        "existing": [15, 70, 80],
        "new":      [25, 35, 45, 55],
    },
    115: {
        "existing": [15, 27, 87, 110],
        "new":      [40, 55, 70],
    },
    145: {
        "existing": [15, 35, 92, 140],
        "new":      [50, 65, 80],
    },
}

SIGMA_FRACTIONS = np.concatenate([[-10, -7], np.linspace(-5, 5, 16), [7, 10]])


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def calculate_extended_bins(mA, voigt_width):
    return mA + SIGMA_FRACTIONS * voigt_width


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mass_point(template_dir, masspoint):
    """Load Voigt fit params, histogram from a single template directory."""
    with open(os.path.join(template_dir, "signal_fit.json")) as f:
        fit = json.load(f)

    rf = ROOT.TFile.Open(os.path.join(template_dir, "shapes.root"), "READ")
    if not rf or rf.IsZombie():
        raise RuntimeError(f"Cannot open shapes.root in {template_dir}")
    h = rf.Get(masspoint)
    if not h:
        raise RuntimeError(f"Histogram '{masspoint}' not found")
    h.SetDirectory(0)
    rf.Close()

    nbins = h.GetNbinsX()
    bin_edges = np.array(
        [h.GetBinLowEdge(i + 1) for i in range(nbins)]
        + [h.GetBinLowEdge(nbins) + h.GetBinWidth(nbins)]
    )
    contents = np.array([h.GetBinContent(i + 1) for i in range(nbins)])
    errors   = np.array([h.GetBinError(i + 1)   for i in range(nbins)])

    voigt_width = sqrt(fit["width"]**2 + fit["sigma"]**2)
    return {
        "mass":        fit["mass"],
        "width":       fit["width"],
        "sigma":       fit["sigma"],
        "voigt_width": voigt_width,
        "integral":    float(contents.sum()),
        "bin_edges":   bin_edges,
        "contents":    contents,
        "errors":      errors,
    }


def load_mhc_data(base_dir, mhc, ma_list, era, channel):
    data = {}
    for ma in ma_list:
        masspoint = f"MHc{mhc}_MA{ma}"
        tdir = os.path.join(base_dir, era, channel, masspoint, "Baseline", "extended")
        if not os.path.isdir(tdir):
            raise RuntimeError(f"Template directory not found: {tdir}")
        data[ma] = load_mass_point(tdir, masspoint)
    return data


# ---------------------------------------------------------------------------
# Double Gaussian fitting
# ---------------------------------------------------------------------------

def fit_double_gaussian(bin_edges, contents, errors, mu0, sigma0):
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths  = bin_edges[1:] - bin_edges[:-1]
    rate    = contents.sum()
    mask    = contents > 0

    def model(x, mu, sigma1, sigma2, frac):
        frac   = np.clip(frac, 0.01, 0.99)
        sigma1 = abs(sigma1)
        sigma2 = abs(sigma2)
        if sigma1 > sigma2:
            sigma1, sigma2 = sigma2, sigma1
        pdf = frac * norm.pdf(x, mu, sigma1) + (1.0 - frac) * norm.pdf(x, mu, sigma2)
        return pdf * widths * rate

    p0     = [mu0, sigma0 * 0.7, sigma0 * 1.5, 0.75]
    bounds = (
        [mu0 - 2 * sigma0, 1e-3,          sigma0 * 0.5, 0.05],
        [mu0 + 2 * sigma0, sigma0 * 2.0,  sigma0 * 5.0, 0.95],
    )
    try:
        sig = errors[mask]
        sig = np.where(sig > 0, sig, contents[mask] * 0.1)
        popt, _ = curve_fit(
            lambda x, mu, s1, s2, f: model(x, mu, s1, s2, f)[mask],
            centers[mask], contents[mask],
            p0=p0, bounds=bounds, sigma=sig,
            absolute_sigma=True, maxfev=5000,
        )
        mu, sigma1, sigma2, frac = popt
        if sigma1 > sigma2:
            sigma1, sigma2 = sigma2, sigma1
            frac = 1.0 - frac
        return {"mu": mu, "sigma1": float(sigma1), "sigma2": float(sigma2),
                "frac": float(frac), "ok": True}
    except RuntimeError as e:
        print(f"    DG fit failed ({e}), using fallback")
        return {"mu": mu0, "sigma1": sigma0 * 0.7, "sigma2": sigma0 * 1.5,
                "frac": 0.75, "ok": False}


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def make_spline(x_arr, y_arr):
    """Adaptive-degree spline: cubic for ≥4 points, quadratic for 3, linear for 2."""
    k = min(3, len(x_arr) - 1)
    return make_interp_spline(x_arr, y_arr, k=k)


def build_splines(data):
    """Return dict of splines for each interpolated parameter."""
    x = np.array(sorted(data.keys()), dtype=float)
    keys = ["mass", "voigt_width", "integral",
            "dg_mu", "dg_sigma1", "dg_sigma2", "dg_frac"]
    splines = {}
    for key in keys:
        y = np.array([data[ma][key] for ma in sorted(data.keys())])
        splines[key] = make_spline(x, y)
    return splines


# ---------------------------------------------------------------------------
# Histogram generation
# ---------------------------------------------------------------------------

def dg_bin_integral(edges, mu, sigma1, sigma2, frac):
    n = len(edges) - 1
    c = np.empty(n)
    for i in range(n):
        g1 = norm.cdf(edges[i+1], mu, sigma1) - norm.cdf(edges[i], mu, sigma1)
        g2 = norm.cdf(edges[i+1], mu, sigma2) - norm.cdf(edges[i], mu, sigma2)
        c[i] = frac * g1 + (1.0 - frac) * g2
    return c


def generate_dg_hist(mu, sigma1, sigma2, frac, rate, bin_edges):
    contents = dg_bin_integral(bin_edges, mu, sigma1, sigma2, frac)
    total = contents.sum()
    return contents * (rate / total) if total > 0 else contents


# ---------------------------------------------------------------------------
# ROOT file output
# ---------------------------------------------------------------------------

def save_dg_template(masspoint, bin_edges, contents, outdir):
    """Save a TH1D signal histogram to shapes_dg.root."""
    os.makedirs(outdir, exist_ok=True)
    nbins = len(bin_edges) - 1
    edges_arr = np.array(bin_edges, dtype=np.float64)

    rf = ROOT.TFile.Open(os.path.join(outdir, "shapes_dg.root"), "RECREATE")
    h = ROOT.TH1D(masspoint, masspoint, nbins, edges_arr)
    for i, c in enumerate(contents):
        h.SetBinContent(i + 1, max(float(c), 0.0))
        h.SetBinError(i + 1, 0.0)  # analytic shape; stat error assigned separately
    h.SetDirectory(rf)
    rf.Write()
    rf.Close()


# ---------------------------------------------------------------------------
# Metrics and plotting
# ---------------------------------------------------------------------------

def ks_stat(real, pred):
    r = real / real.sum() if real.sum() > 0 else real
    p = pred / pred.sum() if pred.sum() > 0 else pred
    return float(np.max(np.abs(np.cumsum(r) - np.cumsum(p))))


def mean_rel(real, pred):
    r = real / real.sum() if real.sum() > 0 else real
    p = pred / pred.sum() if pred.sum() > 0 else pred
    core = slice(2, 17)
    peak = r[core].max()
    mask = r[core] >= 0.05 * peak
    if not mask.any():
        return 0.0
    return float(np.abs(p[core][mask] / r[core][mask] - 1.0).mean())


def plot_single(bin_edges, real_c, real_e, pred_c, title, subtitle, outpath):
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths  = bin_edges[1:] - bin_edges[:-1]
    rate_err = (pred_c.sum() - real_c.sum()) / real_c.sum() * 100 if real_c.sum() > 0 else 0

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7),
                                    height_ratios=[3, 1], sharex=True)
    fig.subplots_adjust(hspace=0.05)

    ax1.bar(centers, real_c, width=widths, fill=False,
            edgecolor="black", linewidth=1.2, label="Real MC")
    ax1.errorbar(centers, real_c, yerr=real_e,
                 fmt="none", ecolor="black", elinewidth=0.8)
    ax1.bar(centers, pred_c, width=widths, fill=False,
            edgecolor="blue", linestyle="--", linewidth=1.4,
            label=f"DG  KS={ks_stat(real_c,pred_c):.3f}  rate={rate_err:+.1f}%")
    ax1.set_ylabel("Events")
    ax1.set_title(f"{title}\n{subtitle}")
    ax1.legend(fontsize=9)

    ratio = np.where(real_c > 0, pred_c / real_c, 1.0)
    rerr  = np.where(real_c > 0, real_e / real_c, 0.0)
    ax2.bar(centers, ratio, width=widths, fill=False, edgecolor="blue", linewidth=1.2)
    ax2.errorbar(centers, np.ones(len(centers)), yerr=rerr,
                 fmt="none", ecolor="gray", elinewidth=0.8)
    ax2.axhline(1.0, color="gray", linestyle="--")
    ax2.set_ylabel("DG / Real")
    ax2.set_xlabel("$m_A$ [GeV]")
    ax2.set_ylim(0.0, 2.0)
    mr = mean_rel(real_c, pred_c)
    ax2.text(0.02, 0.88, f"mean_rel={mr*100:.0f}%", transform=ax2.transAxes, fontsize=8)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_new_template(bin_edges, pred_c, mhc, ma, mA_fit, outpath):
    """Diagnostic plot for a newly generated (no real MC) template."""
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths  = bin_edges[1:] - bin_edges[:-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(centers, pred_c, width=widths, fill=False,
           edgecolor="blue", linewidth=1.4)
    ax.set_xlabel("$m_A$ [GeV]")
    ax.set_ylabel("Events")
    ax.set_title(f"Interpolated template: MHc{mhc} MA{ma}\n"
                 f"(DG, fitted mA={mA_fit:.2f} GeV, rate={pred_c.sum():.3f})")
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_overlay_all(data, new_ma_list, new_templates, mhc, outpath):
    """Overlay existing + new interpolated templates for a given MHc."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Existing (real MC, normalized to unit area)
    for ma in sorted(data.keys()):
        d = data[ma]
        centers = 0.5 * (d["bin_edges"][:-1] + d["bin_edges"][1:])
        widths  = d["bin_edges"][1:] - d["bin_edges"][:-1]
        norm_c  = d["contents"] / d["contents"].sum()
        ax.step(np.append(d["bin_edges"][:-1], d["bin_edges"][-1]),
                np.append(norm_c, norm_c[-1]),
                where="post", color="black", alpha=0.6, linewidth=1.2,
                label=f"MA={ma} (real)" if ma == sorted(data.keys())[0] else f"MA={ma}")

    # New interpolated (blue dashed)
    for ma, (edges, contents) in zip(new_ma_list, new_templates):
        norm_c = contents / contents.sum()
        ax.step(np.append(edges[:-1], edges[-1]),
                np.append(norm_c, norm_c[-1]),
                where="post", color="royalblue", linestyle="--", alpha=0.8,
                linewidth=1.2,
                label=f"MA={ma} (interp)" if ma == new_ma_list[0] else f"MA={ma} (interp)")

    ax.set_xlabel("$m_A$ [GeV]")
    ax.set_ylabel("Normalized events")
    ax.set_title(f"MHc={mhc}: existing (solid) + interpolated (dashed) templates")

    # Build legend with proper labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, ncol=2)

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved overlay: {outpath}")


# ---------------------------------------------------------------------------
# Per-MHc pipeline
# ---------------------------------------------------------------------------

def run_mhc(mhc, era, channel, base_dir, plot_dir):
    cfg = CONFIG[mhc]
    existing = cfg["existing"]
    new_ma_list = cfg["new"]

    print(f"\n{'='*70}")
    print(f"MHc = {mhc} GeV  |  era = {era}  |  channel = {channel}")
    print(f"  Existing MA: {existing}")
    print(f"  New MA:      {new_ma_list}")
    print(f"{'='*70}")

    # Load existing templates
    data = load_mhc_data(base_dir, mhc, existing, era, channel)

    # Fit double Gaussians
    print("\n  Double Gaussian fits:")
    print(f"  {'MA':>4}  {'mu':>8}  {'sigma1':>8}  {'sigma2':>8}  {'frac':>6}  {'rate':>8}  {'OK':>4}")
    for ma in existing:
        d = data[ma]
        dg = fit_double_gaussian(d["bin_edges"], d["contents"], d["errors"],
                                 d["mass"], d["voigt_width"])
        d["dg"]       = dg
        d["dg_mu"]    = dg["mu"]
        d["dg_sigma1"] = dg["sigma1"]
        d["dg_sigma2"] = dg["sigma2"]
        d["dg_frac"]   = dg["frac"]
        print(f"  {ma:4d}  {dg['mu']:8.3f}  {dg['sigma1']:8.3f}  "
              f"{dg['sigma2']:8.3f}  {dg['frac']:6.3f}  {d['integral']:8.3f}  "
              f"{'YES' if dg['ok'] else 'NO':>4}")

    # Leave-one-out validation (only if ≥3 training points after removal, i.e. ≥4 existing)
    do_loo = len(existing) >= 4
    if do_loo:
        print(f"\n  Leave-one-out validation:")
        print(f"  {'MA':>4}  {'DG KS':>7}  {'mean_rel':>9}  {'rate_err%':>10}")
        for holdout in existing[1:-1]:  # skip boundary points
            train = {ma: d for ma, d in data.items() if ma != holdout}
            sp = build_splines(train)
            d  = data[holdout]

            i_mu     = float(sp["dg_mu"](holdout))
            i_s1     = abs(float(sp["dg_sigma1"](holdout)))
            i_s2     = abs(float(sp["dg_sigma2"](holdout)))
            i_frac   = float(np.clip(sp["dg_frac"](holdout), 0.01, 0.99))
            i_rate   = float(sp["integral"](holdout))
            i_vw     = float(sp["voigt_width"](holdout))
            if i_s1 > i_s2:
                i_s1, i_s2 = i_s2, i_s1
                i_frac = 1.0 - i_frac

            pred = generate_dg_hist(i_mu, i_s1, i_s2, i_frac, i_rate, d["bin_edges"])
            ks   = ks_stat(d["contents"], pred)
            mr   = mean_rel(d["contents"], pred)
            re   = (pred.sum() - d["integral"]) / d["integral"] * 100
            print(f"  {holdout:4d}  {ks:7.4f}  {mr*100:9.1f}%  {re:+10.2f}%")

            plot_single(
                d["bin_edges"], d["contents"], d["errors"], pred,
                title=f"LOO validation: MHc{mhc} MA{holdout}",
                subtitle=f"KS={ks:.3f}  mean_rel={mr*100:.0f}%  rate={re:+.1f}%",
                outpath=os.path.join(plot_dir, f"interp_loo_MHc{mhc}_MA{holdout}.png"),
            )
            print(f"    Saved: interp_loo_MHc{mhc}_MA{holdout}.png")
    else:
        print(f"\n  (Only {len(existing)} existing points — skipping leave-one-out)")

    # Self-closure check (DG from own fit vs MC)
    print(f"\n  Self-closure check:")
    print(f"  {'MA':>4}  {'DG KS':>7}  {'mean_rel':>9}")
    for ma in existing:
        d  = data[ma]
        dg = d["dg"]
        pred = generate_dg_hist(dg["mu"], dg["sigma1"], dg["sigma2"],
                                dg["frac"], d["integral"], d["bin_edges"])
        ks = ks_stat(d["contents"], pred)
        mr = mean_rel(d["contents"], pred)
        print(f"  {ma:4d}  {ks:7.4f}  {mr*100:9.1f}%")

        plot_single(
            d["bin_edges"], d["contents"], d["errors"], pred,
            title=f"Self-closure: MHc{mhc} MA{ma}",
            subtitle=f"KS={ks:.3f}  mean_rel={mr*100:.0f}%",
            outpath=os.path.join(plot_dir, f"interp_selfclosure_MHc{mhc}_MA{ma}.png"),
        )
        print(f"    Saved: interp_selfclosure_MHc{mhc}_MA{ma}.png")

    # Build splines from ALL existing points
    splines = build_splines(data)

    # Generate new templates
    print(f"\n  Generating {len(new_ma_list)} new templates:")
    print(f"  {'MA':>4}  {'mA_fit':>8}  {'sigma1':>8}  {'sigma2':>8}  {'frac':>6}  {'rate':>8}")
    new_templates = []  # list of (bin_edges, contents) for overlay plot

    for new_ma in new_ma_list:
        masspoint = f"MHc{mhc}_MA{new_ma}"

        i_mA   = float(splines["mass"](new_ma))
        i_vw   = float(splines["voigt_width"](new_ma))
        i_mu   = float(splines["dg_mu"](new_ma))
        i_s1   = abs(float(splines["dg_sigma1"](new_ma)))
        i_s2   = abs(float(splines["dg_sigma2"](new_ma)))
        i_frac = float(np.clip(splines["dg_frac"](new_ma), 0.01, 0.99))
        i_rate = float(splines["integral"](new_ma))
        if i_s1 > i_s2:
            i_s1, i_s2 = i_s2, i_s1
            i_frac = 1.0 - i_frac

        bin_edges = calculate_extended_bins(i_mA, i_vw)
        contents  = generate_dg_hist(i_mu, i_s1, i_s2, i_frac, i_rate, bin_edges)
        new_templates.append((bin_edges, contents))

        print(f"  {new_ma:4d}  {i_mA:8.3f}  {i_s1:8.3f}  {i_s2:8.3f}  "
              f"{i_frac:6.3f}  {i_rate:8.3f}")

        # Save ROOT file
        tdir = os.path.join(base_dir, era, channel, masspoint, "Baseline", "extended")
        save_dg_template(masspoint, bin_edges, contents, tdir)

        # Save DG params as JSON
        dg_params = {
            "method": "double_gaussian_interpolated",
            "ma_nominal": new_ma,
            "mass": i_mA,
            "voigt_width_proxy": i_vw,
            "sigma1": i_s1,
            "sigma2": i_s2,
            "frac": i_frac,
            "rate": i_rate,
            "interpolated_from": existing,
        }
        with open(os.path.join(tdir, "signal_fit_dg.json"), "w") as f:
            json.dump(dg_params, f, indent=2)

        # Diagnostic plot
        plot_new_template(
            bin_edges, contents, mhc, new_ma, i_mA,
            outpath=os.path.join(plot_dir, f"interp_new_MHc{mhc}_MA{new_ma}.png"),
        )
        print(f"    Saved: interp_new_MHc{mhc}_MA{new_ma}.png  → {tdir}")

    # Overlay: existing + new
    plot_overlay_all(
        data, new_ma_list, new_templates, mhc,
        outpath=os.path.join(plot_dir, f"interp_overlay_MHc{mhc}.png"),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate interpolated DG signal templates")
    parser.add_argument("--mhc",     default="all",
                        help="MHc value (85, 115, 145, or 'all')")
    parser.add_argument("--era",     default="2018")
    parser.add_argument("--channel", default="SR1E2Mu")
    args = parser.parse_args()

    workdir = os.environ.get("WORKDIR", ".")
    base_dir = os.path.join(workdir, "SignalRegionStudyV2", "templates")
    if not os.path.isdir(base_dir):
        base_dir = "templates"
    if not os.path.isdir(base_dir):
        raise RuntimeError("Cannot find templates/ directory. Run from SignalRegionStudyV2.")

    plot_dir = "test"

    if args.mhc == "all":
        mhc_list = [85, 115, 145]
    else:
        mhc_list = [int(args.mhc)]

    for mhc in mhc_list:
        if mhc not in CONFIG:
            raise ValueError(f"MHc={mhc} not in CONFIG. Supported: {list(CONFIG.keys())}")
        run_mhc(mhc, args.era, args.channel, base_dir, plot_dir)

    print("\nAll done.")


if __name__ == "__main__":
    main()
