#!/usr/bin/env python3
"""
Signal template interpolation test: Voigt vs Double Gaussian.

Validates whether analytic signal templates can be generated at arbitrary
MA values by interpolating fit parameters from known mass points.

Two shape models compared:
  - Voigt: uses existing RooFit parameters (mass, width, sigma)
  - Double Gaussian: fit directly to each MC histogram (mu, sigma1, sigma2, frac)

Two tests per model:
  1. Self-closure: generate from own fit params vs real MC
  2. Leave-one-out: interpolate params from 6 other points vs held-out MC

Test case: MHc=130, 2018, SR1E2Mu (7 MA points: 15, 30, 55, 83, 90, 100, 125)
"""

import json
import os
from math import sqrt

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ROOT
ROOT.gROOT.SetBatch(True)
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from scipy.special import voigt_profile
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MHC = 130
MA_LIST = [15, 30, 55, 83, 90, 100, 125]
HOLDOUT_LIST = [55, 83, 90]

SIGMA_FRACTIONS = np.concatenate([[-10, -7], np.linspace(-5, 5, 16), [7, 10]])  # 20 edges → 19 bins


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------

def calculate_extended_bins(mA, voigt_width):
    """Replicate calculateExtendedBins from makeBinnedTemplates.py."""
    return mA + SIGMA_FRACTIONS * voigt_width


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_mass_point_data(base_dir, mhc, ma_list):
    """Load Voigt fit params, histogram contents/edges/errors for each MA."""
    data = {}
    for ma in ma_list:
        masspoint = f"MHc{mhc}_MA{ma}"
        template_dir = os.path.join(base_dir, masspoint, "Baseline", "extended")

        with open(os.path.join(template_dir, "signal_fit.json")) as f:
            fit = json.load(f)

        rf = ROOT.TFile.Open(os.path.join(template_dir, "shapes.root"), "READ")
        if not rf or rf.IsZombie():
            raise RuntimeError(f"Cannot open shapes.root for {masspoint}")
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

        data[ma] = {
            # Voigt params from RooFit
            "mass":     fit["mass"],
            "width":    fit["width"],
            "sigma":    fit["sigma"],
            # Histogram
            "integral": contents.sum(),
            "bin_edges": bin_edges,
            "contents":  contents,
            "errors":    errors,
        }
    return data


# ---------------------------------------------------------------------------
# Double Gaussian fitting
# ---------------------------------------------------------------------------

def _dg_pdf(x, mu, sigma1, sigma2, frac):
    """Double Gaussian PDF (unit-area) evaluated at points x."""
    return frac * norm.pdf(x, mu, sigma1) + (1.0 - frac) * norm.pdf(x, mu, sigma2)


def _dg_bin_integral(edges, mu, sigma1, sigma2, frac):
    """Integrate double Gaussian in each bin; return array of bin contents (unit area)."""
    n = len(edges) - 1
    contents = np.empty(n)
    for i in range(n):
        lo, hi = edges[i], edges[i + 1]
        g1 = norm.cdf(hi, mu, sigma1) - norm.cdf(lo, mu, sigma1)
        g2 = norm.cdf(hi, mu, sigma2) - norm.cdf(lo, mu, sigma2)
        contents[i] = frac * g1 + (1.0 - frac) * g2
    return contents


def fit_double_gaussian(bin_edges, contents, errors, mu0, sigma0):
    """
    Fit a double Gaussian to a histogram using chi2 minimization.

    Returns dict with keys: mu, sigma1, sigma2, frac  (+ fit success flag).
    Initial guess: narrow component = sigma0, wide = 2*sigma0, frac=0.6.
    """
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths  = bin_edges[1:] - bin_edges[:-1]
    rate    = contents.sum()

    # Fit only bins where content > 0
    mask = contents > 0

    def model(x_centers, mu, sigma1, sigma2, frac):
        """Evaluate DG PDF at bin centers (approximation; good for narrow bins)."""
        frac = np.clip(frac, 0.01, 0.99)
        sigma1 = abs(sigma1)
        sigma2 = abs(sigma2)
        # Ensure sigma1 < sigma2 (narrow < wide)
        if sigma1 > sigma2:
            sigma1, sigma2 = sigma2, sigma1
        pdf = _dg_pdf(x_centers, mu, sigma1, sigma2, frac)
        return pdf * widths * rate  # convert PDF → expected bin content

    p0 = [mu0, sigma0 * 0.7, sigma0 * 1.5, 0.6]
    bounds = (
        [mu0 - 2 * sigma0, 1e-3, sigma0 * 0.5, 0.05],
        [mu0 + 2 * sigma0, sigma0 * 2.0, sigma0 * 5.0, 0.95],
    )

    try:
        sigma_fit = errors[mask]
        sigma_fit = np.where(sigma_fit > 0, sigma_fit, contents[mask] * 0.1)
        popt, _ = curve_fit(
            lambda x, mu, s1, s2, f: model(x, mu, s1, s2, f)[mask],
            centers[mask], contents[mask],
            p0=p0, bounds=bounds,
            sigma=sigma_fit, absolute_sigma=True,
            maxfev=5000,
        )
        mu, sigma1, sigma2, frac = popt
        if sigma1 > sigma2:
            sigma1, sigma2 = sigma2, sigma1
            frac = 1.0 - frac
        return {"mu": mu, "sigma1": sigma1, "sigma2": sigma2, "frac": frac, "ok": True}
    except RuntimeError as e:
        print(f"  DG fit failed: {e} — using fallback")
        return {"mu": mu0, "sigma1": sigma0 * 0.7, "sigma2": sigma0 * 1.5, "frac": 0.6, "ok": False}


def fit_all_double_gaussians(data):
    """Fit a double Gaussian to every mass point's histogram. Stores results in data."""
    for ma, d in data.items():
        sigma0 = sqrt(d["width"]**2 + d["sigma"]**2)
        dg = fit_double_gaussian(d["bin_edges"], d["contents"], d["errors"], d["mass"], sigma0)
        d["dg"] = dg
        status = "OK" if dg["ok"] else "FALLBACK"
        print(f"  MA={ma:3d}  [{status}]  mu={dg['mu']:.3f}  "
              f"sigma1={dg['sigma1']:.3f}  sigma2={dg['sigma2']:.3f}  frac={dg['frac']:.3f}")


# ---------------------------------------------------------------------------
# Histogram generation
# ---------------------------------------------------------------------------

def generate_voigt_hist(mA, width, sigma, rate, bin_edges):
    """Bin-integrated Voigt profile, normalized to rate."""
    contents = np.array([
        max(quad(lambda x: voigt_profile(x - mA, sigma, width),
                 bin_edges[i], bin_edges[i + 1])[0], 0.0)
        for i in range(len(bin_edges) - 1)
    ])
    total = contents.sum()
    return contents * (rate / total) if total > 0 else contents


def generate_dg_hist(mu, sigma1, sigma2, frac, rate, bin_edges):
    """Bin-integrated double Gaussian, normalized to rate."""
    contents = _dg_bin_integral(bin_edges, mu, sigma1, sigma2, frac)
    total = contents.sum()
    return contents * (rate / total) if total > 0 else contents


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_shape_metrics(real_contents, pred_contents):
    """
    KS statistic + relative differences in significant core bins (≥5% of peak).
    Core = bins 2..16 (indices, i.e. the ±5σ uniform region).
    """
    r = real_contents / real_contents.sum() if real_contents.sum() > 0 else real_contents
    p = pred_contents / pred_contents.sum() if pred_contents.sum() > 0 else pred_contents

    core = slice(2, 17)
    peak = r[core].max()
    mask = r[core] >= 0.05 * peak
    rel = np.abs(p[core][mask] / r[core][mask] - 1.0)
    max_rel  = float(rel.max())  if rel.size > 0 else 0.0
    mean_rel = float(rel.mean()) if rel.size > 0 else 0.0

    ks = float(np.max(np.abs(np.cumsum(r) - np.cumsum(p))))
    return max_rel, mean_rel, ks


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison_two(bin_edges, real_contents, real_errors,
                        voigt_contents, dg_contents,
                        title, output_path):
    """Three-panel plot: overlay (top), ratio Voigt (mid), ratio DG (bot)."""
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    widths  = bin_edges[1:] - bin_edges[:-1]

    mv_max, mv_mean, mv_ks = compute_shape_metrics(real_contents, voigt_contents)
    md_max, md_mean, md_ks = compute_shape_metrics(real_contents, dg_contents)
    rv = (voigt_contents.sum() - real_contents.sum()) / real_contents.sum() * 100
    rd = (dg_contents.sum()   - real_contents.sum()) / real_contents.sum() * 100

    fig, axes = plt.subplots(3, 1, figsize=(9, 9),
                             height_ratios=[3, 1, 1], sharex=True)
    fig.subplots_adjust(hspace=0.05)
    ax1, ax2, ax3 = axes

    # Top: overlay
    ax1.bar(centers, real_contents, width=widths, fill=False,
            edgecolor="black", linewidth=1.2, label="Real MC")
    ax1.errorbar(centers, real_contents, yerr=real_errors,
                 fmt="none", ecolor="black", elinewidth=0.8)
    ax1.bar(centers, voigt_contents, width=widths, fill=False,
            edgecolor="red", linestyle="--", linewidth=1.2,
            label=f"Voigt  KS={mv_ks:.3f}  rate={rv:+.1f}%")
    ax1.bar(centers, dg_contents, width=widths, fill=False,
            edgecolor="blue", linestyle=":", linewidth=1.4,
            label=f"DblGaus  KS={md_ks:.3f}  rate={rd:+.1f}%")
    ax1.set_ylabel("Events")
    ax1.set_title(title)
    ax1.legend(fontsize=9)

    # Middle: Voigt ratio
    ratio_v = np.where(real_contents > 0, voigt_contents / real_contents, 1.0)
    rerr    = np.where(real_contents > 0, real_errors / real_contents, 0.0)
    ax2.bar(centers, ratio_v, width=widths, fill=False, edgecolor="red", linewidth=1.2)
    ax2.errorbar(centers, np.ones(len(centers)), yerr=rerr,
                 fmt="none", ecolor="gray", elinewidth=0.8)
    ax2.axhline(1.0, color="gray", linestyle="--")
    ax2.set_ylabel("Voigt/Real")
    ax2.set_ylim(0.0, 2.0)
    ax2.text(0.02, 0.88, f"max={mv_max*100:.0f}%  mean={mv_mean*100:.0f}%",
             transform=ax2.transAxes, fontsize=8)

    # Bottom: DG ratio
    ratio_d = np.where(real_contents > 0, dg_contents / real_contents, 1.0)
    ax3.bar(centers, ratio_d, width=widths, fill=False, edgecolor="blue", linewidth=1.2)
    ax3.errorbar(centers, np.ones(len(centers)), yerr=rerr,
                 fmt="none", ecolor="gray", elinewidth=0.8)
    ax3.axhline(1.0, color="gray", linestyle="--")
    ax3.set_ylabel("DG/Real")
    ax3.set_xlabel("$m_A$ [GeV]")
    ax3.set_ylim(0.0, 2.0)
    ax3.text(0.02, 0.88, f"max={md_max*100:.0f}%  mean={md_mean*100:.0f}%",
             transform=ax3.transAxes, fontsize=8)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Interpolation helper
# ---------------------------------------------------------------------------

def make_splines(train_data, keys):
    """Return {key: CubicSpline} from train_data dict keyed by MA."""
    x = np.array(sorted(train_data.keys()), dtype=float)
    splines = {}
    for key in keys:
        y = np.array([train_data[m][key] for m in sorted(train_data.keys())])
        splines[key] = CubicSpline(x, y)
    return splines


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    workdir = os.environ.get("WORKDIR", ".")
    base_dir = os.path.join(workdir, "SignalRegionStudyV2", "templates", "2018", "SR1E2Mu")
    if not os.path.isdir(base_dir):
        base_dir = os.path.join("templates", "2018", "SR1E2Mu")
    if not os.path.isdir(base_dir):
        raise RuntimeError("Cannot find templates directory. Run from SignalRegionStudyV2.")

    print(f"Loading data from: {base_dir}")
    data = load_mass_point_data(base_dir, MHC, MA_LIST)

    print("\nFitting double Gaussians to each mass point histogram:")
    fit_all_double_gaussians(data)

    plot_dir = "test"

    # -----------------------------------------------------------------------
    # Self-closure test
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("SELF-CLOSURE TEST")
    print(f"{'MA':>4}  {'Voigt KS':>9}  {'Voigt mean_rel':>14}  "
          f"{'DG KS':>7}  {'DG mean_rel':>11}  {'DG fit OK':>9}")
    print("=" * 80)

    for ma in MA_LIST:
        d = data[ma]
        dg = d["dg"]

        voigt_c = generate_voigt_hist(d["mass"], d["width"], d["sigma"], d["integral"], d["bin_edges"])
        dg_c    = generate_dg_hist(dg["mu"], dg["sigma1"], dg["sigma2"], dg["frac"], d["integral"], d["bin_edges"])

        _, mv_mean, mv_ks = compute_shape_metrics(d["contents"], voigt_c)
        _, md_mean, md_ks = compute_shape_metrics(d["contents"], dg_c)

        print(f"{ma:4d}  {mv_ks:9.4f}  {mv_mean*100:14.1f}%  "
              f"{md_ks:7.4f}  {md_mean*100:11.1f}%  {'YES' if dg['ok'] else 'NO':>9}")

        plot_comparison_two(
            d["bin_edges"], d["contents"], d["errors"], voigt_c, dg_c,
            title=f"Self-closure: MHc{MHC} MA{ma}",
            output_path=os.path.join(plot_dir, f"interp_selfclosure_MHc{MHC}_MA{ma}.png"),
        )

    # -----------------------------------------------------------------------
    # Leave-one-out interpolation test
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("LEAVE-ONE-OUT INTERPOLATION TEST")
    print(f"{'MA':>4}  {'Voigt KS':>9}  {'Voigt rate%':>11}  "
          f"{'DG KS':>7}  {'DG rate%':>8}  {'DG mean_rel':>11}")
    print("=" * 80)

    voigt_keys = ["mass", "width", "sigma", "integral"]
    dg_keys    = ["dg_mu", "dg_sigma1", "dg_sigma2", "dg_frac", "integral"]

    # Flatten DG params into top-level data entries for spline fitting
    for ma, d in data.items():
        d["dg_mu"]     = d["dg"]["mu"]
        d["dg_sigma1"] = d["dg"]["sigma1"]
        d["dg_sigma2"] = d["dg"]["sigma2"]
        d["dg_frac"]   = d["dg"]["frac"]

    for holdout_ma in HOLDOUT_LIST:
        train = {ma: d for ma, d in data.items() if ma != holdout_ma}
        d = data[holdout_ma]

        # Voigt interpolation
        vs = make_splines(train, voigt_keys)
        i_mass  = float(vs["mass"](holdout_ma))
        i_width = float(vs["width"](holdout_ma))
        i_sigma = float(vs["sigma"](holdout_ma))
        i_rate  = float(vs["integral"](holdout_ma))
        voigt_c = generate_voigt_hist(i_mass, i_width, i_sigma, i_rate, d["bin_edges"])

        # Double Gaussian interpolation
        ds = make_splines(train, dg_keys)
        i_mu     = float(ds["dg_mu"](holdout_ma))
        i_sigma1 = abs(float(ds["dg_sigma1"](holdout_ma)))
        i_sigma2 = abs(float(ds["dg_sigma2"](holdout_ma)))
        i_frac   = float(np.clip(ds["dg_frac"](holdout_ma), 0.01, 0.99))
        i_rate_dg = float(ds["integral"](holdout_ma))  # same spline as Voigt
        if i_sigma1 > i_sigma2:
            i_sigma1, i_sigma2 = i_sigma2, i_sigma1
            i_frac = 1.0 - i_frac
        dg_c = generate_dg_hist(i_mu, i_sigma1, i_sigma2, i_frac, i_rate_dg, d["bin_edges"])

        _, mv_mean, mv_ks = compute_shape_metrics(d["contents"], voigt_c)
        _, md_mean, md_ks = compute_shape_metrics(d["contents"], dg_c)
        v_rate_err = (voigt_c.sum()  - d["integral"]) / d["integral"] * 100
        d_rate_err = (dg_c.sum()     - d["integral"]) / d["integral"] * 100

        print(f"{holdout_ma:4d}  {mv_ks:9.4f}  {v_rate_err:+11.2f}%  "
              f"{md_ks:7.4f}  {d_rate_err:+8.2f}%  {md_mean*100:11.1f}%")
        print(f"       Voigt interp:  mA={i_mass:.3f} (true {d['mass']:.3f})  "
              f"width={i_width:.3f}  sigma={i_sigma:.3f}")
        print(f"       DG interp:     mu={i_mu:.3f}   sigma1={i_sigma1:.3f}  "
              f"sigma2={i_sigma2:.3f}  frac={i_frac:.3f}")

        plot_comparison_two(
            d["bin_edges"], d["contents"], d["errors"], voigt_c, dg_c,
            title=f"Interpolation holdout: MHc{MHC} MA{holdout_ma}",
            output_path=os.path.join(plot_dir, f"interp_leaveoneout_MHc{MHC}_MA{holdout_ma}.png"),
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
