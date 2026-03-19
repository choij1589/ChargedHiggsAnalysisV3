#!/usr/bin/env python3
"""
Cross-MHc rate ratio validation — PCHIP and quadratic fit comparison.

Tests the hypothesis: rate(MHc, MA) ≈ f(MHc) × g(MA)  [factorization]

For each MHc, two models are fit to the normalised rate vs MA:
  1. PCHIP  — monotone-preserving interpolation (no chi², passes through points)
  2. Quadratic polynomial (degree 2) — chi²/ndf from weighted least-squares fit

Cross-MHc comparison: MHc=130's model (PCHIP or QuadFit) is used as an
external predictor for the other MHc values; chi²/ndf is computed with no
free parameters (ndf = number of non-anchor points).

Usage:
  python3 python/test_rate_ratio.py [--era 2018] [--channel SR1E2Mu]
"""

import argparse
import ctypes
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ROOT
ROOT.gROOT.SetBatch(True)
from scipy.interpolate import PchipInterpolator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MHC_MA_MAP = {
    70:  [15, 18, 40, 55, 65],
    85:  [15, 70, 80],
    100: [15, 24, 60, 75, 95],
    115: [15, 27, 87, 110],
    130: [15, 30, 55, 83, 90, 100, 125],
    145: [15, 35, 92, 140],
    160: [15, 50, 85, 98, 120, 135, 155],
}

ANCHOR_MA     = 15    # normalise all curves to this MA
REFERENCE_MHC = 130   # densest MHc used as cross-MHc predictor

COLORS = ["#e41a1c", "#ff7f00", "#e6ab02", "#4daf4a",
          "#377eb8", "#984ea3", "#a65628"]


# ---------------------------------------------------------------------------
# Data loading — returns (integral, stat_error) per mass point
# ---------------------------------------------------------------------------

def load_rates(base_dir, mhc_ma_map, method="Baseline", binning="extended"):
    """Return rates[(mhc, ma)] = (integral, stat_error)."""
    rates = {}
    for mhc, ma_list in mhc_ma_map.items():
        for ma in ma_list:
            masspoint = f"MHc{mhc}_MA{ma}"
            shapes_path = os.path.join(
                base_dir, masspoint, method, binning, "shapes.root"
            )
            if not os.path.isfile(shapes_path):
                raise RuntimeError(f"shapes.root not found: {shapes_path}")

            rf = ROOT.TFile.Open(shapes_path, "READ")
            if not rf or rf.IsZombie():
                raise RuntimeError(f"Cannot open {shapes_path}")
            h = rf.Get(masspoint)
            if not h:
                rf.Close()
                raise RuntimeError(f"Histogram '{masspoint}' not found in {shapes_path}")
            h.SetDirectory(0)
            rf.Close()

            err = ctypes.c_double(0)
            integral = h.IntegralAndError(1, h.GetNbinsX(), err)
            rates[(mhc, ma)] = (float(integral), float(err.value))

    return rates


def normalise(rates, mhc_ma_map, anchor_ma):
    """
    Normalise each MHc curve to its anchor MA value, propagating errors.

    Returns norm_rates[(mhc, ma)] = (norm_val, norm_err)
    """
    norm = {}
    for mhc, ma_list in mhc_ma_map.items():
        a_val, a_err = rates[(mhc, anchor_ma)]
        if a_val <= 0:
            raise RuntimeError(f"Anchor rate zero for MHc={mhc}")
        for ma in ma_list:
            r_val, r_err = rates[(mhc, ma)]
            n_val = r_val / a_val
            # relative errors add in quadrature
            n_err = n_val * np.sqrt((r_err / r_val) ** 2 + (a_err / a_val) ** 2) \
                    if r_val > 0 else 0.0
            norm[(mhc, ma)] = (n_val, n_err)
    return norm


# ---------------------------------------------------------------------------
# Quadratic fit helpers
# ---------------------------------------------------------------------------

def quad_fit(ma_arr, r_arr, err_arr):
    """
    Weighted quadratic fit.  Returns (coeffs, chi2, ndf).
    coeffs follow np.polyval convention: [a, b, c] for a*x^2 + b*x + c.
    ndf = n_points - 3.
    """
    ndf = len(ma_arr) - 3
    if ndf < 0:
        return None, np.nan, ndf          # underdetermined

    w = 1.0 / err_arr                     # weights = 1/sigma (polyfit uses w*y, w*Vandermonde)
    coeffs = np.polyfit(ma_arr, r_arr, 2, w=w)
    r_fit  = np.polyval(coeffs, ma_arr)
    chi2   = float(np.sum(((r_arr - r_fit) / err_arr) ** 2))
    return coeffs, chi2, ndf


def chi2_vs_model(ma_arr, r_arr, err_arr, model_fn):
    """
    chi²/ndf of data vs an external model (no free parameters).
    ndf = number of points.
    """
    pred = model_fn(ma_arr)
    chi2 = float(np.sum(((r_arr - pred) / err_arr) ** 2))
    ndf  = len(ma_arr)
    return chi2, ndf


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def _save(fig, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_normalized_with_fits(norm_rates, mhc_ma_map, anchor_ma, quad_results, plot_dir):
    """
    Data points + PCHIP curve + quadratic curve per MHc.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    for (mhc, ma_list), color in zip(sorted(mhc_ma_map.items()), COLORS):
        ma_arr = np.array(sorted(ma_list))
        r_arr  = np.array([norm_rates[(mhc, ma)][0] for ma in ma_arr])
        e_arr  = np.array([norm_rates[(mhc, ma)][1] for ma in ma_arr])

        ax.errorbar(ma_arr, r_arr, yerr=e_arr, fmt="o", color=color,
                    markersize=5, capsize=3, zorder=3)

        # PCHIP curve
        pchip  = PchipInterpolator(ma_arr, r_arr)
        ma_fine = np.linspace(ma_arr[0], ma_arr[-1], 200)
        ax.plot(ma_fine, pchip(ma_fine), "-", color=color, linewidth=1.5,
                label=f"MHc={mhc}")

        # Quadratic curve (dashed), only if determined
        coeffs, _, ndf = quad_results[mhc]
        if coeffs is not None:
            ax.plot(ma_fine, np.polyval(coeffs, ma_fine), "--", color=color,
                    linewidth=1.0, alpha=0.7)

    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("MA [GeV]")
    ax.set_ylabel(f"Rate / Rate(MA={anchor_ma})")
    ax.set_title("Normalised signal rates  —  solid: PCHIP,  dashed: quadratic fit")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    _save(fig, os.path.join(plot_dir, "rate_ratio_normalized.png"))


def plot_deviation(norm_rates, mhc_ma_map, anchor_ma, reference_mhc,
                   ref_pchip, ref_quad_coeffs, plot_dir):
    """
    Two-panel deviation plot: PCHIP(130) reference | QuadFit(130) reference.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax in axes:
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xlabel("MA [GeV]")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("(Rate − Ref) / Ref  [%]")
    axes[0].set_title(f"vs PCHIP(MHc={reference_mhc})")
    axes[1].set_title(f"vs QuadFit(MHc={reference_mhc})")

    colors_other = [(mhc, c) for (mhc, c) in zip(sorted(mhc_ma_map.keys()), COLORS)
                    if mhc != reference_mhc]

    for (mhc, color) in colors_other:
        # non-anchor MA values
        ma_arr = np.array([ma for ma in sorted(mhc_ma_map[mhc]) if ma != anchor_ma])
        if len(ma_arr) == 0:
            continue
        r_arr = np.array([norm_rates[(mhc, ma)][0] for ma in ma_arr])

        for ax, model_fn in zip(axes, [ref_pchip,
                                        lambda x: np.polyval(ref_quad_coeffs, x)]):
            pred = model_fn(ma_arr)
            with np.errstate(invalid="ignore", divide="ignore"):
                dev = np.where(np.abs(pred) > 1e-9, (r_arr - pred) / pred * 100, np.nan)
            ax.plot(ma_arr, dev, "o-", color=color, label=f"MHc={mhc}",
                    linewidth=1.5, markersize=5)

    axes[0].legend(fontsize=8)
    fig.suptitle(f"Deviation from MHc={reference_mhc} reference  "
                 f"(normalised to MA={anchor_ma})", y=1.01)
    fig.tight_layout()

    _save(fig, os.path.join(plot_dir, "rate_ratio_vs_reference.png"))


# ---------------------------------------------------------------------------
# Leave-one-out interpolation test (within same MHc)
# ---------------------------------------------------------------------------

def leave_one_out(norm_rates, mhc_ma_map, anchor_ma):
    """
    For each MHc, hold out each non-anchor MA point in turn.
    Fit PCHIP and quadratic to the remaining points and evaluate at the holdout.

    Returns:
        loo: {mhc: [(holdout_ma, dev_pchip%, dev_quad%), ...]}
    """
    loo = {}
    for mhc, ma_list in mhc_ma_map.items():
        non_anchor = sorted(ma for ma in ma_list if ma != anchor_ma)
        loo[mhc] = []

        for holdout_ma in non_anchor:
            train = [anchor_ma] + [ma for ma in non_anchor if ma != holdout_ma]
            train = sorted(train)

            ma_tr = np.array(train, dtype=float)
            r_tr  = np.array([norm_rates[(mhc, ma)][0] for ma in train])
            e_tr  = np.array([norm_rates[(mhc, ma)][1] for ma in train])
            e_tr  = np.where(e_tr > 1e-9, e_tr, 1e-6)

            true_val = norm_rates[(mhc, holdout_ma)][0]

            # PCHIP (always works with ≥2 points)
            pchip    = PchipInterpolator(ma_tr, r_tr, extrapolate=True)
            pred_p   = float(pchip(holdout_ma))
            dev_p    = (pred_p - true_val) / true_val * 100 if abs(true_val) > 1e-9 else np.nan

            # Quadratic: needs ≥3 training points
            if len(train) >= 3:
                coeffs_q = np.polyfit(ma_tr, r_tr, 2, w=1.0 / e_tr)
                pred_q   = float(np.polyval(coeffs_q, holdout_ma))
                dev_q    = (pred_q - true_val) / true_val * 100 if abs(true_val) > 1e-9 else np.nan
            else:
                dev_q = np.nan

            loo[mhc].append((holdout_ma, dev_p, dev_q))

    return loo


def plot_loo(loo, mhc_ma_map, plot_dir):
    """
    One subplot per MHc: deviation (%) vs holdout MA for PCHIP and quadratic.
    """
    mhc_list = sorted(mhc_ma_map.keys())
    ncols = 4
    nrows = (len(mhc_list) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows), sharey=False)
    axes_flat = axes.flatten()

    for ax, (mhc, color) in zip(axes_flat, zip(mhc_list, COLORS)):
        entries = loo[mhc]
        if not entries:
            ax.set_visible(False)
            continue

        holdout_mas = [e[0] for e in entries]
        devs_p      = [e[1] for e in entries]
        devs_q      = [e[2] for e in entries]

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.plot(holdout_mas, devs_p, "o-", color=color,
                label="PCHIP", linewidth=1.5, markersize=6)
        ax.plot(holdout_mas, devs_q, "s--", color=color, alpha=0.6,
                label="Quad", linewidth=1.2, markersize=6, markerfacecolor="none")

        ax.set_title(f"MHc={mhc}  (n={len(mhc_ma_map[mhc])} pts)", fontsize=10)
        ax.set_xlabel("Holdout MA [GeV]", fontsize=8)
        ax.set_ylabel("Dev [%]", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[len(mhc_list):]:
        ax.set_visible(False)

    fig.suptitle("Leave-one-out interpolation: PCHIP vs Quadratic (within same MHc)",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, os.path.join(plot_dir, "rate_loo_per_mhc.png"))


def print_loo_summary(loo):
    print()
    print("=" * 72)
    print("  LEAVE-ONE-OUT  deviation (%) within same MHc")
    print("=" * 72)
    print(f"  {'MHc':>5}  {'holdout MA':>10}  {'PCHIP dev%':>11}  {'Quad dev%':>10}")
    print("-" * 72)
    for mhc in sorted(loo.keys()):
        for holdout_ma, dev_p, dev_q in loo[mhc]:
            sp = f"{dev_p:+.1f}%" if not np.isnan(dev_p) else "   n/a"
            sq = f"{dev_q:+.1f}%" if not np.isnan(dev_q) else "   n/a"
            print(f"  {mhc:5d}  {holdout_ma:10d}  {sp:>11}  {sq:>10}")

        devs_p = [e[1] for e in loo[mhc] if not np.isnan(e[1])]
        devs_q = [e[2] for e in loo[mhc] if not np.isnan(e[2])]
        rms_p  = np.sqrt(np.mean(np.array(devs_p)**2)) if devs_p else np.nan
        rms_q  = np.sqrt(np.mean(np.array(devs_q)**2)) if devs_q else np.nan
        rp = f"{rms_p:.1f}%" if not np.isnan(rms_p) else "  n/a"
        rq = f"{rms_q:.1f}%" if not np.isnan(rms_q) else "  n/a"
        print(f"  {'':5}  {'RMS':>10}  {rp:>11}  {rq:>10}")
        print()
    print("=" * 72)


# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------

def print_summary(norm_rates, mhc_ma_map, anchor_ma, reference_mhc,
                  quad_results, ref_pchip, ref_quad_coeffs):
    ref_quad_fn = lambda x: np.polyval(ref_quad_coeffs, x)

    print()
    print("=" * 90)
    print(f"  QUADRATIC SELF-FIT  chi2/ndf  per MHc  (anchor MA={anchor_ma} excluded)")
    print("=" * 90)
    print(f"  {'MHc':>5}  {'n_pts':>5}  {'ndf':>4}  {'chi2':>8}  {'chi2/ndf':>9}  note")
    print("-" * 90)
    for mhc in sorted(mhc_ma_map.keys()):
        ma_list = sorted(mhc_ma_map[mhc])
        ma_arr = np.array([ma for ma in ma_list if ma != anchor_ma])
        r_arr  = np.array([norm_rates[(mhc, ma)][0] for ma in ma_arr])
        e_arr  = np.array([norm_rates[(mhc, ma)][1] for ma in ma_arr])

        # use stored quad_results (fit on all points incl. anchor for shape;
        # but chi2 computed here on non-anchor only for consistency)
        coeffs, _, _ = quad_results[mhc]
        if coeffs is not None and len(ma_arr) >= 3:
            r_fit = np.polyval(coeffs, ma_arr)
            chi2  = float(np.sum(((r_arr - r_fit) / e_arr) ** 2))
            ndf   = len(ma_arr) - 3
            note  = ""
        else:
            chi2 = np.nan
            ndf  = len(ma_arr) - 3
            note = "underdetermined" if ndf < 0 else "few pts"

        chi2_ndf = chi2 / ndf if ndf > 0 else np.nan
        print(f"  {mhc:5d}  {len(ma_arr):5d}  {ndf:4d}  {chi2:8.2f}  {chi2_ndf:9.3f}  {note}")
    print()

    print("=" * 90)
    print(f"  CROSS-MHc PREDICTION chi2/ndf  (MHc={reference_mhc} model → other MHc)")
    print(f"  ndf = number of non-anchor points  (no free parameters)")
    print("=" * 90)
    print(f"  {'MHc':>5}  {'n_pts':>5}  {'ndf':>4}  "
          f"{'chi2_PCHIP':>11}  {'chi2/ndf':>9}  "
          f"{'chi2_Quad':>10}  {'chi2/ndf':>9}")
    print("-" * 90)
    for mhc in sorted(mhc_ma_map.keys()):
        if mhc == reference_mhc:
            continue
        ma_arr = np.array([ma for ma in sorted(mhc_ma_map[mhc]) if ma != anchor_ma])
        r_arr  = np.array([norm_rates[(mhc, ma)][0] for ma in ma_arr])
        e_arr  = np.array([norm_rates[(mhc, ma)][1] for ma in ma_arr])

        chi2_p, ndf_p = chi2_vs_model(ma_arr, r_arr, e_arr, ref_pchip)
        chi2_q, ndf_q = chi2_vs_model(ma_arr, r_arr, e_arr, ref_quad_fn)

        chi2_p_ndf = chi2_p / ndf_p if ndf_p > 0 else np.nan
        chi2_q_ndf = chi2_q / ndf_q if ndf_q > 0 else np.nan
        print(f"  {mhc:5d}  {len(ma_arr):5d}  {ndf_p:4d}  "
              f"{chi2_p:11.1f}  {chi2_p_ndf:9.1f}  "
              f"{chi2_q:10.1f}  {chi2_q_ndf:9.1f}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--era",     default="2018")
    parser.add_argument("--channel", default="SR1E2Mu")
    parser.add_argument("--method",  default="Baseline")
    parser.add_argument("--binning", default="extended")
    args = parser.parse_args()

    workdir  = os.environ.get("WORKDIR", ".")
    base_dir = os.path.join(workdir, "SignalRegionStudyV2", "templates",
                            args.era, args.channel)
    if not os.path.isdir(base_dir):
        base_dir = os.path.join("templates", args.era, args.channel)
    if not os.path.isdir(base_dir):
        raise RuntimeError(
            f"Cannot find templates/{args.era}/{args.channel}. "
            "Run from SignalRegionStudyV2 after generating templates."
        )

    plot_dir = "test"

    print(f"Loading rates from: {base_dir}")
    rates      = load_rates(base_dir, MHC_MA_MAP, method=args.method, binning=args.binning)
    norm_rates = normalise(rates, MHC_MA_MAP, ANCHOR_MA)

    # Quadratic self-fit per MHc (fit on all points including anchor)
    quad_results = {}
    for mhc, ma_list in MHC_MA_MAP.items():
        ma_arr = np.array(sorted(ma_list), dtype=float)
        r_arr  = np.array([norm_rates[(mhc, ma)][0] for ma in ma_arr])
        e_arr  = np.array([norm_rates[(mhc, ma)][1] for ma in ma_arr])
        # Guard: replace zero errors with a small floor to avoid division issues
        e_arr  = np.where(e_arr > 1e-9, e_arr, np.nanmin(e_arr[e_arr > 1e-9]) * 0.1
                          if np.any(e_arr > 1e-9) else 1e-6)
        quad_results[mhc] = quad_fit(ma_arr, r_arr, e_arr)

    # Reference models from MHc=130
    ref_ma   = np.array(sorted(MHC_MA_MAP[REFERENCE_MHC]), dtype=float)
    ref_r    = np.array([norm_rates[(REFERENCE_MHC, ma)][0] for ma in ref_ma])
    ref_pchip        = PchipInterpolator(ref_ma, ref_r, extrapolate=False)
    ref_quad_coeffs, _, _ = quad_results[REFERENCE_MHC]

    # Leave-one-out test within each MHc
    loo = leave_one_out(norm_rates, MHC_MA_MAP, ANCHOR_MA)

    print("\nGenerating plots...")
    plot_normalized_with_fits(norm_rates, MHC_MA_MAP, ANCHOR_MA, quad_results, plot_dir)
    plot_deviation(norm_rates, MHC_MA_MAP, ANCHOR_MA, REFERENCE_MHC,
                   ref_pchip, ref_quad_coeffs, plot_dir)
    plot_loo(loo, MHC_MA_MAP, plot_dir)

    print_summary(norm_rates, MHC_MA_MAP, ANCHOR_MA, REFERENCE_MHC,
                  quad_results, ref_pchip, ref_quad_coeffs)
    print_loo_summary(loo)


if __name__ == "__main__":
    main()
