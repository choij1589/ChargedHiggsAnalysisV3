#!/usr/bin/env python3
"""
Unbinned fitting function comparison: Voigtian vs Double Gaussian vs DCB.

Each model uses its own unbinned pre-fit (mA +/- 20 GeV) followed by
an unbinned narrow fit (fitted_mA +/- 10*vw). Documentation-quality plots.

MHc=160 GeV, era=2018, both SR1E2Mu and SR3Mu channels.

Usage:
    cd SignalRegionStudyV2
    source setup.sh
    python3 fitting/test_fitting_functions.py
"""
import os
from collections import OrderedDict
from math import sqrt

import ROOT

ROOT.gROOT.SetBatch(True)
ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

from plotter import FitCanvasWithRatio

MHC = 160
MA_LIST = [15, 50, 85, 98, 120, 135, 155]
CHANNELS = ["SR1E2Mu", "SR3Mu"]
ERA = "2018"

WORKDIR = os.getenv("WORKDIR")
if not WORKDIR:
    raise EnvironmentError("WORKDIR not set. Please run 'source setup.sh'")

PLOT_DIR = os.path.join(WORKDIR, "SignalRegionStudyV2", "fitting", "plots", "fitting_functions")
os.makedirs(PLOT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Unbinned pre-fit + narrow fit pipelines (one per model)
# ---------------------------------------------------------------------------

def _make_dataset(tree, lo, hi):
    """Create a weighted RooDataSet from a TTree in [lo, hi]."""
    mass = ROOT.RooRealVar("mass", "mass", lo, hi)
    weight = ROOT.RooRealVar("weight", "weight", -10, 10)
    ds = ROOT.RooDataSet("ds", "", tree, ROOT.RooArgSet(mass, weight),
                          f"mass >= {lo} && mass <= {hi}", "weight")
    return ds, mass


def fit_voigt(tree, mA_nominal):
    """Voigtian pre-fit -> Voigtian narrow fit (unbinned)."""
    wide_lo = max(mA_nominal - mA_nominal / 3.0, 12.0)
    wide_hi = mA_nominal + mA_nominal / 3.0
    ds_w, mass_w = _make_dataset(tree, wide_lo, wide_hi)

    pre_mA = ROOT.RooRealVar("vp_mA", "mA", mA_nominal, wide_lo, wide_hi)
    pre_w = ROOT.RooRealVar("vp_w", "w", 0.1, 0.0, 5.0)
    pre_s = ROOT.RooRealVar("vp_s", "s", 0.1, 0.0, 5.0)
    pre_voigt = ROOT.RooVoigtian("vp_voigt", "", mass_w, pre_mA, pre_w, pre_s)
    pre_voigt.fitTo(ds_w, ROOT.RooFit.SumW2Error(True),
                    ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))
    fitted_mA = pre_mA.getVal()
    vw = sqrt(pre_w.getVal()**2 + pre_s.getVal()**2)

    fit_lo = max(fitted_mA - 10.0 * vw, 12.0)
    fit_hi = fitted_mA + 10.0 * vw
    ds_n, mass_n = _make_dataset(tree, fit_lo, fit_hi)

    v_mA = ROOT.RooRealVar("v_mA", "mA", fitted_mA, fit_lo, fit_hi)
    v_w = ROOT.RooRealVar("v_w", "w", pre_w.getVal(), 0.0, 5.0 * vw)
    v_s = ROOT.RooRealVar("v_s", "s", pre_s.getVal(), 0.0, 5.0 * vw)
    voigt = ROOT.RooVoigtian("v_voigt", "", mass_n, v_mA, v_w, v_s)
    voigt.fitTo(ds_n, ROOT.RooFit.SumW2Error(True),
                ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

    sigma_eff = sqrt(v_w.getVal()**2 + v_s.getVal()**2)
    return {
        "x0": v_mA.getVal(), "width": v_w.getVal(), "sigma": v_s.getVal(),
        "sigma_eff": sigma_eff, "fit_lo": fit_lo, "fit_hi": fit_hi,
    }


def fit_dg(tree, mA_nominal):
    """Double Gaussian pre-fit -> DG narrow fit (unbinned)."""
    wide_lo = max(mA_nominal - mA_nominal / 3.0, 12.0)
    wide_hi = mA_nominal + mA_nominal / 3.0
    ds_w, mass_w = _make_dataset(tree, wide_lo, wide_hi)

    pre_mu = ROOT.RooRealVar("dgp_mu", "mu", mA_nominal, wide_lo, wide_hi)
    pre_s1 = ROOT.RooRealVar("dgp_s1", "s1", 0.5, 0.01, 10.0)
    pre_s2 = ROOT.RooRealVar("dgp_s2", "s2", 2.0, 0.1, 15.0)
    pre_f = ROOT.RooRealVar("dgp_f", "f", 0.6, 0.01, 0.99)
    pre_g1 = ROOT.RooGaussian("dgp_g1", "", mass_w, pre_mu, pre_s1)
    pre_g2 = ROOT.RooGaussian("dgp_g2", "", mass_w, pre_mu, pre_s2)
    pre_dg = ROOT.RooAddPdf("dgp_dg", "", pre_g1, pre_g2, pre_f)
    pre_dg.fitTo(ds_w, ROOT.RooFit.SumW2Error(True),
                 ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

    fitted_mA = pre_mu.getVal()
    s1v, s2v, fv = pre_s1.getVal(), pre_s2.getVal(), pre_f.getVal()
    if s1v > s2v:
        s1v, s2v, fv = s2v, s1v, 1.0 - fv
    vw = sqrt(fv * s1v**2 + (1.0 - fv) * s2v**2)

    fit_lo = max(fitted_mA - 10.0 * vw, 12.0)
    fit_hi = fitted_mA + 10.0 * vw
    ds_n, mass_n = _make_dataset(tree, fit_lo, fit_hi)

    dg_mu = ROOT.RooRealVar("dg_mu", "mu", fitted_mA, fit_lo, fit_hi)
    dg_s1 = ROOT.RooRealVar("dg_s1", "s1", s1v, 0.01 * vw, 3.0 * vw)
    dg_s2 = ROOT.RooRealVar("dg_s2", "s2", s2v, 0.1 * vw, 5.0 * vw)
    dg_frac = ROOT.RooRealVar("dg_f", "f", fv, 0.01, 0.99)
    g1 = ROOT.RooGaussian("dg_g1", "", mass_n, dg_mu, dg_s1)
    g2 = ROOT.RooGaussian("dg_g2", "", mass_n, dg_mu, dg_s2)
    dg = ROOT.RooAddPdf("dg", "", g1, g2, dg_frac)
    dg.fitTo(ds_n, ROOT.RooFit.SumW2Error(True),
             ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

    s1v = dg_s1.getVal()
    s2v = dg_s2.getVal()
    fv = dg_frac.getVal()
    if s1v > s2v:
        s1v, s2v, fv = s2v, s1v, 1.0 - fv
    sigma_eff = sqrt(fv * s1v**2 + (1.0 - fv) * s2v**2)

    return {
        "x0": dg_mu.getVal(), "s1": s1v, "s2": s2v, "frac": fv,
        "sigma_eff": sigma_eff, "fit_lo": fit_lo, "fit_hi": fit_hi,
    }


def fit_dcb(tree, mA_nominal):
    """DCB pre-fit -> DCB narrow fit (unbinned)."""
    wide_lo = max(mA_nominal - mA_nominal / 3.0, 12.0)
    wide_hi = mA_nominal + mA_nominal / 3.0
    ds_w, mass_w = _make_dataset(tree, wide_lo, wide_hi)

    pre_x0 = ROOT.RooRealVar("dcbp_x0", "x0", mA_nominal, wide_lo, wide_hi)
    pre_sL = ROOT.RooRealVar("dcbp_sL", "sL", 1.0, 0.01, 10.0)
    pre_sR = ROOT.RooRealVar("dcbp_sR", "sR", 1.0, 0.01, 10.0)
    pre_aL = ROOT.RooRealVar("dcbp_aL", "aL", 1.5, 0.5, 10.0)
    pre_nL = ROOT.RooRealVar("dcbp_nL", "nL", 2.0, 0.1, 50.0)
    pre_aR = ROOT.RooRealVar("dcbp_aR", "aR", 1.5, 0.5, 10.0)
    pre_nR = ROOT.RooRealVar("dcbp_nR", "nR", 2.0, 0.1, 50.0)
    pre_dcb = ROOT.RooCrystalBall("dcbp_dcb", "", mass_w, pre_x0,
                                   pre_sL, pre_sR, pre_aL, pre_nL, pre_aR, pre_nR)
    pre_dcb.fitTo(ds_w, ROOT.RooFit.SumW2Error(True),
                  ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

    fitted_mA = pre_x0.getVal()
    vw = sqrt(0.5 * (pre_sL.getVal()**2 + pre_sR.getVal()**2))

    fit_lo = max(fitted_mA - 10.0 * vw, 12.0)
    fit_hi = fitted_mA + 10.0 * vw
    ds_n, mass_n = _make_dataset(tree, fit_lo, fit_hi)

    dcb_x0 = ROOT.RooRealVar("dcb_x0", "x0", fitted_mA, fit_lo, fit_hi)
    dcb_sL = ROOT.RooRealVar("dcb_sL", "sL", 0.8 * vw, 0.01 * vw, 3.0 * vw)
    dcb_sR = ROOT.RooRealVar("dcb_sR", "sR", 0.8 * vw, 0.01 * vw, 3.0 * vw)
    dcb_aL = ROOT.RooRealVar("dcb_aL", "aL", 1.5, 0.5, 10.0)
    dcb_nL = ROOT.RooRealVar("dcb_nL", "nL", 2.0, 0.1, 50.0)
    dcb_aR = ROOT.RooRealVar("dcb_aR", "aR", 1.5, 0.5, 10.0)
    dcb_nR = ROOT.RooRealVar("dcb_nR", "nR", 2.0, 0.1, 50.0)
    dcb = ROOT.RooCrystalBall("dcb", "", mass_n, dcb_x0,
                               dcb_sL, dcb_sR, dcb_aL, dcb_nL, dcb_aR, dcb_nR)
    dcb.fitTo(ds_n, ROOT.RooFit.SumW2Error(True),
              ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

    sigma_eff = sqrt(0.5 * (dcb_sL.getVal()**2 + dcb_sR.getVal()**2))

    return {
        "x0": dcb_x0.getVal(), "sigmaL": dcb_sL.getVal(), "sigmaR": dcb_sR.getVal(),
        "alphaL": dcb_aL.getVal(), "nL": dcb_nL.getVal(),
        "alphaR": dcb_aR.getVal(), "nR": dcb_nR.getVal(),
        "sigma_eff": sigma_eff, "fit_lo": fit_lo, "fit_hi": fit_hi,
    }


# ---------------------------------------------------------------------------
# Reconstruct constant-parameter PDFs on a common mass variable for plotting
# ---------------------------------------------------------------------------

def build_voigt_const(prefix, mass_var, params):
    mA = ROOT.RooRealVar(f"{prefix}_mA", "mA", params["x0"])
    w = ROOT.RooRealVar(f"{prefix}_w", "w", params["width"])
    s = ROOT.RooRealVar(f"{prefix}_s", "s", params["sigma"])
    for v in [mA, w, s]:
        v.setConstant(True)
    pdf = ROOT.RooVoigtian(f"{prefix}_voigt", "", mass_var, mA, w, s)
    pdf._params = [mA, w, s]
    return pdf


def build_dg_const(prefix, mass_var, params):
    mu = ROOT.RooRealVar(f"{prefix}_mu", "mu", params["x0"])
    s1 = ROOT.RooRealVar(f"{prefix}_s1", "s1", params["s1"])
    s2 = ROOT.RooRealVar(f"{prefix}_s2", "s2", params["s2"])
    f = ROOT.RooRealVar(f"{prefix}_f", "f", params["frac"])
    for v in [mu, s1, s2, f]:
        v.setConstant(True)
    g1 = ROOT.RooGaussian(f"{prefix}_g1", "", mass_var, mu, s1)
    g2 = ROOT.RooGaussian(f"{prefix}_g2", "", mass_var, mu, s2)
    pdf = ROOT.RooAddPdf(f"{prefix}_dg", "", g1, g2, f)
    pdf._params = [mu, s1, s2, f, g1, g2]
    return pdf


def build_dcb_const(prefix, mass_var, params):
    x0 = ROOT.RooRealVar(f"{prefix}_x0", "x0", params["x0"])
    sL = ROOT.RooRealVar(f"{prefix}_sL", "sL", params["sigmaL"])
    sR = ROOT.RooRealVar(f"{prefix}_sR", "sR", params["sigmaR"])
    aL = ROOT.RooRealVar(f"{prefix}_aL", "aL", params["alphaL"])
    nL = ROOT.RooRealVar(f"{prefix}_nL", "nL", params["nL"])
    aR = ROOT.RooRealVar(f"{prefix}_aR", "aR", params["alphaR"])
    nR = ROOT.RooRealVar(f"{prefix}_nR", "nR", params["nR"])
    for v in [x0, sL, sR, aL, nL, aR, nR]:
        v.setConstant(True)
    pdf = ROOT.RooCrystalBall(f"{prefix}_dcb", "", mass_var, x0, sL, sR, aL, nL, aR, nR)
    pdf._params = [x0, sL, sR, aL, nL, aR, nR]
    return pdf


# ---------------------------------------------------------------------------
# Main fitting + plotting
# ---------------------------------------------------------------------------

def run_fits(file_path, mA_nominal, channel, masspoint):
    """Run all three unbinned pipelines and produce a comparison plot."""
    f = ROOT.TFile.Open(file_path)
    tree = f.Get("Central")

    voigt_params = fit_voigt(tree, mA_nominal)
    dg_params = fit_dg(tree, mA_nominal)
    dcb_params = fit_dcb(tree, mA_nominal)

    f.Close()

    # Use DCB fit window for plotting range
    fit_lo = dcb_params["fit_lo"]
    fit_hi = dcb_params["fit_hi"]
    nbins = 100

    # Binned histogram for visualization
    rdf = ROOT.RDataFrame("Central", file_path)
    rdf = rdf.Filter(f"mass >= {fit_lo} && mass <= {fit_hi}")
    hist = rdf.Histo1D(
        ROOT.RDF.TH1DModel("h_plot", "", nbins, fit_lo, fit_hi), "mass", "weight"
    ).GetValue().Clone("h_plot_c")
    hist.SetDirectory(0)

    mass = ROOT.RooRealVar("mass_plot", "mass", fit_lo, fit_hi)
    roo_data = ROOT.RooDataHist("data_plot", "", ROOT.RooArgList(mass), hist)

    # Reconstruct constant-parameter PDFs on common mass variable
    voigt_pdf = build_voigt_const("plt_v", mass, voigt_params)
    dg_pdf = build_dg_const("plt_dg", mass, dg_params)
    dcb_pdf = build_dcb_const("plt_dcb", mass, dcb_params)

    fit_models = OrderedDict([
        ("Voigtian", (voigt_pdf, 3, ROOT.kSolid)),
        ("DG",       (dg_pdf,   4, ROOT.kSolid)),
        ("DCB",      (dcb_pdf,  7, ROOT.kSolid)),
    ])

    config = {
        "era": ERA,
        "xTitle": "m_{A} [GeV]",
        "yTitle": "Events",
        "rTitle": "Fit / MC",
        "rRange": [0.5, 1.5],
        "channel": channel,
        "channelPosX": 0.2, "channelPosY": 0.74,
        "channelFont": 61, "channelSize": 0.04,
        "masspoint": masspoint,
        "masspointPosX": 0.2, "masspointPosY": 0.69,
        "masspointFont": 61, "masspointSize": 0.04,
        "legend": [0.2, 0.5, 0.45, 0.65],
        "legendTextSize": 0.03,
        "iPos": 0, "maxDigits": 3,
    }

    canvas = FitCanvasWithRatio(roo_data, mass, hist, fit_models, config)
    canvas.drawPadUp()
    canvas.drawPadDown()
    canvas.drawMasspoint()

    outpath = os.path.join(PLOT_DIR, f"fit_{channel}_{masspoint}.png")
    canvas.canv.SaveAs(outpath)

    return {
        "channel": channel, "masspoint": masspoint, "mA_nominal": mA_nominal,
        "v_sigma_eff": voigt_params["sigma_eff"], "v_chi2": canvas.chi2_values["Voigtian"],
        "dg_s1": dg_params["s1"], "dg_s2": dg_params["s2"], "dg_frac": dg_params["frac"],
        "dg_sigma_eff": dg_params["sigma_eff"], "dg_chi2": canvas.chi2_values["DG"],
        "dcb_sL": dcb_params["sigmaL"], "dcb_sR": dcb_params["sigmaR"],
        "dcb_sigma_eff": dcb_params["sigma_eff"], "dcb_chi2": canvas.chi2_values["DCB"],
        "plot": outpath,
    }


def main():
    results = []

    for channel in CHANNELS:
        for ma in MA_LIST:
            masspoint = f"MHc{MHC}_MA{ma}"
            file_path = os.path.join(
                WORKDIR, "SignalRegionStudyV2", "samples",
                ERA, channel, masspoint, f"{masspoint}.root"
            )
            if not os.path.exists(file_path):
                print(f"SKIP {channel}/{masspoint}: file not found")
                continue

            print(f"Fitting {channel}/{masspoint}...", end=" ", flush=True)
            res = run_fits(file_path, float(ma), channel, masspoint)
            results.append(res)
            print(f"done -> {os.path.basename(res['plot'])}")

    # Summary table
    print("\n" + "=" * 130)
    print(f"{'Channel':>10} {'MassPoint':>15} | "
          f"{'V:chi2':>7} {'V:se':>6} | "
          f"{'DG:chi2':>7} {'DG:s1':>6} {'DG:s2':>6} {'DG:f':>5} {'DG:se':>6} | "
          f"{'DCB:chi2':>8} {'DCB:sL':>7} {'DCB:sR':>7} {'DCB:se':>7} | "
          f"{'Best':>5}")
    print("=" * 130)

    for r in results:
        chi2s = {"V": r["v_chi2"], "DG": r["dg_chi2"], "DCB": r["dcb_chi2"]}
        best = min(chi2s, key=chi2s.get)
        print(f"{r['channel']:>10} {r['masspoint']:>15} | "
              f"{r['v_chi2']:7.1f} {r['v_sigma_eff']:6.3f} | "
              f"{r['dg_chi2']:7.1f} {r['dg_s1']:6.3f} {r['dg_s2']:6.3f} "
              f"{r['dg_frac']:5.2f} {r['dg_sigma_eff']:6.3f} | "
              f"{r['dcb_chi2']:8.1f} {r['dcb_sL']:7.3f} {r['dcb_sR']:7.3f} "
              f"{r['dcb_sigma_eff']:7.3f} | {best:>5}")

    # sigma_eff comparison between channels
    print("\n" + "=" * 70)
    print("sigma_eff comparison between channels:")
    print(f"{'MA':>5} {'1E2Mu V':>8} {'1E2Mu DCB':>10} "
          f"{'3Mu V':>8} {'3Mu DCB':>8} {'DCB ratio':>10}")
    print("-" * 55)
    for ma in MA_LIST:
        r1 = next((r for r in results if r["channel"] == "SR1E2Mu" and r["mA_nominal"] == ma), None)
        r3 = next((r for r in results if r["channel"] == "SR3Mu" and r["mA_nominal"] == ma), None)
        if r1 and r3:
            rdcb = r3["dcb_sigma_eff"] / r1["dcb_sigma_eff"] if r1["dcb_sigma_eff"] > 0 else 0
            print(f"{ma:5.0f} {r1['v_sigma_eff']:8.3f} {r1['dcb_sigma_eff']:10.3f} "
                  f"{r3['v_sigma_eff']:8.3f} {r3['dcb_sigma_eff']:8.3f} {rdcb:10.3f}")

    print(f"\nPlots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
