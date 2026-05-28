#!/usr/bin/env python3
"""
Test DCB pre-fit (unbinned) vs Voigtian pre-fit (binned), and independent SR3Mu fitting.

Test 1: Compare binned (Voigt pre-fit + DCB) vs unbinned (DCB pre-fit + DCB).
Test 2: Compare SR1E2Mu fit vs independent SR3Mu fit (can we drop the dependency?).

MHc=160 GeV, era=2018, both SR1E2Mu and SR3Mu channels.

Usage:
    cd SignalRegionStudyV3
    source setup.sh
    python3 fitting/test_dcb_prefit.py
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

PLOT_DIR = os.path.join(WORKDIR, "SignalRegionStudyV3", "fitting", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def build_dcb_from_params(prefix, mass_var, params):
    """Construct a RooCrystalBall with fixed parameters (for plotting only)."""
    x0 = ROOT.RooRealVar(f"{prefix}_x0", "x0", params["x0"])
    sL = ROOT.RooRealVar(f"{prefix}_sL", "sL", params["sigmaL"])
    sR = ROOT.RooRealVar(f"{prefix}_sR", "sR", params["sigmaR"])
    aL = ROOT.RooRealVar(f"{prefix}_aL", "aL", params["alphaL"])
    nL = ROOT.RooRealVar(f"{prefix}_nL", "nL", params["nL"])
    aR = ROOT.RooRealVar(f"{prefix}_aR", "aR", params["alphaR"])
    nR = ROOT.RooRealVar(f"{prefix}_nR", "nR", params["nR"])
    for v in [x0, sL, sR, aL, nL, aR, nR]:
        v.setConstant(True)
    dcb = ROOT.RooCrystalBall(f"{prefix}_dcb", "", mass_var, x0, sL, sR, aL, nL, aR, nR)
    # keep references alive
    dcb._params = [x0, sL, sR, aL, nL, aR, nR]
    return dcb


def fit_binned_pipeline(file_path, mA_nominal):
    """Current production approach: Voigt pre-fit (binned) + DCB fit (binned)."""

    # Step 1: wide Voigt pre-fit
    wide_lo, wide_hi = mA_nominal - 20.0, mA_nominal + 20.0
    rdf = ROOT.RDataFrame("Central", file_path)
    rdf = rdf.Filter(f"mass >= {wide_lo} && mass <= {wide_hi}")
    h_wide = rdf.Histo1D(
        ROOT.RDF.TH1DModel("h_wide", "", 400, wide_lo, wide_hi), "mass", "weight"
    ).GetValue().Clone("h_wide_c")
    h_wide.SetDirectory(0)

    mass_w = ROOT.RooRealVar("mass_w", "mass", wide_lo, wide_hi)
    data_w = ROOT.RooDataHist("data_w", "", ROOT.RooArgList(mass_w), h_wide)
    pre_mA = ROOT.RooRealVar("pre_mA", "mA", mA_nominal, wide_lo, wide_hi)
    pre_w = ROOT.RooRealVar("pre_w", "width", 0.1, 0.0, 5.0)
    pre_s = ROOT.RooRealVar("pre_s", "sigma", 0.1, 0.0, 5.0)
    pre_voigt = ROOT.RooVoigtian("pre_voigt", "", mass_w, pre_mA, pre_w, pre_s)
    pre_voigt.fitTo(data_w, ROOT.RooFit.SumW2Error(True),
                    ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))
    fitted_mA = pre_mA.getVal()
    vw = sqrt(pre_w.getVal()**2 + pre_s.getVal()**2)

    # Step 2: narrow DCB fit (binned)
    fit_lo = fitted_mA - 10.0 * vw
    fit_hi = fitted_mA + 10.0 * vw
    nbins = 100

    rdf2 = ROOT.RDataFrame("Central", file_path)
    rdf2 = rdf2.Filter(f"mass >= {fit_lo} && mass <= {fit_hi}")
    hist = rdf2.Histo1D(
        ROOT.RDF.TH1DModel("h_fit", "", nbins, fit_lo, fit_hi), "mass", "weight"
    ).GetValue().Clone("h_fit_c")
    hist.SetDirectory(0)

    mass = ROOT.RooRealVar("mass_b", "mass", fit_lo, fit_hi)
    roo_data = ROOT.RooDataHist("data_b", "", ROOT.RooArgList(mass), hist)

    dcb_x0 = ROOT.RooRealVar("b_x0", "x0", fitted_mA, fit_lo, fit_hi)
    dcb_sL = ROOT.RooRealVar("b_sL", "sL", 0.8 * vw, 0.01 * vw, 3.0 * vw)
    dcb_sR = ROOT.RooRealVar("b_sR", "sR", 0.8 * vw, 0.01 * vw, 3.0 * vw)
    dcb_aL = ROOT.RooRealVar("b_aL", "aL", 1.5, 0.5, 10.0)
    dcb_nL = ROOT.RooRealVar("b_nL", "nL", 2.0, 0.1, 50.0)
    dcb_aR = ROOT.RooRealVar("b_aR", "aR", 1.5, 0.5, 10.0)
    dcb_nR = ROOT.RooRealVar("b_nR", "nR", 2.0, 0.1, 50.0)
    dcb = ROOT.RooCrystalBall("b_dcb", "", mass, dcb_x0,
                               dcb_sL, dcb_sR, dcb_aL, dcb_nL, dcb_aR, dcb_nR)
    dcb.fitTo(roo_data, ROOT.RooFit.SumW2Error(True),
              ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

    sigma_eff = sqrt(0.5 * (dcb_sL.getVal()**2 + dcb_sR.getVal()**2))

    return {
        "x0": dcb_x0.getVal(), "sigmaL": dcb_sL.getVal(), "sigmaR": dcb_sR.getVal(),
        "alphaL": dcb_aL.getVal(), "nL": dcb_nL.getVal(),
        "alphaR": dcb_aR.getVal(), "nR": dcb_nR.getVal(),
        "sigma_eff": sigma_eff,
        "prefit_mA": fitted_mA, "prefit_vw": vw,
        "fit_lo": fit_lo, "fit_hi": fit_hi,
    }


def fit_unbinned_pipeline(file_path, mA_nominal):
    """New approach: DCB pre-fit (unbinned) + DCB fit (unbinned)."""

    f = ROOT.TFile.Open(file_path)
    tree = f.Get("Central")

    # Step 1: wide DCB pre-fit (unbinned)
    wide_lo, wide_hi = mA_nominal - 20.0, mA_nominal + 20.0
    mass_w = ROOT.RooRealVar("mass", "mass", wide_lo, wide_hi)
    weight_w = ROOT.RooRealVar("weight", "weight", -10, 10)
    ds_wide = ROOT.RooDataSet("ds_wide", "", tree, ROOT.RooArgSet(mass_w, weight_w),
                               f"mass >= {wide_lo} && mass <= {wide_hi}", "weight")

    pre_x0 = ROOT.RooRealVar("u_pre_x0", "x0", mA_nominal, wide_lo, wide_hi)
    pre_sL = ROOT.RooRealVar("u_pre_sL", "sL", 1.0, 0.01, 10.0)
    pre_sR = ROOT.RooRealVar("u_pre_sR", "sR", 1.0, 0.01, 10.0)
    pre_aL = ROOT.RooRealVar("u_pre_aL", "aL", 1.5, 0.5, 10.0)
    pre_nL = ROOT.RooRealVar("u_pre_nL", "nL", 2.0, 0.1, 50.0)
    pre_aR = ROOT.RooRealVar("u_pre_aR", "aR", 1.5, 0.5, 10.0)
    pre_nR = ROOT.RooRealVar("u_pre_nR", "nR", 2.0, 0.1, 50.0)
    pre_dcb = ROOT.RooCrystalBall("u_pre_dcb", "", mass_w, pre_x0,
                                   pre_sL, pre_sR, pre_aL, pre_nL, pre_aR, pre_nR)
    pre_dcb.fitTo(ds_wide, ROOT.RooFit.SumW2Error(True),
                  ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

    fitted_mA = pre_x0.getVal()
    vw = sqrt(0.5 * (pre_sL.getVal()**2 + pre_sR.getVal()**2))

    # Step 2: narrow DCB fit (unbinned)
    fit_lo = fitted_mA - 10.0 * vw
    fit_hi = fitted_mA + 10.0 * vw

    mass_n = ROOT.RooRealVar("mass", "mass", fit_lo, fit_hi)
    weight_n = ROOT.RooRealVar("weight", "weight", -10, 10)
    ds_narrow = ROOT.RooDataSet("ds_narrow", "", tree, ROOT.RooArgSet(mass_n, weight_n),
                                 f"mass >= {fit_lo} && mass <= {fit_hi}", "weight")

    dcb_x0 = ROOT.RooRealVar("u_x0", "x0", fitted_mA, fit_lo, fit_hi)
    dcb_sL = ROOT.RooRealVar("u_sL", "sL", 0.8 * vw, 0.01 * vw, 3.0 * vw)
    dcb_sR = ROOT.RooRealVar("u_sR", "sR", 0.8 * vw, 0.01 * vw, 3.0 * vw)
    dcb_aL = ROOT.RooRealVar("u_aL", "aL", 1.5, 0.5, 10.0)
    dcb_nL = ROOT.RooRealVar("u_nL", "nL", 2.0, 0.1, 50.0)
    dcb_aR = ROOT.RooRealVar("u_aR", "aR", 1.5, 0.5, 10.0)
    dcb_nR = ROOT.RooRealVar("u_nR", "nR", 2.0, 0.1, 50.0)
    dcb = ROOT.RooCrystalBall("u_dcb", "", mass_n, dcb_x0,
                               dcb_sL, dcb_sR, dcb_aL, dcb_nL, dcb_aR, dcb_nR)
    dcb.fitTo(ds_narrow, ROOT.RooFit.SumW2Error(True),
              ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

    f.Close()

    sigma_eff = sqrt(0.5 * (dcb_sL.getVal()**2 + dcb_sR.getVal()**2))

    return {
        "x0": dcb_x0.getVal(), "sigmaL": dcb_sL.getVal(), "sigmaR": dcb_sR.getVal(),
        "alphaL": dcb_aL.getVal(), "nL": dcb_nL.getVal(),
        "alphaR": dcb_aR.getVal(), "nR": dcb_nR.getVal(),
        "sigma_eff": sigma_eff,
        "prefit_mA": fitted_mA, "prefit_vw": vw,
        "fit_lo": fit_lo, "fit_hi": fit_hi,
    }


def make_comparison_plot(binned_params, unbinned_params, file_path, channel, masspoint):
    """Overlay binned vs unbinned DCB fits on a common histogram."""

    # Common range: union of both fit windows
    fit_lo = min(binned_params["fit_lo"], unbinned_params["fit_lo"])
    fit_hi = max(binned_params["fit_hi"], unbinned_params["fit_hi"])
    nbins = 100

    rdf = ROOT.RDataFrame("Central", file_path)
    rdf = rdf.Filter(f"mass >= {fit_lo} && mass <= {fit_hi}")
    hist = rdf.Histo1D(
        ROOT.RDF.TH1DModel("h_cmp", "", nbins, fit_lo, fit_hi), "mass", "weight"
    ).GetValue().Clone("h_cmp_c")
    hist.SetDirectory(0)

    mass = ROOT.RooRealVar("mass_cmp", "mass", fit_lo, fit_hi)
    roo_data = ROOT.RooDataHist("data_cmp", "", ROOT.RooArgList(mass), hist)

    dcb_binned = build_dcb_from_params("cmp_bin", mass, binned_params)
    dcb_unbinned = build_dcb_from_params("cmp_unb", mass, unbinned_params)

    fit_models = OrderedDict([
        ("Binned (Voigt+DCB)", (dcb_binned, 7, ROOT.kSolid)),
        ("Unbinned (DCB+DCB)", (dcb_unbinned, 7, ROOT.kDashed)),
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
        "legend": [0.2, 0.5, 0.55, 0.65],
        "legendTextSize": 0.025,
        "iPos": 0, "maxDigits": 3,
    }

    canvas = FitCanvasWithRatio(roo_data, mass, hist, fit_models, config)
    canvas.drawPadUp()
    canvas.drawPadDown()
    canvas.drawMasspoint()

    outpath = os.path.join(PLOT_DIR, f"test1_{channel}_{masspoint}.png")
    canvas.canv.SaveAs(outpath)
    return canvas.chi2_values


def make_channel_plot(sr1e2mu_params, sr3mu_params, sr3mu_file, masspoint):
    """Overlay SR1E2Mu fit vs independent SR3Mu fit on SR3Mu data."""

    # Use SR3Mu fit window
    fit_lo = sr3mu_params["fit_lo"]
    fit_hi = sr3mu_params["fit_hi"]
    nbins = 100

    rdf = ROOT.RDataFrame("Central", sr3mu_file)
    rdf = rdf.Filter(f"mass >= {fit_lo} && mass <= {fit_hi}")
    hist = rdf.Histo1D(
        ROOT.RDF.TH1DModel("h_ch", "", nbins, fit_lo, fit_hi), "mass", "weight"
    ).GetValue().Clone("h_ch_c")
    hist.SetDirectory(0)

    mass = ROOT.RooRealVar("mass_ch", "mass", fit_lo, fit_hi)
    roo_data = ROOT.RooDataHist("data_ch", "", ROOT.RooArgList(mass), hist)

    dcb_1e2mu = build_dcb_from_params("ch_1e2mu", mass, sr1e2mu_params)
    dcb_3mu = build_dcb_from_params("ch_3mu", mass, sr3mu_params)

    fit_models = OrderedDict([
        ("SR1E2Mu fit", (dcb_1e2mu, 7, ROOT.kSolid)),
        ("SR3Mu indep.", (dcb_3mu, 7, ROOT.kDashed)),
    ])

    config = {
        "era": ERA,
        "xTitle": "m_{A} [GeV]",
        "yTitle": "Events",
        "rTitle": "Fit / MC",
        "rRange": [0.5, 1.5],
        "channel": "SR3Mu",
        "channelPosX": 0.2, "channelPosY": 0.74,
        "channelFont": 61, "channelSize": 0.04,
        "masspoint": masspoint,
        "masspointPosX": 0.2, "masspointPosY": 0.69,
        "masspointFont": 61, "masspointSize": 0.04,
        "legend": [0.2, 0.5, 0.55, 0.65],
        "legendTextSize": 0.025,
        "iPos": 0, "maxDigits": 3,
    }

    canvas = FitCanvasWithRatio(roo_data, mass, hist, fit_models, config)
    canvas.drawPadUp()
    canvas.drawPadDown()
    canvas.drawMasspoint()

    outpath = os.path.join(PLOT_DIR, f"test2_{masspoint}.png")
    canvas.canv.SaveAs(outpath)
    return canvas.chi2_values


def get_file_path(channel, masspoint):
    return os.path.join(
        WORKDIR, "SignalRegionStudyV3", "samples",
        ERA, channel, masspoint, f"{masspoint}.root"
    )


def main():
    # ===== Test 1: Binned (Voigt+DCB) vs Unbinned (DCB+DCB) =====
    print("=" * 120)
    print("Test 1: Binned (Voigt+DCB) vs Unbinned (DCB+DCB)")
    print("=" * 120)

    test1_results = []
    for channel in CHANNELS:
        for ma in MA_LIST:
            masspoint = f"MHc{MHC}_MA{ma}"
            file_path = get_file_path(channel, masspoint)
            if not os.path.exists(file_path):
                print(f"SKIP {channel}/{masspoint}: file not found")
                continue

            print(f"  {channel}/{masspoint}...", end=" ", flush=True)
            binned = fit_binned_pipeline(file_path, float(ma))
            unbinned = fit_unbinned_pipeline(file_path, float(ma))
            chi2s = make_comparison_plot(binned, unbinned, file_path, channel, masspoint)
            print("done")

            test1_results.append({
                "channel": channel, "masspoint": masspoint, "mA": ma,
                "binned": binned, "unbinned": unbinned,
                "chi2_binned": chi2s["Binned (Voigt+DCB)"],
                "chi2_unbinned": chi2s["Unbinned (DCB+DCB)"],
            })

    # Test 1 summary
    print("\n" + "=" * 120)
    print(f"{'Channel':>10} {'MassPoint':>15} | "
          f"{'B:x0':>7} {'B:se':>6} {'B:chi2':>7} | "
          f"{'U:x0':>7} {'U:se':>6} {'U:chi2':>7} | "
          f"{'Dse%':>6}")
    print("-" * 120)
    for r in test1_results:
        b, u = r["binned"], r["unbinned"]
        dse = (u["sigma_eff"] - b["sigma_eff"]) / b["sigma_eff"] * 100 if b["sigma_eff"] > 0 else 0
        print(f"{r['channel']:>10} {r['masspoint']:>15} | "
              f"{b['x0']:7.2f} {b['sigma_eff']:6.3f} {r['chi2_binned']:7.1f} | "
              f"{u['x0']:7.2f} {u['sigma_eff']:6.3f} {r['chi2_unbinned']:7.1f} | "
              f"{dse:+6.1f}")

    # ===== Test 2: Independent SR3Mu fitting =====
    print("\n" + "=" * 120)
    print("Test 2: SR1E2Mu fit vs Independent SR3Mu fit (unbinned DCB pipeline)")
    print("=" * 120)

    test2_results = []
    for ma in MA_LIST:
        masspoint = f"MHc{MHC}_MA{ma}"
        f1e2mu = get_file_path("SR1E2Mu", masspoint)
        f3mu = get_file_path("SR3Mu", masspoint)
        if not os.path.exists(f1e2mu) or not os.path.exists(f3mu):
            print(f"SKIP {masspoint}: file not found")
            continue

        print(f"  {masspoint}...", end=" ", flush=True)
        params_1e2mu = fit_unbinned_pipeline(f1e2mu, float(ma))
        params_3mu = fit_unbinned_pipeline(f3mu, float(ma))
        chi2s = make_channel_plot(params_1e2mu, params_3mu, f3mu, masspoint)
        print("done")

        test2_results.append({
            "masspoint": masspoint, "mA": ma,
            "sr1e2mu": params_1e2mu, "sr3mu": params_3mu,
            "chi2_1e2mu": chi2s["SR1E2Mu fit"],
            "chi2_3mu": chi2s["SR3Mu indep."],
        })

    # Test 2 summary
    print("\n" + "=" * 120)
    print(f"{'MassPoint':>15} | "
          f"{'1E2Mu:x0':>9} {'1E2Mu:se':>9} | "
          f"{'3Mu:x0':>9} {'3Mu:se':>9} | "
          f"{'Dx0':>7} {'Dse%':>6} {'se_ratio':>9} | "
          f"{'chi2(1E2Mu)':>12} {'chi2(3Mu)':>10}")
    print("-" * 120)
    for r in test2_results:
        p1, p3 = r["sr1e2mu"], r["sr3mu"]
        dx0 = p3["x0"] - p1["x0"]
        dse = (p3["sigma_eff"] - p1["sigma_eff"]) / p1["sigma_eff"] * 100 if p1["sigma_eff"] > 0 else 0
        ratio = p3["sigma_eff"] / p1["sigma_eff"] if p1["sigma_eff"] > 0 else 0
        print(f"{r['masspoint']:>15} | "
              f"{p1['x0']:9.3f} {p1['sigma_eff']:9.3f} | "
              f"{p3['x0']:9.3f} {p3['sigma_eff']:9.3f} | "
              f"{dx0:+7.3f} {dse:+6.1f} {ratio:9.3f} | "
              f"{r['chi2_1e2mu']:12.1f} {r['chi2_3mu']:10.1f}")

    print(f"\nPlots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
