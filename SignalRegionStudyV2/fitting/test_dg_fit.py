#!/usr/bin/env python3
"""
Validation plots: Voigtian vs Double Gaussian vs Double Crystal Ball.

MHc=160 GeV, era=2018, both SR1E2Mu and SR3Mu channels.
Uses FitCanvasWithRatio from Common/Tools/plotter.py for CMS-style plots.

Usage:
    cd SignalRegionStudyV2
    source setup.sh
    python3 fitting/test_dg_fit.py
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

PLOT_DIR = os.path.join(WORKDIR, "SignalRegionStudyV2", "fitting", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def run_fits(file_path, mA_nominal, channel, masspoint):
    """Fit Voigt, DG, DCB and produce a CMS-style validation plot."""

    # -- Step 1: wide Voigt pre-fit to get scale --
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

    # -- Step 2: narrow histogram around peak --
    fit_lo = fitted_mA - 10.0 * vw
    fit_hi = fitted_mA + 10.0 * vw
    nbins = 100

    rdf2 = ROOT.RDataFrame("Central", file_path)
    rdf2 = rdf2.Filter(f"mass >= {fit_lo} && mass <= {fit_hi}")
    hist = rdf2.Histo1D(
        ROOT.RDF.TH1DModel("h_fit", "", nbins, fit_lo, fit_hi), "mass", "weight"
    ).GetValue().Clone("h_fit_c")
    hist.SetDirectory(0)

    mass = ROOT.RooRealVar("mass", "mass", fit_lo, fit_hi)
    roo_data = ROOT.RooDataHist("data", "", ROOT.RooArgList(mass), hist)

    # -- Voigt --
    v_mA = ROOT.RooRealVar("v_mA", "mA", fitted_mA, fit_lo, fit_hi)
    v_width = ROOT.RooRealVar("v_width", "w", pre_w.getVal(), 0.0, 5.0 * vw)
    v_sigma = ROOT.RooRealVar("v_sigma", "s", pre_s.getVal(), 0.0, 5.0 * vw)
    voigt = ROOT.RooVoigtian("voigt", "", mass, v_mA, v_width, v_sigma)
    voigt.fitTo(roo_data, ROOT.RooFit.SumW2Error(True),
                ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))
    voigt_w = sqrt(v_width.getVal()**2 + v_sigma.getVal()**2)

    # -- Double Gaussian --
    dg_mu = ROOT.RooRealVar("dg_mu", "mu", fitted_mA, fit_lo, fit_hi)
    dg_s1 = ROOT.RooRealVar("dg_s1", "s1", 0.7 * vw, 0.01 * vw, 2.0 * vw)
    dg_s2 = ROOT.RooRealVar("dg_s2", "s2", 1.5 * vw, 0.5 * vw, 5.0 * vw)
    dg_frac = ROOT.RooRealVar("dg_frac", "f", 0.6, 0.01, 0.99)
    g1 = ROOT.RooGaussian("g1", "", mass, dg_mu, dg_s1)
    g2 = ROOT.RooGaussian("g2", "", mass, dg_mu, dg_s2)
    dg = ROOT.RooAddPdf("dg", "", g1, g2, dg_frac)
    dg.fitTo(roo_data, ROOT.RooFit.SumW2Error(True),
             ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))
    s1v, s2v, fv = dg_s1.getVal(), dg_s2.getVal(), dg_frac.getVal()
    if s1v > s2v:
        s1v, s2v, fv = s2v, s1v, 1.0 - fv
    dg_seff = sqrt(fv * s1v**2 + (1.0 - fv) * s2v**2)

    # -- DCB --
    dcb_x0 = ROOT.RooRealVar("dcb_x0", "x0", fitted_mA, fit_lo, fit_hi)
    dcb_sL = ROOT.RooRealVar("dcb_sL", "sL", 0.8 * vw, 0.01 * vw, 3.0 * vw)
    dcb_sR = ROOT.RooRealVar("dcb_sR", "sR", 0.8 * vw, 0.01 * vw, 3.0 * vw)
    dcb_aL = ROOT.RooRealVar("dcb_aL", "aL", 1.5, 0.5, 10.0)
    dcb_nL = ROOT.RooRealVar("dcb_nL", "nL", 2.0, 0.1, 50.0)
    dcb_aR = ROOT.RooRealVar("dcb_aR", "aR", 1.5, 0.5, 10.0)
    dcb_nR = ROOT.RooRealVar("dcb_nR", "nR", 2.0, 0.1, 50.0)
    dcb = ROOT.RooCrystalBall("dcb", "", mass, dcb_x0,
                               dcb_sL, dcb_sR, dcb_aL, dcb_nL, dcb_aR, dcb_nR)
    dcb.fitTo(roo_data, ROOT.RooFit.SumW2Error(True),
              ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))
    dcb_seff = sqrt(0.5 * (dcb_sL.getVal()**2 + dcb_sR.getVal()**2))

    # -- Build fit_models dict for FitCanvasWithRatio --
    # OrderedDict of {name: (RooAbsPdf, npar, line_style)}
    fit_models = OrderedDict([
        ("Voigtian", (voigt, 3, ROOT.kSolid)),
        ("DG",       (dg,    4, ROOT.kSolid)),
        ("DCB",      (dcb,   6, ROOT.kSolid)),
    ])

    # -- Plot with FitCanvasWithRatio --
    config = {
        "era": ERA,
        "xTitle": "m_{A} [GeV]",
        "yTitle": "Events",
        "rTitle": "Fit / MC",
        "rRange": [0.5, 1.5],
        "channel": channel,
        "channelPosX": 0.2,
        "channelPosY": 0.74,
        "channelFont": 61,
        "channelSize": 0.04,
        "masspoint": masspoint,
        "masspointPosX": 0.2,
        "masspointPosY": 0.69,
        "masspointFont": 61,
        "masspointSize": 0.04,
        "legend": [0.2, 0.5, 0.45, 0.65],
        "legendTextSize": 0.03,
        "iPos": 0,
        "maxDigits": 3,
    }

    canvas = FitCanvasWithRatio(roo_data, mass, hist, fit_models, config)
    canvas.drawPadUp()
    canvas.drawPadDown()
    canvas.drawMasspoint()

    outpath = os.path.join(PLOT_DIR, f"fit_{channel}_{masspoint}.png")
    canvas.canv.SaveAs(outpath)

    # -- Collect results --
    # Get chi2 values from the canvas
    chi2_v = canvas.chi2_values["Voigtian"]
    chi2_dg = canvas.chi2_values["DG"]
    chi2_dcb = canvas.chi2_values["DCB"]

    return {
        "channel": channel, "masspoint": masspoint, "mA_nominal": mA_nominal,
        "v_voigt_width": voigt_w, "v_chi2": chi2_v,
        "dg_s1": s1v, "dg_s2": s2v, "dg_frac": fv,
        "dg_sigma_eff": dg_seff, "dg_chi2": chi2_dg,
        "dcb_sL": dcb_sL.getVal(), "dcb_sR": dcb_sR.getVal(),
        "dcb_sigma_eff": dcb_seff, "dcb_chi2": chi2_dcb,
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
            print(f"done → {os.path.basename(res['plot'])}")

    # Summary table
    print("\n" + "=" * 130)
    print(f"{'Channel':>10} {'MassPoint':>15} | "
          f"{'V:chi2':>7} {'V:sig':>6} | "
          f"{'DG:chi2':>7} {'DG:s1':>6} {'DG:s2':>6} {'DG:f':>5} {'DG:se':>6} | "
          f"{'DCB:chi2':>8} {'DCB:sL':>7} {'DCB:sR':>7} {'DCB:se':>7} | "
          f"{'Best':>5}")
    print("=" * 130)

    for r in results:
        chi2s = {"V": r["v_chi2"], "DG": r["dg_chi2"], "DCB": r["dcb_chi2"]}
        best = min(chi2s, key=chi2s.get)
        print(f"{r['channel']:>10} {r['masspoint']:>15} | "
              f"{r['v_chi2']:7.1f} {r['v_voigt_width']:6.3f} | "
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
            print(f"{ma:5.0f} {r1['v_voigt_width']:8.3f} {r1['dcb_sigma_eff']:10.3f} "
                  f"{r3['v_voigt_width']:8.3f} {r3['dcb_sigma_eff']:8.3f} {rdcb:10.3f}")

    print(f"\nPlots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
