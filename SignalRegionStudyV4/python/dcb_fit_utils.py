"""DCB(+Chebychev2) fitting and plotting helpers for the mA-interpolation
chain.

Two fit entry points:

- ``fit_dcb_with_errors``: a deliberate VERBATIM mirror of the production
  fit core ``makeBinnedTemplates.fit_dcb`` plus read-out of what production
  discards (RooFitResult status/covQual, per-parameter errors, at-bound
  flags, dataset stats). The fit configuration — windows, initial values,
  bounds, SumW2Error — is copied unchanged so central values reproduce the
  production fit exactly. Do not modify; datacard bytes are frozen by the
  V4 reproduction contract.

- ``fit_dcb_bkg``: the adopted interpolation fit — same two-stage
  structure, with optionally frozen nL/nR (breaks the alpha-n degeneracy;
  all tail mA-dependence funnels into the alphas) and an optional 2nd-order
  Chebychev combinatoric background for the SR3Mu pairing variants:
  S(m) = fsig*DCB + (1-fsig)*Chebychev2(c1, c2) — can be flat, matching the
  observed wrong-pairing plateau (docs/INTERPOLATION.md: expo and Bernstein
  backgrounds were tried and rejected). Points where fsig hits
  interpolation_config.FSIG_DROP_THRESHOLD refit as pure DCB (unconstrained
  background parameters otherwise inflate every error and destabilize the
  parametrization anchors).

SumW2 caveat: with SumW2Error(True), covQual often reads -1 ("unknown");
quality gating treats only 0 <= covQual < 2 as a failure.
"""
from math import sqrt

import ROOT

import interpolation_config


def _at_bound(var, rel_tol=1e-3):
    """True when a RooRealVar sits at (or hugs) one of its range bounds."""
    lo, hi = var.getMin(), var.getMax()
    tol = rel_tol * (hi - lo)
    return (var.getVal() - lo) < tol or (hi - var.getVal()) < tol


def fit_dcb_with_errors(chain, mA_nominal):
    """Two-stage Double Crystal Ball fit on a ``TChain('Central')`` holding
    ``mass`` and ``weight`` branches; returns parameters WITH errors.

    Fit configuration mirrors makeBinnedTemplates.fit_dcb verbatim.
    """
    if chain.GetEntries() <= 0:
        raise RuntimeError("No signal entries found for DCB fit")

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

    wide_lo = max(mA_nominal - mA_nominal / 3.0, 12.0)
    wide_hi = mA_nominal + mA_nominal / 3.0

    mass_w = ROOT.RooRealVar("mass", "mass", wide_lo, wide_hi)
    weight_w = ROOT.RooRealVar("weight", "weight", -10, 10)
    ds_wide = ROOT.RooDataSet(
        "ds_wide", "", ROOT.RooArgSet(mass_w, weight_w),
        ROOT.RooFit.Import(chain),
        ROOT.RooFit.Cut(f"mass >= {wide_lo} && mass <= {wide_hi}"),
        ROOT.RooFit.WeightVar("weight"),
    )

    pre_x0 = ROOT.RooRealVar("pre_x0", "x0", mA_nominal, wide_lo, wide_hi)
    pre_sL = ROOT.RooRealVar("pre_sL", "sL", 1.0, 0.01, 10.0)
    pre_sR = ROOT.RooRealVar("pre_sR", "sR", 1.0, 0.01, 10.0)
    pre_aL = ROOT.RooRealVar("pre_aL", "aL", 1.5, 0.5, 10.0)
    pre_nL = ROOT.RooRealVar("pre_nL", "nL", 2.0, 0.1, 50.0)
    pre_aR = ROOT.RooRealVar("pre_aR", "aR", 1.5, 0.5, 10.0)
    pre_nR = ROOT.RooRealVar("pre_nR", "nR", 2.0, 0.1, 50.0)
    pre_dcb = ROOT.RooCrystalBall(
        "pre_dcb", "", mass_w, pre_x0,
        pre_sL, pre_sR, pre_aL, pre_nL, pre_aR, pre_nR
    )
    pre_result = pre_dcb.fitTo(ds_wide, ROOT.RooFit.SumW2Error(True),
                               ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

    fitted_mA = pre_x0.getVal()
    vw = sqrt(0.5 * (pre_sL.getVal()**2 + pre_sR.getVal()**2))

    fit_lo = max(fitted_mA - 10.0 * vw, 12.0)
    fit_hi = fitted_mA + 10.0 * vw

    mass_n = ROOT.RooRealVar("mass", "mass", fit_lo, fit_hi)
    weight_n = ROOT.RooRealVar("weight", "weight", -10, 10)
    ds_narrow = ROOT.RooDataSet(
        "ds_narrow", "", ROOT.RooArgSet(mass_n, weight_n),
        ROOT.RooFit.Import(chain),
        ROOT.RooFit.Cut(f"mass >= {fit_lo} && mass <= {fit_hi}"),
        ROOT.RooFit.WeightVar("weight"),
    )

    dcb_x0 = ROOT.RooRealVar("dcb_x0", "x0", fitted_mA, fit_lo, fit_hi)
    dcb_sL = ROOT.RooRealVar("dcb_sL", "sigmaL", 0.8 * vw, 0.01 * vw, 3.0 * vw)
    dcb_sR = ROOT.RooRealVar("dcb_sR", "sigmaR", 0.8 * vw, 0.01 * vw, 3.0 * vw)
    dcb_aL = ROOT.RooRealVar("dcb_aL", "alphaL", 1.5, 0.5, 10.0)
    dcb_nL = ROOT.RooRealVar("dcb_nL", "nL", 2.0, 0.1, 50.0)
    dcb_aR = ROOT.RooRealVar("dcb_aR", "alphaR", 1.5, 0.5, 10.0)
    dcb_nR = ROOT.RooRealVar("dcb_nR", "nR", 2.0, 0.1, 50.0)
    dcb = ROOT.RooCrystalBall(
        "dcb", "", mass_n, dcb_x0,
        dcb_sL, dcb_sR, dcb_aL, dcb_nL, dcb_aR, dcb_nR
    )
    fit_result = dcb.fitTo(ds_narrow, ROOT.RooFit.SumW2Error(True),
                           ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

    sigma_eff = sqrt(0.5 * (dcb_sL.getVal()**2 + dcb_sR.getVal()**2))

    shape_vars = {
        "x0": dcb_x0, "sigmaL": dcb_sL, "sigmaR": dcb_sR,
        "alphaL": dcb_aL, "nL": dcb_nL, "alphaR": dcb_aR, "nR": dcb_nR,
    }
    params = {
        name: {"value": float(var.getVal()), "error": float(var.getError())}
        for name, var in shape_vars.items()
    }
    at_limit = [name for name, var in shape_vars.items() if _at_bound(var)]

    return {
        "params": params,
        "sigma_eff": float(sigma_eff),
        "fit_lo": float(fit_lo),
        "fit_hi": float(fit_hi),
        "wide_lo": float(wide_lo),
        "wide_hi": float(wide_hi),
        "status": int(fit_result.status()),
        "covQual": int(fit_result.covQual()),
        "pre_status": int(pre_result.status()),
        "entries": int(ds_narrow.numEntries()),
        "sumw": float(ds_narrow.sumEntries()),
        "at_limit": at_limit,
        "window_floored": bool(wide_lo <= 12.0 or fit_lo <= 12.0),
    }


def fit_dcb_bkg(chain, mA_nominal, nL_fixed=None, nR_fixed=None, bkg=None,
                allow_drop=True):
    """Two-stage fit: DCB with optionally frozen nL/nR and an optional
    2nd-order Chebychev combinatoric background (see module docstring).

    bkg: None (pure DCB, SR1E2Mu) or "cheb2" (SR3Mu).
    allow_drop: False disables the FSIG_DROP_THRESHOLD refit-as-pure-DCB
    rule (fit-model variant 'nodrop'): fsig keeps its fitted value and
    error, and c1/c2 stay measured, however small the background is.
    """
    if chain.GetEntries() <= 0:
        raise RuntimeError("No signal entries found for DCB fit")

    ROOT.RooMsgService.instance().setGlobalKillBelow(ROOT.RooFit.WARNING)

    # Wide pre-fit: identical to the production mirror; only sets the
    # narrow window.
    wide_lo = max(mA_nominal - mA_nominal / 3.0, 12.0)
    wide_hi = mA_nominal + mA_nominal / 3.0

    mass_w = ROOT.RooRealVar("mass", "mass", wide_lo, wide_hi)
    weight_w = ROOT.RooRealVar("weight", "weight", -10, 10)
    ds_wide = ROOT.RooDataSet(
        "ds_wide", "", ROOT.RooArgSet(mass_w, weight_w),
        ROOT.RooFit.Import(chain),
        ROOT.RooFit.Cut(f"mass >= {wide_lo} && mass <= {wide_hi}"),
        ROOT.RooFit.WeightVar("weight"),
    )

    pre_x0 = ROOT.RooRealVar("pre_x0", "x0", mA_nominal, wide_lo, wide_hi)
    pre_sL = ROOT.RooRealVar("pre_sL", "sL", 1.0, 0.01, 10.0)
    pre_sR = ROOT.RooRealVar("pre_sR", "sR", 1.0, 0.01, 10.0)
    pre_aL = ROOT.RooRealVar("pre_aL", "aL", 1.5, 0.5, 10.0)
    pre_nL = ROOT.RooRealVar("pre_nL", "nL", 2.0, 0.1, 50.0)
    pre_aR = ROOT.RooRealVar("pre_aR", "aR", 1.5, 0.5, 10.0)
    pre_nR = ROOT.RooRealVar("pre_nR", "nR", 2.0, 0.1, 50.0)
    pre_dcb = ROOT.RooCrystalBall(
        "pre_dcb", "", mass_w, pre_x0,
        pre_sL, pre_sR, pre_aL, pre_nL, pre_aR, pre_nR
    )
    pre_result = pre_dcb.fitTo(ds_wide, ROOT.RooFit.SumW2Error(True),
                               ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

    fitted_mA = pre_x0.getVal()
    vw = sqrt(0.5 * (pre_sL.getVal()**2 + pre_sR.getVal()**2))

    fit_lo = max(fitted_mA - 10.0 * vw, 12.0)
    fit_hi = fitted_mA + 10.0 * vw

    mass_n = ROOT.RooRealVar("mass", "mass", fit_lo, fit_hi)
    weight_n = ROOT.RooRealVar("weight", "weight", -10, 10)
    ds_narrow = ROOT.RooDataSet(
        "ds_narrow", "", ROOT.RooArgSet(mass_n, weight_n),
        ROOT.RooFit.Import(chain),
        ROOT.RooFit.Cut(f"mass >= {fit_lo} && mass <= {fit_hi}"),
        ROOT.RooFit.WeightVar("weight"),
    )

    # Narrow-fit model.
    fix_n = nL_fixed is not None
    dcb_x0 = ROOT.RooRealVar("dcb_x0", "x0", fitted_mA, fit_lo, fit_hi)
    dcb_sL = ROOT.RooRealVar("dcb_sL", "sigmaL", 0.8 * vw, 0.01 * vw, 3.0 * vw)
    dcb_sR = ROOT.RooRealVar("dcb_sR", "sigmaR", 0.8 * vw, 0.01 * vw, 3.0 * vw)
    dcb_aL = ROOT.RooRealVar("dcb_aL", "alphaL", 1.5, 0.5, 10.0)
    dcb_aR = ROOT.RooRealVar("dcb_aR", "alphaR", 1.5, 0.5, 10.0)
    if fix_n:
        dcb_nL = ROOT.RooRealVar("dcb_nL", "nL", nL_fixed)
        dcb_nR = ROOT.RooRealVar("dcb_nR", "nR", nR_fixed)
        dcb_nL.setConstant(True)
        dcb_nR.setConstant(True)
    else:
        dcb_nL = ROOT.RooRealVar("dcb_nL", "nL", 2.0, 0.1, 50.0)
        dcb_nR = ROOT.RooRealVar("dcb_nR", "nR", 2.0, 0.1, 50.0)
    dcb = ROOT.RooCrystalBall(
        "dcb", "", mass_n, dcb_x0,
        dcb_sL, dcb_sR, dcb_aL, dcb_nL, dcb_aR, dcb_nR
    )

    bkg_vars = {}
    if bkg is None:
        model = dcb
        fsig = None
    elif bkg == "cheb2":
        fsig = ROOT.RooRealVar("fsig", "fsig", 0.9, 0.2, 1.0)
        c1 = ROOT.RooRealVar("c1", "c1", 0.0, -1.5, 1.5)
        c2 = ROOT.RooRealVar("c2", "c2", 0.0, -1.5, 1.5)
        bkg_pdf = ROOT.RooChebychev("cheb", "", mass_n,
                                    ROOT.RooArgList(c1, c2))
        bkg_vars = {"c1": c1, "c2": c2}
        model = ROOT.RooAddPdf("model", "", ROOT.RooArgList(dcb, bkg_pdf),
                               ROOT.RooArgList(fsig))
    else:
        raise ValueError(f"Unknown background shape: {bkg}")

    fit_result = model.fitTo(ds_narrow, ROOT.RooFit.SumW2Error(True),
                             ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

    # No-background points: refit as pure DCB (see module docstring).
    bkg_dropped = (allow_drop and fsig is not None
                   and fsig.getVal() > interpolation_config.FSIG_DROP_THRESHOLD)
    if bkg_dropped:
        fit_result = dcb.fitTo(ds_narrow, ROOT.RooFit.SumW2Error(True),
                               ROOT.RooFit.Save(),
                               ROOT.RooFit.PrintLevel(-1))

    sigma_eff = sqrt(0.5 * (dcb_sL.getVal()**2 + dcb_sR.getVal()**2))

    float_vars = {
        "x0": dcb_x0, "sigmaL": dcb_sL, "sigmaR": dcb_sR,
        "alphaL": dcb_aL, "alphaR": dcb_aR,
    }
    if not fix_n:
        float_vars["nL"] = dcb_nL
        float_vars["nR"] = dcb_nR
    params = {
        name: {"value": float(var.getVal()), "error": float(var.getError())}
        for name, var in float_vars.items()
    }
    if fix_n:
        params["nL"] = {"value": float(nL_fixed), "error": 0.0}
        params["nR"] = {"value": float(nR_fixed), "error": 0.0}
    if bkg is not None:
        if bkg_dropped:
            params["fsig"] = {"value": 1.0,
                              "error": interpolation_config.FSIG_ANCHOR_ERROR}
            for name in bkg_vars:
                params[name] = {"value": 0.0, "error": 0.0}
        else:
            params["fsig"] = {"value": float(fsig.getVal()),
                              "error": float(fsig.getError())}
            for name, var in bkg_vars.items():
                params[name] = {"value": float(var.getVal()),
                                "error": float(var.getError())}
    at_limit = [name for name, var in float_vars.items() if _at_bound(var)]

    out = {
        "params": params,
        "sigma_eff": float(sigma_eff),
        "fit_lo": float(fit_lo),
        "fit_hi": float(fit_hi),
        "wide_lo": float(wide_lo),
        "wide_hi": float(wide_hi),
        "status": int(fit_result.status()),
        "covQual": int(fit_result.covQual()),
        "pre_status": int(pre_result.status()),
        "entries": int(ds_narrow.numEntries()),
        "sumw": float(ds_narrow.sumEntries()),
        "at_limit": at_limit,
        "window_floored": bool(wide_lo <= 12.0 or fit_lo <= 12.0),
    }
    if bkg is not None:
        out["bkg_shape"] = bkg
        out["bkg_dropped"] = bkg_dropped
    if fix_n:
        out["fixed_n"] = {"nL": float(nL_fixed), "nR": float(nR_fixed)}
    return out


def bkg_shape_of(fit):
    """Background shape of a stored fit record: None or 'cheb2'."""
    return fit.get("bkg_shape")


def build_model(prefix, mass_var, params):
    """Constant-parameter pdf from a params dict: pure DCB, or DCB+cheb2
    when the dict carries fsig/c1/c2.

    Returns (pdf, keep_alive_objects)."""
    keep = []
    order = ["x0", "sigmaL", "sigmaR", "alphaL", "nL", "alphaR", "nR"]
    for name in order:
        var = ROOT.RooRealVar(f"{prefix}_{name}", name, params[name])
        var.setConstant(True)
        keep.append(var)
    dcb = ROOT.RooCrystalBall(f"{prefix}_dcb", "", mass_var, *keep)
    keep.append(dcb)
    if "fsig" not in params:
        return dcb, keep
    fsig = ROOT.RooRealVar(f"{prefix}_fsig", "fsig", params["fsig"])
    fsig.setConstant(True)
    keep.append(fsig)
    c1 = ROOT.RooRealVar(f"{prefix}_c1", "c1", params["c1"])
    c2 = ROOT.RooRealVar(f"{prefix}_c2", "c2", params["c2"])
    c1.setConstant(True)
    c2.setConstant(True)
    bkg_pdf = ROOT.RooChebychev(f"{prefix}_cheb", "", mass_var,
                                ROOT.RooArgList(c1, c2))
    keep.extend([c1, c2, bkg_pdf])
    model = ROOT.RooAddPdf(f"{prefix}_model", "",
                           ROOT.RooArgList(dcb, bkg_pdf),
                           ROOT.RooArgList(fsig))
    return model, keep


def make_mc_hist(chain, name, fit_lo, fit_hi, nbins=100):
    """Weighted MC mass histogram in the fit window."""
    hist = ROOT.TH1D(name, "", nbins, fit_lo, fit_hi)
    hist.Sumw2()
    chain.Draw(f"mass>>{name}",
               f"weight*(mass >= {fit_lo} && mass <= {fit_hi})", "goff")
    hist.SetDirectory(0)
    return hist


def model_label(fit):
    """Legend label of a stored fit's model."""
    return "DCB+cheb" if bkg_shape_of(fit) == "cheb2" else "DCB"


def bkg_components(fit, model_key, prefix):
    """FitCanvasWithRatio 'components' entries showing the signal and
    background parts of a stored fit's model (empty for pure DCB)."""
    if bkg_shape_of(fit) != "cheb2":
        return []
    return [
        {"model": model_key, "pdf": f"{prefix}_dcb", "label": "DCB (signal)",
         "color": ROOT.kAzure + 1, "style": ROOT.kDashed},
        {"model": model_key, "pdf": f"{prefix}_cheb",
         "label": "cheb (combinatoric)",
         "color": ROOT.kGreen + 2, "style": ROOT.kDotted},
    ]


def canvas_config(era, channel, masspoint, components, **overrides):
    """Common FitCanvasWithRatio config for the interpolation-chain
    canvases."""
    config = {
        "era": era, "xTitle": "m_{A} [GeV]", "yTitle": "Events",
        "rTitle": "Fit / MC", "rRange": [0.5, 1.5],
        "channel": channel, "masspoint": masspoint,
        "channelPosX": 0.2, "channelPosY": 0.74,
        "channelFont": 61, "channelSize": 0.04,
        "masspointPosX": 0.2, "masspointPosY": 0.69,
        "masspointFont": 61, "masspointSize": 0.04,
        "legend": [0.60, 0.62, 0.90, 0.78],
        "legendTextSize": 0.03, "iPos": 0, "maxDigits": 3,
        "colors": [ROOT.kRed],
        "components": components,
    }
    config.update(overrides)
    return config


PARAM_TEX = [
    ("x0", "x_{0}"), ("sigmaL", "#sigma_{L}"), ("sigmaR", "#sigma_{R}"),
    ("alphaL", "#alpha_{L}"), ("nL", "n_{L}"),
    ("alphaR", "#alpha_{R}"), ("nR", "n_{R}"),
    ("fsig", "f_{sig}"), ("c1", "c_{1}"), ("c2", "c_{2}"),
]


def draw_dcb_params(params, sigma_eff=None, x=0.2, y=0.645, size=0.026):
    """Draw fitted parameter values (with errors) on the current pad.

    ``params`` is the {name: {value, error}} dict from the fit record.
    Call with the target pad already selected (canvas.cd(1)).
    """
    lat = ROOT.TLatex()
    lat.SetNDC(True)
    lat.SetTextFont(42)
    lat.SetTextSize(size)
    lat.SetTextAlign(11)
    for name, tex in PARAM_TEX:
        if name not in params:
            continue
        v = params[name]["value"]
        e = params[name]["error"]
        lat.DrawLatex(x, y, f"{tex} = {v:.3f} #pm {e:.3f}")
        y -= size + 0.006
    if sigma_eff is not None:
        lat.DrawLatex(x, y, f"#sigma_{{eff}} = {sigma_eff:.3f}")


def draw_dcb_param_comparison(direct_params, predicted, x=0.2, y=0.645,
                              size=0.024):
    """Draw a direct-fit vs interpolated parameter table on the current pad.

    ``direct_params`` is {name: {value, error}}; ``predicted`` is
    {name: value} from the parametrization evaluation.
    """
    lat = ROOT.TLatex()
    lat.SetNDC(True)
    lat.SetTextFont(42)
    lat.SetTextSize(size)
    lat.SetTextAlign(11)
    lat.DrawLatex(x, y, "direct fit / interpolated")
    y -= size + 0.006
    for name, tex in PARAM_TEX:
        if name not in direct_params or name not in predicted:
            continue
        v = direct_params[name]["value"]
        e = direct_params[name]["error"]
        p = predicted[name]
        lat.DrawLatex(x, y, f"{tex} = {v:.3f} #pm {e:.3f}  /  {p:.3f}")
        y -= size + 0.006


def fit_quality(fit):
    """Classify a fit record: returns (flag, reasons).

    flag is "good" or "bad"; informational conditions (unknown covQual,
    low stats) do not make a fit bad — they are reported by the caller.
    """
    reasons = []
    # A non-zero Minuit status is only disqualifying when the covariance is
    # not trustworthy. status=1 (covariance forced pos-def) and status=3 (EDM
    # above tolerance) routinely accompany covQual>=2, i.e. a full covariance
    # matrix, and those fits are usable; rejecting them throws away anchors
    # that sparse mA grids cannot spare. Genuine failures (observed:
    # status=600/602) carry covQual 0/-1 and are still rejected here.
    if fit["status"] != 0 and not fit["covQual"] >= 2:
        reasons.append(f"status={fit['status']} covQual={fit['covQual']}")
    if 0 <= fit["covQual"] < 2:
        reasons.append(f"covQual={fit['covQual']}")
    if fit["at_limit"]:
        reasons.append("at_limit=" + ",".join(fit["at_limit"]))
    fixed = set(fit.get("fixed_n", {}))
    if fit.get("bkg_dropped"):
        fixed |= set(interpolation_config.BKG_PARAMS)  # zero-error by design
    for name, pv in fit["params"].items():
        if name in fixed:
            continue  # frozen parameter: error 0 by construction
        err = pv["error"]
        if not (err > 0) or err != err:  # zero, negative or NaN
            reasons.append(f"{name}_error={err}")
    return ("bad" if reasons else "good"), reasons
