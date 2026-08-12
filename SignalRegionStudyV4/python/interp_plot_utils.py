"""cmsstyle plotting helpers for the mA-interpolation chain.

Pattern: plain cmsstyle canvases + TGraph/TH1 primitives (the
``plotLimits.py`` graph idiom — ``CMS.cmsCanvas``/``CMS.cmsDiCanvas`` +
``CMS.cmsObjectDraw`` + ``CMS.cmsLeg``), NOT the ``Common/Tools/plotter.py``
histogram canvas classes (those stay reserved for the per-point DCB fit and
shape-closure overlays, which already use ``FitCanvasWithRatio`` and are
untouched by this module). ``Common/Tools/plotter.py`` itself is never
edited here — it is shared across every analysis module.

Every public plotting function is import-guarded at the call site is NOT
required here (cmsstyle/ROOT are hard dependencies of this module, exactly
like the rest of the interpolation chain); callers that want a JSON-only
degrade path should catch ImportError around the whole plotting call.
"""
import os
from array import array

import numpy as np
import ROOT
import cmsstyle as CMS

from plotter import EnergyInfo, LumiInfo, PALETTE_LONG, get_CoM_energy

ROOT.gROOT.SetBatch(True)

_HELD_OUT_COLOR = ROOT.TColor.GetColor("#5790fc")
_CURVE_COLOR = ROOT.TColor.GetColor("#e42536")
_BAND_COLOR = ROOT.TColor.GetColor("#e42536")


# --------------------------------------------------------------- primitives

def _set_lumi_energy(period_or_era):
    CMS.SetLumi(LumiInfo[period_or_era], run=period_or_era)
    CMS.SetEnergy(get_CoM_energy(period_or_era))


def graph_canvas(name, xtitle, ytitle, xmin, xmax, ymin, ymax,
                 period_or_era, logy=False, extra_text="Simulation Preliminary"):
    """Single-pad square cmsstyle canvas for a curve/points plot."""
    CMS.SetExtraText(extra_text)
    _set_lumi_energy(period_or_era)
    canv = CMS.cmsCanvas(name, xmin, xmax, ymin, ymax, xtitle, ytitle,
                         square=True, iPos=11, extraSpace=0.02)
    if logy:
        canv.SetLogy()
    canv.cd()
    return canv


def dicanvas_with_pulls(name, xtitle, ytitle, xmin, xmax, ymin, ymax,
                        period_or_era, logy=False,
                        extra_text="Simulation Preliminary",
                        pull_range=(-5, 5)):
    """Two-pad cmsstyle canvas: value (+band) on top, pulls below."""
    CMS.SetExtraText(extra_text)
    _set_lumi_energy(period_or_era)
    canv = CMS.cmsDiCanvas(name, xmin, xmax, ymin, ymax,
                           pull_range[0], pull_range[1],
                           xtitle, ytitle, "pull",
                           square=True, iPos=11, extraSpace=0.02)
    if logy:
        canv.cd(1).SetLogy()
    return canv


def curve_graph(xs, ys, color=None):
    g = ROOT.TGraph(len(xs), array('d', xs), array('d', ys))
    g.SetLineWidth(2)
    g.SetLineColor(color if color is not None else _CURVE_COLOR)
    return g


def band_graph(xs, ys, err_lo, err_hi=None, color=None):
    """Symmetric (err_hi=None -> err_lo used both sides) or asymmetric
    1-sigma band as a filled TGraphAsymmErrors."""
    n = len(xs)
    g = ROOT.TGraphAsymmErrors(n)
    for i in range(n):
        g.SetPoint(i, xs[i], ys[i])
        hi = err_hi[i] if err_hi is not None else err_lo[i]
        g.SetPointError(i, 0.0, 0.0, err_lo[i], hi)
    g.SetFillColorAlpha(color if color is not None else _BAND_COLOR, 0.25)
    g.SetLineWidth(0)
    return g


def points_graph(xs, ys, yerrs, filled=True, color=ROOT.kBlack):
    n = len(xs)
    g = ROOT.TGraphErrors(n, array('d', xs), array('d', ys),
                          array('d', [0.0] * n), array('d', yerrs))
    g.SetMarkerStyle(ROOT.kFullCircle if filled else ROOT.kOpenSquare)
    g.SetMarkerSize(1.0)
    g.SetMarkerColor(color)
    g.SetLineColor(color)
    return g


def _save(canv, outdir, name):
    os.makedirs(outdir, exist_ok=True)
    canv.SaveAs(os.path.join(outdir, f"{name}.png"))
    canv.Close()


# --------------------------------------------------------- shape parameters

_PARAM_TITLES = {
    "x0": "x_{0} [GeV]", "sigmaL": "#sigma_{L} [GeV]",
    "sigmaR": "#sigma_{R} [GeV]", "alphaL": "#alpha_{L}",
    "alphaR": "#alpha_{R}", "fsig": "f_{sig}",
    "c1": "c_{1}", "c2": "c_{2}",
}


def plot_parameter_vs_mA(cat_key, param, points, fit_info, outdir, all_ma,
                         mhc, period):
    """One (category, parameter) parametrization: curve + band on top,
    fit/held-out points, pulls below."""
    import interpolation_config

    xgrid = np.linspace(min(all_ma) - 5, max(all_ma) + 5, 200)
    ygrid = np.asarray(interpolation_config.eval_param(fit_info, xgrid), float)

    used, held = points["used"], points["held_out"]
    all_y = list(ygrid) + used["value"] + held["value"]
    ymin, ymax = min(all_y), max(all_y)
    pad = 0.15 * (ymax - ymin) if ymax > ymin else 0.1 * max(abs(ymax), 1.0)

    canv = dicanvas_with_pulls(
        f"param_{cat_key}_{param}", "m_{A} [GeV]",
        _PARAM_TITLES.get(param, param), min(all_ma) - 5, max(all_ma) + 5,
        ymin - pad, ymax + pad, period)

    canv.cd(1)
    form = fit_info.get("form")
    if form == "logitpoly":
        z = np.polyval(np.array(fit_info["coeffs"]), xgrid)
        band = _poly_band(fit_info, xgrid)
        lo = 1.0 / (1.0 + np.exp(-(z - band)))
        hi = 1.0 / (1.0 + np.exp(-(z + band)))
    else:
        band = _poly_band(fit_info, xgrid)
        lo, hi = ygrid - band, ygrid + band
    g_band = band_graph(list(xgrid), list((np.asarray(lo) + np.asarray(hi)) / 2.0),
                        list((np.asarray(hi) - np.asarray(lo)) / 2.0))
    CMS.cmsObjectDraw(g_band, "E3")
    g_curve = curve_graph(list(xgrid), list(ygrid))
    CMS.cmsObjectDraw(g_curve, "L")
    leg = CMS.cmsLeg(0.62, 0.68, 0.90, 0.88, textSize=0.030)
    if used["mA"]:
        g_used = points_graph(used["mA"], used["value"], used["error"], filled=True)
        CMS.cmsObjectDraw(g_used, "PE")
        leg.AddEntry(g_used, "fit points", "pe")
    if held["mA"]:
        g_held = points_graph(held["mA"], held["value"], held["error"],
                              filled=False, color=_HELD_OUT_COLOR)
        CMS.cmsObjectDraw(g_held, "PE")
        leg.AddEntry(g_held, "held-out points", "pe")
    label = ROOT.TLatex()
    label.SetNDC(True)
    label.SetTextFont(42)
    label.SetTextSize(0.032)
    order_txt = ("logit" if form == "logitpoly" else "") + f"pol{fit_info['chosen_order']}"
    label.DrawLatex(0.20, 0.83,
                    f"{cat_key}: {order_txt}, "
                    f"#chi^{{2}}/ndf={fit_info['chi2']:.1f}/{fit_info['ndf']}")
    canv.cd(1).RedrawAxis()

    canv.cd(2)
    ref = ROOT.TLine()
    ref.SetLineStyle(ROOT.kDotted)
    ref.DrawLine(min(all_ma) - 5, 0.0, max(all_ma) + 5, 0.0)
    for group, filled, color in ((used, True, ROOT.kBlack),
                                 (held, False, _HELD_OUT_COLOR)):
        if not group["mA"]:
            continue
        pred = np.asarray(interpolation_config.eval_param(
            fit_info, np.array(group["mA"])), float)
        pull = (np.array(group["value"]) - pred) / np.array(group["error"])
        g = points_graph(group["mA"], list(pull), [0.0] * len(pull),
                         filled=filled, color=color)
        CMS.cmsObjectDraw(g, "P")
    canv.cd(2).RedrawAxis()

    _save(canv, outdir, f"{param}_vs_mA.{cat_key}")


def _poly_band(fit_info, xgrid):
    deg = len(fit_info["coeffs"]) - 1
    V = np.vander(np.asarray(xgrid, float), deg + 1)
    var = np.einsum("ij,jk,ik->i", V, np.asarray(fit_info["cov"]), V)
    return np.sqrt(np.clip(var, 0.0, None))


# -------------------------------------------------------------- yield model

def plot_yield_period_model(mhc, period, model, polys, merged, fit_ma,
                            outdir, eval_rec, rec_band, inv_logit, fsig_of):
    """One PNG per yield-model component of one run period: G (x2), S,
    p_high, f_sr1e2mu, SR3Mu f overlay."""
    grid = np.linspace(min(fit_ma) - 3, max(fit_ma) + 3, 200)
    fr = model["fractions"][period]

    for tot_channel in ("SR1E2Mu", "SR3Mu"):
        rec = model["totals"][period][tot_channel]["G"]
        pu = rec["points_used"]
        canv = graph_canvas(f"G_{period}_{tot_channel}", "m_{A} [GeV]",
                            "N_{total}", min(fit_ma) - 3, max(fit_ma) + 3,
                            0.5 * min(np.exp(pu["y"])), 2.0 * max(np.exp(pu["y"])),
                            period, logy=True)
        yv = np.exp(eval_rec(rec, grid))
        band = rec_band(rec, grid)
        g_band = band_graph(list(grid), list(yv),
                            list(yv * (1 - np.exp(-band))),
                            list(yv * (np.exp(band) - 1)))
        CMS.cmsObjectDraw(g_band, "E3")
        CMS.cmsObjectDraw(curve_graph(list(grid), list(yv)), "L")
        pts_y = list(np.exp(np.array(pu["y"])))
        pts_e = [y * e for y, e in zip(pts_y, pu["err"])]
        CMS.cmsObjectDraw(points_graph(pu["x"], pts_y, pts_e), "PE")
        lat = ROOT.TLatex()
        lat.SetNDC(True)
        lat.SetTextFont(42)
        lat.SetTextSize(0.032)
        lat.DrawLatex(0.20, 0.83,
                     f"G {tot_channel}: log-pol{rec['chosen_order']}, "
                     f"#chi^{{2}}/ndf={rec['chi2']:.1f}/{rec['ndf']}")
        canv.RedrawAxis()
        _save(canv, outdir, f"model.{period}.G_{tot_channel}")

    rec = fr["S"]
    pu = rec["points_used"]
    canv = graph_canvas(f"S_{period}", "m_{A} [GeV]", "S (containment)",
                        min(fit_ma) - 3, max(fit_ma) + 3,
                        0.8 * min(pu["y"]), 1.2 * max(pu["y"]), period)
    yv = eval_rec(rec, grid)
    band = rec_band(rec, grid)
    CMS.cmsObjectDraw(band_graph(list(grid), list(yv), list(band)), "E3")
    CMS.cmsObjectDraw(curve_graph(list(grid), list(yv)), "L")
    CMS.cmsObjectDraw(points_graph(pu["x"], pu["y"], pu["err"]), "PE")
    canv.RedrawAxis()
    _save(canv, outdir, f"model.{period}.S")

    rec = fr["p_high_logit"]
    pu = rec["points_used"]
    canv = graph_canvas(f"phigh_{period}", "m_{A} [GeV]", "p_{high}",
                        min(fit_ma) - 3, max(fit_ma) + 3, 0.0, 1.0, period)
    yv = inv_logit(eval_rec(rec, grid))
    CMS.cmsObjectDraw(curve_graph(list(grid), list(yv)), "L")
    CMS.cmsObjectDraw(points_graph(pu["x"], list(inv_logit(np.array(pu["y"]))),
                                   [0.0] * len(pu["x"])), "PE")
    canv.RedrawAxis()
    _save(canv, outdir, f"model.{period}.p_high")

    rec = fr["f_sr1e2mu"]
    pu = rec["points_used"]
    canv = graph_canvas(f"fsr1e2mu_{period}", "m_{A} [GeV]", "f_{SR1E2Mu}",
                        min(fit_ma) - 3, max(fit_ma) + 3,
                        0.8 * min(pu["y"]), 1.2 * max(pu["y"]), period)
    yv = eval_rec(rec, grid)
    band = rec_band(rec, grid)
    CMS.cmsObjectDraw(band_graph(list(grid), list(yv), list(band)), "E3")
    CMS.cmsObjectDraw(curve_graph(list(grid), list(yv)), "L")
    CMS.cmsObjectDraw(points_graph(pu["x"], pu["y"], pu["err"]), "PE")
    canv.RedrawAxis()
    _save(canv, outdir, f"model.{period}.f_SR1E2Mu")

    canv = graph_canvas(f"foverlay_{period}", "m_{A} [GeV]", "f = S #upoint p / f^{sig}",
                        min(fit_ma) - 3, max(fit_ma) + 3, 1e-4, 1.0, period,
                        logy=True)
    leg = CMS.cmsLeg(0.62, 0.72, 0.90, 0.88, textSize=0.030)
    for ch, color in (("SR3Mu_lowM", PALETTE_LONG[1]),
                      ("SR3Mu_highM", PALETTE_LONG[2])):
        ma = sorted(merged[ch])
        if not ma:
            continue
        g = points_graph(ma, [merged[ch][m]["f"] for m in ma],
                         [merged[ch][m]["ferr"] for m in ma], color=color)
        CMS.cmsObjectDraw(g, "PE")
        S = eval_rec(fr["S"], grid)
        ph = inv_logit(eval_rec(fr["p_high_logit"], grid))
        p = ph if ch == "SR3Mu_highM" else 1 - ph
        fs = np.array([fsig_of(polys, f"{ch}_{period}", m) for m in grid])
        curve = curve_graph(list(grid), list(S * p / fs), color=color)
        CMS.cmsObjectDraw(curve, "L")
        leg.AddEntry(g, ch, "pe")
    canv.RedrawAxis()
    _save(canv, outdir, f"model.{period}.f_SR3Mu_overlay")


def plot_yield_era_grid(mhc, channel, yields, model, polys, fit_ma, outdir,
                        predict_yield):
    """One PNG per (channel, era): measured window yield vs model curve."""
    import run_period_utils

    grid = np.linspace(min(fit_ma) - 3, max(fit_ma) + 3, 150)
    for period, suberas in run_period_utils.RUN_PERIODS.items():
        for era in suberas:
            pts = []
            for mp, rec in yields.items():
                r = rec["channels"].get(channel, {}).get(era)
                if r is not None:
                    pts.append((rec["mA"], r["sumw"], r["err"]))
            pts.sort()
            curve = np.array([predict_yield(model, polys, channel, era, m)
                              for m in grid])
            ymin = 0.5 * max(min(curve[:, 0].min(),
                                 min((v for _m, v, _e in pts), default=1e-3)), 1e-3)
            ymax = 2.0 * max(curve[:, 0].max(),
                             max((v for _m, v, _e in pts), default=1.0))
            canv = graph_canvas(f"yield_{channel}_{era}", "m_{A} [GeV]",
                                "N_{window}", min(fit_ma) - 3, max(fit_ma) + 3,
                                ymin, ymax, era, logy=True)
            g_band = band_graph(list(grid), list(curve[:, 0]), list(curve[:, 1]))
            CMS.cmsObjectDraw(g_band, "E3")
            CMS.cmsObjectDraw(curve_graph(list(grid), list(curve[:, 0])), "L")
            fit_pts = [(m, v, e) for m, v, e in pts if m in fit_ma]
            held_pts = [(m, v, e) for m, v, e in pts if m not in fit_ma]
            if fit_pts:
                m, v, e = zip(*fit_pts)
                CMS.cmsObjectDraw(points_graph(m, v, e), "PE")
            if held_pts:
                m, v, e = zip(*held_pts)
                CMS.cmsObjectDraw(points_graph(m, v, e, filled=False,
                                               color=_HELD_OUT_COLOR), "PE")
            lat = ROOT.TLatex()
            lat.SetNDC(True)
            lat.SetTextFont(42)
            lat.SetTextSize(0.032)
            lat.DrawLatex(0.20, 0.83, f"{channel}, {era}")
            canv.RedrawAxis()
            _save(canv, outdir, f"model_grid.{channel}.{era}")


def plot_yield_residuals(closure, mhc, outdir):
    """One PNG per (channel, period): relative residual vs mA, one
    colored series per era (filled = in-sample, open = held-out)."""
    import interpolation_config
    import run_period_utils

    for channel in interpolation_config.STUDY_CHANNELS:
        for period, suberas in run_period_utils.RUN_PERIODS.items():
            canv = graph_canvas(f"resid_{channel}_{period}", "m_{A} [GeV]",
                                "(pred - meas) / meas [%]",
                                0, 200, -20, 20, period)
            leg = CMS.cmsLeg(0.62, 0.90 - 0.045 * len(suberas), 0.90, 0.90,
                             textSize=0.028)
            xmin, xmax = 1e9, -1e9
            for color, era in zip(PALETTE_LONG, suberas):
                fit_pts, held_pts = [], []
                for entry in closure.values():
                    rec = entry["scalar"].get(channel, {}).get(era)
                    if rec is None:
                        continue
                    xmin = min(xmin, entry["mA"])
                    xmax = max(xmax, entry["mA"])
                    target = fit_pts if entry["in_sample"] else held_pts
                    target.append((entry["mA"], 100.0 * rec["rel"]))
                if fit_pts:
                    m, v = zip(*sorted(fit_pts))
                    g = points_graph(m, v, [0.0] * len(m), color=color)
                    CMS.cmsObjectDraw(g, "P")
                    leg.AddEntry(g, era, "p")
                if held_pts:
                    m, v = zip(*sorted(held_pts))
                    CMS.cmsObjectDraw(points_graph(m, v, [0.0] * len(m),
                                                   filled=False, color=color), "P")
            ref = ROOT.TLine()
            ref.SetLineStyle(ROOT.kDotted)
            ref.DrawLine(max(xmin, 0), 0.0, max(xmax, 1), 0.0)
            canv.RedrawAxis()
            _save(canv, outdir, f"residuals.{channel}.{period}")


def plot_yield_loo_grid(mhc, channel, yields, loo, outdir,
                        model=None, polys=None, predict_yield=None):
    """One PNG per (channel, era): measured window yields (filled black)
    with the leave-one-out predictions at every grid point overlaid (open
    red; grid-endpoint extrapolations open grey), and — when the adopted
    model is passed — the full-grid fit curve with its 1-sigma band. The
    visual counterpart of the loo_uncertainties.json norm table."""
    import run_period_utils

    grey = ROOT.TColor.GetColor("#9c9ca1")
    for period, suberas in run_period_utils.RUN_PERIODS.items():
        for era in suberas:
            meas = []
            for rec in yields.values():
                r = rec["channels"].get(channel, {}).get(era)
                if r is not None:
                    meas.append((rec["mA"], r["sumw"], r["err"]))
            meas.sort()
            pred, pred_ex = [], []
            for mA, entry in sorted(loo.items()):
                rec = entry["scalar"].get(channel, {}).get(era)
                if rec is None:
                    continue
                tgt = pred_ex if rec.get("extrapolation") else pred
                tgt.append((mA, rec["n_pred"], rec["err_pred"]))
            if not meas:
                continue
            all_m = [m for m, _v, _e in meas]
            curve = None
            if predict_yield is not None:
                xgrid = np.linspace(min(all_m), max(all_m), 150)
                curve = np.array([predict_yield(model, polys, channel, era, m)
                                  for m in xgrid])
            # Range from the measured points, the usable predictions and
            # the fit curve only: an endpoint extrapolation can be off by
            # orders of magnitude and would flatten everything else.
            all_v = ([v for _m, v, _e in meas]
                     + [v for _m, v, _e in pred]
                     + ([] if curve is None else list(curve[:, 0])))
            ymin = 0.5 * max(min(all_v), 1e-3)
            ymax = 2.0 * max(all_v)
            canv = graph_canvas(f"loo_{channel}_{era}", "m_{A} [GeV]",
                                "N_{window}", min(all_m) - 3, max(all_m) + 3,
                                ymin, ymax, era, logy=True)
            leg = CMS.cmsLeg(0.55, 0.70, 0.90, 0.88, textSize=0.028)
            g_curve = None
            if curve is not None:
                CMS.cmsObjectDraw(band_graph(list(xgrid), list(curve[:, 0]),
                                             list(curve[:, 1])), "E3")
                g_curve = curve_graph(list(xgrid), list(curve[:, 0]))
                CMS.cmsObjectDraw(g_curve, "L")
            m, v, e = zip(*meas)
            g_meas = points_graph(m, v, e)
            CMS.cmsObjectDraw(g_meas, "PE")
            leg.AddEntry(g_meas, "measured MC", "pe")
            if g_curve is not None:
                leg.AddEntry(g_curve, "full-grid model", "l")
            if pred:
                m, v, e = zip(*pred)
                g_pred = points_graph(m, v, e, filled=False,
                                      color=_HELD_OUT_COLOR)
                CMS.cmsObjectDraw(g_pred, "PE")
                leg.AddEntry(g_pred, "LOO prediction", "pe")
            if pred_ex:
                m, v, e = zip(*pred_ex)
                g_ex = points_graph(m, v, e, filled=False, color=grey)
                CMS.cmsObjectDraw(g_ex, "PE")
                leg.AddEntry(g_ex, "LOO (extrapolation)", "pe")
            lat = ROOT.TLatex()
            lat.SetNDC(True)
            lat.SetTextFont(42)
            lat.SetTextSize(0.032)
            lat.DrawLatex(0.20, 0.70, f"{channel}, {era}")
            canv.RedrawAxis()
            _save(canv, outdir, f"loo_grid.{channel}.{era}")


def plot_yield_loo_residuals(mhc, channel, loo, outdir):
    """One PNG per (channel, period): LOO relative residual vs mA, one
    colored series per era (filled = usable, open = extrapolation)."""
    import run_period_utils

    for period, suberas in run_period_utils.RUN_PERIODS.items():
        # Range from the usable points only — an endpoint extrapolation can
        # be off by orders of magnitude (its marker just leaves the frame).
        vals = [100.0 * abs(rec["rel"])
                for entry in loo.values()
                for era in suberas
                for rec in [entry["scalar"].get(channel, {}).get(era)]
                if rec is not None and not rec.get("extrapolation")]
        yr = max(20.0, 1.2 * max(vals, default=0.0))
        xs = sorted(loo)
        canv = graph_canvas(f"loo_resid_{channel}_{period}", "m_{A} [GeV]",
                            "(pred - meas) / meas [%]",
                            min(xs) - 3, max(xs) + 3, -yr, yr, period)
        leg = CMS.cmsLeg(0.62, 0.90 - 0.045 * len(suberas), 0.90, 0.90,
                         textSize=0.028)
        for color, era in zip(PALETTE_LONG, suberas):
            used, extrap = [], []
            for mA, entry in sorted(loo.items()):
                rec = entry["scalar"].get(channel, {}).get(era)
                if rec is None:
                    continue
                tgt = extrap if rec.get("extrapolation") else used
                tgt.append((mA, 100.0 * rec["rel"]))
            if used:
                m, v = zip(*used)
                g = points_graph(m, v, [0.0] * len(m), color=color)
                CMS.cmsObjectDraw(g, "P")
                leg.AddEntry(g, era, "p")
            if extrap:
                m, v = zip(*extrap)
                CMS.cmsObjectDraw(points_graph(m, v, [0.0] * len(m),
                                               filled=False, color=color), "P")
        ref = ROOT.TLine()
        ref.SetLineStyle(ROOT.kDotted)
        ref.DrawLine(min(xs), 0.0, max(xs), 0.0)
        canv.RedrawAxis()
        _save(canv, outdir, f"loo_residuals.{channel}.{period}")


def plot_yield_template_closure(cat_key, mp, hist, pred, n_pred, err_pred,
                                chi2, ndf, period, outdir):
    """Absolute-normalization closure: MC hist vs predicted template, with
    a Pred/MC ratio panel."""
    nbins = hist.GetNbinsX()
    lo, hi = hist.GetXaxis().GetXmin(), hist.GetXaxis().GetXmax()
    mc = np.array([hist.GetBinContent(i) for i in range(1, nbins + 1)])
    canv = CMS.cmsDiCanvas(
        f"tmpl_{cat_key}_{mp}", lo, hi, 0.0, 1.3 * max(mc.max(), pred.max()),
        0.5, 1.5, "m(#mu#mu) [GeV]", "Events / bin", "Pred / MC",
        square=True, iPos=11, extraSpace=0.02)
    CMS.SetExtraText("Simulation Preliminary")
    _set_lumi_energy(period)
    canv.cd(1)
    h_pred = hist.Clone(f"h_pred_{cat_key}_{mp}")
    h_pred.Reset()
    for i in range(1, nbins + 1):
        h_pred.SetBinContent(i, pred[i - 1])
    CMS.cmsObjectDraw(h_pred, "hist", LineColor=_CURVE_COLOR, LineWidth=2)
    CMS.cmsObjectDraw(hist, "PE", MarkerStyle=ROOT.kFullCircle, MarkerSize=0.8)
    leg = CMS.cmsLeg(0.55, 0.72, 0.90, 0.88, textSize=0.030)
    leg.AddEntry(hist, "MC", "pe")
    leg.AddEntry(h_pred, f"predicted (N={n_pred:.1f}#pm{err_pred:.1f})", "l")
    lat = ROOT.TLatex()
    lat.SetNDC(True)
    lat.SetTextFont(42)
    lat.SetTextSize(0.030)
    lat.DrawLatex(0.20, 0.83, f"{cat_key}  {mp}")
    lat.DrawLatex(0.20, 0.79, f"#chi^{{2}}/ndf={chi2:.1f}/{ndf}")
    canv.cd(1).RedrawAxis()

    canv.cd(2)
    ratio = hist.Clone(f"h_ratio_{cat_key}_{mp}")
    for i in range(1, nbins + 1):
        denom = mc[i - 1]
        ratio.SetBinContent(i, pred[i - 1] / denom if denom > 0 else 0.0)
        ratio.SetBinError(i, 0.0)
    ref = ROOT.TLine()
    ref.SetLineStyle(ROOT.kDotted)
    ref.DrawLine(lo, 1.0, hi, 1.0)
    CMS.cmsObjectDraw(ratio, "PE", MarkerStyle=ROOT.kFullCircle, MarkerSize=0.8,
                      MarkerColor=_CURVE_COLOR, LineColor=_CURVE_COLOR)
    canv.cd(2).RedrawAxis()
    _save(canv, outdir, f"closure.{cat_key}.{mp}")


# -------------------------------------------------------------- shape deltas

def plot_shape_delta_series(key, syst, model_key, outdir):
    """One PNG per (era|channel key, systematic): dm/dsig/dN vs mA, Up/Down
    curves + donor points."""
    import interpolation_config

    for bucket in ("systs", "pdf_members"):
        if syst not in model_key.get(bucket, {}):
            continue
        directions = model_key[bucket][syst]
        for quantity in interpolation_config.DELTA_QUANTITIES:
            has_any = any(quantity in directions.get(d, {}) for d in directions)
            if not has_any:
                continue
            canv = graph_canvas(f"delta_{key}_{syst}_{quantity}".replace("|", "_"),
                                "m_{A} [GeV]", f"#delta{quantity[1:]} [%]"
                                if quantity != "dN" else "#deltaN [%]",
                                0, 200, -30, 30,
                                model_key.get("period", "Run2"))
            leg = CMS.cmsLeg(0.65, 0.78, 0.90, 0.88, textSize=0.030)
            for direction, color in (("Up", ROOT.kRed + 1), ("Down", ROOT.kAzure + 1)):
                rec = directions.get(direction, {}).get(quantity)
                if not rec:
                    continue
                pts = np.array(rec["points"], float)
                g = points_graph(list(pts[:, 0]), list(100 * pts[:, 1]),
                                 list(100 * pts[:, 2]), color=color)
                CMS.cmsObjectDraw(g, "PE")
                grid = np.linspace(pts[:, 0].min(), pts[:, 0].max(), 100)
                curve = curve_graph(list(grid),
                                    list(100 * np.polyval(np.array(rec["coeffs"]), grid)),
                                    color=color)
                CMS.cmsObjectDraw(curve, "L")
                leg.AddEntry(g, direction, "pe")
            lat = ROOT.TLatex()
            lat.SetNDC(True)
            lat.SetTextFont(42)
            lat.SetTextSize(0.028)
            lat.DrawLatex(0.20, 0.83, f"{key}  {syst}")
            canv.RedrawAxis()
            _save(canv, outdir, f"deltas.{key}.{syst}.{quantity}".replace("|", "_"))
