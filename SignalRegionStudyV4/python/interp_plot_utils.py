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

import srspaths

from plotter import LumiInfo, PALETTE_LONG, get_CoM_energy

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
                        pull_range=(-5, 5), pull_title="pull"):
    """Two-pad cmsstyle canvas: value (+band) on top, pulls below."""
    CMS.SetExtraText(extra_text)
    _set_lumi_energy(period_or_era)
    canv = CMS.cmsDiCanvas(name, xmin, xmax, ymin, ymax,
                           pull_range[0], pull_range[1],
                           xtitle, ytitle, pull_title,
                           square=True, iPos=11, extraSpace=0.02)
    if logy:
        canv.cd(1).SetLogy()
    return canv


def curve_graph(xs, ys, color=None):
    g = ROOT.TGraph(len(xs), array('d', xs), array('d', ys))
    g.SetLineWidth(2)
    g.SetLineColor(color if color is not None else _CURVE_COLOR)
    return g


def band_graph(xs, ys, err_lo, err_hi=None, color=None, alpha=0.25):
    """Symmetric (err_hi=None -> err_lo used both sides) or asymmetric
    1-sigma band as a filled TGraphAsymmErrors."""
    n = len(xs)
    g = ROOT.TGraphAsymmErrors(n)
    for i in range(n):
        g.SetPoint(i, xs[i], ys[i])
        hi = err_hi[i] if err_hi is not None else err_lo[i]
        g.SetPointError(i, 0.0, 0.0, err_lo[i], hi)
    g.SetFillColorAlpha(color if color is not None else _BAND_COLOR, alpha)
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


def _save_both(canv, outdir, name):
    """PNG + PDF, the results/plots convention (plotBreakdown.py)."""
    os.makedirs(outdir, exist_ok=True)
    base = os.path.join(outdir, name)
    canv.SaveAs(f"{base}.png")
    canv.SaveAs(f"{base}.pdf")
    canv.Close()


# --------------------------------------------------------- shape parameters

_PARAM_TITLES = {
    "x0": "x_{0} [GeV]", "sigmaL": "#sigma_{L} [GeV]",
    "sigmaR": "#sigma_{R} [GeV]", "alphaL": "#alpha_{L}",
    "alphaR": "#alpha_{R}", "fsig": "f_{sig}",
    "c1": "c_{1}", "c2": "c_{2}",
}

# Study channels rendered as final states. The SR3Mu pairing variant is kept:
# it is the one piece of the category key that neither the axes nor the
# luminosity header carries.
_CHANNEL_LABELS = {
    "SR1E2Mu": "e#mu#mu",
    "SR3Mu": "#mu#mu#mu",
    "SR3Mu_lowM": "#mu#mu#mu (lowM)",
    "SR3Mu_highM": "#mu#mu#mu (highM)",
}


# In-plot information text, in the style of the V3 paper figures
# (plotPaper*.py -> ComparisonCanvas._draw_channel_text): a left-aligned block
# under the CMS logo, region tag in bold on the first line and the final state
# in the regular font one text height below. The y start assumes iPos=11, i.e.
# "CMS"/"Simulation Preliminary" inside the frame.
INFO_TEXT_POS = (0.22, 0.72)
INFO_TEXT_SIZE = 0.05
REGION_TAG = "SR"


def draw_info_text(lines, posX=INFO_TEXT_POS[0], posY=INFO_TEXT_POS[1],
                   size=INFO_TEXT_SIZE, step=None):
    """Paper-style information block: first line bold (font 62), the rest in
    the regular font (42), stacked downwards one text height apart.

    `step` overrides the line pitch; lines carrying both a subscript and a
    superscript need more than one text height to clear each other."""
    pitch = size if step is None else step
    for i, line in enumerate(lines):
        CMS.drawText(line, posX=posX, posY=posY - i * pitch,
                     font=62 if i == 0 else 42, align=0, size=size)


def param_title(param):
    """Axis title for a shape parameter."""
    return _PARAM_TITLES.get(param, param)


def channel_label(cat_key):
    """Final-state label for a study channel or a `{channel}_{period}`
    category key; the run period is dropped (the lumi header states it)."""
    name = cat_key
    for period in ("_Run2", "_Run3"):
        if name.endswith(period):
            name = name[:-len(period)]
            break
    return _CHANNEL_LABELS.get(name, name)


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

def plot_yield_period_model(mhc, period, model, merged, fit_ma,
                            outdir, eval_rec, rec_band, inv_logit):
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
                     f"G {tot_channel}: slice of the joint (m_{{H^{{#pm}}}}, m_{{A}}) "
                     f"surface, #chi^{{2}}={rec['chi2']:.1f} over "
                     f"{rec['ndf']} points here")
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

    canv = graph_canvas(f"foverlay_{period}", "m_{A} [GeV]", "f = S #upoint p",
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
        curve = curve_graph(list(grid), list(S * p), color=color)
        CMS.cmsObjectDraw(curve, "L")
        leg.AddEntry(g, ch, "pe")
    canv.RedrawAxis()
    _save(canv, outdir, f"model.{period}.f_SR3Mu_overlay")


def plot_yield_era_grid(mhc, channel, yields, model, fit_ma, outdir,
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
            curve = np.array([predict_yield(model, channel, era, m)
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
                        model=None, predict_yield=None):
    """One PNG per (channel, era), two pads. Top: measured window yields
    (filled black), the adopted full-grid fit curve with its 1-sigma band
    (when the model is passed) and the leave-one-out predictions at every
    grid point (open blue; grid-endpoint extrapolations open grey).
    Bottom: the LOO relative residual (pred - meas)/meas in %. The visual
    counterpart of the loo_uncertainties.json norm table."""
    import run_period_utils

    grey = ROOT.TColor.GetColor("#9c9ca1")
    for period, suberas in run_period_utils.RUN_PERIODS.items():
        for era in suberas:
            meas = {}
            for rec in yields.values():
                r = rec["channels"].get(channel, {}).get(era)
                if r is not None:
                    meas[rec["mA"]] = (r["sumw"], r["err"])
            pred, pred_ex = [], []
            for mA, entry in sorted(loo.items()):
                rec = entry["scalar"].get(channel, {}).get(era)
                if rec is None:
                    continue
                tgt = pred_ex if rec.get("extrapolation") else pred
                tgt.append((mA, rec["n_pred"], rec["err_pred"], rec["rel"],
                            rec["n_meas"]))
            if not meas:
                continue
            all_m = sorted(meas)
            curve = None
            if predict_yield is not None:
                xgrid = np.linspace(min(all_m), max(all_m), 150)
                curve = np.array([predict_yield(model, channel, era, m)
                                  for m in xgrid])
            # Ranges from the measured points, the usable predictions and
            # the fit curve only: an endpoint extrapolation can be off by
            # orders of magnitude and would flatten everything else.
            all_v = ([v for v, _e in meas.values()]
                     + [v for _m, v, _e, _r, _n in pred]
                     + ([] if curve is None else list(curve[:, 0])))
            ymin = 0.5 * max(min(all_v), 1e-3)
            ymax = 2.0 * max(all_v)
            yr = 50.0   # fixed ratio range; outliers leave the frame
            xlo, xhi = min(all_m) - 3, max(all_m) + 3

            canv = dicanvas_with_pulls(
                f"loo_{channel}_{era}", "m_{A} [GeV]", "N_{window}",
                xlo, xhi, ymin, ymax, era, logy=True,
                pull_range=(-yr, yr), pull_title="#DeltaN/N [%]")

            canv.cd(1)
            keep = []   # hold drawn graphs until SaveAs
            leg = CMS.cmsLeg(0.55, 0.68, 0.90, 0.88, textSize=0.030)
            g_curve = None
            if curve is not None:
                g_band = band_graph(list(xgrid), list(curve[:, 0]),
                                    list(curve[:, 1]), alpha=0.35)
                CMS.cmsObjectDraw(g_band, "E3")
                keep.append(g_band)
                g_curve = curve_graph(list(xgrid), list(curve[:, 0]))
                CMS.cmsObjectDraw(g_curve, "L")
            m, v, e = zip(*((mA, *meas[mA]) for mA in all_m))
            g_meas = points_graph(m, v, e)
            CMS.cmsObjectDraw(g_meas, "PE")
            leg.AddEntry(g_meas, "measured MC", "pe")
            if g_curve is not None:
                leg.AddEntry(g_curve, "full-grid model", "l")
            if pred:
                m, v, e = zip(*((p[0], p[1], p[2]) for p in pred))
                g_pred = points_graph(m, v, e, filled=False,
                                      color=_HELD_OUT_COLOR)
                CMS.cmsObjectDraw(g_pred, "PE")
                leg.AddEntry(g_pred, "LOO prediction", "pe")
            if pred_ex:
                m, v, e = zip(*((p[0], p[1], p[2]) for p in pred_ex))
                g_ex = points_graph(m, v, e, filled=False, color=grey)
                CMS.cmsObjectDraw(g_ex, "PE")
                leg.AddEntry(g_ex, "LOO (extrapolation)", "pe")
            lat = ROOT.TLatex()
            lat.SetNDC(True)
            lat.SetTextFont(42)
            lat.SetTextSize(0.036)
            lat.DrawLatex(0.20, 0.70,
                          f"{channel}, m_{{H^{{#pm}}}} = {mhc} GeV")
            canv.cd(1).RedrawAxis()

            canv.cd(2)
            if curve is not None:
                # Model 1-sigma band in relative terms around zero, so the
                # residuals can be read directly against it.
                g_rband = band_graph(
                    list(xgrid), [0.0] * len(xgrid),
                    list(100.0 * curve[:, 1] / np.maximum(curve[:, 0], 1e-9)),
                    alpha=0.35)
                CMS.cmsObjectDraw(g_rband, "E3")
                keep.append(g_rband)
            ref = ROOT.TLine()
            ref.SetLineStyle(ROOT.kDotted)
            ref.DrawLine(xlo, 0.0, xhi, 0.0)
            for group, color in ((pred, _HELD_OUT_COLOR), (pred_ex, grey)):
                if not group:
                    continue
                m = [p[0] for p in group]
                r = [100.0 * p[3] for p in group]
                e = [100.0 * p[2] / p[4] for p in group]
                g_res = points_graph(m, r, e, filled=False, color=color)
                CMS.cmsObjectDraw(g_res, "PE")
                keep.append(g_res)
            canv.cd(2).RedrawAxis()

            _save(canv, outdir, f"loo_grid.{channel}.{era}")


def plot_yield_template_closure(cat_key, mp, hist, pred, n_pred, err_pred,
                                chi2, ndf, period, outdir):
    """Absolute-normalization closure: MC hist vs predicted template, with
    a Pred/MC ratio panel."""
    nbins = hist.GetNbinsX()
    lo, hi = hist.GetXaxis().GetXmin(), hist.GetXaxis().GetXmax()
    mc = np.array([hist.GetBinContent(i) for i in range(1, nbins + 1)])
    # Header state FIRST: cmsstyle draws CMS_lumi inside cmsDiCanvas, so
    # setting these afterwards labels the plot with whatever the previous
    # call left behind (and the very first plot of a process with
    # cmsstyle's defaults).
    CMS.SetExtraText("Simulation Preliminary")
    _set_lumi_energy(period)
    canv = CMS.cmsDiCanvas(
        f"tmpl_{cat_key}_{mp}", lo, hi, 0.0, 1.3 * max(mc.max(), pred.max()),
        0.5, 1.5, "m(#mu#mu) [GeV]", "Events / bin", "Pred / MC",
        square=True, iPos=11, extraSpace=0.02)
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
    # Below the CMS block, not on top of it: at NDC y = 0.83/0.79 this text
    # ran straight through the "CMS" / "Simulation Preliminary" logo drawn
    # inside the frame by iPos=11.
    draw_info_text([f"{cat_key}  {mp}",
                    f"#chi^{{2}}/ndf={chi2:.1f}/{ndf}"],
                   posX=0.20, posY=0.68, size=0.032, step=0.042)
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


def _band_from_hist(hist, name, scale=None):
    """Filled band graph from a TH1's bin contents +- bin errors.

    ``scale`` (a TH1 with the same binning) divides both the central value
    and the error, turning the band into the ratio-panel version; bins where
    it is non-positive are dropped rather than drawn at an arbitrary value.
    """
    graph = ROOT.TGraphAsymmErrors()
    n = 0
    for i in range(1, hist.GetNbinsX() + 1):
        axis = hist.GetXaxis()
        centre = axis.GetBinCenter(i)
        half = 0.5 * axis.GetBinWidth(i)
        value, error = hist.GetBinContent(i), hist.GetBinError(i)
        if scale is not None:
            denom = scale.GetBinContent(i)
            if denom <= 0:
                continue
            value, error = value / denom, error / denom
        graph.SetPoint(n, centre, value)
        graph.SetPointError(n, half, half, error, error)
        n += 1
    graph.SetName(name)
    return graph


def plot_template_closure(cat_key, mp, h_mc, h_interp, summary, period,
                          outdir, ratio_range=(0.5, 1.5)):
    """Signal MC vs the interpolated template, on the PRODUCTION binning.

    ``h_mc`` carries MC statistical errors, ``h_interp`` carries the
    assigned interpolation uncertainty in its bin errors (the quadrature of
    every CMS_interp_* nuisance -- see plotTemplateClosure.py).  The two are
    drawn as separate bands so a discrepancy can be attributed to the model
    or to MC noise.

    The luminosity header and the extra text are set BEFORE the canvas is
    built -- cmsstyle draws the header inside cmsDiCanvas, so setting them
    afterwards labels the plot with whatever the previous call left behind
    (the bug plot_yield_template_closure carried until 2026-08-19).
    """
    lo = h_mc.GetXaxis().GetXmin()
    hi = h_mc.GetXaxis().GetXmax()
    y_max = 1.55 * max(h_mc.GetMaximum(), h_interp.GetMaximum())

    CMS.SetExtraText("Simulation Preliminary")
    _set_lumi_energy(period)
    canv = CMS.cmsDiCanvas(
        f"clos_{cat_key}_{mp}".replace("-", "_"), lo, hi, 0.0, y_max,
        ratio_range[0], ratio_range[1],
        "m(#mu#mu) [GeV]", "Events / bin", "interp. / MC",
        square=True, iPos=11, extraSpace=0.02)

    canv.cd(1)
    band = _band_from_hist(h_interp, f"band_{cat_key}_{mp}")
    band.SetFillColorAlpha(_CURVE_COLOR, 0.30)
    band.SetLineWidth(0)
    band.Draw("2 same")
    CMS.cmsObjectDraw(h_interp, "hist", LineColor=_CURVE_COLOR, LineWidth=2)
    CMS.cmsObjectDraw(h_mc, "PE", MarkerStyle=ROOT.kFullCircle,
                      MarkerSize=0.8, LineColor=ROOT.kBlack,
                      MarkerColor=ROOT.kBlack)

    leg = CMS.cmsLeg(0.53, 0.68, 0.92, 0.88, textSize=0.030)
    leg.AddEntry(h_mc, f"signal MC (N={summary['n_mc']:.1f})", "pe")
    leg.AddEntry(h_interp,
                 f"interpolated (N={summary['n_interp']:.1f})", "l")
    leg.AddEntry(band, "interp. uncertainty", "f")
    leg.Draw("same")

    mhc, mA = srspaths.masspoint_mhc_ma(mp)
    chi2_line = "#chi^{2}/ndf = "
    if summary["ndf"]:
        chi2_line += (f"{summary['chi2_stat'] / summary['ndf']:.2f}"
                      f" (MC stat), "
                      f"{summary['chi2_total'] / summary['ndf']:.2f}"
                      f" (+ unc.)")
    else:
        chi2_line += "n/a"
    draw_info_text([f"{REGION_TAG}, {channel_label(cat_key)}",
                    f"m_{{H^{{#pm}}}} = {mhc:g}, m_{{A}} = {mA:g} GeV",
                    f"N_{{interp}}/N_{{MC}} = {summary['norm_ratio']:.3f}",
                    chi2_line],
                   posX=0.20, posY=0.66, size=0.034, step=0.044)
    canv.cd(1).RedrawAxis()

    canv.cd(2)
    # Grey band at 1: the MC statistical error the model is measured
    # against. Red band: the assigned interpolation uncertainty, drawn
    # around the ratio itself, so the test is whether it reaches the grey.
    unity = h_mc.Clone(f"h_unity_{cat_key}_{mp}")
    unity.SetDirectory(0)
    mc_band = _band_from_hist(h_mc, f"mcband_{cat_key}_{mp}", scale=unity)
    mc_band.SetFillColorAlpha(ROOT.kGray + 1, 0.45)
    mc_band.SetLineWidth(0)
    mc_band.Draw("2 same")
    ratio_band = _band_from_hist(h_interp, f"rband_{cat_key}_{mp}",
                                 scale=h_mc)
    ratio_band.SetFillColorAlpha(_CURVE_COLOR, 0.30)
    ratio_band.SetLineWidth(0)
    ratio_band.Draw("2 same")

    ratio = h_interp.Clone(f"h_ratio_{cat_key}_{mp}")
    ratio.SetDirectory(0)
    for i in range(1, ratio.GetNbinsX() + 1):
        denom = h_mc.GetBinContent(i)
        ratio.SetBinContent(i, h_interp.GetBinContent(i) / denom
                            if denom > 0 else 0.0)
        ratio.SetBinError(i, 0.0)
    ref = ROOT.TLine()
    ref.SetLineStyle(ROOT.kDotted)
    ref.DrawLine(lo, 1.0, hi, 1.0)
    CMS.cmsObjectDraw(ratio, "hist", LineColor=_CURVE_COLOR, LineWidth=2)
    canv.cd(2).RedrawAxis()

    _save_both(canv, outdir, f"closure.{cat_key}")


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


# ------------------------------------------------------ (mHc, mA) surfaces

def _drop_wide_error_points(series, ys_all, max_error_frac, name):
    """Return the series with the points whose error exceeds max_error_frac
    of the plotted value range removed, reporting how many went."""
    if not max_error_frac or max_error_frac <= 0:
        return series
    span = max(ys_all) - min(ys_all)
    if span <= 0:
        return series
    limit = max_error_frac * span
    filtered, n_dropped = {}, 0
    for key, block in series.items():
        px, py, pe = block["points"]
        keep = [i for i, e in enumerate(pe) if abs(e) <= limit]
        n_dropped += len(px) - len(keep)
        filtered[key] = dict(block, points=([px[i] for i in keep],
                                            [py[i] for i in keep],
                                            [pe[i] for i in keep]))
    if n_dropped:
        print(f"  {name}: hid {n_dropped} point(s) with an error bar above "
              f"{max_error_frac:g} x the plotted range")
    return filtered


def mhc_legend_label(mhc):
    return f"m_{{H^{{#pm}}}} = {mhc} GeV"


def plot_surface_slices(title, xtitle, ytitle, series, outdir, name,
                        period_or_era, logy=False, headroom_factor=None,
                        max_error_frac=None, legend_label=mhc_legend_label,
                        key_order=None, yrange=None):
    """One curve per series key plus that series' measured points.

    The readable way to show a (mHc, mA) surface to a physicist: the seven
    mHc slices over a common mA axis, each with its own points in the
    matching colour, so the borrowing across studies is visible directly.
    The same layout carries the era shares of one study, where the series
    are the eras of a run period at fixed mHc.

    series: {key: {"curve": (xs, ys), "points": (xs, ys, yerrs)}}
    title: one information-text line, or a sequence of them (first bold).
    headroom_factor: put the axis maximum at this multiple of the largest
    entry, leaving the top of the frame free for the CMS block, the legend
    and the information text. Multiplicative on both scales; the log axis
    uses 2 when unset.
    max_error_frac: drop points whose error bar is longer than this fraction
    of the plotted value range. Display only — those points were still used
    by the fit; the count dropped is reported on stdout.
    legend_label: key -> legend text.
    key_order: draw order; sorted(series) when unset.
    yrange: explicit (ymin, ymax), overriding the automatic range — for
    quantities with a natural fixed scale.
    """
    xs_all, ys_all = [], []
    for block in series.values():
        xs_all += list(block["curve"][0])
        ys_all += list(block["curve"][1])
        ys_all += list(block["points"][1])
    if not ys_all:
        return

    # An error bar spanning a sizeable fraction of the frame says nothing
    # about the surface and squashes every other point; the fit already
    # down-weighted it. Filter against the full range, then take the axis
    # range from what survives.
    series = _drop_wide_error_points(series, ys_all, max_error_frac, name)
    ys_all = []
    for block in series.values():
        ys_all += list(block["curve"][1]) + list(block["points"][1])
    if not ys_all:
        return

    lo, hi = min(ys_all), max(ys_all)
    pad = 0.10 * (hi - lo) if hi > lo else max(abs(hi), 1.0) * 0.1
    if yrange is not None:
        ymin, ymax = yrange
    elif logy:
        ymin = max(lo * 0.5, 1e-6)
        ymax = hi * (headroom_factor if headroom_factor is not None else 2.0)
    elif headroom_factor is not None and hi > 0:
        ymin, ymax = lo - pad, hi * headroom_factor
    else:
        ymin, ymax = lo - pad, hi + pad * 2
    canv = graph_canvas(name, xtitle, ytitle, min(xs_all) - 2,
                        max(xs_all) + 2, ymin, ymax, period_or_era, logy=logy)
    leg = CMS.cmsLeg(0.62, 0.60, 0.92, 0.90, textSize=0.028)
    keep = []
    for i, key in enumerate(key_order or sorted(series)):
        color = PALETTE_LONG[i % len(PALETTE_LONG)]
        cx, cy = series[key]["curve"]
        g = curve_graph(list(cx), list(cy), color=color)
        CMS.cmsObjectDraw(g, "L")
        px, py, pe = series[key]["points"]
        keep.append(g)
        if len(px):
            gp = points_graph(list(px), list(py), list(pe), color=color)
            CMS.cmsObjectDraw(gp, "PE")
            keep.append(gp)
            leg.AddEntry(gp, legend_label(key), "pe")
        else:
            leg.AddEntry(g, legend_label(key), "l")
    draw_info_text([title] if isinstance(title, str) else list(title))
    canv.RedrawAxis()
    _save(canv, outdir, name)


def plot_nuisance_cell(channel, era, regions, outdir):
    """The rms-then-max rule made visible for one (channel, era).

    Per mA region: each study's rms as a point against mHc, the adopted
    value as a solid line and the pooled rms dashed, so it is obvious which
    study set the nuisance and how far the others sit below it.
    """
    have = [r for r in regions if regions[r].get("per_study_rms")]
    if not have:
        return
    all_v = [v for r in have for v in regions[r]["per_study_rms"].values()]
    all_v += [regions[r]["value"] for r in have]
    canv = graph_canvas(f"nuis_{channel}_{era}", "m_{H^{#pm}} [GeV]",
                        "relative yield deviation", 60, 170, 0.0,
                        1.35 * max(all_v), era)
    leg = CMS.cmsLeg(0.16, 0.66, 0.55, 0.90, textSize=0.028)
    keep = []
    for i, region in enumerate(have):
        color = PALETTE_LONG[i % len(PALETTE_LONG)]
        block = regions[region]
        mh = sorted(int(k.replace("MHc", ""))
                    for k in block["per_study_rms"])
        ys = [block["per_study_rms"][f"MHc{m}"] for m in mh]
        gp = points_graph(mh, ys, [0.0] * len(mh), color=color)
        CMS.cmsObjectDraw(gp, "P")
        adopted = curve_graph([60, 170], [block["value"]] * 2, color=color)
        CMS.cmsObjectDraw(adopted, "L")
        pooled = curve_graph([60, 170], [block["pooled_rms"]] * 2, color=color)
        pooled.SetLineStyle(2)
        CMS.cmsObjectDraw(pooled, "L")
        keep += [gp, adopted, pooled]
        leg.AddEntry(gp, f"{region}: {100 * block['value']:.1f}% "
                         f"(driver {block['driver']})", "p")
    lat = ROOT.TLatex()
    lat.SetNDC(True)
    lat.SetTextFont(42)
    lat.SetTextSize(0.028)
    lat.DrawLatex(0.16, 0.94, f"{channel}  {era}   "
                              "solid = adopted (max over studies), "
                              "dashed = pooled rms")
    canv.RedrawAxis()
    _save(canv, outdir, f"norm.{channel}.{era}")


def mhc_color(mhc):
    """Colour of an mHc study, keyed by its slot in the study grid.

    Keying on the grid rather than on whatever studies happen to populate
    a panel keeps one study the same colour in every panel and in both
    interpolation arms, so panels can be read against each other."""
    import interpolation_config
    grid = interpolation_config.mhc_grid()
    mhc = int(mhc)
    if mhc not in grid:
        raise ValueError(f"mHc={mhc} is not a study of the grid {grid}")
    return PALETTE_LONG[grid.index(mhc) % len(PALETTE_LONG)]


def plot_residual_vs_mA(name, header, series, bands, outdir, period_or_era,
                        xrange, ytitle, info_lines=(), yscale=100.0):
    """Signed LOO residual vs mA for one uncertainty cell.

    The layer under `plot_nuisance_cell`: that one shows the rms-then-max
    rule over per-study summaries, this one shows the residuals the rule
    was applied to, so a cell's adopted size can be read against the
    scatter it is meant to cover.

      series  {mhc: {"used": [(mA, resid), ...],
                     "unused": [(mA, resid), ...]}}
              used = entered the envelope (filled marker); unused = kept
              out by the production-pairing restriction (open marker).
      bands   {"adopted": [(x_lo, x_hi, value, inherited_bool), ...],
               "pooled": float or None}
              `adopted` is a list of segments so a flat scale/res band and
              the mA-binned norm step function share one code path; a
              segment omitted from the list is drawn as a gap, which is how
              structurally unreachable mA bins say "no nuisance here".

    Residuals and band values are in the same raw units and both are
    multiplied by `yscale` for display (100 -> percent, 1 -> sigma_eff).
    """
    used_all = [v for s in series.values() for _, v in s.get("used", ())]
    unused_all = [v for s in series.values() for _, v in s.get("unused", ())]
    band_vals = [v for _, _, v, _ in bands.get("adopted", ())]
    if not used_all and not unused_all and not band_vals:
        return False
    # Wide enough for the outliers, tall enough that the band is not a
    # sliver on the axis, then doubled so the scatter reads as a cloud
    # inside the frame rather than filling it.
    span = max([abs(v) for v in used_all + unused_all] or [0.0])
    ymax = 2.0 * yscale * max(1.25 * span, 3.0 * max(band_vals or [0.0]),
                              1e-3)

    xmin, xmax = xrange
    canv = graph_canvas(name, "m_{A} [GeV]", ytitle, xmin, xmax,
                        -ymax, ymax, period_or_era)
    keep = []

    # The band is the reference the panel is read against, so the channel
    # and the legend are placed above it and the cell's numbers below it,
    # in NDC derived from where the band actually lands in the frame.
    band_half = yscale * max(band_vals or [0.0])
    _frame_lo, _frame_hi = canv.GetBottomMargin(), 1.0 - canv.GetTopMargin()

    def _ndc(y):
        return _frame_lo + (y + ymax) / (2.0 * ymax) * (_frame_hi - _frame_lo)

    band_top_ndc, band_bot_ndc = _ndc(band_half), _ndc(-band_half)

    # Bands first so the points sit on top of them. Hatched rather than
    # solid-filled: the assigned uncertainty is a band, not a measurement,
    # and a hatch cannot be misread as data.
    for x_lo, x_hi, value, inherited in bands.get("adopted", ()):
        lo, hi = max(x_lo, xmin), min(x_hi, xmax)
        if not hi > lo:
            continue
        g = band_graph([lo, hi], [0.0, 0.0], [yscale * value] * 2)
        # 3ijk hatchings take their line colour from SetFillColor; the
        # predefined 30xx stipples ignore it and render black, and the
        # alpha form band_graph uses does too. Opposite diagonals mark an
        # inherited value apart from a measured one.
        g.SetFillColor(_BAND_COLOR)
        g.SetFillStyle(3454 if inherited else 3345)
        CMS.cmsObjectDraw(g, "E3")
        keep.append(g)

    pooled = bands.get("pooled")
    if pooled is not None:
        for sign in (+1.0, -1.0):
            g = curve_graph([xmin, xmax], [sign * yscale * pooled] * 2,
                            color=ROOT.kGray + 2)
            g.SetLineStyle(ROOT.kDotted)
            CMS.cmsObjectDraw(g, "L")
            keep.append(g)

    zero = ROOT.TLine()
    zero.SetLineStyle(ROOT.kDotted)
    zero.DrawLine(xmin, 0.0, xmax, 0.0)

    # Legend above the band, on the right so it clears the CMS block.
    # Anchored at the top of the frame and grown downwards: one column
    # always fits that way, and one column cannot overlap itself.
    n_entries = max(len(series), 1)
    leg_top = _frame_hi - 0.01
    leg = CMS.cmsLeg(0.66, leg_top - n_entries * 0.036, 0.93, leg_top,
                     textSize=0.026)
    for mhc in sorted(series):
        color = mhc_color(mhc)
        block = series[mhc]
        entry = None
        for kind, filled in (("used", True), ("unused", False)):
            pts = sorted(block.get(kind, ()))
            if not pts:
                continue
            xs, ys = zip(*pts)
            g = points_graph(xs, [yscale * y for y in ys], [0.0] * len(xs),
                             filled=filled, color=color)
            CMS.cmsObjectDraw(g, "P")
            keep.append(g)
            if filled or entry is None:
                entry = g
        if entry is not None:
            leg.AddEntry(entry, mhc_legend_label(mhc), "p")

    # Channel block in the paper style (plotPaper*.py -> ComparisonCanvas
    # ._draw_channel_text): region tag bold, final state one text height
    # below, at the shared INFO_TEXT_POS. Held above the band, which the
    # paper position already clears for every cell here.
    if header:
        draw_info_text([REGION_TAG, header], posX=INFO_TEXT_POS[0],
                       posY=max(INFO_TEXT_POS[1],
                                band_top_ndc + 2.0 * INFO_TEXT_SIZE),
                       size=INFO_TEXT_SIZE)
    # The cell's numbers along the bottom of the panel, last line just
    # above the axis.
    for i, line in enumerate(info_lines):
        CMS.drawText(line, posX=INFO_TEXT_POS[0],
                     posY=_frame_lo + 0.020 + (len(info_lines) - 1 - i) * 0.030,
                     font=42, align=0, size=0.030)
    canv.RedrawAxis()
    _save(canv, outdir, name)
    return True
