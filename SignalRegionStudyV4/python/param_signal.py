"""Parametric (interp-signal) signal templates.

Graduated from the four-arm study's make_param_templates.py
(archive/test/interpolation), updated to the frozen production model:

- shape parameters come from the sliced (mHc, mA) surfaces
  (fits/MHc{X}/polynomials.json) via closInterpYields.predict_shape_params
  — no drop/pin rule, pure-DCB lowM falls out naturally;
- per-era yields via fitInterpYieldModel.predict_yield (raw-fraction
  S·p decomposition, k_era surface — no polys argument);
- systematic variations are parameter shifts (dm, dsig, dN) evaluated
  from the per-mHc delta model polynomials
  (fits/MHc{X}/shape_deltas/delta_model.json) AT the target mA — one
  model file serves every grid point;
- interpolation nuisances sized from configs/interpolation_uncertainties.json
  (scale/res per (study channel, run period); norm per era, mA-binned),
  named period-level via interpolation_config.interp_nuisance_names.

A parametric template carries zero bin errors by design: the
interpolation uncertainty is an explicit nuisance, not MC statistics.
"""
from array import array
from collections import OrderedDict

import numpy as np
import ROOT

import interpolation_config
import srspaths
from dcb_fit_utils import build_model
from template_utils import (cap_stat_errors, create_scaled_hist,
                            ensure_positive_integral, iter_shape_directions)


class ParametricSignal:
    """Interpolated signal model of one (category, sub-era).

    Every template it produces is normalized through the SAME reference
    window (the point's own interpolated window, the one the yield model
    is defined in), so a template built on a different binning window —
    e.g. the group seed's — automatically carries the pdf's containment
    ratio instead of the full predicted yield.
    """

    def __init__(self, tag, params, ref_window):
        self.tag = tag.replace("-", "_").replace("|", "_")
        self.params = params
        self.ref_window = (float(ref_window[0]), float(ref_window[1]))
        self._n_pdf = 0

    def _integrals(self, params, edges, window):
        """(reference-window integral, per-bin integrals) of one pdf."""
        lo = min(float(edges[0]), self.ref_window[0], float(window[0]))
        hi = max(float(edges[-1]), self.ref_window[1], float(window[1]))
        self._n_pdf += 1
        prefix = f"{self.tag}_{self._n_pdf}"
        mass = ROOT.RooRealVar(f"m_{prefix}", "mass", lo, hi)
        pdf, _keep = build_model(prefix, mass, params)
        obs = ROOT.RooArgSet(mass)

        def integral(a, b, name):
            if b <= a:
                return 0.0
            mass.setRange(name, float(a), float(b))
            return float(pdf.createIntegral(
                obs, ROOT.RooFit.NormSet(obs),
                ROOT.RooFit.Range(name)).getVal())

        ref = integral(self.ref_window[0], self.ref_window[1],
                       f"{prefix}_ref")
        bins = [integral(max(float(edges[i]), float(window[0])),
                         min(float(edges[i + 1]), float(window[1])),
                         f"{prefix}_b{i}")
                for i in range(len(edges) - 1)]
        return ref, bins

    def histogram(self, name, edges, window, n_ref, params=None):
        """TH1 of the model on `edges`, clipped to `window`, normalized so
        that its integral over the reference window equals `n_ref`.

        Bin errors are exactly zero: a parametric template carries no MC
        statistical uncertainty (the interpolation uncertainty is a
        separate, explicit nuisance).
        """
        ref, bins = self._integrals(params or self.params, edges, window)
        if ref <= 0:
            raise RuntimeError(f"Non-positive pdf normalization for {name}")
        hist = ROOT.TH1D(name, "", len(edges) - 1,
                         array("d", [float(e) for e in edges]))
        hist.SetDirectory(0)
        for i, value in enumerate(bins):
            hist.SetBinContent(i + 1, n_ref * value / ref)
            hist.SetBinError(i + 1, 0.0)
        return hist


def shifted_params(params, dm=0.0, dsig=0.0, dx0_abs=0.0):
    """Shape parameters after a systematic (or interpolation) shift."""
    out = dict(params)
    out["x0"] = params["x0"] * (1.0 + dm) + dx0_abs
    for side in ("sigmaL", "sigmaR"):
        out[side] = max(params[side] * (1.0 + dsig),
                        interpolation_config.PARAM_FLOORS[side])
    return out


# ------------------------------------------------------- systematic deltas

def load_delta_model(mhc):
    """fits/MHc{X}/shape_deltas/delta_model.json -> payload["model"].

    Every (era|study_channel) key holds per-syst polynomial records in mA
    (fitInterpShapeDeltas), usable at any grid point of the mHc."""
    import json
    import os
    path = os.path.join(srspaths.interpolation_fits_dir(mhc),
                        "shape_deltas", "delta_model.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found — run the interpolation chain (deltas stage) "
            "first")
    with open(path) as f:
        return json.load(f)["model"]


def delta_key(era, study_channel):
    return f"{era}|{study_channel}"


def delta_record(delta_model, key, bucket, syst, direction, mA, warnings):
    """(dm, dsig, dN) of one systematic series evaluated at mA; zeros when
    the series was not measured (variation = pure copy of the nominal)."""
    from fitInterpShapeDeltas import eval_series
    entry = delta_model.get(key)
    if entry is None:
        raise RuntimeError(f"No delta model for {key}")
    node = entry[bucket].get(syst, {}).get(direction)
    if node is None:
        warnings.append(f"[{key}] {syst}/{direction}: no delta model; "
                        "variation built as a pure copy of the nominal")
        return 0.0, 0.0, 0.0
    values = []
    for quantity in interpolation_config.DELTA_QUANTITIES:
        rec = node.get(quantity)
        if rec is None:
            warnings.append(f"[{key}] {syst}/{direction}/{quantity}: "
                            "missing; treated as 0")
            values.append(0.0)
        else:
            value, _band = eval_series(rec, mA)
            values.append(float(value))
    return tuple(values)


def build_signal_component(model, component, dkey, delta_model,
                           syst_categories, edges, window, n_ref, mA,
                           interp_terms, warnings):
    """nominal + every variation histogram of one signal component."""
    proc_map = OrderedDict()
    central = model.histogram(component, edges, window, n_ref)
    ensure_positive_integral(central)
    cap_stat_errors(central)
    proc_map["nominal"] = central

    for syst, variations, group in syst_categories["preprocessed_shape"]:
        if "signal" not in group:
            continue
        for direction in iter_shape_directions(variations):
            dm, dsig, dn = delta_record(delta_model, dkey, "systs",
                                        syst, direction, mA, warnings)
            hist = model.histogram(f"{component}_{syst}{direction}", edges,
                                   window, n_ref * (1.0 + dn),
                                   shifted_params(model.params, dm, dsig))
            ensure_positive_integral(hist)
            cap_stat_errors(hist)
            proc_map[f"{syst}{direction}"] = hist

    for syst, value, group in syst_categories["valued_shape"]:
        if "signal" not in group:
            continue
        for direction in ("up", "down"):
            hist = create_scaled_hist(central, component, syst, value,
                                      direction)
            ensure_positive_integral(hist)
            cap_stat_errors(hist)
            proc_map[f"{syst}{'Up' if direction == 'up' else 'Down'}"] = hist

    for syst, variations, group in syst_categories["multi_variation"]:
        if "signal" not in group:
            continue
        members = []
        for member in variations:
            dm, dsig, dn = delta_record(delta_model, dkey, "pdf_members",
                                        syst, member, mA, warnings)
            members.append(model.histogram(
                f"{component}_{syst}_{member}", edges, window,
                n_ref * (1.0 + dn), shifted_params(model.params, dm, dsig)))
        if not members:
            warnings.append(f"[{dkey}] {syst}: no members; skipped")
            continue
        values = np.array([[h.GetBinContent(i) for i in
                            range(1, h.GetNbinsX() + 1)] for h in members])
        for direction, contents in (("Up", values.max(axis=0)),
                                    ("Down", values.min(axis=0))):
            hist = central.Clone(f"{component}_{syst}{direction}")
            hist.SetDirectory(0)
            errors = values.std(axis=0)
            for i, content in enumerate(contents):
                hist.SetBinContent(i + 1, float(content))
                hist.SetBinError(i + 1, float(errors[i]))
            ensure_positive_integral(hist)
            cap_stat_errors(hist)
            proc_map[f"{syst}{direction}"] = hist

    for name, kwargs_up, kwargs_down in interp_terms:
        for direction, kwargs in (("Up", kwargs_up), ("Down", kwargs_down)):
            hist = model.histogram(f"{component}_{name}{direction}", edges,
                                   window, n_ref,
                                   shifted_params(model.params, **kwargs))
            ensure_positive_integral(hist)
            cap_stat_errors(hist)
            proc_map[f"{name}{direction}"] = hist

    return proc_map


# --------------------------------------------- interpolation nuisances

def load_interp_uncertainties():
    """configs/interpolation_uncertainties.json (derived, ceil3-rounded)."""
    import json
    with open(srspaths.interpolation_uncertainties_path()) as f:
        return json.load(f)


def interp_shape_terms(uncertainties, study_channel, prod_channel, period,
                       params):
    """(name, up-kwargs, down-kwargs) of the scale/res shape nuisances.

    Sizes are keyed by STUDY channel (SR3Mu_lowM/highM — one datacard
    holds one mass point, so exactly one pairing applies); the names use
    the production channel and are period-level."""
    names = interpolation_config.interp_nuisance_names(prod_channel, period)
    scale_val = uncertainties["scale"][study_channel][period]
    res_val = uncertainties["res"][study_channel][period]
    sigma_eff = float(np.sqrt(0.5 * (params["sigmaL"] ** 2
                                     + params["sigmaR"] ** 2)))
    shift = scale_val * sigma_eff
    return [
        (names["scale"], {"dx0_abs": shift}, {"dx0_abs": -shift}),
        (names["res"], {"dsig": res_val}, {"dsig": -res_val}),
    ]


def interp_systematics_block(uncertainties, study_channel, prod_channel,
                             period, era, mA):
    """Datacard entries for one (sub-era, channel)'s interp nuisances.

    The norm lnN value is this ERA's, selected by the target mA's bin;
    the period-level nuisance name is shared across the period's eras, so
    one nuisance row carries per-era values in its columns."""
    names = interpolation_config.interp_nuisance_names(prod_channel, period)
    ma_bin = interpolation_config.norm_ma_bin(mA)
    norm_by_bin = uncertainties["norm"][study_channel][era]
    if ma_bin not in norm_by_bin:
        raise KeyError(
            f"norm bin {ma_bin!r} missing for {study_channel}/{era} "
            "(unreachable bin?) — refusing to guess")
    return {
        names["scale"]: {"source": "parametric", "type": "shape",
                         "group": ["signal"]},
        names["res"]: {"source": "parametric", "type": "shape",
                       "group": ["signal"]},
        names["norm"]: {"source": "valued", "type": "lnN",
                        "group": ["signal"],
                        "value": float(norm_by_bin[ma_bin])},
    }
