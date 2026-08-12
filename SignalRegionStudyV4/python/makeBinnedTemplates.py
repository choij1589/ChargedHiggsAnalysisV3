#!/usr/bin/env python3
"""
Generate binned histogram templates for HiggsCombine.

Usage:
    python makeBinnedTemplates.py --era Run2 --channel Combined --masspoint MHc130_MA90 --method Baseline
"""
import os
import shutil
import logging
import argparse
from collections import OrderedDict
import ROOT
import numpy as np
from math import sqrt

from template_utils import (
    save_json,
    ensure_positive_integral, cap_stat_errors,
    create_scaled_hist, categorize_systematics,
    calculate_adaptive_bins, apply_syst_driven_merging,
    iter_shape_directions
)
from run_period_utils import (
    category_name,
    component_name,
    resolve_channels,
    resolve_run_periods,
)
from plotter import FitCanvasWithRatio
import srspaths

# Signal-independent machinery lives in binned_template_core (shared with
# the parametric-signal producer); names are re-exported here so existing
# importers (measInterpShapeDeltas, wrappers, plot scripts) keep working.
from binned_template_core import (          # noqa: F401  (re-exports)
    SYST_MERGE_THRESHOLD, MIN_CORE_BINS,
    getHist, createEnvelopeHists, getDataHist,
    validateBackgroundStatistics,
    load_systematics_block, load_sample_block,
    category_background_processes, hist_exists,
    build_component_shape_templates, write_run_period_shapes,
    validate_category_backgrounds, scan_binning, sum_data_obs,
    apply_floor_fallback, write_template_sidecars,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate binned histogram templates for HiggsCombine")
    parser.add_argument("--era", required=True, type=str, help="Run-period target: Run2, Run3, or All")
    parser.add_argument("--channel", required=True, type=str, help="Analysis channel (SR1E2Mu, SR3Mu, Combined)")
    parser.add_argument("--masspoint", required=True, type=str, help="Signal mass point (e.g., MHc130_MA90)")
    parser.add_argument("--method", required=True, type=str, help="Template method (Baseline, ParticleNet)")
    parser.add_argument("--blind", action="store_true",
                        help="Asimov data_obs (MC sum) instead of real data; "
                             "writes to the {method}_blind template segment")
    parser.add_argument("--signal-source", default="mc-signal",
                        choices=["mc-signal", "interp-signal"],
                        help="mc-signal: direct-MC signal templates "
                             "(default). interp-signal: parametric signal "
                             "from the mA-interpolation surfaces (Baseline "
                             "only; masspoint must be on configs/grid.json)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    return parser.parse_args()

# =============================================================================
# Histogram Creation Functions
# =============================================================================


# =============================================================================
# Background Validation Functions
# =============================================================================

# =============================================================================
# ParticleNet Optimization Functions
# =============================================================================

# Mapping from config categories to ParticleNet classes
# ParticleNet has 3 background classes: nonprompt, diboson, ttX
PARTICLENET_CLASS_MAPPING = {
    "nonprompt": ["nonprompt"],
    "diboson": ["diboson", "WZ", "ZZ"],
    "ttX": ["ttX", "ttW", "ttZ", "ttH", "tZq"],
}


def loadDataset(basedir, process, masspoint, mass_min, mass_max, bg_weights=None):
    """Load events with ParticleNet scores from preprocessed samples.

    Vectorized (RDataFrame filter + AsNumpy + elementwise numpy). Every
    operation is elementwise in the same order as the original per-entry
    loop, so the returned arrays are bitwise identical — only faster."""
    file_path = f"{basedir}/{process}.root"
    if not os.path.exists(file_path):
        logging.warning(f"Sample file not found for optimization: {file_path}")
        return np.array([]), np.array([]), np.array([])

    rfile = ROOT.TFile.Open(file_path, "READ")
    tree = rfile.Get("Central")

    if not tree:
        logging.warning(f"Central tree not found in {file_path}")
        rfile.Close()
        return np.array([]), np.array([]), np.array([])

    score_sig = f"score_{masspoint}_signal"
    score_nonprompt = f"score_{masspoint}_nonprompt"
    score_diboson = f"score_{masspoint}_diboson"
    score_ttZ = f"score_{masspoint}_ttZ"

    branches = [b.GetName() for b in tree.GetListOfBranches()]
    rfile.Close()
    if score_sig not in branches:
        logging.warning(f"ParticleNet scores not found in {file_path}")
        return np.array([]), np.array([]), np.array([])

    rdf = (ROOT.RDataFrame("Central", file_path)
           .Filter(f"mass >= {mass_min!r} && mass <= {mass_max!r}"))
    cols = rdf.AsNumpy([score_sig, score_nonprompt, score_diboson,
                        score_ttZ, "weight"])
    s0 = np.asarray(cols[score_sig], dtype=float)
    s1 = np.asarray(cols[score_nonprompt], dtype=float)
    s2 = np.asarray(cols[score_diboson], dtype=float)
    s3 = np.asarray(cols[score_ttZ], dtype=float)
    weights = np.asarray(cols["weight"], dtype=float)

    if bg_weights:
        w1 = bg_weights.get("nonprompt", 1.0)
        w2 = bg_weights.get("diboson", 1.0)
        w3 = bg_weights.get("ttX", 1.0)
        score_denom = s0 + w1 * s1 + w2 * s2 + w3 * s3
    else:
        score_denom = s0 + s1 + s2 + s3

    scores = np.zeros_like(s0)
    np.divide(s0, score_denom, out=scores, where=score_denom > 0)

    labels = np.full(len(scores), 1 if process == masspoint else 0)

    return scores, weights, labels


def evalSensitivity(y_true, y_pred, weights, threshold=0.):
    """Calculate significance Z using Asimov formula."""
    signal_mask = (y_true == 1) & (y_pred > threshold)
    background_mask = (y_true == 0) & (y_pred > threshold)

    S = np.sum(weights[signal_mask])
    B = np.sum(weights[background_mask])

    if B <= 0:
        return 0.0

    return np.sqrt(2 * ((S + B) * np.log(1 + S / B) - S))


def getOptimizedThreshold(scores_sig, weights_sig, scores_bkg, weights_bkg):
    """Find optimal ParticleNet score threshold to maximize sensitivity."""
    y_pred = np.concatenate([scores_sig, scores_bkg])
    y_true = np.concatenate([np.ones(len(scores_sig)), np.zeros(len(scores_bkg))])
    weights = np.concatenate([weights_sig, weights_bkg])

    thresholds = np.linspace(0, 1, 101)
    sensitivities = [evalSensitivity(y_true, y_pred, weights, threshold) for threshold in thresholds]

    best_idx = np.argmax(sensitivities)
    best_threshold = thresholds[best_idx]
    initial_sensitivity = sensitivities[0]
    max_sensitivity = sensitivities[best_idx]

    logging.info("Threshold optimization:")
    logging.info(f"  Best threshold: {best_threshold:.3f}")
    logging.info(f"  Initial sensitivity (no cut): {initial_sensitivity:.3f}")
    logging.info(f"  Max sensitivity: {max_sensitivity:.3f}")
    if initial_sensitivity > 0:
        logging.info(f"  Improvement: {(max_sensitivity / initial_sensitivity - 1) * 100:.2f}%")

    return best_threshold, initial_sensitivity, max_sensitivity


# =============================================================================
# Run-period component template construction
# =============================================================================

def fit_dcb(chain, mA_nominal):
    """Two-stage Double Crystal Ball fit on a ``TChain('Central')`` holding
    ``mass`` and ``weight`` branches.

    Returns the fitted parameters plus the narrow fit window (``fit_lo``,
    ``fit_hi``). This is the pure-fit core shared by ``getFitResultDCBMulti``
    (which adds diagnostic plotting) and the standalone signal-efficiency tool;
    it has no plotting side effects.
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
    pre_dcb.fitTo(ds_wide, ROOT.RooFit.SumW2Error(True),
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
    dcb.fitTo(ds_narrow, ROOT.RooFit.SumW2Error(True),
              ROOT.RooFit.Save(), ROOT.RooFit.PrintLevel(-1))

    sigma_eff = sqrt(0.5 * (dcb_sL.getVal()**2 + dcb_sR.getVal()**2))

    return {
        "x0": float(dcb_x0.getVal()),
        "sigmaL": float(dcb_sL.getVal()),
        "sigmaR": float(dcb_sR.getVal()),
        "alphaL": float(dcb_aL.getVal()),
        "nL": float(dcb_nL.getVal()),
        "alphaR": float(dcb_aR.getVal()),
        "nR": float(dcb_nR.getVal()),
        "sigma_eff": float(sigma_eff),
        "fit_lo": float(fit_lo),
        "fit_hi": float(fit_hi),
    }


def getFitResultDCBMulti(input_paths, mA_nominal, outdir, era, masspoint, channel=""):
    """Fit the summed/appended signal trees for one merged Run-period category."""
    if not input_paths:
        raise FileNotFoundError(f"No signal files provided for {era}/{channel}/{masspoint}")
    missing = [path for path in input_paths if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError("Missing signal file(s): " + ", ".join(missing))

    logging.info(f"Fitting category signal with DCB, nominal mA = {mA_nominal} GeV")

    chain = ROOT.TChain("Central")
    for path in input_paths:
        chain.Add(path)
    if chain.GetEntries() <= 0:
        raise RuntimeError(f"No signal entries found for {era}/{channel}/{masspoint}")

    fit = fit_dcb(chain, mA_nominal)
    fit_lo = fit["fit_lo"]
    fit_hi = fit["fit_hi"]

    hist_name = f"h_fit_{era}_{channel}".replace("-", "_")
    hist = ROOT.TH1D(hist_name, "", 100, fit_lo, fit_hi)
    hist.Sumw2()
    chain.Draw(f"mass>>{hist_name}", f"weight*(mass >= {fit_lo} && mass <= {fit_hi})", "goff")
    hist.SetDirectory(0)

    mass_plot = ROOT.RooRealVar("mass_plot", "mass", fit_lo, fit_hi)
    roo_data = ROOT.RooDataHist("data_plot", "", ROOT.RooArgList(mass_plot), hist)

    p_x0 = ROOT.RooRealVar("p_x0", "x0", fit["x0"])
    p_sL = ROOT.RooRealVar("p_sL", "sL", fit["sigmaL"])
    p_sR = ROOT.RooRealVar("p_sR", "sR", fit["sigmaR"])
    p_aL = ROOT.RooRealVar("p_aL", "aL", fit["alphaL"])
    p_nL = ROOT.RooRealVar("p_nL", "nL", fit["nL"])
    p_aR = ROOT.RooRealVar("p_aR", "aR", fit["alphaR"])
    p_nR = ROOT.RooRealVar("p_nR", "nR", fit["nR"])
    for var in [p_x0, p_sL, p_sR, p_aL, p_nL, p_aR, p_nR]:
        var.setConstant(True)
    dcb_plot = ROOT.RooCrystalBall(
        "dcb_plot", "", mass_plot, p_x0,
        p_sL, p_sR, p_aL, p_nL, p_aR, p_nR
    )

    fit_models = OrderedDict([("DCB", (dcb_plot, 7, ROOT.kSolid))])
    config = {
        "era": era, "xTitle": "m_{A} [GeV]", "yTitle": "Events",
        "rTitle": "Fit / MC", "rRange": [0.5, 1.5],
        "channel": channel, "masspoint": masspoint,
        "channelPosX": 0.2, "channelPosY": 0.74,
        "channelFont": 61, "channelSize": 0.04,
        "masspointPosX": 0.2, "masspointPosY": 0.69,
        "masspointFont": 61, "masspointSize": 0.04,
        "legend": [0.60, 0.65, 0.90, 0.78],
        "legendTextSize": 0.03, "iPos": 0, "maxDigits": 3,
        "colors": [ROOT.kRed],
    }
    canvas = FitCanvasWithRatio(roo_data, mass_plot, hist, fit_models, config)
    canvas.drawPadUp()
    canvas.drawPadDown()
    canvas.drawMasspoint()
    canvas.canv.SaveAs(f"{outdir}/signal_fit.{category_name(channel, era)}.png")

    result = {
        "x0": fit["x0"],
        "sigmaL": fit["sigmaL"],
        "sigmaR": fit["sigmaR"],
        "alphaL": fit["alphaL"],
        "nL": fit["nL"],
        "alphaR": fit["alphaR"],
        "nR": fit["nR"],
        "sigma_eff": fit["sigma_eff"],
    }
    logging.info(
        "Category DCB fit result: x0=%.2f, sigma_eff=%.3f GeV",
        result["x0"], result["sigma_eff"]
    )
    return result


def getCategoryBackgroundWeights(basedirs, mass_min, mass_max, outdir, category):
    """Calculate ParticleNet background weights over all suberas in one category.

    Vectorized reads (RDataFrame filter + AsNumpy); the accumulation stays a
    sequential python sum in the original file/entry order, so the totals
    are bitwise identical to the old per-entry loop (numpy's pairwise sum
    would differ in the last ulp)."""
    logging.info("Calculating category background weights for %s", category)
    weights = {}
    for pn_class, possible_categories in PARTICLENET_CLASS_MAPPING.items():
        total_weight = 0.0
        found_any = False
        for process in possible_categories:
            for basedir in basedirs:
                file_path = f"{basedir}/{process}.root"
                if not os.path.exists(file_path):
                    continue
                rfile = ROOT.TFile.Open(file_path, "READ")
                has_tree = bool(rfile.Get("Central"))
                rfile.Close()
                if not has_tree:
                    continue
                found_any = True
                arr = (ROOT.RDataFrame("Central", file_path)
                       .Filter(f"mass >= {mass_min!r} && mass <= {mass_max!r}")
                       .AsNumpy(["weight"]))["weight"]
                for w in arr.tolist():
                    total_weight += w
        weights[pn_class] = total_weight if found_any else 1.0 / 3.0

    total = sum(weights.values())
    if total > 0:
        weights = {key: val / total for key, val in weights.items()}
    else:
        weights = {key: 1.0 / 3.0 for key in weights}

    save_json({
        "category": category,
        "weights": weights,
        "mass_window": [float(mass_min), float(mass_max)],
    }, f"{outdir}/background_weights.{category}.json")
    return weights


def optimizeCategoryParticleNetThreshold(basedirs, masspoint, mass_min, mass_max, bg_weights, outdir, category):
    sig_scores = []
    sig_weights = []
    for basedir in basedirs:
        scores, weights, _ = loadDataset(basedir, masspoint, masspoint, mass_min, mass_max, bg_weights)
        if len(scores) > 0:
            sig_scores.append(scores)
            sig_weights.append(weights)
    if not sig_scores:
        logging.warning("ParticleNet signal scores not found for %s; no threshold applied", category)
        return -999., None

    bkg_processes = ["nonprompt", "WZ", "ZZ", "ttW", "ttZ", "ttH", "tZq", "conversion", "others"]
    bkg_scores = []
    bkg_weights = []
    for basedir in basedirs:
        for process in bkg_processes:
            scores, weights, _ = loadDataset(basedir, process, masspoint, mass_min, mass_max, bg_weights)
            if len(scores) > 0:
                bkg_scores.append(scores)
                bkg_weights.append(weights)

    if not bkg_scores:
        logging.warning("ParticleNet background scores not found for %s; no threshold applied", category)
        return -999., None

    best_threshold, initial_sensitivity, max_sensitivity = getOptimizedThreshold(
        np.concatenate(sig_scores),
        np.concatenate(sig_weights),
        np.concatenate(bkg_scores),
        np.concatenate(bkg_weights),
    )
    payload = {
        "category": category,
        "masspoint": masspoint,
        "threshold": float(best_threshold),
        "initial_sensitivity": float(initial_sensitivity),
        "max_sensitivity": float(max_sensitivity),
        "improvement": float(max_sensitivity / initial_sensitivity - 1) if initial_sensitivity > 0 else 0.0,
    }
    save_json(payload, f"{outdir}/threshold.{category}.json")
    return best_threshold, payload


def build_signal_component_templates(basedir, component, bin_edges, mass_min, mass_max,
                                     syst_categories, threshold, upper_threshold,
                                     bg_weights, masspoint):
    proc_map = {}
    central = getHist(
        basedir, masspoint, bin_edges, mass_min, mass_max,
        "Central", threshold, upper_threshold, bg_weights, masspoint
    )
    central.SetName(component)
    ensure_positive_integral(central)
    cap_stat_errors(central)
    proc_map["nominal"] = central

    for syst_name, variations, group in syst_categories["preprocessed_shape"]:
        if "signal" not in group:
            continue
        for direction in iter_shape_directions(variations):
            read_tree = f"{syst_name}_{direction}"
            combine_suffix = f"{syst_name}{direction}"
            try:
                hist = getHist(
                    basedir, masspoint, bin_edges, mass_min, mass_max,
                    read_tree, threshold, upper_threshold, bg_weights, masspoint
                )
                hist.SetName(f"{component}_{combine_suffix}")
                ensure_positive_integral(hist)
                cap_stat_errors(hist)
                proc_map[combine_suffix] = hist
            except (FileNotFoundError, RuntimeError) as exc:
                logging.warning("    Skipping %s/%s/%s: %s", component, syst_name, direction, exc)

    for syst_name, value, group in syst_categories["valued_shape"]:
        if "signal" not in group:
            continue
        for direction in ["up", "down"]:
            hist = create_scaled_hist(central, component, syst_name, value, direction)
            ensure_positive_integral(hist)
            cap_stat_errors(hist)
            suffix = f"{syst_name}Up" if direction == "up" else f"{syst_name}Down"
            proc_map[suffix] = hist

    for syst_name, variations, group in syst_categories["multi_variation"]:
        if "signal" not in group:
            continue
        tree_variations = []
        for var in variations:
            if var.startswith("pdf_"):
                tree_variations.append(f"PDF_{int(var.replace('pdf_', ''))}")
            else:
                tree_variations.append(var)
        try:
            hist_up, hist_down = createEnvelopeHists(
                basedir, masspoint, bin_edges, mass_min, mass_max,
                tree_variations, syst_name, threshold, upper_threshold, bg_weights, masspoint
            )
            hist_up.SetName(f"{component}_{syst_name}Up")
            hist_down.SetName(f"{component}_{syst_name}Down")
            ensure_positive_integral(hist_up)
            ensure_positive_integral(hist_down)
            cap_stat_errors(hist_up)
            cap_stat_errors(hist_down)
            proc_map[f"{syst_name}Up"] = hist_up
            proc_map[f"{syst_name}Down"] = hist_down
        except RuntimeError as exc:
            logging.warning("    Skipping envelope %s/%s: %s", component, syst_name, exc)

    return proc_map


def build_run_period_templates(args):
    """Build merged Run-period categories with subera component processes."""
    outdir = srspaths.template_dir(args.masspoint, args.method, args.era,
                                   args.channel, blind=args.blind)
    periods = resolve_run_periods(args.era)
    channels = resolve_channels(args.channel)
    mA_nominal = float(args.masspoint.split("_")[1].replace("MA", ""))

    logging.info("Starting Run-period component template generation")
    logging.info("  Era request: %s", args.era)
    logging.info("  Channel request: %s", args.channel)
    logging.info("  Mass point: %s", args.masspoint)
    logging.info("  Output directory: %s", outdir)

    if os.path.exists(outdir):
        logging.info("Removing existing output directory: %s", outdir)
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    category_outputs = OrderedDict()
    categories_json = OrderedDict()
    binning_json = {
        "construction": "run_period_components",
        "binning_type": "extended",
        "min_core_bins": MIN_CORE_BINS,
        "categories": OrderedDict(),
    }
    process_components = []
    background_validation = OrderedDict()
    threshold_summary = {"construction": "run_period_components", "categories": OrderedDict()}
    background_weight_summary = {"construction": "run_period_components", "categories": OrderedDict()}

    for period, suberas in periods:
        for channel in channels:
            cat = category_name(channel, period)
            logging.info("=" * 60)
            logging.info("Building category %s", cat)
            logging.info("=" * 60)

            basedirs = [srspaths.sample_dir(subera, channel, args.masspoint, args.method) for subera in suberas]
            signal_paths = [f"{basedir}/{args.masspoint}.root" for basedir in basedirs]
            fit_result = getFitResultDCBMulti(signal_paths, mA_nominal, outdir, period, args.masspoint, channel)
            x0 = fit_result["x0"]
            sigma_eff = fit_result["sigma_eff"]
            mass_min = max(x0 - 10.0 * sigma_eff, 12.0)
            mass_max = x0 + 10.0 * sigma_eff

            bg_weights = None
            best_threshold = -999.
            upper_threshold = None
            if args.method == "ParticleNet":
                bg_weights = getCategoryBackgroundWeights(basedirs, mass_min, mass_max, outdir, cat)
                best_threshold, threshold_payload = optimizeCategoryParticleNetThreshold(
                    basedirs, args.masspoint, mass_min, mass_max, bg_weights, outdir, cat
                )
                if threshold_payload:
                    threshold_summary["categories"][cat] = threshold_payload
                background_weight_summary["categories"][cat] = {"weights": bg_weights}

            all_bg_processes, bg_by_subera = category_background_processes(suberas, channel)

            def basedir_of(subera):
                return srspaths.sample_dir(subera, channel, args.masspoint, args.method)

            background_validation[cat], active_by_subera = \
                validate_category_backgrounds(
                    bg_by_subera, basedir_of,
                    calculate_adaptive_bins(x0, sigma_eff, 15),
                    mass_min, mass_max, args.masspoint,
                    best_threshold, upper_threshold, bg_weights,
                )

            bin_edges, n_core_final, apply_floor = scan_binning(
                cat, active_by_subera, basedir_of, x0, sigma_eff,
                mass_min, mass_max, args.masspoint,
                best_threshold, upper_threshold, bg_weights,
            )

            nbins = len(bin_edges) - 1
            bin_edges_vector = ROOT.std.vector["double"](bin_edges)
            templates = {}
            process_order = []
            category_processes = []

            if not args.blind:
                data_obs = sum_data_obs(
                    suberas, basedir_of, bin_edges, mass_min, mass_max,
                    best_threshold, upper_threshold, bg_weights,
                    args.masspoint,
                )
            else:
                data_obs = ROOT.TH1D("data_obs", "data_obs", nbins, bin_edges_vector.data())
                data_obs.SetDirectory(0)

            for subera in suberas:
                basedir = srspaths.sample_dir(subera, channel, args.masspoint, args.method)
                syst_config = load_systematics_block(subera, channel)
                syst_categories = categorize_systematics(syst_config)

                sig_comp = component_name("signal", subera, is_signal=True)
                logging.info("Processing %s", sig_comp)
                sig_map = build_signal_component_templates(
                    basedir, sig_comp, bin_edges, mass_min, mass_max,
                    syst_categories, best_threshold, upper_threshold,
                    bg_weights, args.masspoint
                )
                templates[sig_comp] = sig_map
                process_order.append(sig_comp)
                category_processes.append({
                    "name": sig_comp,
                    "base_process": "signal",
                    "physics_group": "signal",
                    "subera": subera,
                    "is_signal": True,
                })

                for proc in active_by_subera[subera]:
                    comp = component_name(proc, subera)
                    logging.info("Processing %s", comp)
                    floor_mode = "floor" if proc == "others" else "zero"
                    proc_map = build_component_shape_templates(
                        basedir, proc, comp, bin_edges, mass_min, mass_max,
                        syst_categories, best_threshold, upper_threshold,
                        bg_weights, args.masspoint, floor_mode
                    )
                    templates[comp] = proc_map
                    process_order.append(comp)
                    category_processes.append({
                        "name": comp,
                        "base_process": proc,
                        "physics_group": proc,
                        "subera": subera,
                        "is_signal": False,
                    })
                    if args.blind:
                        data_obs.Add(proc_map["nominal"])

            templates["data_obs"] = data_obs

            if apply_floor:
                logging.warning("Applying floor/zero fallback after failed binning scan for %s", cat)
                apply_floor_fallback(templates)

            bkg_process_list = [p["name"] for p in category_processes if not p["is_signal"]]
            pre_merge_nbins = len(bin_edges) - 1
            bin_edges, templates, n_syst_merges = apply_syst_driven_merging(
                bin_edges, templates, bkg_process_list,
                max_rel_syst=SYST_MERGE_THRESHOLD, logger=logging
            )
            if n_syst_merges:
                logging.warning(
                    "%s syst-merge: %d bins -> %d bins",
                    cat, pre_merge_nbins, len(bin_edges) - 1
                )

            category_outputs[cat] = {
                "templates": templates,
                "process_order": process_order,
                "processes": category_processes,
            }
            categories_json[cat] = {
                "category": cat,
                "channel": channel,
                "run_period": period,
                "suberas": suberas,
                "processes": category_processes,
            }
            binning_json["categories"][cat] = {
                "nbins": len(bin_edges) - 1,
                "bin_edges": [float(e) for e in bin_edges],
                "method": "AdaptiveExtendedBins",
                "sigma_eff": float(sigma_eff),
                "mass_min": float(mass_min),
                "mass_max": float(mass_max),
                "n_core_bins": int(n_core_final),
                "fit_model": "dcb",
                "fit_result": fit_result,
                "floor_applied": bool(apply_floor),
                "syst_merge_applied": n_syst_merges > 0,
                "n_bins_merged": int(n_syst_merges),
                "syst_merge_threshold": SYST_MERGE_THRESHOLD,
            }

            for proc_meta in category_processes:
                process_components.append({"category": cat, **proc_meta})

            logging.info("Finished %s: data_obs=%.4f, processes=%d",
                         cat, templates["data_obs"].Integral(), len(process_order))

    write_run_period_shapes(outdir, category_outputs)

    write_template_sidecars(outdir, categories_json, binning_json,
                            background_validation, process_components)
    if threshold_summary["categories"]:
        save_json(threshold_summary, f"{outdir}/threshold.json")
    if background_weight_summary["categories"]:
        save_json(background_weight_summary, f"{outdir}/background_weights.json")

    logging.info("Run-period component template generation complete: %s/shapes.root", outdir)


def build_interp_period_templates(args):
    """Interp-signal (parametric) templates for one scan-grid mass point.

    Group SEED: full build — backgrounds/data/binning with the seed's own
    interpolated mean/sigma (the group's shared payload), plus the seed's
    parametric signal. Group MEMBER: clones the seed's background/data
    histograms and binning verbatim and injects only its own parametric
    signal, into {seed_dir}/points/{member}. The seed job must have run
    first (the DAG orders it)."""
    import json

    import interpolation_config
    import param_signal
    from closInterpYields import predict_shape_params
    from fitInterpYieldModel import predict_yield

    masspoint = args.masspoint
    seed = interpolation_config.group_seed(masspoint)
    is_seed = seed == masspoint
    mhc, mA = srspaths.masspoint_mhc_ma(masspoint)
    mA = float(mA)

    if is_seed:
        outdir = srspaths.template_dir(masspoint, args.method, args.era,
                                       args.channel, blind=args.blind,
                                       source="interp-signal")
        seed_dir = None
    else:
        seed_dir = srspaths.template_dir(seed, args.method, args.era,
                                         args.channel, blind=args.blind,
                                         source="interp-signal")
        outdir = srspaths.interp_member_dir(seed, masspoint, args.era,
                                            args.channel, blind=args.blind)
        if not os.path.exists(f"{seed_dir}/shapes.root"):
            raise FileNotFoundError(
                f"{seed_dir}/shapes.root not found — build the group seed "
                f"{seed} first (members clone its backgrounds)")

    polys, _ = interpolation_config.load_shape_polynomials(mhc)
    with open(os.path.join(srspaths.interpolation_fits_dir(mhc),
                           "yields", "yield_model.json")) as f:
        yield_model = json.load(f)["model"]
    delta_model = param_signal.load_delta_model(mhc)
    uncertainties = param_signal.load_interp_uncertainties()

    logging.info("Interp-signal templates for %s (group seed %s, %s)",
                 masspoint, seed, "seed build" if is_seed else "member")

    if os.path.exists(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)

    periods = resolve_run_periods(args.era)
    channels = resolve_channels(args.channel)

    category_outputs = OrderedDict()
    categories_json = OrderedDict()
    binning_json = {
        "construction": "run_period_components",
        "binning_type": "extended",
        "min_core_bins": MIN_CORE_BINS,
        "categories": OrderedDict(),
    }
    process_components = []
    background_validation = OrderedDict()
    extra_systematics = OrderedDict()
    param_meta = OrderedDict()
    warnings = []

    seed_file = None
    seed_sidecars = {}
    if not is_seed:
        seed_file = ROOT.TFile.Open(f"{seed_dir}/shapes.root", "READ")
        for name in ("categories", "binning", "background_validation",
                     "process_list"):
            with open(f"{seed_dir}/{name}.json") as f:
                seed_sidecars[name] = json.load(f)

    for period, suberas in periods:
        for channel in channels:
            cat = category_name(channel, period)
            study_channel = interpolation_config.study_channel_for(
                channel, masspoint)
            cat_key = interpolation_config.category_key(study_channel, period)
            params, clipped = predict_shape_params(polys[cat_key], mA)
            if clipped:
                warnings.append(f"[{cat}] clipped params: {clipped}")
            interp_win = interpolation_config.interp_window(polys[cat_key], mA)
            sigma_eff = float(np.sqrt(0.5 * (params["sigmaL"] ** 2
                                             + params["sigmaR"] ** 2)))
            yields = {subera: predict_yield(yield_model, study_channel,
                                            subera, mA)
                      for subera in suberas}

            if is_seed:
                x0 = params["x0"]
                mass_min, mass_max = interp_win

                def basedir_of(subera, _ch=channel):
                    return srspaths.sample_dir(subera, _ch, masspoint,
                                               "Baseline")

                _all_bg, bg_by_subera = category_background_processes(
                    suberas, channel)
                background_validation[cat], active_by_subera = \
                    validate_category_backgrounds(
                        bg_by_subera, basedir_of,
                        calculate_adaptive_bins(x0, sigma_eff, 15),
                        mass_min, mass_max, masspoint)
                bin_edges, n_core_final, apply_floor = scan_binning(
                    cat, active_by_subera, basedir_of, x0, sigma_eff,
                    mass_min, mass_max, masspoint)

                nbins = len(bin_edges) - 1
                if not args.blind:
                    data_obs = sum_data_obs(suberas, basedir_of, bin_edges,
                                            mass_min, mass_max,
                                            masspoint=masspoint)
                else:
                    edges_vec = ROOT.std.vector["double"](
                        [float(e) for e in bin_edges])
                    data_obs = ROOT.TH1D("data_obs", "data_obs", nbins,
                                         edges_vec.data())
                    data_obs.SetDirectory(0)
            else:
                seed_binning = seed_sidecars["binning"]["categories"][cat]
                bin_edges = [float(e) for e in seed_binning["bin_edges"]]
                mass_min = float(seed_binning["mass_min"])
                mass_max = float(seed_binning["mass_max"])
                n_core_final = int(seed_binning["n_core_bins"])
                apply_floor = bool(seed_binning["floor_applied"])
                background_validation[cat] = \
                    seed_sidecars["background_validation"][cat]

            templates = {}
            process_order = []
            category_processes = []
            member_signal = {}

            for subera in suberas:
                syst_config = load_systematics_block(subera, channel)
                syst_categories = categorize_systematics(syst_config)

                sig_comp = component_name("signal", subera, is_signal=True)
                model = param_signal.ParametricSignal(
                    f"{cat}_{subera}_{masspoint}", params, interp_win)
                interp_terms = param_signal.interp_shape_terms(
                    uncertainties, study_channel, channel, period, params)
                n_pred, _err_pred = yields[subera]
                proc_map = param_signal.build_signal_component(
                    model, sig_comp, param_signal.delta_key(subera,
                                                            study_channel),
                    delta_model, syst_categories, bin_edges,
                    (mass_min, mass_max), n_pred, mA, interp_terms, warnings)
                templates[sig_comp] = proc_map
                process_order.append(sig_comp)
                category_processes.append({
                    "name": sig_comp, "base_process": "signal",
                    "physics_group": "signal", "subera": subera,
                    "is_signal": True,
                })
                for key, hist in proc_map.items():
                    name = sig_comp if key == "nominal" else f"{sig_comp}{key}"
                    member_signal[name] = hist

                if is_seed:
                    for proc in active_by_subera[subera]:
                        comp = component_name(proc, subera)
                        floor_mode = "floor" if proc == "others" else "zero"
                        proc_map_bkg = build_component_shape_templates(
                            basedir_of(subera), proc, comp, bin_edges,
                            mass_min, mass_max, syst_categories,
                            -999., None, None, masspoint, floor_mode)
                        templates[comp] = proc_map_bkg
                        process_order.append(comp)
                        category_processes.append({
                            "name": comp, "base_process": proc,
                            "physics_group": proc, "subera": subera,
                            "is_signal": False,
                        })
                        if args.blind:
                            data_obs.Add(proc_map_bkg["nominal"])

                extra_systematics[f"{subera}|{channel}"] = \
                    param_signal.interp_systematics_block(
                        uncertainties, study_channel, channel, period,
                        subera, mA)

            if is_seed:
                templates["data_obs"] = data_obs

                if apply_floor:
                    logging.warning("Floor fallback for %s", cat)
                    apply_floor_fallback(templates)

                bkg_names = [p["name"] for p in category_processes
                             if not p["is_signal"]]
                pre_merge = len(bin_edges) - 1
                bin_edges, templates, n_syst_merges = \
                    apply_syst_driven_merging(
                        bin_edges, templates, bkg_names,
                        max_rel_syst=SYST_MERGE_THRESHOLD, logger=logging)
                if n_syst_merges:
                    logging.warning("%s syst-merge: %d -> %d bins", cat,
                                    pre_merge, len(bin_edges) - 1)
                category_outputs[cat] = {
                    "templates": templates,
                    "process_order": process_order,
                    "processes": category_processes,
                }
                categories_processes = category_processes
            else:
                # Member: seed's category payload + this point's signal.
                seed_cat = seed_sidecars["categories"]["categories"][cat]
                categories_processes = seed_cat["processes"]
                # structural check: same signal component list
                seed_signal_names = {p["name"] for p in categories_processes
                                     if p["is_signal"]}
                own_signal_names = {p["name"] for p in category_processes}
                if seed_signal_names != own_signal_names:
                    raise RuntimeError(
                        f"{cat}: member signal components {own_signal_names} "
                        f"!= seed's {seed_signal_names}")
                n_syst_merges = int(seed_binning.get("n_bins_merged", 0))
                category_outputs[cat] = {
                    "seed_tdir": cat,
                    "member_signal": member_signal,
                    "nbins": len(bin_edges) - 1,
                }

            categories_json[cat] = {
                "category": cat,
                "channel": channel,
                "run_period": period,
                "suberas": suberas,
                "processes": categories_processes,
            }
            binning_json["categories"][cat] = {
                "nbins": len(bin_edges) - 1,
                "bin_edges": [float(e) for e in bin_edges],
                "method": "AdaptiveExtendedBins",
                "sigma_eff": float(sigma_eff),
                "mass_min": float(mass_min),
                "mass_max": float(mass_max),
                "n_core_bins": int(n_core_final),
                "fit_model": "interpolated",
                "fit_result": {**{k: float(v) for k, v in params.items()},
                               "sigma_eff": float(sigma_eff)},
                "floor_applied": bool(apply_floor),
                "syst_merge_applied": n_syst_merges > 0,
                "n_bins_merged": int(n_syst_merges),
                "syst_merge_threshold": SYST_MERGE_THRESHOLD,
                "binning_source": "interpolated" if is_seed
                                  else f"seed:{seed}",
            }
            for proc_meta in categories_processes:
                process_components.append({"category": cat, **proc_meta})
            param_meta[cat] = {
                "study_channel": study_channel,
                "params": {k: float(v) for k, v in params.items()},
                "clipped": clipped,
                "interp_window": [float(interp_win[0]),
                                  float(interp_win[1])],
                "template_window": [float(mass_min), float(mass_max)],
                "yields": {subera: {"n_pred": float(n), "err_pred": float(e)}
                           for subera, (n, e) in yields.items()},
            }

    if is_seed:
        write_run_period_shapes(outdir, category_outputs)
    else:
        # Copy the seed's category payloads, replacing the signal hists.
        fout = ROOT.TFile.Open(f"{outdir}/shapes.root", "RECREATE")
        for cat, payload in category_outputs.items():
            tdir_in = seed_file.Get(cat)
            if not tdir_in:
                raise RuntimeError(f"seed shapes.root has no category {cat}")
            tdir_out = fout.mkdir(cat)
            tdir_out.cd()
            member_signal = payload["member_signal"]
            written = set()
            for key in tdir_in.GetListOfKeys():
                name = key.GetName()
                if name in member_signal:
                    hist = member_signal[name]
                    if hist.GetNbinsX() != payload["nbins"]:
                        raise RuntimeError(
                            f"{cat}/{name}: member nbins "
                            f"{hist.GetNbinsX()} != seed {payload['nbins']}")
                    hist.Write(name)
                    written.add(name)
                elif name.startswith("signal_"):
                    warnings.append(f"[{cat}] seed histogram {name} has no "
                                    "member counterpart; dropped")
                else:
                    obj = key.ReadObj()
                    obj.Write(name)
            for name in sorted(set(member_signal) - written):
                member_signal[name].Write(name)
        fout.Close()
        seed_file.Close()

    write_template_sidecars(outdir, categories_json, binning_json,
                            background_validation, process_components)
    tag = f"{args.era}.{args.channel}"
    save_json({"systematics": extra_systematics},
              f"{outdir}/extra_systematics.{tag}.json")
    save_json({
        "meta": {
            "masspoint": masspoint, "group_seed": seed,
            "mhc": mhc, "mA": mA,
            "signal_source": "interp-signal",
            "model_inputs": {
                "polynomials": f"fits/MHc{mhc}/polynomials.json",
                "yield_model": f"fits/MHc{mhc}/yields/yield_model.json",
                "delta_model": f"fits/MHc{mhc}/shape_deltas/delta_model.json",
                "uncertainties": "configs/interpolation_uncertainties.json",
            },
        },
        "categories": param_meta,
        "warnings": warnings,
    }, f"{outdir}/param_signal.{tag}.json")
    for w in warnings:
        logging.warning("%s", w)
    logging.info("Interp-signal templates complete: %s/shapes.root", outdir)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(levelname)s - %(message)s')

    if args.method not in ("Baseline", "ParticleNet"):
        raise ValueError(
            f"Unknown --method '{args.method}' (expected Baseline or ParticleNet)"
        )

    if args.signal_source == "interp-signal":
        if args.method != "Baseline":
            raise ValueError("interp-signal templates exist for Baseline only")
        build_interp_period_templates(args)
    else:
        build_run_period_templates(args)

if __name__ == "__main__":
    main()
