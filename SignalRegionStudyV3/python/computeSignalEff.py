#!/usr/bin/env python3
"""Compute per-mass-point signal efficiencies for SignalRegionStudyV3.

The total signal efficiency factorizes into three independent stages, each read
from a different place:

1. Generator filter efficiency -- ``filterEff`` field per sample in the era's
   ``CommonSampleInfo.json`` (baked into ``MCweight()`` as a constant multiplier,
   so it cancels in the cutflow ratio and is applied here explicitly).
2. Baseline selection efficiency -- weighted ratio of the PromptAnalyzer cutflow
   histograms in the histogram-mode ``_RunSyst_RunTheoryUnc`` production:
   ``{SR1E2Mu|SR3Mu}/Central/cutflow`` Final bin over
   ``{Run1E2Mu|Run3Mu}/Central/cutflow`` Initial bin.
3. Mass-window efficiency -- weighted fraction of SR events (``Events_Central``,
   reconstructed A mass) inside ``x0 +/- 5 * sigma_eff`` from the Double Crystal
   Ball fit (reusing ``fit_dcb`` from ``makeBinnedTemplates``).

Output: one JSON per mass point at ``results/signal_eff/<masspoint>.json``,
keyed by channel then era.

Usage:
    python3 python/computeSignalEff.py --masspoint MHc130_MA90
    python3 python/computeSignalEff.py --all
    python3 python/computeSignalEff.py --all --era 2018 --channel SR1E2Mu
"""
import os
import re
import glob
import json
import logging
import argparse

import ROOT

from makeBinnedTemplates import fit_dcb
from template_utils import save_json

ROOT.gROOT.SetBatch(True)

# =============================================================================
# Constants
# =============================================================================

# Channel -> PromptAnalyzer input channel directory. Only these two channels
# carry signal. The cutflow lives in the histogram-mode production, which has no
# MHc*/NoHistMode userflag, so always use the standard "_RunSyst_RunTheoryUnc"
# directory regardless of whether the mass point is ParticleNet-trained.
CHANNEL_INPUT_MAP = {
    "SR1E2Mu": "Run1E2Mu",
    "SR3Mu": "Run3Mu",
}
HISTMODE_SUFFIX = "_RunSyst_RunTheoryUnc"
SIGNAL_PREFIX = "TTToHcToWAToMuMu-"
COMMONINFO_BASE = "SKNanoAnalyzer/data/Run3_v13_Run2_v9"

# Count SR events within x0 +/- SIGMA_WINDOW * sigma_eff (the extended-binning
# core window; see AGENTS.md "Binning and Signal Fits").
SIGMA_WINDOW = 5.0

_MASSPOINT_RE = re.compile(r"^MHc\d+_MA\d+$")


# =============================================================================
# Helpers
# =============================================================================

def require_workdir():
    """Return $WORKDIR, failing fast if it is not set."""
    workdir = os.environ.get("WORKDIR")
    if not workdir:
        raise RuntimeError("WORKDIR not set; source the module-local setup.sh first")
    return workdir


def load_eras(workdir):
    """Return the analysis eras from samplegroups.json (excludes 'aliases')."""
    path = os.path.join(workdir, "SignalRegionStudyV3", "configs", "samplegroups.json")
    with open(path) as f:
        groups = json.load(f)
    return [key for key in groups.keys() if key != "aliases"]


def parse_masspoint(masspoint):
    """Return (mHc, mA) integers parsed from a mass-point label."""
    parts = masspoint.split("_")
    if len(parts) != 2 or not parts[0].startswith("MHc") or not parts[1].startswith("MA"):
        raise ValueError(f"Invalid mass point format: {masspoint}")
    return int(parts[0].replace("MHc", "")), int(parts[1].replace("MA", ""))


def signal_path(workdir, era, channel, masspoint):
    """Return the histogram-mode signal ROOT file path (may not exist)."""
    input_channel = CHANNEL_INPUT_MAP[channel]
    return os.path.join(
        workdir, "SKNanoOutput", "PromptAnalyzer",
        f"{input_channel}{HISTMODE_SUFFIX}", era, f"{SIGNAL_PREFIX}{masspoint}.root",
    )


def mass_expression(channel, mHc, mA):
    """Reconstructed A-mass expression (RDataFrame/C++), mirroring the rule in
    preprocess.py:303-312 (BasePreprocessor._select_mass)."""
    if "1E2Mu" in channel:
        return "mass1"
    if "3Mu" in channel:
        if mHc >= 100 and mA >= 60:
            return "std::max(mass1, mass2)"
        return "std::min(mass1, mass2)"
    raise ValueError(f"Unknown channel: {channel}")


def discover_masspoints(workdir, eras, channels):
    """Return the sorted union of mass points that have a signal file on disk."""
    found = set()
    for channel in channels:
        input_channel = CHANNEL_INPUT_MAP[channel]
        for era in eras:
            pattern = os.path.join(
                workdir, "SKNanoOutput", "PromptAnalyzer",
                f"{input_channel}{HISTMODE_SUFFIX}", era, f"{SIGNAL_PREFIX}*.root",
            )
            for path in glob.glob(pattern):
                token = os.path.basename(path)[len(SIGNAL_PREFIX):-len(".root")]
                if _MASSPOINT_RE.match(token):
                    found.add(token)
    return sorted(found, key=parse_masspoint)


# =============================================================================
# Efficiency stages
# =============================================================================

def gen_filter_eff(workdir, era, masspoint):
    """Return (filterEff, xsec) from the era's CommonSampleInfo.json, or
    (None, None) if the file or sample key is absent."""
    path = os.path.join(workdir, COMMONINFO_BASE, era, "Sample", "CommonSampleInfo.json")
    if not os.path.exists(path):
        logging.warning("CommonSampleInfo.json not found: %s", path)
        return None, None
    with open(path) as f:
        info = json.load(f)
    entry = info.get(f"{SIGNAL_PREFIX}{masspoint}")
    if entry is None:
        return None, None
    return entry.get("filterEff"), entry.get("xsec")


def baseline_eff(root_path, channel):
    """Return (initial, final, final/initial) from the cutflow histograms.

    Denominator = {Run1E2Mu|Run3Mu}/Central/cutflow Initial (ROOT bin 1);
    numerator   = {SR1E2Mu|SR3Mu}/Central/cutflow Final (ROOT bin 9). Both are
    filled with the same initialWeight, so the ratio is a clean efficiency.
    """
    run_dir = CHANNEL_INPUT_MAP[channel]
    tfile = ROOT.TFile.Open(root_path, "READ")
    if not tfile or tfile.IsZombie():
        raise IOError(f"Failed to open input file: {root_path}")
    try:
        h_initial = tfile.Get(f"{run_dir}/Central/cutflow")
        h_final = tfile.Get(f"{channel}/Central/cutflow")
        if not h_initial or not h_final:
            raise RuntimeError(f"Cutflow histogram(s) missing in {root_path}")
        initial = h_initial.GetBinContent(1)  # CutStage.Initial
        final = h_final.GetBinContent(9)       # CutStage.Final (cutIndex 8)
    finally:
        tfile.Close()
    if initial <= 0.0:
        raise RuntimeError(f"Non-positive Initial cutflow ({initial}) in {root_path}")
    return float(initial), float(final), float(final / initial)


def mass_window_eff(root_path, channel, mHc, mA, tmp_path):
    """Fit the DCB and count weighted SR events within x0 +/- 5*sigma_eff.

    Returns a dict with x0, sigma_eff, mass_window, n_sr, n_window and the ratio.
    Materializes a temporary 'Central' tree (reconstructed mass + weight) so the
    validated fit_dcb helper can be reused verbatim.
    """
    expr = mass_expression(channel, mHc, mA)
    df = ROOT.RDataFrame("Events_Central", root_path).Define("mass", expr)

    snapshot_opts = ROOT.RDF.RSnapshotOptions()
    snapshot_opts.fMode = "RECREATE"
    df.Snapshot("Central", tmp_path, ["mass", "weight"], snapshot_opts)

    chain = ROOT.TChain("Central")
    chain.Add(tmp_path)
    n_entries = chain.GetEntries()
    if n_entries <= 0:
        return {"status": "empty_sr"}

    fit = fit_dcb(chain, float(mA))
    x0 = fit["x0"]
    sigma_eff = fit["sigma_eff"]
    lo = x0 - SIGMA_WINDOW * sigma_eff
    hi = x0 + SIGMA_WINDOW * sigma_eff

    count_df = ROOT.RDataFrame("Central", tmp_path)
    n_sr = count_df.Sum("weight").GetValue()
    n_window = count_df.Filter(f"mass >= {lo} && mass <= {hi}").Sum("weight").GetValue()
    if n_sr <= 0.0:
        return {"status": "empty_sr"}

    return {
        "status": "ok",
        "x0": float(x0),
        "sigma_eff": float(sigma_eff),
        "mass_window": [float(lo), float(hi)],
        "n_sr": float(n_sr),
        "n_window": float(n_window),
        "mass_window_eff": float(n_window / n_sr),
    }


def blank_entry(status, gen_eff=None, xsec=None):
    """Return an era entry with all efficiency fields defined (None where N/A)."""
    return {
        "status": status,
        "gen_filter_eff": gen_eff,
        "xsec": xsec,
        "baseline_eff": None,
        "cutflow_initial": None,
        "cutflow_final": None,
        "mass_window_eff": None,
        "x0": None,
        "sigma_eff": None,
        "mass_window": None,
        "n_sr": None,
        "n_window": None,
        "total_eff": None,
    }


def compute_entry(workdir, era, channel, masspoint, mHc, mA, tmp_path):
    """Compute the full per-(channel, era) efficiency entry for one mass point."""
    gen_eff, xsec = gen_filter_eff(workdir, era, masspoint)

    root_path = signal_path(workdir, era, channel, masspoint)
    if not os.path.exists(root_path):
        return blank_entry("missing_file", gen_eff, xsec)

    entry = blank_entry("ok", gen_eff, xsec)

    try:
        initial, final, base_eff = baseline_eff(root_path, channel)
    except Exception as exc:  # noqa: BLE001 - corrupt/unopenable files must not abort the run
        logging.warning("Cutflow read failed for %s %s %s: %s", masspoint, channel, era, exc)
        entry["status"] = "read_error"
        return entry
    entry["cutflow_initial"] = initial
    entry["cutflow_final"] = final
    entry["baseline_eff"] = base_eff

    try:
        mw = mass_window_eff(root_path, channel, mHc, mA, tmp_path)
    except Exception as exc:  # noqa: BLE001 - low-stat fits can fail in many ways
        logging.warning("DCB fit failed for %s %s %s: %s", masspoint, channel, era, exc)
        entry["status"] = "fit_failed"
        return entry

    if mw["status"] != "ok":
        entry["status"] = mw["status"]
        return entry

    entry["mass_window_eff"] = mw["mass_window_eff"]
    entry["x0"] = mw["x0"]
    entry["sigma_eff"] = mw["sigma_eff"]
    entry["mass_window"] = mw["mass_window"]
    entry["n_sr"] = mw["n_sr"]
    entry["n_window"] = mw["n_window"]

    if gen_eff is not None:
        entry["total_eff"] = float(gen_eff * base_eff * mw["mass_window_eff"])
    return entry


def compute_masspoint(workdir, masspoint, eras, channels, tmp_path):
    """Build the full result dict for one mass point."""
    mHc, mA = parse_masspoint(masspoint)
    result = {
        "masspoint": masspoint,
        "mHc": mHc,
        "mA": mA,
        "sigma_window": SIGMA_WINDOW,
        "channels": {},
    }
    for channel in channels:
        result["channels"][channel] = {}
        for era in eras:
            logging.info("Processing %s / %s / %s", masspoint, channel, era)
            result["channels"][channel][era] = compute_entry(
                workdir, era, channel, masspoint, mHc, mA, tmp_path
            )
    return result


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Compute signal efficiencies for SignalRegionStudyV3")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--masspoint", type=str, help="single mass point, e.g. MHc130_MA90")
    group.add_argument("--all", action="store_true", help="auto-discover and process all mass points")
    parser.add_argument("--era", type=str, default=None, help="restrict to a single era")
    parser.add_argument("--channel", type=str, default=None, choices=list(CHANNEL_INPUT_MAP.keys()),
                        help="restrict to a single channel")
    parser.add_argument("--outdir", type=str, default=None,
                        help="output directory (default: results/signal_eff)")
    parser.add_argument("--debug", action="store_true", help="verbose logging")
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    workdir = require_workdir()
    module_dir = os.path.join(workdir, "SignalRegionStudyV3")

    eras = load_eras(workdir)
    if args.era:
        if args.era not in eras:
            raise ValueError(f"Unknown era '{args.era}'; known eras: {eras}")
        eras = [args.era]

    channels = list(CHANNEL_INPUT_MAP.keys())
    if args.channel:
        channels = [args.channel]

    if args.all:
        masspoints = discover_masspoints(workdir, eras, channels)
        if not masspoints:
            raise RuntimeError("No signal files discovered for the requested eras/channels")
        logging.info("Discovered %d mass points", len(masspoints))
    else:
        masspoints = [args.masspoint]

    outdir = args.outdir or os.path.join(module_dir, "results", "signal_eff")
    os.makedirs(outdir, exist_ok=True)

    scratch = os.environ.get("TMPDIR", "/tmp")
    tmp_path = os.path.join(scratch, f"signal_eff_tmp_{os.getpid()}.root")

    try:
        for masspoint in masspoints:
            result = compute_masspoint(workdir, masspoint, eras, channels, tmp_path)
            out_path = os.path.join(outdir, f"{masspoint}.json")
            save_json(result, out_path)
            logging.info("Wrote %s", out_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    main()
