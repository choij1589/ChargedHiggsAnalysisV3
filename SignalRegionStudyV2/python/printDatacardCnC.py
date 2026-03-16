#!/usr/bin/env python3
"""
Generate Cut-and-Count (CnC) HiggsCombine datacard for cross-checking binned template limits.

Reads existing shapes.root and integrates histograms within a ±3σ_voigt mass window
to produce a counting-experiment datacard (no shape information, single bin).

Usage:
    printDatacardCnC.py --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 \
        --method Baseline --binning extended
"""
import os
import sys
import json
import math
import logging
import argparse
import ROOT

# Argument parsing
parser = argparse.ArgumentParser(description="Generate Cut-and-Count datacard from templates")
parser.add_argument("--era", required=True, type=str, help="Data-taking period")
parser.add_argument("--channel", required=True, type=str, help="Analysis channel (SR1E2Mu, SR3Mu)")
parser.add_argument("--masspoint", required=True, type=str, help="Signal mass point (e.g., MHc130_MA90)")
parser.add_argument("--method", required=True, type=str, help="Template method (Baseline, ParticleNet)")
parser.add_argument("--binning", default="uniform", choices=["uniform", "extended"],
                    help="Binning method: 'uniform' or 'extended'")
parser.add_argument("--unblind", action="store_true", help="Use unblind templates")
parser.add_argument("--partial-unblind", action="store_true", dest="partial_unblind",
                    help="Use partial-unblind templates")
parser.add_argument("--nsigma", type=float, default=3.0, help="Mass window half-width in sigma_voigt (default: 3.0)")
parser.add_argument("--output", type=str, default=None, help="Output datacard path (default: auto)")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args = parser.parse_args()

if args.unblind and args.partial_unblind:
    raise ValueError("--unblind and --partial-unblind are mutually exclusive")

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                    format='%(levelname)s - %(message)s')

WORKDIR = os.getenv("WORKDIR")
if not WORKDIR:
    raise EnvironmentError("WORKDIR not set. Please run 'source setup.sh'")

# Template directory (same logic as printDatacard.py)
binning_suffix = args.binning
if args.unblind:
    binning_suffix = f"{args.binning}_unblind"
elif args.partial_unblind:
    binning_suffix = f"{args.binning}_partial_unblind"
TEMPLATE_DIR = (f"{WORKDIR}/SignalRegionStudyV2/templates"
                f"/{args.era}/{args.channel}/{args.masspoint}/{args.method}/{binning_suffix}")

ROOT.gROOT.SetBatch(True)

# Threshold constants (same as printDatacard.py)
MIN_YIELD_THRESHOLD = 1e-6
MAX_LNN_VALUE = 2.0


def load_signal_fit():
    """Load signal fit parameters from signal_fit.json."""
    fit_path = f"{TEMPLATE_DIR}/signal_fit.json"
    if not os.path.exists(fit_path):
        raise FileNotFoundError(f"signal_fit.json not found: {fit_path}")
    with open(fit_path) as f:
        return json.load(f)


def compute_window(fit_params, nsigma=3.0):
    """
    Compute ±nsigma Voigt mass window.

    sigma_voigt = sqrt(width^2 + sigma^2)
    Returns (mass_min, mass_max).
    """
    mA = fit_params["mass"]
    width = fit_params["width"]
    sigma = fit_params["sigma"]
    sigma_voigt = math.sqrt(width ** 2 + sigma ** 2)
    mass_min = mA - nsigma * sigma_voigt
    mass_max = mA + nsigma * sigma_voigt
    logging.info(f"Mass window ±{nsigma}σ_voigt: [{mass_min:.2f}, {mass_max:.2f}] GeV "
                 f"(mA={mA:.2f}, width={width:.3f}, sigma={sigma:.3f}, σ_voigt={sigma_voigt:.3f})")
    return mass_min, mass_max


def load_systematics_config(era, channel):
    """Load systematic configuration for the given era and channel."""
    config_path = f"{WORKDIR}/SignalRegionStudyV2/configs/systematics.{era}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Systematics config not found: {config_path}")
    with open(config_path) as f:
        config = json.load(f)
    if channel not in config:
        raise ValueError(f"Channel '{channel}' not found in {config_path}")
    return config[channel]


def load_process_list():
    """Load process list from template directory."""
    path = f"{TEMPLATE_DIR}/process_list.json"
    if not os.path.exists(path):
        raise FileNotFoundError(f"process_list.json not found: {path}")
    with open(path) as f:
        return json.load(f)


def load_lowstat_json():
    """Load lowstat.json if it exists (may not exist if no low-stat processes)."""
    path = f"{TEMPLATE_DIR}/lowstat.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def integrate_in_window(hist, mass_min, mass_max):
    """Sum bin contents for bins whose centers fall within [mass_min, mass_max]."""
    if not hist:
        return 0.0
    total = 0.0
    for i in range(1, hist.GetNbinsX() + 1):
        center = hist.GetBinCenter(i)
        if mass_min <= center <= mass_max:
            total += hist.GetBinContent(i)
    return total


class CnCDatacardManager:
    """Generates counting-experiment datacard by integrating histograms in ±3σ window."""

    def __init__(self, era, channel, masspoint, method, binning, mass_min, mass_max):
        self.era = era
        self.channel = channel
        self.signal = masspoint
        self.method = method
        self.binning = binning
        self.mass_min = mass_min
        self.mass_max = mass_max
        self.backgrounds = []
        self.yields = {}   # process -> yield in window

        shapes_path = f"{TEMPLATE_DIR}/shapes.root"
        if not os.path.exists(shapes_path):
            raise FileNotFoundError(f"shapes.root not found: {shapes_path}")
        self.rtfile = ROOT.TFile.Open(shapes_path, "READ")

        # Load process list
        process_config = load_process_list()
        separate_processes = process_config.get("separate_processes", [])
        all_backgrounds = separate_processes + ["others"]

        for bkg in all_backgrounds:
            if bkg in self.backgrounds:
                continue
            hist = self.rtfile.Get(bkg)
            rate = integrate_in_window(hist, mass_min, mass_max)
            if rate > 0:
                self.backgrounds.append(bkg)
                self.yields[bkg] = rate
                logging.debug(f"Background '{bkg}': yield in window = {rate:.4f}")
            else:
                logging.warning(f"Background '{bkg}' has zero/negative yield in window ({rate:.4f}), dropping")

        # Signal
        sig_hist = self.rtfile.Get(self.signal)
        self.yields["signal"] = integrate_in_window(sig_hist, mass_min, mass_max)

        # data_obs
        data_hist = self.rtfile.Get("data_obs")
        self.yields["data_obs"] = integrate_in_window(data_hist, mass_min, mass_max)

        if not self.backgrounds:
            raise ValueError("No backgrounds with positive yields in window!")

        logging.info(f"Active backgrounds: {self.backgrounds}")
        logging.info(f"Signal yield in window: {self.yields['signal']:.4f}")
        logging.info(f"Data obs in window:     {self.yields['data_obs']:.4f}")

    def _get_hist(self, process, variation="Central"):
        """Get histogram from ROOT file."""
        base = self.signal if process == "signal" else process
        name = base if variation == "Central" else f"{base}_{variation}"
        return self.rtfile.Get(name)

    def _integrate(self, process, variation="Central"):
        """Integrate histogram in window."""
        hist = self._get_hist(process, variation)
        return integrate_in_window(hist, self.mass_min, self.mass_max)

    def compute_shape_lnn(self, process, syst_name):
        """
        Convert shape systematic to lnN by integrating up/down histograms in window.

        Returns formatted value string or "-" if not applicable.
        """
        central = self._integrate(process)
        if central < MIN_YIELD_THRESHOLD:
            return "-"

        up = self._integrate(process, f"{syst_name}Up")
        down = self._integrate(process, f"{syst_name}Down")

        # Check that at least one variation exists (non-zero integration means hist found)
        base = self.signal if process == "signal" else process
        has_up = self.rtfile.Get(f"{base}_{syst_name}Up") is not None
        has_down = self.rtfile.Get(f"{base}_{syst_name}Down") is not None

        if not has_up and not has_down:
            return "-"

        effects = []
        if has_up:
            effects.append(abs(up - central) / central)
        if has_down and central > MIN_YIELD_THRESHOLD:
            effects.append(abs(down - central) / central)

        if not effects:
            return "-"

        rate_effect = max(effects)
        if rate_effect < 0.001:
            return "-"

        lnn_value = min(1.0 + rate_effect, MAX_LNN_VALUE)
        return f"{lnn_value:.3f}"

    def format_syst_value(self, process, syst_name, syst_config, lowstat_info):
        """
        Format systematic value for a process in CnC datacard.

        All systematics are converted to lnN:
        - lnN type: pass through config value
        - shape type: integrate up/down in window → compute lnN
        - low-stat processes: use lowstat.json fallback values
        """
        group = syst_config.get("group", [])
        syst_type = syst_config.get("type")
        proc_check = "signal" if process == "signal" else process

        if proc_check not in group:
            return "-"

        if syst_type == "lnN":
            value = syst_config.get("value", 1.0)
            return f"{value:.3f}"

        if syst_type == "shape":
            # Check if this process has a lowstat fallback
            if lowstat_info and process in lowstat_info.get("processes", []):
                fallbacks = lowstat_info.get("fallbacks", {})
                proc_fallbacks = fallbacks.get(process, {})
                return proc_fallbacks.get(syst_name, "-")

            # Compute lnN from window integration of up/down histograms
            return self.compute_shape_lnn(process, syst_name)

        return "-"

    def generate_datacard(self, syst_config, lowstat_info):
        """Generate the full counting-experiment datacard string."""
        processes = ["signal"] + self.backgrounds
        nproc = len(processes)
        nbkg = len(self.backgrounds)

        lines = []

        # Header
        lines += [
            f"# Cut-and-Count datacard for cross-checking binned template limits",
            f"# Era: {self.era}, Channel: {self.channel}, Signal: {self.signal}",
            f"# Method: {self.method}, Binning: {self.binning}",
            f"# Mass window: [{self.mass_min:.2f}, {self.mass_max:.2f}] GeV (±3σ_voigt)",
            "",
            f"imax 1",
            f"jmax {nbkg}",
            f"kmax *",
            "-" * 80,
        ]

        # Observation
        lines += [
            f"bin          {self.channel}",
            f"observation  {self.yields['data_obs']:.4f}",
            "-" * 80,
        ]

        # Rates
        bin_line = "bin          " + "".join(f"{self.channel:<15}" for _ in processes)
        proc_line = "process      " + "".join(f"{p:<15}" for p in processes)
        idx_line = "process      " + "".join(f"{i:<15}" for i in range(nproc))
        rate_line = "rate         " + "".join(f"{self.yields[p]:<15.4f}" for p in processes)
        lines += [bin_line, proc_line, idx_line, rate_line, "-" * 80]

        # Systematics
        for syst_name, syst_cfg in syst_config.items():
            group = syst_cfg.get("group", [])
            # Check relevance
            relevant = any(
                ("signal" if p == "signal" else p) in group
                for p in processes
            )
            if not relevant:
                logging.debug(f"Skipping {syst_name}: no relevant processes")
                continue

            values = [self.format_syst_value(p, syst_name, syst_cfg, lowstat_info)
                      for p in processes]

            if all(v == "-" for v in values):
                logging.debug(f"Skipping {syst_name}: all '-'")
                continue

            syst_line = f"{syst_name:<50} {'lnN':<8}" + "".join(f"{v:<15}" for v in values)
            lines.append(syst_line)

        return "\n".join(lines) + "\n"

    def close(self):
        if self.rtfile:
            self.rtfile.Close()


def main():
    logging.info(f"Generating CnC datacard for {args.masspoint}")
    logging.info(f"  Era: {args.era}, Channel: {args.channel}")
    logging.info(f"  Method: {args.method}, Binning: {args.binning}")
    logging.info(f"  Template dir: {TEMPLATE_DIR}")

    # Load fit parameters and compute window
    fit_params = load_signal_fit()
    mass_min, mass_max = compute_window(fit_params, nsigma=args.nsigma)

    # Load configs
    syst_config = load_systematics_config(args.era, args.channel)
    logging.info(f"Loaded {len(syst_config)} systematics")

    lowstat_info = load_lowstat_json()
    if lowstat_info:
        logging.info(f"Loaded lowstat.json: {lowstat_info.get('processes', [])}")

    # Build manager and generate datacard
    manager = CnCDatacardManager(
        args.era, args.channel, args.masspoint, args.method, args.binning,
        mass_min, mass_max
    )
    datacard = manager.generate_datacard(syst_config, lowstat_info)

    # Determine output path
    nsigma_tag = f"{args.nsigma:g}sigma"
    output_path = args.output if args.output else f"{TEMPLATE_DIR}/datacard_cnc_{nsigma_tag}.txt"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(datacard)
    logging.info(f"CnC datacard saved to: {output_path}")

    # Summary
    logging.info("=" * 60)
    logging.info("Yields in ±3σ window:")
    logging.info(f"  signal:   {manager.yields['signal']:.4f}")
    logging.info(f"  data_obs: {manager.yields['data_obs']:.4f}")
    for bkg in manager.backgrounds:
        logging.info(f"  {bkg}: {manager.yields[bkg]:.4f}")

    manager.close()


if __name__ == "__main__":
    main()
