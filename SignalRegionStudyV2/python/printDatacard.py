#!/usr/bin/env python3
"""
Generate HiggsCombine datacards from shape templates.

This script reads the shapes.root file and systematics configuration to produce
a properly formatted datacard for limit extraction.

Usage:
    printDatacard.py --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90 --method Baseline --binning uniform
"""
import os
import sys
import json
import logging
import argparse
import ROOT

# Argument parsing
parser = argparse.ArgumentParser(description="Generate HiggsCombine datacard from templates")
parser.add_argument("--era", required=True, type=str, help="Data-taking period (2016preVFP, 2017, 2018, 2022, etc.)")
parser.add_argument("--channel", required=True, type=str, help="Analysis channel (SR1E2Mu, SR3Mu)")
parser.add_argument("--masspoint", required=True, type=str, help="Signal mass point (e.g., MHc130_MA90)")
parser.add_argument("--method", required=True, type=str, help="Template method (Baseline, ParticleNet)")
parser.add_argument("--binning", default="uniform", choices=["uniform", "extended"],
                    help="Binning method: 'uniform' (15 bins, default) or 'extended' (15 bins + tails)")
parser.add_argument("--unblind", action="store_true",
                    help="Generate datacard from unblind run")
parser.add_argument("--partial-unblind", action="store_true", dest="partial_unblind",
                    help="Generate datacard from partial-unblind run")
parser.add_argument("--output", type=str, default=None, help="Output datacard path (default: auto-determined)")
parser.add_argument("--debug", action="store_true", help="Enable debug logging")
args = parser.parse_args()

# Validate unblind options
if args.unblind and args.partial_unblind:
    raise ValueError("--unblind and --partial-unblind are mutually exclusive")

# Setup logging
logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                    format='%(levelname)s - %(message)s')

# Path setup
WORKDIR = os.getenv("WORKDIR")
if not WORKDIR:
    raise EnvironmentError("WORKDIR environment variable not set. Please run 'source setup.sh'")

# Template directory
binning_suffix = args.binning
if args.unblind:
    binning_suffix = f"{args.binning}_unblind"
elif args.partial_unblind:
    binning_suffix = f"{args.binning}_partial_unblind"
TEMPLATE_DIR = f"{WORKDIR}/SignalRegionStudyV2/templates/{args.era}/{args.channel}/{args.masspoint}/{args.method}/{binning_suffix}"

# Setup ROOT
ROOT.gROOT.SetBatch(True)

# Threshold for using shape vs lnN systematics
# Backgrounds with relative statistical error above this threshold will use lnN fallback
# Relative error = sqrt(sum of bin_error^2) / integral
SHAPE_REL_ERR_THRESHOLD = 0.30  # 30%

# Constants for lnN fallback computation
MAX_LNN_VALUE = 2.0           # Cap lnN fallback (100% uncertainty max)
MIN_YIELD_THRESHOLD = 1e-6    # Below this yield, skip systematic entirely ("-")


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
    """Load the process list from template directory."""
    process_list_path = f"{TEMPLATE_DIR}/process_list.json"
    if not os.path.exists(process_list_path):
        raise FileNotFoundError(f"Process list not found: {process_list_path}")

    with open(process_list_path) as f:
        return json.load(f)


class DatacardManager:
    """Manages datacard generation from ROOT templates."""

    def __init__(self, era, channel, masspoint, method, binning):
        self.era = era
        self.channel = channel
        self.signal = masspoint
        self.method = method
        self.binning = binning
        self.backgrounds = []
        self.rtfile = None

        # Open shapes file
        shapes_path = f"{TEMPLATE_DIR}/shapes.root"
        if not os.path.exists(shapes_path):
            raise FileNotFoundError(f"Template file not found: {shapes_path}")

        self.rtfile = ROOT.TFile.Open(shapes_path, "READ")

        # Cache for process yields and relative errors
        self.process_yields = {}
        self.process_rel_errors = {}
        # Cache for lnN fallback values: (process, syst_name) -> value string
        self._lnn_fallback_cache = {}

        # Load process list
        process_config = load_process_list()
        separate_processes = process_config.get("separate_processes", [])

        # Check which backgrounds are present with positive yields
        # Always check "others" as it's the catch-all
        all_backgrounds = separate_processes + ["others"]

        for bkg in all_backgrounds:
            if bkg not in self.backgrounds:  # Avoid duplicates
                rate = self.get_event_rate(bkg)
                if rate > 0:
                    self.backgrounds.append(bkg)
                    self.process_yields[bkg] = rate
                    self.process_rel_errors[bkg] = self.get_relative_error(bkg)
                    logging.debug(f"Background '{bkg}' has rate {rate:.4f}, rel_err {self.process_rel_errors[bkg]*100:.1f}%")
                else:
                    logging.warning(f"Background '{bkg}' has non-positive yield ({rate:.4f}), dropping from datacard")

        # Cache signal yield and relative error
        self.process_yields["signal"] = self.get_event_rate("signal")
        self.process_rel_errors["signal"] = self.get_relative_error("signal")

        if len(self.backgrounds) == 0:
            raise ValueError("No backgrounds with positive yields found!")

        logging.info(f"Active backgrounds: {self.backgrounds}")

    def get_event_rate(self, process):
        """Get event rate (integral) for a process."""
        if process == "data_obs":
            hist = self.rtfile.Get("data_obs")
        elif process == "signal":
            hist = self.rtfile.Get(self.signal)
        else:
            hist = self.rtfile.Get(process)

        if not hist:
            return 0.0

        return hist.Integral()

    def get_relative_error(self, process):
        """Get relative statistical error for a process: sqrt(sum of bin_error^2) / integral."""
        if process == "data_obs":
            hist = self.rtfile.Get("data_obs")
        elif process == "signal":
            hist = self.rtfile.Get(self.signal)
        else:
            hist = self.rtfile.Get(process)

        if not hist:
            return float('inf')

        integral = hist.Integral()
        if integral <= 0:
            return float('inf')

        sum_err2 = 0.0
        for i in range(1, hist.GetNbinsX() + 1):
            sum_err2 += hist.GetBinError(i) ** 2

        return (sum_err2 ** 0.5) / integral

    def get_hist_name(self, process, syst="Central"):
        """Get histogram name for process and systematic."""
        base = self.signal if process == "signal" else process
        return base if syst == "Central" else f"{base}_{syst}"

    def check_histogram_exists(self, hist_name):
        """Check if histogram exists in ROOT file."""
        hist = self.rtfile.Get(hist_name)
        return hist is not None

    def compute_lnn_fallback_value(self, process, syst_name):
        """
        Compute per-systematic lnN fallback value for a low-stat process.

        Uses Up/Down histogram integrals to determine the rate effect,
        then returns a formatted lnN value string.

        Args:
            process: Process name
            syst_name: Systematic name

        Returns:
            str: Formatted lnN value (e.g., "1.050") or "-" if negligible
        """
        base = self.signal if process == "signal" else process
        central_hist = self.rtfile.Get(base)
        up_hist = self.rtfile.Get(f"{base}_{syst_name}Up")
        down_hist = self.rtfile.Get(f"{base}_{syst_name}Down")

        if not central_hist:
            return "-"

        central_int = central_hist.Integral()
        if central_int < MIN_YIELD_THRESHOLD:
            return "-"

        fractional_effects = []
        if up_hist:
            up_int = up_hist.Integral()
            fractional_effects.append(abs(up_int - central_int) / central_int)
        if down_hist:
            down_int = down_hist.Integral()
            fractional_effects.append(abs(down_int - central_int) / central_int)

        if not fractional_effects:
            return "-"

        rate_effect = max(fractional_effects)

        # Skip if effect is negligible (< 0.1%)
        if rate_effect < 0.001:
            return "-"

        lnn_value = 1.0 + rate_effect
        if lnn_value > MAX_LNN_VALUE:
            logging.warning(f"Capping lnN fallback for {process}/{syst_name}: "
                          f"{lnn_value:.3f} -> {MAX_LNN_VALUE:.3f}")
            lnn_value = MAX_LNN_VALUE

        return f"{lnn_value:.3f}"

    def precompute_lnn_fallbacks(self, syst_config_all):
        """
        Pre-compute lnN fallback values for all (low-stat process, shape systematic) pairs.

        Must be called while Up/Down histograms still exist in shapes.root.

        Args:
            syst_config_all: Full systematics configuration dict
        """
        lowstat_backgrounds = [
            proc for proc in self.backgrounds
            if self.process_rel_errors.get(proc, float('inf')) > SHAPE_REL_ERR_THRESHOLD
        ]

        if not lowstat_backgrounds:
            return

        logging.info(f"Low-stat backgrounds (rel_err > {SHAPE_REL_ERR_THRESHOLD*100:.0f}%):")
        for proc in lowstat_backgrounds:
            logging.info(f"  {proc}: rel_err={self.process_rel_errors[proc]*100:.1f}%, "
                        f"yield={self.process_yields[proc]:.6f}")

        for syst_name, syst_config in syst_config_all.items():
            if syst_config.get("type") != "shape":
                continue

            group = syst_config.get("group", [])
            for proc in lowstat_backgrounds:
                if proc not in group:
                    continue

                fallback = self.compute_lnn_fallback_value(proc, syst_name)
                self._lnn_fallback_cache[(proc, syst_name)] = fallback
                if fallback != "-":
                    logging.debug(f"  lnN fallback {proc}/{syst_name}: {fallback}")

        logging.info(f"Pre-computed {len(self._lnn_fallback_cache)} lnN fallback values")

    def rewrite_shapes_root(self, syst_config_all):
        """
        Rewrite shapes.root removing Up/Down histograms for low-stat processes.

        The original shapes.root is preserved as shapes_original.root.
        A new shapes.root is written with low-stat shape histograms removed,
        so Combine's shape? directive falls back to lnN for these processes.

        Args:
            syst_config_all: Full systematics configuration dict
        """
        lowstat_backgrounds = [
            proc for proc in self.backgrounds
            if self.process_rel_errors.get(proc, float('inf')) > SHAPE_REL_ERR_THRESHOLD
        ]

        if not lowstat_backgrounds:
            return

        # Collect histogram names to remove
        hists_to_remove = set()
        for syst_name, syst_config in syst_config_all.items():
            if syst_config.get("type") != "shape":
                continue
            group = syst_config.get("group", [])
            for proc in lowstat_backgrounds:
                if proc not in group:
                    continue
                hists_to_remove.add(f"{proc}_{syst_name}Up")
                hists_to_remove.add(f"{proc}_{syst_name}Down")

        if not hists_to_remove:
            return

        # Read all histograms from current file
        shapes_path = f"{TEMPLATE_DIR}/shapes.root"
        original_path = f"{TEMPLATE_DIR}/shapes_original.root"
        all_hists = {}
        for key in self.rtfile.GetListOfKeys():
            name = key.GetName()
            if name not in hists_to_remove:
                hist = key.ReadObj()
                hist.SetDirectory(0)
                all_hists[name] = hist

        # Close current file handle
        self.rtfile.Close()

        # Rename original shapes.root to shapes_original.root (handle re-run)
        if os.path.exists(original_path):
            os.remove(original_path)
        os.rename(shapes_path, original_path)
        logging.info(f"Preserved original as: {original_path}")

        # Write filtered histograms to new shapes.root
        outfile = ROOT.TFile.Open(shapes_path, "RECREATE")
        for name, hist in all_hists.items():
            hist.Write(name)
        outfile.Close()

        logging.info(f"Removed {len(hists_to_remove)} low-stat shape histograms from shapes.root")

        # Reopen for further reads
        self.rtfile = ROOT.TFile.Open(shapes_path, "READ")

    def part1_header(self):
        """Generate part 1 of datacard: header and shapes directive."""
        lines = [
            f"# Datacard for B2G-25-013",
            f"# Era: {self.era}, Channel: {self.channel}, Signal: {self.signal}",
            f"# Method: {self.method}, Binning: {self.binning}",
            "",
            f"imax 1 number of bins",
            f"jmax {len(self.backgrounds)} number of backgrounds",
            f"kmax * number of nuisance parameters",
            "-" * 80,
            f"shapes * * shapes.root $PROCESS $PROCESS_$SYSTEMATIC",
            f"shapes signal * shapes.root {self.signal} {self.signal}_$SYSTEMATIC",
            "-" * 80,
        ]
        return "\n".join(lines)

    def part2_observation(self):
        """Generate part 2: observation."""
        observation = self.get_event_rate("data_obs")
        lines = [
            f"bin          {self.channel}",
            f"observation  {observation:.4f}",
            "-" * 80,
        ]
        return "\n".join(lines)

    def part3_rates(self):
        """Generate part 3: process rates."""
        nproc = len(self.backgrounds) + 1

        # Build bin line
        bin_line = "bin          " + f"{self.channel:<15}" * nproc

        # Build process name line
        proc_names = "process      signal         "
        for bkg in self.backgrounds:
            proc_names += f"{bkg:<15}"

        # Build process index line
        proc_indices = "process      0              "
        for idx in range(1, nproc):
            proc_indices += f"{idx:<15}"

        # Build rate line (use -1 for shape analysis)
        rate_line = "rate         -1             " + "-1             " * len(self.backgrounds)

        lines = [bin_line, proc_names, proc_indices, rate_line, "-" * 80]
        return "\n".join(lines)

    def format_syst_value(self, process, syst_name, syst_config):
        """
        Format systematic value for a specific process.

        For shape systematics on low-stat backgrounds (rel_err > threshold),
        returns the pre-computed lnN fallback value (used with shape? type).

        Args:
            process: Process name
            syst_name: Systematic name
            syst_config: Systematic configuration dict

        Returns:
            str: The value to put in the datacard column ("-", "1", or lnN fallback value)
        """
        group = syst_config.get("group", [])
        source = syst_config.get("source")
        syst_type = syst_config.get("type")

        # Map "signal" in group to actual process check
        proc_check = "signal" if process == "signal" else process

        # Check if this systematic applies to this process
        if proc_check not in group:
            return "-"

        # For lnN systematics
        if syst_type == "lnN":
            value = syst_config.get("value", 1.0)
            return f"{value:.3f}"

        # For shape systematics
        if syst_type == "shape":
            # Check if this is a low-stat background (high relative error)
            is_background = process != "signal"
            rel_err = self.process_rel_errors.get(process, float('inf'))

            if is_background and rel_err > SHAPE_REL_ERR_THRESHOLD:
                # Return pre-computed lnN fallback value for shape? mechanism
                return self._lnn_fallback_cache.get((process, syst_name), "-")

            # Normal shape systematic handling
            if source == "preprocessed":
                # Check if the shape variation exists in the ROOT file
                variations = syst_config.get("variations", [])
                if len(variations) >= 2:
                    base = self.signal if process == "signal" else process
                    up_hist_name = f"{base}_{syst_name}Up"

                    if self.check_histogram_exists(up_hist_name):
                        return "1"
                    else:
                        logging.debug(f"Shape {up_hist_name} not found, skipping for {process}")
                        return "-"
            return "1"

        return "-"

    def generate_systematic_lines(self, syst_config_all):
        """
        Generate all systematic uncertainty lines.

        Uses Combine's shape? type for shape systematics:
        - Processes with shape histograms in shapes.root get full shape treatment
        - Low-stat processes (histograms removed) get lnN fallback values
        """
        lines = []
        processes = ["signal"] + self.backgrounds

        for syst_name, syst_config in syst_config_all.items():
            syst_type = syst_config.get("type")
            group = syst_config.get("group", [])

            # Check if any of our processes are in the group
            relevant_processes = []
            for proc in processes:
                proc_check = "signal" if proc == "signal" else proc
                if proc_check in group:
                    relevant_processes.append(proc)

            if not relevant_processes:
                logging.debug(f"Skipping {syst_name}: no relevant processes")
                continue

            if syst_type == "shape":
                # Build shape? line: "1" for shape processes, lnN fallback for low-stat
                shape_values = []
                for proc in processes:
                    val = self.format_syst_value(proc, syst_name, syst_config)
                    shape_values.append(val)

                # Only add line if not all values are "-"
                if not all(v == "-" for v in shape_values):
                    syst_line = f"{syst_name:<50} {'shape?':<8}"
                    for val in shape_values:
                        syst_line += f"{val:<15}"
                    lines.append(syst_line)

            else:
                # lnN systematics: pass through config values directly
                values = []
                for proc in processes:
                    proc_check = "signal" if proc == "signal" else proc
                    if proc_check not in group:
                        values.append("-")
                        continue
                    base_value = syst_config.get("value", 1.0)
                    values.append(f"{base_value:.3f}")

                if all(v == "-" for v in values):
                    logging.debug(f"Skipping {syst_name}: all processes excluded")
                    continue

                syst_line = f"{syst_name:<50} {syst_type:<8}"
                for val in values:
                    syst_line += f"{val:<15}"
                lines.append(syst_line)

        return "\n".join(lines)

    def generate_automc_line(self, threshold=10):
        """Generate autoMCStats line."""
        return f"{self.channel} autoMCStats {threshold}"

    def write_lowstat_json(self):
        """
        Write lowstat.json with metadata about low-stat processes and their lnN fallbacks.

        This file is consumed by checkTemplates.py to correctly compute systematic
        error bands using lnN fallback values instead of missing shape histograms.
        """
        lowstat_backgrounds = [
            proc for proc in self.backgrounds
            if self.process_rel_errors.get(proc, float('inf')) > SHAPE_REL_ERR_THRESHOLD
        ]

        if not lowstat_backgrounds:
            return

        # Build fallbacks dict: {process: {syst_name: value_string}}
        fallbacks = {}
        for (proc, syst_name), value in self._lnn_fallback_cache.items():
            if proc not in fallbacks:
                fallbacks[proc] = {}
            fallbacks[proc][syst_name] = value

        lowstat_info = {
            "threshold": SHAPE_REL_ERR_THRESHOLD,
            "processes": lowstat_backgrounds,
            "fallbacks": fallbacks
        }

        lowstat_path = f"{TEMPLATE_DIR}/lowstat.json"
        with open(lowstat_path, 'w') as f:
            json.dump(lowstat_info, f, indent=2)

        logging.info(f"Wrote lowstat.json: {lowstat_path}")

    def generate_datacard(self, syst_config):
        """Generate complete datacard string."""
        # Pre-compute lnN fallbacks while Up/Down histograms still exist
        self.precompute_lnn_fallbacks(syst_config)
        # Write lowstat.json with fallback metadata
        self.write_lowstat_json()
        # Remove low-stat shape histograms so shape? falls back to lnN
        self.rewrite_shapes_root(syst_config)

        parts = [
            self.part1_header(),
            self.part2_observation(),
            self.part3_rates(),
            self.generate_systematic_lines(syst_config),
            self.generate_automc_line(),
        ]
        return "\n".join(parts) + "\n"

    def close(self):
        """Close the ROOT file."""
        if self.rtfile:
            self.rtfile.Close()


def main():
    logging.info(f"Generating datacard for {args.masspoint}")
    logging.info(f"  Era: {args.era}")
    logging.info(f"  Channel: {args.channel}")
    logging.info(f"  Method: {args.method}")
    logging.info(f"  Binning: {args.binning}")
    logging.info(f"  Template dir: {TEMPLATE_DIR}")

    # Load systematics config
    try:
        syst_config = load_systematics_config(args.era, args.channel)
        logging.info(f"Loaded {len(syst_config)} systematics from config")
    except Exception as e:
        logging.error(f"Failed to load systematics config: {e}")
        sys.exit(1)

    # Create datacard manager
    try:
        manager = DatacardManager(args.era, args.channel, args.masspoint, args.method, args.binning)
    except Exception as e:
        logging.error(f"Failed to create DatacardManager: {e}")
        sys.exit(1)

    # Generate datacard
    datacard = manager.generate_datacard(syst_config)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"{TEMPLATE_DIR}/datacard.txt"

    # Save datacard
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(datacard)

    logging.info(f"Datacard saved to: {output_path}")

    # Print summary
    logging.info("=" * 60)
    logging.info("Summary:")
    logging.info(f"  Signal rate: {manager.get_event_rate('signal'):.4f}")
    logging.info(f"  Data obs: {manager.get_event_rate('data_obs'):.4f}")
    for bkg in manager.backgrounds:
        logging.info(f"  {bkg} rate: {manager.get_event_rate(bkg):.4f}")

    manager.close()


if __name__ == "__main__":
    main()
