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
# Backgrounds with fewer events than this threshold will use lnN instead of shape
# to avoid statistical artifacts from limited MC statistics
SHAPE_THRESHOLD = 10


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

        # Cache for process yields (used for shape vs lnN decision)
        self.process_yields = {}

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
                    logging.debug(f"Background '{bkg}' has rate {rate:.4f}")

        # Cache signal yield
        self.process_yields["signal"] = self.get_event_rate("signal")

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

    def get_hist_name(self, process, syst="Central"):
        """Get histogram name for process and systematic."""
        base = self.signal if process == "signal" else process
        return base if syst == "Central" else f"{base}_{syst}"

    def check_histogram_exists(self, hist_name):
        """Check if histogram exists in ROOT file."""
        hist = self.rtfile.Get(hist_name)
        return hist is not None

    def estimate_lnN_from_shape(self, process, syst_name, default=1.05):
        """
        Estimate lnN value from shape systematic histograms.

        Calculates the rate change from Up/Down variations relative to Central.
        Returns the larger of (Up/Central, Central/Down) as the lnN value.

        Args:
            process: Process name
            syst_name: Systematic name
            default: Default value if histograms not found

        Returns:
            float: lnN value (e.g., 1.05 for 5% uncertainty)
        """
        base = self.signal if process == "signal" else process
        central_name = base
        up_name = f"{base}_{syst_name}Up"
        down_name = f"{base}_{syst_name}Down"

        central_hist = self.rtfile.Get(central_name)
        up_hist = self.rtfile.Get(up_name)
        down_hist = self.rtfile.Get(down_name)

        if not central_hist:
            return default

        central_int = central_hist.Integral()
        if central_int <= 0:
            return default

        ratios = []
        if up_hist:
            up_int = up_hist.Integral()
            if up_int > 0:
                ratios.append(max(up_int / central_int, central_int / up_int))
        if down_hist:
            down_int = down_hist.Integral()
            if down_int > 0:
                ratios.append(max(down_int / central_int, central_int / down_int))

        if ratios:
            # Use the maximum ratio as the lnN value
            lnN_value = max(ratios)
            # Cap at reasonable values (1.01 to 2.0)
            lnN_value = max(1.01, min(2.0, lnN_value))
            return lnN_value

        return default

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

    def format_syst_value(self, process, syst_name, syst_config, use_lnN_for_lowstat=False):
        """
        Format systematic value for a specific process.

        For shape systematics on backgrounds with low statistics (< SHAPE_THRESHOLD),
        returns "-" to skip shape effect (use_lnN_for_lowstat=False) or the lnN value
        will be handled separately (use_lnN_for_lowstat=True).

        Args:
            process: Process name
            syst_name: Systematic name
            syst_config: Systematic configuration dict
            use_lnN_for_lowstat: If True, return lnN value for low-stat backgrounds

        Returns:
            str: The value to put in the datacard column ("-", "1", or numeric value)
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
            # Check if this is a low-stat background that should use lnN instead
            is_background = process != "signal"
            process_yield = self.process_yields.get(process, 0)

            if is_background and process_yield < SHAPE_THRESHOLD:
                if use_lnN_for_lowstat:
                    # Return lnN value estimated from shape histograms
                    lnN_value = self.estimate_lnN_from_shape(process, syst_name)
                    logging.debug(f"Low-stat background '{process}' ({process_yield:.2f} events): "
                                  f"using lnN={lnN_value:.3f} for {syst_name}")
                    return f"{lnN_value:.3f}"
                else:
                    # Skip shape for this low-stat background
                    return "-"

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

        For shape systematics, creates two lines if needed:
        1. Shape line for signal and high-stat backgrounds (>= SHAPE_THRESHOLD events)
        2. lnN line for low-stat backgrounds (< SHAPE_THRESHOLD events)
        """
        lines = []
        processes = ["signal"] + self.backgrounds

        # Identify low-stat backgrounds once
        lowstat_backgrounds = [
            proc for proc in self.backgrounds
            if self.process_yields.get(proc, 0) < SHAPE_THRESHOLD
        ]
        if lowstat_backgrounds:
            logging.info(f"Low-stat backgrounds (< {SHAPE_THRESHOLD} events): {lowstat_backgrounds}")
            logging.info("These will use lnN instead of shape for applicable systematics")

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

            # For shape systematics, we may need to split into shape + lnN lines
            if syst_type == "shape":
                # Check if any low-stat backgrounds are affected
                affected_lowstat = [
                    proc for proc in lowstat_backgrounds
                    if proc in group
                ]

                # Build shape line (signal + high-stat backgrounds)
                shape_values = []
                for proc in processes:
                    val = self.format_syst_value(proc, syst_name, syst_config, use_lnN_for_lowstat=False)
                    shape_values.append(val)

                # Only add shape line if not all values are "-"
                if not all(v == "-" for v in shape_values):
                    syst_line = f"{syst_name:<50} {'shape':<8}"
                    for val in shape_values:
                        syst_line += f"{val:<15}"
                    lines.append(syst_line)

                # Build lnN line for low-stat backgrounds if any are affected
                if affected_lowstat:
                    lnN_values = []
                    for proc in processes:
                        if proc in affected_lowstat:
                            val = self.format_syst_value(proc, syst_name, syst_config, use_lnN_for_lowstat=True)
                        else:
                            val = "-"
                        lnN_values.append(val)

                    # Only add lnN line if not all values are "-"
                    if not all(v == "-" for v in lnN_values):
                        lnN_syst_name = f"{syst_name}_lowstat"
                        syst_line = f"{lnN_syst_name:<50} {'lnN':<8}"
                        for val in lnN_values:
                            syst_line += f"{val:<15}"
                        lines.append(syst_line)
                        logging.debug(f"Added lnN systematic '{lnN_syst_name}' for low-stat backgrounds")

            else:
                # Non-shape systematics (lnN) - standard handling
                values = []
                for proc in processes:
                    val = self.format_syst_value(proc, syst_name, syst_config)
                    values.append(val)

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

    def generate_datacard(self, syst_config):
        """Generate complete datacard string."""
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
