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
# Backgrounds with relative statistical error above this threshold will skip shape systematics
# Relative error = sqrt(sum of bin_error^2) / integral
SHAPE_REL_ERR_THRESHOLD = 0.30  # 30%


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

    def estimate_rate_effect_from_shape(self, process, syst_name):
        """
        Estimate fractional rate effect from shape systematic histograms.

        Calculates the rate change from Up/Down variations relative to Central.
        Returns the fractional uncertainty (e.g., 0.05 for 5% effect).

        Args:
            process: Process name
            syst_name: Systematic name

        Returns:
            float: Fractional uncertainty (e.g., 0.05 for 5%), or 0.0 if not found
        """
        base = self.signal if process == "signal" else process
        central_name = base
        up_name = f"{base}_{syst_name}Up"
        down_name = f"{base}_{syst_name}Down"

        central_hist = self.rtfile.Get(central_name)
        up_hist = self.rtfile.Get(up_name)
        down_hist = self.rtfile.Get(down_name)

        if not central_hist:
            return 0.0

        central_int = central_hist.Integral()
        if central_int <= 0:
            return 0.0

        fractional_effects = []
        if up_hist:
            up_int = up_hist.Integral()
            if up_int > 0:
                fractional_effects.append(abs(up_int - central_int) / central_int)
        if down_hist:
            down_int = down_hist.Integral()
            if down_int > 0:
                fractional_effects.append(abs(down_int - central_int) / central_int)

        if fractional_effects:
            # Use the maximum fractional effect
            return max(fractional_effects)

        return 0.0

    def collect_absorbed_rate_effects(self, process, syst_config_all):
        """
        For low-stat backgrounds, collect rate effects from ALL shape systematics
        that apply to this process (excluding theory uncertainties which only apply to signal).

        Returns total fractional uncertainty to add in quadrature.

        Args:
            process: Process name (background)
            syst_config_all: Full systematics configuration dict

        Returns:
            float: sqrt(sum of squares) of all rate effects
        """
        if process == "signal":
            return 0.0

        fractional_effects_squared = []

        for syst_name, syst_config in syst_config_all.items():
            syst_type = syst_config.get("type")
            source = syst_config.get("source")
            group = syst_config.get("group", [])

            # Only consider shape systematics
            if syst_type != "shape":
                continue

            # Check if this systematic applies to this process
            if process not in group:
                continue

            # Get rate effect from this shape systematic
            rate_effect = self.estimate_rate_effect_from_shape(process, syst_name)
            if rate_effect > 0:
                fractional_effects_squared.append(rate_effect ** 2)
                logging.debug(f"  Absorbed {syst_name} rate effect: {rate_effect:.4f} ({rate_effect*100:.1f}%)")

        if fractional_effects_squared:
            total_frac = (sum(fractional_effects_squared)) ** 0.5
            logging.debug(f"  Total absorbed rate effect for {process}: {total_frac:.4f} ({total_frac*100:.1f}%)")
            return total_frac

        return 0.0

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

        For shape systematics on backgrounds with high relative error (> SHAPE_REL_ERR_THRESHOLD),
        returns "-" (shape effects are absorbed into normalization lnN).

        Args:
            process: Process name
            syst_name: Systematic name
            syst_config: Systematic configuration dict

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
            # Check if this is a low-stat background (high relative error)
            is_background = process != "signal"
            rel_err = self.process_rel_errors.get(process, float('inf'))

            if is_background and rel_err > SHAPE_REL_ERR_THRESHOLD:
                # Skip shape for low-stat background - effects absorbed into normalization lnN
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

        For shape systematics on low-stat backgrounds (rel_err > threshold):
        - Shape line shows "-" (no shape effect)
        - Rate effects are absorbed into normalization lnN systematics

        For lnN normalization systematics (CMS_B2G25013_Norm_*):
        - Low-stat backgrounds get enhanced value = sqrt(config_value^2 + absorbed_effects^2)
        - High-stat backgrounds and signal get config value as-is
        """
        lines = []
        processes = ["signal"] + self.backgrounds

        # Identify low-stat backgrounds (high relative error)
        lowstat_backgrounds = [
            proc for proc in self.backgrounds
            if self.process_rel_errors.get(proc, float('inf')) > SHAPE_REL_ERR_THRESHOLD
        ]
        if lowstat_backgrounds:
            logging.info(f"Low-stat backgrounds (rel_err > {SHAPE_REL_ERR_THRESHOLD*100:.0f}%):")
            for proc in lowstat_backgrounds:
                logging.info(f"  {proc}: {self.process_rel_errors[proc]*100:.1f}%")
            logging.info("Shape effects will be absorbed into normalization lnN for these backgrounds")

        # Pre-compute absorbed rate effects for low-stat backgrounds
        absorbed_effects = {}
        for proc in lowstat_backgrounds:
            absorbed_effects[proc] = self.collect_absorbed_rate_effects(proc, syst_config_all)
            if absorbed_effects[proc] > 0:
                logging.info(f"  {proc}: absorbed rate effect = {absorbed_effects[proc]:.4f} "
                            f"({absorbed_effects[proc]*100:.1f}%)")

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

            # For shape systematics
            if syst_type == "shape":
                # Build shape line (signal + high-stat backgrounds get shape, low-stat get "-")
                shape_values = []
                for proc in processes:
                    val = self.format_syst_value(proc, syst_name, syst_config)
                    shape_values.append(val)

                # Only add shape line if not all values are "-"
                if not all(v == "-" for v in shape_values):
                    syst_line = f"{syst_name:<50} {'shape':<8}"
                    for val in shape_values:
                        syst_line += f"{val:<15}"
                    lines.append(syst_line)

            else:
                # Non-shape systematics (lnN)
                # Check if this is a normalization systematic for a low-stat background
                is_norm_syst = syst_name.startswith("CMS_B2G25013_Norm_")

                values = []
                for proc in processes:
                    proc_check = "signal" if proc == "signal" else proc

                    # Check if this systematic applies to this process
                    if proc_check not in group:
                        values.append("-")
                        continue

                    base_value = syst_config.get("value", 1.0)

                    # For normalization systematics on low-stat backgrounds,
                    # combine base value with absorbed shape effects
                    if is_norm_syst and proc in lowstat_backgrounds and absorbed_effects.get(proc, 0) > 0:
                        # Convert lnN value to fractional uncertainty
                        # lnN value of 1.10 means 10% uncertainty
                        base_frac = base_value - 1.0
                        absorbed_frac = absorbed_effects[proc]

                        # Combine in quadrature
                        combined_frac = (base_frac**2 + absorbed_frac**2) ** 0.5
                        combined_value = 1.0 + combined_frac

                        logging.debug(f"  {syst_name} for {proc}: base={base_value:.3f}, "
                                     f"absorbed={absorbed_frac:.4f}, combined={combined_value:.3f}")
                        values.append(f"{combined_value:.3f}")
                    else:
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
