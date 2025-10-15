#!/usr/bin/env python3
import os
import ROOT
import argparse
import shutil
import sys
import json

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel")
parser.add_argument("--masspoint", required=True, type=str, help="masspoint")
parser.add_argument("--method", required=True, type=str, help="Baseline / ParticleNet / GBDT")
parser.add_argument("--output", type=str, default=None, help="Output datacard path (default: auto-determined)")
args = parser.parse_args()


def get_era_suffix(era):
    """Get era suffix for systematic naming in datacard"""
    era_map = {
        "2016preVFP": "16a", "2016postVFP": "16b",
        "2017": "17", "2018": "18",
        "2022": "22", "2022EE": "22EE",
        "2023": "23", "2023BPix": "23BPix"
    }
    if era not in era_map:
        raise ValueError(f"Unknown era: {era}")
    return era_map[era]


def load_datacard_systematics(era, channel):
    """Load systematic configuration for datacard generation"""
    config_path = "configs/systematics.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Systematic configuration not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)

    run_period = "Run2" if era in ["2016preVFP", "2016postVFP", "2017", "2018"] else "Run3"

    if run_period not in config:
        raise ValueError(f"Run period {run_period} not found in systematics config")
    if channel not in config[run_period]:
        raise ValueError(f"Channel {channel} not found in systematics config for {run_period}")

    return config[run_period][channel]


class DatacardManager():
    def __init__(self, era, channel, masspoint, method, backgrounds):
        self.era = era
        self.channel = channel
        self.signal = masspoint
        self.method = method
        self.rtfile = None
        self.backgrounds = []
        self.systematics_with_era = []

        # Open shapes file
        rtfile_path = f"templates/{era}/{channel}/{masspoint}/Shape/{method}/shapes.root"
        if not os.path.exists(rtfile_path):
            raise FileNotFoundError(f"Template file not found: {rtfile_path}")

        self.rtfile = ROOT.TFile.Open(rtfile_path)

        # Check which backgrounds are present with positive yields
        for bkg in backgrounds:
            if self.get_event_rate(bkg) > 0:
                self.backgrounds.append(bkg)

        if len(self.backgrounds) == 0:
            raise ValueError("No backgrounds with positive yields found!")

    def _get_hist_name(self, process, syst="Central"):
        """Get histogram name for process and systematic"""
        base = self.signal if process == "signal" else process
        return base if syst == "Central" else f"{base}_{syst}"

    def _format_col(self, text):
        """Format column with consistent tab spacing"""
        return f"{text}\t\t" if len(str(text)) < 8 else f"{text}\t"

    def get_event_rate(self, process, syst="Central"):
        """Get event rate (integral) for a process"""
        if process == "data_obs":
            h = self.rtfile.Get("data_obs")
            if not h:
                raise ValueError("data_obs histogram not found")
            return h.Integral()

        hist_name = self._get_hist_name(process, syst)
        h = self.rtfile.Get(hist_name)
        if not h:
            return 0.0

        return h.Integral()

    def part1string(self):
        """Generate part 1 of datacard: header and shapes"""
        lines = [
            f"imax\t\t\t1 number of bins",
            f"jmax\t\t\t{len(self.backgrounds)} number of backgrounds",
            f"kmax\t\t\t* number of nuisance parameters",
            "-" * 80,
            "shapes\t*\t*\tshapes.root\t$PROCESS\t$PROCESS_$SYSTEMATIC",
            f"shapes\tsignal\t*\tshapes.root\t{self.signal}\t{self.signal}_$SYSTEMATIC",
            "-" * 80
        ]
        return "\n".join(lines)

    def part2string(self):
        """Generate part 2: observation"""
        observation = self.get_event_rate("data_obs")
        lines = [
            "bin\t\t\tsignal_region",
            f"observation\t\t{observation:.4f}",
            "-" * 80
        ]
        return "\n".join(lines)

    def part3string(self):
        """Generate part 3: process rates"""
        nproc = len(self.backgrounds) + 1

        # Build process name line
        proc_names = "process\t\t\tsignal\t\t"
        for bkg in self.backgrounds:
            proc_names += self._format_col(bkg)

        # Build process index line
        proc_indices = "process\t\t\t0\t\t"
        for idx in range(1, nproc):
            proc_indices += f"{idx}\t\t"

        lines = [
            "bin\t\t\t" + "signal_region\t" * nproc,
            proc_names,
            proc_indices,
            "rate\t\t\t" + "-1\t\t" * nproc,
            "-" * 80
        ]
        return "\n".join(lines)

    def autoMCstring(self, threshold):
        """Generate autoMCStats line"""
        return f"signal_region\tautoMCStats\t{threshold}"

    def check_systematic_validity(self, syst_name, applies_to):
        """Check if systematic variations produce negative normalizations"""
        has_negative = False
        details = []

        for proc in applies_to:
            if proc not in ["signal"] + self.backgrounds:
                continue

            # Check up variation
            rate_up = self.get_event_rate(proc, f"{syst_name}Up")
            rate_central = self.get_event_rate(proc, "Central")

            if rate_up <= 0 and rate_central > 0:
                has_negative = True
                details.append(f"{proc} Up: {rate_up:.4e}")

            # Check down variation
            rate_down = self.get_event_rate(proc, f"{syst_name}Down")
            if rate_down <= 0 and rate_central > 0:
                has_negative = True
                details.append(f"{proc} Down: {rate_down:.4e}")

        return has_negative, details

    def syststring(self, syst, sysType, value=None, skip=None, denoteEra=False):
        """Generate systematic line with automatic shape->lnN conversion for negative rates"""
        if skip is None:
            skip = []

        # Check if systematic applies to any process
        if syst == "Nonprompt" and "nonprompt" not in self.backgrounds:
            return ""
        if syst == "Conversion" and "conversion" not in self.backgrounds:
            return ""

        # Apply era suffix
        alias = f"{syst}_{get_era_suffix(self.era)}" if denoteEra else syst
        if denoteEra and syst not in self.systematics_with_era:
            self.systematics_with_era.append(syst)

        # For shape systematics, check for negative normalizations
        original_sysType = sysType
        if sysType == "shape":
            applies_to = [p for p in ["signal"] + self.backgrounds if p not in skip]
            has_negative, details = self.check_systematic_validity(syst, applies_to)

            if has_negative:
                print(f"WARNING: Negative normalization detected for {syst}, switching to lnN", file=sys.stderr)
                for detail in details:
                    print(f"  {detail}", file=sys.stderr)
                sysType = "lnN"
                # Calculate appropriate lnN values
                value = 1.0  # Default conservative value

        # Build values for each process
        processes = ["signal"] + self.backgrounds
        values = []
        for proc in processes:
            if proc in skip:
                values.append("-")
            elif sysType == "lnN":
                if original_sysType == "shape" and value == 1.0:
                    # Calculate actual variation for this process
                    rate_central = self.get_event_rate(proc, "Central")
                    if rate_central > 0:
                        rate_up = self.get_event_rate(proc, f"{syst}Up")
                        rate_down = self.get_event_rate(proc, f"{syst}Down")

                        # Use the larger variation, but ensure positive
                        var_up = abs(rate_up / rate_central - 1.0) if rate_up > 0 else 0.0
                        var_down = abs(rate_down / rate_central - 1.0) if rate_down > 0 else 0.0
                        max_var = max(var_up, var_down, 0.001)  # Minimum 0.1% variation

                        values.append(f"{1.0 + max_var:.3f}")
                    else:
                        values.append("-")
                else:
                    values.append(f"{value:.3f}" if value else "1.000")
            else:  # shape
                values.append("1")

        # Format line
        alias_col = self._format_col(alias)
        value_cols = "\t\t".join(values)
        return f"{alias_col}{sysType}\t{value_cols}\t\t"

    def generate_datacard(self, syst_config):
        """Generate complete datacard string from configuration"""
        lines = []

        # Header
        lines.append("# Datacard for charged Higgs search")
        lines.append(f"# Era: {self.era}, Channel: {self.channel}, Masspoint: {self.signal}, Method: {self.method}")
        lines.append("# Signal cross-section scaled to 5 fb")
        lines.append(self.part1string())
        lines.append(self.part2string())
        lines.append(self.part3string())
        lines.append(self.autoMCstring(threshold=10))

        # Systematics from configuration
        all_processes = ["signal", "nonprompt", "conversion", "diboson", "ttX", "others"]

        for category_name in ["experimental", "datadriven", "normalization"]:
            if category_name not in syst_config:
                continue

            for syst_name, syst_props in syst_config[category_name].items():
                syst_line = self.syststring(
                    syst=syst_name,
                    sysType=syst_props.get("type", "shape"),
                    value=syst_props.get("value"),
                    skip=[p for p in all_processes if p not in syst_props.get("applies_to", all_processes)],
                    denoteEra=(syst_props.get("correlation") == "uncorrelated")
                )
                if syst_line:
                    lines.append(syst_line)

        return "\n".join(lines) + "\n"

    def update_root_file_era_suffix(self):
        """Update ROOT file histogram names to include era suffix for tracked systematics"""
        if len(self.systematics_with_era) == 0:
            return

        # Close the current file
        self.rtfile.Close()

        # Paths
        template_dir = f"templates/{self.era}/{self.channel}/{self.signal}/Shape/{self.method}"
        shapes_path = f"{template_dir}/shapes.root"
        backup_path = f"{template_dir}/shapes_noera.root"

        # Create backup
        if os.path.exists(backup_path):
            sys.stderr.write(f"Warning: Backup already exists, overwriting: {backup_path}\n")
        shutil.copy(shapes_path, backup_path)

        # Get era suffix
        era_suffix = get_era_suffix(self.era)

        # Open files
        original = ROOT.TFile.Open(backup_path, "READ")
        updated = ROOT.TFile.Open(shapes_path, "RECREATE")

        # Update histograms
        for key in original.GetListOfKeys():
            hist_name = key.GetName()
            new_hist_name = hist_name

            # Check if this is a systematic variation histogram (no underscore before Up/Down)
            if hist_name.endswith("Up") or hist_name.endswith("Down"):
                direction = "Up" if hist_name.endswith("Up") else "Down"
                base_name = hist_name[:-len(direction)]

                # Check each tracked systematic
                matched_syst = None
                for syst in self.systematics_with_era:
                    if base_name.endswith(syst):
                        matched_syst = syst
                        break

                if matched_syst:
                    # Extract process name by removing the systematic suffix
                    process_name = base_name[:-len(matched_syst)-1]
                    new_hist_name = f"{process_name}_{matched_syst}_{era_suffix}{direction}"

            # Clone and write histogram
            h = original.Get(hist_name)
            if h:
                h_clone = h.Clone(new_hist_name)
                updated.cd()
                h_clone.Write()

        original.Close()
        updated.Close()

        # Reopen updated file for reading
        self.rtfile = ROOT.TFile.Open(shapes_path, "READ")


if __name__ == "__main__":
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"templates/{args.era}/{args.channel}/{args.masspoint}/Shape/{args.method}/datacard.txt"

    # Load systematic configuration
    try:
        syst_config = load_datacard_systematics(args.era, args.channel)
    except Exception as e:
        print(f"ERROR: Failed to load systematic configuration: {e}", file=sys.stderr)
        exit(1)

    # Create datacard manager
    backgrounds = ["nonprompt", "conversion", "diboson", "ttX", "others"]
    try:
        manager = DatacardManager(args.era, args.channel, args.masspoint, args.method, backgrounds)
    except Exception as e:
        print(f"ERROR: Failed to create DatacardManager: {e}", file=sys.stderr)
        exit(1)

    # Generate datacard
    datacard = manager.generate_datacard(syst_config)

    # Update ROOT file with era suffixes
    manager.update_root_file_era_suffix()

    # Save datacard
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(datacard)

    print(f"Datacard saved to: {output_path}", file=sys.stderr)
