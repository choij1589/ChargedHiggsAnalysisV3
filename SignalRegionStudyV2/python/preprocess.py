#!/usr/bin/env python3
"""
Preprocess signal and background samples for SignalRegionStudyV2.

This script preprocesses ROOT files with systematic variations,
loading era-specific systematics configuration from configs/systematics.{era}.json.

Processes:
- Signal: from RunSyst_RunTheoryUnc (with theory uncertainties); requires real MC for the requested era
- Backgrounds: from RunSyst (WZ, ZZ, ttW, ttZ, etc.)
- Nonprompt: from MatrixAnalyzer (data-driven)
- Data: from PromptAnalyzer (Central only)

Channels:
- SR1E2Mu: Signal region with 1 electron + 2 muons
- SR3Mu: Signal region with 3 muons
- TTZ2E1Mu: TTZ control region with 2 electrons + 1 muon (no signal, for ParticleNet validation)

Usage:
    python preprocess.py --era 2018   --channel SR1E2Mu  --masspoint MHc130_MA90
    python preprocess.py --era 2022EE --channel SR3Mu    --masspoint MHc130_MA90
    python preprocess.py --era 2018   --channel TTZ2E1Mu --masspoint MHc130_MA90
"""
import os
import argparse
import logging
import json
import subprocess
from array import array
import ROOT

from template_utils import (
    parse_variations,
    iter_shape_variations,
    calculate_weight_scale,
    ensure_directory,
    categorize_systematics,
)


# =============================================================================
# ParticleNet trained mass points (loaded from configs/masspoints.json)
# =============================================================================
_MASSPOINTS_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "masspoints.json")
with open(_MASSPOINTS_JSON) as _f:
    _MASSPOINTS_CONFIG = json.load(_f)
PN_TRAINED_MASSPOINTS = set(_MASSPOINTS_CONFIG["particlenet"])


# =============================================================================
# Channel Mappings
# =============================================================================

# Channel -> input channel mapping (SKNanoOutput directory name)
CHANNEL_INPUT_MAP = {
    "SR1E2Mu": "Run1E2Mu",
    "SR3Mu": "Run3Mu",
    "TTZ2E1Mu": "Run2E1Mu",
}

# Channel -> config channel mapping (for samplegroups/systematics lookup)
CHANNEL_CONFIG_MAP = {
    "SR1E2Mu": "SR1E2Mu",
    "SR3Mu": "SR3Mu",
    "TTZ2E1Mu": "SR1E2Mu",  # Reuse SR1E2Mu config (has electron systematics)
}

# Channels that have signal samples
CHANNELS_WITH_SIGNAL = {"SR1E2Mu", "SR3Mu"}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess samples for SignalRegionStudyV2")
    parser.add_argument("--era", required=True, type=str, help="era (e.g., 2018, 2022EE)")
    parser.add_argument("--channel", required=True, type=str,
                        choices=list(CHANNEL_INPUT_MAP.keys()),
                        help="channel (SR1E2Mu, SR3Mu, or TTZ2E1Mu)")
    parser.add_argument("--masspoint", required=True, type=str, help="signal mass point (e.g., MHc130_MA90)")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    return parser.parse_args()


def is_run3_era(era):
    """Check if era is a Run3 era."""
    return era in ["2022", "2022EE", "2023", "2023BPix"]


def load_config(workdir, era, channel):
    """Load systematics and sample group configurations."""
    # Map to config channel
    config_channel = CHANNEL_CONFIG_MAP.get(channel, channel)

    # Load systematics config
    config_path = f"{workdir}/SignalRegionStudyV2/configs/systematics.{era}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Systematics config not found: {config_path}")

    with open(config_path) as f:
        json_systematics = json.load(f)

    if config_channel not in json_systematics:
        raise ValueError(f"Channel '{config_channel}' not found in {config_path}")

    # Load sample groups config
    samplegroups_path = f"{workdir}/SignalRegionStudyV2/configs/samplegroups.json"
    if not os.path.exists(samplegroups_path):
        raise FileNotFoundError(f"Sample groups config not found: {samplegroups_path}")

    with open(samplegroups_path) as f:
        json_samplegroups = json.load(f)

    if era not in json_samplegroups:
        raise ValueError(f"Era '{era}' not found in {samplegroups_path}")
    if config_channel not in json_samplegroups[era]:
        raise ValueError(f"Channel '{config_channel}' not found for era '{era}'")

    return {
        'systematics': json_systematics[config_channel],
        'samples': json_samplegroups[era][config_channel],
        'aliases': json_samplegroups.get("aliases", {})
    }


def load_convSF(workdir, era, channel):
    """Load conversion scale factor from Common/Data/ConvSF.json."""
    config_channel = CHANNEL_CONFIG_MAP.get(channel, channel)
    channel_key = config_channel.replace("SR", "")  # SR1E2Mu → 1E2Mu, SR3Mu → 3Mu

    convSF_file = f"{workdir}/Common/Data/ConvSF.json"
    with open(convSF_file) as f:
        data = json.load(f)

    central_sf = data[channel_key][era]["central"]
    logging.info(f"Loaded ConvSF for {era} {channel}: {central_sf:.6f}")
    return central_sf


def load_kfactors(workdir, era):
    """
    Load K-factors from Common/Data/KFactors.json.

    Returns a dict mapping sample names to their K-factors.
    Only applies K-factors to samples with exact matching names in the JSON.
    """
    kfactor_path = f"{workdir}/Common/Data/KFactors.json"
    if not os.path.exists(kfactor_path):
        logging.warning(f"KFactors file not found: {kfactor_path}")
        return {}

    with open(kfactor_path) as f:
        kfactor_data = json.load(f)

    # Determine run period
    run_period = "Run2" if era in ["2016preVFP", "2016postVFP", "2017", "2018"] else "Run3"
    kfactors_for_period = kfactor_data.get(run_period, {})

    if not kfactors_for_period:
        logging.warning(f"No K-factors defined for {run_period}")
        return {}

    # Extract K-factors with exact sample name matching
    sample_to_kfactor = {}
    for sample_name, kfactor_info in kfactors_for_period.items():
        kfactor_value = kfactor_info["kFactor"]
        sample_to_kfactor[sample_name] = kfactor_value
        logging.debug(f"K-factor for {sample_name}: {kfactor_value}")

    logging.info(f"Loaded {len(sample_to_kfactor)} K-factors for {run_period}")
    return sample_to_kfactor


def hadd_files(output_path, input_files, cleanup=True):
    """Merge ROOT files using hadd."""
    existing_files = [f for f in input_files if os.path.exists(f)]
    if not existing_files:
        logging.warning(f"No files to merge for {output_path}")
        return False

    result = subprocess.run(["hadd", "-f", output_path] + existing_files, capture_output=True, text=True)
    if result.returncode != 0:
        logging.error(f"hadd failed: {result.stderr}")
        return False

    if cleanup:
        for f in existing_files:
            os.remove(f)

    logging.info(f"Merged {len(existing_files)} files -> {output_path}")
    return True


# =============================================================================
# Base Preprocessor Class
# =============================================================================

class BasePreprocessor:
    """Base class with shared file/branch operations."""

    def __init__(self, era, channel, masspoint):
        self.era = era
        self.channel = channel
        self.masspoint = masspoint
        self.in_file = None
        self.out_file = None

        # Parse mass point parameters
        self.mHc = int(masspoint.split("_")[0].replace("MHc", ""))
        self.mA = int(masspoint.split("_")[1].replace("MA", ""))
        self.is_trained_sample = masspoint in PN_TRAINED_MASSPOINTS

    def set_input_file(self, path):
        """Open input ROOT file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")
        self.in_file = ROOT.TFile(path, "READ")
        if self.in_file.IsZombie():
            raise IOError(f"Failed to open input file: {path}")

    def set_output_file(self, path):
        """Open output ROOT file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.out_file = ROOT.TFile(path, "RECREATE")

    def close_files(self):
        """Close input and output files."""
        if self.in_file:
            self.in_file.Close()
            self.in_file = None
        if self.out_file:
            self.out_file.Close()
            self.out_file = None

    def _setup_output_branches(self, out_tree):
        """Setup output branches and return (out_vars, score_vars) arrays."""
        out_vars = {name: array('d', [0.0]) for name in ['mass', 'mass1', 'mass2', 'MT1', 'MT2', 'weight']}
        score_vars = {}
        if self.is_trained_sample:
            for suffix in ['signal', 'nonprompt', 'diboson', 'ttZ']:
                score_vars[suffix] = array('d', [0.0])

        for name, arr in out_vars.items():
            out_tree.Branch(name, arr, f"{name}/D")
        for suffix, arr in score_vars.items():
            out_tree.Branch(f"score_{self.masspoint}_{suffix}", arr, f"score_{self.masspoint}_{suffix}/D")

        return out_vars, score_vars

    def _setup_input_branches(self, in_tree, include_mass=False):
        """Setup input branch addresses and return (in_vars, in_scores) arrays."""
        var_names = ['mass', 'mass1', 'mass2', 'MT1', 'MT2', 'weight'] if include_mass else ['mass1', 'mass2', 'MT1', 'MT2', 'weight']
        in_vars = {name: array('d', [0.0]) for name in var_names}
        in_scores = {}
        if self.is_trained_sample:
            for suffix in ['signal', 'nonprompt', 'diboson', 'ttZ']:
                in_scores[suffix] = array('d', [0.0])

        for name, arr in in_vars.items():
            in_tree.SetBranchAddress(name, arr)
        for suffix, arr in in_scores.items():
            in_tree.SetBranchAddress(f"score_{self.masspoint}_{suffix}", arr)

        return in_vars, in_scores

    def _select_mass(self, mass1, mass2):
        """Select appropriate mass based on channel and mass point."""
        if "1E2Mu" in self.channel or "2E1Mu" in self.channel:
            return mass1
        elif "3Mu" in self.channel:
            if self.mHc >= 100 and self.mA >= 60:
                return max(mass1, mass2)
            return min(mass1, mass2)
        else:
            raise ValueError(f"Unknown channel: {self.channel}")


class SamplePreprocessor(BasePreprocessor):
    """Preprocessor for samples with systematic variations."""

    def __init__(self, era, channel, masspoint, convSF=1.0):
        super().__init__(era, channel, masspoint)
        self.convSF = convSF

    def process_tree(self, input_tree_name, output_tree_name, weight_scale=1.0,
                     is_signal=False, apply_convSF=False, kfactor=1.0):
        """Process a single tree from input to output."""
        in_tree = self.in_file.Get(input_tree_name)
        if not in_tree:
            raise RuntimeError(f"Tree '{input_tree_name}' not found in input file")

        out_tree = ROOT.TTree(output_tree_name, "")

        out_vars, score_vars = self._setup_output_branches(out_tree)
        in_vars, in_scores = self._setup_input_branches(in_tree, include_mass=False)

        for i in range(in_tree.GetEntries()):
            in_tree.GetEntry(i)

            # Copy kinematic variables
            for name in ['mass1', 'mass2', 'MT1', 'MT2']:
                out_vars[name][0] = in_vars[name][0]

            # Calculate weight
            w = in_vars['weight'][0] * weight_scale
            if is_signal:
                w /= 3.0  # Signal cross-section normalization to 5 fb
            if apply_convSF:
                w *= self.convSF
            w *= kfactor  # Apply K-factor
            out_vars['weight'][0] = w

            # Copy scores
            for suffix in score_vars:
                score_vars[suffix][0] = in_scores[suffix][0]

            # Select mass
            out_vars['mass'][0] = self._select_mass(in_vars['mass1'][0], in_vars['mass2'][0])

            out_tree.Fill()

        self.out_file.cd()
        out_tree.Write()
        logging.debug(f"Processed {in_tree.GetEntries()} entries for {output_tree_name}")

    def process_systematics(self, category, syst_categories, **kwargs):
        """Process all shape systematics for a category."""
        # Preprocessed shape systematics (Up/Down pairs)
        for syst_name, variations, group in syst_categories['preprocessed_shape']:
            if category not in group:
                continue
            for input_tree, output_tree in iter_shape_variations(syst_name, variations):
                self.process_tree(input_tree, output_tree, **kwargs)

        # Multi-variation systematics (PDF, Scale)
        for syst_name, variations, group in syst_categories['multi_variation']:
            if category not in group:
                continue
            for var in variations:
                if var.startswith("pdf_"):
                    num = int(var.replace("pdf_", ""))
                    input_tree, output_tree = f"Events_PDF_{num}", f"PDF_{num}"
                elif var.startswith("Scale_"):
                    input_tree, output_tree = f"Events_{var}", var
                else:
                    input_tree, output_tree = f"Events_{var}", var
                try:
                    self.process_tree(input_tree, output_tree, **kwargs)
                except RuntimeError as e:
                    logging.warning(f"    Skipping {var}: {e}")


# =============================================================================
# Batch Processing Helpers
# =============================================================================

def process_samples_batch(preprocessor, samples, input_base_path, output_path,
                          process_func, temp_prefix, aliases=None):
    """
    Generic batch processor for multiple samples with hadd merging.

    Args:
        preprocessor: SamplePreprocessor instance
        samples: List of sample names
        input_base_path: Base path for input files
        output_path: Final merged output path
        process_func: Function(preprocessor, sample) to call for each sample
        temp_prefix: Prefix for temp files
        aliases: Sample name aliases dict
    """
    temp_files = []
    aliases = aliases or {}

    for sample in samples:
        input_path = f"{input_base_path}/Skim_TriLep_{sample}.root"
        if not os.path.exists(input_path):
            logging.warning(f"  Sample not found: {input_path}")
            continue

        alias = aliases.get(sample, sample)
        temp_output = f"{os.path.dirname(output_path)}/_temp_{temp_prefix}_{alias}.root"
        temp_files.append(temp_output)

        preprocessor.set_input_file(input_path)
        preprocessor.set_output_file(temp_output)

        process_func(preprocessor, sample)

        preprocessor.close_files()
        logging.debug(f"  Processed {sample}")

    if temp_files:
        hadd_files(output_path, temp_files, cleanup=True)
        logging.info(f"  Output: {output_path}")


# =============================================================================
# Region Processing
# =============================================================================

def process_backgrounds(workdir, era, channel, masspoint, basedir, preprocessor,
                        config, syst_categories, kfactors):
    """Process all background samples for a channel."""
    input_channel = CHANNEL_INPUT_MAP[channel]
    reserved_keys = {"data", "nonprompt"}

    bkg_base_path = f"{workdir}/SKNanoOutput/PromptAnalyzer/{input_channel}_RunSyst/{era}"

    for category in [k for k in config['samples'] if k not in reserved_keys]:
        output_name = category
        apply_convSF = (category == "conversion")

        logging.info("=" * 60)
        logging.info(f"Processing {output_name}")
        logging.info("=" * 60)

        def process_bkg(proc, sample, cat=output_name, conv=apply_convSF, kf=kfactors):
            sample_kfactor = kf.get(sample, 1.0)
            if sample_kfactor != 1.0:
                logging.info(f"  Applying K-factor {sample_kfactor:.3f} to {sample}")
            proc.process_tree("Events_Central", "Central", apply_convSF=conv, kfactor=sample_kfactor)
            proc.process_systematics(cat, syst_categories, apply_convSF=conv, kfactor=sample_kfactor)

        process_samples_batch(
            preprocessor, config['samples'][category], bkg_base_path,
            f"{basedir}/{output_name}.root", process_bkg, output_name, config['aliases']
        )


def process_nonprompt(workdir, era, channel, masspoint, basedir, preprocessor, config):
    """Process nonprompt samples for a channel."""
    input_channel = CHANNEL_INPUT_MAP[channel]

    logging.info("=" * 60)
    logging.info("Processing Nonprompt")
    logging.info("=" * 60)

    def process_np(proc, sample):
        proc.process_tree("Events", "Central")

    nonprompt_base = f"{workdir}/SKNanoOutput/MatrixAnalyzer/{input_channel}/{era}"
    process_samples_batch(
        preprocessor, config['samples']['nonprompt'], nonprompt_base,
        f"{basedir}/nonprompt.root", process_np, "nonprompt"
    )


def process_data(workdir, era, channel, masspoint, basedir, preprocessor, config):
    """Process data samples for a channel."""
    input_channel = CHANNEL_INPUT_MAP[channel]

    logging.info("=" * 60)
    logging.info("Processing Data")
    logging.info("=" * 60)

    def process_d(proc, sample):
        proc.process_tree("Events_Central", "Central")

    data_base = f"{workdir}/SKNanoOutput/PromptAnalyzer/{input_channel}/{era}"
    process_samples_batch(
        preprocessor, config['samples']['data'], data_base,
        f"{basedir}/data.root", process_d, "data"
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(levelname)s - %(message)s')

    workdir = os.getenv("WORKDIR")
    if not workdir:
        raise EnvironmentError("WORKDIR environment variable is not set. Run 'source setup.sh' first.")

    basedir = f"{workdir}/SignalRegionStudyV2/samples/{args.era}/{args.channel}/{args.masspoint}"

    # Validate TTZ2E1Mu masspoint (only ParticleNet masspoints)
    if args.channel == "TTZ2E1Mu":
        mA = int(args.masspoint.split("_")[1].replace("MA", ""))
        if not (83 < mA < 100):
            raise ValueError(
                f"TTZ2E1Mu channel is only for ParticleNet masspoints (83 < mA < 100).\n"
                f"  Requested masspoint: {args.masspoint} (mA={mA})"
            )

    # Load configurations
    config = load_config(workdir, args.era, args.channel)
    syst_categories = categorize_systematics(config['systematics'])
    convSF = load_convSF(workdir, args.era, args.channel)
    kfactors = load_kfactors(workdir, args.era)

    logging.info(f"Preprocessing {args.masspoint} for {args.era} era and {args.channel} channel")
    logging.info(f"Input channel: {CHANNEL_INPUT_MAP[args.channel]}")
    logging.info(f"Config channel: {CHANNEL_CONFIG_MAP[args.channel]}")
    logging.info(f"Found {len(syst_categories['preprocessed_shape'])} preprocessed shape systematics")
    logging.info(f"Found {len(syst_categories['valued_shape'])} valued shape systematics")
    logging.info(f"Found {len(syst_categories['multi_variation'])} multi-variation systematics")
    logging.info(f"Found {len(syst_categories['valued_lnN'])} valued lnN systematics (skipped)")

    # Clean and create output directory
    ensure_directory(basedir, clean=True)

    # Initialize preprocessor
    preprocessor = SamplePreprocessor(args.era, args.channel, args.masspoint, convSF)

    # === 1. Process Signal (only for signal channels) ===
    if args.channel in CHANNELS_WITH_SIGNAL:
        input_channel = CHANNEL_INPUT_MAP[args.channel]

        logging.info("=" * 60)
        logging.info("Processing Signal")
        logging.info("=" * 60)

        signal_input = f"{workdir}/SKNanoOutput/PromptAnalyzer/{input_channel}_RunSyst_RunTheoryUnc/{args.era}/TTToHcToWAToMuMu-{args.masspoint}.root"
        if not os.path.exists(signal_input):
            raise FileNotFoundError(f"Signal file not found: {signal_input}")

        preprocessor.set_input_file(signal_input)
        preprocessor.set_output_file(f"{basedir}/{args.masspoint}.root")

        preprocessor.process_tree("Events_Central", "Central", is_signal=True)
        preprocessor.process_systematics("signal", syst_categories, is_signal=True)

        preprocessor.close_files()
        logging.info(f"  Output: {basedir}/{args.masspoint}.root")

    # === 2. Process Backgrounds, Nonprompt, Data ===
    process_backgrounds(workdir, args.era, args.channel, args.masspoint, basedir,
                        preprocessor, config, syst_categories, kfactors)
    process_nonprompt(workdir, args.era, args.channel, args.masspoint, basedir,
                      preprocessor, config)
    process_data(workdir, args.era, args.channel, args.masspoint, basedir,
                 preprocessor, config)

    logging.info("=" * 60)
    logging.info(f"Preprocessing complete! Output directory: {basedir}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
