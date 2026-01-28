#!/usr/bin/env python3
"""
Preprocess signal and background samples for SignalRegionStudyV2.

This script preprocesses ROOT files with systematic variations,
loading era-specific systematics configuration from configs/systematics.{era}.json.

Key addition in V2: Run3 signal scaling from Run2 (2018) samples.

Processes:
- Signal: from RunSyst_RunTheoryUnc (with theory uncertainties)
  - For Run3: optionally scale from 2018 using --scale-from-run2 flag
- Backgrounds: from RunSyst (WZ, ZZ, ttW, ttZ, etc.)
- Nonprompt: from MatrixTreeProducer (data-driven)
- Data: from PromptTreeProducer (Central only)

Usage:
    # Run2 processing (standard)
    python preprocess.py --era 2018 --channel SR1E2Mu --masspoint MHc130_MA90

    # Run3 processing with scaled signal from 2018
    python preprocess.py --era 2022EE --channel SR1E2Mu --masspoint MHc130_MA90 --scale-from-run2
"""
import os
import re
import shutil
import argparse
import logging
import json
import subprocess
from array import array
import ROOT


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Preprocess samples for SignalRegionStudyV2")
    parser.add_argument("--era", required=True, type=str, help="era (e.g., 2018, 2022EE)")
    parser.add_argument("--channel", required=True, type=str, help="channel (SR1E2Mu or SR3Mu)")
    parser.add_argument("--masspoint", required=True, type=str, help="signal mass point (e.g., MHc130_MA90)")
    parser.add_argument("--scale-from-run2", action="store_true",
                        help="Scale Run3 signal from 2018 samples (for Run3 eras without signal MC)")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    return parser.parse_args()


def is_run3_era(era):
    """Check if era is a Run3 era."""
    return era in ["2022", "2022EE", "2023", "2023BPix"]


def load_scaling_config(workdir):
    """Load scaling configuration for Run3 signal from Run2."""
    config_path = f"{workdir}/SignalRegionStudyV2/configs/scaling.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Scaling config not found: {config_path}")

    with open(config_path) as f:
        return json.load(f)


def calculate_run3_scale_factor(scaling_config, target_era):
    """
    Calculate scale factor for Run3 signal from 2018.

    scale_factor = (xsec_Run3 / xsec_Run2) * (lumi_target / lumi_source)
    """
    xsec_run2 = scaling_config["ttbar_xsec"]["Run2"]
    xsec_run3 = scaling_config["ttbar_xsec"]["Run3"]
    source_era = scaling_config["source_era_for_run3"]
    lumi_source = scaling_config["luminosity"][source_era]
    lumi_target = scaling_config["luminosity"][target_era]

    scale_factor = (xsec_run3 / xsec_run2) * (lumi_target / lumi_source)

    logging.info(f"Run3 signal scaling from {source_era} to {target_era}:")
    logging.info(f"  xsec ratio: {xsec_run3}/{xsec_run2} = {xsec_run3/xsec_run2:.4f}")
    logging.info(f"  lumi ratio: {lumi_target}/{lumi_source} = {lumi_target/lumi_source:.4f}")
    logging.info(f"  total scale factor: {scale_factor:.4f}")

    return scale_factor, source_era


def load_config(workdir, era, channel):
    """Load systematics and sample group configurations."""
    # Load systematics config
    config_path = f"{workdir}/SignalRegionStudyV2/configs/systematics.{era}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Systematics config not found: {config_path}")

    with open(config_path) as f:
        json_systematics = json.load(f)

    if channel not in json_systematics:
        raise ValueError(f"Channel '{channel}' not found in {config_path}")

    # Load sample groups config
    samplegroups_path = f"{workdir}/SignalRegionStudyV2/configs/samplegroups.json"
    if not os.path.exists(samplegroups_path):
        raise FileNotFoundError(f"Sample groups config not found: {samplegroups_path}")

    with open(samplegroups_path) as f:
        json_samplegroups = json.load(f)

    if era not in json_samplegroups:
        raise ValueError(f"Era '{era}' not found in {samplegroups_path}")
    if channel not in json_samplegroups[era]:
        raise ValueError(f"Channel '{channel}' not found for era '{era}'")

    return {
        'systematics': json_systematics[channel],
        'samples': json_samplegroups[era][channel],
        'aliases': json_samplegroups.get("aliases", {})
    }


def load_convSF(workdir, era, channel):
    """Load conversion scale factor from TriLepton ZG results."""
    convSF_file = f"{workdir}/TriLepton/results/{channel.replace('SR', 'ZG')}/{era}/ConvSF.json"

    if not os.path.exists(convSF_file):
        logging.warning(f"ConvSF file not found: {convSF_file}")
        logging.warning("Using default ConvSF = 1.0 +/- 0.3")
        return 1.0, 0.3

    with open(convSF_file) as f:
        data = json.load(f)

    central_corrections = [c for c in data["corrections"] if c["name"].endswith("_Central")]
    if not central_corrections:
        logging.error(f"No Central correction found in {convSF_file}")
        return 1.0, 0.3

    central_sf = float(central_corrections[0]["data"]["expression"])

    syst_sfs = [
        float(c["data"]["expression"])
        for c in data["corrections"]
        if not c["name"].endswith("_Central")
        and "nonprompt" not in c["name"].lower()
        and "prompt" not in c["name"].lower()
    ]

    sf_err = max(abs(central_sf - min(syst_sfs)), abs(max(syst_sfs) - central_sf)) if syst_sfs else 0.3 * central_sf

    logging.info(f"Loaded ConvSF for {era} {channel}: {central_sf:.4f} +/- {sf_err:.4f}")
    return central_sf, sf_err


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


def parse_variations(variation_spec):
    """
    Parse variation specification strings from config.

    Supports:
    - ["Var_Up", "Var_Down"] - simple list
    - ["Scale_{0..8}//5,7"] - range with exclusions
    - ["pdf_{00..99}"] - range with zero-padded numbers
    """
    if not isinstance(variation_spec, list):
        return []

    if len(variation_spec) == 1 and '{' in variation_spec[0]:
        return _expand_range_pattern(variation_spec[0])
    return variation_spec


def _expand_range_pattern(pattern):
    """Expand range pattern like 'Scale_{0..8}//5,7' or 'pdf_{00..99}'."""
    exclusions = set()
    if '//' in pattern:
        pattern, excl_str = pattern.split('//')
        exclusions = set(int(x) for x in excl_str.split(','))

    match = re.search(r'\{(\d+)\.\.(\d+)\}', pattern)
    if not match:
        return [pattern]

    start, end = int(match.group(1)), int(match.group(2))
    start_str = match.group(1)
    pad_width = len(start_str) if start_str.startswith('0') and len(start_str) > 1 else 0

    prefix, suffix = pattern[:match.start()], pattern[match.end():]
    return [
        f"{prefix}{str(i).zfill(pad_width) if pad_width else str(i)}{suffix}"
        for i in range(start, end + 1) if i not in exclusions
    ]


def get_output_tree_name(syst_name, variation):
    """
    Get output tree name from systematic name and variation.

    Args:
        syst_name: Systematic name (e.g., 'CMS_pileup_13TeV')
        variation: Variation name (e.g., 'PileupReweight_Up')

    Returns:
        Output tree name (e.g., 'CMS_pileup_13TeV_Up')
    """
    if variation.endswith("_Up") or variation.endswith("Up"):
        return f"{syst_name}_Up"
    elif variation.endswith("_Down") or variation.endswith("Down"):
        return f"{syst_name}_Down"
    return variation


def categorize_systematics(config):
    """
    Categorize systematics from config into processing groups.

    Returns dict with keys:
    - preprocessed_shape: list of (syst_name, [variations], group)
    - valued_shape: list of (syst_name, value, group)
    - multi_variation: list of (syst_name, [variations], group)
    - valued_lnN: list of (syst_name, value, group)
    """
    result = {'preprocessed_shape': [], 'valued_shape': [], 'multi_variation': [], 'valued_lnN': []}

    for syst_name, syst_config in config.items():
        source = syst_config.get('source')
        syst_type = syst_config.get('type')
        group = syst_config.get('group', [])

        if source == 'preprocessed' and syst_type == 'shape':
            variations = parse_variations(syst_config.get('variations', []))
            if len(variations) > 2:
                result['multi_variation'].append((syst_name, variations, group))
            elif len(variations) == 2:
                result['preprocessed_shape'].append((syst_name, variations, group))
            else:
                logging.warning(f"Unexpected variation count for {syst_name}: {variations}")

        elif source == 'valued' and syst_type == 'shape':
            result['valued_shape'].append((syst_name, syst_config.get('value'), group))

        elif source == 'valued' and syst_type == 'lnN':
            result['valued_lnN'].append((syst_name, syst_config.get('value'), group))

    return result


def calculate_weight_scale(value, direction):
    """
    Calculate weight scale for valued+shape systematics.

    For value >= 1: up = value, down = 2 - value
    For value < 1: up = 1 + value, down = 1 - value
    """
    if value >= 1.0:
        return value if direction == 'up' else 2.0 - value
    return 1.0 + value if direction == 'up' else 1.0 - value


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


class SamplePreprocessor:
    """Preprocessor for samples with systematic variations."""

    def __init__(self, era, channel, masspoint, convSF=1.0):
        self.era = era
        self.channel = channel
        self.masspoint = masspoint
        self.convSF = convSF
        self.in_file = None
        self.out_file = None

        # Parse mass point parameters
        self.mHc = int(masspoint.split("_")[0].replace("MHc", ""))
        self.mA = int(masspoint.split("_")[1].replace("MA", ""))
        self.is_trained_sample = (83 < self.mA < 100)

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

    def process_tree(self, input_tree_name, output_tree_name, weight_scale=1.0,
                     is_signal=False, apply_convSF=False, kfactor=1.0):
        """Process a single tree from input to output."""
        in_tree = self.in_file.Get(input_tree_name)
        if not in_tree:
            raise RuntimeError(f"Tree '{input_tree_name}' not found in input file")

        out_tree = ROOT.TTree(output_tree_name, "")

        # Output arrays
        out_vars = {name: array('d', [0.0]) for name in ['mass', 'mass1', 'mass2', 'MT1', 'MT2', 'weight']}
        score_vars = {}
        if self.is_trained_sample:
            for suffix in ['signal', 'nonprompt', 'diboson', 'ttZ']:
                score_vars[suffix] = array('d', [0.0])

        # Setup output branches
        for name, arr in out_vars.items():
            out_tree.Branch(name, arr, f"{name}/D")
        for suffix, arr in score_vars.items():
            out_tree.Branch(f"score_{self.masspoint}_{suffix}", arr, f"score_{self.masspoint}_{suffix}/D")

        # Input arrays
        in_vars = {name: array('d', [0.0]) for name in ['mass1', 'mass2', 'MT1', 'MT2', 'weight']}
        in_scores = {suffix: array('d', [0.0]) for suffix in score_vars}

        # Setup input branch addresses
        for name, arr in in_vars.items():
            in_tree.SetBranchAddress(name, arr)
        for suffix, arr in in_scores.items():
            in_tree.SetBranchAddress(f"score_{self.masspoint}_{suffix}", arr)

        # Process entries
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

            # Select mass based on channel and mass point
            if "1E2Mu" in self.channel:
                out_vars['mass'][0] = in_vars['mass1'][0]
            elif "3Mu" in self.channel:
                if self.mHc >= 100 and self.mA >= 60:
                    out_vars['mass'][0] = max(in_vars['mass1'][0], in_vars['mass2'][0])
                else:
                    out_vars['mass'][0] = min(in_vars['mass1'][0], in_vars['mass2'][0])
            else:
                raise ValueError(f"Unknown channel: {self.channel}")

            out_tree.Fill()

        self.out_file.cd()
        out_tree.Write()
        logging.debug(f"Processed {in_tree.GetEntries()} entries for {output_tree_name}")

    def process_preprocessed_shape(self, category, syst_categories, **kwargs):
        """Process all preprocessed shape systematics for a category."""
        for syst_name, variations, group in syst_categories['preprocessed_shape']:
            if category not in group:
                continue

            for var in variations:
                try:
                    self.process_tree(f"Events_{var}", get_output_tree_name(syst_name, var), **kwargs)
                except RuntimeError:
                    pass  # Tree may not exist

    def process_valued_shape(self, category, syst_categories, **kwargs):
        """Process all valued shape systematics for a category."""
        for syst_name, value, group in syst_categories['valued_shape']:
            if category not in group:
                continue

            for direction in ['up', 'down']:
                scale = calculate_weight_scale(value, direction)
                self.process_tree("Events_Central", f"{syst_name}_{direction.capitalize()}",
                                  weight_scale=scale, **kwargs)

    def process_multi_variation(self, category, syst_categories, **kwargs):
        """Process multi-variation systematics (PDF, Scale) for a category."""
        for syst_name, variations, group in syst_categories['multi_variation']:
            if category not in group:
                continue

            for var in variations:
                # Determine input/output tree names based on variation pattern
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


class ScaledSignalPreprocessor:
    """Preprocessor for scaling Run2 signal samples to Run3."""

    def __init__(self, era, channel, masspoint, scale_factor):
        self.era = era
        self.channel = channel
        self.masspoint = masspoint
        self.scale_factor = scale_factor
        self.in_file = None
        self.out_file = None

        # Parse mass point parameters
        self.mHc = int(masspoint.split("_")[0].replace("MHc", ""))
        self.mA = int(masspoint.split("_")[1].replace("MA", ""))
        self.is_trained_sample = (83 < self.mA < 100)

    def set_input_file(self, path):
        """Open input ROOT file (from 2018 samples)."""
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

    def scale_all_trees(self):
        """Scale all trees from input file to output, applying the scale factor to weights."""
        keys = self.in_file.GetListOfKeys()
        tree_names = [key.GetName() for key in keys if key.GetClassName() == "TTree"]

        logging.info(f"  Found {len(tree_names)} trees to scale")

        for tree_name in tree_names:
            self._scale_tree(tree_name)

    def _scale_tree(self, tree_name):
        """Scale a single tree."""
        in_tree = self.in_file.Get(tree_name)
        if not in_tree:
            raise RuntimeError(f"Tree '{tree_name}' not found in input file")

        out_tree = ROOT.TTree(tree_name, "")

        # Output arrays
        out_vars = {name: array('d', [0.0]) for name in ['mass', 'mass1', 'mass2', 'MT1', 'MT2', 'weight']}
        score_vars = {}
        if self.is_trained_sample:
            for suffix in ['signal', 'nonprompt', 'diboson', 'ttZ']:
                score_vars[suffix] = array('d', [0.0])

        # Setup output branches
        for name, arr in out_vars.items():
            out_tree.Branch(name, arr, f"{name}/D")
        for suffix, arr in score_vars.items():
            out_tree.Branch(f"score_{self.masspoint}_{suffix}", arr, f"score_{self.masspoint}_{suffix}/D")

        # Input arrays
        in_vars = {name: array('d', [0.0]) for name in ['mass', 'mass1', 'mass2', 'MT1', 'MT2', 'weight']}
        in_scores = {suffix: array('d', [0.0]) for suffix in score_vars}

        # Setup input branch addresses
        for name, arr in in_vars.items():
            in_tree.SetBranchAddress(name, arr)
        for suffix, arr in in_scores.items():
            in_tree.SetBranchAddress(f"score_{self.masspoint}_{suffix}", arr)

        # Process entries
        for i in range(in_tree.GetEntries()):
            in_tree.GetEntry(i)

            # Copy kinematic variables
            for name in ['mass', 'mass1', 'mass2', 'MT1', 'MT2']:
                out_vars[name][0] = in_vars[name][0]

            # Apply scale factor to weight
            out_vars['weight'][0] = in_vars['weight'][0] * self.scale_factor

            # Copy scores
            for suffix in score_vars:
                score_vars[suffix][0] = in_scores[suffix][0]

            out_tree.Fill()

        self.out_file.cd()
        out_tree.Write()
        logging.debug(f"  Scaled {in_tree.GetEntries()} entries for {tree_name}")


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


def process_signal_from_run2(workdir, era, channel, masspoint, scale_factor, source_era, basedir):
    """
    Process Run3 signal by scaling from Run2 (2018) preprocessed samples.

    Args:
        workdir: Working directory
        era: Target Run3 era (e.g., 2022EE)
        channel: Channel (SR1E2Mu or SR3Mu)
        masspoint: Mass point (e.g., MHc130_MA90)
        scale_factor: Scale factor to apply
        source_era: Source era (2018)
        basedir: Output base directory
    """
    logging.info("=" * 60)
    logging.info(f"Processing Signal (scaled from {source_era})")
    logging.info("=" * 60)

    # Try V2 samples first, then V1
    # Also check samples_source for HTCondor jobs where pnfs is symlinked
    source_paths = [
        f"{workdir}/SignalRegionStudyV2/samples/{source_era}/{channel}/{masspoint}/{masspoint}.root",
        f"{workdir}/SignalRegionStudyV2/samples_source/{source_era}/{channel}/{masspoint}/{masspoint}.root",
        f"{workdir}/SignalRegionStudyV1/samples/{source_era}/{channel}/{masspoint}/{masspoint}.root",
    ]

    source_path = None
    for path in source_paths:
        if os.path.exists(path):
            source_path = path
            break

    if source_path is None:
        raise FileNotFoundError(
            f"Source signal file not found. Tried:\n" +
            "\n".join(f"  - {p}" for p in source_paths) +
            f"\n\nPlease run preprocessing for {source_era} first."
        )

    logging.info(f"  Source: {source_path}")

    processor = ScaledSignalPreprocessor(era, channel, masspoint, scale_factor)
    processor.set_input_file(source_path)
    processor.set_output_file(f"{basedir}/{masspoint}.root")
    processor.scale_all_trees()
    processor.close_files()

    logging.info(f"  Output: {basedir}/{masspoint}.root")


def main():
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(levelname)s - %(message)s')

    workdir = os.getenv("WORKDIR")
    if not workdir:
        raise EnvironmentError("WORKDIR environment variable is not set. Run 'source setup.sh' first.")

    basedir = f"{workdir}/SignalRegionStudyV2/samples/{args.era}/{args.channel}/{args.masspoint}"

    # Validate --scale-from-run2 usage
    if args.scale_from_run2 and not is_run3_era(args.era):
        raise ValueError(f"--scale-from-run2 can only be used with Run3 eras. Got: {args.era}")

    # Load configurations
    config = load_config(workdir, args.era, args.channel)
    syst_categories = categorize_systematics(config['systematics'])
    convSF, _ = load_convSF(workdir, args.era, args.channel)
    kfactors = load_kfactors(workdir, args.era)

    logging.info(f"Preprocessing {args.masspoint} for {args.era} era and {args.channel} channel")
    logging.info(f"Found {len(syst_categories['preprocessed_shape'])} preprocessed shape systematics")
    logging.info(f"Found {len(syst_categories['valued_shape'])} valued shape systematics")
    logging.info(f"Found {len(syst_categories['multi_variation'])} multi-variation systematics")
    logging.info(f"Found {len(syst_categories['valued_lnN'])} valued lnN systematics (skipped)")

    # Clean and create output directory
    if os.path.exists(basedir):
        logging.info(f"Removing existing directory {basedir}")
        shutil.rmtree(basedir)
    os.makedirs(basedir, exist_ok=True)

    # Initialize preprocessor
    preprocessor = SamplePreprocessor(args.era, args.channel, args.masspoint, convSF)
    input_channel = args.channel.replace("SR", "Run")

    # === 1. Process Signal ===
    if args.scale_from_run2:
        # Scale Run3 signal from Run2 (2018)
        scaling_config = load_scaling_config(workdir)
        scale_factor, source_era = calculate_run3_scale_factor(scaling_config, args.era)
        process_signal_from_run2(workdir, args.era, args.channel, args.masspoint,
                                  scale_factor, source_era, basedir)
    else:
        # Standard signal processing from SKNanoOutput
        logging.info("=" * 60)
        logging.info("Processing Signal")
        logging.info("=" * 60)

        signal_input = f"{workdir}/SKNanoOutput/PromptTreeProducer/{input_channel}_RunSyst_RunTheoryUnc/{args.era}/TTToHcToWAToMuMu-{args.masspoint}.root"
        if not os.path.exists(signal_input):
            raise FileNotFoundError(f"Signal file not found: {signal_input}")

        preprocessor.set_input_file(signal_input)
        preprocessor.set_output_file(f"{basedir}/{args.masspoint}.root")

        preprocessor.process_tree("Events_Central", "Central", is_signal=True)
        preprocessor.process_preprocessed_shape("signal", syst_categories, is_signal=True)
        preprocessor.process_multi_variation("signal", syst_categories, is_signal=True)
        # Note: valued_shape systematics are created in makeBinnedTemplates.py by scaling Central

        preprocessor.close_files()
        logging.info(f"  Output: {basedir}/{args.masspoint}.root")

    # === 2. Process Backgrounds ===
    bkg_base_path = f"{workdir}/SKNanoOutput/PromptTreeProducer/{input_channel}_RunSyst/{args.era}"
    reserved_keys = {"data", "nonprompt"}

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
            proc.process_preprocessed_shape(cat, syst_categories, apply_convSF=conv, kfactor=sample_kfactor)
            # Note: valued_shape systematics are created in makeBinnedTemplates.py

        process_samples_batch(
            preprocessor, config['samples'][category], bkg_base_path,
            f"{basedir}/{output_name}.root", process_bkg, output_name, config['aliases']
        )

    # === 3. Process Nonprompt ===
    logging.info("=" * 60)
    logging.info("Processing Nonprompt")
    logging.info("=" * 60)

    def process_nonprompt(proc, sample):
        proc.process_tree("Events", "Central")
        # Note: valued_shape systematics (nonprompt normalization) are created in makeBinnedTemplates.py

    nonprompt_base = f"{workdir}/SKNanoOutput/MatrixTreeProducer/{input_channel}/{args.era}"
    process_samples_batch(
        preprocessor, config['samples']['nonprompt'], nonprompt_base,
        f"{basedir}/nonprompt.root", process_nonprompt, "nonprompt"
    )

    # === 4. Process Data ===
    logging.info("=" * 60)
    logging.info("Processing Data")
    logging.info("=" * 60)

    def process_data(proc, sample):
        proc.process_tree("Events_Central", "Central")

    data_base = f"{workdir}/SKNanoOutput/PromptTreeProducer/{input_channel}/{args.era}"
    process_samples_batch(
        preprocessor, config['samples']['data'], data_base,
        f"{basedir}/data.root", process_data, "data"
    )

    logging.info("=" * 60)
    logging.info(f"Preprocessing complete! Output directory: {basedir}")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
