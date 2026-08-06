#!/usr/bin/env python3
"""
Preprocess signal and background samples for SignalRegionStudyV4.

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


def masspoint_mhc_prefix(masspoint):
    """Return the MHc prefix from a mass point label."""
    parts = masspoint.split("_")
    if len(parts) < 2 or not parts[0].startswith("MHc"):
        raise ValueError(f"Invalid mass point format: {masspoint}")
    return parts[0]


def input_channel_dir(channel, masspoint):
    """Return SKNanoOutput input channel directory for this mass point."""
    input_channel = CHANNEL_INPUT_MAP[channel]
    if masspoint in PN_TRAINED_MASSPOINTS:
        return f"{input_channel}_{masspoint_mhc_prefix(masspoint)}"
    return input_channel


def input_mode(masspoint):
    """Return the SKNanoOutput production mode used by this mass point."""
    return "NoHistMode" if masspoint in PN_TRAINED_MASSPOINTS else "standard"


def prompt_input_dir(channel, masspoint, suffix=""):
    """Return PromptAnalyzer directory name for this channel/mass point."""
    input_channel = input_channel_dir(channel, masspoint)
    directory = f"{input_channel}{suffix}"
    if masspoint in PN_TRAINED_MASSPOINTS:
        directory = f"{directory}_NoHistMode"
    return directory


def matrix_input_dir(channel, masspoint):
    """Return MatrixAnalyzer directory name for this channel/mass point."""
    input_channel = input_channel_dir(channel, masspoint)
    if masspoint in PN_TRAINED_MASSPOINTS:
        return f"{input_channel}_NoHistMode"
    return input_channel


def parse_args():
    """Parse command line arguments.

    Three production modes:
      (default)             per-masspoint dir with ParticleNet scores; only
                            valid for ParticleNet-trained mass points
                            (samples/{era}/{channel}/{masspoint}/).
      --shared-backgrounds  mass-independent backgrounds/nonprompt/data into
                            the shared dir (samples/{era}/SR1E2Mu or
                            samples/{era}/SR3Mu_{lowM,highM}; --pairing
                            required for SR3Mu). No --masspoint.
      --shared-signal       signal only, into the shared dir(s) as
                            {masspoint}.root, from the standard skims.
                            SR3Mu writes BOTH lowM and highM variants (the
                            interpolation study needs signals under both
                            pairing selections).
    """
    parser = argparse.ArgumentParser(description="Preprocess samples for SignalRegionStudyV4")
    parser.add_argument("--era", required=True, type=str, help="era (e.g., 2018, 2022EE)")
    parser.add_argument("--channel", required=True, type=str,
                        choices=list(CHANNEL_INPUT_MAP.keys()),
                        help="channel (SR1E2Mu, SR3Mu, or TTZ2E1Mu)")
    parser.add_argument("--masspoint", type=str, default=None,
                        help="signal mass point (e.g., MHc130_MA90)")
    parser.add_argument("--shared-backgrounds", action="store_true", dest="shared_backgrounds",
                        help="produce shared (mass-independent) backgrounds/nonprompt/data")
    parser.add_argument("--shared-signal", action="store_true", dest="shared_signal",
                        help="produce only the signal into the shared dir(s)")
    parser.add_argument("--pairing", choices=["lowM", "highM"], default=None,
                        help="SR3Mu pairing variant (required for --shared-backgrounds on SR3Mu)")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()

    if args.shared_backgrounds and args.shared_signal:
        parser.error("--shared-backgrounds and --shared-signal are mutually exclusive")
    if args.shared_backgrounds:
        if args.masspoint:
            parser.error("--shared-backgrounds takes no --masspoint")
        if args.channel == "SR3Mu" and not args.pairing:
            parser.error("--shared-backgrounds on SR3Mu requires --pairing lowM|highM")
        if args.channel == "TTZ2E1Mu":
            parser.error("TTZ2E1Mu has no shared layout (ParticleNet-only channel)")
    elif args.shared_signal:
        if not args.masspoint:
            parser.error("--shared-signal requires --masspoint")
        if args.channel == "TTZ2E1Mu":
            parser.error("TTZ2E1Mu has no shared layout (ParticleNet-only channel)")
    else:
        if not args.masspoint:
            parser.error("--masspoint is required (or use a --shared-* mode)")
        if args.masspoint not in PN_TRAINED_MASSPOINTS:
            parser.error(
                f"{args.masspoint} is not ParticleNet-trained; per-masspoint "
                "production is ParticleNet-only. Use --shared-signal for the "
                "baseline layout."
            )
    return args


def is_run3_era(era):
    """Check if era is a Run3 era."""
    return era in ["2022", "2022EE", "2023", "2023BPix"]


def load_config(workdir, era, channel):
    """Load systematics and sample group configurations."""
    # Map to config channel
    config_channel = CHANNEL_CONFIG_MAP.get(channel, channel)

    # Load systematics config
    config_path = f"{workdir}/SignalRegionStudyV4/configs/systematics.{era}.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Systematics config not found: {config_path}")

    with open(config_path) as f:
        json_systematics = json.load(f)

    if config_channel not in json_systematics:
        raise ValueError(f"Channel '{config_channel}' not found in {config_path}")

    # Load sample groups config
    samplegroups_path = f"{workdir}/SignalRegionStudyV4/configs/samplegroups.json"
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

def pairing_for(channel, masspoint=None, pairing=None):
    """Resolve the dimuon-pairing rule: 'mass1' | 'low' | 'high'.

    SR1E2Mu/TTZ2E1Mu always store mass1. SR3Mu stores the higher-mass
    pairing iff mHc >= 100 && mA >= 60 (the pairing rule — not a pure mA
    threshold), or an explicit lowM/highM override for shared-background
    production where no mass point exists."""
    if "1E2Mu" in channel or "2E1Mu" in channel:
        return "mass1"
    if "3Mu" in channel:
        if pairing is not None:
            return {"lowM": "low", "highM": "high"}[pairing]
        if masspoint is None:
            raise ValueError("SR3Mu needs a mass point or an explicit --pairing")
        mhc = int(masspoint.split("_")[0].replace("MHc", ""))
        ma = int(masspoint.split("_")[1].replace("MA", ""))
        return "high" if (mhc >= 100 and ma >= 60) else "low"
    raise ValueError(f"Unknown channel: {channel}")


class BasePreprocessor:
    """Base class with shared file/branch operations."""

    def __init__(self, era, channel, masspoint=None, pairing=None):
        self.era = era
        self.channel = channel
        self.masspoint = masspoint
        self.in_file = None
        self.in_path = None
        self.out_path = None

        self.pairing = pairing_for(channel, masspoint=masspoint, pairing=pairing)
        self.is_trained_sample = (masspoint in PN_TRAINED_MASSPOINTS
                                  if masspoint else False)

    def set_input_file(self, path):
        """Open input ROOT file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")
        self.in_path = path
        self.in_file = ROOT.TFile(path, "READ")
        if self.in_file.IsZombie():
            raise IOError(f"Failed to open input file: {path}")

    def set_output_file(self, path):
        """Register (and truncate) the output ROOT file.

        Trees are written by RDataFrame Snapshot in UPDATE mode, so the file
        is not held open here; RECREATE-and-close truncates any stale file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.out_path = path
        out = ROOT.TFile(path, "RECREATE")
        out.Close()

    def close_files(self):
        """Close the input file."""
        if self.in_file:
            self.in_file.Close()
            self.in_file = None
        self.in_path = None
        self.out_path = None

class SamplePreprocessor(BasePreprocessor):
    """Preprocessor for samples with systematic variations."""

    def __init__(self, era, channel, masspoint=None, convSF=1.0, pairing=None,
                 shared=False):
        super().__init__(era, channel, masspoint, pairing=pairing)
        self.convSF = convSF
        # Shared-layout production reads the standard (non-NoHistMode) skims,
        # which carry no ParticleNet score branches.
        self.include_scores = self.is_trained_sample and not shared

    def process_tree(self, input_tree_name, output_tree_name, weight_scale=1.0,
                     is_signal=False, apply_convSF=False, kfactor=1.0):
        """Process a single tree from input to output.

        Vectorized with an RDataFrame Snapshot: the Define expressions
        replicate the exact per-entry operation order of the original python
        loop (weight multiplications are not associative in floating point),
        so output values are bitwise identical — just written by a compiled
        loop instead of a python one."""
        in_tree = self.in_file.Get(input_tree_name)
        if not in_tree:
            raise RuntimeError(f"Tree '{input_tree_name}' not found in input file")
        n_entries = in_tree.GetEntries()

        rdf = ROOT.RDataFrame(input_tree_name, self.in_path)
        input_columns = {str(c) for c in rdf.GetColumnNames()}

        def define(df, name, expr):
            return df.Redefine(name, expr) if name in input_columns else df.Define(name, expr)

        # Weight: same operation order as the original loop:
        # ((weight * scale) [/ 3.0] [* convSF]) * kfactor
        weight_expr = f"(weight * {weight_scale!r})"
        if is_signal:
            weight_expr = f"({weight_expr} / 3.0)"  # signal normalization to 5 fb
        if apply_convSF:
            weight_expr = f"({weight_expr} * {self.convSF!r})"
        weight_expr = f"({weight_expr} * {kfactor!r})"

        # Select mass and the pT of the same dimuon pairing (see pairing_for)
        if self.pairing == "mass1":
            mass_expr, pt_expr = "mass1", "pT1"
        elif self.pairing == "high":
            mass_expr = "(mass1 >= mass2 ? mass1 : mass2)"
            pt_expr = "(mass1 >= mass2 ? pT1 : pT2)"
        else:
            mass_expr = "(mass1 <= mass2 ? mass1 : mass2)"
            pt_expr = "(mass1 <= mass2 ? pT1 : pT2)"

        rdf = define(rdf, "mass", mass_expr)
        rdf = define(rdf, "pT", pt_expr)
        rdf = define(rdf, "weight", weight_expr)

        columns = ['mass', 'pT', 'mass1', 'mass2', 'weight']
        if self.include_scores:
            columns += [f"score_{self.masspoint}_{suffix}"
                        for suffix in ['signal', 'nonprompt', 'diboson', 'ttZ']]

        if n_entries == 0:
            # RDF Snapshot writes a BRANCHLESS tree for zero-entry inputs;
            # downstream readers expect the branch schema to exist (the old
            # per-entry loop always declared branches before filling).
            out = ROOT.TFile(self.out_path, "UPDATE")
            tree = ROOT.TTree(output_tree_name, "")
            from array import array
            buf = {name: array('d', [0.0]) for name in columns}
            for name in columns:
                tree.Branch(name, buf[name], f"{name}/D")
            out.cd()
            tree.Write()
            out.Close()
            logging.debug(f"Wrote empty (schema-only) tree for {output_tree_name}")
            return

        opts = ROOT.RDF.RSnapshotOptions()
        opts.fMode = "UPDATE"
        columns_vec = ROOT.std.vector('string')(columns)
        rdf.Snapshot(output_tree_name, self.out_path, columns_vec, opts)
        logging.debug(f"Processed {n_entries} entries for {output_tree_name}")

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
    input_channel = prompt_input_dir(channel, masspoint, "_RunSyst")
    reserved_keys = {"data", "nonprompt"}

    bkg_base_path = f"{workdir}/SKNanoOutput/PromptAnalyzer/{input_channel}/{era}"

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
    input_channel = matrix_input_dir(channel, masspoint)

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
    input_channel = prompt_input_dir(channel, masspoint)

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

def shared_dirname(channel, pairing):
    """Shared-layout directory name (mirrors srspaths.shared_channel_dirname)."""
    return f"SR3Mu_{pairing}" if channel == "SR3Mu" else channel


def run_signal(preprocessor, workdir, era, channel, masspoint, basedir,
               syst_categories, input_masspoint_hint):
    """Process the signal sample into {basedir}/{masspoint}.root.

    input_masspoint_hint selects the input skim flavor: pass the mass point
    for per-masspoint (NoHistMode) inputs, or None to force the standard
    skims (shared layout)."""
    input_channel = prompt_input_dir(channel, input_masspoint_hint, "_RunSyst_RunTheoryUnc")

    logging.info("=" * 60)
    logging.info("Processing Signal")
    logging.info("=" * 60)

    signal_input = f"{workdir}/SKNanoOutput/PromptAnalyzer/{input_channel}/{era}/TTToHcToWAToMuMu-{masspoint}.root"
    if not os.path.exists(signal_input):
        raise FileNotFoundError(f"Signal file not found: {signal_input}")

    preprocessor.set_input_file(signal_input)
    preprocessor.set_output_file(f"{basedir}/{masspoint}.root")

    preprocessor.process_tree("Events_Central", "Central", is_signal=True)
    preprocessor.process_systematics("signal", syst_categories, is_signal=True)

    preprocessor.close_files()
    logging.info(f"  Output: {basedir}/{masspoint}.root")


def main():
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO,
                        format='%(levelname)s - %(message)s')

    workdir = os.getenv("WORKDIR")
    if not workdir:
        raise EnvironmentError("WORKDIR environment variable is not set. Run 'source setup.sh' first.")

    # Load configurations
    config = load_config(workdir, args.era, args.channel)
    syst_categories = categorize_systematics(config['systematics'])
    convSF = load_convSF(workdir, args.era, args.channel)
    kfactors = load_kfactors(workdir, args.era)

    logging.info(f"Config channel: {CHANNEL_CONFIG_MAP[args.channel]}")
    logging.info(f"Found {len(syst_categories['preprocessed_shape'])} preprocessed shape systematics")
    logging.info(f"Found {len(syst_categories['valued_shape'])} valued shape systematics")
    logging.info(f"Found {len(syst_categories['multi_variation'])} multi-variation systematics")
    logging.info(f"Found {len(syst_categories['valued_lnN'])} valued lnN systematics (skipped)")

    if args.shared_backgrounds:
        # === Shared layout: mass-independent backgrounds/nonprompt/data ===
        basedir = (f"{workdir}/SignalRegionStudyV4/samples/{args.era}/"
                   f"{shared_dirname(args.channel, args.pairing)}")
        logging.info(f"Shared backgrounds for {args.era}/{shared_dirname(args.channel, args.pairing)}")
        ensure_directory(basedir, clean=False)
        preprocessor = SamplePreprocessor(args.era, args.channel, masspoint=None,
                                          convSF=convSF, pairing=args.pairing,
                                          shared=True)
        process_backgrounds(workdir, args.era, args.channel, None, basedir,
                            preprocessor, config, syst_categories, kfactors)
        process_nonprompt(workdir, args.era, args.channel, None, basedir,
                          preprocessor, config)
        process_data(workdir, args.era, args.channel, None, basedir,
                     preprocessor, config)

    elif args.shared_signal:
        # === Shared layout: signal only, from the standard skims.
        # SR3Mu stores BOTH pairing variants for the interpolation study. ===
        pairings = ["lowM", "highM"] if args.channel == "SR3Mu" else [None]
        for pairing in pairings:
            basedir = (f"{workdir}/SignalRegionStudyV4/samples/{args.era}/"
                       f"{shared_dirname(args.channel, pairing)}")
            logging.info(f"Shared signal {args.masspoint} -> {basedir}")
            ensure_directory(basedir, clean=False)
            preprocessor = SamplePreprocessor(args.era, args.channel,
                                              masspoint=args.masspoint,
                                              convSF=convSF, pairing=pairing,
                                              shared=True)
            run_signal(preprocessor, workdir, args.era, args.channel,
                       args.masspoint, basedir, syst_categories,
                       input_masspoint_hint=None)

    else:
        # === ParticleNet per-masspoint production (scores + NoHistMode skims) ===
        basedir = f"{workdir}/SignalRegionStudyV4/samples/{args.era}/{args.channel}/{args.masspoint}"

        # Validate TTZ2E1Mu masspoint (only ParticleNet masspoints)
        if args.channel == "TTZ2E1Mu":
            mA = int(args.masspoint.split("_")[1].replace("MA", ""))
            if not (83 < mA < 100):
                raise ValueError(
                    f"TTZ2E1Mu channel is only for ParticleNet masspoints (83 < mA < 100).\n"
                    f"  Requested masspoint: {args.masspoint} (mA={mA})"
                )

        logging.info(f"Preprocessing {args.masspoint} for {args.era} era and {args.channel} channel")
        logging.info(f"Input directory channel: {input_channel_dir(args.channel, args.masspoint)}")
        logging.info(f"Input mode: {input_mode(args.masspoint)}")

        ensure_directory(basedir, clean=True)
        preprocessor = SamplePreprocessor(args.era, args.channel, args.masspoint, convSF)

        if args.channel in CHANNELS_WITH_SIGNAL:
            run_signal(preprocessor, workdir, args.era, args.channel,
                       args.masspoint, basedir, syst_categories,
                       input_masspoint_hint=args.masspoint)

        process_backgrounds(workdir, args.era, args.channel, args.masspoint, basedir,
                            preprocessor, config, syst_categories, kfactors)
        process_nonprompt(workdir, args.era, args.channel, args.masspoint, basedir,
                          preprocessor, config)
        process_data(workdir, args.era, args.channel, args.masspoint, basedir,
                     preprocessor, config)

    logging.info("=" * 60)
    logging.info("Preprocessing complete!")
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
