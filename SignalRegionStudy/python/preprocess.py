#!/usr/bin/env python3
import os, shutil
import argparse
import logging
import json
import ROOT

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel (SR1E2Mu or SR3Mu)")
parser.add_argument("--masspoint", required=True, type=str, help="signal mass point")
parser.add_argument("--method", required=True, type=str, help="method (Baseline or ParticleNet)")
parser.add_argument("--debug", action="store_true", help="debug mode")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
WORKDIR = os.getenv("WORKDIR")
BASEDIR = f"{WORKDIR}/SignalRegionStudy/samples/{args.era}/{args.channel}/{args.masspoint}/{args.method}"

# Load SignalRegionStudy library
ROOT.gSystem.Load(f"{WORKDIR}/SignalRegionStudy/lib/libSignalRegionStudy.so")

# Load JSON configurations
json_samplegroups = json.load(open(f"{WORKDIR}/SignalRegionStudy/configs/samplegroups.json"))
json_systematics = json.load(open(f"{WORKDIR}/SignalRegionStudy/configs/systematics.json"))

# Helper function to load ConvSF from TriLepton results
def load_convSF(era, channel):
    """
    Load conversion scale factor from TriLepton ZG results.
    Returns (central_sf, envelope_error)
    """
    convSF_file = f"{WORKDIR}/TriLepton/results/{channel.replace('SR', 'ZG')}/{era}/ConvSF.json"

    if not os.path.exists(convSF_file):
        logging.warning(f"ConvSF file not found: {convSF_file}")
        logging.warning("Using default ConvSF = 1.0 ± 0.3")
        return 1.0, 0.3

    with open(convSF_file) as f:
        data = json.load(f)

    # Extract Central value
    central_corrections = [c for c in data["corrections"]
                           if c["name"].endswith("_Central")]
    if not central_corrections:
        logging.error(f"No Central correction found in {convSF_file}")
        return 1.0, 0.3

    central_sf = float(central_corrections[0]["data"]["expression"])

    # Compute envelope from all systematics (exclude nonprompt/prompt special variations)
    syst_sfs = [float(c["data"]["expression"])
                for c in data["corrections"]
                if not c["name"].endswith("_Central")
                and "nonprompt" not in c["name"].lower()
                and "prompt" not in c["name"].lower()]

    if syst_sfs:
        sf_err = max(abs(central_sf - min(syst_sfs)),
                     abs(max(syst_sfs) - central_sf))
    else:
        # Fallback: use 30% uncertainty
        sf_err = 0.3 * central_sf

    logging.info(f"Loaded ConvSF for {era} {channel}: {central_sf:.4f} ± {sf_err:.4f}")
    return central_sf, sf_err

# Load conversion scale factors
convSF, convSFerr = load_convSF(args.era, args.channel)

# Determine if sample is in training region (80 < mA < 100)
mA = int(args.masspoint.split("_")[1].replace("MA", ""))
isTrainedSample = (80 < mA < 100)

# Determine run period (for systematics configuration)
if args.era in ["2016preVFP", "2016postVFP", "2017", "2018"]:
    RUN = "Run2"
elif args.era in ["2022", "2022EE", "2023", "2023BPix"]:
    RUN = "Run3"
else:
    raise ValueError(f"Invalid era: {args.era}")

# Load sample lists from JSON config
DATAPERIODs = json_samplegroups[args.era][args.channel]["data"]
NONPROMPTSAMPLEs = json_samplegroups[args.era][args.channel]["nonprompt"]

## Helper functions
def hadd(file_path):
    """Hadd multiple files if output doesn't exist"""
    if not os.path.exists(file_path):
        dir_path = os.path.dirname(file_path)
        if os.path.exists(dir_path) and len(os.listdir(dir_path)) > 0:
            logging.info(f"Hadding files in {dir_path}...")
            os.system(f"hadd -f {file_path} {dir_path}/*")
        else:
            logging.warning(f"No files to hadd in {dir_path}")

def getSampleAlias(sample):
    """Map full sample names to short aliases"""
    return json_samplegroups["aliases"].get(sample, sample)

def get_systematics_list(syst_dict):
    """Flatten systematics dictionary into list of tuples"""
    syst_list = [("Central",)]

    for syst_name, syst_config in syst_dict.items():
        if isinstance(syst_config, list):
            # Simple list of variations
            syst_list.append(tuple(syst_config))
        elif isinstance(syst_config, dict):
            # Complex systematic with pattern
            if "count" in syst_config:
                # e.g., PDFReweight with count
                pattern = syst_config["pattern"]
                count = syst_config["count"]
                syst_list.append(tuple([pattern.format(i=i) for i in range(count)]))
            elif "indices" in syst_config:
                # e.g., ScaleVar with specific indices
                pattern = syst_config["pattern"]
                indices = syst_config["indices"]
                syst_list.append(tuple([pattern.format(i=i) for i in indices]))

    return syst_list

def hadd_and_cleanup(output_file, input_files, category_name):
    """
    Merge ROOT files using hadd and cleanup intermediate files.

    Args:
        output_file: Path to merged output file
        input_files: List of input file paths to merge
        category_name: Category name for logging
    """
    # Filter to existing files
    existing_files = [f for f in input_files if os.path.exists(f)]

    if not existing_files:
        logging.warning(f"No {category_name} samples found to hadd")
        return

    # Build and execute hadd command
    hadd_cmd = f"hadd -f {output_file} " + " ".join(existing_files)
    logging.info(f"Merging {len(existing_files)} {category_name} files...")
    os.system(hadd_cmd)

    # Clean up intermediate files
    for f in existing_files:
        os.remove(f)
        logging.debug(f"Removed {f}")

def process_samples_with_systematics(processor, samples, category_name, input_base_path,
                                     systematics, isTrainedSample, masspoint,
                                     applyConvSF=False, use_alias=True):
    """
    Process samples with full systematic variations.

    Args:
        processor: ROOT.Preprocessor instance
        samples: List of sample names
        category_name: Category name for output (e.g., "conversion", "others")
        input_base_path: Base path template for input files
        systematics: List of systematic variations
        isTrainedSample: Whether sample is in training region
        masspoint: Signal mass point
        applyConvSF: Whether to apply conversion scale factor
        use_alias: Whether to use sample aliases

    Returns:
        List of output file paths
    """
    output_files = []

    for sample in samples:
        input_path = f"{input_base_path}/Skim_TriLep_{sample}.root"
        output_path = f"{BASEDIR}/{sample}.root"

        if not os.path.exists(input_path):
            logging.warning(f"{category_name.capitalize()} sample not found: {input_path}")
            continue

        processor.setInputFile(input_path)
        processor.setOutputFile(output_path)

        # Process all systematics
        for syst in [syst for systset in systematics for syst in systset]:
            processor.setInputTree(syst)
            output_name = getSampleAlias(sample) if use_alias else category_name
            processor.fillOutTree(output_name, masspoint, syst, applyConvSF=applyConvSF, isTrainedSample=isTrainedSample)
            processor.saveTree()

        processor.closeInputFile()
        processor.closeOutputFile()
        output_files.append(output_path)

    # Rename to aliases if needed
    if use_alias:
        renamed_files = []
        for sample in samples:
            alias = getSampleAlias(sample)
            if alias != sample and os.path.exists(f"{BASEDIR}/{sample}.root"):
                alias_path = f"{BASEDIR}/{alias}.root"
                shutil.move(f"{BASEDIR}/{sample}.root", alias_path)
                renamed_files.append(alias_path)
            elif os.path.exists(f"{BASEDIR}/{alias}.root"):
                renamed_files.append(f"{BASEDIR}/{alias}.root")
        return renamed_files

    return output_files

def process_nonprompt_samples(processor, samples, input_channel, masspoint, run_period, isTrainedSample):
    """
    Process nonprompt samples with Central/Up/Down variations.

    Args:
        processor: ROOT.Preprocessor instance
        samples: List of sample names (data periods)
        input_channel: Input channel name
        masspoint: Signal mass point
        run_period: "Run2" or "Run3" for weight scaling
        isTrainedSample: Whether sample is in training region

    Returns:
        List of output file paths
    """
    # Determine weight scale based on run period
    if run_period == "Run2":
        weight_up, weight_down = 1.25, 0.75
    else:  # Run3
        weight_up, weight_down = 1.15, 0.85

    output_files = []

    for sample in samples:
        # Set datastream for this sample
        datastream = sample.split("_")[0]
        processor.setDatastream(datastream)

        input_path = f"{WORKDIR}/SKNanoOutput/MatrixTreeProducer/{input_channel}/{args.era}/Skim_TriLep_{sample}.root"
        output_path = f"{BASEDIR}/{sample}_nonprompt.root"

        if not os.path.exists(input_path):
            logging.warning(f"Nonprompt sample not found: {input_path}")
            continue

        processor.setInputFile(input_path)
        processor.setOutputFile(output_path)
        processor.setInputTree("Central")

        # Central
        processor.fillOutTree("nonprompt", masspoint, "Central", applyConvSF=False,
                            isTrainedSample=isTrainedSample, weightScale=1.0)
        processor.saveTree()

        # Nonprompt_Up
        processor.fillOutTree("nonprompt", masspoint, "Nonprompt_Up", applyConvSF=False,
                            isTrainedSample=isTrainedSample, weightScale=weight_up)
        processor.saveTree()

        # Nonprompt_Down
        processor.fillOutTree("nonprompt", masspoint, "Nonprompt_Down", applyConvSF=False,
                            isTrainedSample=isTrainedSample, weightScale=weight_down)
        processor.saveTree()

        processor.closeInputFile()
        processor.closeOutputFile()
        output_files.append(output_path)

    return output_files

def process_data_samples(processor, samples, input_channel, masspoint, isTrainedSample):
    """
    Process data samples (Central only, no systematics).

    Args:
        processor: ROOT.Preprocessor instance
        samples: List of data period names
        input_channel: Input channel name
        masspoint: Signal mass point
        isTrainedSample: Whether sample is in training region

    Returns:
        List of output file paths
    """
    output_files = []

    for sample in samples:
        # Set datastream for this sample
        datastream = sample.split("_")[0]
        processor.setDatastream(datastream)

        input_path = f"{WORKDIR}/SKNanoOutput/PromptTreeProducer/{input_channel}/{args.era}/Skim_TriLep_{sample}.root"
        output_path = f"{BASEDIR}/{sample}_data.root"

        if not os.path.exists(input_path):
            logging.warning(f"Data sample not found: {input_path}")
            continue

        processor.setInputFile(input_path)
        processor.setOutputFile(output_path)
        processor.setInputTree("Central")
        processor.fillOutTree("data", masspoint, "Central", applyConvSF=False, isTrainedSample=isTrainedSample)
        processor.saveTree()
        processor.closeInputFile()
        processor.closeOutputFile()
        output_files.append(output_path)

    return output_files

## Load systematics from JSON
def collect_preprocessed_systematics(syst_config):
    """
    Collect all systematics with source='preprocessed' from configuration.
    Returns a list suitable for get_systematics_list().
    """
    preprocessed = {}
    for category in syst_config.values():
        for syst_name, syst_props in category.items():
            if syst_props.get("source") == "preprocessed":
                preprocessed[syst_name] = syst_props.get("variations", [])
    return preprocessed

# Collect experimental systematics (prompt systematics)
channel_config = json_systematics[RUN][args.channel]
experimental_systs = {k: v.get("variations", [])
                      for k, v in channel_config.get("experimental", {}).items()
                      if v.get("source") == "preprocessed"}
promptSysts = get_systematics_list(experimental_systs)

# Collect theory systematics (for signal only, not yet implemented in RunSyst)
theory_systs = {k: v.get("variations", [])
                for k, v in channel_config.get("theory", {}).items()
                if v.get("source") == "preprocessed"}
theorySysts = get_systematics_list(theory_systs)

if __name__ == "__main__":
    logging.info(f"Preprocessing signal {args.masspoint} for {args.era} era and {args.channel} channel")
    if os.path.exists(BASEDIR):
        logging.info(f"Removing existing directory {BASEDIR}")
        shutil.rmtree(BASEDIR)
    os.makedirs(BASEDIR, exist_ok=True)

    # Construct input channel (Run1E2Mu, Run3Mu)
    input_channel = args.channel.replace("SR", "Run")

    # Preprocessor expects Skim channel naming
    preprocess_channel = args.channel.replace("SR", "Skim")

    # Initialize processor without datastream (will be set per-file)
    # Use first data sample to infer initial datastream
    initial_datastream = DATAPERIODs[0].split("_")[0] if DATAPERIODs else "MuonEG"
    processor = ROOT.Preprocessor(args.era, preprocess_channel, initial_datastream)
    processor.setConvSF(convSF, convSFerr)

    ## Signal
    logging.info(f"Processing signal {args.masspoint}")
    input_path = f"{WORKDIR}/SKNanoOutput/PromptTreeProducer/{input_channel}_RunSyst/{args.era}/TTToHcToWAToMuMu-{args.masspoint}.root"
    output_path = f"{BASEDIR}/{args.masspoint}.root"
    assert os.path.exists(input_path), f"Input file {input_path} does not exist"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    processor.setInputFile(input_path)
    processor.setOutputFile(output_path)

    # Process only prompt systematics (theory systematics not yet implemented)
    for syst in [syst for systset in promptSysts for syst in systset]:
        processor.setInputTree(syst)
        processor.fillOutTree(args.masspoint, args.masspoint, syst, applyConvSF=False, isTrainedSample=isTrainedSample)
        processor.saveTree()

    # TODO: Theory systematics will be added when implemented in RunSyst directory
    # for syst in [syst for systset in theorySysts for syst in systset]:
    #     processor.setInputTree(syst)
    #     processor.fillOutTree(args.masspoint, args.masspoint, syst, applyConvSF=False, isTrainedSample=isTrainedSample)
    #     processor.saveTree()

    processor.closeInputFile()
    processor.closeOutputFile()

    ## Nonprompt
    logging.info("Processing nonprompt...")
    nonprompt_files = process_nonprompt_samples(
        processor, NONPROMPTSAMPLEs, input_channel, args.masspoint, RUN, isTrainedSample
    )
    hadd_and_cleanup(f"{BASEDIR}/nonprompt.root", nonprompt_files, "nonprompt")

    ## Conversion
    logging.info("Processing conversion...")
    CONVBKGs = json_samplegroups[args.era][args.channel]["conv"]
    input_base_path = f"{WORKDIR}/SKNanoOutput/PromptTreeProducer/{input_channel}_RunSyst/{args.era}"
    conversion_files = process_samples_with_systematics(
        processor, CONVBKGs, "conversion", input_base_path, promptSysts,
        isTrainedSample, args.masspoint, applyConvSF=True, use_alias=False
    )
    hadd_and_cleanup(f"{BASEDIR}/conversion.root", conversion_files, "conversion")

    ## Diboson
    logging.info("Processing diboson...")
    DIBOSONBKGs = json_samplegroups[args.era][args.channel]["diboson"]
    diboson_files = process_samples_with_systematics(
        processor, DIBOSONBKGs, "diboson", input_base_path, promptSysts,
        isTrainedSample, args.masspoint, applyConvSF=False, use_alias=True
    )
    hadd_and_cleanup(f"{BASEDIR}/diboson.root", diboson_files, "diboson")

    ## ttX
    logging.info("Processing ttX...")
    TTXBKGs = json_samplegroups[args.era][args.channel]["ttX"]
    ttX_files = process_samples_with_systematics(
        processor, TTXBKGs, "ttX", input_base_path, promptSysts,
        isTrainedSample, args.masspoint, applyConvSF=False, use_alias=True
    )
    hadd_and_cleanup(f"{BASEDIR}/ttX.root", ttX_files, "ttX")

    ## Others
    logging.info("Processing others...")
    OTHERSBKGs = json_samplegroups[args.era][args.channel]["others"]
    others_files = process_samples_with_systematics(
        processor, OTHERSBKGs, "others", input_base_path, promptSysts,
        isTrainedSample, args.masspoint, applyConvSF=False, use_alias=False
    )
    hadd_and_cleanup(f"{BASEDIR}/others.root", others_files, "others")

    ## Data (for future unblinding)
    logging.info("Processing data...")
    data_files = process_data_samples(
        processor, DATAPERIODs, input_channel, args.masspoint, isTrainedSample
    )
    hadd_and_cleanup(f"{BASEDIR}/data.root", data_files, "data")

    logging.info(f"Preprocessing complete! Output: {BASEDIR}")
