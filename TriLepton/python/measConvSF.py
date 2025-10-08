#!/usr/bin/env python
import os
import argparse
import logging
import json
from re import I
import ROOT
from math import sqrt
import correctionlib.schemav2 as cs

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel (ZG1E2Mu, ZG3Mu, or ZGCombined)")
parser.add_argument("--debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
WORKDIR = os.environ["WORKDIR"]

if args.channel not in ["ZG1E2Mu", "ZG3Mu", "ZGCombined"]:
    raise ValueError(f"Invalid channel: {args.channel}")

if args.era in ["Run2", "2016preVFP", "2016postVFP", "2017", "2018"]:
    RUN = "Run2"
elif args.era in ["Run3", "2022", "2022EE", "2023", "2023BPix"]:
    RUN = "Run3"
else:
    raise ValueError(f"Invalid era: {args.era}")

eralist = [args.era]
if args.era == "Run2":
    eralist = ["2016preVFP", "2016postVFP", "2017", "2018"]
if args.era == "Run3":
    eralist = ["2022", "2022EE", "2023", "2023BPix"]

if args.channel == "ZG1E2Mu":
    FLAG = "Run1E2Mu"
    CHANNELS = ["ZG1E2Mu"]
elif args.channel == "ZG3Mu":
    FLAG = "Run3Mu"
    CHANNELS = ["ZG3Mu"]
elif args.channel == "ZGCombined":
    FLAG = None  # Will handle both FLAGs
    CHANNELS = ["ZG1E2Mu", "ZG3Mu"]

json_samplegroup = json.load(open(f"configs/samplegroup.json"))
json_systematics = json.load(open(f"configs/systematics.json"))

def get_yield_data(channels, era):
    """Get data yield for one or more channels"""
    if isinstance(channels, str):
        channels = [channels]
    
    total_yield = 0.0
    for channel in channels:
        # Determine the correct FLAG for this channel
        if channel == "ZG1E2Mu":
            flag = "Run1E2Mu"
        elif channel == "ZG3Mu":
            flag = "Run3Mu"
        else:
            raise ValueError(f"Unknown channel: {channel}")
        
        DATAPERIODs = json_samplegroup[era][channel.replace("ZG", "")]["data"]
        for sample in DATAPERIODs:
            file_path = f"{WORKDIR}/SKNanoOutput/CRPromptSelector/{flag}/{era}/Skim_TriLep_{sample}.root"
            assert os.path.exists(file_path), f"file {file_path} does not exist"
            f = ROOT.TFile.Open(file_path)
            h = f.Get(f"{channel}/Central/ZCand/mass")
            h.SetDirectory(0)
            # Integrate the histogram to get total yield
            yield_value = h.Integral()
            f.Close()
            total_yield += yield_value
    
    return total_yield

def get_yield_nonprompt(channels, era):
    """Get nonprompt yield for one or more channels"""
    if isinstance(channels, str):
        channels = [channels]
    
    total_yield = 0.0
    for channel in channels:
        # Determine the correct FLAG for this channel
        if channel == "ZG1E2Mu":
            flag = "Run1E2Mu"
        elif channel == "ZG3Mu":
            flag = "Run3Mu"
        else:
            raise ValueError(f"Unknown channel: {channel}")
        
        nonprompt = json_samplegroup[era][channel.replace("ZG", "")]["nonprompt"]
        for sample in nonprompt:
            file_path = f"{WORKDIR}/SKNanoOutput/CRMatrixSelector/{flag}/{era}/Skim_TriLep_{sample}.root"
            assert os.path.exists(file_path), f"file {file_path} does not exist"
            f = ROOT.TFile.Open(file_path)
            h = f.Get(f"{channel}/Central/ZCand/mass")
            h.SetDirectory(0)
            # Integrate the histogram to get total yield
            yield_value = h.Integral()
            f.Close()
            total_yield += yield_value
    
    return total_yield

def get_yield_mc(channels, era, mc, syst="Central"):
    """Get MC yield for one or more channels"""
    if isinstance(channels, str):
        channels = [channels]
    
    total_yield = 0.0
    for channel in channels:
        # Determine the correct FLAG for this channel
        if channel == "ZG1E2Mu":
            flag = "Run1E2Mu"
        elif channel == "ZG3Mu":
            flag = "Run3Mu"
        else:
            raise ValueError(f"Unknown channel: {channel}")
        
        # Get the list of samples for this MC category
        mc_samples = json_samplegroup[era][channel.replace("ZG", "")].get(mc, [])
        if not mc_samples:
            logging.warning(f"No {mc} samples found for {channel} in era {era}")
            continue
            
        for sample in mc_samples:
            file_path = f"{WORKDIR}/SKNanoOutput/CRPromptSelector/{flag}_RunSyst/{era}/Skim_TriLep_{sample}.root"
            if not os.path.exists(file_path):
                logging.warning(f"File {file_path} does not exist, skipping")
                continue
                
            f = ROOT.TFile.Open(file_path)
            
            # Try to get the systematic
            h = f.Get(f"{channel}/{syst}/ZCand/mass")
            if not h:
                if syst != "Central":
                    # For non-Central systematics, skip samples that don't have the systematic
                    logging.warning(f"Systematic {syst} not found for {channel}/{sample}, skipping this sample")
                    f.Close()
                    continue
                else:
                    # For Central systematic, this is an error - the histogram should exist
                    logging.warning(f"Cannot find {channel}/Central/ZCand/mass for sample {sample}")
                    f.Close()
                    continue
            
            h.SetDirectory(0)
            # Integrate the histogram to get total yield
            yield_value = h.Integral()
            f.Close()
            total_yield += yield_value
    
    return total_yield

def get_systematics_for_channel(channels, run):
    """Get systematic variations for the current channel(s) and run"""
    if isinstance(channels, str):
        channels = [channels]
    
    systematics = {}
    
    # For combined channels, merge systematics from both channels
    for channel in channels:
        channel_key = channel.replace("ZG", "")
        if run in json_systematics and channel_key in json_systematics[run]:
            # Merge systematics (union of all systematic sources)
            for syst_name, variations in json_systematics[run][channel_key].items():
                if syst_name not in systematics:
                    systematics[syst_name] = variations
                # If already exists, keep the existing one (they should be identical)
    
    # Add nonprompt systematics if relevant (may not be relevant for conv measurements)
    # We'll keep it for consistency but it might be zero
    systematics["nonprompt"] = ["nonprompt_up", "nonprompt_down"]
    
    # Add prompt systematics (combination of all MC systematics)
    systematics["prompt"] = ["prompt_uncertainty"]
    
    return systematics

def calculate_scale_factor(channels, eralist, syst="Central", run="Run3"):
    """Calculate scale factor for a given systematic variation"""
    total_data = 0.0
    total_conv = 0.0
    total_non_conv = 0.0
    
    # Handle nonprompt systematic variations specially
    is_nonprompt_syst = syst in ["nonprompt_up", "nonprompt_down"]
    mc_syst = "Central" if is_nonprompt_syst else syst
    
    for era in eralist:
        # Data doesn't have systematic variations
        data_yield = get_yield_data(channels, era)
        total_data += data_yield
        
        # Get conversion yield
        conv_yield = get_yield_mc(channels, era, "conv", mc_syst)
        total_conv += conv_yield
        
        # Get all non-conversion MC yields
        # These might include: nonprompt, ZZ, ttX, others, etc.
        # We need to check what's available in the samplegroup.json
        non_conv_categories = []
        
        # Check which categories exist for these channels
        for channel in channels:
            channel_key = channel.replace("ZG", "")
            if era in json_samplegroup and channel_key in json_samplegroup[era]:
                for category in json_samplegroup[era][channel_key].keys():
                    if category not in ["data", "conv"] and category not in non_conv_categories:
                        non_conv_categories.append(category)
        
        # Sum all non-conversion backgrounds
        for category in non_conv_categories:
            if category == "nonprompt":
                # Nonprompt uses CRMatrixSelector
                nonprompt_yield = get_yield_nonprompt(channels, era)
                if is_nonprompt_syst:
                    # Handle nonprompt systematic variation
                    nonprompt_uncertainty = 0.25 if run == "Run2" else 0.35  # 25% for Run2, 35% for Run3
                    if syst == "nonprompt_up":
                        scale_factor = 1.0 + nonprompt_uncertainty
                    elif syst == "nonprompt_down":
                        scale_factor = 1.0 - nonprompt_uncertainty
                    nonprompt_yield *= scale_factor
                    if nonprompt_yield > 0:
                        print(f"  Applied {nonprompt_uncertainty*100:.0f}% variation to nonprompt: scale factor = {scale_factor:.3f}")
                total_non_conv += nonprompt_yield
            else:
                # Normal MC categories from CRPromptSelector
                cat_yield = get_yield_mc(channels, era, category, mc_syst)
                total_non_conv += cat_yield
    
    # Calculate scale factor
    if total_conv > 0:
        sf = (total_data - total_non_conv) / total_conv
    else:
        logging.error("Total conversion yield is zero!")
        sf = 1.0
    
    return sf

def calculate_prompt_uncertainty(sf_central, systematics_data):
    """Calculate symmetric prompt uncertainty: sqrt(sum of squared deviations)"""
    # Calculate systematic uncertainty (symmetric)
    syst_squared = 0.0
    
    # Loop over all systematic sources except nonprompt and prompt
    for syst_name, variations in systematics_data.items():
        if syst_name in ["nonprompt", "prompt"]:
            continue
        
        # For each systematic source, find up and down variations
        max_deviation = 0.0
        
        for var_name, sf_value in variations.items():
            if sf_value is None:
                continue
            
            deviation = abs(sf_value - sf_central)
            max_deviation = max(max_deviation, deviation)
        
        # Add squared deviation
        syst_squared += max_deviation * max_deviation
    
    # Take square root to get total systematic uncertainty
    syst_uncertainty = sqrt(syst_squared)
    
    # Create up and down variations with symmetric uncertainty
    prompt_up = sf_central + syst_uncertainty
    prompt_down = sf_central - syst_uncertainty
    
    return prompt_up, prompt_down

def create_correction(name, description, scale_factor):
    """Create a correctionlib Correction object for a single scale factor"""
    # For a single scale factor, we use a simple formula that returns a constant
    data = cs.Formula(
        nodetype="formula",
        expression=str(scale_factor),
        parser="TFormula",
        variables=[]
    )
    
    # Create the correction
    correction = cs.Correction(
        name=name,
        version=1,
        description=description,
        inputs=[],  # No inputs needed for a constant scale factor
        output=cs.Variable(
            name="sf", 
            type="real", 
            description="Scale factor"
        ),
        data=data
    )
    
    return correction

def main():
    # Calculate central scale factor
    channel_str = "+".join(CHANNELS) if len(CHANNELS) > 1 else CHANNELS[0]
    print(f"Calculating Central scale factor for {channel_str}...")
    sf_central = calculate_scale_factor(CHANNELS, eralist, "Central", RUN)
    print(f"Central SF = {sf_central:.4f}")
    
    # Get systematic variations for this channel/run
    systematics = get_systematics_for_channel(CHANNELS, RUN)
    
    # Store all systematic variations  
    systematics_data = {}
    
    # Calculate scale factors for each systematic variation
    for syst_name, variations in systematics.items():
        if syst_name == "prompt":
            # Skip prompt for now, will calculate it after all other systematics
            continue
            
        print(f"Calculating {syst_name} systematic variations...")
        systematics_data[syst_name] = {}
        
        for variation in variations:
            try:
                print(f"  Processing {variation}...")
                sf_syst = calculate_scale_factor(CHANNELS, eralist, variation, RUN)
                systematics_data[syst_name][variation] = sf_syst
                print(f"  {variation} SF = {sf_syst:.4f}")
            except Exception as e:
                print(f"  WARNING: Failed to process {variation}: {e}")
                # Set to None to indicate missing data
                systematics_data[syst_name][variation] = None
    
    # Calculate prompt systematic uncertainty (combination of all MC systematics)
    print("Calculating prompt systematic variations...")
    try:
        prompt_up, prompt_down = calculate_prompt_uncertainty(
            sf_central, 
            systematics_data
        )
        systematics_data["prompt"] = {
            "prompt_up": prompt_up,
            "prompt_down": prompt_down
        }
        print(f"  prompt_up SF = {prompt_up:.4f}")
        print(f"  prompt_down SF = {prompt_down:.4f}")
        print("  Combined symmetric uncertainties from all systematic sources")
    except Exception as e:
        print(f"  WARNING: Failed to calculate prompt variations: {e}")
        systematics_data["prompt"] = {
            "prompt_up": None,
            "prompt_down": None
        }
    
    # Create correctionlib format
    corrections = []
    
    # Central correction
    central_corr = create_correction(
        name=f"ConvSF_{args.channel}_{args.era}_Central",
        description=f"Conversion scale factor for {args.channel} {args.era} (Central)",
        scale_factor=sf_central
    )
    corrections.append(central_corr)
    
    # Systematic corrections
    for syst_name, variations in systematics_data.items():
        for var_name, sf_value in variations.items():
            if sf_value is None:
                continue
            syst_corr = create_correction(
                name=f"ConvSF_{args.channel}_{args.era}_{var_name}",
                description=f"Conversion scale factor for {args.channel} {args.era} ({var_name})",
                scale_factor=sf_value
            )
            corrections.append(syst_corr)
    
    # Create correction set
    nonprompt_uncertainty = 0.25 if RUN == "Run2" else 0.35
    description = f"Conversion scale factors for {args.channel} {args.era}"
    if len(CHANNELS) > 1:
        description += f" (combined {'+'.join(CHANNELS)})"
    description += f". Measured in ZG control region. Nonprompt uncertainty: ±{nonprompt_uncertainty*100:.0f}%"
    
    cset = cs.CorrectionSet(
        schema_version=2,
        description=description,
        corrections=corrections
    )
    
    # Save the file in correctionlib JSON format
    OUTPUTPATH = f"{WORKDIR}/TriLepton/results/{args.channel}/{args.era}/ConvSF.json"
    os.makedirs(os.path.dirname(OUTPUTPATH), exist_ok=True)
    
    # Save to JSON
    with open(OUTPUTPATH, 'w') as f:
        f.write(cset.model_dump_json(exclude_unset=True, indent=2))
    
    print(f"\nCorrectionlib-compatible scale factors saved to: {OUTPUTPATH}")
    print(f"Created {len(corrections)} corrections ({len(systematics)} systematic sources)")
    print(f"Combined channels: {'+'.join(CHANNELS) if len(CHANNELS) > 1 else CHANNELS[0]}")
    print(f"Nonprompt uncertainty: ±{nonprompt_uncertainty*100:.0f}% ({RUN})")
    if len(CHANNELS) > 1:
        print(f"Statistical precision improved by combining {len(CHANNELS)} channels")

if __name__ == "__main__":
    main()
