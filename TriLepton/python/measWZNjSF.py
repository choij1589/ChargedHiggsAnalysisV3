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
parser.add_argument("--channel", required=True, type=str, help="channel (WZ1E2Mu, WZ3Mu, or WZCombined)")
parser.add_argument("--debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
WORKDIR = os.environ["WORKDIR"]

if args.channel not in ["WZ1E2Mu", "WZ3Mu", "WZCombined"]:
    raise ValueError(f"Invalid channel: {args.channel}")

if args.era in ["Run2", "2016preVFP", "2016postVFP", "2017", "2018"]:
    RUN = "Run2"
    samplename_WZ = "WZTo3LNu_amcatnlo"
    max_nj = 5
elif args.era in ["Run3", "2022", "2022EE", "2023", "2023BPix"]:
    RUN = "Run3"
    samplename_WZ = "WZTo3LNu_powheg"
    max_nj = 3
else:
    raise ValueError(f"Invalid era: {args.era}")

eralist = [args.era]
if args.era == "Run2":
    eralist = ["2016preVFP", "2016postVFP", "2017", "2018"]
if args.era == "Run3":
    eralist = ["2022", "2022EE", "2023", "2023BPix"]

if args.channel == "WZ1E2Mu":
    FLAG = "Run1E2Mu"
    CHANNELS = ["WZ1E2Mu"]
elif args.channel == "WZ3Mu":
    FLAG = "Run3Mu"
    CHANNELS = ["WZ3Mu"]
elif args.channel == "WZCombined":
    FLAG = None  # Will handle both FLAGs
    CHANNELS = ["WZ1E2Mu", "WZ3Mu"]

json_samplegroup = json.load(open(f"configs/samplegroup.json"))
json_systematics = json.load(open(f"configs/systematics.json"))

def get_hist_data(channels, era):
    """Get data histogram for one or more channels"""
    if isinstance(channels, str):
        channels = [channels]
    
    hist = None
    for channel in channels:
        # Determine the correct FLAG for this channel
        if channel == "WZ1E2Mu":
            flag = "Run1E2Mu"
        elif channel == "WZ3Mu":
            flag = "Run3Mu"
        else:
            raise ValueError(f"Unknown channel: {channel}")
        
        DATAPERIODs = json_samplegroup[era][channel.replace("WZ", "")]["data"]
        for sample in DATAPERIODs:
            file_path = f"{WORKDIR}/SKNanoOutput/CRPromptSelector/{flag}/{era}/Skim_TriLep_{sample}.root"
            assert os.path.exists(file_path), f"file {file_path} does not exist"
            f = ROOT.TFile.Open(file_path)
            h = f.Get(f"{channel}/Central/jets/size")
            h.SetDirectory(0)
            f.Close()
            if hist is None:
                hist = h.Clone("hist")
            else:
                hist.Add(h)
    hist.SetDirectory(0)
    return hist

def get_hist_nonprompt(channels, era):
    """Get nonprompt histogram for one or more channels"""
    if isinstance(channels, str):
        channels = [channels]
    
    hist = None
    for channel in channels:
        # Determine the correct FLAG for this channel
        if channel == "WZ1E2Mu":
            flag = "Run1E2Mu"
        elif channel == "WZ3Mu":
            flag = "Run3Mu"
        else:
            raise ValueError(f"Unknown channel: {channel}")
        
        nonprompt = json_samplegroup[era][channel.replace("WZ", "")]["nonprompt"]
        for sample in nonprompt:
            file_path = f"{WORKDIR}/SKNanoOutput/CRMatrixSelector/{flag}/{era}/Skim_TriLep_{sample}.root"
            assert os.path.exists(file_path), f"file {file_path} does not exist"
            f = ROOT.TFile.Open(file_path)
            h = f.Get(f"{channel}/Central/jets/size")
            h.SetDirectory(0)
            f.Close()
            if hist is None:
                hist = h.Clone("hist")
                hist.SetDirectory(0)
            else:
                hist.Add(h)
    hist.SetDirectory(0)
    return hist

def get_hist_mc(channels, era, mc, syst="Central"):
    """Get MC histogram for one or more channels"""
    if isinstance(channels, str):
        channels = [channels]
    
    hist = None
    for channel in channels:
        # Determine the correct FLAG for this channel
        if channel == "WZ1E2Mu":
            flag = "Run1E2Mu"
        elif channel == "WZ3Mu":
            flag = "Run3Mu"
        else:
            raise ValueError(f"Unknown channel: {channel}")
        
        pred = json_samplegroup[era][channel.replace("WZ", "")][mc]
        for sample in pred:
            file_path = f"{WORKDIR}/SKNanoOutput/CRPromptSelector/{flag}_RunSyst/{era}/Skim_TriLep_{sample}.root"
            assert os.path.exists(file_path), f"file {file_path} does not exist"
            f = ROOT.TFile.Open(file_path)
            
            # Try to get the systematic
            h = f.Get(f"{channel}/{syst}/jets/size")
            if not h:
                if syst != "Central":
                    # For non-Central systematics, try Central as fallback
                    h = f.Get(f"{channel}/Central/jets/size")
                    if h:
                        logging.info(f"Systematic {syst} not found for {channel}/{sample}, using Central")
                    else:
                        logging.warning(f"Cannot find {channel}/Central/jets/size for sample {sample}")
                        f.Close()
                        continue
                else:
                    # For Central systematic, this is an error - the histogram should exist
                    logging.warning(f"Cannot find {channel}/Central/jets/size for sample {sample}")
                    f.Close()
                    continue
                
            h.SetDirectory(0)
            f.Close()
            if hist is None:
                hist = h.Clone("hist")
                hist.SetDirectory(0)
            else:
                hist.Add(h)
    if hist is not None:
        hist.SetDirectory(0)
    return hist

def get_hist_by_name(channels, era, name, syst="Central"):
    """Get histogram by sample name for one or more channels"""
    if isinstance(channels, str):
        channels = [channels]
    
    hist = None
    for channel in channels:
        # Determine the correct FLAG for this channel
        if channel == "WZ1E2Mu":
            flag = "Run1E2Mu"
            if name == samplename_WZ:
                flag += "_RunNoWZSF"
        elif channel == "WZ3Mu":
            flag = "Run3Mu"
            if name == samplename_WZ:
                flag += "_RunNoWZSF"
        else:
            raise ValueError(f"Unknown channel: {channel}")
       
        file_path = f"{WORKDIR}/SKNanoOutput/CRPromptSelector/{flag}_RunSyst/{era}/Skim_TriLep_{name}.root"
        assert os.path.exists(file_path), f"file {file_path} does not exist"
        f = ROOT.TFile.Open(file_path)
        
        # Try to get the systematic
        h = f.Get(f"{channel}/{syst}/jets/size")
        if not h:
            if syst != "Central":
                # For non-Central systematics, try Central as fallback
                h = f.Get(f"{channel}/Central/jets/size")
                if h:
                    logging.info(f"Systematic {syst} not found for {channel}, using Central")
                else:
                    logging.warning(f"Cannot find {channel}/Central/jets/size for sample {name}")
                    f.Close()
                    continue
            else:
                # For Central systematic, this is an error - the histogram should exist
                logging.warning(f"Cannot find {channel}/Central/jets/size for sample {name}")
                f.Close()
                return None
        
        h.SetDirectory(0)
        f.Close()
        
        if hist is None:
            hist = h.Clone("hist")
        else:
            hist.Add(h)
    if hist is not None:
        hist.SetDirectory(0)
    return hist

def add_hist(name, target, hist):
    if target is None:
        target = hist.Clone(name)
    else:
        target.Add(hist)
    target.SetDirectory(0)
    return target

def merge_high_njet_bins(hist, max_nj):
    """Merge bins for nJets >= max_nj into a single bin at max_nj"""
    if hist is None:
        return None
    
    # Clone the histogram to avoid modifying the original
    merged_hist = hist.Clone(hist.GetName() + "_merged")
    
    # Get the bin corresponding to max_nj (assuming bins start from 0 jets)
    # Bin 1 = 0 jets, Bin 2 = 1 jet, ..., Bin (max_nj+1) = max_nj jets
    target_bin = max_nj + 1
    
    # Sum content and errors from bins >= max_nj
    total_content = 0
    total_error_sq = 0
    
    for bin_idx in range(target_bin, merged_hist.GetNbinsX() + 1):
        content = merged_hist.GetBinContent(bin_idx)
        error = merged_hist.GetBinError(bin_idx)
        
        if bin_idx == target_bin:
            # Keep the content in the target bin
            total_content += content
            total_error_sq += error * error
        else:
            # Add content to target bin and zero out this bin
            total_content += content
            total_error_sq += error * error
            merged_hist.SetBinContent(bin_idx, 0)
            merged_hist.SetBinError(bin_idx, 0)
    
    # Set the merged content and error in the target bin
    merged_hist.SetBinContent(target_bin, total_content)
    merged_hist.SetBinError(target_bin, sqrt(total_error_sq))
    
    return merged_hist

def get_systematics_for_channel(channels, run):
    """Get systematic variations for the current channel(s) and run"""
    if isinstance(channels, str):
        channels = [channels]
    
    systematics = {}
    
    if len(channels) == 1:
        # Single channel: use all systematics from that channel
        channel_key = channels[0].replace("WZ", "")
        if run in json_systematics and channel_key in json_systematics[run]:
            systematics = json_systematics[run][channel_key].copy()
    else:
        # Combined channels: use union of all systematics
        # Channel-specific systematics will use Central values for irrelevant channels
        all_systematics = set()
        channel_systematics = {}
        
        for channel in channels:
            channel_key = channel.replace("WZ", "")
            if run in json_systematics and channel_key in json_systematics[run]:
                channel_systematics[channel_key] = json_systematics[run][channel_key]
                all_systematics.update(json_systematics[run][channel_key].keys())
        
        # Add all systematics from all channels
        for syst_name in all_systematics:
            # Use the systematic definition from any channel that has it
            for channel_key, syst_dict in channel_systematics.items():
                if syst_name in syst_dict:
                    systematics[syst_name] = syst_dict[syst_name]
                    break
        
        # Report which systematics are channel-specific
        common_systematics = set(channel_systematics[list(channel_systematics.keys())[0]].keys())
        for channel_key, syst_dict in list(channel_systematics.items())[1:]:
            common_systematics = common_systematics.intersection(set(syst_dict.keys()))
        
        channel_specific = all_systematics - common_systematics
        if channel_specific:
            channel_names = [ch.replace("WZ", "") for ch in channels]
            print(f"Channel-specific systematics for {'+'.join(channel_names)} (using Central for irrelevant channels): {sorted(channel_specific)}")
    
    # Add nonprompt systematics (manual variations)
    nonprompt_uncertainty = 0.25 if run == "Run2" else 0.35  # 25% for Run2, 35% for Run3
    systematics["nonprompt"] = ["nonprompt_up", "nonprompt_down"]
    
    # Add prompt systematics (combination of all MC systematics + statistical)
    systematics["prompt"] = ["prompt_uncertainty"]
    
    return systematics

def calculate_scale_factors(channels, eralist, samplename_WZ, max_nj, syst="Central", run="Run3"):
    """Calculate scale factors for a given systematic variation"""
    h_data_total = None
    h_nonprompt_total = None
    h_WZ_total = None
    h_ZZ_total = None
    h_conv_total = None
    h_ttX_total = None
    h_others_total = None
    
    # Handle nonprompt systematic variations specially
    is_nonprompt_syst = syst in ["nonprompt_up", "nonprompt_down"]
    mc_syst = "Central" if is_nonprompt_syst else syst
    
    for era in eralist:
        # Data doesn't have systematic variations
        h_data = get_hist_data(channels, era); h_data_total = add_hist("data", h_data_total, h_data)
        h_nonprompt = get_hist_nonprompt(channels, era); h_nonprompt_total = add_hist("nonprompt", h_nonprompt_total, h_nonprompt)
        
        # MC samples use Central for nonprompt systematics, otherwise use the specified systematic
        h_WZ = get_hist_by_name(channels, era, samplename_WZ, mc_syst); h_WZ_total = add_hist("WZ", h_WZ_total, h_WZ)
        h_ZZ = get_hist_by_name(channels, era, "ZZTo4L_powheg", mc_syst); h_ZZ_total = add_hist("ZZ", h_ZZ_total, h_ZZ)
        h_conv = get_hist_mc(channels, era, "conv", mc_syst); h_conv_total = add_hist("conv", h_conv_total, h_conv)
        h_ttX = get_hist_mc(channels, era, "ttX", mc_syst); h_ttX_total = add_hist("ttX", h_ttX_total, h_ttX)
        h_others = get_hist_mc(channels, era, "others", mc_syst); h_others_total = add_hist("others", h_others_total, h_others)
    
    # Apply nonprompt systematic variation if needed
    if is_nonprompt_syst:
        nonprompt_uncertainty = 0.25 if run == "Run2" else 0.35  # 25% for Run2, 35% for Run3
        if syst == "nonprompt_up":
            scale_factor = 1.0 + nonprompt_uncertainty
        elif syst == "nonprompt_down":
            scale_factor = 1.0 - nonprompt_uncertainty
        
        # Scale the nonprompt histogram
        h_nonprompt_total.Scale(scale_factor)
        print(f"  Applied {nonprompt_uncertainty*100:.0f}% variation to nonprompt: scale factor = {scale_factor:.3f}")
    
    # Merge high nJet bins for all histograms
    if syst == "Central":
        print(f"Merging bins for nJets >= {max_nj}")
    h_data_total = merge_high_njet_bins(h_data_total, max_nj)
    h_nonprompt_total = merge_high_njet_bins(h_nonprompt_total, max_nj)
    h_WZ_total = merge_high_njet_bins(h_WZ_total, max_nj)
    h_ZZ_total = merge_high_njet_bins(h_ZZ_total, max_nj)
    h_conv_total = merge_high_njet_bins(h_conv_total, max_nj)
    h_ttX_total = merge_high_njet_bins(h_ttX_total, max_nj)
    h_others_total = merge_high_njet_bins(h_others_total, max_nj)
        
    # Subtract bkgs from data
    SF = h_data_total.Clone("SF")
    SF.Add(h_nonprompt_total, -1)
    SF.Add(h_ZZ_total, -1)
    SF.Add(h_conv_total, -1)
    SF.Add(h_ttX_total, -1)
    SF.Add(h_others_total, -1)
    SF.Divide(h_WZ_total)
    
    return SF

def extract_histogram_data(SF, max_nj):
    """Extract histogram data for correctionlib format"""
    bin_contents = []
    bin_labels = []
    
    # Extract bin data and create labels
    for i in range(1, SF.GetNbinsX() + 1):
        content = SF.GetBinContent(i)
        
        # Only include bins with non-zero content (skip merged empty bins)
        if content != 0 or i <= max_nj + 1:
            bin_contents.append(float(content))
            
            # Create appropriate bin labels
            njet = i - 1  # Bin 1 = 0 jets, Bin 2 = 1 jet, etc.
            bin_labels.append(f"{min(njet, max_nj)}j")
    
    return bin_contents, bin_labels

def create_correction(name, description, bin_contents, bin_labels):
    """Create a correctionlib Correction object using binning with clamping"""
    # Create bin edges for jet multiplicity
    # For n bins (0j, 1j, 2j, 3j), we need n+1 edges
    nbins = len(bin_contents)
    edges = list(range(nbins + 1))  # [0, 1, 2, 3, 4] for 4 bins
    
    # Use Binning with clamping for overflow
    data = cs.Binning(
        nodetype="binning",
        input="njets",
        edges=edges,
        content=[float(value) for value in bin_contents],
        flow="clamp"  # Values >= max edge use the last bin
    )
    
    # Create the correction
    correction = cs.Correction(
        name=name,
        version=1,
        description=description,
        inputs=[
            cs.Variable(
                name="njets", 
                type="real", 
                description="Number of jets (integer, overflow uses highest bin with clamping)"
            )
        ],
        output=cs.Variable(
            name="sf", 
            type="real", 
            description="Scale factor"
        ),
        data=data
    )
    
    return correction

def calculate_prompt_uncertainty(sf_central_data, systematics_data, max_nj):
    """Calculate symmetric prompt uncertainty: sqrt(stat^2 + max(up, down)^2) for each systematic"""
    # Get central values and bin labels
    central_contents, bin_labels = sf_central_data
    nbins = len(central_contents)
    
    # Calculate systematic uncertainty for each bin (symmetric)
    syst_squared = [0.0] * nbins
    
    # Loop over all systematic sources except nonprompt and prompt
    for syst_name, variations in systematics_data.items():
        if syst_name in ["nonprompt", "prompt"]:
            continue
        
        # For each systematic source, find up and down variations
        up_variations = []
        down_variations = []
        
        for var_name, var_data in variations.items():
            if var_data is None:
                continue
                
            var_contents, _ = var_data
            
            if "_Up" in var_name or "up" in var_name:
                up_variations.append(var_contents)
            elif "_Down" in var_name or "down" in var_name:
                down_variations.append(var_contents)
            else:
                # For symmetric variations, treat as both up and down
                up_variations.append(var_contents)
                down_variations.append(var_contents)
        
        # Calculate maximum deviation for this systematic source
        for i in range(nbins):
            max_up_diff = 0.0
            max_down_diff = 0.0
            
            # Find maximum up deviation
            for up_var in up_variations:
                diff = abs(up_var[i] - central_contents[i])
                max_up_diff = max(max_up_diff, diff)
            
            # Find maximum down deviation  
            for down_var in down_variations:
                diff = abs(down_var[i] - central_contents[i])
                max_down_diff = max(max_down_diff, diff)
            
            # Take the maximum of up and down deviations for symmetric uncertainty
            max_deviation = max(max_up_diff, max_down_diff)
            syst_squared[i] += max_deviation * max_deviation
    
    # Take square root to get total systematic uncertainty
    syst_uncertainty = [sqrt(val) for val in syst_squared]
    
    # Create up and down variations with symmetric uncertainty
    prompt_up_contents = [central_contents[i] + syst_uncertainty[i] for i in range(nbins)]
    prompt_down_contents = [central_contents[i] - syst_uncertainty[i] for i in range(nbins)]
    
    return (prompt_up_contents, bin_labels), (prompt_down_contents, bin_labels)

def main():
    # Calculate central scale factors
    channel_str = "+".join(CHANNELS) if len(CHANNELS) > 1 else CHANNELS[0]
    print(f"Calculating Central scale factors for {channel_str}...")
    SF_central = calculate_scale_factors(CHANNELS, eralist, samplename_WZ, max_nj, "Central", RUN)
    
    # Get systematic variations for this channel/run
    systematics = get_systematics_for_channel(CHANNELS, RUN)
    
    # Extract central scale factors
    central_data = extract_histogram_data(SF_central, max_nj)
    
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
                SF_syst = calculate_scale_factors(CHANNELS, eralist, samplename_WZ, max_nj, variation, RUN)
                systematics_data[syst_name][variation] = extract_histogram_data(SF_syst, max_nj)
            except Exception as e:
                print(f"  WARNING: Failed to process {variation}: {e}")
                # Set to None to indicate missing data
                systematics_data[syst_name][variation] = None
    
    # Calculate prompt systematic uncertainty (combination of all MC systematics + statistical)
    print("Calculating prompt systematic variations...")
    try:
        prompt_up_data, prompt_down_data = calculate_prompt_uncertainty(
            central_data, 
            systematics_data, 
            max_nj
        )
        systematics_data["prompt"] = {
            "prompt_up": prompt_up_data,
            "prompt_down": prompt_down_data
        }
        print("  Combined symmetric uncertainties: sqrt(stat^2 + max(up,down)^2) for each systematic")
    except Exception as e:
        print(f"  WARNING: Failed to calculate prompt variations: {e}")
        systematics_data["prompt"] = {
            "prompt_up": None,
            "prompt_down": None
        }
    
    # Create correctionlib format
    corrections = []
    
    # Central correction
    central_contents, bin_labels = central_data
    central_corr = create_correction(
        name=f"WZNjetsSF_{args.channel}_{args.era}_Central",
        description=f"WZ N-jets scale factors for {args.channel} {args.era} (Central)",
        bin_contents=central_contents,
        bin_labels=bin_labels
    )
    corrections.append(central_corr)
    
    # Systematic corrections
    for syst_name, variations in systematics_data.items():
        for var_name, var_data in variations.items():
            if var_data is None:
                continue
            var_contents, var_labels = var_data
            syst_corr = create_correction(
                name=f"WZNjetsSF_{args.channel}_{args.era}_{var_name}",
                description=f"WZ N-jets scale factors for {args.channel} {args.era} ({var_name})",
                bin_contents=var_contents,
                bin_labels=var_labels
            )
            corrections.append(syst_corr)
    
    # Create correction set
    nonprompt_uncertainty = 0.25 if RUN == "Run2" else 0.35
    description = f"WZ N-jets scale factors for {args.channel} {args.era}"
    if len(CHANNELS) > 1:
        description += f" (combined {'+'.join(CHANNELS)})"
    description += f". Nonprompt uncertainty: ±{nonprompt_uncertainty*100:.0f}%"
    
    cset = cs.CorrectionSet(
        schema_version=2,
        description=description,
        corrections=corrections
    )
    
    # Save the file in correctionlib JSON format
    OUTPUTPATH = f"{WORKDIR}/TriLepton/results/{args.channel}/{args.era}/WZNjetsSF.json"
    os.makedirs(os.path.dirname(OUTPUTPATH), exist_ok=True)
    
    # Save to JSON
    with open(OUTPUTPATH, 'w') as f:
        f.write(cset.model_dump_json(exclude_unset=True, indent=2))
    
    print(f"Correctionlib-compatible scale factors saved to: {OUTPUTPATH}")
    total_variations = sum(len(v) for v in systematics.values())
    print(f"Created {len(corrections)} corrections ({len(systematics)} systematic sources)")
    print(f"Combined channels: {'+'.join(CHANNELS) if len(CHANNELS) > 1 else CHANNELS[0]}")
    print(f"Binning: 0j, 1j, 2j, {max_nj}j (with clamping for >={max_nj}j)")
    print(f"Nonprompt uncertainty: ±{nonprompt_uncertainty*100:.0f}% ({RUN})")
    if len(CHANNELS) > 1:
        print(f"Statistical precision improved by combining {len(CHANNELS)} channels")
        print(f"Note: Channel-specific systematics use Central values for irrelevant channels")

if __name__ == "__main__":
    main()
        
