#!/usr/bin/env python
import os
import sys
import argparse
import logging
import json
import ctypes
import ROOT
from math import sqrt
import correctionlib.schemav2 as cs

WORKDIR = os.environ["WORKDIR"]
from utils import build_sknanoutput_path

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel (ZG1E2Mu, ZG3Mu, or ZGCombined)")
parser.add_argument("--debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

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
    FLAG = None
    CHANNELS = ["ZG1E2Mu", "ZG3Mu"]

json_samplegroup = json.load(open(f"configs/samplegroup.json"))

KFACTORS_PATH = f"{WORKDIR}/Common/Data/KFactors.json"
with open(KFACTORS_PATH) as f:
    KFACTORS = json.load(f)

FAKENORM_PATH = f"{WORKDIR}/Common/Data/FakeNorm.json"
with open(FAKENORM_PATH) as f:
    FAKENORM = json.load(f)

# Fallback theory norm for rare processes absent from KFactors.json (others group)
OTHERS_THEORY_NORM = 0.50
RUN3_WZ_SAMPLE = "WZTo3LNu_powheg"


def get_theory_norm_fraction(sample, run):
    """Return fractional theory cross-section uncertainty for a sample.
    Uses xsecErr from KFactors.json; falls back to OTHERS_THEORY_NORM."""
    if run in KFACTORS and sample in KFACTORS[run]:
        return KFACTORS[run][sample]["xsecErr"] - 1.0
    return OTHERS_THEORY_NORM


def get_yield_data_with_error(channels, era):
    """Get data yield with statistical error for one or more channels."""
    if isinstance(channels, str):
        channels = [channels]

    total_yield = 0.0
    total_error_sq = 0.0
    for channel in channels:
        if channel == "ZG1E2Mu":
            flag = "Run1E2Mu"
        elif channel == "ZG3Mu":
            flag = "Run3Mu"
        else:
            raise ValueError(f"Unknown channel: {channel}")

        DATAPERIODs = json_samplegroup[era][channel.replace("ZG", "")]["data"]
        for sample in DATAPERIODs:
            file_path = build_sknanoutput_path(WORKDIR, channel, flag, era, sample)
            assert os.path.exists(file_path), f"file {file_path} does not exist"
            f = ROOT.TFile.Open(file_path)
            h = f.Get(f"{channel}/Central/ZCand/mass")
            h.SetDirectory(0)
            error = ctypes.c_double(0.0)
            yield_value = h.IntegralAndError(0, h.GetNbinsX() + 1, error)
            f.Close()
            total_yield += yield_value
            total_error_sq += error.value * error.value

    return total_yield, sqrt(total_error_sq)


def get_yield_nonprompt_with_error(channels, era, syst="Central"):
    """Get nonprompt yield with statistical error. Applies per-era uncertainty from FakeNorm.json."""
    if isinstance(channels, str):
        channels = [channels]

    if era in ["Run2", "2016preVFP", "2016postVFP", "2017", "2018"]:
        run = "Run2"
    else:
        run = "Run3"

    total_yield = 0.0
    total_error_sq = 0.0
    for channel in channels:
        if channel == "ZG1E2Mu":
            flag = "Run1E2Mu"
        elif channel == "ZG3Mu":
            flag = "Run3Mu"
        else:
            raise ValueError(f"Unknown channel: {channel}")

        nonprompt = json_samplegroup[era][channel.replace("ZG", "")]["nonprompt"]
        channel_yield = 0.0
        channel_error_sq = 0.0

        for sample in nonprompt:
            file_path = build_sknanoutput_path(WORKDIR, channel, flag, era, sample, is_nonprompt=True)
            assert os.path.exists(file_path), f"file {file_path} does not exist"
            f = ROOT.TFile.Open(file_path)
            h = f.Get(f"{channel}/Central/ZCand/mass")
            h.SetDirectory(0)
            error = ctypes.c_double(0.0)
            yield_value = h.IntegralAndError(0, h.GetNbinsX() + 1, error)
            f.Close()
            channel_yield += yield_value
            channel_error_sq += error.value * error.value

        # Apply per-era nonprompt uncertainty from FakeNorm.json
        if syst in ["nonprompt_up", "nonprompt_down"]:
            try:
                uncertainty = FAKENORM[flag][era]
                scale_factor = 1.0 + uncertainty if syst == "nonprompt_up" else 1.0 - uncertainty
                channel_yield *= scale_factor
                channel_error_sq *= scale_factor * scale_factor
                logging.debug(f"Applied {syst} to {channel}/{era}: factor {scale_factor:.3f} (unc {uncertainty})")
            except KeyError:
                logging.warning(f"No nonprompt uncertainty found for {flag}/{era} in FakeNorm.json, using 1.0")

        total_yield += channel_yield
        total_error_sq += channel_error_sq

    return total_yield, sqrt(total_error_sq)


def get_yield_mc_with_error(channels, era, mc, syst="Central", theory_scales=None):
    """Get MC yield with statistical error. theory_scales maps sample -> scale factor."""
    if isinstance(channels, str):
        channels = [channels]
    if theory_scales is None:
        theory_scales = {}

    if era in ["Run2", "2016preVFP", "2016postVFP", "2017", "2018"]:
        run = "Run2"
    else:
        run = "Run3"

    total_yield = 0.0
    total_error_sq = 0.0
    for channel in channels:
        if channel == "ZG1E2Mu":
            flag = "Run1E2Mu"
        elif channel == "ZG3Mu":
            flag = "Run3Mu"
        else:
            raise ValueError(f"Unknown channel: {channel}")

        mc_samples = json_samplegroup[era][channel.replace("ZG", "")].get(mc, [])
        if not mc_samples:
            logging.warning(f"No {mc} samples found for {channel} in era {era}")
            continue

        for sample in mc_samples:
            use_no_wzsf = run == "Run3" and sample == RUN3_WZ_SAMPLE
            file_path = build_sknanoutput_path(
                WORKDIR, channel, flag, era, sample,
                run_syst=True, no_wzsf=use_no_wzsf
            )
            if not os.path.exists(file_path):
                logging.warning(f"File {file_path} does not exist, skipping")
                continue

            f = ROOT.TFile.Open(file_path)
            h = f.Get(f"{channel}/{syst}/ZCand/mass")
            if not h:
                if syst != "Central":
                    logging.warning(f"Systematic {syst} not found for {channel}/{sample}, skipping")
                    f.Close()
                    continue
                else:
                    logging.warning(f"Cannot find {channel}/Central/ZCand/mass for sample {sample}")
                    f.Close()
                    continue

            h.SetDirectory(0)
            error = ctypes.c_double(0.0)
            yield_value = h.IntegralAndError(0, h.GetNbinsX() + 1, error)
            f.Close()

            # Apply K-factor
            if run in KFACTORS and sample in KFACTORS[run]:
                kfactor = KFACTORS[run][sample]["kFactor"]
                yield_value *= kfactor
                error.value *= kfactor

            # Apply per-sample theory norm scale if provided
            if sample in theory_scales:
                scale = theory_scales[sample]
                yield_value *= scale
                error.value *= scale

            total_yield += yield_value
            total_error_sq += error.value * error.value

    return total_yield, sqrt(total_error_sq)


def build_theory_scales(channels, era, category, direction, run):
    """Build per-sample theory norm scale factors for a given category and direction (+1 or -1)."""
    theory_scales = {}
    for channel in channels:
        channel_key = channel.replace("ZG", "")
        samples = json_samplegroup[era][channel_key].get(category, [])
        for sample in samples:
            fraction = get_theory_norm_fraction(sample, run)
            theory_scales[sample] = 1.0 + fraction * direction
    return theory_scales


def calculate_scale_factor(channels, eralist, syst="Central", run="Run3",
                            theory_category=None, theory_direction=0):
    """Calculate ConvSF with optional nonprompt or theory norm variation."""
    total_data = 0.0
    total_data_error_sq = 0.0
    total_conv = 0.0
    total_conv_error_sq = 0.0
    total_non_conv = 0.0
    total_non_conv_error_sq = 0.0

    is_nonprompt_syst = syst in ["nonprompt_up", "nonprompt_down"]
    mc_syst = "Central" if is_nonprompt_syst else syst

    for era in eralist:
        data_yield, data_error = get_yield_data_with_error(channels, era)
        total_data += data_yield
        total_data_error_sq += data_error * data_error

        conv_yield, conv_error = get_yield_mc_with_error(channels, era, "conv", mc_syst)
        total_conv += conv_yield
        total_conv_error_sq += conv_error * conv_error

        non_conv_categories = []
        for channel in channels:
            channel_key = channel.replace("ZG", "")
            if era in json_samplegroup and channel_key in json_samplegroup[era]:
                for category in json_samplegroup[era][channel_key].keys():
                    if category not in ["data", "conv"] and category not in non_conv_categories:
                        non_conv_categories.append(category)

        for category in non_conv_categories:
            if category == "nonprompt":
                y, e = get_yield_nonprompt_with_error(channels, era, syst)
            else:
                # Build per-sample theory scales if this is the varied category
                theory_scales = {}
                if theory_category is not None and category == theory_category and theory_direction != 0:
                    theory_scales = build_theory_scales(channels, era, category, theory_direction, run)
                y, e = get_yield_mc_with_error(channels, era, category, mc_syst, theory_scales=theory_scales)
            total_non_conv += y
            total_non_conv_error_sq += e * e

    if total_conv > 0:
        sf = (total_data - total_non_conv) / total_conv
        numerator = total_data - total_non_conv
        numerator_error = sqrt(total_data_error_sq + total_non_conv_error_sq)
        total_conv_error = sqrt(total_conv_error_sq)
        if abs(numerator) > 0:
            sf_error = abs(sf) * sqrt((numerator_error / numerator)**2 + (total_conv_error / total_conv)**2)
        else:
            logging.warning("Numerator (Data - Bkg) is zero!")
            sf_error = 0.0
    else:
        logging.error("Total conversion yield is zero!")
        sf = 1.0
        sf_error = 0.0

    return sf, sf_error


def create_correction(name, description, scale_factor):
    """Create a correctionlib Correction object for a single scale factor."""
    data = cs.Formula(
        nodetype="formula",
        expression=str(scale_factor),
        parser="TFormula",
        variables=[]
    )
    correction = cs.Correction(
        name=name,
        version=1,
        description=description,
        inputs=[],
        output=cs.Variable(name="sf", type="real", description="Scale factor"),
        data=data
    )
    return correction


def main():
    channel_str = "+".join(CHANNELS) if len(CHANNELS) > 1 else CHANNELS[0]
    print(f"Calculating Central scale factor for {channel_str}...")
    sf_central, sf_central_error = calculate_scale_factor(CHANNELS, eralist, "Central", RUN)
    print(f"Central SF = {sf_central:.4f} ± {sf_central_error:.4f} (stat)")

    # Statistical variations (symmetric from CR measurement)
    stat_up = sf_central + sf_central_error
    stat_down = sf_central - sf_central_error
    rel_stat = 100 * sf_central_error / sf_central if sf_central != 0 else 0
    print(f"Statistical: ±{sf_central_error:.4f} ({rel_stat:.1f}%)")

    # Nonprompt variations (per-era uncertainties from FakeNorm.json)
    print("Calculating nonprompt variations...")
    sf_nonprompt_up, _ = calculate_scale_factor(CHANNELS, eralist, "nonprompt_up", RUN)
    sf_nonprompt_down, _ = calculate_scale_factor(CHANNELS, eralist, "nonprompt_down", RUN)
    nonprompt_unc = abs(sf_nonprompt_up - sf_central)
    print(f"  nonprompt_up   = {sf_nonprompt_up:.4f}  (unc {nonprompt_unc:.4f})")
    print(f"  nonprompt_down = {sf_nonprompt_down:.4f}")

    # Theory norm variations for subtracted backgrounds (per sample via KFactors.json)
    # conv is what we measure — no theory norm needed for it here
    theory_categories = ["diboson", "ttX", "others"]
    theory_variations = {}
    print("Calculating theory norm variations...")
    for cat in theory_categories:
        sf_up, _ = calculate_scale_factor(CHANNELS, eralist, "Central", RUN,
                                          theory_category=cat, theory_direction=+1)
        sf_down, _ = calculate_scale_factor(CHANNELS, eralist, "Central", RUN,
                                            theory_category=cat, theory_direction=-1)
        theory_variations[cat] = (sf_up, sf_down)
        print(f"  theory_{cat}_up = {sf_up:.4f}, theory_{cat}_down = {sf_down:.4f}")

    # Total uncertainty: quadrature sum of stat + nonprompt + theory components
    stat_unc = sf_central_error
    theory_unc = sqrt(sum(abs(sf_up - sf_central)**2 for sf_up, _ in theory_variations.values()))
    total_unc = sqrt(stat_unc**2 + nonprompt_unc**2 + theory_unc**2)
    total_up = sf_central + total_unc
    total_down = sf_central - total_unc
    print(f"Total uncertainty: ±{total_unc:.4f} ({100*total_unc/sf_central:.1f}%)")
    print(f"  Components: stat={stat_unc:.4f}, nonprompt={nonprompt_unc:.4f}, theory={theory_unc:.4f}")

    # Build correctionlib output
    corrections = []

    corrections.append(create_correction(
        name=f"ConvSF_{args.channel}_{args.era}_Central",
        description=f"Conversion scale factor for {args.channel} {args.era} (Central)",
        scale_factor=sf_central
    ))
    corrections.append(create_correction(
        name=f"ConvSF_{args.channel}_{args.era}_statistical_up",
        description=f"Conversion scale factor for {args.channel} {args.era} (statistical_up)",
        scale_factor=stat_up
    ))
    corrections.append(create_correction(
        name=f"ConvSF_{args.channel}_{args.era}_statistical_down",
        description=f"Conversion scale factor for {args.channel} {args.era} (statistical_down)",
        scale_factor=stat_down
    ))
    corrections.append(create_correction(
        name=f"ConvSF_{args.channel}_{args.era}_nonprompt_up",
        description=f"Conversion scale factor for {args.channel} {args.era} (nonprompt_up)",
        scale_factor=sf_nonprompt_up
    ))
    corrections.append(create_correction(
        name=f"ConvSF_{args.channel}_{args.era}_nonprompt_down",
        description=f"Conversion scale factor for {args.channel} {args.era} (nonprompt_down)",
        scale_factor=sf_nonprompt_down
    ))
    for cat, (sf_up, sf_down) in theory_variations.items():
        corrections.append(create_correction(
            name=f"ConvSF_{args.channel}_{args.era}_theory_{cat}_up",
            description=f"Conversion scale factor for {args.channel} {args.era} (theory_{cat}_up)",
            scale_factor=sf_up
        ))
        corrections.append(create_correction(
            name=f"ConvSF_{args.channel}_{args.era}_theory_{cat}_down",
            description=f"Conversion scale factor for {args.channel} {args.era} (theory_{cat}_down)",
            scale_factor=sf_down
        ))
    corrections.append(create_correction(
        name=f"ConvSF_{args.channel}_{args.era}_total_up",
        description=f"Conversion scale factor for {args.channel} {args.era} (total_up)",
        scale_factor=total_up
    ))
    corrections.append(create_correction(
        name=f"ConvSF_{args.channel}_{args.era}_total_down",
        description=f"Conversion scale factor for {args.channel} {args.era} (total_down)",
        scale_factor=total_down
    ))

    description = (
        f"Conversion K-factor for {args.channel} {args.era}. "
        f"Uncertainties: stat (CR measurement), nonprompt (per-era from FakeNorm.json), "
        f"theory norms for subtracted backgrounds (per-sample xsecErr from KFactors.json)."
    )
    cset = cs.CorrectionSet(
        schema_version=2,
        description=description,
        corrections=corrections
    )

    OUTPUTPATH = f"{WORKDIR}/TriLepton/results/{args.channel}/{args.era}/ConvSF.json"
    os.makedirs(os.path.dirname(OUTPUTPATH), exist_ok=True)
    with open(OUTPUTPATH, 'w') as f:
        f.write(cset.model_dump_json(exclude_unset=True, indent=2))

    print(f"\nSaved to: {OUTPUTPATH}")
    print(f"Corrections: {[c.name for c in corrections]}")


if __name__ == "__main__":
    main()
