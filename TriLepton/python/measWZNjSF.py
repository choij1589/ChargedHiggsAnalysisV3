#!/usr/bin/env python
import os
import sys
import argparse
import logging
import json
import ROOT
from math import sqrt
import correctionlib.schemav2 as cs

WORKDIR = os.environ["WORKDIR"]
from utils import build_sknanoutput_path

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--channel", required=True, type=str, help="channel (WZ1E2Mu, WZ3Mu, or WZCombined)")
parser.add_argument("--debug", action="store_true", default=False, help="debug")
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

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
    FLAG = None
    CHANNELS = ["WZ1E2Mu", "WZ3Mu"]

json_samplegroup = json.load(open(f"configs/samplegroup.json"))

KFACTORS_PATH = f"{WORKDIR}/Common/Data/KFactors.json"
with open(KFACTORS_PATH) as f:
    KFACTORS = json.load(f)

FAKENORM_PATH = f"{WORKDIR}/Common/Data/FakeNorm.json"
with open(FAKENORM_PATH) as f:
    FAKENORM = json.load(f)

# Fallback theory norm for rare processes absent from KFactors.json (others group)
OTHERS_THEORY_NORM = 0.50
CONV_SF_FALLBACK = {"central": 1.0, "total": 0.20}
CONV_SF_CACHE = {}


def get_theory_norm_fraction(sample, run):
    """Return fractional theory cross-section uncertainty for a sample.
    Uses xsecErr from KFactors.json; falls back to OTHERS_THEORY_NORM."""
    if run in KFACTORS and sample in KFACTORS[run]:
        return KFACTORS[run][sample]["xsecErr"] - 1.0
    return OTHERS_THEORY_NORM


def get_conv_sf(channel, era):
    """Return central ConvSF and total uncertainty fraction for a WZ channel/era."""
    zg_channel = channel.replace("WZ", "ZG")
    cache_key = (zg_channel, era)
    if cache_key in CONV_SF_CACHE:
        return CONV_SF_CACHE[cache_key]

    path = f"{WORKDIR}/TriLepton/results/{zg_channel}/{era}/ConvSF.json"
    if not os.path.exists(path):
        logging.warning(f"ConvSF.json not found at {path}, using fallback")
        CONV_SF_CACHE[cache_key] = CONV_SF_FALLBACK
        return CONV_SF_CACHE[cache_key]

    try:
        with open(path) as f:
            cset = json.load(f)
        corrections = {c["name"]: c for c in cset["corrections"]}
        central_key = next(k for k in corrections if k.endswith("_Central"))
        total_up_key = next(k for k in corrections if k.endswith("_total_up"))
        central = float(corrections[central_key]["data"]["expression"])
        total_up = float(corrections[total_up_key]["data"]["expression"])
        if central <= 0:
            raise ValueError(f"nonpositive central ConvSF {central}")
        conv_sf = {
            "central": central,
            "total": max((total_up - central) / central, 0.0),
        }
    except (KeyError, StopIteration, TypeError, ValueError) as e:
        logging.warning(f"Failed to parse ConvSF from {path}: {e}; using fallback")
        conv_sf = CONV_SF_FALLBACK

    CONV_SF_CACHE[cache_key] = conv_sf
    return conv_sf


def get_conv_sf_uncertainty_fraction(channels, eralist):
    """Return the max ConvSF total uncertainty fraction across eras.
    Used as the theory norm for the conv background in the WZ CR."""
    fractions = [get_conv_sf(channel, era)["total"] for channel in channels for era in eralist]
    fraction = max(fractions) if fractions else CONV_SF_FALLBACK["total"]
    logging.debug(f"Conv theory norm fraction: {fraction:.4f} (max across eras/channels)")
    return fraction


def apply_conv_scale_factor(hist, channel, era):
    """Apply central ConvSF to a conversion histogram in-place."""
    conv_sf = get_conv_sf(channel, era)
    hist.Scale(conv_sf["central"])
    logging.debug(f"Applied ConvSF {conv_sf['central']:.4f} to {channel}/{era}")
    return hist


def get_hist_data(channels, era):
    """Get data histogram for one or more channels."""
    if isinstance(channels, str):
        channels = [channels]

    hist = None
    for channel in channels:
        if channel == "WZ1E2Mu":
            flag = "Run1E2Mu"
        elif channel == "WZ3Mu":
            flag = "Run3Mu"
        else:
            raise ValueError(f"Unknown channel: {channel}")

        DATAPERIODs = json_samplegroup[era][channel.replace("WZ", "")]["data"]
        for sample in DATAPERIODs:
            file_path = build_sknanoutput_path(WORKDIR, channel, flag, era, sample)
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


def get_hist_nonprompt(channels, era, syst="Central"):
    """Get nonprompt histogram. Applies per-era uncertainty from FakeNorm.json."""
    if isinstance(channels, str):
        channels = [channels]

    if era in ["Run2", "2016preVFP", "2016postVFP", "2017", "2018"]:
        run = "Run2"
    else:
        run = "Run3"

    hist = None
    for channel in channels:
        if channel == "WZ1E2Mu":
            flag = "Run1E2Mu"
        elif channel == "WZ3Mu":
            flag = "Run3Mu"
        else:
            raise ValueError(f"Unknown channel: {channel}")

        nonprompt = json_samplegroup[era][channel.replace("WZ", "")]["nonprompt"]
        h_channel = None
        for sample in nonprompt:
            file_path = build_sknanoutput_path(WORKDIR, channel, flag, era, sample, is_nonprompt=True)
            assert os.path.exists(file_path), f"file {file_path} does not exist"
            f = ROOT.TFile.Open(file_path)
            h = f.Get(f"{channel}/Central/jets/size")
            h.SetDirectory(0)
            f.Close()
            if h_channel is None:
                h_channel = h.Clone(f"hist_{channel}")
                h_channel.SetDirectory(0)
            else:
                h_channel.Add(h)

        # Apply per-era nonprompt uncertainty from FakeNorm.json
        if syst in ["nonprompt_up", "nonprompt_down"]:
            try:
                uncertainty = FAKENORM[flag][era]
                scale_factor = 1.0 + uncertainty if syst == "nonprompt_up" else 1.0 - uncertainty
                h_channel.Scale(scale_factor)
                logging.debug(f"Applied {syst} to {channel}/{era}: factor {scale_factor:.3f} (unc {uncertainty})")
            except KeyError:
                logging.warning(f"No nonprompt uncertainty found for {flag}/{era} in FakeNorm.json, using 1.0")

        if hist is None:
            hist = h_channel.Clone("hist")
            hist.SetDirectory(0)
        else:
            hist.Add(h_channel)

    hist.SetDirectory(0)
    return hist


def get_hist_mc(channels, era, mc, syst="Central", theory_scales=None):
    """Get MC histogram. theory_scales maps sample -> scale factor for theory norm variations."""
    if isinstance(channels, str):
        channels = [channels]
    if theory_scales is None:
        theory_scales = {}

    hist = None
    for channel in channels:
        if channel == "WZ1E2Mu":
            flag = "Run1E2Mu"
        elif channel == "WZ3Mu":
            flag = "Run3Mu"
        else:
            raise ValueError(f"Unknown channel: {channel}")

        pred = json_samplegroup[era][channel.replace("WZ", "")][mc]
        for sample in pred:
            file_path = build_sknanoutput_path(WORKDIR, channel, flag, era, sample, run_syst=True)
            assert os.path.exists(file_path), f"file {file_path} does not exist"
            f = ROOT.TFile.Open(file_path)

            h = f.Get(f"{channel}/{syst}/jets/size")
            if not h:
                if syst != "Central":
                    h = f.Get(f"{channel}/Central/jets/size")
                    if h:
                        logging.info(f"Systematic {syst} not found for {channel}/{sample}, using Central")
                    else:
                        logging.warning(f"Cannot find {channel}/Central/jets/size for sample {sample}")
                        f.Close()
                        continue
                else:
                    logging.warning(f"Cannot find {channel}/Central/jets/size for sample {sample}")
                    f.Close()
                    continue

            h.SetDirectory(0)
            f.Close()

            # Apply K-factor
            if RUN in KFACTORS and sample in KFACTORS[RUN]:
                kfactor = KFACTORS[RUN][sample]["kFactor"]
                h.Scale(kfactor)

            if mc == "conv":
                h = apply_conv_scale_factor(h, channel, era)

            # Apply per-sample theory norm scale if provided
            if sample in theory_scales:
                h.Scale(theory_scales[sample])

            if hist is None:
                hist = h.Clone("hist")
                hist.SetDirectory(0)
            else:
                hist.Add(h)

    if hist is not None:
        hist.SetDirectory(0)
    return hist


def get_hist_by_name(channels, era, name, syst="Central", theory_scale=1.0):
    """Get histogram by sample name. theory_scale applies a flat theory norm variation."""
    if isinstance(channels, str):
        channels = [channels]

    hist = None
    for channel in channels:
        if channel == "WZ1E2Mu":
            flag = "Run1E2Mu"
        elif channel == "WZ3Mu":
            flag = "Run3Mu"
        else:
            raise ValueError(f"Unknown channel: {channel}")

        file_path = build_sknanoutput_path(WORKDIR, channel, flag, era, name,
                                           run_syst=True, no_wzsf=(name == samplename_WZ))
        assert os.path.exists(file_path), f"file {file_path} does not exist"
        f = ROOT.TFile.Open(file_path)

        h = f.Get(f"{channel}/{syst}/jets/size")
        if not h:
            if syst != "Central":
                h = f.Get(f"{channel}/Central/jets/size")
                if h:
                    logging.info(f"Systematic {syst} not found for {channel}, using Central")
                else:
                    logging.warning(f"Cannot find {channel}/Central/jets/size for sample {name}")
                    f.Close()
                    continue
            else:
                logging.warning(f"Cannot find {channel}/Central/jets/size for sample {name}")
                f.Close()
                return None

        h.SetDirectory(0)
        f.Close()

        # Apply K-factor
        if RUN in KFACTORS and name in KFACTORS[RUN]:
            kfactor = KFACTORS[RUN][name]["kFactor"]
            h.Scale(kfactor)

        # Apply flat theory norm scale
        if theory_scale != 1.0:
            h.Scale(theory_scale)

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
    """Merge bins for nJets >= max_nj into a single bin."""
    if hist is None:
        return None

    merged_hist = hist.Clone(hist.GetName() + "_merged")
    target_bin = max_nj + 1

    total_content = 0
    total_error_sq = 0

    for bin_idx in range(target_bin, merged_hist.GetNbinsX() + 1):
        content = merged_hist.GetBinContent(bin_idx)
        error = merged_hist.GetBinError(bin_idx)
        total_content += content
        total_error_sq += error * error
        if bin_idx != target_bin:
            merged_hist.SetBinContent(bin_idx, 0)
            merged_hist.SetBinError(bin_idx, 0)

    merged_hist.SetBinContent(target_bin, total_content)
    merged_hist.SetBinError(target_bin, sqrt(total_error_sq))
    return merged_hist


def build_theory_scales(channels, era, category, direction, run):
    """Build per-sample theory norm scale factors for a given category and direction (+1 or -1)."""
    theory_scales = {}
    for channel in channels:
        channel_key = channel.replace("WZ", "")
        samples = json_samplegroup[era][channel_key].get(category, [])
        for sample in samples:
            fraction = get_theory_norm_fraction(sample, run)
            theory_scales[sample] = 1.0 + fraction * direction
    return theory_scales


def calculate_scale_factors(channels, eralist, samplename_WZ, max_nj, syst="Central", run="Run3",
                             theory_category=None, theory_direction=0, conv_theory_fraction=0.0):
    """Calculate WZNjSF histograms with optional nonprompt or theory norm variation.

    theory_category: one of "ZZ", "conv", "ttX", "others" (or None for no theory variation)
    theory_direction: +1 or -1
    conv_theory_fraction: fractional uncertainty on conv (from ConvSF total unc), used when theory_category="conv"
    """
    h_data_total = None
    h_nonprompt_total = None
    h_WZ_total = None
    h_ZZ_total = None
    h_conv_total = None
    h_ttX_total = None
    h_others_total = None

    is_nonprompt_syst = syst in ["nonprompt_up", "nonprompt_down"]
    mc_syst = "Central" if is_nonprompt_syst else syst

    # Determine theory scales for ZZ (single named sample)
    zz_theory_scale = 1.0
    if theory_category == "ZZ" and theory_direction != 0:
        zz_fraction = get_theory_norm_fraction("ZZTo4L_powheg", run)
        zz_theory_scale = 1.0 + zz_fraction * theory_direction

    # Determine theory scale for conv (flat fraction from ConvSF measurement)
    conv_theory_scale = 1.0
    if theory_category == "conv" and theory_direction != 0:
        conv_theory_scale = 1.0 + conv_theory_fraction * theory_direction

    for era in eralist:
        h_data = get_hist_data(channels, era)
        h_data_total = add_hist("data", h_data_total, h_data)

        h_nonprompt = get_hist_nonprompt(channels, era, syst)
        h_nonprompt_total = add_hist("nonprompt", h_nonprompt_total, h_nonprompt)

        # WZ: signal, no theory variation
        h_WZ = get_hist_by_name(channels, era, samplename_WZ, mc_syst)
        h_WZ_total = add_hist("WZ", h_WZ_total, h_WZ)

        # ZZ: separate named sample with optional theory scale
        h_ZZ = get_hist_by_name(channels, era, "ZZTo4L_powheg", mc_syst, theory_scale=zz_theory_scale)
        h_ZZ_total = add_hist("ZZ", h_ZZ_total, h_ZZ)

        # conv: subtracted background with optional flat theory scale from ConvSF uncertainty
        conv_theory_scales_era = {}
        if theory_category == "conv" and theory_direction != 0:
            channel_key = channels[0].replace("WZ", "")
            for sample in json_samplegroup[era][channel_key].get("conv", []):
                conv_theory_scales_era[sample] = conv_theory_scale
        h_conv = get_hist_mc(channels, era, "conv", mc_syst, theory_scales=conv_theory_scales_era)
        h_conv_total = add_hist("conv", h_conv_total, h_conv)

        # ttX: per-sample theory scales from KFactors.json
        ttX_theory_scales = {}
        if theory_category == "ttX" and theory_direction != 0:
            ttX_theory_scales = build_theory_scales(channels, era, "ttX", theory_direction, run)
        h_ttX = get_hist_mc(channels, era, "ttX", mc_syst, theory_scales=ttX_theory_scales)
        h_ttX_total = add_hist("ttX", h_ttX_total, h_ttX)

        # others: per-sample theory scales (fallback OTHERS_THEORY_NORM)
        others_theory_scales = {}
        if theory_category == "others" and theory_direction != 0:
            others_theory_scales = build_theory_scales(channels, era, "others", theory_direction, run)
        h_others = get_hist_mc(channels, era, "others", mc_syst, theory_scales=others_theory_scales)
        h_others_total = add_hist("others", h_others_total, h_others)

    # Merge high njet bins
    if syst == "Central" and theory_category is None:
        print(f"Merging bins for nJets >= {max_nj}")
    h_data_total = merge_high_njet_bins(h_data_total, max_nj)
    h_nonprompt_total = merge_high_njet_bins(h_nonprompt_total, max_nj)
    h_WZ_total = merge_high_njet_bins(h_WZ_total, max_nj)
    h_ZZ_total = merge_high_njet_bins(h_ZZ_total, max_nj)
    h_conv_total = merge_high_njet_bins(h_conv_total, max_nj)
    h_ttX_total = merge_high_njet_bins(h_ttX_total, max_nj)
    h_others_total = merge_high_njet_bins(h_others_total, max_nj)

    # SF = (Data - nonprompt - ZZ - conv - ttX - others) / WZ
    SF = h_data_total.Clone("SF")
    SF.Add(h_nonprompt_total, -1)
    SF.Add(h_ZZ_total, -1)
    SF.Add(h_conv_total, -1)
    SF.Add(h_ttX_total, -1)
    SF.Add(h_others_total, -1)
    SF.Divide(h_WZ_total)

    return SF


def extract_histogram_data_with_errors(SF, max_nj):
    """Extract bin contents, errors, and labels from SF histogram."""
    bin_contents = []
    bin_errors = []
    bin_labels = []

    for i in range(1, SF.GetNbinsX() + 1):
        content = SF.GetBinContent(i)
        error = SF.GetBinError(i)
        if content != 0 or i <= max_nj + 1:
            bin_contents.append(float(content))
            bin_errors.append(float(error))
            njet = i - 1
            bin_labels.append(f"{min(njet, max_nj)}j")

    return bin_contents, bin_errors, bin_labels


def create_correction(name, description, bin_contents, bin_labels):
    """Create a correctionlib Correction object using jet-bin lookup with clamping."""
    nbins = len(bin_contents)
    edges = list(range(nbins + 1))

    data = cs.Binning(
        nodetype="binning",
        input="njets",
        edges=edges,
        content=[float(v) for v in bin_contents],
        flow="clamp"
    )
    correction = cs.Correction(
        name=name,
        version=1,
        description=description,
        inputs=[cs.Variable(name="njets", type="real",
                            description="Number of jets (integer, overflow uses highest bin)")],
        output=cs.Variable(name="sf", type="real", description="Scale factor"),
        data=data
    )
    return correction


def main():
    channel_str = "+".join(CHANNELS) if len(CHANNELS) > 1 else CHANNELS[0]
    print(f"Calculating Central scale factors for {channel_str}...")
    SF_central = calculate_scale_factors(CHANNELS, eralist, samplename_WZ, max_nj, "Central", RUN)
    central_contents, central_errors, bin_labels = extract_histogram_data_with_errors(SF_central, max_nj)
    nbins = len(central_contents)

    print(f"Central SFs (bin-by-bin):")
    for i, label in enumerate(bin_labels):
        print(f"  {label}: {central_contents[i]:.4f} ± {central_errors[i]:.4f} (stat)")

    # Statistical variations (bin-by-bin, symmetric from CR measurement)
    stat_up_contents = [central_contents[i] + central_errors[i] for i in range(nbins)]
    stat_down_contents = [central_contents[i] - central_errors[i] for i in range(nbins)]

    # Nonprompt variations (per-era from FakeNorm.json)
    print("Calculating nonprompt variations...")
    SF_np_up = calculate_scale_factors(CHANNELS, eralist, samplename_WZ, max_nj, "nonprompt_up", RUN)
    SF_np_down = calculate_scale_factors(CHANNELS, eralist, samplename_WZ, max_nj, "nonprompt_down", RUN)
    np_up_contents, _, _ = extract_histogram_data_with_errors(SF_np_up, max_nj)
    np_down_contents, _, _ = extract_histogram_data_with_errors(SF_np_down, max_nj)

    # Conv theory norm fraction from ConvSF measurement
    conv_fraction = get_conv_sf_uncertainty_fraction(CHANNELS, eralist)
    print(f"Conv theory norm fraction: {conv_fraction:.4f}")

    # Theory norm variations for subtracted backgrounds (bin-by-bin)
    # ZZ: single named sample (xsecErr from KFactors.json)
    # conv: flat fraction from ConvSF total uncertainty
    # ttX: per-sample from KFactors.json
    # others: fallback OTHERS_THEORY_NORM
    theory_categories = ["ZZ", "conv", "ttX", "others"]
    theory_up_contents = {}
    theory_down_contents = {}

    print("Calculating theory norm variations...")
    for cat in theory_categories:
        SF_up = calculate_scale_factors(CHANNELS, eralist, samplename_WZ, max_nj, "Central", RUN,
                                        theory_category=cat, theory_direction=+1,
                                        conv_theory_fraction=conv_fraction)
        SF_down = calculate_scale_factors(CHANNELS, eralist, samplename_WZ, max_nj, "Central", RUN,
                                          theory_category=cat, theory_direction=-1,
                                          conv_theory_fraction=conv_fraction)
        up_c, _, _ = extract_histogram_data_with_errors(SF_up, max_nj)
        down_c, _, _ = extract_histogram_data_with_errors(SF_down, max_nj)
        theory_up_contents[cat] = up_c
        theory_down_contents[cat] = down_c
        print(f"  theory_{cat}: up={[f'{v:.4f}' for v in up_c]}, down={[f'{v:.4f}' for v in down_c]}")

    # Total uncertainty bin-by-bin: quadrature sum of stat + nonprompt + theory components
    total_up_contents = []
    total_down_contents = []
    print("Total uncertainties (bin-by-bin):")
    for i in range(nbins):
        stat_unc_i = central_errors[i]
        nonprompt_unc_i = abs(np_up_contents[i] - central_contents[i])
        theory_unc_i = sqrt(sum(
            abs(theory_up_contents[cat][i] - central_contents[i])**2
            for cat in theory_categories
        ))
        total_unc_i = sqrt(stat_unc_i**2 + nonprompt_unc_i**2 + theory_unc_i**2)
        total_up_contents.append(central_contents[i] + total_unc_i)
        total_down_contents.append(central_contents[i] - total_unc_i)
        rel = 100 * total_unc_i / central_contents[i] if central_contents[i] != 0 else 0
        print(f"  {bin_labels[i]}: ±{total_unc_i:.4f} ({rel:.1f}%)  "
              f"[stat={stat_unc_i:.4f}, nonprompt={nonprompt_unc_i:.4f}, theory={theory_unc_i:.4f}]")

    # Build correctionlib output
    corrections = []
    zero_errors = [0.0] * nbins

    def add_corr(suffix, contents):
        corrections.append(create_correction(
            name=f"WZNjetsSF_{args.channel}_{args.era}_{suffix}",
            description=f"WZ N-jets scale factors for {args.channel} {args.era} ({suffix})",
            bin_contents=contents,
            bin_labels=bin_labels
        ))

    add_corr("Central", central_contents)
    add_corr("statistical_up", stat_up_contents)
    add_corr("statistical_down", stat_down_contents)
    add_corr("nonprompt_up", np_up_contents)
    add_corr("nonprompt_down", np_down_contents)
    for cat in theory_categories:
        add_corr(f"theory_{cat}_up", theory_up_contents[cat])
        add_corr(f"theory_{cat}_down", theory_down_contents[cat])
    add_corr("total_up", total_up_contents)
    add_corr("total_down", total_down_contents)

    description = (
        f"WZ N-jets K-factors for {args.channel} {args.era}. "
        f"Uncertainties: stat (CR measurement, bin-by-bin), nonprompt (per-era from FakeNorm.json), "
        f"theory norms for subtracted backgrounds (per-sample xsecErr from KFactors.json; "
        f"conv uses ConvSF total uncertainty). All uncertainties bin-by-bin."
    )
    cset = cs.CorrectionSet(
        schema_version=2,
        description=description,
        corrections=corrections
    )

    OUTPUTPATH = f"{WORKDIR}/TriLepton/results/{args.channel}/{args.era}/WZNjetsSF.json"
    os.makedirs(os.path.dirname(OUTPUTPATH), exist_ok=True)
    with open(OUTPUTPATH, 'w') as f:
        f.write(cset.model_dump_json(exclude_unset=True, indent=2))

    print(f"\nSaved to: {OUTPUTPATH}")
    print(f"Binning: {', '.join(bin_labels)} (clamping for >={max_nj}j)")
    print(f"Corrections: {[c.name for c in corrections]}")


if __name__ == "__main__":
    main()
