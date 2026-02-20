#!/usr/bin/env python
import os
import argparse
import logging
import json
import ROOT
from itertools import product
from array import array
from common import findbin, get_run_period

parser = argparse.ArgumentParser()
parser.add_argument("--era", required=True, type=str, help="era")
parser.add_argument("--measure", required=True, type=str, help="electron / muon")
parser.add_argument("--isQCD", default=False, action="store_true", help="isQCD")
parser.add_argument("--isTT", default=False, action="store_true", help="Measure from TTJJ_powheg")
parser.add_argument("--isMC", default=False, action="store_true", help="Measure from both QCD and TTJJ separately")
parser.add_argument("--debug", action="store_true", default=False, help="debug")
parser.add_argument("--noHEMVeto", default=False, action="store_true", help="Use RunNoHEMVeto variant")
args = parser.parse_args()

# Validate mutual exclusivity of MC flags
mc_flags = sum([args.isQCD, args.isTT, args.isMC])
if mc_flags > 1:
    raise ValueError("Only one of --isQCD, --isTT, --isMC can be specified")

logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)

#### Settings
WORKDIR = os.environ['WORKDIR']
TrigPrefixes = []
ptcorr_bins = []
abseta_bins = []

if args.measure == "electron":
    ptcorr_bins = [15., 17., 20., 25., 35., 50., 100., 200.]
    abseta_bins = [0., 0.8, 1.479, 2.5]
    TrigPrefixes = ["El8", "El12", "El23"]
    QCD = ["QCD_EMEnriched", "QCD_bcToE"]
elif args.measure == "muon":
    ptcorr_bins = [10., 12., 14., 17., 20., 30., 50., 100., 200.]
    abseta_bins = [0., 0.9, 1.6, 2.4]
    TrigPrefixes = ["Mu8", "Mu17"]
    QCD = ["QCD_MuEnriched"]
else:
    raise KeyError(f"Wrong measure {args.measure}")

# Flavour definitions for MC-based measurements
FLAVOURS = ["ljet", "cjet", "bjet", "pujet"]
FLAVOURS_WITH_INCLUSIVE = ["inclusive", "ljet", "cjet", "bjet", "pujet"]
TTJJ = "TTJJ_powheg"

with open("configs/samplegroup.json") as f:
    SAMPLEGROUP = json.load(f)[args.era][args.measure]
PhysicsSystematics = json.load(open("configs/systematics.json"))[get_run_period(args.era)][args.measure]
PhysicsVariations = []
for syst_name, sources in PhysicsSystematics.items():
    PhysicsVariations.extend(sources)
SelectionVariations = ["Central", "PromptNorm_Up", "PromptNorm_Down", "MotherJetPt_Up", "MotherJetPt_Down", "RequireHeavyTag"]

DATAPERIODs = SAMPLEGROUP["data"]
W = SAMPLEGROUP["W"]
Z = SAMPLEGROUP["Z"]
TT = SAMPLEGROUP["TT"]
ST = SAMPLEGROUP["ST"]
VV = SAMPLEGROUP["VV"]
MCList = W + Z + TT + ST + VV

WPs = ["loose", "tight"]

#### first evaluate central scale for product(trig_prefix, wp, syst)
def get_prompt_scale(trig_prefix, wp, syst):
    """Compute prompt normalization scale factor.

    Returns:
        tuple: (scale, stat_pct) where stat_pct is the fractional stat uncertainty in %
    """
    from ctypes import c_double
    import math

    is_physics_syst = syst in PhysicsVariations

    # For physics systematics: data uses Central, MC uses the systematic path
    # For selection variations: both data and MC use the same path
    if is_physics_syst:
        histkey_data = f"ZEnriched/{trig_prefix}/{wp}/Central/ZCand/mass"
        histkey_mc = f"ZEnriched/{trig_prefix}/{wp}/{syst}/ZCand/mass"
    elif syst in ["PromptNorm_Up", "PromptNorm_Down"]:
        histkey_data = f"ZEnriched/{trig_prefix}/{wp}/Central/ZCand/mass"
        histkey_mc = f"ZEnriched/{trig_prefix}/{wp}/Central/ZCand/mass"
    elif syst == "Stat":
        histkey_data = f"ZEnriched/{trig_prefix}/{wp}/Central/ZCand/mass"
        histkey_mc = f"ZEnriched/{trig_prefix}/{wp}/Central/ZCand/mass"
    else:
        histkey_data = f"ZEnriched/{trig_prefix}/{wp}/{syst}/ZCand/mass"
        histkey_mc = f"ZEnriched/{trig_prefix}/{wp}/{syst}/ZCand/mass"

    # Use MeasFakeRateV4 with consolidated lepton type file
    lepton_type = "MeasFakeEl" if args.measure == "electron" else "MeasFakeMu"
    suffix = "_RunNoHEMVeto" if args.noHEMVeto else ""

    rate_data = 0.
    rate_data_err_sq = 0.
    for sample in DATAPERIODs:
        file_path = f"{WORKDIR}/SKNanoOutput/MeasFakeRateV4/{lepton_type}_RunSyst{suffix}/{args.era}/{sample}.root"
        assert os.path.exists(file_path), f"File not found: {file_path}"
        f = ROOT.TFile.Open(file_path)
        try:
            h = f.Get(histkey_data); h.SetDirectory(0)
            err = c_double(0.)
            rate_data += h.IntegralAndError(0, h.GetNbinsX()+1, err)
            rate_data_err_sq += err.value ** 2
            f.Close()
        except:
            logging.debug(f"Cannot find {histkey_data} for sample {sample}")
            f.Close()
            continue

    rate_mc = 0.
    rate_mc_err_sq = 0.
    for sample in MCList:
        file_path = f"{WORKDIR}/SKNanoOutput/MeasFakeRateV4/{lepton_type}_RunSyst{suffix}/{args.era}/{sample}.root"
        assert os.path.exists(file_path), f"File not found: {file_path}"
        f = ROOT.TFile.Open(file_path)
        try:
            h = f.Get(histkey_mc); h.SetDirectory(0)
            err = c_double(0.)
            rate_mc += h.IntegralAndError(0, h.GetNbinsX()+1, err)
            rate_mc_err_sq += err.value ** 2
            f.Close()
        except:
            logging.debug(f"Cannot find {histkey_mc} for sample {sample}")
            f.Close()
            continue

    scale = rate_data / rate_mc

    # Fractional stat uncertainty: delta(scale)/scale = sqrt((delta_data/data)^2 + (delta_mc/mc)^2)
    stat_pct = 0.
    if rate_data > 0 and rate_mc > 0:
        stat_pct = math.sqrt(rate_data_err_sq / rate_data**2 + rate_mc_err_sq / rate_mc**2) * 100.

    if syst == "PromptNorm_Up": scale *= 1.15
    if syst == "PromptNorm_Down": scale *= 0.85

    return scale, stat_pct

def extract_fake_from_data(ptcorr, abseta, wp, syst):
    """
    Extract fake lepton count and error from data minus scaled prompt MC.

    Returns:
        tuple: (fake_count, error) where error is propagated statistical uncertainty
    """
    import math
    prefix = findbin(ptcorr, abseta, ptcorr_bins, abseta_bins)

    # get integral and error for data
    subdir = "noHEMVeto/" if args.noHEMVeto else ""
    data = 0.
    data_err_sq = 0.
    for sample in DATAPERIODs:
        json_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/JSON/{args.measure}/{subdir}{sample}_{wp}.json"
        with open(json_path, 'r') as f:
            data_dict = json.load(f)
        if syst in ["PromptNorm_Up", "PromptNorm_Down"]:
            data += data_dict["Central"][prefix]
        else:
            data += data_dict[syst][prefix]
        data_err_sq += data_dict["Stat"][prefix] ** 2

    # get MCs integral and error
    prompt = 0.
    prompt_err_sq = 0.
    for sample in MCList:
        json_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/JSON/{args.measure}/{subdir}{sample}_{wp}.json"
        with open(json_path, 'r') as f:
            data_dict = json.load(f)
        if syst in ["PromptNorm_Up", "PromptNorm_Down"]:
            prompt += data_dict["Central"][prefix]
        else:
            prompt += data_dict[syst][prefix]
        prompt_err_sq += data_dict["Stat"][prefix] ** 2

    # get prompt scale
    trig_prefix = get_trig_prefix(ptcorr)
    scale, _ = get_prompt_scale(trig_prefix, wp, syst)
    logging.debug(f"{prefix} {data} {prompt} {scale} {prompt*scale}")

    # fake = data - prompt * scale
    # error = sqrt(data_err^2 + (scale * prompt_err)^2)
    fake = data - prompt * scale
    error = math.sqrt(data_err_sq + (scale ** 2) * prompt_err_sq)

    return fake, error

def get_trig_prefix(ptcorr):
    """Get trigger prefix based on ptcorr and measurement type."""
    if args.measure == "electron":
        if ptcorr < 20.: return "El8"
        elif ptcorr < 35.: return "El12"
        else: return "El23"
    elif args.measure == "muon":
        if ptcorr < 30.: return "Mu8"
        else: return "Mu17"
    else:
        raise KeyError(f"Wrong measure {args.measure}")

def extract_fake_from_mc(ptcorr, abseta, wp, sample_list, flavour=None):
    """
    Extract fake lepton integral and error from MC samples.

    Args:
        ptcorr: pT corrected value (lower bin edge)
        abseta: absolute eta value (lower bin edge)
        wp: working point ("loose" or "tight")
        sample_list: list of sample names to sum over
        flavour: None for inclusive, or one of ["ljet", "cjet", "bjet", "pujet"]

    Returns:
        tuple: (integral, error) summed from all samples
    """
    import math
    from ctypes import c_double

    prefix = findbin(ptcorr, abseta, ptcorr_bins, abseta_bins)
    trig_prefix = get_trig_prefix(ptcorr)
    lepton_type = "MeasFakeEl" if args.measure == "electron" else "MeasFakeMu"

    # Build histogram key based on flavour
    if flavour is None or flavour == "inclusive":
        histkey = f"{prefix}/Inclusive/{trig_prefix}/{wp}/Central/MT"
    else:
        histkey = f"{prefix}/Inclusive/{trig_prefix}/{wp}/Central/{flavour}/MT"

    integral = 0.
    error_sq = 0.
    for sample in sample_list:
        file_path = f"{WORKDIR}/SKNanoOutput/MeasFakeRateV4/{lepton_type}/{args.era}/{sample}.root"
        if not os.path.exists(file_path):
            logging.warning(f"File not found: {file_path}")
            continue
        f = ROOT.TFile.Open(file_path)
        try:
            h = f.Get(histkey)
            if h is None:
                logging.debug(f"Cannot find {histkey} for sample {sample}")
                f.Close()
                continue
            h.SetDirectory(0)
            err = c_double(0.)
            integral += h.IntegralAndError(0, h.GetNbinsX()+1, err)
            error_sq += err.value ** 2
            f.Close()
        except Exception as e:
            logging.debug(f"Error reading {histkey} for sample {sample}: {e}")
            f.Close()
            continue

    return integral, math.sqrt(error_sq)

def extract_fake_from_qcd(ptcorr, abseta, wp, samplegroup, flavour=None):
    """Backward-compatible wrapper for QCD extraction."""
    return extract_fake_from_mc(ptcorr, abseta, wp, SAMPLEGROUP[samplegroup], flavour)

def extract_fake_from_ttjj(ptcorr, abseta, wp, flavour=None):
    """Extract fake lepton integral from TTJJ_powheg sample."""
    return extract_fake_from_mc(ptcorr, abseta, wp, [TTJJ], flavour)

def get_fake_rate(mode="data", sample_identifier="Central", flavour=None):
    """
    Calculate fake rate histogram with proper error propagation.

    Args:
        mode: One of "data", "qcd", "ttjj"
        sample_identifier:
            - For "data" mode: systematic name (e.g., "Central", "PromptNorm_Up")
            - For "qcd" mode: sample group name (e.g., "QCD_MuEnriched")
            - For "ttjj" mode: not used (always TTJJ_powheg)
        flavour: None for inclusive or one of ["ljet", "cjet", "bjet", "pujet"]

    Returns:
        TH2D: Fake rate histogram with statistical errors
    """
    h_loose = ROOT.TH2D("h_loose", "h_loose", len(abseta_bins)-1, array('d', abseta_bins), len(ptcorr_bins)-1, array('d', ptcorr_bins))
    h_tight = ROOT.TH2D("h_tight", "h_tight", len(abseta_bins)-1, array('d', abseta_bins), len(ptcorr_bins)-1, array('d', ptcorr_bins))
    h_loose.SetDirectory(0)
    h_tight.SetDirectory(0)
    h_loose.Sumw2()
    h_tight.Sumw2()

    for ptcorr, abseta in product(ptcorr_bins[:-1], abseta_bins[:-1]):
        if mode == "data":
            loose, loose_err = extract_fake_from_data(ptcorr, abseta, "loose", sample_identifier)
            tight, tight_err = extract_fake_from_data(ptcorr, abseta, "tight", sample_identifier)
        elif mode == "qcd":
            loose, loose_err = extract_fake_from_qcd(ptcorr, abseta, "loose", sample_identifier, flavour)
            tight, tight_err = extract_fake_from_qcd(ptcorr, abseta, "tight", sample_identifier, flavour)
        elif mode == "ttjj":
            loose, loose_err = extract_fake_from_ttjj(ptcorr, abseta, "loose", flavour)
            tight, tight_err = extract_fake_from_ttjj(ptcorr, abseta, "tight", flavour)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Find bin and set content with error
        bin_loose = h_loose.FindBin(abseta, ptcorr)
        bin_tight = h_tight.FindBin(abseta, ptcorr)
        h_loose.SetBinContent(bin_loose, loose)
        h_loose.SetBinError(bin_loose, loose_err)
        h_tight.SetBinContent(bin_tight, tight)
        h_tight.SetBinError(bin_tight, tight_err)

    # Build histogram name with simplified naming
    if mode == "data":
        hist_name = f"fake rate - ({sample_identifier})"
    elif mode == "qcd":
        flavour_suffix = f"_{flavour}" if flavour and flavour != "inclusive" else ""
        hist_name = f"fake rate - ({sample_identifier}{flavour_suffix})"
    elif mode == "ttjj":
        flavour_suffix = f"_{flavour}" if flavour and flavour != "inclusive" else ""
        hist_name = f"fake rate - (TT{flavour_suffix})"

    # Calculate fake rate with proper error for correlated numerator/denominator
    # tight ⊂ loose, so ROOT's Divide() would overestimate error by assuming independence
    import math
    fake_rate = h_tight.Clone(hist_name)
    fake_rate.SetDirectory(0)
    for bx in range(1, fake_rate.GetNbinsX() + 1):
        for by in range(1, fake_rate.GetNbinsY() + 1):
            t = h_tight.GetBinContent(bx, by)
            l = h_loose.GetBinContent(bx, by)
            t_err = h_tight.GetBinError(bx, by)
            l_err = h_loose.GetBinError(bx, by)

            if l > 0:
                ratio = t / l
                # Binomial-like error for efficiency (tight ⊂ loose)
                # For weighted events, use effective N from loose error
                n_eff = (l / l_err) ** 2 if l_err > 0 else l
                err = math.sqrt(ratio * (1 - ratio) / n_eff) if 0 < ratio < 1 else t_err / l
            else:
                ratio = 0.
                err = 0.

            fake_rate.SetBinContent(bx, by, ratio)
            fake_rate.SetBinError(bx, by, err)

    fake_rate.SetTitle(hist_name)
    return fake_rate

if __name__ == "__main__":
    #### Collect scales (only for data-based measurement)
    subdir = "noHEMVeto/" if args.noHEMVeto else ""
    if not (args.isQCD or args.isTT or args.isMC):
        import math
        scale_dict = {}
        stat_dict = {}  # store stat uncertainty for central
        AllVariationsForScale = SelectionVariations + PhysicsVariations
        for trig_prefix, wp, syst in product(TrigPrefixes, WPs, AllVariationsForScale):
            scale, stat_pct = get_prompt_scale(trig_prefix, wp, syst)
            scale_dict[f"{trig_prefix}_{wp}_{syst}"] = scale
            if syst == "Central":
                stat_dict[f"{trig_prefix}_{wp}"] = stat_pct

        # Build summary: per-systematic % variation and total uncertainty
        summary = {}
        for trig_prefix, wp in product(TrigPrefixes, WPs):
            key = f"{trig_prefix}_{wp}"
            central = scale_dict[f"{key}_Central"]
            stat_pct = stat_dict[key]

            entry = {"central": central, "stat": round(stat_pct, 4)}
            syst_sq_sum = 0.
            for syst in PhysicsVariations:
                variation_pct = (scale_dict[f"{key}_{syst}"] - central) / central * 100.
                entry[syst] = round(variation_pct, 4)
                syst_sq_sum += variation_pct ** 2
            # total = quadrature sum of all physics syst deviations + stat
            total_pct = math.sqrt(syst_sq_sum + stat_pct ** 2)
            entry["total"] = round(total_pct, 4)
            summary[key] = entry

        scale_dict["summary"] = summary

        #### Save scales
        json_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/JSON/{args.measure}/{subdir}prompt_scale.json"
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(scale_dict, f, indent=2)

    #### Determine output path based on mode
    base_output_path = f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/ROOT/{args.measure}/{subdir.rstrip('/')}" if subdir else f"{WORKDIR}/MeasFakeRateV4/results/{args.era}/ROOT/{args.measure}"
    os.makedirs(base_output_path, exist_ok=True)

    if args.isTT:
        # TTJJ_powheg measurement with flavour breakdown
        output_path = f"{base_output_path}/fakerate_TTJJ.root"
        out = ROOT.TFile.Open(output_path, "RECREATE")
        for flavour in FLAVOURS_WITH_INCLUSIVE:
            flav = flavour if flavour != "inclusive" else None
            h = get_fake_rate(mode="ttjj", flavour=flav)
            out.cd()
            h.Write()
        out.Close()
        logging.info(f"Written: {output_path}")

    elif args.isMC:
        # Combined QCD + TTJJ measurement (separately, not merged)
        output_path = f"{base_output_path}/fakerate_MC.root"
        out = ROOT.TFile.Open(output_path, "RECREATE")

        # Write QCD samples with flavours
        for sample in QCD:
            for flavour in FLAVOURS_WITH_INCLUSIVE:
                flav = flavour if flavour != "inclusive" else None
                h = get_fake_rate(mode="qcd", sample_identifier=sample, flavour=flav)
                out.cd()
                h.Write()

        # Write TTJJ with flavours
        for flavour in FLAVOURS_WITH_INCLUSIVE:
            flav = flavour if flavour != "inclusive" else None
            h = get_fake_rate(mode="ttjj", flavour=flav)
            out.cd()
            h.Write()

        out.Close()
        logging.info(f"Written: {output_path}")

    elif args.isQCD:
        # QCD measurement with flavour breakdown
        output_path = f"{base_output_path}/fakerate_qcd.root"
        out = ROOT.TFile.Open(output_path, "RECREATE")
        for sample in QCD:
            for flavour in FLAVOURS_WITH_INCLUSIVE:
                flav = flavour if flavour != "inclusive" else None
                h = get_fake_rate(mode="qcd", sample_identifier=sample, flavour=flav)
                out.cd()
                h.Write()
        out.Close()
        logging.info(f"Written: {output_path}")

    else:
        # Data-based measurement (no flavour breakdown)
        output_path = f"{base_output_path}/fakerate.root"
        out = ROOT.TFile.Open(output_path, "RECREATE")
        for syst in SelectionVariations:
            h = get_fake_rate(mode="data", sample_identifier=syst)
            out.cd()
            h.Write()
        out.Close()
        logging.info(f"Written: {output_path}")
