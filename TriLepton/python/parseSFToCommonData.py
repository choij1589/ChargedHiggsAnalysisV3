#!/usr/bin/env python
"""Parse ConvSF and WZNjSF correctionlib results into simple JSON files
in the same format as Common/Data/FakeNorm.json for downstream consumption.

Values stored as fractional uncertainties (e.g. 0.10 = 10%):
  stat     — statistical uncertainty from CR measurement
  nonprompt — nonprompt normalization uncertainty
  prompt   — combined theory norm uncertainty for subtracted backgrounds (quadrature sum)
  total    — total uncertainty (quadrature sum of all three)

Outputs:
  Common/Data/ConvSF.json   — per-era, per-channel
  Common/Data/WZNjSF.json   — per-run, per-jet-bin
"""
import os
import json
from math import sqrt

WORKDIR = os.environ["WORKDIR"]
RESULTS_DIR = f"{WORKDIR}/TriLepton/results"
COMMON_DATA_DIR = f"{WORKDIR}/Common/Data"

RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]

# Theory norm correction suffixes present in ConvSF output
CONV_THEORY_SUFFIXES = ["_theory_diboson_up", "_theory_ttX_up", "_theory_others_up"]
# Theory norm correction suffixes present in WZNjSF output
WZNJ_THEORY_SUFFIXES = ["_theory_ZZ_up", "_theory_conv_up", "_theory_ttX_up", "_theory_others_up"]


def load_corrections(path):
    """Load correctionlib JSON and return {name: correction} dict."""
    with open(path) as f:
        cset = json.load(f)
    return {c["name"]: c for c in cset["corrections"]}


def get_formula(corrections, suffix):
    """Extract float from a formula correction matching the given name suffix."""
    key = next((k for k in corrections if k.endswith(suffix)), None)
    if key is None:
        raise KeyError(f"No correction ending with '{suffix}' found")
    return float(corrections[key]["data"]["expression"])


def get_binning(corrections, suffix):
    """Extract content list and edge-derived labels from a binning correction."""
    key = next((k for k in corrections if k.endswith(suffix)), None)
    if key is None:
        raise KeyError(f"No correction ending with '{suffix}' found")
    data = corrections[key]["data"]
    content = data["content"]
    edges = data["edges"]
    # Last bin is the merged high-nj overflow; label as "{n}j+"
    nbins = len(content)
    labels = [f"{int(edges[i])}j" for i in range(nbins - 1)] + [f"{int(edges[-2])}j+"]
    return content, labels


def frac(up_val, central):
    """Fractional deviation of up variation from central."""
    return abs(up_val - central) / central if central != 0 else 0.0


def prompt_frac_formula(corr, theory_suffixes, central):
    """Quadrature sum of fractional theory norm deviations (ConvSF — scalar)."""
    syst_sq = 0.0
    for sfx in theory_suffixes:
        try:
            up = get_formula(corr, sfx)
            syst_sq += frac(up, central) ** 2
        except KeyError:
            pass
    return sqrt(syst_sq)


def prompt_frac_binning(corr, theory_suffixes, central_bins):
    """Quadrature sum of fractional theory norm deviations per bin (WZNjSF — binned)."""
    nbins = len(central_bins)
    syst_sq = [0.0] * nbins
    for sfx in theory_suffixes:
        try:
            up_bins, _ = get_binning(corr, sfx)
            for i in range(nbins):
                syst_sq[i] += frac(up_bins[i], central_bins[i]) ** 2
        except KeyError:
            pass
    return [sqrt(v) for v in syst_sq]


# ── ConvSF ────────────────────────────────────────────────────────────────────

conv_data = {
    "description": (
        "Conversion scale factors (K-factors for DYJets/TTG/WWG) measured in ZG control regions. "
        "Per-era, per-channel. Uncertainties stored as fractions (e.g. 0.10 = 10%): "
        "stat (CR measurement), nonprompt (per-era FakeNorm.json), "
        "prompt (quadrature sum of theory norms for subtracted backgrounds), "
        "total (quadrature sum of all three)."
    ),
    "1E2Mu": {},
    "3Mu": {},
}

channel_map = {"ZG1E2Mu": "1E2Mu", "ZG3Mu": "3Mu"}

for src_channel, dst_channel in channel_map.items():
    for era in RUN2_ERAS + RUN3_ERAS:
        path = f"{RESULTS_DIR}/{src_channel}/{era}/ConvSF.json"
        if not os.path.exists(path):
            print(f"WARNING: {path} not found, skipping")
            continue
        try:
            corr = load_corrections(path)
            central    = get_formula(corr, "_Central")
            stat_up    = get_formula(corr, "_statistical_up")
            np_up      = get_formula(corr, "_nonprompt_up")
            total_up   = get_formula(corr, "_total_up")
        except (KeyError, ValueError) as e:
            print(f"WARNING: Failed to parse {path}: {e}")
            continue

        stat_f     = round(frac(stat_up, central), 6)
        nonprompt_f = round(frac(np_up, central), 6)
        prompt_f   = round(prompt_frac_formula(corr, CONV_THEORY_SUFFIXES, central), 6)
        total_f    = round(frac(total_up, central), 6)

        conv_data[dst_channel][era] = {
            "central":   round(central, 6),
            "stat":      stat_f,
            "nonprompt": nonprompt_f,
            "prompt":    prompt_f,
            "total":     total_f,
        }
        print(f"ConvSF {dst_channel}/{era}: {central:.4f}  "
              f"stat={stat_f:.3f}  nonprompt={nonprompt_f:.3f}  "
              f"prompt={prompt_f:.3f}  total={total_f:.3f}")

conv_out = f"{COMMON_DATA_DIR}/ConvSF.json"
os.makedirs(os.path.dirname(conv_out), exist_ok=True)
with open(conv_out, "w") as f:
    json.dump(conv_data, f, indent=4)
print(f"Written: {conv_out}\n")


# ── WZNjSF ────────────────────────────────────────────────────────────────────

wznj_data = {
    "description": (
        "WZ N-jets scale factors (K-factors for WZ) measured in WZ combined control regions. "
        "Per-run, per-jet-bin. Last bin (Nj+) is the merged overflow. "
        "Uncertainties stored as fractions (e.g. 0.10 = 10%), all bin-by-bin: "
        "stat (CR measurement), nonprompt (per-era FakeNorm.json), "
        "prompt (quadrature sum of theory norms for subtracted backgrounds), "
        "total (quadrature sum of all three)."
    ),
    "Run2": {},
    "Run3": {},
}

for run in ["Run2", "Run3"]:
    path = f"{RESULTS_DIR}/WZCombined/{run}/WZNjetsSF.json"
    if not os.path.exists(path):
        print(f"WARNING: {path} not found, skipping")
        continue
    try:
        corr = load_corrections(path)
        central_bins, labels = get_binning(corr, "_Central")
        stat_up_bins,   _   = get_binning(corr, "_statistical_up")
        np_up_bins,     _   = get_binning(corr, "_nonprompt_up")
        total_up_bins,  _   = get_binning(corr, "_total_up")
    except (KeyError, ValueError) as e:
        print(f"WARNING: Failed to parse {path}: {e}")
        continue

    prompt_fracs = prompt_frac_binning(corr, WZNJ_THEORY_SUFFIXES, central_bins)

    print(f"WZNjSF {run}:")
    for i, label in enumerate(labels):
        c = central_bins[i]
        stat_f     = round(frac(stat_up_bins[i], c), 6)
        nonprompt_f = round(frac(np_up_bins[i], c), 6)
        prompt_f   = round(prompt_fracs[i], 6)
        total_f    = round(frac(total_up_bins[i], c), 6)
        wznj_data[run][label] = {
            "central":   round(c, 6),
            "stat":      stat_f,
            "nonprompt": nonprompt_f,
            "prompt":    prompt_f,
            "total":     total_f,
        }
        print(f"  {label}: {c:.4f}  stat={stat_f:.3f}  nonprompt={nonprompt_f:.3f}  "
              f"prompt={prompt_f:.3f}  total={total_f:.3f}")

wznj_out = f"{COMMON_DATA_DIR}/WZNjSF.json"
with open(wznj_out, "w") as f:
    json.dump(wznj_data, f, indent=4)
print(f"\nWritten: {wznj_out}")
