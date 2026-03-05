#!/usr/bin/env python3
"""Compute training budget: raw event counts and effective sumW for all categories.

Categories and selections:
  Signal       -- Tight+Bjet, weight = genWeight * puWeight * prefireWeight
  Nonprompt    -- LNT+Bjet,   weight = genWeight * puWeight * prefireWeight * FR_weight
  Diboson      -- Tight+0tag+nJets>0, weight = genWeight * puWeight * prefireWeight
  ttX          -- Tight+Bjet, weight = genWeight * puWeight * prefireWeight

Output: DataAugment/training_budget.json
"""

import os
import json
import ROOT

ROOT.gROOT.SetBatch(True)

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
WORKDIR = os.environ.get("WORKDIR")
if not WORKDIR:
    raise RuntimeError("WORKDIR not set. Run 'source setup.sh' first.")

BASEDIR = os.path.join(WORKDIR, "SKNanoOutput", "EvtTreeProducer")
OUTDIR = os.path.join(WORKDIR, "ParticleNetMD", "DataAugment")

CHANNELS = ["Run1E2Mu", "Run3Mu"]
RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]
ALL_ERAS = RUN2_ERAS + RUN3_ERAS

# --- Signal samples (no Skim_TriLep_ prefix) ---
SIGNAL_SAMPLES = {
    "MHc100_MA95":  {"file": "TTToHcToWAToMuMu-MHc100_MA95",  "eras": ALL_ERAS},
    "MHc115_MA87":  {"file": "TTToHcToWAToMuMu-MHc115_MA87",  "eras": RUN2_ERAS},
    "MHc130_MA90":  {"file": "TTToHcToWAToMuMu-MHc130_MA90",  "eras": ALL_ERAS},
    "MHc145_MA92":  {"file": "TTToHcToWAToMuMu-MHc145_MA92",  "eras": RUN2_ERAS},
    "MHc160_MA85":  {"file": "TTToHcToWAToMuMu-MHc160_MA85",  "eras": ALL_ERAS},
    "MHc160_MA98":  {"file": "TTToHcToWAToMuMu-MHc160_MA98",  "eras": RUN2_ERAS},
}

# --- Nonprompt samples (9 TTLL, no Skim_TriLep_ prefix) ---
NONPROMPT_SAMPLES = [
    "TTLL_powheg",
    "TTLL_mtop171p5_powheg",
    "TTLL_mtop173p5_powheg",
    "TTLL_TuneCP5up_powheg",
    "TTLL_TuneCP5down_powheg",
    "TTLL_TuneCP5CR1_powheg",
    "TTLL_TuneCP5CR2_powheg",
    "TTLL_hdamp_up_powheg",
    "TTLL_hdamp_down_powheg",
]

# --- Diboson samples (Skim_TriLep_ prefix) ---
WZ_SAMPLES = {
    "2016preVFP":  "Skim_TriLep_WZTo3LNu_amcatnlo",
    "2016postVFP": "Skim_TriLep_WZTo3LNu_amcatnlo",
    "2017":        "Skim_TriLep_WZTo3LNu_amcatnlo",
    "2018":        "Skim_TriLep_WZTo3LNu_amcatnlo",
    "2022":        "Skim_TriLep_WZTo3LNu_powheg",
    "2022EE":      "Skim_TriLep_WZTo3LNu_powheg",
    "2023":        "Skim_TriLep_WZTo3LNu_powheg",
    "2023BPix":    "Skim_TriLep_WZTo3LNu_powheg",
}
ZZ_SAMPLE = "Skim_TriLep_ZZTo4L_powheg"

# --- ttX samples (Skim_TriLep_ prefix) ---
TTX_SAMPLES = [
    {"name": "TTZ",
     "file": {"Run2": "Skim_TriLep_TTZToLLNuNu", "Run3": "Skim_TriLep_TTZ_M50"}},
    {"name": "tZq", "file": "Skim_TriLep_tZq"},
    {"name": "TTH", "file": "Skim_TriLep_TTHToNonbb"},
]

# --- Fake rate bin edges (matching correctionlib JSONs) ---
MU_PTCORR_EDGES = [10., 12., 14., 17., 20., 30., 50., 100.]
MU_ABSETA_EDGES = [0., 0.9, 1.6, 2.4]
EL_PTCORR_EDGES = [15., 17., 20., 25., 35., 50., 100.]
EL_ABSETA_EDGES = [0., 0.8, 1.479, 2.5]

N_MU_ETA = len(MU_ABSETA_EDGES) - 1   # 3
N_MU_PT = len(MU_PTCORR_EDGES) - 1    # 7
N_EL_ETA = len(EL_ABSETA_EDGES) - 1   # 3
N_EL_PT = len(EL_PTCORR_EDGES) - 1    # 6

# Correctionlib data path
SKNANO_DATA = os.environ.get("SKNANO_DATA", "")
if not SKNANO_DATA:
    SKNANO_DATA = os.path.join(
        os.path.dirname(WORKDIR), "SKNanoAnalyzer", "data",
        "Run3_v13_Run2_v9")


# ---------------------------------------------------------------------------
# C++ helpers
# ---------------------------------------------------------------------------
ROOT.gInterpreter.Declare("""
bool allTrue(const ROOT::VecOps::RVec<bool>& v) {
    for (auto x : v) { if (!x) return false; }
    return true;
}
bool anyTrue(const ROOT::VecOps::RVec<bool>& v) {
    for (auto x : v) { if (x) return true; }
    return false;
}
bool isLNT(const ROOT::VecOps::RVec<bool>& muIsTight,
           const ROOT::VecOps::RVec<bool>& elIsTight) {
    for (auto t : muIsTight) { if (!t) return true; }
    for (auto t : elIsTight) { if (!t) return true; }
    return false;
}

// Fake rate bin finders
const double gMuPtEdges[] = {10., 12., 14., 17., 20., 30., 50., 100.};
const int gNMuPt = 7;
const double gMuEtaEdges[] = {0., 0.9, 1.6, 2.4};
const int gNMuEta = 3;
const double gElPtEdges[] = {15., 17., 20., 25., 35., 50., 100.};
const int gNElPt = 6;
const double gElEtaEdges[] = {0., 0.8, 1.479, 2.5};
const int gNElEta = 3;

int findMuPtBin(double ptCorr) {
    if (ptCorr < gMuPtEdges[0]) return 0;
    for (int i = 0; i < gNMuPt; i++) {
        if (ptCorr < gMuPtEdges[i+1]) return i;
    }
    return gNMuPt - 1;
}
int findMuEtaBin(double absEta) {
    if (absEta < gMuEtaEdges[0]) return 0;
    for (int i = 0; i < gNMuEta; i++) {
        if (absEta < gMuEtaEdges[i+1]) return i;
    }
    return gNMuEta - 1;
}
int findElPtBin(double ptCorr) {
    if (ptCorr < gElPtEdges[0]) return 0;
    for (int i = 0; i < gNElPt; i++) {
        if (ptCorr < gElPtEdges[i+1]) return i;
    }
    return gNElPt - 1;
}
int findElEtaBin(double absEta) {
    if (absEta < gElEtaEdges[0]) return 0;
    for (int i = 0; i < gNElEta; i++) {
        if (absEta < gElEtaEdges[i+1]) return i;
    }
    return gNElEta - 1;
}
""")


# ---------------------------------------------------------------------------
# Fake rate loading & registration (reused from nonpromptPromotion.py)
# ---------------------------------------------------------------------------
def _extract_rates(json_path, correction_name, n_eta, n_pt):
    """Extract 2D fake rate array from correctionlib JSON."""
    with open(json_path) as f:
        data = json.load(f)
    for corr in data["corrections"]:
        if corr["name"] == correction_name:
            content = corr["data"]["content"]
            if len(content) != n_eta * n_pt:
                raise ValueError(
                    f"{correction_name}: expected {n_eta*n_pt} values, "
                    f"got {len(content)}")
            rates = []
            for ie in range(n_eta):
                row = content[ie * n_pt : (ie + 1) * n_pt]
                rates.append(row)
            return rates
    raise KeyError(f"{correction_name} not found in {json_path}")


def load_and_register_fakerates():
    """Load fake rates from correctionlib JSONs and register C++ functions."""
    registered = []
    for era in ALL_ERAS:
        mu_path = os.path.join(SKNANO_DATA, era, "MUO", "fakerate_TopHNT.json")
        el_path = os.path.join(SKNANO_DATA, era, "EGM", "fakerate_TopHNT.json")

        if not os.path.exists(mu_path) or not os.path.exists(el_path):
            print(f"  WARNING: Fake rate JSONs not found for {era}, skipping")
            continue

        mu_rates = _extract_rates(mu_path, "fakerate_muon_TT",
                                  N_MU_ETA, N_MU_PT)
        el_rates = _extract_rates(el_path, "fakerate_electron_TT",
                                  N_EL_ETA, N_EL_PT)

        is_run3 = era in RUN3_ERAS
        is_run3_str = "true" if is_run3 else "false"
        s = era

        mu_rows = []
        for ie in range(N_MU_ETA):
            row = ", ".join(f"{mu_rates[ie][ip]:.15f}" for ip in range(N_MU_PT))
            mu_rows.append("{" + row + "}")
        mu_str = ",\n        ".join(mu_rows)

        el_rows = []
        for ie in range(N_EL_ETA):
            row = ", ".join(f"{el_rates[ie][ip]:.15f}" for ip in range(N_EL_PT))
            el_rows.append("{" + row + "}")
        el_str = ",\n        ".join(el_rows)

        ROOT.gInterpreter.Declare(f"""
        const double gMuFR_{s}[{N_MU_ETA}][{N_MU_PT}] = {{
            {mu_str}
        }};
        const double gElFR_{s}[{N_EL_ETA}][{N_EL_PT}] = {{
            {el_str}
        }};
        double getFakeRateWeight_{s}(
                const ROOT::VecOps::RVec<float>& muPt,
                const ROOT::VecOps::RVec<float>& muEta,
                const ROOT::VecOps::RVec<float>& muMiniIso,
                const ROOT::VecOps::RVec<bool>&  muIsTight,
                const ROOT::VecOps::RVec<float>& elPt,
                const ROOT::VecOps::RVec<float>& elScEta,
                const ROOT::VecOps::RVec<float>& elMiniIso,
                const ROOT::VecOps::RVec<bool>&  elIsTight) {{
            double w = -1.0;
            for (int i = 0; i < (int)muPt.size(); i++) {{
                if (muIsTight[i]) continue;
                double ptCorr = muPt[i] * (1.0 + std::max(0.0, (double)muMiniIso[i] - 0.1));
                if ({is_run3_str} && ptCorr > 50.0) ptCorr = 49.0;
                double absEta = std::abs((double)muEta[i]);
                int etaBin = findMuEtaBin(absEta);
                int ptBin  = findMuPtBin(ptCorr);
                double f = gMuFR_{s}[etaBin][ptBin];
                w *= -1.0 * f / (1.0 - f);
            }}
            for (int i = 0; i < (int)elPt.size(); i++) {{
                if (elIsTight[i]) continue;
                double ptCorr = elPt[i] * (1.0 + std::max(0.0, (double)elMiniIso[i] - 0.1));
                double absEta = std::abs((double)elScEta[i]);
                int etaBin = findElEtaBin(absEta);
                int ptBin  = findElPtBin(ptCorr);
                double f = gElFR_{s}[etaBin][ptBin];
                w *= -1.0 * f / (1.0 - f);
            }}
            return w;
        }}
        """)
        registered.append(era)

    print(f"  Registered fake rate functions for {len(registered)} eras")
    return registered


# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------
def process_tight_bjet(filepath):
    """Tight+Bjet selection. Returns (count, sumW) or None."""
    if not os.path.exists(filepath):
        return None
    rdf = ROOT.RDataFrame("Events", filepath)
    rdf = rdf.Filter(
        "allTrue(MuonIsTightColl) && allTrue(ElectronIsTightColl)")
    rdf = rdf.Filter("anyTrue(JetIsBtaggedColl)")
    rdf = rdf.Define("evtW", "genWeight * puWeight * prefireWeight")
    n = rdf.Count()
    s = rdf.Sum("evtW")
    return (n.GetValue(), s.GetValue())


def process_lnt_bjet(filepath, era):
    """LNT+Bjet with fake rate weight. Returns (count, sumW_base, sumW_fr) or None."""
    if not os.path.exists(filepath):
        return None
    rdf = ROOT.RDataFrame("Events", filepath)
    rdf = rdf.Filter("isLNT(MuonIsTightColl, ElectronIsTightColl)")
    rdf = rdf.Filter("anyTrue(JetIsBtaggedColl)")
    rdf = rdf.Define("evtW", "genWeight * puWeight * prefireWeight")
    rdf = rdf.Define("frW",
        f"evtW * getFakeRateWeight_{era}("
        f"MuonPtColl, MuonEtaColl, MuonMiniIsoColl, MuonIsTightColl, "
        f"ElectronPtColl, ElectronScEtaColl, ElectronMiniIsoColl, "
        f"ElectronIsTightColl)")
    n = rdf.Count()
    s_base = rdf.Sum("evtW")
    s_fr = rdf.Sum("frW")
    return (n.GetValue(), s_base.GetValue(), s_fr.GetValue())


def process_tight_0tag(filepath):
    """Tight + 0-tag + nJets>0. Returns (count, sumW) or None."""
    if not os.path.exists(filepath):
        return None
    rdf = ROOT.RDataFrame("Events", filepath)
    rdf = rdf.Filter(
        "allTrue(MuonIsTightColl) && allTrue(ElectronIsTightColl)")
    rdf = rdf.Filter("!anyTrue(JetIsBtaggedColl) && nJets > 0")
    rdf = rdf.Define("evtW", "genWeight * puWeight * prefireWeight")
    n = rdf.Count()
    s = rdf.Sum("evtW")
    return (n.GetValue(), s.GetValue())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("Training Budget Computation")
    print("  Signal:    Tight+Bjet")
    print("  Nonprompt: LNT+Bjet (9 TTLL, FR-weighted)")
    print("  Diboson:   Tight+0tag+nJets>0 (WZ+ZZ)")
    print("  ttX:       Tight+Bjet (TTZ+tZq+TTH)")
    print("=" * 70)

    # Register fake rates for nonprompt
    print("\nLoading fake rates...")
    fr_eras = load_and_register_fakerates()

    output = {"signal": {}, "nonprompt": {}, "diboson": {}, "ttX": {}}

    # ==========================================
    # Signal
    # ==========================================
    print("\n" + "=" * 60)
    print("SIGNAL (Tight+Bjet)")
    print("=" * 60)

    for mass_point, cfg in SIGNAL_SAMPLES.items():
        run2 = {"count": 0, "sumW": 0.0}
        run3 = {"count": 0, "sumW": 0.0}

        for channel in CHANNELS:
            for era in cfg["eras"]:
                filepath = os.path.join(
                    BASEDIR, channel, era, f"{cfg['file']}.root")
                result = process_tight_bjet(filepath)
                if result is None:
                    continue
                count, sumW = result
                bucket = run2 if era in RUN2_ERAS else run3
                bucket["count"] += count
                bucket["sumW"] += sumW

        total_count = run2["count"] + run3["count"]
        total_sumW = run2["sumW"] + run3["sumW"]
        output["signal"][mass_point] = {
            "Run2": {"count": run2["count"],
                     "sumW": round(run2["sumW"], 4)},
            "Run3": {"count": run3["count"],
                     "sumW": round(run3["sumW"], 4)},
        }
        print(f"  {mass_point:<15} Run2={run2['count']:>7} ({run2['sumW']:>8.2f})  "
              f"Run3={run3['count']:>7} ({run3['sumW']:>8.2f})  "
              f"Total={total_count:>7} ({total_sumW:>8.2f})")

    # ==========================================
    # Nonprompt
    # ==========================================
    print("\n" + "=" * 60)
    print("NONPROMPT (9 TTLL, LNT+Bjet)")
    print("=" * 60)

    np_total_run2 = {"count": 0, "sumW_base": 0.0, "sumW_fr": 0.0}
    np_total_run3 = {"count": 0, "sumW_base": 0.0, "sumW_fr": 0.0}

    for sample in NONPROMPT_SAMPLES:
        run2 = {"count": 0, "sumW_base": 0.0, "sumW_fr": 0.0}
        run3 = {"count": 0, "sumW_base": 0.0, "sumW_fr": 0.0}

        for channel in CHANNELS:
            for era in ALL_ERAS:
                if era not in fr_eras:
                    continue
                filepath = os.path.join(
                    BASEDIR, channel, era, f"Skim_TriLep_{sample}.root")
                if not os.path.exists(filepath) or os.path.getsize(filepath) < 10_000:
                    continue
                result = process_lnt_bjet(filepath, era)
                if result is None:
                    continue
                count, sumW_base, sumW_fr = result
                bucket = run2 if era in RUN2_ERAS else run3
                bucket["count"] += count
                bucket["sumW_base"] += sumW_base
                bucket["sumW_fr"] += sumW_fr

        output["nonprompt"][sample] = {
            "Run2": {"count": run2["count"],
                     "sumW_base": round(run2["sumW_base"], 4),
                     "sumW_fr": round(run2["sumW_fr"], 4)},
            "Run3": {"count": run3["count"],
                     "sumW_base": round(run3["sumW_base"], 4),
                     "sumW_fr": round(run3["sumW_fr"], 4)},
        }
        for k in ("count", "sumW_base", "sumW_fr"):
            np_total_run2[k] += run2[k]
            np_total_run3[k] += run3[k]

        print(f"  {sample:<30} Run2={run2['count']:>6} (FR:{run2['sumW_fr']:>8.2f})  "
              f"Run3={run3['count']:>6} (FR:{run3['sumW_fr']:>8.2f})")

    print(f"  {'TOTAL':<30} Run2={np_total_run2['count']:>6} "
          f"(FR:{np_total_run2['sumW_fr']:>8.2f})  "
          f"Run3={np_total_run3['count']:>6} "
          f"(FR:{np_total_run3['sumW_fr']:>8.2f})")

    # ==========================================
    # Diboson
    # ==========================================
    print("\n" + "=" * 60)
    print("DIBOSON (WZ+ZZ, Tight+0tag+nJets>0)")
    print("=" * 60)

    for label in ("WZ", "ZZ"):
        run2 = {"count": 0, "sumW": 0.0}
        run3 = {"count": 0, "sumW": 0.0}

        for channel in CHANNELS:
            for era in ALL_ERAS:
                if label == "WZ":
                    sample_name = WZ_SAMPLES[era]
                else:
                    sample_name = ZZ_SAMPLE
                filepath = os.path.join(
                    BASEDIR, channel, era, f"{sample_name}.root")
                result = process_tight_0tag(filepath)
                if result is None:
                    continue
                count, sumW = result
                bucket = run2 if era in RUN2_ERAS else run3
                bucket["count"] += count
                bucket["sumW"] += sumW

        output["diboson"][label] = {
            "Run2": {"count": run2["count"],
                     "sumW": round(run2["sumW"], 4)},
            "Run3": {"count": run3["count"],
                     "sumW": round(run3["sumW"], 4)},
        }
        total_count = run2["count"] + run3["count"]
        total_sumW = run2["sumW"] + run3["sumW"]
        print(f"  {label:<6} Run2={run2['count']:>7} ({run2['sumW']:>8.2f})  "
              f"Run3={run3['count']:>7} ({run3['sumW']:>8.2f})  "
              f"Total={total_count:>7} ({total_sumW:>8.2f})")

    # ==========================================
    # ttX
    # ==========================================
    print("\n" + "=" * 60)
    print("ttX (TTZ+tZq+TTH, Tight+Bjet)")
    print("=" * 60)

    for sample_cfg in TTX_SAMPLES:
        name = sample_cfg["name"]
        run2 = {"count": 0, "sumW": 0.0}
        run3 = {"count": 0, "sumW": 0.0}

        for channel in CHANNELS:
            for era in ALL_ERAS:
                if isinstance(sample_cfg["file"], dict):
                    era_group = "Run2" if era in RUN2_ERAS else "Run3"
                    file_sample = sample_cfg["file"][era_group]
                else:
                    file_sample = sample_cfg["file"]
                filepath = os.path.join(
                    BASEDIR, channel, era, f"{file_sample}.root")
                result = process_tight_bjet(filepath)
                if result is None:
                    continue
                count, sumW = result
                bucket = run2 if era in RUN2_ERAS else run3
                bucket["count"] += count
                bucket["sumW"] += sumW

        output["ttX"][name] = {
            "Run2": {"count": run2["count"],
                     "sumW": round(run2["sumW"], 4)},
            "Run3": {"count": run3["count"],
                     "sumW": round(run3["sumW"], 4)},
        }
        total_count = run2["count"] + run3["count"]
        total_sumW = run2["sumW"] + run3["sumW"]
        print(f"  {name:<6} Run2={run2['count']:>7} ({run2['sumW']:>8.2f})  "
              f"Run3={run3['count']:>7} ({run3['sumW']:>8.2f})  "
              f"Total={total_count:>7} ({total_sumW:>8.2f})")

    # ==========================================
    # Save JSON
    # ==========================================
    os.makedirs(OUTDIR, exist_ok=True)
    json_path = os.path.join(OUTDIR, "training_budget.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nOutput saved to {json_path}")

    # ==========================================
    # Print summary table
    # ==========================================
    print("\n" + "=" * 70)
    print("SUMMARY (per-fold = total / 5)")
    print("=" * 70)
    print(f"{'Category':<14} {'Run2/fold':>10} {'Run3/fold':>10} "
          f"{'Total/fold':>11} {'Run2 sumW':>10} {'Run3 sumW':>10}")
    print("-" * 70)

    for mass_point in SIGNAL_SAMPLES:
        s = output["signal"][mass_point]
        r2 = s["Run2"]["count"]
        r3 = s["Run3"]["count"]
        print(f"  Sig {mass_point:<9} {r2//5:>8}   {r3//5:>8}   "
              f"{(r2+r3)//5:>9}   {s['Run2']['sumW']:>8.1f}   "
              f"{s['Run3']['sumW']:>8.1f}")

    # Nonprompt total
    np_r2 = sum(output["nonprompt"][s]["Run2"]["count"]
                for s in NONPROMPT_SAMPLES)
    np_r3 = sum(output["nonprompt"][s]["Run3"]["count"]
                for s in NONPROMPT_SAMPLES)
    np_fr2 = sum(output["nonprompt"][s]["Run2"]["sumW_fr"]
                 for s in NONPROMPT_SAMPLES)
    np_fr3 = sum(output["nonprompt"][s]["Run3"]["sumW_fr"]
                 for s in NONPROMPT_SAMPLES)
    print(f"  Nonprompt    {np_r2//5:>8}   {np_r3//5:>8}   "
          f"{(np_r2+np_r3)//5:>9}   {np_fr2:>8.1f}   {np_fr3:>8.1f}")

    # Diboson total
    db_r2 = sum(output["diboson"][s]["Run2"]["count"] for s in ("WZ", "ZZ"))
    db_r3 = sum(output["diboson"][s]["Run3"]["count"] for s in ("WZ", "ZZ"))
    db_sw2 = sum(output["diboson"][s]["Run2"]["sumW"] for s in ("WZ", "ZZ"))
    db_sw3 = sum(output["diboson"][s]["Run3"]["sumW"] for s in ("WZ", "ZZ"))
    print(f"  Diboson      {db_r2//5:>8}   {db_r3//5:>8}   "
          f"{(db_r2+db_r3)//5:>9}   {db_sw2:>8.1f}   {db_sw3:>8.1f}")

    # ttX total
    tx_r2 = sum(output["ttX"][s]["Run2"]["count"]
                for s in ("TTZ", "tZq", "TTH"))
    tx_r3 = sum(output["ttX"][s]["Run3"]["count"]
                for s in ("TTZ", "tZq", "TTH"))
    tx_sw2 = sum(output["ttX"][s]["Run2"]["sumW"]
                 for s in ("TTZ", "tZq", "TTH"))
    tx_sw3 = sum(output["ttX"][s]["Run3"]["sumW"]
                 for s in ("TTZ", "tZq", "TTH"))
    print(f"  ttX          {tx_r2//5:>8}   {tx_r3//5:>8}   "
          f"{(tx_r2+tx_r3)//5:>9}   {tx_sw2:>8.1f}   {tx_sw3:>8.1f}")

    print("=" * 70)


if __name__ == "__main__":
    main()
