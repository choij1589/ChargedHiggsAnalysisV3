#!/usr/bin/env python
"""Nonprompt (TTLL) LNT promotion validation.

Validates Loose-Not-Tight (LNT) + b-jet events reweighted with MC fake rates
from TTJJ as augmentation candidates for the nonprompt training class.

Four-way normalized comparison:
  1. Genuine (Tight+Bjet)     — target distribution
  2. LNT (raw pT)             — uncorrected LNT events
  3. LNT (ptCorr)             — kinematic correction only
  4. LNT (ptCorr + FR wt)     — fully corrected (ptCorr + fake rate weight)

Fake rates from correctionlib JSON:
  SKNanoAnalyzer/data/Run3_v13_Run2_v9/{era}/MUO/fakerate_TopHNT.json
  SKNanoAnalyzer/data/Run3_v13_Run2_v9/{era}/EGM/fakerate_TopHNT.json

Output:
  DataAugment/nonprompt/plots/lnt_promote/*.png
  DataAugment/nonprompt/plots/lnt_promote/summary.json
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import json
import ROOT
from collections import OrderedDict

ROOT.gROOT.SetBatch(True)

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
WORKDIR = os.environ.get("WORKDIR")
if not WORKDIR:
    raise RuntimeError("WORKDIR not set. Run 'source setup.sh' first.")

BASEDIR = os.path.join(WORKDIR, "SKNanoOutput", "EvtTreeProducer")
PLOTDIR = os.path.join(WORKDIR, "ParticleNetMD", "DataAugment", "nonprompt",
                       "plots", "lnt_promote")

# Correctionlib data path
SKNANO_DATA = os.environ.get("SKNANO_DATA", "")
if not SKNANO_DATA:
    SKNANO_DATA = os.path.join(
        os.path.dirname(WORKDIR), "SKNanoAnalyzer", "data",
        "Run3_v13_Run2_v9")
    if not os.path.isdir(SKNANO_DATA):
        raise RuntimeError(
            f"Cannot find correctionlib data at {SKNANO_DATA}. "
            "Set SKNANO_DATA environment variable.")

CHANNELS = ["Run1E2Mu", "Run3Mu"]
RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]
ALL_ERAS = RUN2_ERAS + RUN3_ERAS

TTLL_SAMPLES = [
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

# Fake rate bin edges (matching correctionlib JSONs)
MU_PTCORR_EDGES = [10., 12., 14., 17., 20., 30., 50., 100.]
MU_ABSETA_EDGES = [0., 0.9, 1.6, 2.4]
EL_PTCORR_EDGES = [15., 17., 20., 25., 35., 50., 100.]
EL_ABSETA_EDGES = [0., 0.8, 1.479, 2.5]

N_MU_ETA = len(MU_ABSETA_EDGES) - 1   # 3
N_MU_PT = len(MU_PTCORR_EDGES) - 1    # 7
N_EL_ETA = len(EL_ABSETA_EDGES) - 1   # 3
N_EL_PT = len(EL_PTCORR_EDGES) - 1    # 6

_JETS_BJETS = [
    ("jet1_pt",   {"xTitle": "Leading jet p_{T} [GeV]",       "xRange": [0, 200]}),
    ("jet1_eta",  {"xTitle": "Leading jet #eta",                "xRange": [-2.5, 2.5]}),
    ("jet2_pt",   {"xTitle": "Subleading jet p_{T} [GeV]",    "xRange": [0, 150]}),
    ("jet2_eta",  {"xTitle": "Subleading jet #eta",             "xRange": [-2.5, 2.5]}),
    ("jet3_pt",   {"xTitle": "Third jet p_{T} [GeV]",         "xRange": [0, 100]}),
    ("jet3_eta",  {"xTitle": "Third jet #eta",                  "xRange": [-2.5, 2.5]}),
    ("bjet1_pt",  {"xTitle": "Leading b-jet p_{T} [GeV]",     "xRange": [0, 200]}),
    ("bjet1_eta", {"xTitle": "Leading b-jet #eta",              "xRange": [-2.5, 2.5]}),
    ("bjet2_pt",  {"xTitle": "Subleading b-jet p_{T} [GeV]",  "xRange": [0, 200]}),
    ("bjet2_eta", {"xTitle": "Subleading b-jet #eta",           "xRange": [-2.5, 2.5]}),
    ("nJets_f",   {"xTitle": "Number of jets",                  "xRange": [0, 10]}),
    ("n_bjets_f", {"xTitle": "Number of b-jets",                "xRange": [0, 5]}),
]
PLOT_CONFIGS_E2MU = [
    ("el_pt",   {"xTitle": "Electron p_{T} [GeV]",          "xRange": [0, 200]}),
    ("el_eta",  {"xTitle": "Electron #eta",                   "xRange": [-2.5, 2.5]}),
    ("mu1_pt",  {"xTitle": "Leading muon p_{T} [GeV]",       "xRange": [0, 200]}),
    ("mu1_eta", {"xTitle": "Leading muon #eta",               "xRange": [-2.5, 2.5]}),
    ("mu2_pt",  {"xTitle": "Subleading muon p_{T} [GeV]",    "xRange": [0, 150]}),
    ("mu2_eta", {"xTitle": "Subleading muon #eta",            "xRange": [-2.5, 2.5]}),
] + _JETS_BJETS

PLOT_CONFIGS_3MU = [
    ("mu1_pt",  {"xTitle": "Leading muon p_{T} [GeV]",       "xRange": [0, 200]}),
    ("mu1_eta", {"xTitle": "Leading muon #eta",               "xRange": [-2.5, 2.5]}),
    ("mu2_pt",  {"xTitle": "Subleading muon p_{T} [GeV]",    "xRange": [0, 150]}),
    ("mu2_eta", {"xTitle": "Subleading muon #eta",            "xRange": [-2.5, 2.5]}),
    ("mu3_pt",  {"xTitle": "Third muon p_{T} [GeV]",         "xRange": [0, 100]}),
    ("mu3_eta", {"xTitle": "Third muon #eta",                  "xRange": [-2.5, 2.5]}),
] + _JETS_BJETS

PLOT_CONFIGS_BY_CHANNEL = {"Run1E2Mu": PLOT_CONFIGS_E2MU, "Run3Mu": PLOT_CONFIGS_3MU}


# ---------------------------------------------------------------------------
# Section 2: Load Fake Rates from Correctionlib JSON
# ---------------------------------------------------------------------------
def load_fakerate_tables():
    """Load TT fake rates from correctionlib JSONs for all 8 eras.

    Returns dict: era -> {"muon": 2D list [n_eta][n_pt],
                           "electron": 2D list [n_eta][n_pt]}
    """
    tables = {}
    for era in ALL_ERAS:
        mu_path = os.path.join(SKNANO_DATA, era, "MUO", "fakerate_TopHNT.json")
        el_path = os.path.join(SKNANO_DATA, era, "EGM", "fakerate_TopHNT.json")

        if not os.path.exists(mu_path):
            raise FileNotFoundError(f"Muon fake rate not found: {mu_path}")
        if not os.path.exists(el_path):
            raise FileNotFoundError(f"Electron fake rate not found: {el_path}")

        mu_rates = _extract_rates(mu_path, "fakerate_muon_TT",
                                  N_MU_ETA, N_MU_PT)
        el_rates = _extract_rates(el_path, "fakerate_electron_TT",
                                  N_EL_ETA, N_EL_PT)
        tables[era] = {"muon": mu_rates, "electron": el_rates}

    return tables


def _extract_rates(json_path, correction_name, n_eta, n_pt):
    """Extract 2D fake rate array from correctionlib JSON.

    Content is stored row-major: eta varies slowest, pt varies fastest.
    Returns list[n_eta][n_pt].
    """
    with open(json_path) as f:
        data = json.load(f)

    for corr in data["corrections"]:
        if corr["name"] == correction_name:
            content = corr["data"]["content"]
            if len(content) != n_eta * n_pt:
                raise ValueError(
                    f"{correction_name}: expected {n_eta*n_pt} values, "
                    f"got {len(content)}")
            # Reshape row-major to 2D
            rates = []
            for ie in range(n_eta):
                row = content[ie * n_pt : (ie + 1) * n_pt]
                rates.append(row)
            return rates

    raise KeyError(f"{correction_name} not found in {json_path}")


# ---------------------------------------------------------------------------
# Section 3: C++ Declarations
# ---------------------------------------------------------------------------
ROOT.gInterpreter.Declare("""
// --- Shared helpers (same as dibosonRankPromote.py) ---

bool allTrue(const ROOT::VecOps::RVec<bool>& v) {
    for (auto x : v) { if (!x) return false; }
    return true;
}
bool anyTrue(const ROOT::VecOps::RVec<bool>& v) {
    for (auto x : v) { if (x) return true; }
    return false;
}

ROOT::VecOps::RVec<int> ptSortedIdx(const ROOT::VecOps::RVec<float>& pt) {
    ROOT::VecOps::RVec<int> idx(pt.size());
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](int a, int b){ return pt[a] > pt[b]; });
    return idx;
}

// --- Lepton kinematics (3 leading, pt + eta) ---
struct LepKin { float pt1, eta1, pt2, eta2, pt3, eta3; };

LepKin getLeptonKin(const ROOT::VecOps::RVec<float>& muPt,
                    const ROOT::VecOps::RVec<float>& muEta,
                    const ROOT::VecOps::RVec<float>& elPt,
                    const ROOT::VecOps::RVec<float>& elEta) {
    std::vector<std::pair<float,float>> leps;
    for (int i = 0; i < (int)muPt.size(); i++)
        leps.push_back({muPt[i], muEta[i]});
    for (int i = 0; i < (int)elPt.size(); i++)
        leps.push_back({elPt[i], elEta[i]});
    std::sort(leps.begin(), leps.end(),
              [](auto& a, auto& b){ return a.first > b.first; });
    LepKin k{-1,-99, -1,-99, -1,-99};
    if (leps.size() >= 1) { k.pt1 = leps[0].first; k.eta1 = leps[0].second; }
    if (leps.size() >= 2) { k.pt2 = leps[1].first; k.eta2 = leps[1].second; }
    if (leps.size() >= 3) { k.pt3 = leps[2].first; k.eta3 = leps[2].second; }
    return k;
}

// --- getLeptonKin variant using ptCorr for non-tight leptons ---
LepKin getLeptonKinPtCorr(
        const ROOT::VecOps::RVec<float>& muPt,
        const ROOT::VecOps::RVec<float>& muEta,
        const ROOT::VecOps::RVec<float>& muMiniIso,
        const ROOT::VecOps::RVec<bool>&  muIsTight,
        const ROOT::VecOps::RVec<float>& elPt,
        const ROOT::VecOps::RVec<float>& elEta,
        const ROOT::VecOps::RVec<float>& elMiniIso,
        const ROOT::VecOps::RVec<bool>&  elIsTight,
        bool isRun3) {
    std::vector<std::pair<float,float>> leps;
    for (int i = 0; i < (int)muPt.size(); i++) {
        float pt = muPt[i];
        if (!muIsTight[i]) {
            pt = pt * (1.0f + std::max(0.0f, muMiniIso[i] - 0.1f));
        }
        leps.push_back({pt, muEta[i]});
    }
    for (int i = 0; i < (int)elPt.size(); i++) {
        float pt = elPt[i];
        if (!elIsTight[i]) {
            pt = pt * (1.0f + std::max(0.0f, elMiniIso[i] - 0.1f));
        }
        leps.push_back({pt, elEta[i]});
    }
    std::sort(leps.begin(), leps.end(),
              [](auto& a, auto& b){ return a.first > b.first; });
    LepKin k{-1,-99, -1,-99, -1,-99};
    if (leps.size() >= 1) { k.pt1 = leps[0].first; k.eta1 = leps[0].second; }
    if (leps.size() >= 2) { k.pt2 = leps[1].first; k.eta2 = leps[1].second; }
    if (leps.size() >= 3) { k.pt3 = leps[2].first; k.eta3 = leps[2].second; }
    return k;
}

// --- Flavor-split lepton kinematics ---
struct FlavorLepKin {
    float mu1_pt, mu1_eta, mu2_pt, mu2_eta, mu3_pt, mu3_eta;
    float el_pt, el_eta;
};
FlavorLepKin getFlavorLeptonKin(
        const ROOT::VecOps::RVec<float>& muPt, const ROOT::VecOps::RVec<float>& muEta,
        const ROOT::VecOps::RVec<float>& elPt, const ROOT::VecOps::RVec<float>& elEta) {
    auto muIdx = ptSortedIdx(muPt); auto elIdx = ptSortedIdx(elPt);
    FlavorLepKin k{-1,-99,-1,-99,-1,-99,-1,-99};
    if (muIdx.size()>=1){k.mu1_pt=muPt[muIdx[0]];k.mu1_eta=muEta[muIdx[0]];}
    if (muIdx.size()>=2){k.mu2_pt=muPt[muIdx[1]];k.mu2_eta=muEta[muIdx[1]];}
    if (muIdx.size()>=3){k.mu3_pt=muPt[muIdx[2]];k.mu3_eta=muEta[muIdx[2]];}
    if (elIdx.size()>=1){k.el_pt=elPt[elIdx[0]];k.el_eta=elEta[elIdx[0]];}
    return k;
}
FlavorLepKin getFlavorLeptonKinPtCorr(
        const ROOT::VecOps::RVec<float>& muPt, const ROOT::VecOps::RVec<float>& muEta,
        const ROOT::VecOps::RVec<float>& muMiniIso, const ROOT::VecOps::RVec<bool>& muIsTight,
        const ROOT::VecOps::RVec<float>& elPt, const ROOT::VecOps::RVec<float>& elEta,
        const ROOT::VecOps::RVec<float>& elMiniIso, const ROOT::VecOps::RVec<bool>& elIsTight,
        bool isRun3) {
    ROOT::VecOps::RVec<float> muPtCorr(muPt.size()), elPtCorr(elPt.size());
    for (int i = 0; i < (int)muPt.size(); i++) {
        float pt = muPt[i];
        if (!muIsTight[i]) {
            pt = pt * (1.0f + std::max(0.0f, muMiniIso[i] - 0.1f));
        }
        muPtCorr[i] = pt;
    }
    for (int i = 0; i < (int)elPt.size(); i++) {
        float pt = elPt[i];
        if (!elIsTight[i])
            pt = pt * (1.0f + std::max(0.0f, elMiniIso[i] - 0.1f));
        elPtCorr[i] = pt;
    }
    auto muIdx = ptSortedIdx(muPtCorr); auto elIdx = ptSortedIdx(elPtCorr);
    FlavorLepKin k{-1,-99,-1,-99,-1,-99,-1,-99};
    if (muIdx.size()>=1){k.mu1_pt=muPtCorr[muIdx[0]];k.mu1_eta=muEta[muIdx[0]];}
    if (muIdx.size()>=2){k.mu2_pt=muPtCorr[muIdx[1]];k.mu2_eta=muEta[muIdx[1]];}
    if (muIdx.size()>=3){k.mu3_pt=muPtCorr[muIdx[2]];k.mu3_eta=muEta[muIdx[2]];}
    if (elIdx.size()>=1){k.el_pt=elPtCorr[elIdx[0]];k.el_eta=elEta[elIdx[0]];}
    return k;
}

// --- Jet kinematics (3 leading, pt + eta) ---
struct JetKin { float pt1, eta1, pt2, eta2, pt3, eta3; };
JetKin getJetKin(const ROOT::VecOps::RVec<float>& pt,
                 const ROOT::VecOps::RVec<float>& eta) {
    auto idx = ptSortedIdx(pt);
    JetKin k{-1,-99, -1,-99, -1,-99};
    if (idx.size() >= 1) { k.pt1 = pt[idx[0]]; k.eta1 = eta[idx[0]]; }
    if (idx.size() >= 2) { k.pt2 = pt[idx[1]]; k.eta2 = eta[idx[1]]; }
    if (idx.size() >= 3) { k.pt3 = pt[idx[2]]; k.eta3 = eta[idx[2]]; }
    return k;
}

// --- Multi b-jet kinematics ---
struct MultiBjetKin {
    int n_bjets;
    float bjet1_pt, bjet1_eta, bjet2_pt, bjet2_eta;
};

MultiBjetKin getGenuineMultiBjetKin(const ROOT::VecOps::RVec<float>& pt,
                                     const ROOT::VecOps::RVec<float>& eta,
                                     const ROOT::VecOps::RVec<bool>& btag) {
    std::vector<std::pair<float,float>> bjets;
    for (int i = 0; i < (int)pt.size(); i++) {
        if (btag[i]) bjets.push_back({pt[i], eta[i]});
    }
    std::sort(bjets.begin(), bjets.end(),
              [](auto& a, auto& b){ return a.first > b.first; });
    MultiBjetKin k;
    k.n_bjets   = (int)bjets.size();
    k.bjet1_pt  = k.n_bjets >= 1 ? bjets[0].first  : -1;
    k.bjet1_eta = k.n_bjets >= 1 ? bjets[0].second  : -99;
    k.bjet2_pt  = k.n_bjets >= 2 ? bjets[1].first  : -1;
    k.bjet2_eta = k.n_bjets >= 2 ? bjets[1].second  : -99;
    return k;
}

// --- Check if event has any non-tight lepton ---
bool isLNT(const ROOT::VecOps::RVec<bool>& muIsTight,
           const ROOT::VecOps::RVec<bool>& elIsTight) {
    for (auto t : muIsTight) { if (!t) return true; }
    for (auto t : elIsTight) { if (!t) return true; }
    return false;
}

// --- Fake rate bin finders ---
const double gMuPtEdges[] = {10., 12., 14., 17., 20., 30., 50., 100.};
const int gNMuPt = 7;
const double gMuEtaEdges[] = {0., 0.9, 1.6, 2.4};
const int gNMuEta = 3;

const double gElPtEdges[] = {15., 17., 20., 25., 35., 50., 100.};
const int gNElPt = 6;
const double gElEtaEdges[] = {0., 0.8, 1.479, 2.5};
const int gNElEta = 3;

int findMuPtBin(double ptCorr) {
    // clamp: below first edge -> bin 0, above last edge -> last bin
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
# Section 4: Python Helper Functions
# ---------------------------------------------------------------------------
def register_fakerate_cpp(era, mu_rates, el_rates, is_run3):
    """Inject C++ fake rate arrays and register per-era getFakeRateWeight function.

    mu_rates: list[3][7], el_rates: list[3][6]
    """
    safe_era = era.replace("preVFP", "preVFP").replace("postVFP", "postVFP")
    safe_era = safe_era.replace("BPix", "BPix").replace("EE", "EE")
    s = safe_era

    # Format muon FR array
    mu_rows = []
    for ie in range(N_MU_ETA):
        row = ", ".join(f"{mu_rates[ie][ip]:.15f}" for ip in range(N_MU_PT))
        mu_rows.append("{" + row + "}")
    mu_str = ",\n        ".join(mu_rows)

    # Format electron FR array
    el_rows = []
    for ie in range(N_EL_ETA):
        row = ", ".join(f"{el_rates[ie][ip]:.15f}" for ip in range(N_EL_PT))
        el_rows.append("{" + row + "}")
    el_str = ",\n        ".join(el_rows)

    is_run3_str = "true" if is_run3 else "false"

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


def collect_files(eras, channels):
    """Collect TTLL ROOT file paths for given eras and channels.

    Skips files smaller than 10 KB (empty/failed jobs).
    """
    files = []
    for channel in channels:
        for era in eras:
            for sample in TTLL_SAMPLES:
                fpath = os.path.join(BASEDIR, channel, era,
                                     f"Skim_TriLep_{sample}.root")
                if os.path.exists(fpath) and os.path.getsize(fpath) > 10_000:
                    files.append(fpath)
    return files


def make_rdf(files):
    """Create RDataFrame from file list. Returns (rdf, chain)."""
    chain = ROOT.TChain("Events")
    for f in files:
        chain.Add(f)
    return ROOT.RDataFrame(chain), chain


def apply_tight(rdf):
    """Apply tight lepton selection."""
    return rdf.Filter(
        "allTrue(MuonIsTightColl) && allTrue(ElectronIsTightColl)")


def apply_lnt_bjet(rdf):
    """Filter: at least one non-tight lepton AND at least one b-jet."""
    return rdf.Filter(
        "isLNT(MuonIsTightColl, ElectronIsTightColl) "
        "&& anyTrue(JetIsBtaggedColl)")


def define_columns_raw(rdf):
    """Define lepton (raw pT), jet, and multiplicity columns."""
    rdf = rdf.Define("flavKin",
                      "getFlavorLeptonKin(MuonPtColl, MuonEtaColl, "
                      "ElectronPtColl, ElectronEtaColl)")
    rdf = rdf.Define("mu1_pt", "flavKin.mu1_pt").Define("mu1_eta", "flavKin.mu1_eta")
    rdf = rdf.Define("mu2_pt", "flavKin.mu2_pt").Define("mu2_eta", "flavKin.mu2_eta")
    rdf = rdf.Define("mu3_pt", "flavKin.mu3_pt").Define("mu3_eta", "flavKin.mu3_eta")
    rdf = rdf.Define("el_pt",  "flavKin.el_pt" ).Define("el_eta",  "flavKin.el_eta" )
    rdf = rdf.Define("jetKin", "getJetKin(JetPtColl, JetEtaColl)")
    rdf = rdf.Define("jet1_pt", "jetKin.pt1").Define("jet1_eta", "jetKin.eta1")
    rdf = rdf.Define("jet2_pt", "jetKin.pt2").Define("jet2_eta", "jetKin.eta2")
    rdf = rdf.Define("jet3_pt", "jetKin.pt3").Define("jet3_eta", "jetKin.eta3")
    rdf = rdf.Define("nJets_f", "(float)nJets")
    return rdf


def define_columns_ptcorr(rdf, is_run3):
    """Define lepton columns using ptCorr for non-tight leptons."""
    is_run3_str = "true" if is_run3 else "false"
    rdf = rdf.Define("flavKin",
                      f"getFlavorLeptonKinPtCorr(MuonPtColl, MuonEtaColl, "
                      f"MuonMiniIsoColl, MuonIsTightColl, "
                      f"ElectronPtColl, ElectronEtaColl, "
                      f"ElectronMiniIsoColl, ElectronIsTightColl, "
                      f"{is_run3_str})")
    rdf = rdf.Define("mu1_pt", "flavKin.mu1_pt").Define("mu1_eta", "flavKin.mu1_eta")
    rdf = rdf.Define("mu2_pt", "flavKin.mu2_pt").Define("mu2_eta", "flavKin.mu2_eta")
    rdf = rdf.Define("mu3_pt", "flavKin.mu3_pt").Define("mu3_eta", "flavKin.mu3_eta")
    rdf = rdf.Define("el_pt",  "flavKin.el_pt" ).Define("el_eta",  "flavKin.el_eta" )
    rdf = rdf.Define("jetKin", "getJetKin(JetPtColl, JetEtaColl)")
    rdf = rdf.Define("jet1_pt", "jetKin.pt1").Define("jet1_eta", "jetKin.eta1")
    rdf = rdf.Define("jet2_pt", "jetKin.pt2").Define("jet2_eta", "jetKin.eta2")
    rdf = rdf.Define("jet3_pt", "jetKin.pt3").Define("jet3_eta", "jetKin.eta3")
    rdf = rdf.Define("nJets_f", "(float)nJets")
    return rdf


def define_genuine_bjets(rdf):
    """Define genuine multi-b-jet kinematic columns."""
    rdf = rdf.Define("mbjetKin",
                      "getGenuineMultiBjetKin(JetPtColl, JetEtaColl, "
                      "JetIsBtaggedColl)")
    rdf = rdf.Define("bjet1_pt", "mbjetKin.bjet1_pt")
    rdf = rdf.Define("bjet1_eta", "mbjetKin.bjet1_eta")
    rdf = rdf.Define("bjet2_pt", "mbjetKin.bjet2_pt")
    rdf = rdf.Define("bjet2_eta", "mbjetKin.bjet2_eta")
    rdf = rdf.Define("n_bjets_f", "(float)mbjetKin.n_bjets")
    return rdf


def book_histos(rdf, prefix, channel, weight_col="evt_weight"):
    """Book 1D kinematic histograms (lazy RDF actions).

    bjet2 histograms are booked on a filtered RDF (bjet2_pt > 0) to avoid
    underflow from events without a second b-jet, which would distort
    normalization (plotter normalizes including overflow/underflow).
    """
    histos = {}
    if channel == "Run1E2Mu":
        lep_specs = [
            ("el_pt",   50, 0, 200),  ("el_eta",   50, -2.5, 2.5),
            ("mu1_pt",  50, 0, 200),  ("mu1_eta",  50, -2.5, 2.5),
            ("mu2_pt",  50, 0, 150),  ("mu2_eta",  50, -2.5, 2.5),
        ]
    else:  # Run3Mu
        lep_specs = [
            ("mu1_pt",  50, 0, 200),  ("mu1_eta",  50, -2.5, 2.5),
            ("mu2_pt",  50, 0, 150),  ("mu2_eta",  50, -2.5, 2.5),
            ("mu3_pt",  50, 0, 100),  ("mu3_eta",  50, -2.5, 2.5),
        ]
    jet_specs = [
        ("jet1_pt",   50, 0, 200),  ("jet1_eta",  50, -2.5, 2.5),
        ("jet2_pt",   50, 0, 150),  ("jet2_eta",  50, -2.5, 2.5),
        ("jet3_pt",   50, 0, 100),  ("jet3_eta",  50, -2.5, 2.5),
        ("bjet1_pt",  50, 0, 200),  ("bjet1_eta", 50, -2.5, 2.5),
        ("nJets_f",   10, 0, 10),   ("n_bjets_f",  5, 0, 5),
    ]
    for var, nbins, xlo, xhi in lep_specs + jet_specs:
        hname = f"h_{prefix}_{var}"
        histos[var] = rdf.Histo1D((hname, "", nbins, xlo, xhi),
                                   var, weight_col)
    # Book bjet2 on filtered RDF to exclude events without a second b-jet
    rdf_2b = rdf.Filter("bjet2_pt > 0")
    for var, nbins, xlo, xhi in [("bjet2_pt", 50, 0, 200),
                                   ("bjet2_eta", 50, -2.5, 2.5)]:
        hname = f"h_{prefix}_{var}"
        histos[var] = rdf_2b.Histo1D((hname, "", nbins, xlo, xhi),
                                      var, weight_col)
    return histos


def evaluate_histos(booked):
    """Force RDF evaluation. Returns {var: TH1D} with SetDirectory(0)."""
    result = {}
    for var, h_ptr in booked.items():
        h = h_ptr.GetValue().Clone()
        h.SetDirectory(0)
        result[var] = h
    return result


def sum_histos(h1, h2):
    """Sum two histogram dicts {var: TH1D}."""
    result = {}
    for var in h1:
        h = h1[var].Clone(h1[var].GetName() + "_sum")
        h.Add(h2[var])
        h.SetDirectory(0)
        result[var] = h
    return result


# ---------------------------------------------------------------------------
# Section 5: Plotting
# ---------------------------------------------------------------------------
CHANNEL_LABEL = {"Run1E2Mu": "SR1E2Mu", "Run3Mu": "SR3Mu"}

def make_comparison_plots(genuine, lnt_raw, lnt_ptcorr, lnt_fr, channel,
                          plot_dir, extra_config=None):
    """Produce 4-way comparison plots."""
    from plotter import KinematicCanvasWithRatio

    os.makedirs(plot_dir, exist_ok=True)
    plot_configs = PLOT_CONFIGS_BY_CHANNEL[channel]

    for var, extra_cfg in plot_configs:
        if var not in genuine or var not in lnt_raw:
            continue

        hists = OrderedDict()
        hists["Genuine (Tight+Bjet)"] = genuine[var].Clone(f"p_{var}_g")
        hists["LNT (raw p_{T})"] = lnt_raw[var].Clone(f"p_{var}_raw")
        hists["LNT (p_{T}^{corr})"] = lnt_ptcorr[var].Clone(f"p_{var}_ptc")
        hists["LNT (p_{T}^{corr} + FR wt)"] = lnt_fr[var].Clone(f"p_{var}_fr")
        for h in hists.values():
            h.SetDirectory(0)

        is_pt_var = var.endswith("_pt")
        config = {
            "era": "Run2",
            "channel": CHANNEL_LABEL.get(channel, channel),
            "channelPosX": 0.18,
            "channelPosY": 0.82,
            "iPos": 0,
            "yTitle": "Normalized",
            "rTitle": "LNT/Genuine",
            "rRange": [0.0, 2.0],
            "normalize": True,
            "overflow": is_pt_var,
            "legend": [0.50, 0.60, 0.92, 0.88],
        }
        config.update(extra_cfg)
        if extra_config:
            config.update(extra_config)

        c = KinematicCanvasWithRatio(hists, config)
        import cmsstyle as CMS
        hdf_r = CMS.GetCmsCanvasHist(c.canv.cd(2))
        hdf_r.GetYaxis().SetTitleSize(0.1)
        hdf_r.GetYaxis().SetTitleOffset(0.6)
        hdf_r.GetYaxis().CenterTitle()
        c.drawPadUp()
        c.drawPadDown()
        c.canv.SaveAs(os.path.join(plot_dir, f"{var}.png"))

    print(f"  Saved {len(plot_configs)} plots to {plot_dir}")


# ---------------------------------------------------------------------------
# Section 6: Main Function
# ---------------------------------------------------------------------------
def save_histos_to_root(histos_dict, filepath):
    """Save {var: TH1D} dict to a ROOT file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    f = ROOT.TFile.Open(filepath, "RECREATE")
    for var, h in histos_dict.items():
        h.Write(var)
    f.Close()


def load_histos_from_root(filepath):
    """Load all TH1D from a ROOT file into {name: TH1D} dict."""
    f = ROOT.TFile.Open(filepath)
    result = {}
    for key in f.GetListOfKeys():
        h = key.ReadObj()
        h.SetDirectory(0)
        result[key.GetName()] = h
    f.Close()
    return result


HISTO_DIR = os.path.join(PLOTDIR, "histograms")


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Nonprompt (TTLL) LNT promotion validation")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip computation, re-plot from saved histograms")
    args = parser.parse_args()

    if args.plot_only:
        # =============================================================
        # Plot-only mode: load saved histograms and re-plot
        # =============================================================
        print("=" * 60)
        print("Plot-only mode: loading saved histograms")
        print("=" * 60)

        ERA_GROUPS = [("Run2", RUN2_ERAS), ("Run3", RUN3_ERAS)]
        for suffix, _ in ERA_GROUPS:
            for channel in CHANNELS:
                root_path = os.path.join(HISTO_DIR,
                                         f"{suffix}_{channel}.root")
                if not os.path.exists(root_path):
                    print(f"  SKIP {suffix}/{channel}: {root_path} not found")
                    continue
                all_h = load_histos_from_root(root_path)
                g, raw, ptcorr, fr = {}, {}, {}, {}
                for name, h in all_h.items():
                    prefix, var = name.split("/", 1)
                    if prefix == "genuine":
                        g[var] = h
                    elif prefix == "lnt_raw":
                        raw[var] = h
                    elif prefix == "lnt_ptcorr":
                        ptcorr[var] = h
                    elif prefix == "lnt_fr":
                        fr[var] = h

                out_dir = os.path.join(PLOTDIR, suffix, channel)
                make_comparison_plots(g, raw, ptcorr, fr, channel=channel,
                                      plot_dir=out_dir,
                                      extra_config={"era": suffix})

        print("\n" + "=" * 60)
        print("Done! Check plots in:")
        print(f"  {PLOTDIR}")
        print("=" * 60)
        return

    # =================================================================
    # Full mode
    # =================================================================
    print("=" * 60)
    print("Nonprompt (TTLL) LNT promotion validation")
    print("  ptCorr + fake rate weight from correctionlib")
    print("=" * 60)

    # =================================================================
    # Phase 1: Per-era processing
    # =================================================================
    print("\nLoading fake rate tables from correctionlib...")
    fr_tables = load_fakerate_tables()
    print(f"  Loaded fake rates for {len(fr_tables)} eras")

    # Print sample fake rates for sanity
    for era in ["2017", "2022"]:
        mu = fr_tables[era]["muon"]
        el = fr_tables[era]["electron"]
        print(f"\n  {era} muon FR (eta0,pt0)={mu[0][0]:.4f} "
              f"(eta0,pt6)={mu[0][6]:.4f}")
        print(f"  {era} elec FR (eta0,pt0)={el[0][0]:.4f} "
              f"(eta0,pt5)={el[0][5]:.4f}")

    # Register C++ functions for each era
    print("\nRegistering per-era C++ fake rate functions...")
    for era in ALL_ERAS:
        is_run3 = era in RUN3_ERAS
        register_fakerate_cpp(era, fr_tables[era]["muon"],
                              fr_tables[era]["electron"], is_run3)
        print(f"  Registered getFakeRateWeight_{era}()")

    # Accumulate histograms per (era_group, channel)
    slice_histos = {}  # (era_group, channel) -> (g, raw, ptcorr, fr)
    all_chains = []    # keep alive

    summary_data = {}

    for era in ALL_ERAS:
        is_run3 = era in RUN3_ERAS
        era_group = "Run3" if is_run3 else "Run2"
        s = era  # C++ suffix

        print(f"\n{'='*60}")
        print(f"  Processing {era} ({'Run3' if is_run3 else 'Run2'})")
        print(f"{'='*60}")

        for channel in CHANNELS:
            files = collect_files([era], [channel])
            if not files:
                print(f"  {channel}: no files found, skipping")
                continue

            rdf, chain = make_rdf(files)
            all_chains.append(chain)
            rdf = rdf.Define("evt_weight",
                             "genWeight * puWeight * prefireWeight")

            # --- Genuine: tight + bjet ---
            rdf_tight = apply_tight(rdf).Filter("anyTrue(JetIsBtaggedColl)")
            rdf_g = define_genuine_bjets(define_columns_raw(rdf_tight))

            # --- LNT + bjet ---
            rdf_lnt = apply_lnt_bjet(rdf)

            # LNT with raw pT
            rdf_lnt_raw = define_genuine_bjets(define_columns_raw(rdf_lnt))

            # LNT with ptCorr
            rdf_lnt_ptcorr = define_genuine_bjets(
                define_columns_ptcorr(rdf_lnt, is_run3))

            # LNT with ptCorr + FR weight
            rdf_lnt_fr = rdf_lnt_ptcorr.Define(
                "fr_weight",
                f"evt_weight * getFakeRateWeight_{s}("
                f"MuonPtColl, MuonEtaColl, MuonMiniIsoColl, MuonIsTightColl, "
                f"ElectronPtColl, ElectronScEtaColl, ElectronMiniIsoColl, "
                f"ElectronIsTightColl)")

            # Count events
            n_g_ptr = rdf_g.Count()
            n_lnt_ptr = rdf_lnt.Count()

            # Book histograms
            g_booked = book_histos(rdf_g, f"g_{era}_{channel}", channel)
            raw_booked = book_histos(rdf_lnt_raw, f"raw_{era}_{channel}",
                                      channel)
            ptcorr_booked = book_histos(rdf_lnt_ptcorr,
                                         f"ptcorr_{era}_{channel}", channel)
            fr_booked = book_histos(rdf_lnt_fr, f"fr_{era}_{channel}",
                                     channel, weight_col="fr_weight")

            # Evaluate all
            g_h = evaluate_histos(g_booked)
            raw_h = evaluate_histos(raw_booked)
            ptcorr_h = evaluate_histos(ptcorr_booked)
            fr_h = evaluate_histos(fr_booked)

            n_g = n_g_ptr.GetValue()
            n_lnt = n_lnt_ptr.GetValue()

            print(f"  {channel}: Genuine={n_g}, LNT+Bjet={n_lnt}")

            summary_data[f"{era}_{channel}"] = {
                "n_genuine": int(n_g),
                "n_lnt_bjet": int(n_lnt),
            }

            # Accumulate into (era_group, channel) slice
            key = (era_group, channel)
            if key not in slice_histos:
                slice_histos[key] = (g_h, raw_h, ptcorr_h, fr_h)
            else:
                prev = slice_histos[key]
                slice_histos[key] = (
                    sum_histos(prev[0], g_h),
                    sum_histos(prev[1], raw_h),
                    sum_histos(prev[2], ptcorr_h),
                    sum_histos(prev[3], fr_h),
                )

    # =================================================================
    # Phase 2: (era_group, channel) breakdown plots + save histograms
    # =================================================================
    print("\n--- (era_group, channel) breakdown ---")
    for (era_group, channel), (g, raw, ptcorr, fr) in slice_histos.items():
        # Save histograms for --plot-only reuse
        root_path = os.path.join(HISTO_DIR, f"{era_group}_{channel}.root")
        prefixed = {}
        for var, h in g.items():
            prefixed[f"genuine/{var}"] = h
        for var, h in raw.items():
            prefixed[f"lnt_raw/{var}"] = h
        for var, h in ptcorr.items():
            prefixed[f"lnt_ptcorr/{var}"] = h
        for var, h in fr.items():
            prefixed[f"lnt_fr/{var}"] = h
        save_histos_to_root(prefixed, root_path)

        out_dir = os.path.join(PLOTDIR, era_group, channel)
        make_comparison_plots(g, raw, ptcorr, fr, channel=channel,
                              plot_dir=out_dir,
                              extra_config={"era": era_group})

    del all_chains

    # =================================================================
    # Phase 3: Summary JSON
    # =================================================================
    os.makedirs(PLOTDIR, exist_ok=True)

    # Add aggregate stats
    total_genuine = sum(v["n_genuine"]
                        for v in summary_data.values())
    total_lnt = sum(v["n_lnt_bjet"]
                    for v in summary_data.values())
    summary_data["total"] = {
        "n_genuine": total_genuine,
        "n_lnt_bjet": total_lnt,
        "augmentation_ratio": (total_lnt / total_genuine
                               if total_genuine > 0 else 0),
    }

    # Add fake rate ranges
    fr_summary = {}
    for era in ALL_ERAS:
        mu = fr_tables[era]["muon"]
        el = fr_tables[era]["electron"]
        mu_flat = [v for row in mu for v in row]
        el_flat = [v for row in el for v in row]
        fr_summary[era] = {
            "muon_fr_range": [min(mu_flat), max(mu_flat)],
            "electron_fr_range": [min(el_flat), max(el_flat)],
        }
    summary_data["fake_rates"] = fr_summary

    json_path = os.path.join(PLOTDIR, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary_data, f, indent=2)
    print(f"\nSummary saved to {json_path}")

    print("\n" + "=" * 60)
    print("Done! Check plots in:")
    print(f"  {PLOTDIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
