#!/usr/bin/env python
"""Conditional rank-based b-jet promotion for diboson augmentation.

Two-step conditional decomposition:
  Step 1: Sample n_bjets from P(n_bjets | nJets) learned from genuine events
  Step 2a (n_bjets=1): Sample rank from P(rank | n_bjets=1, nJets)
  Step 2b (n_bjets>=2): Sample (r1,r2) pair from P(pair | n_bjets>=2, nJets)

nJets shape correction via per-event weight.
Inclusive b-jet (pT, eta) calibration via normalized 2D shape ratio
(computed on top of nJets-reweighted promoted), applied jet-by-jet.

Output:
  DataAugment/diboson/plots/rank_promote/*.png
  DataAugment/diboson/plots/rank_promote/conditional_tables.json
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib'))
import json
import array
import ROOT
from collections import OrderedDict

ROOT.gROOT.SetBatch(True)

# ---------------------------------------------------------------------------
# Conditional promotion constants
# ---------------------------------------------------------------------------
NJ_MAX_RUN2 = 5  # nJets groups 1..5 (5 means >=5)
NJ_MAX_RUN3 = 3  # nJets groups 1..3 (3 means >=3)
NJ_MAX = {"Run2": NJ_MAX_RUN2, "Run3": NJ_MAX_RUN3}
RANK_MAX = 10    # ranks 0..9
PAIR_BINS = 100  # pair codes 0..99 (encoding: r1*10 + r2, r1 < r2)

# b-jet (pT, eta) calibration binning (same as MC b-tag efficiency measurement)
ETA_BINS = [0.0, 0.8, 1.6, 2.1, 2.5]
PT_BINS = [20., 30., 50., 70., 100., 140., 200., 300., 10000.]
N_ETA = len(ETA_BINS) - 1  # 4
N_PT = len(PT_BINS) - 1    # 9

# ---------------------------------------------------------------------------
# Paths and samples
# ---------------------------------------------------------------------------
WORKDIR = os.environ.get("WORKDIR")
if not WORKDIR:
    raise RuntimeError("WORKDIR not set. Run 'source setup.sh' first.")

BASEDIR = os.path.join(WORKDIR, "SKNanoOutput", "EvtTreeProducer")
PLOTDIR = os.path.join(WORKDIR, "ParticleNetMD", "DataAugment", "diboson",
                       "plots", "rank_promote")

CHANNELS = ["Run1E2Mu", "Run3Mu"]
RUN2_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018"]
RUN3_ERAS = ["2022", "2022EE", "2023", "2023BPix"]
ALL_ERAS = RUN2_ERAS + RUN3_ERAS

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
# C++ helpers — first block (structs + kinematic extractors)
# ---------------------------------------------------------------------------
ROOT.gInterpreter.Declare("""
#include "TRandom3.h"

const int gNjMax = 7;
const int gRankMax = 10;
const int gPairBins = 100;

// b-jet calibration bin edges (same as MC b-tag efficiency measurement)
const int gNEta = 4;
const int gNPt = 8;
const double gEtaEdges[5] = {0.0, 0.8, 1.6, 2.1, 2.5};
const double gPtEdges[9] = {20., 30., 50., 70., 100., 140., 200., 300., 10000.};

int findBin(double val, const double* edges, int nbins) {
    for (int i = 0; i < nbins; i++) {
        if (val < edges[i+1]) return i;
    }
    return nbins - 1;
}

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
    std::sort(idx.begin(), idx.end(), [&](int a, int b){ return pt[a] > pt[b]; });
    return idx;
}

// --- Lepton kinematics (3 leading, pt + eta) ---
struct LepKin { float pt1, eta1, pt2, eta2, pt3, eta3; };
LepKin getLeptonKin(const ROOT::VecOps::RVec<float>& muPt, const ROOT::VecOps::RVec<float>& muEta,
                    const ROOT::VecOps::RVec<float>& elPt, const ROOT::VecOps::RVec<float>& elEta) {
    std::vector<std::pair<float,float>> leps;
    for (int i = 0; i < (int)muPt.size(); i++) leps.push_back({muPt[i], muEta[i]});
    for (int i = 0; i < (int)elPt.size(); i++) leps.push_back({elPt[i], elEta[i]});
    std::sort(leps.begin(), leps.end(), [](auto& a, auto& b){ return a.first > b.first; });
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

// --- Jet kinematics (3 leading, pt + eta) ---
struct JetKin { float pt1, eta1, pt2, eta2, pt3, eta3; };
JetKin getJetKin(const ROOT::VecOps::RVec<float>& pt, const ROOT::VecOps::RVec<float>& eta) {
    auto idx = ptSortedIdx(pt);
    JetKin k{-1,-99, -1,-99, -1,-99};
    if (idx.size() >= 1) { k.pt1 = pt[idx[0]]; k.eta1 = eta[idx[0]]; }
    if (idx.size() >= 2) { k.pt2 = pt[idx[1]]; k.eta2 = eta[idx[1]]; }
    if (idx.size() >= 3) { k.pt3 = pt[idx[2]]; k.eta3 = eta[idx[2]]; }
    return k;
}

// --- Multi b-jet kinematics (all genuine b-tagged jets, sorted by pT) ---
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

// --- Helpers for conditional promotion ---
int getNBjetsFromBtag(const ROOT::VecOps::RVec<bool>& btag) {
    int n = 0;
    for (auto b : btag) if (b) n++;
    return n;
}

// For n_bjets==1 events: return the pT rank of the single b-jet (capped at 9)
int getSingleBjetRank(const ROOT::VecOps::RVec<float>& pt,
                      const ROOT::VecOps::RVec<bool>& btag) {
    auto idx = ptSortedIdx(pt);
    for (int r = 0; r < (int)idx.size(); r++) {
        if (btag[idx[r]]) return std::min(r, 9);
    }
    return -1;
}

// For n_bjets>=2 events: return pair code of top 2 b-jet ranks
// pair_code = r1 * 10 + r2 where r1 < r2 (both capped at 9)
int getDoubleBjetPairCode(const ROOT::VecOps::RVec<float>& pt,
                           const ROOT::VecOps::RVec<bool>& btag) {
    auto idx = ptSortedIdx(pt);
    int first = -1, second = -1;
    for (int r = 0; r < (int)idx.size(); r++) {
        if (btag[idx[r]]) {
            if (first < 0) first = std::min(r, 9);
            else { second = std::min(r, 9); break; }
        }
    }
    if (first >= 0 && second >= 0 && first < second) {
        return first * 10 + second;
    }
    return -1;
}
""")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def collect_files(eras=None, channels=None):
    """Collect diboson ROOT file paths for given eras and channels."""
    if eras is None:
        eras = ALL_ERAS
    if channels is None:
        channels = CHANNELS
    files = []
    for channel in channels:
        for era in eras:
            for sample in [WZ_SAMPLES[era], ZZ_SAMPLE]:
                fpath = os.path.join(BASEDIR, channel, era, f"{sample}.root")
                if os.path.exists(fpath):
                    files.append(fpath)
    return files


def make_rdf(files):
    """Create RDataFrame from file list. Returns (rdf, chain).

    Caller must keep `chain` alive for the lifetime of the RDataFrame.
    """
    chain = ROOT.TChain("Events")
    for f in files:
        chain.Add(f)
    return ROOT.RDataFrame(chain), chain


def apply_tight(rdf):
    """Apply tight lepton selection."""
    return rdf.Filter("allTrue(MuonIsTightColl) && allTrue(ElectronIsTightColl)")


def define_columns(rdf):
    """Define lepton, jet, and multiplicity columns."""
    rdf = rdf.Define("flavKin", "getFlavorLeptonKin(MuonPtColl, MuonEtaColl, "
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


def define_genuine_bjets(rdf):
    """Define genuine multi-b-jet kinematic columns."""
    rdf = rdf.Define("mbjetKin",
                      "getGenuineMultiBjetKin(JetPtColl, JetEtaColl, JetIsBtaggedColl)")
    rdf = rdf.Define("bjet1_pt", "mbjetKin.bjet1_pt")
    rdf = rdf.Define("bjet1_eta", "mbjetKin.bjet1_eta")
    rdf = rdf.Define("bjet2_pt", "mbjetKin.bjet2_pt")
    rdf = rdf.Define("bjet2_eta", "mbjetKin.bjet2_eta")
    rdf = rdf.Define("n_bjets_f", "(float)mbjetKin.n_bjets")
    rdf = rdf.Define("abs_bjet1_eta", "std::abs(bjet1_eta)")
    rdf = rdf.Define("abs_bjet2_eta", "std::abs(bjet2_eta)")
    return rdf


def define_rank_promoted_bjets(rdf, suffix):
    """Apply conditional rank promotion and define b-jet columns."""
    rdf = rdf.Define("rpResult",
                      f"conditionalPromote{suffix}(JetPtColl, JetEtaColl, rdfentry_)")
    rdf = rdf.Filter("rpResult.n_bjets > 0")
    rdf = rdf.Define("bjet1_pt", "rpResult.bjet1_pt")
    rdf = rdf.Define("bjet1_eta", "rpResult.bjet1_eta")
    rdf = rdf.Define("bjet2_pt", "rpResult.bjet2_pt")
    rdf = rdf.Define("bjet2_eta", "rpResult.bjet2_eta")
    rdf = rdf.Define("n_bjets_f", "(float)rpResult.n_bjets")
    rdf = rdf.Define("abs_bjet1_eta", "std::abs(bjet1_eta)")
    rdf = rdf.Define("abs_bjet2_eta", "std::abs(bjet2_eta)")
    return rdf


# ---------------------------------------------------------------------------
# Conditional probability tables
# ---------------------------------------------------------------------------
def compute_conditional_tables(rdf_genuine, nj_max):
    """Compute conditional probability tables from genuine events.

    Tables:
      P(n_bjets=1 | nJets group g)
      P(rank | n_bjets=1, nJets group g)   — CDF for single b-jet rank
      P(pair | n_bjets>=2, nJets group g)  — CDF for rank pair code

    Returns dict with keys: p_1b, cdf_rank, cdf_pair, rank_probs, pair_probs.
    """
    rdf = rdf_genuine
    rdf = rdf.Define("n_bjets_g", "getNBjetsFromBtag(JetIsBtaggedColl)")
    rdf = rdf.Define("nj_group", f"std::min((int)nJets, {nj_max})")
    rdf = rdf.Define("nj_group_f", "(float)nj_group")

    # Split by n_bjets multiplicity
    rdf_1b = rdf.Filter("n_bjets_g == 1")
    rdf_2bp = rdf.Filter("n_bjets_g >= 2")

    # Define rank / pair code columns
    rdf_1b = rdf_1b.Define("single_rank_f",
        "(float)getSingleBjetRank(JetPtColl, JetIsBtaggedColl)")
    rdf_2bp = rdf_2bp.Define("pair_code_i",
        "getDoubleBjetPairCode(JetPtColl, JetIsBtaggedColl)")
    rdf_2bp = rdf_2bp.Filter("pair_code_i >= 0")
    rdf_2bp = rdf_2bp.Define("pair_code_f", "(float)pair_code_i")

    # Book histograms (all lazy)
    h_nb1_ptr = rdf_1b.Histo1D(
        ("h_nb1", "", nj_max, 0.5, nj_max + 0.5),
        "nj_group_f", "evt_weight")
    h_nb2p_ptr = rdf_2bp.Histo1D(
        ("h_nb2p", "", nj_max, 0.5, nj_max + 0.5),
        "nj_group_f", "evt_weight")
    h_rank_ptr = rdf_1b.Histo2D(
        ("h_rank", "", nj_max, 0.5, nj_max + 0.5,
         RANK_MAX, -0.5, RANK_MAX - 0.5),
        "nj_group_f", "single_rank_f", "evt_weight")
    h_pair_ptr = rdf_2bp.Histo2D(
        ("h_pair", "", nj_max, 0.5, nj_max + 0.5,
         PAIR_BINS, -0.5, PAIR_BINS - 0.5),
        "nj_group_f", "pair_code_f", "evt_weight")

    # Evaluate all lazy actions
    h_nb1 = h_nb1_ptr.GetValue()
    h_nb2p = h_nb2p_ptr.GetValue()
    h_rank = h_rank_ptr.GetValue()
    h_pair = h_pair_ptr.GetValue()

    # --- Build probability tables ---

    # P(n_bjets=1 | nJets group g), index 0 unused
    p_1b = [0.0] * (nj_max + 1)
    for g in range(1, nj_max + 1):
        v1 = h_nb1.GetBinContent(g)
        v2p = h_nb2p.GetBinContent(g)
        total = v1 + v2p
        p_1b[g] = v1 / total if total > 0 else 1.0

    # CDF for rank given n_bjets=1 and nJets group g
    cdf_rank = [[0.0] * RANK_MAX for _ in range(nj_max + 1)]
    rank_probs = {}
    for g in range(1, nj_max + 1):
        raw = []
        total = 0
        for r in range(RANK_MAX):
            val = h_rank.GetBinContent(g, r + 1)
            raw.append(val)
            total += val
        prob_norm = [v / total if total > 0 else 0 for v in raw]
        cumul = 0
        for r in range(RANK_MAX):
            cumul += prob_norm[r]
            cdf_rank[g][r] = cumul
        cdf_rank[g][RANK_MAX - 1] = 1.0
        rank_probs[g] = prob_norm
    cdf_rank[0] = [1.0] * RANK_MAX  # dummy for g=0

    # CDF for pair code given n_bjets>=2 and nJets group g
    cdf_pair = [[0.0] * PAIR_BINS for _ in range(nj_max + 1)]
    pair_probs = {}
    for g in range(1, nj_max + 1):
        raw = []
        total = 0
        for pc in range(PAIR_BINS):
            val = h_pair.GetBinContent(g, pc + 1)
            raw.append(val)
            total += val
        probs_dict = {}
        cumul = 0
        for pc in range(PAIR_BINS):
            p = raw[pc] / total if total > 0 else 0
            if p > 0:
                probs_dict[pc] = p
            cumul += p
            cdf_pair[g][pc] = cumul
        cdf_pair[g][PAIR_BINS - 1] = 1.0
        pair_probs[g] = probs_dict
    cdf_pair[0] = [1.0] * PAIR_BINS  # dummy for g=0

    # Print summary
    print("  Conditional probability tables:")
    for g in range(1, nj_max + 1):
        label = f"nJets={g}" if g < nj_max else f"nJets>={g}"
        print(f"\n  {label}:")
        print(f"    P(1b) = {p_1b[g]:.4f}, P(2b+) = {1 - p_1b[g]:.4f}")
        top_ranks = [(r, p) for r, p in enumerate(rank_probs[g]) if p > 0.01]
        if top_ranks:
            rank_str = ", ".join(f"r{r}:{p:.3f}" for r, p in top_ranks)
            print(f"    1b ranks: {rank_str}")
        top_pairs = sorted(pair_probs[g].items(), key=lambda x: -x[1])[:5]
        if top_pairs:
            pair_str = ", ".join(
                f"({pc // 10},{pc % 10}):{p:.3f}" for pc, p in top_pairs)
            print(f"    2b+ pairs: {pair_str}")

    return {
        'p_1b': p_1b,
        'cdf_rank': cdf_rank,
        'cdf_pair': cdf_pair,
        'rank_probs': rank_probs,
        'pair_probs': pair_probs,
    }


# ---------------------------------------------------------------------------
# Register conditional promotion C++ function
# ---------------------------------------------------------------------------
def register_conditional_promote(tables, suffix, nj_max):
    """Inject conditional probability CDFs and register conditionalPromote{suffix}()."""
    nj = nj_max + 1  # indices 0..nj_max
    s = suffix  # shorthand for C++ variable names

    # Format p_1b array
    p_1b_str = ", ".join(f"{tables['p_1b'][g]:.10f}" for g in range(nj))

    # Format cdf_rank 2D array
    cdf_rank_rows = []
    for g in range(nj):
        row = ", ".join(f"{tables['cdf_rank'][g][r]:.10f}"
                        for r in range(RANK_MAX))
        cdf_rank_rows.append("{" + row + "}")
    cdf_rank_str = ",\n        ".join(cdf_rank_rows)

    # Format cdf_pair 2D array
    cdf_pair_rows = []
    for g in range(nj):
        row = ", ".join(f"{tables['cdf_pair'][g][pc]:.10f}"
                        for pc in range(PAIR_BINS))
        cdf_pair_rows.append("{" + row + "}")
    cdf_pair_str = ",\n        ".join(cdf_pair_rows)

    ROOT.gInterpreter.Declare(f"""
    const double gP1b_{s}[{nj}] = {{{p_1b_str}}};
    const double gCdfRank_{s}[{nj}][{RANK_MAX}] = {{
        {cdf_rank_str}
    }};
    const double gCdfPair_{s}[{nj}][{PAIR_BINS}] = {{
        {cdf_pair_str}
    }};

    MultiBjetKin conditionalPromote{s}(const ROOT::VecOps::RVec<float>& pt,
                                        const ROOT::VecOps::RVec<float>& eta,
                                        ULong64_t entry) {{
        auto idx = ptSortedIdx(pt);
        int nJ = (int)idx.size();
        MultiBjetKin result = {{0, -1, -99, -1, -99}};
        if (nJ <= 0) return result;

        int g = std::min(nJ, {nj_max});
        TRandom3 rng((UInt_t)(entry * 2654435761u + 12345));

        if (nJ == 1) {{
            result.n_bjets = 1;
            result.bjet1_pt = pt[idx[0]];
            result.bjet1_eta = eta[idx[0]];
            return result;
        }}

        double u1 = rng.Rndm();
        bool is_1b = (u1 < gP1b_{s}[g]);

        if (is_1b) {{
            double u2 = rng.Rndm();
            int rank = 0;
            for (int r = 0; r < gRankMax; r++) {{
                if (u2 < gCdfRank_{s}[g][r]) {{
                    rank = r;
                    break;
                }}
            }}
            if (rank >= nJ) rank = nJ - 1;
            int j = idx[rank];
            result.n_bjets = 1;
            result.bjet1_pt = pt[j];
            result.bjet1_eta = eta[j];
        }} else {{
            double u2 = rng.Rndm();
            int pair_code = 1;
            for (int pc = 0; pc < gPairBins; pc++) {{
                if (u2 < gCdfPair_{s}[g][pc]) {{
                    pair_code = pc;
                    break;
                }}
            }}
            int r1 = pair_code / 10;
            int r2 = pair_code % 10;
            if (r1 >= nJ) r1 = 0;
            if (r2 >= nJ) r2 = nJ - 1;
            if (r1 >= r2) {{ r1 = 0; r2 = 1; }}

            int j1 = idx[r1];
            int j2 = idx[r2];
            result.n_bjets = 2;
            result.bjet1_pt = pt[j1];
            result.bjet1_eta = eta[j1];
            result.bjet2_pt = pt[j2];
            result.bjet2_eta = eta[j2];
        }}
        return result;
    }}
    """)


# ---------------------------------------------------------------------------
# Histogram booking and evaluation
# ---------------------------------------------------------------------------
def book_histos(rdf, prefix, channel, weight_col="evt_weight"):
    """Book 1D kinematic histograms (lazy RDF actions)."""
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
        ("bjet2_pt",  50, 0, 200),  ("bjet2_eta", 50, -2.5, 2.5),
        ("nJets_f",   10, 0, 10),   ("n_bjets_f",  5, 0, 5),
    ]
    for var, nbins, xlo, xhi in lep_specs + jet_specs:
        hname = f"h_{prefix}_{var}"
        histos[var] = rdf.Histo1D((hname, "", nbins, xlo, xhi), var, weight_col)
    return histos


def evaluate_histos(booked):
    """Force RDF evaluation. Returns {var: TH1D} with SetDirectory(0)."""
    result = {}
    for var, h_ptr in booked.items():
        h = h_ptr.GetValue().Clone()
        h.SetDirectory(0)
        result[var] = h
    return result


def book_bjet_pteta_2d(rdf, prefix, weight_col="evt_weight"):
    """Book 2D (|eta|, pT) histograms for inclusive b-jets.

    Returns (h1_ptr, h2_ptr) for bjet1 and bjet2 respectively.
    """
    eta_arr = array.array('d', ETA_BINS)
    pt_arr = array.array('d', PT_BINS)

    h1 = rdf.Filter("bjet1_pt > 0").Histo2D(
        ROOT.RDF.TH2DModel(f"h2d_{prefix}_b1", "",
                            N_ETA, eta_arr, N_PT, pt_arr),
        "abs_bjet1_eta", "bjet1_pt", weight_col)

    h2 = rdf.Filter("bjet2_pt > 0").Histo2D(
        ROOT.RDF.TH2DModel(f"h2d_{prefix}_b2", "",
                            N_ETA, eta_arr, N_PT, pt_arr),
        "abs_bjet2_eta", "bjet2_pt", weight_col)

    return h1, h2


def evaluate_sum_2d(h1_ptr, h2_ptr):
    """Evaluate and sum two lazy 2D histograms into one."""
    h1 = h1_ptr.GetValue().Clone()
    h2 = h2_ptr.GetValue()
    h1.Add(h2)
    h1.SetDirectory(0)
    return h1


# ---------------------------------------------------------------------------
# nJets reweighting
# ---------------------------------------------------------------------------
def compute_njets_weights(g_histos, rp_histos):
    """Compute nJets shape ratio: genuine_norm / promoted_norm.

    Returns dict {nJets_int: weight}.
    """
    h_g = g_histos["nJets_f"].Clone("h_njets_ratio_g")
    h_rp = rp_histos["nJets_f"].Clone("h_njets_ratio_rp")
    h_g.SetDirectory(0)
    h_rp.SetDirectory(0)

    g_int = h_g.Integral()
    rp_int = h_rp.Integral()
    if g_int > 0:
        h_g.Scale(1.0 / g_int)
    if rp_int > 0:
        h_rp.Scale(1.0 / rp_int)

    weights = {}
    print(f"\n  nJets reweighting factors:")
    for nj in range(10):
        bin_idx = h_g.FindBin(float(nj))
        vg = h_g.GetBinContent(bin_idx)
        vrp = h_rp.GetBinContent(bin_idx)
        if vrp > 0 and vg > 0:
            weights[nj] = vg / vrp
        else:
            weights[nj] = 1.0
        if nj >= 1 and nj <= 7:
            print(f"    nJets={nj}: {weights[nj]:.4f}")

    return weights


def register_njets_weight(weights, suffix):
    """Register C++ getNjetsWeight{suffix}() lookup function."""
    s = suffix
    w_str = ", ".join(f"{weights.get(i, 1.0):.8f}" for i in range(10))
    ROOT.gInterpreter.Declare(f"""
    const double gNjetsWeights_{s}[] = {{{w_str}}};

    double getNjetsWeight{s}(int nJets) {{
        if (nJets < 0 || nJets >= 10) return 1.0;
        return gNjetsWeights_{s}[nJets];
    }}
    """)


# ---------------------------------------------------------------------------
# b-jet (pT, eta) calibration
# ---------------------------------------------------------------------------
def compute_bjet_pteta_weights(h2d_genuine, h2d_promoted):
    """Compute (pT, |eta|) calibration weights from ratio of normalized 2D histograms.

    Returns 2D list weights[eta_bin][pt_bin].
    """
    h_g = h2d_genuine.Clone("h2d_pteta_g_norm")
    h_rp = h2d_promoted.Clone("h2d_pteta_rp_norm")
    h_g.SetDirectory(0)
    h_rp.SetDirectory(0)

    # Normalize to unit integral
    g_int = h_g.Integral()
    rp_int = h_rp.Integral()
    if g_int > 0:
        h_g.Scale(1.0 / g_int)
    if rp_int > 0:
        h_rp.Scale(1.0 / rp_int)

    weights = [[1.0] * N_PT for _ in range(N_ETA)]
    print(f"\n  b-jet (pT, |eta|) calibration weights:")
    pt_labels = [f"[{PT_BINS[i]:.0f},{PT_BINS[i+1]:.0f})" for i in range(N_PT)]
    print(f"    {'|eta|':<16} " + " ".join(f"{l:>12}" for l in pt_labels))
    for ie in range(N_ETA):
        for ip in range(N_PT):
            vg = h_g.GetBinContent(ie + 1, ip + 1)
            vrp = h_rp.GetBinContent(ie + 1, ip + 1)
            if vrp > 0 and vg > 0:
                weights[ie][ip] = vg / vrp
            else:
                weights[ie][ip] = 1.0
        eta_label = f"[{ETA_BINS[ie]:.1f},{ETA_BINS[ie+1]:.1f})"
        w_str = " ".join(f"{weights[ie][ip]:>12.3f}" for ip in range(N_PT))
        print(f"    {eta_label:<16} {w_str}")

    return weights


def register_bjet_pteta_weight(weights, suffix):
    """Register C++ getBjetPtEtaWeight{suffix}(MultiBjetKin) function."""
    s = suffix

    # Format 2D weight array
    rows = []
    for ie in range(N_ETA):
        row = ", ".join(f"{weights[ie][ip]:.8f}" for ip in range(N_PT))
        rows.append("{" + row + "}")
    w_str = ",\n        ".join(rows)

    ROOT.gInterpreter.Declare(f"""
    const double gBjetW_{s}[{N_ETA}][{N_PT}] = {{
        {w_str}
    }};

    double getBjetPtEtaWeight{s}(const MultiBjetKin& kin) {{
        double w = 1.0;
        if (kin.n_bjets >= 1 && kin.bjet1_pt > 0) {{
            double pt = std::max(20.0, std::min((double)kin.bjet1_pt, 9999.9));
            double ae = std::min(std::abs((double)kin.bjet1_eta), 2.49);
            w *= gBjetW_{s}[findBin(ae, gEtaEdges, gNEta)][findBin(pt, gPtEdges, gNPt)];
        }}
        if (kin.n_bjets >= 2 && kin.bjet2_pt > 0) {{
            double pt = std::max(20.0, std::min((double)kin.bjet2_pt, 9999.9));
            double ae = std::min(std::abs((double)kin.bjet2_eta), 2.49);
            w *= gBjetW_{s}[findBin(ae, gEtaEdges, gNEta)][findBin(pt, gPtEdges, gNPt)];
        }}
        return w;
    }}
    """)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
CHANNEL_LABEL = {"Run1E2Mu": "SR1E2Mu", "Run3Mu": "SR3Mu"}

def make_comparison_plots(genuine, raw, nj_reweighted, calibrated, channel,
                          plot_dir, extra_config=None):
    """Produce 4-way comparison (genuine / raw / nJets rw / nJets rw + pT,eta cal)."""
    from plotter import KinematicCanvasWithRatio

    os.makedirs(plot_dir, exist_ok=True)
    plot_configs = PLOT_CONFIGS_BY_CHANNEL[channel]

    for var, extra_cfg in plot_configs:
        if var not in genuine or var not in raw:
            continue

        hists = OrderedDict()
        hists["Genuine (Tight+Bjet)"] = genuine[var].Clone(f"p_{var}_g")
        hists["Promoted (raw)"] = raw[var].Clone(f"p_{var}_raw")
        hists["Promoted (nJets rw)"] = nj_reweighted[var].Clone(f"p_{var}_nj")
        hists["Promoted (+ p_{T},#eta cal.)"] = calibrated[var].Clone(f"p_{var}_cal")
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
            "rTitle": "Promoted/Genuine",
            "rRange": [0.0, 2.0],
            "normalize": True,
            "overflow": is_pt_var,
            "legend": [0.55, 0.60, 0.92, 0.88],
        }
        config.update(extra_cfg)
        if extra_config:
            config.update(extra_config)

        c = KinematicCanvasWithRatio(hists, config)
        # Shrink ratio y-axis title for long label
        import cmsstyle as CMS
        hdf_r = CMS.GetCmsCanvasHist(c.canv.cd(2))
        hdf_r.GetYaxis().SetTitleSize(0.1)
        hdf_r.GetYaxis().SetTitleOffset(0.6)
        hdf_r.GetYaxis().CenterTitle()
        c.drawPadUp()
        c.drawPadDown()
        c.canv.SaveAs(os.path.join(plot_dir, f"{var}.png"))

    print(f"  Saved {len(plot_configs)} plots to {plot_dir}")


def sum_histos(h1, h2):
    """Sum two histogram dicts {var: TH1D}."""
    result = {}
    for var in h1:
        h = h1[var].Clone(h1[var].GetName() + "_sum")
        h.Add(h2[var])
        h.SetDirectory(0)
        result[var] = h
    return result


def build_slice_histos(eras, channels, suffix, tag):
    """Build genuine, nJets-reweighted, and calibrated histograms for one slice.

    Requires conditionalPromote{suffix}, getNjetsWeight{suffix}, and
    getBjetPtEtaWeight{suffix} to be registered before calling.
    Returns (g, nj, cal, chain) or None.
    """
    files = collect_files(eras=eras, channels=channels)
    if not files:
        return None
    channel = channels[0]

    rdf, chain = make_rdf(files)
    rdf_tight = apply_tight(rdf)
    rdf_tight = rdf_tight.Define("evt_weight",
                                  "genWeight * puWeight * prefireWeight")

    rdf_g = define_genuine_bjets(define_columns(
        rdf_tight.Filter("anyTrue(JetIsBtaggedColl)")))
    rdf_rp = define_rank_promoted_bjets(define_columns(
        rdf_tight.Filter("!anyTrue(JetIsBtaggedColl) && nJets > 0")), suffix)

    rdf_nj = rdf_rp.Define("nj_weight",
                             f"evt_weight * getNjetsWeight{suffix}(nJets)")
    rdf_cal = rdf_rp.Define(
        "cal_weight",
        f"evt_weight * getNjetsWeight{suffix}(nJets)"
        f" * getBjetPtEtaWeight{suffix}(rpResult)")

    g = evaluate_histos(book_histos(rdf_g, f"g_{tag}", channel))
    rp = evaluate_histos(book_histos(rdf_rp, f"rp_{tag}", channel))
    nj = evaluate_histos(book_histos(rdf_nj, f"nj_{tag}", channel,
                                      weight_col="nj_weight"))
    cal = evaluate_histos(book_histos(rdf_cal, f"cal_{tag}", channel,
                                       weight_col="cal_weight"))
    return g, rp, nj, cal, chain


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def print_nbj_table(g_histos, rp_histos, label=""):
    """Print n_bjets distribution comparison."""
    h_g = g_histos["n_bjets_f"]
    h_rp = rp_histos["n_bjets_f"]
    tot_g = h_g.Integral()
    tot_rp = h_rp.Integral()
    if label:
        print(f"\n  n_bjets distribution ({label}, weighted):")
    else:
        print(f"\n  n_bjets distribution (weighted):")
    print(f"  {'n_bjets':<10} {'Genuine':>20} {'Promoted':>20}")
    for nb in range(1, 5):
        vg = h_g.GetBinContent(h_g.FindBin(float(nb)))
        vrp = h_rp.GetBinContent(h_rp.FindBin(float(nb)))
        pg = vg / tot_g * 100 if tot_g > 0 else 0
        prp = vrp / tot_rp * 100 if tot_rp > 0 else 0
        print(f"  {nb:<10} {vg:>12.0f} ({pg:>5.1f}%)"
              f" {vrp:>12.0f} ({prp:>5.1f}%)")


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
        description="Conditional rank-based b-jet promotion for diboson augmentation")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip computation, re-plot from saved histograms")
    args = parser.parse_args()

    ERA_GROUPS = [("Run2", RUN2_ERAS), ("Run3", RUN3_ERAS)]

    if args.plot_only:
        # =============================================================
        # Plot-only mode: load saved histograms and re-plot
        # =============================================================
        print("=" * 60)
        print("Plot-only mode: loading saved histograms")
        print("=" * 60)

        for suffix, _ in ERA_GROUPS:
            for channel in CHANNELS:
                root_path = os.path.join(HISTO_DIR,
                                         f"{suffix}_{channel}.root")
                if not os.path.exists(root_path):
                    print(f"  SKIP {suffix}/{channel}: {root_path} not found")
                    continue
                all_h = load_histos_from_root(root_path)
                # Split by prefix
                g, rp, nj, cal = {}, {}, {}, {}
                for name, h in all_h.items():
                    prefix, var = name.split("/", 1)
                    if prefix == "genuine":
                        g[var] = h
                    elif prefix == "raw":
                        rp[var] = h
                    elif prefix == "nj_rw":
                        nj[var] = h
                    elif prefix == "calibrated":
                        cal[var] = h

                out_dir = os.path.join(PLOTDIR, suffix, channel)
                make_comparison_plots(g, rp, nj, cal, channel=channel,
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
    print("Conditional rank-based b-jet promotion (Run2/Run3 split)")
    print("  + nJets reweight + (pT,eta) calibration")
    print("=" * 60)

    # =================================================================
    # Phase 1: For each era group, compute tables, register functions,
    #          build histograms, compute calibration weights.
    # =================================================================
    all_tables = {}
    all_njets_w = {}
    all_pteta_w = {}
    era_histos = {}   # suffix -> (g, nj, cal)
    era_chains = []   # keep alive until plots are done

    for suffix, eras in ERA_GROUPS:
        jet_type = "CHS jets" if suffix == "Run2" else "PUPPI jets"
        print(f"\n{'='*60}")
        print(f"  {suffix} ({jet_type})")
        print(f"{'='*60}")

        files = collect_files(eras=eras)
        print(f"  Found {len(files)} ROOT files")

        rdf, chain = make_rdf(files)
        era_chains.append(chain)
        rdf_tight = apply_tight(rdf)
        rdf_tight = rdf_tight.Define("evt_weight",
                                      "genWeight * puWeight * prefireWeight")

        rdf_genuine = rdf_tight.Filter("anyTrue(JetIsBtaggedColl)")
        rdf_0tag = rdf_tight.Filter(
            "!anyTrue(JetIsBtaggedColl) && nJets > 0")

        # --- Step 1: Conditional tables from genuine events ---
        nj_max = NJ_MAX[suffix]
        print(f"\n--- {suffix}: Conditional probability tables (nJets max={nj_max}) ---")
        tables = compute_conditional_tables(rdf_genuine, nj_max)
        register_conditional_promote(tables, suffix, nj_max)
        all_tables[suffix] = tables

        # --- Step 2: Build genuine + raw promoted histograms ---
        print(f"\n--- {suffix}: Building histograms ---")
        rdf_g = define_genuine_bjets(define_columns(rdf_genuine))
        rdf_rp = define_rank_promoted_bjets(define_columns(rdf_0tag), suffix)

        n_g_ptr = rdf_g.Count()
        n_0tag_ptr = rdf_0tag.Count()
        n_rp_ptr = rdf_rp.Count()

        # Book genuine 1D + 2D histos in one round (use Run1E2Mu as dummy channel;
        # only nJets_f and n_bjets_f are used from these histos in Phase 1)
        g_booked = book_histos(rdf_g, f"g_{suffix}", "Run1E2Mu")
        rp_booked = book_histos(rdf_rp, f"rp_{suffix}", "Run1E2Mu")
        g_2d_h1, g_2d_h2 = book_bjet_pteta_2d(rdf_g, f"g2d_{suffix}")

        g_histos = evaluate_histos(g_booked)
        rp_histos = evaluate_histos(rp_booked)
        h2d_g = evaluate_sum_2d(g_2d_h1, g_2d_h2)

        n_g = n_g_ptr.GetValue()
        n_0tag = n_0tag_ptr.GetValue()
        n_rp = n_rp_ptr.GetValue()
        print(f"  Genuine: {n_g},  0-tag: {n_0tag},  Promoted: {n_rp}")
        print_nbj_table(g_histos, rp_histos, suffix)

        # --- Step 3: nJets reweighting ---
        print(f"\n--- {suffix}: nJets reweighting ---")
        njets_w = compute_njets_weights(g_histos, rp_histos)
        register_njets_weight(njets_w, suffix)
        all_njets_w[suffix] = njets_w

        # --- Step 4: (pT, eta) calibration on top of nJets reweight ---
        # Build 2D histo for promoted with nJets weight applied
        print(f"\n--- {suffix}: b-jet (pT, eta) calibration ---")
        rdf_nj = rdf_rp.Define("nj_weight",
                                f"evt_weight * getNjetsWeight{suffix}(nJets)")
        nj_booked = book_histos(rdf_nj, f"nj_{suffix}", "Run1E2Mu",
                                weight_col="nj_weight")
        rp_2d_h1, rp_2d_h2 = book_bjet_pteta_2d(
            rdf_nj, f"rp2d_{suffix}", weight_col="nj_weight")

        nj_histos = evaluate_histos(nj_booked)
        h2d_rp = evaluate_sum_2d(rp_2d_h1, rp_2d_h2)

        pteta_w = compute_bjet_pteta_weights(h2d_g, h2d_rp)
        register_bjet_pteta_weight(pteta_w, suffix)
        all_pteta_w[suffix] = pteta_w

        # --- Step 5: Calibrated histograms (nJets rw + pT,eta cal) ---
        rdf_cal = rdf_rp.Define(
            "cal_weight",
            f"evt_weight * getNjetsWeight{suffix}(nJets)"
            f" * getBjetPtEtaWeight{suffix}(rpResult)")
        cal_histos = evaluate_histos(book_histos(
            rdf_cal, f"cal_{suffix}", "Run1E2Mu", weight_col="cal_weight"))

        era_histos[suffix] = (g_histos, rp_histos, nj_histos, cal_histos)

    del era_chains

    # =================================================================
    # Phase 2: (era_group, channel) breakdown plots + save histograms
    # =================================================================
    print("\n--- (era_group, channel) breakdown ---")
    slice_chains = []
    for suffix, eras in ERA_GROUPS:
        for channel in CHANNELS:
            result = build_slice_histos(eras, [channel], suffix,
                                        f"{channel}_{suffix}")
            if result is None:
                continue
            g, rp, nj, cal, chain_ref = result
            slice_chains.append(chain_ref)

            # Save histograms for --plot-only reuse
            root_path = os.path.join(HISTO_DIR, f"{suffix}_{channel}.root")
            prefixed = {}
            for var, h in g.items():
                prefixed[f"genuine/{var}"] = h
            for var, h in rp.items():
                prefixed[f"raw/{var}"] = h
            for var, h in nj.items():
                prefixed[f"nj_rw/{var}"] = h
            for var, h in cal.items():
                prefixed[f"calibrated/{var}"] = h
            save_histos_to_root(prefixed, root_path)

            out_dir = os.path.join(PLOTDIR, suffix, channel)
            make_comparison_plots(g, rp, nj, cal, channel=channel,
                                  plot_dir=out_dir,
                                  extra_config={"era": suffix})
    del slice_chains

    # =================================================================
    # Save JSON
    # =================================================================
    os.makedirs(PLOTDIR, exist_ok=True)
    output = {}
    for suffix in ("Run2", "Run3"):
        t = all_tables[suffix]
        nj_max = NJ_MAX[suffix]
        output[suffix] = {
            "p_1b": {str(g): t['p_1b'][g]
                     for g in range(1, nj_max + 1)},
            "rank_probs": {str(g): t['rank_probs'][g]
                           for g in range(1, nj_max + 1)},
            "pair_probs": {str(g): {str(pc): p
                                    for pc, p in t['pair_probs'][g].items()}
                           for g in range(1, nj_max + 1)},
            "njets_weights": {str(k): v
                              for k, v in all_njets_w[suffix].items()},
            "bjet_pteta_weights": {
                "eta_bins": ETA_BINS,
                "pt_bins": PT_BINS,
                "weights": all_pteta_w[suffix],
            },
        }
    json_path = os.path.join(PLOTDIR, "conditional_tables.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {json_path}")

    print("\n" + "=" * 60)
    print("Done! Check plots in:")
    print(f"  {PLOTDIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
