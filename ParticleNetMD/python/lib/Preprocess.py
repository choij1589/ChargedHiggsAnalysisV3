"""
Preprocess.py - Mass-Decorrelated ParticleNet preprocessing

Modifications from ParticleNet:
- Computes and stores OS pair masses (mass1, mass2) for decorrelation
- B-jets are separate particles (no btagScore feature)
- Node features: [E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]
- Data augmentation: nonprompt (LNT+FR weights), diboson (rank-based promotion)
"""
import os
import json
import logging
import random
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from ROOT import TLorentzVector, TRandom3
from DataFormat import (getMuons, getElectrons, getJets,
                        getAllMuons, getAllElectrons,
                        Particle, Muon, Electron, Jet, Bjet)


def compute_os_pair_masses(muons):
    """
    Compute invariant masses of opposite-sign muon pairs.

    For 3mu events: 2 OS pairs exist (return both masses)
    For 1e2mu events: 1 OS pair (return mass1, mass2=-1)

    Args:
        muons: List of Muon objects with Charge() method

    Returns:
        mass1, mass2: Two OS pair masses sorted (mass1 <= mass2)
                      mass2 = -1 if only one OS pair exists
    """
    os_pairs = []
    for i, mu1 in enumerate(muons):
        for j, mu2 in enumerate(muons):
            if i >= j:
                continue
            if mu1.Charge() * mu2.Charge() < 0:  # Opposite sign
                pair_mass = (mu1 + mu2).M()
                os_pairs.append(pair_mass)

    if len(os_pairs) == 0:
        return -1., -1.
    elif len(os_pairs) == 1:
        return os_pairs[0], -1.
    else:
        # Sort: mass1 <= mass2
        os_pairs.sort()
        return os_pairs[0], os_pairs[1]

def getEdgeIndices(nodeList, k=4):
    edgeIndex = []
    edgeAttribute = []
    for i, node in enumerate(nodeList):
        distances = {}
        for j, neigh in enumerate(nodeList):
            # avoid same node
            if node is neigh: continue
            thisPart = TLorentzVector()
            neighPart = TLorentzVector()
            thisPart.SetPxPyPzE(node[1], node[2], node[3], node[0])
            neighPart.SetPxPyPzE(neigh[1], neigh[2], neigh[3], neigh[0])
            distances[j] = thisPart.DeltaR(neighPart)
        distances = dict(sorted(distances.items(), key=lambda item: item[1]))
        for n in list(distances.keys())[:k]:
            edgeIndex.append([i, n])
            edgeAttribute.append([distances[n]])

    return (torch.tensor(edgeIndex, dtype=torch.long), torch.tensor(edgeAttribute, dtype=torch.float))

def evtToGraph(nodeList, weight, sample_info, era, mass1, mass2, k=4):
    """
    Convert event node list to PyTorch Geometric Data object.

    Args:
        nodeList: List of node features [E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]
        weight: Event weight (genWeight * puWeight * prefireWeight)
        sample_info: Dict with sample metadata
        era: Data era string
        mass1: First OS muon pair mass (smaller)
        mass2: Second OS muon pair mass (larger), -1 if only one pair
        k: Number of nearest neighbors for edge construction

    Returns:
        PyTorch Geometric Data object
    """
    x = torch.tensor(nodeList, dtype=torch.float)
    edgeIndex, edgeAttribute = getEdgeIndices(nodeList, k=k)

    # Create era-encoded graph-level features
    # 8-dim one-hot: [2016preVFP, 2016postVFP, 2017, 2018,
    #                  2022, 2022EE, 2023, 2023BPix]
    ERA_INDEX = {
        "2016preVFP": 0, "2016postVFP": 1, "2017": 2, "2018": 3,
        "2022": 4, "2022EE": 5, "2023": 6, "2023BPix": 7,
    }
    vec = [0.0] * 8
    if era in ERA_INDEX:
        vec[ERA_INDEX[era]] = 1.0
    else:
        logging.warning(f"Unknown era {era}, using zero graphInput")
    graphInput = torch.tensor([vec], dtype=torch.float)

    data = Data(x=x,
                edge_index=edgeIndex.t().contiguous(),
                edge_attribute=edgeAttribute,
                weight=torch.tensor(weight, dtype=torch.float),
                graphInput=graphInput,
                sample_info=sample_info,
                era=era,
                mass1=torch.tensor([mass1], dtype=torch.float),
                mass2=torch.tensor([mass2], dtype=torch.float))
    return data

def _assign_folds(flat_data_list, nFolds=5, seed=42):
    """Shuffle and split a flat list of Data objects into nFolds.

    Two-pass approach:
    1. Shuffle with fixed seed for reproducibility
    2. Round-robin assignment: fold = i % nFolds

    Returns: list of nFolds lists
    """
    rng = random.Random(seed)
    indices = list(range(len(flat_data_list)))
    rng.shuffle(indices)

    dataList = [[] for _ in range(nFolds)]
    for i, idx in enumerate(indices):
        fold = i % nFolds
        dataList[fold].append(flat_data_list[idx])

    for i, data in enumerate(dataList):
        logging.debug(f"Fold {i}: {len(data)} events")

    return dataList


def rtfileToDataList(rtfile, sample_name, channel, era, maxSize=-1, nFolds=5):
    """
    Convert ROOT file to list of PyTorch Geometric Data objects.
    Uses Tight lepton selection + b-jet requirement (signal/ttX).

    Args:
        rtfile: ROOT file object
        sample_name: MC sample name (e.g., "TTToHcToWAToMuMu-MHc130MA100", "Skim_TriLep_TTLL_powheg")
        channel: Channel name (e.g., "Run1E2Mu", "Run3Mu")
        era: Data era (2016preVFP, ..., 2023BPix)
        maxSize: Maximum number of events to process (-1 for all)
        nFolds: Number of cross-validation folds

    Note:
        Node features: [E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]
        B-jets are separate particles (no btagScore in features)
        Fold assignment: shuffle+split (replaces METvPt-based seeding)
    """
    flat_data = []
    for evt in rtfile.Events:
        muons = getMuons(evt)
        electrons = getElectrons(evt)
        jets, bjets = getJets(evt)

        # Validate lepton count matches channel requirement
        if channel == "Run3Mu":
            if len(muons) != 3 or len(electrons) != 0:
                continue
        elif channel == "Run1E2Mu":
            if len(muons) != 2 or len(electrons) != 1:
                continue
        else:
            raise ValueError(f"Unknown channel: {channel}")

        METv = Particle(evt.METvPt, 0., evt.METvPhi, 0.)

        # Compute OS muon pair masses for decorrelation
        mass1, mass2 = compute_os_pair_masses(muons)

        # Calculate event weight: genWeight * puWeight * prefireWeight
        weight = evt.genWeight * evt.puWeight * evt.prefireWeight

        # Sample information
        sample_info = {
            'sample_name': sample_name,
            'channel': channel
        }

        # Convert event to graph
        objects = muons + electrons + jets + bjets
        objects.append(METv)

        nodeList = []
        for obj in objects:
            nodeList.append([obj.E(), obj.Px(), obj.Py(), obj.Pz(),
                            obj.Charge(), obj.IsMuon(), obj.IsElectron(),
                            obj.IsJet(), obj.IsBjet()])

        data = evtToGraph(nodeList, weight, sample_info, era, mass1, mass2)
        flat_data.append(data)

        if maxSize > 0 and len(flat_data) >= maxSize:
            break

    logging.info(f"Collected {len(flat_data)} events from {era}/{sample_name}")
    return _assign_folds(flat_data, nFolds)


# ---------------------------------------------------------------------------
# Fake rate helpers (for nonprompt augmentation)
# ---------------------------------------------------------------------------

# Fake rate bin edges (matching correctionlib JSONs)
MU_PTCORR_EDGES = [10., 12., 14., 17., 20., 30., 50., 100.]
MU_ABSETA_EDGES = [0., 0.9, 1.6, 2.4]
EL_PTCORR_EDGES = [15., 17., 20., 25., 35., 50., 100.]
EL_ABSETA_EDGES = [0., 0.8, 1.479, 2.5]


def _find_bin(val, edges):
    """Find bin index for value in bin edges (clamp to valid range)."""
    if val < edges[0]:
        return 0
    for i in range(len(edges) - 1):
        if val < edges[i + 1]:
            return i
    return len(edges) - 2


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
                    f"{correction_name}: expected {n_eta * n_pt} values, "
                    f"got {len(content)}")
            rates = []
            for ie in range(n_eta):
                row = content[ie * n_pt: (ie + 1) * n_pt]
                rates.append(row)
            return rates

    raise KeyError(f"{correction_name} not found in {json_path}")


def load_fakerate_tables(sknano_data_dir):
    """Load correctionlib fake rate JSONs for all 8 eras.
    Path: {sknano_data_dir}/{era}/{MUO|EGM}/fakerate_TopHNT.json
    Returns: {era: {"muon": 2D_rates, "electron": 2D_rates}}
    """
    ALL_ERAS = ["2016preVFP", "2016postVFP", "2017", "2018",
                "2022", "2022EE", "2023", "2023BPix"]
    n_mu_eta = len(MU_ABSETA_EDGES) - 1
    n_mu_pt = len(MU_PTCORR_EDGES) - 1
    n_el_eta = len(EL_ABSETA_EDGES) - 1
    n_el_pt = len(EL_PTCORR_EDGES) - 1

    tables = {}
    for era in ALL_ERAS:
        mu_path = os.path.join(sknano_data_dir, era, "MUO", "fakerate_TopHNT.json")
        el_path = os.path.join(sknano_data_dir, era, "EGM", "fakerate_TopHNT.json")

        if not os.path.exists(mu_path):
            raise FileNotFoundError(f"Muon fake rate not found: {mu_path}")
        if not os.path.exists(el_path):
            raise FileNotFoundError(f"Electron fake rate not found: {el_path}")

        mu_rates = _extract_rates(mu_path, "fakerate_muon_TT", n_mu_eta, n_mu_pt)
        el_rates = _extract_rates(el_path, "fakerate_electron_TT", n_el_eta, n_el_pt)
        tables[era] = {"muon": mu_rates, "electron": el_rates}

    return tables


def compute_fake_rate_weight(muons, muon_is_tight, electrons, electron_is_tight,
                             evt, fr_tables, era, is_run3):
    """Compute FR weight for one LNT event.
    w = -1 × Π[-f/(1-f)] over non-tight leptons
    Uses raw pT and miniIso for FR lookup (50 GeV cap for Run3 muons).
    The muons/electrons passed in already have ptCorr applied to their 4-vectors.
    """
    rates = fr_tables[era]
    w = -1.0

    # Process muons
    for i, (mu_pt_raw, mu_eta, mu_miniIso, tight) in enumerate(
            zip(evt.MuonPtColl, evt.MuonEtaColl, evt.MuonMiniIsoColl, evt.MuonIsTightColl)):
        if tight:
            continue
        ptCorr = mu_pt_raw * (1.0 + max(0.0, mu_miniIso - 0.1))
        # 50 GeV cap for Run3 muons ONLY for FR lookup
        if is_run3 and ptCorr > 50.0:
            ptCorr = 49.0
        absEta = abs(mu_eta)
        eta_bin = _find_bin(absEta, MU_ABSETA_EDGES)
        pt_bin = _find_bin(ptCorr, MU_PTCORR_EDGES)
        f = rates["muon"][eta_bin][pt_bin]
        w *= -1.0 * f / (1.0 - f)

    # Process electrons — use ScEta (supercluster eta) for FR lookup;
    # the eta bins [0, 0.8, 1.479, 2.5] are ECAL boundaries in |scEta|
    for i, (el_pt_raw, el_eta, el_miniIso, tight) in enumerate(
            zip(evt.ElectronPtColl, evt.ElectronScEtaColl, evt.ElectronMiniIsoColl, evt.ElectronIsTightColl)):
        if tight:
            continue
        ptCorr = el_pt_raw * (1.0 + max(0.0, el_miniIso - 0.1))
        absEta = abs(el_eta)
        eta_bin = _find_bin(absEta, EL_ABSETA_EDGES)
        pt_bin = _find_bin(ptCorr, EL_PTCORR_EDGES)
        f = rates["electron"][eta_bin][pt_bin]
        w *= -1.0 * f / (1.0 - f)

    return w


# ---------------------------------------------------------------------------
# Conditional promotion helpers (for diboson augmentation)
# ---------------------------------------------------------------------------

# b-jet calibration bin edges
BJET_ETA_BINS = [0.0, 0.8, 1.6, 2.1, 2.5]
BJET_PT_BINS = [20., 30., 50., 70., 100., 140., 200., 300., 10000.]


def load_conditional_tables(json_path):
    """Load conditional_tables.json from dibosonRankPromote.py output.
    Returns: {"Run2": {p_1b, cdf_rank, cdf_pair, njets_weights, bjet_pteta_weights},
              "Run3": {...}}
    """
    with open(json_path) as f:
        raw = json.load(f)

    tables = {}
    for era_group in ("Run2", "Run3"):
        d = raw[era_group]

        # Parse p_1b: {str(g): float} → list indexed by nj_group
        p_1b_dict = d["p_1b"]
        max_g = max(int(k) for k in p_1b_dict.keys())
        p_1b = [0.0] * (max_g + 1)
        for g_str, val in p_1b_dict.items():
            p_1b[int(g_str)] = val

        # Parse rank_probs → build CDF
        rank_probs = d["rank_probs"]
        rank_max = 10
        cdf_rank = [[1.0] * rank_max for _ in range(max_g + 1)]  # dummy for g=0
        for g_str, probs in rank_probs.items():
            g = int(g_str)
            cumul = 0.0
            for r in range(rank_max):
                cumul += probs[r] if r < len(probs) else 0.0
                cdf_rank[g][r] = cumul
            cdf_rank[g][rank_max - 1] = 1.0

        # Parse pair_probs → build CDF
        pair_probs = d["pair_probs"]
        pair_bins = 100
        cdf_pair = [[1.0] * pair_bins for _ in range(max_g + 1)]  # dummy for g=0
        for g_str, probs_dict in pair_probs.items():
            g = int(g_str)
            raw_probs = [0.0] * pair_bins
            for pc_str, p in probs_dict.items():
                raw_probs[int(pc_str)] = p
            cumul = 0.0
            for pc in range(pair_bins):
                cumul += raw_probs[pc]
                cdf_pair[g][pc] = cumul
            cdf_pair[g][pair_bins - 1] = 1.0

        # Parse njets_weights
        njets_weights = {}
        for k, v in d["njets_weights"].items():
            njets_weights[int(k)] = v

        # Parse bjet_pteta_weights
        pteta_data = d["bjet_pteta_weights"]
        eta_bins = pteta_data["eta_bins"]
        pt_bins = pteta_data["pt_bins"]
        pteta_weights = pteta_data["weights"]

        tables[era_group] = {
            'p_1b': p_1b,
            'cdf_rank': cdf_rank,
            'cdf_pair': cdf_pair,
            'nj_max': max_g,
            'njets_weights': njets_weights,
            'bjet_pteta_weights': pteta_weights,
            'eta_bins': eta_bins,
            'pt_bins': pt_bins,
        }

    return tables


def promote_event(jet_pts, jet_etas, nj_max, p_1b, cdf_rank, cdf_pair, entry_seed):
    """Conditional rank-based promotion for one event.
    Returns: list of promoted jet indices (becomes b-jets).
    Uses deterministic seeding: TRandom3(entry_seed * 2654435761 + 12345).
    """
    nJ = len(jet_pts)
    if nJ <= 0:
        return []

    # Sort jets by pT descending to get rank
    sorted_indices = sorted(range(nJ), key=lambda i: jet_pts[i], reverse=True)

    g = min(nJ, nj_max)
    rng = TRandom3(int((entry_seed * 2654435761 + 12345) & 0xFFFFFFFF))

    if nJ == 1:
        return [sorted_indices[0]]

    u1 = rng.Rndm()
    is_1b = (u1 < p_1b[g])

    if is_1b:
        u2 = rng.Rndm()
        rank = 0
        for r in range(10):
            if u2 < cdf_rank[g][r]:
                rank = r
                break
        if rank >= nJ:
            rank = nJ - 1
        return [sorted_indices[rank]]
    else:
        u2 = rng.Rndm()
        pair_code = 1
        for pc in range(100):
            if u2 < cdf_pair[g][pc]:
                pair_code = pc
                break
        r1 = pair_code // 10
        r2 = pair_code % 10
        if r1 >= nJ:
            r1 = 0
        if r2 >= nJ:
            r2 = nJ - 1
        if r1 >= r2:
            r1 = 0
            r2 = 1
        return [sorted_indices[r1], sorted_indices[r2]]


def compute_calibration_weight(bjet_pts, bjet_etas, nJets,
                                njets_weights, pteta_weights, eta_bins, pt_bins):
    """Compute njets_weight × product of pteta_weight per promoted b-jet."""
    # nJets weight
    w = njets_weights.get(nJets, 1.0)

    # Per-b-jet (pT, eta) calibration weight
    for pt, eta in zip(bjet_pts, bjet_etas):
        absEta = min(abs(eta), eta_bins[-1] - 0.01)
        pt_clamped = max(pt_bins[0], min(pt, pt_bins[-1] - 0.1))
        # Find eta bin
        ie = 0
        for i in range(len(eta_bins) - 1):
            if absEta < eta_bins[i + 1]:
                ie = i
                break
        else:
            ie = len(eta_bins) - 2
        # Find pt bin
        ip = 0
        for i in range(len(pt_bins) - 1):
            if pt_clamped < pt_bins[i + 1]:
                ip = i
                break
        else:
            ip = len(pt_bins) - 2
        w *= pteta_weights[ie][ip]

    return w


# ---------------------------------------------------------------------------
# Nonprompt conversion function (LNT+Bjet with FR weights)
# ---------------------------------------------------------------------------
def rtfileToDataList_nonprompt(rtfile, sample_name, channel, era,
                                fr_tables, is_run3, maxSize=-1, nFolds=5):
    """LNT+Bjet event conversion with FR weights.
    Selection: !allTrue(isTight) AND anyTrue(JetIsBtaggedColl)
    Leptons: getAllMuons/getAllElectrons (ptCorr applied, no cap)
    Weight: genWeight × puWeight × prefireWeight × FR_weight
    FR lookup uses capped ptCorr (50 GeV for Run3 muons)
    """
    flat_data = []
    for evt in rtfile.Events:
        # Check LNT: at least one non-tight lepton
        all_mu_tight = all(bool(t) for t in evt.MuonIsTightColl)
        all_el_tight = all(bool(t) for t in evt.ElectronIsTightColl)
        if all_mu_tight and all_el_tight:
            continue

        # Check b-jet requirement
        has_bjet = any(bool(b) for b in evt.JetIsBtaggedColl)
        if not has_bjet:
            continue

        # Get all leptons with ptCorr
        muons, mu_is_tight = getAllMuons(evt)
        electrons, el_is_tight = getAllElectrons(evt)

        # Validate lepton count
        if channel == "Run3Mu":
            if len(muons) != 3 or len(electrons) != 0:
                continue
        elif channel == "Run1E2Mu":
            if len(muons) != 2 or len(electrons) != 1:
                continue
        else:
            raise ValueError(f"Unknown channel: {channel}")

        jets, bjets = getJets(evt)
        METv = Particle(evt.METvPt, 0., evt.METvPhi, 0.)

        # Compute OS muon pair masses for decorrelation
        mass1, mass2 = compute_os_pair_masses(muons)

        # Calculate event weight with FR weight
        base_weight = evt.genWeight * evt.puWeight * evt.prefireWeight
        fr_weight = compute_fake_rate_weight(
            muons, mu_is_tight, electrons, el_is_tight,
            evt, fr_tables, era, is_run3)
        weight = base_weight * fr_weight

        sample_info = {
            'sample_name': sample_name,
            'channel': channel
        }

        objects = muons + electrons + jets + bjets
        objects.append(METv)

        nodeList = []
        for obj in objects:
            nodeList.append([obj.E(), obj.Px(), obj.Py(), obj.Pz(),
                            obj.Charge(), obj.IsMuon(), obj.IsElectron(),
                            obj.IsJet(), obj.IsBjet()])

        data = evtToGraph(nodeList, weight, sample_info, era, mass1, mass2)
        flat_data.append(data)

        if maxSize > 0 and len(flat_data) >= maxSize:
            break

    logging.info(f"Collected {len(flat_data)} nonprompt (LNT+Bjet) events from {era}/{sample_name}")
    return _assign_folds(flat_data, nFolds)


# ---------------------------------------------------------------------------
# Diboson conversion function (0-tag promoted with calibration weights)
# ---------------------------------------------------------------------------
def rtfileToDataList_diboson(rtfile, sample_name, channel, era,
                              cond_tables, era_group, entry_offset=0,
                              maxSize=-1, nFolds=5):
    """0-tag promoted event conversion with calibration weights.
    Selection: allTrue(isTight) AND !anyTrue(JetIsBtaggedColl) AND nJets>0
    Promotion: conditional rank-based → promoted jets become Bjet in node features
    Weight: genWeight × puWeight × prefireWeight × njets_w × pteta_w
    entry_offset: for deterministic promotion seeding across multiple files
    """
    tables = cond_tables[era_group]
    p_1b = tables['p_1b']
    cdf_rank = tables['cdf_rank']
    cdf_pair = tables['cdf_pair']
    nj_max = tables['nj_max']
    njets_weights = tables['njets_weights']
    pteta_weights = tables['bjet_pteta_weights']
    eta_bins = tables['eta_bins']
    pt_bins = tables['pt_bins']

    flat_data = []
    entry_idx = 0
    for evt in rtfile.Events:
        # Check tight lepton selection
        all_mu_tight = all(bool(t) for t in evt.MuonIsTightColl)
        all_el_tight = all(bool(t) for t in evt.ElectronIsTightColl)
        if not (all_mu_tight and all_el_tight):
            entry_idx += 1
            continue

        # Check 0-tag with jets
        has_bjet = any(bool(b) for b in evt.JetIsBtaggedColl)
        if has_bjet:
            entry_idx += 1
            continue
        if evt.nJets <= 0:
            entry_idx += 1
            continue

        muons = getMuons(evt)
        electrons = getElectrons(evt)

        # Validate lepton count
        if channel == "Run3Mu":
            if len(muons) != 3 or len(electrons) != 0:
                entry_idx += 1
                continue
        elif channel == "Run1E2Mu":
            if len(muons) != 2 or len(electrons) != 1:
                entry_idx += 1
                continue
        else:
            raise ValueError(f"Unknown channel: {channel}")

        # Get jet info for promotion
        jet_pts = list(evt.JetPtColl)
        jet_etas = list(evt.JetEtaColl)
        jet_phis = list(evt.JetPhiColl)
        jet_masses = list(evt.JetMassColl)
        jet_charges = list(evt.JetChargeColl)
        jet_btagScores = list(evt.JetBtagScoreColl)

        # Promote jets to b-jets
        promoted_indices = promote_event(
            jet_pts, jet_etas, nj_max, p_1b, cdf_rank, cdf_pair,
            entry_offset + entry_idx)

        if len(promoted_indices) == 0:
            entry_idx += 1
            continue

        # Build jet/bjet lists with promotion applied
        promoted_set = set(promoted_indices)
        jets = []
        bjets = []
        bjet_pts_promoted = []
        bjet_etas_promoted = []
        for j in range(len(jet_pts)):
            if j in promoted_set:
                # This jet is promoted to b-jet
                thisBjet = Bjet(jet_pts[j], jet_etas[j], jet_phis[j], jet_masses[j])
                thisBjet.SetCharge(jet_charges[j])
                thisBjet.SetBtagScore(jet_btagScores[j])
                bjets.append(thisBjet)
                bjet_pts_promoted.append(jet_pts[j])
                bjet_etas_promoted.append(jet_etas[j])
            else:
                thisJet = Jet(jet_pts[j], jet_etas[j], jet_phis[j], jet_masses[j])
                thisJet.SetCharge(jet_charges[j])
                thisJet.SetBtagScore(jet_btagScores[j])
                jets.append(thisJet)

        METv = Particle(evt.METvPt, 0., evt.METvPhi, 0.)

        # Compute OS muon pair masses for decorrelation
        mass1, mass2 = compute_os_pair_masses(muons)

        # Calculate event weight with calibration
        base_weight = evt.genWeight * evt.puWeight * evt.prefireWeight
        cal_weight = compute_calibration_weight(
            bjet_pts_promoted, bjet_etas_promoted, evt.nJets,
            njets_weights, pteta_weights, eta_bins, pt_bins)
        weight = base_weight * cal_weight

        sample_info = {
            'sample_name': sample_name,
            'channel': channel
        }

        objects = muons + electrons + jets + bjets
        objects.append(METv)

        nodeList = []
        for obj in objects:
            nodeList.append([obj.E(), obj.Px(), obj.Py(), obj.Pz(),
                            obj.Charge(), obj.IsMuon(), obj.IsElectron(),
                            obj.IsJet(), obj.IsBjet()])

        data = evtToGraph(nodeList, weight, sample_info, era, mass1, mass2)
        flat_data.append(data)
        entry_idx += 1

        if maxSize > 0 and len(flat_data) >= maxSize:
            break

    logging.info(f"Collected {len(flat_data)} diboson (0-tag promoted) events from {era}/{sample_name}")
    return _assign_folds(flat_data, nFolds)

class GraphDataset(InMemoryDataset):
    def __init__(self, data_list):
        super(GraphDataset, self).__init__("./tmp/data")
        self.data_list = data_list
        self.data, self.slices = self.collate(data_list)


class SharedBatchDataset(torch.utils.data.Dataset):
    """
    Dataset wrapper for pre-batched shared memory Batch objects.

    This class provides efficient per-graph access from a large pre-batched
    Batch object without unbatching all events into individual Data objects.
    This eliminates the memory overhead of creating hundreds of thousands of
    Python objects.

    Usage:
        shared_batch = Batch(...)  # Large batch with all events
        dataset = SharedBatchDataset(shared_batch)
        loader = DataLoader(dataset, batch_size=1024, shuffle=True,
                          collate_fn=Batch.from_data_list)

    Memory savings:
        - Without this: ~1.5 GB per worker for 491K Data objects
        - With this: ~100 MB per worker for Batch metadata only
    """

    def __init__(self, shared_batch):
        """
        Initialize dataset from pre-batched Batch object.

        Args:
            shared_batch: PyTorch Geometric Batch object containing all events
                         with tensors in shared memory
        """
        from torch_geometric.data import Batch

        if not isinstance(shared_batch, Batch):
            raise TypeError(f"Expected Batch object, got {type(shared_batch)}")

        self.batch = shared_batch
        self.num_graphs = shared_batch.num_graphs

    def __len__(self):
        """Return number of graphs in the dataset."""
        return self.num_graphs

    def __getitem__(self, idx):
        """
        Get individual graph by index.

        DataLoader will call this method to retrieve individual examples,
        then collate them into mini-batches.

        Args:
            idx: Integer index of the graph to retrieve

        Returns:
            Data object for the requested graph
        """
        return self.batch.get_example(idx)
