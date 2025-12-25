"""
Preprocess.py - Mass-Decorrelated ParticleNet preprocessing

Modifications from ParticleNet:
- Computes and stores OS pair masses (mass1, mass2) for decorrelation
- B-jets are separate particles (no btagScore feature)
- Node features: [E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]
"""
import logging
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from ROOT import TLorentzVector, TRandom3
from DataFormat import getMuons, getElectrons, getJets, Particle


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

def evtToGraph(nodeList, weight, sample_info, era, mass1, mass2, has_bjet, k=4):
    """
    Convert event node list to PyTorch Geometric Data object.

    Args:
        nodeList: List of node features [E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]
        weight: Event weight (genWeight * puWeight * prefireWeight)
        sample_info: Dict with sample metadata
        era: Data era string
        mass1: First OS muon pair mass (smaller)
        mass2: Second OS muon pair mass (larger), -1 if only one pair
        has_bjet: Float (1.0 if event has b-jets, 0.0 otherwise)
        k: Number of nearest neighbors for edge construction

    Returns:
        PyTorch Geometric Data object
    """
    x = torch.tensor(nodeList, dtype=torch.float)
    edgeIndex, edgeAttribute = getEdgeIndices(nodeList, k=k)

    # Create era-encoded graph-level features (matching V2 format)
    if era == "2016preVFP":
        graphInput = torch.tensor([[1, 0, 0, 0]], dtype=torch.float)
    elif era == "2016postVFP":
        graphInput = torch.tensor([[0, 1, 0, 0]], dtype=torch.float)
    elif era == "2017":
        graphInput = torch.tensor([[0, 0, 1, 0]], dtype=torch.float)
    elif era == "2018":
        graphInput = torch.tensor([[0, 0, 0, 1]], dtype=torch.float)
    else:
        # Default for Run3 or unknown eras - could be extended
        graphInput = torch.tensor([[0, 0, 0, 0]], dtype=torch.float)
        logging.warning(f"Unknown era {era}, using default graphInput")

    data = Data(x=x,
                edge_index=edgeIndex.t().contiguous(),
                edge_attribute=edgeAttribute,
                weight=torch.tensor(weight, dtype=torch.float),
                graphInput=graphInput,
                sample_info=sample_info,
                era=era,
                mass1=torch.tensor([mass1], dtype=torch.float),
                mass2=torch.tensor([mass2], dtype=torch.float),
                has_bjet=torch.tensor([has_bjet], dtype=torch.float))
    return data

def rtfileToDataList(rtfile, sample_name, channel, era, is_diboson=False, maxSize=-1, nFolds=5):
    """
    Convert ROOT file to list of PyTorch Geometric Data objects.

    Args:
        rtfile: ROOT file object
        sample_name: MC sample name (e.g., "TTToHcToWAToMuMu-MHc130MA100", "Skim_TriLep_TTLL_powheg")
        channel: Channel name (e.g., "Run1E2Mu", "Run3Mu")
        era: Data era (2016preVFP, 2016postVFP, 2017, 2018)
        is_diboson: Whether sample is diboson (WZ, ZZ) - relaxes b-jet requirement
        maxSize: Maximum number of events to process (-1 for all)
        nFolds: Number of cross-validation folds

    Note:
        Node features: [E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]
        B-jets are separate particles (no btagScore in features)
        For diboson samples, b-jet requirement is relaxed for DisCo decorrelation.
    """
    dataList = [[] for _ in range(nFolds)]
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

        # Compute has_bjet flag for DisCo decorrelation
        has_bjet = 1.0 if len(bjets) > 0 else 0.0

        # Require at least one b-jet (except for diboson samples)
        # Diboson samples keep events without b-jets for DisCo decorrelation
        if len(bjets) == 0 and not is_diboson:
            continue

        METv = Particle(evt.METvPt, 0., evt.METvPhi, 0.)
        METvPt = evt.METvPt
        nJets = evt.nJets

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
        # Include muons, electrons, jets, b-jets, and MET as particles
        objects = muons + electrons + jets + bjets
        objects.append(METv)

        # Feature construction: [E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]
        nodeList = []
        for obj in objects:
            nodeList.append([obj.E(), obj.Px(), obj.Py(), obj.Pz(),
                            obj.Charge(), obj.IsMuon(), obj.IsElectron(),
                            obj.IsJet(), obj.IsBjet()])

        # Create graph with weight, sample info, masses, and bjet flag
        data = evtToGraph(nodeList, weight, sample_info, era, mass1, mass2, has_bjet)

        # Get a random number and save folds (deterministic based on METvPt)
        randGen = TRandom3()
        seed = int(METvPt)+1  # add one to avoid an automatically computed seed
        randGen.SetSeed(seed)
        fold = -999
        for _ in range(nJets):
            fold = randGen.Integer(nFolds)
        dataList[fold].append(data)
        if max(len(data) for data in dataList) == maxSize:
            break

    for i, data in enumerate(dataList):
        logging.debug(f"no. of dataList ends with {len(data)} for fold {i}")
    logging.debug("=========================================")

    return dataList

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
