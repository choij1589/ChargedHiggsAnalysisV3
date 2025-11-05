import logging
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from ROOT import TLorentzVector, TRandom3
from DataFormat import getMuons, getElectrons, getJets, Particle

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

def evtToGraph(nodeList, weight, sample_info, era, separate_bjets=False, k=4):
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
                era=era)
    return data

def rtfileToDataList(rtfile, sample_name, channel, era, maxSize=-1, nFolds=5, separate_bjets=False):
    """
    Convert ROOT file to list of PyTorch Geometric Data objects.

    Args:
        rtfile: ROOT file object
        sample_name: MC sample name (e.g., "TTToHcToWAToMuMu-MHc130MA100", "Skim_TriLep_TTLL_powheg")
        channel: Channel name (e.g., "Run1E2Mu", "Run3Mu")
        era: Data era (2016preVFP, 2016postVFP, 2017, 2018)
        maxSize: Maximum number of events to process (-1 for all)
        nFolds: Number of cross-validation folds
    """
    dataList = [[] for _ in range(nFolds)]
    for evt in rtfile.Events:
        muons = getMuons(evt)
        electrons = getElectrons(evt)
        jets, bjets = getJets(evt, separate_bjets=separate_bjets)
        METv = Particle(evt.METvPt, 0., evt.METvPhi, 0.)
        METvPt = evt.METvPt
        nJets = evt.nJets

        # Calculate event weight: genWeight * puWeight * prefireWeight
        weight = evt.genWeight * evt.puWeight * evt.prefireWeight

        # Sample information
        sample_info = {
            'sample_name': sample_name,
            'channel': channel
        }

        # convert event to a graph
        nodeList = []
        if separate_bjets:
            # Include both regular jets and b-jets as separate particles
            objects = muons + electrons + jets + bjets
            objects.append(METv)
            # Feature construction: [E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]
            for obj in objects:
                nodeList.append([obj.E(), obj.Px(), obj.Py(), obj.Pz(),
                                obj.Charge(), obj.IsMuon(), obj.IsElectron(),
                                obj.IsJet(), obj.IsBjet()])
        else:
            # Current behavior: all jets together with btagScore
            objects = muons + electrons + jets
            objects.append(METv)
            # Feature construction: [E, Px, Py, Pz, charge, btagScore, isMuon, isElectron, isJet]
            for obj in objects:
                nodeList.append([obj.E(), obj.Px(), obj.Py(), obj.Pz(),
                                obj.Charge(), obj.BtagScore(),
                                obj.IsMuon(), obj.IsElectron(), obj.IsJet()])

        # Create graph with weight and sample info (no labels)
        data = evtToGraph(nodeList, weight, sample_info, era, separate_bjets=separate_bjets)

        # Get a random number and save folds (deterministic based on METvPt)
        randGen = TRandom3()
        seed = int(METvPt)+1 # add one to avoid an automatically computed seed
        randGen.SetSeed(seed)
        fold = -999
        for _ in range(nJets):
            fold = randGen.Integer(nFolds)
        dataList[fold].append(data)
        if max(len(data) for data in dataList) == maxSize: break

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
