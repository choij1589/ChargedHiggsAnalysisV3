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

def evtToGraph(nodeList, weight, sample_info, era, k=4):
    x = torch.tensor(nodeList, dtype=torch.float)
    edgeIndex, edgeAttribute = getEdgeIndices(nodeList, k=k)
    data = Data(x=x,
                edge_index=edgeIndex.t().contiguous(),
                edge_attribute=edgeAttribute,
                weight=torch.tensor(weight, dtype=torch.float),
                sample_info=sample_info,
                era=era)
    return data

def rtfileToDataList(rtfile, sample_name, channel, era, maxSize=-1, nFolds=5):
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
        jets, bjets = getJets(evt)
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
        objects = muons+electrons+jets; objects.append(METv)
        for obj in objects:
            nodeList.append([obj.E(), obj.Px(), obj.Py(), obj.Pz(),
                             obj.Charge(), obj.BtagScore(),
                             obj.IsMuon(), obj.IsElectron(), obj.IsJet()])

        # Create graph with weight and sample info (no labels)
        data = evtToGraph(nodeList, weight, sample_info, era)

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
