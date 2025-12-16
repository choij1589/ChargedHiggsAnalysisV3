#!/usr/bin/env python
"""
B-jet subset utilities for ParticleNet training.

Provides functions to filter datasets to events containing b-jets
and evaluate model performance on b-jet subsets.
"""

import logging
from typing import List, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from Preprocess import GraphDataset


def has_bjet_in_event(node_features: torch.Tensor) -> bool:
    """
    Check if an event contains b-jets based on node features.

    ParticleNetMD uses node features: [E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]
    B-jets are identified by the isBjet flag at index 8.

    Args:
        node_features: Node feature tensor (num_nodes, num_features)

    Returns:
        True if event contains at least one b-jet
    """
    # Check feature index 8 (isBjet flag)
    return (node_features[:, 8] > 0.5).any().item()


def filter_bjet_events(data_list: List[Data]) -> Tuple[List[Data], int, int]:
    """
    Filter a list of graph data to only events containing b-jets.

    Args:
        data_list: List of PyTorch Geometric Data objects

    Returns:
        Tuple of (filtered_data_list, original_count, filtered_count)
    """
    filtered_data = []

    for data in data_list:
        if has_bjet_in_event(data.x):
            filtered_data.append(data)

    original_count = len(data_list)
    filtered_count = len(filtered_data)

    logging.info(f"Filtered to b-jet events: {filtered_count}/{original_count} "
                f"({filtered_count/original_count*100:.1f}%)")

    return filtered_data, original_count, filtered_count


def create_bjet_subset_loader(data_list: List[Data],
                              batch_size: int = 1024) -> Tuple[DataLoader, int, int]:
    """
    Create a DataLoader containing only events with b-jets.

    Args:
        data_list: List of PyTorch Geometric Data objects
        batch_size: Batch size for data loader

    Returns:
        Tuple of (bjet_loader, original_count, filtered_count)
    """
    # Filter events
    filtered_data, original_count, filtered_count = filter_bjet_events(data_list)

    if filtered_count == 0:
        logging.warning("No events with b-jets found in dataset!")
        return None, original_count, 0

    # Create dataset and loader
    bjet_dataset = GraphDataset(filtered_data)
    bjet_loader = DataLoader(
        bjet_dataset, batch_size=batch_size, pin_memory=True, shuffle=False
    )

    logging.info(f"Created b-jet subset loader with {filtered_count} events")

    return bjet_loader, original_count, filtered_count


def get_bjet_subset_statistics(data_list: List[Data]) -> dict:
    """
    Get statistics about b-jet events in a dataset.

    Args:
        data_list: List of PyTorch Geometric Data objects

    Returns:
        Dictionary with statistics
    """
    total_events = len(data_list)
    bjet_events = 0
    total_bjets = 0

    for data in data_list:
        if has_bjet_in_event(data.x):
            bjet_events += 1
            # Count number of b-jets in this event (isBjet flag at index 8)
            num_bjets = (data.x[:, 8] > 0.5).sum().item()
            total_bjets += num_bjets

    stats = {
        'total_events': total_events,
        'bjet_events': bjet_events,
        'bjet_event_fraction': bjet_events / total_events if total_events > 0 else 0,
        'avg_bjets_per_bjet_event': total_bjets / bjet_events if bjet_events > 0 else 0,
        'total_bjets': total_bjets
    }

    return stats
