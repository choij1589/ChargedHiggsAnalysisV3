#!/usr/bin/env python
"""Shared utilities for multi-worker training with shared datasets.

Imported by both GA (launchGAOptim.py / trainWorker.py) and Lambda sweep
(launchLambdaSweep.py / lambdaSweepWorker.py) to avoid code duplication.
"""

import torch.multiprocessing as mp
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from Preprocess import SharedBatchDataset


def setup_spawn_method():
    """Configure multiprocessing start method to 'spawn' (required for CUDA)."""
    if mp.get_start_method(allow_none=True) != 'spawn':
        try:
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # already set by another module


def make_dataloader_from_batch(shared_batch: Batch, batch_size: int,
                                shuffle: bool, pin_memory: bool = True) -> DataLoader:
    """Create a DataLoader from a shared-memory Batch (no data copying).

    Used by both GA workers (trainWorker.py) and Lambda sweep workers
    (lambdaSweepWorker.py).
    """
    dataset = SharedBatchDataset(shared_batch)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=0,
        collate_fn=Batch.from_data_list,
    )
