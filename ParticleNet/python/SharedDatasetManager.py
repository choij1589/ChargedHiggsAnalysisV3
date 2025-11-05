#!/usr/bin/env python
"""
Shared Dataset Manager for multi-process GA optimization.

This module handles loading datasets once and preparing them for shared memory access
across multiple training processes, significantly reducing memory usage.
"""

import os
import logging
import torch
import torch.multiprocessing as mp
from typing import List, Dict, Optional, Tuple
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader

class SharedDatasetManager:
    """
    Manages dataset loading and shared memory preparation for multi-process training.

    This class ensures that large datasets are loaded only once and shared across
    multiple training processes using PyTorch's shared memory mechanism.
    """

    def __init__(self):
        """Initialize the shared dataset manager."""
        self.shared_cache = {}
        self.logger = logging.getLogger(__name__)

    def prepare_shared_datasets(self,
                              loader,
                              signal_sample: str,
                              background_groups: Dict[str, List[str]],
                              channel: str,
                              train_folds: List[int],
                              valid_folds: List[int],
                              pilot: bool = False,
                              max_events_per_fold: Optional[int] = None,
                              balance_weights: bool = True,
                              random_state: int = 42) -> Tuple[Batch, Batch]:
        """
        Load and prepare datasets for shared memory access.

        Args:
            loader: DynamicDatasetLoader instance
            signal_sample: Signal sample name
            background_groups: Dictionary of background groups
            channel: Channel name (Run1E2Mu, Run3Mu, etc.)
            train_folds: List of training fold indices
            valid_folds: List of validation fold indices
            pilot: Whether to use pilot datasets
            max_events_per_fold: Maximum events per fold per class
            balance_weights: Whether to balance class weights
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_batch, valid_batch) with all tensors in shared memory
        """

        # Create cache key
        cache_key = f"{signal_sample}_{channel}_{'_'.join(map(str, train_folds))}_{pilot}"

        # Check if already in cache
        if cache_key in self.shared_cache:
            self.logger.info(f"Using cached shared datasets for key: {cache_key}")
            return self.shared_cache[cache_key]

        self.logger.info("=" * 60)
        self.logger.info("Loading datasets into shared memory...")
        self.logger.info(f"Signal: {signal_sample}")
        self.logger.info(f"Channel: {channel}")
        self.logger.info(f"Pilot mode: {pilot}")
        self.logger.info("=" * 60)

        # Load training data
        train_data_list = loader.load_multiclass_with_subsampling(
            signal_sample=signal_sample,
            background_groups=background_groups,
            channel=channel,
            fold_list=train_folds,
            pilot=pilot,
            max_events_per_fold=max_events_per_fold,
            balance_weights=balance_weights,
            random_state=random_state
        )

        # Load validation data
        valid_data_list = loader.load_multiclass_with_subsampling(
            signal_sample=signal_sample,
            background_groups=background_groups,
            channel=channel,
            fold_list=valid_folds,
            pilot=pilot,
            max_events_per_fold=max_events_per_fold,
            balance_weights=balance_weights,
            random_state=random_state
        )

        # Move all tensors to shared memory
        self.logger.info("Moving tensors to shared memory...")
        train_shared = self._move_to_shared_memory(train_data_list, "train")
        valid_shared = self._move_to_shared_memory(valid_data_list, "valid")

        # Cache for reuse
        self.shared_cache[cache_key] = (train_shared, valid_shared)

        self.logger.info(f"Shared memory preparation complete!")
        self.logger.info(f"Train dataset: {train_shared.num_graphs} events")
        self.logger.info(f"Valid dataset: {valid_shared.num_graphs} events")
        self.logger.info("=" * 60)

        return train_shared, valid_shared

    def _move_to_shared_memory(self, data_list: List[Data], dataset_type: str) -> Batch:
        """
        Convert graph dataset to shared tensors by pre-batching all data.

        This creates a single large batch containing all events, then shares the resulting
        tensors. This reduces the number of share_memory_() calls dramatically.

        Args:
            data_list: List of PyTorch Geometric Data objects
            dataset_type: Type of dataset (train/valid) for logging

        Returns:
            Batch object with all tensors in shared memory
        """
        self.logger.info(f"Pre-batching {len(data_list)} events into single shared batch...")

        # Collate ALL events into one large Batch
        full_batch = Batch.from_data_list(data_list)

        # Share all tensors in the batch
        tensor_count = 0
        total_bytes = 0

        for key in full_batch.keys():
            value = getattr(full_batch, key)
            if torch.is_tensor(value):
                # Share this tensor in-place
                value.share_memory_()
                tensor_count += 1
                total_bytes += value.element_size() * value.nelement()

        self.logger.info(f"Created single shared batch for {dataset_type} dataset")
        self.logger.info(f"  Events: {len(data_list)}")
        self.logger.info(f"  Shared tensors: {tensor_count}")
        self.logger.info(f"  Total size: {total_bytes / 1024**3:.2f} GB")

        return full_batch

    @staticmethod
    def unbatch_shared_data(shared_batch: Batch) -> List[Data]:
        """
        Unbatch a shared Batch object back into individual Data objects.

        Workers receive a single large batch for memory efficiency,
        but need individual Data objects for training.

        Args:
            shared_batch: Batch object with shared tensors from entire dataset

        Returns:
            List of individual Data objects (still sharing the underlying tensors)
        """
        # Convert Batch back to list of Data objects
        # The underlying tensors remain in shared memory
        return shared_batch.to_data_list()

    def clear_cache(self):
        """Clear the shared memory cache."""
        self.shared_cache.clear()
        self.logger.info("Shared memory cache cleared")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get current shared memory usage statistics.

        Returns:
            Dictionary with memory usage information in GB
        """
        stats = {"total_gb": 0, "num_datasets": len(self.shared_cache)}

        for key, (train_batch, valid_batch) in self.shared_cache.items():
            dataset_bytes = 0

            for batch in [train_batch, valid_batch]:
                for batch_key in batch.keys():
                    value = getattr(batch, batch_key)
                    if torch.is_tensor(value):
                        dataset_bytes += value.element_size() * value.nelement()

            stats[key] = dataset_bytes / 1024**3
            stats["total_gb"] += stats[key]

        return stats

    @staticmethod
    def validate_shared_memory():
        """
        Validate that shared memory works correctly with spawn method.

        Returns:
            bool: True if validation passes
        """
        def worker(rank, shared_tensor, result_queue):
            """Worker function to test shared memory access."""
            try:
                # Verify we can read the shared tensor
                assert shared_tensor[rank] == 0, f"Initial value should be 0, got {shared_tensor[rank]}"

                # Modify the shared tensor
                shared_tensor[rank] = rank + 1

                # Verify modification
                assert shared_tensor[rank] == rank + 1, f"Modified value should be {rank + 1}"

                result_queue.put((rank, True, "Success"))
            except Exception as e:
                result_queue.put((rank, False, str(e)))

        try:
            # Use spawn method (required for CUDA compatibility)
            mp.set_start_method('spawn', force=True)
        except RuntimeError:
            # Already set, that's fine
            pass

        # Create test tensor and move to shared memory
        test_tensor = torch.zeros(4)
        test_tensor.share_memory_()

        # Create queue for results
        result_queue = mp.Queue()

        # Spawn test workers
        processes = []
        for rank in range(4):
            p = mp.Process(target=worker, args=(rank, test_tensor, result_queue))
            p.start()
            processes.append(p)

        # Wait for workers
        for p in processes:
            p.join()

        # Check results
        success = True
        while not result_queue.empty():
            rank, passed, msg = result_queue.get()
            if not passed:
                logging.error(f"Worker {rank} failed: {msg}")
                success = False

        # Verify final tensor state
        expected = torch.tensor([1., 2., 3., 4.])
        if not torch.allclose(test_tensor, expected):
            logging.error(f"Final tensor incorrect. Expected {expected}, got {test_tensor}")
            success = False

        if success:
            logging.info("Shared memory validation passed! Spawn method works correctly.")

        return success


if __name__ == "__main__":
    # Test shared memory functionality
    logging.basicConfig(level=logging.INFO)

    manager = SharedDatasetManager()
    if manager.validate_shared_memory():
        print("✅ Shared memory is working correctly with spawn method!")
    else:
        print("❌ Shared memory validation failed!")