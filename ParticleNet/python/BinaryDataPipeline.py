#!/usr/bin/env python
"""
Binary data pipeline for ParticleNet signal vs. single background training.

Handles dataset creation and validation for binary classification
with integration to the existing DynamicDatasetLoader.
"""

import logging
from typing import Tuple, List, Dict, Any

import torch
from torch_geometric.loader import DataLoader

from DynamicDatasetLoader import DynamicDatasetLoader
from Preprocess import GraphDataset
from BinaryTrainingConfig import BinaryTrainingConfig


class BinaryDataPipeline:
    """
    Manages binary data loading pipeline for signal vs. single background training.

    Integrates with DynamicDatasetLoader to create datasets for binary classification,
    with comprehensive validation and statistics.
    """

    def __init__(self, config: BinaryTrainingConfig):
        """
        Initialize binary data pipeline.

        Args:
            config: BinaryTrainingConfig instance with parsed arguments
        """
        self.config = config
        dataset_dir = "dataset_bjets" if config.args.separate_bjets else "dataset"
        self.dataset_root = f"{config.workdir}/ParticleNet/{dataset_dir}"
        self.loader = DynamicDatasetLoader(self.dataset_root, separate_bjets=config.args.separate_bjets)

        # Dataset storage
        self.train_data = None
        self.valid_data = None
        self.test_data = None

        # DataLoader storage
        self.train_loader = None
        self.valid_loader = None
        self.test_loader = None

    def create_datasets(self) -> Tuple[List[Any], List[Any], List[Any]]:
        """
        Create training, validation, and test datasets for binary classification.

        Returns:
            Tuple of (train_data, valid_data, test_data) lists

        Raises:
            Exception: If dataset creation fails
            ValueError: If empty datasets are created
        """
        logging.info(f"Creating binary training splits for signal: {self.config.signal_full_name}")
        logging.info(f"Background category: {self.config.background_category}")
        logging.info(f"Channel: {self.config.args.channel}, Fold: {self.config.args.fold}")

        try:
            # Define fold assignments for proper train/valid/test splitting
            # For K-fold cross-validation (K=5):
            # - Training uses 3 folds
            # - Validation uses 1 fold
            # - Test uses 1 fold (the current fold parameter)
            train_folds = [(self.config.args.fold+1)%5, (self.config.args.fold+2)%5, (self.config.args.fold+3)%5]
            valid_fold = (self.config.args.fold+4)%5
            test_fold = self.config.args.fold

            logging.info(f"Fold assignment - Train: {train_folds}, Valid: {valid_fold}, Test: {test_fold}")

            # Create training set by combining multiple folds
            self.train_data = []
            for fold_idx in train_folds:
                fold_data = self.loader.create_binary_dataset(
                    signal_sample=self.config.signal_full_name,
                    background_category=self.config.background_category,
                    channel=self.config.args.channel,
                    fold=fold_idx,
                    balance=self.config.args.balance,
                    pilot=self.config.args.pilot
                )
                self.train_data.extend(fold_data)
                logging.debug(f"Loaded {len(fold_data)} events for train fold {fold_idx}")

            # Create validation set
            self.valid_data = self.loader.create_binary_dataset(
                signal_sample=self.config.signal_full_name,
                background_category=self.config.background_category,
                channel=self.config.args.channel,
                fold=valid_fold,
                balance=self.config.args.balance,
                pilot=self.config.args.pilot
            )

            # Create test set
            self.test_data = self.loader.create_binary_dataset(
                signal_sample=self.config.signal_full_name,
                background_category=self.config.background_category,
                channel=self.config.args.channel,
                fold=test_fold,
                balance=self.config.args.balance,
                pilot=self.config.args.pilot
            )

        except Exception as e:
            logging.error(f"Failed to create binary datasets: {e}")
            raise

        # Validate datasets
        if not self.train_data or not self.valid_data or not self.test_data:
            raise ValueError("Empty binary datasets created - check sample availability")

        logging.info(f"Binary dataset sizes - Train: {len(self.train_data)}, "
                    f"Valid: {len(self.valid_data)}, Test: {len(self.test_data)}")

        return self.train_data, self.valid_data, self.test_data

    def create_data_loaders(self, batch_size: int = 1024) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch Geometric data loaders from binary datasets.

        Args:
            batch_size: Batch size for data loaders

        Returns:
            Tuple of (train_loader, valid_loader, test_loader)

        Raises:
            ValueError: If datasets haven't been created yet
        """
        if self.train_data is None or self.valid_data is None or self.test_data is None:
            raise ValueError("Datasets must be created before creating data loaders. Call create_datasets() first.")

        # Create graph datasets
        train_dataset = GraphDataset(self.train_data)
        valid_dataset = GraphDataset(self.valid_data)
        test_dataset = GraphDataset(self.test_data)

        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset, batch_size=batch_size, pin_memory=True, shuffle=True
        )
        self.valid_loader = DataLoader(
            valid_dataset, batch_size=batch_size, pin_memory=True, shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=batch_size, pin_memory=True, shuffle=False
        )

        logging.info(f"Created binary data loaders with batch size {batch_size}")
        logging.info(f"DataLoader sizes - Train: {len(train_dataset)}, "
                    f"Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

        return self.train_loader, self.valid_loader, self.test_loader

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive binary dataset statistics.

        Returns:
            Dictionary containing dataset statistics

        Raises:
            ValueError: If datasets haven't been created yet
        """
        if self.train_data is None or self.valid_data is None or self.test_data is None:
            raise ValueError("Datasets must be created before getting statistics. Call create_datasets() first.")

        # Count class distributions
        def count_classes(data_list):
            counts = {0: 0, 1: 0}  # Signal: 0, Background: 1
            for data in data_list:
                label = data.y.item()
                counts[label] = counts.get(label, 0) + 1
            return counts

        train_counts = count_classes(self.train_data)
        valid_counts = count_classes(self.valid_data)
        test_counts = count_classes(self.test_data)

        stats = {
            'train_size': len(self.train_data),
            'valid_size': len(self.valid_data),
            'test_size': len(self.test_data),
            'total_size': len(self.train_data) + len(self.valid_data) + len(self.test_data),
            'signal_sample': self.config.signal_full_name,
            'background_category': self.config.background_category,
            'channel': self.config.args.channel,
            'fold': self.config.args.fold,
            'num_classes': self.config.num_classes,
            'pilot': self.config.args.pilot,
            'balance': self.config.args.balance,
            'train_distribution': train_counts,
            'valid_distribution': valid_counts,
            'test_distribution': test_counts
        }

        return stats

    def log_dataset_info(self) -> None:
        """Log comprehensive binary dataset information."""
        stats = self.get_dataset_statistics()

        logging.info("=" * 50)
        logging.info("BINARY DATASET INFORMATION")
        logging.info("=" * 50)
        logging.info(f"Signal: {stats['signal_sample']}")
        logging.info(f"Background: {stats['background_category']}")
        logging.info(f"Channel: {stats['channel']}, Fold: {stats['fold']}")
        logging.info(f"Total events: {stats['total_size']:,}")
        logging.info(f"  Train: {stats['train_size']:,} ({stats['train_size']/stats['total_size']*100:.1f}%)")
        logging.info(f"  Valid: {stats['valid_size']:,} ({stats['valid_size']/stats['total_size']*100:.1f}%)")
        logging.info(f"  Test:  {stats['test_size']:,} ({stats['test_size']/stats['total_size']*100:.1f}%)")

        logging.info(f"Class distributions:")
        logging.info(f"  Train - Signal: {stats['train_distribution'][0]:,}, Background: {stats['train_distribution'][1]:,}")
        logging.info(f"  Valid - Signal: {stats['valid_distribution'][0]:,}, Background: {stats['valid_distribution'][1]:,}")
        logging.info(f"  Test  - Signal: {stats['test_distribution'][0]:,}, Background: {stats['test_distribution'][1]:,}")

        logging.info(f"Binary classification: {stats['num_classes']} classes")
        logging.info(f"Pilot mode: {stats['pilot']}, Balance: {stats['balance']}")
        logging.info("=" * 50)

    def validate_data_integrity(self) -> bool:
        """
        Validate binary data integrity and consistency.

        Returns:
            True if data passes all validation checks

        Raises:
            ValueError: If validation fails
        """
        if not self.train_loader or not self.valid_loader or not self.test_loader:
            raise ValueError("Data loaders must be created before validation. Call create_data_loaders() first.")

        logging.info("Validating binary data integrity...")

        # Check for empty loaders
        if len(self.train_loader) == 0:
            raise ValueError("Train loader is empty")
        if len(self.valid_loader) == 0:
            raise ValueError("Valid loader is empty")
        if len(self.test_loader) == 0:
            raise ValueError("Test loader is empty")

        # Sample one batch from each loader to check structure
        try:
            train_batch = next(iter(self.train_loader))
            valid_batch = next(iter(self.valid_loader))
            test_batch = next(iter(self.test_loader))

            # Check required attributes
            for batch_name, batch in [("Train", train_batch), ("Valid", valid_batch), ("Test", test_batch)]:
                if not hasattr(batch, 'x'):
                    raise ValueError(f"{batch_name} batch missing node features (x)")
                if not hasattr(batch, 'edge_index'):
                    raise ValueError(f"{batch_name} batch missing edge indices")
                if not hasattr(batch, 'graphInput'):
                    raise ValueError(f"{batch_name} batch missing graph features")
                if not hasattr(batch, 'y'):
                    raise ValueError(f"{batch_name} batch missing labels (y)")
                if not hasattr(batch, 'batch'):
                    raise ValueError(f"{batch_name} batch missing batch indices")

                # Check binary label range (0 for signal, 1 for background)
                if batch.y.min() < 0 or batch.y.max() >= 2:
                    raise ValueError(f"{batch_name} batch has invalid binary label range. "
                                   f"Expected [0, 1], got [{batch.y.min().item()}, {batch.y.max().item()}]")

            logging.info("Binary data integrity validation passed")
            return True

        except Exception as e:
            logging.error(f"Binary data integrity validation failed: {e}")
            raise

    def get_sample_batch_info(self) -> Dict[str, Any]:
        """
        Get information about a sample binary batch for debugging.

        Returns:
            Dictionary with batch structure information
        """
        if not self.train_loader:
            raise ValueError("Train loader must be created first")

        batch = next(iter(self.train_loader))
        info = {
            'batch_size': batch.y.size(0),
            'num_nodes': batch.x.size(0),
            'node_features': batch.x.size(1),
            'num_edges': batch.edge_index.size(1),
            'graph_features': batch.graphInput.size(1),
            'num_classes': 2,  # Binary classification
            'signal_count': (batch.y == 0).sum().item(),
            'background_count': (batch.y == 1).sum().item()
        }

        # Check for weights
        info['has_weights'] = hasattr(batch, 'weight')
        if info['has_weights']:
            info['weight_range'] = [batch.weight.min().item(), batch.weight.max().item()]

        return info


def create_binary_data_pipeline(config: BinaryTrainingConfig) -> BinaryDataPipeline:
    """
    Factory function to create and initialize binary data pipeline.

    Args:
        config: BinaryTrainingConfig instance

    Returns:
        Initialized BinaryDataPipeline instance
    """
    pipeline = BinaryDataPipeline(config)
    return pipeline