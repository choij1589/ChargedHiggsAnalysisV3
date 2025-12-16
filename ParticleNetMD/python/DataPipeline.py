#!/usr/bin/env python
"""
Data pipeline management for Mass-Decorrelated ParticleNet training.

Handles dataset creation, validation, and DataLoader configuration
for both grouped and individual background training modes.

Modified from ParticleNet to use ParticleNetMD dataset paths.
"""

import logging
from typing import Tuple, List, Dict, Any

import torch
from torch_geometric.loader import DataLoader

from DynamicDatasetLoader import DynamicDatasetLoader
from Preprocess import GraphDataset


class DataPipeline:
    """
    Manages data loading pipeline for multi-class ParticleNet training.

    Handles both grouped background and individual background modes,
    with comprehensive dataset validation and loader configuration.
    """

    def __init__(self, config):
        """
        Initialize data pipeline with training configuration.

        Args:
            config: Configuration object with args namespace and sample information
        """
        self.config = config
        # ParticleNetMD always uses "dataset" directory
        self.dataset_root = f"{config.workdir}/ParticleNetMD/dataset"
        self.loader = DynamicDatasetLoader(self.dataset_root)

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
        Create training, validation, and test datasets.

        Returns:
            Tuple of (train_data, valid_data, test_data) lists

        Raises:
            Exception: If dataset creation fails
            ValueError: If empty datasets are created
        """
        logging.info(f"Creating multi-class training splits for signal: {self.config.signal_full_name}")
        logging.info(f"Channel: {self.config.args.channel}, Fold: {self.config.args.fold}")

        # Check if max_events_per_fold_per_class is specified
        max_events = self.config.args.max_events_per_fold_per_class
        use_subsampling = (max_events is not None)

        try:
            if self.config.use_groups:
                # Create grouped background training splits
                background_groups_full = self.config.get_background_groups_full()

                logging.info(f"Using grouped backgrounds: {len(background_groups_full)} groups")
                for group_name, samples in background_groups_full.items():
                    logging.info(f"  {group_name}: {[s.replace(self.config.args.background_prefix, '') for s in samples]}")

                if use_subsampling:
                    logging.info(f"Using subsampling with max_events_per_fold_per_class={max_events}")
                    # Use subsampling path (same as GA optimization)
                    # Determine fold indices based on standard 5-fold scheme
                    train_folds = self.config.args.train_folds
                    valid_folds = self.config.args.valid_folds
                    test_folds = self.config.args.test_folds

                    # Load training data with subsampling
                    self.train_data = self.loader.load_multiclass_with_subsampling(
                        signal_sample=self.config.signal_full_name,
                        background_groups=background_groups_full,
                        channel=self.config.args.channel,
                        fold_list=train_folds,
                        pilot=self.config.args.pilot,
                        max_events_per_fold=max_events,
                        balance_weights=self.config.args.balance,
                        random_state=42
                    )

                    # Load validation data with subsampling
                    self.valid_data = self.loader.load_multiclass_with_subsampling(
                        signal_sample=self.config.signal_full_name,
                        background_groups=background_groups_full,
                        channel=self.config.args.channel,
                        fold_list=valid_folds,
                        pilot=self.config.args.pilot,
                        max_events_per_fold=max_events,
                        balance_weights=self.config.args.balance,
                        random_state=42
                    )

                    # Load test data with subsampling
                    self.test_data = self.loader.load_multiclass_with_subsampling(
                        signal_sample=self.config.signal_full_name,
                        background_groups=background_groups_full,
                        channel=self.config.args.channel,
                        fold_list=test_folds,
                        pilot=self.config.args.pilot,
                        max_events_per_fold=max_events,
                        balance_weights=self.config.args.balance,
                        random_state=42
                    )
                else:
                    # Use original path without subsampling (backward compatibility)
                    self.train_data, self.valid_data, self.test_data = self.loader.create_grouped_multiclass_training_splits(
                        signal_sample=self.config.signal_full_name,
                        background_groups=background_groups_full,
                        channel=self.config.args.channel,
                        fold=self.config.args.fold,
                        balance=self.config.args.balance,
                        pilot=self.config.args.pilot
                    )
            else:
                # Use individual background training splits (backward compatibility)
                logging.info(f"Using individual backgrounds: {len(self.config.background_full_names)} samples")
                logging.info(f"  Samples: {[s.replace(self.config.args.background_prefix, '') for s in self.config.background_full_names]}")

                if use_subsampling:
                    logging.warning("Subsampling is only supported with grouped backgrounds mode. Falling back to standard loading.")

                self.train_data, self.valid_data, self.test_data = self.loader.create_multiclass_training_splits(
                    signal_sample=self.config.signal_full_name,
                    background_samples=self.config.background_full_names,
                    channel=self.config.args.channel,
                    fold=self.config.args.fold,
                    balance=self.config.args.balance,
                    pilot=self.config.args.pilot
                )

        except Exception as e:
            logging.error(f"Failed to create datasets: {e}")
            raise

        # Validate datasets
        if not self.train_data or not self.valid_data or not self.test_data:
            raise ValueError("Empty datasets created - check sample availability")

        logging.info(f"Dataset sizes - Train: {len(self.train_data)}, "
                    f"Valid: {len(self.valid_data)}, Test: {len(self.test_data)}")

        return self.train_data, self.valid_data, self.test_data

    def create_data_loaders(self, batch_size: int = 1024) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch Geometric data loaders from datasets.

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

        logging.info(f"Created data loaders with batch size {batch_size}")
        logging.info(f"DataLoader sizes - Train: {len(train_dataset)}, "
                    f"Valid: {len(valid_dataset)}, Test: {len(test_dataset)}")

        return self.train_loader, self.valid_loader, self.test_loader

    def get_dataset_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive dataset statistics for logging and validation.

        Returns:
            Dictionary containing dataset statistics

        Raises:
            ValueError: If datasets haven't been created yet
        """
        if self.train_data is None or self.valid_data is None or self.test_data is None:
            raise ValueError("Datasets must be created before getting statistics. Call create_datasets() first.")

        stats = {
            'train_size': len(self.train_data),
            'valid_size': len(self.valid_data),
            'test_size': len(self.test_data),
            'total_size': len(self.train_data) + len(self.valid_data) + len(self.test_data),
            'signal_sample': self.config.signal_full_name,
            'channel': self.config.args.channel,
            'fold': self.config.args.fold,
            'use_groups': self.config.use_groups,
            'num_classes': self.config.num_classes,
            'pilot': self.config.args.pilot,
            'balance': self.config.args.balance
        }

        if self.config.use_groups:
            stats['background_groups'] = self.config.background_groups
            stats['num_groups'] = len(self.config.background_groups)
        else:
            stats['background_samples'] = self.config.backgrounds_list
            stats['num_backgrounds'] = len(self.config.backgrounds_list)

        return stats

    def log_dataset_info(self) -> None:
        """Log comprehensive dataset information for debugging and monitoring."""
        stats = self.get_dataset_statistics()

        logging.info("=" * 50)
        logging.info("DATASET INFORMATION")
        logging.info("=" * 50)
        logging.info(f"Signal: {stats['signal_sample']}")
        logging.info(f"Channel: {stats['channel']}, Fold: {stats['fold']}")
        logging.info(f"Total events: {stats['total_size']:,}")
        logging.info(f"  Train: {stats['train_size']:,} ({stats['train_size']/stats['total_size']*100:.1f}%)")
        logging.info(f"  Valid: {stats['valid_size']:,} ({stats['valid_size']/stats['total_size']*100:.1f}%)")
        logging.info(f"  Test:  {stats['test_size']:,} ({stats['test_size']/stats['total_size']*100:.1f}%)")

        if stats['use_groups']:
            logging.info(f"Background mode: GROUPED ({stats['num_groups']} groups, {stats['num_classes']} classes)")
            for group_name, samples in stats['background_groups'].items():
                logging.info(f"  {group_name}: {samples}")
        else:
            logging.info(f"Background mode: INDIVIDUAL ({stats['num_backgrounds']} samples, {stats['num_classes']} classes)")
            logging.info(f"  Samples: {stats['background_samples']}")

        logging.info(f"Pilot mode: {stats['pilot']}, Balance: {stats['balance']}")
        logging.info("=" * 50)

    def validate_data_integrity(self) -> bool:
        """
        Validate data integrity and consistency.

        Returns:
            True if data passes all validation checks

        Raises:
            ValueError: If validation fails
        """
        if not self.train_loader or not self.valid_loader or not self.test_loader:
            raise ValueError("Data loaders must be created before validation. Call create_data_loaders() first.")

        logging.info("Validating data integrity...")

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

                # Check label range
                if batch.y.min() < 0 or batch.y.max() >= self.config.num_classes:
                    raise ValueError(f"{batch_name} batch has invalid label range. "
                                   f"Expected [0, {self.config.num_classes-1}], "
                                   f"got [{batch.y.min().item()}, {batch.y.max().item()}]")

            logging.info("Data integrity validation passed")
            return True

        except Exception as e:
            logging.error(f"Data integrity validation failed: {e}")
            raise

    def get_sample_batch_info(self) -> Dict[str, Any]:
        """
        Get information about a sample batch for debugging.

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
            'num_classes': self.config.num_classes,
            'label_distribution': {}
        }

        # Calculate label distribution
        for class_idx in range(self.config.num_classes):
            count = (batch.y == class_idx).sum().item()
            info['label_distribution'][class_idx] = count

        # Check for weights
        info['has_weights'] = hasattr(batch, 'weight')
        if info['has_weights']:
            info['weight_range'] = [batch.weight.min().item(), batch.weight.max().item()]

        return info


def create_data_pipeline(config) -> DataPipeline:
    """
    Factory function to create and initialize data pipeline.

    Args:
        config: Configuration object with args namespace and sample information

    Returns:
        Initialized DataPipeline instance
    """
    pipeline = DataPipeline(config)
    return pipeline