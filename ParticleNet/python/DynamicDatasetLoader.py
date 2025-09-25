#!/usr/bin/env python
import os
import logging
import torch
from sklearn.utils import shuffle
from torch_geometric.data import Data, InMemoryDataset
import numpy as np

class DynamicDatasetLoader:
    """
    Dynamic dataset loader for MC sample-based ParticleNet datasets.
    Maps MC samples to background categories and loads/combines samples at training time.
    """

    def __init__(self, dataset_root):
        """
        Initialize the dynamic dataset loader.

        Args:
            dataset_root: Root directory containing samples/ subdirectory
        """
        self.dataset_root = dataset_root
        self.samples_dir = os.path.join(dataset_root, "samples")
        self.signals_dir = os.path.join(self.samples_dir, "signals")
        self.backgrounds_dir = os.path.join(self.samples_dir, "backgrounds")

        if not os.path.exists(self.samples_dir):
            raise ValueError(f"Samples directory not found: {self.samples_dir}")

        # Background category mapping: MC sample name -> physics category
        self.background_categories = {
            "Skim_TriLep_TTLL_powheg": "nonprompt",
            "Skim_TriLep_DYJets": "prompt",
            "Skim_TriLep_DYJets10to50": "prompt",
            "Skim_TriLep_WZTo3LNu_amcatnlo": "diboson",
            "Skim_TriLep_TTZToLLNuNu": "ttZ",
            "Skim_TriLep_TTWToLNu": "ttW",
            "Skim_TriLep_tZq": "rare_top",
            # Add more mappings as needed
        }

    def get_available_samples(self):
        """Get lists of available signal and background samples."""
        signals = []
        backgrounds = []

        if os.path.exists(self.signals_dir):
            signals = [d for d in os.listdir(self.signals_dir)
                      if os.path.isdir(os.path.join(self.signals_dir, d)) and not d.endswith('_pilot')]

        if os.path.exists(self.backgrounds_dir):
            backgrounds = [d for d in os.listdir(self.backgrounds_dir)
                          if os.path.isdir(os.path.join(self.backgrounds_dir, d)) and not d.endswith('_pilot')]

        return signals, backgrounds

    def get_background_categories(self):
        """Get available background categories."""
        return list(set(self.background_categories.values()))

    def get_samples_for_category(self, category):
        """Get MC samples for a given background category."""
        return [sample for sample, cat in self.background_categories.items() if cat == category]

    def load_sample_data(self, sample_name, sample_type, channel, fold, pilot=False):
        """
        Load data for a specific MC sample, channel, and fold.

        Args:
            sample_name: Name of the MC sample (e.g., "TTToHcToWAToMuMu-MHc130MA100")
            sample_type: "signal" or "background"
            channel: Channel name (e.g., "Run1E2Mu", "Run3Mu")
            fold: Fold number (0-4)
            pilot: Whether to load pilot datasets

        Returns:
            List of PyTorch Geometric Data objects
        """
        if sample_type == "signal":
            base_dir = self.signals_dir
        else:
            base_dir = self.backgrounds_dir

        sample_dir = os.path.join(base_dir, sample_name)
        if pilot:
            sample_dir += "_pilot"

        filename = f"{channel}_fold-{fold}.pt"
        filepath = os.path.join(sample_dir, filename)

        if not os.path.exists(filepath):
            logging.warning(f"Dataset file not found: {filepath}")
            return []

        try:
            dataset = torch.load(filepath)
            return dataset.data_list if hasattr(dataset, 'data_list') else []
        except Exception as e:
            logging.error(f"Error loading {filepath}: {e}")
            return []

    def load_background_category_data(self, category, channel, fold, pilot=False):
        """
        Load all samples for a given background category.

        Args:
            category: Background category ("nonprompt", "diboson", "ttZ")
            channel: Channel name
            fold: Fold number
            pilot: Whether to use pilot datasets

        Returns:
            List of Data objects from all samples in the category
        """
        category_data = []
        samples_in_category = self.get_samples_for_category(category)

        for sample_name in samples_in_category:
            sample_data = self.load_sample_data(sample_name, "background", channel, fold, pilot)
            category_data.extend(sample_data)
            if sample_data:
                logging.info(f"Loaded {len(sample_data)} events from {sample_name} for category {category}")

        return category_data

    def create_binary_dataset(self, signal_sample, background_category, channel, fold,
                            balance=True, pilot=False, random_state=42):
        """
        Create a binary classification dataset from signal sample and background category.

        Args:
            signal_sample: Signal sample name (e.g., "TTToHcToWAToMuMu-MHc130MA100")
            background_category: Background category ("nonprompt", "diboson", "ttZ")
            channel: Channel name
            fold: Fold number
            balance: Whether to balance classes
            pilot: Whether to use pilot datasets
            random_state: Random state for shuffling

        Returns:
            List of Data objects with assigned labels
        """
        # Load signal data
        signal_data = self.load_sample_data(signal_sample, "signal", channel, fold, pilot)

        # Load background category data
        background_data = self.load_background_category_data(background_category, channel, fold, pilot)

        if not signal_data or not background_data:
            logging.warning(f"No data loaded for {signal_sample} vs {background_category}, {channel}, fold {fold}")
            return []

        # Assign labels
        for data in signal_data:
            data.y = torch.tensor(0, dtype=torch.long)  # Signal = 0
        for data in background_data:
            data.y = torch.tensor(1, dtype=torch.long)  # Background = 1

        # Balance classes if requested
        if balance:
            min_size = min(len(signal_data), len(background_data))
            signal_data = shuffle(signal_data, random_state=random_state)[:min_size]
            background_data = shuffle(background_data, random_state=random_state)[:min_size]
            logging.info(f"Balanced to {min_size} samples per class")

        # Combine and shuffle
        combined_data = signal_data + background_data
        combined_data = shuffle(combined_data, random_state=random_state)

        logging.info(f"Created binary dataset: {len(signal_data)} signal + {len(background_data)} background = {len(combined_data)} total")
        return combined_data

    def create_multiclass_dataset(self, signal_sample, channel, fold,
                                 balance=True, pilot=False, random_state=42):
        """
        Create a multi-class classification dataset (signal vs all background categories).

        Args:
            signal_sample: Signal sample name
            channel: Channel name
            fold: Fold number
            balance: Whether to balance classes
            pilot: Whether to use pilot datasets
            random_state: Random state for shuffling

        Returns:
            List of Data objects with assigned labels
        """
        # Load signal data
        signal_data = self.load_sample_data(signal_sample, "signal", channel, fold, pilot)

        # Load all background categories
        background_data = {}
        available_categories = self.get_background_categories()

        for i, bg_category in enumerate(available_categories):
            bg_data = self.load_background_category_data(bg_category, channel, fold, pilot)
            if bg_data:
                # Assign multi-class labels
                for data in bg_data:
                    data.y = torch.tensor(i + 1, dtype=torch.long)  # Background labels 1, 2, 3, ...
                background_data[bg_category] = bg_data

        # Assign signal label
        for data in signal_data:
            data.y = torch.tensor(0, dtype=torch.long)  # Signal = 0

        # Combine all data
        all_data = signal_data[:]  # Copy to avoid modifying original
        for bg_data in background_data.values():
            all_data.extend(bg_data)

        # Balance classes if requested
        if balance:
            class_data = {0: signal_data}
            # Add background categories dynamically
            for i, bg_category in enumerate(available_categories):
                class_data[i + 1] = background_data.get(bg_category, [])

            # Find minimum class size
            class_sizes = {k: len(v) for k, v in class_data.items() if v}
            if class_sizes:
                min_size = min(class_sizes.values())

                # Sample equal number from each class
                balanced_data = []
                for class_label, data_list in class_data.items():
                    if data_list:
                        sampled = shuffle(data_list, random_state=random_state)[:min_size]
                        balanced_data.extend(sampled)
                        logging.info(f"Class {class_label}: {len(sampled)} samples")

                all_data = shuffle(balanced_data, random_state=random_state)
                logging.info(f"Balanced multiclass dataset: {len(all_data)} total samples")
        else:
            all_data = shuffle(all_data, random_state=random_state)
            class_counts = {}
            for data in all_data:
                label = data.y.item()
                class_counts[label] = class_counts.get(label, 0) + 1
            logging.info(f"Unbalanced multiclass dataset: {class_counts}")

        return all_data

    def create_training_dataset(self, config):
        """
        Create a training dataset based on configuration.

        Args:
            config: Dictionary with keys:
                - mode: "binary" or "multiclass"
                - signal: Signal sample name
                - background: Background category (for binary mode)
                - channel: Channel name
                - fold: Fold number
                - balance: Whether to balance classes (default: True)
                - pilot: Whether to use pilot datasets (default: False)
                - random_state: Random state (default: 42)

        Returns:
            GraphDataset object ready for training
        """
        mode = config.get("mode", "binary")
        signal = config["signal"]
        channel = config["channel"]
        fold = config["fold"]
        balance = config.get("balance", True)
        pilot = config.get("pilot", False)
        random_state = config.get("random_state", 42)

        if mode == "binary":
            if "background" not in config:
                raise ValueError("Background category must be specified for binary mode")
            data_list = self.create_binary_dataset(
                signal, config["background"], channel, fold,
                balance, pilot, random_state
            )
        elif mode == "multiclass":
            data_list = self.create_multiclass_dataset(
                signal, channel, fold, balance, pilot, random_state
            )
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'binary' or 'multiclass'")

        if not data_list:
            raise ValueError("No data loaded - check sample names and file availability")

        # Create and return GraphDataset
        from Preprocess import GraphDataset
        return GraphDataset(data_list)

# Example usage
if __name__ == "__main__":
    # Example usage of the dynamic dataset loader
    WORKDIR = os.environ.get("WORKDIR", "/path/to/workdir")
    dataset_root = f"{WORKDIR}/ParticleNet/dataset"

    loader = DynamicDatasetLoader(dataset_root)

    # Check available samples and categories
    signals, backgrounds = loader.get_available_samples()
    categories = loader.get_background_categories()

    print(f"Available signal samples: {signals}")
    print(f"Available background samples: {backgrounds}")
    print(f"Available background categories: {categories}")

    # Show sample-to-category mapping
    print(f"\nBackground sample mappings:")
    for category in categories:
        samples = loader.get_samples_for_category(category)
        print(f"  {category}: {samples}")

    # Create binary dataset
    if signals and categories:
        config = {
            "mode": "binary",
            "signal": signals[0],
            "background": categories[0],  # Use category, not sample name
            "channel": "Run1E2Mu",
            "fold": 0,
            "balance": True
        }

        try:
            dataset = loader.create_training_dataset(config)
            print(f"\nCreated binary dataset with {len(dataset)} samples")
        except Exception as e:
            print(f"Error creating dataset: {e}")