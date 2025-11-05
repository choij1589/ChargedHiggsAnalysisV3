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

    def __init__(self, dataset_root, separate_bjets=False):
        """
        Initialize the dynamic dataset loader.

        Args:
            dataset_root: Root directory containing samples/ subdirectory
            separate_bjets: Whether to use separate b-jets dataset directory
        """
        # Modify dataset root based on separate_bjets flag
        if separate_bjets:
            # Use dataset_bjets directory for separate b-jets datasets
            self.dataset_root = dataset_root.replace('/dataset/', '/dataset_bjets/')
            if not self.dataset_root.endswith('dataset_bjets'):
                self.dataset_root = dataset_root.rstrip('/') + '_bjets'
        else:
            self.dataset_root = dataset_root

        self.samples_dir = os.path.join(self.dataset_root, "samples")
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
            channel: Channel name (e.g., "Run1E2Mu", "Run3Mu", "Combined")
            fold: Fold number (0-4)
            pilot: Whether to load pilot datasets

        Returns:
            List of PyTorch Geometric Data objects
        """
        # Handle Combined channel by loading both Run1E2Mu and Run3Mu
        if channel == "Combined":
            data_1e2mu = self.load_sample_data(sample_name, sample_type, "Run1E2Mu", fold, pilot)
            data_3mu = self.load_sample_data(sample_name, sample_type, "Run3Mu", fold, pilot)
            combined_data = data_1e2mu + data_3mu
            if combined_data:
                logging.info(f"Combined channel: loaded {len(data_1e2mu)} Run1E2Mu + {len(data_3mu)} Run3Mu = {len(combined_data)} events")
            return combined_data

        if sample_type == "signal":
            base_dir = self.signals_dir
        else:
            base_dir = self.backgrounds_dir

        sample_dir = os.path.join(base_dir, sample_name)
        if pilot:
            sample_dir += "_pilot"
            logging.info(f"PILOT MODE: Loading from {sample_dir}")

        filename = f"{channel}_fold-{fold}.pt"
        filepath = os.path.join(sample_dir, filename)

        if not os.path.exists(filepath):
            logging.warning(f"Dataset file not found: {filepath}")
            if pilot:
                logging.error(f"PILOT DATASET MISSING: {filepath}")
                logging.error("Make sure pilot datasets exist by running: ./scripts/saveDatasets.sh --pilot")
            return []

        try:
            # Note: weights_only=False is required for PyTorch Geometric Data objects
            # These are trusted dataset files created by our own saveDataset.py
            dataset = torch.load(filepath, weights_only=False)
            return dataset.data_list if hasattr(dataset, 'data_list') else []
        except Exception as e:
            logging.error(f"Error loading {filepath}: {e}")
            return []

    def load_background_category_data(self, category, channel, fold, pilot=False):
        """
        Load all samples for a given background category.

        Args:
            category: Background category ("nonprompt", "diboson", "ttZ")
            channel: Channel name (e.g., "Run1E2Mu", "Run3Mu", "Combined")
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

    def create_multiclass_training_splits(self, signal_sample, background_samples, channel, fold,
                                         balance=True, pilot=False, random_state=42):
        """
        Create training/validation/test splits for multi-class classification using 5-fold scheme.

        Uses V2's proven 3/1/1 fold splitting:
        - Train: 3 folds ((fold+1)%5, (fold+2)%5, (fold+3)%5)
        - Valid: 1 fold ((fold+4)%5)
        - Test: 1 fold (fold)

        Args:
            signal_sample: Signal sample name (e.g., "TTToHcToWAToMuMu-MHc130MA100")
            background_samples: List of 3 background sample names
                               ["Skim_TriLep_TTLL_powheg", "Skim_TriLep_WZTo3LNu_amcatnlo", "Skim_TriLep_TTZToLLNuNu"]
            channel: Channel name (e.g., "Run1E2Mu", "Run3Mu")
            fold: Current test fold (0-4)
            balance: Whether to balance classes by normalizing sample weights
            pilot: Whether to use pilot datasets
            random_state: Random state for shuffling

        Returns:
            Tuple of (train_data, valid_data, test_data) lists with assigned labels:
            - Signal: label 0
            - Background samples: labels 1, 2, 3, ... respectively
        """
        if len(background_samples) < 1:
            raise ValueError("Multi-class training requires at least 1 background sample")

        logging.info(f"Creating {len(background_samples)+1}-class training splits (1 signal + {len(background_samples)} backgrounds)")

        # Define fold splits using V2's scheme
        train_folds = [(fold+1)%5, (fold+2)%5, (fold+3)%5]
        valid_folds = [(fold+4)%5]
        test_folds = [fold]

        logging.info(f"Creating multi-class splits for fold {fold}: train={train_folds}, valid={valid_folds}, test={test_folds}")

        def load_folds_for_sample(sample_name, sample_type, fold_list):
            """Load multiple folds for a sample and return combined data."""
            combined_data = []
            for f in fold_list:
                fold_data = self.load_sample_data(sample_name, sample_type, channel, f, pilot)
                combined_data.extend(fold_data)
                if fold_data:
                    logging.debug(f"Loaded {len(fold_data)} events from {sample_name} fold {f}")
            return combined_data

        # Load signal data for all splits
        signal_train = load_folds_for_sample(signal_sample, "signal", train_folds)
        signal_valid = load_folds_for_sample(signal_sample, "signal", valid_folds)
        signal_test = load_folds_for_sample(signal_sample, "signal", test_folds)

        # Assign signal labels (class 0)
        for data in signal_train + signal_valid + signal_test:
            data.y = torch.tensor(0, dtype=torch.long)

        # Load background data and assign labels
        bg_train, bg_valid, bg_test = [], [], []
        sample_weights = {}  # Track total weights per sample for normalization

        for bg_idx, bg_sample in enumerate(background_samples):
            bg_label = bg_idx + 1  # Background labels: 1, 2, 3, ...

            # Load background data for all splits
            bg_sample_train = load_folds_for_sample(bg_sample, "background", train_folds)
            bg_sample_valid = load_folds_for_sample(bg_sample, "background", valid_folds)
            bg_sample_test = load_folds_for_sample(bg_sample, "background", test_folds)

            # Assign labels
            for data in bg_sample_train + bg_sample_valid + bg_sample_test:
                data.y = torch.tensor(bg_label, dtype=torch.long)

            # Collect for weight normalization
            all_bg_data = bg_sample_train + bg_sample_valid + bg_sample_test
            total_weight = sum(data.weight.item() for data in all_bg_data)
            sample_weights[bg_sample] = total_weight

            bg_train.extend(bg_sample_train)
            bg_valid.extend(bg_sample_valid)
            bg_test.extend(bg_sample_test)

            logging.info(f"Loaded {bg_sample} (label {bg_label}): "
                        f"train={len(bg_sample_train)}, valid={len(bg_sample_valid)}, test={len(bg_sample_test)}")

        # Calculate signal weights for normalization
        all_signal_data = signal_train + signal_valid + signal_test
        signal_total_weight = sum(data.weight.item() for data in all_signal_data)
        sample_weights[signal_sample] = signal_total_weight

        # Apply weight normalization if requested
        if balance:
            # Normalize so each sample class has equal total weight
            max_weight = max(sample_weights.values())

            # Normalize signal weights
            signal_norm_factor = max_weight / signal_total_weight if signal_total_weight > 0 else 1.0
            for data in all_signal_data:
                data.weight = data.weight * signal_norm_factor

            # Normalize background weights
            for bg_idx, bg_sample in enumerate(background_samples):
                bg_norm_factor = max_weight / sample_weights[bg_sample] if sample_weights[bg_sample] > 0 else 1.0

                # Apply to corresponding background data
                for data in bg_train + bg_valid + bg_test:
                    if data.y.item() == bg_idx + 1:  # Match background label
                        data.weight = data.weight * bg_norm_factor

            logging.info(f"Applied weight normalization - signal: {signal_norm_factor:.4f}")
            for bg_idx, bg_sample in enumerate(background_samples):
                bg_norm_factor = max_weight / sample_weights[bg_sample] if sample_weights[bg_sample] > 0 else 1.0
                logging.info(f"  {bg_sample}: {bg_norm_factor:.4f}")

        # Combine and shuffle each split
        train_data = shuffle(signal_train + bg_train, random_state=random_state)
        valid_data = shuffle(signal_valid + bg_valid, random_state=random_state)
        test_data = shuffle(signal_test + bg_test, random_state=random_state)

        # Log final statistics
        def count_classes(data_list):
            counts = {}
            for data in data_list:
                label = data.y.item()
                counts[label] = counts.get(label, 0) + 1
            return counts

        train_counts = count_classes(train_data)
        valid_counts = count_classes(valid_data)
        test_counts = count_classes(test_data)

        logging.info(f"Multi-class splits created:")
        logging.info(f"  Train: {len(train_data)} events, class distribution: {train_counts}")
        logging.info(f"  Valid: {len(valid_data)} events, class distribution: {valid_counts}")
        logging.info(f"  Test: {len(test_data)} events, class distribution: {test_counts}")

        return train_data, valid_data, test_data

    def create_grouped_multiclass_training_splits(self, signal_sample, background_groups, channel, fold,
                                                 balance=True, pilot=False, random_state=42):
        """
        Create training splits for multi-class classification with grouped backgrounds.

        Groups multiple background samples under a single class label and applies proper
        weight normalization both within groups and between groups.

        Args:
            signal_sample: Signal sample name
            background_groups: Dict of {group_name: [list_of_samples]}
                             e.g., {'nonprompt': ['TTLL_powheg'], 'diboson': ['WZTo3LNu_amcatnlo', 'ZZTo4L_powheg']}
            channel: Channel name
            fold: Current test fold (0-4)
            balance: Whether to balance groups by normalizing sample weights
            pilot: Whether to use pilot datasets
            random_state: Random state for shuffling

        Returns:
            Tuple of (train_data, valid_data, test_data) lists with assigned labels:
            - Signal: label 0
            - Background groups: labels 1, 2, 3, ... respectively
        """
        if len(background_groups) < 1:
            raise ValueError("Multi-class training requires at least 1 background group")

        group_names = list(background_groups.keys())
        logging.info(f"Creating {len(group_names)+1}-class training splits (1 signal + {len(group_names)} background groups)")
        logging.info(f"Background groups: {group_names}")

        # Define fold splits using V2's scheme
        train_folds = [(fold+1)%5, (fold+2)%5, (fold+3)%5]
        valid_folds = [(fold+4)%5]
        test_folds = [fold]

        logging.info(f"Creating grouped multi-class splits for fold {fold}: train={train_folds}, valid={valid_folds}, test={test_folds}")

        def load_folds_for_sample(sample_name, sample_type, fold_list):
            """Load multiple folds for a sample and return combined data."""
            combined_data = []
            for f in fold_list:
                fold_data = self.load_sample_data(sample_name, sample_type, channel, f, pilot)
                combined_data.extend(fold_data)
                if fold_data:
                    logging.debug(f"Loaded {len(fold_data)} events from {sample_name} fold {f}")
            return combined_data

        # Load signal data for all splits
        signal_train = load_folds_for_sample(signal_sample, "signal", train_folds)
        signal_valid = load_folds_for_sample(signal_sample, "signal", valid_folds)
        signal_test = load_folds_for_sample(signal_sample, "signal", test_folds)

        # Assign signal labels (class 0)
        for data in signal_train + signal_valid + signal_test:
            data.y = torch.tensor(0, dtype=torch.long)

        # Load background data by groups
        bg_train, bg_valid, bg_test = [], [], []
        group_weights = {}  # Track total weights per group for normalization
        sample_weights = {}  # Track individual sample weights within groups

        for group_idx, (group_name, sample_list) in enumerate(background_groups.items()):
            group_label = group_idx + 1  # Background group labels: 1, 2, 3, ...

            group_train, group_valid, group_test = [], [], []
            group_sample_weights = {}

            # Load all samples in this group
            for sample_name in sample_list:
                full_sample_name = sample_name  # Assuming full names are passed

                # Load sample data for all splits
                sample_train = load_folds_for_sample(full_sample_name, "background", train_folds)
                sample_valid = load_folds_for_sample(full_sample_name, "background", valid_folds)
                sample_test = load_folds_for_sample(full_sample_name, "background", test_folds)

                # Assign group labels to all events from this sample
                for data in sample_train + sample_valid + sample_test:
                    data.y = torch.tensor(group_label, dtype=torch.long)

                # Calculate sample weight within group
                all_sample_data = sample_train + sample_valid + sample_test
                sample_total_weight = sum(data.weight.item() for data in all_sample_data)
                group_sample_weights[full_sample_name] = sample_total_weight

                group_train.extend(sample_train)
                group_valid.extend(sample_valid)
                group_test.extend(sample_test)

                logging.info(f"Loaded {full_sample_name} in group '{group_name}' (label {group_label}): "
                            f"train={len(sample_train)}, valid={len(sample_valid)}, test={len(sample_test)}, weight={sample_total_weight:.2f}")

            # Skip within-group normalization to preserve cross-section ratios
            # Each sample's weight already represents its proper cross-section contribution

            # Calculate total group weight after within-group normalization
            group_total_weight = sum(data.weight.item() for data in group_train + group_valid + group_test)
            group_weights[group_name] = group_total_weight
            sample_weights.update(group_sample_weights)

            bg_train.extend(group_train)
            bg_valid.extend(group_valid)
            bg_test.extend(group_test)

            logging.info(f"Group '{group_name}' total: train={len(group_train)}, valid={len(group_valid)}, test={len(group_test)}, weight={group_total_weight:.2f}")

        # Calculate signal weights for normalization
        all_signal_data = signal_train + signal_valid + signal_test
        signal_total_weight = sum(data.weight.item() for data in all_signal_data)
        group_weights['signal'] = signal_total_weight

        # Apply between-group weight normalization if requested
        if balance:
            # Normalize so each group has equal total weight
            max_group_weight = max(group_weights.values())

            # Normalize signal weights
            signal_norm_factor = max_group_weight / signal_total_weight if signal_total_weight > 0 else 1.0
            for data in all_signal_data:
                data.weight = data.weight * signal_norm_factor

            logging.info(f"Applied between-group weight normalization:")
            logging.info(f"  signal: factor {signal_norm_factor:.4f}")

            # Normalize background group weights
            for group_idx, (group_name, _) in enumerate(background_groups.items()):
                group_norm_factor = max_group_weight / group_weights[group_name] if group_weights[group_name] > 0 else 1.0
                group_label = group_idx + 1

                # Apply to corresponding background data
                for data in bg_train + bg_valid + bg_test:
                    if data.y.item() == group_label:
                        data.weight = data.weight * group_norm_factor

                logging.info(f"  {group_name}: factor {group_norm_factor:.4f}")

        # Combine and shuffle each split
        train_data = shuffle(signal_train + bg_train, random_state=random_state)
        valid_data = shuffle(signal_valid + bg_valid, random_state=random_state)
        test_data = shuffle(signal_test + bg_test, random_state=random_state)

        # Log final statistics
        def count_classes(data_list):
            counts = {}
            for data in data_list:
                label = data.y.item()
                counts[label] = counts.get(label, 0) + 1
            return counts

        train_counts = count_classes(train_data)
        valid_counts = count_classes(valid_data)
        test_counts = count_classes(test_data)

        logging.info(f"Grouped multi-class splits created:")
        logging.info(f"  Train: {len(train_data)} events, class distribution: {train_counts}")
        logging.info(f"  Valid: {len(valid_data)} events, class distribution: {valid_counts}")
        logging.info(f"  Test: {len(test_data)} events, class distribution: {test_counts}")

        return train_data, valid_data, test_data

    def load_multiclass_with_subsampling(self, signal_sample, background_groups, channel, fold_list,
                                         pilot=False, max_events_per_fold=None, balance_weights=True,
                                         random_state=42):
        """
        Load multi-class data with optional per-fold subsampling and weight normalization.

        This method combines loading, subsampling, and weight balancing in a single call.
        Used by trainMultiClassForGA.py and visualizeGAIteration.py.

        Args:
            signal_sample: Signal sample name (e.g., "TTToHcToWAToMuMu-MHc130_MA90")
            background_groups: Dict of {group_name: [list_of_samples]}
                             e.g., {'nonprompt': ['Skim_TriLep_TTLL_powheg'],
                                    'diboson': ['Skim_TriLep_WZTo3LNu_amcatnlo', 'Skim_TriLep_ZZTo4L_powheg']}
            channel: Channel name (e.g., "Run1E2Mu", "Run3Mu", "Combined")
            fold_list: List of fold numbers to load (e.g., [0, 1, 2])
            pilot: Whether to use pilot datasets
            max_events_per_fold: Maximum events per fold per class (None = no limit)
            balance_weights: Whether to normalize weights so each class has equal total weight
            random_state: Random state for shuffling and subsampling

        Returns:
            List of Data objects with:
            - Signal: label 0
            - Background groups: labels 1, 2, 3, ...
            - Weights normalized across classes (if balance_weights=True)

        Note:
            Weight rescaling is applied once after all folds are loaded, not per-fold.
            This simplifies the logic while achieving the same training objective.
        """
        from sklearn.utils import resample

        all_data = []

        # Load signal data with per-fold subsampling
        for fold in fold_list:
            signal_data = self.load_sample_data(signal_sample, "signal", channel, fold, pilot)
            for data in signal_data:
                data.y = torch.tensor(0, dtype=torch.long)  # Signal = class 0

            # Subsample if needed (weight rescaling deferred to final class balancing)
            original_count = len(signal_data)
            if max_events_per_fold and original_count > max_events_per_fold:
                # Randomly subsample to max_events_per_fold
                signal_data = resample(signal_data, n_samples=max_events_per_fold, replace=False, random_state=random_state)
                logging.info(f"Subsampled signal fold {fold}: {original_count} → {len(signal_data)} events")
            else:
                logging.info(f"Loaded {len(signal_data)} signal events from fold {fold}")

            all_data.extend(signal_data)

        # Load background groups with per-fold subsampling at group level
        for group_idx, (group_name, sample_list) in enumerate(background_groups.items()):
            group_label = group_idx + 1  # Background groups: 1, 2, 3, ...

            for fold in fold_list:
                # Load all samples in this group for this fold
                group_fold_data = []
                for sample_name in sample_list:
                    sample_data = self.load_sample_data(sample_name, "background", channel, fold, pilot)
                    group_fold_data.extend(sample_data)

                # Assign group labels
                for data in group_fold_data:
                    data.y = torch.tensor(group_label, dtype=torch.long)

                # Subsample at group level if needed (weight rescaling deferred to final class balancing)
                original_count = len(group_fold_data)
                if max_events_per_fold and original_count > max_events_per_fold:
                    # Randomly subsample to max_events_per_fold
                    group_fold_data = resample(group_fold_data, n_samples=max_events_per_fold, replace=False, random_state=random_state)
                    logging.info(f"Subsampled {group_name} fold {fold}: {original_count} → {len(group_fold_data)} events")
                else:
                    logging.info(f"Loaded {len(group_fold_data)} events from {group_name} (label {group_label}) fold {fold}")

                all_data.extend(group_fold_data)

        # Apply weight normalization to balance classes (if requested)
        # This makes each class have equal total weight for balanced training
        if balance_weights:
            # Calculate total weight per class across ALL folds
            class_weights = {}
            for data in all_data:
                label = data.y.item()
                class_weights[label] = class_weights.get(label, 0.0) + data.weight.item()

            # Normalize so each class has equal total weight
            if class_weights:
                max_class_weight = max(class_weights.values())
                logging.info(f"Applying weight normalization across classes:")

                for class_label in sorted(class_weights.keys()):
                    if class_weights[class_label] > 0:
                        norm_factor = max_class_weight / class_weights[class_label]

                        # Apply normalization to all events of this class
                        for data in all_data:
                            if data.y.item() == class_label:
                                data.weight = data.weight * norm_factor

                        # Log the normalization
                        if class_label == 0:
                            class_name = "signal"
                        else:
                            class_name = list(background_groups.keys())[class_label - 1]
                        logging.info(f"  Class {class_label} ({class_name}): factor {norm_factor:.4f}, weight {class_weights[class_label]:.2f} → {max_class_weight:.2f}")

        return shuffle(all_data, random_state=random_state)

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