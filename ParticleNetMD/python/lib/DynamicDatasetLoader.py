#!/usr/bin/env python
import gc
import os
import pickle
import sys
import time
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import torch
from sklearn.utils import shuffle
from sklearn.utils import resample
import numpy as np


# ---------------------------------------------------------------------------
# Process-pool helpers for parallel (class, fold) loading.
# Unpickling lists of PyG Data objects holds the GIL, so threads do not help;
# each (class, fold) task runs in its own spawned process instead.
# ---------------------------------------------------------------------------
_WORKER_LOG = []


class _ListLogHandler(logging.Handler):
    """Collect a worker's log records so the parent can re-emit them in task order."""

    def emit(self, record):
        _WORKER_LOG.append((record.levelno, record.getMessage()))


def _init_load_worker(parent_sys_path):
    """Pool initializer: same import path as the parent, logs captured, one torch thread."""
    sys.path[:] = parent_sys_path
    root = logging.getLogger()
    root.handlers[:] = [_ListLogHandler()]
    root.setLevel(logging.INFO)
    torch.set_num_threads(1)


def _run_class_fold_task(dataset_root, task):
    """Pool entry point: run one (class, fold) task, return (pickled data_list, log_records).

    The list is returned as plain pickle bytes: multiprocessing's default torch reducer
    would put every small tensor into its own shared-memory segment (~10^5 per task),
    which exhausts the mmap limit.
    """
    _WORKER_LOG.clear()
    # The cyclic GC rescans the millions of freshly unpickled objects many times
    # (~1.5x slower load); none of them form cycles, and this process exits afterwards.
    gc.disable()
    try:
        data = DynamicDatasetLoader(dataset_root)._load_class_fold(**task)
        blob = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        gc.enable()
    return blob, list(_WORKER_LOG)


def _available_cpus():
    """CPUs this process may run on (respects SLURM/cgroup affinity)."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


class DynamicDatasetLoader:
    """
    Dynamic dataset loader for MC sample-based ParticleNet datasets.
    Maps MC samples to background categories and loads/combines samples at training time.
    """

    def __init__(self, dataset_root):
        self.dataset_root = dataset_root
        self.samples_dir = os.path.join(self.dataset_root, "samples")
        self.signals_dir = os.path.join(self.samples_dir, "signals")
        self.backgrounds_dir = os.path.join(self.samples_dir, "backgrounds")

        if not os.path.exists(self.samples_dir):
            raise ValueError(f"Samples directory not found: {self.samples_dir}")

    def get_available_samples(self):
        """Get lists of available signal and background samples."""
        signals = []
        backgrounds = []

        if os.path.exists(self.signals_dir):
            signals = [d for d in os.listdir(self.signals_dir)
                      if os.path.isdir(os.path.join(self.signals_dir, d))]

        if os.path.exists(self.backgrounds_dir):
            backgrounds = [d for d in os.listdir(self.backgrounds_dir)
                          if os.path.isdir(os.path.join(self.backgrounds_dir, d))]

        return signals, backgrounds

    def load_sample_data(self, sample_name, sample_type, channel, fold):
        """
        Load data for a specific MC sample, channel, and fold.

        Args:
            sample_name: Name of the MC sample (e.g., "TTToHcToWAToMuMu-MHc130MA100")
            sample_type: "signal" or "background"
            channel: Channel name (e.g., "Run1E2Mu", "Run3Mu", "Combined")
            fold: Fold number (0-4)

        Returns:
            List of PyTorch Geometric Data objects
        """
        # Handle Combined channel by loading both Run1E2Mu and Run3Mu
        if channel == "Combined":
            data_1e2mu = self.load_sample_data(sample_name, sample_type, "Run1E2Mu", fold)
            data_3mu = self.load_sample_data(sample_name, sample_type, "Run3Mu", fold)
            combined_data = data_1e2mu + data_3mu
            if combined_data:
                logging.info(f"Combined channel [{sample_name}]: loaded {len(data_1e2mu)} Run1E2Mu + {len(data_3mu)} Run3Mu = {len(combined_data)} events")
            return combined_data

        if sample_type == "signal":
            base_dir = self.signals_dir
        else:
            base_dir = self.backgrounds_dir

        sample_dir = os.path.join(base_dir, sample_name)
        filename = f"{channel}_fold-{fold}.pt"
        filepath = os.path.join(sample_dir, filename)

        if not os.path.exists(filepath):
            logging.warning(f"Dataset file not found: {filepath}")
            return []

        try:
            # Note: weights_only=False is required for PyTorch Geometric Data objects
            # These are trusted dataset files created by our own saveDataset.py
            dataset = torch.load(filepath, weights_only=False)
            return dataset.data_list if hasattr(dataset, 'data_list') else []
        except Exception as e:
            logging.error(f"Error loading {filepath}: {e}")
            return []

    def create_multiclass_training_splits(self, signal_sample, background_samples, channel, fold,
                                         balance=True, random_state=42):
        """
        Create training/validation/test splits for multi-class classification using 5-fold scheme.

        Uses V2's proven 3/1/1 fold splitting:
        - Train: 3 folds ((fold+1)%5, (fold+2)%5, (fold+3)%5)
        - Valid: 1 fold ((fold+4)%5)
        - Test: 1 fold (fold)

        Args:
            signal_sample: Signal sample name (e.g., "TTToHcToWAToMuMu-MHc130MA100")
            background_samples: List of background sample names
            channel: Channel name (e.g., "Run1E2Mu", "Run3Mu")
            fold: Current test fold (0-4)
            balance: Whether to balance classes by normalizing sample weights
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
            combined_data = []
            for f in fold_list:
                fold_data = self.load_sample_data(sample_name, sample_type, channel, f)
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

            bg_sample_train = load_folds_for_sample(bg_sample, "background", train_folds)
            bg_sample_valid = load_folds_for_sample(bg_sample, "background", valid_folds)
            bg_sample_test = load_folds_for_sample(bg_sample, "background", test_folds)

            for data in bg_sample_train + bg_sample_valid + bg_sample_test:
                data.y = torch.tensor(bg_label, dtype=torch.long)

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
            max_weight = max(sample_weights.values())

            signal_norm_factor = max_weight / signal_total_weight if signal_total_weight > 0 else 1.0
            for data in all_signal_data:
                data.weight = data.weight * signal_norm_factor

            for bg_idx, bg_sample in enumerate(background_samples):
                bg_norm_factor = max_weight / sample_weights[bg_sample] if sample_weights[bg_sample] > 0 else 1.0
                for data in bg_train + bg_valid + bg_test:
                    if data.y.item() == bg_idx + 1:
                        data.weight = data.weight * bg_norm_factor

            logging.info(f"Applied weight normalization - signal: {signal_norm_factor:.4f}")
            for bg_idx, bg_sample in enumerate(background_samples):
                bg_norm_factor = max_weight / sample_weights[bg_sample] if sample_weights[bg_sample] > 0 else 1.0
                logging.info(f"  {bg_sample}: {bg_norm_factor:.4f}")

        # Combine and shuffle each split
        train_data = shuffle(signal_train + bg_train, random_state=random_state)
        valid_data = shuffle(signal_valid + bg_valid, random_state=random_state)
        test_data = shuffle(signal_test + bg_test, random_state=random_state)

        def count_classes(data_list):
            counts = {}
            for data in data_list:
                label = data.y.item()
                counts[label] = counts.get(label, 0) + 1
            return counts

        logging.info(f"Multi-class splits created:")
        logging.info(f"  Train: {len(train_data)} events, class distribution: {count_classes(train_data)}")
        logging.info(f"  Valid: {len(valid_data)} events, class distribution: {count_classes(valid_data)}")
        logging.info(f"  Test: {len(test_data)} events, class distribution: {count_classes(test_data)}")

        return train_data, valid_data, test_data

    def create_grouped_multiclass_training_splits(self, signal_sample, background_groups, channel, fold,
                                                 balance=True, random_state=42):
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

        train_folds = [(fold+1)%5, (fold+2)%5, (fold+3)%5]
        valid_folds = [(fold+4)%5]
        test_folds = [fold]

        logging.info(f"Creating grouped multi-class splits for fold {fold}: train={train_folds}, valid={valid_folds}, test={test_folds}")

        def load_folds_for_sample(sample_name, sample_type, fold_list):
            combined_data = []
            for f in fold_list:
                fold_data = self.load_sample_data(sample_name, sample_type, channel, f)
                combined_data.extend(fold_data)
                if fold_data:
                    logging.debug(f"Loaded {len(fold_data)} events from {sample_name} fold {f}")
            return combined_data

        # Load signal data for all splits
        signal_train = load_folds_for_sample(signal_sample, "signal", train_folds)
        signal_valid = load_folds_for_sample(signal_sample, "signal", valid_folds)
        signal_test = load_folds_for_sample(signal_sample, "signal", test_folds)

        for data in signal_train + signal_valid + signal_test:
            data.y = torch.tensor(0, dtype=torch.long)

        bg_train, bg_valid, bg_test = [], [], []
        group_weights = {}
        sample_weights = {}

        for group_idx, (group_name, sample_list) in enumerate(background_groups.items()):
            group_label = group_idx + 1

            group_train, group_valid, group_test = [], [], []
            group_sample_weights = {}

            for sample_name in sample_list:
                sample_train = load_folds_for_sample(sample_name, "background", train_folds)
                sample_valid = load_folds_for_sample(sample_name, "background", valid_folds)
                sample_test = load_folds_for_sample(sample_name, "background", test_folds)

                for data in sample_train + sample_valid + sample_test:
                    data.y = torch.tensor(group_label, dtype=torch.long)

                all_sample_data = sample_train + sample_valid + sample_test
                sample_total_weight = sum(data.weight.item() for data in all_sample_data)
                group_sample_weights[sample_name] = sample_total_weight

                group_train.extend(sample_train)
                group_valid.extend(sample_valid)
                group_test.extend(sample_test)

                logging.info(f"Loaded {sample_name} in group '{group_name}' (label {group_label}): "
                            f"train={len(sample_train)}, valid={len(sample_valid)}, test={len(sample_test)}, weight={sample_total_weight:.2f}")

            group_total_weight = sum(data.weight.item() for data in group_train + group_valid + group_test)
            group_weights[group_name] = group_total_weight
            sample_weights.update(group_sample_weights)

            bg_train.extend(group_train)
            bg_valid.extend(group_valid)
            bg_test.extend(group_test)

            logging.info(f"Group '{group_name}' total: train={len(group_train)}, valid={len(group_valid)}, test={len(group_test)}, weight={group_total_weight:.2f}")

        all_signal_data = signal_train + signal_valid + signal_test
        signal_total_weight = sum(data.weight.item() for data in all_signal_data)
        group_weights['signal'] = signal_total_weight

        if balance:
            max_group_weight = max(group_weights.values())

            signal_norm_factor = max_group_weight / signal_total_weight if signal_total_weight > 0 else 1.0
            for data in all_signal_data:
                data.weight = data.weight * signal_norm_factor

            logging.info(f"Applied between-group weight normalization:")
            logging.info(f"  signal: factor {signal_norm_factor:.4f}")

            for group_idx, (group_name, _) in enumerate(background_groups.items()):
                group_norm_factor = max_group_weight / group_weights[group_name] if group_weights[group_name] > 0 else 1.0
                group_label = group_idx + 1
                for data in bg_train + bg_valid + bg_test:
                    if data.y.item() == group_label:
                        data.weight = data.weight * group_norm_factor
                logging.info(f"  {group_name}: factor {group_norm_factor:.4f}")

        train_data = shuffle(signal_train + bg_train, random_state=random_state)
        valid_data = shuffle(signal_valid + bg_valid, random_state=random_state)
        test_data = shuffle(signal_test + bg_test, random_state=random_state)

        def count_classes(data_list):
            counts = {}
            for data in data_list:
                label = data.y.item()
                counts[label] = counts.get(label, 0) + 1
            return counts

        logging.info(f"Grouped multi-class splits created:")
        logging.info(f"  Train: {len(train_data)} events, class distribution: {count_classes(train_data)}")
        logging.info(f"  Valid: {len(valid_data)} events, class distribution: {count_classes(valid_data)}")
        logging.info(f"  Test: {len(test_data)} events, class distribution: {count_classes(test_data)}")

        return train_data, valid_data, test_data

    def _load_class_fold(self, class_name, label, sample_type, sample_list, channel, fold,
                         max_events_per_fold, random_state):
        """Load one class for one fold: concatenate its samples, label, subsample.

        This is exactly one iteration of the original serial loop, so serial and
        parallel loading select the same events in the same order.
        """
        fold_data = []
        sample_counts = []
        for sample_name in sample_list:
            sample_data = self.load_sample_data(sample_name, sample_type, channel, fold)
            sample_counts.append((sample_name, len(sample_data)))
            fold_data.extend(sample_data)

        for data in fold_data:
            data.y = torch.tensor(label, dtype=torch.long)

        original_count = len(fold_data)
        if sample_type == "signal":
            if max_events_per_fold and original_count > max_events_per_fold:
                fold_data = resample(fold_data, n_samples=max_events_per_fold, replace=False, random_state=random_state)
                logging.info(f"Subsampled signal fold {fold}: {original_count} → {len(fold_data)} events")
            else:
                logging.info(f"Loaded {len(fold_data)} signal events from fold {fold}")
            return fold_data

        if len(sample_list) > 1:
            sample_breakdown = " + ".join([f"{count} {name}" for name, count in sample_counts])
            logging.info(f"Group '{class_name}' fold {fold}: {sample_breakdown} = {original_count} events")

        if max_events_per_fold and original_count > max_events_per_fold:
            fold_data = resample(fold_data, n_samples=max_events_per_fold, replace=False, random_state=random_state)
            logging.info(f"Subsampled {class_name} fold {fold}: {original_count} → {len(fold_data)} events")
        else:
            if len(sample_list) == 1:
                logging.info(f"Loaded {len(fold_data)} events from {class_name} (label {label}) fold {fold}")
        return fold_data

    def _run_class_fold_tasks(self, tasks, num_workers):
        """Run (class, fold) tasks serially or in a process pool; results keep task order."""
        if num_workers is None:
            num_workers = min(len(tasks), _available_cpus())
            # Nested pools are not allowed inside daemonic workers (e.g. multiprocessing.Pool)
            if mp.current_process().daemon:
                num_workers = 1
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")

        start = time.time()
        if num_workers == 1 or len(tasks) == 1:
            results = [self._load_class_fold(**task) for task in tasks]
            logging.info(f"[PROFILE] load tasks: {len(tasks)} (class, fold) tasks serially in {time.time() - start:.1f} s")
            return results

        logging.info(f"[PROFILE] load tasks: starting {len(tasks)} (class, fold) tasks with {num_workers} processes")
        results = []
        # Receiving the subsampled lists unpickles ~10^5 objects; pause the cyclic GC meanwhile
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            with ProcessPoolExecutor(max_workers=num_workers, mp_context=mp.get_context("spawn"),
                                     initializer=_init_load_worker, initargs=(list(sys.path),)) as pool:
                for blob, records in pool.map(_run_class_fold_task,
                                              [self.dataset_root] * len(tasks), tasks):
                    for levelno, message in records:
                        logging.log(levelno, message)
                    results.append(pickle.loads(blob))
        finally:
            if gc_was_enabled:
                gc.enable()
        logging.info(f"[PROFILE] load tasks: {len(tasks)} (class, fold) tasks in parallel in {time.time() - start:.1f} s")
        return results

    def load_multiclass_with_subsampling(self, signal_sample, background_groups, channel, fold_list,
                                         max_events_per_fold=None, balance_weights=True,
                                         random_state=42, num_workers=None):
        """
        Load multi-class data with optional per-fold subsampling and weight normalization.

        This method combines loading, subsampling, and weight balancing in a single call.
        Used by DataPipeline, SharedDatasetManager, OverfittingDetector, and
        visualizeGAIteration.

        Args:
            signal_sample: Signal sample name (e.g., "TTToHcToWAToMuMu-MHc130_MA90")
            background_groups: Dict of {group_name: [list_of_samples]}
                             e.g., {'nonprompt': ['Skim_TriLep_TTLL_powheg'],
                                    'diboson': ['Skim_TriLep_WZTo3LNu_amcatnlo', 'Skim_TriLep_ZZTo4L_powheg']}
            channel: Channel name (e.g., "Run1E2Mu", "Run3Mu", "Combined")
            fold_list: List of fold numbers to load (e.g., [0, 1, 2])
            max_events_per_fold: Maximum events per fold per class (None = no limit)
            balance_weights: Whether to normalize weights so each class has equal total weight
            random_state: Random state for shuffling and subsampling
            num_workers: Processes for loading (one task per (class, fold)).
                         None = min(#tasks, available CPUs); 1 = serial, in-process.

        Returns:
            List of Data objects with:
            - Signal: label 0
            - Background groups: labels 1, 2, 3, ...
            - Weights normalized across classes (if balance_weights=True)
        """
        return self.load_multiclass_splits(
            signal_sample, background_groups, channel,
            splits={None: (fold_list, max_events_per_fold)},
            balance_weights=balance_weights, random_state=random_state, num_workers=num_workers,
        )[None]

    def load_multiclass_splits(self, signal_sample, background_groups, channel, splits,
                               balance_weights=True, random_state=42, num_workers=None):
        """Load several splits (e.g. train and valid) with ONE process pool.

        All (class, fold) tasks of all splits run concurrently; weight balancing and the
        final shuffle are then applied per split exactly as load_multiclass_with_subsampling
        does, so each split is identical to loading it on its own.

        Args:
            splits: Ordered dict {split_name: (fold_list, max_events_per_fold)},
                    e.g. {"train": ([0, 1, 2], 20000), "valid": ([3], 20000)}
            (other arguments as in load_multiclass_with_subsampling)

        Returns:
            Dict {split_name: list of Data objects}
        """
        if not splits:
            raise ValueError("load_multiclass_splits: no splits given")

        tasks, owners = [], []
        for split_name, (fold_list, max_events_per_fold) in splits.items():
            split_tasks = self._build_class_fold_tasks(signal_sample, background_groups, channel,
                                                       fold_list, max_events_per_fold, random_state)
            tasks += split_tasks
            owners += [split_name] * len(split_tasks)

        collected = {split_name: [] for split_name in splits}
        for split_name, fold_data in zip(owners, self._run_class_fold_tasks(tasks, num_workers)):
            collected[split_name].extend(fold_data)

        results = {}
        for split_name in splits:
            start = time.time()
            results[split_name] = self._balance_and_shuffle(collected[split_name], background_groups,
                                                            balance_weights, random_state, split_name)
            logging.info(f"[PROFILE] balance+shuffle{f' {split_name}' if split_name else ''}: "
                         f"{len(results[split_name])} events in {time.time() - start:.1f} s")
        return results

    @staticmethod
    def _build_class_fold_tasks(signal_sample, background_groups, channel, fold_list,
                                max_events_per_fold, random_state):
        """One task per (class, fold), in the order of the original serial loop:
        signal folds first, then each background group fold by fold."""
        tasks = [dict(class_name="signal", label=0, sample_type="signal", sample_list=[signal_sample],
                      channel=channel, fold=fold, max_events_per_fold=max_events_per_fold,
                      random_state=random_state)
                 for fold in fold_list]
        for group_idx, (group_name, sample_list) in enumerate(background_groups.items()):
            tasks += [dict(class_name=group_name, label=group_idx + 1, sample_type="background",
                           sample_list=list(sample_list), channel=channel, fold=fold,
                           max_events_per_fold=max_events_per_fold, random_state=random_state)
                      for fold in fold_list]
        return tasks

    @staticmethod
    def _balance_and_shuffle(all_data, background_groups, balance_weights, random_state, split_name=None):
        """Normalize weights so each class has the same total weight, then shuffle."""
        if balance_weights:
            class_weights = {}
            for data in all_data:
                label = data.y.item()
                class_weights[label] = class_weights.get(label, 0.0) + data.weight.item()

            if class_weights:
                max_class_weight = max(class_weights.values())
                suffix = f" ({split_name})" if split_name else ""
                logging.info(f"Applying weight normalization across classes{suffix}:")

                for class_label in sorted(class_weights.keys()):
                    if class_weights[class_label] > 0:
                        norm_factor = max_class_weight / class_weights[class_label]
                        for data in all_data:
                            if data.y.item() == class_label:
                                data.weight = data.weight * norm_factor

                        if class_label == 0:
                            class_name = "signal"
                        else:
                            class_name = list(background_groups.keys())[class_label - 1]
                        logging.info(f"  Class {class_label} ({class_name}): factor {norm_factor:.4f}, weight {class_weights[class_label]:.2f} → {max_class_weight:.2f}")

        return shuffle(all_data, random_state=random_state)
