#!/usr/bin/env python
"""Parametric data pipeline for fixed-mHc ParticleNetMD studies."""

import logging
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from typing import Any, Dict, List, Tuple

import torch
from sklearn.utils import resample, shuffle
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

from DataPipeline import DataPipeline
from DynamicDatasetLoader import DynamicDatasetLoader


class ParametricHypothesisDataset(Dataset):
    """Lazy parametric dataset that materializes mA-conditioned graphs per item."""

    def __init__(self, records, references, ma_center, ma_scale, class_weight_scales=None):
        self.records = records
        self.references = references
        self.ma_center = float(ma_center)
        self.ma_scale = float(ma_scale)
        self.class_weight_scales = class_weight_scales or {}

    def __len__(self):
        return len(self.references)

    def __bool__(self):
        return len(self) > 0

    def _ma_norm(self, ma_value: float) -> float:
        return (float(ma_value) - self.ma_center) / self.ma_scale

    def __getitem__(self, index):
        record_index, ma_value = self.references[index]
        data, label = self.records[record_index]

        cloned = data.clone()
        cloned.y = torch.tensor(label, dtype=torch.long)

        scale = float(self.class_weight_scales.get(label, 1.0))
        if hasattr(cloned, "weight"):
            cloned.weight = cloned.weight * scale

        ma_norm = self._ma_norm(ma_value)
        base_graph_input = cloned.graphInput
        if base_graph_input.dim() == 1:
            base_graph_input = base_graph_input.view(1, -1)

        ma_feature = torch.tensor([[ma_norm]], dtype=base_graph_input.dtype)
        cloned.graphInput = torch.cat([base_graph_input, ma_feature], dim=1)
        cloned.param_mA = torch.tensor([float(ma_value)], dtype=torch.float)
        cloned.param_mA_norm = torch.tensor([float(ma_norm)], dtype=torch.float)
        return cloned


class ParametricDataPipeline(DataPipeline):
    """
    Build P(class | event, mA) training splits from existing ParticleNetMD datasets.

    Signal events are loaded from one sample per mA hypothesis and receive their
    matching mA. Background events are loaded once per split and cloned for all
    mA hypotheses.
    """

    def __init__(self, config):
        super().__init__(config)
        self.loader = DynamicDatasetLoader(self.dataset_root)

    def _ma_norm(self, ma_value: float) -> float:
        return (float(ma_value) - float(self.config.args.ma_center)) / float(self.config.args.ma_scale)

    def _data_loading_workers(self) -> int:
        return max(1, int(getattr(self.config.args, "data_loading_workers", 1)))

    def _load_sample_fold(self, sample_name: str, sample_type: str, fold: int) -> List[Any]:
        return self.loader.load_sample_data(
            sample_name=sample_name,
            sample_type=sample_type,
            channel=self.config.args.channel,
            fold=fold,
        )

    def _run_load_jobs(self, jobs: List[Tuple[str, str, int]]) -> List[List[Any]]:
        if not jobs:
            return []

        results = [None] * len(jobs)
        for index, data in self._iter_load_jobs(jobs):
            results[index] = data

        return results

    def _iter_load_jobs(self, jobs: List[Tuple[str, str, int]]):
        if not jobs:
            return

        workers = min(self._data_loading_workers(), len(jobs))
        if workers <= 1:
            for index, (sample, sample_type, fold) in enumerate(jobs):
                yield index, self._load_sample_fold(sample, sample_type, fold)
            return

        with ThreadPoolExecutor(max_workers=workers) as executor:
            job_iter = iter(enumerate(jobs))
            future_to_index = {}

            for _ in range(workers):
                try:
                    index, (sample, sample_type, fold) = next(job_iter)
                except StopIteration:
                    break
                future = executor.submit(self._load_sample_fold, sample, sample_type, fold)
                future_to_index[future] = index

            while future_to_index:
                completed, _ = wait(future_to_index, return_when=FIRST_COMPLETED)
                for future in completed:
                    index = future_to_index.pop(future)
                    result = future.result()

                    try:
                        next_index, (sample, sample_type, fold) = next(job_iter)
                    except StopIteration:
                        pass
                    else:
                        next_future = executor.submit(self._load_sample_fold, sample, sample_type, fold)
                        future_to_index[next_future] = next_index

                    yield index, result

    def _cap_events(self, data_list: List[Any], cap: int, random_state: int, label: str) -> List[Any]:
        original = len(data_list)
        if cap and original > cap:
            capped = resample(data_list, n_samples=cap, replace=False, random_state=random_state)
            logging.info(f"Subsampled {label}: {original} -> {len(capped)} events")
            return capped
        logging.info(f"Loaded {original} events for {label}")
        return data_list

    def _get_class_weight_scales(self, class_weights: Dict[int, float],
                                 background_groups: Dict[str, List[str]]) -> Dict[int, float]:
        if not self.config.args.balance:
            return {}

        if not class_weights:
            return {}

        max_class_weight = max(class_weights.values())
        logging.info("Applying parametric class weight normalization:")
        class_weight_scales = {}

        for class_label in sorted(class_weights):
            if class_weights[class_label] == 0:
                continue

            norm_factor = max_class_weight / class_weights[class_label]
            class_weight_scales[class_label] = norm_factor

            if class_label == 0:
                class_name = "signal"
            else:
                class_name = list(background_groups.keys())[class_label - 1]
            logging.info(
                f"  Class {class_label} ({class_name}): factor {norm_factor:.4f}, "
                f"weight {class_weights[class_label]:.2f} -> {max_class_weight:.2f}"
            )

        return class_weight_scales

    def _load_split(self, fold_list: List[int], max_events_per_fold: int, random_state: int) -> ParametricHypothesisDataset:
        records = []
        references = []
        class_weights = {}
        background_groups_full = self.config.get_background_groups_full()

        def add_record(data, label: int, ma_values: List[int]):
            record_index = len(records)
            records.append((data, label))
            weight = float(data.weight.item()) if hasattr(data, "weight") else 1.0
            class_weights[label] = class_weights.get(label, 0.0) + weight * len(ma_values)
            for ma_value in ma_values:
                references.append((record_index, ma_value))

        # Signal: cap each mA sample independently so the signal class is
        # event-count balanced across mass hypotheses before class balancing.
        signal_metadata = []
        signal_jobs = []
        for ma_value, signal_sample in zip(self.config.args.ma_values, self.config.signal_full_names):
            for fold in fold_list:
                signal_metadata.append((ma_value, fold))
                signal_jobs.append((signal_sample, "signal", fold))

        if self._data_loading_workers() > 1:
            logging.info(f"Loading {len(signal_jobs)} signal sample/fold jobs with up to {self._data_loading_workers()} workers")

        for index, signal_data in self._iter_load_jobs(signal_jobs):
            ma_value, fold = signal_metadata[index]
            signal_data = self._cap_events(
                signal_data,
                max_events_per_fold,
                random_state + fold + int(ma_value),
                f"signal mA={ma_value} fold {fold}",
            )
            for data in signal_data:
                add_record(data, 0, [ma_value])

        # Backgrounds: cap each grouped class per fold, then expose selected
        # events for every mA hypothesis lazily through dataset references.
        for group_idx, (group_name, sample_list) in enumerate(background_groups_full.items()):
            group_label = group_idx + 1

            for fold in fold_list:
                sample_jobs = [(sample_name, "background", fold) for sample_name in sample_list]
                if self._data_loading_workers() > 1:
                    logging.info(
                        f"Loading {group_name} fold {fold}: "
                        f"{len(sample_jobs)} sample jobs with up to {self._data_loading_workers()} workers"
                    )

                sample_data_lists = self._run_load_jobs(sample_jobs)
                sample_counts = [
                    (sample_name, len(sample_data))
                    for sample_name, sample_data in zip(sample_list, sample_data_lists)
                ]
                group_fold_data = [
                    data
                    for sample_data in sample_data_lists
                    for data in sample_data
                ]

                if len(sample_counts) > 1:
                    breakdown = " + ".join(f"{count} {name}" for name, count in sample_counts)
                    logging.info(f"Group '{group_name}' fold {fold}: {breakdown} = {len(group_fold_data)} events")

                group_fold_data = self._cap_events(
                    group_fold_data,
                    max_events_per_fold,
                    random_state + fold + group_label * 1000,
                    f"{group_name} fold {fold}",
                )

                for data in group_fold_data:
                    add_record(data, group_label, self.config.args.ma_values)

        class_weight_scales = self._get_class_weight_scales(class_weights, background_groups_full)
        references = shuffle(references, random_state=random_state)
        return ParametricHypothesisDataset(
            records=records,
            references=references,
            ma_center=self.config.args.ma_center,
            ma_scale=self.config.args.ma_scale,
            class_weight_scales=class_weight_scales,
        )

    def create_datasets(self) -> Tuple[List[Any], List[Any], List[Any]]:
        logging.info("Creating parametric ParticleNetMD training splits")
        logging.info(f"mHc: {self.config.args.mhc}, mA values: {self.config.args.ma_values}")
        logging.info(f"Channel: {self.config.args.channel}, Fold: {self.config.args.fold}")

        if not self.config.use_groups:
            raise ValueError("ParametricPN currently supports grouped background mode only")

        max_events = self.config.args.max_events_per_fold_per_class
        train_folds = self.config.args.train_folds
        valid_folds = self.config.args.valid_folds
        test_folds = self.config.args.test_folds

        if self.config.args.pilot:
            train_folds = [train_folds[0]]
            max_train = self.config.args.pilot_max_train_events_per_fold_per_class
            max_eval = self.config.args.pilot_max_eval_events_per_fold_per_class
            if max_train <= 0 or max_eval <= 0:
                raise ValueError("Pilot event caps must be positive")
            logging.info(
                f"PILOT MODE: train_folds={train_folds} valid_folds={valid_folds} "
                f"test_folds={test_folds} | caps: train={max_train} valid/test={max_eval} "
                f"| workers={self._data_loading_workers()}"
            )
        else:
            max_train = max_events
            max_eval = max_events

        self.train_data = self._load_split(train_folds, max_train, random_state=42)
        self.valid_data = self._load_split(valid_folds, max_eval, random_state=43)
        self.test_data = self._load_split(test_folds, max_eval, random_state=44)

        if not self.train_data or not self.valid_data or not self.test_data:
            raise ValueError("Empty parametric datasets created - check sample availability")

        logging.info(
            f"Parametric dataset sizes - Train: {len(self.train_data)}, "
            f"Valid: {len(self.valid_data)}, Test: {len(self.test_data)}"
        )
        return self.train_data, self.valid_data, self.test_data

    def create_data_loaders(self, batch_size: int = 1024):
        if self.train_data is None or self.valid_data is None or self.test_data is None:
            raise ValueError("Datasets must be created before creating data loaders. Call create_datasets() first.")

        self.train_loader = DataLoader(
            self.train_data, batch_size=batch_size, pin_memory=True, shuffle=True
        )
        self.valid_loader = DataLoader(
            self.valid_data, batch_size=batch_size, pin_memory=True, shuffle=False
        )
        self.test_loader = DataLoader(
            self.test_data, batch_size=batch_size, pin_memory=True, shuffle=False
        )

        logging.info(f"Created lazy parametric data loaders with batch size {batch_size}")
        logging.info(
            f"DataLoader sizes - Train: {len(self.train_data)}, "
            f"Valid: {len(self.valid_data)}, Test: {len(self.test_data)}"
        )

        return self.train_loader, self.valid_loader, self.test_loader


def create_parametric_data_pipeline(config) -> ParametricDataPipeline:
    return ParametricDataPipeline(config)
