#!/usr/bin/env python
"""
Overfitting detection for GA optimization in ParticleNetMD.

This module performs 16 KS tests (4 true classes x 4 output scores) to detect
overfitting in trained models. A model is considered overfitted if ANY of the
16 tests fails (p-value < threshold).

Key features:
- 16 KS tests for comprehensive overfitting detection
- Adaptive bin merging to handle negative MC weights
- ROOT-based histogram comparison
"""

import os
import json
import logging
import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from typing import Dict, List, Tuple, Optional

import ROOT
ROOT.gROOT.SetBatch(True)

# Class names for ParticleNetMD multi-class training
CLASS_NAMES = ['signal', 'nonprompt', 'diboson', 'ttX']
SCORE_NAMES = ['signal_score', 'nonprompt_score', 'diboson_score', 'ttX_score']
CLASS_DISPLAY_NAMES = ['Signal', 'Nonprompt', 'Diboson', 'TTX']


class OverfittingDetector:
    """Detector for model overfitting using KS tests."""

    def __init__(self, ga_config, signal: str, channel: str, device: str = 'cuda', pilot: bool = False):
        """
        Initialize overfitting detector.

        Args:
            ga_config: GAConfigLoader instance
            signal: Signal sample name (e.g., "MHc130_MA90")
            channel: Channel name (Run1E2Mu, Run3Mu, Combined)
            device: Device to use for evaluation
            pilot: Whether to use pilot datasets
        """
        self.ga_config = ga_config
        self.signal = signal
        self.channel = channel
        self.device = torch.device(device)
        self.pilot = pilot

        # Get configurations
        self.train_params = ga_config.get_training_parameters()
        self.overfitting_config = ga_config.get_overfitting_config()
        self.dataset_config = ga_config.get_dataset_config()
        self.model_config = ga_config.get_model_config()
        self.bg_groups = ga_config.get_background_groups()

        self.num_classes = 1 + len(self.bg_groups)
        self.p_threshold = self.overfitting_config.get('p_value_threshold', 0.05)
        self.bin_merge_threshold = self.overfitting_config.get('bin_merge_threshold', 1e-6)
        self.bin_merge_max_iterations = self.overfitting_config.get('bin_merge_max_iterations', 100)

        # Datasets (loaded lazily)
        self._train_data = None
        self._test_data = None

    def _load_datasets(self):
        """Load training and test datasets."""
        if self._train_data is not None:
            return

        from DynamicDatasetLoader import DynamicDatasetLoader

        WORKDIR = os.environ.get("WORKDIR")
        if not WORKDIR:
            raise EnvironmentError("WORKDIR not set. Run 'source setup.sh'")

        dataset_root = f"{WORKDIR}/ParticleNetMD/dataset"
        signal_full = self.dataset_config['signal_prefix'] + self.signal

        loader = DynamicDatasetLoader(
            dataset_root=dataset_root,
            background_groups=self.bg_groups,
            background_prefix=self.dataset_config['background_prefix']
        )

        # Load training data (for comparison)
        train_folds = self.train_params['train_folds']
        self._train_data = loader.load_multiclass_with_subsampling(
            signal_sample=signal_full,
            background_groups=self.bg_groups,
            channel=self.channel,
            fold_list=train_folds,
            pilot=self.pilot,
            max_events_per_fold=self.train_params.get('max_events_per_fold_per_class'),
            balance_weights=self.train_params.get('balance_weights', True)
        )

        # Load test data (for overfitting detection)
        test_folds = self.overfitting_config.get('test_folds', [4])
        self._test_data = loader.load_multiclass_with_subsampling(
            signal_sample=signal_full,
            background_groups=self.bg_groups,
            channel=self.channel,
            fold_list=test_folds,
            pilot=self.pilot,
            max_events_per_fold=self.train_params.get('max_events_per_fold_per_class'),
            balance_weights=self.train_params.get('balance_weights', True)
        )

        logging.info(f"Loaded {len(self._train_data)} train events, {len(self._test_data)} test events")

    def check_overfitting(self, model_path: str, iteration: int, idx: int,
                          output_dir: str) -> Tuple[bool, Dict]:
        """
        Check if model is overfitted using 16 KS tests.

        Args:
            model_path: Path to model checkpoint
            iteration: GA iteration number
            idx: Model index in population
            output_dir: Directory to save results

        Returns:
            Tuple of (is_overfitted, ks_results_dict)
        """
        from MultiClassModels import create_multiclass_model
        from Preprocess import SharedBatchDataset

        self._load_datasets()

        # Load checkpoint first to infer nNodes from state_dict
        checkpoint = torch.load(model_path, weights_only=True)

        # Infer nNodes from conv1.mlp.0.weight shape: [nNodes, num_node_features * 2]
        # The first dimension is nNodes (hidden dimension)
        nNodes = checkpoint['model_state_dict']['conv1.mlp.0.weight'].shape[0]

        # Create model with correct architecture
        model = create_multiclass_model(
            model_type=self.model_config['default_model'],
            num_node_features=9,
            num_graph_features=4,
            num_classes=self.num_classes,
            num_hidden=nNodes,
            dropout_p=self.train_params['dropout_p']
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Get predictions on train and test sets
        train_predictions = self._get_predictions(model, self._train_data)
        test_predictions = self._get_predictions(model, self._test_data)

        # Perform 16 KS tests
        ks_results, histograms = perform_ks_tests_comprehensive(
            y_true_train=train_predictions['labels'],
            y_scores_train=train_predictions['scores'],
            weights_train=train_predictions['weights'],
            y_true_test=test_predictions['labels'],
            y_scores_test=test_predictions['scores'],
            weights_test=test_predictions['weights'],
            p_threshold=self.p_threshold,
            bin_merge_threshold=self.bin_merge_threshold,
            bin_merge_max_iterations=self.bin_merge_max_iterations
        )

        # Check if any test failed
        is_overfitted = any(result['is_overfitted'] for result in ks_results.values())

        # Save results
        model_output_dir = os.path.join(output_dir, f"model{idx}")
        os.makedirs(model_output_dir, exist_ok=True)
        save_ks_results(ks_results, histograms, model_output_dir)

        # Clean up
        del model
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        return is_overfitted, ks_results

    def _get_predictions(self, model, data_list: List) -> Dict[str, np.ndarray]:
        """Get model predictions on dataset."""
        from torch_geometric.data import Batch

        # Create batch
        batch = Batch.from_data_list(data_list)
        batch = batch.to(self.device)

        with torch.no_grad():
            logits = model(batch.x, batch.edge_index, batch.graphInput, batch.batch)
            probs = torch.softmax(logits, dim=1)

        return {
            'labels': batch.y.cpu().numpy(),
            'scores': probs.cpu().numpy(),
            'weights': batch.weight.cpu().numpy()
        }


def create_weighted_histogram(scores: np.ndarray, weights: np.ndarray, name: str,
                              title: str, nbins: int = 50) -> ROOT.TH1D:
    """Create weighted histogram from scores."""
    hist = ROOT.TH1D(name, title, nbins, 0.0, 1.0)
    hist.SetDirectory(0)

    for score, weight in zip(scores, weights):
        hist.Fill(float(score), float(weight))

    # Normalize to unit area
    if hist.Integral() > 0:
        hist.Scale(1.0 / hist.Integral())

    return hist


def identify_problematic_bins(hist: ROOT.TH1D, threshold: float = 1e-6) -> List[int]:
    """Identify bins with negative content or very small absolute content."""
    problematic_bins = []
    nbins = hist.GetNbinsX()

    for i in range(1, nbins + 1):
        content = hist.GetBinContent(i)
        if content < 0 or abs(content) < threshold:
            problematic_bins.append(i)

    return problematic_bins


def create_merge_groups(hist_train: ROOT.TH1D, hist_test: ROOT.TH1D,
                        threshold: float = 1e-6) -> List[Tuple[int, int]]:
    """Create groups of consecutive bins to merge."""
    prob_train = set(identify_problematic_bins(hist_train, threshold))
    prob_test = set(identify_problematic_bins(hist_test, threshold))
    all_prob = prob_train | prob_test

    if not all_prob:
        return []

    nbins = hist_train.GetNbinsX()
    merge_groups = []
    sorted_prob = sorted(all_prob)

    i = 0
    while i < len(sorted_prob):
        start = sorted_prob[i]
        end = start

        # Extend group to include consecutive problematic bins
        while i + 1 < len(sorted_prob) and sorted_prob[i + 1] == end + 1:
            i += 1
            end = sorted_prob[i]

        # Extend to include adjacent bin if needed for stability
        if start > 1:
            start -= 1
        if end < nbins:
            end += 1

        merge_groups.append((start, end))
        i += 1

    return merge_groups


def apply_bin_merging(h_train: ROOT.TH1D, h_test: ROOT.TH1D,
                      merge_groups: List[Tuple[int, int]]) -> Tuple[ROOT.TH1D, ROOT.TH1D]:
    """Apply bin merging to both histograms."""
    if not merge_groups:
        return h_train, h_test

    # Get original binning
    nbins = h_train.GetNbinsX()
    xmin = h_train.GetXaxis().GetXmin()
    xmax = h_train.GetXaxis().GetXmax()

    # Determine new bin edges
    old_edges = [h_train.GetXaxis().GetBinLowEdge(i) for i in range(1, nbins + 2)]

    # Merge groups affect bin edges
    skip_bins = set()
    for start, end in merge_groups:
        for b in range(start, end):
            if b <= nbins:
                skip_bins.add(b)

    new_edges = [old_edges[0]]
    for i in range(1, nbins + 1):
        if i not in skip_bins:
            new_edges.append(old_edges[i])

    if len(new_edges) < 2:
        return h_train, h_test

    # Create new histograms with merged binning
    new_edges_array = np.array(new_edges, dtype=np.float64)
    n_new_bins = len(new_edges) - 1

    h_train_new = ROOT.TH1D(f"{h_train.GetName()}_merged", h_train.GetTitle(),
                            n_new_bins, new_edges_array)
    h_test_new = ROOT.TH1D(f"{h_test.GetName()}_merged", h_test.GetTitle(),
                           n_new_bins, new_edges_array)
    h_train_new.SetDirectory(0)
    h_test_new.SetDirectory(0)

    # Rebin by adding content
    for i in range(1, nbins + 1):
        x = h_train.GetXaxis().GetBinCenter(i)
        h_train_new.Fill(x, h_train.GetBinContent(i))
        h_test_new.Fill(x, h_test.GetBinContent(i))

    return h_train_new, h_test_new


def merge_histograms_iteratively(hist_train: ROOT.TH1D, hist_test: ROOT.TH1D,
                                 threshold: float = 1e-6, max_iterations: int = 100) -> Tuple[ROOT.TH1D, ROOT.TH1D, int]:
    """Iteratively merge bins in both histograms until no problematic bins remain."""
    h_train = hist_train.Clone(f"{hist_train.GetName()}_itermerge")
    h_test = hist_test.Clone(f"{hist_test.GetName()}_itermerge")
    h_train.SetDirectory(0)
    h_test.SetDirectory(0)

    iteration = 0

    while iteration < max_iterations:
        prob_train = identify_problematic_bins(h_train, threshold)
        prob_test = identify_problematic_bins(h_test, threshold)

        if not prob_train and not prob_test:
            break

        merge_groups = create_merge_groups(h_train, h_test, threshold)

        if not merge_groups:
            break

        h_train, h_test = apply_bin_merging(h_train, h_test, merge_groups)
        iteration += 1

    if iteration >= max_iterations:
        logging.warning(f"Bin merging reached max iterations ({max_iterations}) for {hist_train.GetName()}")

    return h_train, h_test, iteration


def perform_ks_tests_comprehensive(y_true_train: np.ndarray, y_scores_train: np.ndarray, weights_train: np.ndarray,
                                   y_true_test: np.ndarray, y_scores_test: np.ndarray, weights_test: np.ndarray,
                                   p_threshold: float = 0.05,
                                   bin_merge_threshold: float = 1e-6,
                                   bin_merge_max_iterations: int = 100) -> Tuple[Dict, Dict]:
    """
    Perform comprehensive KS tests: all 4 output scores for each of 4 true classes.

    Returns:
        Tuple of (ks_results_dict, histograms_dict)
    """
    num_classes = 4
    ks_results = {}
    histograms = {}

    for true_class in range(num_classes):
        train_mask = (y_true_train == true_class)
        test_mask = (y_true_test == true_class)

        train_scores_class = y_scores_train[train_mask]
        test_scores_class = y_scores_test[test_mask]
        train_weights_class = weights_train[train_mask]
        test_weights_class = weights_test[test_mask]

        for score_idx in range(num_classes):
            train_scores = train_scores_class[:, score_idx]
            test_scores = test_scores_class[:, score_idx]

            key = f"{CLASS_NAMES[true_class]}_{SCORE_NAMES[score_idx]}"

            h_train = create_weighted_histogram(
                train_scores, train_weights_class,
                f"h_train_{key}", f"Train: {CLASS_DISPLAY_NAMES[true_class]} - {SCORE_NAMES[score_idx]}"
            )
            h_test = create_weighted_histogram(
                test_scores, test_weights_class,
                f"h_test_{key}", f"Test: {CLASS_DISPLAY_NAMES[true_class]} - {SCORE_NAMES[score_idx]}"
            )

            h_train_merged, h_test_merged, n_merge_iterations = merge_histograms_iteratively(
                h_train, h_test, threshold=bin_merge_threshold, max_iterations=bin_merge_max_iterations
            )

            p_value = h_train_merged.KolmogorovTest(h_test_merged, option="X")

            ks_results[key] = {
                'true_class': CLASS_NAMES[true_class],
                'score_type': SCORE_NAMES[score_idx],
                'p_value': float(p_value),
                'is_overfitted': bool(p_value < p_threshold),
                'n_train': int(train_mask.sum()),
                'n_test': int(test_mask.sum()),
                'n_merge_iterations': int(n_merge_iterations),
                'n_bins_original': int(h_train.GetNbinsX()),
                'n_bins_merged': int(h_train_merged.GetNbinsX())
            }

            histograms[key] = {
                'train': h_train_merged,
                'test': h_test_merged
            }

    return ks_results, histograms


def save_ks_results(ks_results: Dict, histograms: Dict, output_dir: str):
    """Save KS test results to JSON and ROOT files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save JSON results
    json_path = os.path.join(output_dir, "kolmogorov.json")
    with open(json_path, 'w') as f:
        json.dump(ks_results, f, indent=2)

    # Save ROOT file with histograms
    root_path = os.path.join(output_dir, "kolmogorov.root")
    root_file = ROOT.TFile(root_path, "RECREATE")

    for key, hists in histograms.items():
        hists['train'].Write(f"h_train_{key}")
        hists['test'].Write(f"h_test_{key}")

    root_file.Close()

    any_overfitted = any(result['is_overfitted'] for result in ks_results.values())
    min_p_value = min(result['p_value'] for result in ks_results.values())

    logging.info(f"KS results saved to {output_dir}")
    logging.info(f"  Overfitted: {any_overfitted}, Min p-value: {min_p_value:.4f}")
