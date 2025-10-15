#!/usr/bin/env python
"""
Overfitting Detection Module for GA Optimization.

Uses Kolmogorov-Smirnov test on per-class score distributions to detect
overfitting by comparing train and test set distributions.

A model is considered overfitted if ANY class score distribution shows
statistically significant difference (p < threshold) between train and test sets.
"""

import os
import json
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import ROOT

from GAConfig import load_ga_config
from MultiClassModels import create_multiclass_model
from DynamicDatasetLoader import DynamicDatasetLoader
from Preprocess import GraphDataset
from sklearn.utils import shuffle


class OverfittingDetector:
    """
    Detect overfitting using Kolmogorov-Smirnov test on score distributions.

    For each class (4 total: signal, nonprompt, diboson, ttX):
        - Create score distribution histograms for train and test sets
        - Perform K-S test comparing distributions
        - Flag as overfitted if p-value < threshold

    A model fails if ANY class shows overfitting.
    """

    def __init__(self, config, signal, channel, device, pilot=False):
        """
        Initialize overfitting detector.

        Args:
            config: GAConfigLoader instance
            signal: Signal mass point (e.g., MHc130_MA100)
            channel: Channel name (Run1E2Mu, Run3Mu, Combined)
            device: Computation device (cuda or cpu)
            pilot: Whether to use pilot datasets (default: False)
        """
        self.config = config
        self.signal = signal
        self.channel = channel
        self.device = device
        self.pilot = pilot

        # Get configuration
        self.train_params = config.get_training_parameters()
        self.bg_groups = config.get_background_groups()
        self.dataset_config = config.get_dataset_config()
        self.model_config = config.get_model_config()
        self.overfitting_config = config.get_overfitting_config()
        self.output_config = config.get_output_config()

        # Fold assignments
        self.train_folds = self.train_params['train_folds']  # [0, 2, 4]
        self.test_folds = self.overfitting_config['test_folds']  # [3]
        self.p_threshold = self.overfitting_config['p_value_threshold']  # 0.05

        # Class names
        self.class_names = ['signal', 'nonprompt', 'diboson', 'ttX']
        self.num_classes = len(self.class_names)

        # Store last results
        self.last_results = None

        logging.info(f"OverfittingDetector initialized:")
        logging.info(f"  Train folds: {self.train_folds}")
        logging.info(f"  Test folds: {self.test_folds}")
        logging.info(f"  p-value threshold: {self.p_threshold}")
        logging.info(f"  Pilot mode: {self.pilot}")

    def load_model_and_data(self, model_path, iteration, idx):
        """
        Load trained model checkpoint and datasets.

        Args:
            model_path: Path to model checkpoint (.pt file)
            iteration: GA iteration number
            idx: Model index in population

        Returns:
            model: Loaded model
            train_loader: DataLoader for train set
            test_loader: DataLoader for test set
        """
        logging.debug(f"Loading model {idx} from {model_path}")

        # Check if checkpoint exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

        # Load hyperparameters from config JSON
        # Replace models/ directory with json/ directory and .pt extension with .json
        models_subdir = self.output_config['models_subdir']
        json_subdir = self.output_config['json_subdir']
        config_path = model_path.replace(f'/{models_subdir}/', f'/{json_subdir}/').replace('.pt', '.json')
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config not found: {config_path}")

        with open(config_path, 'r') as f:
            config_data = json.load(f)

        hyperparams = config_data['hyperparameters']
        num_hidden = hyperparams['num_hidden']
        logging.debug(f"Loaded hyperparameters: num_hidden={num_hidden}")

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Extract model state
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            model_state = checkpoint

        # Create model architecture with correct num_hidden
        model = create_multiclass_model(
            model_type=self.model_config['default_model'],
            num_node_features=9,
            num_graph_features=4,
            num_classes=self.num_classes,
            num_hidden=num_hidden,  # Use value from config
            dropout_p=self.train_params['dropout_p']
        ).to(self.device)

        # Load weights
        model.load_state_dict(model_state)
        model.eval()

        logging.debug(f"Model loaded successfully")

        # Load datasets
        WORKDIR = os.environ.get("WORKDIR")
        dataset_root = f"{WORKDIR}/ParticleNet/dataset"

        # Initialize data loader
        loader = DynamicDatasetLoader(
            dataset_root=dataset_root,
            separate_bjets=self.dataset_config['use_bjets']
        )

        # Construct sample names
        signal_full = self.dataset_config['signal_prefix'] + self.signal
        background_groups_full = {
            group_name: [self.dataset_config['background_prefix'] + sample for sample in samples]
            for group_name, samples in self.bg_groups.items()
        }

        # Load data function
        def load_multiclass_data_for_folds(fold_list):
            """Load and combine data from multiple folds."""
            all_data = []

            # Load signal data
            for fold in fold_list:
                signal_data = loader.load_sample_data(signal_full, "signal", self.channel, fold, pilot=self.pilot)
                for data in signal_data:
                    data.y = torch.tensor(0, dtype=torch.long)  # Signal = class 0
                all_data.extend(signal_data)

            # Load background groups
            for group_idx, (group_name, sample_list) in enumerate(background_groups_full.items()):
                group_label = group_idx + 1  # Background groups: 1, 2, 3

                for fold in fold_list:
                    group_fold_data = []
                    for sample_name in sample_list:
                        sample_data = loader.load_sample_data(sample_name, "background", self.channel, fold, pilot=self.pilot)
                        group_fold_data.extend(sample_data)

                    # Assign group labels
                    for data in group_fold_data:
                        data.y = torch.tensor(group_label, dtype=torch.long)

                    all_data.extend(group_fold_data)

            # Shuffle combined data
            all_data = shuffle(all_data, random_state=42)
            return all_data

        # Load train and test datasets
        logging.debug(f"Loading train folds {self.train_folds}...")
        train_data_all = load_multiclass_data_for_folds(self.train_folds)
        train_dataset = GraphDataset(train_data_all)

        logging.debug(f"Loading test folds {self.test_folds}...")
        test_data_all = load_multiclass_data_for_folds(self.test_folds)
        test_dataset = GraphDataset(test_data_all)

        # Create data loaders
        batch_size = self.train_params['batch_size']
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

        logging.info(f"Loaded model {idx}: train={len(train_dataset)}, test={len(test_dataset)} events")

        return model, train_loader, test_loader

    def evaluate_model(self, model, loader, device):
        """
        Evaluate model on dataset.

        Args:
            model: Model to evaluate
            loader: DataLoader for dataset
            device: Computation device

        Returns:
            y_true: True class labels (numpy array)
            y_scores: Predicted class probabilities (numpy array, shape [n_samples, n_classes])
            weights: Event weights (numpy array)
        """
        model.eval()
        y_true_list = []
        y_scores_list = []
        weights_list = []

        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.graphInput, data.batch)
                scores = F.softmax(out, dim=1)  # Convert to probabilities

                y_true_list.append(data.y.cpu().numpy())
                y_scores_list.append(scores.cpu().numpy())
                weights_list.append(data.weight.cpu().numpy())

        y_true = np.concatenate(y_true_list)
        y_scores = np.concatenate(y_scores_list)
        weights = np.concatenate(weights_list)

        return y_true, y_scores, weights

    def create_score_histograms(self, y_true, y_scores, weights, class_idx, dataset_name):
        """
        Create TH1D histogram for score of a specific class with event weights.

        Args:
            y_true: True class labels
            y_scores: Predicted class probabilities [n_samples, n_classes]
            weights: Event weights [n_samples]
            class_idx: Index of class to create histograms for
            dataset_name: "train" or "test"

        Returns:
            TH1D histogram
        """
        class_name = self.class_names[class_idx]

        # Extract scores for this class
        scores = y_scores[:, class_idx]

        # Create histogram
        hist_name = f"h_{dataset_name}_{class_name}"
        hist_title = f"{dataset_name} {class_name} score"
        hist = ROOT.TH1D(hist_name, hist_title, 50, 0, 1)

        # Fill histogram with weights
        for score, weight in zip(scores, weights):
            hist.Fill(score, weight)

        # Normalize to unit area
        if hist.Integral() > 0:
            hist.Scale(1.0 / hist.Integral())

        return hist

    def perform_kolmogorov_tests(self, train_hists_all, test_hists_all):
        """
        Perform Kolmogorov-Smirnov tests for all classes.

        Args:
            train_hists_all: Dict {class_idx: TH1D}
            test_hists_all: Dict {class_idx: TH1D}

        Returns:
            Dictionary with per-class K-S test results:
            {
                class_name: {
                    'p_value': float,
                    'is_overfitted': bool
                }
            }
        """
        results = {}

        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx]

            if class_idx not in train_hists_all or class_idx not in test_hists_all:
                logging.warning(f"Missing histogram for {class_name} score")
                results[class_name] = {
                    'p_value': None,
                    'is_overfitted': False
                }
                continue

            h_train = train_hists_all[class_idx]
            h_test = test_hists_all[class_idx]

            # Perform Kolmogorov test
            p_value = h_train.KolmogorovTest(h_test)
            is_overfitted = (p_value < self.p_threshold)

            logging.debug(f"  {class_name} score: p = {p_value:.4f}")

            results[class_name] = {
                'p_value': float(p_value),
                'is_overfitted': is_overfitted
            }

        return results

    def save_diagnostics(self, results, train_hists_all, test_hists_all, output_dir, idx):
        """
        Save diagnostic ROOT file and JSON with K-S test results.

        Args:
            results: K-S test results from perform_kolmogorov_tests()
            train_hists_all: All train histograms
            test_hists_all: All test histograms
            output_dir: Output directory
            idx: Model index
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save ROOT file with histograms
        root_file_path = os.path.join(output_dir, f"model{idx}_kolmogorov.root")
        root_file = ROOT.TFile(root_file_path, "RECREATE")

        # Create canvas with 2x2 grid for 4 classes
        canvas = ROOT.TCanvas("c_all_classes", "Score distributions (Train vs Test)", 1200, 1000)
        canvas.Divide(2, 2)

        # Save all histograms and create overlays
        for class_idx in range(self.num_classes):
            class_name = self.class_names[class_idx]
            canvas.cd(class_idx + 1)

            if class_idx in train_hists_all and class_idx in test_hists_all:
                h_train = train_hists_all[class_idx]
                h_test = test_hists_all[class_idx]

                # Write histograms
                h_train.Write()
                h_test.Write()

                # Draw overlays
                h_train.SetLineColor(ROOT.kBlue)
                h_train.SetLineWidth(2)
                h_test.SetLineColor(ROOT.kRed)
                h_test.SetLineWidth(2)

                h_train.Draw("HIST")
                h_test.Draw("HIST SAME")

                # Add legend
                legend = ROOT.TLegend(0.6, 0.7, 0.88, 0.88)
                legend.AddEntry(h_train, "Train", "l")
                legend.AddEntry(h_test, "Test", "l")
                legend.Draw()

                # Add p-value text
                p_value = results[class_name]['p_value']
                text = ROOT.TLatex()
                text.SetNDC()
                text.SetTextSize(0.04)
                if p_value is not None:
                    text.DrawLatex(0.6, 0.65, f"K-S p = {p_value:.4f}")

        canvas.Write()
        root_file.Close()
        logging.info(f"Saved ROOT diagnostics to {root_file_path}")

        # Save JSON results
        json_file_path = os.path.join(output_dir, f"model{idx}_kolmogorov.json")

        # Check if any class failed
        failed_classes = [name for name, res in results.items() if res['is_overfitted']]
        overall_min_p = min((res['p_value'] for res in results.values() if res['p_value'] is not None), default=None)
        is_overfitted = len(failed_classes) > 0

        json_data = {
            'model_idx': idx,
            'is_overfitted': is_overfitted,
            'threshold': self.p_threshold,
            'per_class_results': results,
            'overall_min_p_value': overall_min_p,
            'failed_classes': failed_classes
        }

        with open(json_file_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        logging.info(f"Saved JSON diagnostics to {json_file_path}")

    def check_overfitting(self, model_path, iteration, idx, output_dir):
        """
        Main method to check if model is overfitted.

        Args:
            model_path: Path to model checkpoint
            iteration: GA iteration number
            idx: Model index
            output_dir: Output directory for diagnostics

        Returns:
            is_overfitted: True if ANY class has min_p < threshold
        """
        logging.info(f"Checking overfitting for model {idx}...")

        # Load model and data
        model, train_loader, test_loader = self.load_model_and_data(model_path, iteration, idx)

        # Evaluate on train and test sets with weights
        logging.debug("Evaluating on train set...")
        y_true_train, y_scores_train, weights_train = self.evaluate_model(model, train_loader, self.device)

        logging.debug("Evaluating on test set...")
        y_true_test, y_scores_test, weights_test = self.evaluate_model(model, test_loader, self.device)

        # Create histograms for all classes with event weights
        train_hists_all = {}
        test_hists_all = {}

        logging.debug("Creating weighted score histograms...")
        for class_idx in range(self.num_classes):
            train_hists_all[class_idx] = self.create_score_histograms(
                y_true_train, y_scores_train, weights_train, class_idx, "train"
            )
            test_hists_all[class_idx] = self.create_score_histograms(
                y_true_test, y_scores_test, weights_test, class_idx, "test"
            )

        # Perform Kolmogorov tests
        logging.debug("Performing Kolmogorov-Smirnov tests...")
        results = self.perform_kolmogorov_tests(train_hists_all, test_hists_all)

        # Save diagnostics
        self.save_diagnostics(results, train_hists_all, test_hists_all, output_dir, idx)

        # Check if overfitted
        is_overfitted = any(res['is_overfitted'] for res in results.values())

        # Store results
        self.last_results = results

        if is_overfitted:
            failed_classes = [name for name, res in results.items() if res['is_overfitted']]
            logging.warning(f"Model {idx} is OVERFITTED (failed classes: {', '.join(failed_classes)})")
        else:
            logging.info(f"Model {idx} passed overfitting check")

        return is_overfitted
