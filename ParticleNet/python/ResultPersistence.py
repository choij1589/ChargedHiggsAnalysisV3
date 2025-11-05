#!/usr/bin/env python
"""
Result persistence manager for ParticleNet multi-class training.

Handles saving of model predictions, performance metrics, and training artifacts
to ROOT files and JSON formats with support for both grouped and individual backgrounds.
"""

import os
import json
import logging
from array import array
from typing import Dict, Any, List

import torch
import ROOT

from DataPipeline import DataPipeline
from TrainingUtilities import extract_weights_from_batch
from BjetSubsetUtils import has_bjet_in_event


class ResultPersistence:
    """
    Manages persistence of training results and model predictions.

    Handles ROOT tree creation, prediction saving, and performance metrics
    persistence with dynamic branch creation for different background modes.
    """

    def __init__(self, config):
        """
        Initialize result persistence manager.

        Args:
            config: Configuration object with args namespace and sample information
        """
        self.config = config

    def save_predictions_to_root(self, model: torch.nn.Module, data_pipeline: DataPipeline,
                                device: torch.device, tree_path: str) -> None:
        """
        Save model predictions to ROOT tree following V2 pattern.

        Args:
            model: Trained PyTorch model
            data_pipeline: Data pipeline with train/valid/test loaders
            device: Device for model inference
            tree_path: Path to save ROOT file
        """
        logging.info("Saving score distributions to ROOT tree...")
        model.eval()

        # Create output directory
        os.makedirs(os.path.dirname(tree_path), exist_ok=True)

        f = ROOT.TFile(tree_path, "RECREATE")
        tree = ROOT.TTree("Events", "Multi-class training results")

        # Define branches dynamically based on background configuration
        score_branches = {}
        score_arrays = {}

        # Signal score (always present)
        score_arrays['signal'] = array("f", [0.])
        tree.Branch("score_signal", score_arrays['signal'], "score_signal/F")

        # Background scores (dynamic based on groups or individual backgrounds)
        if self.config.use_groups:
            self._create_grouped_score_branches(tree, score_arrays)
        else:
            self._create_individual_score_branches(tree, score_arrays)

        # Metadata branches
        true_label = array("i", [0])
        train_mask = array("B", [False])
        valid_mask = array("B", [False])
        test_mask = array("B", [False])
        has_bjet = array("B", [False])

        tree.Branch("true_label", true_label, "true_label/I")
        tree.Branch("train_mask", train_mask, "train_mask/O")
        tree.Branch("valid_mask", valid_mask, "valid_mask/O")
        tree.Branch("test_mask", test_mask, "test_mask/O")
        tree.Branch("has_bjet", has_bjet, "has_bjet/O")

        # Weight branches for class imbalance analysis
        event_weight = array("f", [0.])
        tree.Branch("weight", event_weight, "weight/F")

        # Save predictions for all data splits
        self._save_split_predictions(model, data_pipeline.train_loader, device, tree,
                                   score_arrays, true_label, event_weight, has_bjet,
                                   train_mask, valid_mask, test_mask,
                                   is_train=True)

        self._save_split_predictions(model, data_pipeline.valid_loader, device, tree,
                                   score_arrays, true_label, event_weight, has_bjet,
                                   train_mask, valid_mask, test_mask,
                                   is_valid=True)

        self._save_split_predictions(model, data_pipeline.test_loader, device, tree,
                                   score_arrays, true_label, event_weight, has_bjet,
                                   train_mask, valid_mask, test_mask,
                                   is_test=True)

        # Write and close file
        f.cd()
        tree.Write()
        f.Close()

        logging.info(f"Predictions saved to: {tree_path}")

    def _create_grouped_score_branches(self, tree: ROOT.TTree, score_arrays: Dict[str, array]) -> None:
        """Create score branches for grouped backgrounds."""
        for group_name in self.config.background_groups.keys():
            branch_name = group_name
            score_arrays[branch_name] = array("f", [0.])
            tree.Branch(f"score_{branch_name}", score_arrays[branch_name], f"score_{branch_name}/F")
            logging.debug(f"Created score branch: score_{branch_name}")

    def _create_individual_score_branches(self, tree: ROOT.TTree, score_arrays: Dict[str, array]) -> None:
        """Create score branches for individual backgrounds (backward compatibility)."""
        for i, bg_name in enumerate(self.config.backgrounds_list):
            branch_name = self._get_background_branch_name(bg_name)
            score_arrays[branch_name] = array("f", [0.])
            tree.Branch(f"score_{branch_name}", score_arrays[branch_name], f"score_{branch_name}/F")
            logging.debug(f"Created score branch: score_{branch_name} (from {bg_name})")

    def _get_background_branch_name(self, bg_name: str) -> str:
        """Generate standardized branch names from background sample names."""
        # Map background sample names to short branch names
        name_mapping = {
            "TTLL": "ttll",
            "WZTo3LNu": "wz",
            "ZZTo4L": "zz",
            "TTZToLLNuNu": "ttz",
            "TTWToLNu": "ttw",
            "tZq": "tzq",
            "DYJets10to50": "dy10to50",
            "DYJets": "dy"
        }

        for key, branch_name in name_mapping.items():
            if key in bg_name:
                return branch_name

        # Fallback: use index-based naming
        bg_index = self.config.backgrounds_list.index(bg_name) if bg_name in self.config.backgrounds_list else 0
        return f"bg{bg_index + 1}"

    def _save_split_predictions(self, model: torch.nn.Module, data_loader,
                               device: torch.device, tree: ROOT.TTree,
                               score_arrays: Dict[str, array], true_label: array,
                               event_weight: array, has_bjet: array,
                               train_mask: array, valid_mask: array, test_mask: array,
                               is_train: bool = False, is_valid: bool = False,
                               is_test: bool = False) -> None:
        """Save predictions for a specific data split."""
        # Reset all masks first, then set the active one
        train_mask[0] = False
        valid_mask[0] = False
        test_mask[0] = False

        # Set split masks
        train_mask[0] = is_train
        valid_mask[0] = is_valid
        test_mask[0] = is_test

        split_name = "train" if is_train else ("valid" if is_valid else "test")
        logging.info(f"Saving {split_name} predictions...")

        with torch.no_grad():
            for batch_idx, batch in enumerate(data_loader):
                batch = batch.to(device)
                logits = model(batch.x, batch.edge_index, batch.graphInput, batch.batch)

                # Apply softmax to get probabilities (model returns logits)
                out = torch.softmax(logits, dim=1)

                # Extract weights from batch
                batch_weights = extract_weights_from_batch(batch)

                # Extract node features for each event in batch
                # batch.batch indicates which nodes belong to which event
                batch_node_indices = batch.batch.cpu()

                for i, (label, scores) in enumerate(zip(batch.y, out)):
                    true_label[0] = int(label.cpu().numpy())

                    # Save signal score (always first)
                    score_arrays['signal'][0] = float(scores[0].cpu().numpy())

                    # Save background scores dynamically
                    bg_score_keys = list(score_arrays.keys())[1:]  # Skip signal
                    for j, score_key in enumerate(bg_score_keys):
                        score_arrays[score_key][0] = float(scores[j + 1].cpu().numpy())

                    # Save weight information (final training weight after normalization)
                    event_weight[0] = float(batch_weights[i].cpu().numpy())

                    # Calculate and save has_bjet flag
                    # Extract node features for this event
                    event_node_mask = (batch_node_indices == i)
                    event_node_features = batch.x[event_node_mask]
                    has_bjet[0] = has_bjet_in_event(event_node_features,
                                                    separate_bjets=self.config.args.separate_bjets)

                    tree.Fill()

                if batch_idx % 100 == 0:
                    logging.debug(f"Processed {batch_idx + 1}/{len(data_loader)} batches for {split_name}")

    def save_performance_summary(self, training_results: Dict[str, Any],
                                model_name: str, output_path: str) -> None:
        """
        Save comprehensive performance summary to JSON.

        Args:
            training_results: Results from training orchestrator
            model_name: Model name for file naming
            output_path: Base output directory
        """
        # Create performance summary
        perf_summary = {
            'model_name': model_name,
            'signal_sample': self.config.signal_full_name,
            'channel': self.config.args.channel,
            'fold': self.config.args.fold,
            'pilot_mode': self.config.args.pilot,
            'training_config': {
                'model_type': self.config.args.model,
                'num_classes': self.config.num_classes,
                'use_groups': self.config.use_groups,
                'optimizer': self.config.args.optimizer,
                'scheduler': self.config.args.scheduler,
                'loss_type': self.config.args.loss_type,
                'initial_lr': self.config.args.initLR,
                'weight_decay': self.config.args.weight_decay,
                'dropout_p': self.config.args.dropout_p,
                'max_epochs': self.config.args.max_epochs,
                'batch_size': 1024  # Default batch size
            },
            'background_info': self._get_background_info(),
            'training_results': training_results
        }

        # Save performance summary
        perf_path = os.path.join(output_path, f"{model_name}_performance.json")
        os.makedirs(os.path.dirname(perf_path), exist_ok=True)

        with open(perf_path, 'w') as f:
            json.dump(perf_summary, f, indent=2)

        logging.info(f"Performance metrics saved to: {perf_path}")

    def _get_background_info(self) -> Dict[str, Any]:
        """Get background configuration information."""
        if self.config.use_groups:
            return {
                'mode': 'grouped',
                'num_groups': len(self.config.background_groups),
                'groups': self.config.background_groups
            }
        else:
            return {
                'mode': 'individual',
                'num_backgrounds': len(self.config.backgrounds_list),
                'backgrounds': self.config.backgrounds_list
            }

    def save_model_info(self, model: torch.nn.Module, model_name: str, output_path: str) -> None:
        """
        Save detailed model architecture information.

        Args:
            model: PyTorch model
            model_name: Model name
            output_path: Output directory
        """
        model_info = {
            'model_name': model_name,
            'model_type': self.config.args.model,
            'num_parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'num_classes': self.config.num_classes,
            'input_features': {
                'node_features': 9,
                'graph_features': 4
            },
            'architecture': {
                'hidden_nodes': self.config.args.nNodes,
                'dropout_p': self.config.args.dropout_p
            },
            'training_mode': 'grouped' if self.config.use_groups else 'individual'
        }

        info_path = os.path.join(output_path, f"{model_name}_model_info.json")
        os.makedirs(os.path.dirname(info_path), exist_ok=True)

        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        logging.info(f"Model info saved to: {info_path}")

    def create_output_directories(self, output_paths: tuple) -> None:
        """
        Create all necessary output directories.

        Args:
            output_paths: Tuple of (output_path, checkpoint_path, summary_path, tree_path)
        """
        output_path, checkpoint_path, summary_path, tree_path = output_paths

        directories_to_create = [
            os.path.dirname(checkpoint_path),
            os.path.dirname(summary_path),
            os.path.dirname(tree_path),
            output_path  # Base output directory
        ]

        for directory in directories_to_create:
            os.makedirs(directory, exist_ok=True)

        logging.info(f"Output directories created under: {output_path}")

    def log_output_paths(self, output_paths: tuple, model_name: str) -> None:
        """
        Log all output file paths for reference.

        Args:
            output_paths: Tuple of output paths
            model_name: Model name
        """
        output_path, checkpoint_path, summary_path, tree_path = output_paths

        logging.info("=" * 60)
        logging.info("OUTPUT FILE PATHS")
        logging.info("=" * 60)
        logging.info(f"Base directory: {output_path}")
        logging.info(f"Model checkpoint: {checkpoint_path}")
        logging.info(f"Training summary: {summary_path}")
        logging.info(f"Predictions tree: {tree_path}")
        logging.info(f"Performance JSON: {os.path.join(output_path, f'{model_name}_performance.json')}")
        logging.info(f"Model info JSON: {os.path.join(output_path, f'{model_name}_model_info.json')}")
        logging.info("=" * 60)


def create_result_persistence(config) -> ResultPersistence:
    """
    Factory function to create result persistence manager.

    Args:
        config: Configuration object with args namespace and sample information

    Returns:
        Initialized ResultPersistence instance
    """
    return ResultPersistence(config)