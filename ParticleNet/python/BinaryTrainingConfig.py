#!/usr/bin/env python
"""
Binary training configuration for ParticleNet signal vs. single background training.

Extends the modular TrainingConfig to handle binary classification
with specialized background category handling.
"""

import argparse
import logging
from typing import Dict, List, Tuple

from TrainingConfig import TrainingConfig


class BinaryTrainingConfig(TrainingConfig):
    """
    Configuration manager for binary ParticleNet training (signal vs. single background).

    Extends TrainingConfig to handle binary classification scenarios
    where we train signal vs. one specific background category.
    """

    def __init__(self):
        super().__init__()
        self.background_category = ""

    def parse_arguments(self) -> argparse.Namespace:
        """Parse and validate command line arguments for binary training."""
        parser = argparse.ArgumentParser(description="Binary ParticleNet training (Signal vs. Single Background)")

        # Required arguments
        parser.add_argument("--signal", required=True, type=str,
                          help="Signal sample name without prefix (e.g., MHc130_MA100)")
        parser.add_argument("--background", required=True, type=str,
                          help="Background category (nonprompt, diboson, ttZ)")
        parser.add_argument("--channel", required=True, type=str,
                          help="Channel (e.g., Run1E2Mu, Run3Mu)")
        parser.add_argument("--fold", required=True, type=int,
                          help="Test fold (0-4)")

        # Sample prefixes
        parser.add_argument("--signal_prefix", default="TTToHcToWAToMuMu-", type=str,
                          help="Prefix for signal samples")

        # Training hyperparameters (same as multi-class)
        parser.add_argument("--max_epochs", default=81, type=int,
                          help="Maximum training epochs")
        parser.add_argument("--model", default="ParticleNet", type=str,
                          help="Model type (ParticleNet, ParticleNetV2, etc.)")
        parser.add_argument("--nNodes", default=128, type=int,
                          help="Number of hidden nodes")
        parser.add_argument("--dropout_p", default=0.25, type=float,
                          help="Dropout probability")

        # Optimization parameters (same as multi-class)
        parser.add_argument("--optimizer", default="Adam", type=str,
                          help="Optimizer (Adam, RMSprop, Adadelta)")
        parser.add_argument("--initLR", default=0.001, type=float,
                          help="Initial learning rate")
        parser.add_argument("--weight_decay", default=1e-4, type=float,
                          help="Weight decay")
        parser.add_argument("--scheduler", default="StepLR", type=str,
                          help="LR scheduler (StepLR, ExponentialLR, CyclicLR, ReduceLROnPlateau)")
        parser.add_argument("--loss_type", default="weighted_ce", type=str,
                          help="Loss function (weighted_ce, sample_normalized, focal)")

        # System parameters
        parser.add_argument("--device", default="cuda", type=str,
                          help="Device (cpu or cuda)")
        parser.add_argument("--pilot", action="store_true", default=False,
                          help="Use pilot datasets")
        parser.add_argument("--debug", action="store_true", default=False,
                          help="Debug mode")
        parser.add_argument("--balance", action="store_true", default=True,
                          help="Balance sample weights")
        parser.add_argument("--separate_bjets", action="store_true", default=False,
                          help="Use separate b-jets as distinct particles")

        self.args = parser.parse_args()
        self._validate_binary_arguments()
        self._process_binary_backgrounds()
        self._setup_paths()

        return self.args

    def _validate_binary_arguments(self) -> None:
        """Validate arguments specific to binary training."""
        # Channel validation
        valid_channels = ["Run1E2Mu", "Run3Mu", "Combined"]
        if self.args.channel not in valid_channels:
            raise ValueError(f"Invalid channel {self.args.channel}. Must be one of: {valid_channels}")

        # Fold validation
        if self.args.fold not in range(5):
            raise ValueError(f"Invalid fold {self.args.fold}, must be 0-4")

        # Background category validation
        valid_backgrounds = ["nonprompt", "diboson", "ttZ"]
        if self.args.background not in valid_backgrounds:
            raise ValueError(f"Invalid background category {self.args.background}. Must be one of: {valid_backgrounds}")

        # Model validation (same as multi-class)
        valid_models = ["ParticleNet", "ParticleNetV2", "OptimizedParticleNet",
                       "EfficientParticleNet", "EnhancedParticleNet", "HierarchicalParticleNet"]
        if self.args.model not in valid_models:
            raise ValueError(f"Invalid model {self.args.model}. Must be one of: {valid_models}")

        # Other validations (same as multi-class)
        valid_optimizers = ["Adam", "RMSprop", "Adadelta"]
        if self.args.optimizer not in valid_optimizers:
            raise ValueError(f"Invalid optimizer {self.args.optimizer}. Must be one of: {valid_optimizers}")

        valid_schedulers = ["StepLR", "ExponentialLR", "CyclicLR", "ReduceLROnPlateau"]
        if self.args.scheduler not in valid_schedulers:
            raise ValueError(f"Invalid scheduler {self.args.scheduler}. Must be one of: {valid_schedulers}")

        valid_losses = ["weighted_ce", "sample_normalized", "focal"]
        if self.args.loss_type not in valid_losses:
            raise ValueError(f"Invalid loss type {self.args.loss_type}. Must be one of: {valid_losses}")

    def _process_binary_backgrounds(self) -> None:
        """Process binary background configuration."""
        self.use_groups = False  # Binary training doesn't use groups
        self.background_category = self.args.background
        self.background_groups = {}  # Not used in binary mode
        self.backgrounds_list = [self.background_category]  # Single background category

        # Construct full sample names
        self.signal_full_name = self.args.signal_prefix + self.args.signal
        self.background_full_names = []  # Will be populated by data pipeline

        # Binary classification: signal (0) vs background (1)
        self.num_classes = 2

        logging.info(f"Binary training configuration:")
        logging.info(f"  Signal: {self.signal_full_name}")
        logging.info(f"  Background category: {self.background_category}")
        logging.info(f"  Classes: {self.num_classes} (signal=0, background=1)")

    def get_model_name(self) -> str:
        """Generate standardized model name for binary classification."""
        model_name = (f"{self.args.model}-nNodes{self.args.nNodes}-{self.args.optimizer}-"
                     f"initLR{str(format(self.args.initLR, '.4f')).replace('.','p')}-"
                     f"decay{str(format(self.args.weight_decay, '.5f')).replace('.', 'p')}-"
                     f"{self.args.scheduler}-{self.args.loss_type}-binary-{self.background_category}")

        return model_name

    def get_output_paths(self, model_name: str) -> Tuple[str, str, str, str]:
        """Get standardized output paths for binary model artifacts."""
        results_base = "results_bjets" if self.args.separate_bjets else "results"
        output_path = f"{self.workdir}/ParticleNet/{results_base}/{self.args.channel}/binary/{self.signal_full_name}/fold-{self.args.fold}"
        if self.args.pilot:
            output_path = f"{self.workdir}/ParticleNet/{results_base}/{self.args.channel}/binary/{self.signal_full_name}/pilot"

        checkpoint_path = f"{output_path}/models/{model_name}.pt"
        summary_path = f"{output_path}/CSV/{model_name}.csv"
        tree_path = f"{output_path}/trees/{model_name}.root"

        return output_path, checkpoint_path, summary_path, tree_path

    def get_background_category(self) -> str:
        """Get the background category for binary training."""
        return self.background_category

    def log_configuration(self) -> None:
        """Log comprehensive binary training configuration."""
        logging.info("=" * 60)
        logging.info("BINARY TRAINING CONFIGURATION")
        logging.info("=" * 60)
        logging.info(f"Signal: {self.signal_full_name}")
        logging.info(f"Background: {self.background_category}")
        logging.info(f"Channel: {self.args.channel}, Fold: {self.args.fold}")
        logging.info(f"Classes: {self.num_classes} (binary classification)")

        logging.info(f"Model: {self.args.model} ({self.args.nNodes} nodes, {self.args.dropout_p:.2f} dropout)")
        logging.info(f"Optimization: {self.args.optimizer} (LR: {self.args.initLR}, decay: {self.args.weight_decay})")
        logging.info(f"Schedule: {self.args.scheduler}, Loss: {self.args.loss_type}")
        logging.info(f"Device: {self.args.device}")
        logging.info(f"Pilot mode: {self.args.pilot}, Balance: {self.args.balance}")
        logging.info(f"Separate b-jets: {self.args.separate_bjets}")
        logging.info("=" * 60)

        logging.info("Using BINARY classification:")
        logging.info("  - Signal detection: standard weighted accuracy")
        logging.info("  - Background rejection: single category focus")
        logging.info(f"  - Target: signal vs. {self.background_category}")


def create_binary_training_config() -> BinaryTrainingConfig:
    """Factory function to create and initialize binary training configuration."""
    config = BinaryTrainingConfig()
    config.parse_arguments()
    return config