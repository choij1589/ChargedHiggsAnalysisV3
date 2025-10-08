#!/usr/bin/env python
"""
Training configuration management for ParticleNet multi-class training.

Handles argument parsing, validation, and configuration organization
for both grouped and individual background training modes.
"""

import os
import argparse
import logging
from typing import Dict, List, Tuple


class TrainingConfig:
    """
    Manages training configuration and validation for multi-class ParticleNet training.

    Handles both grouped backgrounds and individual background modes,
    with comprehensive validation and path management.
    """

    def __init__(self):
        self.args = None
        self.use_groups = False
        self.background_groups = {}
        self.backgrounds_list = []
        self.signal_full_name = ""
        self.background_full_names = []
        self.num_classes = 0
        self.workdir = ""

    def parse_arguments(self) -> argparse.Namespace:
        """Parse and validate command line arguments."""
        parser = argparse.ArgumentParser(description="Multi-class ParticleNet training")

        # Required arguments
        parser.add_argument("--signal", required=True, type=str,
                          help="Signal sample name without prefix (e.g., MHc130_MA100)")
        parser.add_argument("--channel", required=True, type=str,
                          help="Channel (e.g., Run1E2Mu, Run3Mu)")
        parser.add_argument("--fold", required=True, type=int,
                          help="Test fold (0-4)")

        # Background specification (mutually exclusive)
        parser.add_argument("--backgrounds", nargs='+', type=str,
                          help="Background sample names without prefix")
        parser.add_argument("--background_groups", nargs='+', type=str,
                          help="Background groups in format 'groupname:sample1,sample2,...'")

        # Sample prefixes
        parser.add_argument("--signal_prefix", default="TTToHcToWAToMuMu-", type=str,
                          help="Prefix for signal samples")
        parser.add_argument("--background_prefix", default="Skim_TriLep_", type=str,
                          help="Prefix for background samples")

        # Training hyperparameters
        parser.add_argument("--max_epochs", default=81, type=int,
                          help="Maximum training epochs")
        parser.add_argument("--model", default="ParticleNet", type=str,
                          help="Model type (ParticleNet, ParticleNetV2, etc.)")
        parser.add_argument("--nNodes", default=128, type=int,
                          help="Number of hidden nodes")
        parser.add_argument("--dropout_p", default=0.25, type=float,
                          help="Dropout probability")

        # Optimization parameters
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
        self._validate_arguments()
        self._process_backgrounds()
        self._setup_paths()

        return self.args

    def _validate_arguments(self) -> None:
        """Validate parsed arguments."""
        # Background specification validation
        if not self.args.backgrounds and not self.args.background_groups:
            raise ValueError("Either --backgrounds or --background_groups must be provided")
        if self.args.backgrounds and self.args.background_groups:
            raise ValueError("Cannot specify both --backgrounds and --background_groups")

        # Channel validation
        valid_channels = ["Run1E2Mu", "Run3Mu", "Combined"]
        if self.args.channel not in valid_channels:
            raise ValueError(f"Invalid channel {self.args.channel}. Must be one of: {valid_channels}")

        # Fold validation
        if self.args.fold not in range(5):
            raise ValueError(f"Invalid fold {self.args.fold}, must be 0-4")

        # Model validation
        valid_models = ["ParticleNet", "ParticleNetV2", "OptimizedParticleNet",
                       "EfficientParticleNet", "EnhancedParticleNet", "HierarchicalParticleNet"]
        if self.args.model not in valid_models:
            raise ValueError(f"Invalid model {self.args.model}. Must be one of: {valid_models}")

        # Optimizer validation
        valid_optimizers = ["Adam", "RMSprop", "Adadelta"]
        if self.args.optimizer not in valid_optimizers:
            raise ValueError(f"Invalid optimizer {self.args.optimizer}. Must be one of: {valid_optimizers}")

        # Scheduler validation
        valid_schedulers = ["StepLR", "ExponentialLR", "CyclicLR", "ReduceLROnPlateau"]
        if self.args.scheduler not in valid_schedulers:
            raise ValueError(f"Invalid scheduler {self.args.scheduler}. Must be one of: {valid_schedulers}")

        # Loss function validation
        valid_losses = ["weighted_ce", "sample_normalized", "focal"]
        if self.args.loss_type not in valid_losses:
            raise ValueError(f"Invalid loss type {self.args.loss_type}. Must be one of: {valid_losses}")

    def _process_backgrounds(self) -> None:
        """Process background groups or individual backgrounds."""
        self.use_groups = bool(self.args.background_groups)
        self.background_groups = {}
        self.backgrounds_list = []

        if self.use_groups:
            # Parse background groups
            for group_arg in self.args.background_groups:
                if ':' not in group_arg:
                    raise ValueError(f"Invalid group format '{group_arg}'. Expected 'groupname:sample1,sample2,...'")

                group_name, samples_str = group_arg.split(':', 1)
                samples = [s.strip() for s in samples_str.split(',')]

                if not group_name or not samples:
                    raise ValueError(f"Invalid group format '{group_arg}'. Group name and samples are required.")

                self.background_groups[group_name] = samples
                self.backgrounds_list.extend(samples)

            logging.info(f"Using background groups: {list(self.background_groups.keys())}")
            for group_name, samples in self.background_groups.items():
                logging.info(f"  {group_name}: {samples}")
        else:
            # Handle individual backgrounds (backward compatibility)
            for bg_arg in self.args.backgrounds:
                if ' ' in bg_arg:
                    self.backgrounds_list.extend(bg_arg.split())
                else:
                    self.backgrounds_list.append(bg_arg)

            logging.info(f"Using individual background samples: {self.backgrounds_list}")

        # Construct full sample names with prefixes
        self.signal_full_name = self.args.signal_prefix + self.args.signal
        self.background_full_names = [self.args.background_prefix + bg for bg in self.backgrounds_list]

        # Calculate number of classes
        if self.use_groups:
            self.num_classes = 1 + len(self.background_groups)
        else:
            self.num_classes = 1 + len(self.backgrounds_list)

    def _setup_paths(self) -> None:
        """Setup working directory and validate environment."""
        try:
            self.workdir = os.environ["WORKDIR"]
        except KeyError:
            raise EnvironmentError("WORKDIR environment variable not set. Run 'source setup.sh' first.")

    def get_model_name(self) -> str:
        """Generate standardized model name for file naming."""
        if self.use_groups:
            bg_identifier = f"{len(self.background_groups)}grp-" + "-".join(self.background_groups.keys())
        else:
            bg_identifier = f"{len(self.backgrounds_list)}bg"

        model_name = (f"{self.args.model}-nNodes{self.args.nNodes}-{self.args.optimizer}-"
                     f"initLR{str(format(self.args.initLR, '.4f')).replace('.','p')}-"
                     f"decay{str(format(self.args.weight_decay, '.5f')).replace('.', 'p')}-"
                     f"{self.args.scheduler}-{self.args.loss_type}-{bg_identifier}")

        return model_name

    def get_output_paths(self, model_name: str) -> Tuple[str, str, str, str]:
        """Get standardized output paths for model artifacts."""
        results_dir = "results_bjets" if self.args.separate_bjets else "results"
        output_path = f"{self.workdir}/ParticleNet/{results_dir}/{self.args.channel}/multiclass/{self.signal_full_name}/fold-{self.args.fold}"
        if self.args.pilot:
            output_path = f"{self.workdir}/ParticleNet/{results_dir}/{self.args.channel}/multiclass/{self.signal_full_name}/pilot"

        checkpoint_path = f"{output_path}/models/{model_name}.pt"
        summary_path = f"{output_path}/CSV/{model_name}.csv"
        tree_path = f"{output_path}/trees/{model_name}.root"

        return output_path, checkpoint_path, summary_path, tree_path

    def get_background_groups_full(self) -> Dict[str, List[str]]:
        """Get background groups with full sample names (including prefixes)."""
        if not self.use_groups:
            return {}

        background_groups_full = {}
        for group_name, samples in self.background_groups.items():
            background_groups_full[group_name] = [self.args.background_prefix + sample for sample in samples]

        return background_groups_full

    def log_configuration(self) -> None:
        """Log comprehensive configuration information."""
        logging.info("=" * 60)
        logging.info("TRAINING CONFIGURATION")
        logging.info("=" * 60)
        logging.info(f"Signal: {self.signal_full_name}")
        logging.info(f"Channel: {self.args.channel}, Fold: {self.args.fold}")

        if self.use_groups:
            logging.info(f"Background mode: GROUPED ({len(self.background_groups)} groups)")
            for group_name, samples in self.background_groups.items():
                logging.info(f"  {group_name}: {samples}")
        else:
            logging.info(f"Background mode: INDIVIDUAL ({len(self.backgrounds_list)} samples)")
            logging.info(f"  Samples: {self.backgrounds_list}")

        logging.info(f"Model: {self.args.model} ({self.args.nNodes} nodes, {self.args.dropout_p:.2f} dropout)")
        logging.info(f"Optimization: {self.args.optimizer} (LR: {self.args.initLR}, decay: {self.args.weight_decay})")
        logging.info(f"Schedule: {self.args.scheduler}, Loss: {self.args.loss_type}")
        logging.info(f"Classes: {self.num_classes}, Device: {self.args.device}")
        logging.info(f"Pilot mode: {self.args.pilot}, Balance: {self.args.balance}")
        logging.info(f"Separate b-jets: {self.args.separate_bjets}")
        logging.info("=" * 60)

        # Log accuracy calculation method
        if self.use_groups:
            logging.info("Using GROUP-BALANCED accuracy calculation:")
            logging.info("  - Within each group: accuracy weighted by sample weights")
            logging.info(f"  - Between groups: each group contributes equally (1/{self.num_classes})")
            logging.info(f"  - Groups: {list(self.background_groups.keys())}")
        else:
            logging.info("Using STANDARD weighted accuracy calculation")


def create_training_config() -> TrainingConfig:
    """Factory function to create and initialize training configuration."""
    config = TrainingConfig()
    config.parse_arguments()
    return config