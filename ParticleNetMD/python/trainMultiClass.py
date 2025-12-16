#!/usr/bin/env python
"""
Multi-class training script for Mass-Decorrelated ParticleNet.

Trains a 4-class classifier (signal vs 3 backgrounds) using DisCo loss
(Distance Correlation regularization) for mass decorrelation.

Key features:
- Default loss_type is 'disco' with event-weighted DisCo regularization
- Dataset includes mass1, mass2 (OS muon pair masses) for decorrelation
- Node features: [E, Px, Py, Pz, charge, isMuon, isElectron, isJet, isBjet]
"""

import argparse
import logging
import os
from typing import Dict, List, Tuple

from SglConfig import load_sgl_config
from DataPipeline import create_data_pipeline
from TrainingOrchestrator import create_training_orchestrator
from ResultPersistence import create_result_persistence


class Config:
    """
    Simple configuration container that mimics TrainingConfig structure.

    Provides compatibility with DataPipeline, TrainingOrchestrator, and ResultPersistence
    which expect a config object with args namespace and helper properties.
    """

    def __init__(self, signal: str, channel: str, config_path: str = None):
        """Initialize configuration from command-line args and JSON config."""
        # Load JSON configuration
        self.sgl_config = load_sgl_config(config_path)

        # Get all config sections
        train_params = self.sgl_config.get_training_parameters()
        model_config = self.sgl_config.get_model_config()
        optim_config = self.sgl_config.get_optimization_config()
        bg_config = self.sgl_config.get_background_config()
        dataset_config = self.sgl_config.get_dataset_config()
        system_config = self.sgl_config.get_system_config()
        output_config = self.sgl_config.get_output_config()

        # Create args namespace compatible with existing modules
        self.args = argparse.Namespace()
        self.args.signal = signal
        self.args.channel = channel

        # Training parameters - handle folds
        self.args.train_folds = train_params['train_folds']
        self.args.valid_folds = train_params['valid_folds']
        self.args.test_folds = train_params['test_folds']
        # For compatibility, create a single 'fold' for output paths (use first test fold)
        self.args.fold = train_params['test_folds'][0] if train_params['test_folds'] else 4
        self.args.max_epochs = train_params['max_epochs']
        self.args.batch_size = train_params['batch_size']
        self.args.dropout_p = train_params['dropout_p']
        self.args.early_stopping_patience = train_params['early_stopping_patience']
        self.args.loss_type = train_params.get('loss_type', 'disco')  # Default: disco
        self.args.balance = train_params['balance_weights']
        self.args.max_events_per_fold_per_class = train_params.get('max_events_per_fold_per_class')

        # DisCo parameters (for mass decorrelation)
        disco_params = self.sgl_config.config.get('disco_parameters', {})
        self.args.disco_lambda = disco_params.get('disco_lambda', 0.1)

        # Model configuration
        self.args.model = model_config['default_model']
        self.args.nNodes = model_config['nNodes']

        # Optimization configuration
        self.args.optimizer = optim_config['optimizer']
        self.args.initLR = optim_config['initLR']
        self.args.weight_decay = optim_config['weight_decay']
        self.args.scheduler = optim_config['scheduler']

        # Dataset configuration
        self.args.signal_prefix = dataset_config['signal_prefix']
        self.args.background_prefix = dataset_config['background_prefix']

        # System configuration
        self.args.device = system_config['device']
        self.args.pilot = system_config['pilot']
        self.args.debug = system_config['debug']

        # Output configuration - ParticleNetMD always uses "results" directory
        self.args.results_dir = output_config.get('results_dir', 'results')

        # Process background configuration
        self.use_groups = (bg_config['mode'] == 'groups')
        if self.use_groups:
            self.background_groups = bg_config['background_groups']
            # Create flat list of all background samples
            self.backgrounds_list = []
            for samples in self.background_groups.values():
                self.backgrounds_list.extend(samples)
        else:
            self.background_groups = {}
            self.backgrounds_list = bg_config['backgrounds_list']

        # Construct full sample names with prefixes
        self.signal_full_name = self.args.signal_prefix + self.args.signal
        self.background_full_names = [
            self.args.background_prefix + bg for bg in self.backgrounds_list
        ]

        # Calculate number of classes
        if self.use_groups:
            self.num_classes = 1 + len(self.background_groups)
        else:
            self.num_classes = 1 + len(self.backgrounds_list)

        # Validate environment
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

        model_name = (
            f"{self.args.model}-nNodes{self.args.nNodes}-{self.args.optimizer}-"
            f"initLR{str(format(self.args.initLR, '.4f')).replace('.','p')}-"
            f"decay{str(format(self.args.weight_decay, '.5f')).replace('.', 'p')}-"
            f"{self.args.scheduler}-{self.args.loss_type}-{bg_identifier}"
        )

        return model_name

    def get_output_paths(self, model_name: str) -> Tuple[str, str, str, str]:
        """Get standardized output paths for model artifacts."""
        # Use results_dir from config
        results_dir = self.args.results_dir
        output_path = f"{self.workdir}/ParticleNetMD/{results_dir}/{self.args.channel}/multiclass/{self.signal_full_name}/fold-{self.args.fold}"
        if self.args.pilot:
            output_path = f"{self.workdir}/ParticleNetMD/{results_dir}/{self.args.channel}/multiclass/{self.signal_full_name}/pilot"

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
            background_groups_full[group_name] = [
                self.args.background_prefix + sample for sample in samples
            ]

        return background_groups_full

    def log_configuration(self) -> None:
        """Log comprehensive configuration information."""
        logging.info("=" * 60)
        logging.info("TRAINING CONFIGURATION")
        logging.info("=" * 60)
        logging.info(f"Signal: {self.signal_full_name}")
        logging.info(f"Channel: {self.args.channel}")
        logging.info(f"Train folds: {self.args.train_folds}, Valid folds: {self.args.valid_folds}, Test folds: {self.args.test_folds}")

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
        if self.args.loss_type == 'disco':
            logging.info(f"DisCo lambda: {self.args.disco_lambda}")
        logging.info(f"Classes: {self.num_classes}, Device: {self.args.device}")
        logging.info(f"Batch size: {self.args.batch_size}, Max epochs: {self.args.max_epochs}")
        logging.info(f"Pilot mode: {self.args.pilot}, Balance: {self.args.balance}")
        logging.info("=" * 60)

        # Log accuracy calculation method
        if self.use_groups:
            logging.info("Using GROUP-BALANCED accuracy calculation:")
            logging.info("  - Within each group: accuracy weighted by sample weights")
            logging.info(f"  - Between groups: each group contributes equally (1/{self.num_classes})")
            logging.info(f"  - Groups: {list(self.background_groups.keys())}")
        else:
            logging.info("Using STANDARD weighted accuracy calculation")


def parse_arguments() -> Tuple[str, str, str]:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Multi-class ParticleNet training")

    # Required arguments
    parser.add_argument("--signal", required=True, type=str,
                       help="Signal sample name without prefix (e.g., MHc130_MA100)")
    parser.add_argument("--channel", required=True, type=str,
                       help="Channel (e.g., Run1E2Mu, Run3Mu)")

    # Optional config file path
    parser.add_argument("--config", default=None, type=str,
                       help="Path to configuration JSON file (default: configs/SglConfig.json)")

    args = parser.parse_args()

    # Validate channel
    valid_channels = ["Run1E2Mu", "Run3Mu", "Combined"]
    if args.channel not in valid_channels:
        raise ValueError(f"Invalid channel {args.channel}. Must be one of: {valid_channels}")

    return args.signal, args.channel, args.config


def main():
    """
    Main training function using modular architecture.

    Orchestrates the complete training pipeline through specialized modules:
    - Config: JSON configuration loading and management
    - DataPipeline: Dataset creation and data loader management
    - TrainingOrchestrator: Training loop and metrics collection
    - ResultPersistence: Model saving and output management
    """

    # 1. Parse command-line arguments
    signal, channel, config_path = parse_arguments()

    # 2. Load configuration from JSON
    config = Config(signal, channel, config_path)

    # Setup logging based on debug flag
    logging.basicConfig(
        level=logging.DEBUG if config.args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Log comprehensive configuration
    config.log_configuration()

    try:
        # 3. Data Pipeline Setup
        logging.info("Setting up data pipeline...")
        data_pipeline = create_data_pipeline(config)

        # Create datasets
        data_pipeline.create_datasets()
        data_pipeline.log_dataset_info()

        # Create data loaders with batch size from config
        data_pipeline.create_data_loaders(batch_size=config.args.batch_size)
        data_pipeline.validate_data_integrity()

        # Log sample batch information
        batch_info = data_pipeline.get_sample_batch_info()
        logging.info(f"Sample batch: {batch_info['batch_size']} events, "
                    f"{batch_info['num_nodes']} nodes, "
                    f"{batch_info['node_features']} node features")
        logging.info(f"Label distribution: {batch_info['label_distribution']}")
        if batch_info['has_weights']:
            logging.info(f"Weight range: [{batch_info['weight_range'][0]:.6f}, "
                        f"{batch_info['weight_range'][1]:.6f}]")

        # 4. Training Orchestration Setup
        logging.info("Setting up training orchestration...")
        orchestrator = create_training_orchestrator(config, data_pipeline)

        # Generate model name and output paths
        model_name = config.get_model_name()
        output_paths = config.get_output_paths(model_name)
        output_path, checkpoint_path, summary_path, tree_path = output_paths

        logging.info(f"Model: {model_name}")
        logging.info(f"Output: {output_path}")

        # 5. Result Persistence Setup
        persistence = create_result_persistence(config)
        persistence.create_output_directories(output_paths)
        persistence.log_output_paths(output_paths, model_name)

        # 6. Training Infrastructure Setup
        orchestrator.setup_training_infrastructure(model_name, checkpoint_path)

        # 7. Execute Training
        logging.info("Starting training process...")
        training_results = orchestrator.train()

        # 8. Final Evaluation
        test_results = orchestrator.evaluate_final_performance()
        training_results.update(test_results)

        # 9. B-Jet Subset Evaluation
        logging.info("Evaluating performance on b-jet subset...")
        bjet_results = orchestrator.evaluate_bjet_subset_performance()
        training_results['bjet_subset_results'] = bjet_results

        # 10. Save Training Summary
        orchestrator.save_training_summary(summary_path)

        # 11. Save Predictions to ROOT Tree
        model = orchestrator.get_model()
        device = orchestrator.device
        persistence.save_predictions_to_root(model, data_pipeline, device, tree_path)

        # 12. Save Performance Metrics
        persistence.save_performance_summary(training_results, model_name, output_path)
        persistence.save_model_info(model, model_name, output_path)

        # 13. Save GA-compatible JSON (for visualizeGAIteration.py compatibility)
        persistence.save_ga_compatible_json(training_results, model_name, output_path)

        # Final Summary
        logging.info("=" * 60)
        logging.info("TRAINING COMPLETED SUCCESSFULLY!")
        logging.info("=" * 60)
        logging.info(f"Final test accuracy: {training_results['test_accuracy']*100:.2f}%")
        logging.info(f"Total training time: {training_results['total_training_time']:.1f} seconds")
        logging.info(f"Epochs completed: {training_results['epochs_completed']}")
        logging.info(f"Early stopped: {'Yes' if training_results['early_stopped'] else 'No'}")
        logging.info(f"Model parameters: {training_results['model_parameters']:,}")
        logging.info(f"Results saved to: {output_path}")
        logging.info("=" * 60)

    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()