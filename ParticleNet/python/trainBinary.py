#!/usr/bin/env python
"""
Binary training script for ParticleNet (Signal vs. Single Background).

Trains a 2-class classifier (signal vs single background) using the same
modular architecture as multi-class training for direct comparison studies.
"""

import logging
from BinaryTrainingConfig import create_binary_training_config
from BinaryDataPipeline import create_binary_data_pipeline
from TrainingOrchestrator import create_training_orchestrator
from ResultPersistence import create_result_persistence


def main():
    """
    Main binary training function using modular architecture.

    Orchestrates the complete binary training pipeline through specialized modules:
    - BinaryTrainingConfig: Binary-specific configuration management
    - BinaryDataPipeline: Signal vs. single background dataset creation
    - TrainingOrchestrator: Training loop (reused from multi-class)
    - ResultPersistence: Model saving and output management (reused)
    """

    # 1. Binary Configuration Management
    config = create_binary_training_config()

    # Setup logging based on debug flag
    logging.basicConfig(
        level=logging.DEBUG if config.args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Log comprehensive binary configuration
    config.log_configuration()

    try:
        # 2. Binary Data Pipeline Setup
        logging.info("Setting up binary data pipeline...")
        data_pipeline = create_binary_data_pipeline(config)

        # Create binary datasets
        data_pipeline.create_datasets()
        data_pipeline.log_dataset_info()

        # Create data loaders
        data_pipeline.create_data_loaders(batch_size=1024)
        data_pipeline.validate_data_integrity()

        # Log sample batch information
        batch_info = data_pipeline.get_sample_batch_info()
        logging.info(f"Sample batch: {batch_info['batch_size']} events, "
                    f"{batch_info['num_nodes']} nodes, "
                    f"{batch_info['node_features']} node features")
        logging.info(f"Binary distribution: Signal={batch_info['signal_count']}, "
                    f"Background={batch_info['background_count']}")
        if batch_info['has_weights']:
            logging.info(f"Weight range: [{batch_info['weight_range'][0]:.6f}, "
                        f"{batch_info['weight_range'][1]:.6f}]")

        # 3. Training Orchestration Setup (reuse multi-class orchestrator)
        logging.info("Setting up binary training orchestration...")
        orchestrator = create_training_orchestrator(config, data_pipeline)

        # Generate model name and output paths
        model_name = config.get_model_name()
        output_paths = config.get_output_paths(model_name)
        output_path, checkpoint_path, summary_path, tree_path = output_paths

        logging.info(f"Model: {model_name}")
        logging.info(f"Output: {output_path}")

        # 4. Result Persistence Setup (reuse multi-class persistence)
        persistence = create_result_persistence(config)
        persistence.create_output_directories(output_paths)
        persistence.log_output_paths(output_paths, model_name)

        # 5. Training Infrastructure Setup
        orchestrator.setup_training_infrastructure(model_name, checkpoint_path)

        # 6. Execute Binary Training
        logging.info("Starting binary training process...")
        training_results = orchestrator.train()

        # 7. Final Evaluation
        test_results = orchestrator.evaluate_final_performance()
        training_results.update(test_results)

        # 8. Save Training Summary
        orchestrator.save_training_summary(summary_path)

        # 9. Save Predictions to ROOT Tree
        model = orchestrator.get_model()
        device = orchestrator.device
        persistence.save_predictions_to_root(model, data_pipeline, device, tree_path)

        # 10. Save Performance Metrics
        persistence.save_performance_summary(training_results, model_name, output_path)
        persistence.save_model_info(model, model_name, output_path)

        # Final Summary
        logging.info("=" * 60)
        logging.info("BINARY TRAINING COMPLETED SUCCESSFULLY!")
        logging.info("=" * 60)
        logging.info(f"Signal vs. {config.background_category} classification")
        logging.info(f"Final test accuracy: {training_results['test_accuracy']*100:.2f}%")
        logging.info(f"Total training time: {training_results['total_training_time']:.1f} seconds")
        logging.info(f"Epochs completed: {training_results['epochs_completed']}")
        logging.info(f"Early stopped: {'Yes' if training_results['early_stopped'] else 'No'}")
        logging.info(f"Model parameters: {training_results['model_parameters']:,}")
        logging.info(f"Results saved to: {output_path}")
        logging.info("=" * 60)

        # Log comparison information
        logging.info("Binary Training Summary:")
        logging.info(f"  Classification: {config.signal_full_name} vs. {config.background_category}")
        logging.info(f"  Model architecture: {config.args.model} ({training_results['model_parameters']:,} parameters)")
        logging.info(f"  Training efficiency: {training_results['avg_epoch_time']:.1f}s per epoch")
        logging.info(f"  Final performance: {training_results['test_accuracy']*100:.2f}% test accuracy")

    except Exception as e:
        logging.error(f"Binary training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()