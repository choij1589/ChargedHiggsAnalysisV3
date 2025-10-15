#!/usr/bin/env python
"""
Multi-class training script for ParticleNet (Refactored Version).

Trains a 4-class classifier (signal vs 3 backgrounds) using weighted loss functions
and the proven 5-fold cross-validation scheme from V2.

This refactored version uses modular components for better maintainability,
testability, and code reuse while maintaining full backward compatibility.
"""

import logging
from TrainingConfig import create_training_config
from DataPipeline import create_data_pipeline
from TrainingOrchestrator import create_training_orchestrator
from ResultPersistence import create_result_persistence


def main():
    """
    Main training function using modular architecture.

    Orchestrates the complete training pipeline through specialized modules:
    - TrainingConfig: Argument parsing and validation
    - DataPipeline: Dataset creation and data loader management
    - TrainingOrchestrator: Training loop and metrics collection
    - ResultPersistence: Model saving and output management
    """

    # 1. Configuration Management
    config = create_training_config()

    # Setup logging based on debug flag
    logging.basicConfig(
        level=logging.DEBUG if config.args.debug else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Log comprehensive configuration
    config.log_configuration()

    try:
        # 2. Data Pipeline Setup
        logging.info("Setting up data pipeline...")
        data_pipeline = create_data_pipeline(config)

        # Create datasets
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
        logging.info(f"Label distribution: {batch_info['label_distribution']}")
        if batch_info['has_weights']:
            logging.info(f"Weight range: [{batch_info['weight_range'][0]:.6f}, "
                        f"{batch_info['weight_range'][1]:.6f}]")

        # 3. Training Orchestration Setup
        logging.info("Setting up training orchestration...")
        orchestrator = create_training_orchestrator(config, data_pipeline)

        # Generate model name and output paths
        model_name = config.get_model_name()
        output_paths = config.get_output_paths(model_name)
        output_path, checkpoint_path, summary_path, tree_path = output_paths

        logging.info(f"Model: {model_name}")
        logging.info(f"Output: {output_path}")

        # 4. Result Persistence Setup
        persistence = create_result_persistence(config)
        persistence.create_output_directories(output_paths)
        persistence.log_output_paths(output_paths, model_name)

        # 5. Training Infrastructure Setup
        orchestrator.setup_training_infrastructure(model_name, checkpoint_path)

        # 6. Execute Training
        logging.info("Starting training process...")
        training_results = orchestrator.train()

        # 7. Final Evaluation
        test_results = orchestrator.evaluate_final_performance()
        training_results.update(test_results)

        # 7.5. B-Jet Subset Evaluation
        logging.info("Evaluating performance on b-jet subset...")
        bjet_results = orchestrator.evaluate_bjet_subset_performance()
        training_results['bjet_subset_results'] = bjet_results

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