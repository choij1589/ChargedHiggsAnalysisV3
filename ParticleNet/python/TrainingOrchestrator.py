#!/usr/bin/env python
"""
Training orchestrator for ParticleNet multi-class training.

Manages the main training loop, early stopping, learning rate scheduling,
and metrics collection with self-contained training infrastructure.
"""

import os
import logging
import time
from typing import Dict, Any

import torch

from TrainingConfig import TrainingConfig
from DataPipeline import DataPipeline
from TrainingUtilities import (
    train_epoch, evaluate_model, get_detailed_predictions,
    log_detailed_class_accuracies, PerformanceMonitor,
    create_optimizer, create_scheduler, setup_device
)
from MultiClassModels import create_multiclass_model
from WeightedLoss import create_loss_function
from MLTools import EarlyStopper, SummaryWriter
from BjetSubsetUtils import create_bjet_subset_loader, get_bjet_subset_statistics


class TrainingOrchestrator:
    """
    Orchestrates the complete training process for multi-class ParticleNet training.

    Manages model creation, optimization setup, training loop execution,
    and metrics collection with self-contained infrastructure.
    """

    def __init__(self, config: TrainingConfig, data_pipeline: DataPipeline):
        """
        Initialize training orchestrator.

        Args:
            config: Training configuration
            data_pipeline: Data pipeline with prepared data loaders
        """
        self.config = config
        self.data_pipeline = data_pipeline

        # Training infrastructure
        self.device = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.loss_fn = None
        self.early_stopper = None
        self.summary_writer = None
        self.perf_monitor = None

        # Training state
        self.current_epoch = 0
        self.total_training_time = 0.0
        self.best_valid_loss = float('inf')
        self.training_history = []

    def setup_training_infrastructure(self, model_name: str, checkpoint_path: str) -> None:
        """
        Setup all training infrastructure components.

        Args:
            model_name: Name of the model for logging
            checkpoint_path: Path for saving model checkpoints
        """
        logging.info("Setting up training infrastructure...")

        # Setup device
        self.device = setup_device(self.config.args.device)

        # Create model
        self._create_model()

        # Create optimizer and scheduler
        self._create_optimizer()
        self._create_scheduler()

        # Create loss function
        self.loss_fn = create_loss_function(self.config.args.loss_type,
                                          num_classes=self.config.num_classes)
        logging.info(f"Using {self.config.args.loss_type} loss function")

        # Setup training utilities
        self.early_stopper = EarlyStopper(patience=7, path=checkpoint_path)
        self.summary_writer = SummaryWriter(name=model_name)
        self.perf_monitor = PerformanceMonitor(self.config.args.model)

        logging.info("Training infrastructure setup complete")

    def _create_model(self) -> None:
        """Create and configure the model."""
        num_node_features = 9
        num_graph_features = 4

        self.model = create_multiclass_model(
            model_type=self.config.args.model,
            num_node_features=num_node_features,
            num_graph_features=num_graph_features,
            num_classes=self.config.num_classes,
            num_hidden=self.config.args.nNodes,
            dropout_p=self.config.args.dropout_p
        ).to(self.device)

        param_count = sum(p.numel() for p in self.model.parameters())
        logging.info(f"Created {self.config.args.model} with {param_count:,} parameters")

    def _create_optimizer(self) -> None:
        """Create optimizer based on configuration."""
        self.optimizer = create_optimizer(
            self.config.args.optimizer,
            self.model.parameters(),
            self.config.args.initLR,
            self.config.args.weight_decay
        )
        logging.info(f"Created {self.config.args.optimizer} optimizer")

    def _create_scheduler(self) -> None:
        """Create learning rate scheduler based on configuration."""
        self.scheduler = create_scheduler(
            self.config.args.scheduler,
            self.optimizer,
            self.config.args.initLR
        )
        logging.info(f"Created {self.config.args.scheduler} scheduler")

    def train(self) -> Dict[str, Any]:
        """
        Execute the main training loop.

        Returns:
            Dictionary containing training results and statistics
        """
        if not self._is_infrastructure_ready():
            raise RuntimeError("Training infrastructure not properly initialized. Call setup_training_infrastructure() first.")

        logging.info("=" * 60)
        logging.info("STARTING TRAINING")
        logging.info("=" * 60)
        logging.info(f"Training for {self.config.args.max_epochs} epochs")

        # Training loop
        for epoch in range(self.config.args.max_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            # Train one epoch
            train_loss, train_acc = train_epoch(
                self.model, self.data_pipeline.train_loader,
                self.optimizer, self.scheduler, self.loss_fn, self.device,
                self.config.use_groups, self.config.num_classes,
                self.config.args.scheduler
            )

            # Validate
            valid_loss, valid_acc = evaluate_model(
                self.model, self.data_pipeline.valid_loader,
                self.loss_fn, self.device,
                self.config.use_groups, self.config.num_classes
            )

            epoch_time = time.time() - epoch_start_time
            self.total_training_time += epoch_time

            # Performance monitoring
            perf_stats = self.perf_monitor.get_stats()

            # Log metrics to summary writer
            if self.summary_writer:
                self.summary_writer.log_metrics(epoch, train_loss, valid_loss, train_acc, valid_acc)

            # Log to console
            self._log_epoch_progress(epoch, train_loss, train_acc, valid_loss, valid_acc,
                                   epoch_time, perf_stats)

            # Detailed class accuracy logging (every 10 epochs or at the end)
            if self.config.use_groups and (epoch % 10 == 0 or epoch == self.config.args.max_epochs - 1):
                self._log_detailed_class_accuracies(epoch)

            # Store training history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'valid_loss': valid_loss,
                'valid_acc': valid_acc,
                'epoch_time': epoch_time,
                'memory_mb': perf_stats['memory_mb'],
                'cpu_percent': perf_stats['cpu_percent']
            })

            # Early stopping
            if self.early_stopper(valid_loss, self.model):
                logging.info(f"Early stopping at epoch {epoch}")
                break

        logging.info("Training completed!")
        return self._get_training_summary()

    def evaluate_final_performance(self) -> Dict[str, Any]:
        """
        Evaluate final model performance on test set.

        Returns:
            Dictionary containing test performance metrics
        """
        logging.info("Evaluating final model performance...")

        test_loss, test_acc = evaluate_model(
            self.model, self.data_pipeline.test_loader,
            self.loss_fn, self.device,
            self.config.use_groups, self.config.num_classes
        )

        results = {
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'total_training_time': self.total_training_time,
            'epochs_completed': self.current_epoch + 1,
            'best_valid_loss': self.early_stopper.best_loss if self.early_stopper else float('inf')
        }

        logging.info(f"Final test accuracy: {test_acc*100:.2f}%")
        logging.info(f"Training completed in {self.total_training_time:.1f} seconds")
        logging.info(f"Average epoch time: {self.total_training_time/(self.current_epoch+1):.1f} seconds")

        return results

    def evaluate_bjet_subset_performance(self) -> Dict[str, Any]:
        """
        Evaluate model performance on subset of train and test events containing b-jets.

        Returns:
            Dictionary containing b-jet subset performance metrics and statistics
        """
        logging.info("=" * 60)
        logging.info("EVALUATING B-JET SUBSET PERFORMANCE")
        logging.info("=" * 60)

        # ========== TRAIN SET EVALUATION ==========
        logging.info("\n[TRAIN SET B-JET SUBSET EVALUATION]")

        # Get b-jet subset statistics for train data
        train_stats = get_bjet_subset_statistics(
            self.data_pipeline.train_data,
            separate_bjets=self.config.args.separate_bjets
        )

        logging.info(f"Train set b-jet statistics:")
        logging.info(f"  Total events: {train_stats['total_events']}")
        logging.info(f"  Events with b-jets: {train_stats['bjet_events']} "
                    f"({train_stats['bjet_event_fraction']*100:.1f}%)")
        logging.info(f"  Total b-jets: {train_stats['total_bjets']}")
        logging.info(f"  Avg b-jets per b-jet event: {train_stats['avg_bjets_per_bjet_event']:.2f}")

        train_bjet_results = None
        if train_stats['bjet_events'] > 0:
            # Create b-jet subset loader for train data
            train_bjet_loader, train_original_count, train_filtered_count = create_bjet_subset_loader(
                self.data_pipeline.train_data,
                separate_bjets=self.config.args.separate_bjets,
                batch_size=1024
            )

            if train_bjet_loader is not None:
                # Evaluate on train b-jet subset
                train_bjet_loss, train_bjet_acc = evaluate_model(
                    self.model, train_bjet_loader,
                    self.loss_fn, self.device,
                    self.config.use_groups, self.config.num_classes
                )

                # Get full train set performance for comparison
                full_train_loss, full_train_acc = evaluate_model(
                    self.model, self.data_pipeline.train_loader,
                    self.loss_fn, self.device,
                    self.config.use_groups, self.config.num_classes
                )

                # Calculate deltas
                train_acc_delta = train_bjet_acc - full_train_acc
                train_loss_delta = train_bjet_loss - full_train_loss

                train_bjet_results = {
                    'has_bjet_events': True,
                    'statistics': train_stats,
                    'bjet_subset_loss': train_bjet_loss,
                    'bjet_subset_accuracy': train_bjet_acc,
                    'full_train_loss': full_train_loss,
                    'full_train_accuracy': full_train_acc,
                    'accuracy_delta': train_acc_delta,
                    'loss_delta': train_loss_delta,
                    'bjet_event_count': train_filtered_count,
                    'total_train_count': train_original_count
                }

                # Log comparison
                logging.info("\nTrain Set Performance Comparison:")
                logging.info(f"  Full train set: Acc={full_train_acc*100:.2f}%, Loss={full_train_loss:.4f}")
                logging.info(f"  B-jet subset:   Acc={train_bjet_acc*100:.2f}%, Loss={train_bjet_loss:.4f}")
                logging.info(f"  Delta:          Acc={train_acc_delta*100:+.2f}pp, Loss={train_loss_delta:+.4f}")

                if abs(train_acc_delta) > 0.05:  # More than 5pp difference
                    if train_acc_delta > 0:
                        logging.info("  => Model performs BETTER on train events with b-jets")
                    else:
                        logging.info("  => Model performs WORSE on train events with b-jets")
                else:
                    logging.info("  => Performance is similar on train events with and without b-jets")
            else:
                logging.warning("Failed to create train b-jet subset loader.")
        else:
            logging.warning("No events with b-jets found in train set.")

        # ========== TEST SET EVALUATION ==========
        logging.info("\n[TEST SET B-JET SUBSET EVALUATION]")

        # Get b-jet subset statistics for test data
        test_stats = get_bjet_subset_statistics(
            self.data_pipeline.test_data,
            separate_bjets=self.config.args.separate_bjets
        )

        logging.info(f"Test set b-jet statistics:")
        logging.info(f"  Total events: {test_stats['total_events']}")
        logging.info(f"  Events with b-jets: {test_stats['bjet_events']} "
                    f"({test_stats['bjet_event_fraction']*100:.1f}%)")
        logging.info(f"  Total b-jets: {test_stats['total_bjets']}")
        logging.info(f"  Avg b-jets per b-jet event: {test_stats['avg_bjets_per_bjet_event']:.2f}")

        test_bjet_results = None
        if test_stats['bjet_events'] > 0:
            # Create b-jet subset loader for test data
            test_bjet_loader, test_original_count, test_filtered_count = create_bjet_subset_loader(
                self.data_pipeline.test_data,
                separate_bjets=self.config.args.separate_bjets,
                batch_size=1024
            )

            if test_bjet_loader is not None:
                # Evaluate on test b-jet subset
                test_bjet_loss, test_bjet_acc = evaluate_model(
                    self.model, test_bjet_loader,
                    self.loss_fn, self.device,
                    self.config.use_groups, self.config.num_classes
                )

                # Get full test set performance for comparison
                full_test_loss, full_test_acc = evaluate_model(
                    self.model, self.data_pipeline.test_loader,
                    self.loss_fn, self.device,
                    self.config.use_groups, self.config.num_classes
                )

                # Calculate deltas
                test_acc_delta = test_bjet_acc - full_test_acc
                test_loss_delta = test_bjet_loss - full_test_loss

                test_bjet_results = {
                    'has_bjet_events': True,
                    'statistics': test_stats,
                    'bjet_subset_loss': test_bjet_loss,
                    'bjet_subset_accuracy': test_bjet_acc,
                    'full_test_loss': full_test_loss,
                    'full_test_accuracy': full_test_acc,
                    'accuracy_delta': test_acc_delta,
                    'loss_delta': test_loss_delta,
                    'bjet_event_count': test_filtered_count,
                    'total_test_count': test_original_count
                }

                # Log comparison
                logging.info("\nTest Set Performance Comparison:")
                logging.info(f"  Full test set: Acc={full_test_acc*100:.2f}%, Loss={full_test_loss:.4f}")
                logging.info(f"  B-jet subset:  Acc={test_bjet_acc*100:.2f}%, Loss={test_bjet_loss:.4f}")
                logging.info(f"  Delta:         Acc={test_acc_delta*100:+.2f}pp, Loss={test_loss_delta:+.4f}")

                if abs(test_acc_delta) > 0.05:  # More than 5pp difference
                    if test_acc_delta > 0:
                        logging.info("  => Model performs BETTER on test events with b-jets")
                    else:
                        logging.info("  => Model performs WORSE on test events with b-jets")
                else:
                    logging.info("  => Performance is similar on test events with and without b-jets")
            else:
                logging.warning("Failed to create test b-jet subset loader.")
        else:
            logging.warning("No events with b-jets found in test set.")

        # ========== SUMMARY ==========
        logging.info("\n[B-JET SUBSET EVALUATION SUMMARY]")

        # Construct comprehensive results
        results = {
            'train_results': train_bjet_results if train_bjet_results else {
                'has_bjet_events': False,
                'statistics': train_stats
            },
            'test_results': test_bjet_results if test_bjet_results else {
                'has_bjet_events': False,
                'statistics': test_stats
            }
        }

        # Compare train vs test b-jet performance if both are available
        if train_bjet_results and test_bjet_results:
            train_test_acc_gap = train_bjet_results['bjet_subset_accuracy'] - test_bjet_results['bjet_subset_accuracy']
            train_test_loss_gap = train_bjet_results['bjet_subset_loss'] - test_bjet_results['bjet_subset_loss']

            results['train_test_comparison'] = {
                'bjet_accuracy_gap': train_test_acc_gap,
                'bjet_loss_gap': train_test_loss_gap
            }

            logging.info("Train vs Test B-jet Subset Performance:")
            logging.info(f"  Train B-jet Acc: {train_bjet_results['bjet_subset_accuracy']*100:.2f}%")
            logging.info(f"  Test B-jet Acc:  {test_bjet_results['bjet_subset_accuracy']*100:.2f}%")
            logging.info(f"  Gap (Train-Test): {train_test_acc_gap*100:+.2f}pp")

            if abs(train_test_acc_gap) > 0.10:  # More than 10pp gap
                logging.info("  => Significant train-test performance gap on b-jet events")
            else:
                logging.info("  => Train-test performance is consistent on b-jet events")

        logging.info("=" * 60)

        return results

    def save_training_summary(self, summary_path: str) -> None:
        """
        Save training summary to CSV file.

        Args:
            summary_path: Path to save CSV summary
        """
        if self.summary_writer:
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            saved_path = self.summary_writer.save_metrics_csv()
            if saved_path:
                # Copy to the expected location
                import shutil
                shutil.copy2(saved_path, summary_path)
            logging.info(f"Training summary saved to: {summary_path}")

    def _is_infrastructure_ready(self) -> bool:
        """Check if all training infrastructure is properly initialized."""
        return all([
            self.device is not None,
            self.model is not None,
            self.optimizer is not None,
            self.scheduler is not None,
            self.loss_fn is not None,
            self.early_stopper is not None,
            self.summary_writer is not None,
            self.perf_monitor is not None
        ])


    def _log_epoch_progress(self, epoch: int, train_loss: float, train_acc: float,
                           valid_loss: float, valid_acc: float,
                           epoch_time: float, perf_stats: Dict[str, Any]) -> None:
        """Log epoch progress to console."""
        logging.info(f"[EPOCH {epoch}] Train Acc: {train_acc*100:.2f}% | Train Loss: {train_loss:.4e}")
        logging.info(f"[EPOCH {epoch}] Valid Acc: {valid_acc*100:.2f}% | Valid Loss: {valid_loss:.4e}")
        logging.info(f"[EPOCH {epoch}] Time: {epoch_time:.1f}s | "
                    f"Memory: {perf_stats['memory_mb']:.1f}MB | "
                    f"CPU: {perf_stats['cpu_percent']:.1f}%")

    def _log_detailed_class_accuracies(self, epoch: int) -> None:
        """Log detailed per-class accuracies for monitoring."""
        val_predictions, val_labels, val_weights = get_detailed_predictions(
            self.model, self.data_pipeline.valid_loader, self.device,
            self.config.use_groups, self.config.num_classes
        )

        log_detailed_class_accuracies(
            val_predictions, val_labels, val_weights,
            self.config.use_groups, self.config.num_classes,
            self.config.background_groups, epoch, "VALID"
        )

    def _get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        return {
            'model_type': self.config.args.model,
            'num_classes': self.config.num_classes,
            'use_groups': self.config.use_groups,
            'background_groups': self.config.background_groups if self.config.use_groups else None,
            'epochs_completed': self.current_epoch + 1,
            'total_training_time': self.total_training_time,
            'avg_epoch_time': self.total_training_time / (self.current_epoch + 1) if self.current_epoch >= 0 else 0,
            'best_valid_loss': self.early_stopper.best_loss if self.early_stopper else float('inf'),
            'early_stopped': self.early_stopper.early_stop if self.early_stopper else False,
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'final_memory_mb': self.perf_monitor.get_stats()['memory_mb'] if self.perf_monitor else 0,
            'training_history': self.training_history
        }

    def get_model(self) -> torch.nn.Module:
        """Get the trained model."""
        return self.model

    def get_training_history(self) -> list:
        """Get complete training history."""
        return self.training_history


def create_training_orchestrator(config: TrainingConfig, data_pipeline: DataPipeline) -> TrainingOrchestrator:
    """
    Factory function to create training orchestrator.

    Args:
        config: Training configuration
        data_pipeline: Data pipeline with prepared data

    Returns:
        Initialized TrainingOrchestrator instance
    """
    return TrainingOrchestrator(config, data_pipeline)