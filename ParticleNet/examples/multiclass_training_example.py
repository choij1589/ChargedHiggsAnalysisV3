#!/usr/bin/env python
"""
Example script demonstrating multi-class ParticleNet training usage.

Shows how to use the new multi-class training system with weighted loss functions
and 5-fold cross-validation.
"""

import os
import sys
import logging

# Add ParticleNet python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from DynamicDatasetLoader import DynamicDatasetLoader
from MultiClassModels import create_multiclass_model
from WeightedLoss import create_loss_function

def example_dataset_loading():
    """Example of loading multi-class training splits."""
    print("=== Multi-Class Dataset Loading Example ===")

    # Setup
    WORKDIR = os.environ.get("WORKDIR", os.path.join(os.path.dirname(__file__), "../.."))
    dataset_root = os.path.join(WORKDIR, "ParticleNet/dataset")

    # Initialize loader
    loader = DynamicDatasetLoader(dataset_root)

    # Check available samples
    signals, backgrounds = loader.get_available_samples()
    print(f"Available signals: {signals[:3]}...")  # Show first 3
    print(f"Available backgrounds: {backgrounds[:3]}...")

    # Example: Create multi-class training splits
    if signals and len(backgrounds) >= 3:
        signal_sample = signals[0] if signals else "TTToHcToWAToMuMu-MHc130MA100"
        background_samples = [
            "Skim_TriLep_TTLL_powheg",
            "Skim_TriLep_WZTo3LNu_amcatnlo",
            "Skim_TriLep_TTZToLLNuNu"
        ]

        print(f"\nCreating multi-class splits:")
        print(f"Signal: {signal_sample}")
        print(f"Backgrounds: {background_samples}")

        try:
            train_data, valid_data, test_data = loader.create_multiclass_training_splits(
                signal_sample=signal_sample,
                background_samples=background_samples,
                channel="Run1E2Mu",
                fold=0,
                balance=True
            )

            print(f"Dataset sizes: Train={len(train_data)}, Valid={len(valid_data)}, Test={len(test_data)}")

            # Show class distribution
            def count_classes(data_list):
                counts = {}
                for data in data_list:
                    label = data.y.item()
                    counts[label] = counts.get(label, 0) + 1
                return counts

            print(f"Training class distribution: {count_classes(train_data)}")
            print(f"Validation class distribution: {count_classes(valid_data)}")
            print(f"Test class distribution: {count_classes(test_data)}")

        except Exception as e:
            print(f"Dataset loading failed (this is expected if datasets don't exist yet): {e}")

    print()

def example_model_creation():
    """Example of creating multi-class models."""
    print("=== Multi-Class Model Creation Example ===")

    # Model parameters
    num_node_features = 9
    num_graph_features = 4
    num_classes = 4
    num_hidden = 64  # Smaller for example

    print(f"Creating models with {num_classes} classes, {num_hidden} hidden units")

    # Create different model types
    for model_type in ['ParticleNet', 'ParticleNetV2']:
        model = create_multiclass_model(
            model_type=model_type,
            num_node_features=num_node_features,
            num_graph_features=num_graph_features,
            num_classes=num_classes,
            num_hidden=num_hidden
        )

        total_params = sum(p.numel() for p in model.parameters())
        print(f"{model_type}: {total_params:,} parameters")

    print()

def example_loss_functions():
    """Example of using weighted loss functions."""
    print("=== Weighted Loss Functions Example ===")

    import torch

    # Create example data
    batch_size = 8
    num_classes = 4

    predictions = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, num_classes, (batch_size,))
    weights = torch.abs(torch.randn(batch_size)) + 0.1  # Positive weights

    print(f"Example batch: {batch_size} samples, {num_classes} classes")
    print(f"Targets: {targets.tolist()}")
    print(f"Weights: {[f'{w:.3f}' for w in weights.tolist()]}")

    # Test different loss functions
    loss_types = ['weighted_ce', 'sample_normalized', 'focal']

    for loss_type in loss_types:
        loss_fn = create_loss_function(loss_type, num_classes=num_classes)
        loss = loss_fn(predictions, targets, weights)
        print(f"{loss_type}: {loss.item():.4f}")

    print()

def example_training_command():
    """Show example training commands."""
    print("=== Training Command Examples ===")

    print("1. Single model training:")
    print("   python trainMultiClass.py \\")
    print("     --signal TTToHcToWAToMuMu-MHc130MA100 \\")
    print("     --channel Run1E2Mu \\")
    print("     --fold 0 \\")
    print("     --model ParticleNet \\")
    print("     --loss_type weighted_ce")
    print()

    print("2. Batch training all folds:")
    print("   ./scripts/trainMultiClass.sh \\")
    print("     --channel Run1E2Mu \\")
    print("     --model ParticleNet \\")
    print("     --loss-type sample_normalized")
    print()

    print("3. Pilot run for testing:")
    print("   ./scripts/trainMultiClass.sh \\")
    print("     --channel Run1E2Mu \\")
    print("     --pilot \\")
    print("     --dry-run")
    print()

def main():
    """Run all examples."""
    print("Multi-Class ParticleNet Training System Examples")
    print("=" * 50)
    print()

    # Set up logging
    logging.basicConfig(level=logging.INFO)

    # Run examples
    example_dataset_loading()
    example_model_creation()
    example_loss_functions()
    example_training_command()

    print("=" * 50)
    print("All examples completed!")
    print()
    print("Next steps:")
    print("1. Ensure datasets are created using saveDataset.py")
    print("2. Run training with: ./scripts/trainMultiClass.sh --channel <channel>")
    print("3. Check results in $WORKDIR/ParticleNet/results/multiclass/")

if __name__ == "__main__":
    main()