#!/usr/bin/env python
"""
Example usage of the new per-process ParticleNet dataset system.
This script demonstrates how to:
1. Create per-process datasets
2. Load and combine processes dynamically
3. Create balanced training datasets
"""

import os
import sys
import logging
from DynamicDatasetLoader import DynamicDatasetLoader

# Setup logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    # Get WORKDIR environment variable
    WORKDIR = os.environ.get("WORKDIR")
    if not WORKDIR:
        print("ERROR: WORKDIR environment variable not set. Please run 'source setup.sh' first.")
        return 1

    print("Per-Process ParticleNet Dataset Usage Example")
    print("=" * 50)

    # Initialize the dynamic dataset loader
    dataset_root = f"{WORKDIR}/ParticleNet/dataset"
    loader = DynamicDatasetLoader(dataset_root)

    try:
        # 1. Check what processes are available
        signals, backgrounds = loader.get_available_processes()
        print(f"\nAvailable processes:")
        print(f"  Signals: {signals}")
        print(f"  Backgrounds: {backgrounds}")

        if not signals or not backgrounds:
            print("\nNO DATASETS FOUND!")
            print("Please create per-process datasets first using:")
            print("  ./scripts/saveDatasets.sh --pilot")
            return 1

        # Get available background categories
        categories = loader.get_background_categories()
        print(f"  Available background categories: {categories}")

        # 2. Example 1: Binary classification dataset
        print(f"\n1. Creating Binary Classification Dataset")
        print(f"   Signal: {signals[0]}")
        print(f"   Background category: {categories[0] if categories else 'N/A'}")

        if categories:
            binary_config = {
                "mode": "binary",
                "signal": signals[0],
                "background": categories[0],  # Use category, not sample name
                "channel": "Run1E2Mu",
                "fold": 0,
                "balance": True,
                "pilot": False
            }

        try:
            binary_dataset = loader.create_training_dataset(binary_config)
            print(f"   ✓ Created binary dataset with {len(binary_dataset)} samples")

            # Analyze the binary dataset
            if len(binary_dataset) > 0:
                sample = binary_dataset[0]
                print(f"   Sample properties:")
                print(f"     - Node features shape: {sample.x.shape}")
                print(f"     - Edge indices shape: {sample.edge_index.shape}")
                print(f"     - Has weight: {hasattr(sample, 'weight')}")
                print(f"     - Has process_info: {hasattr(sample, 'process_info')}")
                print(f"     - Era: {getattr(sample, 'era', 'N/A')}")
                print(f"     - Label: {sample.y.item()}")
                if hasattr(sample, 'weight'):
                    print(f"     - Weight: {sample.weight.item():.4f}")

        except Exception as e:
            print(f"   ✗ Failed to create binary dataset: {e}")

        # 3. Example 2: Multi-class classification dataset
        print(f"\n2. Creating Multi-class Classification Dataset")
        print(f"   Signal: {signals[0]} vs all backgrounds")

        multiclass_config = {
            "mode": "multiclass",
            "signal": signals[0],
            "channel": "Run1E2Mu",
            "fold": 0,
            "balance": True,
            "pilot": False
        }

        try:
            multiclass_dataset = loader.create_training_dataset(multiclass_config)
            print(f"   ✓ Created multiclass dataset with {len(multiclass_dataset)} samples")

            # Analyze class distribution
            if len(multiclass_dataset) > 0:
                class_counts = {}
                total_weight = 0.0
                for data in multiclass_dataset:
                    label = data.y.item()
                    weight = data.weight.item() if hasattr(data, 'weight') else 1.0
                    class_counts[label] = class_counts.get(label, 0) + 1
                    total_weight += weight

                print(f"   Class distribution: {class_counts}")
                print(f"   Total weight: {total_weight:.2f}")

        except Exception as e:
            print(f"   ✗ Failed to create multiclass dataset: {e}")

        # 4. Example 3: Loading specific sample data
        print(f"\n3. Loading Individual Sample Data")
        try:
            signal_data = loader.load_sample_data(signals[0], "signal", "Run1E2Mu", 0)

            print(f"   {signals[0]} (signal): {len(signal_data)} events")

            # Show samples for each category
            for category in categories[:3]:  # Show first 3 categories
                category_data = loader.load_background_category_data(category, "Run1E2Mu", 0)
                samples_in_cat = loader.get_samples_for_category(category)
                print(f"   {category} category ({samples_in_cat}): {len(category_data)} events")

            # Show weight statistics
            if signal_data:
                weights = [data.weight.item() for data in signal_data if hasattr(data, 'weight')]
                if weights:
                    print(f"   Signal weights: min={min(weights):.4f}, max={max(weights):.4f}, mean={sum(weights)/len(weights):.4f}")

        except Exception as e:
            print(f"   ✗ Failed to load individual sample data: {e}")

        # 5. Example 4: Demonstrating flexibility
        print(f"\n4. Demonstrating Flexibility")
        print("   You can now easily create any combination:")

        if signals and categories:
            # Show a few different combinations
            for i, category in enumerate(categories[:2]):  # Show first 2 categories
                signal = signals[min(i, len(signals)-1)]
                try:
                    config = {
                        "mode": "binary",
                        "signal": signal,
                        "background": category,
                        "channel": "Run1E2Mu",
                        "fold": 0,
                        "balance": True,
                        "pilot": False
                    }
                    dataset = loader.create_training_dataset(config)
                    samples_in_cat = loader.get_samples_for_category(category)
                    print(f"   ✓ {signal} vs {category} ({samples_in_cat}): {len(dataset)} samples")
                except Exception as e:
                    print(f"   ✗ {signal} vs {category}: {e}")

        print(f"\n5. Key Advantages of Per-Process Storage:")
        print("   ✓ No duplicate signal data across different backgrounds")
        print("   ✓ Flexible runtime combination of any signal vs background(s)")
        print("   ✓ Proper event weights preserved for statistical analysis")
        print("   ✓ Clean separation of physics processes and ML labels")
        print("   ✓ Easy to add new processes without regenerating everything")

    except Exception as e:
        print(f"ERROR: {e}")
        return 1

    print(f"\nExample completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())