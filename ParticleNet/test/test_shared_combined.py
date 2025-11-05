#!/usr/bin/env python
"""
Test SharedDatasetManager with Combined channel.
"""

import os
import sys
import logging

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from SharedDatasetManager import SharedDatasetManager
from DynamicDatasetLoader import DynamicDatasetLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_shared_manager_combined():
    """Test SharedDatasetManager with Combined channel."""

    WORKDIR = os.environ.get("WORKDIR")
    if not WORKDIR:
        print("ERROR: WORKDIR not set. Run 'source setup.sh'")
        return False

    # Initialize loader and manager
    dataset_root = f"{WORKDIR}/ParticleNet/dataset"
    loader = DynamicDatasetLoader(dataset_root=dataset_root, separate_bjets=False)
    manager = SharedDatasetManager()

    # Test parameters
    signal_sample = "TTToHcToWAToMuMu-MHc130_MA90"
    background_groups = {
        'nonprompt': ['Skim_TriLep_TTLL_powheg']
    }

    print("=" * 70)
    print("Testing SharedDatasetManager with Combined channel")
    print("=" * 70)

    # Test 1: Load with Run1E2Mu (baseline)
    print("\nTest 1: SharedDatasetManager with Run1E2Mu...")
    train_1e2mu, valid_1e2mu = manager.prepare_shared_datasets(
        loader=loader,
        signal_sample=signal_sample,
        background_groups=background_groups,
        channel="Run1E2Mu",
        train_folds=[0, 1, 2],
        valid_folds=[3],
        pilot=True,
        max_events_per_fold=None,
        balance_weights=False,
        random_state=42
    )

    print(f"  Train events: {train_1e2mu.num_graphs}")
    print(f"  Valid events: {valid_1e2mu.num_graphs}")

    # Test 2: Load with Run3Mu
    print("\nTest 2: SharedDatasetManager with Run3Mu...")
    train_3mu, valid_3mu = manager.prepare_shared_datasets(
        loader=loader,
        signal_sample=signal_sample,
        background_groups=background_groups,
        channel="Run3Mu",
        train_folds=[0, 1, 2],
        valid_folds=[3],
        pilot=True,
        max_events_per_fold=None,
        balance_weights=False,
        random_state=42
    )

    print(f"  Train events: {train_3mu.num_graphs}")
    print(f"  Valid events: {valid_3mu.num_graphs}")

    # Calculate expected Combined total
    expected_train_total = train_1e2mu.num_graphs + train_3mu.num_graphs
    expected_valid_total = valid_1e2mu.num_graphs + valid_3mu.num_graphs

    print(f"\nExpected Combined totals:")
    print(f"  Train: {expected_train_total}")
    print(f"  Valid: {expected_valid_total}")

    # Clear cache to test fresh load
    manager.clear_cache()

    # Test 3: Load with Combined channel
    print("\nTest 3: SharedDatasetManager with Combined channel...")
    train_combined, valid_combined = manager.prepare_shared_datasets(
        loader=loader,
        signal_sample=signal_sample,
        background_groups=background_groups,
        channel="Combined",
        train_folds=[0, 1, 2],
        valid_folds=[3],
        pilot=True,
        max_events_per_fold=None,
        balance_weights=False,
        random_state=42
    )

    print(f"  Train events: {train_combined.num_graphs}")
    print(f"  Valid events: {valid_combined.num_graphs}")

    # Verify counts match
    train_match = (train_combined.num_graphs == expected_train_total)
    valid_match = (valid_combined.num_graphs == expected_valid_total)

    if train_match:
        print(f"  ✅ PASS: Train events match ({train_combined.num_graphs} == {expected_train_total})")
    else:
        print(f"  ❌ FAIL: Train events mismatch ({train_combined.num_graphs} != {expected_train_total})")
        return False

    if valid_match:
        print(f"  ✅ PASS: Valid events match ({valid_combined.num_graphs} == {expected_valid_total})")
    else:
        print(f"  ❌ FAIL: Valid events mismatch ({valid_combined.num_graphs} != {expected_valid_total})")
        return False

    # Test 4: Verify tensors are in shared memory
    print("\nTest 4: Verify tensors are in shared memory...")

    # Check if tensors have is_shared() attribute
    shared_count = 0
    total_tensors = 0

    for batch in [train_combined, valid_combined]:
        for key in batch.keys():
            value = getattr(batch, key)
            import torch
            if torch.is_tensor(value):
                total_tensors += 1
                if value.is_shared():
                    shared_count += 1

    print(f"  Shared tensors: {shared_count}/{total_tensors}")

    if shared_count == total_tensors:
        print(f"  ✅ PASS: All tensors are in shared memory")
    else:
        print(f"  ⚠️  WARNING: Not all tensors are shared ({shared_count}/{total_tensors})")

    # Test 5: Unbatch and verify data integrity
    print("\nTest 5: Unbatch shared data...")
    train_unbatched = SharedDatasetManager.unbatch_shared_data(train_combined)
    valid_unbatched = SharedDatasetManager.unbatch_shared_data(valid_combined)

    print(f"  Unbatched train events: {len(train_unbatched)}")
    print(f"  Unbatched valid events: {len(valid_unbatched)}")

    if len(train_unbatched) == train_combined.num_graphs:
        print(f"  ✅ PASS: Unbatching preserves all train events")
    else:
        print(f"  ❌ FAIL: Lost events during unbatching")
        return False

    if len(valid_unbatched) == valid_combined.num_graphs:
        print(f"  ✅ PASS: Unbatching preserves all valid events")
    else:
        print(f"  ❌ FAIL: Lost events during unbatching")
        return False

    # Test 6: Check memory statistics
    print("\nTest 6: Memory usage statistics...")
    memory_stats = manager.get_memory_usage()
    print(f"  Total shared memory: {memory_stats['total_gb']:.2f} GB")
    print(f"  Number of cached datasets: {memory_stats['num_datasets']}")

    if memory_stats['total_gb'] > 0:
        print(f"  ✅ PASS: Shared memory is being used")
    else:
        print(f"  ❌ FAIL: No shared memory allocated")
        return False

    print("\n" + "=" * 70)
    print("✅ All SharedDatasetManager tests with Combined channel PASSED!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = test_shared_manager_combined()
    sys.exit(0 if success else 1)
