#!/usr/bin/env python
"""
Test script to verify Combined channel functionality in DynamicDatasetLoader.
"""

import os
import sys
import logging

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

from DynamicDatasetLoader import DynamicDatasetLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def test_combined_channel():
    """Test loading data with Combined channel."""

    WORKDIR = os.environ.get("WORKDIR")
    if not WORKDIR:
        print("ERROR: WORKDIR not set. Run 'source setup.sh'")
        return False

    # Initialize loader
    dataset_root = f"{WORKDIR}/ParticleNet/dataset"
    loader = DynamicDatasetLoader(dataset_root=dataset_root, separate_bjets=False)

    # Test parameters
    test_signal = "TTToHcToWAToMuMu-MHc130_MA90"
    test_background = "Skim_TriLep_TTLL_powheg"
    fold = 0
    pilot = True  # Use pilot datasets for quick testing

    print("=" * 70)
    print("Testing Combined channel loading")
    print("=" * 70)

    # Test 1: Load signal with individual channels
    print("\nTest 1: Loading signal with individual channels...")
    signal_1e2mu = loader.load_sample_data(test_signal, "signal", "Run1E2Mu", fold, pilot)
    signal_3mu = loader.load_sample_data(test_signal, "signal", "Run3Mu", fold, pilot)

    print(f"  Run1E2Mu: {len(signal_1e2mu)} events")
    print(f"  Run3Mu:   {len(signal_3mu)} events")
    expected_total = len(signal_1e2mu) + len(signal_3mu)
    print(f"  Expected combined total: {expected_total} events")

    # Test 2: Load signal with Combined channel
    print("\nTest 2: Loading signal with Combined channel...")
    signal_combined = loader.load_sample_data(test_signal, "signal", "Combined", fold, pilot)

    print(f"  Combined: {len(signal_combined)} events")

    # Verify
    if len(signal_combined) == expected_total:
        print("  ✅ PASS: Combined channel loaded correct number of events")
    else:
        print(f"  ❌ FAIL: Expected {expected_total}, got {len(signal_combined)}")
        return False

    # Test 3: Load background with individual channels
    print("\nTest 3: Loading background with individual channels...")
    bg_1e2mu = loader.load_sample_data(test_background, "background", "Run1E2Mu", fold, pilot)
    bg_3mu = loader.load_sample_data(test_background, "background", "Run3Mu", fold, pilot)

    print(f"  Run1E2Mu: {len(bg_1e2mu)} events")
    print(f"  Run3Mu:   {len(bg_3mu)} events")
    expected_bg_total = len(bg_1e2mu) + len(bg_3mu)
    print(f"  Expected combined total: {expected_bg_total} events")

    # Test 4: Load background with Combined channel
    print("\nTest 4: Loading background with Combined channel...")
    bg_combined = loader.load_sample_data(test_background, "background", "Combined", fold, pilot)

    print(f"  Combined: {len(bg_combined)} events")

    # Verify
    if len(bg_combined) == expected_bg_total:
        print("  ✅ PASS: Combined channel loaded correct number of events")
    else:
        print(f"  ❌ FAIL: Expected {expected_bg_total}, got {len(bg_combined)}")
        return False

    # Test 5: Load multiclass with Combined channel
    print("\nTest 5: Loading multiclass dataset with Combined channel...")
    background_groups = {
        'nonprompt': ['Skim_TriLep_TTLL_powheg']
    }

    combined_data = loader.load_multiclass_with_subsampling(
        signal_sample=test_signal,
        background_groups=background_groups,
        channel="Combined",
        fold_list=[0],
        pilot=True,
        max_events_per_fold=None,
        balance_weights=False,
        random_state=42
    )

    print(f"  Multiclass combined: {len(combined_data)} events")

    if len(combined_data) > 0:
        print("  ✅ PASS: Multiclass loading with Combined channel works")
    else:
        print("  ❌ FAIL: No data loaded")
        return False

    print("\n" + "=" * 70)
    print("✅ All tests PASSED!")
    print("=" * 70)

    return True


if __name__ == "__main__":
    success = test_combined_channel()
    sys.exit(0 if success else 1)
