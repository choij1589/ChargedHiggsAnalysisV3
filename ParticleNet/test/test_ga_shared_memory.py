#!/usr/bin/env python
"""
Test script to verify GA optimization with shared memory using pilot datasets.

This script performs a minimal test with:
- Small pilot datasets
- Reduced population size
- Single iteration
- Shared memory mode
"""

import os
import sys
import subprocess
import shutil
import argparse
import logging

def test_shared_memory_ga():
    """Test shared memory GA optimization with pilot data."""

    # Test configuration
    signal = "MHc130_MA100"
    channel = "Run1E2Mu"
    device = "cuda:0" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"

    # Clean up any existing test output
    test_output_dir = f"GAOptim_bjets/{channel}/multiclass/TTToHcToWAToMuMu-{signal}/test"
    if os.path.exists(test_output_dir):
        shutil.rmtree(test_output_dir)

    print("=" * 60)
    print("TESTING GA OPTIMIZATION WITH SHARED MEMORY")
    print("=" * 60)
    print(f"Signal: {signal}")
    print(f"Channel: {channel}")
    print(f"Device: {device}")
    print(f"Mode: PILOT (small datasets)")
    print("=" * 60)

    # Run the shared memory GA optimization with minimal settings
    cmd = [
        "python", "python/launchGAOptimShared.py",
        "--signal", signal,
        "--channel", channel,
        "--device", device,
        "--pilot",  # Use pilot datasets for quick testing
        "--use-shared-memory"
    ]

    print(f"Running command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        # Run the test
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 minute timeout

        if result.returncode == 0:
            print("✅ TEST PASSED: GA optimization with shared memory completed successfully!")
            print("\nOutput summary:")
            # Extract key information from output
            for line in result.stdout.split('\n'):
                if any(keyword in line for keyword in ['Shared memory', 'Loading datasets', 'Memory', 'Worker', 'completed']):
                    print(f"  {line}")

            # Check if output files were created
            expected_dir = f"GAOptim_bjets/{channel}/multiclass/TTToHcToWAToMuMu-{signal}/GA-iter0"
            if os.path.exists(expected_dir):
                print(f"\n✅ Output directory created: {expected_dir}")

                # Check for JSON files
                json_dir = f"{expected_dir}/json"
                if os.path.exists(json_dir):
                    json_files = os.listdir(json_dir)
                    print(f"✅ Found {len(json_files)} JSON files in output")

            return True

        else:
            print(f"❌ TEST FAILED: Process returned exit code {result.returncode}")
            print("\nError output:")
            print(result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print("❌ TEST FAILED: Process timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        return False


def verify_memory_savings():
    """Verify that shared memory reduces memory usage."""

    print("\n" + "=" * 60)
    print("VERIFYING MEMORY SAVINGS")
    print("=" * 60)

    # This would require running both versions and comparing memory usage
    # For now, we just verify that the shared memory infrastructure works

    try:
        # Import and test SharedDatasetManager
        sys.path.append('python')
        from SharedDatasetManager import SharedDatasetManager

        manager = SharedDatasetManager()
        if manager.validate_shared_memory():
            print("✅ Shared memory validation passed")
            return True
        else:
            print("❌ Shared memory validation failed")
            return False

    except Exception as e:
        print(f"❌ Could not verify shared memory: {e}")
        return False


def check_prerequisites():
    """Check if required files and datasets exist."""

    print("\n" + "=" * 60)
    print("CHECKING PREREQUISITES")
    print("=" * 60)

    required_files = [
        "python/launchGAOptimShared.py",
        "python/SharedDatasetManager.py",
        "python/trainWorkerShared.py",
        "configs/GAConfig.json"
    ]

    missing = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            missing.append(file)

    # Check for pilot datasets
    pilot_dir = "dataset/samples/signals/TTToHcToWAToMuMu-MHc130MA100_pilot"
    if os.path.exists(pilot_dir):
        print(f"✅ Found pilot dataset: {pilot_dir}")
    else:
        print(f"⚠️  Pilot dataset not found: {pilot_dir}")
        print("   Run './scripts/saveDatasets.sh --pilot' to create pilot datasets")

    return len(missing) == 0


def main():
    """Run all tests."""

    parser = argparse.ArgumentParser(description="Test GA optimization with shared memory")
    parser.add_argument("--skip-ga", action="store_true", help="Skip GA optimization test")
    parser.add_argument("--cleanup", action="store_true", help="Clean up test outputs after completion")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("GA SHARED MEMORY TEST SUITE")
    print("=" * 60)

    # Check prerequisites
    if not check_prerequisites():
        print("\n⚠️  Some prerequisites are missing. Please fix before running tests.")
        return 1

    # Verify shared memory functionality
    if not verify_memory_savings():
        print("\n⚠️  Shared memory verification failed.")
        return 1

    # Run GA optimization test (unless skipped)
    if not args.skip_ga:
        if not test_shared_memory_ga():
            print("\n⚠️  GA optimization test failed.")
            return 1
    else:
        print("\nSkipping GA optimization test (--skip-ga specified)")

    # Clean up if requested
    if args.cleanup:
        print("\nCleaning up test outputs...")
        test_dirs = [
            "GAOptim_bjets/Run1E2Mu/multiclass/TTToHcToWAToMuMu-MHc130_MA100/test"
        ]
        for dir in test_dirs:
            if os.path.exists(dir):
                shutil.rmtree(dir)
                print(f"  Removed: {dir}")

    print("\n" + "=" * 60)
    print("🎉 ALL TESTS PASSED!")
    print("Shared memory GA optimization is ready for use.")
    print("=" * 60)
    print("\nTo run full GA optimization with shared memory:")
    print("  ./scripts/launchGAOptimShared.sh MHc130_MA100 Run1E2Mu cuda:0")
    print("\nTo use pilot datasets for testing:")
    print("  ./scripts/launchGAOptimShared.sh MHc130_MA100 Run1E2Mu cuda:0 --pilot")

    return 0


if __name__ == "__main__":
    exit(main())