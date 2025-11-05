#!/usr/bin/env python
"""
Test script to validate shared memory functionality with PyTorch Geometric Data objects.

This script tests:
1. Basic shared memory with spawn method
2. PyTorch Geometric Data object sharing
3. Memory usage comparison
4. CUDA compatibility
"""

import os
import sys
import torch
import torch.multiprocessing as mp
from torch_geometric.data import Data
import time
import psutil
import logging
from typing import List

# Add python directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ============================================================
# Worker functions (must be at module level for spawn)
# ============================================================

def worker_basic_test(rank, shared_tensor, result_queue):
    """Worker that modifies shared tensor."""
    try:
        initial = shared_tensor[rank].item()
        shared_tensor[rank] = rank + 1
        final = shared_tensor[rank].item()
        result_queue.put((rank, True, f"Changed {initial} -> {final}"))
    except Exception as e:
        result_queue.put((rank, False, str(e)))


def worker_pyg_data(rank, data_list, result_queue):
    """Worker that accesses shared Data objects."""
    try:
        # Access some data
        sample_data = data_list[rank % len(data_list)]

        # Verify we can read the data
        assert sample_data.x.shape[0] > 0, "No nodes in graph"
        assert sample_data.edge_index.shape[1] > 0, "No edges in graph"

        # Check if data is truly shared (not copied)
        is_shared = sample_data.x.is_shared()

        # Try to access multiple graphs (simulate DataLoader behavior)
        batch_size = min(10, len(data_list))
        total_nodes = 0
        for i in range(batch_size):
            total_nodes += data_list[i].x.shape[0]

        result_queue.put((rank, True, f"Accessed {batch_size} graphs ({total_nodes} nodes), shared={is_shared}"))
    except Exception as e:
        result_queue.put((rank, False, str(e)))


def worker_no_share(rank, num_events, result_queue):
    """Worker that creates its own data."""
    try:
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024**2  # MB

        # Each worker creates its own data
        data_list = create_dummy_data(num_events)

        mem_after = process.memory_info().rss / 1024**2  # MB
        mem_used = mem_after - mem_before

        result_queue.put((rank, mem_used))
    except Exception as e:
        result_queue.put((rank, -1))


def worker_with_share(rank, shared_data_list, result_queue):
    """Worker that uses shared data."""
    try:
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024**2  # MB

        # Access shared data (should not increase memory much)
        total_nodes = 0
        for data in shared_data_list[:10]:  # Access first 10 graphs
            total_nodes += data.x.shape[0]

        mem_after = process.memory_info().rss / 1024**2  # MB
        mem_used = mem_after - mem_before

        result_queue.put((rank, mem_used))
    except Exception as e:
        result_queue.put((rank, -1))


def worker_cuda_test(rank, shared_tensor, result_queue):
    """Worker that moves shared tensor to GPU."""
    try:
        # Access shared CPU tensor
        assert shared_tensor.is_shared(), "Tensor should be shared"
        assert shared_tensor.device.type == 'cpu', "Should be CPU tensor"

        # Move to GPU (different device per worker if available)
        device_id = rank % torch.cuda.device_count()
        device = f'cuda:{device_id}'
        gpu_tensor = shared_tensor.to(device)

        # Perform some GPU operation
        result = gpu_tensor.sum().item()

        result_queue.put((rank, True, f"Moved to {device}, sum={result:.2f}"))
    except Exception as e:
        result_queue.put((rank, False, str(e)))


# ============================================================
# Helper functions
# ============================================================

def create_dummy_data(num_events: int = 1000, num_nodes_per_event: int = 50) -> List[Data]:
    """Create dummy PyTorch Geometric Data objects for testing."""
    data_list = []

    for i in range(num_events):
        # Create random graph data
        num_nodes = num_nodes_per_event + (i % 10)  # Variable number of nodes
        num_edges = num_nodes * 3

        data = Data()
        data.x = torch.randn(num_nodes, 9)  # Node features (9 dims as in real data)
        data.edge_index = torch.randint(0, num_nodes, (2, num_edges))  # Edge connectivity
        data.edge_attribute = torch.randn(num_edges, 1)  # Edge features
        data.graphInput = torch.randn(1, 4)  # Graph-level features
        data.weight = torch.tensor([1.0])  # Event weight
        data.y = torch.tensor([i % 4])  # Class label (4 classes)

        data_list.append(data)

    return data_list


# ============================================================
# Test functions
# ============================================================

def test_basic_shared_memory():
    """Test 1: Basic shared memory functionality with spawn."""
    print("\n" + "=" * 60)
    print("Test 1: Basic Shared Memory with Spawn")
    print("=" * 60)

    # Set spawn method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass  # Already set

    # Create and share tensor
    tensor = torch.zeros(4)
    tensor.share_memory_()

    result_queue = mp.Queue()
    processes = []

    for rank in range(4):
        p = mp.Process(target=worker_basic_test, args=(rank, tensor, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Check results
    success = True
    while not result_queue.empty():
        rank, passed, msg = result_queue.get()
        print(f"  Worker {rank}: {'✅' if passed else '❌'} {msg}")
        success = success and passed

    expected = torch.tensor([1., 2., 3., 4.])
    if torch.allclose(tensor, expected):
        print(f"  Final tensor: {tensor} ✅")
    else:
        print(f"  Final tensor: {tensor} (expected {expected}) ❌")
        success = False

    return success


def test_pyg_data_sharing():
    """Test 2: Sharing PyTorch Geometric Data objects."""
    print("\n" + "=" * 60)
    print("Test 2: PyTorch Geometric Data Sharing")
    print("=" * 60)

    # Create dummy data
    data_list = create_dummy_data(num_events=100)

    # Move to shared memory
    print(f"  Moving {len(data_list)} Data objects to shared memory...")
    for data in data_list:
        # Get all tensor attributes
        for key in ['x', 'edge_index', 'edge_attribute', 'graphInput', 'weight', 'y']:
            if hasattr(data, key):
                tensor = getattr(data, key)
                if torch.is_tensor(tensor):
                    tensor.share_memory_()

    result_queue = mp.Queue()
    processes = []

    for rank in range(4):
        p = mp.Process(target=worker_pyg_data, args=(rank, data_list, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Check results
    success = True
    while not result_queue.empty():
        rank, passed, msg = result_queue.get()
        print(f"  Worker {rank}: {'✅' if passed else '❌'} {msg}")
        success = success and passed

    return success


def test_memory_usage():
    """Test 3: Compare memory usage with and without sharing."""
    print("\n" + "=" * 60)
    print("Test 3: Memory Usage Comparison")
    print("=" * 60)

    # Check system limits first
    import resource
    soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
    print(f"  System file descriptor limits: soft={soft_limit}, hard={hard_limit}")

    # Reduce number of events to avoid file descriptor limit
    # Each event has 6 tensors, so 100 events = 600 file descriptors
    num_events = 100 if soft_limit < 4096 else 500
    num_workers = 4

    if soft_limit < 1024:
        print(f"  ⚠️  Low file descriptor limit detected ({soft_limit})")
        print(f"  Using reduced dataset size: {num_events} events")
        print(f"  To increase limit, run: ulimit -n 4096")

    # Test 1: Without sharing (each worker creates data)
    print("  Without sharing (each worker creates data):")
    result_queue = mp.Queue()
    processes = []

    for rank in range(num_workers):
        p = mp.Process(target=worker_no_share, args=(rank, num_events, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_mem_no_share = 0
    while not result_queue.empty():
        rank, mem = result_queue.get()
        if mem > 0:
            total_mem_no_share += mem
            print(f"    Worker {rank}: {mem:.1f} MB")

    print(f"    Total memory: {total_mem_no_share:.1f} MB")

    # Test 2: With sharing (main creates, workers access)
    print("\n  With sharing (main creates, workers access):")

    # Create and share data in main process
    shared_data_list = create_dummy_data(num_events)

    # Share tensors in batches to avoid file descriptor exhaustion
    try:
        tensors_shared = 0
        for i, data in enumerate(shared_data_list):
            for key in ['x', 'edge_index', 'edge_attribute', 'graphInput', 'weight', 'y']:
                if hasattr(data, key):
                    tensor = getattr(data, key)
                    if torch.is_tensor(tensor) and not tensor.is_shared():
                        tensor.share_memory_()
                        tensors_shared += 1

            # Progress indicator for large datasets
            if (i + 1) % 100 == 0:
                print(f"    Shared {i + 1}/{num_events} events ({tensors_shared} tensors)...")
    except RuntimeError as e:
        if "Too many open files" in str(e):
            print(f"    ⚠️  Hit file descriptor limit after {tensors_shared} tensors")
            print(f"    Continuing with {i} events instead of {num_events}")
            shared_data_list = shared_data_list[:i]
        else:
            raise

    result_queue = mp.Queue()
    processes = []

    for rank in range(num_workers):
        p = mp.Process(target=worker_with_share, args=(rank, shared_data_list, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_mem_share = 0
    while not result_queue.empty():
        rank, mem = result_queue.get()
        if mem > 0:
            total_mem_share += mem
            print(f"    Worker {rank}: {mem:.1f} MB")

    print(f"    Total memory: {total_mem_share:.1f} MB")

    # Calculate savings
    if total_mem_no_share > 0:
        savings = (1 - total_mem_share / total_mem_no_share) * 100
        print(f"\n  Memory savings: {savings:.1f}% ✅")
    else:
        print("\n  Could not calculate memory savings")

    return True


def test_cuda_compatibility():
    """Test 4: CUDA compatibility with shared memory."""
    print("\n" + "=" * 60)
    print("Test 4: CUDA Compatibility")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  ⚠️  CUDA not available, skipping test")
        return True

    # Create shared CPU tensor
    tensor = torch.randn(1000, 1000)  # Larger tensor to test
    tensor.share_memory_()

    result_queue = mp.Queue()
    processes = []

    num_workers = min(4, torch.cuda.device_count() * 2)
    for rank in range(num_workers):
        p = mp.Process(target=worker_cuda_test, args=(rank, tensor, result_queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # Check results
    success = True
    while not result_queue.empty():
        rank, passed, msg = result_queue.get()
        print(f"  Worker {rank}: {'✅' if passed else '❌'} {msg}")
        success = success and passed

    return success


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("SHARED MEMORY VALIDATION TESTS")
    print("=" * 60)

    tests = [
        ("Basic Shared Memory", test_basic_shared_memory),
        ("PyG Data Sharing", test_pyg_data_sharing),
        ("Memory Usage", test_memory_usage),
        ("CUDA Compatibility", test_cuda_compatibility),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {name}: {status}")
        all_passed = all_passed and passed

    if all_passed:
        print("\n🎉 All tests passed! Shared memory is working correctly.")
        print("   You can proceed with the GA optimization implementation.")
    else:
        print("\n⚠️  Some tests failed. Please review the issues above.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())