#!/usr/bin/env python
"""Preload module for the multiprocessing forkserver used by launchGAOptim.py.

The forkserver imports this once; every worker is then forked from it, so torch, PyG,
ROOT and the analysis modules are not re-imported per worker (~45 s each on Perlmutter
GPU nodes, where the conda env in $HOME is slow to read).

The forkserver must never initialize CUDA, or forked workers fail with
"CUDA error: initialization error". Importing torch_geometric does initialize it:
torch_geometric/typing.py calls torch.__config__.show() (which queries the CUDA
driver/runtime/cuDNN versions) only to look for 'USE_MKL=OFF'. While importing PyG we
therefore swap show() for a CPU-only answer to that same question, then restore it.
"""

import torch

if torch.cuda.is_initialized():
    raise RuntimeError("forkserverPreload: CUDA is already initialized; forked workers could not use it")

_original_show = torch.__config__.show


def _show_without_cuda():
    """Same answer to PyG's MKL check as torch.__config__.show(), without touching CUDA."""
    return "" if torch.backends.mkl.is_available() else "USE_MKL=OFF"


torch.__config__.show = _show_without_cuda
try:
    import torch_geometric  # noqa: F401  (typing.py calls torch.__config__.show())
finally:
    torch.__config__.show = _original_show

# Everything launchGAOptim.py imports (ROOT comes in via evaluateGAModels -> OverfittingDetector)
import GAConfig  # noqa: F401,E402
import GATools  # noqa: F401,E402
import evaluateGAModels  # noqa: F401,E402
import SharedDatasetManager  # noqa: F401,E402
import DynamicDatasetLoader  # noqa: F401,E402
import trainWorker  # noqa: F401,E402
import SharedWorkerUtils  # noqa: F401,E402
