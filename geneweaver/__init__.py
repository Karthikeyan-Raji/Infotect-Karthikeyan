"""
geneweaver – GPU compute environment package.

Sub-modules
-----------
gpu_info   : Detect and describe CUDA devices.
numba_kernels : Example Numba CUDA kernels.
dask_gpu   : Dask cluster helpers (CPU + GPU modes).
"""

__version__ = "0.1.0"
__all__ = ["gpu_info", "numba_kernels", "dask_gpu"]
