"""
dask_gpu.py – Dask cluster helpers for CPU and GPU workloads.

Provides
--------
make_local_cluster(n_workers, threads_per_worker, **kwargs)
    : Spin up a Dask ``LocalCluster`` for CPU-based parallel work.

make_gpu_cluster(n_gpus, **kwargs)
    : Spin up a Dask cluster where each worker owns one GPU.
      Falls back to CPU LocalCluster if dask_cuda is unavailable.

demo_dask_array(size, chunks)
    : Run a small Dask array computation and return the result.

demo_dask_dataframe(n_rows, n_partitions)
    : Build a Dask DataFrame, do a groupby-mean, return result.

Notes
-----
dask-cuda (RAPIDS) is the preferred GPU-aware scheduler on Linux.
On Windows, dask-cuda is unsupported; we fall back to a LocalCluster
with ``CUDA_VISIBLE_DEVICES`` set per worker via a preload plugin.
"""

from __future__ import annotations

import os
import logging
from contextlib import contextmanager
from typing import Generator

import numpy as np
import dask
import dask.array as da
import dask.dataframe as dd
from distributed import Client, LocalCluster

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cluster factories
# ---------------------------------------------------------------------------


def make_local_cluster(
    n_workers: int = 2,
    threads_per_worker: int = 2,
    memory_limit: str = "2GB",
    dashboard_address: str = "localhost:8787",
    **kwargs,
) -> LocalCluster:
    """
    Create a CPU-based Dask ``LocalCluster``.

    Parameters
    ----------
    n_workers           : Number of worker processes.
    threads_per_worker  : Threads per worker (keep low for GIL-heavy tasks).
    memory_limit        : Per-worker memory cap (e.g. '4GB').
    dashboard_address   : Dask dashboard URL.

    Returns
    -------
    distributed.LocalCluster  (call ``.close()`` when done).
    """
    logger.info(
        "Starting LocalCluster: %d workers × %d threads (mem: %s)",
        n_workers, threads_per_worker, memory_limit,
    )
    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        dashboard_address=dashboard_address,
        **kwargs,
    )
    logger.info("Dashboard: %s", cluster.dashboard_link)
    return cluster


def make_gpu_cluster(n_gpus: int = 1, **kwargs) -> LocalCluster:
    """
    Create a GPU-aware Dask cluster.

    On Linux with dask-cuda installed: uses ``LocalCUDACluster``.
    On Windows (or without dask-cuda): falls back to ``LocalCluster``
    and sets ``CUDA_VISIBLE_DEVICES`` via worker env.

    Parameters
    ----------
    n_gpus : int – Number of GPUs to use (one worker per GPU).

    Returns
    -------
    distributed.LocalCluster or dask_cuda.LocalCUDACluster
    """
    try:
        from dask_cuda import LocalCUDACluster  # type: ignore[import]
        logger.info("dask-cuda available – starting LocalCUDACluster with %d GPU(s)", n_gpus)
        cluster = LocalCUDACluster(n_workers=n_gpus, **kwargs)
        logger.info("GPU Dashboard: %s", cluster.dashboard_link)
        return cluster
    except ImportError:
        logger.warning(
            "dask-cuda not available (Windows / not installed). "
            "Falling back to LocalCluster with CUDA_VISIBLE_DEVICES per worker."
        )
        # Build per-worker env: worker 0 → GPU 0, worker 1 → GPU 1, …
        worker_kwargs = {
            "env": {"CUDA_VISIBLE_DEVICES": str(i % n_gpus)}
        }
        cluster = LocalCluster(
            n_workers=n_gpus,
            threads_per_worker=1,
            worker_kwargs=worker_kwargs,
            dashboard_address="localhost:8787",
            **kwargs,
        )
        logger.info("Fallback Dashboard: %s", cluster.dashboard_link)
        return cluster


@contextmanager
def local_client(
    n_workers: int = 2,
    threads_per_worker: int = 2,
    **kwargs,
) -> Generator[Client, None, None]:
    """
    Context manager: spin up a LocalCluster + Client, tear down on exit.

    Usage
    -----
    >>> with local_client(n_workers=4) as client:
    ...     result = client.submit(my_func, *args).result()
    """
    cluster = make_local_cluster(n_workers=n_workers,
                                 threads_per_worker=threads_per_worker, **kwargs)
    client = Client(cluster)
    try:
        logger.info("Dask client ready: %s", client)
        yield client
    finally:
        client.close()
        cluster.close()
        logger.info("Dask cluster shut down.")


# ---------------------------------------------------------------------------
# Demo / smoke-test helpers
# ---------------------------------------------------------------------------


def demo_dask_array(
    size: int = 10_000,
    chunks: int = 1_000,
    client: Client | None = None,
) -> np.ndarray:
    """
    Create a random Dask array, compute ``x**2 + x`` and return as NumPy.

    Parameters
    ----------
    size   : Total number of elements.
    chunks : Chunk size.
    client : Optional existing Dask ``Client``.

    Returns
    -------
    np.ndarray (float64)
    """
    x = da.random.random(size, chunks=chunks)
    result = (x ** 2 + x).compute()
    return result


def demo_dask_dataframe(
    n_rows: int = 100_000,
    n_partitions: int = 4,
) -> "pd.DataFrame":  # type: ignore[name-defined]  # noqa: F821
    """
    Build a synthetic Dask DataFrame, compute a groupby-mean.

    Returns
    -------
    pandas.DataFrame  with columns 'group', 'value_mean'.
    """
    import pandas as pd

    pdf = pd.DataFrame({
        "group": np.random.choice(["A", "B", "C", "D"], size=n_rows),
        "value": np.random.randn(n_rows),
    })
    ddf = dd.from_pandas(pdf, npartitions=n_partitions)
    result = ddf.groupby("group")["value"].mean().compute().reset_index()
    result.columns = ["group", "value_mean"]
    return result


def cluster_info(client: Client) -> dict:
    """Return a dict describing the Dask cluster connected to *client*."""
    info = client.scheduler_info()
    workers = info.get("workers", {})
    return {
        "scheduler_address": client.scheduler.address,
        "dashboard_link": client.dashboard_link,
        "n_workers": len(workers),
        "total_threads": sum(w.get("nthreads", 0) for w in workers.values()),
        "total_memory_gb": round(
            sum(w.get("memory_limit", 0) for w in workers.values()) / 1e9, 2
        ),
    }
