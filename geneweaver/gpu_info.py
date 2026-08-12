"""
gpu_info.py – Runtime CUDA / GPU environment diagnostics.

Provides:
  - probe()         : Returns a dict of GPU info; safe on CPU-only machines.
  - print_summary() : Pretty-prints GPU/CUDA environment to console.
  - require_gpu()   : Raises RuntimeError if no usable GPU is found.

Notes
-----
Numba in WDDM mode (Windows display driver) reports ``cuda.is_available()``
as False even when a GPU exists because WDDM enables a watchdog timer that
prevents persistent kernel contexts.  We work around this by attempting to
select device 0 directly and treating success as "GPU available".
"""

from __future__ import annotations

import os
import sys
from typing import Any

_NUMBA_AVAILABLE = False
_CUPY_AVAILABLE = False

try:
    from numba import cuda as _numba_cuda
    _NUMBA_AVAILABLE = True
except ImportError:
    _numba_cuda = None  # type: ignore[assignment]

try:
    import cupy as _cupy  # type: ignore[import-untyped]
    _CUPY_AVAILABLE = True
except ImportError:
    _cupy = None  # type: ignore[assignment]


def _try_select_numba_device(device_id: int = 0) -> bool:
    """
    Attempt to select a CUDA device via Numba.

    Returns True on success, False on any error.  Handles the WDDM
    'is_available() == False but GPU exists' quirk on Windows.
    """
    if not _NUMBA_AVAILABLE:
        return False
    try:
        _numba_cuda.select_device(device_id)
        _numba_cuda.get_current_device()
        return True
    except Exception:
        return False


def probe(verbose: bool = False) -> dict[str, Any]:
    """
    Probe the CUDA environment and return a structured info dict.

    Parameters
    ----------
    verbose : bool
        If True, include per-device attribute detail.

    Returns
    -------
    dict with keys:
        numba_available  bool  – Numba importable
        cupy_available   bool  – CuPy importable
        gpu_usable       bool  – At least one GPU can be initialised
        device_count     int   – Number of CUDA GPUs detected
        devices          list  – Per-device detail dicts
        wddm_mode        bool  – Windows Display Driver Model active
        cuda_driver_ver  str   – CUDA driver version string (from nvidia-smi)
        python_version   str   – Running Python version
    """
    info: dict[str, Any] = {
        "numba_available": _NUMBA_AVAILABLE,
        "cupy_available": _CUPY_AVAILABLE,
        "gpu_usable": False,
        "device_count": 0,
        "devices": [],
        "wddm_mode": sys.platform == "win32",
        "cuda_driver_ver": _driver_version(),
        "python_version": sys.version,
    }

    if not _NUMBA_AVAILABLE:
        return info

    # --- Enumerate devices via numba.cuda.detect() output ---
    try:
        gpu_list = _numba_cuda.gpus.lst
        info["device_count"] = len(gpu_list)
    except Exception:
        return info

    for idx, _dev in enumerate(gpu_list):
        try:
            _numba_cuda.select_device(idx)
            dev = _numba_cuda.get_current_device()
            cc = dev.compute_capability
            dev_info: dict[str, Any] = {
                "id": idx,
                "name": dev.name.decode() if isinstance(dev.name, bytes) else dev.name,
                "compute_capability": f"{cc[0]}.{cc[1]}",
                "usable": True,
            }
            if verbose:
                try:
                    ctx = dev.get_primary_context()
                    dev_info["pci_bus_id"] = getattr(dev, "pci_bus_id", "N/A")
                except Exception:
                    pass
            info["devices"].append(dev_info)
            info["gpu_usable"] = True
        except Exception as exc:
            info["devices"].append({"id": idx, "usable": False, "error": str(exc)})

    return info


def _driver_version() -> str:
    """Return CUDA driver version from nvidia-smi, or 'unknown'."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def print_summary(verbose: bool = False) -> None:
    """Pretty-print GPU environment info using Rich (falls back to plain print)."""
    info = probe(verbose=verbose)

    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        console.rule("[bold cyan]geneweaver – GPU Environment[/bold cyan]")

        table = Table(box=box.ROUNDED, show_header=True, header_style="bold magenta")
        table.add_column("Property", style="cyan", min_width=24)
        table.add_column("Value", style="white")

        status = lambda v: "[green]✓ Yes[/green]" if v else "[red]✗ No[/red]"  # noqa: E731

        table.add_row("Python", info["python_version"].split(" ")[0])
        table.add_row("Numba available", status(info["numba_available"]))
        table.add_row("CuPy available", status(info["cupy_available"]))
        table.add_row("GPU usable", status(info["gpu_usable"]))
        table.add_row("CUDA driver version", info["cuda_driver_ver"])
        table.add_row("WDDM mode (Windows)", status(info["wddm_mode"]))
        table.add_row("Device count", str(info["device_count"]))

        for d in info["devices"]:
            prefix = f"  GPU {d['id']}"
            if d.get("usable"):
                table.add_row(prefix, f"{d['name']}  [CC {d['compute_capability']}]")
            else:
                table.add_row(prefix, f"[red]UNAVAILABLE – {d.get('error', '?')}[/red]")

        console.print(table)
        console.rule()

    except ImportError:
        # Plain fallback
        print("=" * 50)
        print("geneweaver – GPU Environment")
        print("=" * 50)
        for k, v in info.items():
            print(f"  {k}: {v}")
        print("=" * 50)


def require_gpu(min_devices: int = 1) -> None:
    """
    Assert that at least *min_devices* GPUs are usable.

    Raises
    ------
    RuntimeError if the requirement is not met.
    """
    info = probe()
    usable = sum(1 for d in info["devices"] if d.get("usable", False))
    if usable < min_devices:
        raise RuntimeError(
            f"require_gpu: need {min_devices} GPU(s), found {usable} usable. "
            f"Total detected: {info['device_count']}. "
            f"Numba available: {info['numba_available']}."
        )
