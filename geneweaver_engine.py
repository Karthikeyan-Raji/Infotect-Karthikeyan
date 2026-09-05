"""
GeneWeaver — Member 1: HPC / GPU Compute Engine
Capabilities:
  • Week 1: CPU Baseline Matchers (Brute-Force & Space-Optimized Smith-Waterman)
  • Week 2: Numba CUDA Kernel + 2-Bit Nucleotide Compression (H2D & D2H Orchestration)
  • Week 3: Multi-GPU Load-Balanced Dask Scheduling with 23-bp Boundary Overlaps
  • Week 4: Block-Level Shared Memory Optimization to Minimize VRAM Bus Latency
"""

from __future__ import annotations

import os
import time
import math
import psutil
import numpy as np
from typing import List, Tuple, Dict, Any, Optional

# -------------------------------------------------------------------------
# CUDA Availability & Safety Guard
# -------------------------------------------------------------------------
cuda = None
_NUMBA_CUDA_ERROR = None

try:
    import numba
    try:
        from numba import cuda as _cuda
        cuda = _cuda
    except Exception as exc:
        _NUMBA_CUDA_ERROR = exc
except Exception as exc:
    _NUMBA_CUDA_ERROR = exc


def is_cuda_available() -> bool:
    """Verifies that Numba CUDA is installed and an accessible GPU device exists."""
    if cuda is None:
        return False
    try:
        return cuda.is_available()
    except Exception:
        return False


def get_gpu_count() -> int:
    """Returns number of physical CUDA GPUs available."""
    if not is_cuda_available():
        return 0
    try:
        return len(cuda.gpus)
    except Exception:
        return 1


# -------------------------------------------------------------------------
# Global Parameters & Encoding LUTs
# -------------------------------------------------------------------------
THREADS_PER_BLOCK = 256
MAX_HITS_CAPACITY = 200_000

# 2-Bit nucleotide map: A=00, C=01, G=10, T=11
NUC_BIT_MAP = np.full(256, 0, dtype=np.uint8)
NUC_BIT_MAP[ord("A")] = 0b00
NUC_BIT_MAP[ord("a")] = 0b00
NUC_BIT_MAP[ord("C")] = 0b01
NUC_BIT_MAP[ord("c")] = 0b01
NUC_BIT_MAP[ord("G")] = 0b10
NUC_BIT_MAP[ord("g")] = 0b10
NUC_BIT_MAP[ord("T")] = 0b11
NUC_BIT_MAP[ord("t")] = 0b11

BIT_TO_NUC = {0b00: "A", 0b01: "C", 0b10: "G", 0b11: "T"}


def pack_dna_sequence(dna_str: str) -> np.ndarray:
    """
    Compresses ASCII nucleotide strings into 2-bit packed uint8 arrays.
    Reduces memory footprint by 75% (4 bases per byte).
    """
    ascii_bytes = np.frombuffer(dna_str.encode("ascii"), dtype=np.uint8)
    mapped = NUC_BIT_MAP[ascii_bytes]
    remainder = len(mapped) % 4
    if remainder != 0:
        pad_len = 4 - remainder
        mapped = np.pad(mapped, (0, pad_len), mode="constant", constant_values=0)

    packed = (
        (mapped[0::4] << 6)
        | (mapped[1::4] << 4)
        | (mapped[2::4] << 2)
        | (mapped[3::4])
    )
    return packed.astype(np.uint8)


def encode_target_to_uint64(target_20bp: str) -> np.uint64:
    """Packs a 20-bp guide RNA string into a 64-bit unsigned integer register."""
    if len(target_20bp) != 20:
        raise ValueError(f"sgRNA must be exactly 20 bp, received {len(target_20bp)} bp.")
    packed = pack_dna_sequence(target_20bp)
    target_val = np.uint64(0)
    for b in range(5):
        target_val = (target_val << np.uint64(8)) | np.uint64(packed[b])
    return target_val


# -------------------------------------------------------------------------
# Week 1: CPU Baseline Alignment Engines
# -------------------------------------------------------------------------
def run_brute_force_cpu(
    genome: str,
    query: str,
    max_mismatches: int = 4
) -> List[Tuple[int, int]]:
    """Linear character-by-character CPU sliding window baseline."""
    n = len(genome)
    m = len(query)
    hits = []

    for i in range(n - m + 1):
        mismatches = 0
        for j in range(m - 3):
            if genome[i + j] != query[j]:
                mismatches += 1
                if mismatches > max_mismatches:
                    break

        if mismatches <= max_mismatches:
            # Check SpCas9 PAM ('NGG')
            if genome[i + 21] == "G" and genome[i + 22] == "G":
                hits.append((i, mismatches))

    return hits


def run_smith_waterman_cpu(
    genome: str,
    query: str,
    max_mismatches: int = 4
) -> List[Tuple[int, int]]:
    """Space-optimized Smith-Waterman local alignment using alternating rows."""
    n, m = len(genome), len(query)
    prev_row = [0] * (m + 1)
    curr_row = [0] * (m + 1)
    hits = []
    threshold = (m - max_mismatches) * 2 - (max_mismatches * 1)

    for i in range(1, n + 1):
        g_char = genome[i - 1]
        for j in range(1, m + 1):
            score = 2 if g_char == query[j - 1] else -1
            match = prev_row[j - 1] + score
            delete = prev_row[j] - 2
            insert = curr_row[j - 1] - 2
            curr_row[j] = max(0, match, delete, insert)

            if j == m and curr_row[j] >= threshold:
                hits.append((i - m, curr_row[j]))

        prev_row[:] = curr_row[:]
        curr_row = [0] * (m + 1)

    return hits


# -------------------------------------------------------------------------
# Week 2 & Week 4: Numba CUDA Kernel (Shared Memory Accelerated)
# -------------------------------------------------------------------------
if is_cuda_available():
    @cuda.jit
    def crispr_shared_memory_kernel(
        d_genomic_packed,
        packed_len,
        genome_bp_len,
        target_64,
        max_mismatches,
        d_out_indices,
        d_out_mismatches,
        d_hit_count,
        chunk_offset_bp,
    ):
        """
        Shared-memory accelerated kernel.
        Caches 256 bytes per block + 8-byte boundary tail into __shared__ memory.
        """
        s_mem = cuda.shared.array(shape=264, dtype=numba.uint8)

        tid = cuda.threadIdx.x
        bdim = cuda.blockDim.x
        gid = cuda.grid(1)

        block_byte_start = (cuda.blockIdx.x * bdim) // 4

        # Collaborative memory load into on-chip cache
        if block_byte_start + tid < packed_len:
            s_mem[tid] = d_genomic_packed[block_byte_start + tid]
        else:
            s_mem[tid] = 0

        # Boundary tail for 23-bp window spanning past the block
        if tid < 8:
            tail_idx = block_byte_start + bdim + tid
            if tail_idx < packed_len:
                s_mem[bdim + tid] = d_genomic_packed[tail_idx]
            else:
                s_mem[bdim + tid] = 0

        cuda.syncthreads()

        if gid + 23 > genome_bp_len:
            return

        local_byte_offset = tid // 4
        bit_shift = (gid % 4) * 2

        # Assemble 7 consecutive bytes (56 bits) into a 64-bit register
        raw_64 = numba.uint64(0)
        for b in range(7):
            raw_64 = (raw_64 << numba.uint64(8)) | numba.uint64(s_mem[local_byte_offset + b])

        shifted_window = raw_64 << numba.uint64(bit_shift)
        sgrna_window = shifted_window >> numba.uint64(24)

        # Bitwise XOR comparison
        diff = sgrna_window ^ numba.uint64(target_64)
        mismatches = 0
        for i in range(20):
            pair = (diff >> numba.uint64(38 - 2 * i)) & numba.uint64(0b11)
            if pair != 0:
                mismatches += 1
                if mismatches > max_mismatches:
                    return

        # Verify PAM sequence 'GG' (0b10 0b10)
        pam1 = (shifted_window >> numba.uint64(22)) & numba.uint64(0b11)
        pam2 = (shifted_window >> numba.uint64(20)) & numba.uint64(0b11)

        if pam1 == 0b10 and pam2 == 0b10:
            slot = cuda.atomic.add(d_hit_count, 0, 1)
            if slot < d_out_indices.size:
                d_out_indices[slot] = gid + chunk_offset_bp
                d_out_mismatches[slot] = mismatches
else:
    def crispr_shared_memory_kernel(*args, **kwargs):
        raise RuntimeError("CUDA unavailable.")


# -------------------------------------------------------------------------
# Single-Device Execution Engine
# -------------------------------------------------------------------------
def execute_device_chunk(
    packed_chunk: np.ndarray,
    chunk_bp_len: int,
    target_64: int,
    max_mismatches: int,
    chunk_offset: int,
    device_id: int = 0,
) -> Dict[str, Any]:
    """Manages H2D transfer, kernel execution, and D2H collection for a slice."""
    if not is_cuda_available():
        raise RuntimeError("Cannot execute CUDA chunk: no compatible GPU detected.")

    with cuda.gpus[device_id]:
        stream = cuda.stream()
        d_packed = cuda.to_device(packed_chunk, stream=stream)
        d_indices = cuda.device_array(MAX_HITS_CAPACITY, dtype=np.int32, stream=stream)
        d_mismatches = cuda.device_array(MAX_HITS_CAPACITY, dtype=np.int32, stream=stream)
        d_hits = cuda.to_device(np.zeros(1, dtype=np.int32), stream=stream)

        blocks = (chunk_bp_len - 23 + THREADS_PER_BLOCK) // THREADS_PER_BLOCK
        crispr_shared_memory_kernel[blocks, THREADS_PER_BLOCK, stream](
            d_packed,
            len(packed_chunk),
            chunk_bp_len,
            target_64,
            max_mismatches,
            d_indices,
            d_mismatches,
            d_hits,
            chunk_offset,
        )

        stream.synchronize()
        total_hits = int(d_hits.copy_to_host(stream=stream)[0])
        hits_to_pull = min(total_hits, MAX_HITS_CAPACITY)

        out_indices = d_indices[:hits_to_pull].copy_to_host(stream=stream)
        out_mismatches = d_mismatches[:hits_to_pull].copy_to_host(stream=stream)

    return {
        "count": total_hits,
        "indices": out_indices,
        "mismatches": out_mismatches,
        "device_id": device_id,
    }


# -------------------------------------------------------------------------
# Week 3: Distributed Multi-GPU Coordination (Dask)
# -------------------------------------------------------------------------
def run_dask_multigpu_pipeline(
    genome_str: str,
    target_sgrna: str,
    max_mismatches: int = 4,
    chunk_size: int = 2_500_000,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Coordinates genome sequence search across available GPUs via Dask.
    Maintains a 23-bp boundary overlap between sequential chunks.
    """
    import dask
    from dask import delayed

    num_gpus = max(1, get_gpu_count())
    target_64 = encode_target_to_uint64(target_sgrna)

    tasks = []
    total_len = len(genome_str)
    t_start = time.perf_counter()

    chunk_idx = 0
    for start_pos in range(0, total_len, chunk_size):
        end_pos = min(start_pos + chunk_size + 23, total_len)
        chunk_str = genome_str[start_pos:end_pos]
        packed_data = pack_dna_sequence(chunk_str)
        assigned_device = chunk_idx % num_gpus

        task = delayed(execute_device_chunk)(
            packed_data,
            len(chunk_str),
            target_64,
            max_mismatches,
            start_pos,
            assigned_device,
        )
        tasks.append(task)
        chunk_idx += 1

    results = dask.compute(*tasks, scheduler="threads")
    elapsed = time.perf_counter() - t_start

    total_hits = sum(r["count"] for r in results)
    if total_hits > 0:
        all_indices = np.concatenate([r["indices"] for r in results])
        all_mismatches = np.concatenate([r["mismatches"] for r in results])
    else:
        all_indices = np.array([], dtype=np.int32)
        all_mismatches = np.array([], dtype=np.int32)

    return all_indices, all_mismatches, elapsed


# -------------------------------------------------------------------------
# Clean API Interface for Member 2
# -------------------------------------------------------------------------
class GeneWeaverHPC:
    """Unified entry point consumed by Member 2's UI and scoring pipeline."""

    @staticmethod
    def align_sequence(
        genome: str,
        sgrna: str,
        max_mismatches: int = 4,
        prefer_gpu: bool = True,
    ) -> Dict[str, Any]:
        """
        Executes alignment via GPU or falls back to CPU if hardware is absent.
        Returns:
            {
                'status': 'cuda' | 'cpu',
                'runtime_sec': float,
                'indices': np.ndarray,
                'mismatches': np.ndarray,
                'total_hits': int
            }
        """
        if prefer_gpu and is_cuda_available():
            indices, mismatches, elapsed = run_dask_multigpu_pipeline(
                genome, sgrna, max_mismatches
            )
            return {
                "status": "cuda",
                "runtime_sec": elapsed,
                "indices": indices,
                "mismatches": mismatches,
                "total_hits": len(indices),
            }

        # CPU Fallback
        t0 = time.perf_counter()
        hits = run_brute_force_cpu(genome, sgrna + "AGG", max_mismatches)
        elapsed = time.perf_counter() - t0
        return {
            "status": "cpu",
            "runtime_sec": elapsed,
            "indices": np.array([idx for idx, _ in hits], dtype=np.int32),
            "mismatches": np.array([mm for _, mm in hits], dtype=np.int32),
            "total_hits": len(hits),
        }


# -----------------------------------------------------------------------
# Verification & Self-Test Harnes
# -----------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 65)
    print("  GENEWEAVER: MEMBER 1 UNIFIED VERIFICATION SUITE  ")
    print("=" * 65)

    TEST_SIZE = 5_000_000
    TARGET_RNA = "GAGTCCGAGCAGAAGAAGAA"
    PAM = "AGG"
    QUERY = TARGET_RNA + PAM

    print(f"Generating {TEST_SIZE:,} bp synthetic sequence...")
    np.random.seed(1337)
    bases = np.array(["A", "C", "G", "T"])
    genome_sample = "".join(np.random.choice(bases, size=TEST_SIZE))

    # Inject validation targets
    target_pos1 = 250_000
    target_pos2 = 2_750_000
    mutated_sgrna = "GAGTCCTAGCAGAAGAAGAA"  # 1 mismatch

    genome_sample = (
        genome_sample[:target_pos1]
        + (TARGET_RNA + PAM)
        + genome_sample[target_pos1 + 23 :]
    )
    genome_sample = (
        genome_sample[:target_pos2]
        + (mutated_sgrna + PAM)
        + genome_sample[target_pos2 + 23 :]
    )

    # 1. CPU Benchmark on a 500k slice
    slice_len = 500_000
    t0 = time.perf_counter()
    _ = run_brute_force_cpu(genome_sample[:slice_len], QUERY, max_mismatches=4)
    t_cpu_slice = time.perf_counter() - t0
    extrapolated_cpu = (t_cpu_slice / slice_len) * TEST_SIZE
    print(f"\n[Week 1 Benchmark] CPU Brute-Force Extrapolated: {extrapolated_cpu:.2f} s")

    # 2. End-to-End Run (GPU if available, CPU if not)
    print(f"\n[Engine Run] Executing GeneWeaverHPC API...")
    res = GeneWeaverHPC.align_sequence(genome_sample, TARGET_RNA, max_mismatches=4)

    print(f"  ├── Engine Selected: {res['status'].upper()}")
    print(f"  ├── Total Hits Detected: {res['total_hits']}")
    print(f"  └── Execution Wall Time: {res['runtime_sec']:.4f} s")

    if res["status"] == "cuda" and res["runtime_sec"] > 0:
        speedup = extrapolated_cpu / res["runtime_sec"]
        print(f"  ★ Projected Speedup vs CPU: {speedup:.1f}x")

    assert target_pos1 in res["indices"], f"Target lost at position {target_pos1}"
    assert target_pos2 in res["indices"], f"Off-target lost at position {target_pos2}"
    print("\n[SUCCESS] Correctness check passed. All injected sites detected.")