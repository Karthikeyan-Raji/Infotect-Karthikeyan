import os
import time
import psutil
import numpy as np
from typing import List, Tuple

# Try importing Numba CUDA safely for environments without GPU
cuda = None
_NUMBA_CUDA_IMPORT_ERROR = None
try:
    # Import numba and attempt to access its cuda submodule.
    # include numba without CUDA support, so guard this separately.
    import numba  # noqa: F401
    try:
        from numba import cud
    except Exception as sub_err:
        cuda = None
        _NUMBA_CUDA_IMPORT_ERROR = sub_err
except Exception as err:
    cuda = None
    _NUMBA_CUDA_IMPORT_ERROR = err

# Hyperparameters & Global Configuration
# =====================================================================
GENOME_SIZE_SIMULATION = 5_000_000  # 5 Million Base Pairs
TARGET_SGRNA = "GAGTCCGAGCAGAAGAAGAA"  # 20-bp guide RNA
PAM_PATTERN = "AGG"                     # 3-bp PAM site (NGG)
QUERY_SEQ = TARGET_SGRNA + PAM_PATTERN  # Total sequence length M = 23
MAX_MISMATCHES = 4                     # CRISPR mismatch threshold
THREADS_PER_BLOCK = 256
MAX_HITS_CAPACITY = 100_000            # VRAM buffer capacity for hit indices

# 2-Bit Nucleotide Encoding Lookup Table: A=00, C=01, G=10, T=11
NUC_BIT_MAP = np.full(256, 0, dtype=np.uint8)
NUC_BIT_MAP[ord('A')] = 0b00
NUC_BIT_MAP[ord('C')] = 0b01
NUC_BIT_MAP[ord('G')] = 0b10
NUC_BIT_MAP[ord('T')] = 0b11

BIT_TO_NUC = {0b00: 'A', 0b01: 'C', 0b10: 'G', 0b11: 'T'}


# ====================================================================
# System Diagnostic & Utility Functions
# ====================================================================
def get_process_memory_mb() -> float:
    """Returns current process Resident Set Size (RSS) memory in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def cuda_available() -> bool:
    """Checks if Numba CUDA is importable and an active GPU device is detected."""
    if cuda is None:
        return False
    try:
        return cuda.is_available()
    except Exception:
        return False


def cuda_error_message() -> str:
    """Provides detailed diagnostic output if CUDA initialization fails."""
    if cuda is None:
        if _NUMBA_CUDA_IMPORT_ERROR is not None:
            return str(_NUMBA_CUDA_IMPORT_ERROR)
        return "Numba CUDA library could not be imported."
    try:
        if not cuda.is_available():
            return "CUDA library detected, but no compatible GPU driver/device was found."
    except Exception as err:
        return str(err)
    return "Unknown CUDA initialization error."


def generate_synthetic_dna(length: int) -> str:
    """Generates a deterministic pseudo-random DNA sequence composed of {A, C, G, T}."""
    np.random.seed(42)
    bases = np.array(['A', 'C', 'G', 'T'])
    return "".join(np.random.choice(bases, size=length))


# ====================================================================
# Bit-Packing Utilities (4x Compression Ratio)
# ====================================================================
def pack_dna_sequence(dna_str: str) -> np.ndarray:
    """Packs ASCII DNA text into a 2-bit uint8 NumPy array (4 bases per byte)."""
    ascii_bytes = np.frombuffer(dna_str.upper().encode('ascii'), dtype=np.uint8)
    mapped_2bit = NUC_BIT_MAP[ascii_bytes]
    pad_len = (4 - (len(mapped_2bit) % 4)) % 4
    if pad_len > 0:
        mapped_2bit = np.pad(mapped_2bit, (0, pad_len), 'constant', constant_values=0)
    
    packed = (mapped_2bit[0::4] << 6) | \
             (mapped_2bit[1::4] << 4) | \
             (mapped_2bit[2::4] << 2) | \
             (mapped_2bit[3::4])
             
    return packed.astype(np.uint8)


def unpack_2bit_window(packed_arr: np.ndarray, byte_idx: int, nucleotide_len: int = 23) -> str:
    """Reconstructs ASCII text from packed bytes for correctness verification."""
    chars = []
    for i in range(nucleotide_len):
        b_idx = byte_idx + (i // 4)
        shift = 6 - 2 * (i % 4)
        bit_val = (packed_arr[b_idx] >> shift) & 0b11
        chars.append(BIT_TO_NUC[bit_val])
    return ''.join(chars)


# =======================================================================
# 1. CPU Alignment Algorithms (Baseline Implementations
# =======================================================================
def run_brute_force_matcher(genome: str, query: str, max_mismatches: int) -> List[Tuple[int, int]]:
    """Linear character-by-character CPU sliding window matcher O(N * M)."""
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
            if genome[i + 21] == 'G' and genome[i + 22] == 'G':
                hits.append((i, mismatches))
                
    return hits


def run_smith_waterman_scores(
    genome: str, 
    query: str, 
    match_score: int = 2, 
    mismatch_penalty: int = -1, 
    gap_penalty: int = -2
) -> List[Tuple[int, int]]:
    """Space-optimized Smith-Waterman local alignment scoring using 2-row buffers O(N * M)."""
    n = len(genome)
    m = len(query)
    
    prev_row = [0] * (m + 1)
    curr_row = [0] * (m + 1)
    
    high_scoring_sites = []
    threshold = (m - MAX_MISMATCHES) * match_score + (MAX_MISMATCHES * mismatch_penalty)
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if genome[i - 1] == query[j - 1]:
                score = match_score
            else:
                score = mismatch_penalty
                
            match = prev_row[j - 1] + score
            delete = prev_row[j] + gap_penalty
            insert = curr_row[j - 1] + gap_penalty
            
            curr_row[j] = max(0, match, delete, insert)
            
            if j == m and curr_row[j] >= threshold:
                high_scoring_sites.append((i - m, curr_row[j]))
                
        prev_row = list(curr_row)
        curr_row = [0] * (m + 1)
        
    return high_scoring_sites


# ========================================================================
# 2. CUDA Kernel Engine (Parallel Bitwise Off-Target Matcher)
# ========================================================================
if cuda is not None:
    @cuda.jit
    def crispr_offtarget_kernel(
        d_genomic_packed,
        packed_len,
        genome_bp_len,
        target_encoded,
        max_mismatches,
        d_out_indices,
        d_out_mismatches,
        d_hit_count,
    ):
        """GPU Kernel evaluating target alignment & PAM presence per thread index."""
        global_bp_idx = cuda.grid(1)
        if global_bp_idx + 23 > genome_bp_len:
            return

        byte_offset = global_bp_idx // 4
        bit_shift_offset = (global_bp_idx % 4) * 2
        if byte_offset + 6 >= packed_len:
            return

        # Fetch 7 consecutive bytes into a 64-bit register
        raw_64 = np.uint64(0)
        for b in range(7):
            raw_64 = (raw_64 << np.uint64(8)) | np.uint64(d_genomic_packed[byte_offset + b])

        shifted_window = raw_64 << np.uint64(bit_shift_offset)
        sgrna_window = np.uint64(shifted_window >> np.uint64(24))
        target_sgrna = np.uint64(target_encoded)

        # Bitwise XOR comparison
        xor_diff = sgrna_window ^ target_sgrna
        mismatches = 0
        for i in range(20):
            pair_diff = (xor_diff >> np.uint64(38 - 2 * i)) & np.uint64(0b11)
            if pair_diff != np.uint64(0):
                mismatches += 1
                if mismatches > max_mismatches:
                    return

        # PAM Site NGG Verification
        pam_bp1 = (shifted_window >> np.uint64(22)) & np.uint64(0b11)
        pam_bp2 = (shifted_window >> np.uint64(20)) & np.uint64(0b11)
        if pam_bp1 == np.uint64(0b10) and pam_bp2 == np.uint64(0b10):
            hit_slot = cuda.atomic.add(d_hit_count, 0, 1)
            if hit_slot < d_out_indices.size:
                d_out_indices[hit_slot] = global_bp_idx
                d_out_mismatches[hit_slot] = mismatches
else:
    def crispr_offtarget_kernel(*args, **kwargs):
        raise RuntimeError("CUDA kernel unavailable: Numba CUDA library not installed or configured.")


def run_cpu_fallback(genome_str: str, target_sgrna_str: str, max_mismatch: int = 4):
    """Fallback executor when CUDA hardware is unavailable."""
    print("[WARNING] Executing CPU fallback pipeline...")
    query = target_sgrna_str + "AGG"
    start_time = time.perf_counter()
    hits = run_brute_force_matcher(genome_str, query, max_mismatch)
    duration = time.perf_counter() - start_time
    
    return {
        'hits_count': len(hits),
        'indices': np.array([pos for pos, _ in hits], dtype=np.int32),
        'mismatches': np.array([mm for _, mm in hits], dtype=np.int32),
        'h2d_time': 0.0,
        'kernel_time': duration,
        'd2h_time': 0.0,
        'total_gpu_time': duration,
    }


def run_gpu_alignment(genome_str: str, target_sgrna_str: str, max_mismatch: int = 4):
    """Pipeline coordinator managing memory transfers, kernel execution, and response parsing."""
    if len(genome_str) < 23:
        return {
            'hits_count': 0,
            'indices': np.array([], dtype=np.int32),
            'mismatches': np.array([], dtype=np.int32),
            'h2d_time': 0.0,
            'kernel_time': 0.0,
            'd2h_time': 0.0,
            'total_gpu_time': 0.0,
        }

    if not cuda_available():
        print(f"[WARNING] {cuda_error_message()}")
        return run_cpu_fallback(genome_str, target_sgrna_str, max_mismatch)

    print(f"1. Packing genomic string ({len(genome_str):,} bp) into 2-bit representation...")
    packed_genome = pack_dna_sequence(genome_str)
    packed_len = len(packed_genome)
    genome_bp_len = len(genome_str)

    packed_target = pack_dna_sequence(target_sgrna_str)
    if len(packed_target) < 5:
        raise ValueError("Target sgRNA sequence must be at least 20 base pairs long.")

    target_64 = np.uint64(0)
    for b in range(5):
        target_64 = (target_64 << np.uint64(8)) | np.uint64(packed_target[b])

    print("2. Transferring data from Host RAM to Device VRAM (H2D)...")
    h2d_start = time.perf_counter()

    d_genomic_packed = cuda.to_device(packed_genome)
    d_out_indices = cuda.device_array(MAX_HITS_CAPACITY, dtype=np.int32)
    d_out_mismatches = cuda.device_array(MAX_HITS_CAPACITY, dtype=np.int32)
    d_hit_count = cuda.to_device(np.zeros(1, dtype=np.int32))

    cuda.synchronize()
    h2d_time = time.perf_counter() - h2d_start

    total_threads = genome_bp_len - 23 + 1
    blocks_per_grid = (total_threads + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    print(f"3. Launching CUDA Kernel across {blocks_per_grid:,} thread blocks ({THREADS_PER_BLOCK} threads/block)...")
    kernel_start = time.perf_counter()
    try:
        crispr_offtarget_kernel[blocks_per_grid, THREADS_PER_BLOCK](
            d_genomic_packed,
            packed_len,
            genome_bp_len,
            target_64,
            max_mismatch,
            d_out_indices,
            d_out_mismatches,
            d_hit_count,
        )
        cuda.synchronize()
    except Exception as err:
        print(f"[WARNING] CUDA kernel launch failed: {err}")
        return run_cpu_fallback(genome_str, target_sgrna_str, max_mismatch)

    kernel_time = time.perf_counter() - kernel_start

    print("4. Transferring results from Device VRAM back to Host RAM (D2H)...")
    d2h_start = time.perf_counter()

    total_hits = d_hit_count.copy_to_host()[0]
    hits_indices = d_out_indices[:total_hits].copy_to_host()
    hits_mismatches = d_out_mismatches[:total_hits].copy_to_host()
    d2h_time = time.perf_counter() - d2h_start

    return {
        'hits_count': total_hits,
        'indices': hits_indices,
        'mismatches': hits_mismatches,
        'h2d_time': h2d_time,
        'kernel_time': kernel_time,
        'd2h_time': d2h_time,
        'total_gpu_time': h2d_time + kernel_time + d2h_time,
    }


# =========================================================================
# Execution Main Method & Benchmarking Harness
# =========================================================================
if __name__ == '__main__':
    print("=" * 65)
    print("        GeneWeaver: Unified CPU & GPU Alignment Suite        ")
    print("=" * 65)

    print(f"Generating synthetic DNA dataset ({GENOME_SIZE_SIMULATION:,} bp)...")
    mem_before_gen = get_process_memory_mb()
    genome_dna = generate_synthetic_dna(GENOME_SIZE_SIMULATION)

    # Inject known off-target mutations for accuracy validation
    pos1 = 500_000
    genome_dna = genome_dna[:pos1] + (TARGET_SGRNA + PAM_PATTERN) + genome_dna[pos1 + 23:]

    pos2 = 2_500_000
    mutated_sgRNA = "GCGTCCGACTAGAAGAAGAA"  # 2 mismatches
    genome_dna = genome_dna[:pos2] + (mutated_sgRNA + PAM_PATTERN) + genome_dna[pos2 + 23:]

    print(f"Memory Overhead: {get_process_memory_mb() - mem_before_gen:.2f} MB")
    print(f"Target sgRNA + PAM: {QUERY_SEQ} (Length: {len(QUERY_SEQ)} bp)")
    print("-" * 65)

    # 1. CPU Brute-Force Matcher Benchmark
    print("\n[Benchmark 1] Executing CPU Naive Brute-Force Matcher...")
    start_time = time.perf_counter()
    bf_hits = run_brute_force_matcher(genome_dna, QUERY_SEQ, MAX_MISMATCHES)
    bf_duration = time.perf_counter() - start_time
    print(f"  └── Time Elapsed: {bf_duration:.4f} seconds | Rate: {(GENOME_SIZE_SIMULATION / bf_duration) / 1e6:.3f} MB/s")

    # 2. CPU Smith-Waterman Benchmark
    print("\n[Benchmark 2] Executing CPU Space-Optimized Smith-Waterman...")
    start_time = time.perf_counter()
    sw_hits = run_smith_waterman_scores(genome_dna, QUERY_SEQ)
    sw_duration = time.perf_counter() - start_time
    print(f"  └── Time Elapsed: {sw_duration:.4f} seconds | Rate: {(GENOME_SIZE_SIMULATION / sw_duration) / 1e6:.3f} MB/s")

    # 3. GPU Alignment Kernel Execution
    print("\n[Benchmark 3] Executing Numba CUDA Alignment Kernel...")
    gpu_results = run_gpu_alignment(genome_dna, TARGET_SGRNA, max_mismatch=MAX_MISMATCHES)

    print("\n" + "=" * 65)
    print("                      PIPELINE RUN SUMMARY                     ")
    print("=" * 65)
    print(f"Total Off-Target Hits Found: {gpu_results['hits_count']}")
    print(f"  ├── Host-to-Device (H2D) Transfer Time: {gpu_results['h2d_time']:.4f} s")
    print(f"  ├── Pure Kernel Execution Time:       {gpu_results['kernel_time']:.4f} s")
    print(f"  ├── Device-to-Host (D2H) Transfer Time: {gpu_results['d2h_time']:.4f} s")
    print(f"  └── Total GPU Stage Runtime:          {gpu_results['total_gpu_time']:.4f} s")
    
    if gpu_results['kernel_time'] > 0 and cuda_available():
        speedup = bf_duration / gpu_results['total_gpu_time']
        kernel_speedup = bf_duration / gpu_results['kernel_time']
        print(f"\n  ★ End-to-End GPU Speedup: {speedup:.2f}x vs CPU Brute-Force")
        print(f"  ★ Pure Kernel GPU Speedup: {kernel_speedup:.2f}x vs CPU Brute-Force")

    print("-" * 65)
    print("Verifying Detected Off-Target Sequence Locations:")
    for idx, mismatches in zip(gpu_results['indices'], gpu_results['mismatches']):
        extracted = genome_dna[idx : idx + 23]
        print(f"  • Position {idx:10,d} | Mismatches: {mismatches} | Sequence: {extracted}")

    assert pos1 in gpu_results['indices'], f"Error: Target lost at index {pos1}"
    assert pos2 in gpu_results['indices'], f"Error: Target lost at index {pos2}"
    print("\n[SUCCESS] Correctness check passed! All injected targets accurately detected.")